"""Small supervised-attention reproduction for MNIST and HARPL animal sprites.

This is intentionally separate from cli_attention.py. It tests whether a
bio-attention-style top-down decoder can learn full-resolution attention masks
before we draw conclusions from the harder moving-animal/RPL setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from harpl.data.attention_sprites_dataset import MovingAnimalAttentionDataset


BIO_TASKS = {
    "ior": {
        "class": "IOR_DS",
        "params": {"n_digits": 3, "n_attend": 2, "noise": 0.25, "overlap": 1.0},
        "mask_slice": None,
    },
    "arrow": {
        "class": "Arrow_DS",
        "params": {"n_iter": 3, "noise": 0.25},
        "mask_slice": None,
    },
    "cue": {
        "class": "Cue_DS",
        "params": {"fix_attend": (2, 3), "n_digits": 4, "noise": 0.25, "overlap": 0.0},
        "mask_slice": slice(1, None),
    },
    "tracking": {
        "class": "Tracking_DS",
        "params": {"fix_attend": (2, 5), "n_digits": 4, "noise": 0.25},
        "mask_slice": slice(1, None),
    },
    "recognition": {
        "class": "Recognition_DS",
        "params": {"n_iter": 3, "stride": 16, "blank": False, "static": False, "noise": 0.25},
        "mask_slice": slice(1, None),
    },
    "search": {
        "class": "Search_DS",
        "params": {"n_iter": 2, "n_digits": 4, "noise": 0.25, "overlap": 1.0},
        "mask_slice": slice(1, None),
    },
    "popout": {
        "class": "Popout_DS",
        "params": {"n_iter": 2, "noise": 0.25},
        "mask_slice": None,
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_mnist_root() -> Path:
    candidates = (
        Path("/Users/matushalak/Documents/coala/data"),
        Path("/Users/matushalak/Documents/harpl/datasets/mnist"),
        Path("./datasets/mnist"),
        Path("./data"),
    )
    for candidate in candidates:
        if (candidate / "MNIST" / "raw").exists():
            return candidate
    return candidates[-1]


def make_mnist_base(root: Path, train: bool, download: bool) -> Dataset:
    return datasets.MNIST(root=root, train=train, download=download, transform=transforms.ToTensor())


def class_indices(dataset: Dataset, n_classes: int = 10) -> list[list[int]]:
    indices: list[list[int]] = [[] for _ in range(n_classes)]
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        indices[int(label)].append(idx)
    return indices


class BuiltinSearchDataset(Dataset):
    """COALA/PMNIST-style Search task with bio-attention-shaped outputs."""

    def __init__(
        self,
        mnist: Dataset,
        *,
        n_samples: int,
        n_digits: int = 4,
        n_iter: int = 2,
        image_size: int = 96,
        noise: float = 0.25,
        seed: int = 0,
    ):
        self.mnist = mnist
        self.n_samples = n_samples
        self.n_digits = n_digits
        self.n_iter = n_iter
        self.image_size = image_size
        self.noise = noise
        self.seed = seed
        self.pad = (image_size - 28) // 2
        self.classes = class_indices(mnist, 10)

    def __len__(self) -> int:
        return self.n_samples

    def _sample_digit(self, label: int, generator: torch.Generator) -> tuple[torch.Tensor, int]:
        choices = self.classes[label]
        sample_idx = choices[int(torch.randint(len(choices), (1,), generator=generator).item())]
        image, sampled_label = self.mnist[sample_idx]
        return image.float(), int(sampled_label)

    def __getitem__(self, idx: int):
        generator = torch.Generator().manual_seed(self.seed + idx)
        components = torch.zeros(self.n_digits, 1, self.image_size, self.image_size)
        labels = torch.randperm(10, generator=generator)[: self.n_digits].tolist()

        for digit_idx, label in enumerate(labels):
            image, sampled_label = self._sample_digit(label, generator)
            if sampled_label != label:
                raise RuntimeError("MNIST class index cache returned the wrong label.")

            # Match the bio-attention Search geometry: center padded MNIST plus
            # moderate random translation/scale by choosing a bounded top-left.
            size = int(torch.randint(24, 36, (1,), generator=generator).item())
            resized = F.interpolate(image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
            canvas = torch.zeros(1, self.image_size, self.image_size)
            max_xy = self.image_size - size
            for _ in range(64):
                top = int(torch.randint(0, max_xy + 1, (1,), generator=generator).item())
                left = int(torch.randint(0, max_xy + 1, (1,), generator=generator).item())
                overlap = (components.sum(0)[:, top : top + size, left : left + size] * resized[0]).sum()
                if overlap.item() <= 1.0:
                    break
            canvas[:, top : top + size, left : left + size] = resized[0]
            components[digit_idx] = canvas

        target_id = int(torch.randint(self.n_digits, (1,), generator=generator).item())
        target_label = labels[target_id]
        colors = torch.rand(self.n_digits, 3, 1, 1, generator=generator)
        colors = colors / colors.amax(dim=1, keepdim=True).clamp_min(1e-6)
        background = torch.rand(3, 1, 1, generator=generator) * 0.5
        occupancy = components.sum(0).clamp(0.0, 1.0)
        image = (components * colors).sum(0) + (1.0 - occupancy) * background
        if self.noise > 0.0:
            image = image + self.noise * torch.rand((), generator=generator) * torch.rand(
                image.shape, generator=generator
            )
        image = image.clamp(0.0, 1.0)
        mask = components[target_id].clamp(0.0, 1.0)

        x = image.unsqueeze(0).repeat(self.n_iter, 1, 1, 1)
        y = torch.full((self.n_iter,), target_label, dtype=torch.long)
        m = mask.mul(2.0).sub(1.0).unsqueeze(0).repeat(self.n_iter, 1, 1, 1)
        hot_y = F.one_hot(y, 10).float()
        return x, y, m, components, hot_y


class BioTaskDataset(Dataset):
    """Thin wrapper around local bio-attention task composers."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def build_valid_test(self) -> None:
        if hasattr(self.dataset, "build_valid_test"):
            self.dataset.build_valid_test()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


def import_bio_composer(bio_attention_root: Path):
    sys.path.insert(0, str(bio_attention_root))
    from src import composer  # type: ignore

    return composer


def build_bio_dataset(
    *,
    task: str,
    split: str,
    mnist_root: Path,
    bio_attention_root: Path,
    data_root: Path,
    n_samples: int,
    download: bool,
) -> Dataset:
    composer = import_bio_composer(bio_attention_root)
    base = make_mnist_base(mnist_root, train=(split != "test"), download=download)
    if split == "train":
        base, _ = random_split(base, (50000, 10000), generator=torch.Generator().manual_seed(1821))
    elif split == "val":
        _, base = random_split(base, (50000, 10000), generator=torch.Generator().manual_seed(1821))

    if n_samples < len(base):
        base = Subset(base, range(n_samples))

    spec = BIO_TASKS[task]
    params = dict(spec["params"])
    if task == "arrow":
        params["directory"] = str(data_root)
    cls = getattr(composer, spec["class"])
    dataset = BioTaskDataset(cls(base, **params))
    if split != "train":
        dataset.build_valid_test()
    return dataset


def build_builtin_dataset(
    *,
    split: str,
    mnist_root: Path,
    n_samples: int,
    n_digits: int,
    n_iter: int,
    image_size: int,
    noise: float,
    seed: int,
    download: bool,
) -> Dataset:
    base = make_mnist_base(mnist_root, train=(split != "test"), download=download)
    if split == "train":
        base, _ = random_split(base, (50000, 10000), generator=torch.Generator().manual_seed(1821))
    elif split == "val":
        _, base = random_split(base, (50000, 10000), generator=torch.Generator().manual_seed(1821))
    return BuiltinSearchDataset(
        base,
        n_samples=n_samples,
        n_digits=n_digits,
        n_iter=n_iter,
        image_size=image_size,
        noise=noise if split == "train" else 0.0,
        seed=seed + {"train": 0, "val": 1_000_000, "test": 2_000_000}[split],
    )


class AnimalSearchMaskDataset(Dataset):
    """HARPL moving-animal top-down search with reconstructed target masks."""

    def __init__(
        self,
        *,
        data_dir: Path,
        split: str,
        n_samples: int,
        seq_len: int,
        image_size: int,
        crowd_size: int,
        noise: float,
        seed: int,
        max_sprites: int | None,
        sprite_img_dir: str,
        cue_frames: int,
        task: str,
    ):
        self.base = MovingAnimalAttentionDataset(
            data_dir=data_dir,
            split=split,
            task=task,
            output_size=(image_size, image_size),
            base_output_size=(64, 64),
            seq_len=seq_len,
            num_sequences=n_samples,
            sprite_img_dir=sprite_img_dir,
            max_sprites=max_sprites,
            seed=seed,
            noise_type="gaussian" if noise > 0.0 else None,
            noise_level=noise,
            noise_on_top=True,
            crowd_size=crowd_size,
            cue_frames=cue_frames,
            normalize=False,
            return_metadata=True,
            device="cpu",
        )
        self.n_classes = len(self.base.sprites)

    def __len__(self) -> int:
        return len(self.base)

    def _target_mask(self, labels: dict) -> torch.Tensor:
        target_index = int(torch.nonzero(labels["is_target"], as_tuple=False)[0].item())
        class_id = int(labels["object_class"][target_index].item())
        positions = labels["positions"][:, target_index].float()
        rotations = labels["rotations"][:, target_index].float()
        scales = labels["scales"][:, target_index].float()
        sprite = self.base.sprites[class_id].unsqueeze(0).repeat(self.base.seq_len, 1, 1, 1)
        transformed = self.base._batch_apply_transform(sprite, positions, rotations, scales)
        mask = transformed[:, 3:4].clamp(0.0, 1.0)
        visible = labels["visible"][:, target_index].bool().view(-1, 1, 1, 1)
        return torch.where(visible, mask, torch.zeros_like(mask))

    def __getitem__(self, idx: int):
        video, labels = self.base[idx]
        target_class = int(labels["target_class"].item())
        mask = self._target_mask(labels)
        y = torch.full((video.size(0),), target_class, dtype=torch.long)
        hot_y = F.one_hot(y, self.n_classes).float()
        return video, y, mask.mul(2.0).sub(1.0), torch.empty(0), hot_y


def build_animals_dataset(
    *,
    split: str,
    data_dir: Path,
    n_samples: int,
    seq_len: int,
    image_size: int,
    crowd_size: int,
    noise: float,
    seed: int,
    max_sprites: int | None,
    sprite_img_dir: str,
    cue_frames: int,
    task: str,
) -> Dataset:
    return AnimalSearchMaskDataset(
        data_dir=data_dir,
        split=split,
        n_samples=n_samples,
        seq_len=seq_len,
        image_size=image_size,
        crowd_size=crowd_size,
        noise=noise if split == "train" else 0.0,
        seed=seed + {"train": 0, "val": 1_000_000, "test": 2_000_000}[split],
        max_sprites=max_sprites,
        sprite_img_dir=sprite_img_dir,
        cue_frames=cue_frames,
        task=task,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BioLikeAttentionNet(nn.Module):
    """Compact encoder/decoder with prompt-conditioned top-down mask output."""

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 10,
        prompt_dim: int = 10,
        base_channels: int = 32,
        prompt_at_input: bool = True,
    ):
        super().__init__()
        self.prompt_at_input = prompt_at_input
        self.prompt_dim = prompt_dim
        first_channels = in_channels + prompt_dim if prompt_at_input else in_channels
        self.enc1 = ConvBlock(first_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        bottleneck_channels = base_channels * 4
        self.prompt_gain = nn.Linear(prompt_dim, bottleneck_channels)
        self.prompt_bias = nn.Linear(prompt_dim, bottleneck_channels)

        self.up3 = nn.ConvTranspose2d(bottleneck_channels, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels)
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.mask_head = nn.Conv2d(base_channels, 1, 1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_channels, n_classes),
        )

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prompt_at_input:
            prompt_planes = prompt[:, :, None, None].expand(-1, -1, x.size(-2), x.size(-1))
            x = torch.cat([x, prompt_planes], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.enc4(self.pool(e3))

        gain = torch.tanh(self.prompt_gain(prompt)).unsqueeze(-1).unsqueeze(-1)
        bias = self.prompt_bias(prompt).unsqueeze(-1).unsqueeze(-1)
        z_prompted = z * (1.0 + gain) + bias

        d3 = self.up3(z_prompted)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.mask_head(d1), self.classifier(z)


@dataclass
class Metrics:
    loss: float
    cls_loss: float
    mask_loss: float
    cls_acc: float
    mask_mse: float
    mask_iou: float
    center_error: float
    peak_hit: float
    pred_area: float
    target_area: float


def flatten_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, m, _, hot_y = batch
    x = x.to(device).float()
    y = y.to(device).long()
    m = m.to(device).float()
    if not torch.is_tensor(hot_y):
        hot_y = F.one_hot(y, 10).float()
    hot_y = hot_y.to(device).float()
    if hot_y.ndim < 3:
        hot_y = F.one_hot(y, 10).float()
    if x.ndim != 5:
        raise ValueError(f"Expected x as B,T,C,H,W; got {tuple(x.shape)}")
    if y.ndim == 1:
        y = y[:, None].expand(-1, x.size(1))
    if hot_y.ndim == 2:
        hot_y = hot_y[:, None, :].expand(-1, x.size(1), -1)
    bsz, steps = x.shape[:2]
    return (
        x.reshape(bsz * steps, *x.shape[2:]),
        y.reshape(bsz * steps),
        m.reshape(bsz * steps, *m.shape[2:]),
        hot_y.reshape(bsz * steps, hot_y.shape[-1]),
    )


def select_steps(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    prompt: torch.Tensor,
    mask_slice: slice | None,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask_slice is None:
        return x, y, mask, prompt
    bsz_steps = x.size(0)
    batch_size = bsz_steps // n_steps
    keep = torch.arange(n_steps, device=x.device)[mask_slice]
    ids = (torch.arange(batch_size, device=x.device)[:, None] * n_steps + keep[None, :]).flatten()
    return x[ids], y[ids], mask[ids], prompt[ids]


def mask_prediction(mask_logits: torch.Tensor, mask_loss_kind: str) -> torch.Tensor:
    if mask_loss_kind == "mse":
        return torch.tanh(mask_logits).add(1.0).mul(0.5)
    return torch.sigmoid(mask_logits)


def compute_mask_loss(mask_logits: torch.Tensor, target_tanh: torch.Tensor, mask_loss_kind: str, pos_weight: float):
    if mask_loss_kind == "mse":
        return F.mse_loss(torch.tanh(mask_logits), target_tanh)
    target = target_tanh.add(1.0).mul(0.5)
    weight = torch.ones_like(target)
    if pos_weight != 1.0:
        weight = torch.where(target > 0.5, torch.full_like(weight, pos_weight), weight)
    return F.binary_cross_entropy_with_logits(mask_logits, target, weight=weight)


def centers(mask: torch.Tensor) -> torch.Tensor:
    bsz, _, height, width = mask.shape
    yy = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    xx = torch.arange(width, device=mask.device, dtype=mask.dtype).view(1, 1, 1, width)
    mass = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    cy = (mask * yy).sum(dim=(2, 3), keepdim=True) / mass
    cx = (mask * xx).sum(dim=(2, 3), keepdim=True) / mass
    return torch.cat([cy.view(bsz, 1), cx.view(bsz, 1)], dim=1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    mask_loss_kind: str,
    pos_weight: float,
    cls_weight: float,
    mask_weight: float,
    mask_slice: slice | None,
) -> Metrics:
    model.eval()
    totals = {
        "loss": 0.0,
        "cls_loss": 0.0,
        "mask_loss": 0.0,
        "cls_acc": 0.0,
        "mask_mse": 0.0,
        "mask_iou": 0.0,
        "center_error": 0.0,
        "peak_hit": 0.0,
        "pred_area": 0.0,
        "target_area": 0.0,
    }
    n = 0
    for batch in loader:
        x, y, target_tanh, prompt = flatten_batch(batch, device)
        n_steps = batch[0].shape[1]
        x, y, target_tanh, prompt = select_steps(x, y, target_tanh, prompt, mask_slice, n_steps)
        mask_logits, logits = model(x, prompt)
        cls_loss = F.cross_entropy(logits, y)
        att_loss = compute_mask_loss(mask_logits, target_tanh, mask_loss_kind, pos_weight)
        loss = cls_weight * cls_loss + mask_weight * att_loss
        pred = mask_prediction(mask_logits, mask_loss_kind)
        target = target_tanh.add(1.0).mul(0.5)
        pred_bin = pred > 0.5
        target_bin = target > 0.5
        intersection = (pred_bin & target_bin).sum(dim=(1, 2, 3)).float()
        union = (pred_bin | target_bin).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
        center_error = (centers(pred) - centers(target)).norm(dim=1)
        flat_peak = pred.flatten(start_dim=1).argmax(dim=1)
        flat_target = target_bin.flatten(start_dim=1)
        peak_hit = flat_target.gather(1, flat_peak[:, None]).float().squeeze(1)
        batch_n = x.size(0)
        totals["loss"] += loss.item() * batch_n
        totals["cls_loss"] += cls_loss.item() * batch_n
        totals["mask_loss"] += att_loss.item() * batch_n
        totals["cls_acc"] += (logits.argmax(dim=1) == y).float().sum().item()
        totals["mask_mse"] += F.mse_loss(pred, target).item() * batch_n
        totals["mask_iou"] += (intersection / union).sum().item()
        totals["center_error"] += center_error.sum().item()
        totals["peak_hit"] += peak_hit.sum().item()
        totals["pred_area"] += pred_bin.float().mean(dim=(1, 2, 3)).sum().item()
        totals["target_area"] += target_bin.float().mean(dim=(1, 2, 3)).sum().item()
        n += batch_n
    return Metrics(**{key: value / max(n, 1) for key, value in totals.items()})


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    mask_loss_kind: str,
    pos_weight: float,
    cls_weight: float,
    mask_weight: float,
    mask_slice: slice | None,
    grad_clip: float,
) -> Metrics:
    model.train()
    totals = {"loss": 0.0, "cls_loss": 0.0, "mask_loss": 0.0, "cls_acc": 0.0}
    n = 0
    for batch in loader:
        x, y, target_tanh, prompt = flatten_batch(batch, device)
        n_steps = batch[0].shape[1]
        x, y, target_tanh, prompt = select_steps(x, y, target_tanh, prompt, mask_slice, n_steps)
        mask_logits, logits = model(x, prompt)
        cls_loss = F.cross_entropy(logits, y)
        att_loss = compute_mask_loss(mask_logits, target_tanh, mask_loss_kind, pos_weight)
        loss = cls_weight * cls_loss + mask_weight * att_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_n = x.size(0)
        totals["loss"] += loss.item() * batch_n
        totals["cls_loss"] += cls_loss.item() * batch_n
        totals["mask_loss"] += att_loss.item() * batch_n
        totals["cls_acc"] += (logits.argmax(dim=1) == y).float().sum().item()
        n += batch_n

    basic = {key: value / max(n, 1) for key, value in totals.items()}
    return Metrics(
        loss=basic["loss"],
        cls_loss=basic["cls_loss"],
        mask_loss=basic["mask_loss"],
        cls_acc=basic["cls_acc"],
        mask_mse=float("nan"),
        mask_iou=float("nan"),
        center_error=float("nan"),
        peak_hit=float("nan"),
        pred_area=float("nan"),
        target_area=float("nan"),
    )


@torch.no_grad()
def save_panel(
    model: nn.Module,
    loader: DataLoader,
    *,
    output_path: Path,
    device: torch.device,
    mask_loss_kind: str,
    n_examples: int,
    mask_slice: slice | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    batch = next(iter(loader))
    x, y, target_tanh, prompt = flatten_batch(batch, device)
    n_steps = batch[0].shape[1]
    x, y, target_tanh, prompt = select_steps(x, y, target_tanh, prompt, mask_slice, n_steps)
    mask_logits, logits = model(x, prompt)
    pred = mask_prediction(mask_logits, mask_loss_kind)
    target = target_tanh.add(1.0).mul(0.5)
    count = min(n_examples, x.size(0))

    fig, axes = plt.subplots(count, 4, figsize=(8, 2 * count), squeeze=False)
    for row in range(count):
        image = x[row].detach().cpu()
        if image.size(0) == 1:
            image_np = image[0].clamp(0.0, 1.0).numpy()
            axes[row, 0].imshow(image_np, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axes[row, 0].imshow(image.permute(1, 2, 0).clamp(0.0, 1.0).numpy())
        axes[row, 0].set_title(f"input y={int(y[row])} p={int(logits[row].argmax())}")
        axes[row, 1].imshow(target[row, 0].detach().cpu().numpy(), cmap="magma", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title("target")
        axes[row, 2].imshow(pred[row, 0].detach().cpu().numpy(), cmap="magma", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("pred")
        axes[row, 3].imshow(image.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy())
        axes[row, 3].imshow(pred[row, 0].detach().cpu().numpy(), cmap="viridis", alpha=0.45, vmin=0.0, vmax=1.0)
        axes[row, 3].set_title("overlay")
        for col in range(4):
            axes[row, col].axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, slice | None]:
    mnist_root = Path(args.mnist_root).expanduser().resolve()
    if args.backend == "bio":
        mask_slice = BIO_TASKS[args.task]["mask_slice"]
        train_ds = build_bio_dataset(
            task=args.task,
            split="train",
            mnist_root=mnist_root,
            bio_attention_root=Path(args.bio_attention_root).expanduser().resolve(),
            data_root=Path(args.bio_data_root).expanduser().resolve(),
            n_samples=args.train_samples,
            download=args.download,
        )
        val_ds = build_bio_dataset(
            task=args.task,
            split="val",
            mnist_root=mnist_root,
            bio_attention_root=Path(args.bio_attention_root).expanduser().resolve(),
            data_root=Path(args.bio_data_root).expanduser().resolve(),
            n_samples=args.val_samples,
            download=args.download,
        )
    elif args.backend == "animals":
        mask_slice = slice(args.animal_loss_start, None)
        data_dir = Path(args.animals_data_dir).expanduser().resolve()
        train_ds = build_animals_dataset(
            split="train",
            data_dir=data_dir,
            n_samples=args.train_samples,
            seq_len=args.animal_seq_len,
            image_size=args.image_size,
            crowd_size=args.n_digits,
            noise=args.noise,
            seed=args.seed,
            max_sprites=args.max_sprites,
            sprite_img_dir=args.sprite_img_dir,
            cue_frames=args.animal_cue_frames,
            task=args.animal_task,
        )
        val_ds = build_animals_dataset(
            split="val",
            data_dir=data_dir,
            n_samples=args.val_samples,
            seq_len=args.animal_seq_len,
            image_size=args.image_size,
            crowd_size=args.n_digits,
            noise=args.noise,
            seed=args.seed,
            max_sprites=args.max_sprites,
            sprite_img_dir=args.sprite_img_dir,
            cue_frames=args.animal_cue_frames,
            task=args.animal_task,
        )
    else:
        mask_slice = slice(1, None)
        train_ds = build_builtin_dataset(
            split="train",
            mnist_root=mnist_root,
            n_samples=args.train_samples,
            n_digits=args.n_digits,
            n_iter=args.n_iter,
            image_size=args.image_size,
            noise=args.noise,
            seed=args.seed,
            download=args.download,
        )
        val_ds = build_builtin_dataset(
            split="val",
            mnist_root=mnist_root,
            n_samples=args.val_samples,
            n_digits=args.n_digits,
            n_iter=args.n_iter,
            image_size=args.image_size,
            noise=args.noise,
            seed=args.seed,
            download=args.download,
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, mask_slice


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("builtin", "bio", "animals"), default="builtin")
    parser.add_argument("--task", choices=tuple(BIO_TASKS), default="search")
    parser.add_argument("--mnist_root", type=str, default=str(default_mnist_root()))
    parser.add_argument("--bio_attention_root", type=str, default="/Users/matushalak/Documents/bio-attention")
    parser.add_argument("--bio_data_root", type=str, default="/Users/matushalak/Documents/bio-attention/data")
    parser.add_argument("--animals_data_dir", type=str, default="/Users/matushalak/Documents/harpl/datasets")
    parser.add_argument("--sprite_img_dir", type=str, default="animals")
    parser.add_argument("--output_dir", type=str, default="reports/mnist_attention_repro/search_supervised")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=1821)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_samples", type=int, default=4096)
    parser.add_argument("--val_samples", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--no_prompt_at_input", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--mask_loss", choices=("mse", "bce"), default="mse")
    parser.add_argument("--mask_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--pos_weight", type=float, default=1.0)

    parser.add_argument("--n_digits", type=int, default=4)
    parser.add_argument("--n_iter", type=int, default=2)
    parser.add_argument("--animal_seq_len", type=int, default=2)
    parser.add_argument("--animal_cue_frames", type=int, default=1)
    parser.add_argument(
        "--animal_task",
        choices=tuple(MovingAnimalAttentionDataset.valid_tasks),
        default="top_down_search",
    )
    parser.add_argument("--animal_loss_start", type=int, default=0)
    parser.add_argument("--max_sprites", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--panel_examples", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, mask_slice = build_loaders(args)
    first_batch = next(iter(train_loader))
    first_x = first_batch[0]
    in_channels = int(first_x.shape[2])
    prompt_dim = int(first_batch[4].shape[-1])
    model = BioLikeAttentionNet(
        in_channels=in_channels,
        n_classes=prompt_dim,
        prompt_dim=prompt_dim,
        base_channels=args.base_channels,
        prompt_at_input=not args.no_prompt_at_input,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["mask_slice"] = None if mask_slice is None else (mask_slice.start, mask_slice.stop, mask_slice.step)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    rows: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            mask_loss_kind=args.mask_loss,
            pos_weight=args.pos_weight,
            cls_weight=args.cls_weight,
            mask_weight=args.mask_weight,
            mask_slice=mask_slice,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            mask_loss_kind=args.mask_loss,
            pos_weight=args.pos_weight,
            cls_weight=args.cls_weight,
            mask_weight=args.mask_weight,
            mask_slice=mask_slice,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in asdict(train_metrics).items()},
            **{f"val_{key}": value for key, value in asdict(val_metrics).items()},
        }
        rows.append(row)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics.loss:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.cls_acc:.3f} "
            f"val_iou={val_metrics.mask_iou:.3f} "
            f"val_center={val_metrics.center_error:.2f}",
            flush=True,
        )
        write_metrics(output_dir / "metrics.csv", rows)

    save_panel(
        model,
        val_loader,
        output_path=output_dir / "attention_panel.png",
        device=device,
        mask_loss_kind=args.mask_loss,
        n_examples=args.panel_examples,
        mask_slice=mask_slice,
    )
    torch.save(model.state_dict(), output_dir / "model.pt")


if __name__ == "__main__":
    main()
