import argparse
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from harpl.data.attention_sprites_dataset import (
    MovingAnimalAttentionDataset,
    TASK_TO_ID,
)
from harpl.data.synthetic_sprites_dataset import SpriteVideoDataset
from harpl.scripts.eval_utils import (
    evaluate_attention_model,
    evaluate_cross_decode_sprites,
    evaluate_pretrained_attention_tasks,
)
from harpl.networks.harpl import (
    ARPLmodel,
    AttentionDecoder,
    ClassificationHead,
)
from harpl.networks.backbones import Conv2dDecoder
from harpl.networks.utils import additional_data_process, prepare_model
from harpl.scripts.args import (
    add_model_args,
    add_optimization_args,
    add_reproducibility_args,
)
from harpl.scripts.utils import (
    close_logger,
    cuda_memory_stats,
    get_data_specs,
    init_logger,
    is_cuda_device,
    log_variable,
    seed_everything,
    select_device,
)


def _normalize_output_size(value):
    if isinstance(value, int):
        return (value, value)
    value = list(value)
    if len(value) == 1:
        return (value[0], value[0])
    if len(value) == 2:
        return (value[0], value[1])
    raise ValueError("--spritevid_output_size expects one value or two values: height width.")


def _parse_task_values(values):
    if values is None:
        return None
    parsed = []
    for value in values:
        if isinstance(value, str) and value.lstrip("-").isdigit():
            parsed.append(int(value))
        else:
            parsed.append(value)
    return parsed


def _parse_optional_scalar_or_channels(value):
    if value is None:
        return None
    value = list(value)
    if len(value) == 1:
        return value[0]
    if len(value) == 3:
        return tuple(value)
    raise ValueError("Expected either one scalar value or three channel values.")


def _dataloader_kwargs(args, device, shuffle=True):
    if torch.device(args.spritevid_device).type == "cuda" and args.num_workers != 0:
        raise ValueError("CUDA attention dataset rendering requires --num_workers 0.")
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and device.type == "cuda",
        "drop_last": False,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor is not None:
            kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def _infer_context_dim(integrator, fallback):
    if hasattr(integrator, "get_output_dim"):
        try:
            return int(integrator.get_output_dim())
        except NotImplementedError:
            pass
    backbone = getattr(integrator, "backbone", None)
    hidden_size = getattr(backbone, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    return int(fallback)


def _infer_first_encoder_shape(encoder, input_size, n_in_channels):
    encoder_first, _, _ = ARPLmodel._split_encoder(encoder)
    encoder_first.eval()
    height, width = _normalize_output_size(input_size)
    dummy = torch.zeros(1, 1, n_in_channels, height, width)
    with torch.no_grad():
        features = ARPLmodel._forward_video_layers(encoder_first, dummy)
    if features.ndim != 5:
        raise RuntimeError(
            "Expected first encoder block to return spatial features shaped "
            f"(B, T, C, H, W), got {tuple(features.shape)}"
        )
    return tuple(int(dim) for dim in features.shape[2:])


def _infer_full_encoder_shape(encoder, input_size, n_in_channels):
    encoder.eval()
    height, width = _normalize_output_size(input_size)
    dummy = torch.zeros(1, 1, n_in_channels, height, width)
    with torch.no_grad():
        features = encoder(dummy)
    if features.ndim == 5:
        return tuple(int(dim) for dim in features.shape[2:])
    return None


def _infer_encoder_stage_input_shapes(encoder, input_size, n_in_channels):
    stages = ARPLmodel._split_encoder_stages(encoder)
    height, width = _normalize_output_size(input_size)
    x = torch.zeros(1, 1, n_in_channels, height, width)
    shapes = []
    with torch.no_grad():
        for stage in stages:
            shapes.append(tuple(int(dim) for dim in x.shape[2:]))
            x = ARPLmodel._forward_video_layers(stage, x)
    return shapes


def _attention_output_shape(first_encoder_shape, attention_dims):
    channels, height, width = first_encoder_shape
    if attention_dims == "features":
        return (channels, 1, 1)
    if attention_dims == "spatial":
        return (1, height, width)
    if attention_dims == "features+spatial":
        return (channels, height, width)
    raise ValueError("--attention_dims must be one of: features, spatial, features+spatial")


def _expand_encoder_layer_arg(value, n_layers, name):
    if value is None:
        return None
    value = list(value)
    if len(value) == 1 and n_layers > 1:
        return value * n_layers
    if len(value) != n_layers:
        raise ValueError(f"--{name} must provide 1 value or {n_layers} values.")
    return value


def _encoder_tail_decoder_args(args):
    if args.enc_n_layers < 2:
        raise ValueError("--attention_conv2d_decoder requires at least two encoder layers.")
    n_layers = args.enc_n_layers - 1
    return {
        "n_layers": n_layers,
        "kernel_size": _expand_encoder_layer_arg(args.enc_kernel_size, args.enc_n_layers, "enc_kernel_size")[1:],
        "stride": _expand_encoder_layer_arg(args.enc_stride, args.enc_n_layers, "enc_stride")[1:],
        "padding": (
            _expand_encoder_layer_arg(args.enc_padding, args.enc_n_layers, "enc_padding")[1:]
            if args.enc_padding is not None
            else None
        ),
        "max_pool_size": (
            _expand_encoder_layer_arg(args.enc_pool_size, args.enc_n_layers, "enc_pool_size")[1:]
            if args.enc_pool_size is not None
            else None
        ),
    }


def _encoder_all_decoder_args(args):
    return {
        "n_layers": args.enc_n_layers,
        "kernel_size": _expand_encoder_layer_arg(args.enc_kernel_size, args.enc_n_layers, "enc_kernel_size"),
        "stride": _expand_encoder_layer_arg(args.enc_stride, args.enc_n_layers, "enc_stride"),
        "padding": _expand_encoder_layer_arg(args.enc_padding, args.enc_n_layers, "enc_padding")
        if args.enc_padding is not None
        else None,
        "max_pool_size": _expand_encoder_layer_arg(args.enc_pool_size, args.enc_n_layers, "enc_pool_size")
        if args.enc_pool_size is not None
        else None,
    }


class Conv2dAttentionDecoder(nn.Module):
    """Conv2d attention decoder that mirrors the encoder tail."""

    def __init__(self, input_shape, output_shape, attention_dims, use_batch_norm=False, **decoder_kwargs):
        super().__init__()
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.output_shape = tuple(int(dim) for dim in output_shape)
        self.attention_dims = attention_dims
        self.input_dim = int(torch.tensor(self.input_shape).prod().item())
        out_channels = 1 if attention_dims == "spatial" else self.output_shape[0]
        self.decoder = Conv2dDecoder(
            n_in_channels=self.input_shape[0],
            channel_dim=out_channels,
            use_batch_norm=use_batch_norm,
            return_full_feature_map=True,
            **decoder_kwargs,
        )

    def forward(self, x):
        if x.shape[-1] != self.input_dim:
            raise RuntimeError(
                "conv2d attention decoder input must match flattened encoder output; "
                f"got {x.shape[-1]} features, expected {self.input_dim} from {self.input_shape}"
            )
        attention = x.reshape(*x.shape[:-1], *self.input_shape)
        attention = self.decoder(attention)
        if self.attention_dims == "features":
            attention = attention.mean(dim=(-1, -2), keepdim=True)
        if tuple(attention.shape[-3:]) != self.output_shape:
            raise RuntimeError(
                "conv2d attention decoder output shape does not match first encoder features; "
                f"got {tuple(attention.shape[-3:])}, expected {self.output_shape}"
            )
        return 1.0 + torch.tanh(attention)


class BioInspiredAttentionDecoder(nn.Module):
    """Top-down decoder with a bottom-up skip, following bio-attention's mask path."""

    uses_skip_features = True

    def __init__(
        self,
        input_shape,
        skip_shape,
        output_shape,
        attention_dims,
        use_batch_norm=False,
        skip_hidden_channels=None,
        attention_gain=0.5,
        **decoder_kwargs,
    ):
        super().__init__()
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.skip_shape = tuple(int(dim) for dim in skip_shape)
        self.output_shape = tuple(int(dim) for dim in output_shape)
        self.attention_dims = attention_dims
        self.input_dim = int(torch.tensor(self.input_shape).prod().item())
        self.attention_gain = float(attention_gain)
        decoded_channels = self.skip_shape[0]
        hidden_channels = skip_hidden_channels or decoded_channels
        self.top_down = Conv2dDecoder(
            n_in_channels=self.input_shape[0],
            channel_dim=decoded_channels,
            use_batch_norm=use_batch_norm,
            return_full_feature_map=True,
            **decoder_kwargs,
        )
        layers = [
            nn.Conv2d(decoded_channels + self.skip_shape[0], hidden_channels, kernel_size=3, padding=1),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.extend(
            [
                nn.ReLU(),
                nn.Conv2d(hidden_channels, self.output_shape[0], kernel_size=3, padding=1),
            ]
        )
        self.skip_readout = nn.Sequential(*layers)

    def forward(self, x, skip_features):
        if x.shape[-1] != self.input_dim:
            raise RuntimeError(
                "bio-inspired attention decoder input must match flattened encoder output; "
                f"got {x.shape[-1]} features, expected {self.input_dim} from {self.input_shape}"
            )
        if skip_features.ndim != 5:
            raise RuntimeError(
                "bio-inspired attention decoder expects skip features shaped "
                f"(batch, time, channels, height, width), got {tuple(skip_features.shape)}"
            )
        batch_size, seq_len = x.shape[:2]
        top_down = x.reshape(batch_size, seq_len, *self.input_shape)
        top_down = self.top_down(top_down)
        if tuple(top_down.shape[-3:]) != self.skip_shape or tuple(skip_features.shape[-3:]) != self.skip_shape:
            raise RuntimeError(
                "bio-inspired top-down decoder output and skip features must match skip_shape; "
                f"got top-down {tuple(top_down.shape[-3:])}, skip {tuple(skip_features.shape[-3:])}, "
                f"expected {self.skip_shape}"
            )
        merged = torch.cat((top_down, skip_features), dim=2)
        merged = merged.reshape(batch_size * seq_len, *merged.shape[2:])
        attention = self.skip_readout(merged)
        attention = attention.reshape(batch_size, seq_len, *attention.shape[1:])
        if self.attention_dims == "features":
            attention = attention.mean(dim=(-1, -2), keepdim=True)
        elif self.attention_dims == "spatial":
            attention = attention.mean(dim=2, keepdim=True)
        if tuple(attention.shape[-3:]) != self.output_shape:
            raise RuntimeError(
                "bio-inspired attention decoder output shape does not match first encoder features; "
                f"got {tuple(attention.shape[-3:])}, expected {self.output_shape}"
            )
        return 1.0 + (self.attention_gain * torch.tanh(attention))


class BioEncoderLayerAttentionDecoder(nn.Module):
    """Bio-attention style decoder that emits recurrent masks for every encoder stage input."""

    uses_skip_features = True
    returns_attention_list = True

    def __init__(
        self,
        input_shape,
        stage_input_shapes,
        attention_dims,
        kernel_size,
        stride,
        padding=None,
        max_pool_size=None,
        n_layers=None,
        use_batch_norm=False,
        skip_hidden_channels=None,
        attention_gain=0.5,
    ):
        super().__init__()
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.stage_input_shapes = [tuple(int(dim) for dim in shape) for shape in stage_input_shapes]
        self.attention_dims = attention_dims
        self.input_dim = int(torch.tensor(self.input_shape).prod().item())
        self.attention_gain = float(attention_gain)
        self.n_layers = len(self.stage_input_shapes)
        if self.n_layers < 1:
            raise ValueError("BioEncoderLayerAttentionDecoder requires at least one encoder stage.")
        if n_layers is not None and int(n_layers) != self.n_layers:
            raise ValueError(
                f"n_layers={n_layers} does not match inferred encoder stage count {self.n_layers}."
            )

        self.kernel_size = list(kernel_size)
        self.stride = list(stride)
        self.padding = list(padding) if padding is not None else [(0, 0)] * self.n_layers
        self.max_pool_size = list(max_pool_size) if max_pool_size is not None else None
        if len(self.kernel_size) != self.n_layers or len(self.stride) != self.n_layers or len(self.padding) != self.n_layers:
            raise ValueError("Decoder layer args must match the number of encoder stages.")
        if self.max_pool_size is not None and len(self.max_pool_size) != self.n_layers:
            raise ValueError("max_pool_size must match the number of encoder stages.")

        self.up_blocks = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()
        self.readouts = nn.ModuleList()
        top_down_channels = self.input_shape[0]
        for enc_i in range(self.n_layers - 1, -1, -1):
            skip_channels = self.stage_input_shapes[enc_i][0]
            up_layers = []
            if self.max_pool_size is not None:
                up_layers.append(nn.Upsample(scale_factor=self.max_pool_size[enc_i], mode="nearest"))
            up_layers.append(
                nn.ConvTranspose2d(
                    in_channels=top_down_channels,
                    out_channels=skip_channels,
                    kernel_size=self.kernel_size[enc_i],
                    stride=self.stride[enc_i],
                    padding=self.padding[enc_i],
                    output_padding=(
                        (self.stride[enc_i][0] - 1, self.stride[enc_i][1] - 1)
                        if self.stride[enc_i][0] > 1 or self.stride[enc_i][1] > 1
                        else 0
                    ),
                )
            )
            hidden_channels = int(skip_hidden_channels or skip_channels)
            merge_layers = [nn.Conv2d(skip_channels * 2, hidden_channels, kernel_size=3, padding=1)]
            if use_batch_norm:
                merge_layers.append(nn.BatchNorm2d(hidden_channels))
            merge_layers.extend([nn.Tanh(), nn.Conv2d(hidden_channels, skip_channels, kernel_size=3, padding=1), nn.Tanh()])
            mask_channels = 1 if attention_dims == "spatial" else skip_channels
            self.up_blocks.append(nn.Sequential(*up_layers))
            self.merge_blocks.append(nn.Sequential(*merge_layers))
            self.readouts.append(nn.Conv2d(skip_channels, mask_channels, kernel_size=3, padding=1))
            top_down_channels = skip_channels

    def _reshape_video(self, x, batch_size, seq_len):
        return x.reshape(batch_size, seq_len, *x.shape[1:])

    def forward(self, x, skip_features):
        if x.shape[-1] != self.input_dim:
            raise RuntimeError(
                "bio encoder-layer decoder input must match flattened encoder output; "
                f"got {x.shape[-1]} features, expected {self.input_dim} from {self.input_shape}"
            )
        if len(skip_features) != self.n_layers:
            raise RuntimeError(
                f"Expected {self.n_layers} skip feature tensors, got {len(skip_features)}."
            )
        batch_size, seq_len = x.shape[:2]
        h = x.reshape(batch_size * seq_len, *self.input_shape)
        masks = [None] * self.n_layers
        for rev_i, enc_i in enumerate(range(self.n_layers - 1, -1, -1)):
            skip = skip_features[enc_i]
            if skip.ndim != 5:
                raise RuntimeError(
                    "bio encoder-layer decoder expects each skip tensor shaped "
                    f"(batch, time, channels, height, width), got {tuple(skip.shape)}"
                )
            skip_flat = skip.reshape(batch_size * seq_len, *skip.shape[2:])
            h = self.up_blocks[rev_i](h)
            if h.shape[-2:] != skip_flat.shape[-2:]:
                h = F.interpolate(h, size=skip_flat.shape[-2:], mode="bilinear", align_corners=False)
            if h.shape[1] != skip_flat.shape[1]:
                raise RuntimeError(
                    "Top-down feature channels must match skip channels before merging; "
                    f"got top-down {h.shape[1]}, skip {skip_flat.shape[1]} at encoder layer {enc_i}."
                )
            h = self.merge_blocks[rev_i](torch.cat((h, skip_flat), dim=1))
            mask = self.readouts[rev_i](h)
            if self.attention_dims == "features":
                mask = mask.mean(dim=(-1, -2), keepdim=True)
            masks[enc_i] = 1.0 + (self.attention_gain * torch.tanh(self._reshape_video(mask, batch_size, seq_len)))
        return masks


def _infer_predictor_output_dim(predictor, fallback):
    last_linear = None
    for module in predictor.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        return int(last_linear.out_features)
    return int(fallback)


def _default_readout_path(model_path):
    return Path(model_path).parent / "online_ctx_readout.pt"


def _get_attention_normalization_stats(args, output_size):
    if hasattr(args, "_attention_normalization_stats"):
        return args._attention_normalization_stats

    stats_dataset = SpriteVideoDataset(
        data_dir=args.data_input_dir,
        split="train",
        output_size=output_size,
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        background=args.attention_background,
        grayscale=args.grayscale,
        device=args.spritevid_device,
        seed=args.seed,
        max_sprites=args.spritevid_max_sprites,
        sprite_img_dir=args.sprite_img_dir,
        discretize_latents=getattr(args, "spritevid_discretize_latents", False),
        noise_type=args.spritevid_noise_type,
        noise_intensity=args.spritevid_noise_level,
        freeze_noise=args.spritevid_frozen_noise,
        noise_on_top=args.sprite_noise_on_top,
    )
    mean = stats_dataset.mean.detach().cpu()
    std = stats_dataset.std.detach().cpu()
    if args.grayscale:
        mean = float(mean.reshape(-1)[0].item())
        std = float(std.reshape(-1)[0].item())
    else:
        mean = tuple(float(value) for value in mean.reshape(-1).tolist())
        std = tuple(float(value) for value in std.reshape(-1).tolist())
    args._attention_normalization_stats = (mean, std)
    print(f"attention_normalization_mean={mean} std={std}")
    return args._attention_normalization_stats


def _attention_normalization_args(args, output_size):
    if not args.attention_normalize:
        return False, None, None
    mean = _parse_optional_scalar_or_channels(args.attention_mean)
    std = _parse_optional_scalar_or_channels(args.attention_std)
    if mean is None or std is None:
        mean, std = _get_attention_normalization_stats(args, output_size)
    return True, mean, std


def _load_frozen_classification_head(head, readout_path, device):
    readout_path = Path(readout_path)
    if not readout_path.exists():
        raise FileNotFoundError(
            f"Frozen online readout checkpoint not found: {readout_path}. "
            "Pass --attention_readout_path to use a different readout."
        )
    state_dict = torch.load(readout_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict-like readout checkpoint, got {type(state_dict).__name__}.")

    candidates = [
        ("0.layers.readout.weight", "0.layers.readout.bias"),
        ("layers.readout.weight", "layers.readout.bias"),
        ("readout.weight", "readout.bias"),
    ]
    for weight_key, bias_key in candidates:
        if weight_key in state_dict and bias_key in state_dict:
            readout_state = {
                "weight": state_dict[weight_key],
                "bias": state_dict[bias_key],
            }
            break
    else:
        raise KeyError(
            "Could not find a sprite-classification readout in "
            f"{readout_path}; expected 0.layers.readout.*, layers.readout.*, or readout.* keys."
        )

    expected_weight_shape = tuple(head.readout.weight.shape)
    actual_weight_shape = tuple(readout_state["weight"].shape)
    if actual_weight_shape != expected_weight_shape:
        raise RuntimeError(
            f"Readout weight shape {actual_weight_shape} does not match attention head "
            f"shape {expected_weight_shape}."
        )
    head.readout.load_state_dict(readout_state)
    head.to(device)
    head.eval()
    for param in head.parameters():
        param.requires_grad_(False)
    return readout_path


def _make_attention_targets(task_info, seq_len, cue_frames):
    targets = task_info[:, 1].unsqueeze(1).expand(-1, seq_len)
    mask = torch.ones_like(targets, dtype=torch.bool)
    grouping = task_info[:, 0] == TASK_TO_ID["perceptual_grouping"]
    if grouping.any():
        mask[grouping, :cue_frames] = False
    return targets, mask


def _attention_time_weights(seq_len, args, device):
    weights = torch.ones(seq_len, dtype=torch.float32, device=device)
    warmup_frames = max(0, int(getattr(args, "attention_loss_warmup_frames", 0)))
    end_weight = float(getattr(args, "attention_loss_end_weight", 1.0))
    if warmup_frames > 0:
        weights[: min(warmup_frames, seq_len)] = 0.0
    if end_weight != 1.0:
        ramp_start = min(max(warmup_frames, 0), seq_len)
        if ramp_start < seq_len:
            ramp = torch.linspace(1.0, end_weight, seq_len - ramp_start, device=device)
            weights[ramp_start:] = ramp
    return weights


def _weighted_classification_loss(logits, targets, mask, criterion, args):
    item_loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).reshape_as(targets)
    weights = _attention_time_weights(logits.size(1), args, logits.device).view(1, -1).expand_as(item_loss)
    weights = weights * mask.to(dtype=weights.dtype)
    denom = weights.sum().clamp_min(1.0)
    return (item_loss * weights).sum() / denom, weights > 0


def _attention_regularization_loss(model, args):
    attention = getattr(model, "last_attention_maps", None)
    if attention is None:
        return None
    deviation = attention - 1.0
    spatial = deviation.abs().mean(dim=2)
    reg = attention.new_tensor(0.0)
    if args.attention_area_weight:
        reg = reg + args.attention_area_weight * spatial.mean()
    if args.attention_tv_weight:
        tv_h = spatial[:, :, 1:, :].sub(spatial[:, :, :-1, :]).abs().mean()
        tv_w = spatial[:, :, :, 1:].sub(spatial[:, :, :, :-1]).abs().mean()
        reg = reg + args.attention_tv_weight * (tv_h + tv_w)
    return reg


def _target_object_positions(labels):
    target_index = labels["is_target"].to(dtype=torch.float32).argmax(dim=1)
    batch_index = torch.arange(target_index.size(0), device=target_index.device)
    return labels["positions"][batch_index, :, target_index, :]


def _target_object_visible(labels):
    target_index = labels["is_target"].to(dtype=torch.float32).argmax(dim=1)
    batch_index = torch.arange(target_index.size(0), device=target_index.device)
    return labels["visible"][batch_index, :, target_index]


def _attention_target_from_mass(mass, args):
    if getattr(args, "attention_supervision_scale", "modulation") == "bio":
        gain = float(getattr(args, "attention_bio_gain", 0.5))
        return 1.0 + gain * (2.0 * mass - 1.0)
    return 1.0 + mass


def _attention_probability(attention, args):
    if getattr(args, "attention_supervision_scale", "modulation") == "bio":
        gain = max(float(getattr(args, "attention_bio_gain", 0.5)), 1e-6)
        return ((attention - (1.0 - gain)) / (2.0 * gain)).clamp(min=1e-6, max=1.0 - 1e-6)
    return (attention - 1.0).clamp(min=1e-6, max=1.0 - 1e-6)


def _make_gaussian_attention_target(attention, labels, input_size, sigma, args):
    _, att_seq_len, channels, height, width = attention.shape
    positions = _target_object_positions(labels)[:, 1 : att_seq_len + 1].to(
        device=attention.device,
        dtype=attention.dtype,
    )
    visible = _target_object_visible(labels)[:, 1 : att_seq_len + 1].to(device=attention.device)

    input_height, input_width = input_size
    target_x = positions[..., 0] * (width / float(input_width))
    target_y = positions[..., 1] * (height / float(input_height))
    yy = torch.arange(height, device=attention.device, dtype=attention.dtype).view(1, 1, height, 1)
    xx = torch.arange(width, device=attention.device, dtype=attention.dtype).view(1, 1, 1, width)
    sigma = max(float(sigma), 1e-6)
    gaussian = torch.exp(
        -(
            (xx - target_x[..., None, None]).square()
            + (yy - target_y[..., None, None]).square()
        )
        / (2.0 * sigma * sigma)
    )
    gaussian = gaussian[:, :, None, :, :]
    target = _attention_target_from_mass(gaussian, args)
    if channels != 1:
        target = target.expand(-1, -1, channels, -1, -1)
        gaussian = gaussian.expand(-1, -1, channels, -1, -1)
    visible = visible[:, :, None, None, None]
    return target, visible, gaussian


def _make_silhouette_attention_target(attention, labels, dataset, args=None):
    _, att_seq_len, channels, height, width = attention.shape
    device = attention.device
    dtype = attention.dtype
    target_index = labels["is_target"].to(device=device, dtype=torch.float32).argmax(dim=1)
    batch_index = torch.arange(target_index.size(0), device=device)
    class_ids = labels["object_class"].to(device=device)[batch_index, target_index].long()
    positions = labels["positions"].to(device=device, dtype=torch.float32)[batch_index, : att_seq_len + 1, target_index, :]
    rotations = labels["rotations"].to(device=device, dtype=torch.float32)[batch_index, : att_seq_len + 1, target_index]
    scales = labels["scales"].to(device=device, dtype=torch.float32)[batch_index, : att_seq_len + 1, target_index]
    visible = labels["visible"].to(device=device)[batch_index, 1 : att_seq_len + 1, target_index].bool()

    # Attention at recurrent step t is applied to input frame t, while the first
    # generated attention map is for frame 1. Match the existing Gaussian target
    # convention by dropping the frame-0 transform.
    positions = positions[:, 1 : att_seq_len + 1]
    rotations = rotations[:, 1 : att_seq_len + 1]
    scales = scales[:, 1 : att_seq_len + 1]

    sprites = torch.stack(
        [dataset.sprites[int(class_id.item())].to(device=device, dtype=dtype) for class_id in class_ids],
        dim=0,
    )
    sprites = sprites[:, None].expand(-1, att_seq_len, -1, -1, -1).reshape(
        class_ids.size(0) * att_seq_len,
        *sprites.shape[1:],
    )
    transformed = dataset._batch_apply_transform(
        sprites,
        positions.reshape(-1, 2),
        rotations.reshape(-1),
        scales.reshape(-1),
    )
    silhouette = transformed[:, 3:4].clamp(0.0, 1.0)
    silhouette = silhouette.reshape(class_ids.size(0), att_seq_len, 1, *silhouette.shape[-2:])
    if silhouette.shape[-2:] != (height, width):
        silhouette = F.interpolate(
            silhouette.reshape(-1, 1, *silhouette.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(class_ids.size(0), att_seq_len, 1, height, width)
    if channels != 1:
        silhouette = silhouette.expand(-1, -1, channels, -1, -1)
    visible = visible[:, :, None, None, None]
    target = 1.0 + silhouette if args is None else _attention_target_from_mass(silhouette, args)
    return target, visible, silhouette


def _single_attention_supervision_loss(attention, labels, args, dataset=None):
    if not isinstance(labels, dict):
        raise ValueError("--attention_supervision_weight requires --attention_return_metadata.")
    if args.attention_supervision_target == "silhouette":
        if dataset is None:
            raise ValueError("silhouette attention supervision requires the attention dataset.")
        target, mask, target_mass_source = _make_silhouette_attention_target(attention, labels, dataset, args)
    else:
        input_size = _normalize_output_size(args.spritevid_output_size)
        target, mask, target_mass_source = _make_gaussian_attention_target(
            attention,
            labels,
            input_size,
            args.attention_supervision_sigma,
            args,
        )
    if not mask.any():
        return attention.new_tensor(0.0)
    warmup_frames = max(0, int(getattr(args, "attention_loss_warmup_frames", 0)))
    if warmup_frames > 1:
        frame_index = torch.arange(1, attention.size(1) + 1, device=attention.device)
        mask = mask & (frame_index.view(1, -1, 1, 1, 1) > warmup_frames)
    if not mask.any():
        return attention.new_tensor(0.0)
    weights = 1.0 + args.attention_supervision_positive_weight * target_mass_source
    loss = attention.new_tensor(0.0)
    if args.attention_supervision_mse_weight:
        mse = (attention - target).square() * weights
        loss = loss + args.attention_supervision_mse_weight * mse[mask.expand_as(mse)].mean()
    if args.attention_supervision_spatial_ce_weight:
        target_mass = target_mass_source / target_mass_source.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        logits = (attention - 1.0) / max(float(args.attention_supervision_temperature), 1e-6)
        log_probs = torch.log_softmax(logits.flatten(-2), dim=-1).reshape_as(logits)
        spatial_ce = -(target_mass * log_probs).sum(dim=(-1, -2), keepdim=True)
        loss = loss + args.attention_supervision_spatial_ce_weight * spatial_ce[mask.expand_as(spatial_ce)].mean()
    if args.attention_supervision_bce_weight:
        bce_target = target_mass_source
        if args.attention_supervision_bce_threshold > 0.0:
            bce_target = (target_mass_source >= args.attention_supervision_bce_threshold).to(dtype=attention.dtype)
        probs = _attention_probability(attention, args)
        bce_weights = 1.0 + args.attention_supervision_positive_weight * bce_target
        bce = F.binary_cross_entropy(probs, bce_target, weight=bce_weights, reduction="none")
        loss = loss + args.attention_supervision_bce_weight * bce[mask.expand_as(bce)].mean()
    return loss


def _attention_supervision_loss(model, labels, args, dataset=None):
    if not args.attention_supervision_weight:
        return None
    attention = getattr(model, "last_attention_maps", None)
    if attention is None:
        return None
    attention_layers = [attention]
    if getattr(args, "attention_supervise_all_layers", True):
        model_layers = getattr(model, "last_attention_maps_by_layer", None)
        if model_layers:
            attention_layers = model_layers
    losses = [
        _single_attention_supervision_loss(attention_layer, labels, args, dataset)
        for attention_layer in attention_layers
    ]
    losses = [loss for loss in losses if loss is not None]
    if not losses:
        return None
    return torch.stack(losses).mean()


def _attention_alignment_metrics(attention, labels, dataset):
    _, _, _, height, width = attention.shape
    _, visible, silhouette = _make_silhouette_attention_target(attention, labels, dataset)
    silhouette = silhouette.mean(dim=2, keepdim=True)
    score = attention.mean(dim=2, keepdim=True)
    valid = visible.expand_as(silhouette).bool()
    target_mask = (silhouette > 0.5) & valid
    predicted_mask = (score > 1.0).expand_as(target_mask) & valid
    intersection = (predicted_mask & target_mask).sum(dtype=torch.float32)
    union = (predicted_mask | target_mask).sum(dtype=torch.float32).clamp_min(1.0)

    flat_target = target_mask.reshape(target_mask.size(0), target_mask.size(1), -1)
    flat_score = score.reshape(score.size(0), score.size(1), -1)
    peak_index = flat_score.argmax(dim=-1, keepdim=True)
    peak_hit = flat_target.gather(-1, peak_index).squeeze(-1)
    frame_valid = visible.reshape(visible.size(0), visible.size(1))
    valid_frames = frame_valid.sum(dtype=torch.float32).clamp_min(1.0)

    foreground = target_mask
    background = (~target_mask) & valid
    foreground_count = foreground.sum(dtype=torch.float32).clamp_min(1.0)
    background_count = background.sum(dtype=torch.float32).clamp_min(1.0)
    positive_count = predicted_mask.sum(dtype=torch.float32)
    valid_count = valid.sum(dtype=torch.float32).clamp_min(1.0)
    return {
        "iou_num": intersection,
        "iou_den": union,
        "peak_hit_num": (peak_hit & frame_valid).sum(dtype=torch.float32),
        "peak_hit_den": valid_frames,
        "foreground_score_sum": score.expand_as(foreground)[foreground].sum(),
        "foreground_score_count": foreground_count,
        "background_score_sum": score.expand_as(background)[background].sum(),
        "background_score_count": background_count,
        "predicted_area_sum": positive_count / valid_count,
        "batch_count": torch.tensor(1.0, device=attention.device),
        "height": torch.tensor(float(height), device=attention.device),
        "width": torch.tensor(float(width), device=attention.device),
    }


def _denormalize_video_frames(frames, dataset):
    if not getattr(dataset, "normalize", False):
        return frames.clamp(0.0, 1.0)
    mean = torch.as_tensor(dataset.mean, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    std = torch.as_tensor(dataset.std, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    return (frames * std + mean).clamp(0.0, 1.0)


def _save_attention_alignment_panel(args, video, labels, attention, dataset, output_path):
    import matplotlib.pyplot as plt

    max_examples = min(int(args.attention_panel_examples), video.size(0))
    max_frames = min(8, attention.size(1))
    frames = _denormalize_video_frames(video[:max_examples, 1 : max_frames + 1].detach().cpu(), dataset)
    panel_labels = {
        key: value[:max_examples] if torch.is_tensor(value) and value.size(0) == video.size(0) else value
        for key, value in labels.items()
    }
    _, _, silhouette = _make_silhouette_attention_target(attention[:max_examples, :max_frames], panel_labels, dataset)
    silhouette = silhouette.mean(dim=2, keepdim=True)
    score = attention[:max_examples, :max_frames].mean(dim=2, keepdim=True)
    target_size = tuple(frames.shape[-2:])
    if score.shape[-2:] != target_size:
        score = F.interpolate(
            score.reshape(-1, 1, *score.shape[-2:]),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).reshape(max_examples, max_frames, 1, *target_size)
    if silhouette.shape[-2:] != target_size:
        silhouette = F.interpolate(
            silhouette.reshape(-1, 1, *silhouette.shape[-2:]),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).reshape(max_examples, max_frames, 1, *target_size)

    fig, axes = plt.subplots(max_examples * 3, max_frames, figsize=(1.8 * max_frames, 4.8 * max_examples))
    if max_examples * 3 == 1:
        axes = axes[None, :]
    for example_idx in range(max_examples):
        for frame_idx in range(max_frames):
            row = example_idx * 3
            axes[row, frame_idx].imshow(frames[example_idx, frame_idx].permute(1, 2, 0))
            axes[row + 1, frame_idx].imshow(score[example_idx, frame_idx, 0].detach().cpu(), cmap="magma", vmin=0.0, vmax=2.0)
            axes[row + 2, frame_idx].imshow(silhouette[example_idx, frame_idx, 0].detach().cpu(), cmap="gray", vmin=0.0, vmax=1.0)
            for offset in range(3):
                axes[row + offset, frame_idx].set_axis_off()
            if frame_idx == 0:
                axes[row, frame_idx].set_ylabel(f"ex {example_idx} frame", fontsize=8)
                axes[row + 1, frame_idx].set_ylabel("attention", fontsize=8)
                axes[row + 2, frame_idx].set_ylabel("target", fontsize=8)
    fig.tight_layout(pad=0.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _accumulate_attention_alignment(totals, attention, labels, dataset):
    metrics = _attention_alignment_metrics(attention, labels, dataset)
    if totals is None:
        return {key: value.detach().clone() for key, value in metrics.items()}
    for key, value in metrics.items():
        totals[key] = totals[key] + value.detach()
    return totals


def _print_attention_alignment_totals(totals, split):
    iou = (totals["iou_num"] / totals["iou_den"]).item()
    peak_hit = (totals["peak_hit_num"] / totals["peak_hit_den"].clamp_min(1.0)).item()
    foreground = (totals["foreground_score_sum"] / totals["foreground_score_count"].clamp_min(1.0)).item()
    background = (totals["background_score_sum"] / totals["background_score_count"].clamp_min(1.0)).item()
    predicted_area = (totals["predicted_area_sum"] / totals["batch_count"].clamp_min(1.0)).item()
    print(
        f"{split}_attention_iou={iou:.4f} "
        f"{split}_attention_peak_hit={peak_hit:.4f} "
        f"{split}_attention_fg={foreground:.4f} "
        f"{split}_attention_bg={background:.4f} "
        f"{split}_attention_pred_area={predicted_area:.4f} "
        f"attention_hw={int(totals['height'].item() / totals['batch_count'].item())}x"
        f"{int(totals['width'].item() / totals['batch_count'].item())}"
    )


def evaluate_attention_alignment(args, model, loader, dataset, device, split):
    if not args.attention_report_batches and not args.attention_panel_examples:
        return
    if args.attention_supervision_target != "silhouette":
        print(f"{split}_attention_alignment=skipped target={args.attention_supervision_target}")
        return

    model.eval()
    non_blocking = args.pin_memory and is_cuda_device(device)
    totals = None
    layer_totals = None
    panel_saved = False
    max_batches = args.attention_report_batches or len(loader)
    with torch.no_grad():
        for batch_idx, (video, labels) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            video = video.to(device, non_blocking=non_blocking)
            task_info = _move_labels(labels, device, non_blocking=non_blocking)
            model((video, task_info), return_logits_only=True)
            attention = getattr(model, "last_attention_maps", None)
            if attention is None:
                continue
            totals = _accumulate_attention_alignment(totals, attention, labels, dataset)
            attention_layers = getattr(model, "last_attention_maps_by_layer", None)
            if attention_layers:
                if layer_totals is None:
                    layer_totals = [None for _ in attention_layers]
                for layer_idx, attention_layer in enumerate(attention_layers):
                    layer_totals[layer_idx] = _accumulate_attention_alignment(
                        layer_totals[layer_idx],
                        attention_layer,
                        labels,
                        dataset,
                    )
            if args.attention_panel_examples and not panel_saved:
                checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
                output_path = checkpoint_dir / f"{split}_attention_panel.png"
                _save_attention_alignment_panel(args, video, labels, attention, dataset, output_path)
                print(f"{split}_attention_panel={output_path}")
                panel_saved = True

    if totals is None:
        print(f"{split}_attention_alignment=skipped no_attention_maps")
        return
    _print_attention_alignment_totals(totals, split)
    if layer_totals is not None:
        for layer_idx, totals_for_layer in enumerate(layer_totals):
            if totals_for_layer is not None:
                _print_attention_alignment_totals(totals_for_layer, f"{split}_attention_layer{layer_idx}")


def _move_labels(labels, device, non_blocking):
    if isinstance(labels, dict):
        labels = labels["task_info"]
    return labels.to(device, non_blocking=non_blocking)


def _snapshot_named_params(named_params):
    return {
        name: param.detach().cpu().clone()
        for name, param in named_params
    }


def _max_abs_delta(snapshot, named_params):
    max_delta = 0.0
    for name, param in named_params:
        before = snapshot[name].to(device=param.device, dtype=param.dtype)
        delta = (param.detach() - before).abs().max().item()
        max_delta = max(max_delta, delta)
    return max_delta


def _release_memory(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    elif torch.device(device).type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _mps_memory_stats(device):
    if torch.device(device).type != "mps" or not hasattr(torch, "mps"):
        return {}
    stats = {}
    if hasattr(torch.mps, "current_allocated_memory"):
        stats["MPS/memory_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
    if hasattr(torch.mps, "driver_allocated_memory"):
        stats["MPS/driver_allocated_gb"] = torch.mps.driver_allocated_memory() / 1e9
    if hasattr(torch.mps, "recommended_max_memory"):
        stats["MPS/recommended_max_memory_gb"] = torch.mps.recommended_max_memory() / 1e9
    return stats


def _prepare_attention_dataset(args, split="train", num_sequences=None):
    output_size = _normalize_output_size(args.spritevid_output_size)
    base_output_size = _normalize_output_size(args.attention_base_output_size)
    normalize, mean, std = _attention_normalization_args(args, output_size)
    return MovingAnimalAttentionDataset(
        data_dir=args.data_input_dir,
        split=split,
        task=args.attention_task,
        tasks=_parse_task_values(args.attention_tasks),
        output_size=output_size,
        base_output_size=base_output_size,
        scale_pixel_parameters=args.attention_scale_pixel_parameters,
        seq_len=args.seq_len,
        num_sequences=args.num_sequences if num_sequences is None else num_sequences,
        sprite_img_dir=args.sprite_img_dir,
        max_sprites=args.spritevid_max_sprites,
        seed=args.seed,
        background=args.attention_background,
        device=args.spritevid_device,
        noise_type=args.spritevid_noise_type,
        noise_level=args.spritevid_noise_level,
        training_noise_level=args.attention_training_noise_level,
        object_recognition_noise_level=args.attention_object_recognition_noise_level,
        object_recognition_matches_pretraining=args.attention_object_recognition_matches_pretraining,
        freeze_noise=args.spritevid_frozen_noise,
        noise_on_top=args.sprite_noise_on_top,
        popout_mode=args.popout_mode,
        num_distractors=args.num_distractors,
        crowd_size=args.crowd_size,
        cue_frames=args.cue_frames,
        occluder_count=args.attention_occluder_count,
        occluder_min_size=args.attention_occluder_min_size,
        occluder_max_size=args.attention_occluder_max_size,
        fixation_size=args.attention_fixation_size,
        scale_range=tuple(args.attention_scale_range),
        velocity_range=tuple(args.attention_velocity_range),
        scale_velocity_range=tuple(args.attention_scale_velocity_range),
        slow_speed_range=tuple(args.attention_slow_speed_range),
        fast_speed_range=tuple(args.attention_fast_speed_range),
        angular_speed_range=tuple(args.attention_angular_speed_range),
        slow_rotation_speed_range=tuple(args.attention_slow_rotation_speed_range),
        fast_rotation_speed_range=tuple(args.attention_fast_rotation_speed_range),
        velocity_popout_kind=args.attention_velocity_popout_kind,
        rotation_popout_kind=args.attention_rotation_popout_kind,
        normalize=normalize,
        mean=mean,
        std=std,
        return_metadata=args.attention_return_metadata,
    )


def _prepare_arpl_model(args, dataset, device):
    if args.model_path is None:
        raise ValueError("Specify a pretrained RePL checkpoint with --model_path.")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    output_size = _normalize_output_size(args.spritevid_output_size)
    input_size, _ = get_data_specs(
        dataset="animals",
        target_label="multitask",
        spritevid_num_sprites=args.spritevid_max_sprites,
        spritevid_output_size=output_size,
        flatten_images=args.flatten_images,
    )
    preprocess, postprocess = additional_data_process("animals", args.flatten_enc_output)
    return_full_features = args.return_full_features
    if return_full_features is None:
        return_full_features = args.flatten_enc_output

    state_dict = torch.load(args.model_path, map_location="cpu")
    repl = prepare_model(
        encoder_kind=args.encoder,
        integrator_kind=args.integrator,
        predictor_kind=args.predictor,
        input_size=input_size,
        n_in_channels=1 if args.grayscale else 3,
        enc_dim=args.enc_output_dim,
        ctx_dim=args.ctx_dim,
        ctx_n_layers=args.ctx_n_layers,
        pred_n_hidden_layers=args.pred_n_hidden_layers,
        pred_hidden_dim=args.pred_hidden_dim,
        pred_steps=args.pred_steps,
        dense_prediction=args.dense_prediction,
        prediction_target=args.prediction_target,
        pred_target_dim_override=args.pred_target_dim_override,
        use_bn_enc=args.use_bn,
        enc_n_layers=args.enc_n_layers,
        enc_kernel_size=args.enc_kernel_size,
        enc_stride=args.enc_stride,
        enc_padding=args.enc_padding,
        enc_pool_size=args.enc_pool_size,
        preprocess=preprocess,
        postprocess=postprocess,
        return_full_features=return_full_features,
        flatten_enc_output=args.flatten_enc_output,
        state_dict=state_dict,
    )

    context_dim = _infer_context_dim(repl.integrator, args.ctx_dim)
    predictor_output_dim = _infer_predictor_output_dim(repl.predictor, context_dim)
    first_encoder_shape = _infer_first_encoder_shape(
        repl.encoder,
        output_size,
        n_in_channels=1 if args.grayscale else 3,
    )
    attention_apply_stage = getattr(args, "attention_apply_stage", "first_encoder")
    if attention_apply_stage in ("input", "encoder_layers"):
        attention_shape = (1, output_size[0], output_size[1])
    else:
        attention_shape = _attention_output_shape(first_encoder_shape, args.attention_dims)
    head = ClassificationHead(input_dim=context_dim, num_classes=len(dataset.sprites))
    readout_path = args.attention_readout_path or _default_readout_path(args.model_path)
    _load_frozen_classification_head(head, readout_path, device)
    attention_decoder_kind = getattr(args, "attention_decoder_kind", None)
    if attention_decoder_kind is None:
        attention_decoder_kind = "conv2d" if getattr(args, "attention_conv2d_decoder", False) else "mlp"
    elif getattr(args, "attention_conv2d_decoder", False) and attention_decoder_kind == "mlp":
        attention_decoder_kind = "conv2d"
    if attention_decoder_kind in ("conv2d", "bio"):
        full_encoder_shape = _infer_full_encoder_shape(
            repl.encoder,
            output_size,
            n_in_channels=1 if args.grayscale else 3,
        )
        if full_encoder_shape is None:
            raise ValueError(
                "--attention_conv2d_decoder requires the encoder to return full spatial "
                "features so the predictor output can be reshaped before decoding."
            )
        full_encoder_dim = int(torch.tensor(full_encoder_shape).prod().item())
        if predictor_output_dim != full_encoder_dim:
            raise ValueError(
                "--attention_conv2d_decoder requires predictor output dim to match the "
                f"flattened encoder feature map; got {predictor_output_dim}, expected "
                f"{full_encoder_dim} from {full_encoder_shape}."
            )
        if attention_decoder_kind == "bio":
            if attention_apply_stage == "encoder_layers":
                stage_input_shapes = _infer_encoder_stage_input_shapes(
                    repl.encoder,
                    output_size,
                    n_in_channels=1 if args.grayscale else 3,
                )
                decoder = BioEncoderLayerAttentionDecoder(
                    input_shape=full_encoder_shape,
                    stage_input_shapes=stage_input_shapes,
                    attention_dims=args.attention_dims,
                    use_batch_norm=args.use_bn,
                    skip_hidden_channels=args.attention_bio_hidden_channels,
                    attention_gain=args.attention_bio_gain,
                    **_encoder_all_decoder_args(args),
                )
            elif attention_apply_stage == "input":
                decoder_kwargs = _encoder_all_decoder_args(args)
                skip_shape = (1 if args.grayscale else 3, output_size[0], output_size[1])
                decoder_attention_dims = "spatial"
                decoder = BioInspiredAttentionDecoder(
                    input_shape=full_encoder_shape,
                    skip_shape=skip_shape,
                    output_shape=attention_shape,
                    attention_dims=decoder_attention_dims,
                    use_batch_norm=args.use_bn,
                    skip_hidden_channels=args.attention_bio_hidden_channels,
                    attention_gain=args.attention_bio_gain,
                    **decoder_kwargs,
                )
            else:
                decoder_kwargs = _encoder_tail_decoder_args(args)
                skip_shape = first_encoder_shape
                decoder_attention_dims = args.attention_dims
                decoder = BioInspiredAttentionDecoder(
                    input_shape=full_encoder_shape,
                    skip_shape=skip_shape,
                    output_shape=attention_shape,
                    attention_dims=decoder_attention_dims,
                    use_batch_norm=args.use_bn,
                    skip_hidden_channels=args.attention_bio_hidden_channels,
                    attention_gain=args.attention_bio_gain,
                    **decoder_kwargs,
                )
        else:
            if attention_apply_stage in ("input", "encoder_layers"):
                raise ValueError(
                    "--attention_apply_stage input/encoder_layers is only implemented for --attention_decoder_kind bio."
                )
            decoder = Conv2dAttentionDecoder(
                input_shape=full_encoder_shape,
                output_shape=attention_shape,
                attention_dims=args.attention_dims,
                use_batch_norm=args.use_bn,
                **_encoder_tail_decoder_args(args),
            )
    else:
        decoder = AttentionDecoder(
            input_dim=predictor_output_dim,
            output_dim=None,
            output_shape=attention_shape,
            hidden_dim=args.attention_hidden_dim or context_dim,
            n_hidden_layers=args.attention_decoder_layers,
        )
    model = ARPLmodel(
        encoder=repl.encoder,
        integrator=repl.integrator,
        predictor=repl.predictor,
        head=head,
        decoder=decoder,
        num_tasks=len(TASK_TO_ID),
        preprocess=repl.preprocess,
        postprocess=repl.postprocess,
        freeze_repl=True,
        eval_frozen=True,
        decoder_input_dim=predictor_output_dim,
        use_task_embedding=getattr(args, "attention_use_task_embedding", True),
        class_prompt_value=getattr(args, "attention_class_prompt_value", 10.0),
        attention_update_mode="bio" if attention_decoder_kind == "bio" else "predictor",
        class_feedback_mode=getattr(args, "attention_class_feedback_mode", "prompt"),
        attention_apply_stage=attention_apply_stage,
    )
    model.head.eval()
    for param in model.head.parameters():
        param.requires_grad_(False)
    return model.to(device)


def _prepare_optimizer(args, model):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable ARPL attention parameters were found.")
    if args.optimizer == "adam":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def _expected_trainable_prefixes(args):
    prefixes = ["decoder.", "class_feedback.", "decoder_class_feedback."]
    if getattr(args, "attention_use_task_embedding", True):
        prefixes.append("task_embedding.")
    return tuple(prefixes)


def _validate_trainable_scope(model, args):
    trainable_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    frozen_named = [(name, param) for name, param in model.named_parameters() if not param.requires_grad]
    expected_prefixes = _expected_trainable_prefixes(args)
    unexpected = [
        name
        for name, _ in trainable_named
        if not name.startswith(expected_prefixes)
    ]
    if unexpected:
        raise RuntimeError(
            "Unexpected trainable parameters outside attention decoder/feedback modules: "
            + ", ".join(unexpected)
        )
    for prefix in expected_prefixes:
        if not any(name.startswith(prefix) for name, _ in trainable_named):
            raise RuntimeError(f"No trainable parameters found under expected prefix {prefix!r}.")
    return trainable_named, frozen_named


def _validate_optimizer_scope(optimizer, trainable_named):
    trainable_param_ids = {id(param) for _, param in trainable_named}
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    if optimizer_param_ids != trainable_param_ids:
        raise RuntimeError("Optimizer parameter set does not exactly match trainable ARPL parameters.")


def train(args, device):
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    if args.attention_supervision_weight and not args.attention_return_metadata:
        args.attention_return_metadata = True

    dataset = _prepare_attention_dataset(args)
    val_dataset = _prepare_attention_dataset(args, split="val", num_sequences=args.attention_val_sequences)
    test_dataset = _prepare_attention_dataset(args, split="test", num_sequences=args.attention_test_sequences)
    loader = DataLoader(dataset, **_dataloader_kwargs(args, device, shuffle=True))
    val_loader = DataLoader(val_dataset, **_dataloader_kwargs(args, device, shuffle=False))
    test_loader = DataLoader(test_dataset, **_dataloader_kwargs(args, device, shuffle=False))
    model = _prepare_arpl_model(args, dataset, device)
    trainable_named, frozen_named = _validate_trainable_scope(model, args)
    optimizer = _prepare_optimizer(args, model)
    _validate_optimizer_scope(optimizer, trainable_named)
    criterion = nn.CrossEntropyLoss(reduction="none")

    trainable_snapshot = _snapshot_named_params(trainable_named)
    frozen_snapshot = _snapshot_named_params(frozen_named)

    trainable_count = sum(param.numel() for _, param in trainable_named)
    frozen_count = sum(param.numel() for _, param in frozen_named)
    print(f"trainable_params={trainable_count} frozen_params={frozen_count}")
    print("trainable_modules=" + ", ".join(name for name, _ in trainable_named[:12]))
    if len(trainable_named) > 12:
        print(f"trainable_modules_truncated={len(trainable_named) - 12}")

    non_blocking = args.pin_memory and is_cuda_device(device)
    first_loss = None
    last_loss = None
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        model.encoder_first.eval()
        model.encoder_tail.eval()
        model.integrator.eval()
        model.predictor.eval()
        model.head.eval()

        total_loss = 0.0
        total_correct = 0
        total_items = 0
        total_batches = 0

        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        last_batch_end = time.perf_counter()
        for batch_idx, (video, labels) in enumerate(progress):
            data_time = time.perf_counter() - last_batch_end
            step_start = time.perf_counter()
            video = video.to(device, non_blocking=non_blocking)
            task_info = _move_labels(labels, device, non_blocking=non_blocking)

            logits = model((video, task_info), return_logits_only=True)
            targets, mask = _make_attention_targets(task_info, logits.size(1), args.cue_frames)
            class_loss, loss_mask = _weighted_classification_loss(logits, targets, mask, criterion, args)
            reg_loss = _attention_regularization_loss(model, args)
            supervision_loss = _attention_supervision_loss(model, labels, args, dataset)
            loss = args.attention_class_loss_weight * class_loss
            if reg_loss is not None:
                loss = loss + reg_loss
            if supervision_loss is not None:
                loss = loss + args.attention_supervision_weight * supervision_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
            step_time = time.perf_counter() - step_start

            with torch.no_grad():
                preds = logits[loss_mask].argmax(dim=-1)
                correct = (preds == targets[loss_mask]).sum().item()
                count = int(loss_mask.sum().item())
                loss_value = loss.item()
                class_loss_value = class_loss.item()
                reg_loss_value = 0.0 if reg_loss is None else reg_loss.item()
                supervision_loss_value = 0.0 if supervision_loss is None else supervision_loss.item()
                if first_loss is None:
                    first_loss = loss_value
                last_loss = loss_value
                batch_acc = correct / max(count, 1)
                samples_per_sec = video.shape[0] / step_time if step_time > 0 else 0.0
                total_loss += loss_value
                total_correct += correct
                total_items += count
                total_batches += 1
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    acc=f"{batch_acc:.3f}",
                )
                if not args.nolog:
                    log_variable(epoch + 1, "Attention/epoch", commit=False)
                    log_variable(loss_value, "Attention/train_loss_step", commit=False)
                    log_variable(class_loss_value, "Attention/train_class_loss_step", commit=False)
                    log_variable(reg_loss_value, "Attention/train_reg_loss_step", commit=False)
                    log_variable(supervision_loss_value, "Attention/train_supervision_loss_step", commit=False)
                    log_variable(batch_acc, "Attention/train_acc_step", commit=False)
                    log_variable(count, "Attention/supervised_items_step", commit=False)
                    log_variable(data_time, "Attention/timing_data_sec", commit=False)
                    log_variable(step_time, "Attention/timing_step_sec", commit=False)
                    log_variable(samples_per_sec, "Attention/timing_samples_per_sec", commit=False)
                    for metric_name, metric_value in cuda_memory_stats(device).items():
                        log_variable(metric_value, f"Attention/{metric_name}", commit=False)
                    for metric_name, metric_value in _mps_memory_stats(device).items():
                        log_variable(metric_value, f"Attention/{metric_name}", commit=False)
                    log_variable(global_step, "Attention/global_step", commit=True)

            stop_after_batch = args.max_batches and batch_idx + 1 >= args.max_batches
            del video, labels, task_info, logits, targets, mask, loss_mask, loss
            global_step += 1
            last_batch_end = time.perf_counter()
            if stop_after_batch:
                break

        avg_loss = total_loss / max(total_batches, 1)
        avg_acc = total_correct / max(total_items, 1)
        print(
            f"epoch={epoch + 1} train_loss={avg_loss:.6f} "
            f"train_acc={avg_acc:.4f} batches={total_batches}"
        )
        if not args.nolog:
            log_variable(epoch + 1, "Attention/epoch_summary", commit=False)
            log_variable(avg_loss, "Attention/train_loss_epoch", commit=False)
            log_variable(avg_acc, "Attention/train_acc_epoch", commit=False)
            log_variable(total_batches, "Attention/batches_epoch", commit=True)
        if args.attention_eval_every and (epoch + 1) % args.attention_eval_every == 0:
            evaluate_attention_model(
                args,
                model,
                val_loader,
                device,
                "val",
                _make_attention_targets,
                _move_labels,
                use_attention=True,
            )
        progress.close()
        _release_memory(device)

    if first_loss is None or last_loss is None:
        raise RuntimeError("No training batches were processed.")

    frozen_delta = _max_abs_delta(frozen_snapshot, frozen_named)
    trainable_delta = _max_abs_delta(trainable_snapshot, trainable_named)
    print(
        f"first_batch_loss={first_loss:.6f} last_batch_loss={last_loss:.6f} "
        f"trainable_max_delta={trainable_delta:.6g} frozen_max_delta={frozen_delta:.6g}"
    )
    if not args.nolog:
        log_variable(first_loss, "Attention/first_batch_loss", commit=False)
        log_variable(last_loss, "Attention/last_batch_loss", commit=False)
        log_variable(trainable_delta, "Attention/trainable_max_delta", commit=False)
        log_variable(frozen_delta, "Attention/frozen_max_delta", commit=True)
    if frozen_delta > args.freeze_tolerance:
        raise RuntimeError(
            f"Frozen RePL parameters changed by {frozen_delta:.6g}, "
            f"above --freeze_tolerance {args.freeze_tolerance}."
        )
    if trainable_delta <= 0.0:
        raise RuntimeError("No trainable attention decoder or feedback parameter changed during training.")

    if args.checkpoint_dir is not None:
        checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "attention_model_final.pt")
        torch.save(optimizer.state_dict(), checkpoint_dir / "attention_optimizer_final.pt")
        print(f"saved_checkpoint={checkpoint_dir / 'attention_model_final.pt'}")
    evaluate_attention_model(
        args,
        model,
        test_loader,
        device,
        "test",
        _make_attention_targets,
        _move_labels,
        use_attention=True,
    )
    evaluate_attention_alignment(args, model, test_loader, test_dataset, device, "test")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train ARPL attention decoder/head on MovingAnimalAttentionDataset."
    )
    add_reproducibility_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)

    # Paths and sprite rendering.
    parser.add_argument("--model_path", type=str, required=True, help="Pretrained RePL checkpoint.")
    parser.add_argument("--data_input_dir", type=str, default="datasets")
    parser.add_argument("--sprite_img_dir", type=str, default="animals")
    parser.add_argument("--spritevid_max_sprites", type=int, default=8)
    parser.add_argument("--spritevid_output_size", type=int, nargs="+", default=[64])
    parser.add_argument("--spritevid_device", type=str, default="cpu")

    # Dataset size and loader behavior.
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--num_sequences", type=int, default=16000)
    parser.add_argument("--attention_val_sequences", type=int, default=1024)
    parser.add_argument("--attention_test_sequences", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin-memory", "--pin_memory", dest="pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-workers", "--persistent_workers", dest="persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-factor", "--prefetch_factor", dest="prefetch_factor", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    # Attention dataset noise and normalization.
    parser.add_argument("--spritevid_noise_type", choices=["gaussian", "salt_pepper", "none"], default="gaussian")
    parser.add_argument("--spritevid_noise_level", type=float, default=0.1)
    parser.add_argument("--spritevid_frozen_noise", action="store_true")
    parser.add_argument("--sprite_noise_on_top", action="store_true")
    parser.add_argument("--attention_training_noise_level", type=float, default=0.1)
    parser.add_argument("--attention_object_recognition_noise_level", type=float, default=0.35)
    parser.add_argument("--attention_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_mean", type=float, nargs="+", default=None, help="Scalar or three channel means for attention dataset normalization.")
    parser.add_argument("--attention_std", type=float, nargs="+", default=None, help="Scalar or three channel stds for attention dataset normalization.")

    # Base model input options.
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--flatten_images", action="store_true")
    parser.add_argument("--return_full_features", action=argparse.BooleanOptionalAction, default=None)

    # Attention task composition.
    parser.add_argument("--attention_task", default="mixed")
    parser.add_argument("--attention_tasks", nargs="*", default=None)
    parser.add_argument("--attention_object_recognition_matches_pretraining", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_base_output_size", type=int, nargs="+", default=[64])
    parser.add_argument("--attention_scale_pixel_parameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_background", type=float, default=0.5)
    parser.add_argument("--popout_mode", choices=["class", "rotation", "velocity", "mixed"], default="class")
    parser.add_argument("--num_distractors", type=int, default=2)
    parser.add_argument("--crowd_size", type=int, default=3)
    parser.add_argument("--cue_frames", type=int, default=5)

    # Occluders and fixation cue.
    parser.add_argument("--attention_occluder_count", type=int, default=0)
    parser.add_argument("--attention_occluder_min_size", type=int, default=8)
    parser.add_argument("--attention_occluder_max_size", type=int, default=18)
    parser.add_argument("--attention_fixation_size", type=float, default=4.0)

    # Motion ranges. Defaults match the continuous SpriteVideoDataset where possible.
    parser.add_argument("--attention_scale_range", type=float, nargs=2, default=[0.2, 1.0])
    parser.add_argument("--attention_velocity_range", type=float, nargs=2, default=[-8.0, 8.0], help="Per-axis x/y velocity component range in pixels per frame.")
    parser.add_argument("--attention_scale_velocity_range", type=float, nargs=2, default=[-0.125, 0.125], help="Z/scale velocity range per frame.")
    parser.add_argument("--attention_angular_speed_range", type=float, nargs=2, default=[-30.0, 30.0])

    # Popout ranges. Slow/fast speed ranges are absolute movement or rotation speeds.
    parser.add_argument("--attention_slow_speed_range", type=float, nargs=2, default=[1.0, 3.0])
    parser.add_argument("--attention_fast_speed_range", type=float, nargs=2, default=[7.0, 8.0])
    parser.add_argument("--attention_velocity_popout_kind", choices=["fast", "slow", "mixed"], default="fast")
    parser.add_argument("--attention_slow_rotation_speed_range", type=float, nargs=2, default=[5.0, 15.0])
    parser.add_argument("--attention_fast_rotation_speed_range", type=float, nargs=2, default=[25.0, 30.0])
    parser.add_argument("--attention_rotation_popout_kind", choices=["fast", "slow", "mixed"], default="fast")
    parser.add_argument("--attention_return_metadata", action=argparse.BooleanOptionalAction, default=False)

    # Attention model and evaluation.
    parser.add_argument("--attention_hidden_dim", type=int, default=None)
    parser.add_argument("--attention_decoder_layers", type=int, default=1)
    parser.add_argument("--attention_dims", choices=["features", "spatial", "features+spatial"], default="features+spatial")
    parser.add_argument("--attention_decoder_kind", choices=["mlp", "conv2d", "bio"], default="mlp")
    parser.add_argument("--attention_conv2d_decoder", action="store_true", help="Use Conv2dDecoder mirrored from the Conv2dEncoder tail for attention masks.")
    parser.add_argument("--attention_apply_stage", choices=["first_encoder", "input", "encoder_layers"], default="first_encoder")
    parser.add_argument("--attention_bio_hidden_channels", type=int, default=None)
    parser.add_argument("--attention_bio_gain", type=float, default=0.5)
    parser.add_argument("--attention_area_weight", type=float, default=0.0)
    parser.add_argument("--attention_tv_weight", type=float, default=0.0)
    parser.add_argument("--attention_class_loss_weight", type=float, default=1.0)
    parser.add_argument("--attention_loss_warmup_frames", type=int, default=0)
    parser.add_argument("--attention_loss_end_weight", type=float, default=1.0)
    parser.add_argument("--attention_supervision_weight", type=float, default=0.0)
    parser.add_argument("--attention_supervision_target", choices=["gaussian", "silhouette"], default="gaussian")
    parser.add_argument("--attention_supervision_scale", choices=["modulation", "bio"], default="modulation")
    parser.add_argument("--attention_supervision_sigma", type=float, default=3.0)
    parser.add_argument("--attention_supervision_mse_weight", type=float, default=1.0)
    parser.add_argument("--attention_supervision_spatial_ce_weight", type=float, default=0.0)
    parser.add_argument("--attention_supervision_bce_weight", type=float, default=0.0)
    parser.add_argument("--attention_supervision_bce_threshold", type=float, default=0.0)
    parser.add_argument("--attention_supervision_temperature", type=float, default=0.25)
    parser.add_argument("--attention_supervision_positive_weight", type=float, default=20.0)
    parser.add_argument("--attention_supervise_all_layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_readout_path", type=str, default=None, help="Frozen online sprite-classification readout checkpoint. Defaults to online_ctx_readout.pt beside --model_path.")
    parser.add_argument("--attention_use_task_embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_class_feedback_mode", choices=["prompt", "target", "logits"], default="prompt")
    parser.add_argument("--attention_class_prompt_value", type=float, default=10.0)
    parser.add_argument("--attention_eval_every", type=int, default=5, help="Run attention validation every N epochs; 0 disables periodic validation.")
    parser.add_argument("--attention_report_batches", type=int, default=0, help="Evaluate silhouette-attention alignment on this many final test batches; 0 disables.")
    parser.add_argument("--attention_panel_examples", type=int, default=0, help="Save this many final test examples as frame/attention/target panels; 0 disables.")
    parser.add_argument("--max_batches", type=int, default=0, help="Stop each epoch after this many batches; 0 means full epoch.")
    parser.add_argument("--freeze_tolerance", type=float, default=0.0)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="attention")
    parser.add_argument("--torch_num_threads", type=int, default=0)
    parser.add_argument("--nolog", action="store_true", help="disable experiment logging")
    parser.add_argument("--logger", choices=["tensorboard", "wandb", "none"], default="wandb", help="experiment logger backend")
    parser.add_argument("--wandb_project", type=str, default="HARPL", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Weights & Biases run id to resume")
    parser.add_argument("--wandb_resume", choices=["allow", "must", "never", "auto"], default=None, help="Weights & Biases resume mode")
    parser.add_argument("--wandb_group", type=str, default="attention", help="Weights & Biases run group")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log root directory")

    # Eval of pretrained models
    parser.add_argument("--eval_pretrained_attention_tasks", action="store_true", help="Evaluate the frozen pretrained model and online readout on each attention task with attention disabled, then exit.")
    parser.add_argument("--eval_cross_decode_sprites", action="store_true", help="Evaluate frozen pretrained ctx readouts on generated held-out sprite identities, then exit.")
    parser.add_argument("--cross_decode_sprite_indices", type=int, nargs="+", default=[8, 9])
    parser.add_argument("--cross_decode_sequences", type=int, default=512)
    parser.add_argument("--cross_decode_readout_path", type=str, default=None, help="Readout checkpoint for cross-decoding. Defaults to online_ctx_readout.pt beside --model_path.")
    parser.add_argument("--cross_decode_readout_stats_path", type=str, default=None, help="Optional mean/std stats for offline readouts.")
    
    parser.set_defaults(
        encoder="conv2d",
        use_bn=True,
        enc_n_layers=6,
        enc_kernel_size=[(5, 5)] * 6,
        enc_stride=[(2, 2), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1)],
        enc_padding=[(2, 2)] * 6,
        enc_output_dim=32,
        flatten_enc_output=True,
        integrator="lstm",
        ctx_dim=512,
        predictor="mlp",
        pred_hidden_dim=512,
        pred_steps=1,
        prediction_target="enc",
        dense_prediction=False,
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.spritevid_noise_type == "none":
        args.spritevid_noise_type = None
    seed_everything(args.seed, args.deterministic)
    init_logger(args)
    device = select_device(args.device)
    try:
        if args.eval_pretrained_attention_tasks:
            evaluate_pretrained_attention_tasks(
                args,
                device,
                _prepare_attention_dataset,
                _prepare_arpl_model,
                _dataloader_kwargs,
                _make_attention_targets,
                _move_labels,
            )
        elif args.eval_cross_decode_sprites:
            evaluate_cross_decode_sprites(
                args,
                device,
                _prepare_attention_dataset,
                _prepare_arpl_model,
                _default_readout_path,
                _normalize_output_size,
            )
        else:
            train(args, device)
    finally:
        close_logger()


if __name__ == "__main__":
    main()
