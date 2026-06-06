import torch
import torchvision.transforms as transforms

from harpl.data.datasets import MNISTSequencesDataset
from harpl.data.synthetic_sprites_dataset import SpriteVideoDataset
from harpl.data.utils import create_validation_sampler


class ImageDataLoader:
    """Generic image DataLoader wrapper."""

    def __init__(
        self,
        dataset,
        data_dir,
        grayscale=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        augmentations=None,
        **kwargs,
    ):
        self.dataset = dataset
        self.data_dir = data_dir
        self.grayscale = grayscale
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.augmentations = augmentations or {}
        self._dataset_kwargs = kwargs
        self._train_subset = "train"
        self._val_subset = "train"
        self._test_subset = "test"

    def get_transforms(self, eval=False):
        transforms_list = []
        if self.augmentations.get("randcrop") and not eval:
            transforms_list.append(transforms.RandomCrop(self.augmentations["randcrop"]))
        if self.augmentations.get("randcrop") and eval:
            transforms_list.append(transforms.CenterCrop(self.augmentations["randcrop"]))
        if self.augmentations.get("grayscale"):
            transforms_list.append(transforms.Grayscale())
        transforms_list.append(transforms.ToTensor())
        return transforms.Compose(transforms_list)

    def _make_datasets(self, splits):
        return {
            split: self.dataset(
                self.data_dir,
                split=split,
                download=True,
                transform=self.get_transforms(eval=not split.startswith("train")),
                **self._dataset_kwargs,
            )
            for split in splits
        }

    def _get_loader(self, split, batch_size, shuffle, sampler=None):
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle and sampler is None,
            "sampler": sampler,
            "num_workers": self.num_workers,
            "drop_last": True,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return torch.utils.data.DataLoader(self.datasets[split], **loader_kwargs), sampler

    def get_train(self, batch_size, sampler=None):
        return self._get_loader(self._train_subset, batch_size=batch_size, shuffle=True, sampler=sampler)

    def get_validation(self, batch_size, sampler=None):
        return self._get_loader(self._val_subset, batch_size=batch_size, shuffle=False, sampler=sampler)

    def get_test(self, batch_size):
        return self._get_loader(self._test_subset, batch_size=batch_size, shuffle=False)


class ImageSequencesDataLoader(ImageDataLoader):
    """DataLoader for generated MNIST triplet sequences."""

    def __init__(
        self,
        data_dir,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        augmentations=None,
        val_size=0.1,
        seq_type="triplets",
        seq_len=64,
        num_sequences=10000,
        inter_trial_interval=0,
    ):
        super().__init__(
            dataset=MNISTSequencesDataset,
            data_dir=data_dir,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            augmentations=augmentations,
            seq_type=seq_type,
            seq_len=seq_len,
            num_sequences=num_sequences,
            inter_trial_interval=inter_trial_interval,
        )
        self.datasets = self._make_datasets([self._train_subset, self._test_subset])
        self.train_sampler, self.val_sampler = create_validation_sampler(
            len(self.datasets[self._train_subset]), val_size
        )

    def get_train(self, batch_size):
        return super().get_train(batch_size, self.train_sampler)

    def get_validation(self, batch_size):
        return super().get_validation(batch_size, self.val_sampler)


class SpriteVideoDataLoader(ImageDataLoader):
    """DataLoader for synthetic moving-animal sprite videos."""

    def __init__(
        self,
        data_dir,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        augmentations=None,
        val_size=0.1,
        output_size=(64, 64),
        seq_len=32,
        num_sequences=100000,
        seed=42,
        background=0.5,
        max_sprites=16,
        exclude_latent_regions=False,
        discretize_latents=False,
        noise_type=None,
        noise_level=0.1,
        frozen_noise=False,
        noise_on_top=False,
        grid_enabled=False,
        frozen_grid=False,
        sprite_imgs="animals",
        grayscale=False,
        occlude_n_frames=0,
        device="cpu",
    ):
        super().__init__(
            dataset=SpriteVideoDataset,
            data_dir=data_dir,
            grayscale=grayscale,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            augmentations=augmentations,
            output_size=output_size,
            seq_len=seq_len,
            num_sequences=num_sequences,
            seed=seed,
            background=background,
            sprite_img_dir=sprite_imgs,
            discretize_latents=discretize_latents,
            max_sprites=max_sprites,
            noise_type=noise_type,
            noise_intensity=noise_level,
            freeze_noise=frozen_noise,
            noise_on_top=noise_on_top,
            grid_enabled=grid_enabled,
            freeze_grid=frozen_grid,
            occlude_n_frames=occlude_n_frames,
            device=device,
        )
        self.exclude_latent_regions = exclude_latent_regions
        self.grayscale = grayscale
        self.datasets = self._make_datasets([self._train_subset, self._test_subset])
        self.train_sampler, self.val_sampler = create_validation_sampler(
            len(self.datasets[self._train_subset]), val_size
        )

    def _make_datasets(self, splits):
        datasets = {}
        if "train" in splits:
            datasets["train"] = self.dataset(
                self.data_dir,
                split="train",
                download=True,
                transform=self.get_transforms(eval=False),
                exclude_latent_regions=self.exclude_latent_regions,
                grayscale=self.grayscale,
                **self._dataset_kwargs,
            )
            mean, std = datasets["train"].mean, datasets["train"].std
        for split in splits:
            if split == "train":
                continue
            datasets[split] = self.dataset(
                self.data_dir,
                split=split,
                download=True,
                transform=self.get_transforms(eval=not split.startswith("train")),
                mean=mean,
                std=std,
                exclude_latent_regions=False,
                grayscale=self.grayscale,
                **self._dataset_kwargs,
            )
        return datasets

    def get_train(self, batch_size):
        return super().get_train(batch_size, self.train_sampler)

    def get_validation(self, batch_size):
        return super().get_validation(batch_size, self.val_sampler)
