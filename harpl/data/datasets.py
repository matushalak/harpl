import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class ImageSequencesDataset(Dataset):
    """Base class for generated image sequence datasets."""

    valid_splits = ["train", "test"]

    def __init__(
        self,
        root,
        split="train",
        download=False,
        transform=None,
        seq_type="triplets",
        seq_len=64,
        num_sequences=10000,
        num_classes=10,
        inter_trial_interval=0,
    ):
        super().__init__()
        if split not in self.valid_splits:
            raise ValueError(f"Split {split} not valid. Use one of {self.valid_splits}")
        if seq_type != "triplets":
            raise ValueError("HARPL keeps only the MNIST triplets sequence type")

        self.root = os.fspath(root)
        self.split = split
        self.seq_type = seq_type
        self.seqlen = seq_len
        self.num_sequences = num_sequences
        self.num_classes = num_classes
        self.transform = transform
        self.inter_trial_interval = inter_trial_interval

        self._load_base_dataset(download=download)
        self._generate_sequences()
        self._sample_images_for_sequences()

    def _load_base_dataset(self, download):
        raise NotImplementedError

    def _sample_images_for_sequences(self):
        raise NotImplementedError

    def _generate_sequences(self):
        self.dense_labels = self._generate_triplet_sequences()
        self.labels = None
        self.sequence_sample_labels = self.dense_labels[:, :, 0]
        if self.dense_labels is not None and len(self.dense_labels.shape) == 2:
            self.dense_labels = self.dense_labels.unsqueeze(-1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, n):
        x = self.data[n].unsqueeze(1)
        y = self.labels[n] if self.labels is not None else torch.tensor([])
        dense_y = self.dense_labels[n] if self.dense_labels is not None else torch.tensor([])
        return x, (y, dense_y, torch.tensor([]), torch.tensor([]), torch.tensor([]))

    def _generate_triplet_sequences(self, n=None):
        if n is None:
            n = self.num_sequences

        dense_labels = torch.zeros((n, self.seqlen, 3), dtype=torch.long)
        clusters = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        for i in range(n):
            pos = 0
            current_cluster_idx = np.random.randint(0, 3)

            while pos < self.seqlen:
                if pos > 0 and np.random.rand() < 0.2:
                    choices = [idx for idx in range(3) if idx != current_cluster_idx]
                    current_cluster_idx = np.random.choice(choices)
                    dense_labels[i, pos, 0] = 0
                    dense_labels[i, pos, 1] = 0
                    dense_labels[i, pos, 2] = -1
                    pos += 1
                    if pos >= self.seqlen:
                        break

                cluster = clusters[current_cluster_idx]
                digits_perm = np.random.permutation(cluster)
                permutations = [
                    [cluster[0], cluster[1], cluster[2]],
                    [cluster[0], cluster[2], cluster[1]],
                    [cluster[1], cluster[0], cluster[2]],
                    [cluster[1], cluster[2], cluster[0]],
                    [cluster[2], cluster[0], cluster[1]],
                    [cluster[2], cluster[1], cluster[0]],
                ]
                perm_type = next(
                    (idx for idx, perm in enumerate(permutations) if np.array_equal(digits_perm, perm)),
                    0,
                )
                triplet_type = current_cluster_idx * 6 + perm_type

                for k in range(3):
                    if pos >= self.seqlen:
                        break
                    digit = digits_perm[k]
                    dense_labels[i, pos, 0] = digit
                    dense_labels[i, pos, 1] = current_cluster_idx + 1
                    dense_labels[i, pos, 2] = triplet_type if k == 1 else -1
                    pos += 1

        return dense_labels


class MNISTSequencesDataset(ImageSequencesDataset):
    """MNIST triplet sequence dataset."""

    DATASET_DIR = "mnist"

    def _load_base_dataset(self, download):
        if self.split == "train":
            dataset = MNIST(root=os.path.join(self.root, self.DATASET_DIR), train=True, download=download)
        elif self.split == "test":
            dataset = MNIST(root=os.path.join(self.root, self.DATASET_DIR), train=False, download=download)
        else:
            raise ValueError(f"Split {self.split} not valid. Use one of {self.valid_splits}")

        self.dataset = dataset
        self.img_labels = dataset.targets.long()
        self.imgs = dataset.data.float() / 255.0
        self.imgs = (self.imgs - 0.1307) / 0.3081
        self.imgs_by_class = [self.imgs[self.img_labels == i] for i in range(10)]

    def _sample_images_for_sequences(self):
        h, w = 28, 28
        self.data = torch.zeros((self.num_sequences, self.seqlen, h, w), dtype=torch.float32)

        for i in range(self.num_sequences):
            for j in range(self.seqlen):
                class_id = self.sequence_sample_labels[i, j].item()
                if class_id == -1:
                    continue
                img_idx = np.random.randint(self.imgs_by_class[class_id].shape[0])
                self.data[i, j] = self.imgs_by_class[class_id][img_idx]
