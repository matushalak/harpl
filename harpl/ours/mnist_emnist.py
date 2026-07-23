"""Loading and class-subsetting helpers for MNIST (digits) and EMNIST (letters).

MNIST provides handwritten digits 0-9. EMNIST's "letters" split reuses the
same image format for handwritten letters a-z (26 classes, labeled 1-26).
EMNIST images are stored transposed relative to MNIST, so `load_emnist`
corrects the orientation before returning.
"""

import os
import string

import torch
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST, MNIST

DIGIT_CLASSES = list(range(10))
LETTER_CLASSES = list(string.ascii_lowercase)
_LETTER_TO_LABEL = {letter: idx + 1 for idx, letter in enumerate(LETTER_CLASSES)}


def load_mnist(root, train=True, download=False):
    """Load MNIST digits as (images, labels) tensors: images (N, 28, 28) in [0, 1], labels 0-9."""
    dataset = MNIST(root=os.path.join(root, "mnist"), train=train, download=download)
    images = dataset.data.float() / 255.0
    labels = dataset.targets.long()
    return images, labels


def load_emnist(root, train=True, download=False):
    """Load EMNIST letters as (images, labels) tensors: images (N, 28, 28) in [0, 1], labels 1-26 (a=1)."""
    dataset = EMNIST(root=os.path.join(root, "emnist"), split="letters", train=train, download=download)
    images = dataset.data.float().transpose(-2, -1) / 255.0
    labels = dataset.targets.long()
    return images, labels


def letters_to_labels(letters):
    """Convert single letters (any case) to their EMNIST 'letters' split label ids."""
    return [_LETTER_TO_LABEL[letter.lower()] for letter in letters]


class ClassSubsetDataset(Dataset):
    """Filters an (images, labels) pair down to the given classes.

    Labels are remapped to 0..len(classes)-1 following the order of `classes`,
    so the same class list always produces the same contiguous label space
    regardless of the original label ids (digit ids 0-9 or letter ids 1-26).
    """

    def __init__(self, images, labels, classes, transform=None):
        super().__init__()
        self.classes = list(classes)
        self.transform = transform

        mask = torch.zeros(labels.shape[0], dtype=torch.bool)
        for cls in self.classes:
            mask |= labels == cls
        label_map = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = images[mask]
        self.labels = torch.tensor([label_map[int(label)] for label in labels[mask]], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]
