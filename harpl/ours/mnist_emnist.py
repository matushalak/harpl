"""Loading and class-subsetting helpers for MNIST (digits) and EMNIST (letters).

MNIST provides handwritten digits 0-9. For letters we use EMNIST's "balanced"
split rather than "letters": the "letters" split merges every letter's
upper- and lower-case forms into a single class (26 classes total), while
"balanced" keeps upper/lower distinct wherever they actually look different
(e.g. 'A' vs 'a'), only merging the case pairs that look identical (e.g.
'C'/'c', 'O'/'o') -- 37 letter classes total. `load_emnist` returns just the
letter portion of "balanced" (dropping its digit classes), so its label space
never collides with MNIST's.

EMNIST images are stored transposed relative to MNIST, so `load_emnist`
corrects the orientation before returning.
"""

import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST, MNIST

DIGIT_CLASSES = list(range(10))

# torchvision's EMNIST "balanced" split class order: 10 digits (label ids
# 0-9), then 26 uppercase letters, then the 11 lowercase letters that look
# distinct from their uppercase counterpart. Letters not listed in lowercase
# here (c, i, j, k, l, m, o, p, s, u, v, w, x, y, z) look the same in both
# cases and only have an uppercase class.
LETTER_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t",
]
_LETTER_TO_LABEL = {letter: idx + len(DIGIT_CLASSES) for idx, letter in enumerate(LETTER_CLASSES)}


def load_mnist(root, train=True, download=False):
    """Load MNIST digits as (images, labels) tensors: images (N, 28, 28) in [0, 1], labels 0-9."""
    dataset = MNIST(root=os.path.join(root, "mnist"), train=train, download=download)
    images = dataset.data.float() / 255.0
    labels = dataset.targets.long()
    return images, labels


def load_emnist(root, train=True, download=False):
    """Load EMNIST letters as (images, labels) tensors: images (N, 28, 28) in [0, 1].

    Uses the "balanced" split and drops its digit classes, keeping only the
    37 letter classes (label ids 10-46; see LETTER_CLASSES / letters_to_labels).
    """
    dataset = EMNIST(root=os.path.join(root, "emnist"), split="balanced", train=train, download=download)
    images = dataset.data.float().transpose(-2, -1) / 255.0
    labels = dataset.targets.long()
    letter_mask = labels >= len(DIGIT_CLASSES)
    return images[letter_mask], labels[letter_mask]


def letters_to_labels(letters):
    """Convert letters (case matters) to their EMNIST 'balanced' split label ids.

    Case-ambiguous letters (c, i, j, k, l, m, o, p, s, u, v, w, x, y, z) only
    have an uppercase class in the balanced split, so either case works for
    those; for the other letters, 'A' and 'a' resolve to distinct classes.
    """
    labels = []
    for letter in letters:
        if letter in _LETTER_TO_LABEL:
            labels.append(_LETTER_TO_LABEL[letter])
        elif letter.upper() in _LETTER_TO_LABEL:
            labels.append(_LETTER_TO_LABEL[letter.upper()])
        else:
            raise KeyError(f"Unknown EMNIST balanced letter class: {letter!r}")
    return labels


class ClassSubsetDataset(Dataset):
    """Filters an (images, labels) pair down to the given classes.

    Labels are remapped to 0..len(classes)-1 following the order of `classes`,
    so the same class list always produces the same contiguous label space
    regardless of the original label ids (digit ids 0-9 or letter ids 10-46).
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
