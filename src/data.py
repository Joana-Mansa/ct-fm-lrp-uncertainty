"""NoduleMNIST3D loaders.

NoduleMNIST3D holds 3D chest CT patches centred on lung nodules from LIDC-IDRI,
labelled benign or malignant. We use the 64^3 release because CT-FM downsamples
by a factor of 16 and the 28^3 release is not divisible by it.

The dataset is imbalanced, 863 benign against 295 malignant in the train split,
so every result in this repository reports balanced accuracy and AUC alongside
plain accuracy.
"""

import numpy as np
import torch
from medmnist import NoduleMNIST3D
from torch.utils.data import DataLoader, Dataset

CLASSES = ["benign", "malignant"]
SIZE = 64
ROOT = "data/medmnist"


class Nodules(Dataset):
    """NoduleMNIST3D split as float volumes in [-1, 1].

    `fraction` keeps a stratified subset of the training data, which is how the
    label-efficiency ablation is run. The subset is drawn with a fixed seed so
    the pretrained and from-scratch arms see exactly the same volumes.
    """

    def __init__(self, split, root=ROOT, size=SIZE, fraction=1.0, seed=0):
        ds = NoduleMNIST3D(split=split, download=True, root=root, size=size)
        imgs = ds.imgs.astype(np.float32) / 127.5 - 1.0
        labels = ds.labels.astype(np.int64).squeeze(-1)

        if fraction < 1.0:
            rng = np.random.default_rng(seed)
            keep = []
            for cls in np.unique(labels):
                idx = np.flatnonzero(labels == cls)
                n = max(1, int(round(len(idx) * fraction)))
                keep.append(rng.choice(idx, size=n, replace=False))
            keep = np.sort(np.concatenate(keep))
            imgs, labels = imgs[keep], labels[keep]

        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.from_numpy(self.imgs[i]).unsqueeze(0), int(self.labels[i])


def loader(split, batch_size, shuffle=None, workers=4, fraction=1.0, seed=0,
           root=ROOT, size=SIZE):
    """DataLoader for one split, optionally on a stratified fraction of it."""
    ds = Nodules(split, root=root, size=size, fraction=fraction, seed=seed)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )


def split_counts(split, root=ROOT, size=SIZE):
    """Class counts for the dataset table in the README."""
    ds = Nodules(split, root=root, size=size)
    counts = np.bincount(ds.labels, minlength=2)
    return {CLASSES[i]: int(c) for i, c in enumerate(counts)}
