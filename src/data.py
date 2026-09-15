"""NoduleMNIST3D 64-cube loaders with rating-derived binary labels.

The short class names follow MedMNIST; they do not imply biopsy confirmation.
Training subsets are stratified and shared between matched comparison arms.
"""

import os
from pathlib import Path

import numpy as np
import torch
from medmnist import NoduleMNIST3D
from torch.utils.data import DataLoader, Dataset

CLASSES = ["benign", "malignant"]
SIZE = 64
ROOT = os.environ.get("MEDMNIST_ROOT", str(Path(__file__).resolve().parents[1] / "data" / "medmnist"))


class Nodules(Dataset):
    """NoduleMNIST3D split as float volumes in [-1, 1].

    `fraction` keeps a stratified subset of the training data, which is how the
    label-efficiency ablation is run. The subset is drawn with a fixed seed so
    the pretrained and from-scratch arms see exactly the same volumes.
    """

    def __init__(self, split, root=ROOT, size=SIZE, fraction=1.0, seed=0):
        Path(root).mkdir(parents=True, exist_ok=True)
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
