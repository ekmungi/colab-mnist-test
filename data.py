"""Data helpers: reproducible non-overlapping splits, MNIST loading, and the one place that decides the transforms."""
import torch
from torch.utils.data import Subset

import augment


def split_indices(n_items, train_size, val_size, seed):
    """Returns (train_indices, val_indices) as lists drawn from range(n_items) with no overlap, repeatable for a seed."""
    if train_size + val_size > n_items:
        raise ValueError(f"{train_size} + {val_size} items requested but only {n_items} exist")
    order = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[:train_size], order[train_size:train_size + val_size]


def load_mnist(data_dir, augment_train=False):
    """Downloads MNIST into data_dir if needed and returns (train_full, val_source, test) datasets.

    val_source holds the same images as train_full without augmentation, so validation never sees random changes.
    """
    from torchvision import datasets  # imported here so the splitting helper works without torchvision

    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=augment.build_transform(augment_train))
    val_source = datasets.MNIST(data_dir, train=True, download=True, transform=augment.build_transform(False))
    test = datasets.MNIST(data_dir, train=False, download=True, transform=augment.build_transform(False))
    return train_full, val_source, test


def make_sets(train_full, val_source, train_size, val_size, seed):
    """Returns (train_set, val_set, train_indices, val_indices): two Subsets over disjoint items."""
    train_idx, val_idx = split_indices(len(train_full), train_size, val_size, seed)
    return Subset(train_full, train_idx), Subset(val_source, val_idx), train_idx, val_idx
