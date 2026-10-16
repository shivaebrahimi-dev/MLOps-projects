"""Fashion-MNIST loading and the seeded train/validation split."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from fashion_mlops.config import TrainConfig
from fashion_mlops.preprocessing import to_model_input


def split_indices(n: int, val_split: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle ``range(n)`` with ``seed`` and return (train indices, validation indices)."""
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator)
    n_val = max(1, round(n * val_split))
    return permutation[n_val:], permutation[:n_val]


def make_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Download Fashion-MNIST if needed and return (train loader, validation loader)."""
    # Imported here so that code which only needs the split logic does not require torchvision.
    from torchvision.datasets import FashionMNIST

    raw = FashionMNIST(root=config.data_dir, train=True, download=True)
    # The whole dataset fits in memory, so it is normalised once up front.
    images = to_model_input(raw.data)
    labels = raw.targets

    train_idx, val_idx = split_indices(len(labels), config.val_split, config.seed)
    train_set = TensorDataset(images[train_idx], labels[train_idx])
    val_set = TensorDataset(images[val_idx], labels[val_idx])

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(val_set, batch_size=1024, num_workers=config.num_workers)
    return train_loader, val_loader
