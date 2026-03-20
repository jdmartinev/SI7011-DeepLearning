"""CIFAR-10 dataset creation and DataLoader factory."""
from __future__ import annotations

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from src.data.transforms import get_inference_transform, get_augmented_transform

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_dataloaders(
    root: str = "data/",
    val_split: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 123,
    augment_train: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader).

    The validation set is carved out of the training set using a fixed
    random seed for reproducibility across runs.
    """
    train_transform = get_augmented_transform() if augment_train else get_inference_transform()
    eval_transform  = get_inference_transform()

    train_full = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    # Separate val split uses eval transforms — no augmentation
    val_full = torchvision.datasets.CIFAR10(
        root=root, train=True, download=False, transform=eval_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=eval_transform
    )

    n_val = int(len(train_full) * val_split)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_full), generator=generator)

    train_dataset = Subset(train_full, indices[n_val:])
    val_dataset   = Subset(val_full,   indices[:n_val])

    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
