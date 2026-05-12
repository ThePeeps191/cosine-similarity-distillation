"""CIFAR‑100 data loaders with standard training augmentations."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from config import Config


def _train_transform(config: Config) -> T.Compose:
    return T.Compose([
        T.RandomCrop(config.image_size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(config.cifar_mean, config.cifar_std),
    ])


def _test_transform(config: Config) -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(config.cifar_mean, config.cifar_std),
    ])


def _fingerprint_transform(config: Config) -> T.Compose:
    """Augmentations used during fingerprint generation.

    Must exactly match the student's training augmentations so that
    fingerprint targets are consistent with what the student sees
    during CSD training.  Called once per augmented view per image.
    """
    return T.Compose([
        T.RandomCrop(config.image_size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(config.cifar_mean, config.cifar_std),
    ])


def get_cifar100_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    """Return train and test DataLoaders for CIFAR-100.

    Training set uses augmentations; test set is plain.
    """
    train_set = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=_train_transform(config),
    )
    test_set = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=_test_transform(config),
    )
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=(config.device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=(config.device.type == "cuda"),
    )
    return train_loader, test_loader


class IndexedCIFAR100(Dataset):
    """Wraps CIFAR-100 to also return the dataset index of each sample.

    This enables per‑sample fingerprint lookup during CSD training.
    """

    def __init__(self, root: str, train: bool, transform=None, download: bool = True):
        self.dataset = torchvision.datasets.CIFAR100(
            root=root, train=train, transform=transform, download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        return img, label, idx


def get_indexed_cifar100_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    """Return train loader (with index) and plain test loader for CIFAR-100."""
    train_set = IndexedCIFAR100(
        root="./data", train=True, transform=_train_transform(config),
    )
    test_set = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=_test_transform(config),
    )
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=(config.device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=(config.device.type == "cuda"),
    )
    return train_loader, test_loader
