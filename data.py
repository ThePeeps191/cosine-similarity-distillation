"""CIFAR-100 data loaders: standard PyTorch DataLoaders + fast vectorized GPU/tensor loaders.

The fast tensor loader pre-loads CIFAR-100 (~153 MB in uint8) into memory once and
performs exact RandomCrop(32, padding=4) + RandomHorizontalFlip(p=0.5) + Normalize
as vectorized tensor ops directly on CPU/MPS/CUDA, eliminating the Python/PIL
DataLoader bottleneck while preserving exact CIFAR-100 augmentation semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator
import numpy as np
import torch
import torch.nn.functional as F
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
    """Augmentations used during fingerprint generation (matches student training)."""
    return T.Compose([
        T.RandomCrop(config.image_size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(config.cifar_mean, config.cifar_std),
    ])


def get_cifar100_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    """Return standard PyTorch train and test DataLoaders for CIFAR-100."""
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
    """Wraps CIFAR-100 to also return the dataset index of each sample."""

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


def augment_cifar_batch(
    images: torch.Tensor,
    pad: int = 4,
    flip_prob: float = 0.5,
) -> torch.Tensor:
    """Vectorized RandomCrop(32, padding=4, value=0) + RandomHorizontalFlip on a batch.

    Operates directly on (B, C, H, W) tensors on any device (CPU/MPS/CUDA).
    """
    B, C, H, W = images.shape
    device = images.device
    padded = F.pad(images, (pad, pad, pad, pad), mode="constant", value=0.0)
    # Per-image random crop coordinates
    top = torch.randint(0, 2 * pad + 1, (B,), device=device)
    left = torch.randint(0, 2 * pad + 1, (B,), device=device)

    # Vectorized gather via grid indexing
    h_idx = torch.arange(H, device=device).view(1, H, 1) + top.view(B, 1, 1)   # (B, H, 1)
    w_idx = torch.arange(W, device=device).view(1, 1, W) + left.view(B, 1, 1)  # (B, 1, W)
    b_idx = torch.arange(B, device=device).view(B, 1, 1)
    # padded: (B, C, H+2p, W+2p) -> permute to (B, H+2p, W+2p, C) for fast advanced indexing
    cropped = padded.permute(0, 2, 3, 1)[b_idx, h_idx, w_idx].permute(0, 3, 1, 2).contiguous()

    # Random horizontal flip per image
    if flip_prob > 0.0:
        flip_mask = (torch.rand(B, device=device) < flip_prob).view(B, 1, 1, 1)
        cropped = torch.where(flip_mask, cropped.flip(-1), cropped)
    return cropped


class FastCIFAR100TensorLoader:
    """Zero-overhead in-memory CIFAR-100 batch iterator with vectorized GPU augmentations."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        batch_size: int = 128,
        shuffle: bool = True,
        augment: bool = False,
        return_index: bool = False,
        device: torch.device | None = None,
        original_indices: torch.Tensor | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        # Keep float32 [0,1] on device (50k * 3 * 32 * 32 * 4B = 586 MB; tiny on 24GB M6!)
        self.images = images.to(self.device)
        self.labels = labels.to(self.device)
        self.original_indices = (
            original_indices.to(self.device)
            if original_indices is not None
            else torch.arange(len(images), device=self.device, dtype=torch.long)
        )
        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.return_index = return_index
        self.n = len(images)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        if self.shuffle:
            perm = torch.randperm(self.n, device=self.device)
        else:
            perm = torch.arange(self.n, device=self.device)
        for start in range(0, self.n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            x = self.images[idx]
            if self.augment:
                # Pad with 0 in [0,1] space (matches PIL RandomCrop padding=4 before Normalize)
                x = augment_cifar_batch(x, pad=4, flip_prob=0.5)
            x = (x - self.mean) / self.std
            y = self.labels[idx]
            if self.return_index:
                yield x, y, self.original_indices[idx]
            else:
                yield x, y


_CIFAR100_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _load_raw_cifar100(root: str = "./data", train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    key = f"{root}_{train}"
    if key not in _CIFAR100_CACHE:
        ds = torchvision.datasets.CIFAR100(root=root, train=train, download=True)
        # ds.data: (N, 32, 32, 3) uint8 -> (N, 3, 32, 32) float32 in [0, 1]
        imgs = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div_(255.0).contiguous()
        labels = torch.tensor(ds.targets, dtype=torch.long)
        _CIFAR100_CACHE[key] = (imgs, labels)
    return _CIFAR100_CACHE[key]


def get_fast_cifar100_loaders(
    config: Config,
    return_index: bool = False,
    val_fraction: float = 0.0,
    split_seed: int = 0,
) -> tuple[FastCIFAR100TensorLoader, FastCIFAR100TensorLoader] | tuple[FastCIFAR100TensorLoader, FastCIFAR100TensorLoader, FastCIFAR100TensorLoader]:
    """Return fast in-memory CIFAR-100 loaders.

    If ``val_fraction > 0`` (e.g. 0.1 for 45k train / 5k val), returns
    ``(train_loader, val_loader, test_loader)`` using a deterministic stratified split
    so hyperparameters can be tuned without touching the test set.
    """
    train_imgs, train_labels = _load_raw_cifar100("./data", train=True)
    test_imgs, test_labels = _load_raw_cifar100("./data", train=False)

    test_loader = FastCIFAR100TensorLoader(
        test_imgs, test_labels, config.cifar_mean, config.cifar_std,
        batch_size=256, shuffle=False, augment=False, return_index=False,
        device=config.device,
    )

    if val_fraction > 0.0:
        rng = np.random.RandomState(split_seed)
        train_idx_list, val_idx_list = [], []
        labels_np = train_labels.numpy()
        for c in range(config.num_classes):
            c_idx = np.where(labels_np == c)[0]
            rng.shuffle(c_idx)
            n_val = int(round(len(c_idx) * val_fraction))
            val_idx_list.append(c_idx[:n_val])
            train_idx_list.append(c_idx[n_val:])
        tr_idx = torch.from_numpy(np.concatenate(train_idx_list)).long()
        va_idx = torch.from_numpy(np.concatenate(val_idx_list)).long()

        train_loader = FastCIFAR100TensorLoader(
            train_imgs[tr_idx], train_labels[tr_idx], config.cifar_mean, config.cifar_std,
            batch_size=config.batch_size, shuffle=True, augment=True,
            return_index=return_index, device=config.device, original_indices=tr_idx,
        )
        val_loader = FastCIFAR100TensorLoader(
            train_imgs[va_idx], train_labels[va_idx], config.cifar_mean, config.cifar_std,
            batch_size=256, shuffle=False, augment=False,
            return_index=False, device=config.device, original_indices=va_idx,
        )
        return train_loader, val_loader, test_loader

    train_loader = FastCIFAR100TensorLoader(
        train_imgs, train_labels, config.cifar_mean, config.cifar_std,
        batch_size=config.batch_size, shuffle=True, augment=True,
        return_index=return_index, device=config.device,
    )
    return train_loader, test_loader
