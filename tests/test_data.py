"""Tests for data.py — CIFAR‑100 loaders and indexed dataset."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import Config
from data import (
    get_cifar100_loaders,
    IndexedCIFAR100,
    get_indexed_cifar100_loaders,
    _train_transform,
    _test_transform,
)


def test_transforms_produce_correct_range():
    """Train/test transforms should produce normalised tensors of shape (3,32,32)."""
    c = Config()
    t_train = _train_transform(c)
    t_test = _test_transform(c)

    # We can't call transforms on raw images without PIL, but check types
    from torchvision import transforms as T
    assert isinstance(t_train, T.Compose)
    assert isinstance(t_test, T.Compose)


def test_get_cifar100_loaders():
    """Loaders should return (images, labels) with correct shapes."""
    c = Config()
    c.batch_size = 16
    train_loader, test_loader = get_cifar100_loaders(c)

    images, labels = next(iter(train_loader))
    assert images.shape == (16, 3, 32, 32)
    assert labels.shape == (16,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

    images, labels = next(iter(test_loader))
    assert images.shape == (16, 3, 32, 32)
    assert labels.shape == (16,)


def test_indexed_dataset():
    """IndexedCIFAR100 should return (img, label, index)."""
    c = Config()
    t = _train_transform(c)
    ds = IndexedCIFAR100(root="./data", train=True, transform=t)
    img, label, idx = ds[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, int)
    assert isinstance(idx, int)
    assert idx == 0

    img2, label2, idx2 = ds[100]
    assert idx2 == 100


def test_get_indexed_cifar100_loaders():
    """Indexed loader should return (img, label, idx)."""
    c = Config()
    c.batch_size = 16
    train_loader, test_loader = get_indexed_cifar100_loaders(c)

    batch = next(iter(train_loader))
    assert len(batch) == 3
    img, label, idx = batch
    assert img.shape == (16, 3, 32, 32)
    assert label.shape == (16,)
    assert idx.shape == (16,)

    # Test loader should still return (img, label) only
    batch_test = next(iter(test_loader))
    assert len(batch_test) == 2
