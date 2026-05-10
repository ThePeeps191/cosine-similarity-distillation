"""Tests for models.py — ResNet architecture shapes and factory functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models import BasicBlock, CIFARResNet, resnet20, resnet56


def test_basicblock_identity_shortcut():
    """BasicBlock with matching dimensions should have Identity shortcut."""
    block = BasicBlock(16, 16, stride=1)
    x = torch.randn(4, 16, 8, 8)
    out = block(x)
    assert out.shape == (4, 16, 8, 8)


def test_basicblock_projection_shortcut():
    """BasicBlock with stride=2 or channel mismatch should project."""
    block = BasicBlock(16, 32, stride=2)
    x = torch.randn(4, 16, 8, 8)
    out = block(x)
    assert out.shape == (4, 32, 4, 4)


def test_basicblock_no_nans():
    """Forward pass should not produce NaNs."""
    block = BasicBlock(16, 16)
    x = torch.randn(4, 16, 8, 8)
    out = block(x)
    assert not torch.isnan(out).any()


def test_resnet20_output_shapes():
    """ResNet‑20 logits shape (B,100), features shape (B,64,8,8)."""
    model = resnet20(num_classes=100)
    x = torch.randn(2, 3, 32, 32)

    logits = model(x)
    assert logits.shape == (2, 100)

    logits, features = model(x, return_features=True)
    assert logits.shape == (2, 100)
    assert features.shape == (2, 64, 8, 8)


def test_resnet56_output_shapes():
    """ResNet‑56 logits shape (B,100), features shape (B,64,8,8)."""
    model = resnet56(num_classes=100)
    x = torch.randn(2, 3, 32, 32)

    logits = model(x)
    assert logits.shape == (2, 100)

    logits, features = model(x, return_features=True)
    assert logits.shape == (2, 100)
    assert features.shape == (2, 64, 8, 8)


def test_resnet20_parameter_count():
    """ResNet‑20 should have roughly 0.27M parameters."""
    model = resnet20()
    n = sum(p.numel() for p in model.parameters())
    assert 260_000 < n < 300_000, f"Unexpected param count: {n}"


def test_resnet56_parameter_count():
    """ResNet‑56 should have roughly 0.86M parameters."""
    model = resnet56()
    n = sum(p.numel() for p in model.parameters())
    assert 840_000 < n < 900_000, f"Unexpected param count: {n}"


def test_resnet20_train_mode():
    """Model should be trainable (gradients flow)."""
    model = resnet20(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    # Check that at least some gradients exist
    grads = [p.grad is not None for p in model.parameters()]
    assert any(grads)


def test_different_num_classes():
    """Models should support custom num_classes."""
    for nc in [10, 50, 100, 200]:
        m = resnet20(num_classes=nc)
        assert m(torch.randn(1, 3, 32, 32)).shape == (1, nc)
