"""Tests for utils.py — seeding and parameter counting."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import numpy as np
import torch
import torch.nn as nn
from utils import set_seed, count_parameters, get_model_size_mb, get_tensor_size


def test_set_seed_reproducibility():
    """Two calls with the same seed should produce the same random numbers."""
    set_seed(123)
    a = random.random()
    b = np.random.rand()
    c = torch.rand(1).item()

    set_seed(123)
    a2 = random.random()
    b2 = np.random.rand()
    c2 = torch.rand(1).item()

    assert a == a2
    assert b == b2
    assert c == c2


def test_count_parameters():
    """count_parameters should match manual computation."""
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
    model = TinyNet()
    n = count_parameters(model)
    # Linear(10,5): weights 10*5=50 + bias 5 = 55
    assert n == 55, f"Expected 55, got {n}"


def test_get_model_size_mb():
    """get_model_size_mb should be close to params*4/(1024**2)."""
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 10)
    model = TinyNet()
    size_mb = get_model_size_mb(model)
    expected = 1010 * 4 / (1024 * 1024)  # 100*10 weights + 10 biases = 1010
    assert abs(size_mb - expected) < 0.001


def test_get_tensor_size():
    """get_tensor_size should format MB or KB appropriately."""
    small = torch.zeros(100, dtype=torch.float32)
    s1 = get_tensor_size(small)
    assert "KB" in s1

    large = torch.zeros(300_000, dtype=torch.float32)  # ~1.14 MB
    s2 = get_tensor_size(large)
    assert "MB" in s2
