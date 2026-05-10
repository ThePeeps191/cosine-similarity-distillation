"""Utility helpers: seeding, parameter counting, and tensor-size formatting."""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Approximate size of model parameters in megabytes (float32)."""
    return count_parameters(model) * 4 / (1024 * 1024)


def get_tensor_size(tensor: torch.Tensor) -> str:
    """Return a human-readable size string (MB or KB) for a tensor."""
    size_bytes = tensor.numel() * tensor.element_size()
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / 1024:.2f} KB"
