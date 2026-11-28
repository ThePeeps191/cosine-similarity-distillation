"""Utility helpers: seeding, device memory cleanup, parameter counting, and exact byte formatting."""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, PyTorch (CPU/CUDA/MPS).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def empty_cache(device: torch.device | None = None) -> None:
    """Safely clear accelerator cache on CUDA or Apple MPS."""
    dev_type = device.type if device is not None else None
    if (dev_type is None or dev_type == "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (dev_type is None or dev_type == "mps") and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def sync_device(device: torch.device) -> None:
    """Synchronize accelerator for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_bytes(model: nn.Module) -> int:
    """Exact float32 parameter storage in bytes (includes both trainable and frozen params)."""
    return sum(p.numel() for p in model.parameters()) * 4


def get_model_size_mb(model: nn.Module) -> float:
    """Approximate size of model parameters in binary megabytes (MiB, float32)."""
    return get_model_size_bytes(model) / (1024 * 1024)


def format_bytes(size_bytes: int | float) -> str:
    """Format an exact byte count using binary KiB/MiB units (consistent with PyTorch)."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{int(size_bytes)} B"


def get_tensor_size(tensor: torch.Tensor) -> str:
    """Return a human-readable size string (MB or KB) for a tensor."""
    size_bytes = tensor.numel() * tensor.element_size()
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / 1024:.2f} KB"
