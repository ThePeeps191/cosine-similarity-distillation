"""Generate cosine-similarity fingerprints for the entire CIFAR‑100 training set.

This module implements the core CSD pre‑computation:
    1. Load the trained teacher.
    2. Create a frozen random reference matrix **R**.
    3. Forward every training image, pool layer‑3 features, L2‑normalise,
       and compute φ = f_norm @ R_norm.
    4. Save per‑sample and per‑class fingerprints, then discard the teacher.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import Config
from models import resnet56
from utils import get_tensor_size, get_model_size_mb


@torch.no_grad()
def generate_fingerprints(
    config: Config,
    r_override: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre‑compute teacher fingerprints for every training sample.

    Args:
        config: Experiment configuration.
        r_override: If given, override ``config.r`` (used in ablation studies).

    Returns:
        Tuple of ``(R_norm, all_fingerprints, class_fingerprints, labels)``.
    """
    r = r_override if r_override is not None else config.r
    os.makedirs(config.results_dir, exist_ok=True)
    device = config.device

    # ------------------------------------------------------------------
    # 1. Load teacher
    # ------------------------------------------------------------------
    if not os.path.exists(config.teacher_ckpt):
        raise FileNotFoundError(
            f"Teacher checkpoint not found at {config.teacher_ckpt}. "
            "Run train_teacher() first."
        )
    teacher = resnet56(num_classes=config.num_classes).to(device)
    teacher.load_state_dict(torch.load(config.teacher_ckpt, map_location=device))
    teacher.eval()

    # ------------------------------------------------------------------
    # 2. Create frozen random reference matrix R (C x r), columns L2‑normed
    # ------------------------------------------------------------------
    R = torch.randn(config.C, r, device=device)
    R_norm = F.normalize(R, p=2, dim=0)  # unit‑length columns
    torch.save(R_norm.cpu(), config.random_matrix_path)

    # ------------------------------------------------------------------
    # 3. Collect fingerprints (non‑shuffled loader → indices match dataset)
    # ------------------------------------------------------------------
    train_set = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize(config.cifar_mean, config.cifar_std),
        ]),
    )
    ordered_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    all_fps: list[torch.Tensor] = []
    all_labels: list[int] = []

    for images, labels in ordered_loader:
        images = images.to(device)
        _, features = teacher(images, return_features=True)  # (B, 64, 8, 8)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 64)
        f_norm = F.normalize(pooled, p=2, dim=1)  # unit vectors
        phi = f_norm @ R_norm  # (B, r) — cosine similarities
        all_fps.append(phi.cpu())
        all_labels.extend(labels.tolist())

    all_fingerprints = torch.cat(all_fps, dim=0)  # (50000, r)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)  # (50000,)

    torch.save(all_fingerprints, config.all_fingerprints_path)

    # ------------------------------------------------------------------
    # 4. Per‑class averaged fingerprints
    # ------------------------------------------------------------------
    class_fps = torch.zeros(config.num_classes, r)
    for c in range(config.num_classes):
        mask = labels_tensor == c
        if mask.sum() > 0:
            class_fps[c] = all_fingerprints[mask].mean(dim=0)
    torch.save(class_fps, config.class_fingerprints_path)

    # ------------------------------------------------------------------
    # 5. Storage statistics
    # ------------------------------------------------------------------
    teacher_mb = get_model_size_mb(teacher)
    all_fps_size = get_tensor_size(all_fingerprints)
    class_fps_size = get_tensor_size(class_fps)

    print(f"\n[Fingerprints] Teacher model size:     {teacher_mb:.2f} MB")
    print(f"[Fingerprints] Per‑sample fingerprints: {all_fps_size}")
    print(f"[Fingerprints] Per‑class fingerprints:  {class_fps_size}")
    print(f"[Fingerprints] Random matrix R:         {get_tensor_size(R_norm.cpu())}")

    # Clean up
    del teacher
    torch.cuda.empty_cache()

    return R_norm.cpu(), all_fingerprints, class_fps, labels_tensor
