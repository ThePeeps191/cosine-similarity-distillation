"""Generate cosine-similarity fingerprints for the entire CIFAR-100 training set.

This module implements the core CSD pre-computation:
    1. Load the trained teacher.
    2. Create frozen random reference matrices:
       - R3 (64 x r) for layer-3 features (abstraction level, existing)
       - R2 (32 x r) for layer-2 features (texture / edge level, multi-layer)
    3. For each training image, create N augmented views (matching student
       augmentations), compute pooled features from layer-2 and layer-3,
       L2-normalise, compute phi = f_norm @ R_norm for each layer, and
       average across views.
    4. Save per-sample and per-class fingerprints for both layers.
    5. Discard the teacher.

Multi-layer fingerprints provide complementary views of the teacher's
representational geometry at different abstraction levels.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader

from config import Config
from data import _fingerprint_transform
from models import resnet56
from utils import get_tensor_size, get_model_size_mb


@torch.no_grad()
def generate_fingerprints(
    config: Config,
    r_override: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Pre-compute teacher fingerprints for every training sample.

    For each training image, generates N augmented views (matching the
    student's training augmentations), computes fingerprints for each
    layer (layer-2 and layer-3 when multi-layer is enabled), averages
    across views, and saves per-sample and per-class fingerprints.

    Args:
        config: Experiment configuration.
        r_override: If given, override ``config.r`` (used in ablation studies).

    Returns:
        Dict with keys ``R2_norm``, ``R3_norm``, ``all_fp_l2``,
        ``all_fp_l3``, ``class_fp_l2``, ``class_fp_l3``, ``labels``.
        When multi-layer is disabled, ``all_fp_l2`` / ``class_fp_l2``
        are None.
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

    multi = config.use_multi_layer

    # ------------------------------------------------------------------
    # 2. Create frozen random reference matrices
    #    R3: (64 x r) for layer-3 features
    #    R2: (32 x r) for layer-2 features (multi-layer only)
    #    Columns are L2-normalised.
    # ------------------------------------------------------------------
    R3 = torch.randn(config.C, r, device=device)
    R3_norm = F.normalize(R3, p=2, dim=0)
    torch.save(R3_norm.cpu(), config.random_matrix_path)

    R2_norm = None
    if multi:
        R2 = torch.randn(32, r, device=device)  # layer2 has 32 channels
        R2_norm = F.normalize(R2, p=2, dim=0)
        torch.save(R2_norm.cpu(), "results/random_matrix_R2.pt")

    # ------------------------------------------------------------------
    # 3. Collect augmentation-averaged fingerprints
    #    Non-shuffled loader so index i matches dataset order.
    # ------------------------------------------------------------------
    aug_transform = _fingerprint_transform(config)
    N_AUG = config.n_augmentations

    train_set = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True,
        transform=T.ToTensor(),
    )
    ordered_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    all_fps_l3: list[torch.Tensor] = []
    all_fps_l2: list[torch.Tensor] = [] if multi else None
    all_labels: list[int] = []

    for images, labels in ordered_loader:
        B = len(images)
        augmented_views = []
        for i in range(B):
            img_pil = TF.to_pil_image(images[i])
            for _ in range(N_AUG):
                augmented_views.append(aug_transform(img_pil))

        augmented_batch = torch.stack(augmented_views).to(device)

        if multi:
            _, l2, l3 = teacher(augmented_batch, return_features="multi")
            pooled_l2 = F.adaptive_avg_pool2d(l2, (1, 1)).squeeze(-1).squeeze(-1)
            pooled_l3 = F.adaptive_avg_pool2d(l3, (1, 1)).squeeze(-1).squeeze(-1)
            f_norm_l2 = F.normalize(pooled_l2, p=2, dim=1)
            f_norm_l3 = F.normalize(pooled_l3, p=2, dim=1)
            phi_l2 = f_norm_l2 @ R2_norm  # (B * N_AUG, r)
            phi_l3 = f_norm_l3 @ R3_norm  # (B * N_AUG, r)
            phi_l2 = phi_l2.view(B, N_AUG, r).mean(dim=1)
            phi_l3 = phi_l3.view(B, N_AUG, r).mean(dim=1)
            all_fps_l2.append(phi_l2.cpu())
            all_fps_l3.append(phi_l3.cpu())
        else:
            _, features = teacher(augmented_batch, return_features=True)
            pooled = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
            f_norm = F.normalize(pooled, p=2, dim=1)
            phi = f_norm @ R3_norm
            phi = phi.view(B, N_AUG, r).mean(dim=1)
            all_fps_l3.append(phi.cpu())

        all_labels.extend(labels.tolist())

    all_fp_l3 = torch.cat(all_fps_l3, dim=0)  # (50000, r)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    torch.save(all_fp_l3, config.all_fingerprints_path)

    if multi:
        all_fp_l2 = torch.cat(all_fps_l2, dim=0)
        torch.save(all_fp_l2, "results/all_fingerprints_l2.pt")
    else:
        all_fp_l2 = None

    # ------------------------------------------------------------------
    # 4. Per-class averaged fingerprints
    # ------------------------------------------------------------------
    class_fp_l3 = torch.zeros(config.num_classes, r)
    class_fp_l2 = torch.zeros(config.num_classes, r) if multi else torch.zeros(0)
    for c in range(config.num_classes):
        mask = labels_tensor == c
        if mask.sum() > 0:
            class_fp_l3[c] = all_fp_l3[mask].mean(dim=0)
            if multi:
                class_fp_l2[c] = all_fp_l2[mask].mean(dim=0)

    torch.save(class_fp_l3, config.class_fingerprints_path)
    if multi:
        torch.save(class_fp_l2, "results/class_fingerprints_l2.pt")

    # ------------------------------------------------------------------
    # 5. Storage statistics
    # ------------------------------------------------------------------
    teacher_mb = get_model_size_mb(teacher)
    all_fps_size = get_tensor_size(all_fp_l3)
    class_fps_size = get_tensor_size(class_fp_l3)

    print(f"\n[Fingerprints] Teacher model size:     {teacher_mb:.2f} MB")
    print(f"[Fingerprints] Per-sample fingerprints: {all_fps_size}")
    print(f"[Fingerprints] Per-class fingerprints:  {class_fps_size}")
    print(f"[Fingerprints] Random matrix R3:         {get_tensor_size(R3_norm.cpu())}")
    if multi:
        print(f"[Fingerprints] Random matrix R2:         {get_tensor_size(R2_norm.cpu())}")

    # Clean up
    del teacher
    torch.cuda.empty_cache()

    return {
        "R3_norm": R3_norm.cpu(),
        "R2_norm": R2_norm.cpu() if multi else torch.zeros(0),
        "all_fp_l3": all_fp_l3,
        "all_fp_l2": all_fp_l2 if multi else torch.zeros(0),
        "class_fp_l3": class_fp_l3,
        "class_fp_l2": class_fp_l2 if multi else torch.zeros(0),
        "labels": labels_tensor,
    }
