"""Train a ResNet-20 student with Cosine Similarity Distillation (CSD).

The teacher is **never loaded**.  Instead, pre-computed fingerprints are read from
disk and used as regression targets alongside the classification loss:

    L = CrossEntropy(logits, labels)
        + lambda * (1 - cos_sim(phi_S_l2, phi_T_l2) + 1 - cos_sim(phi_S_l3, phi_T_l3))

Uses multi-layer fingerprints (layer-2 + layer-3) to capture complementary
teacher geometry at texture and abstraction levels.  Cosine-similarity loss
focuses on directional agreement, and a lambda warmup lets early classification
stabilise before fingerprint alignment begins.

Supports both per-sample and per-class fingerprint variants.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import Config
from data import get_cifar100_loaders, get_indexed_cifar100_loaders
from models import resnet20
from utils import set_seed


def train_epoch_csd(
    student: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    R3_norm: torch.Tensor,
    fp_lookup_l3: torch.Tensor,
    lambda_csd: float,
    use_per_class: bool,
    multi_layer: bool = True,
    R2_norm: torch.Tensor | None = None,
    fp_lookup_l2: torch.Tensor | None = None,
    l2_weight: float = 0.2,
    l3_weight: float = 1.0,
) -> tuple[float, float, float]:
    """Single training epoch for CSD.

    Args:
        R3_norm: Layer-3 random reference matrix (64, r).
        fp_lookup_l3: Layer-3 fingerprint table.
        R2_norm: Layer-2 random reference matrix (32, r), None if single-layer.
        fp_lookup_l2: Layer-2 fingerprint table, None if single-layer.
        multi_layer: If True, compute combined loss from both layers.
        l2_weight: Weight for layer-2 fingerprint loss (default 0.2).
        l3_weight: Weight for layer-3 fingerprint loss (default 1.0).

    Returns:
        (average_loss, accuracy_percent, mean_fingerprint_cosine_similarity)
    """
    student.train()
    running_loss = 0.0
    correct, total = 0, 0
    running_fp_sim = 0.0
    fp_count = 0

    for batch in loader:
        if use_per_class:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
        else:
            images, labels, indices = batch
            images, labels, indices = (
                images.to(device), labels.to(device), indices.to(device),
            )

        optimizer.zero_grad()

        if multi_layer:
            logits, feat_l2, feat_l3 = student(images, return_features="multi")
            pooled_l2 = F.adaptive_avg_pool2d(feat_l2, (1, 1)).squeeze(-1).squeeze(-1)
            pooled_l3 = F.adaptive_avg_pool2d(feat_l3, (1, 1)).squeeze(-1).squeeze(-1)
            f_norm_l2 = F.normalize(pooled_l2, p=2, dim=1)
            f_norm_l3 = F.normalize(pooled_l3, p=2, dim=1)
            phi_S_l2 = f_norm_l2 @ R2_norm  # (B, r)
            phi_S_l3 = f_norm_l3 @ R3_norm  # (B, r)

            if use_per_class:
                phi_T_l2 = fp_lookup_l2[labels]
                phi_T_l3 = fp_lookup_l3[labels]
            else:
                phi_T_l2 = fp_lookup_l2[indices]
                phi_T_l3 = fp_lookup_l3[indices]

            distil_l2 = 1.0 - F.cosine_similarity(phi_S_l2, phi_T_l2, dim=1).mean()
            distil_l3 = 1.0 - F.cosine_similarity(phi_S_l3, phi_T_l3, dim=1).mean()
            distil = l2_weight * distil_l2 + l3_weight * distil_l3

            with torch.no_grad():
                sim_l2 = F.cosine_similarity(phi_S_l2, phi_T_l2, dim=1).mean().item()
                sim_l3 = F.cosine_similarity(phi_S_l3, phi_T_l3, dim=1).mean().item()
                batch_fp_sim = (sim_l2 + sim_l3) / 2.0
        else:
            logits, features = student(images, return_features=True)
            pooled = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
            f_norm = F.normalize(pooled, p=2, dim=1)
            phi_S_l3 = f_norm @ R3_norm  # (B, r)

            if use_per_class:
                phi_T_l3 = fp_lookup_l3[labels]
            else:
                phi_T_l3 = fp_lookup_l3[indices]

            distil = 1.0 - F.cosine_similarity(phi_S_l3, phi_T_l3, dim=1).mean()

            with torch.no_grad():
                batch_fp_sim = F.cosine_similarity(phi_S_l3, phi_T_l3, dim=1).mean().item()

        ce = F.cross_entropy(logits, labels)
        loss = ce + lambda_csd * distil

        running_fp_sim += batch_fp_sim * images.size(0)
        fp_count += images.size(0)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total, running_fp_sim / fp_count


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * inputs.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def train_student_csd(
    config: Config,
    use_per_class: bool = False,
    lambda_csd: Optional[float] = None,
    r_override: Optional[int] = None,
    epochs_override: Optional[int] = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Train ResNet-20 with Cosine Similarity Distillation.

    Args:
        config: Experiment configuration.
        use_per_class: If True, use per-class averaged fingerprints.
        lambda_csd: CSD loss weight; defaults to 0.1.
        r_override: Override fingerprint dimension (for ablation).
        epochs_override: Override number of training epochs.
        save_outputs: If False, skip saving checkpoint and history (for ablations).

    Returns:
        Training history dict including ``best_acc``.
    """
    set_seed(config.seed)
    os.makedirs(config.results_dir, exist_ok=True)
    r = r_override if r_override is not None else config.r
    lam = lambda_csd if lambda_csd is not None else 0.1
    epochs = epochs_override if epochs_override is not None else config.epochs_student
    multi = config.use_multi_layer

    # ------------------------------------------------------------------
    # Load reference matrices and fingerprints
    # ------------------------------------------------------------------
    R3_norm = torch.load(config.random_matrix_path, map_location="cpu")
    R3_norm = R3_norm.to(config.device)

    R2_norm = None
    fp_lookup_l2 = None
    if multi:
        R2_norm = torch.load("results/random_matrix_R2.pt", map_location="cpu")
        R2_norm = R2_norm.to(config.device)

    if use_per_class:
        fp_lookup_l3 = torch.load(config.class_fingerprints_path, map_location="cpu")
        fp_lookup_l3 = fp_lookup_l3.to(config.device)
        if multi:
            fp_lookup_l2 = torch.load("results/class_fingerprints_l2.pt", map_location="cpu")
            fp_lookup_l2 = fp_lookup_l2.to(config.device)
        train_loader, test_loader = get_cifar100_loaders(config)
    else:
        fp_lookup_l3 = torch.load(config.all_fingerprints_path, map_location="cpu")
        fp_lookup_l3 = fp_lookup_l3.to(config.device)
        if multi:
            fp_lookup_l2 = torch.load("results/all_fingerprints_l2.pt", map_location="cpu")
            fp_lookup_l2 = fp_lookup_l2.to(config.device)
        train_loader, test_loader = get_indexed_cifar100_loaders(config)

    assert R3_norm.shape[1] == fp_lookup_l3.shape[1] == r, \
        f"Dimension mismatch: R3={R3_norm.shape[1]}, fp={fp_lookup_l3.shape[1]}, r={r}"

    # ------------------------------------------------------------------
    # Build student
    # ------------------------------------------------------------------
    student = resnet20(num_classes=config.num_classes).to(config.device)

    optimizer = optim.SGD(
        student.parameters(), lr=config.lr, momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "test_acc": [], "fp_alignment": [],
    }
    best_acc = 0.0

    suffix = "per_class" if use_per_class else "per_sample"
    ckpt_path = (
        config.csd_class_ckpt if use_per_class else config.csd_sample_ckpt
    )

    for epoch in range(1, epochs + 1):
        warmup_end = min(config.warmup_epochs, epochs)
        if epoch <= warmup_end:
            effective_lam = lam * (epoch / warmup_end)
        else:
            effective_lam = lam

        train_loss, train_acc, fp_align = train_epoch_csd(
            student, train_loader, optimizer, config.device,
            R3_norm, fp_lookup_l3, effective_lam, use_per_class,
            multi_layer=multi, R2_norm=R2_norm, fp_lookup_l2=fp_lookup_l2,
            l2_weight=config.lambda_l2_weight, l3_weight=config.lambda_l3_weight,
        )
        _, test_acc = evaluate(student, test_loader, config.device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["fp_alignment"].append(fp_align)

        if test_acc > best_acc:
            best_acc = test_acc
            if save_outputs:
                torch.save(student.state_dict(), ckpt_path)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[CSD {suffix} λ={effective_lam:.4f}] Epoch {epoch:3d}/{epochs} | "
            f"LR: {lr:.5f} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | FP Align: {fp_align:.4f} | "
            f"Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%"
        )

    history["best_acc"] = best_acc
    history["lambda"] = lam
    history["r"] = r
    if save_outputs:
        history_path = os.path.join(
            config.results_dir, f"csd_{suffix}_history.json",
        )
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    return history
