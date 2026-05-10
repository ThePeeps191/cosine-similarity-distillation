"""Train a ResNet‑20 student with FitNet feature‑matching distillation (Romero et al., 2015).

Loss = CE + β·MSE(student_pooled_features, teacher_pooled_features).
A grid search over *β* is performed; the best model is kept.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import Config
from data import get_cifar100_loaders
from models import resnet20, resnet56
from utils import set_seed


def train_epoch_fitnet(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    beta: float,
) -> tuple[float, float]:
    student.train()
    teacher.eval()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        s_logits, s_feat = student(inputs, return_features=True)
        with torch.no_grad():
            _, t_feat = teacher(inputs, return_features=True)

        # Pool both feature maps (B, 64, 8, 8) → (B, 64)
        s_pooled = F.adaptive_avg_pool2d(s_feat, (1, 1)).squeeze(-1).squeeze(-1)
        t_pooled = F.adaptive_avg_pool2d(t_feat, (1, 1)).squeeze(-1).squeeze(-1)

        ce = F.cross_entropy(s_logits, labels)
        fm = F.mse_loss(s_pooled, t_pooled)
        loss = ce + beta * fm
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = s_logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


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


def train_student_fitnet(config: Config) -> dict[str, Any]:
    """Train ResNet‑20 with FitNet, sweeping over betas.

    Returns the history of the best run (by test accuracy across all betas).
    """
    set_seed(config.seed)
    os.makedirs(config.results_dir, exist_ok=True)
    train_loader, test_loader = get_cifar100_loaders(config)

    teacher = resnet56(num_classes=config.num_classes).to(config.device)
    teacher.load_state_dict(torch.load(config.teacher_ckpt, map_location=config.device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    best_acc_overall = 0.0
    best_history: dict[str, Any] = {}
    best_state: dict[str, torch.Tensor] | None = None

    for beta in config.fitnet_beta_values:
        print(f"\n[FitNet] Trying beta = {beta}")
        set_seed(config.seed)  # reset seed for fair comparison
        student = resnet20(num_classes=config.num_classes).to(config.device)
        optimizer = optim.SGD(
            student.parameters(), lr=config.lr, momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        scheduler = MultiStepLR(
            optimizer, milestones=config.milestones, gamma=config.gamma,
        )

        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "test_acc": [],
        }
        best_acc = 0.0
        best_state_beta: dict[str, torch.Tensor] | None = None

        for epoch in range(1, config.epochs_student + 1):
            train_loss, train_acc = train_epoch_fitnet(
                student, teacher, train_loader, optimizer, config.device, beta,
            )
            _, test_acc = evaluate(student, test_loader, config.device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                best_state_beta = {k: v.cpu().clone() for k, v in student.state_dict().items()}

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[FitNet b={beta}] Epoch {epoch:3d}/{config.epochs_student} | "
                f"LR: {lr:.5f} | Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
                f"Best: {best_acc:.2f}%"
            )

        history["best_acc"] = best_acc
        history["beta"] = beta
        if best_acc > best_acc_overall:
            best_acc_overall = best_acc
            best_history = history
            best_state = best_state_beta

    # Save the checkpoint from the best beta run
    if best_state is not None:
        torch.save(best_state, config.fitnet_ckpt)

    best_history["best_acc_overall"] = best_acc_overall
    with open(os.path.join(config.results_dir, "fitnet_history.json"), "w") as f:
        json.dump(best_history, f, indent=2)

    print(f"\n[FitNet] Best beta run achieves {best_acc_overall:.2f}%")

    return best_history
