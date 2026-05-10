"""Train a ResNet‑20 student with standard Knowledge Distillation (Hinton et al., 2015).

The teacher is loaded and used at every batch:  L = (1‑α)·CE + α·T²·KL.
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


def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float,
    alpha: float,
) -> torch.Tensor:
    """Standard KD loss (Hinton et al., 2015).

    ``L = (1-α)·CE(student, labels) + α·T²·KL(σ_student/T || σ_teacher/T)``.
    """
    ce = F.cross_entropy(student_logits, labels)
    kd = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    return (1.0 - alpha) * ce + alpha * kd


def train_epoch_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    T: float,
    alpha: float,
) -> tuple[float, float]:
    student.train()
    teacher.eval()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        s_logits = student(inputs)
        with torch.no_grad():
            t_logits = teacher(inputs)
        loss = kd_loss_fn(s_logits, t_logits, labels, T=T, alpha=alpha)
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


def train_student_kd(config: Config) -> dict[str, Any]:
    """Train ResNet‑20 with KD.  Teacher is loaded and kept in eval mode."""
    set_seed(config.seed)
    os.makedirs(config.results_dir, exist_ok=True)

    train_loader, test_loader = get_cifar100_loaders(config)

    teacher = resnet56(num_classes=config.num_classes).to(config.device)
    teacher.load_state_dict(torch.load(config.teacher_ckpt, map_location=config.device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = resnet20(num_classes=config.num_classes).to(config.device)

    optimizer = optim.SGD(
        student.parameters(), lr=config.lr, momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)

    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, config.epochs_student + 1):
        train_loss, train_acc = train_epoch_kd(
            student, teacher, train_loader, optimizer, config.device,
            T=config.kd_temperature, alpha=config.kd_alpha,
        )
        _, test_acc = evaluate(student, test_loader, config.device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), config.kd_ckpt)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[KD] Epoch {epoch:3d}/{config.epochs_student} | "
            f"LR: {lr:.5f} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
            f"Best: {best_acc:.2f}%"
        )

    history["best_acc"] = best_acc
    with open(os.path.join(config.results_dir, "kd_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history
