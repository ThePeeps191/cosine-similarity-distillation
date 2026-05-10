"""Train a ResNet‑56 teacher on CIFAR‑100.

Saves the best checkpoint (by test accuracy) and training history.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import Config
from data import get_cifar100_loaders
from models import resnet56
from utils import set_seed


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.  Returns (average_loss, accuracy_percent)."""
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a data loader.  Returns (average_loss, accuracy_percent)."""
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


def train_teacher(config: Config) -> dict[str, Any]:
    """Train a ResNet‑56 teacher and save the best checkpoint.

    Returns:
        Dictionary with keys ``train_loss``, ``train_acc``, ``test_acc``
        (lists of per‑epoch values) and ``best_acc``.
    """
    set_seed(config.seed)
    os.makedirs(config.results_dir, exist_ok=True)

    train_loader, test_loader = get_cifar100_loaders(config)

    model = resnet56(num_classes=config.num_classes).to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = MultiStepLR(
        optimizer, milestones=config.milestones, gamma=config.gamma,
    )

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "test_acc": [],
    }
    best_acc = 0.0

    for epoch in range(1, config.epochs_teacher + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device,
        )
        _, test_acc = evaluate(model, test_loader, config.device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), config.teacher_ckpt)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Teacher] Epoch {epoch:3d}/{config.epochs_teacher} | "
            f"LR: {lr:.5f} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
            f"Best: {best_acc:.2f}%"
        )

    history["best_acc"] = best_acc
    with open(config.teacher_history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history
