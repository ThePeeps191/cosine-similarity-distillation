"""Train a ResNet‑20 student without distillation (baseline lower bound)."""

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
from models import resnet20
from utils import set_seed


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
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


def train_student_baseline(
    config: Config,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train ResNet‑20 with standard cross‑entropy (no distillation).

    Args:
        config: Experiment configuration.
        epochs_override: If given, use this many epochs instead of config value.

    Returns:
        Dictionary with training history and best test accuracy.
    """
    set_seed(config.seed)
    os.makedirs(config.results_dir, exist_ok=True)
    epochs = epochs_override if epochs_override is not None else config.epochs_student

    train_loader, test_loader = get_cifar100_loaders(config)
    model = resnet20(num_classes=config.num_classes).to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)

    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
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
            torch.save(model.state_dict(), config.baseline_ckpt)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Baseline] Epoch {epoch:3d}/{epochs} | "
            f"LR: {lr:.5f} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
            f"Best: {best_acc:.2f}%"
        )

    history["best_acc"] = best_acc
    with open(os.path.join(config.results_dir, "baseline_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history
