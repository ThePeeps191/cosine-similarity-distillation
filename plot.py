"""Plotting utilities for CSD experiments.

All plots are saved as 300‑dpi PNGs in the ``results/`` directory.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non‑interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from models import resnet20


def _save(name: str, config: Config) -> str:
    path = os.path.join(config.results_dir, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# 1. Teacher training curves
# ---------------------------------------------------------------------------
def plot_teacher_training(history: dict[str, Any], config: Config) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Teacher Training Loss")

    ax2.plot(epochs, history["train_acc"], "b-", label="Train", linewidth=1.5)
    ax2.plot(epochs, history["test_acc"], "r-", label="Test", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Teacher Accuracy")
    ax2.legend()

    plt.suptitle("ResNet‑56 Teacher Training on CIFAR‑100", fontweight="bold")
    _save("teacher_training.png", config)


# ---------------------------------------------------------------------------
# 2. Test accuracy vs. epoch for all methods
# ---------------------------------------------------------------------------
def plot_accuracy_comparison(
    hist_dict: dict[str, list[float]], config: Config,
) -> None:
    plt.figure(figsize=(10, 6))
    colors = {
        "Student‑only": "tab:gray",
        "KD": "tab:orange",
        "FitNet": "tab:green",
        "CSD per‑sample": "tab:blue",
        "CSD per‑class": "tab:red",
    }
    for label, test_acc in hist_dict.items():
        epochs = range(1, len(test_acc) + 1)
        plt.plot(epochs, test_acc, label=label, color=colors.get(label),
                 linewidth=1.5)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Comparison Across Distillation Methods")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save("accuracy_comparison.png", config)


# ---------------------------------------------------------------------------
# 3. Horizontal bar chart of final accuracies
# ---------------------------------------------------------------------------
def plot_accuracy_barplot(acc_dict: dict[str, float], config: Config) -> None:
    methods = list(acc_dict.keys())
    values = list(acc_dict.values())

    present_methods = [m for m, v in zip(methods, values) if v is not None]
    present_values = [v for v in values if v is not None]

    if not present_values:
        print("[WARN] No accuracy values available — skipping bar plot.")
        return

    plt.figure(figsize=(10, 5))
    bars = plt.barh(present_methods, present_values,
                    color=plt.cm.viridis(np.linspace(0.2, 0.9, len(present_methods))))
    plt.xlabel("Top‑1 Accuracy (%)")
    plt.title("Final Test Accuracy by Method")

    for bar, val in zip(bars, present_values):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}%", va="center", fontsize=9)

    _save("accuracy_barplot.png", config)


# ---------------------------------------------------------------------------
# 4. Storage comparison (log scale)
# ---------------------------------------------------------------------------
def plot_storage_comparison(config: Config) -> None:
    if not os.path.exists(config.all_fingerprints_path):
        print("[WARN] Missing all_fingerprints.pt — skipping storage plot.")
        return
    if not os.path.exists(config.class_fingerprints_path):
        print("[WARN] Missing class_fingerprints.pt — skipping storage plot.")
        return

    all_fp = torch.load(config.all_fingerprints_path, map_location="cpu")
    class_fp = torch.load(config.class_fingerprints_path, map_location="cpu")

    # Re‑compute teacher size
    from models import resnet56
    teacher = resnet56()
    from utils import get_model_size_mb
    teacher_mb = get_model_size_mb(teacher)

    all_fp_kb = all_fp.numel() * all_fp.element_size() / 1024
    class_fp_kb = class_fp.numel() * class_fp.element_size() / 1024
    teacher_kb = teacher_mb * 1024

    items = {
        "ResNet‑56 Teacher": teacher_kb,
        "CSD Per‑sample (r=128)": all_fp_kb,
        "CSD Per‑class (r=128)": class_fp_kb,
    }

    labels = list(items.keys())
    sizes = list(items.values())

    plt.figure(figsize=(10, 5))
    bars = plt.barh(labels, sizes, color=["#e74c3c", "#3498db", "#2ecc71"])
    plt.xscale("log")
    plt.xlabel("Storage (KB, log scale)")
    plt.title("Storage Required for Knowledge Transfer")

    for bar, siz in zip(bars, sizes):
        if siz > 1024:
            text = f"{siz / 1024:.2f} MB"
        else:
            text = f"{siz:.2f} KB"
        plt.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                 text, va="center", fontsize=10)

    _save("storage_comparison.png", config)


# ---------------------------------------------------------------------------
# 5. t‑SNE visualisation of pooled features
# ---------------------------------------------------------------------------
def plot_tsne(
    config: Config,
    baseline_model_path: str,
    csd_model_path: str,
    test_loader: DataLoader,
    csd_label: str = "CSD Student (per-sample)",
) -> None:
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[WARN] scikit-learn not installed — skipping t‑SNE plot.")
        return

    device = config.device

    def extract_features(model_path: str) -> tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = resnet20(num_classes=config.num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        feats, labs = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                _, fmap = model(images, return_features=True)
                pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).squeeze(-1).squeeze(-1)
                feats.append(pooled.cpu().numpy())
                labs.append(labels.numpy())
        return np.concatenate(feats), np.concatenate(labs)

    if not os.path.exists(baseline_model_path) or not os.path.exists(csd_model_path):
        print("[WARN] Missing baseline or CSD checkpoint — skipping t‑SNE plot.")
        return

    f_base, l_base = extract_features(baseline_model_path)
    f_csd, l_csd = extract_features(csd_model_path)

    if len(f_base) == 0:
        print("[WARN] Empty test features — skipping t‑SNE plot.")
        return

    # Use a subset (first 2000) for speed
    n_sub = min(2000, len(f_base))
    idx = np.random.RandomState(42).choice(len(f_base), n_sub, replace=False)

    f_all = np.concatenate([f_base[idx], f_csd[idx]], axis=0)
    l_all = np.concatenate([l_base[idx], l_csd[idx]], axis=0)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    embedded = tsne.fit_transform(f_all)
    e_base = embedded[:n_sub]
    e_csd = embedded[n_sub:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sc1 = ax1.scatter(e_base[:, 0], e_base[:, 1], c=l_base[idx],
                      cmap="tab20", alpha=0.6, s=8)
    ax1.set_title("Baseline Student (no distillation)")
    ax1.axis("off")

    sc2 = ax2.scatter(e_csd[:, 0], e_csd[:, 1], c=l_csd[idx],
                      cmap="tab20", alpha=0.6, s=8)
    ax2.set_title(csd_label)
    ax2.axis("off")

    plt.suptitle("t‑SNE of Pooled Layer‑3 Features (CIFAR‑100 test subset)", fontweight="bold")
    plt.subplots_adjust(right=0.9)
    _save("tsne_features.png", config)


# ---------------------------------------------------------------------------
# 6. Fingerprint dimension ablation plot
# ---------------------------------------------------------------------------
def plot_fingerprint_ablation(
    data: list[dict[str, Any]], config: Config,
) -> None:
    rs = [d["r"] for d in data]
    accs = [d["accuracy"] for d in data]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(rs, accs, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Fingerprint Dimension r")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Effect of Fingerprint Dimension on CSD Accuracy")
    ax1.grid(True, alpha=0.3)

    for r, a in zip(rs, accs):
        ax1.annotate(f"{a:.2f}%", (r, a), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    _save("ablation_r.png", config)


# ---------------------------------------------------------------------------
# 7. Lambda sensitivity plot
# ---------------------------------------------------------------------------
def plot_lambda_sensitivity(
    data: list[dict[str, Any]], config: Config,
) -> None:
    lams = [d["lambda"] for d in data]
    accs = [d["accuracy"] for d in data]

    plt.figure(figsize=(8, 5))
    plt.plot(lams, accs, "rs-", linewidth=2, markersize=8)
    plt.xlabel("CSD Loss Weight λ")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Sensitivity of CSD Accuracy to λ")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)

    for la, a in zip(lams, accs):
        plt.annotate(f"{a:.2f}%", (la, a), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    _save("ablation_lambda.png", config)


# ---------------------------------------------------------------------------
# 8. Training loss vs epoch for all methods
# ---------------------------------------------------------------------------
def plot_training_loss_comparison(
    hist_dict_loss: dict[str, list[float]], config: Config,
) -> None:
    """Line plot of training loss over epochs for every student method."""
    plt.figure(figsize=(10, 6))
    colors = {
        "Student-only": "tab:gray",
        "KD": "tab:orange",
        "FitNet": "tab:green",
        "CSD per-sample": "tab:blue",
        "CSD per-class": "tab:red",
    }
    for label, loss in hist_dict_loss.items():
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, label=label, color=colors.get(label),
                 linewidth=1.5)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison Across Distillation Methods")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save("loss_comparison.png", config)


# ---------------------------------------------------------------------------
# 9. Accuracy improvement over baseline bar chart
# ---------------------------------------------------------------------------
def plot_accuracy_improvement(
    accuracies: dict[str, float | None], config: Config,
) -> None:
    """Bar chart showing the absolute accuracy gain over the student-only baseline."""
    baseline = accuracies.get("student_only")
    if baseline is None:
        print("[WARN] Baseline accuracy missing — skipping improvement plot.")
        return

    methods_order = ["KD", "FitNet", "CSD per-sample", "CSD per-class"]
    label_map = {
        "KD": accuracies.get("kd"),
        "FitNet": accuracies.get("fitnet"),
        "CSD per-sample": accuracies.get("csd_per_sample"),
        "CSD per-class": accuracies.get("csd_per_class"),
    }

    gains = []
    labels_present = []
    for m in methods_order:
        v = label_map[m]
        if v is not None:
            gains.append(v - baseline)
            labels_present.append(m)

    if not gains:
        print("[WARN] No student accuracies available — skipping improvement plot.")
        return

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels_present, gains,
                   color=["tab:orange", "tab:green", "tab:blue", "tab:red"],
                   width=0.5)
    plt.ylabel("Accuracy Gain over Baseline (pp)")
    plt.title("Distillation Improvement Over Student-Only Baseline")
    plt.axhline(y=0, color="gray", linewidth=0.5)

    for bar, g in zip(bars, gains):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"+{g:.2f}pp", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    plt.grid(True, axis="y", alpha=0.3)
    _save("accuracy_improvement.png", config)


# ---------------------------------------------------------------------------
# 10. Fingerprint alignment over CSD training epochs
# ---------------------------------------------------------------------------
def plot_fingerprint_alignment(
    fp_alignment_data: dict[str, list[float]], config: Config,
) -> None:
    """Line plot showing cosine similarity between student and teacher fingerprints over epochs."""
    plt.figure(figsize=(8, 5))
    colors = {"CSD per-sample": "tab:blue", "CSD per-class": "tab:red"}
    for label, fp_align in fp_alignment_data.items():
        epochs = range(1, len(fp_align) + 1)
        plt.plot(epochs, fp_align, label=label, color=colors.get(label),
                 linewidth=1.5)

    plt.xlabel("Epoch")
    plt.ylabel("Mean Cosine Similarity | φ_student, φ_teacher")
    plt.title("Student–Teacher Fingerprint Alignment During CSD Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    _save("fp_alignment.png", config)
