"""Publication-grade figure generation for CSD & CSD++ experiments.

Generates high-DPI (300 dpi) figures suitable for an arXiv / conference paper:
1. fig_main_comparison.png     : Main CIFAR-100 ResNet-56 -> ResNet-20 (multi-seed mean +/- std)
2. fig_pareto_storage.png      : Accuracy vs. Transfer Storage Pareto Frontier (log scale, KB to MB)
3. fig_privacy_pareto.png      : Privacy (Feature Inversion Cosine) & JL Error vs. Dimension r & Quantization
4. fig_cross_arch.png          : Cross-Architecture Generalization (ResNet-56 -> MobileNetV2-CIFAR)
5. fig_component_ablation.png  : Controlled Component Ablation (v4.1 -> Centering -> Projector -> JL-Rel -> Anchor -> CSD++)
6. fig_training_dynamics.png   : Test accuracy curves & fingerprint alignment over training
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "figure.dpi": 300,
    })


def _save_fig(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {path}")


def plot_privacy_and_jl_audit(audit_rows: list[dict[str, Any]], out_path: str) -> None:
    """Two-panel figure: (a) Feature Inversion Cosine vs Storage/r, (b) JL Pairwise Error vs Storage."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    styles = {
        "fp32": ("#d62728", "o", "-", 9.0, 2.8, "FP32 (32-bit)"),
        "int8": ("#1f77b4", "s", ":", 6.2, 1.8, "INT8 (8-bit, overlaps FP32)"),
        "1bit": ("#2ca02c", "^", "--", 7.0, 2.0, "1-bit SimHash"),
    }

    for qmode, (color, marker, ls, ms, lw, label) in styles.items():
        rows = sorted([d for d in audit_rows if d["quant"] == qmode], key=lambda x: x["r"])
        if not rows:
            continue
        rs = [d["r"] for d in rows]
        recon = [d["recon_cosine"] for d in rows]
        jl_mae = [d["jl_pairwise_mae"] for d in rows]
        ax1.plot(rs, recon, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=ms, label=label)
        if qmode == "1bit" and "mlp_recon_cosine" in rows[0]:
            mlp_recon = [d["mlp_recon_cosine"] for d in rows]
            ax1.plot(rs, mlp_recon, color="#1b7837", marker="v", linestyle=":", linewidth=1.6, markersize=5.5, label="1-bit SimHash (2-layer MLP attacker)")
        ax2.plot(rs, jl_mae, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=ms, label=label)

    # Highlight naive overcomplete pointwise RP operating point
    old_pt = [d for d in audit_rows if d["r"] == 128 and d["quant"] == "fp32"]
    if old_pt:
        ax1.annotate(
            "Direct-RP (r=128, FP32)\n100% linearly invertible",
            xy=(128, old_pt[0]["recon_cosine"]),
            xytext=(18, 0.94),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3),
            fontsize=8.3, color="#b22222", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff5f5", edgecolor="#d62728", alpha=0.92),
        )

    all_rs = sorted(list({d["r"] for d in audit_rows}))
    ax1.axvline(64, color="gray", linestyle=":", linewidth=1.2, label="Feature dim C = 64")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(all_rs)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel("Projection Dimension r")
    ax1.set_ylabel("Feature Inversion Cosine Similarity (lower = more private)")
    ax1.set_title("(a) Linear & MLP Feature-Inversion Audit")
    ax1.set_ylim(0.25, 1.07)
    ax1.legend(loc="lower right", fontsize=8.0)

    ax2.axvline(64, color="gray", linestyle=":", linewidth=1.2)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(all_rs)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel("Projection Dimension r")
    ax2.set_ylabel("JL Pairwise Cosine MAE (lower = higher fidelity)")
    ax2.set_title("(b) Johnson–Lindenstrauss Pairwise Geometry Error")
    ax2.legend(loc="upper right", fontsize=8.4)

    plt.tight_layout()
    _save_fig(out_path)


def plot_main_benchmark(summary_rows: list[dict[str, Any]], out_path: str, teacher_acc: float | None = None) -> None:
    """Horizontal bar chart with multi-seed mean +/- std error bars, color-coded by regime."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10.0, 5.4))

    labels = [r["label"] for r in summary_rows]
    means = [r["mean_std"] for r in summary_rows]
    stds = [r["std_std"] for r in summary_rows]
    cats = [r["category"] for r in summary_rows]

    cat_colors = {
        "baseline": "#7f7f7f",
        "online": "#ff7f0e",
        "offline_baseline": "#9467bd",
        "csd_old": "#d62728",
        "csd_new": "#1f77b4",
    }
    colors = [cat_colors.get(c, "#1f77b4") for c in cats]

    y = np.arange(len(labels))
    bars = ax.barh(y, means, xerr=stds, height=0.64, color=colors, capsize=4, alpha=0.92, edgecolor="#222222", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    if teacher_acc is not None:
        ax.axvline(teacher_acc, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Teacher ResNet-56 ({teacher_acc:.2f}%)")

    xmin = max(60.0, min(means) - 1.5)
    xmax = max(max(means) + 2.7, (teacher_acc + 1.5) if teacher_acc else max(means) + 2.7)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("CIFAR-100 Top-1 Test Accuracy (%, Standard Eval)")
    ax.set_title("ResNet-56 → ResNet-20 Distillation on CIFAR-100 (200 Epochs, 3-Seed Mean ± Std)", pad=36)

    for bar, m, s, r in zip(bars, means, stds, summary_rows):
        txt = f"{m:.2f} ± {s:.2f}%" if s > 0 else f"{m:.2f}%"
        if r.get("storage_str"):
            txt += f"  [{r['storage_str']}]"
        ax.text(m + s + 0.10, bar.get_y() + bar.get_height() / 2, txt, va="center", fontsize=8.6)

    # Custom legend placed cleanly above the axes (zero overlap with bars/labels)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=cat_colors["baseline"], label="No-Teacher Baselines"),
        Patch(facecolor=cat_colors["online"], label="Online Teacher KD"),
        Patch(facecolor=cat_colors["offline_baseline"], label="Offline Logit KD"),
        Patch(facecolor=cat_colors["csd_old"], label="Direct-RP (Pointwise)"),
        Patch(facecolor=cat_colors["csd_new"], label="SimHash-CSD (Ours)"),
    ]
    if teacher_acc is not None:
        handles.append(plt.Line2D([0], [0], color="#2ca02c", linestyle="--", lw=1.5, label=f"Teacher ({teacher_acc:.2f}%)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=8.2, frameon=True)

    plt.tight_layout()
    _save_fig(out_path)


def plot_pareto_frontier(points: list[dict[str, Any]], out_path: str, baseline_acc: float, kd_acc: float | None = None) -> None:
    """Accuracy vs. Transfer Storage (log KB) Pareto scatter plot."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    cat_style = {
        "online": ("#ff7f0e", "D", 95, "Online KD / FitNet / SP (Full Teacher)"),
        "offline_baseline": ("#9467bd", "s", 80, "Offline Sparse Logit KD"),
        "csd_old": ("#d62728", "X", 100, "Direct-RP (Pointwise Baseline)"),
        "csd_new": ("#1f77b4", "o", 105, "SimHash-CSD (Ours, 1-bit)"),
    }

    plotted_cats = set()
    for p in points:
        kb = max(1.0, p["storage_bytes"] / 1024.0)
        acc = p["acc"]
        cat = p["category"]
        color, marker, size, cat_lbl = cat_style[cat]
        lbl = cat_lbl if cat not in plotted_cats else None
        plotted_cats.add(cat)
        ax.scatter(kb, acc, color=color, marker=marker, s=size, zorder=4, edgecolors="#222222", linewidths=0.7, label=lbl)
        dx, dy = p.get("offset", (6, 6))
        ax.annotate(p["label"], (kb, acc), textcoords="offset points", xytext=(dx, dy), fontsize=8.5, fontweight="bold" if cat == "csd_new" else "normal")

    # Draw Pareto frontier through CSD++ points
    ours = sorted([p for p in points if p["category"] == "csd_new"], key=lambda x: x["storage_bytes"])
    if len(ours) >= 2:
        ax.plot(
            [p["storage_bytes"] / 1024.0 for p in ours],
            [p["acc"] for p in ours],
            color="#1f77b4", linestyle="--", linewidth=1.6, alpha=0.7, zorder=3,
        )

    ax.axhline(baseline_acc, color="#7f7f7f", linestyle=":", linewidth=1.4, label=f"Student-only Baseline ({baseline_acc:.2f}%)")
    if kd_acc is not None:
        ax.axhline(kd_acc, color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.65, label=f"Online KD ({kd_acc:.2f}%)")

    ax.set_xscale("log")
    all_accs = [p["acc"] for p in points] + [baseline_acc]
    ax.set_ylim(min(all_accs) - 0.18, max(all_accs) + 0.25)
    ax.set_xlabel("Transferred Teacher Artifact Storage (KB, log scale)")
    ax.set_ylabel("CIFAR-100 Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs. Transfer Storage Pareto Frontier (ResNet-56 → ResNet-20, 3 Seeds)")
    ax.legend(loc="lower right", fontsize=8.3)

    plt.tight_layout()
    _save_fig(out_path)


def plot_cross_architecture(
    rows: list[dict[str, Any]],
    out_path: str,
    teacher_acc: float | None = None,
    title: str = "Cross-Architecture Distillation: ResNet-56 (C=64) -> VGG-8-BN (C=512)",
) -> None:
    """Bar chart showing Cross-Architecture Generalization (mean +/- std across seeds)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9.4, 4.8))

    labels = [
        f"{r['label']}\n({r['storage_str']})" if r.get("storage_str") and r["storage_str"] != "0 B" else r["label"]
        for r in rows
    ]
    means = [r["mean_std"] for r in rows]
    stds = [r["std_std"] for r in rows]
    colors = ["#7f7f7f" if r["category"] == "baseline" else ("#ff7f0e" if r["category"] == "online" else ("#9467bd" if r["category"] == "offline_baseline" else "#1f77b4")) for r in rows]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, width=0.55, color=colors, capsize=4, edgecolor="#222222", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=14, ha="right", fontsize=9.2)
    ax.set_ylabel("CIFAR-100 Top-1 Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(max(55.0, min(means) - 1.5), max(means) + 2.0)

    if teacher_acc is not None:
        ax.axhline(teacher_acc, color="#2ca02c", linestyle="--", linewidth=1.6, zorder=2,
                   label=f"Teacher ResNet-56 ({teacher_acc:.2f}%)")
        ax.legend(loc="upper right", fontsize=9.0)

    for bar, m, s in zip(bars, means, stds):
        txt = f"{m:.2f}±{s:.2f}%" if s > 0 else f"{m:.2f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.14, txt, ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    ax.annotate(
        "Note: Original CSD v4.1 cannot run here\n(requires C_S == C_T; here 512 != 64)",
        xy=(0.02, 0.87), xycoords="axes fraction",
        fontsize=8.6, color="#b22222",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff0f0", edgecolor="#d62728", alpha=0.92),
    )

    plt.tight_layout()
    _save_fig(out_path)


def plot_component_ablation(rows: list[dict[str, Any]], out_path: str) -> None:
    """Bar chart showing controlled step-by-step ablation gains."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    labels = [r["label"] for r in rows]
    accs = [r["acc"] for r in rows]
    colors = ["#7f7f7f", "#d62728", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"][:len(labels)]

    x = np.arange(len(labels))
    bars = ax.bar(x, accs, width=0.58, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("CIFAR-100 Top-1 Accuracy (%)")
    ax.set_title("Controlled Ablation: Disentangling What Drives CSD++ Performance")
    ax.set_ylim(min(accs) - 1.5, max(accs) + 1.5)

    base = accs[0]
    for bar, a in zip(bars, accs):
        delta = a - base
        sign = "+" if delta >= 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2, a + 0.08,
            f"{a:.2f}%\n({sign}{delta:.2f}pp)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    plt.tight_layout()
    _save_fig(out_path)
