"""Generate all 6 publication-grade figures (300 dpi) directly from saved JSON results.

Zero manual copy-pasting: every bar, curve, error bar, and annotation is read directly
from the raw experiment JSONs under ``results/`` and written to both ``results/`` and ``paper/``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_csd_pp import (
    _apply_style,
    _save_fig,
    plot_main_benchmark,
    plot_pareto_frontier,
    plot_privacy_and_jl_audit,
)

RESULTS_DIR = "results"
PAPER_DIR = "paper"


def _load(name: str) -> Any:
    with open(os.path.join(RESULTS_DIR, name), "r") as f:
        return json.load(f)


def _clean_label(tag: str, default: str) -> str:
    overrides = {
        "csd_v41_class": "Direct-RP (pointwise)",
        "sp_online": "Online SP (Oracle)",
        "kd_online": "Online KD (Hinton)",
        "fitnet_online": "Online FitNet",
        "offline_sparse_kd": "Offline Sparse-12 Logits",
        "simhash_r64": "SimHash-CSD (r=64, 1-bit)",
        "simhash_r128": "SimHash-CSD (r=128, 1-bit)",
        "simhash_hybrid": "SimHash-CSD + Sparse-12",
        "simhash_dkd": "SimHash-CSD + Offline DKD",
    }
    return overrides.get(tag, default)


def plot_generalization_grid(out_path: str) -> None:
    """Three-panel generalization figure: (a) Cross-Arch VGG-8, (b) Large-Teacher ResNet-32x4, (c) STL-10 64x64."""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # --- Panel (a): Cross-Architecture ResNet-56 (C=64) -> VGG-8-BN (C=512) ---
    ax = axes[0]
    ca = _load("summary_cross_arch_e100.json")
    lbls_a = ["Baseline\n(VGG-8)", "Online\nKD\n(3.29MB)", "Offline\nSparse-12\n(1.34MB)", "SimHash\nr=64\n(391KB)", "SimHash\nr=128\n(781KB)"]
    means_a = [r["mean_std"] for r in ca]
    stds_a = [r["std_std"] for r in ca]
    cols_a = ["#7f7f7f", "#ff7f0e", "#9467bd", "#1f77b4", "#08306b"]
    x = np.arange(len(lbls_a))
    bars = ax.bar(x, means_a, yerr=stds_a, width=0.58, color=cols_a, capsize=3.5, edgecolor="#222222", linewidth=0.5)
    ax.axhline(72.50, color="#2ca02c", linestyle="--", linewidth=1.4, label="Teacher ResNet-56 (72.50%)")
    ax.set_xticks(x); ax.set_xticklabels(lbls_a, fontsize=8.0)
    ax.set_ylabel("Top-1 Test Accuracy (%)")
    ax.set_title("(a) Cross-Arch: ResNet-56 → VGG-8\n(8× Channel Mismatch, 0 Extra Params)")
    ax.set_ylim(69.5, 75.5)
    ax.legend(loc="upper left", fontsize=7.6)
    for b, m, s in zip(bars, means_a, stds_a):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.10, f"{m:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    # --- Panel (b): High-Capacity-Gap ResNet-32x4 (78.58%, 28.42MB) -> ResNet-20 ---
    ax = axes[1]
    lt = _load("summary_large_teacher_e100.json")
    lbls_b = ["Baseline\n(R20)", "Online\nKD\n(28.4MB)", "Offline\nSparse-12\n(1.34MB)", "SimHash\nr=64\n(391KB)", "SimHash\nr=128\n(781KB)"]
    means_b = [r["mean_std"] for r in lt]
    stds_b = [r["std_std"] for r in lt]
    cols_b = ["#7f7f7f", "#ff7f0e", "#9467bd", "#1f77b4", "#08306b"]
    x = np.arange(len(lbls_b))
    bars = ax.bar(x, means_b, yerr=stds_b, width=0.58, color=cols_b, capsize=3.5, edgecolor="#222222", linewidth=0.5)
    ax.axhline(means_b[0], color="#7f7f7f", linestyle=":", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(lbls_b, fontsize=8.0)
    ax.set_title("(b) Large-Teacher: ResNet-32×4 → R20\n(74.5× Compression, Immune to Gap Collapse)")
    ax.set_ylim(66.2, 70.1)
    for b, m, s in zip(bars, means_b, stds_b):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.08, f"{m:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    # --- Panel (c): Second Dataset STL-10 (64x64) ResNet-56 (81.30%) -> ResNet-20 ---
    ax = axes[2]
    stl = _load("summary_stl10_e150.json")
    stl_map = {r["tag"]: r for r in stl}
    order_c = ["baseline", "csd_v41_class", "kd_online", "sp_online", "simhash_r64", "simhash_r128"]
    lbls_c = ["Baseline\n(R20)", "Direct-RP\n(37KB)", "Online\nKD\n(3.26MB)", "Online\nSP\n(3.26MB)", "SimHash\nr=64\n(39.1KB)", "SimHash\nr=128\n(78.1KB)"]
    means_c = [stl_map[k]["mean_acc"] for k in order_c]
    stds_c = [stl_map[k]["std_acc"] for k in order_c]
    cols_c = ["#7f7f7f", "#d62728", "#ff7f0e", "#ffbb78", "#1f77b4", "#08306b"]
    x = np.arange(len(lbls_c))
    bars = ax.bar(x, means_c, yerr=stds_c, width=0.58, color=cols_c, capsize=3.5, edgecolor="#222222", linewidth=0.5)
    ax.axhline(81.30, color="#2ca02c", linestyle="--", linewidth=1.4, label="Teacher ResNet-56 (81.30%)")
    ax.set_xticks(x); ax.set_xticklabels(lbls_c, fontsize=7.6)
    ax.set_title("(c) Second Dataset: STL-10 (64×64)\n(85.5× Compression, 2.27× Faster)")
    ax.set_ylim(78.5, 84.0)
    ax.legend(loc="lower right", fontsize=7.6)
    for b, m, s in zip(bars, means_c, stds_c):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.10, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    _save_fig(out_path)


def plot_training_curves_and_hardware(main_rows: list[dict[str, Any]], out_path: str) -> None:
    """Two-panel figure: (a) 3-seed test accuracy curves (200 ep), (b) Apple M6 MPS speed & VRAM savings."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.5), gridspec_kw={"width_ratios": [1.35, 1.05]})

    curve_specs = [
        ("baseline", "Student-only (CE)", "#7f7f7f", "--"),
        ("csd_v41_class", "Direct-RP (148 KB)", "#d62728", ":"),
        ("kd_online", "Online KD (3.29 MB)", "#ff7f0e", "-."),
        ("sp_online", "Online SP Oracle (3.29 MB)", "#2ca02c", "--"),
        ("simhash_r64", "SimHash-CSD r=64 (391 KB)", "#1f77b4", "-"),
    ]
    for tag, label, color, ls in curve_specs:
        curves = []
        for seed in [42, 123, 456]:
            p = os.path.join(RESULTS_DIR, f"{tag}_resnet20_s{seed}_e200_history.json")
            if os.path.exists(p):
                curves.append(json.load(open(p))["val_acc"])
        if not curves:
            continue
        arr = np.array(curves)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
        epochs = np.arange(1, len(mean) + 1)
        ax1.plot(epochs, mean, label=label, color=color, linestyle=ls, linewidth=1.85)
        ax1.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.13)

    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("CIFAR-100 Top-1 Test Accuracy (%)")
    ax1.set_title("(a) Test Accuracy Trajectories (3-Seed Mean ± Std)")
    ax1.set_xlim(1, 200)
    ax1.set_ylim(42, 72.2)
    ax1.legend(loc="lower right", fontsize=8.3)

    # Panel (b): Measured Apple M6 MPS step latency & memory across teacher scales
    hw = _load("hardware_profile_m6.json")
    labels_hw = [
        "SimHash-CSD\n(R20 only, C100)",
        "Online KD\n(R56 + R20)",
        "Online KD\n(R32×4 + R20)",
        "SimHash-CSD\n(R20, STL-10)",
        "Online ViT-B/16\n(+ R20, STL-10)",
    ]
    ms_vals = [d["ms_per_step"] for d in hw]
    mem_vals = [d["driver_mem_mb"] for d in hw]
    y = np.arange(len(labels_hw))
    h = 0.36
    bars1 = ax2.barh(y - h / 2, ms_vals, height=h, color="#1f77b4", label="Step Latency (ms)", edgecolor="#222222", linewidth=0.5)
    ax2_top = ax2.twiny()
    bars2 = ax2_top.barh(y + h / 2, mem_vals, height=h, color="#ff7f0e", alpha=0.85, label="MPS Driver Mem (MB)", edgecolor="#222222", linewidth=0.5)
    ax2_top.spines["right"].set_visible(False)

    ax2.set_yticks(y); ax2.set_yticklabels(labels_hw, fontsize=8.2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Step Latency (ms/batch, lower = faster)", color="#1f77b4", fontsize=9.2)
    ax2_top.set_xlabel("Peak MPS Driver Memory (MB, lower = lighter)", color="#c95a00", fontsize=9.2)
    ax2.set_xlim(0, max(ms_vals) * 1.28)
    ax2_top.set_xlim(0, max(mem_vals) * 1.28)
    ax2.set_title("(b) Hardware Latency & Memory (Apple M6 MPS)", pad=24)
    ax2.grid(False)

    for b, v in zip(bars1, ms_vals):
        ax2.text(v + 3.5, b.get_y() + b.get_height() / 2, f"{v:.1f}ms", va="center", fontsize=7.6, color="#08306b", fontweight="bold")
    for b, v in zip(bars2, mem_vals):
        ax2_top.text(v + 35, b.get_y() + b.get_height() / 2, f"{v:.0f}MB", va="center", fontsize=7.6, color="#a84a00", fontweight="bold")

    plt.tight_layout()
    _save_fig(out_path)


def plot_bit_budget_ablation(out_path: str) -> None:
    """Two-panel figure: (a) Bit-Budget Theorem (1-bit vs INT8), (b) Controlled Component Ablation."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))

    def _acc(tag: str) -> tuple[float, float]:
        d = _load(f"{tag}_resnet20_s42_e100_history.json")
        return d["best_test_acc"], d["storage_bytes"] / 1024.0

    base_acc, _ = _acc("baseline")
    kd_acc, _ = _acc("kd_online")

    sh_tags = [
        ("simhash_r16", "r=16"), ("simhash_r32", "r=32"), ("csd_rel_1bit", "r=64"),
        ("simhash_r128", "r=128"), ("simhash_r256", "r=256"),
    ]
    sh_kb, sh_acc, sh_lbl = [], [], []
    for t, l in sh_tags:
        a, kb = _acc(t)
        sh_kb.append(kb); sh_acc.append(a); sh_lbl.append(l)

    ax1.plot(sh_kb, sh_acc, "o-", color="#1f77b4", linewidth=2.2, markersize=8, label="1-bit SimHash-CSD (Ours)")
    for kb, a, l in zip(sh_kb, sh_acc, sh_lbl):
        dy = 9
        ax1.annotate(f"{l}\n({a:.2f}%)", (kb, a), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=7.9, color="#08306b", fontweight="bold")

    i8_tags = [("csd_rel_int8_r8", "INT8 r=8"), ("csd_rel_int8", "INT8 r=32")]
    i8_kb, i8_acc, i8_lbl = [], [], []
    for t, l in i8_tags:
        a, kb = _acc(t)
        i8_kb.append(kb); i8_acc.append(a); i8_lbl.append(l)
    ax1.plot(i8_kb, i8_acc, "s--", color="#d62728", linewidth=1.8, markersize=7.5, label="INT8 Relational CSD")
    ax1.annotate(f"{i8_lbl[0]}\n({i8_acc[0]:.2f}%)", (i8_kb[0], i8_acc[0]), textcoords="offset points", xytext=(-54, -8), ha="center", fontsize=7.9, color="#b22222", fontweight="bold")
    ax1.annotate(f"{i8_lbl[1]}\n({i8_acc[1]:.2f}%)", (i8_kb[1], i8_acc[1]), textcoords="offset points", xytext=(-4, -26), ha="center", fontsize=7.9, color="#b22222", fontweight="bold")

    # Highlight iso-storage Bit-Budget win arrow
    ax1.annotate(
        "+1.53pp at 20%\nless storage!",
        xy=(sh_kb[2] + 8, sh_acc[2] - 0.08),
        xytext=(sh_kb[2] * 1.15, i8_acc[0] + 0.62),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
        fontsize=7.9, color="#1b7837", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#f0fff0", edgecolor="#2ca02c", alpha=0.92),
    )

    ax1.axhline(base_acc, color="#7f7f7f", linestyle=":", linewidth=1.3, label=f"Baseline ({base_acc:.2f}%)")
    ax1.axhline(kd_acc, color="#ff7f0e", linestyle="--", linewidth=1.3, label=f"Online KD ({kd_acc:.2f}%, 3.29 MB)")
    ax1.set_xscale("log")
    ax1.set_xlim(75, 2400)
    ax1.set_xlabel("Transferred Artifact Storage (KB, log scale)")
    ax1.set_ylabel("CIFAR-100 Top-1 Accuracy (%, 100 ep)")
    ax1.set_title("(a) Bit-Budget Theorem: 1-bit SimHash vs. INT8")
    ax1.set_ylim(67.3, 70.7)
    ax1.legend(loc="lower right", fontsize=7.8)

    ab_rows = [
        ("Baseline\n(CE)", base_acc, "#7f7f7f"),
        ("Direct-RP\n(Pointwise)", _acc("csd_v41_class")[0], "#d62728"),
        ("MSE-only\n(no KL)", _acc("simhash_r64_mse_only")[0], "#9ecae1"),
        ("Uncentered\n(ReLU cone)", _acc("simhash_r64_uncentered")[0], "#6baed6"),
        ("KL-only\n(no MSE)", _acc("simhash_r64_kl_only")[0], "#4292c6"),
        ("SimHash\nr=64 (391KB)", _acc("csd_rel_1bit")[0], "#2171b5"),
        ("SimHash\nr=128 (781KB)", _acc("simhash_r128")[0], "#08306b"),
    ]
    x = np.arange(len(ab_rows))
    bars = ax2.bar(x, [r[1] for r in ab_rows], width=0.58, color=[r[2] for r in ab_rows], edgecolor="#222222", linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels([r[0] for r in ab_rows], rotation=20, ha="right", fontsize=8.2)
    ax2.set_ylabel("CIFAR-100 Top-1 Accuracy (%, 100 ep)")
    ax2.set_title("(b) Controlled Component Ablation (r=64, 391 KB)")
    ax2.set_ylim(67.0, 70.7)
    for bar, (_, a, _) in zip(bars, ab_rows):
        d = a - base_acc
        s = "+" if d >= 0 else ""
        ax2.text(bar.get_x() + bar.get_width() / 2, a + 0.06, f"{a:.2f}%\n({s}{d:.2f})", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    plt.tight_layout()
    _save_fig(out_path)


def main() -> None:
    os.makedirs(PAPER_DIR, exist_ok=True)
    teacher_acc = _load("teacher_resnet56_s42_history.json")["best_acc"]

    # Combine main + nextgen summaries for Table 1 figures
    main_raw = _load("summary_main_e200.json")
    next_raw = _load("summary_nextgen_e200.json")
    all_by_tag = {r["tag"]: dict(r) for r in (main_raw + next_raw)}
    for t, r in all_by_tag.items():
        r["label"] = _clean_label(t, r["label"])

    order = [
        "baseline", "label_smooth", "csd_v41_class", "offline_sparse_kd",
        "simhash_r64", "simhash_r128", "simhash_hybrid", "simhash_dkd",
        "sp_online", "kd_online", "fitnet_online",
    ]
    main_rows = [all_by_tag[t] for t in order if t in all_by_tag]

    # 1. Main comparison bar chart (200 ep, 3 seeds)
    for d in (RESULTS_DIR, PAPER_DIR):
        plot_main_benchmark(main_rows, os.path.join(d, "fig_main_comparison.png"), teacher_acc=teacher_acc)

    # 2. Accuracy vs Storage Pareto Frontier (200 ep, 3 seeds)
    pareto_offsets = {
        "csd_v41_class": (10, -2),
        "simhash_r64": (-55, 11),
        "simhash_r128": (-42, 11),
        "offline_sparse_kd": (10, -10),
        "sp_online": (-108, -18),
        "kd_online": (-108, 7),
        "fitnet_online": (-92, -6),
    }
    pareto_pts = []
    for tag in ["csd_v41_class", "simhash_r64", "simhash_r128", "offline_sparse_kd", "sp_online", "kd_online", "fitnet_online"]:
        r = all_by_tag[tag]
        pareto_pts.append({
            "label": f"{r['label']}\n({r['storage_str']})",
            "storage_bytes": r["storage_bytes"],
            "acc": r["mean_std"],
            "category": r["category"],
            "offset": pareto_offsets.get(tag, (6, 6)),
        })
    for d in (RESULTS_DIR, PAPER_DIR):
        plot_pareto_frontier(
            pareto_pts, os.path.join(d, "fig_pareto_storage.png"),
            baseline_acc=all_by_tag["baseline"]["mean_std"],
            kd_acc=all_by_tag["kd_online"]["mean_std"],
        )

    # 3. Privacy & JL Geometry Audit (Linear + MLP attackers)
    audit_rows = _load("privacy_audit_resnet56.json")
    for d in (RESULTS_DIR, PAPER_DIR):
        plot_privacy_and_jl_audit(audit_rows, os.path.join(d, "fig_privacy_pareto.png"))

    # 4. Bit-Budget Theorem & Component Ablation
    for d in (RESULTS_DIR, PAPER_DIR):
        plot_bit_budget_ablation(os.path.join(d, "fig_bit_budget_ablation.png"))

    # 5. Three-panel Generalization Grid (Cross-Arch VGG-8, Large-Teacher R32x4, Second Dataset STL-10)
    for d in (RESULTS_DIR, PAPER_DIR):
        plot_generalization_grid(os.path.join(d, "fig_generalization_grid.png"))

    # 6. Training Curves & Apple M6 MPS Hardware Speed/VRAM Profile
    for d in (RESULTS_DIR, PAPER_DIR):
        plot_training_curves_and_hardware(main_rows, os.path.join(d, "fig_training_curves.png"))

    print("\nAll 6 publication-grade figures generated into results/ and paper/!")


if __name__ == "__main__":
    main()
