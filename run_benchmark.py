"""Resumable multi-stage experimental benchmark orchestrator for SimHash-CSD / CSD++.

Stages:
    1. screening   : Fast 100-epoch screening of CSD++ variants vs baseline, KD, and v4.1
    2. screening2  : Targeted SimHash-CSD & hybrid screening (100 epochs)
    3. ablations   : Controlled component ablations + (r, quant) storage-privacy-accuracy Pareto sweep
    4. main        : Multi-seed (3 seeds) 200-epoch benchmark on ResNet-56 -> ResNet-20
    5. cross_arch  : Multi-seed (3 seeds) cross-architecture benchmark on ResNet-56 -> VGG-8 (C=64 -> C=512)
    6. summarize   : Aggregate all JSONs, print Markdown/LaTeX tables, and render all publication figures
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from config import Config
from experiments import (
    prepare_teacher_cache,
    run_privacy_audit_cli,
    train_student,
    train_teacher_model,
)
from plot_csd_pp import (
    plot_component_ablation,
    plot_cross_architecture,
    plot_main_benchmark,
    plot_pareto_frontier,
    plot_privacy_and_jl_audit,
)
from utils import format_bytes


def _history_path(results_dir: str, tag: str, student: str, seed: int, epochs: int) -> str:
    return os.path.join(results_dir, f"{tag}_{student}_s{seed}_e{epochs}_history.json")


def _run_if_missing(
    config: Config,
    teacher_data: dict[str, Any],
    spec: dict[str, Any],
    student: str,
    seed: int,
    epochs: int,
    teacher: str = "resnet56",
    save_ckpt: bool = True,
) -> dict[str, Any]:
    tag = spec["tag"]
    hpath = _history_path(config.results_dir, tag, student, seed, epochs)
    if os.path.exists(hpath):
        with open(hpath, "r") as f:
            h = json.load(f)
        print(
            f"[SKIP] {tag} ({student}, s={seed}, e={epochs}) -> "
            f"Std {h['best_test_acc']:.2f}%, TTA {h['tta_test_acc']:.2f}%, "
            f"Storage {h['storage_str']}",
            flush=True,
        )
        return h

    kwargs = {k: v for k, v in spec.items() if k not in ("tag", "label", "category")}
    return train_student(
        config,
        student_arch=student,
        teacher_arch=teacher,
        teacher_data=teacher_data,
        epochs=epochs,
        seed=seed,
        tag=tag,
        save_ckpt=save_ckpt,
        **kwargs,
    )


def get_screening_specs() -> list[dict[str, Any]]:
    return [
        {"tag": "baseline", "method": "baseline", "label": "Student-only (CE)", "category": "baseline"},
        {"tag": "kd_online", "method": "kd_online", "label": "Online KD (Hinton)", "category": "online"},
        {
            "tag": "csd_v41_class", "method": "csd_v41_class",
            "r": 128, "quant": "fp32", "center": False, "use_projector": False,
            "use_multi_layer": True, "lam_proj": 0.3, "warmup_epochs": 20,
            "label": "Original CSD v4.1 (per-class)", "category": "csd_old",
        },
        {
            "tag": "offline_class_kd", "method": "offline_class_kd",
            "kd_alpha": 0.5, "kd_temp": 4.0,
            "label": "Offline Class-Logit KD", "category": "offline_baseline",
        },
        {
            "tag": "offline_sparse_kd", "method": "offline_sparse_kd",
            "top_m": 12, "kd_alpha": 0.9, "kd_temp": 4.0,
            "label": "Offline Sparse-12 Logit KD", "category": "offline_baseline",
        },
        {
            "tag": "csd_proto", "method": "csd_proto",
            "r": 32, "quant": "int8", "orthogonal": True, "center": True,
            "use_projector": True, "use_multi_layer": False,
            "lam_proj": 0.5, "lam_anchor": 6.0, "gram_weight": 1.0,
            "tau_anchor": 0.18, "warmup_epochs": 10,
            "label": "CSD-Proto (Ours, 23 KB)", "category": "csd_new",
        },
        {
            "tag": "csd_rel_int8", "method": "csd_rel",
            "r": 32, "quant": "int8", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15,
            "warmup_epochs": 10,
            "label": "CSD-Rel INT8 (Ours, r=32)", "category": "csd_new",
        },
        {
            "tag": "csd_rel_1bit", "method": "csd_rel",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15,
            "warmup_epochs": 10,
            "label": "SimHash-CSD (1-bit, r=64, 391 KB)", "category": "csd_new",
        },
        {
            "tag": "csd_anchor_sample", "method": "csd_anchor_sample",
            "top_m": 12, "center": True, "lam_anchor": 8.0,
            "gram_weight": 1.0, "tau_anchor": 0.18, "warmup_epochs": 10,
            "label": "CSD-Anchor Sparse-12 (Ours)", "category": "csd_new",
        },
        {
            "tag": "csd_pp", "method": "csd_pp",
            "r": 32, "quant": "int8", "orthogonal": True, "top_m": 12,
            "center": True, "use_projector": True,
            "lam_proj": 0.5, "lam_rel": 3.0, "lam_anchor": 8.0,
            "gram_weight": 1.0, "tau_rel": 0.15, "tau_anchor": 0.18,
            "warmup_epochs": 10,
            "label": "CSD++ Full (Ours)", "category": "csd_new",
        },
    ]


def get_screening2_specs() -> list[dict[str, Any]]:
    return [
        {
            "tag": "simhash_r32", "method": "csd_rel",
            "r": 32, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15,
            "warmup_epochs": 10,
            "label": "SimHash-CSD (1-bit, r=32, 195 KB)", "category": "csd_new",
        },
        {
            "tag": "simhash_r128", "method": "csd_rel",
            "r": 128, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15,
            "warmup_epochs": 10,
            "label": "SimHash-CSD (1-bit, r=128, 781 KB)", "category": "csd_new",
        },
        {
            "tag": "simhash_multi_r64", "method": "csd_rel",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": True, "lam_rel": 5.0, "l2_weight": 0.5, "tau_rel": 0.15,
            "warmup_epochs": 10,
            "label": "SimHash-CSD Multi-Layer (r=64, 781 KB)", "category": "csd_new",
        },
        {
            "tag": "simhash_r64_strong", "method": "csd_rel",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 10.0, "tau_rel": 0.12,
            "warmup_epochs": 10,
            "label": "SimHash-CSD Strong (r=64, lam=10, 391 KB)", "category": "csd_new",
        },
        {
            "tag": "simhash_hybrid", "method": "csd_hybrid",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "top_m": 8,
            "lam_rel": 4.0, "tau_rel": 0.15, "kd_alpha": 0.7, "kd_temp": 4.0,
            "warmup_epochs": 10,
            "label": "SimHash-CSD + Sparse-8 Logits (1.34 MB)", "category": "csd_new",
        },
    ]


def get_ablation_specs() -> list[dict[str, Any]]:
    """Controlled component ablations + complete 1-bit vs INT8 bit-budget Pareto curve (100 ep, s=42)."""
    return [
        # Extreme compression & high-res Pareto points for 1-bit SimHash
        {
            "tag": "simhash_r16", "method": "csd_rel",
            "r": 16, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 10,
            "label": "SimHash-CSD (1-bit, r=16, 98 KB)", "category": "csd_new",
        },
        {
            "tag": "simhash_r256", "method": "csd_rel",
            "r": 256, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 10,
            "label": "SimHash-CSD (1-bit, r=256, 1.53 MB)", "category": "csd_new",
        },
        # Iso-storage Bit-Budget Theorem test: INT8 r=8 (488 KB) vs 1-bit SimHash r=64 (391 KB)
        {
            "tag": "csd_rel_int8_r8", "method": "csd_rel",
            "r": 8, "quant": "int8", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 10,
            "label": "CSD-Rel INT8 (r=8, 488 KB, iso-storage)", "category": "csd_new",
        },
        # Component ablations on SimHash-CSD (r=64, 391 KB)
        {
            "tag": "simhash_r64_uncentered", "method": "csd_rel",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": False,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 10,
            "label": "SimHash-CSD w/o Centering", "category": "ablation",
        },
        {
            "tag": "simhash_r64_gaussian", "method": "csd_rel",
            "r": 64, "quant": "1bit", "orthogonal": False, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 10,
            "label": "SimHash-CSD w/ Gaussian R (non-orth)", "category": "ablation",
        },
        {
            "tag": "simhash_r64_mse_only", "method": "csd_rel",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15,
            "rel_kl_weight": 0.0, "rel_mse_weight": 1.0, "warmup_epochs": 10,
            "label": "SimHash-CSD MSE-only (w/o Rel-KL)", "category": "ablation",
        },
        {
            "tag": "simhash_r64_kl_only", "method": "csd_rel",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15,
            "rel_kl_weight": 1.0, "rel_mse_weight": 0.0, "warmup_epochs": 10,
            "label": "SimHash-CSD KL-only (w/o Gram-MSE)", "category": "ablation",
        },
    ]


def get_main_specs() -> list[dict[str, Any]]:
    """Main Table 1: ResNet-56 -> ResNet-20 (200 epochs, 3 seeds)."""
    return [
        {"tag": "baseline", "method": "baseline", "label": "Student-only (CE)", "category": "baseline"},
        {"tag": "label_smooth", "method": "label_smooth", "label": "Label Smoothing (eps=0.1)", "category": "baseline"},
        {"tag": "kd_online", "method": "kd_online", "kd_alpha": 0.9, "kd_temp": 4.0, "label": "Online KD (Hinton)", "category": "online"},
        {"tag": "fitnet_online", "method": "fitnet_online", "fitnet_beta": 10.0, "label": "Online FitNet (feature MSE)", "category": "online"},
        {
            "tag": "offline_sparse_kd", "method": "offline_sparse_kd",
            "top_m": 12, "kd_alpha": 0.9, "kd_temp": 4.0,
            "label": "Offline Sparse-12 Logit KD", "category": "offline_baseline",
        },
        {
            "tag": "csd_v41_class", "method": "csd_v41_class",
            "r": 128, "quant": "fp32", "center": False, "use_projector": False,
            "use_multi_layer": True, "lam_proj": 0.3, "warmup_epochs": 40,
            "label": "Original CSD v4.1 (per-class)", "category": "csd_old",
        },
        {
            "tag": "simhash_r64", "method": "csd_rel",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 20,
            "label": "SimHash-CSD (1-bit, r=64)", "category": "csd_new",
        },
        {
            "tag": "simhash_r128", "method": "csd_rel",
            "r": 128, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 20,
            "label": "SimHash-CSD (1-bit, r=128)", "category": "csd_new",
        },
        {
            "tag": "simhash_hybrid", "method": "csd_hybrid",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "top_m": 12,
            "lam_rel": 4.0, "tau_rel": 0.15, "kd_alpha": 0.8, "kd_temp": 4.0, "warmup_epochs": 20,
            "label": "SimHash-CSD + Sparse-12 Logits", "category": "csd_new",
        },
    ]


def get_nextgen_specs() -> list[dict[str, Any]]:
    """High-impact additions for Table 1: Offline DKD + SimHash-CSD Hybrid and Online SP Oracle."""
    return [
        {
            "tag": "simhash_dkd", "method": "csd_dkd_hybrid",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "top_m": 12,
            "lam_rel": 4.0, "tau_rel": 0.15, "kd_temp": 4.0,
            "dkd_alpha": 1.0, "dkd_beta": 4.0, "queue_size": 0, "warmup_epochs": 20,
            "label": "SimHash-CSD + Offline DKD", "category": "csd_new",
        },
        {
            "tag": "sp_online", "method": "sp_online",
            "center": True, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 20,
            "label": "Online SP (Full-Precision Oracle)", "category": "online",
        },
    ]


def get_large_teacher_specs() -> list[dict[str, Any]]:
    """High-compression cross-width benchmark: ResNet-32x4 (28.35 MB, C=256) -> ResNet-20 (1.06 MB, C=64)."""
    return [
        {"tag": "lt_baseline", "method": "baseline", "label": "Student-only (ResNet-20)", "category": "baseline"},
        {"tag": "lt_kd_online", "method": "kd_online", "kd_alpha": 0.9, "kd_temp": 4.0, "label": "Online KD (ResNet-32x4)", "category": "online"},
        {
            "tag": "lt_offline_sparse_kd", "method": "offline_sparse_kd",
            "top_m": 12, "kd_alpha": 0.9, "kd_temp": 4.0,
            "label": "Offline Sparse-12 Logit KD", "category": "offline_baseline",
        },
        {
            "tag": "lt_simhash_r64", "method": "csd_rel",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 15,
            "label": "SimHash-CSD (1-bit, r=64)", "category": "csd_new",
        },
        {
            "tag": "lt_simhash_r128", "method": "csd_rel",
            "r": 128, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 15,
            "label": "SimHash-CSD (1-bit, r=128)", "category": "csd_new",
        },
    ]


def get_cross_arch_specs() -> list[dict[str, Any]]:
    """Cross-Architecture Table 2: ResNet-56 (C=64) -> VGG-8-BN (C=512), 100 epochs, 3 seeds."""
    return [
        {"tag": "baseline", "method": "baseline", "lr_override": 0.05, "label": "Student-only (VGG-8)", "category": "baseline"},
        {"tag": "kd_online", "method": "kd_online", "kd_alpha": 0.9, "kd_temp": 4.0, "lr_override": 0.05, "label": "Online KD (Hinton)", "category": "online"},
        {
            "tag": "offline_sparse_kd", "method": "offline_sparse_kd",
            "top_m": 12, "kd_alpha": 0.9, "kd_temp": 4.0, "lr_override": 0.05,
            "label": "Offline Sparse-12 Logit KD", "category": "offline_baseline",
        },
        {
            "tag": "simhash_r64", "method": "csd_rel",
            "r": 64, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 10,
            "lr_override": 0.05,
            "label": "SimHash-CSD (1-bit, r=64)", "category": "csd_new",
        },
        {
            "tag": "simhash_r128", "method": "csd_rel",
            "r": 128, "quant": "1bit", "orthogonal": True, "center": True,
            "use_multi_layer": False, "lam_rel": 5.0, "tau_rel": 0.15, "warmup_epochs": 10,
            "lr_override": 0.05,
            "label": "SimHash-CSD (1-bit, r=128)", "category": "csd_new",
        },
    ]


def run_screening(config: Config, epochs: int = 100, seed: int = 42, specs: list[dict[str, Any]] | None = None, title: str = "SCREENING") -> None:
    teacher_data = prepare_teacher_cache(config, teacher_arch="resnet56")
    if specs is None:
        run_privacy_audit_cli(config, teacher_arch="resnet56")
        specs = get_screening_specs()

    results = []
    for spec in specs:
        h = _run_if_missing(config, teacher_data, spec, student="resnet20", seed=seed, epochs=epochs, save_ckpt=False)
        results.append((spec["label"], h))

    print("\n" + "=" * 96)
    print(f"{title} RESULTS (ResNet-56 -> ResNet-20, {epochs} epochs, seed={seed})")
    print("=" * 96)
    print(f"{'Method':<44s} {'Std Top-1':>10s} {'TTA Top-1':>10s} {'Storage':>12s} {'s/ep':>7s}")
    print("-" * 96)
    for label, h in results:
        print(
            f"{label:<44s} {h['best_test_acc']:9.2f}% {h['tta_test_acc']:9.2f}% "
            f"{h['storage_str']:>12s} {h['sec_per_epoch']:6.2f}s"
        )


def run_multi_seed_stage(
    config: Config,
    specs: list[dict[str, Any]],
    student: str,
    seeds: list[int],
    epochs: int,
    title: str,
    teacher: str = "resnet56",
    save_ckpt: bool = True,
) -> list[dict[str, Any]]:
    teacher_data = prepare_teacher_cache(config, teacher_arch=teacher)
    summary_rows: list[dict[str, Any]] = []

    for spec in specs:
        stds, ttas, lasts, times = [], [], [], []
        storage_bytes = 0
        storage_str = "0 B"
        for seed in seeds:
            h = _run_if_missing(
                config, teacher_data, spec, student=student, seed=seed, epochs=epochs,
                teacher=teacher, save_ckpt=(save_ckpt and seed == seeds[0]),
            )
            stds.append(h["best_test_acc"])
            ttas.append(h["tta_test_acc"])
            lasts.append(h["last_test_acc"])
            times.append(h["sec_per_epoch"])
            storage_bytes = h["storage_bytes"]
            storage_str = h["storage_str"]

        # Fix legacy 0 B display for online KD if an old json exists
        if spec["method"] in ("kd_online", "fitnet_online") and storage_bytes == 0:
            storage_bytes = 861620 * 4
            storage_str = format_bytes(storage_bytes)

        summary_rows.append({
            "tag": spec["tag"],
            "label": spec["label"],
            "category": spec["category"],
            "seeds": seeds,
            "mean_std": float(np.mean(stds)),
            "std_std": float(np.std(stds, ddof=1)) if len(stds) > 1 else 0.0,
            "mean_last": float(np.mean(lasts)),
            "std_last": float(np.std(lasts, ddof=1)) if len(lasts) > 1 else 0.0,
            "mean_tta": float(np.mean(ttas)),
            "std_tta": float(np.std(ttas, ddof=1)) if len(ttas) > 1 else 0.0,
            "mean_sec_ep": float(np.mean(times)),
            "storage_bytes": storage_bytes,
            "storage_str": storage_str,
            "per_seed_std": stds,
        })

    print("\n" + "=" * 108)
    print(f"{title} ({student}, {epochs} epochs, seeds={seeds})")
    print("=" * 108)
    print(f"{'Method':<32s} {'Std Top-1 (%)':>16s} {'Last Top-1 (%)':>16s} {'TTA Top-1 (%)':>16s} {'Storage':>11s} {'s/ep':>7s}")
    print("-" * 108)
    for r in summary_rows:
        print(
            f"{r['label']:<32s} "
            f"{r['mean_std']:6.2f} +/- {r['std_std']:4.2f}% "
            f"{r['mean_last']:6.2f} +/- {r['std_last']:4.2f}% "
            f"{r['mean_tta']:6.2f} +/- {r['std_tta']:4.2f}% "
            f"{r['storage_str']:>11s} {r['mean_sec_ep']:6.2f}s"
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="SimHash-CSD Benchmark Orchestrator")
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["screening", "screening2", "ablations", "main", "nextgen", "cross_arch", "large_teacher"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.scheduler = "cosine"

    if args.stage == "screening":
        run_screening(cfg, epochs=args.epochs or 100, seed=(args.seeds[0] if args.seeds else 42))
    elif args.stage == "screening2":
        run_screening(cfg, epochs=args.epochs or 100, seed=(args.seeds[0] if args.seeds else 42),
                      specs=get_screening2_specs(), title="SCREENING 2 (SIMHASH & HYBRID)")
    elif args.stage == "ablations":
        run_screening(cfg, epochs=args.epochs or 100, seed=(args.seeds[0] if args.seeds else 42),
                      specs=get_ablation_specs(), title="CONTROLLED ABLATIONS & BIT-BUDGET PARETO")
    elif args.stage == "main":
        seeds = args.seeds or [42, 123, 456]
        epochs = args.epochs or 200
        rows = run_multi_seed_stage(
            cfg, get_main_specs(), student="resnet20", seeds=seeds, epochs=epochs,
            title="MAIN BENCHMARK: ResNet-56 -> ResNet-20",
        )
        with open(os.path.join(cfg.results_dir, f"summary_main_e{epochs}.json"), "w") as f:
            json.dump(rows, f, indent=2)
    elif args.stage == "nextgen":
        seeds = args.seeds or [42, 123, 456]
        epochs = args.epochs or 200
        rows = run_multi_seed_stage(
            cfg, get_nextgen_specs(), student="resnet20", seeds=seeds, epochs=epochs,
            title="NEXT-GEN BENCHMARK: Online SP Oracle + SimHash-CSD-Q + Offline DKD",
        )
        with open(os.path.join(cfg.results_dir, f"summary_nextgen_e{epochs}.json"), "w") as f:
            json.dump(rows, f, indent=2)
    elif args.stage == "cross_arch":
        seeds = args.seeds or [42, 123, 456]
        epochs = args.epochs or 100
        rows = run_multi_seed_stage(
            cfg, get_cross_arch_specs(), student="vgg8", seeds=seeds, epochs=epochs,
            title="CROSS-ARCHITECTURE: ResNet-56 (C=64) -> VGG-8 (C=512)",
        )
        with open(os.path.join(cfg.results_dir, f"summary_cross_arch_e{epochs}.json"), "w") as f:
            json.dump(rows, f, indent=2)
    elif args.stage == "large_teacher":
        seeds = args.seeds or [42, 123, 456]
        epochs = args.epochs or 100
        t_ckpt = os.path.join(cfg.results_dir, "teacher_resnet32x4_s42.pth")
        if not os.path.exists(t_ckpt):
            train_teacher_model(cfg, arch="resnet32x4", epochs=120, seed=42, save_path=t_ckpt)
        rows = run_multi_seed_stage(
            cfg, get_large_teacher_specs(), student="resnet20", seeds=seeds, epochs=epochs,
            teacher="resnet32x4",
            title="HIGH-COMPRESSION CROSS-WIDTH: ResNet-32x4 (28.35 MB, C=256) -> ResNet-20 (C=64)",
        )
        with open(os.path.join(cfg.results_dir, f"summary_large_teacher_e{epochs}.json"), "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
