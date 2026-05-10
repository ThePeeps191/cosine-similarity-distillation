"""Cosine Similarity Distillation — end‑to‑end orchestration.

Usage:
    python main.py

This script runs the full CSD pipeline:
    1. Train a ResNet‑56 teacher.
    2. Generate cosine‑similarity fingerprints.
    3. Train baselines (student‑only, KD, FitNet).
    4. Tune CSD hyperparameters and train CSD variants.
    5. Run ablation studies on fingerprint dimension *r* and loss weight *λ*.
    6. Collect results, produce tables & plots.
    7. Save a results summary and zip the output directory.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from typing import Any

from config import Config
from utils import set_seed

# ---------------------------------------------------------------------------
# Import all pipeline stages
# ---------------------------------------------------------------------------
from train_teacher import train_teacher
from generate_fingerprints import generate_fingerprints
from train_student_baseline import train_student_baseline
from train_student_kd import train_student_kd
from train_student_fitnet import train_student_fitnet
from train_student_csd import train_student_csd
from evaluate import collect_all_results
from ablation import run_fingerprint_ablation, run_lambda_ablation
from plot import (
    plot_teacher_training,
    plot_accuracy_comparison,
    plot_accuracy_barplot,
    plot_storage_comparison,
    plot_tsne,
    plot_fingerprint_ablation,
    plot_lambda_sensitivity,
    plot_training_loss_comparison,
    plot_accuracy_improvement,
    plot_fingerprint_alignment,
)
from data import get_cifar100_loaders


def _load_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _find_best_lambda(config: Config) -> float:
    """Use the lambda ablation results to find the best λ, or return default."""
    ablation_path = os.path.join(config.results_dir, "ablation_lambda.json")
    if os.path.exists(ablation_path):
        data = _load_json(ablation_path)
        if data:  # guard against empty list
            best = max(data, key=lambda d: d["accuracy"])
            return best["lambda"]
    return 0.1


def main() -> None:
    start_time = time.time()
    config = Config()
    set_seed(config.seed)
    os.makedirs(config.results_dir, exist_ok=True)

    print("=" * 70)
    print("COSINE SIMILARITY DISTILLATION (CSD) — Full Pipeline")
    print(f"Device: {config.device}")
    print(f"Results directory: {config.results_dir}")
    print("=" * 70)

    # =======================================================================
    # Phase 1 — Teacher training
    # =======================================================================
    print("\n" + "=" * 50)
    print("PHASE 1: Teacher Training")
    print("=" * 50)
    if os.path.exists(config.teacher_ckpt):
        print("[SKIP] Teacher checkpoint already exists.")
    else:
        train_teacher(config)

    # =======================================================================
    # Phase 2 — Fingerprint generation
    # =======================================================================
    print("\n" + "=" * 50)
    print("PHASE 2: Fingerprint Generation")
    print("=" * 50)
    if os.path.exists(config.all_fingerprints_path):
        print("[SKIP] Fingerprints already exist.")
    else:
        generate_fingerprints(config)

    # =======================================================================
    # Phase 3 — Baselines
    # =======================================================================
    print("\n" + "=" * 50)
    print("PHASE 3: Baseline Training")
    print("=" * 50)

    if os.path.exists(config.baseline_ckpt):
        print("[SKIP] Student‑only baseline checkpoint exists.")
    else:
        print("\n--- Student‑only (no distillation) ---")
        train_student_baseline(config)

    if os.path.exists(config.kd_ckpt):
        print("[SKIP] KD checkpoint exists.")
    else:
        print("\n--- KD (Hinton) ---")
        train_student_kd(config)

    if os.path.exists(config.fitnet_ckpt):
        print("[SKIP] FitNet checkpoint exists.")
    else:
        print("\n--- FitNet (feature MSE) ---")
        train_student_fitnet(config)

    # =======================================================================
    # Phase 4 — CSD training (with λ tuning)
    # =======================================================================
    print("\n" + "=" * 50)
    print("PHASE 4: CSD Training")
    print("=" * 50)

    # 4a. Lambda ablation → find best λ
    ablation_lambda_path = os.path.join(config.results_dir, "ablation_lambda.json")
    if os.path.exists(ablation_lambda_path):
        print("[SKIP] Lambda ablation already done.")
    else:
        print("\n--- Lambda tuning for CSD ---")
        run_lambda_ablation(config)

    best_lambda = _find_best_lambda(config)
    print(f"Best λ from tuning: {best_lambda}")

    # 4b. Train CSD per‑sample with best λ
    if os.path.exists(config.csd_sample_ckpt):
        print("[SKIP] CSD per‑sample checkpoint exists.")
    else:
        print(f"\n--- CSD (per‑sample, λ={best_lambda}) ---")
        train_student_csd(config, use_per_class=False, lambda_csd=best_lambda)

    # 4c. Train CSD per‑class with best λ
    if os.path.exists(config.csd_class_ckpt):
        print("[SKIP] CSD per‑class checkpoint exists.")
    else:
        print(f"\n--- CSD (per‑class, λ={best_lambda}) ---")
        train_student_csd(config, use_per_class=True, lambda_csd=best_lambda)

    # =======================================================================
    # Phase 5 — Ablation studies
    # =======================================================================
    print("\n" + "=" * 50)
    print("PHASE 5: Ablation Studies")
    print("=" * 50)

    ablation_r_path = os.path.join(config.results_dir, "ablation_r.json")
    if os.path.exists(ablation_r_path):
        print("[SKIP] Fingerprint dimension ablation done.")
    else:
        print("\n--- Fingerprint dimension (r) ablation ---")
        run_fingerprint_ablation(config)
        # The r ablation overwrites all_fingerprints.pt — restore default r=128
        print("\n--- Regenerating default fingerprints (r=128) after ablation ---")
        generate_fingerprints(config)

    # =======================================================================
    # Phase 6 — Results & plots
    # =======================================================================
    print("\n" + "=" * 50)
    print("PHASE 6: Results Collection & Plotting")
    print("=" * 50)

    _, test_loader = get_cifar100_loaders(config)
    results = collect_all_results(config, test_loader=test_loader)
    if not results:
        print("[FATAL] Cannot collect results — teacher checkpoint missing. Exiting.")
        return

    # Teacher training plots
    teacher_hist = _load_json(config.teacher_history_path)
    plot_teacher_training(teacher_hist, config)

    # Accuracy comparison line plot (load all histories)
    hist_dict: dict[str, list[float]] = {}
    hist_files = {
        "Student‑only": "baseline_history.json",
        "KD": "kd_history.json",
        "FitNet": "fitnet_history.json",
        "CSD per‑sample": "csd_per_sample_history.json",
        "CSD per‑class": "csd_per_class_history.json",
    }
    for label, fname in hist_files.items():
        path = os.path.join(config.results_dir, fname)
        if os.path.exists(path):
            hist_dict[label] = _load_json(path)["test_acc"]

    if hist_dict:
        plot_accuracy_comparison(hist_dict, config)

    # Bar plot
    acc_dict = {
        "Student‑only": results["accuracies"]["student_only"],
        "KD": results["accuracies"]["kd"],
        "FitNet": results["accuracies"]["fitnet"],
        "CSD per‑sample": results["accuracies"]["csd_per_sample"],
        "CSD per‑class": results["accuracies"]["csd_per_class"],
    }
    plot_accuracy_barplot(acc_dict, config)

    # Training loss comparison (uses same history JSONs as accuracy)
    loss_hist_dict: dict[str, list[float]] = {}
    for label, fname in hist_files.items():
        path = os.path.join(config.results_dir, fname)
        if os.path.exists(path):
            loss_hist_dict[label] = _load_json(path)["train_loss"]

    if loss_hist_dict:
        plot_training_loss_comparison(loss_hist_dict, config)

    # Accuracy improvement over baseline
    plot_accuracy_improvement(results["accuracies"], config)

    # Fingerprint alignment over training (CSD only)
    fp_align_dict: dict[str, list[float]] = {}
    for label_variant, fname in [("CSD per-sample", "csd_per_sample_history.json"),
                                  ("CSD per-class", "csd_per_class_history.json")]:
        path = os.path.join(config.results_dir, fname)
        if os.path.exists(path):
            data = _load_json(path)
            if "fp_alignment" in data:
                fp_align_dict[label_variant] = data["fp_alignment"]
    if fp_align_dict:
        plot_fingerprint_alignment(fp_align_dict, config)

    # Storage comparison
    plot_storage_comparison(config)

    # t‑SNE (baseline vs CSD per‑sample)
    if os.path.exists(config.csd_sample_ckpt):
        plot_tsne(config, config.baseline_ckpt, config.csd_sample_ckpt, test_loader)

    # Ablation plots
    if os.path.exists(ablation_r_path):
        r_data = _load_json(ablation_r_path)
        plot_fingerprint_ablation(r_data, config)

    if os.path.exists(ablation_lambda_path):
        lam_data = _load_json(ablation_lambda_path)
        plot_lambda_sensitivity(lam_data, config)

    # =======================================================================
    # Phase 7 — Summary & zip
    # =======================================================================
    print("\n" + "=" * 50)
    print("PHASE 7: Final Summary")
    print("=" * 50)

    elapsed = (time.time() - start_time) / 3600
    print(f"Total runtime: {elapsed:.2f} hours")
    print(f"Results saved in: {os.path.abspath(config.results_dir)}")

    # Create zip archive
    zip_path = "csd_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        results_abs = os.path.abspath(config.results_dir)
        for root, _, files in os.walk(config.results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, results_abs)
                zf.write(file_path, arcname=arcname)

    print(f"Results zipped to: {zip_path}")

    # Final summary string
    acc = results["accuracies"]
    def _fa(v: float | None) -> str:
        return f"{v:.2f}%" if v is not None else "N/A"
    def _di(a: float | None, b: float | None) -> str:
        if a is not None and b is not None:
            return f"{a - b:.2f}%"
        return "N/A"

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Teacher accuracy:        {_fa(acc['teacher'])}")
    print(f"  Student‑only:            {_fa(acc['student_only'])}")
    print(f"  KD (Hinton):             {_fa(acc['kd'])}")
    print(f"  FitNet:                  {_fa(acc['fitnet'])}")
    print(f"  CSD per‑sample:          {_fa(acc['csd_per_sample'])}")
    print(f"  CSD per‑class:           {_fa(acc['csd_per_class'])}")
    print(f"  KD improvement:          {_di(acc['kd'], acc['student_only'])}")
    print(f"  CSD improvement:         {_di(acc['csd_per_sample'], acc['student_only'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
