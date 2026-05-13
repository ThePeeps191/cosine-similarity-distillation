"""Ablation studies: fingerprint dimension *r* and CSD loss weight *λ*."""

from __future__ import annotations

import json
import os
from typing import Any

from config import Config
from generate_fingerprints import generate_fingerprints
from train_student_csd import train_student_csd


def run_fingerprint_ablation(config: Config) -> list[dict[str, Any]]:
    """Sweep over ``ablation_r_values``, re‑generate fingerprints, and train CSD.

    Returns a list of dicts with keys: ``r``, ``accuracy``, ``storage``.
    """
    os.makedirs(config.results_dir, exist_ok=True)
    results: list[dict[str, Any]] = []

    for r in config.ablation_r_values:
        print(f"\n{'='*60}")
        print(f"[Ablation r] Running with r = {r}")
        print(f"{'='*60}")

        generate_fingerprints(config, r_override=r)
        history = train_student_csd(
            config, use_per_class=False, lambda_csd=0.1, r_override=r,
            epochs_override=config.epochs_ablation, save_outputs=False,
        )
        acc = history["best_acc"]

        # Storage
        all_fp = f"{50000 * r * 4 / (1024 * 1024):.2f} MB"
        class_fp = f"{100 * r * 4 / 1024:.2f} KB"

        results.append({
            "r": r,
            "accuracy": acc,
            "storage_per_sample": all_fp,
            "storage_per_class": class_fp,
        })

    with open(os.path.join(config.results_dir, "ablation_r.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_lambda_ablation(config: Config) -> list[dict[str, Any]]:
    """Sweep over ``ablation_lambda_values``, training CSD per‑sample with r=128.

    Returns a list of dicts with keys: ``lambda``, ``accuracy``.
    """
    os.makedirs(config.results_dir, exist_ok=True)
    results: list[dict[str, Any]] = []

    for lam in config.ablation_lambda_values:
        print(f"\n{'='*60}")
        print(f"[Ablation λ] Running with λ = {lam}")
        print(f"{'='*60}")

        history = train_student_csd(
            config, use_per_class=False, lambda_csd=lam,
            epochs_override=config.epochs_lambda_ablation, save_outputs=False,
        )
        results.append({"lambda": lam, "accuracy": history["best_acc"]})

    with open(os.path.join(config.results_dir, "ablation_lambda.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_variant_comparison(config: Config) -> dict[str, Any]:
    """Compare per‑sample vs per‑class CSD variants.

    Returns dict with accuracy and storage for each variant.
    """
    os.makedirs(config.results_dir, exist_ok=True)

    # Per‑sample
    hist_sample = train_student_csd(
        config, use_per_class=False, lambda_csd=0.1,
    )
    # Per‑class
    hist_class = train_student_csd(
        config, use_per_class=True, lambda_csd=0.1,
    )

    result = {
        "csd_per_sample_acc": hist_sample["best_acc"],
        "csd_per_class_acc": hist_class["best_acc"],
    }
    with open(os.path.join(config.results_dir, "variant_comparison.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result
