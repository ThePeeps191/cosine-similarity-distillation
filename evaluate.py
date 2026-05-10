"""Collect results from all methods, print a comparison table, and return a dict."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from config import Config
from data import get_cifar100_loaders
from models import resnet20, resnet56
from utils import get_model_size_mb, get_tensor_size


@torch.no_grad()
def _eval_ckpt(
    model: torch.nn.Module, loader, device: torch.device,
) -> float:
    model.eval()
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    if total == 0:
        return 0.0
    return 100.0 * correct / total


def collect_all_results(config: Config, test_loader=None) -> dict[str, Any]:
    """Evaluate all saved models and build the main results table.

    Returns a dict with keys: ``table`` (DataFrame), ``accuracies`` (dict),
    ``storage`` (dict), ``teacher_forwards`` (dict).
    """
    os.makedirs(config.results_dir, exist_ok=True)
    if test_loader is None:
        _, test_loader = get_cifar100_loaders(config)
    device = config.device

    # ------------------------------------------------------------------
    # Teacher
    # ------------------------------------------------------------------
    if not os.path.exists(config.teacher_ckpt):
        print("[WARN] Teacher checkpoint missing; cannot evaluate.")
        return {}
    teacher = resnet56(num_classes=config.num_classes).to(device)
    teacher.load_state_dict(torch.load(config.teacher_ckpt, map_location=device))
    teacher.eval()
    teacher_acc = _eval_ckpt(teacher, test_loader, device)
    teacher_mb = get_model_size_mb(teacher)

    # ------------------------------------------------------------------
    # Student-only (baseline)
    # ------------------------------------------------------------------
    baseline_acc = None
    if os.path.exists(config.baseline_ckpt):
        s_baseline = resnet20(num_classes=config.num_classes).to(device)
        s_baseline.load_state_dict(torch.load(config.baseline_ckpt, map_location=device))
        baseline_acc = _eval_ckpt(s_baseline, test_loader, device)

    # ------------------------------------------------------------------
    # KD
    # ------------------------------------------------------------------
    kd_acc = None
    if os.path.exists(config.kd_ckpt):
        s_kd = resnet20(num_classes=config.num_classes).to(device)
        s_kd.load_state_dict(torch.load(config.kd_ckpt, map_location=device))
        kd_acc = _eval_ckpt(s_kd, test_loader, device)

    # ------------------------------------------------------------------
    # FitNet
    # ------------------------------------------------------------------
    fitnet_acc = None
    if os.path.exists(config.fitnet_ckpt):
        s_fitnet = resnet20(num_classes=config.num_classes).to(device)
        s_fitnet.load_state_dict(torch.load(config.fitnet_ckpt, map_location=device))
        fitnet_acc = _eval_ckpt(s_fitnet, test_loader, device)

    # ------------------------------------------------------------------
    # CSD per‑sample
    # ------------------------------------------------------------------
    csd_sample_acc = None
    if os.path.exists(config.csd_sample_ckpt):
        s_csd_sample = resnet20(num_classes=config.num_classes).to(device)
        s_csd_sample.load_state_dict(torch.load(config.csd_sample_ckpt, map_location=device))
        csd_sample_acc = _eval_ckpt(s_csd_sample, test_loader, device)

    # ------------------------------------------------------------------
    # CSD per‑class
    # ------------------------------------------------------------------
    csd_class_acc = None
    if os.path.exists(config.csd_class_ckpt):
        s_csd_class = resnet20(num_classes=config.num_classes).to(device)
        s_csd_class.load_state_dict(torch.load(config.csd_class_ckpt, map_location=device))
        csd_class_acc = _eval_ckpt(s_csd_class, test_loader, device)

    # ------------------------------------------------------------------
    # Storage sizes
    # ------------------------------------------------------------------
    R_mat_size_str = "N/A"
    storage_per_sample = "N/A"
    storage_per_class = "N/A"

    if os.path.exists(config.random_matrix_path):
        R_mat = torch.load(config.random_matrix_path, map_location="cpu")
        R_mat_size_str = get_tensor_size(R_mat)

    if os.path.exists(config.all_fingerprints_path):
        all_fp = torch.load(config.all_fingerprints_path, map_location="cpu")
        storage_per_sample = get_tensor_size(all_fp)
    if os.path.exists(config.class_fingerprints_path):
        class_fp = torch.load(config.class_fingerprints_path, map_location="cpu")
        storage_per_class = get_tensor_size(class_fp)

    storage_kd = f"{teacher_mb:.2f} MB (teacher)"
    storage_fitnet = f"{teacher_mb:.2f} MB (teacher)"
    storage_csd_sample = f"{storage_per_sample} (fingerprints + {R_mat_size_str} R)"
    storage_csd_class = f"{storage_per_class} (fingerprints + {R_mat_size_str} R)"

    # ------------------------------------------------------------------
    # Teacher forwards during training
    # ------------------------------------------------------------------
    approx_batches = -(-50000 // config.batch_size)  # ceil(50000/128) = 391
    kd_forwards = approx_batches * config.epochs_student
    fitnet_forwards = approx_batches * config.epochs_student * len(config.fitnet_beta_values)

    # ------------------------------------------------------------------
    # Build table (with fallback for unavailable results)
    # ------------------------------------------------------------------
    def _fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    rows = [
        ["Student-only (no distillation)", _fmt(baseline_acc), "--", "0"],
        ["KD (Hinton)", _fmt(kd_acc), storage_kd, f"{kd_forwards:,}+"],
        ["FitNet (feature MSE)", _fmt(fitnet_acc), storage_fitnet, f"{fitnet_forwards:,}+"],
        ["CSD (per-sample)", _fmt(csd_sample_acc), storage_csd_sample, "1 (precomputation)"],
        ["CSD (per-class)", _fmt(csd_class_acc), storage_csd_class, "1 (precomputation)"],
    ]

    accuracies = {
        "teacher": teacher_acc,
        "student_only": baseline_acc,
        "kd": kd_acc,
        "fitnet": fitnet_acc,
        "csd_per_sample": csd_sample_acc,
        "csd_per_class": csd_class_acc,
    }

    # ------------------------------------------------------------------
    # Print table (use pandas if available, otherwise plain text)
    # ------------------------------------------------------------------
    try:
        import pandas as pd
        df = pd.DataFrame(
            rows,
            columns=["Method", "Top-1 Accuracy (%)", "Storage for Transfer", "Teacher Forwards"],
        )
        print("\n" + "=" * 80)
        print("MAIN RESULTS TABLE")
        print("=" * 80)
        print(df.to_string(index=False))
        table_obj = df
    except ImportError:
        print("\n" + "=" * 80)
        print("MAIN RESULTS TABLE")
        print("=" * 80)
        print(f"{'Method':<30} {'Top-1 Acc (%)':<16} {'Storage':<35} {'Teacher Forwards'}")
        print("-" * 95)
        for row in rows:
            print(f"{row[0]:<30} {row[1]:<16} {row[2]:<35} {row[3]}")
        table_obj = rows

    results: dict[str, Any] = {
        "table": table_obj,
        "accuracies": accuracies,
        "storage": {
            "teacher_mb": teacher_mb,
            "all_fingerprints": storage_per_sample,
            "class_fingerprints": storage_per_class,
        },
        "teacher_forwards": {
            "kd": kd_forwards,
            "fitnet": fitnet_forwards,
            "csd": 1,
        },
    }

    # Save to JSON
    with open(config.results_summary_path, "w") as f:
        json_ready = {}
        for k, v in results.items():
            if hasattr(v, "to_dict"):
                json_ready[k] = v.to_dict()
            elif isinstance(v, dict):
                json_ready[k] = {str(inner_k): inner_v for inner_k, inner_v in v.items()}
            else:
                json_ready[k] = str(v)
        json.dump(json_ready, f, indent=2, default=str)

    return results
