"""SimHash-CSD — turnkey end-to-end entry point.

Usage:
    python main.py

Runs the reproducible SimHash-CSD pipeline (auto-selecting CUDA > Apple MPS > CPU):
    1. Train ResNet-56 teacher (if checkpoint missing).
    2. Extract offline 8-view teacher features & run Linear + MLP privacy audit.
    3. Run controlled component ablations & Bit-Budget Pareto curve.
    4. Run 3-seed 200-epoch CIFAR-100 main benchmark + Online SP Oracle.
    5. Run 3-seed cross-architecture (ResNet-56 -> VGG-8-BN) benchmark.
    6. Generate all 6 publication-grade figures into results/ and paper/.

Every step is idempotent and skips any run whose JSON history already exists.
"""

from __future__ import annotations

import os
import time

from config import Config
from experiments import prepare_teacher_cache, run_privacy_audit_cli, train_teacher_model
from generate_all_figures import main as generate_figures
from run_benchmark import (
    get_ablation_specs,
    get_cross_arch_specs,
    get_main_specs,
    get_nextgen_specs,
    run_multi_seed_stage,
    run_screening,
)


def main() -> None:
    t0 = time.time()
    cfg = Config()
    cfg.scheduler = "cosine"
    os.makedirs(cfg.results_dir, exist_ok=True)

    print("=" * 72)
    print(f"SimHash-CSD End-to-End Pipeline | Device: {cfg.device}")
    print("=" * 72)

    t_ckpt = os.path.join(cfg.results_dir, "teacher_resnet56_s42.pth")
    if not os.path.exists(t_ckpt):
        train_teacher_model(cfg, arch="resnet56", epochs=200, seed=42, save_path=t_ckpt)
    else:
        print(f"[SKIP] Teacher checkpoint exists: {t_ckpt}")

    prepare_teacher_cache(cfg, teacher_arch="resnet56", teacher_ckpt=t_ckpt)
    run_privacy_audit_cli(cfg, teacher_arch="resnet56")

    run_screening(cfg, epochs=100, seed=42, specs=get_ablation_specs(), title="CONTROLLED ABLATIONS")
    run_multi_seed_stage(cfg, get_main_specs(), student="resnet20", seeds=[42, 123, 456], epochs=200, title="MAIN BENCHMARK")
    run_multi_seed_stage(cfg, get_nextgen_specs(), student="resnet20", seeds=[42, 123, 456], epochs=200, title="ONLINE SP & HYBRID")
    run_multi_seed_stage(cfg, get_cross_arch_specs(), student="vgg8", seeds=[42, 123, 456], epochs=100, title="CROSS-ARCHITECTURE")

    generate_figures()
    print(f"\nPipeline completed in {(time.time() - t0) / 3600:.2f} hours.")


if __name__ == "__main__":
    main()
