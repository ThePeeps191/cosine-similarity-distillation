"""Configuration for Cosine Similarity Distillation (CSD & CSD++).

Centralised hyperparameters, dataset constants, and hardware device selection
supporting CUDA, Apple Silicon MPS, and CPU.
"""

from __future__ import annotations

import torch


def get_best_device() -> torch.device:
    """Select the fastest available accelerator: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Config:
    """Container for all experiment hyperparameters."""

    def __init__(self) -> None:
        # Reproducibility
        self.seed: int = 42
        self.seeds: list[int] = [42, 123, 456]

        # Data
        self.batch_size: int = 128
        self.cifar_mean: tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
        self.cifar_std: tuple[float, float, float] = (0.2675, 0.2565, 0.2761)
        self.num_classes: int = 100
        self.image_size: int = 32

        # Teacher
        self.epochs_teacher: int = 200

        # Student (shared across methods)
        self.epochs_student: int = 200
        self.epochs_ablation: int = 100
        self.epochs_lambda_ablation: int = 200

        # Optimisation
        self.lr: float = 0.1
        self.momentum: float = 0.9
        self.weight_decay: float = 5e-4
        self.milestones: list[int] = [60, 120, 160]
        self.gamma: float = 0.2
        self.scheduler: str = "cosine"  # "cosine" or "multistep"

        # Hardware / performance
        self.device: torch.device = get_best_device()
        self.use_amp: bool = self.device.type in ("cuda", "mps")
        self.amp_dtype: torch.dtype = torch.bfloat16
        self.use_compile: bool = self.device.type in ("cuda", "mps")
        self.use_fast_loader: bool = True

        # Fingerprint (v4.1 legacy defaults preserved for backward compatibility)
        self.C: int = 64   # channels after layer 3 (before pooling)
        self.r: int = 128  # fingerprint dimension (v4.1 default; CSD++ uses r=32/64)
        self.n_augmentations: int = 8  # augmented views averaged per fingerprint
        self.warmup_epochs: int = 40  # lambda warmup: linear ramp for first N epochs
        self.use_multi_layer: bool = True  # if True, fingerprint layer2 (32-d) + layer3 (64-d)
        self.lambda_l2_weight: float = 0.2  # layer-2 (texture) gets smaller fingerprint weight
        self.lambda_l3_weight: float = 1.0  # layer-3 (object) gets full fingerprint weight
        self.use_contrastive: bool = False  # did not improve accuracy on CIFAR-100; kept for reproducibility
        self.contrastive_margin: float = 0.3
        self.contrastive_push_weight: float = 0.3

        # CSD++ hyperparameters
        self.csd_r: int = 32               # compressed JL sketch dimension (r < C -> true compression & privacy)
        self.csd_quant: str = "int8"       # "fp32", "fp16", "int8", "1bit"
        self.csd_orthogonal: bool = True   # QR-orthogonalized random projection (lower JL variance)
        self.csd_top_m: int = 12           # sparse top-m semantic anchor profile entries per sample
        self.csd_tau_rel: float = 0.15     # temperature for batch-relational cosine KL
        self.csd_tau_anchor: float = 0.18  # calibrated to match KD T=4 dark-knowledge entropy (~3.5 nats)
        self.csd_warmup_epochs: int = 20   # smoother warmup for CSD++

        # KD hyperparameters
        self.kd_temperature: float = 4.0
        self.kd_alpha: float = 0.9

        # FitNet beta sweep
        self.fitnet_beta_values: list[float] = [10.0, 50.0, 100.0]

        # Ablation ranges
        self.ablation_r_values: list[int] = [16, 32, 64, 128, 256]
        self.ablation_lambda_values: list[float] = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

        # Paths
        self.results_dir: str = "results"
        self.teacher_ckpt: str = "results/best_teacher.pth"
        self.baseline_ckpt: str = "results/best_student_baseline.pth"
        self.kd_ckpt: str = "results/best_student_kd.pth"
        self.fitnet_ckpt: str = "results/best_student_fitnet.pth"
        self.csd_sample_ckpt: str = "results/best_student_csd_per_sample.pth"
        self.csd_class_ckpt: str = "results/best_student_csd_per_class.pth"
        self.random_matrix_path: str = "results/random_matrix_R.pt"
        self.all_fingerprints_path: str = "results/all_fingerprints.pt"
        self.class_fingerprints_path: str = "results/class_fingerprints.pt"
        self.teacher_history_path: str = "results/teacher_history.json"
        self.results_summary_path: str = "results/results_summary.json"
