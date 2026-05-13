"""Configuration for Cosine Similarity Distillation (CSD) project.

Centralised hyperparameters, dataset constants, and device selection.
"""

from __future__ import annotations

import torch


class Config:
    """Container for all experiment hyperparameters."""

    def __init__(self) -> None:
        # Reproducibility
        self.seed: int = 42

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
        self.epochs_ablation: int = 100  # shorter runs for hyper-parameter sweeps (one LR drop at 60 + 40 epochs at lr=0.02)
        self.epochs_lambda_ablation: int = 200  # lambda ablation runs full length to match real CSD training

        # Optimisation
        self.lr: float = 0.1
        self.momentum: float = 0.9
        self.weight_decay: float = 5e-4
        self.milestones: list[int] = [60, 120, 160]
        self.gamma: float = 0.2

        # Fingerprint
        self.C: int = 64   # channels after layer 3 (before pooling)
        self.r: int = 128  # fingerprint dimension
        self.n_augmentations: int = 8  # augmented views averaged per fingerprint
        self.warmup_epochs: int = 40  # lambda warmup: linear ramp for first N epochs
        self.use_multi_layer: bool = True  # if True, fingerprint layer2 (32-d) + layer3 (64-d)
        self.lambda_l2_weight: float = 0.2  # layer-2 (texture) gets smaller fingerprint weight
        self.lambda_l3_weight: float = 1.0  # layer-3 (object) gets full fingerprint weight

        # KD hyperparameters
        self.kd_temperature: float = 4.0
        self.kd_alpha: float = 0.9

        # FitNet beta sweep
        self.fitnet_beta_values: list[float] = [10.0, 50.0, 100.0]

        # Ablation ranges
        self.ablation_r_values: list[int] = [16, 32, 64, 128, 256]
        self.ablation_lambda_values: list[float] = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

        # Device
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

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
