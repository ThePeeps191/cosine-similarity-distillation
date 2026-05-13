"""Tests for config.py — verify hyperparameter values and types."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import Config


def test_config_defaults():
    """All Config fields should have the expected types and ranges."""
    c = Config()

    assert isinstance(c.seed, int) and c.seed == 42
    assert c.batch_size == 128
    assert isinstance(c.cifar_mean, tuple) and len(c.cifar_mean) == 3
    assert isinstance(c.cifar_std, tuple) and len(c.cifar_std) == 3
    assert c.num_classes == 100
    assert c.image_size == 32

    assert c.epochs_teacher == 200
    assert c.epochs_student == 200
    assert c.epochs_ablation == 100

    assert c.lr == 0.1
    assert c.momentum == 0.9
    assert c.weight_decay == 5e-4
    assert c.milestones == [60, 120, 160]
    assert c.gamma == 0.2

    assert c.C == 64
    assert c.r == 128
    assert c.n_augmentations == 8
    assert c.warmup_epochs == 40
    assert c.use_multi_layer == True
    assert c.kd_temperature == 4.0
    assert c.kd_alpha == 0.9
    assert len(c.fitnet_beta_values) == 3
    assert len(c.ablation_r_values) == 5
    assert len(c.ablation_lambda_values) == 6

    assert isinstance(c.device, torch.device)
    assert c.results_dir == "results"


def test_config_paths_exist():
    """All path attributes should be non‑empty strings under results/."""
    c = Config()
    paths = [
        c.teacher_ckpt, c.baseline_ckpt, c.kd_ckpt, c.fitnet_ckpt,
        c.csd_sample_ckpt, c.csd_class_ckpt, c.random_matrix_path,
        c.all_fingerprints_path, c.class_fingerprints_path,
        c.teacher_history_path, c.results_summary_path,
    ]
    for p in paths:
        assert p.startswith("results/"), f"{p} should be under results/"
        assert len(p) > len("results/"), f"{p} should have a filename"
