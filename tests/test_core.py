"""Tests for the core CSD math — fingerprint generation and loss computation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F


def test_random_matrix_normalization():
    """Each column of R should be a unit vector after L2‑normalisation."""
    C, r = 64, 128
    R = torch.randn(C, r)
    R_norm = F.normalize(R, p=2, dim=0)
    col_norms = R_norm.norm(p=2, dim=0)
    assert torch.allclose(col_norms, torch.ones(r), atol=1e-6)


def test_fingerprint_range():
    """φ = f_norm @ R_norm should be in [-1, 1] since both inputs are unit‑normed."""
    C, r = 64, 32
    B = 8
    features = torch.randn(B, C)  # pooled features
    f_norm = F.normalize(features, p=2, dim=1)
    R = torch.randn(C, r)
    R_norm = F.normalize(R, p=2, dim=0)
    phi = f_norm @ R_norm
    assert phi.shape == (B, r)
    assert torch.all(phi >= -1.0001) and torch.all(phi <= 1.0001)


def test_csd_loss_gradient_flow():
    """MSE between student and teacher fingerprints should produce gradients."""
    C, r = 64, 128
    B = 16
    f_student = torch.randn(B, C, requires_grad=True)
    f_teacher = torch.randn(B, C)

    R = torch.randn(C, r)
    R_norm = F.normalize(R, p=2, dim=0)

    f_s_norm = F.normalize(f_student, p=2, dim=1)
    f_t_norm = F.normalize(f_teacher, p=2, dim=1)

    phi_s = f_s_norm @ R_norm
    phi_t = f_t_norm @ R_norm

    loss = F.mse_loss(phi_s, phi_t)
    loss.backward()
    assert f_student.grad is not None
    assert not torch.allclose(f_student.grad, torch.zeros_like(f_student.grad))


def test_per_class_averaging():
    """Mean of fingerprints from the same class should be a valid fingerprint."""
    r = 64
    N = 500
    n_classes = 10
    labels = torch.randint(0, n_classes, (N,))
    fingerprints = torch.randn(N, r)

    class_means = torch.zeros(n_classes, r)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_means[c] = fingerprints[mask].mean(dim=0)

    # Check shape
    assert class_means.shape == (n_classes, r)
    # Check non‑zero for classes with samples
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            assert class_means[c].abs().sum() > 0


def test_fingerprint_cosine_similarity_interpretation():
    """φ[i,j] should be the cosine similarity between f[i] and R[:,j]."""
    C, r = 64, 16
    B = 4
    f = torch.randn(B, C)
    R = torch.randn(C, r)

    f_norm = F.normalize(f, p=2, dim=1)
    R_norm = F.normalize(R, p=2, dim=0)
    phi_matrix = f_norm @ R_norm

    # Check one entry
    i, j = 2, 5
    manual_cos = F.cosine_similarity(f[i:i+1], R[:, j:j+1].t(), dim=1)
    assert torch.allclose(phi_matrix[i, j], manual_cos, atol=1e-6)


def test_kd_loss_computation():
    """Verify KD loss formula: (1-α)·CE + α·T²·KL."""
    B, C = 4, 10
    T, alpha = 4.0, 0.9
    student_logits = torch.randn(B, C, requires_grad=True)
    teacher_logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))

    ce = F.cross_entropy(student_logits, labels)
    kd = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)

    loss = (1 - alpha) * ce + alpha * kd
    loss.backward()
    assert student_logits.grad is not None


def test_l2_normalize_unit_norm():
    """F.normalize with p=2, dim=1 should produce unit‑norm rows."""
    x = torch.randn(32, 64)
    x_norm = F.normalize(x, p=2, dim=1)
    row_norms = x_norm.norm(p=2, dim=1)
    assert torch.allclose(row_norms, torch.ones(32), atol=1e-6)
