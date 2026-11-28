"""Unit tests for CSD++ core mathematics, quantization, SimHash, losses, cross-arch models, and privacy audit."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from csd_core import (
    ProjectorHead,
    SimHashRelationalQueue,
    audit_privacy_inversion,
    center_and_normalize,
    charikar_cosine_matrix,
    compute_pack_bytes,
    csd_anchor_semantic_loss,
    csd_jl_relational_loss,
    csd_projected_cosine_loss,
    dequantize_fingerprints,
    dkd_loss_fn,
    make_random_matrix,
    pack_sparse_topk,
    quantize_fingerprints,
    unpack_sparse_topk,
)
from data import augment_cifar_batch
from models import get_model, mobilenetv2_cifar, resnet8x4, resnet20, resnet56


def test_orthogonal_random_matrix_seed_reproducibility():
    R1 = make_random_matrix(64, 32, seed=123, orthogonal=True)
    R2 = make_random_matrix(64, 32, seed=123, orthogonal=True)
    assert torch.allclose(R1, R2)
    # Columns must be orthonormal when r <= in_dim
    gram = R1.T @ R1
    assert torch.allclose(gram, torch.eye(32), atol=1e-5)


def test_quantization_roundtrip_and_byte_savings():
    x = torch.randn(500, 32)
    p_fp32 = quantize_fingerprints(x, "fp32")
    p_fp16 = quantize_fingerprints(x, "fp16")
    p_int8 = quantize_fingerprints(x, "int8")
    p_1bit = quantize_fingerprints(x, "1bit")

    assert compute_pack_bytes(p_fp32) == 500 * 32 * 4
    assert compute_pack_bytes(p_fp16) == 500 * 32 * 2
    assert compute_pack_bytes(p_int8) == 500 * 32 * 1 + 500 * 2
    assert compute_pack_bytes(p_1bit) == 500 * (32 // 8)

    # int8 reconstruction accuracy should be very high (cos > 0.999)
    x_hat = dequantize_fingerprints(p_int8)
    cos = F.cosine_similarity(x, x_hat, dim=1).mean().item()
    assert cos > 0.999, f"Unexpected int8 cosine fidelity: {cos}"


def test_simhash_charikar_cosine_estimator():
    torch.manual_seed(0)
    # Two vectors with known cosine similarity ~0.5
    u = F.normalize(torch.randn(1, 64), dim=1)
    v_orth = F.normalize(torch.randn(1, 64), dim=1)
    v_orth = F.normalize(v_orth - (v_orth @ u.T) * u, dim=1)
    v = 0.5 * u + (0.75 ** 0.5) * v_orth
    X = torch.cat([u, v], dim=0)

    R = make_random_matrix(64, 2048, seed=42, orthogonal=True)
    pack = quantize_fingerprints(X @ R, mode="1bit")
    signs = dequantize_fingerprints(pack)
    G_est = charikar_cosine_matrix(signs)
    assert abs(G_est[0, 1].item() - 0.5) < 0.06


def test_sparse_topk_pack_unpack():
    torch.manual_seed(1)
    mat = torch.randn(64, 100)
    pack = pack_sparse_topk(mat, top_m=12, quant="int8")
    rec = unpack_sparse_topk(pack)
    assert rec.shape == (64, 100)
    # Top-1 argmax must be preserved
    assert (mat.argmax(dim=1) == rec.argmax(dim=1)).all()


def test_cross_architecture_models_and_losses():
    m_mob = mobilenetv2_cifar(num_classes=100)
    m_r8 = resnet8x4(num_classes=100)
    x = torch.randn(8, 3, 32, 32)

    for m in (m_mob, m_r8, resnet20(), resnet56()):
        logits, l2, l3 = m(x, return_features="multi")
        assert logits.shape == (8, 100)
        assert l2.shape[1] == m.feature_dims[0]
        assert l3.shape[1] == m.feature_dims[1]
        assert m.classifier_weight.shape == (100, m.feature_dims[1])

    # Verify gradients flow through all CSD++ loss components for cross-arch (160 != 64)
    logits, l3 = m_mob(x, return_features=True)
    f_s = F.adaptive_avg_pool2d(l3, (1, 1)).flatten(1)
    s_norm = center_and_normalize(f_s)

    proj = ProjectorHead(m_mob.feature_dims[1], 32, hidden_dim=160)
    t_sketch = torch.randn(8, 32)
    t_anchor = torch.randn(8, 100)
    t_gram = torch.eye(100)

    l_proj, sim = csd_projected_cosine_loss(proj(s_norm), t_sketch)
    l_rel = csd_jl_relational_loss(s_norm, t_sketch, is_1bit=False)
    l_anc = csd_anchor_semantic_loss(s_norm, m_mob.classifier_weight, t_anchor, t_class_gram=t_gram)

    total = l_proj + l_rel + l_anc
    total.backward()
    assert m_mob.fc.weight.grad is not None
    assert not torch.isnan(total)


def test_vectorized_augment_shape_and_bounds():
    imgs = torch.rand(16, 3, 32, 32)
    aug = augment_cifar_batch(imgs, pad=4, flip_prob=0.5)
    assert aug.shape == imgs.shape
    assert aug.min() >= 0.0 and aug.max() <= 1.0


def test_privacy_inversion_audit_monotonicity():
    torch.manual_seed(7)
    feats = F.relu(torch.randn(400, 64))
    res = audit_privacy_inversion(feats, r_values=[8, 32, 128], quant_modes=["fp32", "1bit"], seed=42)
    by_key = {(d["r"], d["quant"]): d for d in res}
    # r=128 fp32 (overcomplete, like old CSD) must be nearly 100% invertible
    assert by_key[(128, "fp32")]["recon_cosine"] > 0.99
    # r=8 1bit must be much harder to invert than r=128 fp32
    assert by_key[(8, "1bit")]["recon_cosine"] < by_key[(128, "fp32")]["recon_cosine"] - 0.3


def test_fifo_relational_queue_and_1bit_loss():
    torch.manual_seed(11)
    q = SimHashRelationalQueue(queue_size=40, feat_dim=16, sketch_dim=64, device=torch.device("cpu"))
    assert q.get() == (None, None)

    for step in range(4):
        s = F.normalize(torch.randn(16, 16, requires_grad=True), dim=1)
        t = torch.sign(torch.randn(16, 64))
        t[t == 0] = 1.0
        sq, tq = q.get()
        loss = csd_jl_relational_loss(s, t, is_1bit=True, tau=0.15, s_queue=sq, t_queue=tq)
        loss.backward()
        assert not torch.isnan(loss)
        q.enqueue(s, t)
    assert q.filled == 40


def test_dkd_loss_gradients():
    torch.manual_seed(12)
    s = torch.randn(16, 100, requires_grad=True)
    t = torch.randn(16, 100)
    y = torch.randint(0, 100, (16,))
    loss = dkd_loss_fn(s, t, y, T=4.0, alpha=1.0, beta=4.0)
    loss.backward()
    assert s.grad is not None and not torch.isnan(loss) and loss.item() >= 0.0

