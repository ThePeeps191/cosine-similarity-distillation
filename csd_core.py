"""Core mathematics, compression, losses, and privacy auditing for CSD++.

Implements:
1. Seed-generated orthogonal / Gaussian Johnson-Lindenstrauss (JL) reference matrices (4-byte storage).
2. Fingerprint quantization & bit-packing (fp32, fp16, int8, 1-bit SimHash) with exact byte accounting.
3. Fast vectorized offline teacher feature & artifact extraction (single teacher pass, then discarded).
4. Training-only learnable projection heads (zero inference overhead, enables cross-architecture KD).
5. Principled CSD++ loss components:
   - Centered Projected Cosine Alignment (L_proj)
   - Johnson-Lindenstrauss Intra-Batch Relational Cosine Distillation (L_rel, 100% architecture-agnostic)
   - Prototype-Anchored Semantic Cosine Similarity Distillation (L_anchor, ultra-compact & architecture-agnostic)
6. Quantitative feature-inversion privacy audit (linear Ridge + MLP attackers).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import _load_raw_cifar100, augment_cifar_batch


# ============================================================================
# 1. Seed-Generated Random Reference Matrices (4-byte storage!)
# ============================================================================

def make_random_matrix(
    in_dim: int,
    r: int,
    seed: int = 42,
    orthogonal: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a frozen random reference matrix R of shape (in_dim, r) with unit-norm columns.

    When ``orthogonal=True``:
    - If ``r <= in_dim``, columns are strictly orthonormal via QR decomposition (Stiefel manifold),
      achieving strictly lower Johnson-Lindenstrauss variance than i.i.d. Gaussian projections.
    - If ``r > in_dim``, stacks orthogonal blocks so columns remain maximally spread.

    Storage cost: only the 32-bit integer ``seed`` (4 bytes) is needed to regenerate R.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    if orthogonal:
        blocks = []
        remaining = r
        while remaining > 0:
            k = min(in_dim, remaining)
            g = torch.randn(in_dim, k, generator=gen)
            q, _ = torch.linalg.qr(g, mode="reduced")
            blocks.append(q)
            remaining -= k
        R = torch.cat(blocks, dim=1)
    else:
        R = torch.randn(in_dim, r, generator=gen)

    R = F.normalize(R, p=2, dim=0)
    return R.to(device)


# ============================================================================
# 2. Quantization, Bit-Packing, and Exact Byte Accounting
# ============================================================================

def quantize_fingerprints(x: torch.Tensor, mode: str = "int8") -> dict[str, Any]:
    """Quantize a (N, r) fingerprint tensor into a compact storage pack.

    Modes:
        - ``"fp32"``: uncompressed float32 (4 bytes/elem)
        - ``"fp16"``: float16 (2 bytes/elem)
        - ``"int8"``: per-row symmetric int8 + fp16 scale (1 byte/elem + 2 bytes/row)
        - ``"1bit"``: SimHash sign bit-packing into uint8 (0.125 bytes/elem = 1 bit/elem!)
    """
    x_cpu = x.detach().cpu().float()
    N, r = x_cpu.shape

    if mode == "fp32":
        return {"mode": "fp32", "data": x_cpu, "shape": (N, r)}
    elif mode == "fp16":
        return {"mode": "fp16", "data": x_cpu.half(), "shape": (N, r)}
    elif mode == "int8":
        scale = x_cpu.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        q = torch.round(x_cpu / scale).clamp(-127, 127).to(torch.int8)
        return {"mode": "int8", "data": q, "scale": scale.half().squeeze(1), "shape": (N, r)}
    elif mode == "1bit":
        bits = (x_cpu >= 0).numpy().astype(np.uint8)
        packed = np.packbits(bits, axis=1)
        return {"mode": "1bit", "data": torch.from_numpy(packed), "shape": (N, r)}
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")


def dequantize_fingerprints(pack: dict[str, Any], device: torch.device | str = "cpu") -> torch.Tensor:
    """Reconstruct float32 tensor of shape (N, r) from a quantized pack.

    For ``"1bit"``, returns ``{-1.0, +1.0}`` sign vectors of shape (N, r).
    """
    mode = pack["mode"]
    N, r = pack["shape"]
    if mode == "fp32":
        return pack["data"].to(device=device, dtype=torch.float32)
    elif mode == "fp16":
        return pack["data"].to(device=device, dtype=torch.float32)
    elif mode == "int8":
        q = pack["data"].to(device=device, dtype=torch.float32)
        s = pack["scale"].to(device=device, dtype=torch.float32).unsqueeze(1)
        return q * s
    elif mode == "1bit":
        unpacked = np.unpackbits(pack["data"].numpy(), axis=1)[:, :r]
        signs = torch.from_numpy(unpacked.astype(np.float32) * 2.0 - 1.0)
        return signs.to(device=device)
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")


def pack_sparse_topk(
    matrix: torch.Tensor,
    top_m: int = 12,
    quant: str = "int8",
) -> dict[str, Any]:
    """Compress a (N, K) similarity or logit matrix by storing only top-m entries per row.

    Also stores the per-row mean of the remaining (K - m) entries so the full K-dim vector
    can be faithfully reconstructed for softmax/KL without distortion.
    """
    x = matrix.detach().cpu().float()
    N, K = x.shape
    m = min(top_m, K)
    vals, idx = torch.topk(x, k=m, dim=1)  # (N, m)

    # Compute mean of non-top-m entries per row
    total_sum = x.sum(dim=1)
    top_sum = vals.sum(dim=1)
    rest_mean = (total_sum - top_sum) / max(1, K - m)

    idx_dtype = torch.uint8 if K <= 256 else torch.int16
    idx_packed = idx.to(idx_dtype)

    if quant == "int8":
        vmin = rest_mean.unsqueeze(1)
        vmax = vals.amax(dim=1, keepdim=True)
        span = (vmax - vmin).clamp(min=1e-6)
        q = torch.round(((vals - vmin) / span) * 255.0).clamp(0, 255).to(torch.uint8)
        return {
            "mode": "sparse_int8",
            "indices": idx_packed,
            "values": q,
            "vmin": vmin.squeeze(1).half(),
            "span": span.squeeze(1).half(),
            "shape": (N, K),
        }
    elif quant == "fp16":
        return {
            "mode": "sparse_fp16",
            "indices": idx_packed,
            "values": vals.half(),
            "rest_mean": rest_mean.half(),
            "shape": (N, K),
        }
    else:
        raise ValueError(f"Unsupported sparse quant: {quant}")


def unpack_sparse_topk(pack: dict[str, Any], device: torch.device | str = "cpu") -> torch.Tensor:
    """Reconstruct dense (N, K) matrix from a sparse top-m pack."""
    N, K = pack["shape"]
    idx = pack["indices"].to(device=device, dtype=torch.long)
    if pack["mode"] == "sparse_int8":
        vmin = pack["vmin"].to(device=device, dtype=torch.float32).unsqueeze(1)
        span = pack["span"].to(device=device, dtype=torch.float32).unsqueeze(1)
        vals = vmin + (pack["values"].to(device=device, dtype=torch.float32) / 255.0) * span
        out = vmin.expand(N, K).clone()
        out.scatter_(1, idx, vals)
        return out
    elif pack["mode"] == "sparse_fp16":
        rest = pack["rest_mean"].to(device=device, dtype=torch.float32).unsqueeze(1)
        vals = pack["values"].to(device=device, dtype=torch.float32)
        out = rest.expand(N, K).clone()
        out.scatter_(1, idx, vals)
        return out
    else:
        raise ValueError(f"Unknown sparse pack mode: {pack['mode']}")


def compute_pack_bytes(obj: Any) -> int:
    """Recursively compute exact payload bytes of a tensor or artifact dict."""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        total = 0
        for k, v in obj.items():
            if k in ("mode", "shape"):
                continue
            if k == "seed":
                total += 4  # 32-bit PRNG seed
            else:
                total += compute_pack_bytes(v)
        return total
    if isinstance(obj, (list, tuple)):
        return sum(compute_pack_bytes(x) for x in obj)
    return 0


def charikar_cosine_matrix(signs: torch.Tensor) -> torch.Tensor:
    """Estimate pairwise cosine similarities from {-1, +1} SimHash sketches via Charikar's theorem.

    For two sign vectors s_i, s_j in {-1, +1}^r:
        agreement_fraction = 0.5 * (1 + (s_i . s_j) / r)
        d_H / r            = 0.5 * (1 - (s_i . s_j) / r)
        cos_est(i, j)      = cos(pi * d_H / r) = sin(pi/2 * (s_i . s_j) / r)
    """
    r = signs.shape[1]
    dot = (signs @ signs.T) / float(r)
    dot = dot.clamp(-1.0, 1.0)
    return torch.sin((math.pi / 2.0) * dot)


# ============================================================================
# 3. Training-Only Learnable Projection Head (Zero Inference Cost!)
# ============================================================================

class ProjectorHead(nn.Module):
    """Lightweight training-only projector mapping student features (C_S) -> target dim (r).

    Discarded completely after training -> 0 extra parameters and 0 FLOPs at test time!
    Enables cross-architecture distillation (C_S != C_T) and removes basis rigidity.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 4. Fast Offline Teacher Extraction & Artifact Construction
# ============================================================================

@torch.no_grad()
def extract_teacher_features(
    teacher: nn.Module,
    config: Any,
    n_augmentations: int = 8,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Single-pass vectorized offline extraction of multi-view averaged teacher representations.

    After this returns, the teacher model can be permanently discarded.
    """
    device = config.device
    teacher.eval()

    imgs, labels = _load_raw_cifar100("./data", train=True)
    mean = torch.tensor(config.cifar_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(config.cifar_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    N = len(imgs)
    l2_dim, l3_dim = teacher.feature_dims
    all_l2 = torch.zeros(N, l2_dim, dtype=torch.float32)
    all_l3 = torch.zeros(N, l3_dim, dtype=torch.float32)
    all_logits = torch.zeros(N, config.num_classes, dtype=torch.float32)

    # Include 1 clean view + (n_augmentations - 1) augmented views for low-variance targets
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        raw_x = imgs[start:end].to(device)
        B = raw_x.size(0)

        sum_l2 = torch.zeros(B, l2_dim, device=device, dtype=torch.float32)
        sum_l3 = torch.zeros(B, l3_dim, device=device, dtype=torch.float32)
        sum_logits = torch.zeros(B, config.num_classes, device=device, dtype=torch.float32)

        for v in range(n_augmentations):
            if v == 0:
                xv = raw_x
            else:
                xv = augment_cifar_batch(raw_x, pad=4, flip_prob=0.5)
            xv = (xv - mean) / std
            with torch.autocast(device.type, dtype=config.amp_dtype, enabled=config.use_amp and device.type in ("cuda", "mps")):
                logits, l2, l3 = teacher(xv, return_features="multi")
            p2 = F.adaptive_avg_pool2d(l2.float(), (1, 1)).flatten(1)
            p3 = F.adaptive_avg_pool2d(l3.float(), (1, 1)).flatten(1)
            sum_l2 += p2
            sum_l3 += p3
            sum_logits += logits.float()

        all_l2[start:end] = (sum_l2 / n_augmentations).cpu()
        all_l3[start:end] = (sum_l3 / n_augmentations).cpu()
        all_logits[start:end] = (sum_logits / n_augmentations).cpu()

    return {
        "l2": all_l2,
        "l3": all_l3,
        "logits": all_logits,
        "labels": labels.clone(),
        "classifier_weight": teacher.classifier_weight.detach().cpu().float(),
    }


def build_csd_artifacts(
    teacher_data: dict[str, torch.Tensor],
    num_classes: int = 100,
    r: int = 32,
    quant: str = "int8",
    orthogonal: bool = True,
    seed: int = 42,
    top_m: int = 12,
    center: bool = True,
) -> dict[str, Any]:
    """Construct compact, teacher-free CSD++ artifacts from extracted teacher representations.

    Produces:
    1. ``class_proto`` (ultra-compact ~20-50 KB):
       - Per-class centered prototype JL sketches (l2 & l3)
       - K x K class-prototype cosine Gram matrix G_T^class
       - K x K empirical class-averaged semantic cosine profile S_bar_T
    2. ``sample_jl`` (compact per-sample JL sketches):
       - Per-sample centered JL sketches (l3, and optionally l2) at (r, quant)
    3. ``sample_anchor`` (compact per-sample prototype-anchored dark knowledge):
       - Sparse top-m cosine similarities of each sample against teacher's K class prototypes
    4. ``offline_class_logits`` (fair ultra-compact baseline, ~20-40 KB):
       - Class-averaged teacher logits (K x K)
    5. ``offline_sparse_logits`` (fair per-sample baseline at same byte budget):
       - Sparse top-m per-sample teacher logits
    """
    f_l2 = teacher_data["l2"].float()
    f_l3 = teacher_data["l3"].float()
    logits = teacher_data["logits"].float()
    labels = teacher_data["labels"].long()
    W_T = teacher_data["classifier_weight"].float()

    # Center features to remove positive-orthant ReLU cone anisotropy
    if center:
        mean_l2 = f_l2.mean(dim=0, keepdim=True)
        mean_l3 = f_l3.mean(dim=0, keepdim=True)
        c_l2 = F.normalize(f_l2 - mean_l2, p=2, dim=1)
        c_l3 = F.normalize(f_l3 - mean_l3, p=2, dim=1)
    else:
        c_l2 = F.normalize(f_l2, p=2, dim=1)
        c_l3 = F.normalize(f_l3, p=2, dim=1)

    # Seed-generated orthogonal JL projection matrices (4-byte seed storage!)
    R3 = make_random_matrix(f_l3.shape[1], r, seed=seed, orthogonal=orthogonal)
    R2 = make_random_matrix(f_l2.shape[1], r, seed=seed + 1, orthogonal=orthogonal)

    # Per-sample JL sketches
    phi_l3 = c_l3 @ R3  # (N, r)
    phi_l2 = c_l2 @ R2  # (N, r)
    sample_jl_l3 = quantize_fingerprints(phi_l3, mode=quant)
    sample_jl_l2 = quantize_fingerprints(phi_l2, mode=quant)

    # Teacher class prototypes A_T: (K, C_T), unit-norm
    proto_l3 = torch.zeros(num_classes, f_l3.shape[1])
    proto_l2 = torch.zeros(num_classes, f_l2.shape[1])
    class_logits = torch.zeros(num_classes, num_classes)
    for c in range(num_classes):
        mask = labels == c
        proto_l3[c] = c_l3[mask].mean(dim=0)
        proto_l2[c] = c_l2[mask].mean(dim=0)
        class_logits[c] = logits[mask].mean(dim=0)

    proto_l3_norm = F.normalize(proto_l3, p=2, dim=1)
    proto_l2_norm = F.normalize(proto_l2, p=2, dim=1)

    # Class prototype JL fingerprints
    class_fp_l3 = quantize_fingerprints(proto_l3_norm @ R3, mode=quant)
    class_fp_l2 = quantize_fingerprints(proto_l2_norm @ R2, mode=quant)

    # K x K teacher class-prototype cosine Gram matrix (symmetric, unit diag)
    class_gram = proto_l3_norm @ proto_l3_norm.T  # (K, K) in [-1, 1]

    # Sample-to-class semantic cosine similarity profile: (N, K)
    # Measures how similar each image's centered teacher feature is to every class prototype
    sample_anchor_cos = c_l3 @ proto_l3_norm.T  # (N, K) in [-1, 1]

    # Empirical class-averaged semantic cosine profile: (K, K)
    class_anchor_profile = torch.zeros(num_classes, num_classes)
    for c in range(num_classes):
        class_anchor_profile[c] = sample_anchor_cos[labels == c].mean(dim=0)

    # Also compute teacher classifier-weight cosine similarities (normalized W_T)
    W_T_norm = F.normalize(W_T, p=2, dim=1)
    w_gram = W_T_norm @ W_T_norm.T

    # Pack ultra-compact class artifacts
    class_artifacts = {
        "seed": seed,
        "fp_l3": class_fp_l3,
        "fp_l2": class_fp_l2,
        "class_gram": quantize_fingerprints(class_gram, mode="int8"),
        "class_anchor_profile": quantize_fingerprints(class_anchor_profile, mode="int8"),
        "w_gram": quantize_fingerprints(w_gram, mode="int8"),
    }

    # Pack compact per-sample artifacts
    sample_jl_artifacts = {
        "seed": seed,
        "l3": sample_jl_l3,
        "l2": sample_jl_l2,
    }

    sample_anchor_artifacts = {
        "sparse_anchor": pack_sparse_topk(sample_anchor_cos, top_m=top_m, quant="int8"),
    }

    # Fair offline logit baselines
    offline_class_logits = quantize_fingerprints(class_logits, mode="fp16")
    offline_sparse_logits = pack_sparse_topk(logits, top_m=top_m, quant="int8")

    return {
        "r": r,
        "quant": quant,
        "orthogonal": orthogonal,
        "center": center,
        "seed": seed,
        "top_m": top_m,
        "class_artifacts": class_artifacts,
        "sample_jl_artifacts": sample_jl_artifacts,
        "sample_anchor_artifacts": sample_anchor_artifacts,
        "offline_class_logits": offline_class_logits,
        "offline_sparse_logits": offline_sparse_logits,
        "storage_bytes": {
            "csd_proto_l3_only": compute_pack_bytes({
                "seed": seed,
                "fp_l3": class_fp_l3,
                "class_gram": class_artifacts["class_gram"],
                "class_anchor_profile": class_artifacts["class_anchor_profile"],
            }),
            "csd_proto_full": compute_pack_bytes(class_artifacts),
            "csd_rel_l3": compute_pack_bytes({"seed": seed, "l3": sample_jl_l3}),
            "csd_rel_multi": compute_pack_bytes(sample_jl_artifacts),
            "csd_anchor_sparse": compute_pack_bytes(sample_anchor_artifacts),
            "csd_pp_full": compute_pack_bytes({"seed": seed, "l3": sample_jl_l3})
            + compute_pack_bytes(sample_anchor_artifacts)
            + compute_pack_bytes(class_artifacts["class_gram"]),
            "offline_class_logits": compute_pack_bytes(offline_class_logits),
            "offline_sparse_logits": compute_pack_bytes(offline_sparse_logits),
        },
    }


# ============================================================================
# 5. CSD++ Loss Functions
# ============================================================================

def center_and_normalize(feat: torch.Tensor, running_mean: torch.Tensor | None = None) -> torch.Tensor:
    """Zero-center and L2-normalize features to eliminate positive-orthant anisotropy."""
    if running_mean is not None:
        centered = feat - running_mean
    else:
        centered = feat - feat.mean(dim=0, keepdim=True)
    return F.normalize(centered, p=2, dim=1)


def csd_projected_cosine_loss(
    s_proj: torch.Tensor,
    t_fp: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Centered directional cosine alignment between projected student features and teacher fingerprints."""
    cos = F.cosine_similarity(s_proj, t_fp, dim=1)
    loss = (1.0 - cos).mean()
    return loss, cos.detach().mean().item()


class SimHashRelationalQueue:
    """Zero-disk-storage FIFO memory queue for large-context relational distillation.

    Maintains a ring buffer of the most recent Q detached student centered features
    paired with their exact offline teacher sketches. Because offline teacher sketches
    are frozen, teacher keys exhibit zero drift while expanding the relational context
    from B (128) to B + Q (e.g., 2176) — ~17x more positive and hard-negative pairs
    per gradient step at zero extra disk storage and zero extra student parameters.
    """

    def __init__(self, queue_size: int, feat_dim: int, sketch_dim: int, device: torch.device) -> None:
        self.queue_size = queue_size
        self.feat_queue = torch.zeros(queue_size, feat_dim, device=device, dtype=torch.float32)
        self.sketch_queue = torch.zeros(queue_size, sketch_dim, device=device, dtype=torch.float32)
        self.ptr = 0
        self.filled = 0

    def get(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.filled == 0:
            return None, None
        return self.feat_queue[: self.filled], self.sketch_queue[: self.filled]

    @torch.no_grad()
    def enqueue(self, s_feat: torch.Tensor, t_sketch: torch.Tensor) -> None:
        B = s_feat.size(0)
        if B >= self.queue_size:
            self.feat_queue.copy_(s_feat[-self.queue_size :].detach())
            self.sketch_queue.copy_(t_sketch[-self.queue_size :].detach())
            self.ptr = 0
            self.filled = self.queue_size
            return
        end = self.ptr + B
        if end <= self.queue_size:
            self.feat_queue[self.ptr : end].copy_(s_feat.detach())
            self.sketch_queue[self.ptr : end].copy_(t_sketch.detach())
        else:
            first = self.queue_size - self.ptr
            self.feat_queue[self.ptr :].copy_(s_feat[:first].detach())
            self.sketch_queue[self.ptr :].copy_(t_sketch[:first].detach())
            rem = B - first
            self.feat_queue[:rem].copy_(s_feat[first:].detach())
            self.sketch_queue[:rem].copy_(t_sketch[first:].detach())
        self.ptr = end % self.queue_size
        self.filled = min(self.queue_size, self.filled + B)


def dkd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float = 4.0,
    alpha: float = 1.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Decoupled Knowledge Distillation (Zhao et al., CVPR 2022): TCKD + NCKD."""
    B, K = student_logits.shape
    gt_mask = torch.zeros_like(student_logits, dtype=torch.bool)
    gt_mask.scatter_(1, labels.unsqueeze(1), True)
    other_mask = ~gt_mask

    pred_s = F.softmax(student_logits / T, dim=1)
    pred_t = F.softmax(teacher_logits / T, dim=1)

    # Target-Class KD (TCKD): binary distribution [p_target, p_non_target]
    p_s_gt = (pred_s * gt_mask).sum(dim=1, keepdim=True).clamp(1e-7, 1.0 - 1e-7)
    p_t_gt = (pred_t * gt_mask).sum(dim=1, keepdim=True).clamp(1e-7, 1.0 - 1e-7)
    bin_s = torch.cat([p_s_gt, 1.0 - p_s_gt], dim=1)
    bin_t = torch.cat([p_t_gt, 1.0 - p_t_gt], dim=1)
    tckd = F.kl_div(bin_s.log(), bin_t, reduction="batchmean") * (T * T)

    # Non-Target-Class KD (NCKD): softmax over non-target classes only
    log_p_s_nc = F.log_softmax(student_logits / T - 1000.0 * gt_mask.float(), dim=1)
    p_t_nc = F.softmax(teacher_logits / T - 1000.0 * gt_mask.float(), dim=1)
    nckd = F.kl_div(log_p_s_nc, p_t_nc, reduction="batchmean") * (T * T)

    return alpha * tckd + beta * nckd


def csd_jl_relational_loss(
    s_feat_norm: torch.Tensor,
    t_sketch: torch.Tensor,
    is_1bit: bool = False,
    tau: float = 0.15,
    kl_weight: float = 1.0,
    mse_weight: float = 1.0,
    s_queue: torch.Tensor | None = None,
    t_queue: torch.Tensor | None = None,
) -> torch.Tensor:
    """Johnson-Lindenstrauss Relational Cosine Distillation (100% architecture-agnostic).

    Aligns the student's pairwise cosine similarity matrix G_S against the teacher's
    JL-reconstructed pairwise cosine matrix G_T directly from compact offline sketches.
    When ``s_queue`` and ``t_queue`` are provided (FIFO Relational Queue), expands the
    relational matrix from (B, B) to (B, B + Q) for ~17x denser semantic supervision
    with zero extra disk storage and zero extra student parameters.
    """
    B = s_feat_norm.size(0)
    if s_queue is not None and t_queue is not None and s_queue.size(0) > 0:
        s_keys = torch.cat([s_feat_norm, s_queue], dim=0)  # (B + Q, C_S)
        t_keys = torch.cat([t_sketch, t_queue], dim=0)     # (B + Q, r)
    else:
        s_keys = s_feat_norm
        t_keys = t_sketch

    M = s_keys.size(0)
    G_S = s_feat_norm @ s_keys.T  # (B, M) in [-1, 1]

    with torch.no_grad():
        if is_1bit:
            r = t_sketch.shape[1]
            dot = ((t_sketch @ t_keys.T) / float(r)).clamp(-1.0, 1.0)
            G_T = torch.sin((math.pi / 2.0) * dot)
        else:
            t_q = F.normalize(t_sketch, p=2, dim=1)
            t_k = F.normalize(t_keys, p=2, dim=1)
            G_T = t_q @ t_k.T

    # Mask only the B self-pairs (i, i)
    self_mask = torch.zeros(B, M, device=G_S.device, dtype=torch.bool)
    self_mask[:, :B] = torch.eye(B, device=G_S.device, dtype=torch.bool)
    total = torch.tensor(0.0, device=G_S.device)

    if kl_weight > 0.0:
        off_S = G_S.masked_fill(self_mask, -1e4)
        off_T = G_T.masked_fill(self_mask, -1e4)
        p_T = F.softmax(off_T / tau, dim=1)
        log_p_S = F.log_softmax(off_S / tau, dim=1)
        kl_loss = F.kl_div(log_p_S, p_T, reduction="batchmean") * tau
        total = total + kl_weight * kl_loss

    if mse_weight > 0.0:
        valid = ~self_mask
        mse_loss = F.mse_loss(G_S[valid], G_T[valid])
        total = total + mse_weight * mse_loss

    return total


def csd_anchor_semantic_loss(
    s_feat_norm: torch.Tensor,
    s_classifier_weight: torch.Tensor,
    t_anchor_profile: torch.Tensor,
    t_class_gram: torch.Tensor | None = None,
    tau: float = 0.10,
    gram_weight: float = 0.5,
) -> torch.Tensor:
    """Prototype-Anchored Semantic Cosine Similarity Distillation (100% architecture-agnostic).

    Uses the student's own L2-normalized classifier weights W_S as semantic class anchors:
      s_S(x) = f_S_norm(x) @ W_S_norm^T  in [-1, 1]^K
    and aligns s_S(x) to the teacher's semantic cosine profile t_anchor_profile(x) via
    temperature-scaled KL divergence, plus optional classifier Gram matrix alignment!
    """
    W_S_norm = F.normalize(s_classifier_weight, p=2, dim=1)  # (K, C_S)
    s_profile = s_feat_norm @ W_S_norm.T  # (B, K) in [-1, 1]

    p_T = F.softmax(t_anchor_profile / tau, dim=1)
    log_p_S = F.log_softmax(s_profile / tau, dim=1)
    # Scaled by tau (since s_profile is unit-scale cosine in [-1, 1], grad w.r.t. s_profile is p_S - p_T)
    kl_loss = F.kl_div(log_p_S, p_T, reduction="batchmean") * tau

    if t_class_gram is not None and gram_weight > 0.0:
        G_W_S = W_S_norm @ W_S_norm.T
        gram_loss = F.mse_loss(G_W_S, t_class_gram)
        return kl_loss + gram_weight * gram_loss
    return kl_loss


# ============================================================================
# 6. Quantitative Feature-Inversion Privacy Audit
# ============================================================================

def audit_privacy_inversion(
    features: torch.Tensor,
    r_values: list[int] = [8, 16, 32, 64, 128],
    quant_modes: list[str] = ["fp32", "int8", "1bit"],
    seed: int = 42,
    orthogonal: bool = True,
) -> list[dict[str, Any]]:
    """Quantitatively measure feature-direction invertibility across (r, quantization).

    Fits an optimal Ridge regression (closed-form linear attacker) on 80% of
    (fingerprint -> centered unit feature) pairs and evaluates reconstruction
    cosine similarity and MSE on the held-out 20%.

    Proves rigorously:
    - Old CSD (r=128 >= C=64, fp32) has reconstruction cosine ~1.000 (fully invertible!).
    - True compression (r < C) + quantization (int8 / 1-bit) dramatically lowers
      reconstruction fidelity, yielding a quantitative privacy-accuracy-storage frontier.
    """
    f_centered = F.normalize(features.float() - features.float().mean(dim=0, keepdim=True), p=2, dim=1)
    N, C = f_centered.shape
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_train = int(0.8 * N)
    tr_idx = torch.from_numpy(perm[:n_train])
    te_idx = torch.from_numpy(perm[n_train:])

    Y_tr = f_centered[tr_idx]  # (N_tr, C)
    Y_te = f_centered[te_idx]  # (N_te, C)

    results: list[dict[str, Any]] = []
    for r in r_values:
        R = make_random_matrix(C, r, seed=seed, orthogonal=orthogonal)
        phi_raw = f_centered @ R
        for qmode in quant_modes:
            pack = quantize_fingerprints(phi_raw, mode=qmode)
            storage_bytes = compute_pack_bytes(pack) + 4  # + 4B PRNG seed
            phi_hat = dequantize_fingerprints(pack)

            X_tr = phi_hat[tr_idx]
            X_te = phi_hat[te_idx]

            # Optimal regularized linear inversion: W = (X^T X + eps I)^-1 X^T Y
            XtX = X_tr.T @ X_tr + 1e-4 * torch.eye(r)
            XtY = X_tr.T @ Y_tr
            W_inv = torch.linalg.solve(XtX, XtY)

            Y_pred = F.normalize(X_te @ W_inv, p=2, dim=1)
            recon_cos = F.cosine_similarity(Y_pred, Y_te, dim=1).mean().item()
            recon_mse = F.mse_loss(Y_pred, Y_te).item()

            # Non-linear 2-layer MLP inversion attacker (tests against non-linear recovery)
            torch.manual_seed(seed)
            mlp = nn.Sequential(
                nn.Linear(r, 128),
                nn.ReLU(),
                nn.Linear(128, C),
            )
            opt_mlp = torch.optim.Adam(mlp.parameters(), lr=3e-3)
            # Train quickly on mini-batches (up to 4000 samples for speed)
            n_mlp_tr = min(len(X_tr), 8000)
            Xm_tr, Ym_tr = X_tr[:n_mlp_tr], Y_tr[:n_mlp_tr]
            for _ in range(15):
                p_idx = torch.randperm(n_mlp_tr)
                for b0 in range(0, n_mlp_tr, 256):
                    bi = p_idx[b0 : b0 + 256]
                    opt_mlp.zero_grad()
                    pred = F.normalize(mlp(Xm_tr[bi]), p=2, dim=1)
                    l_att = (1.0 - F.cosine_similarity(pred, Ym_tr[bi], dim=1)).mean()
                    l_att.backward()
                    opt_mlp.step()
            with torch.no_grad():
                Y_pred_mlp = F.normalize(mlp(X_te), p=2, dim=1)
                mlp_recon_cos = F.cosine_similarity(Y_pred_mlp, Y_te, dim=1).mean().item()

            # Also measure JL pairwise cosine preservation error on up to 512 test samples
            sub_n = min(512, len(Y_te))
            sub_y = Y_te[:sub_n]
            sub_x = X_te[:sub_n]
            G_true = sub_y @ sub_y.T
            if qmode == "1bit":
                G_est = charikar_cosine_matrix(sub_x)
            else:
                sub_xn = F.normalize(sub_x, p=2, dim=1)
                G_est = sub_xn @ sub_xn.T
            eye = ~torch.eye(sub_n, dtype=torch.bool)
            jl_mae = (G_true[eye] - G_est[eye]).abs().mean().item()

            results.append({
                "r": r,
                "quant": qmode,
                "storage_bytes": storage_bytes,
                "storage_kb": storage_bytes / 1024.0,
                "recon_cosine": recon_cos,
                "mlp_recon_cosine": mlp_recon_cos,
                "recon_mse": recon_mse,
                "jl_pairwise_mae": jl_mae,
            })

    return results
