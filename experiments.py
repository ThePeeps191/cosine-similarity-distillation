"""Unified, reproducible experiment engine for CSD & CSD++ on Apple MPS / CUDA / CPU.

Supports:
- Teacher training (ResNet-56, ResNet-32x4, etc.) with CosineAnnealingLR / MultiStepLR + AMP + torch.compile
- Single-pass offline teacher artifact extraction + quantitative privacy inversion audit
- All baselines & CSD++ methods with exact byte accounting and wall-clock/memory profiling:
    1. baseline            : Student-only Cross-Entropy
    2. label_smooth        : CE with label smoothing (eps=0.1)
    3. kd_online           : Online Hinton KD (teacher resident every batch)
    4. fitnet_online       : Online FitNet with learnable projector (works cross-architecture!)
    5. offline_class_kd    : Offline Class-Averaged Logit KD (~20 KB, fair ultra-compact baseline)
    6. offline_sparse_kd   : Offline Sparse Top-m Quantized Logit KD (~1.1 MB, fair per-sample baseline)
    7. csd_v41_class       : Faithful reproduction of original CSD v4.1 per-class (uncentered, shared R, r=128)
    8. csd_v41_sample      : Faithful reproduction of original CSD v4.1 per-sample (uncentered, shared R, r=128)
    9. csd_proto_random    : Control with RANDOM orthogonal prototypes (proves teacher geometry matters!)
   10. csd_proto           : CSD++ Ultra-Compact (~25-40 KB, 100% arch-agnostic: centered class prototype
                             JL alignment + KxK semantic anchor KL + classifier Gram alignment)
   11. csd_rel             : CSD++ Pure JL Relational (~0.4-1.5 MB, 0 extra params, 100% arch-agnostic:
                             intra-batch pairwise cosine matrix distillation from quantized/1-bit JL sketches)
   12. csd_pp              : CSD++ Full (~1.2-2.0 MB, STILL < teacher size, 100% arch-agnostic:
                             centered projector JL alignment + JL batch-relational cosine + sparse
                             per-sample prototype-anchored semantic KL + class Gram alignment)

All evaluations report BOTH standard Top-1 accuracy (last epoch & best) AND 10-crop TTA Top-1 accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from config import Config
from csd_core import (
    ProjectorHead,
    SimHashRelationalQueue,
    audit_privacy_inversion,
    build_csd_artifacts,
    center_and_normalize,
    compute_pack_bytes,
    csd_anchor_semantic_loss,
    csd_jl_relational_loss,
    csd_projected_cosine_loss,
    dequantize_fingerprints,
    dkd_loss_fn,
    extract_teacher_features,
    make_random_matrix,
    unpack_sparse_topk,
)
from data import get_fast_cifar100_loaders
from models import get_model
from utils import (
    count_parameters,
    empty_cache,
    format_bytes,
    get_model_size_bytes,
    set_seed,
    sync_device,
)


# ============================================================================
# Evaluation helpers (standard + 10-crop TTA)
# ============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    tta: bool = False,
) -> tuple[float, float]:
    """Evaluate model returning (avg_loss, top1_accuracy_percent)."""
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    for batch in loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        if tta:
            B, C, H, W = inputs.shape
            padded = F.pad(inputs, [4, 4, 4, 4], mode="reflect")
            crops = [
                padded[:, :, :H, :W],
                padded[:, :, :H, 8:],
                padded[:, :, 4 : 4 + H, 4 : 4 + W],
                padded[:, :, 8:, :W],
                padded[:, :, 8:, 8:],
            ]
            tta_batch = torch.cat(crops + [c.flip(-1) for c in crops], dim=0)
            logits = model(tta_batch).view(10, B, -1).mean(dim=0)
        else:
            logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        running_loss += loss.item() * inputs.size(0)
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / max(1, total), 100.0 * correct / max(1, total)


def _maybe_compile(model: nn.Module, config: Config) -> nn.Module:
    if config.use_compile and config.device.type in ("cuda", "mps"):
        try:
            return torch.compile(model)
        except Exception as e:
            print(f"[WARN] torch.compile unavailable ({e}); continuing eagerly.")
    return model


def _build_scheduler(
    optimizer: optim.Optimizer, epochs: int, config: Config,
) -> optim.lr_scheduler.LRScheduler:
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    return MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)


def _get_classifier_weight(model: nn.Module) -> torch.Tensor:
    """Safely access classifier_weight even when model is wrapped by torch.compile."""
    orig = getattr(model, "_orig_mod", model)
    return orig.classifier_weight


def _get_feature_dims(model: nn.Module) -> tuple[int, int]:
    orig = getattr(model, "_orig_mod", model)
    return orig.feature_dims


def _unwrap_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    orig = getattr(model, "_orig_mod", model)
    return {k: v.detach().cpu().clone() for k, v in orig.state_dict().items()}


# ============================================================================
# Teacher training & offline extraction
# ============================================================================

def train_teacher_model(
    config: Config,
    arch: str = "resnet56",
    epochs: int = 200,
    seed: int = 42,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Train a teacher model on CIFAR-100 with fast GPU loader + AMP + torch.compile."""
    set_seed(seed)
    os.makedirs(config.results_dir, exist_ok=True)
    save_path = save_path or os.path.join(config.results_dir, f"teacher_{arch}_s{seed}.pth")

    train_loader, test_loader = get_fast_cifar100_loaders(config, return_index=False)
    raw_model = get_model(arch, num_classes=config.num_classes).to(config.device)
    model = _maybe_compile(raw_model, config)

    optimizer = optim.SGD(
        raw_model.parameters(), lr=config.lr, momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, epochs, config)

    history: dict[str, Any] = {
        "arch": arch, "seed": seed, "epochs": epochs, "scheduler": config.scheduler,
        "params": count_parameters(raw_model),
        "size_bytes": get_model_size_bytes(raw_model),
        "train_loss": [], "train_acc": [], "test_acc": [],
    }
    best_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                config.device.type, dtype=config.amp_dtype,
                enabled=config.use_amp and config.device.type in ("cuda", "mps"),
            ):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            r_loss += loss.item() * x.size(0)
            correct += logits.argmax(dim=1).eq(y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_loss = r_loss / total
        train_acc = 100.0 * correct / total
        _, test_acc = evaluate_model(raw_model, test_loader, config.device, tta=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = _unwrap_state_dict(raw_model)

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            elapsed = (time.time() - t0) / 60.0
            print(
                f"[Teacher {arch}] Ep {epoch:3d}/{epochs} | "
                f"LR {optimizer.param_groups[0]['lr']:.5f} | "
                f"Loss {train_loss:.4f} | Train {train_acc:.2f}% | "
                f"Test {test_acc:.2f}% | Best {best_acc:.2f}% | {elapsed:.1f}m",
                flush=True,
            )

    assert best_state is not None
    torch.save(best_state, save_path)
    raw_model.load_state_dict(best_state)
    _, tta_acc = evaluate_model(raw_model, test_loader, config.device, tta=True)

    history["best_acc"] = best_acc
    history["last_acc"] = history["test_acc"][-1]
    history["tta_acc"] = tta_acc
    history["wall_minutes"] = (time.time() - t0) / 60.0

    hist_path = save_path.replace(".pth", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Teacher {arch}] DONE: Best={best_acc:.2f}%, TTA={tta_acc:.2f}% in {history['wall_minutes']:.1f}m", flush=True)
    empty_cache(config.device)
    return history


def prepare_teacher_cache(
    config: Config,
    teacher_arch: str = "resnet56",
    teacher_ckpt: str | None = None,
    n_augmentations: int = 8,
    force: bool = False,
) -> dict[str, torch.Tensor]:
    """Load or extract multi-view teacher features once, then discard teacher."""
    teacher_ckpt = teacher_ckpt or os.path.join(config.results_dir, f"teacher_{teacher_arch}_s42.pth")
    cache_path = os.path.join(config.results_dir, f"teacher_{teacher_arch}_feats_n{n_augmentations}.pt")

    if os.path.exists(cache_path) and not force:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    t0 = time.time()
    teacher = get_model(teacher_arch, num_classes=config.num_classes).to(config.device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=config.device, weights_only=True))
    teacher.eval()

    data = extract_teacher_features(teacher, config, n_augmentations=n_augmentations)
    torch.save(data, cache_path)
    del teacher
    empty_cache(config.device)
    print(f"[Teacher Cache] Extracted {n_augmentations}-view features in {time.time()-t0:.1f}s -> {cache_path}", flush=True)
    return data


# ============================================================================
# Unified Student Distillation Trainer
# ============================================================================

def train_student(
    config: Config,
    method: str,
    student_arch: str = "resnet20",
    teacher_arch: str = "resnet56",
    teacher_ckpt: str | None = None,
    teacher_data: dict[str, torch.Tensor] | None = None,
    epochs: int = 200,
    seed: int = 42,
    r: int = 32,
    quant: str = "int8",
    orthogonal: bool = True,
    top_m: int = 12,
    center: bool = True,
    use_projector: bool = True,
    use_multi_layer: bool = True,
    # Loss weights
    lam_proj: float = 1.0,
    lam_rel: float = 2.0,
    lam_anchor: float = 2.0,
    rel_kl_weight: float = 1.0,
    rel_mse_weight: float = 1.0,
    gram_weight: float = 0.5,
    l2_weight: float = 0.2,
    tau_rel: float = 0.15,
    tau_anchor: float = 0.10,
    kd_alpha: float = 0.9,
    kd_temp: float = 4.0,
    dkd_alpha: float = 1.0,
    dkd_beta: float = 4.0,
    fitnet_beta: float = 10.0,
    queue_size: int = 0,
    warmup_epochs: int = 20,
    lr_override: float | None = None,
    val_fraction: float = 0.0,
    tag: str | None = None,
    save_ckpt: bool = True,
) -> dict[str, Any]:
    """Train a student model under any baseline or CSD++ method with exact accounting."""
    set_seed(seed)
    os.makedirs(config.results_dir, exist_ok=True)
    device = config.device
    teacher_ckpt = teacher_ckpt or os.path.join(config.results_dir, f"teacher_{teacher_arch}_s42.pth")

    needs_index = method in (
        "csd_v41_sample", "csd_rel", "csd_pp", "csd_sample_proj",
        "csd_anchor_sample", "offline_sparse_kd", "csd_hybrid", "csd_dkd_hybrid",
    )
    if val_fraction > 0.0:
        train_loader, eval_loader, test_loader = get_fast_cifar100_loaders(
            config, return_index=needs_index, val_fraction=val_fraction, split_seed=seed,
        )
    else:
        train_loader, test_loader = get_fast_cifar100_loaders(
            config, return_index=needs_index, val_fraction=0.0,
        )
        eval_loader = test_loader

    raw_student = get_model(student_arch, num_classes=config.num_classes).to(device)
    s_l2_dim, s_l3_dim = raw_student.feature_dims

    # Load online teacher ONLY for online methods (kd_online, fitnet_online)
    online_teacher: nn.Module | None = None
    storage_bytes = 0
    teacher_forwards_train = 0
    extra_modules: list[nn.Module] = []

    if method in ("kd_online", "fitnet_online", "sp_online"):
        online_teacher = get_model(teacher_arch, num_classes=config.num_classes).to(device)
        online_teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
        online_teacher.eval()
        for p in online_teacher.parameters():
            p.requires_grad = False
        storage_bytes = get_model_size_bytes(online_teacher)
        teacher_forwards_train = len(train_loader) * epochs

        if method == "fitnet_online":
            _, t_l3_dim = online_teacher.feature_dims
            if s_l3_dim != t_l3_dim:
                fitnet_proj = nn.Linear(s_l3_dim, t_l3_dim, bias=False).to(device)
                extra_modules.append(fitnet_proj)

    # Prepare offline teacher-free artifacts (teacher NOT loaded!)
    artifacts: dict[str, Any] | None = None
    proj_l3: nn.Module | None = None
    proj_l2: nn.Module | None = None
    R3_shared: torch.Tensor | None = None
    R2_shared: torch.Tensor | None = None
    fp_l3_table: torch.Tensor | None = None
    fp_l2_table: torch.Tensor | None = None
    class_gram_T: torch.Tensor | None = None
    anchor_profile_table: torch.Tensor | None = None
    offline_logits_table: torch.Tensor | None = None
    is_1bit = (quant == "1bit")

    offline_methods = {
        "offline_class_kd", "offline_sparse_kd",
        "csd_v41_class", "csd_v41_sample",
        "csd_proto_random", "csd_proto", "csd_rel",
        "csd_sample_proj", "csd_anchor_sample", "csd_pp",
        "csd_hybrid", "csd_dkd_hybrid",
    }

    if method in offline_methods:
        if teacher_data is None and method != "csd_proto_random":
            teacher_data = prepare_teacher_cache(config, teacher_arch=teacher_arch, teacher_ckpt=teacher_ckpt)

        if method in ("csd_v41_class", "csd_v41_sample"):
            # Faithful reproduction of original CSD v4.1: uncentered, Gaussian R (r=128), fp32, shared R
            assert teacher_data is not None
            t_l2 = F.normalize(teacher_data["l2"].float(), p=2, dim=1)
            t_l3 = F.normalize(teacher_data["l3"].float(), p=2, dim=1)
            assert s_l3_dim == t_l3.shape[1], "Original CSD v4.1 requires C_S == C_T!"
            R3_shared = make_random_matrix(s_l3_dim, r, seed=seed, orthogonal=False, device=device)
            R2_shared = make_random_matrix(s_l2_dim, r, seed=seed + 1, orthogonal=False, device=device)
            phi_l3_all = t_l3 @ R3_shared.cpu()
            phi_l2_all = t_l2 @ R2_shared.cpu()
            if method == "csd_v41_class":
                labels_t = teacher_data["labels"].long()
                cf3 = torch.stack([phi_l3_all[labels_t == c].mean(0) for c in range(config.num_classes)])
                cf2 = torch.stack([phi_l2_all[labels_t == c].mean(0) for c in range(config.num_classes)])
                fp_l3_table = cf3.to(device)
                fp_l2_table = cf2.to(device)
                storage_bytes = (
                    compute_pack_bytes(cf3) + (compute_pack_bytes(cf2) if use_multi_layer else 0)
                    + compute_pack_bytes(R3_shared.cpu()) + (compute_pack_bytes(R2_shared.cpu()) if use_multi_layer else 0)
                )
            else:
                fp_l3_table = phi_l3_all.to(device)
                fp_l2_table = phi_l2_all.to(device)
                storage_bytes = (
                    compute_pack_bytes(phi_l3_all) + (compute_pack_bytes(phi_l2_all) if use_multi_layer else 0)
                    + compute_pack_bytes(R3_shared.cpu()) + (compute_pack_bytes(R2_shared.cpu()) if use_multi_layer else 0)
                )

        elif method == "csd_proto_random":
            # Sanity-check control: random orthogonal class prototypes (ZERO teacher information!)
            # Tests whether teacher geometry actually beats generic ETF/simplex prototype alignment.
            gen = torch.Generator(device="cpu").manual_seed(seed + 999)
            rand_proto = F.normalize(torch.randn(config.num_classes, r, generator=gen), p=2, dim=1)
            fp_l3_table = rand_proto.to(device)
            class_gram_T = (rand_proto @ rand_proto.T).to(device)
            anchor_profile_table = class_gram_T.clone()
            proj_l3 = ProjectorHead(s_l3_dim, r, hidden_dim=max(s_l3_dim, r)).to(device)
            extra_modules.append(proj_l3)
            storage_bytes = 0

        else:
            assert teacher_data is not None
            artifacts = build_csd_artifacts(
                teacher_data, num_classes=config.num_classes,
                r=r, quant=quant, orthogonal=orthogonal, seed=seed,
                top_m=top_m, center=center,
            )

            if method == "offline_class_kd":
                offline_logits_table = dequantize_fingerprints(artifacts["offline_class_logits"], device=device)
                storage_bytes = artifacts["storage_bytes"]["offline_class_logits"]

            elif method == "offline_sparse_kd":
                offline_logits_table = unpack_sparse_topk(artifacts["offline_sparse_logits"], device=device)
                storage_bytes = artifacts["storage_bytes"]["offline_sparse_logits"]

            elif method == "csd_proto":
                ca = artifacts["class_artifacts"]
                fp_l3_table = dequantize_fingerprints(ca["fp_l3"], device=device)
                class_gram_T = dequantize_fingerprints(ca["class_gram"], device=device)
                anchor_profile_table = dequantize_fingerprints(ca["class_anchor_profile"], device=device)
                if use_projector:
                    proj_l3 = ProjectorHead(s_l3_dim, r, hidden_dim=max(s_l3_dim, r)).to(device)
                    extra_modules.append(proj_l3)
                else:
                    R3_shared = make_random_matrix(s_l3_dim, r, seed=seed, orthogonal=orthogonal, device=device)
                if use_multi_layer:
                    fp_l2_table = dequantize_fingerprints(ca["fp_l2"], device=device)
                    if use_projector:
                        proj_l2 = ProjectorHead(s_l2_dim, r).to(device)
                        extra_modules.append(proj_l2)
                    else:
                        R2_shared = make_random_matrix(s_l2_dim, r, seed=seed + 1, orthogonal=orthogonal, device=device)
                    storage_bytes = artifacts["storage_bytes"]["csd_proto_full"]
                else:
                    storage_bytes = artifacts["storage_bytes"]["csd_proto_l3_only"]

            elif method == "csd_rel":
                sja = artifacts["sample_jl_artifacts"]
                fp_l3_table = dequantize_fingerprints(sja["l3"], device=device)
                if use_multi_layer:
                    fp_l2_table = dequantize_fingerprints(sja["l2"], device=device)
                    storage_bytes = artifacts["storage_bytes"]["csd_rel_multi"]
                else:
                    storage_bytes = artifacts["storage_bytes"]["csd_rel_l3"]

            elif method == "csd_sample_proj":
                sja = artifacts["sample_jl_artifacts"]
                fp_l3_table = dequantize_fingerprints(sja["l3"], device=device)
                if use_projector:
                    proj_l3 = ProjectorHead(s_l3_dim, r, hidden_dim=max(s_l3_dim, r)).to(device)
                    extra_modules.append(proj_l3)
                else:
                    R3_shared = make_random_matrix(s_l3_dim, r, seed=seed, orthogonal=orthogonal, device=device)
                storage_bytes = artifacts["storage_bytes"]["csd_rel_l3"]

            elif method == "csd_anchor_sample":
                anchor_profile_table = unpack_sparse_topk(artifacts["sample_anchor_artifacts"]["sparse_anchor"], device=device)
                class_gram_T = dequantize_fingerprints(artifacts["class_artifacts"]["class_gram"], device=device)
                storage_bytes = (
                    artifacts["storage_bytes"]["csd_anchor_sparse"]
                    + compute_pack_bytes(artifacts["class_artifacts"]["class_gram"])
                )

            elif method == "csd_pp":
                sja = artifacts["sample_jl_artifacts"]
                ca = artifacts["class_artifacts"]
                fp_l3_table = dequantize_fingerprints(sja["l3"], device=device)
                class_gram_T = dequantize_fingerprints(ca["class_gram"], device=device)
                anchor_profile_table = unpack_sparse_topk(artifacts["sample_anchor_artifacts"]["sparse_anchor"], device=device)
                if use_projector:
                    proj_l3 = ProjectorHead(s_l3_dim, r, hidden_dim=max(s_l3_dim, r)).to(device)
                    extra_modules.append(proj_l3)
                storage_bytes = artifacts["storage_bytes"]["csd_pp_full"]

            elif method in ("csd_hybrid", "csd_dkd_hybrid"):
                sja = artifacts["sample_jl_artifacts"]
                fp_l3_table = dequantize_fingerprints(sja["l3"], device=device)
                offline_logits_table = unpack_sparse_topk(artifacts["offline_sparse_logits"], device=device)
                storage_bytes = (
                    artifacts["storage_bytes"]["csd_rel_l3"]
                    + artifacts["storage_bytes"]["offline_sparse_logits"]
                )
                if use_multi_layer:
                    fp_l2_table = dequantize_fingerprints(sja["l2"], device=device)
                    storage_bytes = (
                        artifacts["storage_bytes"]["csd_rel_multi"]
                        + artifacts["storage_bytes"]["offline_sparse_logits"]
                    )

    student = _maybe_compile(raw_student, config)
    params = list(raw_student.parameters())
    for m in extra_modules:
        params.extend(list(m.parameters()))

    optimizer = optim.SGD(
        params, lr=(lr_override if lr_override is not None else config.lr),
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, epochs, config)

    # EMA feature means for stable centering across mini-batches
    ema_mean_l3 = torch.zeros(1, s_l3_dim, device=device)
    ema_mean_l2 = torch.zeros(1, s_l2_dim, device=device)
    ema_t_mean_l3 = None
    if online_teacher is not None and method == "sp_online":
        ema_t_mean_l3 = torch.zeros(1, online_teacher.feature_dims[1], device=device)
    ema_initialized = False
    ema_momentum = 0.9

    # Optional zero-disk-storage FIFO relational memory queue
    rel_queue_l3: SimHashRelationalQueue | None = None
    if queue_size > 0 and fp_l3_table is not None:
        rel_queue_l3 = SimHashRelationalQueue(queue_size, s_l3_dim, fp_l3_table.shape[1], device=device)

    history: dict[str, Any] = {
        "method": method, "tag": tag or method,
        "student_arch": student_arch, "teacher_arch": teacher_arch,
        "seed": seed, "epochs": epochs, "r": r, "quant": quant,
        "storage_bytes": storage_bytes, "storage_str": format_bytes(storage_bytes),
        "teacher_forwards_train": teacher_forwards_train,
        "student_params": count_parameters(raw_student),
        "train_loss": [], "train_acc": [], "val_acc": [], "fp_alignment": [],
    }
    best_acc = 0.0
    best_state = None

    sync_device(device)
    t_start = time.time()
    epoch_times: list[float] = []

    for epoch in range(1, epochs + 1):
        sync_device(device)
        t_ep = time.time()
        student.train()
        for m in extra_modules:
            m.train()

        # Linear warmup for distillation losses
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warm_scale = epoch / float(warmup_epochs)
        else:
            warm_scale = 1.0

        r_loss, r_fp_sim, correct, total = 0.0, 0.0, 0, 0

        for batch in train_loader:
            if needs_index:
                x, y, idx = batch
            else:
                x, y = batch
                idx = None

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device.type, dtype=config.amp_dtype,
                enabled=config.use_amp and device.type in ("cuda", "mps"),
            ):
                if method in ("baseline", "label_smooth", "kd_online", "offline_class_kd", "offline_sparse_kd"):
                    logits = student(x)
                    f_l2 = f_l3 = None
                elif use_multi_layer and method in ("csd_v41_class", "csd_v41_sample", "csd_proto", "csd_rel", "csd_hybrid", "csd_dkd_hybrid"):
                    logits, fmap_l2, fmap_l3 = student(x, return_features="multi")
                    f_l2 = F.adaptive_avg_pool2d(fmap_l2, (1, 1)).flatten(1)
                    f_l3 = F.adaptive_avg_pool2d(fmap_l3, (1, 1)).flatten(1)
                else:
                    logits, fmap_l3 = student(x, return_features=True)
                    f_l3 = F.adaptive_avg_pool2d(fmap_l3, (1, 1)).flatten(1)
                    f_l2 = None

            # Compute losses in fp32 for full numerical stability
            logits_f = logits.float()
            f_l3_f = f_l3.float() if f_l3 is not None else None
            f_l2_f = f_l2.float() if f_l2 is not None else None

            # Update EMA feature means for centering
            if f_l3_f is not None and center:
                with torch.no_grad():
                    bm3 = f_l3_f.mean(dim=0, keepdim=True)
                    if not ema_initialized:
                        ema_mean_l3.copy_(bm3)
                        if f_l2_f is not None:
                            ema_mean_l2.copy_(f_l2_f.mean(dim=0, keepdim=True))
                        ema_initialized = True
                    else:
                        ema_mean_l3.mul_(ema_momentum).add_(bm3, alpha=1.0 - ema_momentum)
                        if f_l2_f is not None:
                            ema_mean_l2.mul_(ema_momentum).add_(f_l2_f.mean(dim=0, keepdim=True), alpha=1.0 - ema_momentum)

            fp_sim_val = 0.0

            if method == "baseline":
                loss = F.cross_entropy(logits_f, y)

            elif method == "label_smooth":
                loss = F.cross_entropy(logits_f, y, label_smoothing=0.1)

            elif method == "kd_online":
                assert online_teacher is not None
                with torch.no_grad():
                    with torch.autocast(
                        device.type, dtype=config.amp_dtype,
                        enabled=config.use_amp and device.type in ("cuda", "mps"),
                    ):
                        t_logits = online_teacher(x).float()
                ce = F.cross_entropy(logits_f, y)
                kd = F.kl_div(
                    F.log_softmax(logits_f / kd_temp, dim=1),
                    F.softmax(t_logits / kd_temp, dim=1),
                    reduction="batchmean",
                ) * (kd_temp * kd_temp)
                loss = (1.0 - kd_alpha) * ce + kd_alpha * kd

            elif method == "fitnet_online":
                assert online_teacher is not None and f_l3_f is not None
                with torch.no_grad():
                    with torch.autocast(
                        device.type, dtype=config.amp_dtype,
                        enabled=config.use_amp and device.type in ("cuda", "mps"),
                    ):
                        _, t_fmap = online_teacher(x, return_features=True)
                    t_pooled = F.adaptive_avg_pool2d(t_fmap.float(), (1, 1)).flatten(1)
                ce = F.cross_entropy(logits_f, y)
                s_aligned = extra_modules[0](f_l3_f) if extra_modules else f_l3_f
                loss = ce + fitnet_beta * F.mse_loss(s_aligned, t_pooled)

            elif method == "sp_online":
                # Online Similarity-Preserving KD (full-precision teacher oracle for SimHash-CSD)
                assert online_teacher is not None and f_l3_f is not None
                with torch.no_grad():
                    with torch.autocast(
                        device.type, dtype=config.amp_dtype,
                        enabled=config.use_amp and device.type in ("cuda", "mps"),
                    ):
                        _, t_fmap = online_teacher(x, return_features=True)
                    t_pooled = F.adaptive_avg_pool2d(t_fmap.float(), (1, 1)).flatten(1)
                    if center and ema_t_mean_l3 is not None:
                        tm = t_pooled.mean(dim=0, keepdim=True)
                        if epoch == 1 and total == 0:
                            ema_t_mean_l3.copy_(tm)
                        else:
                            ema_t_mean_l3.mul_(ema_momentum).add_(tm, alpha=1.0 - ema_momentum)
                        t_norm = center_and_normalize(t_pooled, ema_t_mean_l3)
                    else:
                        t_norm = F.normalize(t_pooled, p=2, dim=1)
                ce = F.cross_entropy(logits_f, y)
                s3_norm = center_and_normalize(f_l3_f, ema_mean_l3 if center else None) if center else F.normalize(f_l3_f, p=2, dim=1)
                l_rel = csd_jl_relational_loss(
                    s3_norm, t_norm, is_1bit=False, tau=tau_rel,
                    kl_weight=rel_kl_weight, mse_weight=rel_mse_weight,
                )
                loss = ce + (warm_scale * lam_rel) * l_rel

            elif method == "offline_class_kd":
                assert offline_logits_table is not None
                t_logits = offline_logits_table[y]
                ce = F.cross_entropy(logits_f, y)
                kd = F.kl_div(
                    F.log_softmax(logits_f / kd_temp, dim=1),
                    F.softmax(t_logits / kd_temp, dim=1),
                    reduction="batchmean",
                ) * (kd_temp * kd_temp)
                loss = (1.0 - kd_alpha) * ce + kd_alpha * kd

            elif method == "offline_sparse_kd":
                assert offline_logits_table is not None and idx is not None
                t_logits = offline_logits_table[idx]
                ce = F.cross_entropy(logits_f, y)
                kd = F.kl_div(
                    F.log_softmax(logits_f / kd_temp, dim=1),
                    F.softmax(t_logits / kd_temp, dim=1),
                    reduction="batchmean",
                ) * (kd_temp * kd_temp)
                loss = (1.0 - kd_alpha) * ce + kd_alpha * kd

            elif method in ("csd_v41_class", "csd_v41_sample"):
                assert f_l3_f is not None and R3_shared is not None and fp_l3_table is not None
                ce = F.cross_entropy(logits_f, y)
                phi_s3 = F.normalize(f_l3_f, p=2, dim=1) @ R3_shared
                target3 = fp_l3_table[y] if method == "csd_v41_class" else fp_l3_table[idx]
                d3, sim3 = csd_projected_cosine_loss(phi_s3, target3)
                distil = d3
                fp_sim_val = sim3
                if use_multi_layer and f_l2_f is not None and R2_shared is not None and fp_l2_table is not None:
                    phi_s2 = F.normalize(f_l2_f, p=2, dim=1) @ R2_shared
                    target2 = fp_l2_table[y] if method == "csd_v41_class" else fp_l2_table[idx]
                    d2, sim2 = csd_projected_cosine_loss(phi_s2, target2)
                    distil = distil + l2_weight * d2
                    fp_sim_val = 0.5 * (sim3 + sim2)
                loss = ce + (warm_scale * lam_proj) * distil

            elif method in ("csd_proto", "csd_proto_random"):
                assert f_l3_f is not None and fp_l3_table is not None
                assert anchor_profile_table is not None and class_gram_T is not None
                ce = F.cross_entropy(logits_f, y)
                s3_norm = center_and_normalize(f_l3_f, ema_mean_l3 if center else None)

                # 1. Centered projected class-prototype alignment
                if proj_l3 is not None:
                    phi_s3 = proj_l3(s3_norm)
                else:
                    assert R3_shared is not None
                    phi_s3 = s3_norm @ R3_shared
                l_proj, fp_sim_val = csd_projected_cosine_loss(phi_s3, fp_l3_table[y])

                if use_multi_layer and f_l2_f is not None and fp_l2_table is not None:
                    s2_norm = center_and_normalize(f_l2_f, ema_mean_l2 if center else None)
                    phi_s2 = proj_l2(s2_norm) if proj_l2 is not None else (s2_norm @ R2_shared)
                    l_proj2, sim2 = csd_projected_cosine_loss(phi_s2, fp_l2_table[y])
                    l_proj = l_proj + l2_weight * l_proj2
                    fp_sim_val = 0.5 * (fp_sim_val + sim2)

                # 2. Prototype-anchored semantic similarity KL + classifier Gram alignment
                w_s = _get_classifier_weight(raw_student).float()
                l_anchor = csd_anchor_semantic_loss(
                    s3_norm, w_s, anchor_profile_table[y],
                    t_class_gram=class_gram_T, tau=tau_anchor, gram_weight=gram_weight,
                )

                loss = ce + warm_scale * (lam_proj * l_proj + lam_anchor * l_anchor)

            elif method == "csd_rel":
                assert f_l3_f is not None and fp_l3_table is not None and idx is not None
                ce = F.cross_entropy(logits_f, y)
                s3_norm = center_and_normalize(f_l3_f, ema_mean_l3 if center else None) if center else F.normalize(f_l3_f, p=2, dim=1)
                sq, tq = rel_queue_l3.get() if (rel_queue_l3 is not None and epoch > 5) else (None, None)
                l_rel = csd_jl_relational_loss(
                    s3_norm, fp_l3_table[idx], is_1bit=is_1bit, tau=tau_rel,
                    kl_weight=rel_kl_weight, mse_weight=rel_mse_weight,
                    s_queue=sq, t_queue=tq,
                )
                if rel_queue_l3 is not None:
                    rel_queue_l3.enqueue(s3_norm, fp_l3_table[idx])
                if use_multi_layer and f_l2_f is not None and fp_l2_table is not None:
                    s2_norm = center_and_normalize(f_l2_f, ema_mean_l2 if center else None) if center else F.normalize(f_l2_f, p=2, dim=1)
                    l_rel2 = csd_jl_relational_loss(
                        s2_norm, fp_l2_table[idx], is_1bit=is_1bit, tau=tau_rel,
                        kl_weight=rel_kl_weight, mse_weight=rel_mse_weight,
                    )
                    l_rel = l_rel + l2_weight * l_rel2
                loss = ce + (warm_scale * lam_rel) * l_rel

            elif method == "csd_sample_proj":
                assert f_l3_f is not None and fp_l3_table is not None and idx is not None
                ce = F.cross_entropy(logits_f, y)
                s3_norm = center_and_normalize(f_l3_f, ema_mean_l3 if center else None)
                phi_s3 = proj_l3(s3_norm) if proj_l3 is not None else (s3_norm @ R3_shared)
                l_proj, fp_sim_val = csd_projected_cosine_loss(phi_s3, fp_l3_table[idx])
                loss = ce + (warm_scale * lam_proj) * l_proj

            elif method == "csd_anchor_sample":
                assert f_l3_f is not None and anchor_profile_table is not None and idx is not None
                ce = F.cross_entropy(logits_f, y)
                s3_norm = center_and_normalize(f_l3_f, ema_mean_l3 if center else None)
                w_s = _get_classifier_weight(raw_student).float()
                l_anchor = csd_anchor_semantic_loss(
                    s3_norm, w_s, anchor_profile_table[idx],
                    t_class_gram=class_gram_T, tau=tau_anchor, gram_weight=gram_weight,
                )
                loss = ce + (warm_scale * lam_anchor) * l_anchor

            elif method == "csd_pp":
                assert f_l3_f is not None and fp_l3_table is not None
                assert anchor_profile_table is not None and idx is not None
                ce = F.cross_entropy(logits_f, y)
                s3_norm = center_and_normalize(f_l3_f, ema_mean_l3 if center else None)
                t_sketch = fp_l3_table[idx]

                # A. Centered projected cosine alignment (via training-only projector)
                l_proj = torch.tensor(0.0, device=device)
                if proj_l3 is not None and lam_proj > 0.0 and not is_1bit:
                    phi_s3 = proj_l3(s3_norm)
                    l_proj, fp_sim_val = csd_projected_cosine_loss(phi_s3, t_sketch)

                # B. JL intra-batch relational cosine distillation
                l_rel = torch.tensor(0.0, device=device)
                if lam_rel > 0.0:
                    l_rel = csd_jl_relational_loss(
                        s3_norm, t_sketch, is_1bit=is_1bit, tau=tau_rel, mse_weight=1.0,
                    )

                # C. Per-sample prototype-anchored semantic cosine KL + class Gram alignment
                l_anchor = torch.tensor(0.0, device=device)
                if lam_anchor > 0.0:
                    w_s = _get_classifier_weight(raw_student).float()
                    l_anchor = csd_anchor_semantic_loss(
                        s3_norm, w_s, anchor_profile_table[idx],
                        t_class_gram=class_gram_T, tau=tau_anchor, gram_weight=gram_weight,
                    )

                loss = ce + warm_scale * (lam_proj * l_proj + lam_rel * l_rel + lam_anchor * l_anchor)

            elif method == "csd_hybrid":
                assert f_l3_f is not None and fp_l3_table is not None
                assert offline_logits_table is not None and idx is not None
                t_logits = offline_logits_table[idx]
                ce = F.cross_entropy(logits_f, y)
                kd = F.kl_div(
                    F.log_softmax(logits_f / kd_temp, dim=1),
                    F.softmax(t_logits / kd_temp, dim=1),
                    reduction="batchmean",
                ) * (kd_temp * kd_temp)
                s3_norm = center_and_normalize(f_l3_f, ema_mean_l3 if center else None)
                sq, tq = rel_queue_l3.get() if (rel_queue_l3 is not None and epoch > 5) else (None, None)
                l_rel = csd_jl_relational_loss(
                    s3_norm, fp_l3_table[idx], is_1bit=is_1bit, tau=tau_rel, mse_weight=1.0,
                    s_queue=sq, t_queue=tq,
                )
                if rel_queue_l3 is not None:
                    rel_queue_l3.enqueue(s3_norm, fp_l3_table[idx])
                if use_multi_layer and f_l2_f is not None and fp_l2_table is not None:
                    s2_norm = center_and_normalize(f_l2_f, ema_mean_l2 if center else None)
                    l_rel2 = csd_jl_relational_loss(
                        s2_norm, fp_l2_table[idx], is_1bit=is_1bit, tau=tau_rel, mse_weight=1.0,
                    )
                    l_rel = l_rel + l2_weight * l_rel2
                loss = (1.0 - kd_alpha) * ce + kd_alpha * kd + (warm_scale * lam_rel) * l_rel

            elif method == "csd_dkd_hybrid":
                assert f_l3_f is not None and fp_l3_table is not None
                assert offline_logits_table is not None and idx is not None
                t_logits = offline_logits_table[idx]
                ce = F.cross_entropy(logits_f, y)
                dkd = dkd_loss_fn(logits_f, t_logits, y, T=kd_temp, alpha=dkd_alpha, beta=dkd_beta)
                s3_norm = center_and_normalize(f_l3_f, ema_mean_l3 if center else None)
                sq, tq = rel_queue_l3.get() if (rel_queue_l3 is not None and epoch > 5) else (None, None)
                l_rel = csd_jl_relational_loss(
                    s3_norm, fp_l3_table[idx], is_1bit=is_1bit, tau=tau_rel, mse_weight=1.0,
                    s_queue=sq, t_queue=tq,
                )
                if rel_queue_l3 is not None:
                    rel_queue_l3.enqueue(s3_norm, fp_l3_table[idx])
                loss = ce + warm_scale * (dkd + lam_rel * l_rel)

            else:
                raise ValueError(f"Unknown method: {method}")

            loss.backward()
            optimizer.step()

            bs = x.size(0)
            r_loss += loss.item() * bs
            r_fp_sim += fp_sim_val * bs
            correct += logits_f.argmax(dim=1).eq(y).sum().item()
            total += bs

        scheduler.step()
        sync_device(device)
        epoch_times.append(time.time() - t_ep)

        train_loss = r_loss / total
        train_acc = 100.0 * correct / total
        fp_align = r_fp_sim / total
        _, eval_acc = evaluate_model(raw_student, eval_loader, device, tta=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(eval_acc)
        history["fp_alignment"].append(fp_align)

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_state = _unwrap_state_dict(raw_student)

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            elapsed = (time.time() - t_start) / 60.0
            print(
                f"[{tag or method} {student_arch} s={seed}] Ep {epoch:3d}/{epochs} | "
                f"Loss {train_loss:.4f} | Tr {train_acc:.2f}% | Val {eval_acc:.2f}% | "
                f"Best {best_acc:.2f}% | FP {fp_align:.3f} | {elapsed:.1f}m",
                flush=True,
            )

    # Final evaluation on test set (both last-epoch and best-checkpoint, std & TTA)
    _, last_test_acc = evaluate_model(raw_student, test_loader, device, tta=False)
    assert best_state is not None
    raw_student.load_state_dict(best_state)
    _, best_test_acc = evaluate_model(raw_student, test_loader, device, tta=False)
    _, best_test_tta = evaluate_model(raw_student, test_loader, device, tta=True)

    run_name = f"{tag or method}_{student_arch}_s{seed}_e{epochs}"
    if save_ckpt:
        torch.save(best_state, os.path.join(config.results_dir, f"{run_name}.pth"))

    history["best_val_acc"] = best_acc
    history["best_test_acc"] = best_test_acc
    history["last_test_acc"] = last_test_acc
    history["tta_test_acc"] = best_test_tta
    history["sec_per_epoch"] = float(sum(epoch_times[1:]) / max(1, len(epoch_times) - 1)) if len(epoch_times) > 1 else epoch_times[0]
    history["wall_minutes"] = (time.time() - t_start) / 60.0

    with open(os.path.join(config.results_dir, f"{run_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(
        f"[{run_name}] DONE | Std: {best_test_acc:.2f}% (last {last_test_acc:.2f}%) | "
        f"TTA: {best_test_tta:.2f}% | Storage: {format_bytes(storage_bytes)} | "
        f"{history['sec_per_epoch']:.2f}s/ep ({history['wall_minutes']:.1f}m)",
        flush=True,
    )

    del online_teacher, raw_student, student, extra_modules
    empty_cache(device)
    return history


def run_privacy_audit_cli(config: Config, teacher_arch: str = "resnet56") -> list[dict[str, Any]]:
    """Run quantitative feature-inversion privacy audit and save JSON."""
    teacher_data = prepare_teacher_cache(config, teacher_arch=teacher_arch)
    results = audit_privacy_inversion(
        teacher_data["l3"],
        r_values=[8, 16, 32, 64, 128],
        quant_modes=["fp32", "int8", "1bit"],
        seed=42,
        orthogonal=True,
    )
    out_path = os.path.join(config.results_dir, f"privacy_audit_{teacher_arch}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== QUANTITATIVE PRIVACY INVERSION AUDIT ===")
    print(f"{'r':>4s} {'quant':>6s} {'Storage':>10s} {'Recon Cos':>10s} {'JL MAE':>10s}")
    for row in results:
        print(
            f"{row['r']:4d} {row['quant']:>6s} {format_bytes(row['storage_bytes']):>10s} "
            f"{row['recon_cosine']:10.4f} {row['jl_pairwise_mae']:10.4f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSD++ Experiment Runner")
    parser.add_argument("--mode", type=str, required=True, choices=["teacher", "cache", "privacy", "student"])
    parser.add_argument("--method", type=str, default="csd_pp")
    parser.add_argument("--student", type=str, default="resnet20")
    parser.add_argument("--teacher", type=str, default="resnet56")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--quant", type=str, default="int8")
    parser.add_argument("--top_m", type=int, default=12)
    parser.add_argument("--lam_proj", type=float, default=1.0)
    parser.add_argument("--lam_rel", type=float, default=2.0)
    parser.add_argument("--lam_anchor", type=float, default=2.0)
    parser.add_argument("--gram_weight", type=float, default=0.5)
    parser.add_argument("--tau_rel", type=float, default=0.15)
    parser.add_argument("--tau_anchor", type=float, default=0.10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--no_center", action="store_true")
    parser.add_argument("--no_proj", action="store_true")
    parser.add_argument("--no_multi_layer", action="store_true")
    parser.add_argument("--no_orthogonal", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default="cosine")
    args = parser.parse_args()

    cfg = Config()
    cfg.scheduler = args.scheduler

    if args.mode == "teacher":
        train_teacher_model(cfg, arch=args.teacher, epochs=args.epochs, seed=args.seed)
    elif args.mode == "cache":
        prepare_teacher_cache(cfg, teacher_arch=args.teacher)
    elif args.mode == "privacy":
        run_privacy_audit_cli(cfg, teacher_arch=args.teacher)
    elif args.mode == "student":
        train_student(
            cfg, method=args.method, student_arch=args.student, teacher_arch=args.teacher,
            epochs=args.epochs, seed=args.seed, r=args.r, quant=args.quant,
            orthogonal=not args.no_orthogonal, top_m=args.top_m,
            center=not args.no_center, use_projector=not args.no_proj,
            use_multi_layer=not args.no_multi_layer,
            lam_proj=args.lam_proj, lam_rel=args.lam_rel, lam_anchor=args.lam_anchor,
            gram_weight=args.gram_weight, tau_rel=args.tau_rel, tau_anchor=args.tau_anchor,
            warmup_epochs=args.warmup, tag=args.tag,
        )
