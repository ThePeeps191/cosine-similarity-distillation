"""Capstone experiments: Headless ViT-B/16 -> CNN distillation, Heterogeneous Multi-Teacher
SimHash-CSD Ensembles (CNN+ViT on STL-10 and ResNet-56+ResNet-32x4 on CIFAR-100), and
exact Apple M6 MPS hardware memory/speed profiling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision.models import ViT_B_16_Weights, vit_b_16

from config import Config
from csd_core import (
    center_and_normalize,
    charikar_cosine_matrix,
    compute_pack_bytes,
    dequantize_fingerprints,
    make_random_matrix,
    quantize_fingerprints,
)
from data import FastCIFAR100TensorLoader, get_fast_cifar100_loaders
from experiments import evaluate_model, prepare_teacher_cache
from models import get_model
from stl10_bench import STL_MEAN, STL_STD, _eval_stl, _load_stl10_tensors
from utils import empty_cache, format_bytes, get_model_size_bytes, set_seed, sync_device


def ensemble_simhash_loss(
    s_feat_norm: torch.Tensor,
    t_signs_list: list[torch.Tensor],
    tau: float = 0.15,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Zero-overhead heterogeneous multi-teacher SimHash-CSD loss.

    Averages Charikar cosine similarity matrices across M arbitrary teachers
    (e.g., CNN + Vision Transformer) without any inter-teacher alignment or extra params.
    """
    B = s_feat_norm.size(0)
    G_S = s_feat_norm @ s_feat_norm.T

    if weights is None:
        weights = [1.0 / len(t_signs_list)] * len(t_signs_list)

    with torch.no_grad():
        G_T = torch.zeros(B, B, device=G_S.device, dtype=torch.float32)
        for w, signs in zip(weights, t_signs_list):
            G_T += w * charikar_cosine_matrix(signs)

    eye_mask = torch.eye(B, device=G_S.device, dtype=torch.bool)
    off_S = G_S.masked_fill(eye_mask, -1e4)
    off_T = G_T.masked_fill(eye_mask, -1e4)
    p_T = F.softmax(off_T / tau, dim=1)
    log_p_S = F.log_softmax(off_S / tau, dim=1)
    kl = F.kl_div(log_p_S, p_T, reduction="batchmean") * tau
    mse = F.mse_loss(G_S[~eye_mask], G_T[~eye_mask])
    return kl + mse


# ============================================================================
# 1. Extract Headless Pretrained ViT-B/16 CLS Embeddings on STL-10 (Once!)
# ============================================================================

@torch.no_grad()
def extract_vitb16_stl10(config: Config) -> dict[str, Any]:
    os.makedirs(config.results_dir, exist_ok=True)
    cache_path = os.path.join(config.results_dir, "stl10_vitb16_feats.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    device = config.device
    t0 = time.time()
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    vit_bytes = get_model_size_bytes(model)
    # Remove classification head -> pure headless Vision Transformer backbone (768-d CLS token)
    model.heads = nn.Identity()
    model = model.to(device).eval()

    x_tr, y_tr, _, _ = _load_stl10_tensors(img_size=96)
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    N = len(x_tr)
    feats = torch.zeros(N, 768, dtype=torch.float32)
    bs = 64
    for i in range(0, N, bs):
        bx = x_tr[i : i + bs].to(device)
        bx_224 = F.interpolate(bx, size=(224, 224), mode="bilinear", align_corners=False)
        bx_norm = (bx_224 - im_mean) / im_std
        bx_flip = (bx_224.flip(-1) - im_mean) / im_std
        with torch.autocast(device.type, dtype=config.amp_dtype, enabled=config.use_amp and device.type in ("cuda", "mps")):
            f1 = model(bx_norm).float()
            f2 = model(bx_flip).float()
        feats[i : i + bs] = (0.5 * (f1 + f2)).cpu()

    data = {"cls": feats, "labels": y_tr, "teacher_bytes": vit_bytes}
    torch.save(data, cache_path)
    del model
    empty_cache(device)
    print(
        f"[ViT-B/16 Headless] Extracted 2-view 768-d CLS features on STL-10 in {time.time()-t0:.1f}s "
        f"(Teacher size: {format_bytes(vit_bytes)})",
        flush=True,
    )
    return data


# ============================================================================
# 2. STL-10 Headless ViT-B/16 -> ResNet-20 & CNN+ViT Multi-Teacher Ensemble
# ============================================================================

def run_stl10_vit_and_ensemble(config: Config, epochs: int = 150, seeds: list[int] = [42, 123, 456]) -> None:
    device = config.device
    vit_data = extract_vitb16_stl10(config)
    cnn_data = torch.load(os.path.join(config.results_dir, "stl10_teacher_feats.pt"), map_location="cpu", weights_only=False)
    x_tr, y_tr, x_te, y_te = _load_stl10_tensors(img_size=64)

    vit_bytes = vit_data["teacher_bytes"]
    cnn_bytes = 855754 * 4  # ResNet-56 (10 classes)

    # Build 1-bit SimHash sketches for headless ViT-B/16 (C=768 -> r=64 and r=128)
    f_vit = vit_data["cls"].float()
    f_vit_c = F.normalize(f_vit - f_vit.mean(0, keepdim=True), dim=1)

    R_v64 = make_random_matrix(768, 64, seed=42, orthogonal=True)
    p_v64 = quantize_fingerprints(f_vit_c @ R_v64, mode="1bit")
    signs_v64 = dequantize_fingerprints(p_v64, device=device)
    bytes_v64 = compute_pack_bytes(p_v64) + 4

    R_v128 = make_random_matrix(768, 128, seed=42, orthogonal=True)
    p_v128 = quantize_fingerprints(f_vit_c @ R_v128, mode="1bit")
    signs_v128 = dequantize_fingerprints(p_v128, device=device)
    bytes_v128 = compute_pack_bytes(p_v128) + 4

    # Build 1-bit SimHash sketches for CNN ResNet-56 (C=64 -> r=64)
    f_cnn = cnn_data["l3"].float()
    f_cnn_c = F.normalize(f_cnn - f_cnn.mean(0, keepdim=True), dim=1)
    R_c64 = make_random_matrix(64, 64, seed=42, orthogonal=True)
    p_c64 = quantize_fingerprints(f_cnn_c @ R_c64, mode="1bit")
    signs_c64 = dequantize_fingerprints(p_c64, device=device)
    bytes_c64 = compute_pack_bytes(p_c64) + 4

    specs = [
        ("stl10_vit_r64", "Headless ViT-B/16 -> ResNet-20 (r=64)", [signs_v64], bytes_v64, vit_bytes),
        ("stl10_vit_r128", "Headless ViT-B/16 -> ResNet-20 (r=128)", [signs_v128], bytes_v128, vit_bytes),
        ("stl10_ens_cnn_vit", "Ensemble (ResNet-56 + ViT-B/16) -> ResNet-20", [signs_c64, signs_v128], bytes_c64 + bytes_v128, cnn_bytes + vit_bytes),
    ]

    summary = []
    for tag, label, s_list, sbytes, tbytes in specs:
        accs, sp_times = [], []
        for seed in seeds:
            out_j = os.path.join(config.results_dir, f"{tag}_s{seed}_e{epochs}.json")
            if os.path.exists(out_j):
                h = json.load(open(out_j))
                accs.append(h["best_acc"])
                sp_times.append(h["sec_per_epoch"])
                print(f"[SKIP] {tag} s={seed}: {h['best_acc']:.2f}%", flush=True)
                continue

            set_seed(seed)
            raw_s = get_model("resnet20", num_classes=10).to(device)
            s_model = torch.compile(raw_s) if config.use_compile and device.type in ("cuda", "mps") else raw_s
            opt = optim.SGD(raw_s.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            sched = CosineAnnealingLR(opt, T_max=epochs)
            tr_loader = FastCIFAR100TensorLoader(
                x_tr, y_tr, STL_MEAN, STL_STD, batch_size=64, shuffle=True, augment=True,
                return_index=True, device=device,
            )

            ema_s = torch.zeros(1, 64, device=device)
            ema_init = False
            best_acc = 0.0
            ep_times = []
            for ep in range(1, epochs + 1):
                sync_device(device)
                t_ep = time.time()
                s_model.train()
                warm = min(1.0, ep / 15.0)
                for bx, by, bidx in tr_loader:
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, dtype=config.amp_dtype, enabled=config.use_amp and device.type in ("cuda", "mps")):
                        logits, l3 = s_model(bx, return_features=True)
                        f3 = F.adaptive_avg_pool2d(l3, (1, 1)).flatten(1)
                    logits_f, f3_f = logits.float(), f3.float()
                    with torch.no_grad():
                        bm = f3_f.mean(0, keepdim=True)
                        if not ema_init:
                            ema_s.copy_(bm)
                            ema_init = True
                        else:
                            ema_s.mul_(0.9).add_(bm, alpha=0.1)
                    ce = F.cross_entropy(logits_f, by)
                    sn = center_and_normalize(f3_f, ema_s)
                    l_rel = ensemble_simhash_loss(sn, [s[bidx] for s in s_list], tau=0.15)
                    loss = ce + (5.0 * warm) * l_rel
                    loss.backward()
                    opt.step()
                sched.step()
                sync_device(device)
                ep_times.append(time.time() - t_ep)
                acc = _eval_stl(raw_s, x_te, y_te, device)
                if acc > best_acc:
                    best_acc = acc

            sec_ep = float(np.mean(ep_times[1:])) if len(ep_times) > 1 else ep_times[0]
            json.dump(
                {"tag": tag, "seed": seed, "best_acc": best_acc, "sec_per_epoch": sec_ep, "storage_bytes": sbytes, "teacher_bytes": tbytes},
                open(out_j, "w"), indent=2,
            )
            print(
                f"[{tag} s={seed}] DONE | Best: {best_acc:.2f}% | {sec_ep:.2f}s/ep | "
                f"Storage: {format_bytes(sbytes)} ({tbytes/sbytes:.0f}x smaller!)",
                flush=True,
            )
            accs.append(best_acc)
            sp_times.append(sec_ep)
            del raw_s, s_model
            empty_cache(device)

        summary.append({
            "tag": tag, "label": label,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "mean_sec_ep": float(np.mean(sp_times)),
            "storage_bytes": sbytes, "storage_str": format_bytes(sbytes),
            "teacher_bytes": tbytes, "compression_ratio": float(tbytes / sbytes),
            "per_seed": accs,
        })

    out_s = os.path.join(config.results_dir, f"summary_stl10_vit_ens_e{epochs}.json")
    json.dump(summary, open(out_s, "w"), indent=2)
    print("\n" + "=" * 96)
    print(f"HEADLESS ViT-B/16 & CNN+ViT ENSEMBLE ON STL-10 ({epochs} ep, 3 seeds)")
    print("=" * 96)
    for r in summary:
        print(
            f"  {r['label']:<46s} | {r['mean_acc']:6.2f} +/- {r['std_acc']:4.2f}% | "
            f"{r['storage_str']:>9s} ({r['compression_ratio']:6.0f}x) | {r['mean_sec_ep']:5.2f}s/ep"
        )


# ============================================================================
# 3. CIFAR-100 Multi-Teacher Ensemble: ResNet-56 + ResNet-32x4 -> ResNet-20
# ============================================================================

def run_cifar100_multi_teacher(config: Config, epochs: int = 200, seeds: list[int] = [42, 123, 456]) -> None:
    device = config.device
    d56 = prepare_teacher_cache(config, teacher_arch="resnet56")
    d32 = prepare_teacher_cache(config, teacher_arch="resnet32x4")

    f56 = F.normalize(d56["l3"].float() - d56["l3"].float().mean(0, keepdim=True), dim=1)
    f32 = F.normalize(d32["l3"].float() - d32["l3"].float().mean(0, keepdim=True), dim=1)

    p56 = quantize_fingerprints(f56 @ make_random_matrix(64, 64, seed=42, orthogonal=True), mode="1bit")
    p32 = quantize_fingerprints(f32 @ make_random_matrix(256, 64, seed=42, orthogonal=True), mode="1bit")
    s56 = dequantize_fingerprints(p56, device=device)
    s32 = dequantize_fingerprints(p32, device=device)
    sbytes = compute_pack_bytes(p56) + compute_pack_bytes(p32) + 8
    tbytes = (861620 + 7449188) * 4

    train_loader, test_loader = get_fast_cifar100_loaders(config, return_index=True, val_fraction=0.0)
    stds, ttas, lasts, sp_times = [], [], [], []
    for seed in seeds:
        hpath = os.path.join(config.results_dir, f"ens_r56_r32x4_resnet20_s{seed}_e{epochs}_history.json")
        if os.path.exists(hpath):
            h = json.load(open(hpath))
            stds.append(h["best_test_acc"]); ttas.append(h["tta_test_acc"]); lasts.append(h["last_test_acc"]); sp_times.append(h["sec_per_epoch"])
            print(f"[SKIP] ens_r56_r32x4 s={seed}: Std {h['best_test_acc']:.2f}%, TTA {h['tta_test_acc']:.2f}%", flush=True)
            continue

        set_seed(seed)
        raw_s = get_model("resnet20", num_classes=100).to(device)
        s_model = torch.compile(raw_s) if config.use_compile and device.type in ("cuda", "mps") else raw_s
        opt = optim.SGD(raw_s.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        sched = CosineAnnealingLR(opt, T_max=epochs)

        ema_s = torch.zeros(1, 64, device=device)
        ema_init = False
        best_acc, best_state = 0.0, None
        ep_times, val_hist = [], []
        t0 = time.time()
        for ep in range(1, epochs + 1):
            sync_device(device)
            t_ep = time.time()
            s_model.train()
            warm = min(1.0, ep / 20.0)
            for bx, by, bidx in train_loader:
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=config.amp_dtype, enabled=config.use_amp and device.type in ("cuda", "mps")):
                    logits, l3 = s_model(bx, return_features=True)
                    f3 = F.adaptive_avg_pool2d(l3, (1, 1)).flatten(1)
                logits_f, f3_f = logits.float(), f3.float()
                with torch.no_grad():
                    bm = f3_f.mean(0, keepdim=True)
                    if not ema_init:
                        ema_s.copy_(bm); ema_init = True
                    else:
                        ema_s.mul_(0.9).add_(bm, alpha=0.1)
                ce = F.cross_entropy(logits_f, by)
                sn = center_and_normalize(f3_f, ema_s)
                l_rel = ensemble_simhash_loss(sn, [s56[bidx], s32[bidx]], tau=0.15)
                loss = ce + (5.0 * warm) * l_rel
                loss.backward()
                opt.step()
            sched.step()
            sync_device(device)
            ep_times.append(time.time() - t_ep)
            _, acc = evaluate_model(raw_s, test_loader, device, tta=False)
            val_hist.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in raw_s.state_dict().items()}
            if ep == 1 or ep % 50 == 0 or ep == epochs:
                print(f"[Ensemble R56+R32x4 s={seed}] Ep {ep:3d}/{epochs} | Test {acc:.2f}% | Best {best_acc:.2f}% | {(time.time()-t0)/60:.1f}m", flush=True)

        last_acc = val_hist[-1]
        assert best_state is not None
        raw_s.load_state_dict(best_state)
        _, tta_acc = evaluate_model(raw_s, test_loader, device, tta=True)
        sec_ep = float(np.mean(ep_times[1:]))
        hist = {
            "tag": "ens_r56_r32x4", "seed": seed, "epochs": epochs,
            "best_test_acc": best_acc, "last_test_acc": last_acc, "tta_test_acc": tta_acc,
            "sec_per_epoch": sec_ep, "storage_bytes": sbytes, "storage_str": format_bytes(sbytes),
            "teacher_bytes": tbytes, "val_acc": val_hist,
        }
        json.dump(hist, open(hpath, "w"), indent=2)
        print(f"[ens_r56_r32x4 s={seed}] DONE | Std: {best_acc:.2f}% | TTA: {tta_acc:.2f}% | {sec_ep:.2f}s/ep", flush=True)
        stds.append(best_acc); ttas.append(tta_acc); lasts.append(last_acc); sp_times.append(sec_ep)
        del raw_s, s_model
        empty_cache(device)

    summary = {
        "tag": "ens_r56_r32x4", "label": "SimHash-CSD Multi-Teacher (R56+R32x4)",
        "mean_std": float(np.mean(stds)), "std_std": float(np.std(stds, ddof=1)),
        "mean_last": float(np.mean(lasts)), "std_last": float(np.std(lasts, ddof=1)),
        "mean_tta": float(np.mean(ttas)), "std_tta": float(np.std(ttas, ddof=1)),
        "mean_sec_ep": float(np.mean(sp_times)),
        "storage_bytes": sbytes, "storage_str": format_bytes(sbytes),
        "teacher_bytes": tbytes, "compression_ratio": float(tbytes / sbytes),
        "per_seed_std": stds,
    }
    json.dump(summary, open(os.path.join(config.results_dir, f"summary_cifar100_ensemble_e{epochs}.json"), "w"), indent=2)
    print(f"\n[CIFAR-100 Ensemble R56+R32x4] Std: {summary['mean_std']:.2f} +/- {summary['std_std']:.2f}% | TTA: {summary['mean_tta']:.2f} +/- {summary['std_tta']:.2f}% | {summary['storage_str']} ({summary['compression_ratio']:.1f}x smaller!)")


# ============================================================================
# 4. Exact Apple M6 MPS Memory & Latency Hardware Profiler
# ============================================================================

def run_hardware_profile(config: Config) -> None:
    device = config.device
    results = []

    def _measure(name: str, student_fn, teacher_fn=None, img_size: int = 32, teacher_img_size: int = 32, bs: int = 128, num_classes: int = 100):
        empty_cache(device)
        sync_device(device)
        s = student_fn().to(device)
        t = teacher_fn().to(device).eval() if teacher_fn is not None else None
        if t is not None:
            for p in t.parameters():
                p.requires_grad = False
        opt = optim.SGD(s.parameters(), lr=0.1, momentum=0.9)
        x = torch.randn(bs, 3, img_size, img_size, device=device)
        y = torch.randint(0, num_classes, (bs,), device=device)
        signs = torch.sign(torch.randn(bs, 64, device=device))

        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=config.amp_dtype, enabled=device.type in ("cuda", "mps")):
                lg, l3 = s(x, return_features=True)
                f3 = F.adaptive_avg_pool2d(l3, (1, 1)).flatten(1)
                if t is not None:
                    with torch.no_grad():
                        xt = F.interpolate(x, size=(teacher_img_size, teacher_img_size)) if teacher_img_size != img_size else x
                        t_out = t(xt)
                    loss = F.cross_entropy(lg.float(), y) + t_out.float().mean() * 0.0
                else:
                    sn = center_and_normalize(f3.float())
                    loss = F.cross_entropy(lg.float(), y) + ensemble_simhash_loss(sn, [signs])
            loss.backward()
            opt.step()

        sync_device(device)
        mem_mb = 0.0
        if device.type == "mps":
            mem_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
        elif device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        t0 = time.time()
        n_steps = 15
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=config.amp_dtype, enabled=device.type in ("cuda", "mps")):
                lg, l3 = s(x, return_features=True)
                f3 = F.adaptive_avg_pool2d(l3, (1, 1)).flatten(1)
                if t is not None:
                    with torch.no_grad():
                        xt = F.interpolate(x, size=(teacher_img_size, teacher_img_size)) if teacher_img_size != img_size else x
                        t_out = t(xt)
                    loss = F.cross_entropy(lg.float(), y) + t_out.float().mean() * 0.0
                else:
                    sn = center_and_normalize(f3.float())
                    loss = F.cross_entropy(lg.float(), y) + ensemble_simhash_loss(sn, [signs])
            loss.backward()
            opt.step()
        sync_device(device)
        ms_step = (time.time() - t0) / n_steps * 1000.0
        results.append({"name": name, "driver_mem_mb": mem_mb, "ms_per_step": ms_step})
        print(f"  {name:<42s} | {ms_step:7.2f} ms/step | Driver Mem: {mem_mb:7.1f} MB", flush=True)
        del s, t, opt, x, y
        empty_cache(device)

    print("\n=== APPLE M6 MPS HARDWARE SPEED & MEMORY PROFILE ===")
    _measure("SimHash-CSD (ResNet-20 only, bs=128)", lambda: get_model("resnet20"), None, 32, 32, 128)
    _measure("Online KD (ResNet-56 + ResNet-20, bs=128)", lambda: get_model("resnet20"), lambda: get_model("resnet56"), 32, 32, 128)
    _measure("Online KD (ResNet-32x4 + ResNet-20, bs=128)", lambda: get_model("resnet20"), lambda: get_model("resnet32x4"), 32, 32, 128)
    _measure("SimHash-CSD (ResNet-20, STL-10 64x64, bs=64)", lambda: get_model("resnet20", 10), None, 64, 64, 64, num_classes=10)
    _measure("Online ViT-B/16 + ResNet-20 (224x224, bs=64)", lambda: get_model("resnet20", 10), lambda: vit_b_16(weights=None), 64, 224, 64, num_classes=10)

    out_p = os.path.join(config.results_dir, "hardware_profile_m6.json")
    json.dump(results, open(out_p, "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["download_vit", "stl10_vit", "cifar_ens", "profile", "all"])
    args = parser.parse_args()
    cfg = Config()
    if args.mode == "download_vit":
        _ = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        print("ViT-B/16 weights downloaded and cached cleanly.")
    elif args.mode == "stl10_vit":
        run_stl10_vit_and_ensemble(cfg, epochs=150)
    elif args.mode == "cifar_ens":
        run_cifar100_multi_teacher(cfg, epochs=200)
    elif args.mode == "profile":
        run_hardware_profile(cfg)
    elif args.mode == "all":
        run_stl10_vit_and_ensemble(cfg, epochs=150)
        run_cifar100_multi_teacher(cfg, epochs=200)
        run_hardware_profile(cfg)
