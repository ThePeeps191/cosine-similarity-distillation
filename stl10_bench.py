"""Second-dataset benchmark (STL-10 natural images) & CRD-style Representation Transfer Probing.

Two complementary evaluations:
1. ``--mode transfer``: Freeze trained CIFAR-100 ResNet-20 student backbones and fit an L2-regularized
   linear probe on STL-10 (5,000 train / 8,000 test, resized to 32x32) — standard CRD (ICLR 2020)
   representation transferability benchmark (< 60 seconds).
2. ``--mode stl10``: Full end-to-end distillation benchmark on STL-10 (64x64 natural images, 10 classes,
   5,000 train / 8,000 test) across 3 seeds comparing Student-only, Online KD, Online SP, Old CSD v4.1,
   SimHash-CSD (39.1 KB, 85x smaller than teacher!), and SimHash-CSD-Q (FIFO queue, 39.1 KB!).
"""

from __future__ import annotations

import argparse
import glob
import json
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import Config
from csd_core import (
    SimHashRelationalQueue,
    center_and_normalize,
    compute_pack_bytes,
    csd_jl_relational_loss,
    csd_projected_cosine_loss,
    dequantize_fingerprints,
    make_random_matrix,
    quantize_fingerprints,
)
from data import FastCIFAR100TensorLoader, augment_cifar_batch
from models import get_model
from utils import empty_cache, format_bytes, get_model_size_bytes, set_seed, sync_device


STL_MEAN = (0.4467, 0.4398, 0.4066)
STL_STD = (0.2603, 0.2566, 0.2713)


def _load_stl10_tensors(img_size: int = 64, root: str = "./data") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tr = torchvision.datasets.STL10(root=root, split="train", download=True)
    te = torchvision.datasets.STL10(root=root, split="test", download=True)
    # tr.data: (N, 3, 96, 96) uint8
    x_tr = torch.from_numpy(tr.data).float().div_(255.0)
    y_tr = torch.from_numpy(tr.labels).long()
    x_te = torch.from_numpy(te.data).float().div_(255.0)
    y_te = torch.from_numpy(te.labels).long()
    if img_size != 96:
        x_tr = F.interpolate(x_tr, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x_te = F.interpolate(x_te, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x_tr.contiguous(), y_tr, x_te.contiguous(), y_te


# ============================================================================
# 1. CRD-Style Linear Transfer Probing: CIFAR-100 -> STL-10
# ============================================================================

@torch.no_grad()
def _extract_backbone_feats(ckpt_path: str, arch: str, x: torch.Tensor, config: Config) -> np.ndarray:
    device = config.device
    model = get_model(arch, num_classes=100).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    mean = torch.tensor(config.cifar_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(config.cifar_std, device=device).view(1, 3, 1, 1)
    feats = []
    for i in range(0, len(x), 256):
        bx = (x[i : i + 256].to(device) - mean) / std
        _, l3 = model(bx, return_features=True)
        p = F.adaptive_avg_pool2d(l3, (1, 1)).flatten(1)
        feats.append(p.cpu().numpy())
    return np.concatenate(feats, axis=0)


def run_transfer_probe(config: Config) -> list[dict[str, Any]]:
    print("\n=== CRD-STYLE LINEAR TRANSFER PROBING: CIFAR-100 -> STL-10 (32x32) ===", flush=True)
    x_tr, y_tr, x_te, y_te = _load_stl10_tensors(img_size=32)
    y_tr_np, y_te_np = y_tr.numpy(), y_te.numpy()

    candidates = [
        ("baseline", "Student-only (CE)"),
        ("label_smooth", "Label Smoothing"),
        ("csd_v41_class", "Original CSD v4.1"),
        ("offline_sparse_kd", "Offline Sparse-12 Logit KD"),
        ("kd_online", "Online KD (Hinton)"),
        ("fitnet_online", "Online FitNet"),
        ("sp_online", "Online SP (Oracle)"),
        ("simhash_r64", "SimHash-CSD (1-bit, r=64)"),
        ("simhash_r128", "SimHash-CSD (1-bit, r=128)"),
        ("simhash_q2048", "SimHash-CSD-Q (1-bit, r=64, Q=2048)"),
        ("simhash_hybrid", "SimHash-CSD + Sparse-12 Logits"),
        ("simhash_dkd", "SimHash-CSD-Q + Offline DKD"),
    ]

    results = []
    for tag, label in candidates:
        ckpt = os.path.join(config.results_dir, f"{tag}_resnet20_s42_e200.pth")
        if not os.path.exists(ckpt):
            continue
        f_tr = _extract_backbone_feats(ckpt, "resnet20", x_tr, config)
        f_te = _extract_backbone_feats(ckpt, "resnet20", x_te, config)

        scaler = StandardScaler()
        f_tr_s = scaler.fit_transform(f_tr)
        f_te_s = scaler.transform(f_te)

        clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        clf.fit(f_tr_s, y_tr_np)
        acc = 100.0 * float((clf.predict(f_te_s) == y_te_np).mean())
        results.append({"tag": tag, "label": label, "stl10_linear_acc": acc})
        print(f"  {label:<38s} | STL-10 Linear Probe: {acc:.2f}%", flush=True)

    out_path = os.path.join(config.results_dir, "transfer_cifar100_to_stl10.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================================
# 2. End-to-End STL-10 (64x64) Second-Dataset Distillation Benchmark
# ============================================================================

@torch.no_grad()
def _eval_stl(model: nn.Module, x_te: torch.Tensor, y_te: torch.Tensor, device: torch.device) -> float:
    model.eval()
    mean = torch.tensor(STL_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STL_STD, device=device).view(1, 3, 1, 1)
    correct, total = 0, 0
    for i in range(0, len(x_te), 256):
        bx = (x_te[i : i + 256].to(device) - mean) / std
        by = y_te[i : i + 256].to(device)
        preds = model(bx).argmax(dim=1)
        correct += preds.eq(by).sum().item()
        total += by.size(0)
    return 100.0 * correct / total


def run_stl10_benchmark(config: Config, epochs: int = 150, seeds: list[int] = [42, 123, 456]) -> None:
    os.makedirs(config.results_dir, exist_ok=True)
    device = config.device
    x_tr, y_tr, x_te, y_te = _load_stl10_tensors(img_size=64)
    mean = torch.tensor(STL_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STL_STD, device=device).view(1, 3, 1, 1)

    # 1. Train ResNet-56 teacher on STL-10 (10 classes, 64x64)
    t_ckpt = os.path.join(config.results_dir, "stl10_teacher_resnet56.pth")
    t_hist_path = os.path.join(config.results_dir, "stl10_teacher_resnet56_history.json")
    if not os.path.exists(t_ckpt):
        set_seed(42)
        raw_t = get_model("resnet56", num_classes=10).to(device)
        t_model = torch.compile(raw_t) if config.use_compile and device.type in ("cuda", "mps") else raw_t
        opt = optim.SGD(raw_t.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        sched = CosineAnnealingLR(opt, T_max=epochs)
        tr_loader = FastCIFAR100TensorLoader(x_tr, y_tr, STL_MEAN, STL_STD, batch_size=64, shuffle=True, augment=True, device=device)
        best_acc, best_state = 0.0, None
        t0 = time.time()
        for ep in range(1, epochs + 1):
            t_model.train()
            for bx, by in tr_loader:
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=config.amp_dtype, enabled=config.use_amp and device.type in ("cuda", "mps")):
                    loss = F.cross_entropy(t_model(bx), by)
                loss.backward()
                opt.step()
            sched.step()
            acc = _eval_stl(raw_t, x_te, y_te, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in raw_t.state_dict().items()}
            if ep == 1 or ep % 50 == 0 or ep == epochs:
                print(f"[STL-10 Teacher ResNet-56] Ep {ep:3d}/{epochs} | Test {acc:.2f}% | Best {best_acc:.2f}% | {(time.time()-t0)/60:.1f}m", flush=True)
        assert best_state is not None
        torch.save(best_state, t_ckpt)
        json.dump({"best_acc": best_acc, "size_bytes": get_model_size_bytes(raw_t)}, open(t_hist_path, "w"), indent=2)
        del raw_t, t_model
        empty_cache(device)

    t_info = json.load(open(t_hist_path))
    print(f"[STL-10] Teacher ResNet-56: {t_info['best_acc']:.2f}% ({format_bytes(t_info['size_bytes'])})", flush=True)

    # 2. Extract 8-view teacher features once, then discard teacher
    feat_path = os.path.join(config.results_dir, "stl10_teacher_feats.pt")
    if not os.path.exists(feat_path):
        teacher = get_model("resnet56", num_classes=10).to(device)
        teacher.load_state_dict(torch.load(t_ckpt, map_location=device, weights_only=True))
        teacher.eval()
        N = len(x_tr)
        f3_all = torch.zeros(N, 64)
        logits_all = torch.zeros(N, 10)
        with torch.no_grad():
            for i in range(0, N, 128):
                bx = x_tr[i : i + 128].to(device)
                s_f3 = torch.zeros(bx.size(0), 64, device=device)
                s_lg = torch.zeros(bx.size(0), 10, device=device)
                for v in range(8):
                    xv = bx if v == 0 else augment_cifar_batch(bx, pad=8, flip_prob=0.5)
                    xv = (xv - mean) / std
                    lg, l3 = teacher(xv, return_features=True)
                    s_f3 += F.adaptive_avg_pool2d(l3.float(), (1, 1)).flatten(1)
                    s_lg += lg.float()
                f3_all[i : i + 128] = (s_f3 / 8.0).cpu()
                logits_all[i : i + 128] = (s_lg / 8.0).cpu()
        torch.save({"l3": f3_all, "logits": logits_all, "labels": y_tr}, feat_path)
        del teacher
        empty_cache(device)

    tdata = torch.load(feat_path, map_location="cpu", weights_only=False)
    f3_c = F.normalize(tdata["l3"] - tdata["l3"].mean(0, keepdim=True), dim=1)
    R64 = make_random_matrix(64, 64, seed=42, orthogonal=True)
    simhash_pack = quantize_fingerprints(f3_c @ R64, mode="1bit")
    simhash_bytes = compute_pack_bytes(simhash_pack) + 4
    simhash_signs = dequantize_fingerprints(simhash_pack, device=device)

    R128_orth = make_random_matrix(64, 128, seed=42, orthogonal=True)
    simhash128_pack = quantize_fingerprints(f3_c @ R128_orth, mode="1bit")
    simhash128_bytes = compute_pack_bytes(simhash128_pack) + 4
    simhash128_signs = dequantize_fingerprints(simhash128_pack, device=device)

    # Old CSD v4.1 per-class on STL-10
    f3_u = F.normalize(tdata["l3"], dim=1)
    R128 = make_random_matrix(64, 128, seed=42, orthogonal=False, device=device)
    phi_u = f3_u @ R128.cpu()
    v41_cf = torch.stack([phi_u[y_tr == c].mean(0) for c in range(10)]).to(device)
    v41_bytes = compute_pack_bytes(v41_cf.cpu()) + compute_pack_bytes(R128.cpu())

    methods = [
        ("baseline", "Student-only (ResNet-20)", 0),
        ("csd_v41_class", "Original CSD v4.1 (per-class)", v41_bytes),
        ("kd_online", "Online KD (Hinton)", t_info["size_bytes"]),
        ("sp_online", "Online SP (Oracle)", t_info["size_bytes"]),
        ("simhash_r64", "SimHash-CSD (1-bit, r=64)", simhash_bytes),
        ("simhash_r128", "SimHash-CSD (1-bit, r=128)", simhash128_bytes),
    ]

    summary = []
    for mkey, label, sbytes in methods:
        accs, sp_times = [], []
        for seed in seeds:
            out_j = os.path.join(config.results_dir, f"stl10_{mkey}_s{seed}_e{epochs}.json")
            if os.path.exists(out_j):
                h = json.load(open(out_j))
                accs.append(h["best_acc"])
                sp_times.append(h["sec_per_epoch"])
                print(f"[SKIP STL-10] {mkey} s={seed}: {h['best_acc']:.2f}%", flush=True)
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

            online_t = None
            if mkey in ("kd_online", "sp_online"):
                online_t = get_model("resnet56", num_classes=10).to(device)
                online_t.load_state_dict(torch.load(t_ckpt, map_location=device, weights_only=True))
                online_t.eval()
                for p in online_t.parameters():
                    p.requires_grad = False

            ema_s = torch.zeros(1, 64, device=device)
            ema_t = torch.zeros(1, 64, device=device)
            ema_init = False
            q = SimHashRelationalQueue(1024, 64, 64, device=device) if mkey == "simhash_q1024" else None

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

                    if mkey == "baseline":
                        loss = F.cross_entropy(logits_f, by)
                    elif mkey == "csd_v41_class":
                        ce = F.cross_entropy(logits_f, by)
                        phi_s = F.normalize(f3_f, dim=1) @ R128
                        d_loss, _ = csd_projected_cosine_loss(phi_s, v41_cf[by])
                        loss = ce + (0.3 * warm) * d_loss
                    elif mkey == "kd_online":
                        assert online_t is not None
                        with torch.no_grad():
                            with torch.autocast(device.type, dtype=config.amp_dtype, enabled=config.use_amp and device.type in ("cuda", "mps")):
                                t_lg = online_t(bx).float()
                        ce = F.cross_entropy(logits_f, by)
                        kd = F.kl_div(F.log_softmax(logits_f / 4.0, dim=1), F.softmax(t_lg / 4.0, dim=1), reduction="batchmean") * 16.0
                        loss = 0.1 * ce + 0.9 * kd
                    elif mkey == "sp_online":
                        assert online_t is not None
                        with torch.no_grad():
                            with torch.autocast(device.type, dtype=config.amp_dtype, enabled=config.use_amp and device.type in ("cuda", "mps")):
                                _, tl3 = online_t(bx, return_features=True)
                            tf3 = F.adaptive_avg_pool2d(tl3.float(), (1, 1)).flatten(1)
                            tm = tf3.mean(0, keepdim=True)
                            ema_t.copy_(tm) if (ep == 1 and bidx[0] >= 0 and ema_t.abs().sum() == 0) else ema_t.mul_(0.9).add_(tm, alpha=0.1)
                            tn = center_and_normalize(tf3, ema_t)
                        ce = F.cross_entropy(logits_f, by)
                        sn = center_and_normalize(f3_f, ema_s)
                        l_rel = csd_jl_relational_loss(sn, tn, is_1bit=False, tau=0.15)
                        loss = ce + (5.0 * warm) * l_rel
                    elif mkey in ("simhash_r64", "simhash_r128"):
                        ce = F.cross_entropy(logits_f, by)
                        sn = center_and_normalize(f3_f, ema_s)
                        signs_tbl = simhash_signs if mkey == "simhash_r64" else simhash128_signs
                        l_rel = csd_jl_relational_loss(sn, signs_tbl[bidx], is_1bit=True, tau=0.15)
                        loss = ce + (5.0 * warm) * l_rel
                    else:
                        raise ValueError(mkey)

                    loss.backward()
                    opt.step()
                sched.step()
                sync_device(device)
                ep_times.append(time.time() - t_ep)
                acc = _eval_stl(raw_s, x_te, y_te, device)
                if acc > best_acc:
                    best_acc = acc
            sec_ep = float(np.mean(ep_times[1:])) if len(ep_times) > 1 else ep_times[0]
            json.dump({"method": mkey, "seed": seed, "best_acc": best_acc, "sec_per_epoch": sec_ep, "storage_bytes": sbytes}, open(out_j, "w"), indent=2)
            print(f"[STL-10 {mkey} s={seed}] DONE | Best: {best_acc:.2f}% | {sec_ep:.2f}s/ep | Storage: {format_bytes(sbytes)}", flush=True)
            accs.append(best_acc)
            sp_times.append(sec_ep)
            del raw_s, s_model, online_t
            empty_cache(device)

        summary.append({
            "tag": mkey, "label": label,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "mean_sec_ep": float(np.mean(sp_times)),
            "storage_bytes": sbytes, "storage_str": format_bytes(sbytes),
            "per_seed": accs,
        })

    out_s = os.path.join(config.results_dir, f"summary_stl10_e{epochs}.json")
    json.dump(summary, open(out_s, "w"), indent=2)
    print("\n" + "=" * 88)
    print(f"STL-10 (64x64) BENCHMARK: ResNet-56 ({t_info['best_acc']:.2f}%) -> ResNet-20 ({epochs} ep, 3 seeds)")
    print("=" * 88)
    for r in summary:
        print(f"  {r['label']:<38s} | {r['mean_acc']:6.2f} +/- {r['std_acc']:4.2f}% | {r['storage_str']:>10s} | {r['mean_sec_ep']:5.2f}s/ep")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["transfer", "stl10"])
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    cfg = Config()
    if args.mode == "transfer":
        run_transfer_probe(cfg)
    else:
        run_stl10_benchmark(cfg, epochs=args.epochs)
