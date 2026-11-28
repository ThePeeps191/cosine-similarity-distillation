"""Exact, byte-accurate storage and compute verification for SimHash-CSD.

Computes exact parameter counts, transfer artifact byte sizes (including ALL
tensors, quantization scales, and 4-byte PRNG seeds — zero hidden overhead),
and online teacher-forward counts. No GPU required.

Run:
    python storage_calcs.py
"""

from __future__ import annotations

import torch
from csd_core import build_csd_artifacts
from models import resnet20, resnet32x4, resnet56, vgg8_cifar
from utils import count_parameters, format_bytes, get_model_size_bytes


def main() -> None:
    print("=" * 88)
    print("EXACT BYTE-LEVEL STORAGE & COMPUTE AUDIT — SimHash-CSD")
    print("=" * 88)

    t_r56 = resnet56(num_classes=100)
    t_r32x4 = resnet32x4(num_classes=100)
    s_r20 = resnet20(num_classes=100)
    s_vgg8 = vgg8_cifar(num_classes=100)

    r56_bytes = get_model_size_bytes(t_r56)
    r32x4_bytes = get_model_size_bytes(t_r32x4)
    print("\nModels (float32, CIFAR-100):")
    print(f"  Teacher  ResNet-56   : {count_parameters(t_r56):>10,} params | {format_bytes(r56_bytes)}")
    print(f"  Teacher  ResNet-32x4 : {count_parameters(t_r32x4):>10,} params | {format_bytes(r32x4_bytes)}")
    print(f"  Student  ResNet-20   : {count_parameters(s_r20):>10,} params | {format_bytes(get_model_size_bytes(s_r20))}")
    print(f"  Student  VGG-8-BN    : {count_parameters(s_vgg8):>10,} params | {format_bytes(get_model_size_bytes(s_vgg8))}")

    # Synthetic CIFAR-100 teacher tensors (N=50,000, K=100, C2=32, C3=64)
    N, K, C2, C3 = 50000, 100, 32, 64
    dummy_c100 = {
        "l2": torch.zeros(N, C2),
        "l3": torch.zeros(N, C3),
        "logits": torch.zeros(N, K),
        "labels": torch.arange(N) % K,
        "classifier_weight": torch.zeros(K, C3),
    }

    art_1bit_32 = build_csd_artifacts(dummy_c100, num_classes=K, r=32, quant="1bit", top_m=12)
    art_1bit_64 = build_csd_artifacts(dummy_c100, num_classes=K, r=64, quant="1bit", top_m=12)
    art_1bit_128 = build_csd_artifacts(dummy_c100, num_classes=K, r=128, quant="1bit", top_m=12)
    art_int8_8 = build_csd_artifacts(dummy_c100, num_classes=K, r=8, quant="int8", top_m=12)
    art_int8_32 = build_csd_artifacts(dummy_c100, num_classes=K, r=32, quant="int8", top_m=12)

    direct_rp_class_multi = (K * 128 * 4) * 2 + (C3 + C2) * 128 * 4
    direct_rp_sample_l3 = N * 128 * 4 + C3 * 128 * 4
    offline_dense_logits = N * K * 4

    rows = [
        ("Online KD / SP / FitNet (ResNet-56)", r56_bytes, "78,200 (every batch)"),
        ("Online KD (ResNet-32x4)", r32x4_bytes, "78,200 (every batch)"),
        ("Offline Dense Logits (50k x 100 fp32)", offline_dense_logits, "0 (offline only)"),
        ("Direct-RP Per-Sample (r=128 fp32 + R3)", direct_rp_sample_l3, "0 (offline only)"),
        ("Direct-RP Per-Class Baseline (L2+L3 + R2/R3)", direct_rp_class_multi, "0 (offline only)"),
        ("Offline Sparse-12 Logit KD (int8)", art_1bit_64["storage_bytes"]["offline_sparse_logits"], "0 (offline only)"),
        ("Relational INT8 (r=32, 0 extra params)", art_int8_32["storage_bytes"]["csd_rel_l3"], "0 (offline only)"),
        ("Iso-Storage Relational INT8 (r=8)", art_int8_8["storage_bytes"]["csd_rel_l3"], "0 (offline only)"),
        ("SimHash-CSD (r=32, 1-bit, high privacy)", art_1bit_32["storage_bytes"]["csd_rel_l3"], "0 (offline only)"),
        ("SimHash-CSD (r=64, 1-bit, ours — primary)", art_1bit_64["storage_bytes"]["csd_rel_l3"], "0 (offline only)"),
        ("SimHash-CSD (r=128, 1-bit, ours)", art_1bit_128["storage_bytes"]["csd_rel_l3"], "0 (offline only)"),
        ("SimHash-CSD + Sparse-12 Logits (r=64)", art_1bit_64["storage_bytes"]["csd_rel_l3"] + art_1bit_64["storage_bytes"]["offline_sparse_logits"], "0 (offline only)"),
    ]

    print(f"\n{'Method (CIFAR-100, N=50,000)':<46s} {'Storage':>11s} {'vs R56':>13s} {'vs R32x4':>13s}   {'Train T-Fwds'}")
    print("-" * 108)
    for name, nbytes, fwds in rows:
        r56_ratio = r56_bytes / max(1, nbytes)
        r32_ratio = r32x4_bytes / max(1, nbytes)
        s_r56 = f"{r56_ratio:5.1f}x smaller" if r56_ratio >= 1.0 else f"{1/r56_ratio:5.1f}x LARGER"
        s_r32 = f"{r32_ratio:5.1f}x smaller" if r32_ratio >= 1.0 else f"{1/r32_ratio:5.1f}x LARGER"
        print(f"{name:<46s} {format_bytes(nbytes):>11s} {s_r56:>13s} {s_r32:>13s}   {fwds}")

    # STL-10 (N=5,000, K=10) exact check
    t_r56_stl = resnet56(num_classes=10)
    r56_stl_bytes = get_model_size_bytes(t_r56_stl)
    stl_1bit_64 = 5000 * 8 + 4  # 8 bytes/image + 4-byte seed
    print(f"\nSTL-10 (N=5,000): ResNet-56 = {format_bytes(r56_stl_bytes)} vs SimHash-CSD (r=64) = "
          f"{format_bytes(stl_1bit_64)} ({r56_stl_bytes / stl_1bit_64:.1f}x smaller)")


if __name__ == "__main__":
    main()
