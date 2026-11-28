# SimHash-CSD: Teacher-Free Relational Knowledge Distillation via 1-Bit Orthogonal Hyperplane Sketches

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/template.pdf)
[![Tests](https://img.shields.io/badge/pytest-35%20passed-brightgreen.svg)](tests/)

**SimHash-CSD** eliminates the teacher model completely from the student training loop by compressing centered teacher intermediate representations into compact **1-bit QR-orthogonal SimHash sketches** (just **8 bytes/image** at `r=64`) in a single offline pass. During student training, mini-batch teacher pairwise cosine similarities are reconstructed directly from these binary sketches via **Charikar's angular estimator** and aligned with the student's centered cosine Gram matrix — requiring **zero online teacher forwards**, **zero teacher GPU memory**, **zero extra student parameters**, and **zero teacher logits**.

---

## Key Highlights (3-Seed Mean ± Std)

| Benchmark | Student-only | Online KD (Full Teacher) | Online SP (Oracle) | **SimHash-CSD (Ours, 1-bit)** | Storage Compression | Speedup |
|---|---|---|---|---|---|---|
| **CIFAR-100** (`ResNet-56 -> ResNet-20`, 200 ep) | 69.19 ± 0.57% | 70.49 ± 0.04% | 70.46 ± 0.14% | **70.45 ± 0.16%** *(TTA 70.53%)* | **390.6 KB** (**8.6×** smaller) | **1.82–1.93×** |
| **Cross-Arch** (`ResNet-56 [C=64] -> VGG-8 [C=512]`, 100 ep) | 70.95 ± 0.13% | 74.18 ± 0.30% | — | **73.29 ± 0.08%** *(+2.34pp, beats teacher!)* | **781.3 KB** (**4.3×** smaller) | **1.70×** |
| **Large-Teacher Gap** (`ResNet-32x4 [28.4 MB] -> ResNet-20`, 100 ep) | 68.38 ± 0.53% | 68.37 ± 0.18% *(collapsed)* | — | **69.05 ± 0.06%** *(+0.68pp vs KD, p=0.015)* | **781.3 KB** (**37.2×** smaller; `r=64`: **74.5×**) | **3.58×** |
| **STL-10 64×64** (`ResNet-56 [81.30%] -> ResNet-20`, 150 ep) | 81.05 ± 0.20% | 82.48 ± 0.29% | 82.63 ± 0.30% | **82.43 ± 0.38%** *(beats teacher!)* | **39.1 KB** (**85.5×** smaller) | **2.27×** |
| **Transfer Probe** (`CIFAR-100 -> STL-10` linear probe) | 62.15% | 63.01% | 63.15% | **62.92%** *(+2.96pp vs Sparse-12 Logits!)* | **390.6 KB** | — |

### Core Findings
- **Matches the Full-Precision Online Relational Oracle (`Online SP`) within `0.01pp`** (`70.45%` vs `70.46%`) using only **1 bit per hyperplane** (`390.6 KB` total vs `3.29 MB` live teacher).
- **Bit-Budget Theorem Proven Empirically:** At an iso-storage budget (~390–488 KB), **1-bit SimHash (`r=64`, `390.6 KB`) outperforms INT8 continuous JL sketches (`r=8`, `488.3 KB`) by `+1.53pp`** (`69.62%` vs `68.09%`) with **2.30× lower pairwise cosine MAE** (`0.1165` vs `0.2678`), because 1 bit across 64 orthogonal hyperplanes spans 100% of the teacher subspace instead of truncating 87.5% of it.
- **QR-Orthogonal Variance Reduction:** QR-orthogonal random hyperplanes cut 1-bit pairwise cosine MAE by **22.5%–23.0%** vs. i.i.d. Gaussian hyperplanes across 10 seeds, with **26% lower error on semantically similar positive pairs**.
- **100% Architecture-Agnostic & Logit-Free:** Requires zero learnable projectors (`0` extra params) and handles arbitrary channel mismatches (`C=64 -> C=512`, `C=256 -> C=64`, `C=768 -> C=64`).
- **Honest Quantitative Privacy Audit:** Both closed-form Ridge and non-linear 2-layer MLP feature-inversion attackers are benchmarked across 18 `(r, quantization)` configurations, proving naive overcomplete continuous projections are 100.0% linearly invertible while mapping the exact privacy–storage–accuracy trade-off of 1-bit sketches.

---

## How SimHash-CSD Works

```python
# Phase 1: One-Time Offline Sketching (Teacher permanently discarded afterward!)
f_T_centered = F.normalize(f_T - f_T.mean(dim=0, keepdim=True), p=2, dim=1)
R            = make_random_matrix(C_T, r=64, seed=42, orthogonal=True)   # 4-byte seed!
sketches     = packbits(f_T_centered @ R >= 0)                            # 8 bytes/image (390.6 KB on CIFAR-100)

# Phase 2: Teacher-Free Student Training (0 teacher forwards, 0 extra params, any C_S != C_T!)
f_S_centered = F.normalize(f_S - ema_mean_S, p=2, dim=1)
G_S          = f_S_centered @ f_S_centered.T                              # (B, B) student cosine Gram
G_T_hat      = torch.sin((math.pi / 2.0) * (B_signs @ B_signs.T) / r)     # Charikar exact angular estimator
L_rel        = tau * KL(softmax(G_T_hat_off / tau) || softmax(G_S_off / tau)) + MSE(G_S_off, G_T_hat_off)
loss         = CE(logits, labels) + lambda_t * L_rel
```

---

## Quick Start (Apple Silicon MPS / CUDA / CPU)

```bash
git clone https://github.com/ThePeeps191/cosine-similarity-distillation.git
cd cosine-similarity-distillation

uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# 1. Verify all 35 unit tests & module imports (CPU, ~3s)
pytest tests/ -v
python import_checks.py

# 2. Verify exact byte-level storage calculations (no GPU needed)
python storage_calcs.py

# 3. Train ResNet-56 teacher + run quantitative Linear & MLP privacy audit
python experiments.py --mode teacher --teacher resnet56 --epochs 200 --seed 42
python experiments.py --mode privacy --teacher resnet56

# 4. Run full resumable multi-seed benchmarks (auto-selects CUDA > Apple MPS > CPU)
python run_benchmark.py --stage ablations --epochs 100
python run_benchmark.py --stage main --epochs 200 --seeds 42 123 456
python run_benchmark.py --stage nextgen --epochs 200 --seeds 42 123 456
python run_benchmark.py --stage cross_arch --epochs 100 --seeds 42 123 456
python run_benchmark.py --stage large_teacher --epochs 100 --seeds 42 123 456

# 5. Second dataset (STL-10 64x64), transfer probing, ViT-B/16 & hardware profiler
python stl10_bench.py --mode stl10 --epochs 150
python stl10_bench.py --mode transfer
python vit_and_ensemble_bench.py --mode all

# 6. Regenerate all 6 publication-grade 300-dpi figures directly from results/*.json
python generate_all_figures.py

# 7. Compile the LaTeX paper to PDF (requires tectonic or pdflatex + bibtex)
tectonic paper/template.tex
```

---

## Repository Structure

```
cosine-similarity-distillation/
├── csd_core.py                  # Core math: QR-orthogonal R, 1-bit SimHash, Charikar Gram, losses, Linear+MLP privacy audit
├── models.py                    # CIFARResNet (20/56/110/8x4/32x4), VGG-8-BN, MobileNetV2-CIFAR
├── data.py                      # Standard DataLoaders + zero-overhead vectorized GPU CIFAR-100 loader
├── config.py                    # Unified Config with CUDA > Apple MPS > CPU auto-detection & AMP/compile
├── utils.py                     # Multi-backend seeding, cache cleanup, exact byte-level accounting
├── experiments.py               # Unified trainer for all online/offline baselines & SimHash-CSD variants
├── run_benchmark.py             # Resumable multi-stage orchestrator (ablations, main, cross-arch, large-teacher)
├── stl10_bench.py               # STL-10 (64x64) second-dataset benchmark & CIFAR-100 -> STL-10 transfer probe
├── vit_and_ensemble_bench.py    # Headless ViT-B/16 diagnostic, multi-teacher ensembles & MPS hardware profiler
├── plot_csd_pp.py               # Publication plotting primitives
├── generate_all_figures.py      # Generates all 6 paper figures directly from results/*.json
├── storage_calcs.py             # Exact byte-accurate storage audit across all methods
├── main.py                      # Turnkey end-to-end SimHash-CSD entry point
├── paper/
│   ├── template.pdf             # Compiled 12-page preprint PDF
│   ├── template.tex             # Full LaTeX paper (Propositions 1-2, Theorem 1, 6 figures, 5 tables)
│   ├── references.bib           # Complete bibliography (19 references)
│   └── fig_*.png                # 6 generated 300-dpi publication figures
└── tests/                       # 35 unit tests covering math, bit-packing, Charikar, queues, DKD, privacy audit
```

## License

MIT License — see [LICENSE](LICENSE) for details.
