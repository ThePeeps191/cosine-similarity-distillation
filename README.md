# Cosine Similarity Distillation (CSD)

A novel teacher-free distillation method that replaces live teacher inference with compact precomputed random projection teacher fingerprints, enabling storage-efficient, high-performance knowledge transfer and enhanced student training.

[![arXiv](https://img.shields.io/badge/arXiv-2505.XXXXX-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Cosine Similarity Distillation, abbreviated as **CSD**, is a method of teacher-student model distillation that completely removes the need for live forwarding of the teacher model during student training. Instead of loading the teacher at every batch, CSD precomputes compact "fingerprints", which are the cosine similarities between the teacher's normalized intermediate feature vectors and the columns of a frozen random reference matrix. The student then regresses these fingerprints alongside the classification loss, enabling distillation with **48.9%** of KD's accuracy gain and **67x smaller** storage on the CIFAR-100 vision model dataset.

<!-- Accuracy calculated as (69.65 - 68.77) / (70.57 - 68.77) x 100 = 48.9% -->

This repository houses a demo of the CSD method with CIFAR-100 image classification through PyTorch. Read the CSD paper at <https://arxiv.org>.

**Key contributions:**
- First method to use a frozen random matrix as a shared coordinate system for distillation
- Extreme storage efficiency: CIFAR-100 per-class fingerprints as small as **50 KB** (67x smaller than the teacher)
- Teacher-free training: teacher forwarded once, and never loaded again (vs. every batch for KD/FitNet)
- Privacy benefit: random projections are information-theoretically hard to invert

## Quick Start

```bash
git clone https://github.com/ThePeeps191/cosine-similarity-distillation.git
cd cosine-similarity-distillation

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Run tests (no GPU needed, ~45s)
pytest tests/ -v

# Verify storage numbers (no GPU needed)
python storage_calcs.py

# Verify all imports (no GPU needed)
python import_checks.py

# Run full pipeline (needs GPU, ~8 hours on T4)
python main.py
```

## Results

| Method | Top-1 Accuracy | Storage for Transfer | Teacher Forwards |
|--------|---------------|---------------------|-----------------|
| Teacher (ResNet-56) | 72.61% | -- | -- |
| Student-only (no distillation) | 68.77%  | -- | 0 |
| KD (Hinton) | 70.57%  | 3.29 MB (teacher) | 78,200 (Every batch) |
| FitNet (feature MSE) | 70.83%  | 3.29 MB (teacher) | 234,600 (Every batch) |
| CSD (per-sample) | 69.17%  | 24.41 MB (fingerprints) | 1 |
| CSD (per-class) | **69.65%**  | **50 KB** (fingerprints) | **1** |

All accuracies include test-time augmentation (10-crop, TTA).

## How CSD Works

```
R       = randn(64, r), columns L2-normalized              (frozen, no gradients)
phi_T   = mean_{aug views}(normalize(pool(layer3(aug))) @ R)  (precomputed once)
phi_S   = normalize(pool(layer3(x_aug)), dim=1) @ R           (computed per batch)
Loss    = CE(logits, labels) + lambda * (1 - cos_sim(phi_S, phi_T))
```

The teacher generates fingerprints from the same RandomCrop + RandomHorizontalFlip
augmentations the student sees, averaged over 8 views per image.  This ensures
the student matches a fingerprint target it can physically achieve.  A cosine
similarity loss focuses on directional agreement (what fingerprints encode), and
a lambda warmup (0 for first 40 epochs) lets classification stabilize before
fingerprint alignment begins.  Multi-layer fingerprints (layer-2 + layer-3) capture
complementary teacher geometry at texture and object levels.

## Pipeline Phases

| Phase | Step | Output |
|-------|------|--------|
| 1 | Train teacher (ResNet-56, 200 epochs) | `best_teacher.pth` |
| 2 | Generate augmentation-averaged fingerprints (multi-layer, r=128, 8 views) | fingerprint `.pt` files + random matrices |
| 3 | Train baselines (student-only, KD, FitNet) | `best_student_*.pth` |
| 4 | Lambda tuning + CSD per-sample + per-class | `best_student_csd_*.pth` |
| 5 | Ablation: r {16,32,64,128,256} and lambda {0.01-1.0} | `ablation_*.json` |
| 6 | Evaluate all models, generate 10 plots | `*.png`, `results_summary.json` |
| 7 | Zip results | `csd_results.zip` |

Every phase is **idempotent** -- safe to interrupt and resume. Checkpoints are only computed if missing.

## Architecture

- **Teacher**: ResNet-56, ~862k params, ~3.29 MB
- **Student**: ResNet-20, ~278k params, ~1.06 MB
- **Models**: Custom `CIFARResNet` (NOT torchvision's ImageNet ResNet) with conv1 3x3 stride 1, 16 filters, **no MaxPool**, three stages: [16, 32, 64] channels
- **Feature extraction**: After layer3, before pooling -> (B, 64, 8, 8) -> global avg pool -> 64-d vector
- **Random matrix R**: (64 x r), columns L2-normalized, frozen forever after generation
- **Multi-layer fingerprints**: R2 (32 x r) for layer-2 (texture) + R3 (64 x r) for layer-3 (semantic)
- **Augmentation-averaged fingerprints**: 8 augmented views per image, averaged to produce a single target

## Key improvements

- **Augmentation-aware fingerprints**: teacher fingerprints use same RandomCrop + RandomHorizontalFlip as student training, averaged over 8 views per image
- **Cosine similarity loss**: `1 - cos_sim(phi_S, phi_T)` instead of MSE, for direction-focused alignment
- **Lambda warmup**: linear ramp from 0 over first 40 epochs, letting classification stabilize before fingerprint matching
- **Multi-layer fingerprints**: fingerprints computed from both layer-2 (texture/shape) and layer-3 (object identity), combined with layer-specific weights (0.2 for l2, 1.0 for l3)
- **Per-class lambda tuning**: separate lambda sweep for per-class variant, run at 200 epochs

## Google Colab / Kaggle

Free Colab sessions timeout after ~4-6 hours. The full pipeline runs across multiple sessions with checkpoint persistence. See [COLAB_GUIDE.md](COLAB_GUIDE.md) for detailed cell-by-cell instructions.

## Testing

```bash
pytest tests/ -v
```

26 tests, no GPU needed. CIFAR-100 (~160 MB) is auto-downloaded on first run.

## Project Structure

```
cosine-similarity-distillation/
├── config.py                  # All hyperparameters in one Config class
├── models.py                  # CIFARResNet, BasicBlock, resnet20(), resnet56()
├── data.py                    # CIFAR-100 loaders + IndexedCIFAR100 dataset
├── utils.py                   # set_seed(), count_parameters(), get_tensor_size()
├── train_teacher.py           # Phase 1: teacher training
├── generate_fingerprints.py   # Phase 2: fingerprint precomputation
├── train_student_baseline.py  # Phase 3a: student-only baseline (CE)
├── train_student_kd.py        # Phase 3b: Hinton KD baseline
├── train_student_fitnet.py    # Phase 3c: FitNet (feature MSE) baseline
├── train_student_csd.py       # Phase 4: core CSD training
├── evaluate.py                # Phase 6: results collection + table
├── ablation.py                # Phase 5: r and lambda ablation studies
├── plot.py                    # Phase 6: 10 plots generated
├── storage_calcs.py           # Verifiable storage/compute numbers (no GPU needed)
├── import_checks.py           # Pre-push import verification for all modules
├── COLAB_GUIDE.md             # Complete Google Colab setup and session guide
├── main.py                    # End-to-end orchestration (Phases 1-7)
├── tests/                     # 26 unit tests (no GPU needed)
│   ├── __init__.py
│   ├── test_config.py         # Config hyperparameter validation
│   ├── test_utils.py          # Seeding, param count, tensor size formatting
│   ├── test_models.py         # ResNet-20/56 forward shapes, param counts, gradients
│   ├── test_data.py           # CIFAR-100 loader shapes, indexed dataset
│   └── test_core.py           # CSD math: fingerprints, cosine similarity, KD loss
├── .gitignore                 # Ignores venv/, results/, data/, *.zip, .pytest_cache/
├── requirements.txt
├── LICENSE                    # MIT License
└── README.md
```

## Notes

- **num_workers=0**: Windows multiprocessing compatibility. Data loading is single-process.
- **No MaxPool**: Standard CIFAR ResNet convention -- a 3x3 stride-1 conv in conv1.
- **Multi-layer fingerprints**: 8 views averaged per image, layer-2 (R2, 32-d) + layer-3 (R3, 64-d).
- **Cosine loss**: `1 - cos_sim(phi_S, phi_T)` replaces MSE for direction-focused alignment.
- **Lambda warmup**: Linear ramp from 0 over 40 epochs (config.warmup_epochs=40).
- **Lambda weights**: Layer-2 fingerprint loss weighted at 0.2, layer-3 at 1.0.
- **Augmentations**: Train = RandomCrop(32, pad=4) + RandomHorizontalFlip + Normalize. Test = Normalize only.
- **Learning rate**: MultiStepLR(milestones=[60, 120, 160], gamma=0.2) with initial lr=0.1.
- **PyTorch 2.6+**: Compatible. The removed `manual_seed_all` is not used.
- **Privacy**: Gaussian random projections make teacher reconstruction infeasible.

## References

- Hinton et al., 2015 -- *Distilling the Knowledge in a Neural Network*
- Romero et al., 2015 -- *FitNets: Hints for Thin Deep Nets*
- He et al., 2016 -- *Deep Residual Learning for Image Recognition*
- Mannix et al., 2024 -- *CosPress: Cosine Similarity Preserving Distillation*
- Zhou et al., 2024 -- *TCS: Teacher-free Coordinate System Distillation*
- Ghojogh et al., 2021 -- *Johnson-Lindenstrauss Lemma for Dimensionality Reduction*

## License

MIT License -- see [LICENSE](LICENSE) for details.
