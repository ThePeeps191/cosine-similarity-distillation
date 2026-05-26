# CSD Versions — Development History

This document tracks the evolution of Cosine Similarity Distillation from initial
proof-of-concept through the published results.  Each version represents a specific
hypothesis tested, a problem diagnosed, and an improvement measured.

---

## v1 — Baseline CSD (initial implementation)

**What changed:** First implementation of the CSD concept.

**Fingerprints:** Single-layer (layer-3 only, 64-d).  Clean images only — no
augmentation applied during fingerprint generation (just ToTensor + Normalize).

**Loss:** MSE between student and teacher fingerprints.  No warmup.  No lambda scheduling.

**Training:** Lambda tuned at 100 epochs.  Single R matrix (64 x r).

**Results (200 epochs, no TTA):**

| Variant | Accuracy |
|---------|----------|
| CSD per-sample | 67.99% |
| CSD per-class | 68.57% |
| Baseline | 68.34% |
| KD | 70.46% |

**Diagnosis:** The student sees augmented images (RandomCrop + HorizontalFlip)
but must match fingerprints from clean originals.  The target is physically
impossible — the student is penalised for missing pixels it cannot observe.
FP Align never exceeded 0.60.

---

## v2 — Augmentation-aware + cosine loss + warmup

**What changed:** Three simultaneous fixes.

1. **Augmentation-aware fingerprints:** Fingerprints computed from the same
   RandomCrop + HorizontalFlip augmentations the student sees, averaged over
   4 augmented views per training image.  Targets are now achievable.

2. **Cosine similarity loss:** `1 - cos(phi_S, phi_T)` replaces MSE.  Focuses
   on directional agreement (what fingerprints encode) rather than magnitude
   matching.

3. **Lambda warmup:** lambda linearly ramps from 0 to target value over the
   first 40 epochs, letting classification stabilise before fingerprint
   alignment begins.

**Results (200 epochs, no TTA):**

| Variant | Accuracy | Change from v1 |
|---------|----------|---------------|
| CSD per-sample | 68.66% | +0.67% |
| CSD per-class | 68.88% | +0.31% |
| Baseline | 68.34% | — |

**Diagnosis:** Both variants now beat the baseline.  The augmentation mismatch
was confirmed as the root cause of v1's failure.  FP Align improved to ~0.60
by end of training (vs ~0.57 in v1).

---

## v3 — Multi-layer fingerprints + 8 augmentations

**What changed:**

1. **Multi-layer fingerprints:** Added a second random reference matrix R2
   (32 x r) for layer-2 features alongside the existing R3 (64 x r) for
   layer-3.  The student must now align with the teacher at two abstraction
   levels: texture/pattern (layer-2) and object/semantic (layer-3).

2. **8 augmentations:** Increased from 4 to 8 augmented views per image per
   fingerprint, providing more stable targets.

3. Equal weight on both layers (1.0 for l2, 1.0 for l3).

**Results (200 epochs, no TTA):**

| Variant | Accuracy | Change from v2 |
|---------|----------|---------------|
| CSD per-sample | 68.21% | -0.45% |
| CSD per-class | 68.95% | +0.07% |
| Baseline | 68.34% | — |

**Diagnosis:** Per-sample CSD dropped.  The equal weight on both layers
(1.0 + 1.0 = 2.0 total fingerprint weight) made the total fingerprint loss
too strong, overwhelming CE for per-sample matching.  The per-class variant
benefited from the extra layer because its class-averaged targets smoothed
out the additional noise.  Lambda ablation ran at 100 epochs, where the
40-epoch warmup consumed 40% of training — making lambda rankings unreliable
for the 200-epoch real training.

---

## v4 — Layer-specific lambda + 200-epoch tuning

**What changed:**

1. **Layer-specific lambda weights:** Layer-2 fingerprint loss weighted at 0.2,
   layer-3 at 1.0.  The texture signal (layer-2) carries less discriminative
   power than the semantic signal (layer-3) and should not dominate the loss.

2. **200-epoch lambda ablation:** Increased from 100 to 200 epochs so the 40-epoch
   warmup occupies only 20% of training (vs 40% in v3).  Lambda ranking now
   reflects 160 epochs of full-strength fingerprint matching.

3. **Separate per-class lambda tuning:** The per-class variant was discovered to
   need a different lambda than per-sample.  A dedicated per-class lambda sweep
   identified lambda = 0.3 as optimal (vs lambda = 0.01 for per-sample).

**Results (200 epochs, no TTA):**

| Variant | Accuracy | Change from v3 |
|---------|----------|---------------|
| CSD per-sample | 68.96% | +0.75% |
| CSD per-class | 68.80% | -0.15% |
| Baseline | 68.34% | — |

**Diagnosis:** Per-sample jumped significantly (lambda = 0.01 was found by the
improved 200-epoch ablation).  Per-class dropped slightly because it was still
trained with per-sample's lambda (0.01) — this was fixed in v4.1.

---

## v4.1 — Per-class with own lambda + TTA (final)

**What changed:**

1. **Per-class trained with its own best lambda (0.3)** from the dedicated
   per-class sweep, instead of reusing per-sample's lambda (0.01).

2. **Test-time augmentation (TTA):** 10-crop evaluation (5 crops x 2 flips)
   applied fairly to all methods (teacher, baseline, KD, FitNet, both CSD
   variants).  No model retraining — just a better evaluation strategy.

**Results (200 epochs, with TTA):**

| Method | Accuracy | Storage | Teacher Forwards |
|--------|----------|---------|-----------------|
| Teacher (ResNet-56) | 72.61% | 3.29 MB | — |
| Student-only (baseline) | 68.77% | — | 0 |
| KD (Hinton) | 70.57% | 3.29 MB | 78,200 |
| FitNet | 70.83% | 3.29 MB | 234,600 |
| **CSD per-sample** | **69.17%** | **24.41 MB** | **1** |
| **CSD per-class** | **69.65%** | **50 KB** | **1** |

**Key metrics:**
- CSD per-class recovers 48.9% of KD's accuracy gain (+0.88% over baseline
  vs KD's +1.80%)
- 67x storage reduction (50 KB vs 3.29 MB teacher)
- 78,200x fewer teacher forwards (1 vs 78,200)
- FP Align reaches 0.89 during training

---

## v5 — Contrastive fingerprints (negative result, not used)

**What changed:** Added a hard-negative contrastive push term to the per-class
fingerprint loss.  In addition to pulling toward the correct class fingerprint,
the student is penalised for being too similar to the hardest-negative class
fingerprint (the wrong class it is currently most confused about).

**Formula:**
```
pull  = 1 - cos(phi_S, phi_T_correct)
push  = ReLU(hardest_wrong_cos_sim - 0.3)
loss  = CE + lambda * (pull + 0.3 * push)
```

**Result:** Accuracy dropped to 69.04% (from 69.65% in v4.1).  The push gradient
conflicted with the pull gradient when the hardest-negative class was semantically
similar to the correct class.  At 64-dimensional features projected onto 128
random directions, class separation in fingerprint space is too subtle for a
margin-based push to help.

**Status:** Code retained in the repository for reproducibility with
`config.use_contrastive = False` as the default.

---

## Summary table

| Version | Key changes | Per-sample | Per-class | vs Baseline |
|---------|------------|-----------|-----------|-------------|
| v1 | Clean fingerprints, MSE, single-layer | 67.99% | 68.57% | +0.23pp |
| v2 | Augmentation-aware, cosine loss, warmup | 68.66% | 68.88% | +0.54pp |
| v3 | Multi-layer (R2+R3), 8 augs, equal weights | 68.21% | 68.95% | +0.61pp |
| v4 | Layer-specific lambda (0.2:1.0), 200-epoch tuning | 68.96% | 68.80% | +0.62pp |
| **v4.1** | **Per-class own lambda + TTA** | **69.17%** | **69.65%** | **+0.88pp** |
| v5 | Contrastive push (negative result) | 69.17% | 69.04% | +0.27pp |
