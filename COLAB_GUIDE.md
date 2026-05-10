# Google Colab Guide

Run the full CSD pipeline on a free Google Colab T4 GPU in **4 staged sessions**.
Total GPU time: ~8 hours. Free Colab limits sessions to ~4-6 hours, so checkpoints
are saved to Google Drive between sessions for guaranteed resumption.

**Drive storage needed:** Make sure you have at least **100 MB** free in your Google Drive (Google Drive gives 15 GB free).

Alternatively, if you would like to test the CSD demo on your local machine, follow the **Quick Start** section in [README.md](README.md).

---

## Table of Contents

- [Session overview](#session-overview)
- [Session 1 — Teacher + Fingerprints (~80 min)](#session-1--teacher--fingerprints-80-min)
- [Session 2 — Baseline + KD + Lambda tuning (~180 min)](#session-2--baseline--kd--lambda-tuning-180-min)
- [Session 3 — FitNet (~150 min)](#session-3--fitnet-150-min)
- [Session 4 — CSD + r-ablation + Evaluate + Plots (~135 min)](#session-4--csd--r-ablation--evaluate--plots-135-min)
- [Resuming after a timeout](#resuming-after-a-timeout)
- [Colab Pro alternative](#colab-pro-alternative)
- [Warning signs](#warning-signs)

---

## Session overview

| Session | Cells | What runs | GPU time |
|---------|-------|-----------|----------|
| 1 | Cell 0, 0b, 1a | Teacher training + Fingerprint generation | ~80 min |
| 2 | Cell 0, 0b, 1b–1e | Student baseline + KD + Lambda ablation | ~180 min |
| 3 | Cell 0, 0b, 1f | FitNet (3 betas × 200 epochs) | ~150 min |
| 4 | Cell 0, 0b, 1g–1k | CSD per-sample + per-class + r-ablation + Evaluate + Plots + Download | ~135 min |

Each session is a **new notebook**: open a fresh one, set runtime to T4 GPU, then paste and run the cells.

> **Note:** Ablation sweeps (lambda tuning, r-ablation) use `config.epochs_ablation = 100` (not 200) to keep session times manageable. 100 epochs includes one LR drop at epoch 60, which is sufficient for hyperparameter comparison.

---

## Boilerplate (same for every session)

These two cells are used at the start of **every session**. Paste them first.

**Cell 0 — Mount Google Drive and set up save path:**

```python
from google.colab import drive
drive.mount('/content/drive')
SAVE = '/content/drive/MyDrive/csd_results'
import os, shutil
os.makedirs(SAVE, exist_ok=True)
```

**Cell 0b — Clone the repo and install dependencies:**

```python
!git clone https://github.com/ThePeeps191/cosine-similarity-distillation.git
%cd cosine-similarity-distillation
!pip install -r requirements.txt
os.makedirs('results', exist_ok=True)
```

After these two cells, follow the session-specific cells below.

---

## Session 1 — Teacher + Fingerprints (~80 min)

**Cell 0** and **Cell 0b** from boilerplate above.

**Cell 1a — Train teacher and generate fingerprints:**

```python
from config import Config
from utils import set_seed
from train_teacher import train_teacher
from generate_fingerprints import generate_fingerprints

config = Config()
set_seed(config.seed)

if not os.path.exists(config.teacher_ckpt):
    train_teacher(config)

if not os.path.exists(config.all_fingerprints_path):
    generate_fingerprints(config)

for f in os.listdir('results'):
    shutil.copy(os.path.join('results', f), os.path.join(SAVE, f))
print("Session 1 complete — saved to Google Drive")
```

**What happens:** Trains a ResNet-56 teacher for 200 epochs on CIFAR-100, then forwards all
50k training images through the teacher to compute cosine-similarity fingerprints
against a frozen random matrix R. Output saved to Google Drive.

**Expected output in Drive:** `best_teacher.pth`, `random_matrix_R.pt`,
`all_fingerprints.pt`, `class_fingerprints.pt`, `teacher_history.json`

---

## Session 2 — Baseline + KD + Lambda tuning (~180 min)

**Cell 0** and **Cell 0b** from boilerplate.

**Restore cell — copy saved files back from Drive:**

```python
if os.path.exists(SAVE):
    for f in os.listdir(SAVE):
        shutil.copy(os.path.join(SAVE, f), os.path.join('results', f))
```

**Cell 1b — Student-only baseline (~30 min):**

```python
from config import Config
from train_student_baseline import train_student_baseline

config = Config()

if not os.path.exists(config.baseline_ckpt):
    train_student_baseline(config)
```

**Cell 1c — KD (Hinton) baseline (~50 min):**

```python
from train_student_kd import train_student_kd

if not os.path.exists(config.kd_ckpt):
    train_student_kd(config)
```

**Cell 1d — Lambda ablation for CSD (~90 min):**

```python
from ablation import run_lambda_ablation

if not os.path.exists('results/ablation_lambda.json'):
    run_lambda_ablation(config)
```

**Cell 1e — Save results to Drive:**

```python
for f in os.listdir('results'):
    shutil.copy(os.path.join('results', f), os.path.join(SAVE, f))
print("Session 2 complete")
```

**Expected output adds in Drive:** `best_student_baseline.pth`, `best_student_kd.pth`,
`baseline_history.json`, `kd_history.json`, `ablation_lambda.json`

---

## Session 3 — FitNet (~150 min)

**Cell 0** and **Cell 0b** from boilerplate.

**Restore cell:**

```python
if os.path.exists(SAVE):
    for f in os.listdir(SAVE):
        shutil.copy(os.path.join(SAVE, f), os.path.join('results', f))
```

**Cell 1f — FitNet baseline (3 betas × 200 epochs):**

```python
from config import Config
from train_student_fitnet import train_student_fitnet

config = Config()

if not os.path.exists(config.fitnet_ckpt):
    train_student_fitnet(config)

for f in os.listdir('results'):
    shutil.copy(os.path.join('results', f), os.path.join(SAVE, f))
print("Session 3 complete")
```

**What happens:** Sweeps beta ∈ {10, 50, 100} for FitNet feature-matching
distillation. Saves the checkpoint from the beta with the highest test accuracy.

**Expected output adds in Drive:** `best_student_fitnet.pth`, `fitnet_history.json`

---

## Session 4 — CSD + r-ablation + Evaluate + Plots + Download (~135 min)

**Cell 0** and **Cell 0b** from boilerplate.

**Restore cell:**

```python
if os.path.exists(SAVE):
    for f in os.listdir(SAVE):
        shutil.copy(os.path.join(SAVE, f), os.path.join('results', f))
```

**Cell 1g — Find best lambda from ablation:**

```python
from config import Config
import json

config = Config()

best_lam = 0.1
if os.path.exists('results/ablation_lambda.json'):
    with open('results/ablation_lambda.json') as f:
        d = json.load(f)
    if d:
        best_lam = max(d, key=lambda x: x['accuracy'])['lambda']
print(f'Best lambda from ablation: {best_lam}')
```

**Cell 1h — CSD per-sample (~30 min):**

```python
from train_student_csd import train_student_csd

if not os.path.exists(config.csd_sample_ckpt):
    train_student_csd(config, use_per_class=False, lambda_csd=best_lam)
```

**Cell 1i — CSD per-class (~30 min):**

```python
if not os.path.exists(config.csd_class_ckpt):
    train_student_csd(config, use_per_class=True, lambda_csd=best_lam)
```

**Cell 1j — Fingerprint dimension r-ablation (~60 min):**

```python
from ablation import run_fingerprint_ablation
from generate_fingerprints import generate_fingerprints

if not os.path.exists('results/ablation_r.json'):
    run_fingerprint_ablation(config)
    generate_fingerprints(config)  # Restore default r=128 fingerprints
```

**Cell 1k — Evaluate all models, generate all plots, download zip:**

```python
import os, json
from evaluate import collect_all_results
from data import get_cifar100_loaders
from plot import (
    plot_teacher_training, plot_accuracy_comparison,
    plot_accuracy_barplot, plot_accuracy_improvement,
    plot_storage_comparison, plot_training_loss_comparison,
    plot_tsne, plot_fingerprint_ablation,
    plot_lambda_sensitivity, plot_fingerprint_alignment,
)

_, test_loader = get_cifar100_loaders(config)
results = collect_all_results(config, test_loader=test_loader)

if not results:
    print("No results — teacher checkpoint missing. Did Session 1 complete?")
else:
    hist = json.load(open(config.teacher_history_path))
    plot_teacher_training(hist, config)

    acc = results["accuracies"]
    plot_accuracy_barplot(acc, config)
    plot_accuracy_improvement(acc, config)

    hist_dict = {}
    hist_files = {
        "Student-only": "baseline_history.json",
        "KD": "kd_history.json",
        "FitNet": "fitnet_history.json",
        "CSD per-sample": "csd_per_sample_history.json",
        "CSD per-class": "csd_per_class_history.json",
    }
    for label, fname in hist_files.items():
        path = os.path.join('results', fname)
        if os.path.exists(path):
            hist_dict[label] = json.load(open(path))["test_acc"]
    if hist_dict:
        plot_accuracy_comparison(hist_dict, config)

    loss_hist_dict = {}
    for label, fname in hist_files.items():
        path = os.path.join('results', fname)
        if os.path.exists(path):
            loss_hist_dict[label] = json.load(open(path))["train_loss"]
    if loss_hist_dict:
        plot_training_loss_comparison(loss_hist_dict, config)

    fp_align_dict = {}
    for label_variant, fname in [("CSD per-sample", "csd_per_sample_history.json"),
                                  ("CSD per-class", "csd_per_class_history.json")]:
        path = os.path.join('results', fname)
        if os.path.exists(path):
            data = json.load(open(path))
            if "fp_alignment" in data:
                fp_align_dict[label_variant] = data["fp_alignment"]
    if fp_align_dict:
        plot_fingerprint_alignment(fp_align_dict, config)

    plot_storage_comparison(config)

    if os.path.exists(config.csd_sample_ckpt):
        plot_tsne(config, config.baseline_ckpt, config.csd_sample_ckpt, test_loader)

    if os.path.exists('results/ablation_r.json'):
        plot_fingerprint_ablation(json.load(open('results/ablation_r.json')), config)
    if os.path.exists('results/ablation_lambda.json'):
        plot_lambda_sensitivity(json.load(open('results/ablation_lambda.json')), config)

    print("All plots generated")

for f in os.listdir('results'):
    shutil.copy(os.path.join('results', f), os.path.join(SAVE, f))

!zip -r /content/csd_results.zip results/
from google.colab import files
files.download('/content/csd_results.zip')
print("Session 4 complete — results downloaded")
```

**What you get:**
- `results_summary.json` — the main results table with all accuracies
- `teacher_training.png` — teacher convergence curves
- `accuracy_comparison.png` — test accuracy vs epoch for all student methods
- `loss_comparison.png` — training loss vs epoch for all student methods
- `accuracy_barplot.png` — final accuracy bar chart for paper
- `accuracy_improvement.png` — accuracy gain over baseline by method
- `fp_alignment.png` — student-teacher fingerprint cosine similarity during CSD training
- `storage_comparison.png` — storage comparison (log scale)
- `tsne_features.png` — t-SNE of baseline vs CSD features
- `ablation_r.png` — accuracy vs fingerprint dimension
- `ablation_lambda.png` — accuracy vs CSD loss weight
- `csd_results.zip` — all files bundled for download

---

## Resuming after a timeout

Every training phase checks `os.path.exists()` before running. If Colab
disconnects mid-session:

1. Open a new notebook, set runtime to T4 GPU.
2. Paste Cell 0 and Cell 0b (clone, install, mount Drive).
3. Paste the restore cell (`for f in os.listdir(SAVE): ...`).
4. Re-run the session-specific cells.
5. Completed phases are skipped automatically. Only the current in-progress
   epoch is lost (~10-30 seconds of work).

---

## Colab Pro alternative

For $10/month, Colab Pro grants ~24 hours of continuous GPU (V100 priority).
Combine all cells from all 4 sessions into a single notebook and run overnight.

**Single notebook structure:**
```
Cell 0   → Drive mount + clone + install
Cell A   → Teacher + fingerprints (Phase 1-2)
Cell B   → Student baseline + KD + FitNet (Phase 3)
Cell C   → Lambda ablation + CSD per-sample + per-class (Phase 4)
Cell D   → r-ablation + regenerate r=128 (Phase 5)
Cell E   → Evaluate + all plots + zip + download (Phase 6-7)
```

---

## Results

| Method | Top-1 Accuracy | Storage for Transfer | Teacher Forwards |
|--------|---------------|---------------------|-----------------|
| Teacher (ResNet-56) | XX.X% | N/A | — |
| Student-only (no distillation) | XX.X%  | — | 0 |
| KD (Hinton) | XX.X%  | 3.29 MB (teacher) | 78,200 (Every batch) |
| FitNet (feature MSE) | XX.X%  | 3.29 MB (teacher) | 234,600 (Every batch) |
| CSD (per-sample) | XX.X%  | 24.41 MB (fingerprints) | 1 |
| CSD (per-class) | XX.X%  | 50 KB (fingerprints) | 1 |

---

## Warning signs

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError: best_teacher.pth` | Session 1 didn't finish or Drive files not restored | Re-run Session 1; check with `!ls $SAVE` |
| Low student accuracy | Model has not converged or checkpoint on wrong device | Verify `map_location=device` in `torch.load()` |
| `OutOfMemoryError` | Peak VRAM exceeds T4 16 GB | Runtime → Factory reset runtime, retry |
| Training unusually slow | Background Colab processes or degraded T4 | Factory reset runtime and retry |
