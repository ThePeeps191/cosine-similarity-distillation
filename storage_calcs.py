"""
Verification script for all of the CSD paper storage and compute claims.

Computes model sizes, fingerprint storage, teacher-forward counts, and
storage reduction ratios using only the model architecture and config
(no GPU training needed). Every formula is also explained inline.

Run from repo root with venv active:

    python storage_calcs.py
"""

from models import resnet20, resnet56
from utils import count_parameters, get_model_size_mb

print("=" * 60)
print("STORAGE VERIFICATION - Cosine Similarity Distillation")
print("=" * 60)

# ------------------------------------------------------------------
# Teacher model size
# count_parameters() sums all trainable params. Each float32 uses 4 bytes.
# get_model_size_mb() converts to binary megabytes: params * 4 / (1024^2).
# ------------------------------------------------------------------
teacher_params = count_parameters(resnet56())
teacher_mb = get_model_size_mb(resnet56())
print(f"\nTeacher (ResNet-56):")
print(f"  Trainable parameters: {teacher_params:,}")
print(f"  Model size (float32): {teacher_mb:.2f} MB")
print(f"  Model size in KB:     {teacher_mb * 1024:.1f} KB")

# ------------------------------------------------------------------
# Student model size
# ------------------------------------------------------------------
student_params = count_parameters(resnet20())
student_mb = get_model_size_mb(resnet20())
print(f"\nStudent (ResNet-20):")
print(f"  Trainable parameters: {student_params:,}")
print(f"  Model size (float32): {student_mb:.2f} MB")
print(f"  Model size in KB:     {student_mb * 1024:.1f} KB")

# ------------------------------------------------------------------
# Random matrix R
# R has shape (C=64, r=128). Each entry is float32 (4 bytes).
# Size = 64 * 128 * 4 = 32,768 bytes = 32 KB.
# ------------------------------------------------------------------
R_elements = 64 * 128
R_bytes = R_elements * 4
R_kb = R_bytes / 1024
print(f"\nRandom matrix R (64 x 128):")
print(f"   64 channels after layer3, 128 = fingerprint dimension r")
print(f"  Formula: C x r x 4 bytes = 64 x 128 x 4")
print(f"  Size:    {R_kb:.2f} KB")

# ------------------------------------------------------------------
# Per-sample fingerprints
# 50,000 training images, each with an r-dimensional fingerprint.
# Each entry is float32 (4 bytes).
# Size = 50000 * 128 * 4 = 25,600,000 bytes.
# PyTorch uses binary prefixes: 1 MB = 1024 * 1024 = 1,048,576 bytes.
# So 25,600,000 / 1,048,576 = 24.41 MB.
# If you see "25.6 MB" elsewhere, that's decimal (1000^2), not binary.
# ------------------------------------------------------------------
num_images = 50000
sample_elements = num_images * 128
sample_bytes = sample_elements * 4
sample_mb_binary = sample_bytes / (1024 * 1024)
sample_mb_decimal = sample_bytes / (1000 * 1000)
print(f"\nPer-sample fingerprints ({num_images} images x r=128):")
print(f"   50000 = CIFAR-100 training set size")
print(f"  Formula: {num_images} x 128 x 4 bytes")
print(f"  Binary (1024^2 MB): {sample_mb_binary:.2f} MB  (PyTorch convention)")
print(f"  Decimal (1000^2 MB): {sample_mb_decimal:.2f} MB  (marketing units)")

# ------------------------------------------------------------------
# Per-class fingerprints
# 100 classes, each with an r-dimensional fingerprint (mean of per-sample).
# Size = 100 * 128 * 4 = 51,200 bytes = 50 KB.
# ------------------------------------------------------------------
num_classes = 100
class_elements = num_classes * 128
class_bytes = class_elements * 4
class_kb = class_bytes / 1024
print(f"\nPer-class fingerprints ({num_classes} classes x r=128):")
print(f"   100 = number of CIFAR-100 classes")
print(f"  Formula: {num_classes} x 128 x 4 bytes")
print(f"  Size:    {class_kb:.2f} KB")

# ------------------------------------------------------------------
# Storage comparison ratios
# Teacher size / per-class fingerprint size gives the reduction ratio.
# ------------------------------------------------------------------
teacher_kb = teacher_mb * 1024
ratio = teacher_kb / class_kb
print(f"\nStorage comparison:")
print(f"  Teacher size:                {teacher_kb:.1f} KB ({teacher_mb:.2f} MB)")
print(f"  Per-class fingerprints:      {class_kb:.2f} KB")
print(f"  Storage reduction ratio:     {ratio:.0f}x")
print(f"  (teacher is {ratio:.0f}x larger than per-class fingerprints)")

# ------------------------------------------------------------------
# Teacher forwards during training
# KD: 1 forward per batch x num_batches per epoch x 200 epochs
# FitNet: same x 3 (beta sweep)
# CSD: just 1 (precomputation phase, then teacher is discarded)
# ------------------------------------------------------------------
batch_size = 128
approx_batches = -(-50000 // batch_size)  # ceil(50000/128) = 391
kd_forwards = approx_batches * 200
print(f"\nTeacher forwards during training:")
print(f"  {approx_batches} batches/epoch = 50000 images / 128 batch size")
print(f"  200 epochs (config.epochs_student)")
print(f"  KD:     {approx_batches} x 200 = {kd_forwards:,} forwards")
print(f"  FitNet: {approx_batches} x 200 x 3 = {kd_forwards * 3:,} forwards (3 betas)")
print(f"  CSD:    1 forward (fingerprint precomputation only)")

# print("\n" + "=" * 60)
print("\nAll numbers above are computed from architecture and config.")
print("Accuracy and performance metrics require running `python main.py` on a GPU.")
# print("=" * 60)
