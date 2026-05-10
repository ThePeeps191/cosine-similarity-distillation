"""Import verification for the CSD pipeline.

Checks that every module in the project imports cleanly without runtime errors.

    python import_checks.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

print("=" * 55)
print("CSD Import Verification")
print("=" * 55)

failed = 0
passed = 0

modules = [
    ("config.py",        "from config import Config; c = Config()"),
    ("utils.py",          "from utils import set_seed, count_parameters, get_model_size_mb, get_tensor_size; set_seed(42)"),
    ("models.py",         "from models import resnet20, resnet56; m20 = resnet20(); m56 = resnet56()"),
    ("data.py",           "from data import get_cifar100_loaders, get_indexed_cifar100_loaders, IndexedCIFAR100"),
    ("storage_calcs.py",  "import storage_calcs"),
    ("train_teacher.py",  "from train_teacher import train_teacher"),
    ("generate_fingerprints.py", "from generate_fingerprints import generate_fingerprints"),
    ("train_student_baseline.py", "from train_student_baseline import train_student_baseline"),
    ("train_student_kd.py",       "from train_student_kd import train_student_kd"),
    ("train_student_fitnet.py",   "from train_student_fitnet import train_student_fitnet"),
    ("train_student_csd.py",      "from train_student_csd import train_student_csd"),
    ("ablation.py",      "from ablation import run_fingerprint_ablation, run_lambda_ablation"),
    ("evaluate.py",      "from evaluate import collect_all_results"),
    ("main.py",          "import main"),
]

for name, code in modules:
    try:
        exec(code)
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}  ->  {e}")
        failed += 1

print("-" * 55)
print(f"  {passed} passed, {failed} failed, {len(modules)} total")
print("=" * 55)

if failed:
    print("\nFix the failures above.")
    sys.exit(1)
else:
    print("\nAll imports pass.")
