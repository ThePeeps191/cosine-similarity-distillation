"""Pre-push import verification for all SimHash-CSD modules.

Run from repo root:
    python import_checks.py
"""

from __future__ import annotations

import importlib
import sys

MODULES = [
    "config",
    "utils",
    "models",
    "data",
    "csd_core",
    "experiments",
    "run_benchmark",
    "stl10_bench",
    "vit_and_ensemble_bench",
    "plot_csd_pp",
    "generate_all_figures",
    "main",
    "storage_calcs",
]


def main() -> None:
    failed = []
    for mod in MODULES:
        try:
            importlib.import_module(mod)
            print(f"  [OK]   {mod}")
        except Exception as e:
            print(f"  [FAIL] {mod}: {e}")
            failed.append((mod, e))

    if failed:
        print(f"\n{len(failed)} module(s) failed to import.")
        sys.exit(1)
    print(f"\nAll {len(MODULES)} modules imported cleanly.")


if __name__ == "__main__":
    main()
