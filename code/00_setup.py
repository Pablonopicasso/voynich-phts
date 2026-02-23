#!/usr/bin/env python3
"""
PHTS Theory v3.0 — Replication Package
Script 00: Environment Setup and Dependency Check
================================================
Mamontov, P. (2026). Procedural Humoral-Tabular System Theory v3.0. Cryptologia.

Run this first to verify your environment before executing any other scripts.
"""

import sys
import os

REQUIRED = {
    "numpy": "1.24",
    "pandas": "2.0",
    "scikit-learn": "1.3",
    "scipy": "1.11",
    "matplotlib": "3.7",
}

def check_python():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}", end="")
    if v.major >= 3 and v.minor >= 9:
        print(" ✓")
        return True
    else:
        print(" ✗ (requires ≥ 3.9)")
        return False

def check_packages():
    ok = True
    for pkg, min_ver in REQUIRED.items():
        try:
            import importlib
            mod = importlib.import_module(pkg.replace("-", "_").replace("scikit_learn", "sklearn"))
            ver = getattr(mod, "__version__", "unknown")
            print(f"  {pkg} {ver} ✓")
        except ImportError:
            print(f"  {pkg} ✗ NOT FOUND — run: pip install {pkg}")
            ok = False
    return ok

def check_data_files():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    required_files = [
        "eva_corpus_schema.md",
        "operator_definitions.csv",
        "section_mappings.csv",
        "glyph_positional_data.csv",
    ]
    ok = True
    for f in required_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            print(f"  {f} ✓")
        else:
            print(f"  {f} ✗ MISSING")
            ok = False
    # EVA corpus (must be downloaded separately)
    eva_path = os.path.join(data_dir, "eva_raw.txt")
    if os.path.exists(eva_path):
        size = os.path.getsize(eva_path)
        print(f"  eva_raw.txt ({size//1024} KB) ✓")
    else:
        print("  eva_raw.txt ✗ NOT FOUND")
        print("    → Download from: https://voynich.nu/transcr.html")
        print("    → Save as: data/eva_raw.txt")
        ok = False
    return ok

def main():
    print("=" * 60)
    print("PHTS Theory v3.0 — Replication Package Setup Check")
    print("=" * 60)
    
    print("\n[1] Python version:")
    py_ok = check_python()
    
    print("\n[2] Required packages:")
    pkg_ok = check_packages()
    
    print("\n[3] Data files:")
    data_ok = check_data_files()
    
    print("\n" + "=" * 60)
    if py_ok and pkg_ok and data_ok:
        print("✓ ALL CHECKS PASSED — Ready to run replication scripts")
        print("\nRecommended order:")
        print("  python 01_preprocess_eva.py")
        print("  python 02_glyph_classification.py")
        print("  python 03_grammar_coverage.py")
        print("  python 04_operator_identification.py")
        print("  python 05_entropy_analysis.py")
        print("  python 06_ml_classifier.py")
        print("  python 07_kill_shot_test.py   ← THE KILL-SHOT")
        print("  python 08_falsification_battery.py")
        print("  python 09_full_validation_battery.py")
    else:
        print("✗ SETUP INCOMPLETE — Fix issues above before proceeding")
        if not data_ok:
            print("\nMost likely fix: Download EVA corpus from voynich.nu")
    print("=" * 60)

if __name__ == "__main__":
    main()
