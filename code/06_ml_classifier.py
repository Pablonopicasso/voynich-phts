#!/usr/bin/env python3
"""
PHTS Theory v3.0 — Replication Package
Script 06: Machine Learning Section Classifier
===============================================
Trains a classifier to predict VM section from structural features
(glyph frequency profiles). Uses:
  - Grouped 5-Fold Cross-Validation (grouped by folio)
  - Leave-One-Section-Out (LOSO) validation
  - Operator ablation (removing operators reduces accuracy by ~31pp)

PHTS v3.0 claims (Section 5.5):
  - ML accuracy (Grouped K-Fold): 90.5% ± 4.3%  (threshold: >80%)
  - LOSO accuracy:                68.1%           (threshold: >60%)
  - Chance level:                 ~20% (5 sections)
  - Operator ablation drop:       ~31 pp

Output: results/ml_classification_report.txt, results/tier1_validation_results.csv (partial)
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

OPERATORS = {"o", "k", "d", "c", "p"}
SECTIONS  = ["Herbal", "Pharmaceutical", "Balneological", "Zodiac", "Astronomical"]
TOP_GLYPHS = ["q", "o", "k", "a", "e", "y", "ch", "d", "c", "s",
               "l", "r", "n", "sh", "t", "ee", "i", "p", "f", "m"]

def load_data(data_dir):
    pkl = os.path.join(data_dir, "eva_processed.pkl")
    if not os.path.exists(pkl):
        print("ERROR: Run 01_preprocess_eva.py first.")
        sys.exit(1)
    return pd.read_pickle(pkl)

def build_folio_features(df, include_operators=True):
    """
    Aggregate word-level data to folio-level feature vectors.
    Features: glyph frequency proportions per folio.
    """
    glyphs_to_use = TOP_GLYPHS if include_operators else [g for g in TOP_GLYPHS if g not in OPERATORS]
    
    folios = df.groupby(["folio", "section"])
    X_rows, y_labels, groups = [], [], []
    
    for (folio, section), group in folios:
        if section not in SECTIONS:
            continue
        # Count glyph frequencies
        all_glyphs = [g for word in group["word"] for g in word]
        total = len(all_glyphs) if len(all_glyphs) > 0 else 1
        counts = Counter(all_glyphs)
        
        features = [counts.get(g, 0) / total for g in glyphs_to_use]
        # Additional structural features
        mean_word_len = group["word_length"].mean()
        n_words = len(group)
        op_density = sum(counts.get(g, 0) for g in OPERATORS) / total
        
        features += [mean_word_len, n_words / 100.0, op_density]
        
        X_rows.append(features)
        y_labels.append(section)
        groups.append(folio)
    
    return np.array(X_rows), np.array(y_labels), np.array(groups)

def grouped_kfold_cv(X, y, groups, n_splits=5):
    """Grouped K-Fold CV (grouped by folio, not word)."""
    gkf = GroupKFold(n_splits=n_splits)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    accs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf.fit(X[train_idx], y[train_idx])
        acc = accuracy_score(y[test_idx], clf.predict(X[test_idx]))
        accs.append(acc)
    
    return np.mean(accs) * 100, np.std(accs) * 100

def loso_validation(X, y, groups):
    """Leave-One-Section-Out: train on 4 sections, test on 1."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    section_groups = le.transform(y)
    
    logo = LeaveOneGroupOut()
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    accs = []
    for train_idx, test_idx in logo.split(X, y, section_groups):
        if len(test_idx) == 0:
            continue
        clf.fit(X[train_idx], y[train_idx])
        acc = accuracy_score(y[test_idx], clf.predict(X[test_idx]))
        accs.append(acc)
    
    return np.mean(accs) * 100

def shuffle_sanity(X, y, groups):
    """Shuffle labels → should collapse to chance (~20%)."""
    y_shuffled = y.copy()
    np.random.seed(0)
    np.random.shuffle(y_shuffled)
    mean_acc, _ = grouped_kfold_cv(X, y_shuffled, groups)
    return mean_acc

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print("PHTS Script 06: Machine Learning Section Classifier")
    print("=" * 55)
    print("Chance level (5 sections): ~20%")
    
    df = load_data(data_dir)
    df = df[df["section"] != "Unknown"]
    
    # --- WITH OPERATORS ---
    print("\n[1] Building folio feature vectors (with operators)...")
    X, y, groups = build_folio_features(df, include_operators=True)
    print(f"    {X.shape[0]} folios, {X.shape[1]} features")
    
    print("\n[2] Grouped 5-Fold CV...")
    mean_acc, std_acc = grouped_kfold_cv(X, y, groups)
    kfold_pass = mean_acc >= 80.0
    print(f"    Accuracy: {mean_acc:.1f}% ± {std_acc:.1f}%")
    print(f"    Threshold: >80%")
    print(f"    STATUS: {'✓ PASS' if kfold_pass else '✗ FAIL'}")
    
    print("\n[3] LOSO Validation...")
    loso_acc = loso_validation(X, y, groups)
    loso_pass = loso_acc >= 60.0
    print(f"    Accuracy: {loso_acc:.1f}%")
    print(f"    Threshold: >60%")
    print(f"    STATUS: {'✓ PASS' if loso_pass else '✗ FAIL'}")
    
    print("\n[4] Shuffle Sanity Check...")
    shuffle_acc = shuffle_sanity(X, y, groups)
    shuffle_pass = shuffle_acc <= 30.0  # should be near chance
    print(f"    Accuracy: {shuffle_acc:.1f}%  (should be ~20%)")
    print(f"    STATUS: {'✓ PASS (collapsed to chance)' if shuffle_pass else '✗ FAIL'}")
    
    # --- WITHOUT OPERATORS (ablation) ---
    print("\n[5] Operator Ablation (remove operators, measure drop)...")
    X_no_op, y_no_op, g_no_op = build_folio_features(df, include_operators=False)
    mean_acc_no_op, _ = grouped_kfold_cv(X_no_op, y_no_op, g_no_op)
    drop = mean_acc - mean_acc_no_op
    ablation_pass = drop >= 20.0  # PHTS claims ~31pp drop
    print(f"    Accuracy without operators: {mean_acc_no_op:.1f}%")
    print(f"    Drop: {drop:.1f} percentage points")
    print(f"    PHTS reports: ~31 pp drop")
    print(f"    STATUS: {'✓ PASS' if ablation_pass else '✗ FAIL'}")
    
    # --- FULL CLASSIFICATION REPORT ---
    clf_final = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    gkf = GroupKFold(n_splits=5)
    all_y_true, all_y_pred = [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf_final.fit(X[train_idx], y[train_idx])
        all_y_true.extend(y[test_idx])
        all_y_pred.extend(clf_final.predict(X[test_idx]))
    
    report = classification_report(all_y_true, all_y_pred, target_names=SECTIONS)
    
    report_path = os.path.join(results_dir, "ml_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("PHTS Theory v3.0 — ML Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Grouped 5-Fold CV Accuracy: {mean_acc:.1f}% ± {std_acc:.1f}%\n")
        f.write(f"LOSO Accuracy:              {loso_acc:.1f}%\n")
        f.write(f"Shuffle (sanity) Accuracy:  {shuffle_acc:.1f}%\n")
        f.write(f"Operator Ablation Drop:     {drop:.1f} pp\n\n")
        f.write("Per-Section Classification Report:\n")
        f.write(report)
    print(f"\n✓ Saved: {report_path}")
    
    print("\n── FINAL SUMMARY ──")
    print(f"CV Accuracy:      {mean_acc:.1f}% ✓/✗ (>80%)")
    print(f"LOSO Accuracy:    {loso_acc:.1f}% ✓/✗ (>60%)")
    print(f"Ablation Drop:    {drop:.1f} pp ✓/✗ (>20 pp)")
    print(f"Shuffle Sanity:   {shuffle_acc:.1f}% ✓/✗ (~20%)")
    print("\nRun 07_kill_shot_test.py next — THE KILL-SHOT")

if __name__ == "__main__":
    main()
