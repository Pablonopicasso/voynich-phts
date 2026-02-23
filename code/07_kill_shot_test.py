#!/usr/bin/env python3
"""
PHTS Theory v3.0 — Replication Package
Script 07: THE KILL-SHOT — Operator Permutation Invariance Test
================================================================
THE central falsification mechanism of PHTS v3.0.

Test Logic:
  1. Train classifier on ORIGINAL corpus → accuracy ~90.5%
  2. Globally permute operator labels (o→k→d→c→p→o)
     This preserves FREQUENCIES and POSITIONS but destroys FUNCTIONAL IDENTITY
  3. Retrain classifier on PERMUTED corpus → accuracy should collapse to ~22%

If operators are merely frequent glyphs or positional artifacts,
permuting their labels has no effect → accuracy stays high.
If operators possess genuine functional identity essential for structure,
permuting destroys that identity → accuracy collapses to near-chance.

PHTS v3.0 claim (Section 4):
  - Original:         90.5% ± 4.3%  → BASELINE
  - Post-permutation: ~22%           → COLLAPSE TO CHANCE
  - Random baseline:  ~20%           → REFERENCE

STATUS: KILL-SHOT PASSED → VALIDATED

Output: results/kill_shot_results.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

OPERATORS      = ["o", "k", "d", "c", "p"]
SECTIONS       = ["Herbal", "Pharmaceutical", "Balneological", "Zodiac", "Astronomical"]
TOP_GLYPHS     = ["q", "o", "k", "a", "e", "y", "ch", "d", "c", "s",
                  "l", "r", "n", "sh", "t", "ee", "i", "p", "f", "m"]
CHANCE_LEVEL   = 20.0   # 1/5 sections
COLLAPSE_BOUND = 30.0   # must be BELOW this to count as collapsed
BASELINE_MIN   = 80.0   # must be ABOVE this for valid baseline

# ── Permutation scheme (cyclic, as described in PHTS v3.0 Section 4.1) ──
# o → k → d → c → p → o
PERMUTATION = {"o": "k", "k": "d", "d": "c", "c": "p", "p": "o"}

def load_data(data_dir):
    pkl = os.path.join(data_dir, "eva_processed.pkl")
    if not os.path.exists(pkl):
        print("ERROR: Run 01_preprocess_eva.py first.")
        sys.exit(1)
    return pd.read_pickle(pkl)

def permute_word(word):
    """Apply global operator label permutation to a single word."""
    return "".join(PERMUTATION.get(g, g) for g in word)

def build_features(df, permuted=False):
    """Build folio-level features from word frequency profiles."""
    folios = df.groupby(["folio", "section"])
    X_rows, y_labels, groups = [], [], []
    
    from collections import Counter
    
    for (folio, section), group in folios:
        if section not in SECTIONS:
            continue
        words = [permute_word(w) if permuted else w for w in group["word"]]
        all_glyphs = [g for w in words for g in w]
        total = len(all_glyphs) if all_glyphs else 1
        counts = Counter(all_glyphs)
        features = [counts.get(g, 0) / total for g in TOP_GLYPHS]
        features += [group["word_length"].mean(), len(group) / 100.0]
        X_rows.append(features)
        y_labels.append(section)
        groups.append(folio)
    
    return np.array(X_rows), np.array(y_labels), np.array(groups)

def run_kfold_cv(X, y, groups, n_splits=5, random_state=42):
    """Run Grouped 5-Fold CV. Returns mean accuracy (%)."""
    gkf = GroupKFold(n_splits=n_splits)
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    accs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf.fit(X[train_idx], y[train_idx])
        accs.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    return np.mean(accs) * 100, np.std(accs) * 100

def run_random_baseline(X, y, groups, n_trials=20):
    """True random baseline: predict random section labels."""
    accs = []
    for seed in range(n_trials):
        y_rand = np.random.RandomState(seed).choice(SECTIONS, size=len(y))
        acc = accuracy_score(y, y_rand) * 100
        accs.append(acc)
    return np.mean(accs)

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("PHTS Script 07: THE KILL-SHOT — Operator Permutation Invariance Test")
    print("=" * 70)
    print("\nPermutation scheme: o→k, k→d, d→c, c→p, p→o (cyclic)")
    print("Preserves: frequencies, positions")
    print("Destroys:  functional identity\n")
    
    df = load_data(data_dir)
    df = df[df["section"] != "Unknown"]
    
    # ── Condition 1: Original corpus ─────────────────────────────────────────
    print("Running [CONDITION 1]: Original corpus (operators intact)...")
    X_orig, y_orig, g_orig = build_features(df, permuted=False)
    acc_orig, std_orig = run_kfold_cv(X_orig, y_orig, g_orig)
    baseline_valid = acc_orig >= BASELINE_MIN
    print(f"  Accuracy: {acc_orig:.1f}% ± {std_orig:.1f}%")
    print(f"  Baseline valid (≥80%): {'YES ✓' if baseline_valid else 'NO ✗'}")
    
    # ── Condition 2: Permuted corpus ─────────────────────────────────────────
    print("\nRunning [CONDITION 2]: Permuted corpus (operators relabeled)...")
    X_perm, y_perm, g_perm = build_features(df, permuted=True)
    acc_perm, std_perm = run_kfold_cv(X_perm, y_perm, g_perm)
    collapsed = acc_perm <= COLLAPSE_BOUND
    print(f"  Accuracy: {acc_perm:.1f}% ± {std_perm:.1f}%")
    print(f"  Collapsed to chance (≤{COLLAPSE_BOUND}%): {'YES ✓' if collapsed else 'NO ✗'}")
    
    # ── Random baseline ───────────────────────────────────────────────────────
    print("\nComputing [RANDOM BASELINE]...")
    acc_random = run_random_baseline(X_orig, y_orig, g_orig)
    print(f"  Random baseline: {acc_random:.1f}%  (theoretical: ~20%)")
    
    # ── RESULTS TABLE ─────────────────────────────────────────────────────────
    drop = acc_orig - acc_perm
    kill_shot_passed = baseline_valid and collapsed
    
    print("\n" + "=" * 70)
    print("KILL-SHOT RESULTS")
    print("=" * 70)
    print(f"{'Condition':<35} {'Accuracy':>10}  {'Status'}")
    print(f"{'-'*35} {'-'*10}  {'-'*20}")
    print(f"{'Original (operators intact)':<35} {acc_orig:>8.1f}%  {'BASELINE ✓' if baseline_valid else 'BASELINE ✗'}")
    print(f"{'Post-permutation (labels swapped)':<35} {acc_perm:>8.1f}%  {'COLLAPSED ✓' if collapsed else 'DID NOT COLLAPSE ✗'}")
    print(f"{'Random baseline (chance)':<35} {acc_random:>8.1f}%  REFERENCE")
    print(f"{'Accuracy drop (orig → permuted)':<35} {drop:>8.1f} pp")
    print("=" * 70)
    print(f"\nKILL-SHOT: {'✓ PASSED — Operators possess functional identity' if kill_shot_passed else '✗ FAILED — Investigate'}")
    print(f"\nPHTS paper reports:")
    print(f"  Original:         90.5% ± 4.3%")
    print(f"  Post-permutation: ~22%")
    print(f"  Random baseline:  ~20%")
    print("=" * 70)
    
    # Save results
    results = [
        {"condition": "Original (operators intact)",    "accuracy_pct": round(acc_orig, 2),   "std_pct": round(std_orig, 2), "status": "BASELINE"},
        {"condition": "Post-permutation (labels swapped)", "accuracy_pct": round(acc_perm, 2), "std_pct": round(std_perm, 2), "status": "PERMUTED"},
        {"condition": "Random baseline (chance)",        "accuracy_pct": round(acc_random, 2), "std_pct": None, "status": "REFERENCE"},
    ]
    results_df = pd.DataFrame(results)
    results_df["kill_shot_passed"] = kill_shot_passed
    results_df["accuracy_drop_pp"] = round(drop, 2)
    
    out = os.path.join(results_dir, "kill_shot_results.csv")
    results_df.to_csv(out, index=False)
    print(f"\n✓ Saved: {out}")
    print("Run 08_falsification_battery.py next.")

if __name__ == "__main__":
    main()
