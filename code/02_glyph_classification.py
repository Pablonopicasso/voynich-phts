#!/usr/bin/env python3
"""
PHTS Theory v3.0 — Replication Package
Script 02: Glyph Classification (F/C/S Roles)
==============================================
Classifies every glyph in the EVA corpus by its positional role:
  F = Frame  (word-initial bias ≥ 60%)
  C = Core   (medial bias)
  S = Specifier (word-final bias ≥ 60%)
  OP = Operator (invariance std < 5.0; tested in script 04)

References PHTS v3.0, Section 3.2.
Output: data/glyph_classification_results.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

def load_processed(data_dir):
    pkl = os.path.join(data_dir, "eva_processed.pkl")
    if not os.path.exists(pkl):
        print("ERROR: eva_processed.pkl not found. Run 01_preprocess_eva.py first.")
        sys.exit(1)
    return pd.read_pickle(pkl)

def compute_positional_counts(df):
    """
    For each unique glyph, count occurrences at:
      - initial position (word[0])
      - final position (word[-1])
      - medial position (word[1:-1])
    """
    init_counts  = defaultdict(int)
    final_counts = defaultdict(int)
    medial_counts = defaultdict(int)

    for _, row in df.iterrows():
        word = row["word"]
        if len(word) == 0:
            continue
        # Initial
        init_counts[word[0]] += 1
        # Final (only if word length > 1)
        if len(word) > 1:
            final_counts[word[-1]] += 1
        # Medial
        for g in word[1:-1]:
            medial_counts[g] += 1

    all_glyphs = set(list(init_counts.keys()) +
                     list(final_counts.keys()) +
                     list(medial_counts.keys()))
    
    records = []
    for g in sorted(all_glyphs):
        i = init_counts[g]
        m = medial_counts[g]
        f = final_counts[g]
        total = i + m + f
        if total < 20:  # skip very rare
            continue
        records.append({
            "glyph": g,
            "initial_count": i,
            "medial_count": m,
            "final_count": f,
            "total": total,
            "initial_pct": round(100 * i / total, 1),
            "medial_pct": round(100 * m / total, 1),
            "final_pct": round(100 * f / total, 1),
        })
    
    return pd.DataFrame(records)

def assign_phts_class(row, init_thresh=60.0, final_thresh=60.0, medial_thresh=50.0):
    """
    Assign PHTS positional class:
      F  = initial_pct >= 60%
      S  = final_pct  >= 60%
      C  = medial dominant or flexible
      ?  = unclear
    Operators are identified separately in script 04.
    """
    if row["initial_pct"] >= init_thresh:
        return "F"
    elif row["final_pct"] >= final_thresh:
        return "S"
    elif row["medial_pct"] >= medial_thresh:
        return "C"
    elif row["medial_pct"] >= 40.0:
        return "C"  # flexible but medial-leaning
    else:
        return "?"

def compute_grammar_role(row):
    """Map F/C/S to PHTS grammar grammar W = F C* S*"""
    cls = row["phts_class"]
    if cls == "F":
        return "word_initiator"
    elif cls == "C":
        return "word_core_carrier"
    elif cls == "S":
        return "word_terminator"
    else:
        return "unclassified"

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print("PHTS Script 02: Glyph Classification")
    print("=" * 50)
    
    df = load_processed(data_dir)
    print(f"Loaded {len(df):,} word tokens")
    
    pos_df = compute_positional_counts(df)
    print(f"Computed positional counts for {len(pos_df)} glyph types")
    
    pos_df["phts_class"] = pos_df.apply(assign_phts_class, axis=1)
    pos_df["grammar_role"] = pos_df.apply(compute_grammar_role, axis=1)
    pos_df = pos_df.sort_values("total", ascending=False)
    
    # Print summary
    print("\n── CLASSIFICATION SUMMARY ──")
    print(pos_df["phts_class"].value_counts().to_string())
    
    print("\n── TOP 15 GLYPHS BY FREQUENCY ──")
    top = pos_df.head(15)[["glyph","phts_class","initial_pct","medial_pct","final_pct","total"]]
    print(top.to_string(index=False))
    
    # Save
    out_path = os.path.join(results_dir, "glyph_classification_results.csv")
    pos_df.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")
    print("\nExpected: q,k,t → F; o,a,e,ch → C; y,n,s → S")
    print("Run 03_grammar_coverage.py next.")

if __name__ == "__main__":
    main()
