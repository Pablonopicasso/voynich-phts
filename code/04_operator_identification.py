#!/usr/bin/env python3
"""
PHTS Theory v3.0 — Replication Package
Script 04: Operator Identification by Positional Invariance
============================================================
Identifies the five PHTS operators (o, k, d, c, p) by measuring
their positional invariance *across sections*.

A glyph qualifies as a PHTS operator if:
  1. It has a strong positional signature (not random)
  2. Its positional signature is INVARIANT across sections (std < 5.0)
  3. It is functionally necessary (tested in kill-shot, script 07)

PHTS v3.0 claim: The 5 operators have mean invariance std = 3.4 (< 5.0 threshold)

Output: results/operator_positional_stats.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

# The 5 PHTS operators (as identified in PHTS v3.0 Section 3.3)
PHTS_OPERATORS = {
    "o":  {"name": "Process-Continuation", "expected_pos": "medial",  "expected_pct": (68, 98)},
    "k":  {"name": "Sequence-Initiation",  "expected_pos": "initial", "expected_pct": (88, 96)},
    "d":  {"name": "Sequence-Termination", "expected_pos": "final",   "expected_pct": (80, 90)},
    "c":  {"name": "Structural-Linkage",   "expected_pos": "medial",  "expected_pct": (45, 60)},
    "p":  {"name": "Sectional-Starter",    "expected_pos": "initial", "expected_pct": (70, 85)},
}

SECTIONS = ["Herbal", "Pharmaceutical", "Balneological", "Zodiac", "Astronomical"]

def load_data(data_dir):
    pkl = os.path.join(data_dir, "eva_processed.pkl")
    if not os.path.exists(pkl):
        print("ERROR: Run 01_preprocess_eva.py first.")
        sys.exit(1)
    return pd.read_pickle(pkl)

def compute_positional_by_section(df, glyph):
    """
    For a given glyph, compute initial/medial/final % in each section.
    Returns dict: section -> {initial_pct, medial_pct, final_pct, n}
    """
    section_stats = {}
    for section in SECTIONS:
        sec_df = df[df["section"] == section]
        init = sum(1 for w in sec_df["word"] if w and w[0] == glyph)
        final = sum(1 for w in sec_df["word"] if len(w) > 1 and w[-1] == glyph)
        medial = sum(g == glyph for w in sec_df["word"] for g in w[1:-1])
        total = init + medial + final
        if total == 0:
            section_stats[section] = {"initial_pct": 0, "medial_pct": 0,
                                       "final_pct": 0, "n": 0}
        else:
            section_stats[section] = {
                "initial_pct": round(100 * init / total, 1),
                "medial_pct":  round(100 * medial / total, 1),
                "final_pct":   round(100 * final / total, 1),
                "n": total,
            }
    return section_stats

def compute_invariance(section_stats, target_pos):
    """
    Compute std of target positional % across all sections.
    Low std = high invariance = good operator candidate.
    """
    values = [stats[f"{target_pos}_pct"] for stats in section_stats.values()
              if stats["n"] > 0]
    if len(values) < 2:
        return float("nan")
    return round(float(np.std(values)), 2)

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print("PHTS Script 04: Operator Identification by Positional Invariance")
    print("=" * 65)
    print("PHTS claim: 5 operators have mean invariance std = 3.4 (threshold < 5.0)")
    
    df = load_data(data_dir)
    
    records = []
    inv_stds = []
    
    for glyph, meta in PHTS_OPERATORS.items():
        print(f"\n── Operator '{glyph}' ({meta['name']}) ──")
        section_stats = compute_positional_by_section(df, glyph)
        inv_std = compute_invariance(section_stats, meta["expected_pos"])
        inv_stds.append(inv_std)
        
        passes = inv_std < 5.0 if not np.isnan(inv_std) else False
        
        print(f"  Expected position: {meta['expected_pos']}")
        print(f"  Invariance std:    {inv_std}")
        print(f"  Threshold:         < 5.0")
        print(f"  Status:            {'✓ PASS' if passes else '✗ FAIL'}")
        
        for section, stats in section_stats.items():
            pos_key = f"{meta['expected_pos']}_pct"
            print(f"    {section:<18}: {stats[pos_key]:5.1f}%  (n={stats['n']:,})")
            records.append({
                "operator": glyph,
                "operator_name": meta["name"],
                "section": section,
                "initial_pct": stats["initial_pct"],
                "medial_pct": stats["medial_pct"],
                "final_pct": stats["final_pct"],
                "n": stats["n"],
                "target_position": meta["expected_pos"],
                "target_pct": stats[f"{meta['expected_pos']}_pct"],
                "invariance_std": inv_std,
                "passes_threshold": passes,
            })
    
    mean_std = round(float(np.nanmean(inv_stds)), 2)
    all_pass = all(s < 5.0 for s in inv_stds if not np.isnan(s))
    
    print(f"\n── SUMMARY ──")
    print(f"Mean invariance std across 5 operators: {mean_std}")
    print(f"PHTS paper reports:                     3.4")
    print(f"Threshold:                              < 5.0")
    print(f"Overall status:  {'✓ PASS' if all_pass else '✗ FAIL'}")
    
    results_df = pd.DataFrame(records)
    out = os.path.join(results_dir, "operator_positional_stats.csv")
    results_df.to_csv(out, index=False)
    print(f"\n✓ Saved: {out}")
    print("Run 05_entropy_analysis.py next.")

if __name__ == "__main__":
    main()
