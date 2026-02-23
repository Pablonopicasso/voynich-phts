#!/usr/bin/env python3
"""
PHTS Theory v3.0 — Replication Package
Script 05: Entropy Analysis + OETC Spike Test
==============================================
Computes:
  1. Shannon entropy H per section
  2. Cross-entropy between sections (4.0x ratio)
  3. Operator Erasure and Template Collapse (OETC) test
     → Removing operators should cause >100% entropy spike

PHTS v3.0 claims (Section 5.2–5.4):
  - Herbal H ≈ 1.7 bits, Zodiac H ≈ 0.7 bits (well below random ~4.5)
  - Cross-entropy ratio: 4.0x  (threshold: >1.5x)
  - OETC entropy spike: +124%  (threshold: >100%)

Output: results/entropy_by_section.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
import math

OPERATORS = {"o", "k", "d", "c", "p"}
SECTIONS  = ["Herbal", "Pharmaceutical", "Balneological", "Zodiac", "Astronomical"]

def load_data(data_dir):
    pkl = os.path.join(data_dir, "eva_processed.pkl")
    if not os.path.exists(pkl):
        print("ERROR: Run 01_preprocess_eva.py first.")
        sys.exit(1)
    return pd.read_pickle(pkl)

def glyph_sequence(df_section):
    """Return flat list of all glyphs in a section."""
    glyphs = []
    for word in df_section["word"]:
        glyphs.extend(list(word))
    return glyphs

def shannon_entropy(sequence):
    """Compute Shannon entropy H in bits."""
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    total = sum(counts.values())
    H = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return round(H, 4)

def cross_entropy(p_dist, q_dist):
    """H(p, q) = -sum p(x) log q(x)"""
    all_keys = set(p_dist) | set(q_dist)
    total_p = sum(p_dist.values())
    total_q = sum(q_dist.values())
    smoothing = 1e-10
    H_cross = 0.0
    for k in all_keys:
        p = p_dist.get(k, 0) / total_p
        q = (q_dist.get(k, 0) + smoothing) / (total_q + smoothing * len(all_keys))
        if p > 0:
            H_cross -= p * math.log2(q)
    return round(H_cross, 4)

def remove_operators(glyphs):
    """Remove all operator glyphs from sequence (OETC erasure)."""
    return [g for g in glyphs if g not in OPERATORS]

def compute_section_entropy(df):
    """Compute H per section, with and without operators."""
    results = {}
    for section in SECTIONS:
        sec = df[df["section"] == section]
        glyphs_full = glyph_sequence(sec)
        glyphs_no_op = remove_operators(glyphs_full)
        
        H_full  = shannon_entropy(glyphs_full)
        H_no_op = shannon_entropy(glyphs_no_op)
        spike_pct = round((H_no_op - H_full) / H_full * 100, 1) if H_full > 0 else 0
        
        results[section] = {
            "section": section,
            "n_glyphs_full": len(glyphs_full),
            "n_glyphs_no_op": len(glyphs_no_op),
            "pct_operators": round(100*(1 - len(glyphs_no_op)/len(glyphs_full)), 1),
            "H_full_bits": H_full,
            "H_no_operator_bits": H_no_op,
            "entropy_spike_pct": spike_pct,
            "distribution": Counter(glyphs_full),
        }
    return results

def compute_cross_entropy_ratio(section_results):
    """
    Compute mean cross-entropy between sections / mean self-entropy.
    PHTS claim: ratio = 4.0x (threshold: >1.5x)
    """
    self_entropies = [v["H_full_bits"] for v in section_results.values()]
    mean_self_H = np.mean(self_entropies)
    
    cross_entropies = []
    sections = list(section_results.keys())
    for i, s1 in enumerate(sections):
        for j, s2 in enumerate(sections):
            if i != j:
                hcross = cross_entropy(
                    section_results[s1]["distribution"],
                    section_results[s2]["distribution"]
                )
                cross_entropies.append(hcross)
    
    mean_cross_H = np.mean(cross_entropies)
    ratio = round(mean_cross_H / mean_self_H, 2) if mean_self_H > 0 else 0
    return mean_self_H, mean_cross_H, ratio

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print("PHTS Script 05: Entropy Analysis + OETC Spike Test")
    print("=" * 55)
    
    df = load_data(data_dir)
    df = df[df["section"] != "Unknown"]
    
    print("\n── SECTION ENTROPY (with vs without operators) ──")
    section_results = compute_section_entropy(df)
    
    rows = []
    oetc_spikes = []
    for section, r in section_results.items():
        print(f"  {section:<18}: H={r['H_full_bits']} bits → H(no-op)={r['H_no_operator_bits']} bits  "
              f"spike={r['entropy_spike_pct']}%")
        oetc_spikes.append(r["entropy_spike_pct"])
        rows.append({k: v for k, v in r.items() if k != "distribution"})
    
    mean_spike = round(float(np.mean(oetc_spikes)), 1)
    oetc_pass = mean_spike >= 100.0
    
    print(f"\n── OETC RESULT ──")
    print(f"Mean entropy spike: {mean_spike}%")
    print(f"PHTS reports:       +124%")
    print(f"Threshold:          >100%")
    print(f"STATUS:             {'✓ PASS' if oetc_pass else '✗ FAIL'}")
    
    mean_self, mean_cross, ratio = compute_cross_entropy_ratio(section_results)
    cross_pass = ratio >= 1.5
    
    print(f"\n── CROSS-ENTROPY RATIO ──")
    print(f"Mean self-entropy:    {mean_self:.4f} bits")
    print(f"Mean cross-entropy:   {mean_cross:.4f} bits")
    print(f"Ratio:                {ratio}x")
    print(f"PHTS reports:         4.0x")
    print(f"Threshold:            >1.5x")
    print(f"STATUS:               {'✓ PASS' if cross_pass else '✗ FAIL'}")
    
    results_df = pd.DataFrame(rows)
    results_df["cross_entropy_ratio"] = ratio
    out = os.path.join(results_dir, "entropy_by_section.csv")
    results_df.to_csv(out, index=False)
    print(f"\n✓ Saved: {out}")
    print("Run 06_ml_classifier.py next.")

if __name__ == "__main__":
    main()
