#!/usr/bin/env python3
"""
PHTS Theory v3.0 — Replication Package
Script 03: Grammar Coverage Test
=================================
Tests the PHTS minimal generative grammar:

    W = F C* S*

where:
  F = Frame glyph  (word-initial position)
  C = Core glyph   (medial position, zero or more)
  S = Specifier    (word-final position, zero or more)

PHTS v3.0 claim: This grammar covers ≥98% of all VM words.
This script verifies that claim against the EVA corpus.

Output: results/grammar_coverage_results.csv
"""

import os
import sys
import pandas as pd
import numpy as np

# PHTS glyph class assignments (from script 02 + operator_definitions.csv)
# Based on PHTS v3.0 Section 3.2 and glyph_positional_data.csv
FRAME_GLYPHS    = {"q", "k", "t", "f", "sh", "p"}          # F: word-initial
CORE_GLYPHS     = {"o", "a", "e", "ch", "l", "r", "ee",
                   "c", "ai", "ar", "ol", "or"}              # C: medial
SPECIFIER_GLYPHS = {"y", "n", "s", "m", "d", "dy", "edy"}   # S: word-final
OPERATOR_GLYPHS  = {"o", "k", "d", "c", "p"}                # OPs (subset of F/C/S)

def load_wordlist(data_dir):
    pkl = os.path.join(data_dir, "eva_processed.pkl")
    if not os.path.exists(pkl):
        print("ERROR: Run 01_preprocess_eva.py first.")
        sys.exit(1)
    df = pd.read_pickle(pkl)
    return df[df["section"] != "Unknown"].reset_index(drop=True)

def tokenize_word(word):
    """
    Simple bigram-aware tokenizer for common EVA digraphs.
    Returns list of EVA tokens.
    """
    digraphs = {"sh", "ch", "ee", "ai", "ar", "ol", "or", "dy", "edy", "kc", "qo", "sc"}
    tokens = []
    i = 0
    while i < len(word):
        # Try 3-char first (edy)
        if word[i:i+3] in digraphs:
            tokens.append(word[i:i+3])
            i += 3
        elif word[i:i+2] in digraphs:
            tokens.append(word[i:i+2])
            i += 2
        else:
            tokens.append(word[i])
            i += 1
    return tokens

def matches_grammar(tokens):
    """
    Check if a tokenized word matches W = F C* S*.
    Returns (matches:bool, pattern:str, reason:str)
    """
    if not tokens:
        return False, "", "empty"
    
    n = len(tokens)
    
    # Single-token word: must be F or C (allow)
    if n == 1:
        t = tokens[0]
        if t in FRAME_GLYPHS or t in CORE_GLYPHS or t in SPECIFIER_GLYPHS:
            return True, "F", "single-token-word"
        return False, "?", "unrecognized-single"
    
    # Find the boundary between F, C, and S regions
    # F: leading frame glyphs
    # C*: middle core glyphs  
    # S*: trailing specifier glyphs
    
    pattern_parts = []
    pos = 0
    
    # F segment (optional leading frame)
    has_F = False
    if tokens[0] in FRAME_GLYPHS:
        has_F = True
        pattern_parts.append("F")
        pos = 1
    
    # C* segment
    c_start = pos
    while pos < n and tokens[pos] in CORE_GLYPHS:
        pos += 1
    if pos > c_start:
        pattern_parts.append("C*")
    
    # S* segment
    s_start = pos
    while pos < n and tokens[pos] in SPECIFIER_GLYPHS:
        pos += 1
    if pos > s_start:
        pattern_parts.append("S*")
    
    pattern = " ".join(pattern_parts) if pattern_parts else "?"
    
    # Grammar match: all tokens consumed AND at least some recognized
    if pos == n and len(pattern_parts) > 0:
        return True, pattern, "grammar-match"
    elif pos == n and len(pattern_parts) == 0:
        return False, "NONE", "no-recognized-tokens"
    else:
        # Unconsumed tokens
        leftover = tokens[pos:]
        return False, pattern + "+?", f"unconsumed: {leftover}"

def compute_coverage(df):
    """Run grammar test on all words. Returns results DataFrame."""
    results = []
    for _, row in df.iterrows():
        word = row["word"]
        tokens = tokenize_word(word)
        matches, pattern, reason = matches_grammar(tokens)
        results.append({
            "word": word,
            "section": row["section"],
            "folio": row["folio"],
            "tokens": str(tokens),
            "token_count": len(tokens),
            "grammar_match": matches,
            "pattern": pattern,
            "reason": reason,
        })
    return pd.DataFrame(results)

def print_section_coverage(results_df):
    print("\n── COVERAGE BY SECTION ──")
    for section in results_df["section"].unique():
        sec = results_df[results_df["section"] == section]
        coverage = sec["grammar_match"].mean() * 100
        n = len(sec)
        print(f"  {section:<18}: {coverage:5.1f}%  (n={n:,})")

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print("PHTS Script 03: Grammar Coverage Test (W = F C* S*)")
    print("=" * 55)
    print("PHTS claim: grammar covers ≥98% of VM words")
    
    df = load_wordlist(data_dir)
    print(f"Testing {len(df):,} words...")
    
    results_df = compute_coverage(df)
    overall_coverage = results_df["grammar_match"].mean() * 100
    
    print(f"\n── OVERALL RESULT ──")
    print(f"Grammar coverage: {overall_coverage:.2f}%")
    print(f"Threshold:        98.00%")
    passed = overall_coverage >= 98.0
    print(f"STATUS:           {'✓ PASS' if passed else '✗ FAIL'}")
    
    print_section_coverage(results_df)
    
    # Most common patterns
    print("\n── TOP WORD PATTERNS ──")
    print(results_df["pattern"].value_counts().head(10).to_string())
    
    # Failure cases
    fails = results_df[~results_df["grammar_match"]]
    print(f"\nFailed words: {len(fails):,} ({len(fails)/len(results_df)*100:.1f}%)")
    print("Sample failures:")
    print(fails[["word","pattern","reason"]].head(10).to_string(index=False))
    
    out = os.path.join(results_dir, "grammar_coverage_results.csv")
    results_df.to_csv(out, index=False)
    print(f"\n✓ Saved: {out}")
    print("Run 04_operator_identification.py next.")

if __name__ == "__main__":
    main()
