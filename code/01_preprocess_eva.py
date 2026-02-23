#!/usr/bin/env python3
"""
PHTS Theory v3.0 — Replication Package
Script 01: EVA Corpus Preprocessor
====================================
Loads the raw EVA interlinear transcription from voynich.nu,
cleans it, segments it by folio and section, and outputs a
structured DataFrame for downstream analysis.

Output: data/eva_processed.pkl, data/eva_wordlist.csv
"""

import re
import os
import sys
import pandas as pd

# ── Section boundaries (PHTS v3.0, Section 3.1) ─────────────────────────────
SECTION_MAP = {
    "Herbal":         {"start": 1,  "end": 66},   # f1r–f66v
    "Pharmaceutical": {"start": 87, "end": 102},  # f87r–f102v
    "Balneological":  {"start": 75, "end": 84},   # f75r–f84v
    "Zodiac":         {"start": 70, "end": 73},   # f70r–f73v
    "Astronomical":   {"start": 67, "end": 69},   # f67r–f69v
}

def folio_to_section(folio_str):
    """Map a folio label (e.g., 'f9v') to its PHTS section name."""
    m = re.match(r"f(\d+)", folio_str)
    if not m:
        return "Unknown"
    num = int(m.group(1))
    for section, bounds in SECTION_MAP.items():
        if bounds["start"] <= num <= bounds["end"]:
            return section
    return "Unknown"

def parse_eva_line(line):
    """
    Parse one line of EVA interlinear format.
    Returns list of (word, position_in_line) tuples or None if header.
    """
    line = line.strip()
    if not line or line.startswith("<") or line.startswith("#"):
        return None, None
    # Remove annotations and uncertain markers
    line = re.sub(r"[!%\[\]]", "", line)
    # Split on word boundaries (- or space)
    words = [w for w in re.split(r"[-.\s]+", line) if w]
    return words

def load_eva_corpus(path):
    """
    Load the EVA raw transcription file.
    Returns a list of dicts: {folio, section, line_no, word, word_pos, glyphs}
    """
    records = []
    current_folio = "unknown"
    line_no = 0

    print(f"Loading EVA corpus from: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            # Folio marker
            m = re.match(r"<(f\d+[rv])>", raw_line)
            if m:
                current_folio = m.group(1)
                continue
            words = parse_eva_line(raw_line)[0]
            if not words:
                continue
            line_no += 1
            section = folio_to_section(current_folio)
            for pos, word in enumerate(words):
                if len(word) < 1:
                    continue
                records.append({
                    "folio": current_folio,
                    "section": section,
                    "line_no": line_no,
                    "word": word,
                    "word_pos_in_line": pos,
                    "word_length": len(word),
                    "glyphs": list(word),
                    "first_glyph": word[0] if word else "",
                    "last_glyph": word[-1] if word else "",
                    "middle_glyphs": list(word[1:-1]) if len(word) > 2 else [],
                })
    
    print(f"  Parsed {len(records):,} word tokens")
    print(f"  Folios: {len(set(r['folio'] for r in records))}")
    sections = pd.Series([r['section'] for r in records]).value_counts()
    print(f"  Section distribution:")
    for sec, cnt in sections.items():
        print(f"    {sec}: {cnt:,} words ({cnt/len(records)*100:.1f}%)")
    return records

def save_outputs(records, output_dir):
    """Save processed corpus as pickle and word-level CSV."""
    df = pd.DataFrame(records)
    
    # Explode glyphs to glyph-level for positional analysis
    pkl_path = os.path.join(output_dir, "eva_processed.pkl")
    df.to_pickle(pkl_path)
    print(f"\nSaved word-level DataFrame: {pkl_path}")
    
    # Word-level CSV (without glyph list columns for readability)
    csv_df = df.drop(columns=["glyphs", "middle_glyphs"])
    csv_path = os.path.join(output_dir, "eva_wordlist.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved word-list CSV: {csv_path}")
    
    # Summary stats
    print(f"\n── CORPUS SUMMARY ──")
    print(f"Total word tokens: {len(df):,}")
    print(f"Unique word types: {df['word'].nunique():,}")
    print(f"Unique folios:     {df['folio'].nunique()}")
    print(f"Mean word length:  {df['word_length'].mean():.2f} glyphs")
    
    return df

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    eva_path = os.path.join(data_dir, "eva_raw.txt")
    
    if not os.path.exists(eva_path):
        print(f"ERROR: EVA corpus not found at {eva_path}")
        print("Download from https://voynich.nu/transcr.html")
        sys.exit(1)
    
    records = load_eva_corpus(eva_path)
    df = save_outputs(records, data_dir)
    print("\n✓ Preprocessing complete. Run 02_glyph_classification.py next.")
    return df

if __name__ == "__main__":
    main()
