# PHTS Theory v3.0 — Replication Package
## Procedural Humoral-Tabular System Theory for the Voynich Manuscript
### Companion to: Mamontov, P. (2026). *Cryptologia* Submission

---

**Author:** Pavel Mamontov, Independent Researcher, Badalona, Catalonia, Spain  
**AI Contributors:** Claude (Anthropic), ChatGPT (OpenAI), Grok (xAI)  
**Version:** 3.0  
**Date:** January 2026  
**Archive:** `PHTS_v3_Replication_Package.zip`  
**Target Journal:** *Cryptologia*

---

## What This Package Contains

This replication package provides everything needed for an independent researcher to fully reproduce all Tier-1 (structural) validation results reported in the PHTS Theory v3.0 paper. Tier-2 (semantic) results are documented but cannot be reproduced independently without the bilingual anchor set.

```
PHTS_v3_Replication_Package/
│
├── README.md                          ← This file
│
├── data/
│   ├── eva_corpus_schema.md           ← EVA corpus description and access instructions
│   ├── operator_definitions.csv       ← The 5 core PHTS operators with positional data
│   ├── section_mappings.csv           ← Folio-to-section assignments
│   ├── glyph_positional_data.csv      ← Positional frequency table (initial/medial/final)
│   ├── category_encoding_table.csv    ← Tier-2: Prefix/Core/Suffix semantic mappings
│   └── tacuinum_parallel_table.csv    ← Tier-2: VM–Tacuinum Sanitatis field alignment
│
├── code/
│   ├── 00_setup.py                    ← Environment check and dependency installer
│   ├── 01_preprocess_eva.py           ← Load, clean, and segment EVA corpus
│   ├── 02_glyph_classification.py     ← Positional classification → F/C/S roles
│   ├── 03_grammar_coverage.py         ← W = F C* S* grammar coverage test
│   ├── 04_operator_identification.py  ← Identify 5 operators by invariance
│   ├── 05_entropy_analysis.py         ← Shannon entropy + OETC spike test
│   ├── 06_ml_classifier.py            ← Neural classifier + Grouped K-Fold + LOSO
│   ├── 07_kill_shot_test.py           ← THE KILL-SHOT: Operator Permutation Test
│   ├── 08_falsification_battery.py    ← Runs all 9 falsification criteria
│   ├── 09_full_validation_battery.py  ← Runs complete 30-test battery + report
│   └── utils/
│       ├── corpus_loader.py           ← Shared EVA loader utilities
│       └── stats_helpers.py           ← Chi-square, entropy, cross-entropy helpers
│
├── results/
│   ├── tier1_validation_results.csv   ← Machine-readable 30-test results
│   ├── kill_shot_results.csv          ← Permutation test accuracy table
│   ├── falsification_criteria.csv     ← 9 criteria, thresholds, current values
│   ├── operator_positional_stats.csv  ← Per-operator, per-section positional stats
│   ├── entropy_by_section.csv         ← Shannon entropy H per section
│   └── ml_classification_report.txt   ← Sklearn classification report
│
└── docs/
    ├── PHTS_v3_0_Cryptologia_Submission.docx   ← Full paper (submitted version)
    ├── PHTS_Replication_Guide.docx              ← Step-by-step replication guide
    ├── codebook_operators.md                    ← Operator reference card
    └── sha256_manifest.txt                      ← File integrity hashes
```

---

## Quick Start (Reproduce Kill-Shot in 5 Minutes)

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn scipy matplotlib

# 2. Download EVA corpus (free, public domain)
#    → https://voynich.nu/transcr.html (download interlinear transcription)
#    → Save as data/eva_raw.txt

# 3. Preprocess
python code/01_preprocess_eva.py

# 4. Run THE KILL-SHOT
python code/07_kill_shot_test.py
# Expected output: Original accuracy ~90.5% → Post-permutation ~22%

# 5. Run full validation battery
python code/09_full_validation_battery.py
# Expected output: 30/30 tests PASS, 0/9 falsification criteria met
```

---

## Key Claims and Where to Find Them

| Claim | Where Validated | Script | Expected Result |
|-------|----------------|--------|-----------------|
| Grammar covers 98% of words | Section 3.2 | `03_grammar_coverage.py` | ≥98% |
| Operators have positional invariance (std=3.4) | Section 3.3 | `04_operator_identification.py` | std < 5.0 |
| OETC entropy spike >100% | Section 5.4 | `05_entropy_analysis.py` | +124% |
| ML accuracy 90.5% | Section 5.5 | `06_ml_classifier.py` | >80% |
| **Kill-shot collapses to ~22%** | **Section 4** | **`07_kill_shot_test.py`** | **~20% chance** |
| 0/9 falsification criteria met | Section 7 | `08_falsification_battery.py` | 0 met |
| Cross-entropy ratio 4.0x | Section 5.3 | `05_entropy_analysis.py` | >1.5x |

---

## The Two-Tier Framework

| Tier | Claims | Confidence | Reproducible? |
|------|--------|------------|---------------|
| **Tier-1** | Structural (grammar, operators, entropy, ML) | 99% | ✅ Fully |
| **Tier-2** | Semantic (humoral mappings, decipherments) | 88% | ⚠️ Partially (requires anchor set) |

**Important:** All scripts in this package test Tier-1 claims only, as these are falsifiable through the EVA corpus alone. Tier-2 results are documented in `data/category_encoding_table.csv` and `data/tacuinum_parallel_table.csv`.

---

## Data Availability

The EVA transcription corpus is **publicly available** at:
- voynich.nu/transcr.html (René Zandbergen, 2024)
- Also archived on Zenodo: DOI 10.5281/zenodo.xxxxxxx (placeholder)

The corpus contains ~157,000 glyph tokens across ~240 folios in the standard interlinear format.

---

## Dependencies

```
Python ≥ 3.9
numpy ≥ 1.24
pandas ≥ 2.0
scikit-learn ≥ 1.3
scipy ≥ 1.11
matplotlib ≥ 3.7
```

---

## Citing This Package

Mamontov, P. (2026). PHTS Theory v3.0 Replication Package [Data and Code]. *Zenodo*. DOI: 10.5281/zenodo.xxxxxxx

---

## License

Code: MIT License  
Data (EVA corpus): Public domain (Zandbergen, 2024)  
Paper: © Pavel Mamontov 2026, submitted to *Cryptologia*

---

## Contact

Pavel Mamontov — Independent Researcher, Badalona, Catalonia, Spain  
[Contact via Cryptologia editorial office during peer review]
