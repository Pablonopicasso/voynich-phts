# PHTS Operator Codebook — Reference Card
## PHTS Theory v3.0 | Mamontov (2026) | Cryptologia

---

## The 5 PHTS Operators

| ID | EVA | Name | Primary Role | Positional Signature | Invariance Std | Confidence |
|----|-----|------|-------------|----------------------|----------------|------------|
| OP1 | `o` | Process-Continuation | Drives medial structure | Medial 68–98% | 2.1 | 99% |
| OP2 | `k` | Sequence-Initiation | Opens procedural units | Initial ~93% | 1.9 | 99% |
| OP3 | `d` | Sequence-Termination | Closes procedural units | Final ~85% | 2.8 | 98% |
| OP4 | `c` | Structural-Linkage | Connects sub-units | Flexible (medial ~52%) | 4.1 | 97% |
| OP5 | `p` | Sectional-Starter | Marks section boundaries | Initial ~78% | 3.7 | 96% |

**Mean invariance std: 3.4** (threshold: <5.0 → PASS)

---

## Grammar Model: W = F C* S*

```
W      → Word (any VM word)
F      → Frame glyph (word-initiator): q, k, t, f, sh, p
C*     → Zero or more Core glyphs: o, a, e, ch, l, r, ee, c
S*     → Zero or more Specifier glyphs: y, n, s, d, m
```

**Coverage: ~98% of all VM words** (threshold: ≥98% → PASS)

---

## Glyph Class Assignments

### Frame Glyphs (F) — Word-initial position (≥60%)
| Glyph | Initial % | Notes |
|-------|-----------|-------|
| q | 97.0% | Most common word-initiator |
| k | 92.8% | Also OP2 (sequence-initiator) |
| t | 89.6% | Common gallows |
| f | 88.3% | Rare gallows |
| sh | 86.3% | Ligature, frequent |
| p | 78.2% | Also OP5 (sectional-starter) |

### Core Glyphs (C) — Medial position (≥50%)
| Glyph | Medial % | Notes |
|-------|----------|-------|
| o | 91.4% | OP1 — Process-Continuation |
| ee | 87.5% | Ligature, highly medial |
| a | 79.2% | High-frequency core |
| e | 78.6% | Common medial |
| l | 76.4% | Flexible core |
| r | 78.9% | Flexible core |
| ch | 65.0% | Ligature, core |
| c | 52.3% | OP4 — Structural-Linkage |

### Specifier Glyphs (S) — Word-final position (≥60%)
| Glyph | Final % | Notes |
|-------|---------|-------|
| y | 90.8% | Most common terminator |
| n | 89.1% | Common terminator |
| s | 88.9% | Common terminator |
| d | 85.0% | OP3 — Sequence-Termination |
| m | 61.0% | Occasional specifier |

---

## Kill-Shot: Permutation Scheme

The cyclic permutation used in the Operator Permutation Invariance Test:

```
o → k → d → c → p → o  (cyclic rotation)
```

This preserves:
- Glyph **frequencies** (each operator has same total count)
- Glyph **positions** (each position has same operator count)

This destroys:
- **Functional identity** (each position now has wrong operator)

**Result: 90.5% → ~22% accuracy collapse** → Kill-shot PASSED

---

## Section-Operator Co-occurrence Patterns

| Section | Dominant Operator | Secondary | Notes |
|---------|-------------------|-----------|-------|
| Herbal | OP1 (o) | OP2 (k) | Dense process-continuation |
| Pharmaceutical | OP2 (k) | OP3 (d) | Clear initiation-termination |
| Balneological | OP1 (o) | OP4 (c) | Linked processes |
| Zodiac | OP5 (p) | OP2 (k) | Section-level structure |
| Astronomical | OP5 (p) | OP3 (d) | Compact entries |

---

## Operator Permutation Map (Reference)

For reproducibility, the exact permutation used:

```python
PERMUTATION = {"o": "k", "k": "d", "d": "c", "c": "p", "p": "o"}
# Inverse permutation (for recovery):
INV_PERMUTATION = {"k": "o", "d": "k", "c": "d", "p": "c", "o": "p"}
```

---

*PHTS Theory v3.0 | Pavel Mamontov | Badalona, Spain | January 2026*
