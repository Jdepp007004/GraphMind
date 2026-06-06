# Time Context Coverage Audit

**Date:** 2026-06-06
**Question:** Did time-aware M1 fail due to (A) useless signal or (B) sparsity?

---

## Coverage Statistics

| Granularity | Steps | From Time Table | Fallback | Unseen State | Avg Trans/State |
|-------------|-------|----------------|----------|-------------|----------------|
| TimeAwareM1_6Band | 56447 | 55598 (98.5%) | 849 (1.5%) | 849 (100.0% of fallbacks) | 61.1 |
| TimeAwareM1_12Band | 56447 | 55119 (97.6%) | 1328 (2.4%) | 1328 (100.0% of fallbacks) | 38.6 |
| TimeAwareM1_24Hour | 56447 | 54349 (96.3%) | 2098 (3.7%) | 2098 (100.0% of fallbacks) | 24.6 |
| TimeAwareM1_48Bucket | 56447 | 53231 (94.3%) | 3216 (5.7%) | 3216 (100.0% of fallbacks) | 15.9 |

---

## Interpretation

**Conclusion: Mixed — moderate coverage but signal quality insufficient.**
