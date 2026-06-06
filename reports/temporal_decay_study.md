# Temporal Decay Study (Phase 8)

**Date:** 2026-06-06  
**Source:** `results/v5_temporal_decay.csv` (124 rows: 4 policies × 31 users)

---

## Results

| Policy | Half-life | F1 | ΔF1 | Hit Rate |
|--------|----------|-----|-----|---------|
| **Baseline (GraphMindRL)** | n/a | **0.7424** | — | 0.9357 |
| Decay_30d | 30 days | 0.7286 | -0.0138 | 0.9386 |
| Decay_60d | 60 days | 0.7283 | -0.0141 | 0.9399 |
| Decay_14d | 14 days | 0.7282 | -0.0142 | 0.9384 |
| Decay_7d | 7 days | 0.7263 | -0.0161 | 0.9376 |

**All temporal decay variants underperform the baseline.**

---

## Q: Does behavioral drift matter in UbiqLog?

**Answer: NO — temporal decay consistently hurts performance.**

The monotonic degradation pattern:
- 60-day half-life: −0.0141 (near plain M1)
- 30-day half-life: −0.0138 (also near plain M1)
- 14-day half-life: −0.0142
- 7-day half-life:  −0.0161 (most aggressive decay, worst)

The decay variants all produce F1 ≈ 0.7280 — essentially equivalent to naive M2 (0.7295) but slightly worse due to the decay distorting transition counts.

---

## Why Decay Hurts

### Hit rate paradox

Decay_60d achieves the HIGHEST hit rate (0.9399) but LOWER F1 than baseline (0.7283). This reveals that the decay policy is maximizing cache OCCUPANCY (fewer evictions) at the cost of precision.

The decay re-weights old transitions downward, so the TOP-K predictions contain different apps than the plain M1 top-K. These are often well-known apps (frequent recently) rather than contextually appropriate ones.

### Dataset characteristics

UbiqLog spans ~2 months per user. With a 14-day half-life, the most recent 2 weeks dominate. But the train set is the first 80% of data (~6-7 weeks). The 14-day half-life over-weights the last 2 weeks of training, which is the period MOST SIMILAR to the test set. This should help... but in practice:

- The test set is only the LAST 10% (~1 week), not the next 2 weeks
- Users in UbiqLog have relatively stable behavior (dataset collection bias)
- The transition matrix is already implicitly recency-weighted because the test set follows the training set

### Statistical interpretation

The near-identical F1 values (0.7282–0.7295) across decay and non-decay M2 variants suggest that behavioral drift is NOT a significant factor in this dataset. The UbiqLog collection covers a 2-month window; within this window, user behavior is stable enough that equal-weight historical transitions perform as well as or better than decayed ones.

---

## Conclusion

**Temporal decay provides no benefit for UbiqLog.** This is likely a dataset property: 2 months of stable behavior is not long enough for significant drift to appear. Temporal decay would be valuable for a 12-month+ dataset with seasonal behavioral changes.

For V5 implementation purposes: **do not implement temporal decay.** The engineering cost is not justified by this dataset.

---
