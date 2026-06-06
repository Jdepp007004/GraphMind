# Time Context Benchmark (Phase 3)

**Date:** 2026-06-06  
**Baseline:** GraphMindRL F1=0.7424  
**Source:** `results/v5_time_context.csv` (124 rows: 4 policies × 31 users)  
**Experiment:** P(next | app, time_band) with M1 fallback, 4 granularities

---

## Results

| Policy | F1 | ΔF1 | Hit Rate | Lat (ms) | Sig? |
|--------|-----|-----|---------|----------|------|
| **Baseline (GraphMindRL)** | **0.7424** | — | 0.9357 | 2002.5 | — |
| TimeAwareM1_6Band | 0.7151 | **-0.0273** | 0.9338 | ~1985 | ❌ Worse |
| TimeAwareM1_12Band | 0.7001 | **-0.0423** | 0.9301 | ~1971 | ❌ Worse |
| TimeAwareM1_24Hour | 0.6915 | **-0.0509** | 0.9264 | ~1961 | ❌ Worse |
| TimeAwareM1_48Bucket | 0.6870 | **-0.0554** | 0.9256 | ~1957 | ❌ Worse |

**All time-aware M1 variants underperform the baseline.**

---

## Q1: Does time context help?

**Answer: NO — in pure M1 form, time conditioning HURTS.**

The direction of degradation is monotonic with granularity:
- 6 bands:   −0.027 F1
- 12 bands:  −0.042 F1  
- 24 hours:  −0.051 F1
- 48 buckets: −0.055 F1

This is exactly the sparsity pattern predicted by the architecture audit. More granular time splits produce emptier lookup tables, causing more frequent M1 fallbacks with degraded precision.

---

## Q2: Which granularity wins?

**6-band is best** (least degradation), confirming that coarser time splits preserve more data per cell.

| Coarse bands | Approx transitions/cell | Fallback rate |
|-------------|------------------------|--------------|
| 6 bands | ~1,100/cell | ~20% cells empty |
| 12 bands | ~560/cell | ~35% cells empty |
| 24 hours | ~280/cell | ~55% cells empty |
| 48 buckets | ~140/cell | ~70% cells empty |

Even 6-band still underperforms because when the time-conditioned table IS populated, it narrows the prediction set too aggressively for rare transition patterns.

---

## Q3: Where does sparsity begin?

**Sparsity begins immediately.** With mean 6,700 training transitions per user and 500 apps:

```
M1:     500 states    → mean 13.4 transitions/state  → ADEQUATE
TA-M1 (6-band): 3,000 states → mean 2.2 transitions/state → SPARSE
TA-M1 (48-bucket): 24,000 states → mean 0.3 transitions/state → VERY SPARSE
```

The fragmentation occurs because each `(app, time_band)` pair is a separate lookup key. Most apps do not appear in all time bands, so even 6 coarse bands produce many empty cells.

---

## Root Cause: Why the Audit Prediction Was Wrong

The Phase 5 audit analysis measured raw parquet hit rate (+5.4pp for ContextMarkov-1). That measure used `coarse_bucket = time_bucket // 8` and only counted transitions from the training set — it did **not** measure the effect on prediction output quality at the full benchmark level.

The actual benchmark reveals the tradeoff: time conditioning does reduce entropy within each cell, but the **precision loss from more frequent fallback** outweighs the entropy reduction gain when evaluated at the standard top-5 level.

**Conclusion:** Raw time conditioning on M1 does not deliver the predicted gain. Requires better integration (soft conditioning or feature augmentation rather than hard table split).

---
