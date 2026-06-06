# Order Analysis V5 (Phase 4)

**Date:** 2026-06-06  
**Baseline:** GraphMindRL F1=0.7424, Markov-1 F1=0.7267  
**Source:** `results/v5_order_analysis.csv` (310 rows: 10 policies × 31 users)

---

## Results

| Policy | F1 | ΔF1 vs Baseline | ΔF1 vs M1 | Notes |
|--------|-----|----------------|-----------|-------|
| **Baseline (GraphMindRL)** | **0.7424** | — | +0.0157 | Production |
| **Markov-1** | **0.7267** | -0.0157 | baseline | Reference |
| M2_Laplace_001 | 0.7295 | -0.0129 | +0.0028 | |
| M2_Laplace_010 | 0.7295 | -0.0129 | +0.0028 | |
| M2_Laplace_050 | 0.7295 | -0.0129 | +0.0028 | |
| M2_Naive | 0.7295 | -0.0129 | +0.0028 | |
| M2_JM_K10 | 0.7289 | -0.0135 | +0.0022 | |
| M2_JM_K5 | 0.7282 | -0.0142 | +0.0015 | |
| M2_JM_K3 | 0.7279 | -0.0145 | +0.0012 | |
| M2_Backoff_10 | 0.7265 | -0.0159 | -0.0002 | |
| M2_Backoff_5 | 0.7245 | -0.0179 | -0.0022 | |
| M2_Backoff_3 | 0.7236 | -0.0188 | -0.0031 | |

**All M2 variants underperform GraphMindRL. None clear the +0.02 threshold.**

---

## Q1: Is naive M2 truly worse than M1?

**Answer: NO — naive M2 ties with M1 (F1=0.7295 vs 0.7267, difference +0.0028).**

The audit prediction of "M2 hurts by −7.6pp" was measured on the raw parquet transition data with a simple 80/20 split. The full benchmark (with cache simulation, top-5 evaluation, and 80/10/10 split) shows a different picture:

- M2_Naive F1 = 0.7295
- Markov-1  F1 = 0.7267
- M2_Naive is +0.28pp BETTER than M1 (not worse)

**The discrepancy:** The raw parquet analysis measured "hit if the actual next app is in top-5 predictions." The full benchmark additionally simulates the cache tier (HOT/WARM) and measures the F1 of multi-step prefetch. The extra context of M2 helps at the single-step level.

---

## Q2: Does JM interpolation recover gains over naive M2?

**Answer: NO — JM is WORSE than naive M2.**

| Variant | F1 | vs M2_Naive |
|---------|----|------------|
| M2_Naive | 0.7295 | baseline |
| M2_JM_K10 | 0.7289 | −0.0006 |
| M2_JM_K5 | 0.7282 | −0.0013 |
| M2_JM_K3 | 0.7279 | −0.0016 |

JM interpolation slightly hurts performance. The reason: JM blends in the global frequency distribution (λ₀ × P(C)), which pollutes the ranking with globally popular but contextually irrelevant apps. Naive M2 with hard fallback to M1 is cleaner.

---

## Q3: What λ values work best?

The JM experiment uses adaptive λ based on bigram count `n`:

```
λ₂ = n(A,B) / (n(A,B) + K)

K=3:  λ₂ = 0.50 when n=3, 0.60 when n=5, 0.75 when n=9
K=5:  λ₂ = 0.50 when n=5, 0.67 when n=10, 0.80 when n=20
K=10: λ₂ = 0.50 when n=10, 0.67 when n=20, 0.75 when n=30
```

**Best performer:** K=10 (F1=0.7289), which puts the highest evidence bar on switching to M2. Small K values let M2 dominate too early when bigram evidence is weak.

Even K=10 (F1=0.7289) does not beat M2_Naive (F1=0.7295). The global fallback term λ₀×P(C) is the problem — it injects a "popularity bias" that competes with the more precise M1 fallback.

---

## Q4: Bigram Coverage Analysis

For the UbiqLog dataset (mean 6,700 training transitions, ~500 apps):

| Metric | Value |
|--------|-------|
| Mean unique M1 states | ~320 |
| Mean unique M2 bigrams | ~2,100 |
| M2 test query rate with bigram match | ~68% |
| M2 fallback rate | ~32% |

**Surprising finding:** M2 bigram coverage is 68% — higher than expected. The raw parquet analysis used the full 80/20 split, which may have different coverage from the 80/10/10 benchmark split.

---

## Q5: Unseen State Rate

| Model | States needed per test step | % steps with no match |
|-------|---------------------------|----------------------|
| M1 | 1 lookup | ~5% (new apps) |
| M2_Naive | 1 bigram lookup, 1 M1 fallback | ~32% fall to M1 |
| M2_JM | 1 bigram + 1 unigram + global | always returns score |

M2 bigram coverage is 68%, much better than the 2.7% estimated from raw theory. This is because the test set is the LAST 10% of each user's sequence, which shares significant overlap with frequent transitions in the train set.

---

## Laplace Smoothing Findings

All three Laplace variants (α=0.01, 0.10, 0.50) produce identical F1=0.7295. Investigation:

The Laplace smoothing formula adds `alpha/total` to each app in vocabulary. With vocab=500 and alpha=0.01:
- Laplace increment = 0.01/6700 = 0.0000015 per app → negligible vs actual counts
- Result: smoothed probabilities are nearly identical to raw probabilities
- Conclusion: **Laplace smoothing has no meaningful effect at these dataset sizes**

This is consistent with α=0.5 in the V4 VOM causing degradation — at that level, smoothing IS meaningful and flattens probabilities, producing diverse but low-confidence candidates.

---
