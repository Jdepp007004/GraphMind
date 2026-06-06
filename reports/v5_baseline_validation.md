# V5 Baseline Validation

**Date:** 2026-06-06  
**Purpose:** Confirm that the existing benchmark outputs are internally consistent and match stated baseline before any V5 experiments.

---

## 1. Source Files Verified

| File | Path | Rows | Status |
|------|------|------|--------|
| `benchmark_results_v4.csv` | `results/benchmark_results_v4.csv` | 465 | ✅ Present |
| `user_level_results_v4.csv` | `results/user_level_results_v4.csv` | 465 | ✅ Present |
| `statistical_results_v4.csv` | `results/statistical_results_v4.csv` | 27 | ✅ Present |
| `benchmark_results_fast.csv` | `results/benchmark_results_fast.csv` | — | ⚠️ Not found (fast run was integrated into V4) |

**Coverage:** 31 users × 15 policies = 465 rows. All 31 usable users present.

---

## 2. Baseline Metrics — Confirmed

Aggregated from `benchmark_results_v4.csv` (mean across 31 users, chronological 80/10/10 split):

| Rank | Policy | F1 | Hit Rate | Lat Saved (ms) | N Users |
|------|--------|-----|---------|----------------|---------|
| 1 | **GraphMindRL** | **0.7424** | **0.9357** | **2002.5** | 31 |
| 2 | Graph+Confidence | 0.7408 | 0.9355 | 2002.2 | 31 |
| 3 | Markov-2 | 0.7295 | 0.9297 | 1993.8 | 31 |
| 4 | Markov-1 | 0.7267 | 0.9380 | 2005.7 | 31 |
| 4 | GraphOnly | 0.7267 | 0.9380 | 2005.7 | 31 |
| 6 | GlobalMarkov2 | 0.6790 | 0.9132 | 1969.8 | 31 |
| 7 | VariableOrderMarkov | 0.6249 | 0.9201 | 1979.8 | 31 |
| 8 | RLAdaptiveEnsemble | 0.6170 | 0.9175 | 1976.0 | 31 |
| 9 | LFU | 0.6153 | 0.9291 | 1992.7 | 31 |
| 10 | ClusterMarkov | 0.6117 | 0.9009 | 1951.8 | 31 |
| 11 | ContextMarkov | 0.6095 | 0.9170 | 1975.2 | 31 |
| 12 | RecencyFrequency | 0.5914 | 0.9298 | 1993.8 | 31 |
| 13 | Frequency | 0.5844 | 0.9296 | 1993.4 | 31 |
| 14 | LRU | 0.5838 | 0.9280 | 1991.2 | 31 |
| 15 | Random | 0.2372 | 0.7934 | 1795.4 | 31 |

✅ **GraphMindRL F1 = 0.7424 — CONFIRMED**  
✅ **GraphOnly F1 = 0.7267 — CONFIRMED**  
✅ **Markov-1 F1 = 0.7267 — CONFIRMED**

---

## 3. Baseline Validation Checks

### 3.1 GraphOnly == Markov-1 (exact match)

Per-user comparison of F1 scores:

```
Max difference:  0.000000
Mean difference: 0.000000
All 31 users identical: TRUE
```

**GraphOnly and Markov-1 are mathematically identical in all 31 users.**  
This is not statistical coincidence — it is structural identity (both build `dict[app→dict[app,prob]]`).

### 3.2 GraphMindRL margin over Markov-1

```
GraphMindRL F1 = 0.7424
Markov-1    F1 = 0.7267
Margin      = +0.0157 (not statistically significant, p=0.1196)
```

The GraphMindRL advantage over Markov-1 is small and not statistically significant.  
The statistically significant wins are only against GlobalMarkov2 (p=0.0001, d=0.45).

### 3.3 Split Consistency

All policies use identical:
- 31 users
- 80% train / 10% val / 10% test
- Chronological split (no leakage)
- MAX_GAP = 3600s
- HOT_SIZE = 5, WARM_SIZE = 15
- Galaxy A23 latency model (cold=2763ms, warm=1301ms, hot=274ms)

### 3.4 Latency Model Source

Measured values from Samsung Galaxy A23:
- Cold start: **2,763 ms** (mean across 13 apps, 3,900 measurements)
- Warm start: **1,301 ms**
- Hot start:  **274 ms**

No literature placeholder values used.

---

## 4. Baseline Agreement With Stated Values

| Metric | Stated | Measured | Match |
|--------|--------|----------|-------|
| GraphMindRL F1 | 0.7424 | 0.7424 | ✅ Exact |
| GraphMindRL Hit Rate | 0.9357 | 0.9357 | ✅ Exact |
| GraphMindRL Lat Saved | 2002.5 ms | 2002.5 ms | ✅ Exact |
| Users | 31 | 31 | ✅ Exact |
| Split | 80/10/10 | 80/10/10 | ✅ Exact |

---

## 5. V5 Improvement Target

| Target | Value |
|--------|-------|
| Current best F1 | 0.7424 (GraphMindRL) |
| Minimum upgrade threshold | **+0.02 absolute = F1 ≥ 0.7624** |
| Aspirational target | F1 ≥ 0.78 |
| Significance requirement | p < 0.05 (paired t-test, 31 users) |

---
