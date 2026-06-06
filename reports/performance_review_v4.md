# GraphMind V4 — Performance Review & Decision Gate

**Decision: ✅ KEEP GraphMindRL V3**

---

## Primary Comparison

| Metric | V3 (GraphMindRL) | V4 (RLAdaptiveEnsemble) | Δ | Relative |
|--------|-----------------|-----------------|---|----------|
| **F1** | 0.7424 | 0.6170 | -0.1254 | -16.89% |
| Hit Rate | 0.9357 | 0.9175 | -0.0182 | — |
| Lat Saved (ms) | 2002.5 | 1976.0 | -26.6 | — |

## Statistical Evidence

| Test | Statistic | Threshold | Met? |
|------|----------|-----------|------|
| p-value (paired t-test) | 0.0002 | < 0.05 | ✅ |
| F1 improvement | -0.1254 | ≥ 0.02 | ❌ |
| Cohen's d | -0.741 (medium) | — | — |

## Decision Criteria

```
UPGRADE if:
  F1 improvement >= 0.02  → -0.1254  (NOT MET)
  AND p < 0.05          → 0.0002   (MET)

DECISION: KEEP
```

## Winning Policy: **GraphMindRL**

- F1: **0.7424**
- Hit Rate: **0.9357**
- Latency Saved: **2002.5 ms/launch**

## Full Policy Ranking (all V4 policies by F1)

| Rank | Policy | F1 | Hit Rate | Lat Saved (ms) | Notes |
|------|--------|----|---------:|---------------:|-------|
| 1 | GraphMindRL | 0.7424 | 0.9357 | 2002.5 | ← **Dashboard policy** 🏆 |
| 2 | Graph+Confidence | 0.7408 | 0.9355 | 2002.2 |  |
| 3 | Markov-2 | 0.7295 | 0.9297 | 1993.8 |  |
| 4 | Markov-1 | 0.7267 | 0.9380 | 2005.7 |  |
| 5 | GraphOnly | 0.7267 | 0.9380 | 2005.7 |  |
| 6 | GlobalMarkov2 | 0.6790 | 0.9132 | 1969.8 |  |
| 7 | VariableOrderMarkov | 0.6249 | 0.9201 | 1979.8 |  |
| 8 | RLAdaptiveEnsemble | 0.6170 | 0.9175 | 1976.0 |  |
| 9 | LFU | 0.6153 | 0.9291 | 1992.7 |  |
| 10 | ClusterMarkov | 0.6117 | 0.9009 | 1951.8 |  |
| 11 | ContextMarkov | 0.6095 | 0.9170 | 1975.2 |  |
| 12 | RecencyFrequency | 0.5914 | 0.9298 | 1993.8 |  |
| 13 | Frequency | 0.5844 | 0.9296 | 1993.4 |  |
| 14 | LRU | 0.5838 | 0.9280 | 1991.2 |  |
| 15 | Random | 0.2372 | 0.7934 | 1795.4 |  |

## Dominated Baselines

- **Graph+Confidence** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)
- **Markov-2** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)
- **GlobalMarkov2** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)
- **VariableOrderMarkov** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)
- **RLAdaptiveEnsemble** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)
- **LFU** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)
- **ClusterMarkov** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)
- **ContextMarkov** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)
- **RecencyFrequency** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)
- **Frequency** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)
- **LRU** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)
- **Random** is dominated by **GraphMindRL** (F1, HR, Lat all ≤)

## Deployment Recommendation

**Keep GraphMindRL V3.** The V4 RLAdaptiveEnsemble does not achieve the minimum 2% F1 improvement threshold (-0.1254 absolute). The current V3 system (F1=0.7424, Lat=2002.5ms) represents the production-ready policy for the dashboard.

**Future work:** The new V4 models (VOM, ContextMarkov, ClusterMarkov) may be worth investigating individually in further ablations.
