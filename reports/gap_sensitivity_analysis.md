# Transition Gap Sensitivity Analysis

**Question:** What MAX_GAP threshold best captures meaningful app transitions?

**Method:** Train Markov-1 per user (80% split), evaluate on test split (10%).
Compare Hit Rate, F1, and Latency Saved across 15/30/60 minute thresholds.

---

## Transition Count Statistics

| Threshold | Median Transitions | Mean Unique Apps | Mean Graph Density |
|-----------|-------------------|------------------|-------------------|
| 15min (900s) | 4,085 | 88.1 | 0.0968 |
| 30min (1800s) | 4,411 | 88.8 | 0.0974 |
| 60min (3600s) | 4,675 | 89.2 | 0.0984 |

---

## Evaluation Metrics (Markov-1, mean across users)

| Threshold | Hit Rate | F1 | Latency Saved (ms) |
|-----------|----------|-----|-------------------|
| **15min** | 0.6179 ± 0.1442 | 0.6196 ± 0.1775 | 1817.9 |
| **30min** | 0.6159 ± 0.1509 | 0.6202 ± 0.1838 | 1812.2 |
| **60min** ✅ BEST| 0.6202 ± 0.1456 | 0.6231 ± 0.1806 | 1824.9 |

---

## Selected Threshold: **60min (3600s)**

**Rationale:**

- 60min achieves the best F1 score (mean 0.6231)
- Shorter thresholds (15min) miss valid transitions where the user pauses between apps
- Longer thresholds (60min) include stale context and inflate graph density artificially
- 60min best balances transition recall and precision

**All pipeline outputs use MAX_GAP = 3600s (60min)**
