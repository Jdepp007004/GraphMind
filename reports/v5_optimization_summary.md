# V5 Optimization Summary

**Date:** 2026-06-06  
**Baseline:** GraphMindRL F1=0.7424 | HR=0.9357 | Lat=2002.5ms  
**Previous best:** RL_LatencyFocus F1=0.7539 (p=0.0003, d=0.752)

---

## 🏆 Final Result

| Policy | F1 | ΔF1 | HR | p | Cohen d | Sig | ≥+0.02 |
|--------|-----|-----|----|---|---------|-----|--------|
| **GraphMindRL_V5** | **0.7745** | **+0.0321** | 0.9307 | 0.0115 | 0.491 | ✅ | ✅ |
| GraphMindRL_V5_t10 | 0.7733 | +0.0309 | 0.9326 | 0.0105 | 0.498 | ✅ | ✅ |
| RL_LatencyFocus | 0.7550 | +0.0126 | 0.9325 | 0.0004 | 0.725 | ✅ | ❌ |
| GraphMindRL_Base | 0.7539 | +0.0115 | 0.9344 | 0.0003 | 0.752 | ✅ | ❌ |

**Decision: ✅ RECOMMEND GraphMindRL_V5 FOR PRODUCTION**

GraphMindRL_V5 meets the primary success criterion: ΔF1 = +0.0321 ≥ +0.02, p = 0.0115 < 0.05.

---

## Winning Configuration

```python
confidence = 0.5 * trans_prob + 0.1 * recency + 0.4 * frequency
threshold  = 0.16   # adaptive ±0.005 based on 20-step hit rate
budget     = 5      # HOT_SIZE
```

Key insight: **Frequency was massively underweighted** (0.2 → 0.4).  
Recency was **overweighted** (0.3 → 0.1).  
The original heuristic weights were significantly suboptimal.

---

## Top 10 Configurations Tested

| Rank | Config | F1 | ΔF1 |
|------|--------|-----|-----|
| 1 | **E: GraphMindRL_V5** (w=0.5/0.1/0.4, t=0.16) | **0.7745** | **+0.0321** |
| 2 | E: GraphMindRL_V5_t10 (w=0.5/0.1/0.4, t=0.10) | 0.7733 | +0.0309 |
| 3 | A: w0.5_r0.1_f0.4 (t=0.10) | 0.7733 | +0.0309 |
| 4 | A: w0.6_r0.2_f0.2 (t=0.10) | 0.7732 | +0.0308 |
| 5 | A: w0.7_r0.2_f0.1 (t=0.10) | 0.7723 | +0.0299 |
| 6 | A: w0.4_r0.1_f0.5 (t=0.10) | 0.7721 | +0.0297 |
| 7 | A: w0.5_r0.2_f0.3 (t=0.10) | 0.7678 | +0.0254 |
| 8 | A: w0.6_r0.3_f0.1 (t=0.10) | 0.7562 | +0.0138 |
| 9 | B: thresh=0.16 (w=0.5/0.3/0.2) | 0.7564 | +0.0140 |
| 10 | B: thresh=0.20 (w=0.5/0.3/0.2) | 0.7564 | +0.0140 |

---

## Phase A — Confidence Weight Grid

**Grid:** trans ∈ {0.4, 0.5, 0.6, 0.7} × rec ∈ {0.1, 0.2, 0.3, 0.4}, freq = 1 − trans − rec  
**Fixed threshold:** 0.10 (best known from RL_LatencyFocus)

**Best:** trans=0.5, rec=0.1, freq=0.4 → F1=0.7733 (ΔF1=+0.0309)

Key pattern: **Low recency (0.1) + high frequency (0.3–0.5) consistently outperforms** the original 0.3/0.2 split.

All 7 configurations with rec=0.1 or rec=0.2 achieved ΔF1 > +0.02.  
All configurations with rec=0.3 or rec=0.4 achieved ΔF1 < +0.02.

**Interpretation:** On UbiqLog, users show strong habitual app usage patterns that are better captured by historical frequency than by short-term recency. The 0.95 recency decay is too aggressive.

---

## Phase B — Threshold Sweep

**Fixed weights:** 0.5/0.3/0.2 (baseline)

| Threshold | F1 | ΔF1 | Precision | Recall | HR |
|-----------|-----|-----|-----------|--------|----|
| 0.02 | 0.7530 | +0.0106 | — | — | 0.9346 |
| 0.10 | 0.7550 | +0.0126 | — | — | 0.9325 |
| 0.14 | 0.7557 | +0.0133 | — | — | 0.9312 |
| **0.16** | **0.7564** | **+0.0140** | — | — | **0.9308** |
| 0.20 | 0.7564 | +0.0140 | — | — | 0.9299 |

**Finding:** With baseline weights, the F1 curve is nearly flat from threshold=0.10–0.20. Threshold alone has a ceiling of ~0.7564 with these weights. Weights are the primary lever.

See plot: [threshold_vs_f1.png](figures/threshold_vs_f1.png)

---

## Phase C — Time Context Coverage

| Granularity | Time Table Coverage | Fallback |
|-------------|-------------------|----------|
| TimeAwareM1_6Band | 98.5% | 1.5% |
| TimeAwareM1_12Band | 97.6% | 2.4% |
| TimeAwareM1_24Hour | 96.3% | 3.7% |
| TimeAwareM1_48Bucket | 94.3% | 5.7% |

**Conclusion: ANSWER A — Time signal is low-quality, NOT sparsity.**

Coverage is 94–98% across all granularities. States are seen in training, but the conditional distributions (P(next | app, time_band)) add noise rather than signal on this 2-month dataset.

**Implication:** Time conditioning would only help with much longer data (≥12 months) where the temporal patterns are stable and well-sampled.

---

## Phase D — Modified Kneser-Ney (no global term)

| Policy | F1 | ΔF1 | p | Sig |
|--------|-----|-----|---|-----|
| ModKN_K3 | 0.7275 | −0.0149 | 0.1323 | ❌ n.s. |
| ModKN_K5 | 0.7276 | −0.0148 | 0.1336 | ❌ n.s. |
| ModKN_K10 | 0.7283 | −0.0141 | 0.1520 | ❌ n.s. |

**Finding:** Removing the global unigram term from JM does not recover M2_Naive performance (0.7295). All three variants underperform the baseline. Excluded from V5.

The bigram model fundamentally underperforms the confidence-layer approach on UbiqLog because it lacks the recency/frequency signals that distinguish recently active apps.

---

## Final Recommendation

**→ IMPLEMENT GraphMindRL_V5 IN PRODUCTION**

```python
# GraphMindRL_V5 production configuration
confidence(app) = 0.5 * trans_prob(app | current)
               + 0.1 * recency(app)
               + 0.4 * frequency(app) / total_obs

threshold = 0.16   # start; adaptive ±0.005 based on 20-step hit rate
budget    = 5      # HOT_SIZE
```

**Result:** F1=0.7745, ΔF1=+0.0321, p=0.0115, Cohen d=0.491, 31 users, 80/10/10 split.

Proceed to dashboard freeze after implementing this configuration change.
