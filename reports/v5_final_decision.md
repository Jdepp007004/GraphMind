# V5 Final Decision

**Date:** 2026-06-06
**Baseline:** GraphMindRL F1=0.7424  HR=0.9357  Lat=2002.5ms
**Previous best:** RL_LatencyFocus F1=0.7539 (p=0.0003, d=0.752)

---

## Phase E Results

| Policy | F1 | ΔF1 | HR | p | Cohen d | Sig | ≥+0.02 |
|--------|-----|-----|-----|---|---------|-----|--------|
| GraphMindRL_V5 | 0.7756 | +0.0332 | 0.9359 | 0.0108 | 0.496 | ✅ | ✅ |
| GraphMindRL_V5_t10 | 0.7744 | +0.0320 | 0.9382 | 0.0098 | 0.504 | ✅ | ✅ |
| RL_LatencyFocus | 0.7556 | +0.0132 | 0.9364 | 0.0003 | 0.754 | ✅ | ❌ |
| GraphMindRL_Base | 0.7546 | +0.0122 | 0.9382 | 0.0002 | 0.788 | ✅ | ❌ |

---

## Configuration

- **Best weights (Phase A):** trans=0.5  rec=0.1  freq=0.4
- **Best threshold (Phase B):** 0.16
- **Modified KN:** All variants underperform baseline (excluded from V5)

---

## Decision

**✅ RECOMMEND V5 PRODUCTION DEPLOYMENT**

**GraphMindRL_V5** achieves:
- F1 = 0.7756  (ΔF1 = +0.0332)
- p = 0.0108  Cohen d = 0.496
- Statistically significant, meets +0.02 threshold

**Configuration to promote to production:**
```python
confidence = 0.5*trans_prob + 0.1*recency + 0.4*frequency
threshold  = 0.16  # adaptive ±0.005 based on hit rate
budget     = 5
```