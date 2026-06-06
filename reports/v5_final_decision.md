# V5 Final Decision

**Date:** 2026-06-06
**Baseline:** GraphMindRL F1=0.7424  HR=0.9357  Lat=2002.5ms
**Previous best:** RL_LatencyFocus F1=0.7539 (p=0.0003, d=0.752)

---

## Phase E Results

| Policy | F1 | ΔF1 | HR | p | Cohen d | Sig | ≥+0.02 |
|--------|-----|-----|-----|---|---------|-----|--------|
| GraphMindRL_V5 | 0.7745 | +0.0321 | 0.9307 | 0.0115 | 0.491 | ✅ | ✅ |
| GraphMindRL_V5_t10 | 0.7733 | +0.0309 | 0.9326 | 0.0105 | 0.498 | ✅ | ✅ |
| RL_LatencyFocus | 0.7550 | +0.0126 | 0.9325 | 0.0004 | 0.725 | ✅ | ❌ |
| GraphMindRL_Base | 0.7539 | +0.0115 | 0.9344 | 0.0003 | 0.752 | ✅ | ❌ |

---

## Configuration

- **Best weights (Phase A):** trans=0.5  rec=0.1  freq=0.4
- **Best threshold (Phase B):** 0.16
- **Modified KN:** All variants underperform baseline (excluded from V5)

---

## Decision

**✅ RECOMMEND V5 PRODUCTION DEPLOYMENT**

**GraphMindRL_V5** achieves:
- F1 = 0.7745  (ΔF1 = +0.0321)
- p = 0.0115  Cohen d = 0.491
- Statistically significant, meets +0.02 threshold

**Configuration to promote to production:**
```python
confidence = 0.5*trans_prob + 0.1*recency + 0.4*frequency
threshold  = 0.16  # adaptive ±0.005 based on hit rate
budget     = 5
```