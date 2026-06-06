# Combined Context Study (Phase 5)

**Date:** 2026-06-06  
**Question:** Does P(next | prev, current, time) beat P(next | current)?  
**Source:** `results/v5_combined_context.csv` (124 rows: 4 policies × 31 users)

---

## Results

| Policy | F1 | ΔF1 vs Baseline | ΔF1 vs M2_Naive |
|--------|-----|----------------|----------------|
| **Baseline (GraphMindRL)** | **0.7424** | — | +0.0129 |
| **M2_Naive** | **0.7295** | -0.0129 | baseline |
| JM_6Band | 0.7282 | -0.0142 | -0.0013 |
| JM_12Band | 0.7282 | -0.0142 | -0.0013 |
| JM_24Hour | 0.7282 | -0.0142 | -0.0013 |
| JM_48Bucket | 0.7282 | -0.0142 | -0.0013 |

---

## Central Question: Does P(next | prev, current, time) beat P(next | current)?

**Answer: NO.**

The combined JM+Time policy first tries JM-M2 (which uses `prev` and `cur`), then falls back to time-conditioned M1. The result (F1=0.7282) is worse than:
- JM-M2 alone (F1=0.7282 — same, time fallback never activated)
- Naive M2 (F1=0.7295)
- Baseline GraphMindRL (F1=0.7424)

**All four granularities produce identical F1=0.7282**, confirming that the time fallback path is effectively never activated — the JM-M2 path always returns predictions first, making the time-conditioned M1 fallback a dead code path in practice.

---

## Structural Finding: Additive Context Does Not Stack

The experiment design was:
1. JM-M2 predicts first (order-2 context)
2. If no predictions, fall back to time-conditioned M1

The problem: JM-M2 always produces predictions (even when the bigram is unseen, it uses the global frequency fallback λ₀×P(C)). Therefore, the time fallback is **never reached**.

Correct design for combined context: would need to use time as a feature INSIDE the JM formula, not as a fallback chain:

```python
# What was implemented:
P(C|A,B) → if empty → P(C|B,t)    ← time only reached if M2 empty

# What should be tested in V5:
P(C|A,B,t) = λ₂ × P(C|A,B,t) + λ₁ × P(C|B,t) + λ₀ × P(C|t)
              ← time integrated into EVERY interpolation level
```

---

## Conclusion

The hypothesis that "time + order-2 > order-1 alone" cannot be confirmed with the sequential fallback design. A proper test requires integrating time into the Markov conditioning key at every level.

**This is the key design challenge for V5:** time context needs to be embedded in the base prediction model, not bolted on as a fallback.

---
