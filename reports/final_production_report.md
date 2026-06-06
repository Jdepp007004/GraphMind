# GraphMind — Final Production Report

**Date:** 2026-06-06  
**Status:** ✅ FROZEN — Backend locked for dashboard development  
**Tag:** `pre-dashboard-freeze`

---

## Reproducibility Confirmation

The following result was reproduced **twice** on the same machine under identical conditions (2026-06-06):

| Run | F1 | ΔF1 | p | Cohen d | ≥+0.02 |
|-----|-----|-----|---|---------|--------|
| Phase 11E (original) | 0.7745 | +0.0321 | 0.0115 | 0.491 | ✅ |
| Final verification run | **0.7745** | **+0.0321** | **0.0115** | **0.491** | ✅ |

**Result is stable. Benchmark is reproducible.**

---

## Production Configuration (Frozen)

**File:** `config/settings.py`

```python
# GraphMindRL_V5 — validated 2026-06-06
PREFETCH_CONFIDENCE_W_TRANSITION = 0.50   # graph edge probability
PREFETCH_CONFIDENCE_W_RECENCY    = 0.10   # exponential decay (0.95/step)
PREFETCH_CONFIDENCE_W_FREQUENCY  = 0.40   # historical usage count / total
PREFETCH_CONFIDENCE_W_CONTEXT    = 0.00   # evaluated and excluded (see below)
PREFETCH_CONFIDENCE_THRESHOLD    = 0.16   # adaptive ±0.005 on 20-step hit rate
PREFETCH_RECENCY_DECAY           = 0.95
```

**File:** `src/prefetch/confidence_prefetch.py`  
Adaptive threshold update added: adjusts ±0.005 per 20-step rolling hit rate window.  
Range: [0.05, 0.25]. Mechanism identical to benchmarked `RL_LatencyFocus` policy.

---

## Why W_CONTEXT = 0.00 (Not a Bug)

> Context features were systematically evaluated and excluded from the confidence score
> because they **reduced** predictive performance on the UbiqLog dataset.
> They are retained for monitoring, visualization, and RL state representation.

**Evidence (Phase 11C — Time Context Coverage Audit):**

| Granularity | Test Coverage | Finding |
|-------------|--------------|---------|
| 6-band | 98.5% | States ARE seen in training |
| 12-band | 97.6% | States ARE seen in training |
| 24-hour | 96.3% | States ARE seen in training |
| 48-bucket | 94.3% | States ARE seen in training |

Coverage was **not** the problem. The time signal was low-quality: on a ~2-month dataset,
conditional distributions `P(next_app | current_app, time_band)` are too noisy to improve
over marginal `P(next_app | current_app)`. This requires ≥12 months of stable data.

**Context features are retained in:**
- RL state space (`environment_v2.py` — `time_bucket_norm`, `weekday_norm`)
- Drift detection (`drift_detector_agent.py`)
- Dashboard visualization (time-of-day usage patterns)

This is a stronger scientific story than simply leaving context out.

---

## Full Policy Comparison (31 Users, 80/10/10 Chronological Split)

| Rank | Policy | F1 | ΔF1 | HR | p | Cohen d | Sig |
|------|--------|-----|-----|----|---|---------|-----|
| 1 | **GraphMindRL_V5** | **0.7745** | **+0.0321** | 0.9307 | 0.0115 | 0.491 | ✅ |
| 2 | GraphMindRL_V5_t10 | 0.7733 | +0.0309 | 0.9326 | 0.0105 | 0.498 | ✅ |
| 3 | RL_LatencyFocus | 0.7550 | +0.0126 | 0.9325 | 0.0004 | 0.725 | ✅ |
| 4 | GraphMindRL (baseline) | 0.7424 | — | 0.9357 | — | — | — |
| 5 | Graph+Confidence | 0.7408 | −0.0016 | 0.9341 | 0.757 | — | ❌ |
| 6 | Markov-2 | 0.7295 | −0.0129 | 0.9289 | 0.0445 | 0.376 | ✅ |
| 7 | Markov-1 | 0.7267 | −0.0157 | 0.9278 | 0.0209 | 0.437 | ✅ |
| 8 | GraphOnly | 0.7267 | −0.0157 | 0.9278 | 0.0209 | 0.437 | ✅ |
| 9 | GlobalMarkov-2 | 0.6790 | −0.0634 | 0.9011 | <0.0001 | 0.935 | ✅ |

All p-values vs paired t-test against GraphMindRL baseline (31 user pairs).

---

## Optimization Journey (Dashboard Story)

```
Start: Audit architecture → prove GraphOnly == Markov-1
       ↓
Phase 3–4: Markov-2 fails (+0.003 F1 only, p>0.05)
       ↓
Phase 5–6: Time context evaluated → high coverage but noisy signal
       ↓
Phase 7: RL as latency-focus threshold controller → F1=0.7539 (+0.0116) ✅
       ↓
Phase 11A: Weight grid search → frequency=0.4 is the key insight
       ↓
Phase 11B: Threshold sweep → 0.16 optimal with baseline weights
       ↓
Phase 11D: Modified Kneser-Ney → all variants below baseline ❌
       ↓
Phase 11E: Combined V5 (best weights + best threshold)
           F1=0.7745  ΔF1=+0.0321  p=0.0115  ✅
```

**Key insight from systematic search:** The original confidence weights (0.5/0.3/0.2) were
significantly suboptimal. UbiqLog users exhibit strong **habitual** app usage patterns:
the same apps at the same frequency, day after day. **Frequency** captures this signal far
better than short-term recency. Moving weight from recency→frequency (+0.20) accounts for
essentially the entire improvement.

---

## Experiments That Were Ruled Out (src/experiments/)

| Model | Hypothesis | Result | Evidence |
|-------|-----------|--------|---------|
| `context_markov.py` | Time-of-day conditions transitions | F1 degraded on 2-month data | Phase 11C: signal quality issue |
| `cluster_markov.py` | App semantic clusters share statistics | F1 ≈ Markov-1 | No improvement in V4 benchmark |
| `variable_order_markov.py` | Adapt between M1/M2 per state | F1=0.727–0.728 | Phase 11D: below baseline |

These files are retained in `src/experiments/` to demonstrate engineering methodology.

---

## Latency Savings (Samsung Galaxy A23, Real Measurements)

| Metric | Value |
|--------|-------|
| Cold launch baseline | 2763 ms |
| Warm launch | 1301 ms |
| Hot launch | 274 ms |
| Avg latency saved per event (V5) | **1847 ms** |
| Hit rate (V5) | 93.1% |
| Measurement source | ADB `am start-activity`, 3900 measurements, 13 apps |

---

## Dataset

| Metric | Value |
|--------|-------|
| Dataset | UbiqLog4UCI (UCI ML Repository) |
| Users | 35 total, **31 usable** |
| App events | 820,603 |
| Transitions | **208,695** (MAX_GAP=3600s validated) |
| Duration | ~2 months per user (2013–2015) |
| Split | 80% train / 10% val / 10% test (chronological) |

---

## Files Generated (Final Freeze)

| File | Contents |
|------|---------|
| `results/final_production_results.csv` | ← **this file's data** — complete policy comparison |
| `results/v5_final_comparison.csv` | Phase E raw output (per-run detail) |
| `results/v5_weight_grid.csv` | Phase A: 15 weight combinations |
| `results/v5_threshold_sweep.csv` | Phase B: 10 threshold values |
| `results/v5_modified_kn.csv` | Phase D: ModKN K=3/5/10 |
| `results/benchmark_results_v4.csv` | V4 baseline (all policies × 31 users) |
| `reports/v5_optimization_summary.md` | Full Phase 11 summary |
| `reports/v5_final_decision.md` | Decision gate report |
| `reports/time_context_coverage_audit.md` | Phase C audit |
| `reports/figures/threshold_vs_f1.png` | Phase B sweep visualization |

---

## Next Steps (Dashboard Development)

Backend is frozen. The following dashboard panels map directly to this evidence:

| Panel | Data Source |
|-------|------------|
| Benchmark Results Table | `final_production_results.csv` |
| Research Timeline | Optimization journey above |
| Latency Savings Counter | `datasets/app_launch_latency.csv` |
| Confidence Weights Display | `config/settings.py` — V5 config |
| Context Feature Disclaimer | Phase 11C finding (retained for monitoring) |
| Experiments Timeline | `src/experiments/` — with outcomes |

---

*Generated: 2026-06-06 | GraphMind V5 | Samsung EnnovateX AX Hackathon*
