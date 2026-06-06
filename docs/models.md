# Model Catalogue

> **GraphMindRL V5 — All Models Tested and Evaluated**

---

## Overview

This document catalogues every model variant evaluated during the development of GraphMindRL V5. Nine distinct policies were benchmarked. Three were accepted into the production configuration. The rest were rejected based on empirical evidence.

**Evaluation protocol**: 31 users, 80/10/10 chronological split, paired t-test vs. GraphMindRL Baseline, acceptance requires p < 0.05 and Cohen's d > 0.2.

---

## Model Results Summary

| Model | F1 | ΔF1 vs Baseline | p-value | Cohen's d | Status |
|---|---|---|---|---|---|
| **GraphMindRL_V5** | **0.7745** | **+0.0321** | **0.0115** | **0.491** | **🟢 PRODUCTION** |
| GraphMindRL_V5 (t=0.10) | 0.7733 | +0.0309 | 0.0105 | 0.498 | 🔵 Candidate |
| RL_LatencyFocus | 0.7539 | +0.0116 | 0.0003 | 0.752 | 🔵 Candidate |
| GraphMindRL Baseline | 0.7424 | 0.0000 | — | — | ⚪ Reference |
| Graph+Confidence | 0.7369 | −0.0055 | 0.3421 | 0.187 | 🔴 Rejected |
| Markov-2 | 0.7355 | −0.0069 | 0.2891 | 0.203 | 🔴 Rejected |
| Markov-1 / GraphOnly | 0.7267 | −0.0157 | 0.1123 | 0.291 | ⚪ Baseline |
| Modified KN | 0.7421 | −0.0003 | 0.9400 | 0.010 | 🔴 Rejected |
| GlobalMarkov-2 | 0.7201 | −0.0223 | 0.0731 | 0.342 | 🔴 Rejected |

---

## Model 1: Markov-1 (GraphOnly)

### Purpose

The foundational baseline. Captures only first-order sequential dependency between apps.

### Formula

```
P(next | current) = count(current → next) / Σ_x count(current → x)

Prefetch: top-k apps by P(next | current)
```

### Architecture

- Per-user transition probability table.
- For each source app A, the successors are ranked by transition count.
- Top HOT_CACHE_SIZE + WARM_CACHE_SIZE apps are prefetched.
- No threshold (top-k selection).

### Result

| F1 | Hit Rate | Notes |
|---|---|---|
| 0.7267 | 92.4% | Starting baseline |

### Decision

**Kept as baseline.** This is the simplest possible approach and the anchor for all comparison statistics.

---

## Model 2: Markov-2

### Purpose

Extend Markov-1 to second-order: condition on the last *two* apps rather than just the current app.

### Formula

```
P(next | current, previous) = count(previous → current → next) / Σ_x count(previous → current → x)
```

### Architecture

- Per-user second-order transition table.
- Requires substantially more data per (previous, current) pair to estimate reliably.
- Falls back to Markov-1 when the (previous, current) pair is unseen.

### Result

| F1 | Hit Rate | p-value | Status |
|---|---|---|---|
| 0.7355 | 91.4% | 0.289 | **REJECTED** |

### Decision

**Rejected.** p = 0.289 does not meet the significance threshold. The UbiqLog sessions are too short (~2 months per user) for reliable second-order estimation. Most (previous, current) pairs are seen only once or twice in training, making the probability estimates unreliable.

**Archived in**: `archive/old_results/`

---

## Model 3: GlobalMarkov-2

### Purpose

Learn transition probabilities from all users combined rather than per-user. This pool of cross-user data makes the second-order estimates more reliable.

### Formula

```
P(next | current) = Σ_u count_u(current → next) / Σ_u Σ_x count_u(current → x)
```

### Architecture

- Single global transition table trained on transitions from all 31 users.
- Applied to each test user's sequence.

### Result

| F1 | Hit Rate | p-value | Status |
|---|---|---|---|
| 0.7201 | 91.1% | 0.073 | **REJECTED** |

### Decision

**Rejected.** Global pooling hurts F1. App usage patterns are highly personalised — pooling across users introduces noise that outweighs the benefit of more data. This confirms that per-user models are essential.

---

## Model 4: Graph+Confidence

### Purpose

Augment Markov-1 transition probabilities with recency and frequency signals, using a static confidence threshold.

### Formula

```
score(app) = 0.5 × P(app | current)
           + 0.2 × recency_score(app)
           + 0.2 × freq_score(app)
           + 0.1 × context_score(app)

threshold = static (not adaptive)
```

### Architecture

- Per-user Markov graph with confidence scoring.
- Weights: 0.5/0.2/0.2/0.1 (default, not optimised).
- Fixed threshold (tuned once on the validation set).

### Result

| F1 | Hit Rate | p-value | Status |
|---|---|---|---|
| 0.7369 | 91.8% | 0.342 | **REJECTED** (vs baseline) |

### Decision

**Rejected** relative to GraphMindRL Baseline. The confidence scoring idea is sound, but the weights are sub-optimal and the static threshold does not self-calibrate. This model motivated the subsequent weight grid search (Phase 11A) and the RL threshold controller development.

---

## Model 5: GraphMindRL Baseline

### Purpose

Add an RL-based adaptive threshold controller to the confidence scorer.

### Formula

```
score(app) = 0.5 × P(app | current)
           + 0.2 × recency_score(app)
           + 0.2 × freq_score(app)
           + 0.1 × context_score(app)

threshold_t = adaptive via RL:
  if rolling_hit_rate > 0.80: threshold += 0.005
  if rolling_hit_rate < 0.50: threshold -= 0.005
```

### Architecture

- All components of Graph+Confidence.
- Adaptive threshold controller (20-step rolling window, ±0.005 adjustment).
- Initial threshold = 0.16.

### Result

| F1 | Hit Rate | ΔF1 vs Markov-1 | p-value | Status |
|---|---|---|---|---|
| 0.7424 | 93.6% | +0.0157 | 0.023 | **ACCEPTED — REFERENCE** |

### Decision

**Accepted as reference baseline.** The RL threshold controller is a valuable addition. This model becomes the reference point for all subsequent Phase 11 experiments.

---

## Model 6: Modified Kneser-Ney (Modified KN)

### Purpose

Apply modified Kneser-Ney smoothing to the transition probability estimates to handle unseen transitions.

### Formula

```
P_KN(next | current) = max(count(current → next) - d, 0) / count(current)
                     + λ(current) × P_continuation(next)

where:
  d = discount constant (optimised on validation set)
  λ(current) = d × out_degree(current) / count(current)
  P_continuation(next) = |{A : A→next seen}| / |{(A, B) : A→B seen}|
```

### Architecture

- Standard first-order Markov backbone.
- Discount constant d and back-off weights tuned on the validation set.

### Result

| F1 | Hit Rate | ΔF1 vs Baseline | p-value | Status |
|---|---|---|---|---|
| 0.7421 | 93.4% | −0.0003 | 0.940 | **REJECTED** |

### Decision

**Rejected.** KN smoothing provides virtually zero improvement (ΔF1 = −0.0003). The training data is sufficient to estimate the common transitions reliably; there is no meaningful data sparsity problem for this dataset. KN smoothing is most useful when many transitions are unseen — here, nearly all common transitions appear in training.

---

## Model 7: RL_LatencyFocus

### Purpose

Explore an alternative weight configuration that maximises latency savings (rather than F1).

### Formula

```
score(app) = 0.6 × P(app | current)
           + 0.1 × recency_score(app)
           + 0.3 × freq_score(app)

threshold = adaptive (RL controller)
```

### Architecture

- Same as GraphMindRL Baseline but with weights (0.6/0.1/0.3/0.0).
- Prioritises high-confidence predictions (higher transition weight) to minimise false positives.

### Result

| F1 | Hit Rate | ΔF1 vs Baseline | p-value | Cohen's d | Status |
|---|---|---|---|---|---|
| 0.7539 | 90.7% | +0.0116 | 0.0003 | 0.752 | **SIGNIFICANT** |

### Decision

**Significant improvement**, but ultimately superseded by GraphMindRL_V5 (F1 = 0.7745 > 0.7539). Archived as the best candidate before Phase 11A weight optimisation.

Note: Cohen's d = 0.752 (large effect) reflects that this variant is very consistent in its improvement across users, even though the absolute ΔF1 is smaller than V5.

---

## Model 8: Time-Aware Variants

### Purpose

Add time-of-day and day-of-week features to capture daily behavioural rhythms.

### Formula

```
context_score(app, time_band) = P(app | time_band)
                               × W_CONTEXT

W_CONTEXT tested: {0.05, 0.10, 0.15, 0.20}
Time granularities tested: {6-band, 12-band, 24-hour, 48-bucket}
```

### Architecture

Four variants tested:
- **6-band**: Day divided into six 4-hour slots
- **12-band**: Day divided into twelve 2-hour slots
- **24-hour**: Standard hourly granularity
- **48-bucket**: Day divided into 48 30-minute slots

### Results

| Variant | Coverage | F1 | ΔF1 vs Baseline | Status |
|---|---|---|---|---|
| 6-band | 98.5% | 0.7301 | −0.0123 | **REJECTED** |
| 12-band | 97.6% | 0.7318 | −0.0106 | **REJECTED** |
| 24-hour | 96.3% | 0.7389 | −0.0035 | **REJECTED** |
| 48-bucket | 94.3% | 0.7352 | −0.0072 | **REJECTED** |

### Decision

**All variants rejected.** Time-context features hurt F1 at all granularities. The conditional distributions P(next | app, time_band) are too noisy on 2-month datasets. The data is not the issue (coverage 94–98%); the problem is that the distributions have not converged with 2 months of data. Context weight set to `W_CONTEXT = 0.00` in production.

**Scientific note**: Context features are retained in the RL state representation for monitoring purposes. The zeroing of W_CONTEXT applies only to the confidence scoring function, not the full system state.

**Archived in**: `archive/`

---

## Model 9: GraphMindRL_V5 (Production)

### Purpose

Combine the RL adaptive threshold controller with optimised confidence weights found by Phase 11A grid search and Phase 11B threshold sweep.

### Formula (Production — Frozen)

```
score(app) = W_TRANSITION × P(app | current)
           + W_RECENCY    × recency_score(app)
           + W_FREQUENCY  × freq_score(app)
           + W_CONTEXT    × context_score(app)

W_TRANSITION = 0.50
W_RECENCY    = 0.10   (↓ from 0.20 — recency is noisy relative to frequency)
W_FREQUENCY  = 0.40   (↑ from 0.20 — frequency is the strongest non-sequential signal)
W_CONTEXT    = 0.00   (zeroed — noisy on short datasets)

Initial threshold = 0.16 (Phase 11B optimal)
Adaptive: threshold ± 0.005 per 20-step window

recency_score(app) = exp(-λ × Δt)             λ = 0.0001
freq_score(app)    = count(app) / max_count    normalised by most frequent app
```

### Architecture

- Per-user weighted directed Markov graph (NetworkX DiGraph).
- Confidence scorer with four signal components.
- Adaptive threshold RL controller (20-step rolling window).
- HOT cache (5 slots, LRU).
- WARM cache (15 slots, score-ranked).
- COLD store (SQLite, unlimited).

### Result

| F1 | Hit Rate | Latency | ΔF1 vs Baseline | p | d | Reproduced |
|---|---|---|---|---|---|---|
| **0.7745** | **93.1%** | **1,847ms** | **+0.0321** | **0.0115** | **0.491** | **✓ ×2** |

### Decision

**PRODUCTION MODEL.** All acceptance criteria met. Result reproduced on two independent runs. Configuration frozen. This is the official submission model.

**Files**:
- Implementation: `src/prefetch/confidence_prefetch.py`
- Configuration: `config/settings.py`
- Benchmark: `scripts/run_phase11_e.py`
- Result: `results/final_production_results.csv`

---

## Why W_FREQUENCY = 0.40 Won

The most important finding from the weight grid search was that **frequency substantially outperforms recency** as a secondary signal for app prefetching.

**Intuition**: People's app usage habits are highly repetitive and stable. The apps a user opens most often are the apps they are most likely to open next — regardless of when they last opened them. Recency adds noise because it changes every step, while frequency is a stable signal.

**Evidence**: Setting W_RECENCY = 0.10 and W_FREQUENCY = 0.40 improved F1 by +0.0309 over the default configuration (0.2/0.2). This was the largest single improvement found in the entire project.

---

## Ablation Study

To confirm that each component contributes positively, we performed an ablation study removing one component at a time:

| Configuration | F1 | ΔF1 vs V5 | Note |
|---|---|---|---|
| **Full V5 (0.5/0.1/0.4/0.0 + RL)** | **0.7745** | — | Production |
| No RL (static threshold) | 0.7661 | −0.0084 | RL controller needed |
| No frequency (0.5/0.5/0.0/0.0) | 0.7589 | −0.0156 | Frequency critical |
| No recency (0.5/0.0/0.5/0.0) | 0.7703 | −0.0042 | Recency helpful |
| Transition only (1.0/0.0/0.0/0.0) | 0.7267 | −0.0478 | All signals needed |

**Conclusion**: All three active components (transition, recency, frequency) and the RL controller are necessary. Removing any one of them degrades performance.

---

*All model results are stored in `results/v5_all_experiments.csv`. Individual phase CSVs are in `results/`. Failed model code is archived in `archive/`.*
