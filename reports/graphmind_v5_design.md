# GraphMind V5 — Architecture Design

**Date:** 2026-06-06  
**Basis:** Architecture audit findings, empirical Markov order analysis, time context analysis  
**Constraint:** No deep learning, no LightGBM/XGBoost, no LSTM/Transformer. Must preserve GraphMind architecture.

---

## 1. Design Principles

### What the audit proved

| Finding | V5 Response |
|---------|------------|
| GraphOnly == Markov-1 (F1 identical) | Make graph genuinely add value |
| Time_bucket unused in core policies | Build time into the primary predictor |
| Naive M2 hurts by −7.6pp (sparsity) | Use Jelinek-Mercer interpolation |
| ContextMarkov-1 gains +5.4pp | Make time context the default, not optional |
| RL reward doesn't optimize F1 | Redesign reward to target precision+recall |
| V4 REINFORCE underfits (15-dim linear) | Use a stronger state representation |
| BehaviouralGraph bypassed in benchmark | Align graph with evaluation pipeline |

### What to preserve
- Three-tier memory model (HOT/WARM/COLD)
- EventBus pub-sub architecture
- Personal per-user models (cross-user GlobalMarkov2 proved inferior)
- Graph as core data structure
- RL as adaptive component

---

## 2. V5 Architecture Overview

```
UbiqLog events
    │
    ▼
Time-Aware Transition Graph
  Node: (app, time_bucket)           ← V5: time baked into node identity
  Edge: count + temporal decay       ← V5: recency-weighted probabilities
    │
    ├─── JM-Interpolated Predictor   ← V5 primary predictor
    │      P(C|B,tb) = λ₂ × P(C|A,B,tb) + λ₁ × P(C|B,tb) + λ₀ × P(C)
    │
    ├─── Confidence Scorer           ← existing, time-bucket-aware
    │      conf = 0.5×trans + 0.2×rec + 0.2×freq + 0.1×ctx
    │
    └─── RL Precision Controller     ← V5: optimize precision, not cache budget
           State: 18 dims (context + predictor confidences + precision history)
           Action: confidence threshold (continuous, not discrete)
           Reward: F1-proxy = 2PR/(P+R)
    │
    ▼
Prefetch Decision
  HOT: top-3 by confidence
  WARM: top-12 above threshold
```

---

## 3. Component 1: Time-Aware Transition Graph

### V5 node identity

```python
# Current (BehaviouralGraph): node = (app_id, time_bucket, battery_bucket)
# Problem: battery unused in UbiqLog; node matching broken
# V5: node = (app_id, time_bucket)   ← drop battery
node_key = (app_id, time_bucket)   # 0-47 time buckets
```

Each `(app, time_bucket)` pair is a distinct graph node.

**Memory impact:** With ~50 apps/user × 48 buckets = 2,400 possible nodes. In practice ~200–400 occupied nodes per user (most apps don't span all time periods).

### V5 edge: temporal decay weight

**Problem with current edges:** `transition_prob += 0.01` per occurrence, normalized. This treats a 6-month-old transition identically to a yesterday's transition.

**V5 formula:**

```python
# Edge maintains: raw_count, last_seen_timestamp, base_prob
def get_decayed_prob(edge, current_ts, decay_halflife_days=14):
    days_elapsed = (current_ts - edge.last_seen_ts).total_seconds() / 86400
    decay = 0.5 ** (days_elapsed / decay_halflife_days)
    return edge.base_prob * decay + (1 - decay) * global_fallback_prob
```

**Decay half-life = 14 days:** After 2 weeks without a transition, probability halves. After 2 months (~4 half-lives), probability drops to 6% of original — effectively pruned.

This makes GraphMind adaptive to **behavioural drift** without retraining from scratch.

### V5 edge attributes

```python
@dataclass
class V5GraphEdge:
    source_key: Tuple[str, int]    # (app_id, time_bucket)
    target_key: Tuple[str, int]    # (app_id, time_bucket)
    raw_count: int                  # cumulative transition count
    base_prob: float               # normalized probability at last update
    last_seen_ts: datetime         # for temporal decay
    # REMOVED: battery_cost, time_sensitivity (both unused dead code)
```

---

## 4. Component 2: Jelinek-Mercer Interpolated Predictor

### Formula

```python
def predict_jm(cur: str, tb: int, prev: Optional[str] = None,
               k: int = 5) -> List[Tuple[str, float]]:
    """
    P(next | prev, cur, tb) = λ₂ × P(next|prev,cur,tb)
                             + λ₁ × P(next|cur,tb)
                             + λ₀ × P(next)
    
    λ₂ = count(prev,cur,tb) / (count(prev,cur,tb) + K)  K=5
    λ₁ = (1 - λ₂) × count(cur,tb) / (count(cur,tb) + K)
    λ₀ = 1 - λ₂ - λ₁
    """
    K = 5  # smoothing constant

    # Bigram count
    n2 = count_table.get((prev, cur, coarse_tb), 0) if prev else 0
    λ₂ = n2 / (n2 + K)

    # Unigram count with time bucket
    n1 = count_table.get((cur, coarse_tb), 0)
    λ₁ = (1 - λ₂) * n1 / (n1 + K)

    # Global frequency
    λ₀ = 1 - λ₂ - λ₁

    # Merge scores
    scores = {}
    for app, p in P2.get((prev, cur, coarse_tb), {}).items():
        scores[app] = scores.get(app, 0) + λ₂ * p
    for app, p in P1.get((cur, coarse_tb), {}).items():
        scores[app] = scores.get(app, 0) + λ₁ * p
    for app, p in global_freq.items():
        scores[app] = scores.get(app, 0) + λ₀ * p

    return sorted(scores.items(), key=lambda x: -x[1])[:k]
```

**Why this works:** λ₂ is data-adaptive. When the bigram has 20+ occurrences, λ₂ → 0.8 (order-2 dominates). When the bigram has 0 occurrences, λ₂ = 0 (falls back to time-conditioned M1). This solves the sparsity problem that caused naive M2 to underperform.

### Coarse vs fine time buckets

Use **coarse time buckets** (time_bucket // 8 → 6 bands) for the conditioning key:

| Coarse bucket | Hours | Label |
|---------------|-------|-------|
| 0 | 0:00–3:59 | Late night |
| 1 | 4:00–7:59 | Early morning |
| 2 | 8:00–11:59 | Morning |
| 3 | 12:00–15:59 | Afternoon |
| 4 | 16:00–19:59 | Evening |
| 5 | 20:00–23:59 | Night |

This gives 6× more data per bucket vs 48-bucket conditioning, while preserving the major behavioral distinctions.

---

## 5. Component 3: RL Precision Controller

### V5 RL role: Threshold Optimizer

**What V3 did:** Adjust HOT/WARM cache budget size (useless — F1=0.742, same as Graph+Confidence)  
**What V4 tried:** Weight 5 predictors (failed — underfitted with 15-dim linear, 3 passes)  
**What V5 should do:** Control the **confidence threshold** for prefetch dynamically

**Rationale:** The primary difference between GraphMindRL (F1=0.742) and Graph+Confidence (F1=0.741) is effectively the threshold: GraphMindRL adaptively adjusts `self._thresh` based on hit history. This is the ONLY mechanism that works.

V5 should formalize this into a proper RL problem with a continuous action and an F1-proxy reward.

### V5 Action Space

```
Action: threshold ∈ [0.05, 0.95]  (single continuous value)

Interpretation:
  high threshold → fewer predictions, higher precision, lower recall
  low threshold  → more predictions, higher recall, lower precision
  RL learns: when to be aggressive vs conservative
```

A single continuous threshold is:
- Tractable for a small linear policy
- Directly interpretable
- Provably impactful (varies F1 significantly across [0.05, 0.95])

### V5 State Space (18 dimensions)

```
[0]     current_app_hash_norm         # vocab index / vocab_size
[1]     prev_app_hash_norm
[2]     time_bucket_norm              # time_bucket / 47
[3]     weekday_norm                  # day_of_week / 6
[4]     is_weekend                    # binary
[5]     transition_entropy_norm       # local entropy estimate / 4.0
[6]     predictor_confidence          # JM top-1 probability
[7]     app_frequency_norm            # freq(cur_app) / total_events
[8]     app_recency_norm              # recency score (exp decay)
[9]     graph_out_degree_norm         # out_edges(cur_app) / max_out_degree
[10]    recent_hit_rate               # hits/total last 20 events
[11]    recent_precision              # tp/(tp+fp) last 20 events
[12]    recent_recall                 # tp/(tp+fn) last 20 events
[13]    recent_false_prefetch_rate    # fp/(total predictions) last 20
[14:18] hit history (4 binary)        # last 4 outcomes
```

**Key additions over V4:**
- `recent_precision` and `recent_recall` (obs[11:13]) — RL can now see the precision/recall tradeoff directly
- `graph_out_degree_norm` (obs[9]) — high out-degree = uncertain = lower threshold preferred
- `is_weekend` (obs[4]) — weekend patterns differ

### V5 Reward Function

**Design goal:** Optimize F1, not just hit_rate.

```python
def reward_v5(tp, fp, fn, latency_saved_ms):
    """
    F1-proxy reward with latency bonus.
    
    precision = tp / (tp + fp)   ← penalizes false prefetches
    recall    = tp / (tp + fn)   ← rewards catching the actual next app
    f1_proxy  = 2 * P * R / (P + R)
    """
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    f1 = 2 * P * R / max(P + R, 1e-9)

    latency_bonus = min(1.0, latency_saved_ms / 2489.0) * 0.3

    return f1 + latency_bonus
```

**Reward range:** [0, 1.3] per step. F1 dominates (0–1.0), latency is a secondary bonus.

**Why this is better than current reward:**
- Current V3: `2.0×hit_rate` — ignores false prefetches unless they cause thrash
- V5: F1 directly penalizes both false prefetches (precision) and misses (recall)
- F1 reward aligns RL objective with benchmark metric

### Recommended coefficients from benchmark data

```
reward = 1.0 × f1_proxy + 0.3 × latency_bonus − 0.0 × memory_cost
```

**Memory cost = 0:** UbiqLog phones had no memory pressure in the dataset. Adding an artificial memory cost penalty would penalize the RL for doing its job without grounding.

**Literature support (from reward function analysis):**
- Ran et al. (2012): precision penalty more important than recall penalty for mobile prefetch
- Tossell et al. (2012): users notice false prefetches (battery drain) more than missed prefetches
- Both support higher weight on precision, hence f1 > hit_rate as reward

---

## 6. Expected V5 Impact Estimates

### A. Time buckets only (CtxM1 replacing M1)

**Evidence:** Measured +5.37pp hit rate gain on 31 users.  
**Expected F1 gain:** +3 to +5 pp (F1 < HR gain because precision may slightly drop)  
**Source:** Phase 5 empirical measurement  

```
Current best: GraphMindRL F1 = 0.742
Estimated:    CtxM1-only  F1 ≈ 0.775–0.792
```

### B. Order-2 with Jelinek-Mercer (JM-M2 replacing naive M2)

**Evidence:** Naive M2 lost −7.6pp. Literature shows JM-smoothed M2 gains +1–3pp over M1.  
**Expected F1 gain over CtxM1:** +1 to +3 pp  
**Sources:** Lu (2013), McInerney (2013), Kjaergaard (2012)  

```
Estimated:    CtxM1-only  F1 ≈ 0.775–0.792
Estimated:    CtxM1 + JM-M2 F1 ≈ 0.785–0.810
```

### C. Time-aware JM predictor + temporal decay

**Evidence:** Temporal decay ensures the model tracks drift. No direct measurement available for decay in this dataset.  
**Conservative estimate:** +0.5 to +1.5 pp additional over B.  
**Expected F1:** ≈ 0.790–0.820

### D. Time-aware JM predictor + V5 RL (F1 reward, threshold control)

**Evidence:** V3 RL (cache allocator) delivered only +0.1pp over Graph+Confidence.  
V5 RL has a stronger, aligned reward signal and a smaller, more tractable action space.  
**Conservative RL gain estimate:** +1 to +3 pp over C alone.  
**Expected F1:** ≈ 0.795–0.840

### Summary table

| Variant | Expected F1 | Gain vs Current | Confidence |
|---------|------------|----------------|------------|
| Current best (GraphMindRL V3) | 0.742 | baseline | Measured |
| A: Time context only | 0.775–0.792 | +3.3–5.0 pp | High (measured on data) |
| B: + JM-M2 | 0.785–0.810 | +4.3–6.8 pp | Medium (literature) |
| C: + Temporal decay | 0.790–0.820 | +4.8–7.8 pp | Medium (estimated) |
| D: + V5 RL (F1 reward) | 0.795–0.840 | +5.3–9.8 pp | Medium-Low |

**Best case:** F1 = 0.840 (13pp gain)  
**Conservative estimate:** F1 = 0.795 (5.3pp gain)  
**Minimum threshold for upgrade:** F1 ≥ 0.762 (2pp over current, statistically significant)

---

## 7. V5 Implementation Roadmap

### Priority order (by expected ROI)

**P0 — Highest ROI, low risk:**

1. **Time-conditioned predictor** (`CtxM1` with 6-coarse-bucket conditioning)
   - Files to modify: `scripts/run_benchmark_v4.py` (add a `TimeAwareMarkov1Policy`)
   - Files to modify: replace `GraphConfidencePolicy` and `GraphMindRLPolicy` with time-aware variants
   - Estimated effort: 2 hours

**P1 — High ROI, medium complexity:**

2. **Jelinek-Mercer interpolation** (replaces naive M2)
   - New file: `src/models/jm_interpolated.py`
   - Parameters: K=5 (smoothing), coarse_bucket_size=8
   - Estimated effort: 3 hours

3. **V5 reward function** (`f1_proxy + latency_bonus`)
   - Modify: `src/rl/reward_v2.py` (add `RewardV5` class)
   - Estimated effort: 1 hour

**P2 — Medium ROI, medium complexity:**

4. **Temporal edge decay** on BehaviouralGraph
   - Modify: `src/core/graph_engine.py` (add `last_seen_ts` and decay formula)
   - Align with benchmark pipeline
   - Estimated effort: 4 hours

5. **V5 RL state** (add precision/recall history, graph out-degree)
   - Modify: `src/rl/environment_v2.py` → new `environment_v3.py`
   - Estimated effort: 3 hours

**P3 — Lower ROI (verify only after P0-P2 show gains):**

6. **Continuous threshold RL** (V5 action space redesign)
   - Depends on whether SB3 PPO handles continuous action well
   - Alternative: tabular Q-learning over 20 discrete thresholds
   - Estimated effort: 5 hours

---

## 8. What NOT to Do

| Idea | Reason to avoid |
|------|----------------|
| Markov-3 | Sparsity causes consistent degradation |
| 48-bucket time conditioning | Too sparse; 6-band coarse bucketing is better |
| Cross-user transfer | GlobalMarkov2 F1=0.679 vs personal F1=0.742 (+6.3pp for personal) |
| LSTM / RNN | Violates project constraint; overkill for structured Markov prediction |
| Cluster-level pooling | ClusterMarkov F1=0.612 — pooling hurts |
| More REINFORCE passes | 10+ passes still won't fix 15-dim linear over 500-app vocab |

---

## 9. Design Decisions Summary

| Decision | Choice | Rationale |
|---------|--------|-----------|
| Time granularity | 6 coarse bands (4-hour each) | Balance between data density and temporal resolution |
| Smoothing | Jelinek-Mercer K=5 | Data-adaptive λ, no manual tuning |
| Order | 1+2 interpolated (no 3) | M3 always hurts; JM-M2 recovers M2 without sparsity penalty |
| RL role | Threshold controller | Most impactful lever; tractable 1-D continuous action |
| RL reward | F1-proxy + latency bonus | Aligned with benchmark metric |
| RL state | 18-dim (adds precision/recall history) | Gives RL the signals it needs to balance P/R |
| Edge decay | 14-day half-life | Tracks drift without aggressive forgetting |

---
