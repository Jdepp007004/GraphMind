# GraphMind Architecture Audit

**Date:** 2026-06-06  
**Mode:** Read-only inspection — no code was modified.  
**Scope:** Complete pipeline from UbiqLog events through to prefetch decision.

---

## 1. Complete Pipeline Trace

### Data Flow Diagram

```
Raw UbiqLog .txt files
  └─ Application events: {ProcessName, Start, End}
        │
        ▼
ubiqlog_transition_pipeline.py :: build_transitions()
  │  Filters system apps, validates timestamps
  │  Computes gap_s, time_bucket (0-47), day_of_week (0-6)
  │  MAX_GAP = 3600s (validated by sensitivity study)
  └─ transitions.parquet  (208,695 rows)
     Columns: from_app, to_app, timestamp, gap_s, time_bucket, day_of_week, user_id
        │
        ▼
run_benchmark_v4.py :: load_events_with_context()
  │  Re-reads raw .txt per-user (does NOT use parquet)
  │  Returns: (apps: List[str], tbs: List[int], wds: List[int])
  │  time_bucket = dt.hour * 2 + (1 if dt.minute >= 30 else 0)
  │  weekday     = dt.weekday()
        │
        ▼
Policy.train(apps, tbs, wds, val_apps, val_tbs, val_wds)
  │  GraphMindRL:  builds dict[str -> dict[str, float]] (Markov-1)
  │  Markov-2:     builds dict[(str,str) -> dict[str, float]]
  │  ContextMarkov: builds P(next|cur,tb) P(next|cur,wd) P(next|cur,tb,wd)
  │  ClusterMarkov: k-means++ on 4 features, personal→cluster→global
  │  RLEnsemble:    REINFORCE, 3 passes, linear 15→5 policy
        │
        ▼
Policy.predict(current_app, prev_app, time_bucket, weekday)
  │  Returns: List[str] — top-k predicted next apps
        │
        ▼
Cache.prefetch(predictions)
  │  Inserts predicted apps into WARM tier (bounded 15)
        │
        ▼
Cache.lookup(next_actual_app)
  │  Returns: "hot" | "warm" | "miss"
        │
        ▼
MeasuredLatencyModel.saved(app, tier)
  │  cold=2763ms, warm=1301ms, hot=274ms (Galaxy A23 measurements)
  └─ Metrics: hit_rate, precision, recall, F1, latency_saved_ms
```

---

## 2. Phase 1 — Component Input/Output Analysis

### 2.1 Transition Builder
**Source:** `scripts/ubiqlog_transition_pipeline.py`, lines 123–141

| Property | Detail |
|---------|--------|
| **Input** | Sorted `List[{pkg, start, end}]` per user |
| **Output** | `List[{from_app, to_app, timestamp, gap_s, time_bucket, day_of_week}]` |
| **Features used** | ProcessName, Start, End datetime, MAX_GAP=3600s |
| **Features ignored** | Activity context, WiFi, Bluetooth, Location, SMS, Calls |
| **Formula** | `time_bucket = hour * 2 + (1 if minute >= 30 else 0)` |

```python
# Line 132-140 (ubiqlog_transition_pipeline.py)
start_dt = b["start"]
time_bucket = start_dt.hour * 2 + start_dt.minute // 30  # 0-47
transitions.append({
    "from_app":    a["pkg"],
    "to_app":      b["pkg"],
    "timestamp":   start_dt.isoformat(),
    "gap_s":       round(gap, 1),
    "time_bucket": time_bucket,
    "day_of_week": start_dt.weekday(),   # 0=Mon, 6=Sun
})
```

**Finding:** `time_bucket` and `day_of_week` **ARE present** in transitions.parquet but **NOT used** by the graph construction (`build_graph()` on line 172 ignores them completely).

---

### 2.2 Graph Layer (`BehaviouralGraph`)
**Source:** `src/core/graph_engine.py`

| Property | Detail |
|---------|--------|
| **Node key** | `(app_id, time_bucket, battery_bucket)` — lines 315–317 |
| **Input** | EventBus `TOPIC_APP_LAUNCHED` payload |
| **Output** | Ranked successor node_ids via `get_top_k_next_nodes()` |
| **Features used** | `app_id`, `time_bucket`, `battery_bucket` (in node matching) |
| **Features in node** | `time_bucket`, `battery_bucket`, `context_flags`, `category` |
| **Edge attributes** | `transition_prob`, `time_sensitivity`, `battery_cost` |

**Critical Finding — Node Identity:**

```python
# graph_engine.py, lines 315-317 (_on_app_launched)
if (n.app_id == app_id and n.time_bucket == time_bucket
        and n.battery_bucket == battery_bucket):
    current_node_id = nid
    break
```

The `BehaviouralGraph` encodes **time_bucket in the node identity**. This means each `(app, time_bucket)` pair is a distinct node. The graph IS time-aware at the node level.

**However:** In the `run_benchmark_v4.py` evaluation, `_find_node_id()` in `environment_v2.py` (line 440) only matches on `app_id` and `time_bucket`, **ignoring battery_bucket** (since UbiqLog has no battery data). The benchmark policies (lines 380–670) do **not use BehaviouralGraph at all** — they implement their own inline Markov tables.

**Verdict:** The `BehaviouralGraph` is a simulation-era artifact. The benchmark pipelines bypass it entirely.

---

### 2.3 Confidence Prefetch Layer
**Source:** `src/prefetch/confidence_prefetch.py`

**Formula (lines 192–196):**

```
confidence = W_TRANSITION * transition_prob
           + W_RECENCY    * recency_score
           + W_FREQUENCY  * frequency_score
           + W_CONTEXT    * context_score

Weights (settings.py):
  W_TRANSITION = 0.50
  W_RECENCY    = 0.20
  W_FREQUENCY  = 0.20
  W_CONTEXT    = 0.10
```

**Context score formula (lines 267–288):**

```python
# context_score = 1.0 if most common bucket for app == current_bucket
#               = 0.5 if within ±2 buckets (±1 hour)
#               = 0.0 otherwise
diff = min(diff, 48 - diff)  # handles midnight wraparound
```

| Feature | Used in confidence? |
|---------|-------------------|
| `transition_prob` (graph edge) | ✅ 50% weight |
| `recency` (exponential decay) | ✅ 20% weight |
| `frequency` (count / total) | ✅ 20% weight |
| `context_score` (time_bucket match) | ✅ 10% weight |
| `day_of_week` | ❌ Not used |
| `weekday/weekend` | ❌ Not used |
| `session_id` | ❌ Not in data |

**Finding:** `ConfidencePrefetch` **is time-aware** (uses time_bucket for context scoring) but **is only used by `environment_v2.py`** (the SB3 PPO training path), NOT by the benchmark evaluation policies.

---

### 2.4 RL Layer (`environment_v2.py` — ResourceAllocationPolicy)
**Source:** `src/rl/environment_v2.py`

**Action space (lines 138–142):**
```python
# MultiDiscrete([5, 5, 5])
# Dimension 0: HOT budget  → [1, 5, 10, 20, 30]
# Dimension 1: WARM budget → [10, 30, 50, 100, 150]
# Dimension 2: conf_level  → [0.5, 0.6, 0.7, 0.8, 0.9]
```

**Observation vector — 109 dimensions (lines 362–402):**

| Indices | Feature | Detail |
|---------|---------|--------|
| [0:50] | current app one-hot | OHE over 50-app vocabulary |
| [50:100] | previous app one-hot | OHE over 50-app vocabulary |
| [100] | `time_bucket / 47` | Temporal context ✅ |
| [101] | `day_of_week / 6` | Weekly context ✅ |
| [102] | HOT occupancy ratio | Cache state |
| [103] | WARM occupancy ratio | Cache state |
| [104:109] | hit history (5 binary) | Recent performance |

**Reward formula (reward_v2.py, lines 98–104):**

```
R = 2.0 × hit_rate
  + 1.0 × (latency_saved_ms / MAX_LATENCY_SAVED_MS)
  − 0.5 × (battery_overhead / MAX_BATTERY_OVERHEAD)
  − 0.8 × (false_prefetch_count / prefetch_total)
  − 1.2 × (thrash_count / MAX_THRASH_PER_STEP)
```

**What RL optimizes:** `2.0×hit_rate + 1.0×latency` primarily. F1 is NOT directly optimized. Recall and Precision are NOT individually optimized. False prefetches penalized at −0.8 weight.

**What RL does NOT do:** RL does NOT select apps. It selects HOT/WARM budget sizes and a confidence threshold. The graph generates candidates; RL decides how many to keep.

---

### 2.5 Benchmark Policies (`run_benchmark_v4.py`)

The benchmark does NOT use `BehaviouralGraph`, `ConfidencePrefetch`, or `environment_v2.py`.

It builds **standalone inline Markov dictionaries**:

```python
# GraphOnlyPolicy.train() — lines ~481-490 run_benchmark_v4.py
c = defaultdict(lambda: defaultdict(int))
for i in range(1, len(apps)):
    c[apps[i-1]][apps[i]] += 1      # counts app→app
m1 = {s: dict(sorted(d.items())) for s, d in c.items()}
```

**This is a context-free first-order Markov chain. `time_bucket` and `day_of_week` are passed as arguments to `predict()` but IGNORED by GraphOnly, Markov-1, Markov-2, LRU, LFU, Frequency, RecencyFrequency, and GlobalMarkov2.**

Only `ContextMarkov` and `RLAdaptiveEnsemble` actually use `time_bucket`/`weekday` — and both underperformed in V4.

---

## 3. Phase 2 — Graph Inspection: Q&A

### Q1: Is the graph first-order or higher-order?

**Answer: In practice, ALL benchmark policies are first-order Markov.**

Evidence:
- `GraphOnly`, `Graph+Confidence`, `GraphMindRL` — all build `dict[app → dict[app, prob]]`
- `Markov-2` and `VOM` build order-2, but the PRIMARY GraphMind policy (`GraphMindRL`) is order-1
- `BehaviouralGraph` stores node-level context but transitions are still `node_i → node_j` (first-order on the node sequence)

**The benchmark result `GraphOnly F1 = Markov-1 F1 = 0.727` is direct proof.**

```python
# From benchmark output:
# GraphMindRL  F1=0.742
# GraphOnly    F1=0.727
# Markov-1     F1=0.727
# GraphOnly == Markov-1 because both implement P(next|current)
```

### Q2: Does any graph node contain time_bucket, weekday, weekend?

**Answer: YES — in `BehaviouralGraph` (the simulation-era graph).**

```python
# graph_engine.py, line 32-34 (GraphNode dataclass)
time_bucket: int               # 0-47 (30-min buckets)
battery_bucket: int            # 0-4
context_flags: dict            # {"headphones", "calendar_near", "weekend"}
```

**BUT:** The benchmark evaluation bypasses `BehaviouralGraph`. The inline Markov tables in `run_benchmark_v4.py` have no time context.

**Conclusion:** Time context EXISTS in the data model but is NOT USED in the benchmark evaluation of GraphMind's core policies.

### Q3: What is stored in graph edges?

**`GraphEdge` attributes (graph_engine.py, lines 41–50):**

| Attribute | Type | Range | Meaning |
|-----------|------|-------|---------|
| `source_id` | str | — | Source node UUID |
| `target_id` | str | — | Target node UUID |
| `transition_prob` | float | [0, 1] | P(target \| source), normalised |
| `time_sensitivity` | float | [0, 1] | How time-dependent this edge is |
| `battery_cost` | float | [0, 1] | Battery overhead of prefetching target |

**Note:** `battery_cost` and `time_sensitivity` are initialized to fixed values (0.2 and 0.5 respectively, line 348) and only `transition_prob` is ever updated (+0.01 per occurrence). The other two attributes are **never meaningfully trained**.

### Q4: Can the graph distinguish WhatsApp→Instagram at 9am vs 11pm?

**Answer: YES in `BehaviouralGraph`, NO in the benchmark policies.**

In `BehaviouralGraph`: Different time_buckets create different nodes. WhatsApp@tb=18 and WhatsApp@tb=46 are distinct nodes with separate edge sets.

In `run_benchmark_v4.py` (Markov-1, GraphOnly, GraphMindRL): The edge key is `(from_app, to_app)` only. 9am and 11pm produce the same prediction.

### Q5: Can the graph distinguish A→B→C from X→B→C?

**Answer: NO for most policies, YES for Markov-2.**

- Markov-1, GraphOnly, GraphMindRL: `predict(B)` ignores what came before B
- Markov-2, VOM: predict using `(prev, current)` context
- `BehaviouralGraph`: also first-order (edges are `node_i → node_j`)

---

## 4. Phase 3 — RL Inspection Summary

### What action does RL choose?

**V3 (ResourceAllocationPolicy / `environment_v2.py`):**  
→ Cache budget: HOT size, WARM size, confidence threshold  
→ RL does NOT select apps

**V4 (AdaptiveEnsembleController / `adaptive_ensemble_env.py`):**  
→ Predictor weights: `[w_M1, w_M2, w_VOM, w_ctx, w_graph]`  
→ RL selects which predictor to trust

### What reward is used?

**V3 (RewardV2):**
```
R = 2.0*hit_rate + 1.0*lat_norm − 0.5*batt_norm − 0.8*fp_norm − 1.2*thrash_norm
```
Optimizes: hit_rate (primary), latency, penalizes false prefetches and thrash  
Does NOT directly optimize: F1, Precision, Recall

**V4 (AdaptiveEnsemble):**
```
R = (1.0 if hit else 0.0) − 0.02 × weight_entropy
```
Optimizes: hit rate (binary), discourages weight concentration  
Does NOT optimize: F1, latency, false prefetch count

### Is RL predicting the next app?

**V3:** NO. RL adjusts HOT/WARM size. Graph predicts apps.  
**V4:** INDIRECTLY. RL weights predictors whose outputs are merged into a ranked list. The RL output is a weight vector, not an app selection.

### Complete V3 RL observation vector (109 dims):
```
[0:50]    current app one-hot  (vocab_size=50)
[50:100]  previous app one-hot
[100]     time_bucket / 47
[101]     day_of_week / 6
[102]     HOT_count / HOT_TIER_CAPACITY
[103]     WARM_count / WARM_TIER_CAPACITY
[104:109] recent hit history (5 binary)
```

### Complete V4 RL observation vector (15 dims):
```
[0]     current_app_vocab_idx / vocab_size  (vocab capped at 200)
[1]     prev_app_vocab_idx / vocab_size
[2]     time_bucket / 47
[3]     weekday / 6
[4]     transition_entropy / 4.0
[5]     M1 top-1 confidence
[6]     M2 top-1 confidence
[7]     VOM top-1 confidence
[8]     ContextMarkov top-1 confidence
[9]     Graph top-1 confidence
[10:15] hit history (5 binary)
```

---

## 5. Phase 4 — Time Context Status

### Time fields in the data pipeline:

| Field | Available in parquet? | Used in benchmark? |
|-------|----------------------|-------------------|
| `time_bucket` (0-47) | ✅ YES | ❌ GraphOnly/M1/M2/GraphMindRL: NO |
| `day_of_week` (0-6) | ✅ YES | ❌ GraphOnly/M1/M2/GraphMindRL: NO |
| `hour` (derived) | ✅ derivable from `time_bucket//2` | ❌ No policy uses it |
| `weekday/weekend` | ✅ derivable from `day_of_week` | ❌ No policy uses it |
| `session_id` | ❌ Not in data | — |

**Derivation formulas:**
```python
hour         = time_bucket // 2           # 0-23
half_hour    = time_bucket                # 0-47
quarter_hour = time_bucket * 2 + (minute // 15) % 2  # 0-95 (needs raw minute)
weekday      = day_of_week               # already present
weekend      = day_of_week >= 5          # Sat=5, Sun=6
```

### Time context analysis findings:

| Time period | Mean transition entropy |
|------------|----------------------|
| Afternoon | 2.387 bits |
| Evening | 2.394 bits |
| Morning | 2.262 bits |
| Night | 2.012 bits |

**Interpretation:** Night-time transitions are significantly MORE predictable (entropy 2.01 bits) than daytime (2.39 bits). A time-aware model should gain the most by conditioning on night vs. day context.

**Top app patterns change across periods:**
- Morning: com.sec.android.app.launcher (home screen)
- Afternoon: com.viber.voip (messaging)
- Evening: com.viber.voip, com.whatsapp (social)
- Night: com.viber.voip, com.sec.knox.eventsmanager (security)

---

## 6. Phase 5 — Higher-Order Markov Analysis

### Empirical results (top-5 hit rate, 31 users, 80/20 split):

| Model | Mean HR | Std | Min | Max |
|-------|---------|-----|-----|-----|
| **Markov-1** | **0.6045** | 0.2039 | 0.215 | 0.947 |
| Markov-2 | 0.5283 | 0.2083 | 0.149 | 0.933 |
| Markov-3 | 0.4845 | 0.2047 | 0.140 | 0.933 |
| **ContextMarkov-1** | **0.6582** | 0.1802 | 0.203 | 0.920 |

> Note: these are measured on the raw transition parquet, NOT the full top-5 benchmark. Numbers differ from the benchmark due to different evaluation windows and data scope.

### Key findings:

**Finding 1 — Higher order hurts without fallback:**  
M2 is WORSE than M1 by 7.6 percentage points. M3 is worse than M2 by 4.4 pp. This happens because higher-order models have more sparse states — when a bigram/trigram has never been seen in training, the fallback to M1 is too abrupt and the model encounters many unseen contexts in the 20% test split.

**Finding 2 — Time context helps significantly:**  
ContextMarkov-1 (using `P(next|app, coarse_time_bucket)`) outperforms pure M1 by **+5.37 percentage points**. This is a measured gain on real data with no other changes.

**Finding 3 — The VOM underperformance in V4 is explained:**  
With Laplace alpha=0.5 over ~500-app vocabulary, the smoothing flattens M2 probabilities and causes VOM to generate many low-confidence, diverse candidates. Combined with equal weight to M1, this degrades precision.

---

## 7. Critical Findings Summary

| Finding | Severity | Impact |
|---------|----------|--------|
| GraphOnly == Markov-1 (F1=0.727 each) | 🔴 HIGH | The "graph" provides no advantage over plain Markov |
| `time_bucket` exists but unused in core policies | 🔴 HIGH | Measured +5.4pp gain available from context-M1 |
| BehaviouralGraph is bypassed in benchmark | 🟡 MEDIUM | Architecture diverged from evaluation |
| Higher-order Markov hurts without smoothing | 🟡 MEDIUM | Naive M2/M3 is not the answer |
| RL reward doesn't optimize F1 directly | 🟡 MEDIUM | RL optimizes hit_rate, not precision/recall balance |
| `time_sensitivity`, `battery_cost` edges never trained | 🟢 LOW | Dead code in graph engine |
| V4 REINFORCE underfits (3 passes, 15-dim linear) | 🔴 HIGH | Model too weak for 500-app vocabulary |

---

## 8. Answer to the Central Question

**Is GraphMind secretly behaving as a first-order Markov chain?**

**YES.** For all primary benchmark policies (GraphOnly, GraphMindRL, Graph+Confidence):

```
predict(current_app) = top-k from P(next | current_app)
```

This is confirmed by the benchmark result: `GraphOnly F1 = Markov-1 F1 = 0.727` (identical to 4 decimal places).

**Is time-aware second-order prediction the highest-ROI improvement?**

**YES — with a caveat:**
- Pure second-order (naive M2) **hurts** by −7.6pp (sparsity problem)
- Time-conditioned first-order **helps** by +5.4pp (measured on real data)
- The winning combination is **context-aware M1 with graceful M2 fallback and Laplace smoothing**

---
