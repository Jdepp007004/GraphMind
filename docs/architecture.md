# GraphMind V6 — System Architecture

> **Samsung EnnovateX AX Hackathon 2026 — PS03**

## Architecture Diagram

![GraphMind V6 Architecture](../architecture_diagram.png)

---

## Seven Architectural Layers

GraphMind V6 is organised into seven distinct layers that form a closed-loop agentic pipeline. Each layer has a single responsibility, communicating through well-defined interfaces.

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1: PERCEPTION — EventBus                          │
│  App-switch events → node identity (app, time, battery)  │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│  Layer 2: LONG-TERM MEMORY — BehaviouralGraph            │
│  Per-user NetworkX DiGraph — Markov transition model     │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│  Layer 3: REASONING — ConfidenceScorer                   │
│  Multi-signal fusion: 0.50×transition + 0.40×freq        │
│  + 0.10×recency → ranked candidate list                  │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│  Layer 4: RERANKING — EmbeddingTransformerReranker [V6]  │
│  Per-user Transformer reranks candidates via             │
│  34-dim app embeddings + temporal features               │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│  Layer 5: ACTUATION — MemoryManager + FiveTierCache      │
│  PIN (10ms) → HOT (42ms) → WARM (190ms)                  │
│  → COOL (400ms) → COLD (720ms)                           │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│  Layer 6: PLANNING — RL Environment (PPO)                │
│  AdaptiveThresholdController — MultiDiscrete([5,5,5])    │
│  Adjusts HOT size, WARM size, confidence threshold       │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│  Layer 7: REWARD — RewardV2                              │
│  R = 2.0×hit_rate − 1.2×thrash (multi-component)        │
│  + Gemma 2B explanation layer (post-decision, optional)  │
└──────────────────────────────────────────────────────────┘
```

---

## Layer 1: EventBus (Perception)

**File:** `src/core/event_bus.py`

The EventBus is the system's sensory layer. It consumes the Android `TOPIC_APP_LAUNCHED` event stream (or the UbiqLog replay feed during benchmarking) and extracts the **node identity**:

```python
node_identity = (
    event["app_id"],          # Package name: e.g. "com.google.youtube"
    event["time_bucket"],     # 30-min slot 0–47: e.g. 38 = 7pm
    event["battery_bucket"],  # 0–4 (0=0–20%, 4=80–100%)
)
```

The triple encodes *what*, *when*, and *under what resource constraint* simultaneously.

---

## Layer 2: BehaviouralGraph (Long-Term Memory)

**File:** `src/core/graph_engine.py`

A per-user **NetworkX DiGraph** stores the full history of app transitions as weighted edges:

- **Nodes:** `(app_id, time_bucket, battery_bucket)` tuples
- **Edges:** Directed, weighted by observed transition count
- **Query:** `BehaviouralGraph.query(node)` → `{app_id: probability}` dict
- **Storage:** In-memory graph + SQLite cold store (`cold_graph.db`)
- **Eviction:** Nodes inactive > `NODE_EVICTION_DAYS=15` days removed from cold tier

The graph separates `YouTube@7pm` and `YouTube@9am` into different nodes, capturing time-dependent transition patterns without explicit feature engineering.

---

## Layer 3: ConfidenceScorer (Reasoning)

**File:** `src/prefetch/confidence_prefetch.py`

Multi-signal linear fusion:

```
score(app) = 0.50 × P(app | current_app)   # Markov transition probability
           + 0.40 × freq_score(app)         # normalised historical frequency
           + 0.10 × recency_score(app)      # exponential decay from last use
           + 0.00 × context_score(app)      # zeroed — noisy on short datasets
```

Weights determined by systematic **Phase 11A grid search** (not manual tuning). All candidates above the adaptive threshold are passed downstream.

---

## Layer 4: EmbeddingTransformerReranker (V6 Addition)

**File:** `src/models/transformer_reranker.py`

The key V6 innovation. A **per-user Transformer** reranks the ConfidenceScorer's candidate list using:

- **App embeddings:** 32-dim learned embedding per app package
- **Temporal features:** time_bucket, day_of_week (normalised)
- **Architecture:** 2-layer Transformer encoder + linear head
- **Training:** Per-user, chronological split, ~30 epochs

V6 trains **31 separate rerankers** (one per user), avoiding cross-user contamination.

**Impact:** Cache hit rate 80.51% (V5, no reranker) → **97.92%** (V6, per-user Transformer).

---

## Layer 5: MemoryManager + FiveTierCache (Actuation)

**Files:** `src/core/memory_manager.py`, `src/core/five_tier_cache.py`

V6 implements a **5-tier cache hierarchy** calibrated to Samsung Galaxy A23 latencies:

| Tier | Capacity | Simulated Latency | Eviction Policy |
|---|---|---|---|
| 📌 **PIN** | 2 apps | 10 ms | Never evicted — pinned by user |
| 🔥 **HOT** | 5 apps | 42 ms | LRU from interactions |
| 🌡️ **WARM** | 15 apps | 190 ms | Prefetched by ConfidenceScorer |
| 🌀 **COOL** | 30 apps | 400 ms | Lower-confidence prefetches |
| ❄️ **COLD** | Unlimited | 720 ms | SQLite on-device, evict after 15 days |

**Security flush:** Transition from sensitive app (finance, health) → consumer app triggers HOT flush to prevent cross-context data leakage. See `src/security/sensitivity_model.py`.

---

## Layer 6: RL Environment + PPO Agent (Planning)

**Files:** `src/rl/environment_v2.py`, `src/rl/reward_v2.py`

**Action space:** `MultiDiscrete([5, 5, 5])` = 125 discrete actions

| Dimension | Options |
|---|---|
| `hot_budget` | [1, 5, 10, 20, 30] apps |
| `warm_budget` | [10, 30, 50, 100, 150] apps |
| `conf_threshold` | [0.50, 0.60, 0.70, 0.80, 0.90] |

**State observation:** 109-dimensional vector (app one-hot encodings + temporal features + cache occupancy + 5-step hit history).

**Production controller:** Bang-bang adaptive meta-controller (simpler than full PPO, better convergence on short 2-month sequences):
```
if rolling_hit_rate_20 > 0.80: threshold += 0.005
elif rolling_hit_rate_20 < 0.50: threshold -= 0.005
```

---

## Layer 7: RewardV2 + Gemma 2B (Reward + Explanation)

**Files:** `src/rl/reward_v2.py`, `src/gemma_explainer.py`

**Reward signal:**
```
R = 2.0 × hit_rate
  + 1.0 × (latency_saved_ms / 800)
  - 0.5 × battery_overhead_pct
  - 0.8 × false_prefetch_rate
  - 1.2 × thrash_rate
```

**Gemma 2B (optional):** Post-decision NL explanation — fires async after KPI measurement, zero effect on benchmarks.

```
→ "Preloading Spotify because you typically switch from YouTube in the evening."
```

HuggingFace: [https://huggingface.co/google/gemma-2b](https://huggingface.co/google/gemma-2b)

---

## Data Flow Summary

```
App Launch Event
→ EventBus (extract node identity)
→ BehaviouralGraph.query() [Tool #1]
→ ConfidenceScorer (multi-signal fusion)
→ EmbeddingTransformerReranker.rerank() [Tool #2]
→ AdaptiveThresholdController.filter()
→ FiveTierCache.allocate()
← [KPI metrics measured here]
→ Gemma.generate_explanation() [Tool #3, async]
→ RewardV2.compute()
→ PPOAgent.update_policy()
```

---

*See [ax.md](ax.md) for the detailed agentic workflow narrative.*  
*See [benchmarking.md](benchmarking.md) for evaluation methodology.*
