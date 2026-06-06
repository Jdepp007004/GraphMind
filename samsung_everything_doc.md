# GraphMind — Complete Technical Reference

> **Everything you need to know about GraphMind in a single document.**
> On-device, privacy-preserving, RL-trained predictive app-launch cache for Samsung Android devices.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Feature List & How They Are Implemented](#4-feature-list--how-they-are-implemented)
   - 4.1 [Behavioural Graph Engine](#41-behavioural-graph-engine)
   - 4.2 [Three-Tier Memory Hierarchy](#42-three-tier-memory-hierarchy)
   - 4.3 [EventBus (Pub-Sub Backbone)](#43-eventbus-pub-sub-backbone)
   - 4.4 [Context Encoder (Situational Embeddings)](#44-context-encoder-situational-embeddings)
   - 4.5 [PPO Reinforcement Learning Agent](#45-ppo-reinforcement-learning-agent)
   - 4.6 [LangGraph Multi-Agent Orchestrator](#46-langgraph-multi-agent-orchestrator)
   - 4.7 [Graph Manager Agent (Gemma 2B)](#47-graph-manager-agent-gemma-2b)
   - 4.8 [Drift Detector Agent (KL Divergence)](#48-drift-detector-agent-kl-divergence)
   - 4.9 [Prefetch Daemon](#49-prefetch-daemon)
   - 4.10 [Security Agent & Context Boundary Enforcer](#410-security-agent--context-boundary-enforcer)
   - 4.11 [Synthetic Dataset Generator](#411-synthetic-dataset-generator)
   - 4.12 [Benchmark Evaluation Framework](#412-benchmark-evaluation-framework)
   - 4.13 [Explainability Engine](#413-explainability-engine)
   - 4.14 [Graph Playback System](#414-graph-playback-system)
   - 4.15 [Android Telemetry Integration](#415-android-telemetry-integration)
   - 4.16 [Streamlit Dashboard](#416-streamlit-dashboard)
   - 4.17 [CLI Wizard](#417-cli-wizard)
5. [Benchmark Results](#5-benchmark-results)
   - 5.1 [Cache Hit Rate Comparison](#51-cache-hit-rate-comparison)
   - 5.2 [Launch Speed Gain](#52-launch-speed-gain)
   - 5.3 [Advanced Metrics](#53-advanced-metrics)
   - 5.4 [How Each Metric is Calculated](#54-how-each-metric-is-calculated)
6. [Reward Function Deep-Dive](#6-reward-function-deep-dive)
7. [KL Divergence Drift Detection — Full Calculation](#7-kl-divergence-drift-detection--full-calculation)
8. [RL Environment Specification](#8-rl-environment-specification)
9. [PPO Hyperparameters](#9-ppo-hyperparameters)
10. [Configuration Reference (settings.py)](#10-configuration-reference-settingspy)
11. [Project Folder Structure](#11-project-folder-structure)
12. [Module List — One-Liner Explanations](#12-module-list--one-liner-explanations)
13. [Data Flow (End-to-End)](#13-data-flow-end-to-end)
14. [Security Model](#14-security-model)
15. [Agentic AI Practices (ax.md summary)](#15-agentic-ai-practices-axmd-summary)
16. [10-User Personas](#16-10-user-personas)
17. [App Taxonomy & App ID Vocabulary](#17-app-taxonomy--app-id-vocabulary)
18. [Baseline Policies Compared](#18-baseline-policies-compared)
19. [Installation & Quick Start](#19-installation--quick-start)
20. [Test Suite](#20-test-suite)
21. [What Worked / What Did NOT Work](#21-what-worked--what-did-not-work)
22. [Future Work](#22-future-work)
23. [Dependencies](#23-dependencies)

---

## 1. Project Overview

**GraphMind** is an on-device, privacy-preserving, predictive app-launch memory management system built for Samsung Android devices. It replaces Android's reactive Low Memory Killer Daemon (LMKD) with a proactive, reinforcement-learning-trained three-tier memory hierarchy.

| Property | Value |
|---|---|
| **Language** | Python 3.11/3.12 |
| **RL Framework** | Stable-Baselines3 PPO + Gymnasium |
| **Agent Framework** | LangGraph 0.1.14 |
| **Graph Backend** | NetworkX DiGraph |
| **LLM (optional)** | Google Gemma 2B (HuggingFace Transformers) |
| **Dashboard** | Streamlit + Plotly + PyVis |
| **Database (COLD tier)** | SQLite |
| **Simulation scope** | 10 users × 30 days × ~80 events/day |
| **Total synthetic events** | ~24,000 per dataset |

The system simulates and measures how a graph-based predictive cache (HOT/WARM/COLD tiers) compares against four Android baselines: LMKD, ART Static Profile, UsageStats LRU, and Samsung Bixby Frequency.

---

## 2. Problem Statement

Android's **LMKD (Low Memory Killer Daemon)** kills background apps to free RAM when memory pressure rises. This means the next time you open that app, it must perform a **cold start** — reloading all resources from disk, taking 1–3 seconds. This "launch friction" is a key pain point on Samsung devices.

**GraphMind's approach:**
- Model the user's behavioural patterns as a directed weighted graph
- Use RL (PPO) to learn which apps to keep in memory (HOT tier) and which to pre-warm (WARM tier)
- Proactively load predicted next apps before the user opens them → warm starts (~0.1 s) instead of cold starts (~1–3 s)
- Enforce privacy: flush sensitive (financial/health) data from HOT cache when switching to consumer apps (social/entertainment)

---

## 3. System Architecture

```
+-------------------------------------------------------------------------+
|                          GraphMind System                               |
|                                                                         |
|  ┌─────────────┐  EventBus   ┌──────────────────────────────────────┐  |
|  │  OS Events  │────────────►│      LangGraph Orchestrator          │  |
|  │ (Simulator/ │             │  ┌─────────────┐  ┌──────────────┐   │  |
|  │  ADB Live)  │             │  │Graph Manager│─►│Drift Detector│   │  |
|  └─────────────┘             │  └─────────────┘  └──────┬───────┘   │  |
|                              │      conditional          │           │  |
|  ┌─────────────┐             │  ┌─────────────┐  ┌──────▼───────┐   │  |
|  │Behavioural  │◄────────────│  │  RL Trainer │  │   Prefetch   │   │  |
|  │   Graph     │             │  │    Agent    │─►│    Agent     │   │  |
|  │ (NetworkX)  │             │  └─────────────┘  └──────┬───────┘   │  |
|  └──────┬──────┘             │                   ┌──────▼───────┐   │  |
|         │                   │                   │   Security   │   │  |
|  ┌──────▼──────┐             │                   │    Agent     │   │  |
|  │   Memory   │             │                   └──────────────┘   │  |
|  │  Manager   │             └──────────────────────────────────────┘  |
|  │ HOT/WARM/  │                                                        |
|  │   COLD     │                                                        |
|  └────────────┘                                                        |
+-------------------------------------------------------------------------+
```

**Flow summary:**
```
START → graph_manager → drift_detector → [KL > 0.3] → rl_trainer → prefetch → security → END
                                       → [KL ≤ 0.3] ──────────────────────────────────────►
```

---

## 4. Feature List & How They Are Implemented

---

### 4.1 Behavioural Graph Engine

**File:** [`src/core/graph_engine.py`](src/core/graph_engine.py)

#### What It Is
A directed weighted graph (NetworkX `DiGraph`) where:
- **Nodes** = situations: `(app_id, time_bucket, battery_bucket)` tuples
- **Edges** = transitions: directed links with 3D weights

#### GraphNode Structure
```python
@dataclass
class GraphNode:
    node_id: str               # UUID
    embedding: np.ndarray      # shape (64,) — 64-dim situational embedding
    app_id: str                # e.g. "com.instagram.android"
    time_bucket: int           # 0–47 (30-min buckets in a 24hr day)
    battery_bucket: int        # 0–4 (0=0–20%, 1=20–40%, …, 4=80–100%)
    context_flags: dict        # {"headphones": bool, "calendar_near": bool, "weekend": bool}
    last_seen_day: int         # simulation day of last access (for eviction)
    access_count: int          # total accesses
    category: str             # "social", "financial", "utility", etc.
```

#### GraphEdge Structure
```python
@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    transition_prob: float     # [0.0, 1.0] — probability of going to target from source
    time_sensitivity: float    # [0.0, 1.0] — how time-dependent this transition is
    battery_cost: float        # [0.0, 1.0] — battery penalty for pre-fetching target
```

#### Key Operations

**Top-K Prediction (used by PrefetchDaemon):**
```
score = transition_prob - (battery_cost × (1 - battery_level/100))
```
- Sorts edges by score descending, returns top-k target node IDs
- If `battery_level < BATTERY_SUPPRESS_THRESHOLD (20%)`: `k = max(1, k // 2)`

**Edge Weight Updates:**
- Every time transition `A → B` is observed: `delta_prob += 0.01` (clamped to 1.0)
- After each update: outgoing edges from source are re-normalized so they sum to 1.0

**Edge Pruning (weekly):**
- Removes all edges where `transition_prob < 0.05`

**Node Eviction:**
- Removes nodes where `(current_day - last_seen_day) > NODE_EVICTION_DAYS (15)`

**Serialization:**
- Graph is pickled to disk: `data/base_graphs/{user_id}_base.pkl`
- Snapshot for dashboard: truncated to 200 nodes / 500 edges (JSON-serializable)

---

### 4.2 Three-Tier Memory Hierarchy

**File:** [`src/core/memory_manager.py`](src/core/memory_manager.py)

#### Architecture

| Tier | Implementation | Capacity | Simulated Latency | Backing |
|------|---------------|----------|-------------------|---------|
| **HOT** | Python `dict` + LRU list | 30 nodes | ~0 ms (RAM) | In-process |
| **WARM** | `OrderedDict` LRU | 150 nodes | ~5 ms (cache) | In-process |
| **COLD** | SQLite table `cold_nodes` | Unlimited | ~50 ms (disk) | `data/cold_graph.db` |

#### Promotion / Demotion Flow

```
Access node → check HOT → if miss, check WARM → if miss, check COLD → if miss: graph lookup
                 ↓ on HOT full                ↓ on WARM full
            evict LRU HOT → WARM          evict LRU WARM → COLD
```

**HOT tier promotion (`promote_to_hot`):**
1. If already in HOT → update LRU order (no-op)
2. If HOT at capacity (≥ 30) → evict LRU HOT node to WARM (cascade if WARM also full)
3. Load node from WARM / COLD / graph
4. Insert into HOT dict, append to LRU list
5. Publish `TOPIC_NODE_PROMOTED`

**WARM tier eviction to COLD:**
- Serializes `GraphNode` via `pickle.dumps`
- Stores in SQLite: `INSERT OR REPLACE INTO cold_nodes (user_id, node_id, serialized_node, last_seen_day)`

**Security flush (`flush_hot_by_category`):**
- Finds all HOT nodes whose `category` matches a sensitive category
- Demotes them to WARM (then COLD if WARM is full)
- Called by `SecurityAgent` when sensitive→consumer transition is detected

**Cache hit/miss tracking:**
- Every app launch triggers `check_and_publish_cache_result(node_id)`
- Publishes `TOPIC_CACHE_HIT` (with tier = "hot" or "warm") or `TOPIC_CACHE_MISS`

**WARM rebuild (PrefetchDaemon):**
- Dumps all current WARM nodes to COLD
- Reloads with `predicted_node_ids` from the graph's top-K predictions

---

### 4.3 EventBus (Pub-Sub Backbone)

**File:** [`src/core/event_bus.py`](src/core/event_bus.py)

#### Design
Thread-safe singleton pub-sub bus. **All inter-module communication goes through EventBus** — no direct cross-module imports between unrelated components.

```python
bus = EventBus.get_instance()  # singleton
bus.subscribe("app_launched", callback)
bus.publish("app_launched", {"app_id": "...", "user_id": "...", "timestamp": 0.0})
```

#### Topics

| Topic Constant | String | When Published | Who Subscribes |
|---|---|---|---|
| `TOPIC_APP_LAUNCHED` | `"app_launched"` | Every app foreground event | Graph, Memory, Drift, Security, Prefetch |
| `TOPIC_APP_CLOSED` | `"app_closed"` | App goes to background | — |
| `TOPIC_BATTERY_UPDATED` | `"battery_updated"` | Battery level changes | PrefetchDaemon |
| `TOPIC_HEADPHONES_CONNECTED` | `"headphones_connected"` | Wired/BT headphones connected | PrefetchDaemon |
| `TOPIC_CALENDAR_EVENT` | `"calendar_event_approaching"` | Calendar event ≤ 30 min away | PrefetchDaemon |
| `TOPIC_NODE_PROMOTED` | `"node_promoted_to_hot"` | Node enters HOT tier | Dashboard |
| `TOPIC_NODE_DEMOTED` | `"node_demoted_from_hot"` | Node leaves HOT tier | Dashboard |
| `TOPIC_CACHE_HIT` | `"cache_hit"` | App found in HOT or WARM | RL Env, Dashboard |
| `TOPIC_CACHE_MISS` | `"cache_miss"` | App not found in HOT or WARM | RL Env, Dashboard |
| `TOPIC_DRIFT_DETECTED` | `"drift_detected"` | KL divergence > 0.3 | RLTrainerAgent |
| `TOPIC_SECURITY_FLUSH` | `"security_cache_flush"` | Sensitive→consumer transition | Dashboard |
| `TOPIC_PREFETCH_TRIGGERED` | `"prefetch_triggered"` | Prefetch cycle ran | Dashboard |
| `TOPIC_RL_WEIGHT_UPDATED` | `"rl_weight_updated"` | PPO policy updated | — |

#### Schema Validation
- `EventBus.__init__` builds a schema registry from `event_schema.py`
- Every `publish()` call validates the payload against the schema
- Invalid events are rejected (logged as WARNING), count tracked in `_rejected_event_count`

---

### 4.4 Context Encoder (Situational Embeddings)

**File:** [`src/data/context_encoder.py`](src/data/context_encoder.py)

#### Architecture
A 3-layer MLP (PyTorch) that encodes raw OS event data into a 64-dim embedding vector.

```
Input (35 dims):
  app_id_one_hot[30]   → one-hot from APP_ID_VOCAB (30 known apps)
  time_bucket / 47.0   → normalized [0,1]
  battery / 100.0      → normalized [0,1]
  headphones           → 0.0 or 1.0
  calendar_near        → 1.0 if calendar event ≤ 30 min
  weekend              → 0.0 or 1.0

MLP Architecture:
  Linear(35 → 128) + ReLU
  Linear(128 → 64) + ReLU
  Linear(64 → 64)          ← raw embedding, no activation

Output (64 dims): situational embedding vector
```

#### App ID Vocabulary (30 apps)
```
com.instagram.android, com.google.youtube, com.spotify.music,
com.slack.android, com.google.android.gm, com.linkedin.android,
com.google.android.maps, com.android.calendar, com.tiktok.android,
com.whatsapp, com.netflix.mediaclient, com.amazon.mShop.android,
net.one97.paytm, com.google.android.apps.photos, com.github.android,
com.samsung.health, com.strava, com.myntra.android,
com.zomato.android, com.swiggy.android, com.google.android.apps.docs,
com.adobe.reader, com.phonepe.app, com.hdfcbank.new,
com.samsung.android.messaging, com.booking, com.makemytrip,
com.indiainfoline.trade, com.samsung.android.calendar, unknown
```

- Unknown apps map to index 29 (`"unknown"`)
- Weights saved to `models/encoder.pt` (if trained); random weights otherwise
- Encoder weights are **frozen** during PPO training (simultaneous fine-tuning caused instability)

---

### 4.5 PPO Reinforcement Learning Agent

**Files:** [`src/rl/environment.py`](src/rl/environment.py), [`src/rl/trainer.py`](src/rl/trainer.py), [`src/rl/reward.py`](src/rl/reward.py), [`src/rl/evaluation.py`](src/rl/evaluation.py)

#### Gymnasium Environment (`GraphMindEnv`)

**Observation Space:** `Box(shape=(68,), dtype=float32)`

```
Observation vector (68 dims):
  [0:35]   context embedding from ContextEncoder.encode(last_event)[:35]
  [35:65]  hot_tier_occupancy[30]  → 1.0 for occupied slots, 0.0 for empty
  [65]     battery / 100.0
  [66]     time_bucket / 47.0
  [67]     recent_hit_rate (last 50 events)
```

**Action Space:** `Discrete(31)`

| Action | Effect |
|---|---|
| 0–28 | Promote HOT node at that index to priority front (LRU update) |
| 29 | Call `graph.prune_weak_edges()` |
| 30 | Emergency demote: demote bottom 50% of HOT nodes to WARM |

**Episode:** One simulated day (all events for that day for one user)

**Reset:** Increments to next simulation day; resets cache hit/miss counters

**Step:**
1. Advance one event from `EventSimulator`
2. Apply action (promotion / prune / demote)
3. Track thrash events (nodes that were in HOT and are now missing)
4. Compute reward via `compute_reward()`
5. `terminated = True` when day's events are exhausted

---

### 4.6 LangGraph Multi-Agent Orchestrator

**File:** [`src/agents/orchestrator.py`](src/agents/orchestrator.py)

#### State Schema (`GraphMindState`)
```python
class GraphMindState(TypedDict):
    user_id: str
    current_day: int
    current_event: Optional[dict]
    battery: float
    kl_divergence: float
    cache_hit_rate: float
    security_flush_count: int
    last_agent: str
    messages: List[dict]
```

#### Agent Graph (LangGraph StateGraph)
```
START
  │
  ▼
graph_manager          (always runs)
  │
  ▼
drift_detector         (always runs, computes KL)
  │
  ├─[KL > 0.3]──► rl_trainer ──► prefetch ──► security ──► END
  │
  └─[KL ≤ 0.3]──────────────────► prefetch ──► security ──► END
```

Each agent receives the full `GraphMindState`, updates it, and returns the updated state. All 5 agents are initialized in `GraphMindOrchestrator.__init__` and share references to the same `BehaviouralGraph` and `MemoryManager`.

**run_full_simulation()** runs all `SIMULATION_DAYS (30)` days sequentially:
- Weekly pruning: `graph.prune_weak_edges()` every 7 days
- Daily eviction: `graph.evict_stale_nodes(day)` every day
- Saves complete log to `results/{user_id}_simulation_log.json`

---

### 4.7 Graph Manager Agent (Gemma 2B)

**File:** [`src/agents/graph_manager_agent.py`](src/agents/graph_manager_agent.py)

#### What It Does
Prioritizes which HOT-tier nodes should be kept by reasoning about current context.

**With Gemma 2B (if `models/gemma-2b/` exists):**
```
Prompt → "Time of day bucket: {X}/47. Apps in cache: {Y, Z, ...}. Which 3 apps should be highest priority? List app IDs only."
Response → parsed for known app IDs → re-promote those in LRU order
```

**Without Gemma (rule-based fallback):**
```
Sort HOT nodes by access_count descending → promote top 5
```

**Weekly task:** Calls `graph.prune_weak_edges()` every 7 days.

**Why Gemma is optional:** Gemma 2B is too slow for real-time decisions on CPU. Used only for periodic/offline tasks with a rule-based fallback for latency-sensitive paths.

---

### 4.8 Drift Detector Agent (KL Divergence)

**File:** [`src/agents/drift_detector_agent.py`](src/agents/drift_detector_agent.py)

#### What It Detects
Changes in app usage distribution that indicate the user's behaviour has fundamentally shifted (e.g., started a new job, changed daily routine, went on vacation).

#### Data Structures
```python
transition_history = deque(maxlen=DRIFT_WINDOW_SIZE * 2)  # maxlen=200 — historical window
recent_window      = deque(maxlen=DRIFT_WINDOW_SIZE)      # maxlen=100 — recent window
```

Every `TOPIC_APP_LAUNCHED` event appends `app_id` to both deques.

#### KL Divergence Calculation (Full Algorithm)

```python
def compute_kl_divergence() -> float:
    # 1. Guard: need at least DRIFT_WINDOW_SIZE events
    if len(recent_window) < 100 or len(transition_history) < 100:
        return 0.0

    # 2. Build vocabulary: union of all app IDs seen in both windows
    vocab = sorted(set(recent_window) | set(transition_history))
    eps = 1e-10

    # 3. Historical distribution P: frequency over transition_history (200 events)
    hist_counts = {app: count_in(transition_history, app) for app in vocab}
    hist_total = sum(hist_counts.values())
    P = np.array([(hist_counts[a] / hist_total) + eps for a in vocab])
    P /= P.sum()   # normalize to sum = 1.0

    # 4. Recent distribution Q: frequency over recent_window (100 events)
    rec_counts = {app: count_in(recent_window, app) for app in vocab}
    rec_total = sum(rec_counts.values())
    Q = np.array([(rec_counts[a] / rec_total) + eps for a in vocab])
    Q /= Q.sum()   # normalize to sum = 1.0

    # 5. KL(Q || P) = Σ Q(i) * log(Q(i) / P(i))
    kl = float(scipy.stats.entropy(Q, P))
    return max(0.0, kl)
```

**Interpretation:**
- `KL = 0.0` → recent usage matches history exactly (stable behaviour)
- `KL > 0.3` → significant drift → publish `TOPIC_DRIFT_DETECTED` → trigger `rl_trainer`
- `KL > 0.3` also causes `RLTrainerAgent` to spike learning rate by `DRIFT_LR_SPIKE_MULTIPLIER (5.0×)`

**Why epsilon=1e-10?** Prevents `log(0)` when an app appears in one window but not the other.

---

### 4.9 Prefetch Daemon

**File:** [`src/prefetch/daemon.py`](src/prefetch/daemon.py)

#### What It Does
Proactively pre-warms the WARM and HOT tiers with predicted next apps.

#### Scheduling
- Uses **APScheduler `BackgroundScheduler`**
- Runs `run_prefetch_cycle()` every `PREFETCH_INTERVAL_MINUTES (15)` minutes

#### Prefetch Cycle Logic
```python
def run_prefetch_cycle() -> List[str]:
    # 1. Battery check
    k = 2 if current_battery < 20 else PREFETCH_TOP_K (5)

    # 2. Get predictions from graph
    predicted_ids = graph.get_top_k_next_nodes(current_node_id, k, current_battery)
    # scoring: score = transition_prob - (battery_cost × (1 - battery/100))

    # 3. Rebuild WARM tier with predictions
    memory_manager.rebuild_warm_from_graph(predicted_ids)

    # 4. Promote top-2 predictions to HOT immediately
    for nid in predicted_ids[:2]:
        memory_manager.promote_to_hot(nid)

    # 5. Announce
    bus.publish(TOPIC_PREFETCH_TRIGGERED, {...})
```

#### Context-Triggered Prefetch (Beyond Scheduler)
| Event | Action |
|---|---|
| **Headphones connected** | Immediately promotes top Spotify/YouTube/Netflix/TikTok node to HOT |
| **Calendar event ≤ 30 min** | Promotes top-3 productivity/enterprise category nodes to HOT |
| **Battery update** | Updates `current_battery` for suppression logic |
| **App launched** | Updates `current_node_id` for next prediction cycle |

---

### 4.10 Security Agent & Context Boundary Enforcer

**Files:** [`src/security/context_boundary.py`](src/security/context_boundary.py), [`src/security/classification_guard.py`](src/security/classification_guard.py)

#### Threat Model
When a user switches from a financial/health app (sensitive context) to a social/entertainment app (consumer context), GraphMind ensures no sensitive data remains accessible in the HOT cache — preventing potential data leakage through the memory hierarchy.

#### Sensitive vs Consumer Categories

```python
SENSITIVE_CATEGORIES = ["financial", "health", "enterprise", "government", "unknown_sensitive"]
CONSUMER_CATEGORIES  = ["social", "entertainment", "shopping", "gaming"]
```

**"unknown_sensitive" rule:** Unknown package names are isolated as `unknown_sensitive` until classified. This conservative default prevents potential data leakage from unrecognized apps.

#### Transition Detection
```python
def check_transition(from_category, to_category) -> bool:
    return (from_category in SENSITIVE_CATEGORIES and
            to_category in CONSUMER_CATEGORIES)
```

Examples that trigger a flush:
- `financial → social` ✅ (banking → Instagram)
- `health → entertainment` ✅ (Samsung Health → YouTube)
- `enterprise → shopping` ✅ (Slack → Amazon)
- `social → financial` ❌ (no flush — going to sensitive is fine)

#### Flush Action
```python
def enforce_boundary(from_category, to_category, timestamp):
    for cat in SENSITIVE_CATEGORIES:
        flushed_ids.extend(memory_manager.flush_hot_by_category(cat))
    flush_event = {
        "timestamp": timestamp,
        "from_category": from_category,
        "to_category": to_category,
        "flushed_node_ids": flushed_ids,
        "user_id": user_id
    }
    flush_log.append(flush_event)
    bus.publish(TOPIC_SECURITY_FLUSH, flush_event)
```

#### Retention Policy
```
HOT tier:    500 events max retention
WARM tier:   2000 events max retention
COLD tier:   15 days max retention
Trace log:   1000 events max
```

---

### 4.11 Synthetic Dataset Generator

**File:** [`src/data/dataset_generator.py`](src/data/dataset_generator.py)

#### What It Generates
- 10 user event logs: `data/synthetic/users/user_00.json` through `user_09.json`
- Each file: ~2,400 events (30 days × ~80 events/day)
- Total: ~24,000 events across all users

#### Event Schema (one event)
```json
{
  "day": 0,
  "timestamp": 28800.5,
  "app_id": "com.instagram.android",
  "battery": 87.3,
  "time_bucket": 16,
  "headphones": false,
  "calendar_event_in_mins": null,
  "weekend": false,
  "category": "social"
}
```

#### Generation Algorithm (Rule-Based Fallback)
```
For each day d in 0..29:
  n_events = Normal(mean=80, std=20), clipped to minimum 50
  battery starts at 100%, drains ~90% linearly across the day
  
  For each event:
    1. Compute hour based on event index (spread 6am–midnight)
    2. Add Gaussian jitter ±1 hour
    3. Select app:
       - 30% probability: top apps from persona profile
       - 15% probability: inject sensitive app (banking/health)
       - 15% probability: inject consumer app (social/entertainment)
       - 40% probability: random app from taxonomy
    4. Compute time_bucket = floor((hour×60 + minute) / 30)  → range 0–47
    5. Simulate battery drain per event
    6. Randomly assign headphones (20% probability)
    7. Randomly assign calendar event (10% probability, 5–120 min ahead)
  
  Sort events by timestamp within each day
```

**Reproducibility:** Each user uses seed `RANDOM_SEED (42) + user_number` → fully deterministic.

**Behavioral drift injection:** `drift_factor = day / 30` — later days subtly shift app selection probabilities to simulate real behavioral drift that the KL detector must catch.

**Gemma 2B mode:** If `models/gemma-2b/` exists, prompts Gemma with persona description and attempts JSON parsing. Falls back to rule-based if Gemma response is unparseable.

#### 100-User Expansion
`generate_100_users()` clones the 10 base personas 10 times each, shuffling top apps and shifting peak hours by ±1 hour.

---

### 4.12 Benchmark Evaluation Framework

**Files:** [`src/benchmarks/evaluator.py`](src/benchmarks/evaluator.py), [`src/benchmarks/baselines.py`](src/benchmarks/baselines.py), [`src/benchmarks/advanced_metrics.py`](src/benchmarks/advanced_metrics.py), [`src/benchmarks/graphmind_policy_runner.py`](src/benchmarks/graphmind_policy_runner.py)

#### What It Evaluates
Replays all 10 users × all 5 policies × 300 events per user, measuring:

| KPI | How Measured |
|---|---|
| `cache_hit_rate` | Fraction of events where app was in predicted/pre-warmed set |
| `launch_speed_gain_pct` | `(hit_rate - baseline_hit_rate) × 30.0` |
| `thrash_rate` | Fraction of events where app was in HOT but was just evicted |
| `battery_overhead_pct` | Simulated — 1.5% if hit_rate > 0.5, else 0.5% |
| `graph_node_count` | Unique distinct apps seen in user's event log |

#### Launch Speed Gain Formula
```
gain_pct = (cache_hit_rate - baseline_cache_hit_rate) × 30.0
```
Based on Android ART documentation: a cache hit avoids a full cold start (~30% faster launch).

#### Metric Provenance
Each metric row is tagged: `MEASURED` or `ESTIMATED`
- `MEASURED`: Directly computed from simulation replay
- `ESTIMATED`: Derived/approximated (e.g., battery overhead, launch speed)

---

### 4.13 Explainability Engine

**Files:** [`src/explainability/reasoning_engine.py`](src/explainability/reasoning_engine.py), [`src/explainability/prediction_explainer.py`](src/explainability/prediction_explainer.py), [`src/explainability/decision_trace.py`](src/explainability/decision_trace.py)

#### What It Does
Generates human-readable reason strings for every GraphMind decision. Pure functional — no side effects, no EventBus.

#### Reason Types
- `reasons_for_preload(app_id, transition_prob, battery, time_bucket, access_count, ...)` → explains why an app was prefetched
- `reasons_for_promotion(app_id, from_tier, access_count, time_bucket, kl_divergence)` → explains HOT promotion
- `reasons_for_demotion(app_id, hot_pressure, days_inactive)` → explains WARM demotion
- `reasons_for_flush(from_category, to_category, flushed_count)` → explains security flush
- `reasons_for_prediction(app_id, source_app, transition_prob, rank, battery, time_bucket)` → explains next-app prediction

**Example output:**
```
instagram preloaded because:
  - transition probability 0.42 from previous app
  - frequently accessed (15 times in session)
  - morning commute hours usage pattern detected
  - weekday behavioral pattern
  - high battery (allows prefetch)
  confidence: 84%
```

---

### 4.14 Graph Playback System

**Files:** [`src/graph_playback/timeline_engine.py`](src/graph_playback/timeline_engine.py), [`src/graph_playback/snapshot_manager.py`](src/graph_playback/snapshot_manager.py), [`src/graph_playback/graph_animator.py`](src/graph_playback/graph_animator.py)

#### What It Does
Allows time-travel through graph evolution. Records daily graph snapshots and can replay them in sequence for visualization.

- **`SnapshotManager`:** Saves/loads graph state snapshots to `results/snapshots/`
- **`TimelineEngine`:** Manages the ordered sequence of daily snapshots; supports seek, step-forward, step-back
- **`GraphAnimator`:** Converts snapshot diffs to animation frames for the dashboard

---

### 4.15 Android Telemetry Integration

**Files:** [`src/android/`](src/android/)

All 9 modules in this package enable live connection to a real Samsung device via ADB.

| Module | What It Does |
|---|---|
| `adb_connector.py` | Manages ADB subprocess connection; runs shell commands on device |
| `device_detector.py` | Detects connected Samsung devices via `adb devices`; reads model, serial, API level |
| `battery_collector.py` | Reads battery % and charging state via `adb shell dumpsys battery` |
| `usage_stats_collector.py` | Gets foreground app via `adb shell dumpsys usagestats`; deduplicates |
| `audio_collector.py` | Detects wired/BT headphones via `adb shell dumpsys audio` |
| `screen_collector.py` | Reads screen on/off + locked state via `adb shell dumpsys power` |
| `calendar_collector.py` | Reads upcoming calendar events from device |
| `telemetry_event_adapter.py` | Translates raw ADB data into GraphMind EventBus `TOPIC_APP_LAUNCHED` payloads |
| `telemetry_collector.py` | Orchestrates all collectors in a 5-second polling background thread |

**Live mode:** `TelemetryCollector.start()` runs all sensors in a daemon thread, publishing real app-launch events to the same EventBus that the simulation uses → GraphMind can run live on a real device without code changes.

---

### 4.16 Streamlit Dashboard

**File:** [`src/dashboard/app.py`](src/dashboard/app.py)

Run with: `streamlit run src/dashboard/app.py` → opens at `http://localhost:8501`

#### 9 Dashboard Tabs

| Tab | Content |
|---|---|
| 🔗 Graph Evolution | PyVis interactive graph at days 1, 7, 14, 29 per user |
| 📊 Benchmarks | Plotly bar chart: cache hit rate by policy; provenance table |
| 🎯 RL Training | PPO episode reward curves per user over training steps |
| 🔒 Security Log | Table of all security flush events with colour coding |
| 💾 Memory Tiers | Pie chart of HOT/WARM/COLD node distribution |
| Provenance | Detailed metric provenance (MEASURED vs ESTIMATED) labels |
| Policy Comparison | Random vs NoOp vs Frequency vs LRU vs PPO bar chart |
| Scale Test | Line chart: prediction time vs user count |
| Device | Samsung device diagnostic report JSON |

#### Sidebar Controls
- **User selector:** `user_00` through `user_09`
- **Day slider:** 0–29
- **▶ Run Live Simulation:** Triggers `GraphMindOrchestrator.run_full_simulation()` live
- **📊 Run Benchmarks:** Triggers `BenchmarkEvaluator.run_all()` live

---

### 4.17 CLI Wizard

**Files:** [`src/cli/wizard.py`](src/cli/wizard.py), [`src/cli/connect_samsung.py`](src/cli/connect_samsung.py), [`src/cli/device_setup.py`](src/cli/device_setup.py)

- **`wizard.py`:** Interactive terminal setup wizard — detects device, validates ADB, generates dataset, trains RL, runs benchmark in one guided session
- **`connect_samsung.py`:** `python -m src.cli.connect_samsung [--doctor]` — device health check and diagnostic report
- **`device_setup.py`:** Sets up ADB forwarding and verifies permissions for live telemetry

---

## 5. Benchmark Results

### 5.1 Cache Hit Rate Comparison

*(Averaged across all 10 users, 300 events each, Day 29)*

| Policy | Avg Cache Hit Rate | vs LMKD Baseline |
|---|---|---|
| **GraphMind_RL** | **59.8%** | **+29.3 pp** |
| LMKD_Reactive | 30.5% | — |
| UsageStats_LRU | 30.5% | 0.0 pp |
| Bixby_Frequency | 20.2% | −10.3 pp |
| ART_StaticProfile | 0.0% | −30.5 pp |

> **Note:** ART_StaticProfile scores 0% because in simulation its static profile (built from Days 1–7) does not match the app_id format used in the event replay. In production, ART profiles use compiled code paths rather than package IDs. This is a known limitation of the simulation vs. real-world comparison.

**Per-user breakdown:**

| User | LMKD | UsageStats | Bixby | GraphMind_RL | Gain vs LMKD |
|---|---|---|---|---|---|
| user_00 | 26.0% | 26.0% | 17.0% | **59.7%** | +33.7 pp |
| user_01 | 29.0% | 29.0% | 18.7% | **61.0%** | +32.0 pp |
| user_02 | 34.0% | 34.0% | 21.7% | **63.7%** | +29.7 pp |
| user_03 | 31.3% | 31.3% | 19.0% | **59.7%** | +28.4 pp |
| user_04 | 30.7% | 30.7% | 20.7% | **62.7%** | +32.0 pp |
| user_05 | 28.3% | 28.3% | 21.3% | **58.3%** | +30.0 pp |
| user_06 | 25.7% | 25.7% | 20.7% | **61.0%** | +35.3 pp |
| user_07 | 25.7% | 25.7% | 19.3% | **58.3%** | +32.6 pp |
| user_08 | 28.3% | 28.3% | 20.7% | **57.0%** | +28.7 pp |
| user_09 | 33.7% | 33.7% | 23.0% | **58.0%** | +24.3 pp |

### 5.2 Launch Speed Gain

*(Formula: `(hit_rate - lmkd_rate) × 30.0` → estimated % faster launch)*

| User | Launch Speed Gain (%) |
|---|---|
| user_00 | +10.1% |
| user_01 | +9.6% |
| user_02 | +8.9% |
| user_03 | +8.5% |
| user_04 | +9.6% |
| user_05 | +9.0% |
| user_06 | +10.6% |
| user_07 | +9.8% |
| user_08 | +8.6% |
| user_09 | +7.3% |
| **Average** | **+9.2%** |

### 5.3 Advanced Metrics

*(From `results/advanced_benchmark_results.csv`, averaged across 10 users)*

#### Latency Percentiles (simulated)

| Metric | Value | How Modelled |
|---|---|---|
| **P50 Latency** | ~844 ms | Weighted avg of HOT (45ms), WARM (210ms), COLD (850ms) hits |
| **P95 Latency** | ~962 ms | 95th percentile of simulated latency distribution |
| **P99 Latency** | ~998 ms | 99th percentile of simulated latency distribution |

> The P50 is dominated by COLD misses in the simulation (most nodes not yet warm early in the day). In real deployment, HOT-tier hits dominate for well-trained models → expected real P50 ~45–210 ms.

#### Prefetch Precision / Recall / F1

| Metric | Value |
|---|---|
| Prefetch Precision | ~5.4% |
| Prefetch Recall | ~59.4% |
| Prefetch F1 | ~9.7% |

> Low precision, high recall: the system casts a wide net (pre-warms many nodes), correctly recalling most of the apps that will actually be used. Precision is low because many prefetched nodes are never accessed in the short evaluation window (300 events).

#### Memory Footprint Estimates

| Tier | Estimate |
|---|---|
| HOT (30 nodes × 8 KB) | **~0.23 MB RAM** |
| WARM (varies) | **~0.3 MB** |
| COLD (SQLite) | **~19–25 MB per user** |
| Total per user | **~20–26 MB** |

#### Graph Growth Metrics

| Metric | Value |
|---|---|
| Node growth rate | ~27.5 nodes/day |
| Edge growth rate | ~53.1 edges/day |
| Node churn rate | ~21% of days saw node count decrease (evictions) |
| Edge churn rate | ~7–14% of days saw edge count decrease (pruning) |

#### Security Flush Accuracy

| Metric | Value |
|---|---|
| Flush accuracy | **100%** (all flushes removed ≥1 node) |
| False flush rate | **0%** |
| Flush rate | ~96–129 flushes per 1000 events |

---

### 5.4 How Each Metric is Calculated

#### Cache Hit Rate
```
cache_hit_rate = cache_hits / (cache_hits + cache_misses)

"Hit" = app_id was in the policy's predicted set OR was previously in the hot set
"Miss" = app_id was not predicted
```

#### Thrash Rate
```
thrash_rate = thrash_events / (cache_hits + cache_misses)

"Thrash" = app was in HOT at step i-1, but NOT in predicted set at step i,
           and then the user launched that app at step i (evict-then-need pattern)
```

#### Battery Overhead %
```
battery_overhead_pct = 1.5  (if cache_hit_rate > 0.5)
                     = 0.5  (otherwise)
```
Simulated overhead — higher-performing policies do more prefetching, consuming slightly more battery. In the RL environment, actual battery drain is tracked per action (0.1% per promote action).

#### Launch Speed Gain %
```
gain_pct = (graphmind_cache_hit_rate - lmkd_cache_hit_rate) × 30.0

Basis: Android ART docs state warm start is ~30% faster than cold start.
A 30pp improvement in cache hit rate → approximately 30% × 30 = 9% overall launch time improvement.
```

#### Latency Percentiles (Simulated)
```python
# Build latency array from tier composition
latencies = (
    [45.0  ms] × hot_hits   +
    [210.0 ms] × warm_hits  +
    [850.0 ms] × cold_misses
)
# Add 10% Gaussian jitter (std=0.1 of mean)
latencies_jittered = latencies × clip(Normal(1.0, 0.1), 0.7, 1.3)
P50 = percentile(latencies_jittered, 50)
P95 = percentile(latencies_jittered, 95)
P99 = percentile(latencies_jittered, 99)
```

#### Prefetch Precision / Recall / F1
```
precision = |predicted_set ∩ actual_accessed_set| / |predicted_set|
recall    = |predicted_set ∩ actual_accessed_set| / |actual_accessed_set|
F1        = 2 × (precision × recall) / (precision + recall)
```

#### Memory Footprint
```
bytes_per_node = 8192  (~8 KB: 64-dim float32 embedding + metadata)
RAM estimate (HOT)   = hot_count   × 8 KB / 1024² MB
WARM estimate        = warm_count  × 8 KB / 1024² MB
COLD estimate        = cold_count  × 8 KB / 1024² MB
```

#### Graph Growth / Churn
```
node_growth_rate = mean(max(0, node_counts[i] - node_counts[i-1]) for i in 1..29)
node_churn_rate  = count(days where node_count decreased) / 29
```

---

## 6. Reward Function Deep-Dive

**File:** [`src/rl/reward.py`](src/rl/reward.py)

### Formula

```
R = α × cache_hit_rate
  + β × speed_gain
  - γ × thrash_rate
  - δ × battery_cost
  + ε × friction_saved_rate
  - ζ × fp_rate
```

### Component Definitions

| Component | Formula | Range | Notes |
|---|---|---|---|
| `cache_hit_rate` | `cache_hits / (cache_hits + cache_misses)` | [0, 1] | Primary objective |
| `speed_gain` | `min(1.0, friction_saved / total)` | [0, 1] | = launch friction avoided |
| `thrash_rate` | `min(1.0, thrash_events / 10)` | [0, 1] | 10 thrashes = max penalty |
| `battery_cost` | `min(1.0, battery_consumed / 5.0)` | [0, 1] | 5% drain = max penalty |
| `friction_saved_rate` | `min(1.0, friction_saved / total)` | [0, 1] | Redundant with speed_gain (same value) |
| `fp_rate` | `min(1.0, prefetch_fp_count / 15)` | [0, 1] | 15 false prefetches = max penalty |

### Weights

| Weight | Symbol | Value | Rationale |
|---|---|---|---|
| `REWARD_ALPHA` | α | **1.0** | Cache hit rate is the primary objective |
| `REWARD_BETA` | β | **0.8** | Speed gain is almost as important |
| `REWARD_GAMMA` | γ | **0.5** | Thrashing is penalized but tolerated |
| `REWARD_DELTA` | δ | **0.3** | Battery cost is a light penalty |
| `REWARD_EPSILON` | ε | **0.4** | Friction saved is rewarded |
| `REWARD_ZETA` | ζ | **0.3** | False prefetches are lightly penalized |

### Maximum Possible Reward
```
R_max = 1.0 × 1.0 + 0.8 × 1.0 + 0.4 × 1.0 = 2.2  (with no penalties)
R_min = 0 - 0.5 - 0.3 - 0.3 = -1.1               (all-penalty scenario)
```

---

## 7. KL Divergence Drift Detection — Full Calculation

**File:** [`src/agents/drift_detector_agent.py`](src/agents/drift_detector_agent.py)

### Mathematical Definition
```
KL(Q || P) = Σᵢ Q(i) × log(Q(i) / P(i))
```
Where:
- **P** = historical distribution (200 most recent app launches)
- **Q** = recent distribution (100 most recent app launches)
- **i** iterates over the union vocabulary of all seen app IDs

### What It Measures
How much Q (recent) diverges from P (historical). A higher value means recent usage is increasingly different from historical usage.

### Threshold Logic
```
KL ≤ 0.3 → Stable behaviour → skip RL trainer → go straight to prefetch
KL > 0.3 → Drift detected → run RL trainer → learning_rate × 5.0 spike
```

### Epsilon Smoothing
```
P[i] = (hist_counts[i] / hist_total) + 1e-10
Q[i] = (rec_counts[i] / rec_total) + 1e-10
```
Then re-normalize P and Q to sum to 1.0 after adding epsilon. This prevents `log(0)` when an app appears in one window but not the other.

### Example Calculation
```
vocab = ["instagram", "spotify", "gmail", "maps"]
history (200 events): instagram×80, spotify×60, gmail×40, maps×20
P = [0.40, 0.30, 0.20, 0.10]

recent (100 events): instagram×10, spotify×10, gmail×30, maps×50
Q = [0.10, 0.10, 0.30, 0.50]

KL(Q||P) = 0.10×log(0.10/0.40) + 0.10×log(0.10/0.30) + 0.30×log(0.30/0.20) + 0.50×log(0.50/0.10)
         = 0.10×(−1.386) + 0.10×(−1.099) + 0.30×(0.405) + 0.50×(1.609)
         = −0.139 + (−0.110) + 0.122 + 0.805
         = 0.678  → DRIFT DETECTED (> 0.3)
```

---

## 8. RL Environment Specification

**File:** [`src/rl/environment.py`](src/rl/environment.py)

### Observation Space (68 dimensions)

```
Index  |  Source                         |  Description
-------|-----------------------------------|-----------------------------------------
0–34   |  ContextEncoder.encode(event)    |  35-dim context vector (truncated from 64)
35–64  |  hot_tier_occupancy[30]          |  1.0 if slot occupied, 0.0 if empty
65     |  battery / 100.0                 |  Normalized battery level
66     |  time_bucket / 47.0             |  Normalized time of day
67     |  recent_hit_rate                 |  Cache hits / total over last 50 events
```

### Action Space (Discrete 31)

```
Action 0–28:  Promote HOT[action] to LRU front  → costs 0.1% battery
Action 29:    Call graph.prune_weak_edges()      → 0% battery cost
Action 30:    Demote bottom 50% of HOT to WARM  → counts as thrash events
```

### Episode Lifecycle
```
reset() → increment day, reset counters
step(action) → consume next event from simulator
            → apply action
            → compute reward
            → terminated=True when day's events exhausted
```

### Training Scale
```
PPO_TOTAL_TIMESTEPS = 200,000  per user
PPO_N_STEPS         = 512      (reduced from 2048 to prevent gradient instability on short episodes)
PPO_BATCH_SIZE      = 64
PPO_N_EPOCHS        = 10
PPO_GAMMA           = 0.99     (discount factor)
PPO_LEARNING_RATE   = 3e-4
```

---

## 9. PPO Hyperparameters

| Parameter | Value | Location |
|---|---|---|
| Algorithm | PPO (MlpPolicy) | `stable_baselines3.PPO` |
| Total timesteps | 200,000 per user | `settings.PPO_TOTAL_TIMESTEPS` |
| Learning rate | 3e-4 | `settings.PPO_LEARNING_RATE` |
| n_steps | 512 | `settings.PPO_N_STEPS` (clamped in trainer) |
| Batch size | 64 | `settings.PPO_BATCH_SIZE` |
| n_epochs | 10 | `settings.PPO_N_EPOCHS` |
| Gamma (discount) | 0.99 | `settings.PPO_GAMMA` |
| Seed | 42 | `settings.RANDOM_SEED` |
| Policy network | MLP (default SB3) | 2 hidden layers × 64 units |
| Drift LR spike | 5.0× | `settings.DRIFT_LR_SPIKE_MULTIPLIER` |

**Why `n_steps=512` instead of 2048?** Episodes of ~80 events per day caused gradient instability with larger rollout buffers. Reducing n_steps to 512 ensured the PPO update happens more frequently relative to the episode length.

---

## 10. Configuration Reference (settings.py)

**File:** [`config/settings.py`](config/settings.py)

| Constant | Value | Description |
|---|---|---|
| `NUM_USERS` | 10 | Number of simulated users |
| `SIMULATION_DAYS` | 30 | Days per user simulation |
| `EVENTS_PER_DAY_MEAN` | 80 | Mean app events per day |
| `EVENTS_PER_DAY_STD` | 20 | Standard deviation of events per day |
| `RANDOM_SEED` | 42 | Global random seed |
| `NODE_EMBEDDING_DIM` | 64 | Dimensions of situational embedding |
| `EDGE_PRUNE_THRESHOLD` | 0.05 | Prune edges with prob < 5% |
| `NODE_EVICTION_DAYS` | 15 | Evict nodes inactive > 15 days |
| `MAX_NODES_COLD` | 2000 | Hard cap on COLD graph size |
| `HOT_TIER_CAPACITY` | 30 | Max nodes in HOT |
| `WARM_TIER_CAPACITY` | 150 | Max nodes in WARM |
| `COLD_DB_PATH` | `data/cold_graph.db` | SQLite file path |
| `PPO_TOTAL_TIMESTEPS` | 200,000 | RL training budget per user |
| `PPO_LEARNING_RATE` | 3e-4 | PPO learning rate |
| `PPO_N_STEPS` | 2048 | Rollout buffer (capped to 512 in trainer) |
| `PPO_BATCH_SIZE` | 64 | PPO mini-batch size |
| `PPO_N_EPOCHS` | 10 | PPO epochs per update |
| `PPO_GAMMA` | 0.99 | Discount factor |
| `REWARD_ALPHA` | 1.0 | Cache hit rate weight |
| `REWARD_BETA` | 0.8 | Speed gain weight |
| `REWARD_GAMMA` | 0.5 | Thrash penalty weight |
| `REWARD_DELTA` | 0.3 | Battery cost weight |
| `REWARD_EPSILON` | 0.4 | Friction saved weight |
| `REWARD_ZETA` | 0.3 | False prefetch penalty weight |
| `PREFETCH_INTERVAL_MINUTES` | 15 | Prefetch daemon cycle interval |
| `PREFETCH_TOP_K` | 5 | Nodes to pre-warm per cycle |
| `BATTERY_SUPPRESS_THRESHOLD` | 20 | % below which prefetch is reduced |
| `DRIFT_WINDOW_SIZE` | 100 | Events in recent drift window |
| `DRIFT_KL_THRESHOLD` | 0.3 | KL above this triggers RL fine-tune |
| `DRIFT_LR_SPIKE_MULTIPLIER` | 5.0 | LR multiplier on drift detection |
| `SENSITIVE_CATEGORIES` | financial, health, enterprise, government, unknown_sensitive | Security-sensitive app categories |
| `CONSUMER_CATEGORIES` | social, entertainment, shopping, gaming | Consumer app categories |
| `HOT_RETENTION_EVENTS` | 500 | HOT tier security retention limit |
| `WARM_RETENTION_EVENTS` | 2000 | WARM tier security retention limit |
| `COLD_RETENTION_DAYS` | 15 | COLD tier age limit |
| `TRACE_RETENTION_EVENTS` | 1000 | Security trace log limit |
| `GEMMA_MODEL_ID` | `google/gemma-2b` | HuggingFace model ID |
| `GEMMA_LOCAL_PATH` | `models/gemma-2b` | Local model directory |
| `GEMMA_MAX_NEW_TOKENS` | 128 | Max tokens for Gemma generation |
| `GEMMA_DEVICE` | cpu | Device for PyTorch (env: `DEVICE`) |
| `DASHBOARD_PORT` | 8501 | Streamlit port |
| `LOG_LEVEL` | INFO | Logging level (env: `LOG_LEVEL`) |
| `BASELINE_LMKD` | `"LMKD_Reactive"` | LMKD baseline name |
| `BASELINE_ART` | `"ART_StaticProfile"` | ART baseline name |
| `BASELINE_LRU` | `"UsageStats_LRU"` | LRU baseline name |
| `BASELINE_BIXBY` | `"Bixby_Frequency"` | Bixby baseline name |
| `BASELINE_GRAPHMIND` | `"GraphMind_RL"` | GraphMind policy name |

---

## 11. Project Folder Structure

```
Samsung/                                    ← Project root
│
├── config/
│   ├── __init__.py
│   └── settings.py                        ← Single source of truth for all constants
│
├── data/
│   ├── app_taxonomy.json                  ← App package → category mapping
│   ├── cold_graph.db                      ← SQLite COLD tier (~73 MB)
│   ├── base_graphs/                       ← Pickled base BehaviouralGraph per user
│   └── synthetic/
│       └── users/
│           ├── user_00.json               ← 30-day event log for user 0
│           ├── user_01.json
│           ├── ...
│           └── user_09.json
│
├── docs/
│   ├── architecture.md                    ← System architecture diagram
│   ├── ax.md                              ← Agentic AI practices documentation
│   ├── installation.md                    ← Setup guide
│   └── user_guide.md                      ← Dashboard & CLI usage guide
│
├── logs/                                  ← Runtime log files
│
├── models/
│   ├── gemma-2b/                          ← (optional) Local Gemma 2B weights
│   ├── rl_policies/
│   │   ├── user_00_ppo.zip                ← Trained PPO policy per user
│   │   └── ...
│   └── encoder.pt                         ← ContextEncoder MLP weights
│
├── results/
│   ├── benchmark_results.csv             ← 5-policy × 10-user comparison
│   ├── advanced_benchmark_results.csv    ← Advanced KPIs (latency, memory, etc.)
│   ├── ppo_training_metrics.csv          ← Per-step PPO training logs
│   ├── policy_comparison.csv             ← Random/NoOp/Freq/LRU/PPO comparison
│   ├── scale_test.csv                    ← Prediction time vs user count
│   ├── device_report.json                ← Samsung device diagnostic
│   ├── training_curves.json              ← Training reward curves
│   ├── snapshots/                        ← Daily graph snapshots for playback
│   └── user_NN_simulation_log.json       ← Full 30-day simulation log per user
│
├── scripts/
│   ├── generate_dataset.py               ← Run dataset generation
│   ├── generate_100_users.py             ← Expand to 100 users
│   ├── train_rl.py                       ← Train PPO policies
│   ├── run_simulation.py                 ← Run full simulation for a user
│   ├── run_benchmarks.py                 ← Run all baselines comparison
│   ├── run_dashboard.py                  ← Launch Streamlit dashboard
│   ├── run_scale_test.py                 ← Scale test: prediction time vs N users
│   ├── run_topk_study.py                 ← Study optimal prefetch K value
│   ├── run_iteration2_validation.py      ← Phase 2 validation script
│   ├── run_iteration3_validation.py      ← Phase 3 validation script
│   ├── download_models.py                ← Download Gemma 2B from HuggingFace
│   └── device_validation.py             ← Validate real device connectivity
│
├── src/
│   ├── __init__.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py               ← LangGraph StateGraph (5-agent coordinator)
│   │   ├── graph_manager_agent.py        ← HOT tier prioritization with Gemma 2B
│   │   ├── drift_detector_agent.py       ← KL divergence drift detection
│   │   ├── rl_trainer_agent.py           ← PPO fine-tuning agent
│   │   ├── prefetch_agent.py             ← LangGraph wrapper for PrefetchDaemon
│   │   ├── security_agent.py             ← LangGraph wrapper for ContextBoundaryEnforcer
│   │   └── drift_visualizer.py           ← Visualization helper for drift metrics
│   │
│   ├── android/
│   │   ├── __init__.py
│   │   ├── adb_connector.py              ← ADB subprocess management
│   │   ├── device_detector.py            ← Samsung device discovery
│   │   ├── battery_collector.py          ← ADB battery telemetry
│   │   ├── usage_stats_collector.py      ← Foreground app via UsageStatsManager
│   │   ├── audio_collector.py            ← Headphone detection via ADB
│   │   ├── screen_collector.py           ← Screen on/locked state via ADB
│   │   ├── calendar_collector.py         ← Upcoming calendar events via ADB
│   │   ├── telemetry_event_adapter.py    ← Translates ADB data → EventBus events
│   │   └── telemetry_collector.py        ← Orchestrates all collectors in polling loop
│   │
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── evaluator.py                  ← Main 5-policy × 10-user benchmark runner
│   │   ├── baselines.py                  ← 4 baseline policy implementations
│   │   ├── advanced_metrics.py           ← Latency, memory, precision/recall metrics
│   │   ├── graphmind_policy_runner.py    ← Runs GraphMind through the benchmark harness
│   │   ├── case_study.py                 ← Per-user case study analysis
│   │   └── provenance.py                 ← MEASURED/ESTIMATED metric labelling
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── wizard.py                     ← Interactive setup and demo wizard
│   │   ├── connect_samsung.py            ← Device connection CLI entry point
│   │   └── device_setup.py              ← ADB forwarding and permission setup
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── graph_engine.py               ← BehaviouralGraph (NetworkX DiGraph wrapper)
│   │   ├── memory_manager.py             ← HOT/WARM/COLD three-tier memory hierarchy
│   │   ├── event_bus.py                  ← Thread-safe singleton pub-sub event bus
│   │   └── event_schema.py              ← Event payload schema definitions and registry
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py                        ← Streamlit 9-tab dashboard application
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_generator.py          ← Synthetic 10-user behavioural event generator
│   │   ├── context_encoder.py            ← 3-layer MLP: OS event → 64-dim embedding
│   │   └── event_simulator.py           ← Replays saved event logs through EventBus
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── reasoning_engine.py           ← Human-readable reason generation (pure functions)
│   │   ├── prediction_explainer.py       ← Per-prediction explanation builder
│   │   └── decision_trace.py            ← Decision audit trail recorder
│   │
│   ├── graph_playback/
│   │   ├── __init__.py
│   │   ├── timeline_engine.py            ← Day-by-day graph snapshot navigation
│   │   ├── snapshot_manager.py          ← Saves/loads graph state snapshots
│   │   └── graph_animator.py            ← Converts snapshot diffs to animation frames
│   │
│   ├── prefetch/
│   │   ├── __init__.py
│   │   └── daemon.py                     ← APScheduler 15-min prefetch + context triggers
│   │
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── environment.py               ← Custom Gymnasium env (68-dim obs, 31 actions)
│   │   ├── trainer.py                    ← PPO training manager with W&B callback
│   │   ├── reward.py                     ← Reward function (pure function, no side effects)
│   │   └── evaluation.py               ← Policy evaluation and comparison utilities
│   │
│   └── security/
│       ├── __init__.py
│       ├── context_boundary.py          ← Sensitive→consumer transition detection + flush
│       ├── classification_guard.py      ← App category classification with unknown isolation
│       └── security_visualizer.py       ← Security event visualization helpers
│
├── tests/
│   ├── conftest.py                       ← Pytest fixtures and EventBus reset
│   ├── test_phase1_graph.py             ← Graph engine unit tests
│   ├── test_phase2_memory.py            ← Memory manager unit tests
│   ├── test_phase3_rl.py                ← RL environment and reward tests
│   ├── test_phase4_agents.py            ← Agent unit tests (drift, graph manager, etc.)
│   ├── test_phase5_benchmarks.py        ← Benchmark framework tests
│   ├── test_advanced_benchmarks.py      ← Advanced KPI tests
│   ├── test_android_integration.py      ← Android ADB integration tests
│   ├── test_benchmark_fairness.py       ← Ensures all baselines start from identical state
│   ├── test_benchmark_provenance.py     ← Provenance label tests
│   ├── test_cli_wizard.py               ← CLI wizard tests
│   ├── test_device_validation.py        ← Device detection tests
│   ├── test_drift_visualization.py      ← Drift visualizer tests
│   ├── test_event_validation.py         ← EventBus schema validation tests
│   ├── test_explainability.py           ← Reasoning engine and explainer tests
│   ├── test_graph_playback.py           ← Timeline and snapshot tests
│   ├── test_rl_evaluation.py            ← RL policy evaluation tests
│   ├── test_scale.py                    ← Graph scale and performance tests
│   ├── test_security_hardening.py       ← Security boundary and flush tests
│   └── test_security_visualization.py  ← Security visualizer tests
│
├── .env.example                          ← Environment variable template
├── .gitignore
├── agents.md                             ← Agentic practices summary (user-rules doc)
├── GRAPHMIND_BUILD_SPEC.md              ← Full build specification (~80 KB)
├── GRAPHMIND_FULL_AUDIT_REPORT.md       ← Audit report (~76 KB)
├── GRAPHMIND_HARDCHECK.py               ← 6-phase automated validation script (~83 KB)
├── graphmind_audit.py                   ← Audit runner
├── LICENSE
├── README.md                             ← Brief project overview
├── requirements.txt                     ← Python dependencies
├── retrain_system_python.py             ← Retraining utility
└── retrain_user00.py                    ← Quick user_00 retraining script
```

---

## 12. Module List — One-Liner Explanations

### `config/`
| Module | One-Liner |
|---|---|
| `settings.py` | Single source of truth: all constants, thresholds, paths, and hyperparameters |

### `src/core/`
| Module | One-Liner |
|---|---|
| `graph_engine.py` | Directed weighted NetworkX graph representing user app-usage situations and transitions |
| `memory_manager.py` | HOT/WARM/COLD three-tier memory hierarchy with LRU eviction and SQLite COLD persistence |
| `event_bus.py` | Thread-safe singleton pub-sub bus decoupling all inter-module communication |
| `event_schema.py` | Defines and validates payload schemas for each EventBus topic |

### `src/agents/`
| Module | One-Liner |
|---|---|
| `orchestrator.py` | LangGraph StateGraph wiring all 5 agents with conditional drift→RL routing |
| `graph_manager_agent.py` | Prioritizes HOT tier using Gemma 2B reasoning or access-count fallback, prunes weekly |
| `drift_detector_agent.py` | Computes KL divergence between 100-event recent window and 200-event history window |
| `rl_trainer_agent.py` | LangGraph node that spikes PPO learning rate and optionally runs fine-tuning on drift |
| `prefetch_agent.py` | Thin LangGraph wrapper delegating prefetch trigger to PrefetchDaemon |
| `security_agent.py` | Thin LangGraph wrapper delegating sensitive→consumer transitions to ContextBoundaryEnforcer |
| `drift_visualizer.py` | Generates KL divergence time-series and histogram plots for the dashboard |

### `src/android/`
| Module | One-Liner |
|---|---|
| `adb_connector.py` | Runs `adb shell` subprocesses and manages device connections |
| `device_detector.py` | Discovers connected Samsung devices, reads model name, API level, serial |
| `battery_collector.py` | Polls battery % and charging state via `dumpsys battery` |
| `usage_stats_collector.py` | Gets foreground package name via `dumpsys usagestats` with deduplication |
| `audio_collector.py` | Detects wired and Bluetooth headphone connections via `dumpsys audio` |
| `screen_collector.py` | Reads screen on/off and locked state via `dumpsys power` |
| `calendar_collector.py` | Reads next calendar event title and minutes-until via ADB |
| `telemetry_event_adapter.py` | Translates raw ADB sensor data into EventBus-compatible event payloads |
| `telemetry_collector.py` | Background 5-second polling loop orchestrating all collectors and publishing events |

### `src/benchmarks/`
| Module | One-Liner |
|---|---|
| `evaluator.py` | Replays 10 users × 5 policies, measures cache hit rate, thrash, and battery overhead |
| `baselines.py` | Implements LMKD, ART StaticProfile, UsageStats LRU, and Bixby Frequency baselines |
| `advanced_metrics.py` | Computes prefetch precision/recall/F1, P50/P95/P99 latency, memory footprint estimates |
| `graphmind_policy_runner.py` | Runs GraphMind (graph + memory + prefetch) through the benchmark event-replay harness |
| `case_study.py` | Per-user deep-dive analysis of graph evolution, cache performance, and security events |
| `provenance.py` | Tags every benchmark metric row as MEASURED or ESTIMATED for transparency |

### `src/cli/`
| Module | One-Liner |
|---|---|
| `wizard.py` | Interactive terminal wizard: detect device → generate data → train RL → run benchmarks |
| `connect_samsung.py` | CLI entry point for `--doctor` device health checks and diagnostic reports |
| `device_setup.py` | Configures ADB port-forwarding and USB debugging permissions for live mode |

### `src/dashboard/`
| Module | One-Liner |
|---|---|
| `app.py` | Streamlit 9-tab dashboard: graph visualization, benchmark charts, RL curves, security log |

### `src/data/`
| Module | One-Liner |
|---|---|
| `dataset_generator.py` | Generates 10-user × 30-day synthetic event logs with Gemma 2B or rule-based fallback |
| `context_encoder.py` | 3-layer MLP (35→128→64→64) converting OS events into 64-dim situational embeddings |
| `event_simulator.py` | Replays a saved user event JSON file through EventBus one event at a time |

### `src/explainability/`
| Module | One-Liner |
|---|---|
| `reasoning_engine.py` | Pure-function reason-list generator for preload, promotion, demotion, flush decisions |
| `prediction_explainer.py` | Builds full explanation objects combining graph data, tier info, and reason strings |
| `decision_trace.py` | Records every GraphMind decision in an ordered audit trail |

### `src/graph_playback/`
| Module | One-Liner |
|---|---|
| `timeline_engine.py` | Manages ordered sequence of daily graph snapshots for seek/step playback |
| `snapshot_manager.py` | Saves and loads BehaviouralGraph state snapshots to/from `results/snapshots/` |
| `graph_animator.py` | Computes snapshot diffs and generates animation frame metadata for dashboard |

### `src/prefetch/`
| Module | One-Liner |
|---|---|
| `daemon.py` | APScheduler 15-min cycle prefetch + context-aware triggers (headphones, calendar, battery) |

### `src/rl/`
| Module | One-Liner |
|---|---|
| `environment.py` | Custom Gymnasium env: 68-dim obs, 31 discrete actions, one-day episode |
| `trainer.py` | PPO training manager for all 10 users, saves policies, collects metrics |
| `reward.py` | Computes scalar reward from cache hits, thrash events, battery, speed gain (pure function) |
| `evaluation.py` | Evaluates trained PPO policy against baselines; generates policy_comparison.csv |

### `src/security/`
| Module | One-Liner |
|---|---|
| `context_boundary.py` | Detects sensitive→consumer transitions and flushes HOT cache of sensitive nodes |
| `classification_guard.py` | Classifies app packages via taxonomy; isolates unknowns as `unknown_sensitive` |
| `security_visualizer.py` | Generates security flush timeline and category-transition heatmaps |

---

## 13. Data Flow (End-to-End)

### Simulation Mode

```
scripts/generate_dataset.py
         │
         ▼
data/synthetic/users/user_NN.json      (30 days × ~80 events)
         │
         ▼
EventSimulator.step()                  (replays one event at a time)
         │  publishes TOPIC_APP_LAUNCHED
         ▼
EventBus  ──────────────────────────────────────────────────────────────────
    │           │              │                 │               │
    ▼           ▼              ▼                 ▼               ▼
BehaviouralGraph  MemoryManager  DriftDetectorAgent  ContextBoundaryEnforcer  PrefetchDaemon
 (update node,     (check tier,    (record app_id      (check category         (update
  update edge,      promote HOT,    into windows)       transition, may          current_node)
  normalize)        publish hit/miss)                   flush HOT)
         │
         ▼
LangGraph Orchestrator.run_day(day)
    │
    ├─ graph_manager → prioritize HOT (Gemma or access_count sort)
    ├─ drift_detector → compute KL → update state["kl_divergence"]
    │       │
    │  [KL > 0.3] ──► rl_trainer → load PPO, spike LR, optional fine-tune
    │       │
    │  [KL ≤ 0.3] ──────────────────────────────────────────────►
    │                                                            │
    ├─ prefetch → daemon.run_prefetch_cycle() → rebuild WARM, promote top-2 HOT
    └─ security → log flush events, enforce retention policy
         │
         ▼
results/user_NN_simulation_log.json    (complete 30-day state history)
         │
         ▼
Streamlit Dashboard                    (visualize graph, benchmarks, RL curves)
```

### Live Mode (Real Device)

```
Samsung Galaxy (USB)
         │  ADB
         ▼
TelemetryCollector._poll_loop()   (every 5 seconds)
    ├─ battery_collector.collect()
    ├─ usage_stats_collector.get_foreground_app()
    ├─ audio_collector.collect()
    ├─ screen_collector.collect()
    └─ calendar_collector.collect()
         │
         ▼
TelemetryEventAdapter.publish_app_launched(...)
         │  publishes TOPIC_APP_LAUNCHED
         ▼
EventBus  (same pipeline as simulation mode above)
```

---

## 14. Security Model

### Context Isolation
- GraphMind stores only 64-dim float embeddings — never raw user data or plaintext
- Sensitive context (financial, health, enterprise) nodes are flushed from HOT when user switches to social/entertainment
- Unknown apps default to `unknown_sensitive` category until explicitly classified

### Data Retention Limits
```
HOT tier:    500-event rolling window
WARM tier:   2000-event rolling window
COLD tier:   15-day age limit (NODE_EVICTION_DAYS)
Trace log:   1000-event rolling window
```

### Privacy Guarantees
- All processing is on-device (no cloud calls)
- No user data leaves the device
- Graph contains behavioral patterns only (app IDs + timing), not content
- Flush events are logged locally only

### Security Flush Accuracy (from benchmarks)
- **100% flush accuracy** — every triggered flush removed ≥1 node
- **0% false flush rate** — no unnecessary flushes in simulation
- **~96–129 flushes per 1000 events** (reflecting the frequency of sensitive→consumer transitions in synthetic data)

---

## 15. Agentic AI Practices (ax.md summary)

### ✅ What Worked

| Practice | Implementation | Outcome |
|---|---|---|
| **Agentic Workflow** | LangGraph StateGraph with 5 autonomous agents | Clean separation; agents testable independently |
| **Statistical Reasoning** | KL divergence drift detection | Threshold 0.3 reliably detected genuine pattern shifts |
| **RL Planning** | PPO reward signals for future cache optimization | Outperformed all baselines by 29+ pp hit rate |
| **Tool Chaining via EventBus** | `app_launched → graph → memory → prefetch → security` | Fully decoupled; any module replaceable |
| **Multi-Agent Orchestration** | Conditional routing: drift → RL trainer (only when needed) | Battery-preserving; RL only runs on genuine drift |
| **Three-Tier Memory** | HOT/WARM/COLD hierarchy | +18% cache hit rate vs LMKD baseline |
| **Rule-Based Fallback** | For Gemma and dataset generation | Reproducible, fast, no GPU required |

### ❌ What Did NOT Work

| Issue | Problem | Fix Applied |
|---|---|---|
| **Gemma 2B real-time** | Too slow for sub-100ms decisions on CPU | Relegated to offline/periodic tasks; rule-based fallback |
| **PPO gradient instability** | Long 80-event episodes caused instability | Reduced `n_steps=512`; shorter rollout buffer |
| **SQLite COLD tier scale** | Bottleneck with 10 simultaneous users | Accepted in simulation; RocksDB planned for production |
| **LangGraph 0.1.14 API** | Conditional edge API changed between 0.1.x versions | Pinned to exact 0.1.14 |
| **PyVis in Streamlit** | Large graph rendering slow | Limited to 50 nodes/100 edges for rendering |
| **Concurrent encoder fine-tuning** | ContextEncoder + PPO simultaneous training = instability | Encoder weights frozen during PPO |

---

## 16. 10-User Personas

| User ID | Persona | Sleep Pattern | Peak Hours | Top Apps |
|---|---|---|---|---|
| `user_00` | University student | Irregular | 10, 14, 22 | YouTube, Instagram, Notes, Food Delivery, Spotify |
| `user_01` | Office commuter professional | Regular | 7, 12, 18 | Maps, Gmail, LinkedIn, Slack, News |
| `user_02` | Night shift nurse | Inverted | 0, 6, 20 | Samsung Health, WhatsApp, Calendar, Maps, Banking |
| `user_03` | Work-from-home developer | Flexible | 9, 15, 21 | GitHub, Slack, Browser, Spotify, Docs |
| `user_04` | Retired senior | Early | 6, 10, 16 | News, Gallery, Messaging, Video Call, Health |
| `user_05` | Frequent business traveler | Variable | 5, 13, 20 | Maps, Airline App, Gmail, Booking, Expense |
| `user_06` | Stay-at-home parent | Early + fragmented | 7, 12, 20 | Shopping, Calendar, WhatsApp, YouTube Kids, Food Delivery |
| `user_07` | University researcher | Late | 11, 16, 23 | Browser, Notes, PDF Reader, Gmail, Slack |
| `user_08` | Fitness enthusiast | Early consistent | 5, 12, 19 | Strava, Spotify, Maps, Samsung Health, Food Tracker |
| `user_09` | Social media content creator | Irregular | 9, 15, 22 | Instagram, TikTok, YouTube, Photo Editor, Scheduler |

---

## 17. App Taxonomy & App ID Vocabulary

### Category Mapping (from `data/app_taxonomy.json`)

| Category | Apps |
|---|---|
| **financial** | `net.one97.paytm`, `com.hdfcbank.new`, `com.phonepe.app`, `com.indiainfoline.trade` |
| **health** | `com.samsung.health`, `com.strava` |
| **social** | `com.instagram.android`, `com.whatsapp`, `com.linkedin.android` |
| **entertainment** | `com.google.youtube`, `com.spotify.music`, `com.netflix.mediaclient`, `com.tiktok.android` |
| **productivity** | `com.google.android.gm`, `com.slack.android`, `com.google.android.apps.docs` |
| **navigation** | `com.google.android.maps`, `com.makemytrip`, `com.booking` |
| **shopping** | `com.amazon.mShop.android`, `com.myntra.android`, `com.swiggy.android`, `com.zomato.android` |
| **utility** | `com.android.calendar`, `com.adobe.reader`, `com.google.android.apps.photos` |
| **enterprise** | `com.github.android`, `com.samsung.android.messaging` |

---

## 18. Baseline Policies Compared

### LMKD_Reactive
Simulates Android's Low Memory Killer Daemon.
- Keeps N most-recently-used apps in memory via LRU `OrderedDict`
- Evicts LRU when over capacity
- **No time-of-day awareness, no prediction**
- Cache hit rate: ~25–34% across users

### ART_StaticProfile
Simulates Android Runtime Baseline Profile.
- Builds static frequency profile from Days 1–7: `profile[time_bucket] = [app_id ranked by frequency]`
- After Day 7: **profile is FROZEN** — no further learning
- Represents AOT compilation of hot code paths
- Cache hit rate: ~0% in simulation (profile format mismatch with event replay)

### UsageStats_LRU
Simulates `UsageStatsManager` + LRU process cache.
- Identical to LMKD in simulation (both use recency, no transition modelling)
- Updates continuously but uses recency only
- Cache hit rate: same as LMKD (~25–34%)

### Bixby_Frequency
Simulates Samsung Bixby Routines / One UI app suggestions.
- Uses frequency counts per `(time_bucket, is_weekend)` key
- Updates continuously; time-of-day aware
- **No graph structure**, no transition chains, no RL
- Cache hit rate: ~17–23% across users

### GraphMind_RL
The full GraphMind system.
- Graph-based transition modelling
- PPO RL agent managing HOT/WARM/COLD tiers
- Drift detection + adaptive RL fine-tuning
- Context-aware prefetch (headphones, calendar, battery)
- Cache hit rate: **57–64%** across users

---

## 19. Installation & Quick Start

### Requirements
- Python 3.11 or 3.12
- 8 GB RAM recommended
- 20 GB disk (only if using Gemma 2B)
- CUDA GPU optional (CPU training fully supported)
- Windows 10/11 or Ubuntu 20.04+

### Setup Steps

```bash
# 1. Clone and enter project
git clone <repo-url>
cd Samsung

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env
# Edit .env: set DEVICE=cpu, LOG_LEVEL=INFO

# 5. Generate dataset (~2 min, idempotent)
python scripts/generate_dataset.py

# 6. Train PPO policies (30 min on CPU for all 10 users)
python scripts/train_rl.py --all --timesteps 200000
# Quick test: python scripts/train_rl.py --user user_00 --timesteps 50000

# 7. Run simulation (all users)
# Windows PowerShell:
foreach ($i in 0..9) {
    $user = "user_{0:D2}" -f $i
    python scripts/run_simulation.py --user $user
}

# 8. Run benchmarks
python scripts/run_benchmarks.py
# Output: results/benchmark_results.csv

# 9. Launch dashboard
streamlit run src/dashboard/app.py
# Opens: http://localhost:8501

# 10. Run tests
pytest tests/ -v
```

### Optional: Gemma 2B Setup
```bash
export HF_TOKEN=your_huggingface_token
pip install transformers huggingface_hub
python scripts/download_models.py
```

### Environment Variables (`.env`)
```
DEVICE=cpu              # or cuda:0 for GPU
LOG_LEVEL=INFO          # DEBUG for verbose output
WANDB_API_KEY=          # optional, W&B experiment tracking
WANDB_MODE=offline      # offline W&B logging
```

### Validation (Hardcheck)
```bash
python GRAPHMIND_HARDCHECK.py --phase 1   # Graph engine
python GRAPHMIND_HARDCHECK.py --phase 2   # Memory manager
python GRAPHMIND_HARDCHECK.py --phase 3   # RL environment
python GRAPHMIND_HARDCHECK.py --phase 4   # Agents + orchestrator
python GRAPHMIND_HARDCHECK.py --phase 5   # Benchmarks
python GRAPHMIND_HARDCHECK.py --phase 6   # Android + Security + Explainability
python GRAPHMIND_HARDCHECK.py --verbose   # All phases with details
```

---

## 20. Test Suite

20 test files covering all phases:

| Test File | What It Tests |
|---|---|
| `test_phase1_graph.py` | GraphNode, GraphEdge, add_node, add_edge, prune, evict, top-k scoring |
| `test_phase2_memory.py` | HOT/WARM/COLD promotion, demotion, eviction, flush_hot_by_category |
| `test_phase3_rl.py` | GraphMindEnv reset/step, reward function, observation shape |
| `test_phase4_agents.py` | DriftDetector KL, GraphManagerAgent fallback, Orchestrator state flow |
| `test_phase5_benchmarks.py` | BenchmarkEvaluator, baseline policies, compute_launch_speed_gain |
| `test_advanced_benchmarks.py` | Precision/recall/F1, latency percentiles, memory estimates, graph growth |
| `test_android_integration.py` | ADBConnector, DeviceDetector, TelemetryCollector, EventAdapter |
| `test_benchmark_fairness.py` | Ensures all baselines reset to identical initial state |
| `test_benchmark_provenance.py` | MEASURED/ESTIMATED tags on all metric columns |
| `test_cli_wizard.py` | Wizard device detection, interactive prompts, setup flow |
| `test_device_validation.py` | Device info parsing, serial detection |
| `test_drift_visualization.py` | Drift visualizer chart generation |
| `test_event_validation.py` | EventBus schema validation; reject malformed payloads |
| `test_explainability.py` | ReasoningEngine reason generation for all action types |
| `test_graph_playback.py` | TimelineEngine seek/step, SnapshotManager save/load |
| `test_rl_evaluation.py` | PPO evaluation against baselines, policy comparison |
| `test_scale.py` | Graph prediction time scales sub-linearly with user count |
| `test_security_hardening.py` | Context boundary detection, flush accuracy, retention policy |
| `test_security_visualization.py` | Security timeline and heatmap generation |
| `conftest.py` | Shared fixtures: EventBus.clear_all(), temp graph/memory, user profiles |

Run tests:
```bash
pytest tests/ -v                                  # all tests
pytest tests/test_phase1_graph.py -v             # specific phase
pytest tests/ --cov=src --cov-report=html        # with coverage
```

---

## 21. What Worked / What Did NOT Work

### ✅ What Worked

1. **EventBus pub-sub architecture** — kept all modules decoupled and independently testable; adding a new agent required zero changes to existing code
2. **KL divergence drift detection** — the 0.3 threshold correctly triggered RL retraining on genuine pattern shifts without false positives in simulation
3. **Three-tier memory model** — the HOT/WARM/COLD hierarchy delivered **+29 pp cache hit rate** vs LMKD baseline across all 10 users
4. **Rule-based dataset generation** — fully reproducible across all 10 personas without requiring GPU or Gemma access
5. **LangGraph for agent orchestration** — clean conditional routing API; drift→RL conditional edge was straightforward to implement
6. **Separate reward weight constants** — tuning α, β, γ, δ, ε, ζ without touching reward function logic was crucial for rapid iteration
7. **Context-triggered prefetch** — headphones-connected and calendar-approaching triggers give GraphMind an edge over pure time-based approaches
8. **Security flush accuracy at 100%** — the conservative `unknown_sensitive` classification prevented false negatives

### ❌ What Did NOT Work

1. **Gemma 2B for real-time decisions** — 2B parameters is too slow for sub-100ms decisions on CPU; restricted to offline/periodic tasks
2. **PPO on long episodes** — 80-event episodes caused gradient instability; fixed by setting `n_steps=512`
3. **SQLite COLD tier at scale** — Python's sqlite3 becomes a bottleneck with 10 simultaneous users; production would need RocksDB
4. **LangGraph 0.1.14 conditional edges** — API changed between minor versions; required strict version pinning
5. **PyVis graph rendering in Streamlit** — limited to 50 nodes; `generate_html()` + `st.components.v1.html()` is not ideal for large dynamic graphs
6. **Simultaneous ContextEncoder + PPO training** — caused training instability; encoder weights are now frozen during PPO

---

## 22. Future Work

1. **Replace Gemma 2B with a 1B model** optimized for Samsung Exynos NPU with INT4 quantization to fit in 2 GB RAM
2. **RocksDB COLD tier** replacing SQLite for O(log N) read/write at scale
3. **Federated learning** for cross-device pattern sharing without privacy violation (aggregate embeddings, not raw data)
4. **Real Samsung One UI integration** via system API (`android.app.usage.UsageStatsManager`) instead of ADB simulation
5. **Online PPO learning** — continuous learning from real events without full retraining cycles
6. **App category auto-discovery** — use on-device ML to classify unknown packages instead of defaulting to `unknown_sensitive`
7. **Personalized prefetch budget** — adapt `PREFETCH_TOP_K` dynamically based on user battery habits and charging patterns
8. **Cross-user cold start** — warm-start a new user's graph using aggregate patterns from similar personas

---

## 23. Dependencies

From `requirements.txt`:

| Package | Version | Role |
|---|---|---|
| `networkx` | 3.3 | Directed graph data structure for BehaviouralGraph |
| `numpy` | latest | Numerical arrays, KL divergence, reward computation |
| `pandas` | latest | Benchmark result DataFrames, CSV I/O |
| `scipy` | latest | `scipy.stats.entropy` for KL divergence |
| `gymnasium` | latest | Custom Gym environment for PPO training |
| `stable-baselines3` | latest | PPO implementation |
| `shimmy` | latest | SB3 + Gymnasium compatibility layer |
| `python-dotenv` | latest | `.env` file loading for settings |
| `langgraph` | **0.1.14** | LangGraph StateGraph for multi-agent orchestration (version pinned) |
| `streamlit` | latest | Dashboard web application |
| `pyvis` | latest | Interactive graph visualization in dashboard |
| `plotly` | latest | Charts and plots in dashboard |
| `apscheduler` | latest | Background scheduler for PrefetchDaemon 15-min cycles |
| `pytest` | latest | Test runner |
| `pytest-cov` | latest | Test coverage reporting |
| `torch` | latest | PyTorch for ContextEncoder MLP and Gemma (CPU mode) |
| `transformers` | optional | HuggingFace Transformers for Gemma 2B (optional) |

---

*Generated: 2026-06-02 — GraphMind complete reference document*
*Project: Samsung Axon Hackathon — GraphMind Predictive App Intelligence*
