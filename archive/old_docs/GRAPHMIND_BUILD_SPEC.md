# GRAPHMIND — MASTER BUILD SPECIFICATION
# File 1 of 2: Complete Implementation Guide
# Version: 1.0 | Target: Samsung AX Hackathon 2026 Phase 2
# Deadline: June 22, 2026 2:00 PM IST

---

## CRITICAL RULES FOR THE IMPLEMENTING LLM

1. NEVER re-import or re-connect a module that is already connected. All inter-file connections are defined in SECTION 3 (Connector Registry). Do not create new import paths not listed there.
2. NEVER create a file not listed in SECTION 2 (File Structure). If you need a helper, add it inside an existing file.
3. ALL functions must match EXACTLY the signatures defined in SECTION 4. Do not rename, do not add parameters, do not change return types.
4. Build in PHASE ORDER. Do not start Phase 2 until Phase 1 tests pass. Each phase has a GATE TEST — run it before proceeding.
5. All data flows through the EventBus singleton. No direct cross-module calls except those listed in the Connector Registry.
6. The synthetic dataset is generated ONCE in Phase 1 and never regenerated. All phases consume it from disk.
7. Config values live ONLY in config/settings.py. No magic numbers anywhere else.
8. Every function must have a docstring matching the spec. The LLM hardchecker in File 2 will verify docstring presence.

---

## SECTION 1: ENVIRONMENT SETUP

### 1.1 Python Version
Python 3.11.x (required, not 3.12+, Stable-Baselines3 compatibility)

### 1.2 Directory to work in
/graphmind/   (root of the project, all paths below are relative to this)

### 1.3 Install sequence (run in this exact order)

```bash
# Step 1: Create virtual environment
python3.11 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate   # Windows

# Step 2: Install PyTorch CPU (keeps it lightweight for simulation)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu

# Step 3: Install core dependencies
pip install \
  networkx==3.3 \
  numpy==1.26.4 \
  pandas==2.2.2 \
  scipy==1.13.0 \
  gymnasium==0.29.1 \
  stable-baselines3==2.3.2 \
  shimmy==1.3.0

# Step 4: Install LLM and ML dependencies
pip install \
  transformers==4.41.2 \
  peft==0.11.1 \
  accelerate==0.30.0 \
  huggingface-hub==0.23.2 \
  sentencepiece==0.2.0 \
  protobuf==4.25.3

# Step 5: Install agentic framework
pip install \
  langgraph==0.1.14 \
  langchain==0.2.1 \
  langchain-core==0.2.1 \
  langchain-community==0.2.1

# Step 6: Install dashboard and utils
pip install \
  streamlit==1.35.0 \
  pyvis==0.3.2 \
  plotly==5.22.0 \
  matplotlib==3.9.0 \
  wandb==0.17.0 \
  apscheduler==3.10.4 \
  python-dotenv==1.0.1 \
  pytest==8.2.0 \
  pytest-cov==5.0.0

# Step 7: Verify installation
python -c "import torch; import networkx; import stable_baselines3; import langgraph; import streamlit; print('ALL DEPS OK')"
```

### 1.4 Model Downloads (run after Phase 1 dataset generation)

```bash
# Gemma 2B — requires HuggingFace token with Gemma access
# Set token: export HF_TOKEN=your_token_here
python scripts/download_models.py
```

### 1.5 Environment Variables
Create a file called `.env` in the project root:
```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key_or_disabled
WANDB_MODE=offline   # use offline to avoid network calls during dev
LOG_LEVEL=INFO
DEVICE=cpu
```

---

## SECTION 2: COMPLETE FILE STRUCTURE

```
graphmind/
│
├── .env                          # Environment variables (not committed)
├── requirements.txt              # Frozen from pip freeze after setup
├── README.md                     # GitHub README (fill template fields)
├── agents.md                     # Agentic AI practices documentation
├── LICENSE                       # Apache 2.0
│
├── config/
│   └── settings.py               # ALL constants and config values live here
│
├── data/
│   ├── synthetic/
│   │   ├── users/                # One JSON file per user: user_00.json ... user_09.json
│   │   └── metadata.json         # Dataset summary: user count, days, event counts
│   ├── base_graphs/              # Serialized initial graphs per user (generated Phase 2)
│   └── app_taxonomy.json         # Security: app category definitions
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── event_bus.py          # Singleton EventBus — central nervous system
│   │   ├── graph_engine.py       # Graph data structure, node/edge CRUD
│   │   └── memory_manager.py     # Three-tier HOT/WARM/COLD manager
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_generator.py  # Synthetic 10-user dataset generation via Gemma 2B
│   │   ├── event_simulator.py    # Replays dataset as time-stepped event stream
│   │   └── context_encoder.py    # OS event tuple → 64-dim embedding (MLP)
│   │
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── environment.py        # Custom Gymnasium env wrapping simulator + memory
│   │   ├── trainer.py            # PPO training loop using Stable-Baselines3
│   │   └── reward.py             # Reward function: R = α*cache_hit + β*speed - γ*thrash - δ*battery + ε*friction
│   │
│   ├── prefetch/
│   │   ├── __init__.py
│   │   └── daemon.py             # Pre-fetch daemon: 15-min cycle, battery/time/event gates
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   └── context_boundary.py   # Context isolation: detect sensitive transitions, flush HOT cache
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph_manager_agent.py   # LangGraph agent: node/edge decisions via Gemma 2B
│   │   ├── rl_trainer_agent.py      # LangGraph agent: PPO training and weight updates
│   │   ├── prefetch_agent.py        # LangGraph agent: pre-fetch scheduling
│   │   ├── drift_detector_agent.py  # LangGraph agent: KL-divergence monitoring
│   │   ├── security_agent.py        # LangGraph agent: context boundary enforcement
│   │   └── orchestrator.py          # LangGraph state machine wiring all 5 agents
│   │
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── baselines.py          # 4 baseline policies: LMKD, ART, UsageStats-LRU, Bixby
│   │   └── evaluator.py          # Runs all 5 policies, produces comparison metrics
│   │
│   └── dashboard/
│       ├── __init__.py
│       └── app.py                # Streamlit dashboard: graph viz, RL curves, security log
│
├── scripts/
│   ├── download_models.py        # Downloads Gemma 2B from HuggingFace
│   ├── generate_dataset.py       # Entry point: runs dataset_generator.py
│   ├── train_rl.py               # Entry point: runs RL training for all 10 users
│   ├── run_simulation.py         # Entry point: runs full simulation for one user
│   ├── run_benchmarks.py         # Entry point: runs evaluator.py, saves results CSV
│   └── run_dashboard.py          # Entry point: streamlit run src/dashboard/app.py
│
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_phase1_graph.py      # Phase 1 gate tests
│   ├── test_phase2_memory.py     # Phase 2 gate tests
│   ├── test_phase3_rl.py         # Phase 3 gate tests
│   ├── test_phase4_agents.py     # Phase 4 gate tests
│   └── test_phase5_full.py       # Phase 5 end-to-end gate tests
│
└── docs/
    ├── architecture.md           # Technical architecture document
    ├── installation.md           # Installation guide
    ├── user_guide.md             # How to run the system
    ├── ax.md                     # REQUIRED: Agentic AI practices detail
    └── screenshots/              # Screenshots added after dashboard runs
```

---

## SECTION 3: CONNECTOR REGISTRY
## ALL IMPORT CONNECTIONS ARE DEFINED HERE. IMPLEMENT THESE EXACTLY. DO NOT ADD MORE.

```
FORMAT: SOURCE_FILE imports FROM TARGET_FILE as ALIAS
All connections are ONE-TIME. If already imported, do not re-import.

TIER 0 — Config (imported by everything, never imports from src/)
  config/settings.py  →  imports: os, dotenv only

TIER 1 — Core (imported by all src/ modules, never imports from agents/ or dashboard/)
  src/core/event_bus.py       →  imports: config.settings, threading, queue, logging
  src/core/graph_engine.py    →  imports: config.settings, src.core.event_bus, networkx, numpy, pickle, logging
  src/core/memory_manager.py  →  imports: config.settings, src.core.event_bus, src.core.graph_engine, logging, sqlite3

TIER 2 — Data (imported by rl/, agents/, benchmarks/, dashboard/)
  src/data/dataset_generator.py  →  imports: config.settings, transformers, torch, json, os, logging, random
  src/data/event_simulator.py    →  imports: config.settings, src.core.event_bus, pandas, json, os, logging, time
  src/data/context_encoder.py    →  imports: config.settings, torch, numpy, logging

TIER 3 — RL (imported by agents/, benchmarks/, scripts/)
  src/rl/reward.py         →  imports: config.settings, numpy, logging
  src/rl/environment.py    →  imports: config.settings, src.core.event_bus, src.core.memory_manager, src.data.event_simulator, src.data.context_encoder, src.rl.reward, gymnasium, numpy, logging
  src/rl/trainer.py        →  imports: config.settings, src.rl.environment, stable_baselines3, wandb, os, logging

TIER 4 — Prefetch (imported by agents/, dashboard/)
  src/prefetch/daemon.py  →  imports: config.settings, src.core.event_bus, src.core.memory_manager, src.core.graph_engine, apscheduler, logging, threading

TIER 5 — Security (imported by agents/, dashboard/)
  src/security/context_boundary.py  →  imports: config.settings, src.core.event_bus, src.core.memory_manager, json, logging

TIER 6 — Agents (imported by orchestrator, scripts/, dashboard/)
  src/agents/graph_manager_agent.py   →  imports: config.settings, src.core.graph_engine, src.core.memory_manager, transformers, torch, logging
  src/agents/rl_trainer_agent.py      →  imports: config.settings, src.rl.trainer, src.core.event_bus, logging
  src/agents/prefetch_agent.py        →  imports: config.settings, src.prefetch.daemon, src.core.event_bus, logging
  src/agents/drift_detector_agent.py  →  imports: config.settings, src.core.event_bus, scipy.stats, numpy, logging
  src/agents/security_agent.py        →  imports: config.settings, src.security.context_boundary, src.core.event_bus, logging
  src/agents/orchestrator.py          →  imports: config.settings, langgraph, src.agents.graph_manager_agent, src.agents.rl_trainer_agent, src.agents.prefetch_agent, src.agents.drift_detector_agent, src.agents.security_agent, logging

TIER 7 — Benchmarks (imported by scripts/, dashboard/)
  src/benchmarks/baselines.py   →  imports: config.settings, src.data.event_simulator, numpy, pandas, logging
  src/benchmarks/evaluator.py   →  imports: config.settings, src.benchmarks.baselines, src.rl.environment, src.core.memory_manager, pandas, numpy, logging

TIER 8 — Dashboard (imported only by scripts/run_dashboard.py)
  src/dashboard/app.py  →  imports: config.settings, src.core.graph_engine, src.core.memory_manager, src.rl.trainer, src.benchmarks.evaluator, src.agents.orchestrator, streamlit, pyvis, plotly, pandas, json, os, logging

SCRIPTS (entry points only, import from src/)
  scripts/download_models.py   →  imports: config.settings, huggingface_hub, os
  scripts/generate_dataset.py  →  imports: src.data.dataset_generator
  scripts/train_rl.py          →  imports: config.settings, src.rl.trainer, src.data.event_simulator
  scripts/run_simulation.py    →  imports: config.settings, src.agents.orchestrator, src.data.event_simulator
  scripts/run_benchmarks.py    →  imports: config.settings, src.benchmarks.evaluator
  scripts/run_dashboard.py     →  imports: streamlit (via subprocess call)

TESTS
  tests/conftest.py            →  imports: pytest, config.settings, src.core.graph_engine, src.core.memory_manager, src.data.event_simulator
  tests/test_phase*.py         →  imports: pytest, conftest fixtures, relevant src modules only
```

---

## SECTION 4: FILE-BY-FILE FUNCTION SPECIFICATIONS

---

### FILE: config/settings.py
**Purpose:** Single source of truth for all constants. No functions, only variables.

```python
# config/settings.py

from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
USERS_DIR = os.path.join(SYNTHETIC_DIR, "users")
BASE_GRAPHS_DIR = os.path.join(DATA_DIR, "base_graphs")
APP_TAXONOMY_PATH = os.path.join(DATA_DIR, "app_taxonomy.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ── Dataset ────────────────────────────────────────────────────────────────
NUM_USERS = 10
SIMULATION_DAYS = 30
EVENTS_PER_DAY_MEAN = 80          # mean number of app events per simulated day
EVENTS_PER_DAY_STD = 20
RANDOM_SEED = 42

# ── Graph Engine ───────────────────────────────────────────────────────────
NODE_EMBEDDING_DIM = 64
EDGE_PRUNE_THRESHOLD = 0.05       # delete edge if transition prob < 5%
NODE_EVICTION_DAYS = 45           # evict node from COLD if inactive this many days
MAX_NODES_COLD = 2000             # hard cap on COLD graph size

# ── Memory Manager ─────────────────────────────────────────────────────────
HOT_TIER_CAPACITY = 30            # max nodes in HOT (simulated RAM)
WARM_TIER_CAPACITY = 150          # max nodes in WARM (simulated cache)
COLD_DB_PATH = os.path.join(DATA_DIR, "cold_graph.db")

# ── RL Training ────────────────────────────────────────────────────────────
PPO_TOTAL_TIMESTEPS = 200_000
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
RL_MODELS_DIR = os.path.join(MODELS_DIR, "rl_policies")

# Reward weights
REWARD_ALPHA = 1.0    # cache hit rate weight
REWARD_BETA = 0.8     # launch speed gain weight
REWARD_GAMMA = 0.5    # thrash penalty weight
REWARD_DELTA = 0.3    # battery cost weight
REWARD_EPSILON = 0.4  # friction saved weight

# ── Pre-fetch Daemon ───────────────────────────────────────────────────────
PREFETCH_INTERVAL_MINUTES = 15
PREFETCH_TOP_K = 5                # number of nodes to pre-warm each cycle
BATTERY_SUPPRESS_THRESHOLD = 20  # percent — suppress aggressive pre-fetch below this

# ── Drift Detection ────────────────────────────────────────────────────────
DRIFT_WINDOW_SIZE = 100           # number of recent transitions to track
DRIFT_KL_THRESHOLD = 0.3          # KL divergence above this triggers learning rate spike
DRIFT_LR_SPIKE_MULTIPLIER = 5.0   # multiply learning rate by this on drift

# ── Security ───────────────────────────────────────────────────────────────
SENSITIVE_CATEGORIES = ["financial", "health", "enterprise", "government"]
CONSUMER_CATEGORIES = ["social", "entertainment", "shopping", "gaming"]
# Transition from sensitive → consumer triggers cache flush

# ── Gemma Model ────────────────────────────────────────────────────────────
GEMMA_MODEL_ID = "google/gemma-2b"
GEMMA_LOCAL_PATH = os.path.join(MODELS_DIR, "gemma-2b")
GEMMA_MAX_NEW_TOKENS = 128
GEMMA_DEVICE = os.getenv("DEVICE", "cpu")

# ── Dashboard ──────────────────────────────────────────────────────────────
DASHBOARD_PORT = 8501
DASHBOARD_REFRESH_SECONDS = 5

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ── Baseline Names (used as dict keys throughout) ──────────────────────────
BASELINE_LMKD = "LMKD_Reactive"
BASELINE_ART = "ART_StaticProfile"
BASELINE_LRU = "UsageStats_LRU"
BASELINE_BIXBY = "Bixby_Frequency"
BASELINE_GRAPHMIND = "GraphMind_RL"
```

---

### FILE: src/core/event_bus.py
**Purpose:** Singleton publish-subscribe bus. All inter-module communication goes through this. Prevents direct cross-module coupling.

```python
FUNCTIONS TO IMPLEMENT:

class EventBus (singleton):
    """
    Thread-safe singleton event bus. All modules publish and subscribe here.
    Use EventBus.get_instance() to get the single instance.
    NEVER instantiate EventBus() directly after the first call.
    """

    def get_instance() -> EventBus:
        """
        Class method. Returns the single EventBus instance.
        Creates it on first call, returns existing on subsequent calls.
        Thread-safe using a lock.
        Returns: EventBus singleton instance.
        """

    def subscribe(topic: str, callback: callable) -> None:
        """
        Register a callback to be called when topic is published.
        topic: string event name, e.g. "app_launched", "battery_updated", "drift_detected"
        callback: function(payload: dict) -> None
        Multiple callbacks can be registered for the same topic.
        """

    def publish(topic: str, payload: dict) -> None:
        """
        Publish an event to all subscribers of topic.
        payload: dictionary of event data. Always include {"timestamp": float} key.
        Calls all registered callbacks synchronously in subscription order.
        Logs the publish at DEBUG level: f"EventBus: {topic} -> {list(payload.keys())}"
        """

    def unsubscribe(topic: str, callback: callable) -> None:
        """
        Remove a specific callback from a topic.
        No-op if callback not registered for topic.
        """

    def clear_all() -> None:
        """
        Remove all subscriptions. Used in tests only to reset state between tests.
        """

TOPICS (these are the string constants for topics — define them at module level):
    TOPIC_APP_LAUNCHED = "app_launched"
    TOPIC_APP_CLOSED = "app_closed"
    TOPIC_BATTERY_UPDATED = "battery_updated"
    TOPIC_HEADPHONES_CONNECTED = "headphones_connected"
    TOPIC_CALENDAR_EVENT = "calendar_event_approaching"
    TOPIC_NODE_PROMOTED = "node_promoted_to_hot"
    TOPIC_NODE_DEMOTED = "node_demoted_from_hot"
    TOPIC_CACHE_HIT = "cache_hit"
    TOPIC_CACHE_MISS = "cache_miss"
    TOPIC_DRIFT_DETECTED = "drift_detected"
    TOPIC_SECURITY_FLUSH = "security_cache_flush"
    TOPIC_PREFETCH_TRIGGERED = "prefetch_triggered"
    TOPIC_RL_WEIGHT_UPDATED = "rl_weight_updated"

PAYLOAD SCHEMAS (each publish must include these keys at minimum):
    TOPIC_APP_LAUNCHED:       {"timestamp": float, "app_id": str, "user_id": str, "battery": float, "time_of_day_bucket": int}
    TOPIC_BATTERY_UPDATED:    {"timestamp": float, "battery": float, "user_id": str}
    TOPIC_DRIFT_DETECTED:     {"timestamp": float, "kl_divergence": float, "user_id": str}
    TOPIC_SECURITY_FLUSH:     {"timestamp": float, "user_id": str, "reason": str, "flushed_node_ids": list}
    TOPIC_CACHE_HIT:          {"timestamp": float, "node_id": str, "tier": str, "user_id": str}
    TOPIC_CACHE_MISS:         {"timestamp": float, "node_id": str, "user_id": str}
```

---

### FILE: src/core/graph_engine.py
**Purpose:** Core graph data structure. Nodes are situation embeddings. Edges are 3D weighted directed connections. Handles all graph CRUD, pruning, eviction, serialization.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

class GraphNode:
    """
    Represents a single situation in the user's behavioural graph.
    """
    Fields:
        node_id: str              # UUID string, generated at creation
        embedding: np.ndarray     # shape (NODE_EMBEDDING_DIM,) = (64,)
        app_id: str               # e.g. "com.instagram.android"
        time_bucket: int          # 0-47 (30-min buckets in a 24hr day)
        battery_bucket: int       # 0-4 (0=0-20%, 1=20-40%, ..., 4=80-100%)
        context_flags: dict       # {"headphones": bool, "calendar_near": bool, "weekend": bool}
        last_seen_day: int        # simulation day of last access (for eviction)
        access_count: int         # total number of times this node was accessed
        category: str             # from app_taxonomy: "social", "financial", etc.


class GraphEdge:
    """
    Directed weighted edge between two nodes.
    """
    Fields:
        source_id: str
        target_id: str
        transition_prob: float    # [0.0, 1.0] — probability of going to target from source
        time_sensitivity: float   # [0.0, 1.0] — how time-dependent this transition is
        battery_cost: float       # [0.0, 1.0] — battery penalty for pre-fetching target


class BehaviouralGraph:
    """
    The main directed weighted graph. Wraps NetworkX DiGraph.
    One instance per user.
    """

    def __init__(user_id: str):
        """
        Initialize an empty graph for a user.
        user_id: string identifier, e.g. "user_00"
        Creates an internal nx.DiGraph().
        Subscribes to TOPIC_APP_LAUNCHED on EventBus to call self._on_app_launched().
        """

    def add_node(node: GraphNode) -> None:
        """
        Add a GraphNode to the graph. 
        If node_id already exists, update last_seen_day and access_count only.
        Publishes nothing.
        """

    def add_edge(source_id: str, target_id: str, transition_prob: float, time_sensitivity: float, battery_cost: float) -> None:
        """
        Add or update a directed edge between two existing nodes.
        If edge already exists, update all three weight values.
        Raises ValueError if source_id or target_id not in graph.
        """

    def update_edge_weights(source_id: str, target_id: str, delta_prob: float, delta_time: float, delta_battery: float) -> None:
        """
        Apply additive delta to edge weights. Clamp all values to [0.0, 1.0] after update.
        Raises ValueError if edge does not exist.
        """

    def get_node(node_id: str) -> GraphNode | None:
        """
        Return the GraphNode for node_id, or None if not found.
        """

    def get_edges_from(node_id: str) -> list[GraphEdge]:
        """
        Return all outgoing edges from node_id as a list of GraphEdge objects.
        Returns empty list if node not found or has no outgoing edges.
        """

    def get_top_k_next_nodes(current_node_id: str, k: int, battery_level: float) -> list[str]:
        """
        Return the top-k most likely next node_ids from current_node_id.
        Scoring: score = transition_prob - (battery_cost * (1 - battery_level/100))
        If battery_level < BATTERY_SUPPRESS_THRESHOLD, set k = max(1, k // 2).
        Sort edges by score descending, return top-k target node_ids.
        Returns empty list if current_node_id not in graph.
        """

    def prune_weak_edges() -> int:
        """
        Delete all edges where transition_prob < EDGE_PRUNE_THRESHOLD (0.05).
        Returns the number of edges deleted.
        Does NOT publish an event.
        """

    def evict_stale_nodes(current_day: int) -> int:
        """
        Delete all nodes where (current_day - last_seen_day) > NODE_EVICTION_DAYS (45).
        Also delete all edges connected to evicted nodes.
        Returns the number of nodes evicted.
        """

    def node_count() -> int:
        """Return total number of nodes in the graph."""

    def edge_count() -> int:
        """Return total number of edges in the graph."""

    def save_to_disk(path: str) -> None:
        """
        Serialize the entire graph to a pickle file at path.
        Creates parent directories if they don't exist.
        Raises IOError on failure.
        """

    def load_from_disk(path: str) -> None:
        """
        Load graph state from pickle file at path. Overwrites current state.
        Raises FileNotFoundError if path does not exist.
        """

    def get_graph_snapshot(day: int) -> dict:
        """
        Return a JSON-serializable snapshot of the graph for the dashboard.
        Returns: {
            "day": day,
            "user_id": self.user_id,
            "node_count": int,
            "edge_count": int,
            "nodes": [{"node_id": str, "app_id": str, "category": str, "access_count": int}, ...],
            "edges": [{"source": str, "target": str, "prob": float}, ...]
        }
        Truncates to max 200 nodes/500 edges for rendering performance.
        """

    def _on_app_launched(payload: dict) -> None:
        """
        PRIVATE. EventBus callback for TOPIC_APP_LAUNCHED.
        payload keys: app_id, user_id, battery, time_of_day_bucket
        If payload["user_id"] != self.user_id: return immediately.
        Find or create a GraphNode for this (app_id, time_of_day_bucket, battery_bucket) tuple.
        Update last_seen_day and access_count.
        If a previous node exists in this session, add/update edge from previous → current.
        Increment transition_prob by 0.01 on each occurrence (clamped to 1.0).
        """
```

---

### FILE: src/core/memory_manager.py
**Purpose:** Three-tier memory hierarchy. HOT = dict (simulated RAM), WARM = LRU cache (simulated L3/file cache), COLD = SQLite on disk. Manages promotion, demotion, and eviction.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

class MemoryManager:
    """
    Manages the three-tier memory hierarchy for one user's graph.
    HOT: Python dict, max HOT_TIER_CAPACITY (30) nodes.
    WARM: OrderedDict LRU, max WARM_TIER_CAPACITY (150) nodes.
    COLD: SQLite database, theoretically unlimited.
    """

    def __init__(user_id: str, graph: BehaviouralGraph):
        """
        Initialize tiers. Connect to/create SQLite COLD DB at COLD_DB_PATH.
        Create table: cold_nodes(user_id TEXT, node_id TEXT, serialized_node BLOB, last_seen_day INT)
        Subscribe to TOPIC_APP_LAUNCHED on EventBus to call self._on_app_launched().
        graph: BehaviouralGraph instance for this user. Stored as self.graph.
        """

    def promote_to_hot(node_id: str) -> bool:
        """
        Move node_id to HOT tier.
        If already in HOT: return True (no-op).
        If HOT is full (>= HOT_TIER_CAPACITY): evict the least-recently-used HOT node to WARM first.
        Then move node from WARM to HOT (or load from COLD/graph if not in WARM).
        Publishes TOPIC_NODE_PROMOTED with {"node_id": node_id, "tier": "hot", "user_id": user_id}.
        Returns True on success, False if node_id not found anywhere.
        """

    def demote_from_hot(node_id: str) -> bool:
        """
        Move node_id from HOT to WARM.
        If WARM is full: evict oldest WARM node to COLD.
        Publishes TOPIC_NODE_DEMOTED with {"node_id": node_id, "from_tier": "hot", "to_tier": "warm", "user_id": user_id}.
        Returns True on success, False if node_id not in HOT.
        """

    def is_in_hot(node_id: str) -> bool:
        """Return True if node_id is in the HOT tier."""

    def is_in_warm(node_id: str) -> bool:
        """Return True if node_id is in the WARM tier."""

    def get_hot_node_ids() -> list[str]:
        """Return list of all node_ids currently in HOT tier."""

    def get_warm_node_ids() -> list[str]:
        """Return list of all node_ids currently in WARM tier."""

    def flush_hot_by_category(category: str) -> list[str]:
        """
        Remove all HOT nodes whose GraphNode.category matches category.
        Demote them to WARM (or COLD if WARM is full).
        Returns list of flushed node_ids.
        Used by SecurityAgent on sensitive context transitions.
        """

    def rebuild_warm_from_graph(predicted_node_ids: list[str]) -> None:
        """
        Replace WARM tier content with the given predicted_node_ids.
        Load each node from COLD if not already in HOT or WARM.
        Called by PrefetchDaemon at the start of each session and every 15-min cycle.
        Demotes current WARM nodes to COLD before replacing.
        """

    def get_tier_stats() -> dict:
        """
        Return current tier statistics.
        Returns: {"hot_count": int, "warm_count": int, "cold_count": int,
                  "hot_capacity": int, "warm_capacity": int}
        """

    def check_and_publish_cache_result(node_id: str, user_id: str) -> str:
        """
        Check which tier node_id is in.
        Publish TOPIC_CACHE_HIT if found in HOT or WARM tier.
        Publish TOPIC_CACHE_MISS if not found.
        Returns: "hot", "warm", "cold", or "miss"
        """

    def _on_app_launched(payload: dict) -> None:
        """
        PRIVATE. EventBus callback for TOPIC_APP_LAUNCHED.
        If payload["user_id"] != self.user_id: return.
        Find the node_id for the launched app from self.graph.
        Call check_and_publish_cache_result() for that node_id.
        Call promote_to_hot() for that node_id.
        """

    def _evict_lru_from_warm_to_cold(node_id: str) -> None:
        """
        PRIVATE. Move a node from WARM to COLD SQLite.
        Serialize the GraphNode and store in cold_nodes table.
        Remove from WARM dict.
        """
```

---

### FILE: src/data/dataset_generator.py
**Purpose:** Generates the synthetic 10-user behavioural dataset using Gemma 2B. Run ONCE. Output saved to data/synthetic/users/.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

USER_PROFILES = [
    # List of 10 dicts defining each user's persona for generation
    # These are the FIXED profiles — do not randomize these
    {"user_id": "user_00", "persona": "university student", "sleep_pattern": "irregular", "peak_hours": [10, 14, 22], "top_apps": ["youtube", "instagram", "notes_app", "food_delivery", "music_app"]},
    {"user_id": "user_01", "persona": "office commuter professional", "sleep_pattern": "regular", "peak_hours": [7, 12, 18], "top_apps": ["maps", "email", "linkedin", "slack", "news_app"]},
    {"user_id": "user_02", "persona": "night shift nurse", "sleep_pattern": "inverted", "peak_hours": [0, 6, 20], "top_apps": ["health_app", "messaging", "calendar", "maps", "banking_app"]},
    {"user_id": "user_03", "persona": "work from home developer", "sleep_pattern": "flexible", "peak_hours": [9, 15, 21], "top_apps": ["github_app", "slack", "browser", "music_app", "productivity_app"]},
    {"user_id": "user_04", "persona": "retired senior", "sleep_pattern": "early", "peak_hours": [6, 10, 16], "top_apps": ["news_app", "gallery", "messaging", "video_call", "health_app"]},
    {"user_id": "user_05", "persona": "frequent business traveler", "sleep_pattern": "variable", "peak_hours": [5, 13, 20], "top_apps": ["maps", "airline_app", "email", "booking_app", "expense_app"]},
    {"user_id": "user_06", "persona": "stay at home parent", "sleep_pattern": "early_fragmented", "peak_hours": [7, 12, 20], "top_apps": ["shopping_app", "calendar", "messaging", "youtube_kids", "food_delivery"]},
    {"user_id": "user_07", "persona": "university researcher", "sleep_pattern": "late", "peak_hours": [11, 16, 23], "top_apps": ["browser", "notes_app", "pdf_reader", "email", "slack"]},
    {"user_id": "user_08", "persona": "fitness enthusiast", "sleep_pattern": "early_consistent", "peak_hours": [5, 12, 19], "top_apps": ["fitness_app", "music_app", "maps", "health_app", "food_tracker"]},
    {"user_id": "user_09", "persona": "social media content creator", "sleep_pattern": "irregular", "peak_hours": [9, 15, 22], "top_apps": ["instagram", "tiktok", "youtube", "photo_editor", "scheduling_app"]},
]

class DatasetGenerator:
    """
    Generates synthetic behavioural event logs for all 10 users.
    Uses Gemma 2B to generate realistic per-persona event sequences.
    Falls back to rule-based generation if Gemma not available (for testing).
    """

    def __init__():
        """
        Load Gemma 2B tokenizer and model from GEMMA_LOCAL_PATH.
        If model not found at GEMMA_LOCAL_PATH, try GEMMA_MODEL_ID from HuggingFace.
        If both fail, set self.use_fallback = True (rule-based generation).
        Log which mode is active.
        """

    def generate_all_users() -> None:
        """
        Generate event logs for all 10 users in USER_PROFILES.
        Creates USERS_DIR if it doesn't exist.
        For each user, calls generate_user_events() and saves to USERS_DIR/user_XX.json.
        Also generates and saves data/synthetic/metadata.json.
        Skips generation if output file already exists (idempotent).
        """

    def generate_user_events(profile: dict) -> list[dict]:
        """
        Generate SIMULATION_DAYS * EVENTS_PER_DAY_MEAN events for one user.
        profile: one entry from USER_PROFILES.
        
        Each event is a dict:
        {
            "day": int,           # 0 to SIMULATION_DAYS-1
            "timestamp": float,   # seconds since day start
            "app_id": str,        # e.g. "com.instagram.android"
            "battery": float,     # 0.0 to 100.0
            "time_bucket": int,   # 0-47 (30-min buckets)
            "headphones": bool,
            "calendar_event_in_mins": int | null,  # None or minutes until next event
            "weekend": bool,
            "category": str       # from APP_TAXONOMY lookup
        }
        
        If self.use_fallback = True: call _generate_fallback().
        Else: call _generate_with_gemma().
        """

    def _generate_with_gemma(profile: dict) -> list[dict]:
        """
        PRIVATE. Use Gemma 2B to generate daily app sequences for the given persona.
        Prompts Gemma with the persona description and asks for a JSON list of app sequences.
        Parses the JSON response. Falls back to _generate_fallback() if parsing fails.
        """

    def _generate_fallback(profile: dict) -> list[dict]:
        """
        PRIVATE. Rule-based synthetic generation.
        Uses profile["peak_hours"] and profile["top_apps"] to construct realistic sequences.
        Uses numpy random with seed = RANDOM_SEED + int(profile["user_id"][-2:]) for reproducibility.
        Generates realistic battery drain across the day (start 100%, drain by usage pattern).
        Returns list of events matching the schema in generate_user_events().
        """

    def _app_id_to_package(app_name: str) -> str:
        """
        PRIVATE. Convert human-readable app name to package-style ID.
        e.g. "instagram" → "com.instagram.android"
        Uses a hardcoded mapping dict. Returns "com.unknown.{app_name}" for unmapped names.
        """

    def _save_metadata() -> None:
        """
        PRIVATE. Save data/synthetic/metadata.json with:
        {"num_users": 10, "days_per_user": 30, "total_events": int,
         "generation_mode": "gemma" | "fallback", "created_at": ISO timestamp}
        """
```

---

### FILE: src/data/app_taxonomy.json
**Purpose:** Maps app package IDs to human-readable names and security categories. Create this as a static JSON file, not generated.

```json
CONTENT TO CREATE (data/app_taxonomy.json):
{
  "com.instagram.android": {"name": "Instagram", "category": "social"},
  "com.google.youtube": {"name": "YouTube", "category": "entertainment"},
  "com.spotify.music": {"name": "Spotify", "category": "entertainment"},
  "com.slack.android": {"name": "Slack", "category": "enterprise"},
  "com.google.android.gm": {"name": "Gmail", "category": "enterprise"},
  "com.linkedin.android": {"name": "LinkedIn", "category": "enterprise"},
  "com.google.android.maps": {"name": "Google Maps", "category": "utility"},
  "com.android.calendar": {"name": "Calendar", "category": "productivity"},
  "com.samsung.android.calendar": {"name": "Samsung Calendar", "category": "productivity"},
  "com.tiktok.android": {"name": "TikTok", "category": "social"},
  "com.whatsapp": {"name": "WhatsApp", "category": "social"},
  "com.netflix.mediaclient": {"name": "Netflix", "category": "entertainment"},
  "com.amazon.mShop.android": {"name": "Amazon", "category": "shopping"},
  "net.one97.paytm": {"name": "Paytm", "category": "financial"},
  "com.google.android.apps.photos": {"name": "Google Photos", "category": "productivity"},
  "com.github.android": {"name": "GitHub", "category": "enterprise"},
  "com.samsung.health": {"name": "Samsung Health", "category": "health"},
  "com.strava": {"name": "Strava", "category": "health"},
  "com.myntra.android": {"name": "Myntra", "category": "shopping"},
  "com.zomato.android": {"name": "Zomato", "category": "food"},
  "com.swiggy.android": {"name": "Swiggy", "category": "food"},
  "com.google.android.apps.docs": {"name": "Google Docs", "category": "productivity"},
  "com.adobe.reader": {"name": "Adobe Reader", "category": "productivity"},
  "com.phonepe.app": {"name": "PhonePe", "category": "financial"},
  "com.hdfcbank.new": {"name": "HDFC Bank", "category": "financial"},
  "com.indiainfoline.trade": {"name": "Trading App", "category": "financial"},
  "com.samsung.android.messaging": {"name": "Samsung Messages", "category": "social"},
  "com.booking": {"name": "Booking.com", "category": "travel"},
  "com.makemytrip": {"name": "MakeMyTrip", "category": "travel"},
  "unknown": {"name": "Unknown", "category": "utility"}
}
```

---

### FILE: src/data/event_simulator.py
**Purpose:** Replays a user's saved event log as a real-time stream, publishing EventBus events. This is the "Android OS" for the simulation.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

class EventSimulator:
    """
    Replays the saved synthetic event log for one user.
    Publishes events to the EventBus at each step.
    Tracks current day, time, battery for simulation state.
    """

    def __init__(user_id: str):
        """
        Load event log from USERS_DIR/{user_id}.json.
        Store as self.events: list of event dicts.
        Set self.current_event_index = 0.
        Set self.current_day = 0.
        Set self.bus = EventBus.get_instance().
        Raises FileNotFoundError if user file doesn't exist.
        """

    def step() -> dict | None:
        """
        Advance simulation by one event.
        Publish the current event to the EventBus as TOPIC_APP_LAUNCHED.
        Increment self.current_event_index.
        Returns the event dict that was published, or None if simulation is complete.
        """

    def step_day() -> list[dict]:
        """
        Advance simulation by all events in the next day.
        Calls step() for each event on the current day.
        Increments self.current_day.
        Returns list of all events published for that day.
        """

    def step_all() -> None:
        """
        Replay all events in the entire 30-day log.
        Calls step() for each event sequentially.
        Logs progress every 1000 events.
        """

    def reset() -> None:
        """
        Reset simulator to day 0, event 0.
        Clears any session state.
        """

    def get_current_state() -> dict:
        """
        Return current simulation state.
        Returns: {"user_id": str, "current_day": int, "current_event_index": int,
                  "total_events": int, "battery": float, "last_app_id": str | None}
        """

    def get_events_for_day(day: int) -> list[dict]:
        """
        Return all events for a specific day without publishing them.
        Used by benchmarks to get the ground truth sequence.
        """
```

---

### FILE: src/data/context_encoder.py
**Purpose:** Converts raw OS event tuples into 64-dim situation embeddings. These become graph node features.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

class ContextEncoder:
    """
    Lightweight MLP that encodes OS event tuples into 64-dim embeddings.
    Input: (app_id_onehot[30], time_bucket[1], battery_bucket[1], headphones[1], calendar_near[1], weekend[1]) = 35 dims
    Output: 64-dim embedding vector
    Model is initialized with random weights and updated during RL training.
    """

    def __init__():
        """
        Define the MLP architecture using PyTorch:
        Layer 1: Linear(35, 128) + ReLU
        Layer 2: Linear(128, 64) + ReLU
        Output:  Linear(64, 64)  (no activation — raw embedding)
        Load weights from MODELS_DIR/encoder.pt if file exists.
        Set to eval mode. Use GEMMA_DEVICE for device placement.
        """

    def encode(event: dict) -> np.ndarray:
        """
        Convert an event dict to a 64-dim numpy embedding.
        event keys: app_id (str), time_bucket (int 0-47), battery (float),
                    headphones (bool), calendar_event_in_mins (int|None), weekend (bool)
        
        Encoding steps:
        1. app_id → one-hot vector of size 30 (use APP_ID_VOCAB defined below)
        2. time_bucket → normalize to [0,1] by dividing by 47
        3. battery → normalize to [0,1] by dividing by 100
        4. headphones → float 0.0 or 1.0
        5. calendar_near → 1.0 if calendar_event_in_mins <= 30, else 0.0
        6. weekend → float 0.0 or 1.0
        Concatenate all into tensor of shape (35,), pass through MLP, return as numpy (64,).
        """

    def save_weights(path: str) -> None:
        """Save model state_dict to path."""

    def load_weights(path: str) -> None:
        """Load model state_dict from path. Raise FileNotFoundError if missing."""

APP_ID_VOCAB = [
    # Fixed list of 30 app IDs. Apps not in this list map to index 29 (unknown).
    "com.instagram.android", "com.google.youtube", "com.spotify.music",
    "com.slack.android", "com.google.android.gm", "com.linkedin.android",
    "com.google.android.maps", "com.android.calendar", "com.tiktok.android",
    "com.whatsapp", "com.netflix.mediaclient", "com.amazon.mShop.android",
    "net.one97.paytm", "com.google.android.apps.photos", "com.github.android",
    "com.samsung.health", "com.strava", "com.myntra.android",
    "com.zomato.android", "com.swiggy.android", "com.google.android.apps.docs",
    "com.adobe.reader", "com.phonepe.app", "com.hdfcbank.new",
    "com.samsung.android.messaging", "com.booking", "com.makemytrip",
    "com.indiainfoline.trade", "com.samsung.android.calendar", "unknown"
]
```

---

### FILE: src/rl/reward.py
**Purpose:** Computes the RL reward signal from simulation state. Pure function, no side effects.

```python
FUNCTIONS TO IMPLEMENT:

def compute_reward(
    cache_hits: int,
    cache_misses: int,
    thrash_events: int,
    battery_consumed: float,
    friction_saved: int,
    step_duration_seconds: float
) -> float:
    """
    Compute the scalar reward for one RL step.
    
    Formula: R = α*cache_hit_rate + β*speed_gain - γ*thrash_rate - δ*battery_cost + ε*friction_saved_rate
    
    Where:
        cache_hit_rate = cache_hits / max(1, cache_hits + cache_misses)   [0.0 to 1.0]
        speed_gain = min(1.0, friction_saved / max(1, cache_hits + cache_misses))
        thrash_rate = min(1.0, thrash_events / 10.0)  [normalize: 10 thrashes = max penalty]
        battery_cost = min(1.0, battery_consumed / 5.0)  [normalize: 5% drain = max penalty]
        friction_saved_rate = min(1.0, friction_saved / max(1, cache_hits + cache_misses))
    
    α = REWARD_ALPHA = 1.0
    β = REWARD_BETA = 0.8
    γ = REWARD_GAMMA = 0.5
    δ = REWARD_DELTA = 0.3
    ε = REWARD_EPSILON = 0.4
    
    Returns: float reward value (can be negative if thrash + battery high)
    Logs the breakdown at DEBUG level.
    """

def compute_episode_summary(rewards: list[float]) -> dict:
    """
    Compute summary statistics for a training episode.
    rewards: list of per-step reward values.
    Returns: {"mean": float, "min": float, "max": float, "total": float, "steps": int}
    """
```

---

### FILE: src/rl/environment.py
**Purpose:** Custom Gymnasium environment wrapping the simulator and memory manager. This is what PPO trains on.

```python
CLASSES TO IMPLEMENT:

class GraphMindEnv(gymnasium.Env):
    """
    Custom Gymnasium environment for RL training.
    
    Observation space: Box(shape=(35 + HOT_TIER_CAPACITY + 3,), dtype=float32)
        = context_embedding(35) + hot_tier_occupancy(30) + [battery, time_bucket_norm, cache_hit_rate_recent]
        Total: 68 dimensions
    
    Action space: Discrete(HOT_TIER_CAPACITY + 1)
        Actions 0 to 28: promote node at hot_tier_index to front (signal to prioritize)
        Action 29: "no-op / run prune cycle"
        Action 30: "emergency: demote bottom half of HOT to WARM"
    
    Episode: one simulated day (all events for one day for one user)
    """

    def __init__(user_id: str):
        """
        Initialize the environment for a specific user.
        Create EventSimulator(user_id).
        Create BehaviouralGraph(user_id) and MemoryManager(user_id, graph).
        Load graph from BASE_GRAPHS_DIR/{user_id}_base.pkl if exists, else start empty.
        Set observation_space and action_space per spec above.
        Initialize counters: self.cache_hits=0, self.cache_misses=0, self.thrash_events=0, self.battery_start=100.0
        Subscribe to TOPIC_CACHE_HIT to increment self.cache_hits.
        Subscribe to TOPIC_CACHE_MISS to increment self.cache_misses.
        """

    def reset(seed: int | None = None) -> tuple[np.ndarray, dict]:
        """
        Reset to start of a new day (or day 0 if first call).
        Advance to the next unprocessed day.
        Reset step counters.
        Returns (initial_observation, {})
        """

    def step(action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step: publish one event via simulator.step(), apply the action,
        compute reward.
        
        Action interpretation:
            0-28: promote corresponding HOT node index to priority front
            29: call graph.prune_weak_edges()
            30: call memory_manager to demote bottom 15 HOT nodes to WARM
        
        Returns: (observation, reward, terminated, truncated, info)
            terminated = True when day's events are exhausted
            truncated = False always
            info = {"cache_hits": int, "cache_misses": int, "day": int}
        """

    def _get_observation() -> np.ndarray:
        """
        PRIVATE. Build the 68-dim observation vector from current state.
        Use zeros for context embedding if no event has been published yet.
        Cache hit rate = cache_hits / max(1, cache_hits + cache_misses) for last 50 steps.
        """

    def render() -> None:
        """No-op. Required by Gymnasium interface."""

    def close() -> None:
        """Cleanup. Unsubscribe EventBus callbacks."""
```

---

### FILE: src/rl/trainer.py
**Purpose:** Runs PPO training for all 10 users. Saves trained policy to disk.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

class RLTrainer:
    """
    Manages PPO training for GraphMind across all users.
    """

    def __init__():
        """
        Create MODELS_DIR/rl_policies/ directory if needed.
        Initialize W&B run if WANDB_API_KEY is set, else log offline.
        Set self.trained_users = {} (dict of user_id -> model path)
        """

    def train_user(user_id: str, total_timesteps: int = PPO_TOTAL_TIMESTEPS) -> str:
        """
        Train a PPO agent for one user.
        Creates GraphMindEnv(user_id).
        Wraps with stable_baselines3.PPO using MlpPolicy.
        Training hyperparams from settings.py (PPO_LEARNING_RATE, PPO_N_STEPS, etc.)
        Uses WandbCallback if W&B is active.
        Saves model to RL_MODELS_DIR/{user_id}_ppo.zip.
        Returns the save path.
        Logs training start and completion.
        """

    def train_all_users() -> dict:
        """
        Train PPO for all 10 users in USER_PROFILES order.
        Calls train_user() for each.
        Returns dict: {user_id: model_path}
        Logs total training time at completion.
        """

    def load_policy(user_id: str) -> stable_baselines3.PPO | None:
        """
        Load a saved PPO policy from RL_MODELS_DIR/{user_id}_ppo.zip.
        Returns the PPO model object.
        Returns None if file doesn't exist.
        """

    def get_training_curves() -> dict:
        """
        Return training curve data for dashboard rendering.
        Reads from W&B local logs or from saved CSV in RESULTS_DIR/training_curves.csv.
        Returns: {"user_id": [{"step": int, "reward": float}, ...], ...}
        Returns empty dict if no training data found.
        """
```

---

### FILE: src/prefetch/daemon.py
**Purpose:** Background daemon that proactively warms the HOT/WARM cache based on predicted next nodes.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

class PrefetchDaemon:
    """
    Runs periodic pre-fetching of predicted next nodes into HOT tier.
    Triggered by time, events, and context signals.
    """

    def __init__(user_id: str, graph: BehaviouralGraph, memory_manager: MemoryManager):
        """
        Store references. Do NOT start the scheduler here.
        Subscribe to EventBus:
            TOPIC_APP_LAUNCHED → _on_app_launched()
            TOPIC_HEADPHONES_CONNECTED → _on_headphones_connected()
            TOPIC_CALENDAR_EVENT → _on_calendar_event()
            TOPIC_BATTERY_UPDATED → _on_battery_updated()
        Set self.current_battery = 100.0
        Set self.current_node_id = None
        Set self.scheduler = None (initialized in start())
        """

    def start() -> None:
        """
        Start the APScheduler background scheduler.
        Add a job: call run_prefetch_cycle() every PREFETCH_INTERVAL_MINUTES minutes.
        Start scheduler.
        Log: "PrefetchDaemon started for user {user_id}"
        """

    def stop() -> None:
        """Shutdown the scheduler gracefully."""

    def run_prefetch_cycle() -> list[str]:
        """
        Main prefetch logic. Called every 15 minutes.
        1. If self.current_battery < BATTERY_SUPPRESS_THRESHOLD: k = 2, else k = PREFETCH_TOP_K
        2. If self.current_node_id is None: return []
        3. Call graph.get_top_k_next_nodes(self.current_node_id, k, self.current_battery)
        4. Call memory_manager.rebuild_warm_from_graph(predicted_ids)
        5. For top 2 predicted nodes: call memory_manager.promote_to_hot()
        6. Publish TOPIC_PREFETCH_TRIGGERED with {"user_id": user_id, "prefetched_ids": list, "battery": float}
        7. Returns list of prefetched node_ids.
        """

    def _on_app_launched(payload: dict) -> None:
        """PRIVATE. Update self.current_node_id from the launched app's node."""

    def _on_battery_updated(payload: dict) -> None:
        """PRIVATE. Update self.current_battery."""

    def _on_headphones_connected(payload: dict) -> None:
        """PRIVATE. Immediately promote music/entertainment nodes to HOT."""

    def _on_calendar_event(payload: dict) -> None:
        """
        PRIVATE. If event in <= 30 minutes:
        Identify nodes related to productivity/enterprise apps.
        Promote top-3 to HOT immediately.
        """
```

---

### FILE: src/security/context_boundary.py
**Purpose:** Detects sensitive-to-consumer context transitions and sanitizes the HOT cache.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

class ContextBoundaryEnforcer:
    """
    Monitors app transitions and enforces context isolation.
    When user moves from a SENSITIVE context (financial, health, enterprise)
    to a CONSUMER context (social, entertainment, shopping),
    flush HOT cache of all sensitive-category nodes.
    """

    def __init__(user_id: str, memory_manager: MemoryManager):
        """
        Load app_taxonomy from APP_TAXONOMY_PATH.
        Store memory_manager reference.
        Subscribe to TOPIC_APP_LAUNCHED → _on_app_launched().
        Set self.previous_category = None
        Set self.flush_log = [] (list of flush event dicts)
        """

    def check_transition(from_category: str, to_category: str) -> bool:
        """
        Determine if this transition requires a cache flush.
        Returns True if from_category in SENSITIVE_CATEGORIES AND to_category in CONSUMER_CATEGORIES.
        Returns False otherwise.
        """

    def enforce_boundary(from_category: str, to_category: str, timestamp: float) -> dict | None:
        """
        If check_transition() returns True:
            1. Flush all HOT nodes from SENSITIVE_CATEGORIES via memory_manager.flush_hot_by_category().
            2. Build a flush_event dict:
               {"timestamp": timestamp, "from_category": from_category,
                "to_category": to_category, "flushed_node_ids": list, "user_id": user_id}
            3. Append to self.flush_log.
            4. Publish TOPIC_SECURITY_FLUSH with the flush_event dict.
            5. Return the flush_event dict.
        Else: return None.
        """

    def get_flush_log() -> list[dict]:
        """Return all recorded flush events."""

    def get_app_category(app_id: str) -> str:
        """
        Look up category from app_taxonomy.
        Returns category string or "utility" if app_id not found.
        """

    def _on_app_launched(payload: dict) -> None:
        """
        PRIVATE. EventBus callback.
        Get category for payload["app_id"].
        Call enforce_boundary(self.previous_category, current_category, payload["timestamp"]).
        Update self.previous_category = current_category.
        """
```

---

### FILE: src/agents/orchestrator.py
**Purpose:** LangGraph state machine wiring all 5 agents together. This is the top-level coordinator.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

STATE SCHEMA (TypedDict):
class GraphMindState(TypedDict):
    user_id: str
    current_day: int
    current_event: dict | None
    battery: float
    kl_divergence: float
    cache_hit_rate: float
    security_flush_count: int
    last_agent: str
    messages: list[dict]   # running log of agent actions

class GraphMindOrchestrator:
    """
    LangGraph state machine coordinating all 5 agents.
    Runs one full simulation day as one orchestration cycle.
    """

    def __init__(user_id: str):
        """
        Initialize all 5 agents and their dependencies:
            - BehaviouralGraph(user_id) → shared across agents
            - MemoryManager(user_id, graph) → shared
            - PrefetchDaemon(user_id, graph, memory_manager)
            - ContextBoundaryEnforcer(user_id, memory_manager)
            - GraphManagerAgent(graph, memory_manager)
            - RLTrainerAgent(user_id)
            - PrefetchAgent(daemon)
            - DriftDetectorAgent(user_id)
            - SecurityAgent(enforcer)
        Build the LangGraph graph using build_graph().
        """

    def build_graph() -> langgraph.graph.StateGraph:
        """
        Build and compile the LangGraph StateGraph.
        
        Nodes: "graph_manager", "rl_trainer", "prefetch", "drift_detector", "security"
        
        Edges (sequential with conditional):
            START → graph_manager
            graph_manager → drift_detector
            drift_detector → rl_trainer (if kl_divergence > DRIFT_KL_THRESHOLD)
            drift_detector → prefetch (if kl_divergence <= DRIFT_KL_THRESHOLD)
            rl_trainer → prefetch
            prefetch → security
            security → END
        
        Returns compiled graph.
        """

    def run_day(day: int) -> GraphMindState:
        """
        Run one full simulation day through the state machine.
        Initializes state with current day and user context.
        Invokes the compiled LangGraph graph.
        Returns final state after all agents have run.
        """

    def run_full_simulation() -> list[GraphMindState]:
        """
        Run all SIMULATION_DAYS days sequentially.
        Returns list of daily state snapshots.
        Save snapshots to RESULTS_DIR/{user_id}_simulation_log.json.
        """
```

---

### FILE: src/agents/graph_manager_agent.py
**Purpose:** LangGraph agent node. Uses Gemma 2B to reason about which nodes to promote/demote. Logs reasoning.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

class GraphManagerAgent:
    """
    LangGraph agent that manages graph decisions using Gemma 2B reasoning.
    Gemma is given current HOT tier contents and asked which nodes to keep/evict.
    """

    def __init__(graph: BehaviouralGraph, memory_manager: MemoryManager):
        """
        Store references. Load Gemma tokenizer + model or use fallback.
        If GEMMA_LOCAL_PATH exists: load model. Else set self.use_llm = False.
        """

    def run(state: GraphMindState) -> GraphMindState:
        """
        Main agent function called by LangGraph.
        1. Get HOT tier contents from memory_manager.
        2. If self.use_llm: prompt Gemma with HOT tier context and time_of_day → get node priority decisions.
        3. Else: use rule-based priority (sort by access_count descending).
        4. Reorder HOT tier based on decisions.
        5. Run graph.prune_weak_edges() if current_day % 7 == 0.
        6. Append reasoning to state["messages"].
        7. Update state["last_agent"] = "graph_manager".
        8. Return updated state.
        """

    def _build_gemma_prompt(hot_nodes: list[GraphNode], time_of_day: int) -> str:
        """
        PRIVATE. Build a short prompt for Gemma describing current HOT tier nodes.
        Ask Gemma: "Given these apps in cache and time of day, which should be prioritized?"
        Return prompt string. Keep under 256 tokens for speed.
        """

    def _parse_gemma_response(response: str) -> list[str]:
        """
        PRIVATE. Parse Gemma's response to extract app names or node_ids to prioritize.
        Return list of node_ids in priority order.
        Falls back to original order if parsing fails.
        """
```

---

### FILE: src/agents/drift_detector_agent.py
**Purpose:** Monitors KL-divergence between recent and historical app transition distributions.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

class DriftDetectorAgent:
    """
    Tracks the distribution of app transitions over time.
    Computes KL divergence between a sliding window and historical baseline.
    Triggers learning rate spike if drift detected.
    """

    def __init__(user_id: str):
        """
        Set self.transition_history = deque(maxlen=DRIFT_WINDOW_SIZE * 2)
        Set self.recent_window = deque(maxlen=DRIFT_WINDOW_SIZE)
        Subscribe to TOPIC_APP_LAUNCHED → _record_transition().
        """

    def run(state: GraphMindState) -> GraphMindState:
        """
        Main agent function called by LangGraph.
        Compute KL divergence between recent_window and older half of transition_history.
        Update state["kl_divergence"] = computed KL value.
        If KL > DRIFT_KL_THRESHOLD: publish TOPIC_DRIFT_DETECTED.
        Append detection result to state["messages"].
        Return updated state.
        """

    def compute_kl_divergence() -> float:
        """
        Compute KL divergence between recent and historical transition distributions.
        Convert both windows to probability distributions over app_ids.
        Use scipy.stats.entropy(P, Q) for KL(P||Q).
        Returns 0.0 if insufficient data (< DRIFT_WINDOW_SIZE events).
        Add small epsilon (1e-10) to avoid log(0).
        """

    def _record_transition(payload: dict) -> None:
        """PRIVATE. EventBus callback. Record app_id into both deques."""
```

---

### FILE: src/agents/rl_trainer_agent.py

```python
class RLTrainerAgent:
    def __init__(user_id: str):
        """Store user_id. Load PPO policy from disk if exists."""

    def run(state: GraphMindState) -> GraphMindState:
        """
        If drift was detected (state["kl_divergence"] > DRIFT_KL_THRESHOLD):
            Spike learning rate: multiply current LR by DRIFT_LR_SPIKE_MULTIPLIER.
            Run 1000 additional PPO timesteps.
        Else: no-op (training runs in background via scripts/train_rl.py).
        Update state["last_agent"] = "rl_trainer".
        Return state.
        """
```

---

### FILE: src/agents/prefetch_agent.py

```python
class PrefetchAgent:
    def __init__(daemon: PrefetchDaemon):
        """Store daemon reference."""

    def run(state: GraphMindState) -> GraphMindState:
        """
        Call daemon.run_prefetch_cycle().
        Update state["cache_hit_rate"] from current memory_manager stats.
        Append prefetch log to state["messages"].
        Return state.
        """
```

---

### FILE: src/agents/security_agent.py

```python
class SecurityAgent:
    def __init__(enforcer: ContextBoundaryEnforcer):
        """Store enforcer reference."""

    def run(state: GraphMindState) -> GraphMindState:
        """
        Get flush_log from enforcer since last run.
        Update state["security_flush_count"].
        Append security events to state["messages"].
        Return state.
        """
```

---

### FILE: src/benchmarks/baselines.py
**Purpose:** Implements 4 baseline policies to compare against GraphMind.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

class BaselinePolicy:
    """Abstract base class for all baselines."""
    def predict_next_apps(current_app_id: str, context: dict) -> list[str]:
        """Return list of predicted next app_ids (ordered by confidence)."""
    def update(event: dict) -> None:
        """Update policy state with a new observed event."""
    def reset() -> None:
        """Reset policy to initial state."""
    def get_name() -> str:
        """Return BASELINE_* constant name."""


class LMKDReactiveBaseline(BaselinePolicy):
    """
    Simulates Android LMKD behavior: purely reactive, no prediction.
    Keeps the N most-recently-used apps in memory. Evicts LRU on overflow.
    No time-of-day awareness. No transition modelling.
    capacity: HOT_TIER_CAPACITY
    """
    def predict_next_apps(current_app_id, context) -> list[str]:
        """Returns top-5 most recently used apps regardless of context."""
    def update(event) -> None:
        """Add app_id to front of LRU queue. Evict tail if over capacity."""
    def get_name() -> str: return BASELINE_LMKD


class ARTStaticProfileBaseline(BaselinePolicy):
    """
    Simulates Android ART Baseline Profile behavior:
    Pre-warms the top-N most frequently launched apps per time-of-day bucket.
    Profile is built from Day 1-7 and then FROZEN (static, no further learning).
    Represents ART's AOT compilation of hot code paths.
    """
    def build_profile(events: list[dict]) -> None:
        """
        Build static frequency profile from first 7 days of events.
        profile[time_bucket] = [app_id ordered by frequency]
        """
    def predict_next_apps(current_app_id, context) -> list[str]:
        """Return profile[context["time_bucket"]] top-5."""
    def get_name() -> str: return BASELINE_ART


class UsageStatsLRUBaseline(BaselinePolicy):
    """
    Simulates Android UsageStatsManager + LRU process cache.
    Keeps recently-used apps warm. Updates continuously but uses recency only.
    No transition modelling (doesn't know that Instagram follows WhatsApp).
    """
    def predict_next_apps(current_app_id, context) -> list[str]:
        """Returns top-5 most recently used apps, context-agnostic."""
    def get_name() -> str: return BASELINE_LRU


class BixbyFrequencyBaseline(BaselinePolicy):
    """
    Simulates Samsung Bixby Routines / One UI app suggestions.
    Uses frequency counts per (time_bucket, day_of_week) pair.
    Updates continuously but no RL, no graph structure, no transition chains.
    """
    def predict_next_apps(current_app_id, context) -> list[str]:
        """Return top-5 most frequent apps for current (time_bucket, day_of_week)."""
    def get_name() -> str: return BASELINE_BIXBY
```

---

### FILE: src/benchmarks/evaluator.py
**Purpose:** Runs all 5 policies on all 10 users and produces comparative KPI numbers.

```python
CLASSES AND FUNCTIONS TO IMPLEMENT:

class BenchmarkEvaluator:
    """
    Runs all baselines + GraphMind on all 10 users.
    Measures: cache hit rate, launch speed gain, thrash events, battery overhead.
    """

    def __init__():
        """Initialize all 4 baselines. Load all 10 user datasets."""

    def run_all() -> pd.DataFrame:
        """
        For each user × each policy (5 total), replay 30-day event log.
        Measure at each event:
            - cache_hit: was the next app already in simulated warm/hot cache?
            - thrash: was an app evicted and then immediately needed again?
            - battery_cost: simulated % drain per pre-fetch operation
        
        Returns DataFrame with columns:
        [user_id, policy_name, day, cache_hit_rate, launch_speed_gain_pct,
         thrash_rate, battery_overhead_pct, graph_node_count]
        
        Save to RESULTS_DIR/benchmark_results.csv.
        """

    def run_user_policy(user_id: str, policy: BaselinePolicy, events: list[dict]) -> dict:
        """
        Run one policy on one user's full event log.
        Returns dict of aggregate metrics for this user-policy combination.
        """

    def compute_launch_speed_gain(cache_hit_rate: float, baseline_cache_hit_rate: float) -> float:
        """
        Estimate launch speed gain from cache hit rate improvement.
        Based on Android ART documentation: cache hit → ~30% faster cold start avoided.
        Formula: gain_pct = (cache_hit_rate - baseline_cache_hit_rate) * 30.0
        Returns percentage improvement (can be negative).
        """

    def print_summary_table() -> None:
        """
        Print a formatted comparison table to stdout.
        Columns: Policy, Avg Cache Hit %, Launch Speed Gain %, Thrash Rate %, Battery Overhead %
        Highlight GraphMind row.
        """

    def get_per_user_evolution() -> dict:
        """
        For GraphMind policy only, return cache hit rate by day for each user.
        Used by dashboard to show per-user graph evolution.
        Returns: {"user_00": [{"day": 0, "cache_hit_rate": 0.23}, ...], ...}
        """
```

---

### FILE: src/dashboard/app.py
**Purpose:** Streamlit dashboard. Run via `streamlit run src/dashboard/app.py`.

```python
LAYOUT SECTIONS TO IMPLEMENT:

Section 1 — Sidebar:
    User selector (dropdown: user_00 to user_09)
    Day slider (0 to 29)
    "Run Live Simulation" button
    "Run Benchmarks" button

Section 2 — Top Row (3 columns):
    Col 1: Metric card — Cache Hit Rate (current user, current day)
    Col 2: Metric card — Security Flushes (total for selected user)
    Col 3: Metric card — Graph Size (nodes / edges)

Section 3 — Graph Evolution Tab:
    Four PyVis graph renders: Day 1, Day 7, Day 14, Day 30 snapshots
    Load from RESULTS_DIR/{user_id}_snapshots.json if exists

Section 4 — Benchmark Comparison Tab:
    Load benchmark_results.csv from RESULTS_DIR
    Plotly bar chart: Cache Hit Rate by Policy (5 bars)
    Plotly line chart: Cache Hit Rate over Days for GraphMind per user
    Table: Policy × Metric comparison

Section 5 — RL Training Tab:
    Load training_curves from RLTrainer.get_training_curves()
    Plotly line chart: Reward over training steps per user

Section 6 — Security Log Tab:
    Table of all security flush events: timestamp, from_category, to_category, flushed_node_count
    Color-coded: financial→social = red, health→social = orange, enterprise→social = yellow

FUNCTION to implement:
def load_data(user_id: str, day: int) -> dict:
    """Load all pre-computed results for selected user/day from RESULTS_DIR."""

def render_pyvis_graph(snapshot: dict) -> str:
    """Convert graph snapshot dict to PyVis HTML. Return HTML string for st.components.html."""
```

---

## SECTION 5: PHASE-WISE EXECUTION PLAN

### PHASE 1 — Foundation (Days 1-4)
**Goal:** Graph engine, EventBus, and synthetic dataset working. Everything downstream depends on these.

**Files to implement in order:**
1. `config/settings.py` — copy verbatim from Section 4
2. `data/app_taxonomy.json` — copy verbatim from Section 4
3. `src/core/event_bus.py` — all functions
4. `src/core/graph_engine.py` — all functions
5. `src/core/memory_manager.py` — all functions
6. `src/data/dataset_generator.py` — start with _generate_fallback(), add Gemma later
7. `scripts/generate_dataset.py` — 5 lines, just calls DatasetGenerator().generate_all_users()

**Run to generate data:**
```bash
python scripts/generate_dataset.py
# Expected output: data/synthetic/users/user_00.json ... user_09.json
# Expected: data/synthetic/metadata.json
```

**PHASE 1 GATE TEST — run this before proceeding:**
```bash
pytest tests/test_phase1_graph.py -v
```

**What the gate test checks:**
- EventBus singleton returns same instance on two calls
- EventBus subscribe + publish calls callback correctly
- BehaviouralGraph add_node() stores node correctly
- BehaviouralGraph add_edge() with valid nodes succeeds
- BehaviouralGraph prune_weak_edges() removes edges below 0.05
- BehaviouralGraph evict_stale_nodes() removes nodes inactive > 45 days
- BehaviouralGraph save/load roundtrip preserves all data
- data/synthetic/users/user_00.json exists and has correct schema
- All 10 user files exist with >= 1000 events each

**PHASE 1 SUCCESS CRITERIA:**
```
PASSED tests/test_phase1_graph.py::test_event_bus_singleton
PASSED tests/test_phase1_graph.py::test_event_bus_pubsub
PASSED tests/test_phase1_graph.py::test_graph_add_node
PASSED tests/test_phase1_graph.py::test_graph_add_edge
PASSED tests/test_phase1_graph.py::test_graph_pruning
PASSED tests/test_phase1_graph.py::test_graph_eviction
PASSED tests/test_phase1_graph.py::test_graph_serialization
PASSED tests/test_phase1_graph.py::test_dataset_exists
PASSED tests/test_phase1_graph.py::test_dataset_schema
```

---

### PHASE 2 — Memory + Encoder (Days 5-7)
**Goal:** Three-tier memory manager and context encoder working.

**Files to implement:**
1. `src/data/context_encoder.py` — full implementation
2. `src/data/event_simulator.py` — full implementation

**PHASE 2 GATE TEST:**
```bash
pytest tests/test_phase2_memory.py -v
```

**What the gate test checks:**
- MemoryManager promotes node to HOT correctly
- MemoryManager evicts LRU from HOT to WARM when at capacity
- MemoryManager demotes from HOT to WARM correctly
- MemoryManager flush_hot_by_category removes correct nodes
- MemoryManager tier_stats returns correct counts
- ContextEncoder encode() returns shape (64,) numpy array
- ContextEncoder encode() is deterministic (same input → same output)
- EventSimulator loads user_00.json correctly
- EventSimulator step() publishes TOPIC_APP_LAUNCHED event
- EventSimulator step_day() advances day counter

---

### PHASE 3 — RL Training (Days 8-12)
**Goal:** RL environment, reward, and PPO training working for at least 1 user.

**Files to implement:**
1. `src/rl/reward.py`
2. `src/rl/environment.py`
3. `src/rl/trainer.py`
4. `scripts/train_rl.py`

**Run training for 1 user first:**
```bash
python scripts/train_rl.py --user user_00 --timesteps 50000
# Expected: models/rl_policies/user_00_ppo.zip
```

**PHASE 3 GATE TEST:**
```bash
pytest tests/test_phase3_rl.py -v
```

**What the gate test checks:**
- GraphMindEnv instantiates without error for user_00
- GraphMindEnv.reset() returns observation of shape (68,)
- GraphMindEnv.step() returns (obs, reward, terminated, truncated, info)
- reward.compute_reward() returns float in expected range
- reward.compute_reward() penalizes thrash events
- PPO model file exists at models/rl_policies/user_00_ppo.zip
- Loaded PPO model can predict action from observation

---

### PHASE 4 — Agents + Security + Prefetch (Days 13-17)
**Goal:** All 5 LangGraph agents wired and the security feature working.

**Files to implement (in order):**
1. `src/prefetch/daemon.py`
2. `src/security/context_boundary.py`
3. `src/agents/graph_manager_agent.py`
4. `src/agents/rl_trainer_agent.py`
5. `src/agents/prefetch_agent.py`
6. `src/agents/drift_detector_agent.py`
7. `src/agents/security_agent.py`
8. `src/agents/orchestrator.py`
9. `scripts/run_simulation.py`

**Run simulation for 1 user:**
```bash
python scripts/run_simulation.py --user user_00
# Expected: results/user_00_simulation_log.json
```

**PHASE 4 GATE TEST:**
```bash
pytest tests/test_phase4_agents.py -v
```

**What the gate test checks:**
- ContextBoundaryEnforcer.check_transition("financial", "social") returns True
- ContextBoundaryEnforcer.check_transition("social", "financial") returns False
- ContextBoundaryEnforcer.enforce_boundary() flushes correct nodes
- ContextBoundaryEnforcer.get_flush_log() returns list of events
- DriftDetectorAgent.compute_kl_divergence() returns 0.0 with no data
- DriftDetectorAgent.compute_kl_divergence() returns > 0.0 with divergent data
- GraphMindOrchestrator instantiates without error for user_00
- GraphMindOrchestrator.run_day(0) returns GraphMindState dict
- GraphMindState has all required keys: user_id, current_day, kl_divergence, cache_hit_rate, security_flush_count

---

### PHASE 5 — Benchmarks + Dashboard (Days 18-22)
**Goal:** All 10 users benchmarked, dashboard running, videos ready.

**Files to implement:**
1. `src/benchmarks/baselines.py` — all 4 baseline classes
2. `src/benchmarks/evaluator.py` — full evaluator
3. `src/dashboard/app.py` — full Streamlit dashboard
4. `scripts/run_benchmarks.py`
5. `scripts/run_dashboard.py`

**Run full pipeline:**
```bash
# Train all 10 users (takes longest)
python scripts/train_rl.py --all --timesteps 200000

# Run simulation for all users
for i in $(seq -f "%02g" 0 9); do
    python scripts/run_simulation.py --user user_${i}
done

# Run benchmarks
python scripts/run_benchmarks.py
# Expected: results/benchmark_results.csv

# Launch dashboard
python scripts/run_dashboard.py
# Expected: http://localhost:8501
```

**PHASE 5 GATE TEST:**
```bash
pytest tests/test_phase5_full.py -v
```

**What the gate test checks:**
- results/benchmark_results.csv exists with 50 rows (10 users × 5 policies)
- GraphMind achieves cache_hit_rate > LMKD in benchmark for >= 8/10 users
- GraphMind achieves cache_hit_rate > Bixby baseline for >= 8/10 users
- Security: at least 1 flush event recorded per user on average
- Graph node count < 1000 for all users after 30 days
- Dashboard app.py imports without error
- Streamlit app runs without crashing (subprocess check)

---

## SECTION 6: PREDETERMINED DATA FLOW CONTRACTS

Every inter-module data exchange is defined here. Any LLM implementing this must not deviate from these contracts.

### Contract 1: OS Event (published on TOPIC_APP_LAUNCHED)
```python
{
    "timestamp": float,          # Unix timestamp (sim time)
    "app_id": str,               # package ID e.g. "com.instagram.android"
    "user_id": str,              # "user_00" through "user_09"
    "battery": float,            # 0.0 to 100.0
    "time_of_day_bucket": int,   # 0 to 47
    "headphones": bool,
    "calendar_event_in_mins": int | None,
    "weekend": bool,
    "category": str              # from app_taxonomy
}
```

### Contract 2: Graph Snapshot (returned by BehaviouralGraph.get_graph_snapshot)
```python
{
    "day": int,
    "user_id": str,
    "node_count": int,
    "edge_count": int,
    "nodes": [{"node_id": str, "app_id": str, "category": str, "access_count": int}],
    "edges": [{"source": str, "target": str, "prob": float}]
}
```

### Contract 3: Benchmark Results CSV columns
```
user_id, policy_name, day, cache_hit_rate, launch_speed_gain_pct, thrash_rate, battery_overhead_pct, graph_node_count
```

### Contract 4: LangGraph State (GraphMindState TypedDict)
```python
{
    "user_id": str,
    "current_day": int,
    "current_event": dict | None,
    "battery": float,
    "kl_divergence": float,
    "cache_hit_rate": float,
    "security_flush_count": int,
    "last_agent": str,
    "messages": list[dict]
}
```

### Contract 5: Simulation Log (saved per user)
```json
{
    "user_id": "user_00",
    "days": [
        {
            "day": 0,
            "state": GraphMindState,
            "graph_snapshot": GraphSnapshot,
            "tier_stats": {"hot_count": int, "warm_count": int, "cold_count": int}
        }
    ]
}
```

---

## SECTION 7: REQUIRED NON-CODE FILES

### agents.md (place in project root)
```markdown
# GraphMind — Agentic AI Practices

## Agentic Architecture Overview
GraphMind implements a 5-agent LangGraph state machine...
[Describe each agent's role, inputs, outputs, reasoning approach]

## Agentic Workflows
[Describe the sequential workflow: GraphManager → DriftDetector → RLTrainer → Prefetch → Security]

## Reasoning & Planning Pipelines
[Describe how Gemma 2B reasons about node prioritization]

## Tool Use / Tool Chaining
[Describe the chain: OS metric reader → context encoder → RL reward calculator → graph weight updater → pre-fetch daemon → security gate]

## MCP Servers / Skills
[Describe any MCP-style tool definitions you used]

## Memory / Context Handling
[Explain the three-tier HOT/WARM/COLD architecture]

## Multi-Agent Orchestration
[Explain LangGraph state machine, conditional edges, state passing]

## What Worked
[Fill after implementation]

## What Did NOT Work
[Fill after implementation — judges specifically want this]
```

### docs/ax.md
Identical content to agents.md but expanded with code snippets, actual Gemma prompts used, W&B training curve screenshots.

---

## SECTION 8: KNOWN RISKS AND MITIGATIONS

| Risk | Mitigation |
|---|---|
| Gemma 2B too slow on CPU | Use fallback rule-based logic in all Gemma paths. Set use_llm = False if model load takes > 30s |
| PPO doesn't converge in time | Reduce timesteps to 50k for training, 200k only for final submission run |
| 10-user training takes too long | Train users in parallel using Python multiprocessing.Pool(4) |
| NetworkX graph too slow for 1000+ nodes | Cap COLD graph at MAX_NODES_COLD = 2000. Use eviction aggressively |
| LangGraph version incompatibility | Pin to langgraph==0.1.14 exactly. Do not upgrade |
| Streamlit PyVis rendering slow | Cap graph visualization at 200 nodes as specified in get_graph_snapshot() |
| W&B requires internet | Set WANDB_MODE=offline in .env. All logs go to local ./wandb/ directory |

---

## SECTION 9: SUBMISSION CHECKLIST

Before making repo public and emailing Samsung:

- [ ] README.md: all fields filled (team, problem number, video links, model links, dataset links)
- [ ] src/ folder: all files present and importable
- [ ] docs/ folder: architecture.md, installation.md, user_guide.md, ax.md all present
- [ ] agents.md in project root
- [ ] results/benchmark_results.csv present with 50 rows
- [ ] At least 1 PPO model present in models/rl_policies/
- [ ] Synthetic dataset in data/synthetic/users/ (all 10 users)
- [ ] LICENSE file (Apache 2.0)
- [ ] .env.example file (no real tokens, just variable names)
- [ ] Demo video on YouTube (public or unlisted)
- [ ] Setup & Reproducibility video on YouTube
- [ ] Presentation PDF on Google Drive (openly accessible)
- [ ] Models uploaded to Hugging Face (RL policy + Persona Encoder)
- [ ] Dataset uploaded to Hugging Face (synthetic behavioural graphs)
- [ ] Email sent from institute email to ennovatex.io@samsung.com before Jun 22 2:00 PM IST
- [ ] Email subject: "AX Hackathon Phase 2 Submission | Problem 03 | GraphMind"
- [ ] Email body: Team Name + GitHub repo link ONLY
```
