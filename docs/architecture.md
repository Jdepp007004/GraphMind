# GraphMind Architecture

## Overview

GraphMind is an on-device, privacy-preserving predictive app launch system for Samsung Android devices. It replaces Android's reactive LMKD memory manager with a proactive, RL-trained three-tier memory hierarchy.

## System Architecture

```
+---------------------------------------------------------------------+
|                         GraphMind System                            |
|                                                                     |
|  +--------------+   EventBus   +----------------------------------+|
|  |  OS Events   |------------->|     LangGraph Orchestrator       ||
|  |  (Simulator) |              |  +----------+  +--------------+  ||
|  +--------------+              |  |  Graph   |  | Drift Detect |  ||
|                                |  | Manager  |->|    Agent     |  ||
|  +--------------+              |  +----------+  +------+-------+  ||
|  |Behavioural   |              |          conditional  |           ||
|  |   Graph      |<-------------|  +----------+  +-----v--------+  ||
|  |  (NetworkX)  |              |  |RL Trainer|  |   Prefetch   |  ||
|  +------+-------+              |  |  Agent   |->|    Agent     |  ||
|         |                      |  +----------+  +------+-------+  ||
|  +------v-------+              |                +------v--------+  ||
|  |   Memory     |              |                |   Security    |  ||
|  |   Manager    |              |                |    Agent      |  ||
|  |  HOT/WARM/   |              |                +---------------+  ||
|  |    COLD      |              +----------------------------------+|
|  +--------------+                                                   |
+---------------------------------------------------------------------+
```

## Component Details

### 1. EventBus (`src/core/event_bus.py`)
Singleton publish-subscribe bus. All inter-module communication goes through this. Never import between unrelated modules -- always use EventBus topics.

Topics:
- `TOPIC_APP_LAUNCHED` -- OS event: app launched
- `TOPIC_BATTERY_UPDATED` -- battery level changed
- `TOPIC_SECURITY_FLUSH` -- sensitive context transition detected
- `TOPIC_DRIFT_DETECTED` -- KL divergence threshold exceeded
- `TOPIC_CACHE_HIT` / `TOPIC_CACHE_MISS` -- memory tier results
- `TOPIC_PREFETCH_TRIGGERED` -- daemon pre-fetch cycle ran

### 2. BehaviouralGraph (`src/core/graph_engine.py`)
Directed weighted graph (NetworkX DiGraph). Each node represents a `(app_id, time_bucket, battery_bucket)` situation. Edges store transition probability, time sensitivity, and battery cost.

- Nodes: `GraphNode` dataclass with 64-dim embedding
- Edges: 3-weight directed connections (`transition_prob`, `time_sensitivity`, `battery_cost`)
- Pruning: removes edges with prob < 0.05 weekly
- Eviction: removes nodes inactive for > 45 days

### 3. MemoryManager (`src/core/memory_manager.py`)
Three-tier hierarchy:

| Tier | Implementation | Capacity | Latency Model |
|------|---------------|----------|---------------|
| HOT  | Python dict + LRU | 30 nodes | ~0ms (RAM) |
| WARM | OrderedDict LRU | 150 nodes | ~5ms (cache) |
| COLD | SQLite on-disk | Unlimited | ~50ms (disk) |

### 4. LangGraph Orchestrator (`src/agents/orchestrator.py`)
State machine wiring 5 agents:
```
START -> graph_manager -> drift_detector -[kl > 0.3]-> rl_trainer -> prefetch -> security -> END
                                        +[kl <= 0.3]-> prefetch -> security -> END
```

### 5. RL Environment (`src/rl/environment.py`)
Custom Gymnasium env:
- Observation: 68-dim (35 context + 30 HOT occupancy + 3 state signals)
- Action: Discrete(31) -- node priority, prune, emergency demote
- Episode: one simulated day
- Reward: `R = alpha*hit + beta*speed - gamma*thrash - delta*battery + epsilon*friction`

### 6. Context Encoder (`src/data/context_encoder.py`)
3-layer MLP: 35-dim input -> 64-dim embedding
- Input: app one-hot(30) + time_norm + battery_norm + headphones + calendar_near + weekend

### 7. Drift Detector (`src/agents/drift_detector_agent.py`)
KL divergence between recent 100 app transitions and historical 200 transitions.
Triggers RL learning rate spike if KL > 0.3.

### 8. Security Agent (`src/security/context_boundary.py`)
Detects transitions from SENSITIVE categories (financial, health, enterprise) to CONSUMER categories (social, entertainment, shopping). Flushes HOT cache of sensitive nodes on detection.

## Data Flow

1. OS launches an app -> EventBus publishes `TOPIC_APP_LAUNCHED`
2. BehaviouralGraph subscribes -> creates/updates node, updates edge weights
3. MemoryManager subscribes -> checks cache tier, promotes node to HOT
4. ContextBoundaryEnforcer subscribes -> detects category transition, may flush HOT
5. DriftDetectorAgent subscribes -> records transition in both history windows
6. LangGraph runs -> graph_manager prioritizes, drift_detector checks KL, prefetch daemon pre-warms cache

## Security Model

GraphMind follows Android's process isolation model. No user data is stored in plaintext -- only encoded 64-dim embeddings. The security flush ensures financial/health data is never cached when switching to social apps.

## Scalability

- Graph is pruned weekly (removes low-probability edges)
- Nodes are evicted after 45 days of inactivity
- Maximum graph size: 1000 nodes enforced by evaluator checks
- WARM tier is rebuilt every 15 minutes by PrefetchDaemon
