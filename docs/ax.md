# GraphMind -- Agentic AI Practices (ax.md)

## Overview

GraphMind is built on a foundation of agentic AI patterns. This document explains each agentic practice we use, what worked, and what did not work during development.

---

## 1. Agentic Workflows

GraphMind implements an **agentic pipeline** where autonomous agents monitor OS events, make decisions, and take actions -- without human intervention after deployment.

### Implementation
- **LangGraph StateGraph** orchestrates 5 agents in a directed acyclic workflow
- Each agent receives the full system state and returns an updated version
- State flows: `graph_manager -> drift_detector -> [rl_trainer?] -> prefetch -> security`
- The system runs every 15 minutes via APScheduler (PrefetchDaemon)

### Outcome
[OK] Clean separation of concerns. Agents can be tested independently. The event-driven design means new agents can be added without modifying existing ones.

---

## 2. Reasoning & Planning

The **GraphManagerAgent** and **DriftDetectorAgent** perform structured reasoning about app usage patterns.

### Implementation
- **GraphManagerAgent** uses Gemma 2B (with rule-based fallback) to reason: "Given apps [X, Y, Z] in cache at time-bucket 10 (morning), which 3 should be prioritized?"
- **DriftDetectorAgent** computes KL divergence as a statistical reasoning mechanism to detect when user behaviour has fundamentally changed
- **RLTrainerAgent** uses PPO reward signals to plan which nodes to prioritize based on future expected rewards

### Outcome
[OK] KL divergence-based drift detection worked reliably. The threshold of 0.3 correctly identified genuine pattern shifts.
[FAIL] Gemma 2B reasoning was too slow for real-time decisions -- we use the LLM output only for non-critical periodic tasks and fall back to rule-based for latency-sensitive operations.

---

## 3. Tool Use and Tool Chaining

Each agent uses multiple tools in sequence:

### Tool Chain Example (Prefetch Cycle)
1. `DriftDetectorAgent.compute_kl_divergence()` -> detects drift
2. `RLTrainerAgent.train_user()` -> spikes learning rate, fine-tunes policy
3. `PrefetchDaemon.run_prefetch_cycle()` -> calls `BehaviouralGraph.get_top_k_next_nodes()`
4. `MemoryManager.rebuild_warm_from_graph()` -> promotes predicted nodes
5. `ContextBoundaryEnforcer.enforce_boundary()` -> may flush sensitive nodes

### Outcome
[OK] Tool chaining through EventBus decouples tools from each other -- each tool publishes results as events, avoiding tight coupling.

---

## 4. Coding Assistants

GraphMind was developed using **Antigravity IDE** (Google DeepMind Advanced Agentic Coding) as a coding assistant throughout the entire implementation.

### Usage Patterns
- Generating boilerplate for Gymnasium environments
- Debugging LangGraph version compatibility issues (0.1.14 API)
- Generating the 10-user synthetic dataset schemas
- Writing all test files

### Outcome
[OK] The coding assistant significantly accelerated development -- especially for repetitive but correct code (conftest, all test files, docstrings). The agent was able to understand the full spec and implement from it end-to-end.

---

## 5. MCP Servers (Model Context Protocol)

GraphMind uses MCP-compatible patterns through the EventBus abstraction, which acts as an internal message passing server.

### Implementation
- **EventBus** is our internal MCP -- agents communicate via structured message passing (topic + payload)
- All topics are constants defined in `event_bus.py` -- equivalent to MCP tool definitions
- Payloads follow a fixed schema (always include `timestamp`, `user_id`)

### Outcome
[OK] The EventBus pattern is clean and testable. Each module can be replaced without affecting others.
[WARNING]? True MCP server integration (external tooling) was not implemented -- the EventBus covers all inter-module needs.

---

## 6. Memory and Context Handling

GraphMind implements a sophisticated **three-tier memory hierarchy** as its core contribution.

### Implementation
- **HOT tier** (30 nodes, Python dict): Ultra-fast access. The RL agent actively manages what goes here.
- **WARM tier** (150 nodes, OrderedDict LRU): Pre-fetched predictions. Refreshed every 15 minutes.
- **COLD tier** (SQLite, unlimited): Long-term behavioural memory. Persists across reboots.

### Context Handling
- `ContextEncoder` encodes each app event into a 64-dim situational embedding
- Embeddings capture time-of-day, battery, headphones, and calendar context
- Graph nodes aggregate these embeddings to build rich situational models

### Outcome
[OK] The three-tier model is the key innovation. Simulation shows 18%+ improvement in cache hit rate over LMKD baseline.
[FAIL] COLD-tier SQLite becomes a bottleneck at scale -- a future version would use RocksDB.

---

## 7. Multi-Agent Orchestration

The **LangGraph StateGraph** coordinates 5 specialized agents:

| Agent | Role | Trigger |
|-------|------|---------|
| GraphManagerAgent | Prioritizes HOT tier, prunes graph | Every orchestration cycle |
| DriftDetectorAgent | Computes KL divergence | Subscribes to app events |
| RLTrainerAgent | Fine-tunes PPO on drift | Conditional on KL > 0.3 |
| PrefetchAgent | Pre-warms WARM/HOT cache | Every 15 minutes |
| SecurityAgent | Enforces context boundaries | Sensitive->consumer transition |

### Outcome
[OK] The conditional routing (drift -> RL training) ensures the RL agent only runs when needed, preserving battery.
[FAIL] LangGraph 0.1.14's API has limited debugging tooling -- state inspection required manual logging.

---

## 8. Practical Problem-Solving

GraphMind addresses a real Samsung pain point: cold-start app launch latency.

### Problem
Android's LMKD kills background apps to free memory -> next launch requires full cold start (~1-3 seconds).

### Solution
GraphMind predicts which apps you'll need next based on behavioral patterns, pre-warms them in memory (HOT tier), and keeps them there. Result: warm start (~0.1 seconds).

### Measured Results
- Cache hit rate: +18% vs LMKD
- Launch speed gain: +15-25% (estimated from cache hit improvement)
- Security flushes: 5+ per user per 30 days (correctly detected sensitive transitions)
- Graph stability: < 1000 nodes for all users after 30 days

---

## What Worked

1. **Event-driven architecture via EventBus** -- decoupled, testable, easy to extend
2. **KL divergence drift detection** -- simple, fast, effective at detecting genuine pattern changes
3. **Rule-based dataset generation** -- fast, reproducible, covers all 10 personas without requiring GPU
4. **Three-tier memory model** -- intuitive analogy to CPU cache hierarchy, strong KPI results
5. **LangGraph for agent orchestration** -- clean API for conditional routing between agents
6. **Separate reward weight constants** (alpha, beta, gamma, delta, epsilon) -- easy to tune without touching logic

---

## What Did NOT Work

1. **Gemma 2B for real-time decisions** -- 2B parameters is too slow for sub-100ms latency requirements on CPU. We fall back to rule-based prioritization and use Gemma only for offline tasks.
2. **Stable-Baselines3 PPO on long episodes** -- episodes spanning 80+ events caused gradient instability. Fixed by limiting `n_steps=512` and using shorter episode boundaries.
3. **SQLite COLD tier at scale** -- Python's sqlite3 module becomes a bottleneck with 10 simultaneous users. In production, RocksDB or LevelDB would be used.
4. **LangGraph 0.1.14 conditional edges** -- The API for conditional edges changed between 0.1.x versions. Required careful alignment with the exact spec version.
5. **PyVis graph rendering in Streamlit** -- PyVis's `generate_html()` requires `st.components.v1.html()` which is not ideal for large graphs. Limited to 50 nodes for rendering performance.
6. **Gemma-based embedding updates during RL** -- Attempting to fine-tune the ContextEncoder MLP simultaneously with PPO caused training instability. The encoder weights are frozen in the current implementation.

---

## Future Work

- Replace Gemma 2B with a smaller 1B model optimized for Samsung Exynos NPU
- Use quantization (INT4) to fit the model in 2GB RAM
- Implement federated learning for cross-device pattern sharing without privacy violation
- Integrate with real Samsung One UI app launch events via adb or system API
- Replace SQLite with RocksDB for the COLD tier
