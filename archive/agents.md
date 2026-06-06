# GraphMind — Agentic AI Practices Summary

This document summarizes GraphMind's use of agentic AI practices. Full details in [docs/ax.md](docs/ax.md).

## Agentic Workflow

GraphMind runs a **LangGraph StateGraph** with 5 autonomous agents:
- `graph_manager` — prioritizes HOT tier using Gemma 2B reasoning (fallback: access count sort)
- `drift_detector` — computes KL divergence between recent and historical app transitions
- `rl_trainer` — fine-tunes PPO with learning rate spike when drift is detected
- `prefetch` — pre-warms WARM/HOT memory cache using graph-predicted next nodes
- `security` — enforces context isolation between SENSITIVE and CONSUMER app categories

## Reasoning & Planning

- KL divergence statistical reasoning to detect behavioral drift
- Gemma 2B for natural-language cache prioritization reasoning
- PPO RL agent plans cache management using future reward signals

## Tool Use and Chaining

Agents chain tools through EventBus: `app_launched` → graph update → cache check → prefetch trigger → security flush

## Multi-Agent Orchestration

Sequential + conditional routing: drift detection conditionally triggers RL training before prefetch.

## Memory and Context Handling

Three-tier memory (HOT/WARM/COLD) implements explicit context management. ContextEncoder maintains situational embeddings across 64 dimensions.

## What Worked
- EventBus pub-sub architecture kept all modules decoupled and testable
- KL divergence drift detection worked reliably (threshold 0.3)
- Three-tier memory model delivered +18% cache hit rate vs LMKD baseline
- Rule-based dataset generation was reproducible across all 10 user personas

## What Did NOT Work
- Gemma 2B too slow for real-time decisions on CPU — relegated to offline/periodic tasks
- SB3 PPO gradient instability on long episodes — fixed by n_steps=512
- SQLite COLD tier bottlenecks at scale — would need RocksDB in production
- LangGraph 0.1.14 conditional edge API required careful version pinning
