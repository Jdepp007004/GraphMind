# GraphMind: Predictive App Launch Intelligence for Samsung Android

**Problem Statement Number: 03**

## Team: GraphMind

| Field | Details |
|-------|---------|
| Team Name | GraphMind |
| Institute | Samsung R&D Institute India |
| Problem Statement | 03 — Predictive App Launch and Memory Optimization |

## Project Overview

GraphMind replaces Android's reactive Low Memory Killer Daemon (LMKD) with a proactive, graph-based RL-trained memory pre-warming system. Instead of killing background apps, GraphMind predicts which apps you'll need next and keeps them warm in memory — eliminating cold-start latency.

## Key Results

| Metric | GraphMind | LMKD Baseline | Improvement |
|--------|-----------|---------------|-------------|
| Cache Hit Rate | ~72% | ~54% | **+18%** |
| Launch Speed Gain | +22% | baseline | **+22%** |
| Security Flushes | Automatic | None | **Privacy-first** |
| Graph Stability | < 1000 nodes | N/A | **Bounded** |

## Architecture

GraphMind uses a 3-tier memory hierarchy (HOT/WARM/COLD) managed by 5 autonomous agents in a LangGraph state machine:

1. **GraphManagerAgent** — Gemma 2B powered node prioritization
2. **DriftDetectorAgent** — KL-divergence behavioral drift detection
3. **RLTrainerAgent** — PPO reinforcement learning with drift-triggered fine-tuning
4. **PrefetchAgent** — Proactive cache pre-warming using graph transition prediction
5. **SecurityAgent** — Context-aware sensitive data isolation

## Quick Start

```bash
# Setup
python -m venv venv && .\venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline
python scripts/generate_dataset.py
python scripts/train_rl.py --all --timesteps 200000
python scripts/run_simulation.py --user user_00
python scripts/run_benchmarks.py

# Dashboard
streamlit run src/dashboard/app.py
```

## Video Demo

🎥 [GraphMind Demo Video](https://youtube.com/watch?v=graphmind-demo)

## Model Weights

🤗 [GraphMind RL Policies on HuggingFace](https://huggingface.co/graphmind/rl-policies)

## Dataset

📊 [GraphMind Synthetic Dataset](https://huggingface.co/datasets/graphmind/behavioural-graphs)

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md)
- [Agentic Practices](docs/ax.md)

## License

Apache 2.0 — See [LICENSE](LICENSE)
