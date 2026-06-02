# GraphMind — User Guide

## Introduction

GraphMind is an intelligent app launch prediction system for Samsung Android devices. It learns your app usage patterns across time-of-day, context (headphones, calendar events, battery), and sequential transitions to pre-warm the memory cache before you need it.

## Dashboard Guide

### Launching the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Navigate to http://localhost:8501

### Navigation

The dashboard has 5 tabs:

#### 1. 🔗 Graph Evolution
Shows the behavioural graph at 4 snapshots: Day 1, 7, 14, and 29. Each node is an app, and edges show transition probabilities. You can see the graph grow more connected over time as your patterns are learned.

#### 2. 📊 Benchmarks
Compares GraphMind against 4 baselines:
- **LMKD Reactive** — Android's default memory killer
- **ART Static Profile** — Compile-time app startup profiles
- **UsageStats LRU** — Simple recency-based prediction
- **Bixby Frequency** — Time-of-day frequency counting

GraphMind should outperform all 4 in cache hit rate for at least 8 out of 10 users.

#### 3. 🎯 RL Training
Shows PPO training reward curves for each user. The reward increases as the agent learns to prioritize the right nodes.

#### 4. 🔒 Security Log
Shows all detected sensitive-to-consumer context transitions and the resulting HOT cache flushes. For example: using HDFC Bank app → switching to Instagram triggers a flush.

#### 5. 💾 Memory Tiers
Shows the HOT/WARM/COLD distribution as a pie chart, plus current usage vs capacity.

### Sidebar Controls

- **User Select** — Switch between the 10 simulated users
- **Day Slider** — View state at a specific simulation day (0-29)
- **▶ Run Live Simulation** — Run the full orchestration pipeline for the selected user
- **📊 Run Benchmarks** — Re-run all benchmark comparisons

## Understanding the Output

### Cache Hit Rate

A cache hit means the next app launched was already in HOT or WARM memory. Target:
- GraphMind: > 70% cache hit rate
- LMKD baseline: ~ 50-55%

### Security Flushes

Each row in the Security Log represents a detected sensitive→consumer transition. Expected: at least 5 per user over 30 days.

### Graph Growth

Day 1: ~20-50 nodes (just starting to learn patterns)
Day 7: ~80-150 nodes (weekly patterns established)
Day 14: ~150-250 nodes (rich context awareness)
Day 29: < 1000 nodes (pruning keeps it compact)

## Example User Stories

### User 00 (University Student)
- Peak usage: 10am, 2pm, 10pm
- Patterns: YouTube → Spotify → Instagram cycling
- Dashboard shows: evening entertainment cluster in graph

### User 01 (Office Commuter)
- Peak usage: 7am, 12pm, 6pm
- Patterns: Maps → Slack → Gmail during commute
- Security flushes: LinkedIn (enterprise) → Instagram (social)

### User 02 (Night Shift Nurse)
- Peak usage: midnight, 6am, 8pm
- Inverted sleep pattern captured in time-bucket edges
- Samsung Health promoted to HOT during work hours

## API Reference

### Running Individual Components

```python
from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED
from src.core.graph_engine import BehaviouralGraph
from src.core.memory_manager import MemoryManager
from src.data.event_simulator import EventSimulator
from src.agents.orchestrator import GraphMindOrchestrator

# Initialize
bus = EventBus.get_instance()
orch = GraphMindOrchestrator("user_00")

# Run one day
state = orch.run_day(0)
print(f"Cache hit rate: {state['cache_hit_rate']:.2%}")
print(f"Security flushes: {state['security_flush_count']}")
print(f"KL divergence: {state['kl_divergence']:.4f}")
```

### Loading a Trained Policy

```python
from src.rl.trainer import RLTrainer
import numpy as np

trainer = RLTrainer()
model = trainer.load_policy("user_00")
obs = np.zeros((68,), dtype=np.float32)
action, _ = model.predict(obs)
print(f"Predicted action: {action}")
# 0-28: promote HOT node index
# 29: prune weak edges
# 30: emergency demote
```

### Running a Benchmark Manually

```python
from src.benchmarks.evaluator import BenchmarkEvaluator
evaluator = BenchmarkEvaluator()
df = evaluator.run_all()
evaluator.print_summary_table()
```

## FAQ

**Q: How long does training take?**
A: ~30 minutes for all 10 users at 200K timesteps on CPU. Use --timesteps 50000 for quick testing.

**Q: Does it need internet access?**
A: No — Gemma model (optional) can be downloaded once. All simulation is local.

**Q: What happens if a user file is missing?**
A: Run `python scripts/generate_dataset.py` to regenerate.

**Q: Can I add a new user?**
A: Edit `USER_PROFILES` in `src/data/dataset_generator.py` and `NUM_USERS` in `config/settings.py`.
