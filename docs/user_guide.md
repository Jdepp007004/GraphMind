# User Guide

> **GraphMindRL V5 — Practical User Manual**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Running Benchmarks](#running-benchmarks)
3. [Launching the Dashboard](#launching-the-dashboard)
4. [Viewing Users](#viewing-users)
5. [Inspecting Graphs](#inspecting-graphs)
6. [Viewing Metrics](#viewing-metrics)
7. [Exporting Results](#exporting-results)
8. [Common Tasks](#common-tasks)
9. [Reference: Key Files](#reference-key-files)

---

## Introduction

This guide is for anyone who wants to interact with the GraphMindRL V5 system — whether running the benchmark, exploring results in the dashboard, or exporting data for further analysis.

**Prerequisites**: Python 3.10+ and all dependencies installed (`pip install -r requirements.txt`). See [docs/reproducibility.md](reproducibility.md) for full installation instructions.

---

## Running Benchmarks

### Run the Official Benchmark

The official benchmark evaluates all 9 policies on all 31 users:

```bash
python scripts/run_phase11_e.py
```

Output is written to `results/final_production_results.csv`.

### Run for a Single User

To quickly test the prefetch engine on a single user:

```bash
python scripts/run_v5_rl_graph.py --user <user_id>
```

Replace `<user_id>` with the numeric user identifier (e.g., `1`, `3`, `12`).

### Run with Custom Weights

To experiment with alternative confidence weights (**note: this does not modify the production config**):

```bash
python scripts/run_v5_rl_graph.py \
  --w_transition 0.5 \
  --w_recency 0.1 \
  --w_frequency 0.4 \
  --w_context 0.0 \
  --threshold 0.16
```

Results are written to a timestamped CSV in `results/` and do **not** overwrite `final_production_results.csv`.

### Run the Benchmark for All Phase 11 Variants

```bash
# Weight grid search (Phase 11A)
python scripts/run_phase11_a.py

# Threshold sweep (Phase 11B)
python scripts/run_phase11_b.py

# Final combined benchmark (Phase 11E — official)
python scripts/run_phase11_e.py
```

---

## Launching the Dashboard

### First Time

```bash
cd dashboard
npm install
npm run dev
```

### Subsequent Times

```bash
cd dashboard
npm run dev
```

The dashboard runs at **http://localhost:3000**. It does not require the Python benchmark to be running.

### Regenerate Dashboard Data

If you have re-run the benchmark and want the dashboard to reflect the new results:

```bash
python scripts/generate_dashboard_data.py
```

Then restart the dashboard (`Ctrl+C` and `npm run dev` again).

---

## Viewing Users

### In the Dashboard

1. Navigate to **Cache Simulator** (`http://localhost:3000/simulator`) or **User Playback** (`http://localhost:3000/playback`).
2. Use the **User** dropdown to select any of the 5 pre-loaded users.
3. Click **Play** to watch that user's app sequence.

The 5 users in the dashboard are a representative sample; they were selected based on data quality and event count.

### User Statistics (All 31 Users)

View per-user statistics in CSV form:

```bash
cat data/processed/user_summary.csv
```

Or view them in the dashboard's Research page under the user statistics section.

### User Statistics Fields

| Field | Description |
|---|---|
| `user_id` | Numeric user identifier |
| `n_transitions` | Total transitions in dataset |
| `n_train` | Transitions in training set |
| `n_test` | Transitions in test set |
| `n_unique_apps` | Unique apps used |
| `f1_v5` | F1 score for GraphMindRL_V5 |
| `f1_baseline` | F1 score for GraphMindRL Baseline |
| `delta_f1` | Improvement for this user |

---

## Inspecting Graphs

### In the Dashboard

1. Navigate to **Graph Explorer** (`http://localhost:3000/graph`).
2. The graph for the default user loads automatically.
3. Drag nodes to rearrange. Scroll to zoom.
4. Use the **Search** box to find a specific app (e.g., type "youtube" or "chrome").
5. Click any node to open its detail panel (out-degree, top transitions).
6. Adjust the **Min prob** slider to filter low-probability edges.

### Programmatically

To inspect the graph for a specific user:

```python
from src.data.transition_extractor import load_processed_transitions
from src.models.graph_model import BehaviourGraph

user_id = 3
transitions = load_processed_transitions(user_id, split='train')
graph = BehaviourGraph(transitions)

# Get top candidates from 'youtube'
candidates = graph.get_candidates('com.google.android.youtube')
for app, prob in candidates[:5]:
    print(f"  {app}: {prob:.3f}")
```

### Export Graph as JSON

The dashboard JSON for the graph is at `dashboard/public/data/graph.json`. This contains the top 20 nodes and all edges above a minimum weight threshold. It can be used directly with any graph analysis tool.

---

## Viewing Metrics

### In the Dashboard

**Executive Overview** (`/`) — Key aggregate metrics for the production model.

**Benchmark Explorer** (`/benchmark`) — Full comparison table of all 9 policies. Click column headers to sort. Click rows to expand.

**Research Validation** (`/research`) — Ablation study, weight grid, threshold sweep, statistical significance table.

### From the Command Line

View the final benchmark result:

```bash
cat results/final_production_results.csv
```

View per-user F1 scores:

```bash
cat results/user_level_results_v4.csv | head -40
```

View the weight grid results:

```bash
cat results/v5_weight_grid.csv | sort -t, -k5 -rn | head -10
```

View the threshold sweep results:

```bash
cat results/v5_threshold_sweep.csv
```

### Key Metric Definitions

| Metric | Meaning |
|---|---|
| F1 | Harmonic mean of precision and recall (primary metric) |
| Precision | Fraction of prefetches that the user actually opened |
| Recall | Fraction of app opens that were correctly prefetched |
| Hit Rate | Fraction of all opens served from HOT or WARM cache |
| ΔF1 | Improvement vs. GraphMindRL Baseline (F1 = 0.7424) |
| p-value | Probability of observing this ΔF1 if there were truly no difference |
| Cohen's d | Standardised effect size (0.491 = medium-to-large) |
| Latency Saved | Total ms saved by serving from cache instead of cold load |

---

## Exporting Results

### Export All Results to CSV

All benchmark results are already in `results/`:

| File | Contents |
|---|---|
| `results/final_production_results.csv` | Official frozen result (all 9 policies) |
| `results/v5_all_experiments.csv` | All experiments run during the project |
| `results/v5_weight_grid.csv` | Phase 11A grid search (all weight combinations) |
| `results/v5_threshold_sweep.csv` | Phase 11B threshold sweep |
| `results/v5_final_comparison.csv` | Phase 11E head-to-head comparison |
| `results/statistical_results_v4.csv` | Per-user statistical results |
| `results/user_level_results_v4.csv` | Per-user F1 for all policies |
| `results/benchmark_results_v4.csv` | Full benchmark results V4 |

### Export Dashboard JSON

```bash
python scripts/generate_dashboard_data.py
```

This generates or updates all JSON files in `dashboard/public/data/`.

### Export a Graph for External Analysis

```python
import json
with open('dashboard/public/data/graph.json') as f:
    graph_data = json.load(f)

nodes = graph_data['nodes']   # list of {id, label, frequency, out_degree}
edges = graph_data['edges']   # list of {source, target, probability, count}

print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
```

---

## Common Tasks

### "I want to see which user improved the most."

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/user_level_results_v4.csv')
df['delta'] = df['f1_v5'] - df['f1_baseline']
top = df.nlargest(5, 'delta')[['user_id', 'f1_baseline', 'f1_v5', 'delta']]
print(top.to_string(index=False))
"
```

### "I want to verify the production config is correct."

```bash
python -c "
from config import settings as s
checks = [
    ('W_TRANSITION', s.W_TRANSITION, 0.50),
    ('W_RECENCY',    s.W_RECENCY,    0.10),
    ('W_FREQUENCY',  s.W_FREQUENCY,  0.40),
    ('W_CONTEXT',    s.W_CONTEXT,    0.00),
    ('THRESHOLD',    s.INITIAL_THRESHOLD, 0.16),
    ('HOT_SIZE',     s.HOT_CACHE_SIZE,    5),
    ('WARM_SIZE',    s.WARM_CACHE_SIZE,   15),
]
all_ok = True
for name, actual, expected in checks:
    ok = abs(actual - expected) < 1e-9
    print(f'  {name}: {actual} — {\"OK\" if ok else \"FAIL (expected \" + str(expected) + \")\"}')
    if not ok: all_ok = False
print('\\nResult:', 'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED')
"
```

### "I want to run the benchmark and immediately view it in the dashboard."

```bash
# Step 1: Run benchmark
python scripts/run_phase11_e.py

# Step 2: Regenerate dashboard data
python scripts/generate_dashboard_data.py

# Step 3: Launch dashboard
cd dashboard && npm run dev
# Open http://localhost:3000
```

### "I want to understand why the time context features were removed."

Read the scientific note in the Research Validation dashboard page (`/research`), or the detailed report:

```bash
cat reports/time_context_analysis.md
```

---

## Reference: Key Files

| File | Purpose |
|---|---|
| `config/settings.py` | Production configuration (frozen) |
| `scripts/run_phase11_e.py` | Official benchmark entry point |
| `scripts/generate_dashboard_data.py` | Dashboard JSON generator |
| `results/final_production_results.csv` | Official result (frozen) |
| `results/v5_all_experiments.csv` | Complete experiment log |
| `dashboard/public/data/` | Dashboard JSON files |
| `docs/reproducibility.md` | Full reproduction instructions |
| `docs/architecture.md` | System architecture |
| `docs/benchmarking.md` | Evaluation methodology |
| `reports/final_production_report.md` | Result narrative |
| `reports/v5_decision_gate.md` | Decision documentation |

---

*For technical questions about the implementation, see [docs/architecture.md](architecture.md). For reproduction instructions, see [docs/reproducibility.md](reproducibility.md).*
