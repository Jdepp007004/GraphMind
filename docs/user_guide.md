# GraphMind V5 -- User Guide

> **Samsung EnnovateX AX Hackathon 2026 -- PS03**
> How to run the dashboard, interpret KPI output, and read Gemma explanations.

---

## Table of Contents

1. [Running the Dashboard](#running-the-dashboard)
2. [Dashboard Pages Reference](#dashboard-pages-reference)
3. [Interpreting KPI Output](#interpreting-kpi-output)
4. [Reading Gemma Explanations in User Journey Tab](#reading-gemma-explanations-in-user-journey-tab)
5. [Running the Benchmark](#running-the-benchmark)
6. [Interpreting Benchmark Results](#interpreting-benchmark-results)

---

## Running the Dashboard

### Prerequisites

- Node.js 18+ installed
- Project dependencies installed (see [docs/installation.md](installation.md))

### Launch

```bash
cd dashboard
npm run dev
```

Open **http://localhost:3000** in your browser.

The dashboard uses pre-generated JSON data from `dashboard/public/data/`. If you have just run a new benchmark and want the dashboard to reflect the new results:

```bash
python scripts/generate_dashboard_data.py
```

Then refresh the browser.

---

## Dashboard Pages Reference

### 🏠 Overview (`/`) -- KPI Summary

The Overview page is the entry point. It displays:

1. **7-Row KPI Table**: All 7 PS03 KPIs with Target, Achieved, and Status columns (🟢 PASS / 🔴 FAIL).
2. **System Pipeline Diagram**: Visual representation of the 7-step agentic pipeline.
3. **Production Configuration**: The frozen confidence weights and threshold.
4. **Statistical Proof**: Paired t-test results (p = 0.0115, Cohen's d = 0.491).

**How to read the KPI table**:

| Column | Description |
|---|---|
| KPI | The PS03 target KPI name |
| Target | Minimum required value to pass |
| Achieved | GraphMind V5's measured value |
| Status | 🟢 PASS if Achieved ≥ Target, 🔴 FAIL otherwise |

If a KPI shows 🔴 FAIL, the value may be a placeholder. Run the benchmark first:
```bash
python -m src.benchmarks.evaluator_v2
```

---

### 📊 Benchmark Explorer (`/benchmark`)

Displays an interactive comparison table of all 10 policies evaluated:

- **Policy**: The algorithm name (GraphMind_RL, LRU, Markov-1, etc.)
- **F1 Score**: Harmonic mean of precision and recall for prefetch correctness
- **Hit Rate**: Fraction of app opens served from HOT or WARM cache
- **Latency Saved**: Mean milliseconds saved per launch vs always-cold baseline
- **ΔF1 vs Baseline**: Improvement over GraphMindRL Baseline (reference policy)
- **p-value**: Paired t-test significance vs the reference policy
- **Cohen's d**: Effect size magnitude

**Sorting**: Click any column header to sort. GraphMind_RL will always top the F1 column.

**Chart panel**: Select any two policies to plot a per-user F1 scatter plot, showing exactly which of the 31 users benefit most from GraphMind.

---

### 🗺️ Optimization Journey (`/journey`)

Shows the F1 trajectory across all 8 research phases:

```
F1
0.78 │                                              ●  ← V5 (0.7745)
0.76 │                                         ●
0.74 │                        ●          ●
0.72 │             ●     ●
0.70 │        ●
     └──────────────────────────────────────────────
       Markov-1  Markov-2  G+C  Base  RL-Lat  V5(0.10) V5
```

Each point on the journey shows:
- The experiment name and hypothesis
- The result (F1 value)
- Whether the hypothesis was accepted or rejected
- The decision rationale

**Color coding**:
- 🟢 Green = Accepted hypothesis (merged into production)
- 🔴 Red = Rejected hypothesis (archived, not merged)

---

### 🕸️ Graph Explorer (`/graph`)

Interactive visualisation of a user's Markov behaviour graph:

- **Nodes**: Apps used by the selected user
- **Edges**: Observed transitions (A→B), weighted by transition probability
- **Edge thickness**: Proportional to transition probability
- **Node size**: Proportional to overall usage frequency

**Controls**:
- Select a user from the dropdown to load their personal graph
- Click any node to highlight its outgoing transitions
- Zoom and pan with mouse/trackpad

---

### 🎮 Cache Simulator (`/simulator`)

Live animation of the HOT/WARM/COLD cache system:

- The left panel shows the app event stream being replayed
- The right panel shows the cache state in real time
- **HOT tier**: Top 5 apps (in RAM, instant launch)
- **WARM tier**: Next 15 apps (pre-loaded, ~200ms launch)
- **COLD tier**: All others (~1,800ms cold launch)

**What to look for**: When an app event arrives, watch how the cache is updated. If the app was in HOT or WARM, it counts as a cache hit. If it was in COLD, it counts as a miss.

---

### 📼 User Journey (`/playback`)

Step-through playback of a real user's event sequence. This is where **Gemma explanations** appear.

See [Reading Gemma Explanations](#reading-gemma-explanations-in-user-journey-tab) for details.

**Controls**:
- Select a user from the dropdown
- Use ← → to step through events one at a time
- Or press ▶ to autoplay at 1 event/second

---

### 🔬 Research Validation (`/research`)

Statistical evidence for the benchmark claims:

- **Ablation study results**: What each component (RL, graph, confidence) contributes
- **Bootstrap confidence intervals**: 95% CI for mean F1 of each policy
- **Paired t-test results**: Per-policy comparison table with p-values and Cohen's d
- **Reproducibility log**: Timestamps and checksums for the two official benchmark runs

---

## Interpreting KPI Output

When you run the benchmark, the terminal prints a KPI summary table:

```
STABILITY: PASS -- 0 issues

══════════════════════════════════════════════════════════════════════════════════════
  KPI                                            Target    Achieved    Status
══════════════════════════════════════════════════════════════════════════════════════
  Next Context Prediction Accuracy (F1)           ≥0.75      0.7745  ✅ PASS
  Cache Hit Rate (%)                              ≥85%       93.10%  ✅ PASS
  Memory Thrashing Reduction (%)                  ≥50%        [X]%  [STATUS]
  App Load Time Improvement (%)                   ≥20%        [X]%  [STATUS]
  App Launch Time Improvement (%)                 ≥10%        [X]%  [STATUS]
  System Stability (issues)                       = 0            0  ✅ PASS
  Memory Utilisation Efficiency Improvement (%)   ≥30%        [X]%  [STATUS]
══════════════════════════════════════════════════════════════════════════════════════
  Overall: X/7 KPIs PASS
══════════════════════════════════════════════════════════════════════════════════════
```

This is also saved to `reports/kpi_summary.json`. To inspect the JSON:

```bash
python -c "import json; d = json.load(open('reports/kpi_summary.json')); print(json.dumps(d, indent=2))"
```

### Understanding Each KPI

**Next Context Prediction Accuracy (F1)**: This is the core ML metric. It measures how often GraphMind correctly predicts the next app the user will open. F1 = 0.7745 means that, harmonically averaged, 77.45% of prefetch predictions are both precise (not wasting memory) and recall the user's actual next app.

**Cache Hit Rate (%)**: The fraction of app launches served from the HOT or WARM cache (not cold storage). 93.1% means only 6.9% of launches experience the full 1,800ms cold-start delay.

**Memory Thrashing Reduction (%)**: How much less cache thrashing GraphMind causes vs the LRU baseline. Thrashing = an app was evicted and then immediately re-accessed. Less thrashing = more stable cache = less memory bandwidth waste.

**App Load Time Improvement (%)**: The reduction in time from tap to app-fully-interactive, weighted by the fraction of launches served from cache. Computed against the mean cold-start latency across all apps in the literature table.

**App Launch Time Improvement (%)**: Similar to load time, but specifically measuring the first-frame latency reduction. HOT-tier apps (in-RAM) achieve larger launch time improvements than WARM-tier apps.

**System Stability (issues)**: Counts crashes, OOM errors, and unhandled exceptions during the entire benchmark run. Any non-zero value indicates a system problem.

**Memory Utilisation Efficiency Improvement (%)**: How many of LRU's cold-start failures (cache misses) GraphMind eliminates. Formula: `(LRU_miss_rate - GM_miss_rate) / LRU_miss_rate * 100`. A value of 86% means GraphMind eliminates 86% of the cold-start delays that a basic LRU cache would still cause. Computed on the synthetic benchmark where the LRU baseline has a 80.31% miss rate vs GraphMind's 11.23% miss rate.

---

## Reading Gemma Explanations in User Journey Tab

The **User Journey** page (`/playback`) shows Gemma's natural-language explanations alongside each prefetch event.

### What You See

For each step in the user's event stream:

```
Event 47/208
─────────────────────────────────────────────────────────────
App opened:   YouTube
Time:         7:30 PM (bucket 39)
Battery:      72% (bucket 3)

Prefetch decision:
  → Spotify (confidence: 0.72) ← loaded into WARM
  → WhatsApp (confidence: 0.44) ← loaded into WARM

💬 Gemma says:
  "Preloading Spotify because you typically switch from
   YouTube in the evening when your battery is charged."

Cache result:  ✅ HIT (Spotify was in WARM when opened 2 events later)
─────────────────────────────────────────────────────────────
```

### Understanding the Explanation

The explanation tells you in plain language **why** the system made its prefetch decision. Key things to notice:

1. **App names are human-readable**: The system converts package IDs to display names automatically.

2. **Time context is natural**: "in the evening", "around midday", "late at night" -- derived from the 30-minute bucket index.

3. **Confidence drives the language**: High confidence (≥ 0.60) → "almost always switch from..."; medium confidence (0.35–0.60) → "frequently open it after..."; low confidence → "based on your most-used apps...".

4. **Fallback explanations** (when `ENABLE_GEMMA=false` or model unavailable): These use the same template language but are generated deterministically from the edge weights, not by the Gemma model. The fallback is indicated in the dashboard with a small ℹ️ icon.

### Important: Gemma Does Not Affect Metrics

> The Gemma explanation is generated **after** the prefetch decision is already made and all metrics are recorded. It has zero effect on F1, cache hit rate, or any other KPI. The `gemma_explanation` column in the benchmark CSV is nullable and is not scored.

To verify this: run the benchmark with `ENABLE_GEMMA=false` and `ENABLE_GEMMA=true` -- you will see identical F1 and hit rate values in both runs.

---

## Running the Benchmark

### Quick benchmark (synthetic data, ~2 minutes)

```bash
set ENABLE_GEMMA=false  # Windows
python -m src.benchmarks.evaluator_v2 --dataset synthetic
```

### Full benchmark with UbiqLog data

```bash
set ENABLE_GEMMA=false
python scripts/run_phase11_e.py
```

### With Gemma explanations (slower)

```bash
set ENABLE_GEMMA=true
python -m src.benchmarks.evaluator_v2
```

---

## Interpreting Benchmark Results

The benchmark writes four CSV files to `results/`:

| File | Contents |
|---|---|
| `benchmark_results_v2.csv` | Per-policy metrics (11 metrics + gemma_explanation) |
| `ablation_results_v2.csv` | Ablation variant comparison |
| `statistical_results_v2.csv` | t-test and Cohen's d vs GraphOnly baseline |
| `advanced_metrics_v2.csv` | Additional derived metrics |

And one JSON file to `reports/`:

| File | Contents |
|---|---|
| `reports/kpi_summary.json` | All 7 PS03 KPIs with pass/fail status |

**To check if the official result was reproduced**:

```bash
python -c "
import csv
with open('results/benchmark_results_v2.csv') as f:
    for row in csv.DictReader(f):
        if row.get('policy') in ('GraphMind_RL', 'GraphMindRL_V5'):
            print(f\"Policy: {row['policy']} | F1: {row.get('f1', 'N/A')} | Hit Rate: {float(row.get('cache_hit_rate', 0))*100:.1f}%\")
"
```

Expected output:
```
Policy: GraphMind_RL | F1: 0.7745 | Hit Rate: 93.1%
```

---

*For installation help, see [docs/installation.md](installation.md).*
*For reproduction steps, see [docs/reproducibility.md](reproducibility.md).*
*For architecture details, see [docs/architecture.md](architecture.md).*
