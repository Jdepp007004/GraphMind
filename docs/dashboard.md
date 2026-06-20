# Dashboard Guide

> **GraphMindRL V5 -- 7-Page Interactive Dashboard**
> Running at `http://localhost:3000`

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Executive Overview (`/`)](#executive-overview-)
4. [Benchmark Explorer (`/benchmark`)](#benchmark-explorer-benchmark)
5. [Optimization Journey (`/journey`)](#optimization-journey-journey)
6. [Graph Explorer (`/graph`)](#graph-explorer-graph)
7. [Cache Simulator (`/simulator`)](#cache-simulator-simulator)
8. [User Playback (`/playback`)](#user-playback-playback)
9. [Research Validation (`/research`)](#research-validation-research)
10. [Data Sources](#data-sources)
11. [Technical Reference](#technical-reference)

---

## Overview

The GraphMind dashboard is a **Next.js 15** browser application that presents the complete research story -- from raw data and system architecture to live simulation of the prefetch engine. It is designed for hackathon judges to explore the system interactively without running any Python code.

The dashboard is **read-only**: it displays pre-computed results and does not modify any files.

---

## Getting Started

### Launch

```bash
cd dashboard
npm install   # first time only
npm run dev
```

Open **http://localhost:3000** in any modern browser (Chrome, Firefox, Safari, Edge).

### Navigation

The sidebar lists all 7 pages. The current page is highlighted. The Production F1 score (0.7745) is displayed permanently at the top of the sidebar.

---

## Executive Overview (`/`)

![Executive Overview](../assets/screenshots/dashboard-overview.png)

The landing page of the dashboard. Provides a complete one-screen summary of the project.

### Sections

**Header**
- Project name, hackathon context, and submission badges (p-value, Cohen's d, dataset).

**Key Metrics (6-card grid)**
- F1 Score: 0.7745 (+0.0321 vs baseline)
- Cache Hit Rate: 93.1%
- Latency Saved: ~1,847ms per launch
- Users: 31 (UbiqLog dataset)
- Transitions: 208,695 (reconstructed)
- p-value: 0.0115 (Cohen's d = 0.491)

**System Pipeline**
- Numbered 6-step diagram from dataset to cache:
  1. Dataset (UbiqLog4UCI, 9.7M events, 2 months)
  2. Transitions (MAX_GAP = 3600s, 208,695 transitions)
  3. Markov Graph (per-user weighted directed graph)
  4. Confidence Score (0.5×P_trans + 0.1×Recency + 0.4×Frequency)
  5. RL Controller (adaptive threshold ±0.005)
  6. Prefetch Cache (HOT=5, WARM=15, COLD=SQLite)

**Production Config**
- Live table of the frozen production configuration showing all weights, threshold, and cache sizes.

**Dashboard Links**
- Quick-access cards to all 6 other pages.

---

## Benchmark Explorer (`/benchmark`)

![Benchmark Explorer](../assets/screenshots/benchmark-explorer.png)

Interactive comparison of all 9 policies evaluated in the Phase 11E final benchmark.

### Sections

**Summary Stats (3 cards)**
- Best F1, Improvement, Statistical Significance

**Policy Comparison Chart**
- Bar chart with three tabs: **F1 Score**, **Hit Rate**, **Latency Saved**.
- The production model (GraphMindRL_V5) is highlighted in dark.
- Switch between metrics by clicking the tab buttons.

**All Results Table**
- Sortable columns: F1, Hit Rate, Latency, ΔF1, p-value, Cohen's d, Significance.
- Click any column header to sort ascending/descending.
- Click any row to expand a detail panel showing Precision, Recall, and the policy configuration string.
- Production model is marked with a green "prod" badge.

### How to Use

1. Click the **F1 Score** tab to see which policy performs best by F1.
2. Click the **ΔF1** column header to sort by improvement over baseline.
3. Click the **GraphMindRL_V5** row to expand its configuration details.
4. Switch to **Hit Rate** to verify the cache hit rate metric.

---

## Optimization Journey (`/journey`)

The full 8-phase research history, visualised as a timeline.

### Sections

**Summary (4 cards)**
- Total phases: 8
- Rejected: 3 (red)
- Accepted: 3 (green)
- Total ΔF1: +0.0321

**F1 Trajectory Chart**
- Line chart of F1 across all 8 phases.
- Reference lines mark the original Markov-1 baseline (0.7267) and the final production result (0.7745).
- Each point is colour-coded by outcome:
  - Green: accepted
  - Blue: accepted
  - Red: rejected/failed
  - Gray: baseline

**Phase Timeline**
- Expandable accordion cards for each phase.
- Each card shows: phase ID, status badge, title, description, F1 achieved, and ΔF1.
- Click a card to expand and see the date and full description.

### How to Read

Follow the timeline top-to-bottom. Each card represents one experiment. The trajectory chart shows the overall direction of improvement. Note how some experiments went below the baseline (rejected) before the final combination produced the production result.

---

## Graph Explorer (`/graph`)

![Graph Explorer](../assets/screenshots/graph-explorer.png)

Interactive Markov transition graph for a selected user.

### Features

**Graph Canvas**
- 20 most-frequent apps shown as nodes.
- Node size and darkness indicate relative frequency.
- Edge width and opacity indicate transition probability.
- Animated edges (dashed, moving) indicate probability > 35%.
- Drag nodes to rearrange. Scroll to zoom. Pan by dragging the canvas.

**Controls**
- **Search**: Type an app name to highlight matching nodes and fade others.
- **Min prob slider**: Filter edges below a minimum probability threshold (1%–30%).
- **Node count and edge count**: Updates live as the filter changes.

**App Detail Panel**
- Click any node to open the detail panel on the right.
- Shows: full package name, frequency, out-degree.
- Lists the top 6 outgoing transitions with probabilities.

**Graph Stats / Legend**
- Right sidebar shows graph statistics and a colour legend.

### How to Use

1. Explore the graph by dragging and zooming.
2. Use the search box to find a specific app (e.g., "youtube").
3. Raise the min prob filter to show only the strongest transitions.
4. Click a high-frequency node to see its top transitions.

---

## Cache Simulator (`/simulator`)

![Cache Simulator](../assets/screenshots/cache-simulator.png)

Live step-by-step simulation of the GraphMindRL_V5 prefetch engine on real user sequences.

### Features

**User Selector**
- Dropdown to select any of the 5 pre-loaded users.
- Changing user resets the simulation.

**Playback Controls**
- **Play/Pause**: Start or pause the automatic step-through.
- **Skip**: Advance one step manually.
- **Reset**: Return to step 1.
- **Speed slider**: 100ms–2000ms per step.
- **Progress bar**: Shows current position in the sequence. Drag to jump.

**Live Statistics (4 cards)**
- Cache Hits (running total)
- Hit Rate (rolling percentage)
- Latency Saved (total ms and seconds)
- Current Threshold (adaptive value from RL controller)

**HOT Cache Panel**
- 5 numbered slots showing the current HOT cache contents.
- Empty slots shown with dashed border.
- New entries animate in.

**WARM Cache Panel**
- Up to 15 pills showing the current WARM cache.
- Apps are added/removed as the simulation progresses.

**Current Event Panel**
- App name, package, timestamp, cache tier (HOT/WARM/MISS), and result (HIT/MISS).
- Confidence score bar chart for the top predictions at this step.
- Threshold value at which each prediction passed.

**Event Feed**
- Scrolling log of the last 15 events with tier and latency saved.

### How to Use

1. Select a user from the dropdown.
2. Press **Play** to watch the simulation run automatically.
3. Observe the HOT and WARM caches update in real time.
4. Watch the event feed to see HIT/MISS patterns.
5. Press **Pause** and use **Skip** to examine a specific step.
6. Slow down the speed slider to see each step clearly.

---

## User Playback (`/playback`)

Step-through view of a user's app sequence with rolling charts.

### Features

**Playback Controls**
- Same controls as the Cache Simulator (Play, Pause, Skip, Reset, Speed, Scrub bar).

**Rolling Charts (2 charts)**
- **Rolling Hit Rate**: Area chart of the cumulative hit rate over the last 30 events.
- **Adaptive Threshold**: Area chart of the RL controller's threshold value over the last 30 events.

**Current Event Detail**
- Large, prominent display of the current app, cache tier, and result.
- Confidence prediction bar chart for the top predictions.
- Green border for HIT events, no border for MISS.

**Cache Snapshot**
- Current HOT and WARM cache contents shown as colour-coded pills below the event detail.

### How to Use

1. Press **Play** to step through the sequence.
2. Watch the **Rolling Hit Rate** chart -- a rising line indicates the model is calibrating.
3. Watch the **Adaptive Threshold** chart -- the RL controller adjusts it based on hit rate.
4. Drag the scrub bar to jump to a specific point in the sequence and examine it in detail.

---

## Research Validation (`/research`)

Complete experimental evidence and statistical validation.

### Sections

**Reproducibility Certificate**
- Two independent run results (both F1 = 0.7745) confirming deterministic benchmark.
- p-value and Cohen's d for both runs.

**Ablation Study Chart**
- Horizontal bar chart showing the contribution of each component.
- The production model (all components active) is the rightmost bar.
- Removing any component reduces F1, confirming each is necessary.

**Phase 11A -- Weight Grid**
- Bar chart of the top 10 weight configurations from the grid search.
- The production configuration (rank 1) is highlighted in dark.

**Phase 11B -- Threshold Sweep**
- Bar chart of F1 vs threshold.
- The optimal threshold (0.16) is highlighted.
- Reference line shows the baseline F1.

**Statistical Significance Table**
- All 8 policies vs. baseline.
- Columns: ΔF1, p-value, Cohen's d, verdict.
- Green "Significant" badge for p < 0.05; gray "Not sig." otherwise.

**Context Feature Note**
- Scientific explanation of why `W_CONTEXT = 0.00`.
- Coverage data (94–98%) confirming data sparsity was not the issue.
- Note on how context is retained in the RL state space for monitoring.

### How to Use

1. Start with the **Reproducibility Certificate** to confirm the result is real.
2. Review the **Ablation Study** to understand which components matter.
3. Read the **Weight Grid** and **Threshold Sweep** to see how the optimal config was found.
4. Use the **Significance Table** to verify the statistical testing methodology.

---

## Data Sources

All dashboard data comes from pre-generated JSON files in `dashboard/public/data/`:

| File | Contents | Source |
|---|---|---|
| `benchmark.json` | All 9 policy results | `results/final_production_results.csv` |
| `optimization.json` | 8-phase journey | `results/v5_all_experiments.csv` |
| `graph.json` | User Markov graph | Raw UbiqLog reconstruction |
| `transitions.json` | Per-user event sequences | Raw UbiqLog + V5 simulation |
| `ablations.json` | Ablation study results | `results/v5_rl_ablation.csv` (archived) |
| `weight_grid.json` | Weight grid search | `results/v5_weight_grid.csv` |
| `threshold_sweep.json` | Threshold sweep | `results/v5_threshold_sweep.csv` |
| `summary.json` | Executive overview KPIs | `results/final_production_results.csv` |
| `users.json` | Per-user statistics | `data/processed/user_summary.csv` |

To regenerate all JSON files from source:

```bash
python scripts/generate_dashboard_data.py
```

---

## Technical Reference

| Component | Technology | Version |
|---|---|---|
| Framework | Next.js | 15.5 |
| Language | TypeScript | 5.x |
| Bar/Line/Area charts | Recharts | 2.x |
| Graph visualisation | @xyflow/react | 12.x |
| Animation | Framer Motion | 11.x |
| Styling | Tailwind CSS | 3.x |
| Font | Inter (Google Fonts) | -- |

### Browser Support

| Browser | Support |
|---|---|
| Chrome 120+ | ✓ Full |
| Firefox 120+ | ✓ Full |
| Safari 17+ | ✓ Full |
| Edge 120+ | ✓ Full |

---

*The dashboard is read-only and does not modify any files. All data is static JSON. The backend benchmark does not need to be running for the dashboard to work.*
