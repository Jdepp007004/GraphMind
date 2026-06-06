# GraphMind Dashboard — Final Implementation Plan

**Winning benchmark policy:** `GraphMindRL`  
**F1:** 0.7424 | **Hit Rate:** 0.9357 | **Avg Latency Saved:** 2003 ms/launch  

---

## Technology Stack

- **Framework:** Next.js 14 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Charts:** Recharts
- **Graph:** React Flow

## Data Sources (actual benchmark outputs)

```
results/benchmark_results_v4.csv        ← per-user per-policy
results/user_level_results_v4.csv       ← same, pivoted
results/advanced_metrics_v4.csv         ← mean/std/P50/P90/P95/P99/CI95
results/statistical_results_v4.csv      ← p-values, CI, Cohen's d
results/ablation_results_v2.csv         ← ablation variants
reports/figures/graphmind_vs_markov2.png← user scatter
```

**Dashboard MUST consume these files. No hardcoded values.**

## Pages

### `/` — Homepage
- Hero metric strip: 31 users | 820K events | 208K transitions | F1=0.742
- Avg latency saved: 2003 ms/launch
- Winner badge: GraphMindRL
- Policy comparison bar chart (F1, 6 key policies)
- Animated stat counters on load

### `/benchmark` — Benchmark Results
- All policies ranked by F1 (bar chart with CI error bars)
- Hit Rate ranking (bar chart)
- Latency saved ranking (bar chart)
- `graphmind_vs_markov2.png` scatter embed
- P90/P95/P99 distribution table
- Policy filter dropdown

### `/statistical` — Statistical Analysis
- Significance heatmap (policy × metric → p-value)
- Cohen's d bar chart with effect magnitude labels
- CI overlap visualisation
- Key findings summary

### `/ablation` — Ablation Study
- Step chart: GraphOnly → +Confidence → +RL → Full GraphMind
- Component contribution table

### `/users` — User Overview
- Sortable table: user ID, n_events, n_transitions, best policy
- Bubble chart: events × F1 × cluster

### `/user/[id]` — User Detail
- Top-20 app frequency bar chart
- Transition heatmap (10×10)
- Policy comparison for this user
- Hit rate over test timeline

### `/dataset` — Dataset Overview
- UbiqLog stats: 35 users, 820K events, 208K transitions
- Event type distribution
- Gap sensitivity results (15/30/60 min)
- Latency tier chart (cold/warm/hot)

### `/rl` — RL Details
- V3 vs V4 architecture comparison
- Predictor weight visualisation (learned ensemble weights)
- Training trajectory chart

### `/user/[id]/playback` — Timeline Playback
- Step-by-step app launch replay
- HOT/WARM/COLD tier live display
- Policy prediction vs actual (hit/miss highlight)
- Speed controls: 1× / 5× / 10× / 100×

## Design System

- Dark mode primary (#0f1117 bg, #7c3aed accent, #10b981 success)
- Glassmorphism cards (backdrop-blur, border opacity)
- Micro-animations (framer-motion or CSS transitions)
- Inter font (Google Fonts)
- Responsive: 320px → 4K

## Data API

Create `lib/data.ts` to parse all CSV files at build time.
Use Next.js `generateStaticParams` for user pages.
No external data fetching needed — all data is local CSV.
