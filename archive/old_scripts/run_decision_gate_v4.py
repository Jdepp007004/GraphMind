#!/usr/bin/env python3
"""
scripts/run_decision_gate_v4.py

Phase 7 — Decision Gate.

Reads:
  results/benchmark_results_v4.csv
  results/statistical_results_v4.csv

Decision criteria (RLAdaptiveEnsemble vs best V3 policy):
  UPGRADE: F1 improvement >= 2% AND statistically significant (p < 0.05)
  KEEP:    F1 improvement < 2% OR not significant

Outputs:
  reports/performance_review_v4.md  — full decision report with evidence
  reports/dashboard_plan_final.md   — dashboard plan using winning policy
"""

import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")

UPGRADE_F1_THRESHOLD   = 0.02   # absolute F1
SIGNIFICANCE_THRESHOLD = 0.05   # p-value


def load_bench(path: str) -> Dict[str, Dict[str, List[float]]]:
    data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pol = row["policy"]
            for k, v in row.items():
                if k in ("user_id", "policy"): continue
                try:
                    data[pol][k].append(float(v))
                except ValueError:
                    pass
    return data


def load_stat(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(vals: List[float]) -> float:
    return round(float(np.mean(vals)), 4) if vals else 0.0


def std(vals: List[float]) -> float:
    return round(float(np.std(vals)), 4) if vals else 0.0


def pct(vals: List[float], p: float) -> float:
    return round(float(np.percentile(vals, p)), 4) if vals else 0.0


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    bench_path = os.path.join(RESULTS_DIR, "benchmark_results_v4.csv")
    stat_path  = os.path.join(RESULTS_DIR, "statistical_results_v4.csv")

    if not os.path.exists(bench_path):
        logger.error(f"Missing: {bench_path}")
        return

    bench = load_bench(bench_path)
    stats = load_stat(stat_path)

    # Identify all available policies
    available = sorted(bench.keys())
    logger.info(f"Available policies: {available}")

    # Best V3 policy (by F1)
    v3_candidates = ["GraphMindRL", "Graph+Confidence", "Markov-2", "VariableOrderMarkov"]
    best_v3_name  = max([p for p in v3_candidates if p in bench],
                        key=lambda p: mean(bench[p]["f1"]),
                        default="GraphMindRL")
    v4_name = "RLAdaptiveEnsemble"

    v3_f1  = bench[best_v3_name]["f1"]   if best_v3_name in bench else []
    v4_f1  = bench[v4_name]["f1"]        if v4_name in bench else []
    v3_hr  = bench[best_v3_name]["hit_rate"] if best_v3_name in bench else []
    v4_hr  = bench[v4_name]["hit_rate"]      if v4_name in bench else []
    v3_lat = bench[best_v3_name]["latency_saved_ms"] if best_v3_name in bench else []
    v4_lat = bench[v4_name]["latency_saved_ms"]      if v4_name in bench else []

    # Decision metrics
    if not v3_f1 or not v4_f1:
        logger.warning("RLAdaptiveEnsemble not found in results — defaulting to KEEP")
        decision = "KEEP"
        abs_f1_improvement = 0.0
        rel_f1_improvement = 0.0
        is_significant     = False
        p_value            = 1.0
        cohens_d_val       = 0.0
        cohens_d_mag       = "negligible"
    else:
        abs_f1_improvement = round(mean(v4_f1) - mean(v3_f1), 4)
        rel_f1_improvement = round(abs_f1_improvement / max(mean(v3_f1), 1e-9) * 100, 2)

        # Find stat result for this comparison
        stat_row = None
        for r in stats:
            if (r.get("treatment") == v4_name and
                r.get("control") == best_v3_name and
                r.get("metric") == "f1"):
                stat_row = r
                break

        p_value       = float(stat_row["p_value"])   if stat_row else 1.0
        is_significant = p_value < SIGNIFICANCE_THRESHOLD
        cohens_d_val  = float(stat_row["cohens_d"])  if stat_row else 0.0
        cohens_d_mag  = stat_row["effect_magnitude"] if stat_row else "negligible"

        # Decision logic
        meets_f1_threshold = abs_f1_improvement >= UPGRADE_F1_THRESHOLD
        decision = "UPGRADE" if (meets_f1_threshold and is_significant) else "KEEP"

    logger.info(f"\n{'='*50}")
    logger.info(f"DECISION: {decision}")
    logger.info(f"  V3 best ({best_v3_name}) F1: {mean(v3_f1):.4f}")
    logger.info(f"  V4 ({v4_name}) F1:          {mean(v4_f1):.4f}")
    logger.info(f"  Absolute improvement:         {abs_f1_improvement:+.4f}")
    logger.info(f"  Relative improvement:         {rel_f1_improvement:+.2f}%")
    logger.info(f"  Significant:                  {'Yes' if is_significant else 'No'} (p={p_value:.4f})")
    logger.info(f"  Cohen's d:                    {cohens_d_val:.3f} ({cohens_d_mag})")
    logger.info(f"{'='*50}")

    winning_policy = v4_name if decision == "UPGRADE" else best_v3_name
    winning_f1     = mean(v4_f1) if decision == "UPGRADE" else mean(v3_f1)
    winning_hr     = mean(v4_hr) if decision == "UPGRADE" else mean(v3_hr)
    winning_lat    = mean(v4_lat) if decision == "UPGRADE" else mean(v3_lat)

    # All policies ranked for full table
    all_sorted = sorted(
        [(p, mean(d["f1"]), mean(d["hit_rate"]), mean(d["latency_saved_ms"]))
         for p, d in bench.items()],
        key=lambda x: -x[1]
    )

    # Write performance review
    pr_path = os.path.join(REPORTS_DIR, "performance_review_v4.md")
    with open(pr_path, "w", encoding="utf-8") as f:
        f.write("# GraphMind V4 — Performance Review & Decision Gate\n\n")
        f.write(f"**Decision: {'🚀 UPGRADE to RLAdaptiveEnsemble' if decision == 'UPGRADE' else '✅ KEEP GraphMindRL V3'}**\n\n")
        f.write("---\n\n")
        f.write("## Primary Comparison\n\n")
        f.write(f"| Metric | V3 ({best_v3_name}) | V4 ({v4_name}) | Δ | Relative |\n")
        f.write("|--------|-----------------|-----------------|---|----------|\n")
        f.write(f"| **F1** | {mean(v3_f1):.4f} | {mean(v4_f1):.4f} | {abs_f1_improvement:+.4f} | {rel_f1_improvement:+.2f}% |\n")
        f.write(f"| Hit Rate | {mean(v3_hr):.4f} | {mean(v4_hr):.4f} | {mean(v4_hr)-mean(v3_hr):+.4f} | — |\n")
        f.write(f"| Lat Saved (ms) | {mean(v3_lat):.1f} | {mean(v4_lat):.1f} | {mean(v4_lat)-mean(v3_lat):+.1f} | — |\n")
        f.write("\n## Statistical Evidence\n\n")
        f.write(f"| Test | Statistic | Threshold | Met? |\n")
        f.write("|------|----------|-----------|------|\n")
        f.write(f"| p-value (paired t-test) | {p_value:.4f} | < {SIGNIFICANCE_THRESHOLD} | {'✅' if is_significant else '❌'} |\n")
        f.write(f"| F1 improvement | {abs_f1_improvement:+.4f} | ≥ {UPGRADE_F1_THRESHOLD} | {'✅' if abs_f1_improvement >= UPGRADE_F1_THRESHOLD else '❌'} |\n")
        f.write(f"| Cohen's d | {cohens_d_val:.3f} ({cohens_d_mag}) | — | — |\n")
        f.write("\n## Decision Criteria\n\n")
        f.write("```\n")
        f.write("UPGRADE if:\n")
        f.write(f"  F1 improvement >= {UPGRADE_F1_THRESHOLD}  → {abs_f1_improvement:+.4f}  ({'MET' if abs_f1_improvement >= UPGRADE_F1_THRESHOLD else 'NOT MET'})\n")
        f.write(f"  AND p < {SIGNIFICANCE_THRESHOLD}          → {p_value:.4f}   ({'MET' if is_significant else 'NOT MET'})\n")
        f.write(f"\nDECISION: {decision}\n")
        f.write("```\n\n")
        f.write(f"## Winning Policy: **{winning_policy}**\n\n")
        f.write(f"- F1: **{winning_f1:.4f}**\n")
        f.write(f"- Hit Rate: **{winning_hr:.4f}**\n")
        f.write(f"- Latency Saved: **{winning_lat:.1f} ms/launch**\n\n")
        f.write("## Full Policy Ranking (all V4 policies by F1)\n\n")
        f.write("| Rank | Policy | F1 | Hit Rate | Lat Saved (ms) | Notes |\n")
        f.write("|------|--------|----|---------:|---------------:|-------|\n")
        for rank, (pol, f1, hr, lat) in enumerate(all_sorted, 1):
            note = ""
            if pol == winning_policy: note = "← **Dashboard policy** 🏆"
            elif f1 < all_sorted[-1][1] * 1.01 and pol not in ("Random",): note = "dominated"
            f.write(f"| {rank} | {pol} | {f1:.4f} | {hr:.4f} | {lat:.1f} | {note} |\n")
        f.write("\n## Dominated Baselines\n\n")
        # Flag any policy that is strictly worse than another on all 3 metrics
        dominated = []
        for p1, f1_1, hr1, lat1 in all_sorted:
            for p2, f1_2, hr2, lat2 in all_sorted:
                if p1 == p2: continue
                if f1_2 >= f1_1 and hr2 >= hr1 and lat2 >= lat1 and (f1_2 > f1_1 or hr2 > hr1):
                    dominated.append((p1, p2))
                    break
        if dominated:
            for (weaker, stronger) in dominated:
                f.write(f"- **{weaker}** is dominated by **{stronger}** (F1, HR, Lat all ≤)\n")
        else:
            f.write("_No policy is strictly dominated across all three metrics._\n")
        f.write("\n## Deployment Recommendation\n\n")
        if decision == "UPGRADE":
            f.write(f"**Upgrade to RLAdaptiveEnsemble.** The V4 REINFORCE ensemble controller ")
            f.write(f"delivers a statistically significant F1 improvement of {abs_f1_improvement:+.4f} ")
            f.write(f"({rel_f1_improvement:+.2f}%) over the previous best policy ({best_v3_name}). ")
            f.write(f"Cohen's d = {cohens_d_val:.3f} ({cohens_d_mag} effect). ")
            f.write(f"The RL predictor-weighting role is validated.\n")
        else:
            f.write(f"**Keep GraphMindRL V3.** The V4 RLAdaptiveEnsemble does not achieve ")
            f.write(f"the minimum 2% F1 improvement threshold ({abs_f1_improvement:+.4f} absolute). ")
            if not is_significant:
                f.write(f"The improvement is also not statistically significant (p={p_value:.4f}). ")
            f.write(f"The current V3 system (F1={mean(v3_f1):.4f}, Lat={mean(v3_lat):.1f}ms) ")
            f.write(f"represents the production-ready policy for the dashboard.\n\n")
            f.write(f"**Future work:** The new V4 models (VOM, ContextMarkov, ClusterMarkov) ")
            f.write(f"may be worth investigating individually in further ablations.\n")

    logger.info(f"Written: {pr_path}")

    # Write dashboard plan
    dp_path = os.path.join(REPORTS_DIR, "dashboard_plan_final.md")
    with open(dp_path, "w", encoding="utf-8") as f:
        f.write("# GraphMind Dashboard — Final Implementation Plan\n\n")
        f.write(f"**Winning benchmark policy:** `{winning_policy}`  \n")
        f.write(f"**F1:** {winning_f1:.4f} | **Hit Rate:** {winning_hr:.4f} | "
                f"**Avg Latency Saved:** {winning_lat:.0f} ms/launch  \n\n")
        f.write("---\n\n")
        f.write("## Technology Stack\n\n")
        f.write("- **Framework:** Next.js 14 (App Router)\n")
        f.write("- **Language:** TypeScript\n")
        f.write("- **Styling:** Tailwind CSS\n")
        f.write("- **Charts:** Recharts\n")
        f.write("- **Graph:** React Flow\n\n")
        f.write("## Data Sources (actual benchmark outputs)\n\n")
        f.write("```\n")
        f.write("results/benchmark_results_v4.csv        ← per-user per-policy\n")
        f.write("results/user_level_results_v4.csv       ← same, pivoted\n")
        f.write("results/advanced_metrics_v4.csv         ← mean/std/P50/P90/P95/P99/CI95\n")
        f.write("results/statistical_results_v4.csv      ← p-values, CI, Cohen's d\n")
        f.write("results/ablation_results_v2.csv         ← ablation variants\n")
        f.write("reports/figures/graphmind_vs_markov2.png← user scatter\n")
        f.write("```\n\n")
        f.write("**Dashboard MUST consume these files. No hardcoded values.**\n\n")
        f.write("## Pages\n\n")
        f.write("### `/` — Homepage\n")
        f.write("- Hero metric strip: 31 users | 820K events | 208K transitions | F1={:.3f}\n".format(winning_f1))
        f.write(f"- Avg latency saved: {winning_lat:.0f} ms/launch\n")
        f.write(f"- Winner badge: {winning_policy}\n")
        f.write("- Policy comparison bar chart (F1, 6 key policies)\n")
        f.write("- Animated stat counters on load\n\n")
        f.write("### `/benchmark` — Benchmark Results\n")
        f.write("- All policies ranked by F1 (bar chart with CI error bars)\n")
        f.write("- Hit Rate ranking (bar chart)\n")
        f.write("- Latency saved ranking (bar chart)\n")
        f.write("- `graphmind_vs_markov2.png` scatter embed\n")
        f.write("- P90/P95/P99 distribution table\n")
        f.write("- Policy filter dropdown\n\n")
        f.write("### `/statistical` — Statistical Analysis\n")
        f.write("- Significance heatmap (policy × metric → p-value)\n")
        f.write("- Cohen's d bar chart with effect magnitude labels\n")
        f.write("- CI overlap visualisation\n")
        f.write("- Key findings summary\n\n")
        f.write("### `/ablation` — Ablation Study\n")
        f.write("- Step chart: GraphOnly → +Confidence → +RL → Full GraphMind\n")
        f.write("- Component contribution table\n\n")
        f.write("### `/users` — User Overview\n")
        f.write("- Sortable table: user ID, n_events, n_transitions, best policy\n")
        f.write("- Bubble chart: events × F1 × cluster\n\n")
        f.write("### `/user/[id]` — User Detail\n")
        f.write("- Top-20 app frequency bar chart\n")
        f.write("- Transition heatmap (10×10)\n")
        f.write("- Policy comparison for this user\n")
        f.write("- Hit rate over test timeline\n\n")
        f.write("### `/dataset` — Dataset Overview\n")
        f.write("- UbiqLog stats: 35 users, 820K events, 208K transitions\n")
        f.write("- Event type distribution\n")
        f.write("- Gap sensitivity results (15/30/60 min)\n")
        f.write("- Latency tier chart (cold/warm/hot)\n\n")
        f.write("### `/rl` — RL Details\n")
        f.write("- V3 vs V4 architecture comparison\n")
        f.write("- Predictor weight visualisation (learned ensemble weights)\n")
        f.write("- Training trajectory chart\n\n")
        f.write("### `/user/[id]/playback` — Timeline Playback\n")
        f.write("- Step-by-step app launch replay\n")
        f.write("- HOT/WARM/COLD tier live display\n")
        f.write("- Policy prediction vs actual (hit/miss highlight)\n")
        f.write("- Speed controls: 1× / 5× / 10× / 100×\n\n")
        f.write("## Design System\n\n")
        f.write("- Dark mode primary (#0f1117 bg, #7c3aed accent, #10b981 success)\n")
        f.write("- Glassmorphism cards (backdrop-blur, border opacity)\n")
        f.write("- Micro-animations (framer-motion or CSS transitions)\n")
        f.write("- Inter font (Google Fonts)\n")
        f.write("- Responsive: 320px → 4K\n\n")
        f.write("## Data API\n\n")
        f.write("Create `lib/data.ts` to parse all CSV files at build time.\n")
        f.write("Use Next.js `generateStaticParams` for user pages.\n")
        f.write("No external data fetching needed — all data is local CSV.\n")

    logger.info(f"Written: {dp_path}")
    logger.info(f"\n{'='*40}")
    logger.info(f"FINAL DECISION: {decision}")
    logger.info(f"DASHBOARD POLICY: {winning_policy}")
    logger.info(f"{'='*40}")

    return decision, winning_policy


if __name__ == "__main__":
    main()
