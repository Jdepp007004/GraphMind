#!/usr/bin/env python3
"""
scripts/run_fast_statistical_analysis.py

Phase 5: Statistical analysis on fast benchmark results.

Reads: results/benchmark_results_fast.csv

Comparisons:
  - GraphMindRL vs Markov-2
  - GraphMindRL vs GlobalMarkov2
  - GraphMindRL vs GraphOnly
  - GraphMindRL vs Graph+Confidence

Per comparison:
  - 95% bootstrap CI on mean (2000 iterations)
  - Paired t-test (per-user paired observations)
  - Cohen's d effect size

Outputs:
  results/statistical_results_fast.csv
  reports/statistical_analysis.md
"""

import csv
import json
import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")

N_BOOTSTRAP = 2000
ALPHA       = 0.05
RNG         = np.random.default_rng(42)

METRICS_PRIMARY = ["hit_rate", "f1", "latency_saved_ms"]
COMPARISONS = [
    ("GraphMindRL", "Markov-2"),
    ("GraphMindRL", "GlobalMarkov2"),
    ("GraphMindRL", "GraphOnly"),
    ("GraphMindRL", "Graph+Confidence"),
]


def load_results(path: str) -> Dict[str, Dict[str, List[float]]]:
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


def bootstrap_ci(vals: List[float]) -> Tuple[float, float]:
    arr = np.array(vals)
    if len(arr) < 3:
        return (float("nan"), float("nan"))
    boots = [float(np.mean(RNG.choice(arr, size=len(arr), replace=True))) for _ in range(N_BOOTSTRAP)]
    boots.sort()
    return (round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4))


def paired_t(control: List[float], treatment: List[float]) -> dict:
    c, t = np.array(control), np.array(treatment)
    n = min(len(c), len(t))
    if n < 3:
        return {"t_stat": float("nan"), "p_value": float("nan"), "significant": None,
                "mean_delta": float("nan"), "n": n}
    diffs = t[:n] - c[:n]
    t_stat, p_val = stats.ttest_rel(c[:n], t[:n])
    return {
        "t_stat":      round(float(t_stat), 4),
        "p_value":     round(float(p_val), 6),
        "significant": bool(p_val < ALPHA),
        "mean_delta":  round(float(np.mean(diffs)), 4),
        "n":           n,
    }


def cohens_d(control: List[float], treatment: List[float]) -> dict:
    c, t = np.array(control), np.array(treatment)
    n1, n2 = len(c), len(t)
    if n1 < 2 or n2 < 2:
        return {"d": float("nan"), "magnitude": "insufficient_data"}
    s1 = float(c.std(ddof=1)); s2 = float(t.std(ddof=1))
    pooled = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if pooled == 0:
        return {"d": 0.0, "magnitude": "negligible"}
    d = (float(t.mean()) - float(c.mean())) / pooled
    mag = ("large" if abs(d)>=0.8 else
           "medium" if abs(d)>=0.5 else
           "small" if abs(d)>=0.2 else "negligible")
    return {"d": round(d, 4), "magnitude": mag}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    bench_path = os.path.join(RESULTS_DIR, "benchmark_results_fast.csv")
    if not os.path.exists(bench_path):
        logger.error(f"Missing: {bench_path}")
        return

    data = load_results(bench_path)
    logger.info(f"Loaded {len(data)} policies, running {len(COMPARISONS)*len(METRICS_PRIMARY)} comparisons")

    stat_rows = []
    for metric in METRICS_PRIMARY:
        for (treatment_name, control_name) in COMPARISONS:
            if treatment_name not in data or control_name not in data:
                logger.warning(f"Policy not found: {treatment_name} or {control_name}")
                continue

            control   = data[control_name][metric]
            treatment = data[treatment_name][metric]

            ci_c = bootstrap_ci(control)
            ci_t = bootstrap_ci(treatment)
            tt   = paired_t(control, treatment)
            cd   = cohens_d(control, treatment)
            ci_overlap = (ci_c[1] >= ci_t[0] and ci_t[1] >= ci_c[0])

            row = {
                "metric":              metric,
                "treatment":           treatment_name,
                "control":             control_name,
                "treatment_mean":      round(float(np.mean(treatment)) if treatment else 0, 4),
                "control_mean":        round(float(np.mean(control)) if control else 0, 4),
                "mean_improvement":    tt["mean_delta"],
                "treatment_ci95_lo":   ci_t[0],
                "treatment_ci95_hi":   ci_t[1],
                "control_ci95_lo":     ci_c[0],
                "control_ci95_hi":     ci_c[1],
                "ci_overlap":          ci_overlap,
                "t_statistic":         tt["t_stat"],
                "p_value":             tt["p_value"],
                "significant":         tt["significant"],
                "cohens_d":            cd["d"],
                "effect_magnitude":    cd["magnitude"],
                "n_users":             tt["n"],
            }
            stat_rows.append(row)
            logger.info(
                f"{metric:20s} | {treatment_name:14s} vs {control_name:16s} | "
                f"Δ={tt['mean_delta']:+.4f} p={tt['p_value']:.4f} "
                f"sig={'Y' if tt['significant'] else 'N'} "
                f"d={cd['d']:.2f} ({cd['magnitude']})"
            )

    # Write CSV
    csv_path = os.path.join(RESULTS_DIR, "statistical_results_fast.csv")
    if stat_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
            w.writeheader(); w.writerows(stat_rows)
    logger.info(f"Written: {csv_path}")

    # Write markdown
    md_path = os.path.join(REPORTS_DIR, "statistical_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# GraphMind Statistical Analysis\n\n")
        f.write(f"**Method:** Paired t-test + Bootstrap 95% CI + Cohen's d  \n")
        f.write(f"**n:** {stat_rows[0]['n_users'] if stat_rows else '?'} paired user observations  \n")
        f.write(f"**Significance threshold:** α = {ALPHA}  \n")
        f.write(f"**Bootstrap iterations:** {N_BOOTSTRAP}  \n")
        f.write(f"**Comparisons:** GraphMindRL vs Markov-2, GlobalMarkov2, GraphOnly, Graph+Confidence  \n\n")
        f.write("---\n\n")

        for metric in METRICS_PRIMARY:
            f.write(f"## {metric.replace('_',' ').title()}\n\n")
            f.write("| Comparison | Δ (mean) | 95% CI Treatment | p-value | Sig? | Cohen's d | Effect |\n")
            f.write("|-----------|---------|-----------------|---------|------|----------|--------|\n")
            for row in stat_rows:
                if row["metric"] != metric: continue
                sig  = "✅" if row["significant"] else "❌"
                ci_t = f"[{row['treatment_ci95_lo']:.4f}, {row['treatment_ci95_hi']:.4f}]"
                f.write(
                    f"| {row['treatment']} vs {row['control']} "
                    f"| {row['mean_improvement']:+.4f} "
                    f"| {ci_t} "
                    f"| {row['p_value']:.4f} "
                    f"| {sig} "
                    f"| {row['cohens_d']:.3f} "
                    f"| {row['effect_magnitude']} |\n"
                )
            f.write("\n")

        f.write("---\n\n")
        f.write("## Summary\n\n")
        sig_count = sum(1 for r in stat_rows if r["significant"])
        f.write(f"- **{sig_count}/{len(stat_rows)}** comparisons statistically significant (p < {ALPHA})\n\n")

        f.write("### Key Findings\n\n")
        for row in stat_rows:
            if row["metric"] != "f1": continue
            dir_word = "improvement" if row["mean_improvement"] > 0 else "degradation"
            f.write(
                f"- **F1** — GraphMindRL vs {row['control']}: "
                f"{abs(row['mean_improvement']):.4f} absolute {dir_word}, "
                f"p={row['p_value']:.4f}, d={row['cohens_d']:.2f} ({row['effect_magnitude']}), "
                f"sig={'yes' if row['significant'] else 'no'}\n"
            )

    logger.info(f"Written: {md_path}")
    logger.info(f"Significant: {sig_count}/{len(stat_rows)}")


if __name__ == "__main__":
    main()
