#!/usr/bin/env python3
"""
scripts/run_statistical_analysis.py

Phase 6: Statistical analysis of benchmark results.

Reads: results/benchmark_results_v2.csv
Outputs:
  - results/statistical_results_v2.csv
  - reports/statistical_analysis.md

Comparisons:
  - GraphMindRL vs Markov-2
  - GraphMindRL vs GraphOnly
  - GraphMindRL vs Graph+Confidence
"""

import csv
import json
import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")

N_BOOTSTRAP = 2000
ALPHA       = 0.05
RNG_SEED    = 42
rng = np.random.default_rng(RNG_SEED)

METRICS_PRIMARY = ["hit_rate", "f1", "latency_saved_ms"]
COMPARISONS = [
    ("GraphMindRL", "Markov-2"),
    ("GraphMindRL", "GraphOnly"),
    ("GraphMindRL", "Graph+Confidence"),
]


def load_results(path: str) -> Dict[str, Dict[str, List[float]]]:
    """Load benchmark_results_v2.csv → {policy: {metric: [values per user]}}"""
    data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            policy = row["policy"]
            for k, v in row.items():
                if k in ("user_id", "policy"):
                    continue
                try:
                    data[policy][k].append(float(v))
                except ValueError:
                    pass
    return data


def bootstrap_ci(values: List[float], statistic="mean", n=N_BOOTSTRAP, alpha=ALPHA) -> Tuple[float, float]:
    """Bootstrap confidence interval."""
    arr = np.array(values)
    if len(arr) < 3:
        return (float("nan"), float("nan"))
    boot_stats = []
    for _ in range(n):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats.append(float(np.mean(sample)) if statistic == "mean" else float(np.median(sample)))
    boot_stats.sort()
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (round(float(lo), 4), round(float(hi), 4))


def paired_t_test(control: List[float], treatment: List[float]) -> dict:
    """Paired t-test between two per-user value lists."""
    c, t = np.array(control), np.array(treatment)
    n = min(len(c), len(t))
    if n < 3:
        return {"t_statistic": float("nan"), "p_value": float("nan"), "significant": None, "mean_delta": float("nan"), "n": n}
    diffs = t[:n] - c[:n]
    mean_delta = float(np.mean(diffs))
    t_stat, p_val = stats.ttest_rel(c[:n], t[:n])
    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value":     round(float(p_val), 6),
        "significant": bool(p_val < ALPHA),
        "mean_delta":  round(mean_delta, 4),
        "n":           n,
    }


def cohens_d(control: List[float], treatment: List[float]) -> dict:
    """Cohen's d effect size."""
    c, t = np.array(control), np.array(treatment)
    n1, n2 = len(c), len(t)
    if n1 < 2 or n2 < 2:
        return {"d": float("nan"), "magnitude": "insufficient_data"}
    s1, s2 = float(c.std(ddof=1)), float(t.std(ddof=1))
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return {"d": 0.0, "magnitude": "negligible"}
    d = (float(t.mean()) - float(c.mean())) / pooled
    if abs(d) >= 0.8:
        magnitude = "large"
    elif abs(d) >= 0.5:
        magnitude = "medium"
    elif abs(d) >= 0.2:
        magnitude = "small"
    else:
        magnitude = "negligible"
    return {"d": round(d, 4), "magnitude": magnitude}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    benchmark_path = os.path.join(RESULTS_DIR, "benchmark_results_v2.csv")
    if not os.path.exists(benchmark_path):
        logger.error(f"Missing: {benchmark_path} — run run_benchmarks_v2.py first")
        return

    data = load_results(benchmark_path)
    logger.info(f"Loaded data for {len(data)} policies")

    stat_rows = []
    for metric in METRICS_PRIMARY:
        for (treatment_name, control_name) in COMPARISONS:
            if treatment_name not in data or control_name not in data:
                logger.warning(f"Missing policy: {treatment_name} or {control_name}")
                continue

            control   = data[control_name][metric]
            treatment = data[treatment_name][metric]

            ci_control   = bootstrap_ci(control)
            ci_treatment = bootstrap_ci(treatment)
            t_test       = paired_t_test(control, treatment)
            d_effect     = cohens_d(control, treatment)

            # CI overlap
            ci_overlap = ci_control[1] >= ci_treatment[0] and ci_treatment[1] >= ci_control[0]

            row = {
                "metric":             metric,
                "treatment":          treatment_name,
                "control":            control_name,
                "treatment_mean":     round(float(np.mean(treatment)) if treatment else 0, 4),
                "control_mean":       round(float(np.mean(control)) if control else 0, 4),
                "mean_improvement":   t_test["mean_delta"],
                "treatment_ci_lo":    ci_treatment[0],
                "treatment_ci_hi":    ci_treatment[1],
                "control_ci_lo":      ci_control[0],
                "control_ci_hi":      ci_control[1],
                "ci_overlap":         ci_overlap,
                "t_statistic":        t_test["t_statistic"],
                "p_value":            t_test["p_value"],
                "significant":        t_test["significant"],
                "cohens_d":           d_effect["d"],
                "effect_magnitude":   d_effect["magnitude"],
                "n_users":            t_test["n"],
            }
            stat_rows.append(row)
            logger.info(
                f"{metric}: {treatment_name} vs {control_name} — "
                f"Δ={t_test['mean_delta']:+.4f} p={t_test['p_value']:.4f} "
                f"sig={t_test['significant']} d={d_effect['d']:.2f} ({d_effect['magnitude']})"
            )

    # Write CSV
    csv_path = os.path.join(RESULTS_DIR, "statistical_results_v2.csv")
    if stat_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
            w.writeheader()
            w.writerows(stat_rows)
    logger.info(f"Written: {csv_path}")

    # Write markdown report
    md_path = os.path.join(REPORTS_DIR, "statistical_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# GraphMind Statistical Analysis\n\n")
        f.write(f"**Method:** Paired t-test + Bootstrap 95% CI + Cohen's d  \n")
        f.write(f"**Metric:** Per-user values, n={stat_rows[0]['n_users'] if stat_rows else '?'} paired observations  \n")
        f.write(f"**Significance threshold:** α = {ALPHA}  \n")
        f.write(f"**Bootstrap iterations:** {N_BOOTSTRAP}  \n\n")
        f.write("---\n\n")

        for metric in METRICS_PRIMARY:
            f.write(f"## {metric.replace('_', ' ').title()}\n\n")
            f.write("| Comparison | Δ (mean) | p-value | Significant | CI Overlap | Cohen's d | Effect |\n")
            f.write("|------------|----------|---------|-------------|-----------|----------|--------|\n")
            for row in stat_rows:
                if row["metric"] != metric:
                    continue
                sig_icon = "✅ Yes" if row["significant"] else "❌ No"
                overlap  = "⚠ Yes" if row["ci_overlap"] else "No"
                f.write(
                    f"| {row['treatment']} vs {row['control']} "
                    f"| {row['mean_improvement']:+.4f} "
                    f"| {row['p_value']:.4f} "
                    f"| {sig_icon} "
                    f"| {overlap} "
                    f"| {row['cohens_d']:.3f} "
                    f"| {row['effect_magnitude']} |\n"
                )
            f.write("\n")

        f.write("---\n\n")
        f.write("## Interpretation\n\n")
        significant = [r for r in stat_rows if r["significant"]]
        f.write(f"- **{len(significant)}/{len(stat_rows)}** comparisons are statistically significant (p < {ALPHA})\n")

        for row in stat_rows:
            imp = row["mean_improvement"]
            direction = "improvement" if imp > 0 else "degradation"
            f.write(
                f"- **{row['metric']}** — GraphMindRL vs {row['control']}: "
                f"{abs(imp):.4f} absolute {direction}, "
                f"p={row['p_value']:.4f}, "
                f"Cohen's d={row['cohens_d']:.2f} ({row['effect_magnitude']})\n"
            )

    logger.info(f"Written: {md_path}")


if __name__ == "__main__":
    main()
