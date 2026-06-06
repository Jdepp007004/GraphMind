"""
src/benchmarks/statistics.py

Statistical evaluation for GraphMind v2 benchmark runs.

Provides:
  - Descriptive statistics: mean, median, std, min, max
  - Bootstrap confidence intervals (non-parametric, works for non-normal distributions)
  - Paired t-test (parametric, for normally distributed deltas)
  - Cohen's d effect size (magnitude of improvement)
  - Summary table generation (for markdown reports)

WHY BOOTSTRAP:
  Sequence prediction metrics (hit rate, F1) are not guaranteed to be
  normally distributed. Bootstrap CIs are distribution-free and valid
  even with small sample sizes.

WHY PAIRED T-TEST:
  When comparing two policies on the same event stream, observations are
  paired (same user, same day). The paired t-test accounts for this
  correlation, increasing statistical power.

WHY COHEN'S D:
  Statistical significance (p < 0.05) does not imply practical significance.
  Cohen's d measures the effect size: small (0.2), medium (0.5), large (0.8).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from config import settings

logger = logging.getLogger(__name__)


class StatisticalEvaluator:
    """
    Computes confidence intervals, hypothesis tests, and effect sizes
    for GraphMind v2 benchmark results.
    """

    def __init__(
        self,
        confidence_level: float = settings.STATS_CONFIDENCE_LEVEL,
        n_bootstrap: int = settings.STATS_BOOTSTRAP_N_SAMPLES,
        rng_seed: int = settings.RANDOM_SEED,
    ) -> None:
        """
        Args:
            confidence_level: Target CI level, default 0.95 (95%).
            n_bootstrap:      Bootstrap resampling iterations (default 10,000).
            rng_seed:         Seed for reproducible bootstrap sampling.
        """
        self._cl = confidence_level
        self._n_bootstrap = n_bootstrap
        self._rng = np.random.default_rng(rng_seed)

    # ── Descriptive Statistics ─────────────────────────────────────────────

    def describe(self, values: List[float]) -> dict:
        """
        Compute descriptive statistics for a list of values.

        Returns:
            dict with: n, mean, median, std, min, max, q25, q75
        """
        if not values:
            return {
                "n": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None, "q25": None, "q75": None,
            }
        arr = np.array(values, dtype=np.float64)
        return {
            "n":      int(len(arr)),
            "mean":   round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "std":    round(float(arr.std(ddof=1) if len(arr) > 1 else 0.0), 4),
            "min":    round(float(arr.min()), 4),
            "max":    round(float(arr.max()), 4),
            "q25":    round(float(np.percentile(arr, 25)), 4),
            "q75":    round(float(np.percentile(arr, 75)), 4),
        }

    # ── Bootstrap Confidence Intervals ────────────────────────────────────

    def bootstrap_ci(
        self,
        values: List[float],
        statistic: str = "mean",
    ) -> Tuple[float, float]:
        """
        Compute a bootstrap confidence interval for the given statistic.

        This is the preferred CI method for sequence prediction metrics
        because it makes no distributional assumptions.

        Algorithm:
          1. Resample `values` with replacement N_BOOTSTRAP times.
          2. Compute the statistic on each resample.
          3. Return the (α/2, 1-α/2) percentiles of the bootstrap distribution.

        Args:
            values:    Observed metric values (e.g., per-user hit rates).
            statistic: One of "mean", "median". Default "mean".

        Returns:
            (lower_bound, upper_bound) for the confidence interval.
            Returns (nan, nan) when sample size < 2.
        """
        if len(values) < 2:
            return (float("nan"), float("nan"))

        arr = np.array(values, dtype=np.float64)
        n = len(arr)
        alpha = 1.0 - self._cl

        stat_fn = np.mean if statistic == "mean" else np.median
        bootstrap_stats = np.array([
            stat_fn(self._rng.choice(arr, size=n, replace=True))
            for _ in range(self._n_bootstrap)
        ])

        lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
        return (round(lower, 4), round(upper, 4))

    # ── Paired t-test ──────────────────────────────────────────────────────

    def paired_t_test(
        self,
        control: List[float],
        treatment: List[float],
    ) -> dict:
        """
        Perform a paired two-sided t-test between control and treatment.

        IMPORTANT: Both lists must have the same length (paired observations).
        Each pair corresponds to the same user / same event stream.

        Args:
            control:   Metric values for the baseline policy.
            treatment: Metric values for the proposed policy.

        Returns:
            dict with:
              t_statistic  : float
              p_value      : float
              significant  : bool (True when p < 1 - confidence_level)
              n_pairs      : int
              mean_delta   : float (treatment - control)
              ci_lower/upper: float (bootstrap CI on the difference)
        """
        if len(control) != len(treatment):
            raise ValueError(
                f"paired_t_test: control and treatment must have equal length, "
                f"got {len(control)} vs {len(treatment)}"
            )
        if len(control) < settings.STATS_MIN_SAMPLES_FOR_TEST:
            return {
                "t_statistic": None,
                "p_value": None,
                "significant": None,
                "n_pairs": len(control),
                "mean_delta": None,
                "ci_lower": None,
                "ci_upper": None,
                "note": f"Insufficient samples (n={len(control)} < "
                        f"STATS_MIN_SAMPLES_FOR_TEST={settings.STATS_MIN_SAMPLES_FOR_TEST})",
            }

        c = np.array(control, dtype=np.float64)
        t = np.array(treatment, dtype=np.float64)
        diff = t - c

        t_stat, p_val = scipy_stats.ttest_rel(t, c)
        alpha = 1.0 - self._cl
        ci_lower, ci_upper = self.bootstrap_ci(list(diff), statistic="mean")

        return {
            "t_statistic": round(float(t_stat), 4),
            "p_value":     round(float(p_val), 6),
            "significant": bool(p_val < alpha),
            "n_pairs":     len(control),
            "mean_delta":  round(float(diff.mean()), 4),
            "ci_lower":    ci_lower,
            "ci_upper":    ci_upper,
        }

    # ── Cohen's d Effect Size ─────────────────────────────────────────────

    def cohens_d(
        self,
        control: List[float],
        treatment: List[float],
    ) -> dict:
        """
        Compute Cohen's d effect size for the difference between two groups.

        Formula:
          d = (mean_treatment - mean_control) / pooled_std

          pooled_std = sqrt( ((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2) )

        Interpretation:
          |d| < 0.2:  negligible
          |d| < 0.5:  small
          |d| < 0.8:  medium
          |d| >= 0.8: large

        Args:
            control:   Baseline policy metric values.
            treatment: Proposed policy metric values.

        Returns:
            dict with: d, magnitude (str), mean_control, mean_treatment, pooled_std
        """
        if not control or not treatment:
            return {"d": None, "magnitude": "undefined"}

        c = np.array(control, dtype=np.float64)
        t = np.array(treatment, dtype=np.float64)

        n1, n2 = len(c), len(t)
        s1, s2 = c.std(ddof=1) if n1 > 1 else 0.0, t.std(ddof=1) if n2 > 1 else 0.0

        if n1 + n2 <= 2:
            pooled_std = max(s1, s2, 1e-9)
        else:
            pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
            pooled_std = float(np.sqrt(max(pooled_var, 1e-18)))

        d = float((t.mean() - c.mean()) / pooled_std)
        magnitude = self._magnitude_label(d)

        return {
            "d":               round(d, 4),
            "magnitude":       magnitude,
            "mean_control":    round(float(c.mean()), 4),
            "mean_treatment":  round(float(t.mean()), 4),
            "pooled_std":      round(pooled_std, 4),
        }

    # ── Full Policy Comparison ────────────────────────────────────────────

    def compare_policies(
        self,
        baseline_name: str,
        treatment_name: str,
        baseline_values: List[float],
        treatment_values: List[float],
        metric_name: str = "cache_hit_rate",
    ) -> dict:
        """
        Run the full statistical comparison between two policies.

        Returns a comprehensive dict suitable for inclusion in a paper's
        results table or supplementary material.

        Returns:
            dict with baseline_name, treatment_name, metric_name,
            baseline_stats, treatment_stats, t_test, effect_size,
            bootstrap_ci_baseline, bootstrap_ci_treatment.
        """
        return {
            "baseline_name":          baseline_name,
            "treatment_name":         treatment_name,
            "metric_name":            metric_name,
            "baseline_stats":         self.describe(baseline_values),
            "treatment_stats":        self.describe(treatment_values),
            "t_test":                 self.paired_t_test(baseline_values, treatment_values),
            "effect_size":            self.cohens_d(baseline_values, treatment_values),
            "bootstrap_ci_baseline":  self.bootstrap_ci(baseline_values),
            "bootstrap_ci_treatment": self.bootstrap_ci(treatment_values),
        }

    # ── Summary Table ─────────────────────────────────────────────────────

    def generate_summary_table(
        self,
        policy_metrics: Dict[str, List[float]],
        metric_name: str = "cache_hit_rate",
    ) -> List[dict]:
        """
        Generate a ranked summary table for all policies on one metric.

        Args:
            policy_metrics: {policy_name: [per-user or per-run values]}.
            metric_name:    Name of the metric being compared.

        Returns:
            List of row dicts sorted by mean descending, each containing:
              rank, policy_name, mean, median, std, ci_lower, ci_upper.
        """
        rows = []
        for policy_name, values in policy_metrics.items():
            stats = self.describe(values)
            ci_lower, ci_upper = self.bootstrap_ci(values)
            rows.append({
                "policy_name": policy_name,
                "metric":      metric_name,
                "mean":        stats["mean"],
                "median":      stats["median"],
                "std":         stats["std"],
                "ci_lower":    ci_lower,
                "ci_upper":    ci_upper,
                "n":           stats["n"],
            })

        # Sort by mean descending (best first)
        rows.sort(key=lambda r: r["mean"] if r["mean"] is not None else -1.0, reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank

        return rows

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _magnitude_label(d: float) -> str:
        """Return a human-readable Cohen's d magnitude label."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        if abs_d < 0.5:
            return "small"
        if abs_d < 0.8:
            return "medium"
        return "large"
