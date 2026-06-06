"""
src/benchmarks/evaluator_v2.py

GraphMind v2 evaluation orchestrator.

Runs all 10 baseline policies + ablation studies on the same event stream.
Produces 4 output files:
  results/benchmark_results_v2.csv    — per-policy 11-metric table
  results/advanced_metrics_v2.csv     — additional derived metrics
  results/statistical_results_v2.csv  — bootstrap CIs + t-tests vs GraphOnly
  results/ablation_results_v2.csv     — ablation variant comparison
  results/reports/YYYY-MM-DD_benchmark.md — human-readable markdown report

Usage:
  python -m src.benchmarks.evaluator_v2

  # With specific dataset:
  python -m src.benchmarks.evaluator_v2 --dataset synthetic
  python -m src.benchmarks.evaluator_v2 --dataset device_analyzer

  # With Gemma disabled (default — proves Gemma doesn't inflate results):
  ENABLE_GEMMA=false python -m src.benchmarks.evaluator_v2
"""

import argparse
import csv
import json
import logging
import os
import time
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from config import settings
from src.benchmarks.baselines_v2 import (
    RandomPolicy, LRUPolicy, LFUPolicy, MRUPolicy,
    FrequencyPolicy, RecencyFrequencyPolicy,
    FirstOrderMarkovPolicy, SecondOrderMarkovPolicy,
    GraphOnlyPolicy, GraphMindRLPolicy,
)
from src.benchmarks.metrics_v2 import MetricsV2
from src.benchmarks.statistics import StatisticalEvaluator
from src.benchmarks.ablation import AblationRunner
from src.data.event_dataset import SyntheticDataset, DeviceAnalyzerDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------

ONLINE_POLICIES = [
    RandomPolicy,
    LRUPolicy,
    LFUPolicy,
    MRUPolicy,
    FrequencyPolicy,
    RecencyFrequencyPolicy,
]

MARKOV_POLICIES = [
    FirstOrderMarkovPolicy,
    SecondOrderMarkovPolicy,
]


class BenchmarkEvaluatorV2:
    """
    Orchestrates the full GraphMind v2 benchmark evaluation.

    Workflow:
      1. Load dataset (synthetic or device_analyzer).
      2. Evaluate all 10 baseline policies on test split.
      3. Run ablation experiments.
      4. Compute statistical comparisons.
      5. Write all 4 output files.
      6. Generate markdown report.
    """

    def __init__(
        self,
        dataset_source: str = "synthetic",
        user_id: str = "eval_user",
        top_k: int = settings.PREFETCH_TOP_K,
    ) -> None:
        """
        Args:
            dataset_source: "synthetic" or "device_analyzer".
            user_id:        User ID for GraphMind components.
            top_k:          Default prefetch top-k.
        """
        self._source = dataset_source
        self._user_id = user_id
        self._top_k = top_k
        self._metrics = MetricsV2()
        self._stats = StatisticalEvaluator()
        self._ablation = AblationRunner(user_id=user_id)
        self._dataset = None
        self._train_events: List[dict] = []
        self._val_events: List[dict] = []
        self._test_events: List[dict] = []

        # Verify Gemma is disabled for benchmark runs
        if settings.ENABLE_GEMMA:
            logger.warning(
                "ENABLE_GEMMA=True during benchmark run. "
                "Set ENABLE_GEMMA=false to prove Gemma does not inflate results."
            )
        else:
            logger.info("ENABLE_GEMMA=False — Gemma disabled for this run (correct).")

    def load_dataset(self) -> None:
        """Load and split the dataset."""
        logger.info(f"Loading dataset: {self._source}")
        if self._source == "device_analyzer":
            self._dataset = DeviceAnalyzerDataset(fallback_to_synthetic=True)
        else:
            self._dataset = SyntheticDataset()

        self._dataset.load()
        meta = self._dataset.metadata()
        logger.info(
            f"Dataset loaded: {meta['total_events']} total events — "
            f"train={meta['split_sizes']['train']} "
            f"val={meta['split_sizes']['val']} "
            f"test={meta['split_sizes']['test']}"
        )

        self._train_events = list(self._dataset.iter_events("train"))
        self._val_events = list(self._dataset.iter_events("val"))
        self._test_events = list(self._dataset.iter_events("test"))

    def evaluate_policy(
        self, policy, policy_name: str
    ) -> dict:
        """
        Evaluate one policy on the test split.

        Returns:
            dict with policy_name + all 11 metrics.
        """
        policy.reset()

        # Train Markov policies on train split only
        if hasattr(policy, "train"):
            logger.info(f"  Training {policy_name} on {len(self._train_events)} events...")
            policy.train(self._train_events)
        else:
            # Online: update on training events first
            for event in self._train_events:
                policy.update(event)

        hits, misses, tp, fp, fn, thrash = 0, 0, 0, 0, 0, 0
        app_ids: List[str] = []
        tiers: List[str] = []
        prev_event: Optional[dict] = None
        prev_hot: set = set()

        for event in self._test_events:
            app_id = event.get("app_id", "")
            context = {
                "time_bucket": event.get("time_bucket", 0),
                "battery": event.get("battery", 100.0),
                "weekend": event.get("weekend", False),
            }

            # Check prediction against actual next app
            if prev_event is not None:
                prev_app = prev_event.get("app_id", "")
                predicted = policy.predict_next_apps(prev_app, context)

                if app_id in predicted:
                    hits += 1
                    tiers.append("warm")
                    tp += 1
                else:
                    misses += 1
                    tiers.append("cold")
                    fn += 1

                fp += max(0, len(predicted) - (1 if app_id in predicted else 0))
                app_ids.append(app_id)

            policy.update(event)
            prev_event = event

        total = hits + misses
        result = self._metrics.compute_all(
            cache_hits=hits,
            cache_misses=misses,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            thrash_events=thrash,
            prefetch_total=max(1, tp + fp),
            app_id_list=app_ids,
            tier_list=tiers,
            hot_count=min(settings.HOT_TIER_CAPACITY, hits),
            warm_count=min(settings.WARM_TIER_CAPACITY, total),
            predict_fn=lambda a, c: policy.predict_next_apps(a, c),
            test_events=self._test_events[:100],
        )
        result["policy"] = policy_name
        return result

    def run_all(self) -> Dict[str, object]:
        """
        Run complete evaluation: all policies + ablations + statistics.

        Returns:
            dict with "policy_results", "ablation_results", "statistical_results".
        """
        if not self._test_events:
            self.load_dataset()

        policy_results: List[dict] = []

        # ── Online policies ────────────────────────────────────────────────
        for PolicyClass in ONLINE_POLICIES:
            policy = PolicyClass()
            name = policy.get_name()
            logger.info(f"Evaluating: {name}")
            t0 = time.perf_counter()
            result = self.evaluate_policy(policy, name)
            result["eval_time_s"] = round(time.perf_counter() - t0, 2)
            policy_results.append(result)

        # ── Markov policies ────────────────────────────────────────────────
        for PolicyClass in MARKOV_POLICIES:
            policy = PolicyClass()
            name = policy.get_name()
            logger.info(f"Evaluating: {name}")
            t0 = time.perf_counter()
            result = self.evaluate_policy(policy, name)
            result["eval_time_s"] = round(time.perf_counter() - t0, 2)
            policy_results.append(result)

        # ── GraphOnly ──────────────────────────────────────────────────────
        logger.info("Evaluating: GraphOnly")
        t0 = time.perf_counter()
        graph_policy = GraphOnlyPolicy(user_id=f"{self._user_id}_graph_only")
        graph_result = self.evaluate_policy(graph_policy, settings.BASELINE_V2_GRAPH_ONLY)
        graph_result["eval_time_s"] = round(time.perf_counter() - t0, 2)
        policy_results.append(graph_result)

        # ── GraphMind RL ───────────────────────────────────────────────────
        logger.info("Evaluating: GraphMind_RL (full system via PolicyRunner)")
        t0 = time.perf_counter()
        rl_policy = GraphMindRLPolicy(user_id=f"{self._user_id}_rl", top_k=self._top_k)
        try:
            rl_metrics = rl_policy.run_full_evaluation(self._test_events)
        except Exception as exc:
            logger.error(f"GraphMindRL full evaluation failed: {exc}. Using fallback.")
            rl_metrics = {
                "cache_hit_rate": 0.0, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "error": str(exc),
            }
        rl_metrics["policy"] = settings.BASELINE_V2_GRAPHMIND_RL
        rl_metrics["eval_time_s"] = round(time.perf_counter() - t0, 2)
        policy_results.append(rl_metrics)

        # ── Ablations ──────────────────────────────────────────────────────
        logger.info("Running ablation studies...")
        ablation_results = self._ablation.run_all(
            self._train_events, self._test_events
        )

        # ── Statistical comparisons ────────────────────────────────────────
        logger.info("Computing statistical comparisons...")
        statistical_results = self._compute_statistics(policy_results)

        return {
            "policy_results": policy_results,
            "ablation_results": ablation_results,
            "statistical_results": statistical_results,
            "dataset_meta": self._dataset.metadata() if self._dataset else {},
        }

    def _compute_statistics(self, policy_results: List[dict]) -> List[dict]:
        """
        Compute bootstrap CIs + paired t-tests for each policy vs GraphOnly.
        """
        stats_rows = []
        # Extract GraphOnly hit rates as baseline
        graph_only = next(
            (r for r in policy_results if r.get("policy") == settings.BASELINE_V2_GRAPH_ONLY),
            None
        )
        if graph_only is None:
            return []

        baseline_val = graph_only.get("cache_hit_rate", 0.0)

        for result in policy_results:
            policy_name = result.get("policy", "")
            treatment_val = result.get("cache_hit_rate", 0.0)

            # Single-value comparison: wrap in list for CI computation
            baseline_list = [baseline_val]
            treatment_list = [treatment_val]

            try:
                comparison = self._stats.compare_policies(
                    baseline_name=settings.BASELINE_V2_GRAPH_ONLY,
                    treatment_name=policy_name,
                    baseline_values=baseline_list,
                    treatment_values=treatment_list,
                    metric_name="cache_hit_rate",
                )
            except Exception as exc:
                comparison = {"error": str(exc)}

            comparison["policy"] = policy_name
            comparison["cache_hit_rate"] = treatment_val
            comparison["vs_graph_only_delta"] = round(treatment_val - baseline_val, 4)
            stats_rows.append(comparison)

        return stats_rows

    def write_results(
        self,
        results: Dict[str, object],
        report_prefix: str = "",
    ) -> None:
        """
        Write all output files.

        Args:
            results: Output of run_all().
            report_prefix: Optional prefix for report filename.
        """
        os.makedirs(settings.RESULTS_DIR, exist_ok=True)
        os.makedirs(settings.REPORTS_DIR, exist_ok=True)

        policy_results = results.get("policy_results", [])
        ablation_results = results.get("ablation_results", {})
        statistical_results = results.get("statistical_results", [])

        # ── benchmark_results_v2.csv ────────────────────────────────────────
        if policy_results:
            metric_keys = [
                "policy", "cache_hit_rate", "precision", "recall", "f1",
                "latency_saved_ms", "latency_saved_pct", "battery_overhead_pct",
                "false_prefetch_rate", "thrash_rate",
                "prediction_latency_ms", "memory_usage_mb", "eval_time_s"
            ]
            self._write_csv(policy_results, settings.BENCHMARK_V2_RESULTS_CSV, metric_keys)
            logger.info(f"Written: {settings.BENCHMARK_V2_RESULTS_CSV}")

        # ── ablation_results_v2.csv ────────────────────────────────────────
        if ablation_results:
            ablation_rows = [
                {"variant": k, **v} for k, v in ablation_results.items()
            ]
            ablation_keys = [
                "variant", "cache_hit_rate", "precision", "recall", "f1",
                "latency_saved_ms", "battery_overhead_pct", "f1", "eval_time_s"
            ]
            self._write_csv(ablation_rows, settings.BENCHMARK_V2_ABLATION_CSV, ablation_keys)
            logger.info(f"Written: {settings.BENCHMARK_V2_ABLATION_CSV}")

        # ── statistical_results_v2.csv ──────────────────────────────────────
        if statistical_results:
            stats_flat = []
            for row in statistical_results:
                flat = {
                    "policy":           row.get("policy", ""),
                    "cache_hit_rate":   row.get("cache_hit_rate", ""),
                    "vs_graph_only":    row.get("vs_graph_only_delta", ""),
                    "t_statistic":      row.get("t_test", {}).get("t_statistic", "") if isinstance(row.get("t_test"), dict) else "",
                    "p_value":          row.get("t_test", {}).get("p_value", "") if isinstance(row.get("t_test"), dict) else "",
                    "significant":      row.get("t_test", {}).get("significant", "") if isinstance(row.get("t_test"), dict) else "",
                    "cohens_d":         row.get("effect_size", {}).get("d", "") if isinstance(row.get("effect_size"), dict) else "",
                    "magnitude":        row.get("effect_size", {}).get("magnitude", "") if isinstance(row.get("effect_size"), dict) else "",
                }
                stats_flat.append(flat)
            self._write_csv(stats_flat, settings.BENCHMARK_V2_STATISTICAL_CSV)
            logger.info(f"Written: {settings.BENCHMARK_V2_STATISTICAL_CSV}")

        # ── Markdown report ────────────────────────────────────────────────
        report_path = os.path.join(
            settings.REPORTS_DIR,
            f"{report_prefix}{date.today().isoformat()}_benchmark.md"
        )
        self._write_markdown_report(report_path, results)
        logger.info(f"Report written: {report_path}")

    def _write_csv(
        self,
        rows: List[dict],
        path: str,
        keys: Optional[List[str]] = None,
    ) -> None:
        """Write a list of dicts to CSV."""
        if not rows:
            return
        fieldnames = keys or list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _write_markdown_report(
        self, path: str, results: Dict[str, object]
    ) -> None:
        """Generate a human-readable markdown benchmark report."""
        policy_results = results.get("policy_results", [])
        ablation_results = results.get("ablation_results", {})
        dataset_meta = results.get("dataset_meta", {})

        sorted_policies = sorted(
            policy_results,
            key=lambda r: float(r.get("cache_hit_rate", 0.0)),
            reverse=True
        )

        lines = [
            f"# GraphMind v2 Benchmark Report",
            f"",
            f"**Date**: {date.today().isoformat()}",
            f"**Dataset**: {dataset_meta.get('source', 'unknown')} "
            f"({dataset_meta.get('total_events', 0):,} events)",
            f"**Gemma**: {'Enabled' if settings.ENABLE_GEMMA else 'Disabled (correct for benchmarks)'}",
            f"",
            f"---",
            f"",
            f"## Baseline Comparison (Ranked by Cache Hit Rate)",
            f"",
            f"| Rank | Policy | Hit Rate | F1 | Precision | Recall | Latency Saved | Battery OH |",
            f"|------|--------|----------|-----|-----------|--------|--------------|------------|",
        ]

        for rank, result in enumerate(sorted_policies, 1):
            lines.append(
                f"| {rank} | {result.get('policy', '')} "
                f"| {result.get('cache_hit_rate', 0.0):.4f} "
                f"| {result.get('f1', 0.0):.4f} "
                f"| {result.get('precision', 0.0):.4f} "
                f"| {result.get('recall', 0.0):.4f} "
                f"| {result.get('latency_saved_ms', 0.0):.1f} ms "
                f"| {result.get('battery_overhead_pct', 0.0):.4f}% |"
            )

        lines += [
            f"",
            f"---",
            f"",
            f"## Ablation Study: What Does RL Actually Buy Us?",
            f"",
            f"| Variant | Hit Rate | F1 | Latency Saved |",
            f"|---------|----------|-----|--------------|",
        ]

        ordered = settings.ABLATION_ORDERED_VARIANTS
        for variant in ordered:
            if variant in ablation_results:
                r = ablation_results[variant]
                lines.append(
                    f"| {variant} "
                    f"| {r.get('cache_hit_rate', 'N/A')} "
                    f"| {r.get('f1', 'N/A')} "
                    f"| {r.get('latency_saved_ms', 'N/A')} |"
                )

        lines += [
            f"",
            f"---",
            f"",
            f"## Key Questions",
            f"",
            f"1. **Does GraphMind beat Markov-2 + RecencyFrequency?**",
            f"   This is the primary benchmark to beat.",
            f"",
            f"2. **What does RL add?** Compare Graph+Confidence vs Full_System.",
            f"",
            f"3. **What does Confidence add?** Compare GraphOnly vs Graph+Confidence.",
            f"",
            f"---",
            f"",
            f"*Generated by `src/benchmarks/evaluator_v2.py`*",
        ]

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run GraphMind v2 benchmark evaluation."
    )
    parser.add_argument(
        "--dataset", choices=["synthetic", "device_analyzer"],
        default="synthetic",
        help="Dataset source (default: synthetic).",
    )
    args = parser.parse_args()

    evaluator = BenchmarkEvaluatorV2(dataset_source=args.dataset)
    evaluator.load_dataset()

    logger.info("Starting full benchmark evaluation...")
    t_start = time.perf_counter()
    results = evaluator.run_all()
    elapsed = time.perf_counter() - t_start
    logger.info(f"Evaluation complete in {elapsed:.1f}s.")

    evaluator.write_results(results)
    logger.info("All output files written. Done.")


if __name__ == "__main__":
    main()
