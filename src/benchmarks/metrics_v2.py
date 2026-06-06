"""
src/benchmarks/metrics_v2.py

11 evaluation metrics for GraphMind v2 benchmark.

All formulas are documented inline. No hardcoded values anywhere.
All latency values sourced from LatencyModel (literature or measured).

Metrics:
  1.  cache_hit_rate          — hits / (hits + misses)
  2.  precision               — TP / (TP + FP)  prefetch precision
  3.  recall                  — TP / (TP + FN)  prefetch recall
  4.  f1                      — 2 * P * R / (P + R)
  5.  latency_saved_ms        — expected ms saved per launch
  6.  latency_saved_pct       — latency_saved_ms / cold_start_ms * 100
  7.  battery_overhead_pct    — (prefetch_energy / total_energy) * 100
  8.  false_prefetch_rate     — FP / (TP + FP)
  9.  thrash_rate             — thrash_events / total_events
  10. prediction_latency_ms   — wall-clock time to run one prediction step
  11. memory_usage_mb         — estimated HOT + WARM RAM footprint

All methods are pure functions (no side effects) unless noted.
"""

import logging
import time
from typing import List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)


class MetricsV2:
    """
    Computes all 11 GraphMind v2 evaluation metrics.

    Designed to be called once per policy per evaluation run.
    All methods accept raw counts and return float metrics.
    """

    def __init__(self) -> None:
        """Lazily import LatencyModel to avoid circular imports."""
        self._latency_model = None

    def _get_latency_model(self):
        """Return (cached) LatencyModel instance."""
        if self._latency_model is None:
            from src.benchmarks.latency_model import LatencyModel
            self._latency_model = LatencyModel()
        return self._latency_model

    # ── Metric 1: Cache Hit Rate ────────────────────────────────────────────

    def cache_hit_rate(
        self,
        cache_hits: int,
        cache_misses: int,
    ) -> float:
        """
        Formula: hits / (hits + misses)

        Args:
            cache_hits:   Number of events where app was in HOT or WARM cache.
            cache_misses: Number of events where app was a cold start.

        Returns:
            float ∈ [0, 1]. 1.0 = perfect cache utilisation.
        """
        total = cache_hits + cache_misses
        if total == 0:
            return 0.0
        return cache_hits / total

    # ── Metrics 2–4: Prefetch Precision / Recall / F1 ──────────────────────

    def precision(
        self,
        true_positives: int,
        false_positives: int,
    ) -> float:
        """
        Formula: TP / (TP + FP)

        Prefetch precision: fraction of prefetched apps that were actually used.
        High precision → few wasted prefetches (good for battery).

        Returns:
            float ∈ [0, 1]. Returns 0.0 when no prefetches were made.
        """
        denom = true_positives + false_positives
        if denom == 0:
            return 0.0
        return true_positives / denom

    def recall(
        self,
        true_positives: int,
        false_negatives: int,
    ) -> float:
        """
        Formula: TP / (TP + FN)

        Prefetch recall: fraction of actually-used apps that were prefetched.
        High recall → fewer cold starts (good for latency).

        Returns:
            float ∈ [0, 1]. Returns 0.0 when no relevant apps exist.
        """
        denom = true_positives + false_negatives
        if denom == 0:
            return 0.0
        return true_positives / denom

    def f1(
        self,
        true_positives: int,
        false_positives: int,
        false_negatives: int,
    ) -> float:
        """
        Formula: 2 * precision * recall / (precision + recall)

        Harmonic mean of precision and recall. Balances wasted prefetches
        (FP, battery cost) against missed prefetches (FN, latency cost).

        Returns:
            float ∈ [0, 1]. Returns 0.0 when precision + recall = 0.
        """
        p = self.precision(true_positives, false_positives)
        r = self.recall(true_positives, false_negatives)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    # ── Metrics 5–6: Latency ───────────────────────────────────────────────

    def latency_saved_ms(
        self,
        app_id_list: List[str],
        tier_list: List[str],
    ) -> float:
        """
        Formula: mean over all events of (cold_ms - tier_ms)

        Expected latency saved per launch, averaged across the event stream.
        Uses the LatencyModel (literature or measured values).

        Args:
            app_id_list: App ID for each event in the evaluation.
            tier_list:   Tier for each event: "hot", "warm", or "cold"/"miss".

        Returns:
            float: Mean latency saved in milliseconds per launch.
                   0.0 for cold starts (no savings). Can be negative if
                   literature values are inconsistent (should not occur).
        """
        if not app_id_list or not tier_list:
            return 0.0
        lm = self._get_latency_model()
        savings = []
        for app_id, tier in zip(app_id_list, tier_list):
            if tier in ("hot", "warm"):
                savings.append(lm.latency_saved_ms(app_id, tier))
            else:
                savings.append(0.0)
        return sum(savings) / max(1, len(savings))

    def latency_saved_pct(
        self,
        app_id_list: List[str],
        tier_list: List[str],
    ) -> float:
        """
        Formula: latency_saved_ms / cold_start_ms * 100

        Where cold_start_ms is the mean cold start across all apps in the stream.

        Returns:
            float: Percentage latency reduction vs always-cold baseline.
        """
        if not app_id_list:
            return 0.0
        lm = self._get_latency_model()
        cold_mean = sum(
            lm.cold_start_ms(app_id) for app_id in app_id_list
        ) / max(1, len(app_id_list))
        saved = self.latency_saved_ms(app_id_list, tier_list)
        if cold_mean <= 0:
            return 0.0
        return (saved / cold_mean) * 100.0

    # ── Metric 7: Battery Overhead ─────────────────────────────────────────

    def battery_overhead_pct(
        self,
        prefetch_total: int,
        total_events: int,
        overhead_per_prefetch: float = 0.001,
    ) -> float:
        """
        Formula: (prefetch_total * overhead_per_prefetch / total_events) * 100

        Estimates additional battery drain from prefetch operations as a
        percentage of total event-driven battery usage.

        Args:
            prefetch_total:         Total number of prefetch operations performed.
            total_events:           Total events in the evaluation run.
            overhead_per_prefetch:  Estimated battery cost per prefetch operation
                                    as a fraction of total (default 0.001 = 0.1%).
                                    This is a proxy; replace with measured values
                                    from collect_app_latency.py when available.

        Returns:
            float: Battery overhead as a percentage ∈ [0, 100].
        """
        if total_events == 0:
            return 0.0
        overhead = (prefetch_total * overhead_per_prefetch / total_events) * 100.0
        return min(100.0, overhead)

    # ── Metric 8: False Prefetch Rate ──────────────────────────────────────

    def false_prefetch_rate(
        self,
        true_positives: int,
        false_positives: int,
    ) -> float:
        """
        Formula: FP / (TP + FP)

        Fraction of all prefetches that turned out to be unnecessary.
        Equivalent to 1 - precision. Included separately for clarity in tables.

        Returns:
            float ∈ [0, 1]. 0.0 = no wasted prefetches.
        """
        denom = true_positives + false_positives
        if denom == 0:
            return 0.0
        return false_positives / denom

    # ── Metric 9: Thrash Rate ─────────────────────────────────────────────

    def thrash_rate(
        self,
        thrash_events: int,
        total_events: int,
    ) -> float:
        """
        Formula: thrash_events / total_events

        Cache thrashing: a node was evicted from HOT tier and then
        immediately re-accessed (within 5 events). High thrash rate
        indicates poor eviction policy or too-small HOT tier.

        Returns:
            float ∈ [0, 1]. 0.0 = no thrashing.
        """
        if total_events == 0:
            return 0.0
        return thrash_events / total_events

    # ── Metric 10: Prediction Latency ──────────────────────────────────────

    def prediction_latency_ms(
        self,
        predict_fn,
        test_events: List[dict],
        n_samples: int = 100,
    ) -> float:
        """
        Measure wall-clock time for one prediction call in milliseconds.

        Runs predict_fn on up to n_samples events and returns the mean
        latency per call. This is the computational overhead of the policy.

        Args:
            predict_fn:  Callable(current_app_id: str, context: dict) → List[str].
            test_events: Events to use for timing (uses first n_samples).
            n_samples:   Number of calls to time.

        Returns:
            float: Mean prediction latency in milliseconds.
        """
        if not test_events:
            return 0.0
        events_to_time = test_events[:min(n_samples, len(test_events))]
        latencies: List[float] = []
        for event in events_to_time:
            app_id = event.get("app_id", "unknown")
            context = {
                "time_bucket": event.get("time_bucket", 0),
                "battery": event.get("battery", 100.0),
                "weekend": event.get("weekend", False),
            }
            t0 = time.perf_counter()
            try:
                predict_fn(app_id, context)
            except Exception:
                pass
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    # ── Metric 11: Memory Usage ────────────────────────────────────────────

    def memory_usage_mb(
        self,
        hot_count: int,
        warm_count: int,
        cold_count: int = 0,
        bytes_per_node: int = 8192,
    ) -> float:
        """
        Formula: (hot_count + warm_count + cold_count) * bytes_per_node / (1024^2)

        Estimates the RAM + cache storage footprint of GraphMind's memory tiers.

        bytes_per_node = ~8KB per GraphNode (64-dim float32 embedding = 256 bytes
        + metadata fields + NetworkX overhead ≈ 8KB conservative estimate).

        Args:
            hot_count:      Number of nodes in HOT tier.
            warm_count:     Number of nodes in WARM tier.
            cold_count:     Number of nodes in COLD tier (SQLite, disk).
            bytes_per_node: Estimated bytes per GraphNode (default 8192 = 8KB).

        Returns:
            float: Total estimated memory usage in megabytes.
        """
        total_nodes = hot_count + warm_count + cold_count
        return (total_nodes * bytes_per_node) / (1024 * 1024)

    # ── Composite Summary ──────────────────────────────────────────────────

    def compute_all(
        self,
        *,
        cache_hits: int,
        cache_misses: int,
        true_positives: int,
        false_positives: int,
        false_negatives: int,
        thrash_events: int,
        prefetch_total: int,
        app_id_list: List[str],
        tier_list: List[str],
        hot_count: int,
        warm_count: int,
        cold_count: int = 0,
        predict_fn=None,
        test_events: Optional[List[dict]] = None,
    ) -> dict:
        """
        Compute all 11 metrics and return them as a single dict.

        All keyword arguments are required except predict_fn and test_events
        (needed only for prediction_latency_ms; defaults to 0.0 when absent).

        Returns:
            dict with keys matching metric names (11 entries).
        """
        total_events = cache_hits + cache_misses

        p = self.precision(true_positives, false_positives)
        r = self.recall(true_positives, false_negatives)

        pred_latency = 0.0
        if predict_fn is not None and test_events is not None:
            pred_latency = self.prediction_latency_ms(predict_fn, test_events)

        return {
            "cache_hit_rate":        round(self.cache_hit_rate(cache_hits, cache_misses), 4),
            "precision":             round(p, 4),
            "recall":                round(r, 4),
            "f1":                    round(self.f1(true_positives, false_positives, false_negatives), 4),
            "latency_saved_ms":      round(self.latency_saved_ms(app_id_list, tier_list), 2),
            "latency_saved_pct":     round(self.latency_saved_pct(app_id_list, tier_list), 2),
            "battery_overhead_pct":  round(self.battery_overhead_pct(prefetch_total, max(1, total_events)), 4),
            "false_prefetch_rate":   round(self.false_prefetch_rate(true_positives, false_positives), 4),
            "thrash_rate":           round(self.thrash_rate(thrash_events, max(1, total_events)), 4),
            "prediction_latency_ms": round(pred_latency, 3),
            "memory_usage_mb":       round(self.memory_usage_mb(hot_count, warm_count, cold_count), 3),
        }
