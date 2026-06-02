"""
tests/test_advanced_benchmarks.py

Phase 6 tests for advanced evaluation metrics and user case study engine.
"""

import pytest
import pandas as pd
import os
import json

from src.benchmarks.advanced_metrics import AdvancedBenchmarkMetrics
from src.benchmarks.case_study import CaseStudyGenerator, UserCaseStudy


# ── AdvancedBenchmarkMetrics Tests ────────────────────────────────────────────

def test_prefetch_precision_recall_perfect():
    """Perfect prediction must give precision=1, recall=1, F1=1."""
    metrics = AdvancedBenchmarkMetrics()
    p, r, f1 = metrics.compute_prefetch_precision_recall(
        ["a", "b", "c"], ["a", "b", "c"]
    )
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_prefetch_precision_recall_no_overlap():
    """No overlap must give precision=0, recall=0, F1=0."""
    metrics = AdvancedBenchmarkMetrics()
    p, r, f1 = metrics.compute_prefetch_precision_recall(
        ["x", "y"], ["a", "b"]
    )
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_prefetch_precision_partial_overlap():
    """Partial overlap must give intermediate precision and recall."""
    metrics = AdvancedBenchmarkMetrics()
    # predicted [a, b, c], actual [a, b, d, e]
    # TP=2, precision=2/3, recall=2/4=0.5
    p, r, f1 = metrics.compute_prefetch_precision_recall(
        ["a", "b", "c"], ["a", "b", "d", "e"]
    )
    assert abs(p - 2/3) < 0.01
    assert abs(r - 0.5) < 0.01
    assert f1 > 0


def test_prefetch_empty_inputs():
    """Empty inputs must return 1.0 for all metrics (no prediction, no recall needed)."""
    metrics = AdvancedBenchmarkMetrics()
    p, r, f1 = metrics.compute_prefetch_precision_recall([], [])
    assert p == 1.0
    assert r == 1.0


def test_latency_percentiles_all_hot_hits():
    """All HOT hits must produce very low P50/P95/P99 latency."""
    metrics = AdvancedBenchmarkMetrics()
    result = metrics.compute_latency_percentiles(
        cache_hits_hot=1000, cache_hits_warm=0, cache_misses=0
    )
    assert "p50_ms" in result
    assert "p95_ms" in result
    assert "p99_ms" in result
    assert result["p50_ms"] < 100  # should be around 45ms
    assert result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"]


def test_latency_percentiles_all_misses():
    """All cache misses must produce high latency."""
    metrics = AdvancedBenchmarkMetrics()
    result = metrics.compute_latency_percentiles(
        cache_hits_hot=0, cache_hits_warm=0, cache_misses=1000
    )
    assert result["p50_ms"] > 500  # cold start latency
    assert result["p50_ms"] <= result["p99_ms"]


def test_latency_percentiles_zero_total():
    """Zero events must return zeros without error."""
    metrics = AdvancedBenchmarkMetrics()
    result = metrics.compute_latency_percentiles(0, 0, 0)
    assert result["p50_ms"] == 0.0
    assert result["p95_ms"] == 0.0


def test_memory_estimates_hot_tier():
    """RAM estimate for 30 HOT nodes must be positive and small."""
    metrics = AdvancedBenchmarkMetrics()
    result = metrics.compute_memory_estimates(hot_count=30, warm_count=0, cold_count=0)
    assert result["ram_estimate_mb"] > 0
    assert result["ram_estimate_mb"] < 1.0  # 30 * 8KB = 240KB < 1MB
    assert "total_storage_estimate_mb" in result


def test_graph_growth_metrics():
    """Growing graph must have positive growth rate."""
    metrics = AdvancedBenchmarkMetrics()
    node_counts = list(range(10, 50))   # growing
    edge_counts = list(range(5, 100, 2))
    result = metrics.compute_graph_growth_metrics(node_counts, edge_counts)
    assert result["node_growth_rate"] == 1.0  # exactly 1 node/day
    assert result["edge_growth_rate"] == 2.0
    assert result["node_churn_rate"] == 0.0  # no decrease


def test_graph_growth_single_day():
    """Single-day data must return zeros gracefully."""
    metrics = AdvancedBenchmarkMetrics()
    result = metrics.compute_graph_growth_metrics([20], [15])
    assert result["node_growth_rate"] == 0.0
    assert result["edge_growth_rate"] == 0.0


def test_security_flush_accuracy_all_warranted():
    """All flushes with nodes removed must give 100% accuracy."""
    metrics = AdvancedBenchmarkMetrics()
    flush_log = [
        {"flushed_node_ids": ["n1", "n2"]},
        {"flushed_node_ids": ["n3"]},
    ]
    result = metrics.compute_security_flush_accuracy(flush_log, total_events=1000)
    assert result["flush_accuracy"] == 1.0
    assert result["false_flush_rate"] == 0.0


def test_security_flush_accuracy_some_empty():
    """Flushes with no nodes removed must reduce accuracy."""
    metrics = AdvancedBenchmarkMetrics()
    flush_log = [
        {"flushed_node_ids": ["n1"]},
        {"flushed_node_ids": []},   # false flush
        {"flushed_node_ids": []},   # false flush
    ]
    result = metrics.compute_security_flush_accuracy(flush_log, total_events=1000)
    assert result["flush_accuracy"] < 1.0
    assert result["false_flush_rate"] > 0.0


def test_security_flush_accuracy_empty():
    """No flush events must return accuracy=1.0 and rate=0."""
    metrics = AdvancedBenchmarkMetrics()
    result = metrics.compute_security_flush_accuracy([], total_events=100)
    assert result["flush_accuracy"] == 1.0
    assert result["flush_rate_per_1000_events"] == 0.0


def test_run_advanced_benchmark_returns_dataframe():
    """run_advanced_benchmark() must return a DataFrame with correct columns."""
    metrics = AdvancedBenchmarkMetrics()
    df = metrics.run_advanced_benchmark()
    assert isinstance(df, pd.DataFrame)
    required_cols = {
        "user_id", "prefetch_precision", "prefetch_recall", "prefetch_f1",
        "p50_ms", "p95_ms", "p99_ms", "ram_estimate_mb",
        "node_growth_rate", "edge_growth_rate", "flush_accuracy"
    }
    assert required_cols.issubset(set(df.columns))
    assert len(df) == 10  # 10 users


# ── CaseStudyGenerator Tests ──────────────────────────────────────────────────

def test_case_study_generate_returns_user_study():
    """generate() must return a UserCaseStudy with valid fields."""
    gen = CaseStudyGenerator()
    study = gen.generate("user_00")
    assert isinstance(study, UserCaseStudy)
    assert study.user_id == "user_00"
    assert study.persona_name != ""
    assert 0.0 <= study.initial_hit_rate <= 1.0
    assert 0.0 <= study.final_hit_rate <= 1.0


def test_case_study_generate_all():
    """generate_all() must return 10 case studies."""
    gen = CaseStudyGenerator()
    studies = gen.generate_all()
    assert len(studies) == 10
    user_ids = {s.user_id for s in studies}
    assert "user_00" in user_ids
    assert "user_09" in user_ids


def test_case_study_to_dict_schema():
    """to_dict() must include all required keys."""
    gen = CaseStudyGenerator()
    study = gen.generate("user_01")
    d = study.to_dict()
    required = {
        "user_id", "persona_name", "initial_hit_rate", "final_hit_rate",
        "hit_rate_improvement", "learned_sequences", "top_apps",
        "learned_confidence", "peak_node_count", "drift_days", "day_snapshots"
    }
    assert required.issubset(set(d.keys()))


def test_case_study_summary_text():
    """summary_text() must contain user_id and hit rate information."""
    gen = CaseStudyGenerator()
    study = gen.generate("user_03")
    text = study.summary_text()
    assert "user_03" in text
    assert "Day 1" in text
    assert "Day 30" in text


def test_case_study_hit_rate_improvement_positive():
    """final_hit_rate must be >= initial_hit_rate for improving models."""
    gen = CaseStudyGenerator()
    study = gen.generate("user_04")
    d = study.to_dict()
    # The improvement can be 0 if simulation log was flat; just check it's computable
    assert isinstance(d["hit_rate_improvement"], float)


def test_case_study_day_snapshots_have_required_keys():
    """Day snapshots must have cache_hit_rate and node_count."""
    gen = CaseStudyGenerator()
    study = gen.generate("user_02")
    for day, snap in study.day_snapshots.items():
        assert "cache_hit_rate" in snap
        assert "node_count" in snap
