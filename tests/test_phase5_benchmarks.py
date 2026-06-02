"""
tests/test_phase5_benchmarks.py

Phase 5 gate tests: baselines, evaluator, benchmark results CSV.
"""

import os
import pytest
import pandas as pd

from config import settings
from src.benchmarks.baselines import (
    LMKDReactiveBaseline, ARTStaticProfileBaseline,
    UsageStatsLRUBaseline, BixbyFrequencyBaseline
)
from src.benchmarks.evaluator import BenchmarkEvaluator


def test_baselines_importable():
    """All 4 baseline classes must be importable."""
    from src.benchmarks import baselines
    for cls in ["BaselinePolicy", "LMKDReactiveBaseline", "ARTStaticProfileBaseline",
                "UsageStatsLRUBaseline", "BixbyFrequencyBaseline"]:
        assert hasattr(baselines, cls)


def test_lmkd_baseline_predicts():
    """LMKD must return app predictions after updates."""
    bl = LMKDReactiveBaseline()
    for app in ["com.instagram.android", "com.whatsapp", "com.spotify.music"]:
        bl.update({"app_id": app, "time_bucket": 10, "battery": 80.0, "day": 1, "weekend": False})
    result = bl.predict_next_apps("com.instagram.android", {"time_bucket": 10, "battery": 80.0})
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(a, str) for a in result)


def test_art_profile_frozen():
    """ART profile must be frozen after build_profile()."""
    bl = ARTStaticProfileBaseline()
    events = []
    for day in range(7):
        for _ in range(10):
            events.append({"app_id": "com.instagram.android", "time_bucket": 10,
                           "battery": 80.0, "day": day, "weekend": False})
    bl.build_profile(events)
    predictions_before = bl.predict_next_apps("any", {"time_bucket": 10, "battery": 80.0})
    for _ in range(20):
        bl.update({"app_id": "com.totally.different.app", "time_bucket": 10,
                   "battery": 80.0, "day": 15, "weekend": False})
    predictions_after = bl.predict_next_apps("any", {"time_bucket": 10, "battery": 80.0})
    assert predictions_before == predictions_after


def test_lru_baseline_most_recent_first():
    """LRU must put most recent app first."""
    bl = UsageStatsLRUBaseline()
    apps = ["app_a", "app_b", "app_c", "app_d", "app_e"]
    for app in apps:
        bl.update({"app_id": app, "time_bucket": 5, "battery": 60.0, "day": 1, "weekend": False})
    result = bl.predict_next_apps("app_e", {"time_bucket": 5, "battery": 60.0})
    assert len(result) > 0
    assert result[0] == "app_e"


def test_bixby_time_context():
    """Bixby must return different predictions for different time buckets."""
    bl = BixbyFrequencyBaseline()
    for _ in range(20):
        bl.update({"app_id": "com.instagram.android", "time_bucket": 10,
                   "battery": 80.0, "day": 1, "weekend": False})
    result_morning = bl.predict_next_apps("any", {"time_bucket": 10, "battery": 80.0, "weekend": False})
    result_night = bl.predict_next_apps("any", {"time_bucket": 40, "battery": 80.0, "weekend": False})
    assert "com.instagram.android" in result_morning
    assert result_morning != result_night or len(result_night) == 0


def test_evaluator_importable():
    """BenchmarkEvaluator must instantiate without error."""
    evaluator = BenchmarkEvaluator()
    assert evaluator is not None


def test_benchmark_results_exist():
    """benchmark_results.csv must exist with >= 50 rows."""
    path = os.path.join(settings.RESULTS_DIR, "benchmark_results.csv")
    assert os.path.exists(path), f"Missing: {path}"
    df = pd.read_csv(path)
    assert len(df) >= 50


def test_benchmark_results_schema():
    """benchmark_results.csv must have all required columns and policies."""
    path = os.path.join(settings.RESULTS_DIR, "benchmark_results.csv")
    df = pd.read_csv(path)
    required_cols = {"user_id", "policy_name", "day", "cache_hit_rate",
                     "launch_speed_gain_pct", "thrash_rate", "battery_overhead_pct"}
    assert not (required_cols - set(df.columns))
    policies = set(df["policy_name"].unique())
    expected_policies = {settings.BASELINE_LMKD, settings.BASELINE_ART,
                         settings.BASELINE_LRU, settings.BASELINE_BIXBY,
                         settings.BASELINE_GRAPHMIND}
    assert not (expected_policies - policies)
