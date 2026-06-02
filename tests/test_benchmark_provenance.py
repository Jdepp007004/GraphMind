"""
Tests for benchmark metric provenance.
"""

from src.benchmarks.advanced_metrics import AdvancedBenchmarkMetrics
from src.benchmarks.evaluator import BenchmarkEvaluator
from src.benchmarks.provenance import (
    BENCHMARK_METRICS, MetricProvenance, metrics_missing_provenance,
    provenance_column
)


def test_provenance_enum_values():
    assert MetricProvenance.MEASURED.value == "MEASURED"
    assert MetricProvenance.ESTIMATED.value == "ESTIMATED"
    assert MetricProvenance.SYNTHETIC.value == "SYNTHETIC"
    assert MetricProvenance.UNKNOWN.value == "UNKNOWN"


def test_benchmark_rows_have_provenance(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "RESULTS_DIR", str(tmp_path))
    evaluator = BenchmarkEvaluator()
    evaluator._user_events = {
        "user_00": [
            {"app_id": "com.instagram.android", "category": "social",
             "time_bucket": 10, "battery": 80.0, "day": 0, "weekend": False},
            {"app_id": "com.whatsapp", "category": "social",
             "time_bucket": 10, "battery": 79.0, "day": 0, "weekend": False},
        ]
    }
    df = evaluator.run_all()
    assert metrics_missing_provenance(df) == []
    for metric in BENCHMARK_METRICS:
        col = provenance_column(metric)
        assert col in df.columns
        assert df[col].notna().all()


def test_advanced_benchmark_rows_have_provenance():
    df = AdvancedBenchmarkMetrics().run_advanced_benchmark()
    metric_cols = [c for c in df.columns
                   if c != "user_id" and not c.endswith("_provenance")]
    assert metric_cols
    for metric in metric_cols:
        col = provenance_column(metric)
        assert col in df.columns
        assert df[col].notna().all()
