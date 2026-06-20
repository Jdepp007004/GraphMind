"""
Tests for benchmark metric provenance.
"""

from src.benchmarks.advanced_metrics import AdvancedBenchmarkMetrics
from src.benchmarks.provenance import (
    BENCHMARK_METRICS, MetricProvenance, metrics_missing_provenance,
    provenance_column
)


def test_provenance_enum_values():
    assert MetricProvenance.MEASURED.value == "MEASURED"
    assert MetricProvenance.ESTIMATED.value == "ESTIMATED"
    assert MetricProvenance.SYNTHETIC.value == "SYNTHETIC"
    assert MetricProvenance.UNKNOWN.value == "UNKNOWN"


def test_benchmark_rows_have_provenance():
    from src.benchmarks.provenance import attach_row_provenance, BENCHMARK_METRICS, provenance_column
    import pandas as pd
    row = {
        "cache_hit_rate": 0.5,
        "launch_speed_gain_pct": 0.2,
        "thrash_rate": 0.1,
        "battery_overhead_pct": 0.05,
        "graph_node_count": 10
    }
    row_with_prov = attach_row_provenance(row, measured=["cache_hit_rate"], estimated=["thrash_rate"])
    df = pd.DataFrame([row_with_prov])
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
