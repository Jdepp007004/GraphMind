"""
src/benchmarks/provenance.py

Metric provenance labels for benchmark and dashboard reporting.
"""

from enum import Enum
from typing import Iterable

import pandas as pd


class MetricProvenance(str, Enum):
    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    SYNTHETIC = "SYNTHETIC"
    UNKNOWN = "UNKNOWN"


BENCHMARK_METRICS = [
    "cache_hit_rate",
    "launch_speed_gain_pct",
    "thrash_rate",
    "battery_overhead_pct",
    "graph_node_count",
]


def provenance_column(metric_name: str) -> str:
    """Return the provenance column name for a metric value column."""
    return f"{metric_name}_provenance"


def attach_row_provenance(row: dict, measured: Iterable[str],
                          estimated: Iterable[str] = (),
                          synthetic: Iterable[str] = ()) -> dict:
    """Attach provenance labels to benchmark metric fields in a row."""
    measured_set = set(measured)
    estimated_set = set(estimated)
    synthetic_set = set(synthetic)
    for metric in BENCHMARK_METRICS:
        if metric not in row:
            continue
        if metric in measured_set:
            provenance = MetricProvenance.MEASURED
        elif metric in estimated_set:
            provenance = MetricProvenance.ESTIMATED
        elif metric in synthetic_set:
            provenance = MetricProvenance.SYNTHETIC
        else:
            provenance = MetricProvenance.UNKNOWN
        row[provenance_column(metric)] = provenance.value
    return row


def metrics_missing_provenance(df: pd.DataFrame,
                               metrics: Iterable[str] = BENCHMARK_METRICS) -> list:
    """Return metric columns that do not have matching provenance columns."""
    missing = []
    for metric in metrics:
        if metric in df.columns and provenance_column(metric) not in df.columns:
            missing.append(metric)
    return missing
