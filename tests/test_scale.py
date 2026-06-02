"""
Scale test coverage for graph scalability script.
"""

import csv

from scripts.run_scale_test import run_scale_case, run_scale_test


def test_scale_case_reports_required_metrics():
    row = run_scale_case(10, events_per_user=2)
    required = {
        "user_count", "node_count", "edge_count", "memory_usage_mb",
        "serialization_time_ms", "prediction_time_ms", "survived"
    }
    assert required.issubset(row.keys())
    assert row["user_count"] == 10
    assert row["survived"] is True
    assert row["node_count"] > 0


def test_scale_test_writes_csv(tmp_path):
    out = tmp_path / "scale.csv"
    rows = run_scale_test(str(out), user_counts=[10, 100], events_per_user=1)
    assert out.exists()
    assert len(rows) == 2
    with out.open() as f:
        loaded = list(csv.DictReader(f))
    assert loaded[0]["user_count"] == "10"


def test_scale_survives_1000_users():
    row = run_scale_case(1000, events_per_user=1)
    assert row["survived"] is True
    assert row["user_count"] == 1000
    assert row["node_count"] == 1000
