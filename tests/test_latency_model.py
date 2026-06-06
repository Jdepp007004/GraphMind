"""tests/test_latency_model.py — LatencyModel literature + measured modes."""
import csv
import os
import pytest
import tempfile
from src.benchmarks.latency_model import LatencyModel, _LITERATURE_RECORDS
from config import settings


def test_literature_mode_default():
    model = LatencyModel(force_mode="literature")
    assert model.mode == "literature"


def test_literature_cold_start_known_app():
    model = LatencyModel(force_mode="literature")
    ms = model.cold_start_ms("com.instagram.android")
    assert ms == 820.0


def test_literature_warm_start_known_app():
    model = LatencyModel(force_mode="literature")
    ms = model.warm_start_ms("com.instagram.android")
    assert ms == 210.0


def test_literature_hot_start_known_app():
    model = LatencyModel(force_mode="literature")
    ms = model.hot_start_ms("com.instagram.android")
    assert ms == 45.0


def test_literature_fallback_for_unknown():
    model = LatencyModel(force_mode="literature")
    ms = model.cold_start_ms("com.unknown.package")
    assert ms == _LITERATURE_RECORDS["default"]["cold_ms"]


def test_literature_mode_hot_less_than_warm():
    """Hot start must always be faster than warm start."""
    model = LatencyModel(force_mode="literature")
    for app_id, record in _LITERATURE_RECORDS.items():
        if app_id == "default":
            continue
        assert record["hot_ms"] < record["warm_ms"], (
            f"{app_id}: hot_ms={record['hot_ms']} >= warm_ms={record['warm_ms']}"
        )


def test_literature_mode_warm_less_than_cold():
    """Warm start must always be faster than cold start."""
    model = LatencyModel(force_mode="literature")
    for app_id, record in _LITERATURE_RECORDS.items():
        if app_id == "default":
            continue
        assert record["warm_ms"] < record["cold_ms"], (
            f"{app_id}: warm_ms={record['warm_ms']} >= cold_ms={record['cold_ms']}"
        )


def test_latency_saved_hot():
    model = LatencyModel(force_mode="literature")
    saved = model.latency_saved_ms("com.instagram.android", "hot")
    expected = 820.0 - 45.0
    assert abs(saved - expected) < 1e-6


def test_latency_saved_warm():
    model = LatencyModel(force_mode="literature")
    saved = model.latency_saved_ms("com.instagram.android", "warm")
    expected = 820.0 - 210.0
    assert abs(saved - expected) < 1e-6


def test_latency_saved_cold_returns_zero():
    model = LatencyModel(force_mode="literature")
    saved = model.latency_saved_ms("com.instagram.android", "cold")
    assert saved == 0.0


def test_provenance_fields_present():
    """Every literature record must have provenance metadata."""
    required = {"source", "device_class", "android_version", "app_version",
                "measurement_date", "citation", "cold_ms", "warm_ms", "hot_ms"}
    for app_id, record in _LITERATURE_RECORDS.items():
        missing = required - set(record.keys())
        assert not missing, f"{app_id} missing provenance fields: {missing}"


def test_provenance_source_is_literature():
    for app_id, record in _LITERATURE_RECORDS.items():
        assert record["source"] == "literature", \
            f"{app_id} source should be 'literature', got {record['source']}"


def test_measured_mode_loads_csv():
    """LatencyModel loads measured CSV and parses cold_ms values correctly."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["app_id", "start_type", "device_class", "android_version",
                        "app_version", "measurement_date",
                        "mean_ms", "median_ms", "p50_ms", "p95_ms", "p99_ms"]
        )
        writer.writeheader()
        writer.writerow({
            "app_id": "com.test.app", "start_type": "cold",
            "device_class": "Samsung A23", "android_version": "Android 12",
            "app_version": "1.0", "measurement_date": "2024-01",
            "mean_ms": "750.0", "median_ms": "740.0",
            "p50_ms": "740.0", "p95_ms": "900.0", "p99_ms": "950.0",
        })
        tmp_path = f.name

    try:
        model = LatencyModel(force_mode="literature")  # start in lit mode
        # Directly patch CSV path and reload
        import config.settings as s
        original = s.LATENCY_MEASURED_CSV_PATH
        s.LATENCY_MEASURED_CSV_PATH = tmp_path
        model._mode = LatencyModel.MODE_MEASURED
        model._load_measured()
        s.LATENCY_MEASURED_CSV_PATH = original

        cold = model._get_measured("com.test.app", "cold_ms", "mean")
        assert cold == 750.0
    finally:
        os.unlink(tmp_path)



def test_get_record_has_all_provenance():
    model = LatencyModel(force_mode="literature")
    record = model.get_record("com.instagram.android")
    assert "citation" in record
    assert "device_class" in record
    assert "measurement_date" in record


def test_latency_report_no_default_entry():
    model = LatencyModel(force_mode="literature")
    report = model.latency_report()
    app_ids = [r["app_id"] for r in report]
    assert "default" not in app_ids
