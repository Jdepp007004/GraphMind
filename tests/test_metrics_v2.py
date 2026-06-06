"""tests/test_metrics_v2.py — MetricsV2 formula correctness."""
import pytest
from src.benchmarks.metrics_v2 import MetricsV2


@pytest.fixture
def m():
    return MetricsV2()


def test_cache_hit_rate_formula(m):
    assert m.cache_hit_rate(80, 20) == pytest.approx(0.8)


def test_cache_hit_rate_zero_total(m):
    assert m.cache_hit_rate(0, 0) == 0.0


def test_cache_hit_rate_all_hits(m):
    assert m.cache_hit_rate(100, 0) == 1.0


def test_precision_formula(m):
    # TP=80, FP=20 → 80/100 = 0.8
    assert m.precision(80, 20) == pytest.approx(0.8)


def test_precision_zero_predictions(m):
    assert m.precision(0, 0) == 0.0


def test_recall_formula(m):
    # TP=70, FN=30 → 70/100 = 0.7
    assert m.recall(70, 30) == pytest.approx(0.7)


def test_recall_zero_relevant(m):
    assert m.recall(0, 0) == 0.0


def test_f1_formula(m):
    # P=0.8, R=0.7 → F1 = 2*0.8*0.7/(0.8+0.7) ≈ 0.747
    f1 = m.f1(80, 20, 30)
    expected = 2 * 0.8 * (80/110) / (0.8 + 80/110)
    assert f1 == pytest.approx(expected, abs=1e-4)


def test_f1_zero_when_both_zero(m):
    assert m.f1(0, 0, 0) == 0.0


def test_false_prefetch_rate_formula(m):
    # FP=20, TP=80 → 20/100 = 0.2
    assert m.false_prefetch_rate(80, 20) == pytest.approx(0.2)


def test_false_prefetch_rate_zero_predictions(m):
    assert m.false_prefetch_rate(0, 0) == 0.0


def test_thrash_rate_formula(m):
    assert m.thrash_rate(10, 100) == pytest.approx(0.1)


def test_thrash_rate_zero_events(m):
    assert m.thrash_rate(0, 0) == 0.0


def test_battery_overhead_formula(m):
    # 1 prefetch, 100 events, 0.001 overhead_per_prefetch
    # (1 * 0.001 / 100) * 100 = 0.001%
    result = m.battery_overhead_pct(1, 100, overhead_per_prefetch=0.001)
    assert result == pytest.approx(0.001)


def test_memory_usage_mb_formula(m):
    # (10 + 50 + 0) * 8192 / (1024^2)
    expected = (10 + 50) * 8192 / (1024 * 1024)
    result = m.memory_usage_mb(10, 50, 0)
    assert result == pytest.approx(expected)


def test_latency_saved_ms_uses_model(m):
    """With all cold starts, saved should be 0."""
    app_ids = ["com.instagram.android"] * 10
    tiers = ["cold"] * 10
    saved = m.latency_saved_ms(app_ids, tiers)
    assert saved == 0.0


def test_latency_saved_pct_with_hot_hits(m):
    app_ids = ["com.instagram.android"] * 5
    tiers = ["hot"] * 5
    pct = m.latency_saved_pct(app_ids, tiers)
    # Should be positive since hot_ms < cold_ms
    assert pct > 0


def test_compute_all_returns_11_metrics(m):
    result = m.compute_all(
        cache_hits=80,
        cache_misses=20,
        true_positives=70,
        false_positives=15,
        false_negatives=30,
        thrash_events=5,
        prefetch_total=85,
        app_id_list=["com.instagram.android"] * 100,
        tier_list=["hot"] * 80 + ["cold"] * 20,
        hot_count=25,
        warm_count=100,
    )
    expected_keys = {
        "cache_hit_rate", "precision", "recall", "f1",
        "latency_saved_ms", "latency_saved_pct", "battery_overhead_pct",
        "false_prefetch_rate", "thrash_rate", "prediction_latency_ms", "memory_usage_mb"
    }
    assert expected_keys.issubset(result.keys())


def test_precision_plus_false_prefetch_rate_sums_to_one(m):
    """precision(TP, FP) + false_prefetch_rate(TP, FP) must = 1.0"""
    tp, fp = 70, 30
    p = m.precision(tp, fp)
    fpr = m.false_prefetch_rate(tp, fp)
    assert abs(p + fpr - 1.0) < 1e-9
