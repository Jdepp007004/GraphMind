"""tests/test_ablation.py — Ablation framework smoke tests."""
import pytest
from src.benchmarks.ablation import AblationRunner, run_ablation_comparison_table
from config import settings


def _make_events(n=50, apps=None):
    if apps is None:
        apps = ["com.instagram.android", "com.whatsapp", "com.android.chrome",
                "com.spotify.music", "com.google.youtube"]
    return [
        {
            "app_id": apps[i % len(apps)],
            "battery": 80.0,
            "time_bucket": i % 48,
            "weekend": i % 7 >= 5,
            "headphones": False,
            "calendar_event_in_mins": None,
            "timestamp": float(i),
            "day": i // 10,
            "category": "social",
        }
        for i in range(n)
    ]


@pytest.fixture
def events():
    all_evts = _make_events(100)
    train = all_evts[:80]
    test = all_evts[80:]
    return train, test


def test_ablation_no_rl_runs(events):
    train, test = events
    runner = AblationRunner(user_id="test_no_rl")
    result = runner._run_no_rl(train, test)
    assert "cache_hit_rate" in result
    assert 0.0 <= result["cache_hit_rate"] <= 1.0


def test_ablation_no_graph_runs(events):
    train, test = events
    runner = AblationRunner(user_id="test_no_graph")
    result = runner._run_no_graph(train, test)
    assert "cache_hit_rate" in result


def test_ablation_full_system_runs(events):
    train, test = events
    runner = AblationRunner(user_id="test_full")
    result = runner._run_full_system(train, test)
    assert "cache_hit_rate" in result


def test_ablation_graph_plus_confidence_runs(events):
    train, test = events
    runner = AblationRunner(user_id="test_g_conf")
    result = runner._run_graph_plus_confidence(train, test)
    assert "cache_hit_rate" in result


def test_ablation_graph_confidence_no_rl_runs(events):
    """Graph+Confidence+NoRL variant: the critical 'does RL matter?' experiment."""
    train, test = events
    runner = AblationRunner(user_id="test_gconf_norl")
    result = runner._run_graph_confidence_no_rl(train, test)
    assert "cache_hit_rate" in result


def test_ablation_run_all_returns_all_variants(events):
    train, test = events
    runner = AblationRunner(user_id="test_all")
    results = runner.run_all(train, test)
    expected_variants = {
        settings.ABLATION_NO_RL,
        settings.ABLATION_GRAPH_PLUS_CONFIDENCE,
        settings.ABLATION_GRAPH_CONFIDENCE_NO_RL,
        settings.ABLATION_GRAPH_RL_ONLY,
        settings.ABLATION_FULL_SYSTEM,
        settings.ABLATION_NO_GRAPH,
        settings.ABLATION_NO_CONFIDENCE,
        settings.ABLATION_NO_SECURITY,
        settings.ABLATION_NO_CONTEXT,
    }
    returned = set(results.keys())
    assert expected_variants.issubset(returned), \
        f"Missing variants: {expected_variants - returned}"


def test_ablation_results_include_variant_name(events):
    train, test = events
    runner = AblationRunner(user_id="test_names")
    results = runner.run_all(train, test)
    for variant_name, result in results.items():
        if "error" not in result:
            assert result.get("variant") == variant_name, \
                f"{variant_name}: variant key mismatch"


def test_ablation_results_include_eval_time(events):
    train, test = events
    runner = AblationRunner(user_id="test_time")
    results = runner.run_all(train, test)
    for variant_name, result in results.items():
        assert "eval_time_s" in result, f"{variant_name} missing eval_time_s"


def test_ablation_metrics_in_valid_range(events):
    train, test = events
    runner = AblationRunner(user_id="test_range")
    results = runner.run_all(train, test)
    for variant_name, result in results.items():
        if "error" in result:
            continue
        hit_rate = result.get("cache_hit_rate", 0.0)
        assert 0.0 <= hit_rate <= 1.0, \
            f"{variant_name}: hit_rate {hit_rate} out of [0,1]"
        f1 = result.get("f1", 0.0)
        assert 0.0 <= f1 <= 1.0, \
            f"{variant_name}: f1 {f1} out of [0,1]"


def test_ordered_variants_list_includes_critical_comparison():
    """Verify the ordered variant list for paper table includes critical experiments."""
    assert settings.ABLATION_NO_RL in settings.ABLATION_ORDERED_VARIANTS
    assert settings.ABLATION_GRAPH_PLUS_CONFIDENCE in settings.ABLATION_ORDERED_VARIANTS
    assert settings.ABLATION_GRAPH_CONFIDENCE_NO_RL in settings.ABLATION_ORDERED_VARIANTS
    assert settings.ABLATION_FULL_SYSTEM in settings.ABLATION_ORDERED_VARIANTS


def test_convenience_function(events):
    train, test = events
    results = run_ablation_comparison_table(train, test, user_id="test_conv")
    assert isinstance(results, dict)
    assert len(results) > 0
