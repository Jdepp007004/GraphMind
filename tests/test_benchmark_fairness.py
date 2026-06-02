"""
Benchmark fairness tests for execution-derived GraphMind evaluation.
"""

from pathlib import Path

from src.benchmarks.evaluator import BenchmarkEvaluator
from src.benchmarks.graphmind_policy_runner import GraphMindPolicyRunner


def _sample_events():
    return [
        {"app_id": "com.instagram.android", "category": "social",
         "time_bucket": 10, "battery": 82.0, "day": 0, "weekend": False},
        {"app_id": "com.whatsapp", "category": "social",
         "time_bucket": 10, "battery": 81.0, "day": 0, "weekend": False},
        {"app_id": "com.spotify.music", "category": "entertainment",
         "time_bucket": 11, "battery": 80.0, "day": 0, "weekend": False},
        {"app_id": "com.whatsapp", "category": "social",
         "time_bucket": 10, "battery": 79.0, "day": 0, "weekend": False},
    ]


def test_no_graphmind_boost_constants_in_benchmark_source():
    source_files = [
        Path("src/benchmarks/evaluator.py"),
        Path("src/benchmarks/graphmind_policy_runner.py"),
    ]
    combined = "\n".join(p.read_text(encoding="utf-8") for p in source_files)
    forbidden = ["_GRAPHMIND_HIT_BOOST", "lmkd_rate +", "guarantee we beat"]
    for token in forbidden:
        assert token not in combined


def test_graphmind_runner_derives_results_from_execution():
    runner = GraphMindPolicyRunner("fairness_user")
    result = runner.run(_sample_events())

    assert result["cache_hits"] + result["cache_misses"] == len(_sample_events())
    assert result["graph_node_count"] > 0
    assert result["records"]
    assert all("tier" in r and "cache_hit" in r for r in result["records"])


def test_evaluator_uses_graphmind_execution_path(monkeypatch):
    called = {"count": 0}

    def fake_run(self, events):
        called["count"] += 1
        return {
            "cache_hit_rate": 0.25,
            "thrash_rate": 0.0,
            "battery_overhead_pct": 0.01,
            "graph_node_count": 3,
        }

    monkeypatch.setattr(GraphMindPolicyRunner, "run", fake_run)
    evaluator = BenchmarkEvaluator()
    result = evaluator.run_graphmind_policy("fairness_user", _sample_events())

    assert called["count"] == 1
    assert result["cache_hit_rate"] == 0.25
