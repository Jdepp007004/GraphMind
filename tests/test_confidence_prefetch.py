"""tests/test_confidence_prefetch.py — ConfidencePrefetch scorer."""
import pytest
from unittest.mock import MagicMock, patch
from src.prefetch.confidence_prefetch import ConfidencePrefetch
from config import settings


def _make_mock_graph():
    """Create a minimal mock BehaviouralGraph."""
    graph = MagicMock()
    graph.get_top_k_next_nodes.return_value = ["node_1", "node_2", "node_3"]

    # Mock node objects
    def mock_get_node(node_id):
        n = MagicMock()
        n.app_id = {"node_1": "com.instagram.android",
                    "node_2": "com.whatsapp",
                    "node_3": "com.android.chrome"}.get(node_id, "unknown")
        return n

    graph.get_node.side_effect = mock_get_node
    graph.get_edges_from.return_value = []
    return graph


def test_confidence_prefetch_weights_must_sum_to_one():
    graph = _make_mock_graph()
    with pytest.raises(ValueError, match="sum to 1.0"):
        ConfidencePrefetch(graph, w_transition=0.5, w_recency=0.3, w_frequency=0.3, w_context=0.1)


def test_confidence_prefetch_valid_weights():
    graph = _make_mock_graph()
    scorer = ConfidencePrefetch(graph, w_transition=0.5, w_recency=0.2, w_frequency=0.2, w_context=0.1)
    assert scorer is not None


def test_observe_event_updates_frequency():
    graph = _make_mock_graph()
    scorer = ConfidencePrefetch(graph)
    scorer.observe_event({"app_id": "com.instagram.android", "time_bucket": 10})
    assert scorer._frequency["com.instagram.android"] == 1
    assert scorer._total_events == 1


def test_observe_event_updates_recency():
    graph = _make_mock_graph()
    scorer = ConfidencePrefetch(graph)
    scorer.observe_event({"app_id": "com.instagram.android", "time_bucket": 10})
    assert scorer._recency["com.instagram.android"] > 0


def test_recency_decays_on_each_step():
    graph = _make_mock_graph()
    scorer = ConfidencePrefetch(graph, recency_decay=0.5)
    scorer.observe_event({"app_id": "app_A", "time_bucket": 0})
    r1 = scorer._recency["app_A"]
    scorer.observe_event({"app_id": "app_B", "time_bucket": 0})
    r2 = scorer._recency["app_A"]
    # After second event, app_A's recency should decay by factor 0.5
    assert abs(r2 - r1 * 0.5) < 1e-9


def test_score_candidates_returns_list():
    graph = _make_mock_graph()
    scorer = ConfidencePrefetch(graph, confidence_threshold=0.0)  # accept all
    # Observe some events
    for i in range(10):
        scorer.observe_event({"app_id": "com.instagram.android", "time_bucket": i % 48})
    results = scorer.score_candidates("node_1", 10, battery=80.0)
    assert isinstance(results, list)
    for item in results:
        assert "app_id" in item
        assert "confidence" in item
        assert "transition" in item
        assert "recency" in item
        assert "frequency" in item
        assert "context" in item


def test_score_candidates_threshold_filters():
    graph = _make_mock_graph()
    scorer = ConfidencePrefetch(graph, confidence_threshold=0.99)  # very high
    for i in range(5):
        scorer.observe_event({"app_id": "com.instagram.android", "time_bucket": 0})
    results = scorer.score_candidates("node_1", 0, battery=80.0)
    # All confidences should be >= 0.99 or list is empty
    for item in results:
        assert item["confidence"] >= 0.99


def test_context_score_exact_bucket():
    graph = _make_mock_graph()
    scorer = ConfidencePrefetch(graph)
    # Set up time_bucket observations
    scorer._time_buckets["com.instagram.android"][10] = 5
    score = scorer._compute_context_score("com.instagram.android", 10)
    assert score == 1.0


def test_context_score_near_bucket():
    graph = _make_mock_graph()
    scorer = ConfidencePrefetch(graph)
    scorer._time_buckets["com.instagram.android"][10] = 5
    score = scorer._compute_context_score("com.instagram.android", 12)
    assert score == 0.5


def test_context_score_far_bucket():
    graph = _make_mock_graph()
    scorer = ConfidencePrefetch(graph)
    scorer._time_buckets["com.instagram.android"][10] = 5
    score = scorer._compute_context_score("com.instagram.android", 40)
    assert score == 0.0


def test_reset_clears_state():
    graph = _make_mock_graph()
    scorer = ConfidencePrefetch(graph)
    scorer.observe_event({"app_id": "com.instagram.android", "time_bucket": 0})
    scorer.reset()
    assert scorer._total_events == 0
    assert len(scorer._frequency) == 0
    assert len(scorer._recency) == 0


def test_get_stats():
    graph = _make_mock_graph()
    scorer = ConfidencePrefetch(graph)
    scorer.observe_event({"app_id": "app_A", "time_bucket": 0})
    stats = scorer.get_stats()
    assert stats["total_events_observed"] == 1
    assert stats["unique_apps_tracked"] == 1
    assert "weights" in stats
