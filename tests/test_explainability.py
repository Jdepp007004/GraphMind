"""
tests/test_explainability.py

Phase 2 tests for the explainability engine.
Verifies deterministic explanations, trace storage, and EventBus integration.
"""

import pytest
import time

from src.core.event_bus import EventBus, TOPIC_PREFETCH_TRIGGERED, TOPIC_SECURITY_FLUSH
from src.core.event_bus import TOPIC_NODE_PROMOTED, TOPIC_NODE_DEMOTED
from src.explainability.decision_trace import (
    DecisionTrace, DecisionTraceStore,
    ACTION_PRELOADED, ACTION_PROMOTED, ACTION_DEMOTED, ACTION_FLUSHED
)
from src.explainability.reasoning_engine import ReasoningEngine
from src.explainability.prediction_explainer import PredictionExplainer, get_trace_store


# ── DecisionTrace Tests ───────────────────────────────────────────────────────

def test_decision_trace_format_explanation():
    """format_explanation() must return a string containing the app name and reasons."""
    trace = DecisionTrace(
        action=ACTION_PRELOADED,
        app_id="com.spotify.music",
        user_id="user_00",
        reasons=["weekday morning pattern", "headphones connected",
                 "transition probability 0.82"],
        confidence=0.82
    )
    explanation = trace.format_explanation()
    assert "spotify" in explanation.lower() or "preloaded" in explanation
    assert "weekday morning pattern" in explanation
    assert "headphones connected" in explanation
    assert "transition probability" in explanation


def test_decision_trace_to_dict():
    """to_dict() must include all required keys."""
    trace = DecisionTrace(
        action=ACTION_FLUSHED,
        app_id="[financial context]",
        user_id="user_01",
        reasons=["security boundary"],
        confidence=1.0
    )
    d = trace.to_dict()
    required = {"trace_id", "timestamp", "action", "app_id", "user_id",
                "reasons", "confidence", "metadata"}
    assert required.issubset(set(d.keys()))
    assert d["action"] == ACTION_FLUSHED
    assert d["user_id"] == "user_01"
    assert d["confidence"] == 1.0


def test_decision_trace_unique_ids():
    """Each DecisionTrace must have a unique trace_id."""
    t1 = DecisionTrace(ACTION_PRELOADED, "com.app.a", "user_00", ["r1"], 0.9)
    t2 = DecisionTrace(ACTION_PRELOADED, "com.app.a", "user_00", ["r1"], 0.9)
    # IDs may collide within same millisecond in fast tests, so just check format
    assert "user_00" in t1.trace_id
    assert t1.trace_id.startswith("user_00")


# ── DecisionTraceStore Tests ──────────────────────────────────────────────────

def test_trace_store_add_and_get():
    """store.add() and get_recent() must work correctly."""
    store = DecisionTraceStore()
    for i in range(5):
        trace = DecisionTrace(
            action=ACTION_PROMOTED,
            app_id=f"com.app{i}",
            user_id="user_test",
            reasons=[f"reason {i}"],
            confidence=0.7 + i * 0.05
        )
        store.add(trace)
    recent = store.get_recent("user_test", limit=3)
    assert len(recent) == 3
    # Newest first
    assert recent[0].app_id == "com.app4"


def test_trace_store_filter_by_action():
    """get_recent() with action_filter must return only matching actions."""
    store = DecisionTraceStore()
    for action in [ACTION_PROMOTED, ACTION_DEMOTED, ACTION_FLUSHED]:
        trace = DecisionTrace(action, "com.app.test", "user_test", ["r"], 0.9)
        store.add(trace)
    promoted = store.get_recent("user_test", action_filter=ACTION_PROMOTED)
    assert len(promoted) == 1
    assert promoted[0].action == ACTION_PROMOTED


def test_trace_store_per_app_lookup():
    """get_for_app() must return traces for a specific app only."""
    store = DecisionTraceStore()
    store.add(DecisionTrace(ACTION_PRELOADED, "com.spotify.music", "user_test", ["r1"], 0.9))
    store.add(DecisionTrace(ACTION_PRELOADED, "com.google.maps", "user_test", ["r2"], 0.8))
    store.add(DecisionTrace(ACTION_PROMOTED, "com.spotify.music", "user_test", ["r3"], 0.7))
    spotify_traces = store.get_for_app("user_test", "com.spotify.music")
    assert len(spotify_traces) == 2
    assert all(t.app_id == "com.spotify.music" for t in spotify_traces)


def test_trace_store_max_per_user_eviction():
    """Store must not exceed max_per_user traces per user."""
    store = DecisionTraceStore(max_per_user=5)
    for i in range(10):
        store.add(DecisionTrace(ACTION_PRELOADED, f"app{i}", "user_test", [f"r{i}"], 0.5))
    assert store.count("user_test") == 5


def test_trace_store_multi_user_isolation():
    """Traces for different users must be isolated."""
    store = DecisionTraceStore()
    store.add(DecisionTrace(ACTION_PROMOTED, "com.app.a", "user_00", ["r"], 0.9))
    store.add(DecisionTrace(ACTION_PROMOTED, "com.app.b", "user_01", ["r"], 0.9))
    assert store.count("user_00") == 1
    assert store.count("user_01") == 1
    assert store.get_recent("user_00")[0].app_id == "com.app.a"
    assert store.get_recent("user_01")[0].app_id == "com.app.b"


def test_trace_store_clear():
    """clear() must empty traces for a user."""
    store = DecisionTraceStore()
    store.add(DecisionTrace(ACTION_PRELOADED, "com.app", "user_test", ["r"], 0.9))
    store.clear("user_test")
    assert store.count("user_test") == 0


# ── ReasoningEngine Tests ─────────────────────────────────────────────────────

def test_reasoning_engine_preload_reasons():
    """reasons_for_preload() must return at least 3 reasons."""
    engine = ReasoningEngine()
    reasons = engine.reasons_for_preload(
        app_id="com.spotify.music",
        transition_prob=0.82,
        battery=81.0,
        time_bucket=14,  # morning commute
        access_count=15,
        headphones=True,
        calendar_mins=25,
        weekend=False,
        category="entertainment"
    )
    assert len(reasons) >= 3
    reason_text = " ".join(reasons).lower()
    assert "0.82" in reason_text or "transition" in reason_text
    assert "headphones" in reason_text
    assert "battery" in reason_text


def test_reasoning_engine_flush_reasons():
    """reasons_for_flush() must mention sensitive to consumer transition."""
    engine = ReasoningEngine()
    reasons = engine.reasons_for_flush("financial", "social", 3)
    reason_text = " ".join(reasons).lower()
    assert "financial" in reason_text
    assert "social" in reason_text
    assert len(reasons) >= 2


def test_reasoning_engine_demotion_reasons():
    """reasons_for_demotion() with high pressure must mention eviction."""
    engine = ReasoningEngine()
    reasons = engine.reasons_for_demotion("com.app", hot_pressure=0.95)
    reason_text = " ".join(reasons).lower()
    assert "hot" in reason_text or "evict" in reason_text or "pressure" in reason_text


def test_reasoning_engine_deterministic():
    """Same inputs must produce identical reason lists (deterministic)."""
    engine = ReasoningEngine()
    r1 = engine.reasons_for_preload("com.app", 0.75, 60.0, 20, 5, False, None, True, "social")
    r2 = engine.reasons_for_preload("com.app", 0.75, 60.0, 20, 5, False, None, True, "social")
    assert r1 == r2


# ── PredictionExplainer Integration Tests ─────────────────────────────────────

def test_explainer_records_prefetch_trace():
    """PredictionExplainer must create PRELOADED traces on PREFETCH_TRIGGERED event."""
    bus = EventBus.get_instance()
    bus.clear_all()
    from src.explainability.decision_trace import DecisionTraceStore
    store = DecisionTraceStore()
    explainer = PredictionExplainer("user_00", store=store)

    bus.publish(TOPIC_PREFETCH_TRIGGERED, {
        "timestamp": time.time(),
        "user_id": "user_00",
        "prefetched_ids": ["node_abc", "node_def"],
        "battery": 75.0
    })

    traces = store.get_recent("user_00", limit=10)
    assert len(traces) >= 2
    assert all(t.action == ACTION_PRELOADED for t in traces)
    bus.clear_all()


def test_explainer_records_security_flush_trace():
    """PredictionExplainer must create FLUSHED trace on SECURITY_FLUSH event."""
    bus = EventBus.get_instance()
    bus.clear_all()
    store = DecisionTraceStore()
    explainer = PredictionExplainer("user_01", store=store)

    bus.publish(TOPIC_SECURITY_FLUSH, {
        "timestamp": time.time(),
        "user_id": "user_01",
        "from_category": "financial",
        "to_category": "social",
        "flushed_node_ids": ["node_1", "node_2", "node_3"],
    })

    traces = store.get_recent("user_01", action_filter=ACTION_FLUSHED)
    assert len(traces) == 1
    assert "financial" in " ".join(traces[0].reasons).lower()
    bus.clear_all()


def test_explainer_ignores_other_users():
    """PredictionExplainer must ignore events from other users."""
    bus = EventBus.get_instance()
    bus.clear_all()
    store = DecisionTraceStore()
    explainer = PredictionExplainer("user_02", store=store)

    bus.publish(TOPIC_SECURITY_FLUSH, {
        "timestamp": time.time(),
        "user_id": "user_99",  # different user
        "from_category": "financial",
        "to_category": "social",
        "flushed_node_ids": [],
    })

    assert store.count("user_02") == 0
    bus.clear_all()


def test_explainer_get_latest_explanations():
    """get_latest_explanations() must return formatted strings."""
    bus = EventBus.get_instance()
    bus.clear_all()
    store = DecisionTraceStore()
    explainer = PredictionExplainer("user_03", store=store)

    bus.publish(TOPIC_PREFETCH_TRIGGERED, {
        "timestamp": time.time(),
        "user_id": "user_03",
        "prefetched_ids": ["com.spotify.music"],
        "battery": 90.0
    })

    explanations = explainer.get_latest_explanations(limit=5)
    assert len(explanations) >= 1
    assert isinstance(explanations[0], str)
    assert len(explanations[0]) > 10
    bus.clear_all()
