"""
tests/test_drift_visualization.py

Phase 5 tests for the drift intelligence visualization layer.
Verifies KL history tracking, state classification, and metrics.
"""

import pytest
import time

from src.core.event_bus import EventBus, TOPIC_DRIFT_DETECTED, TOPIC_APP_LAUNCHED
from src.agents.drift_visualizer import (
    DriftVisualizer, DriftEvent,
    STATE_NORMAL, STATE_DRIFT, STATE_KL_SPIKE,
    STATE_ADAPTATION, STATE_CONVERGENCE
)
from config import settings


# ── DriftEvent Tests ──────────────────────────────────────────────────────────

def test_drift_event_to_dict():
    """DriftEvent.to_dict() must include all required keys."""
    event = DriftEvent(timestamp=1000.0, kl_value=0.45,
                       state=STATE_KL_SPIKE, user_id="user_00")
    d = event.to_dict()
    assert d["kl_value"] == 0.45
    assert d["state"] == STATE_KL_SPIKE
    assert d["user_id"] == "user_00"


# ── DriftVisualizer Core Tests ────────────────────────────────────────────────

def test_drift_visualizer_init():
    """DriftVisualizer must initialize with empty history."""
    bus = EventBus.get_instance()
    bus.clear_all()
    viz = DriftVisualizer("user_00")
    assert viz.get_kl_history() == []
    assert viz.get_drift_events() == []
    assert viz.get_current_state() == STATE_NORMAL
    bus.clear_all()


def test_drift_visualizer_inject_normal_kl():
    """inject_kl_reading() with low KL must classify as NORMAL."""
    bus = EventBus.get_instance()
    bus.clear_all()
    viz = DriftVisualizer("user_00")
    viz.inject_kl_reading(0.05)
    history = viz.get_kl_history()
    assert len(history) == 1
    assert history[0]["state"] == STATE_NORMAL
    bus.clear_all()


def test_drift_visualizer_inject_spike_kl():
    """inject_kl_reading() with KL above threshold must classify as KL_SPIKE or DRIFT."""
    bus = EventBus.get_instance()
    bus.clear_all()
    viz = DriftVisualizer("user_05")
    viz.inject_kl_reading(0.6)  # > 0.3 threshold, > 0.45 spike threshold
    history = viz.get_kl_history()
    assert history[0]["state"] in (STATE_KL_SPIKE, STATE_DRIFT)
    bus.clear_all()


def test_drift_visualizer_records_drift_event_on_bus():
    """Drift events published to EventBus must be recorded."""
    bus = EventBus.get_instance()
    bus.clear_all()
    viz = DriftVisualizer("user_02")

    bus.publish(TOPIC_DRIFT_DETECTED, {
        "timestamp": time.time(),
        "user_id": "user_02",
        "kl_divergence": 0.55,
    })

    events = viz.get_drift_events()
    assert len(events) == 1
    assert events[0]["kl_value"] == 0.55
    assert events[0]["state"] == STATE_KL_SPIKE
    bus.clear_all()


def test_drift_visualizer_ignores_other_user_drift():
    """Drift events for other users must not be recorded."""
    bus = EventBus.get_instance()
    bus.clear_all()
    viz = DriftVisualizer("user_03")

    bus.publish(TOPIC_DRIFT_DETECTED, {
        "timestamp": time.time(),
        "user_id": "user_99",
        "kl_divergence": 0.8,
    })

    assert viz.get_drift_events() == []
    bus.clear_all()


def test_drift_visualizer_kl_history_max(monkeypatch):
    """History must not exceed max_history entries."""
    bus = EventBus.get_instance()
    bus.clear_all()
    viz = DriftVisualizer("user_04", max_history=10)
    for i in range(20):
        viz.inject_kl_reading(0.1 * (i % 5))
    assert len(viz.get_kl_history()) == 10
    bus.clear_all()


# ── Adaptation Metrics Tests ──────────────────────────────────────────────────

def test_adaptation_metrics_no_events():
    """get_adaptation_metrics() with no events must return safe defaults."""
    bus = EventBus.get_instance()
    bus.clear_all()
    viz = DriftVisualizer("user_06")
    metrics = viz.get_adaptation_metrics()
    required_keys = {
        "total_drift_events", "avg_recovery_time_s",
        "min_recovery_time_s", "max_recovery_time_s",
        "current_kl", "current_state", "adaptation_half_life_events"
    }
    assert required_keys.issubset(set(metrics.keys()))
    assert metrics["total_drift_events"] == 0
    assert metrics["current_state"] == STATE_NORMAL
    bus.clear_all()


def test_adaptation_metrics_after_spike():
    """get_adaptation_metrics() must reflect drift events after a spike."""
    bus = EventBus.get_instance()
    bus.clear_all()
    viz = DriftVisualizer("user_07")

    bus.publish(TOPIC_DRIFT_DETECTED, {
        "timestamp": time.time(),
        "user_id": "user_07",
        "kl_divergence": 0.7,
    })

    metrics = viz.get_adaptation_metrics()
    assert metrics["total_drift_events"] == 1
    assert metrics["current_kl"] == 0.0  # no inject_kl_reading yet
    bus.clear_all()


# ── State Transitions Tests ────────────────────────────────────────────────────

def test_state_transitions_on_kl_sequence():
    """State transitions must follow NORMAL -> DRIFT -> ADAPTATION -> CONVERGENCE."""
    bus = EventBus.get_instance()
    bus.clear_all()
    viz = DriftVisualizer("user_08")

    # Normal phase
    for _ in range(3):
        viz.inject_kl_reading(0.05)

    # Spike
    viz.inject_kl_reading(0.8)

    # Adaptation
    viz.inject_kl_reading(0.35)

    # Convergence
    viz.inject_kl_reading(0.05)

    history = viz.get_kl_history()
    states = [h["state"] for h in history]
    assert STATE_NORMAL in states
    assert STATE_KL_SPIKE in states or STATE_DRIFT in states
    bus.clear_all()


def test_get_state_transitions_returns_list():
    """get_state_transitions() must return a list of transition records."""
    bus = EventBus.get_instance()
    bus.clear_all()
    viz = DriftVisualizer("user_09")
    viz.inject_kl_reading(0.1)
    viz.inject_kl_reading(0.6)
    transitions = viz.get_state_transitions()
    assert isinstance(transitions, list)
    for t in transitions:
        assert "from_state" in t
        assert "to_state" in t
        assert "kl_value" in t
    bus.clear_all()


def test_timeline_data_matches_kl_history():
    """get_timeline_data() must return same data as get_kl_history()."""
    bus = EventBus.get_instance()
    bus.clear_all()
    viz = DriftVisualizer("user_10")
    for kl in [0.1, 0.2, 0.5, 0.4, 0.1]:
        viz.inject_kl_reading(kl)
    assert viz.get_timeline_data() == viz.get_kl_history(limit=500)
    bus.clear_all()
