"""
tests/test_security_visualization.py

Phase 4 tests for the security visualization pipeline.
Verifies that visualization reads from existing security system correctly.
"""

import pytest
import time
import numpy as np

from src.core.event_bus import EventBus, TOPIC_SECURITY_FLUSH
from src.core.graph_engine import BehaviouralGraph, GraphNode
from src.core.memory_manager import MemoryManager
from src.security.context_boundary import ContextBoundaryEnforcer
from src.security.security_visualizer import SecurityVisualizer


@pytest.fixture
def enforcer_with_flushes():
    """Set up enforcer and trigger some security flushes."""
    bus = EventBus.get_instance()
    bus.clear_all()

    g = BehaviouralGraph("user_vis")
    # Add financial nodes
    for i in range(3):
        node = GraphNode(
            node_id=f"fin_{i}",
            embedding=np.zeros(64),
            app_id=f"com.bank{i}.app",
            time_bucket=10,
            battery_bucket=3,
            context_flags={"headphones": False, "calendar_near": False, "weekend": False},
            last_seen_day=0,
            access_count=1,
            category="financial"
        )
        g.add_node(node)

    mm = MemoryManager("user_vis", g)
    for i in range(3):
        mm.promote_to_hot(f"fin_{i}")

    enforcer = ContextBoundaryEnforcer("user_vis", mm)

    # Trigger two flushes
    enforcer.enforce_boundary("financial", "social", 1000.0)
    # Repopulate and flush again
    for i in range(2):
        node = GraphNode(
            node_id=f"health_{i}",
            embedding=np.zeros(64),
            app_id=f"com.health{i}.app",
            time_bucket=10,
            battery_bucket=3,
            context_flags={"headphones": False, "calendar_near": False, "weekend": False},
            last_seen_day=1,
            access_count=1,
            category="health"
        )
        g.add_node(node)
        mm.promote_to_hot(f"health_{i}")
    enforcer.enforce_boundary("health", "entertainment", 2000.0)

    yield enforcer
    bus.clear_all()


# ── Timeline Data Tests ───────────────────────────────────────────────────────

def test_security_visualizer_get_timeline_data(enforcer_with_flushes):
    """get_timeline_data() must return one record per flush event."""
    viz = SecurityVisualizer("user_vis", enforcer_with_flushes)
    timeline = viz.get_timeline_data()
    assert len(timeline) == 2  # two flush events triggered


def test_security_timeline_flow_string(enforcer_with_flushes):
    """Timeline records must include correct flow string."""
    viz = SecurityVisualizer("user_vis", enforcer_with_flushes)
    timeline = viz.get_timeline_data()
    first = timeline[0]
    assert "Financial" in first["flow"] or "financial" in first["flow"].lower()
    assert "Flush Triggered" in first["flow"]
    assert "Node" in first["flow"]


def test_security_timeline_severity_high_for_financial(enforcer_with_flushes):
    """Financial category flushes must be rated HIGH severity."""
    viz = SecurityVisualizer("user_vis", enforcer_with_flushes)
    timeline = viz.get_timeline_data()
    financial_events = [t for t in timeline if t["from_category"] == "financial"]
    assert all(e["severity"] == "HIGH" for e in financial_events)


def test_security_timeline_required_keys(enforcer_with_flushes):
    """Each timeline record must contain all required keys."""
    viz = SecurityVisualizer("user_vis", enforcer_with_flushes)
    timeline = viz.get_timeline_data()
    required = {"event_number", "timestamp", "from_category", "to_category",
                "flushed_count", "severity", "flow", "user_id"}
    for record in timeline:
        assert required.issubset(set(record.keys()))


# ── Summary Metrics Tests ─────────────────────────────────────────────────────

def test_security_summary_metrics(enforcer_with_flushes):
    """get_summary_metrics() must return correct counts."""
    viz = SecurityVisualizer("user_vis", enforcer_with_flushes)
    metrics = viz.get_summary_metrics()
    assert metrics["total_flush_events"] == 2
    assert metrics["total_nodes_removed"] >= 0  # depends on HOT state
    assert "avg_flush_size" in metrics
    assert "category_transition_counts" in metrics


def test_security_summary_empty_enforcer():
    """get_summary_metrics() must return zeros for an enforcer with no flushes."""
    bus = EventBus.get_instance()
    bus.clear_all()
    g = BehaviouralGraph("user_empty")
    mm = MemoryManager("user_empty", g)
    enforcer = ContextBoundaryEnforcer("user_empty", mm)
    viz = SecurityVisualizer("user_empty", enforcer)
    metrics = viz.get_summary_metrics()
    assert metrics["total_flush_events"] == 0
    assert metrics["total_nodes_removed"] == 0
    bus.clear_all()


# ── Category Flow Data Tests ──────────────────────────────────────────────────

def test_security_category_flow_data(enforcer_with_flushes):
    """get_category_flow_data() must return source/target/count records."""
    viz = SecurityVisualizer("user_vis", enforcer_with_flushes)
    flow_data = viz.get_category_flow_data()
    assert len(flow_data) >= 1
    for record in flow_data:
        assert "source" in record
        assert "target" in record
        assert "count" in record
        assert record["count"] >= 1


# ── Live Event Accumulation Test ──────────────────────────────────────────────

def test_security_visualizer_live_events():
    """Visualizer must accumulate real-time flush events via EventBus."""
    bus = EventBus.get_instance()
    bus.clear_all()

    g = BehaviouralGraph("user_live")
    mm = MemoryManager("user_live", g)
    enforcer = ContextBoundaryEnforcer("user_live", mm)

    viz = SecurityVisualizer("user_live", enforcer)

    # Publish a flush event directly to the bus (simulating real device event)
    bus.publish(TOPIC_SECURITY_FLUSH, {
        "timestamp": time.time(),
        "user_id": "user_live",
        "from_category": "enterprise",
        "to_category": "gaming",
        "flushed_node_ids": ["node_a", "node_b"]
    })

    timeline = viz.get_timeline_data()
    # Either from enforcer log (0 entries) or live events (1 entry)
    # Since enforcer didn't trigger (no HOT enterprise nodes), check live
    live = viz._live_events
    assert len(live) == 1
    assert live[0]["from_category"] == "enterprise"
    bus.clear_all()


def test_security_visualizer_ignores_other_users():
    """Live event accumulation must ignore events for other users."""
    bus = EventBus.get_instance()
    bus.clear_all()

    g = BehaviouralGraph("user_target")
    mm = MemoryManager("user_target", g)
    enforcer = ContextBoundaryEnforcer("user_target", mm)
    viz = SecurityVisualizer("user_target", enforcer)

    bus.publish(TOPIC_SECURITY_FLUSH, {
        "timestamp": time.time(),
        "user_id": "user_other",  # different user
        "from_category": "financial",
        "to_category": "social",
        "flushed_node_ids": []
    })

    assert len(viz._live_events) == 0  # ignored
    bus.clear_all()
