"""
tests/test_phase2_memory.py

Phase 2 gate tests: MemoryManager, ContextEncoder, EventSimulator.
"""

import pytest
import numpy as np

from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED
from src.core.graph_engine import BehaviouralGraph, GraphNode
from src.core.memory_manager import MemoryManager
from src.data.context_encoder import ContextEncoder
from src.data.event_simulator import EventSimulator
from config import settings


def test_memory_manager_hot_promote():
    """promote_to_hot() must add node to HOT tier."""
    g = BehaviouralGraph("user_test")
    node = GraphNode("n1", np.zeros(64), "com.test.app", 5, 2,
                     {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social")
    g.add_node(node)
    mm = MemoryManager("user_test", g)
    result = mm.promote_to_hot("n1")
    assert result is True
    assert mm.is_in_hot("n1")


def test_memory_manager_hot_capacity():
    """HOT tier must not exceed HOT_TIER_CAPACITY."""
    g = BehaviouralGraph("user_cap_test")
    mm = MemoryManager("user_cap_test", g)
    for i in range(settings.HOT_TIER_CAPACITY + 5):
        nid = f"node_{i}"
        g.add_node(GraphNode(nid, np.zeros(64), "app", 5, 2,
                             {"headphones": False, "calendar_near": False, "weekend": False}, i, 1, "social"))
        mm.promote_to_hot(nid)
    stats = mm.get_tier_stats()
    assert stats["hot_count"] <= settings.HOT_TIER_CAPACITY


def test_memory_manager_demote():
    """demote_from_hot() must move node from HOT to WARM."""
    g = BehaviouralGraph("user_test")
    g.add_node(GraphNode("n1", np.zeros(64), "com.test.app", 5, 2,
                         {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social"))
    mm = MemoryManager("user_test", g)
    mm.promote_to_hot("n1")
    result = mm.demote_from_hot("n1")
    assert result is True
    assert not mm.is_in_hot("n1")
    assert mm.is_in_warm("n1")


def test_memory_manager_flush_by_category():
    """flush_hot_by_category() must remove only matching category nodes."""
    old_cap = settings.HOT_TIER_CAPACITY
    settings.HOT_TIER_CAPACITY = 5
    try:
        g = BehaviouralGraph("user_test")
        g.add_node(GraphNode("fin_1", np.zeros(64), "com.hdfcbank.new", 5, 2,
                             {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "financial"))
        g.add_node(GraphNode("soc_1", np.zeros(64), "com.instagram.android", 8, 2,
                             {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social"))
        mm = MemoryManager("user_test", g)
        mm.promote_to_hot("fin_1")
        mm.promote_to_hot("soc_1")
        flushed = mm.flush_hot_by_category("financial")
        assert "fin_1" in flushed
        assert not mm.is_in_hot("fin_1")
        assert mm.is_in_hot("soc_1")
    finally:
        settings.HOT_TIER_CAPACITY = old_cap


def test_memory_manager_tier_stats():
    """get_tier_stats() must return correct schema and capacities."""
    g = BehaviouralGraph("user_test")
    mm = MemoryManager("user_test", g)
    stats = mm.get_tier_stats()
    assert set(stats.keys()) == {"hot_count", "warm_count", "cold_count", "hot_capacity", "warm_capacity"}
    assert stats["hot_capacity"] == settings.HOT_TIER_CAPACITY
    assert stats["warm_capacity"] == settings.WARM_TIER_CAPACITY


def test_context_encoder_output_shape():
    """ContextEncoder.encode() must return numpy array of shape (64,)."""
    enc = ContextEncoder()
    event = {"app_id": "com.instagram.android", "time_bucket": 10, "battery": 75.0,
             "headphones": False, "calendar_event_in_mins": None, "weekend": False}
    result = enc.encode(event)
    assert isinstance(result, np.ndarray)
    assert result.shape == (64,)


def test_context_encoder_deterministic():
    """Same input must produce identical output."""
    enc = ContextEncoder()
    event = {"app_id": "com.slack.android", "time_bucket": 20, "battery": 50.0,
             "headphones": True, "calendar_event_in_mins": 15, "weekend": False}
    r1 = enc.encode(event)
    r2 = enc.encode(event)
    assert np.allclose(r1, r2)


def test_event_simulator_loads():
    """EventSimulator must load events for user_00."""
    sim = EventSimulator("user_00")
    assert len(sim.events) > 0


def test_event_simulator_step_publishes():
    """step() must publish TOPIC_APP_LAUNCHED event."""
    bus = EventBus.get_instance()
    bus.clear_all()
    received = []
    bus.subscribe(TOPIC_APP_LAUNCHED, lambda p: received.append(p))
    sim = EventSimulator("user_00")
    result = sim.step()
    assert result is not None
    assert len(received) == 1
    assert "app_id" in received[0]
    bus.clear_all()


def test_event_simulator_day_advance():
    """step_day() must increment current_day."""
    bus = EventBus.get_instance()
    bus.clear_all()
    sim = EventSimulator("user_00")
    assert sim.current_day == 0
    sim.step_day()
    assert sim.current_day == 1
    sim.step_day()
    assert sim.current_day == 2
    bus.clear_all()
