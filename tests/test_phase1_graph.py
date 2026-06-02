"""
tests/test_phase1_graph.py

Phase 1 gate tests: EventBus, BehaviouralGraph, and dataset validation.
"""

import os
import json
import tempfile
import pytest
import numpy as np

from src.core.event_bus import EventBus
from src.core.graph_engine import BehaviouralGraph, GraphNode, GraphEdge
from config import settings


def test_event_bus_singleton():
    """EventBus.get_instance() must return the same object every time."""
    EventBus._instance = None
    a = EventBus.get_instance()
    b = EventBus.get_instance()
    assert a is b


def test_event_bus_pubsub():
    """Subscribe + publish must invoke callback with correct payload."""
    bus = EventBus.get_instance()
    bus.clear_all()
    received = []
    bus.subscribe("test", lambda p: received.append(p))
    bus.publish("test", {"timestamp": 1.0, "value": 99})
    assert len(received) == 1
    assert received[0]["value"] == 99
    bus.clear_all()


def test_event_bus_unsubscribe():
    """Unsubscribed callbacks must not be called."""
    bus = EventBus.get_instance()
    bus.clear_all()
    received = []
    cb = lambda p: received.append(p)
    bus.subscribe("t", cb)
    bus.unsubscribe("t", cb)
    bus.publish("t", {"timestamp": 1.0})
    assert len(received) == 0
    bus.clear_all()


def test_graph_add_node():
    """add_node() and get_node() must work correctly."""
    g = BehaviouralGraph("user_test")
    node = GraphNode("n1", np.zeros(64), "com.test.app", 5, 2,
                     {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social")
    g.add_node(node)
    assert g.get_node("n1") is not None
    assert g.get_node("n1").app_id == "com.test.app"
    assert g.node_count() == 1


def test_graph_add_edge():
    """add_edge() must create edge; invalid nodes must raise ValueError."""
    g = BehaviouralGraph("user_test")
    for nid in ["n1", "n2"]:
        g.add_node(GraphNode(nid, np.zeros(64), "app", 5, 2,
                             {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social"))
    g.add_edge("n1", "n2", 0.5, 0.3, 0.2)
    edges = g.get_edges_from("n1")
    assert len(edges) == 1
    assert edges[0].transition_prob == 0.5
    with pytest.raises(ValueError):
        g.add_edge("n1", "nonexistent", 0.1, 0.1, 0.1)


def test_graph_pruning():
    """prune_weak_edges() must remove edges below EDGE_PRUNE_THRESHOLD."""
    g = BehaviouralGraph("user_test")
    for nid in ["n1", "n2", "n3"]:
        g.add_node(GraphNode(nid, np.zeros(64), "app", 5, 2,
                             {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social"))
    g.add_edge("n1", "n2", 0.02, 0.5, 0.1)  # weak
    g.add_edge("n1", "n3", 0.8, 0.5, 0.1)   # strong
    pruned = g.prune_weak_edges()
    assert pruned == 1
    assert g.edge_count() == 1


def test_graph_eviction():
    """evict_stale_nodes() must remove nodes inactive for > 45 days."""
    g = BehaviouralGraph("user_test")
    old = GraphNode("old", np.zeros(64), "app", 5, 2,
                    {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social")
    fresh = GraphNode("fresh", np.zeros(64), "app", 5, 2,
                      {"headphones": False, "calendar_near": False, "weekend": False}, 40, 1, "social")
    g.add_node(old)
    g.add_node(fresh)
    g.add_edge("old", "fresh", 0.5, 0.3, 0.1)
    evicted = g.evict_stale_nodes(50)  # old: 50-0=50 > 45, fresh: 50-40=10
    assert evicted == 1
    assert g.get_node("old") is None
    assert g.get_node("fresh") is not None
    assert g.edge_count() == 0


def test_graph_serialization():
    """save_to_disk() / load_from_disk() must preserve all data."""
    g = BehaviouralGraph("user_test")
    node = GraphNode("n1", np.ones(64), "com.test.app", 10, 3,
                     {"headphones": True, "calendar_near": False, "weekend": False}, 5, 7, "social")
    g.add_node(node)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        g.save_to_disk(path)
        g2 = BehaviouralGraph("user_test")
        g2.load_from_disk(path)
        assert g2.node_count() == 1
        assert g2.get_node("n1").app_id == "com.test.app"
        assert g2.get_node("n1").access_count == 7
    finally:
        os.unlink(path)


def test_dataset_exists():
    """All 10 user dataset files must exist."""
    for i in range(10):
        path = os.path.join(settings.USERS_DIR, f"user_{i:02d}.json")
        assert os.path.exists(path), f"Missing: {path}"


def test_dataset_schema():
    """user_00.json must have correct event schema."""
    path = os.path.join(settings.USERS_DIR, "user_00.json")
    with open(path) as f:
        events = json.load(f)
    assert isinstance(events, list)
    assert len(events) > 0
    required = {"day", "timestamp", "app_id", "battery", "time_bucket",
                "headphones", "calendar_event_in_mins", "weekend", "category"}
    sample = events[0]
    assert required.issubset(set(sample.keys()))
    assert 0.0 <= sample["battery"] <= 100.0
    assert 0 <= sample["time_bucket"] <= 47
