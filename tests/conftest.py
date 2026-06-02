"""
tests/conftest.py

Shared fixtures for GraphMind tests.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.core.event_bus import EventBus
from src.core.graph_engine import BehaviouralGraph, GraphNode
from src.core.memory_manager import MemoryManager
from src.data.event_simulator import EventSimulator


@pytest.fixture(autouse=True)
def reset_event_bus():
    """Reset EventBus state before each test."""
    bus = EventBus.get_instance()
    bus.clear_all()
    yield
    bus.clear_all()


@pytest.fixture
def sample_graph():
    """Return a BehaviouralGraph with a few test nodes."""
    g = BehaviouralGraph("user_test")
    for i in range(5):
        node = GraphNode(
            node_id=f"node_{i}",
            embedding=np.zeros(64),
            app_id=f"com.test.app{i}",
            time_bucket=i * 5,
            battery_bucket=2,
            context_flags={"headphones": False, "calendar_near": False, "weekend": False},
            last_seen_day=i,
            access_count=i + 1,
            category="social"
        )
        g.add_node(node)
    return g


@pytest.fixture
def sample_memory(sample_graph):
    """Return a MemoryManager with the sample graph."""
    return MemoryManager("user_test", sample_graph)


@pytest.fixture
def user00_simulator():
    """Return an EventSimulator for user_00 (requires dataset to exist)."""
    return EventSimulator("user_00")
