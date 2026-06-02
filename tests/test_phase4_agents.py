"""
tests/test_phase4_agents.py

Phase 4 gate tests: security, drift detector, prefetch daemon, orchestrator.
"""

import pytest
import numpy as np

from src.core.event_bus import EventBus, TOPIC_SECURITY_FLUSH
from src.core.graph_engine import BehaviouralGraph, GraphNode
from src.core.memory_manager import MemoryManager
from src.security.context_boundary import ContextBoundaryEnforcer
from src.prefetch.daemon import PrefetchDaemon
from src.agents.drift_detector_agent import DriftDetectorAgent
from src.agents.orchestrator import GraphMindOrchestrator, GraphMindState
from config import settings


def test_security_check_transition_sensitive_consumer():
    """Financial -> social must return True."""
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    mm = MemoryManager("user_test", g)
    enforcer = ContextBoundaryEnforcer("user_test", mm)
    assert enforcer.check_transition("financial", "social") is True
    assert enforcer.check_transition("health", "entertainment") is True
    assert enforcer.check_transition("enterprise", "shopping") is True
    EventBus.get_instance().clear_all()


def test_security_no_flush_on_non_sensitive():
    """Social -> financial must NOT trigger flush."""
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    mm = MemoryManager("user_test", g)
    enforcer = ContextBoundaryEnforcer("user_test", mm)
    assert enforcer.check_transition("social", "financial") is False
    assert enforcer.check_transition("entertainment", "productivity") is False
    assert enforcer.check_transition("social", "social") is False
    EventBus.get_instance().clear_all()


def test_security_enforce_boundary_flushes_hot():
    """enforce_boundary must flush financial nodes from HOT."""
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    fin_node = GraphNode("fin_1", np.zeros(64), "com.hdfcbank.new", 5, 2,
                         {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "financial")
    g.add_node(fin_node)
    mm = MemoryManager("user_test", g)
    mm.promote_to_hot("fin_1")
    enforcer = ContextBoundaryEnforcer("user_test", mm)
    flush_events = []
    EventBus.get_instance().subscribe(TOPIC_SECURITY_FLUSH, lambda p: flush_events.append(p))
    result = enforcer.enforce_boundary("financial", "social", 1000.0)
    assert result is not None
    assert not mm.is_in_hot("fin_1")
    assert len(enforcer.get_flush_log()) == 1
    assert len(flush_events) == 1
    EventBus.get_instance().clear_all()


def test_prefetch_daemon_no_auto_start():
    """PrefetchDaemon.scheduler must be None before start()."""
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    mm = MemoryManager("user_test", g)
    daemon = PrefetchDaemon("user_test", g, mm)
    assert daemon.scheduler is None
    EventBus.get_instance().clear_all()


def test_drift_detector_zero_data():
    """compute_kl_divergence() with no data must return 0.0."""
    EventBus.get_instance().clear_all()
    agent = DriftDetectorAgent("user_test")
    result = agent.compute_kl_divergence()
    assert result == 0.0
    EventBus.get_instance().clear_all()


def test_drift_detector_divergent_data():
    """KL divergence with completely different apps must exceed threshold."""
    EventBus.get_instance().clear_all()
    agent = DriftDetectorAgent("user_test")
    for i in range(150):
        apps = ["com.appA.android", "com.appB.android", "com.appC.android"]
        agent.transition_history.append(apps[i % 3])
    for i in range(100):
        apps = ["com.appD.android", "com.appE.android"]
        agent.recent_window.append(apps[i % 2])
    kl = agent.compute_kl_divergence()
    assert kl > settings.DRIFT_KL_THRESHOLD
    EventBus.get_instance().clear_all()


def test_orchestrator_instantiate():
    """GraphMindOrchestrator must instantiate without error."""
    EventBus.get_instance().clear_all()
    orch = GraphMindOrchestrator("user_00")
    assert orch is not None
    EventBus.get_instance().clear_all()


def test_orchestrator_run_day():
    """run_day(0) must return state with all required keys."""
    EventBus.get_instance().clear_all()
    orch = GraphMindOrchestrator("user_00")
    state = orch.run_day(0)
    required = {"user_id", "current_day", "kl_divergence", "cache_hit_rate",
                "security_flush_count", "last_agent", "messages"}
    assert required.issubset(set(state.keys()))
    assert state["user_id"] == "user_00"
    assert state["current_day"] == 0
    EventBus.get_instance().clear_all()


def test_orchestrator_state_schema():
    """GraphMindState must have all required TypedDict fields."""
    import typing
    hints = typing.get_type_hints(GraphMindState)
    required = {"user_id", "current_day", "current_event", "battery", "kl_divergence",
                "cache_hit_rate", "security_flush_count", "last_agent", "messages"}
    assert not (required - set(hints.keys()))


def test_langgraph_compiled_graph():
    """orchestrator.compiled_graph must not be None."""
    EventBus.get_instance().clear_all()
    orch = GraphMindOrchestrator("user_00")
    assert hasattr(orch, 'compiled_graph')
    assert orch.compiled_graph is not None
    EventBus.get_instance().clear_all()
