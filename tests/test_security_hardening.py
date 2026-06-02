"""
Security hardening tests for unknown-app isolation and retention policy.
"""

import numpy as np

from config import settings
from src.core.event_bus import EventBus
from src.core.graph_engine import BehaviouralGraph, GraphNode
from src.core.memory_manager import MemoryManager
from src.security.classification_guard import ClassificationGuard, RetentionPolicy
from src.security.context_boundary import ContextBoundaryEnforcer


def _node(node_id: str, app_id: str, category: str) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        embedding=np.zeros(settings.NODE_EMBEDDING_DIM),
        app_id=app_id,
        time_bucket=1,
        battery_bucket=4,
        context_flags={},
        last_seen_day=0,
        access_count=1,
        category=category,
    )


def test_unknown_apps_classified_as_unknown_sensitive():
    guard = ClassificationGuard({})
    category = guard.classify("com.not.in.taxonomy", payload_category="utility")
    assert category == settings.UNKNOWN_SENSITIVE_CATEGORY
    assert guard.is_sensitive(category)


def test_context_enforcer_unknown_app_isolated():
    EventBus.get_instance().clear_all()
    graph = BehaviouralGraph("security_user")
    mm = MemoryManager("security_user", graph)
    enforcer = ContextBoundaryEnforcer("security_user", mm)

    assert enforcer.get_app_category("com.missing.app") == settings.UNKNOWN_SENSITIVE_CATEGORY
    assert enforcer.check_transition(settings.UNKNOWN_SENSITIVE_CATEGORY, "social")

    EventBus.get_instance().clear_all()


def test_unknown_sensitive_nodes_flush_on_consumer_transition():
    EventBus.get_instance().clear_all()
    graph = BehaviouralGraph("security_user")
    mm = MemoryManager("security_user", graph)
    graph.add_node(_node("unknown_1", "com.missing.app", settings.UNKNOWN_SENSITIVE_CATEGORY))
    assert mm.promote_to_hot("unknown_1")

    enforcer = ContextBoundaryEnforcer("security_user", mm)
    result = enforcer.enforce_boundary(settings.UNKNOWN_SENSITIVE_CATEGORY, "social", 1.0)

    assert result is not None
    assert "unknown_1" in result["flushed_node_ids"]
    assert not mm.is_in_hot("unknown_1")
    EventBus.get_instance().clear_all()


def test_retention_policy_trims_security_logs():
    EventBus.get_instance().clear_all()
    graph = BehaviouralGraph("security_user")
    mm = MemoryManager("security_user", graph)
    enforcer = ContextBoundaryEnforcer("security_user", mm)
    enforcer.retention_policy = RetentionPolicy(trace_retention_events=2)
    enforcer.classification_guard.retention_policy = enforcer.retention_policy
    enforcer.flush_log.extend([{"id": 1}, {"id": 2}, {"id": 3}])
    enforcer.classification_guard.classification_log.extend([
        {"id": 1}, {"id": 2}, {"id": 3}
    ])

    summary = enforcer.enforce_retention_policy()

    assert summary["flush_events_removed"] == 1
    assert summary["classification_events_removed"] == 1
    assert len(enforcer.flush_log) == 2
    assert len(enforcer.classification_guard.classification_log) == 2
    EventBus.get_instance().clear_all()
