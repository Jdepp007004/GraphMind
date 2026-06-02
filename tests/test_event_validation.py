"""
EventBus validation and graph normalization tests.
"""

from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED
from src.core.graph_engine import BehaviouralGraph


def test_invalid_known_event_rejected():
    bus = EventBus.get_instance()
    received = []
    bus.subscribe(TOPIC_APP_LAUNCHED, lambda payload: received.append(payload))

    bus.publish(TOPIC_APP_LAUNCHED, {"user_id": "user_00"})

    stats = bus.get_validation_stats()
    assert received == []
    assert stats["rejected_event_count"] == 1
    assert "missing required fields" in stats["rejected_events"][0]["reason"]


def test_valid_known_event_dispatched():
    bus = EventBus.get_instance()
    received = []
    bus.subscribe(TOPIC_APP_LAUNCHED, lambda payload: received.append(payload))

    bus.publish(TOPIC_APP_LAUNCHED, {
        "timestamp": 1.0,
        "user_id": "user_00",
        "app_id": "com.example",
    })

    assert len(received) == 1
    assert bus.get_validation_stats()["rejected_event_count"] == 0


def test_custom_development_topic_still_allowed():
    bus = EventBus.get_instance()
    received = []
    bus.subscribe("topic_test", lambda payload: received.append(payload))

    bus.publish("topic_test", {"value": 99})

    assert received == [{"value": 99}]


def test_graph_event_edges_are_normalized():
    bus = EventBus.get_instance()
    graph = BehaviouralGraph("norm_user")
    events = [
        ("com.a", 1.0),
        ("com.b", 2.0),
        ("com.a", 3.0),
        ("com.c", 4.0),
    ]
    for app_id, ts in events:
        bus.publish(TOPIC_APP_LAUNCHED, {
            "timestamp": ts,
            "user_id": "norm_user",
            "app_id": app_id,
            "battery": 80.0,
            "time_of_day_bucket": 10,
            "day": 0,
        })

    for node_id in list(graph._graph.nodes()):
        outgoing = graph.get_edges_from(node_id)
        if outgoing:
            total = sum(edge.transition_prob for edge in outgoing)
            assert abs(total - 1.0) < 0.0001
