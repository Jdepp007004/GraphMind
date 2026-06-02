"""
src/core/event_bus.py

Singleton publish-subscribe bus. All inter-module communication goes through this.
Prevents direct cross-module coupling.
"""

import threading
import queue
import logging
import sys
from config import settings
from src.core.event_schema import build_default_registry

logger = logging.getLogger(__name__)

# ── Topic Constants ────────────────────────────────────────────────────────
TOPIC_APP_LAUNCHED = "app_launched"
TOPIC_APP_CLOSED = "app_closed"
TOPIC_BATTERY_UPDATED = "battery_updated"
TOPIC_HEADPHONES_CONNECTED = "headphones_connected"
TOPIC_CALENDAR_EVENT = "calendar_event_approaching"
TOPIC_NODE_PROMOTED = "node_promoted_to_hot"
TOPIC_NODE_DEMOTED = "node_demoted_from_hot"
TOPIC_CACHE_HIT = "cache_hit"
TOPIC_CACHE_MISS = "cache_miss"
TOPIC_DRIFT_DETECTED = "drift_detected"
TOPIC_SECURITY_FLUSH = "security_cache_flush"
TOPIC_PREFETCH_TRIGGERED = "prefetch_triggered"
TOPIC_RL_WEIGHT_UPDATED = "rl_weight_updated"


class EventBus:
    """
    Thread-safe singleton event bus. All modules publish and subscribe here.
    Use EventBus.get_instance() to get the single instance.
    NEVER instantiate EventBus() directly after the first call.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the internal subscription registry."""
        self._subscribers: dict[str, list] = {}
        self._sub_lock = threading.Lock()
        self._schema_registry = build_default_registry(sys.modules[__name__])
        self._rejected_event_count = 0
        self._rejected_events: list[dict] = []

    @classmethod
    def get_instance(cls) -> "EventBus":
        """
        Class method. Returns the single EventBus instance.
        Creates it on first call, returns existing on subsequent calls.
        Thread-safe using a lock.
        Returns: EventBus singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def subscribe(self, topic: str, callback: callable) -> None:
        """
        Register a callback to be called when topic is published.
        topic: string event name, e.g. 'app_launched', 'battery_updated', 'drift_detected'
        callback: function(payload: dict) -> None
        Multiple callbacks can be registered for the same topic.
        """
        with self._sub_lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            if callback not in self._subscribers[topic]:
                self._subscribers[topic].append(callback)

    def publish(self, topic: str, payload: dict) -> None:
        """
        Publish an event to all subscribers of topic.
        payload: dictionary of event data. Always include {'timestamp': float} key.
        Calls all registered callbacks synchronously in subscription order.
        Logs the publish at DEBUG level: f'EventBus: {topic} -> {list(payload.keys())}'
        """
        result = self._schema_registry.validate(topic, payload)
        if not result.valid:
            self._rejected_event_count += 1
            self._rejected_events.append({
                "topic": topic,
                "reason": result.reason,
                "payload": payload,
            })
            logger.warning("EventBus rejected event %s: %s", topic, result.reason)
            return
        logger.debug(f"EventBus: {topic} -> {list(payload.keys())}")
        with self._sub_lock:
            callbacks = list(self._subscribers.get(topic, []))
        for cb in callbacks:
            try:
                cb(payload)
            except Exception as e:
                logger.error(f"EventBus callback error for topic '{topic}': {e}")

    def unsubscribe(self, topic: str, callback: callable) -> None:
        """
        Remove a specific callback from a topic.
        No-op if callback not registered for topic.
        """
        with self._sub_lock:
            if topic in self._subscribers and callback in self._subscribers[topic]:
                self._subscribers[topic].remove(callback)

    def clear_all(self) -> None:
        """
        Remove all subscriptions. Used in tests only to reset state between tests.
        """
        with self._sub_lock:
            self._subscribers.clear()

    def get_validation_stats(self) -> dict:
        """Return invalid event counts and recent rejection records."""
        return {
            "rejected_event_count": self._rejected_event_count,
            "rejected_events": list(self._rejected_events),
        }

    def clear_validation_stats(self) -> None:
        """Reset validation counters without affecting subscriptions."""
        self._rejected_event_count = 0
        self._rejected_events.clear()
