"""
src/core/event_schema.py

Lightweight EventBus schema validation.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set


@dataclass
class EventValidationResult:
    valid: bool
    reason: str = ""


class EventSchemaRegistry:
    """Validates known GraphMind event topics before dispatch."""

    def __init__(self) -> None:
        self._required: Dict[str, Set[str]] = {}

    def register(self, topic: str, required_fields: Iterable[str]) -> None:
        """Register required payload fields for a known topic."""
        self._required[topic] = set(required_fields)

    def is_known(self, topic: str) -> bool:
        """Return True when a topic has a registered schema."""
        return topic in self._required

    def validate(self, topic: str, payload: dict) -> EventValidationResult:
        """Validate a topic and payload against the registered schema."""
        if not isinstance(topic, str) or not topic:
            return EventValidationResult(False, "topic must be a non-empty string")
        if not isinstance(payload, dict):
            return EventValidationResult(False, "payload must be a dict")
        if topic not in self._required:
            return EventValidationResult(True)
        missing = sorted(field for field in self._required[topic] if field not in payload)
        if missing:
            return EventValidationResult(False, f"missing required fields: {missing}")
        if "timestamp" in payload:
            try:
                float(payload["timestamp"])
            except Exception:
                return EventValidationResult(False, "timestamp must be numeric")
        return EventValidationResult(True)


def build_default_registry(topics) -> EventSchemaRegistry:
    """Build the default schema registry from EventBus topic constants."""
    registry = EventSchemaRegistry()
    registry.register(topics.TOPIC_APP_LAUNCHED, ["timestamp", "user_id", "app_id"])
    registry.register(topics.TOPIC_BATTERY_UPDATED, ["timestamp", "user_id", "battery"])
    registry.register(topics.TOPIC_HEADPHONES_CONNECTED, ["timestamp", "user_id"])
    registry.register(topics.TOPIC_CALENDAR_EVENT, ["timestamp", "user_id"])
    registry.register(topics.TOPIC_NODE_PROMOTED, ["timestamp", "user_id", "node_id"])
    registry.register(topics.TOPIC_NODE_DEMOTED, ["timestamp", "user_id", "node_id"])
    registry.register(topics.TOPIC_CACHE_HIT, ["timestamp", "user_id", "node_id"])
    registry.register(topics.TOPIC_CACHE_MISS, ["timestamp", "user_id", "node_id"])
    registry.register(topics.TOPIC_DRIFT_DETECTED, ["timestamp", "user_id"])
    registry.register(topics.TOPIC_SECURITY_FLUSH, ["timestamp", "user_id"])
    registry.register(topics.TOPIC_PREFETCH_TRIGGERED, ["timestamp", "user_id", "prefetched_ids"])
    registry.register(topics.TOPIC_RL_WEIGHT_UPDATED, ["timestamp", "user_id"])
    return registry
