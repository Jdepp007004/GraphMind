"""
src/security/context_boundary.py

Detects sensitive-to-consumer context transitions and sanitizes the HOT cache.
"""

import json
import logging
from typing import Optional, List

from config import settings
from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED, TOPIC_SECURITY_FLUSH
from src.core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class ContextBoundaryEnforcer:
    """
    Monitors app transitions and enforces context isolation.
    When user moves from a SENSITIVE context (financial, health, enterprise)
    to a CONSUMER context (social, entertainment, shopping),
    flush HOT cache of all sensitive-category nodes.
    """

    def __init__(self, user_id: str, memory_manager: MemoryManager) -> None:
        """
        Load app_taxonomy from APP_TAXONOMY_PATH.
        Store memory_manager reference.
        Subscribe to TOPIC_APP_LAUNCHED -> _on_app_launched().
        Set self.previous_category = None
        Set self.flush_log = [] (list of flush event dicts)
        """
        self.user_id = user_id
        self.memory_manager = memory_manager
        self.previous_category: Optional[str] = None
        self.flush_log: List[dict] = []
        try:
            with open(settings.APP_TAXONOMY_PATH) as f:
                self._taxonomy: dict = json.load(f)
        except Exception:
            self._taxonomy = {}
        bus = EventBus.get_instance()
        bus.subscribe(TOPIC_APP_LAUNCHED, self._on_app_launched)

    def check_transition(self, from_category: str, to_category: str) -> bool:
        """
        Determine if this transition requires a cache flush.
        Returns True if from_category in SENSITIVE_CATEGORIES AND to_category in CONSUMER_CATEGORIES.
        Returns False otherwise.
        """
        if from_category is None:
            return False
        return (from_category in settings.SENSITIVE_CATEGORIES and
                to_category in settings.CONSUMER_CATEGORIES)

    def enforce_boundary(self, from_category: str, to_category: str,
                         timestamp: float) -> Optional[dict]:
        """
        If check_transition() returns True:
            1. Flush all HOT nodes from SENSITIVE_CATEGORIES via memory_manager.flush_hot_by_category().
            2. Build a flush_event dict:
               {'timestamp': timestamp, 'from_category': from_category,
                'to_category': to_category, 'flushed_node_ids': list, 'user_id': user_id}
            3. Append to self.flush_log.
            4. Publish TOPIC_SECURITY_FLUSH with the flush_event dict.
            5. Return the flush_event dict.
        Else: return None.
        """
        if not self.check_transition(from_category, to_category):
            return None
        # Flush all sensitive categories
        flushed_ids = []
        for cat in settings.SENSITIVE_CATEGORIES:
            flushed = self.memory_manager.flush_hot_by_category(cat)
            flushed_ids.extend(flushed)
        flush_event = {
            "timestamp": timestamp,
            "from_category": from_category,
            "to_category": to_category,
            "flushed_node_ids": flushed_ids,
            "user_id": self.user_id
        }
        self.flush_log.append(flush_event)
        bus = EventBus.get_instance()
        bus.publish(TOPIC_SECURITY_FLUSH, flush_event)
        logger.info(
            f"Security flush: {from_category} -> {to_category}, "
            f"flushed {len(flushed_ids)} nodes for {self.user_id}"
        )
        return flush_event

    def get_flush_log(self) -> List[dict]:
        """Return all recorded flush events."""
        return self.flush_log

    def get_app_category(self, app_id: str) -> str:
        """
        Look up category from app_taxonomy.
        Returns category string or 'utility' if app_id not found.
        """
        return self._taxonomy.get(app_id, {}).get("category", "utility")

    def _on_app_launched(self, payload: dict) -> None:
        """
        PRIVATE. EventBus callback.
        Get category for payload['app_id'].
        Call enforce_boundary(self.previous_category, current_category, payload['timestamp']).
        Update self.previous_category = current_category.
        """
        if payload.get("user_id") != self.user_id:
            return
        app_id = payload.get("app_id", "unknown")
        current_category = payload.get("category") or self.get_app_category(app_id)
        if self.previous_category is not None:
            self.enforce_boundary(self.previous_category, current_category,
                                  float(payload.get("timestamp", 0.0)))
        self.previous_category = current_category
