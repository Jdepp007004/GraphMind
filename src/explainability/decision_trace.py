"""
src/explainability/decision_trace.py

Stores and retrieves prediction decision traces.
Each trace records WHY a prediction was made (pre-fetch, promotion, demotion, flush).
Thread-safe in-memory store with optional JSON persistence.
"""

import json
import logging
import os
import threading
import time
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Decision action types
ACTION_PRELOADED = "preloaded"
ACTION_PROMOTED = "promoted"
ACTION_DEMOTED = "demoted"
ACTION_FLUSHED = "flushed"
ACTION_PREDICTED = "predicted"


class DecisionTrace:
    """
    Immutable record of a single GraphMind decision with its reasons.
    """

    def __init__(self, action: str, app_id: str, user_id: str,
                 reasons: List[str], confidence: float = 1.0,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        self.trace_id = f"{user_id}_{action}_{int(time.time() * 1000)}"
        self.timestamp = time.time()
        self.action = action           # ACTION_* constant
        self.app_id = app_id           # package name or node_id
        self.user_id = user_id
        self.reasons = reasons         # list of human-readable reason strings
        self.confidence = confidence   # 0.0-1.0 overall confidence
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "app_id": self.app_id,
            "user_id": self.user_id,
            "reasons": self.reasons,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def format_explanation(self) -> str:
        """
        Format as a human-readable explanation string.
        Example:
          Spotify preloaded because:
          - weekday morning pattern
          - headphones connected
          - transition probability 0.82
        """
        action_label = {
            ACTION_PRELOADED: "preloaded",
            ACTION_PROMOTED: "promoted to HOT",
            ACTION_DEMOTED: "demoted to WARM",
            ACTION_FLUSHED: "flushed from cache",
            ACTION_PREDICTED: "predicted as next app",
        }.get(self.action, self.action)

        app_label = self.app_id.split(".")[-1] if "." in self.app_id else self.app_id
        header = f"{app_label} {action_label} because:"
        lines = [header] + [f"  - {r}" for r in self.reasons]
        if self.confidence < 1.0:
            lines.append(f"  - confidence: {self.confidence:.0%}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DecisionTrace(action={self.action!r}, app={self.app_id!r}, reasons={len(self.reasons)})"


class DecisionTraceStore:
    """
    Thread-safe in-memory store for DecisionTrace records.
    Keyed by user_id. Optional disk persistence to JSON.
    """

    def __init__(self, persist_path: Optional[str] = None,
                 max_per_user: int = 500) -> None:
        self._traces: Dict[str, List[DecisionTrace]] = {}
        self._lock = threading.Lock()
        self._persist_path = persist_path
        self._max_per_user = max_per_user

    def add(self, trace: DecisionTrace) -> None:
        """Add a trace to the store. Evicts oldest if over max_per_user."""
        with self._lock:
            uid = trace.user_id
            if uid not in self._traces:
                self._traces[uid] = []
            self._traces[uid].append(trace)
            # Evict oldest if over limit
            if len(self._traces[uid]) > self._max_per_user:
                self._traces[uid] = self._traces[uid][-self._max_per_user:]

    def get_recent(self, user_id: str, limit: int = 20,
                   action_filter: Optional[str] = None) -> List[DecisionTrace]:
        """Return most recent traces for user_id, newest first."""
        with self._lock:
            traces = list(self._traces.get(user_id, []))
        if action_filter:
            traces = [t for t in traces if t.action == action_filter]
        return list(reversed(traces))[:limit]

    def get_for_app(self, user_id: str, app_id: str,
                    limit: int = 10) -> List[DecisionTrace]:
        """Return recent traces for a specific app."""
        with self._lock:
            traces = list(self._traces.get(user_id, []))
        filtered = [t for t in traces if t.app_id == app_id]
        return list(reversed(filtered))[:limit]

    def get_all_users(self) -> List[str]:
        """Return all user_ids that have traces."""
        with self._lock:
            return list(self._traces.keys())

    def count(self, user_id: str) -> int:
        """Return total number of traces for user_id."""
        with self._lock:
            return len(self._traces.get(user_id, []))

    def to_json_list(self, user_id: str, limit: int = 100) -> List[dict]:
        """Export recent traces as list of dicts for dashboard/API."""
        return [t.to_dict() for t in self.get_recent(user_id, limit)]

    def save_to_disk(self) -> None:
        """Persist all traces to JSON file."""
        if not self._persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            with self._lock:
                data = {
                    uid: [t.to_dict() for t in traces]
                    for uid, traces in self._traces.items()
                }
            with open(self._persist_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"DecisionTraceStore: failed to save: {e}")

    def clear(self, user_id: Optional[str] = None) -> None:
        """Clear traces for a user, or all traces if user_id is None."""
        with self._lock:
            if user_id:
                self._traces.pop(user_id, None)
            else:
                self._traces.clear()
