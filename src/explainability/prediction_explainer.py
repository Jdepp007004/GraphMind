"""
src/explainability/prediction_explainer.py

Subscribes to EventBus events and generates DecisionTraces in real time.
This is the live integration layer: it listens to existing events and
produces explanations without modifying any core modules.

Reuses:
  - EventBus.get_instance() (existing)
  - TOPIC_* constants (existing)
  - ReasoningEngine (new, pure)
  - DecisionTraceStore (new)
"""

import logging
import time
from typing import Optional

from src.core.event_bus import (
    EventBus,
    TOPIC_APP_LAUNCHED,
    TOPIC_NODE_PROMOTED,
    TOPIC_NODE_DEMOTED,
    TOPIC_PREFETCH_TRIGGERED,
    TOPIC_SECURITY_FLUSH,
)
from src.explainability.reasoning_engine import ReasoningEngine
from src.explainability.decision_trace import (
    DecisionTraceStore, DecisionTrace,
    ACTION_PRELOADED, ACTION_PROMOTED, ACTION_DEMOTED,
    ACTION_FLUSHED, ACTION_PREDICTED
)

logger = logging.getLogger(__name__)

# Singleton store shared across all PredictionExplainer instances
_global_trace_store = DecisionTraceStore(max_per_user=1000)


def get_trace_store() -> DecisionTraceStore:
    """Return the shared global trace store."""
    return _global_trace_store


class PredictionExplainer:
    """
    Subscribes to all relevant EventBus topics and generates explanation traces.
    One instance per user. Lightweight — only stores data in memory.
    """

    def __init__(self, user_id: str,
                 store: Optional[DecisionTraceStore] = None) -> None:
        self.user_id = user_id
        self.store = store or _global_trace_store
        self.engine = ReasoningEngine()
        self._previous_app: Optional[str] = None
        self._previous_category: Optional[str] = None
        self._last_battery: float = 100.0
        self._last_time_bucket: int = 0

        bus = EventBus.get_instance()
        bus.subscribe(TOPIC_APP_LAUNCHED, self._on_app_launched)
        bus.subscribe(TOPIC_NODE_PROMOTED, self._on_node_promoted)
        bus.subscribe(TOPIC_NODE_DEMOTED, self._on_node_demoted)
        bus.subscribe(TOPIC_PREFETCH_TRIGGERED, self._on_prefetch_triggered)
        bus.subscribe(TOPIC_SECURITY_FLUSH, self._on_security_flush)
        logger.debug(f"PredictionExplainer initialized for {user_id}")

    def _on_app_launched(self, payload: dict) -> None:
        """Record context for subsequent explanation generation."""
        if payload.get("user_id") != self.user_id:
            return
        self._previous_app = payload.get("app_id", "unknown")
        self._previous_category = payload.get("category", "utility")
        self._last_battery = float(payload.get("battery", 100.0))
        self._last_time_bucket = int(payload.get("time_of_day_bucket", 0))

    def _on_node_promoted(self, payload: dict) -> None:
        """Generate a PROMOTED trace when a node enters HOT."""
        if payload.get("user_id") != self.user_id:
            return
        app_id = payload.get("node_id", "unknown")
        reasons = self.engine.reasons_for_promotion(
            app_id=app_id,
            from_tier=payload.get("from_tier", "warm"),
            access_count=payload.get("access_count", 1),
            time_bucket=self._last_time_bucket,
        )
        confidence = min(1.0, 0.6 + 0.1 * payload.get("access_count", 1))
        trace = DecisionTrace(
            action=ACTION_PROMOTED,
            app_id=app_id,
            user_id=self.user_id,
            reasons=reasons,
            confidence=round(confidence, 2),
            metadata={"tier": "hot"}
        )
        self.store.add(trace)
        logger.debug(f"Explainer: PROMOTED trace for {app_id}")

    def _on_node_demoted(self, payload: dict) -> None:
        """Generate a DEMOTED trace when a node leaves HOT."""
        if payload.get("user_id") != self.user_id:
            return
        app_id = payload.get("node_id", "unknown")
        hot_count = payload.get("hot_count", 30)
        hot_capacity = 30
        hot_pressure = hot_count / hot_capacity
        reasons = self.engine.reasons_for_demotion(
            app_id=app_id,
            hot_pressure=hot_pressure,
        )
        trace = DecisionTrace(
            action=ACTION_DEMOTED,
            app_id=app_id,
            user_id=self.user_id,
            reasons=reasons,
            confidence=0.95,
            metadata={"from_tier": "hot", "to_tier": "warm"}
        )
        self.store.add(trace)

    def _on_prefetch_triggered(self, payload: dict) -> None:
        """Generate PRELOADED traces for each prefetched node."""
        if payload.get("user_id") != self.user_id:
            return
        prefetched_ids = payload.get("prefetched_ids", [])
        battery = float(payload.get("battery", self._last_battery))

        for rank, node_id in enumerate(prefetched_ids[:5], 1):
            reasons = self.engine.reasons_for_prediction(
                app_id=node_id,
                source_app=self._previous_app or "unknown",
                transition_prob=max(0.1, 0.9 - rank * 0.1),
                rank=rank,
                battery=battery,
                time_bucket=self._last_time_bucket,
            )
            confidence = round(max(0.1, 0.9 - (rank - 1) * 0.12), 2)
            trace = DecisionTrace(
                action=ACTION_PRELOADED,
                app_id=node_id,
                user_id=self.user_id,
                reasons=reasons,
                confidence=confidence,
                metadata={"rank": rank, "source_app": self._previous_app or ""}
            )
            self.store.add(trace)
        logger.debug(f"Explainer: {len(prefetched_ids)} PRELOADED traces for {self.user_id}")

    def _on_security_flush(self, payload: dict) -> None:
        """Generate FLUSHED trace on security context boundary crossing."""
        if payload.get("user_id") != self.user_id:
            return
        from_cat = payload.get("from_category", "sensitive")
        to_cat = payload.get("to_category", "consumer")
        flushed_ids = payload.get("flushed_node_ids", [])

        reasons = self.engine.reasons_for_flush(
            from_category=from_cat,
            to_category=to_cat,
            flushed_count=len(flushed_ids)
        )
        trace = DecisionTrace(
            action=ACTION_FLUSHED,
            app_id=f"[{from_cat} context]",
            user_id=self.user_id,
            reasons=reasons,
            confidence=1.0,
            metadata={
                "from_category": from_cat,
                "to_category": to_cat,
                "flushed_ids": flushed_ids
            }
        )
        self.store.add(trace)
        logger.info(f"Explainer: FLUSHED trace — {from_cat} to {to_cat}, {len(flushed_ids)} nodes")

    def get_latest_explanations(self, limit: int = 10) -> list:
        """
        Return the latest decision traces for this user as formatted strings.
        Used by the dashboard Explainability tab.
        """
        traces = self.store.get_recent(self.user_id, limit=limit)
        return [t.format_explanation() for t in traces]

    def get_traces_dict(self, limit: int = 50) -> list:
        """Return traces as list of dicts for Streamlit dataframe display."""
        return self.store.to_json_list(self.user_id, limit=limit)
