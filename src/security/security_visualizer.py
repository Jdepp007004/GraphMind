"""
src/security/security_visualizer.py

Visualization layer ONLY for the existing security system.
Reads from ContextBoundaryEnforcer.flush_log — does NOT reimplement any logic.
Provides data transformation methods for dashboard rendering.
"""

import logging
from typing import List, Dict, Any, Optional
import time

from src.core.event_bus import EventBus, TOPIC_SECURITY_FLUSH

logger = logging.getLogger(__name__)


class SecurityVisualizer:
    """
    Reads from the existing ContextBoundaryEnforcer flush_log and transforms
    raw flush events into dashboard-ready visualization data.

    Usage:
        enforcer = ContextBoundaryEnforcer(user_id, memory_manager)
        viz = SecurityVisualizer(user_id, enforcer)
        viz.get_timeline_data()  # for the Security Timeline tab
    """

    def __init__(self, user_id: str, enforcer) -> None:
        """
        enforcer: existing ContextBoundaryEnforcer instance.
        Does NOT subscribe to EventBus (enforcer already does that).
        """
        self.user_id = user_id
        self.enforcer = enforcer
        # Also listen to real-time security flush events for live updates
        self._live_events: List[dict] = []
        bus = EventBus.get_instance()
        bus.subscribe(TOPIC_SECURITY_FLUSH, self._on_security_flush)

    def _on_security_flush(self, payload: dict) -> None:
        """Accumulate real-time flush events for live display."""
        if payload.get("user_id") != self.user_id:
            return
        self._live_events.append(payload)

    def get_flush_log(self) -> List[dict]:
        """
        Return all flush events from the enforcer's log.
        Falls back to live-accumulated events if enforcer has none.
        """
        log = self.enforcer.get_flush_log()
        if log:
            return log
        return list(self._live_events)

    def get_timeline_data(self) -> List[dict]:
        """
        Transform flush events into Security Timeline display records.
        Each record describes one boundary crossing:
        {
          'event_number': int,
          'timestamp': float,
          'from_category': str,
          'to_category': str,
          'flushed_count': int,
          'flushed_ids': list,
          'severity': str,   # 'HIGH' | 'MEDIUM' | 'LOW'
          'flow': str,        # e.g. "financial -> social -> Flush Triggered"
        }
        """
        flush_log = self.get_flush_log()
        result = []
        for i, event in enumerate(flush_log, 1):
            from_cat = event.get("from_category", "unknown")
            to_cat = event.get("to_category", "unknown")
            flushed = event.get("flushed_node_ids", [])
            severity = self._compute_severity(from_cat, len(flushed))
            flow = self._build_flow_string(from_cat, to_cat, len(flushed))
            result.append({
                "event_number": i,
                "timestamp": event.get("timestamp", 0.0),
                "from_category": from_cat,
                "to_category": to_cat,
                "flushed_count": len(flushed),
                "flushed_ids": flushed,
                "severity": severity,
                "flow": flow,
                "user_id": event.get("user_id", self.user_id),
            })
        return result

    def get_summary_metrics(self) -> dict:
        """
        Return aggregate security metrics for the dashboard header.
        {
          'total_flush_events': int,
          'total_nodes_removed': int,
          'avg_flush_size': float,
          'most_common_from_category': str,
          'category_transition_counts': dict,
        }
        """
        flush_log = self.get_flush_log()
        if not flush_log:
            return {
                "total_flush_events": 0,
                "total_nodes_removed": 0,
                "avg_flush_size": 0.0,
                "most_common_from_category": "none",
                "category_transition_counts": {},
            }

        total_nodes = sum(len(e.get("flushed_node_ids", [])) for e in flush_log)
        avg_size = total_nodes / len(flush_log) if flush_log else 0.0

        # Count transitions
        transition_counts: Dict[str, int] = {}
        from_counts: Dict[str, int] = {}
        for event in flush_log:
            key = f"{event.get('from_category','?')} to {event.get('to_category','?')}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
            fc = event.get("from_category", "?")
            from_counts[fc] = from_counts.get(fc, 0) + 1

        most_common_from = max(from_counts, key=from_counts.get) if from_counts else "none"

        return {
            "total_flush_events": len(flush_log),
            "total_nodes_removed": total_nodes,
            "avg_flush_size": round(avg_size, 2),
            "most_common_from_category": most_common_from,
            "category_transition_counts": transition_counts,
        }

    def get_category_flow_data(self) -> List[dict]:
        """
        Return data for a Sankey/flow diagram showing category transitions.
        Each element: {'source': str, 'target': str, 'count': int}
        """
        flush_log = self.get_flush_log()
        flow_counts: Dict[str, int] = {}
        for event in flush_log:
            key = (event.get("from_category", "?"), event.get("to_category", "?"))
            flow_counts[key] = flow_counts.get(key, 0) + 1
        return [
            {"source": k[0], "target": k[1], "count": v}
            for k, v in flow_counts.items()
        ]

    def _compute_severity(self, from_category: str, flushed_count: int) -> str:
        """HIGH for financial/health, MEDIUM for enterprise, LOW for others."""
        if from_category in ("financial", "health", "government"):
            return "HIGH"
        if from_category == "enterprise":
            return "MEDIUM"
        return "LOW"

    def _build_flow_string(self, from_cat: str, to_cat: str, flushed_count: int) -> str:
        """
        Build the canonical flow string shown in the Security Timeline:
          financial -> social -> Flush Triggered -> N Nodes Removed
        """
        parts = [
            from_cat.capitalize(),
            to_cat.capitalize(),
            "Flush Triggered",
            f"{flushed_count} Node{'s' if flushed_count != 1 else ''} Removed"
        ]
        return " -> ".join(parts)
