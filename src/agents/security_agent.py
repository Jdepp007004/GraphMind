"""
src/agents/security_agent.py

LangGraph agent: context boundary enforcement.
"""

import logging
from typing import Dict, Any

from config import settings
from src.security.context_boundary import ContextBoundaryEnforcer
from src.core.event_bus import EventBus

logger = logging.getLogger(__name__)


class SecurityAgent:
    """
    LangGraph agent that monitors security flush events and updates orchestration state.
    """

    def __init__(self, enforcer: ContextBoundaryEnforcer) -> None:
        """Store enforcer reference."""
        self.enforcer = enforcer
        self._last_flush_count = 0

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get flush_log from enforcer since last run.
        Update state['security_flush_count'].
        Append security events to state['messages'].
        Return state.
        """
        flush_log = self.enforcer.get_flush_log()
        new_flushes = flush_log[self._last_flush_count:]
        self._last_flush_count = len(flush_log)
        state["security_flush_count"] = len(flush_log)
        state["last_agent"] = "security"
        state["messages"].append({
            "agent": "security",
            "total_flushes": len(flush_log),
            "new_flushes": len(new_flushes),
            "flush_events": new_flushes
        })
        return state
