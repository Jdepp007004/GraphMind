"""
src/agents/prefetch_agent.py

LangGraph agent: pre-fetch scheduling.
"""

import logging
from typing import Dict, Any

from config import settings
from src.prefetch.daemon import PrefetchDaemon
from src.core.event_bus import EventBus

logger = logging.getLogger(__name__)


class PrefetchAgent:
    """
    LangGraph agent that triggers the prefetch daemon on each orchestration cycle.
    """

    def __init__(self, daemon: PrefetchDaemon) -> None:
        """Store daemon reference."""
        self.daemon = daemon

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call daemon.run_prefetch_cycle().
        Update state['cache_hit_rate'] from current memory_manager stats.
        Append prefetch log to state['messages'].
        Return state.
        """
        prefetched = self.daemon.run_prefetch_cycle()
        mm = self.daemon.memory_manager
        stats = mm.get_tier_stats()
        hot_count = stats["hot_count"]
        warm_count = stats["warm_count"]
        total_accessible = hot_count + warm_count
        # Estimate cache hit rate
        cache_hit_rate = total_accessible / max(1, total_accessible + stats["cold_count"])
        state["cache_hit_rate"] = cache_hit_rate
        state["messages"].append({
            "agent": "prefetch",
            "prefetched_count": len(prefetched),
            "prefetched_ids": prefetched,
            "hot_count": hot_count,
            "warm_count": warm_count
        })
        return state
