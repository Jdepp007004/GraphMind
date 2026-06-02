"""
src/benchmarks/graphmind_policy_runner.py

Execution-derived GraphMind benchmark runner.
"""

import logging
from typing import Dict, List, Optional

from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED
from src.core.graph_engine import BehaviouralGraph
from src.core.memory_manager import MemoryManager
from src.prefetch.daemon import PrefetchDaemon

logger = logging.getLogger(__name__)


LATENCY_BY_TIER_MS = {
    "hot": 45.0,
    "warm": 210.0,
    "cold": 850.0,
    "miss": 850.0,
}


class GraphMindPolicyRunner:
    """
    Replays events through GraphMind's graph, memory manager, and prefetch path.

    This runner intentionally measures GraphMind from execution state. It does
    not receive benchmark boosts, post-processing wins, or fixed policy metrics.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.graph = BehaviouralGraph(user_id)
        self.memory_manager = MemoryManager(user_id, self.graph)
        self.prefetch = PrefetchDaemon(user_id, self.graph, self.memory_manager)
        self.records: List[dict] = []

    def run(self, events: List[dict]) -> dict:
        cache_hits = 0
        cache_misses = 0
        evictions = 0
        prefetched_total = 0
        latency_values: List[float] = []
        previous_hot: set = set()

        for event in events:
            payload = self._build_payload(event)

            before_hot = set(self.memory_manager.get_hot_node_ids())
            before_warm = set(self.memory_manager.get_warm_node_ids())

            EventBus.get_instance().publish(TOPIC_APP_LAUNCHED, payload)

            current_node_id = self.prefetch.current_node_id
            tier = self._tier_for_node(current_node_id, before_hot, before_warm)
            if tier in ("hot", "warm"):
                cache_hits += 1
            else:
                cache_misses += 1

            prefetched = self.prefetch.run_prefetch_cycle()
            prefetched_total += len(prefetched)

            after_hot = set(self.memory_manager.get_hot_node_ids())
            evictions += len(previous_hot - after_hot)
            previous_hot = after_hot

            latency = LATENCY_BY_TIER_MS[tier]
            latency_values.append(latency)
            self.records.append({
                "user_id": self.user_id,
                "day": int(event.get("day", 0)),
                "app_id": event.get("app_id", "unknown"),
                "node_id": current_node_id,
                "tier": tier,
                "cache_hit": tier in ("hot", "warm"),
                "latency_ms": latency,
                "prefetched_ids": prefetched,
                "hot_count": len(after_hot),
                "warm_count": len(self.memory_manager.get_warm_node_ids()),
                "cold_count": self.memory_manager.get_tier_stats()["cold_count"],
            })

        total = max(1, cache_hits + cache_misses)
        avg_latency = sum(latency_values) / max(1, len(latency_values))
        return {
            "cache_hit_rate": cache_hits / total,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "thrash_rate": evictions / total,
            "evictions": evictions,
            "battery_overhead_pct": min(5.0, prefetched_total * 0.001),
            "avg_latency_ms": avg_latency,
            "graph_node_count": self.graph.node_count(),
            "graph_edge_count": self.graph.edge_count(),
            "records": self.records,
        }

    def _tier_for_node(self, node_id: Optional[str], hot_before: set,
                       warm_before: set) -> str:
        if node_id is None:
            return "miss"
        if node_id in hot_before:
            return "hot"
        if node_id in warm_before:
            return "warm"
        return "miss"

    def _build_payload(self, event: dict) -> Dict:
        time_bucket = int(event.get("time_of_day_bucket", event.get("time_bucket", 0)))
        return {
            "timestamp": float(event.get("timestamp", 0.0)),
            "user_id": self.user_id,
            "app_id": event.get("app_id", "unknown"),
            "category": event.get("category", "utility"),
            "battery": float(event.get("battery", 100.0)),
            "time_of_day_bucket": time_bucket,
            "time_bucket": time_bucket,
            "day": int(event.get("day", 0)),
            "weekend": bool(event.get("weekend", False)),
            "headphones": bool(event.get("headphones", False)),
            "calendar_event_in_mins": event.get("calendar_event_in_mins"),
        }
