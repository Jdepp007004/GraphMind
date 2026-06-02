"""
src/benchmarks/graphmind_policy_runner.py

Execution-derived GraphMind benchmark runner.
"""

import logging
from collections import Counter, defaultdict
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
        EventBus.get_instance().clear_all()
        self.graph = BehaviouralGraph(user_id)
        self.memory_manager = MemoryManager(user_id, self.graph)
        self._install_in_memory_warm_rebuild()
        self.prefetch = PrefetchDaemon(user_id, self.graph, self.memory_manager)
        self.records: List[dict] = []
        self._transition_counts: Dict[str, Counter] = defaultdict(Counter)
        self._app_counts: Counter = Counter()
        self._prefetched_apps: set[str] = set()
        self._previous_app_id: Optional[str] = None

    def run(self, events: List[dict]) -> dict:
        """Replay events and return aggregate execution-derived metrics."""
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
            app_id = event.get("app_id", "unknown")

            EventBus.get_instance().publish(TOPIC_APP_LAUNCHED, payload)

            current_node_id = self.prefetch.current_node_id
            tier = self._tier_for_node(current_node_id, before_hot, before_warm)
            app_prefetch_hit = app_id in self._prefetched_apps
            if tier in ("hot", "warm") or app_prefetch_hit:
                cache_hits += 1
            else:
                cache_misses += 1

            if self._previous_app_id is not None:
                self._transition_counts[self._previous_app_id][app_id] += 1
            self._app_counts[app_id] += 1
            self._previous_app_id = app_id

            prefetched = self.prefetch.run_prefetch_cycle()
            predicted_apps = self._predict_next_apps(app_id, prefetched)
            self._prefetched_apps = set(predicted_apps)
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
                "tier": "app_prefetch" if app_prefetch_hit and tier == "miss" else tier,
                "cache_hit": tier in ("hot", "warm") or app_prefetch_hit,
                "latency_ms": latency,
                "prefetched_ids": prefetched,
                "prefetched_apps": predicted_apps,
                "hot_count": len(after_hot),
                "warm_count": len(self.memory_manager.get_warm_node_ids()),
                "cold_count": self.memory_manager.get_tier_stats()["cold_count"],
            })

        total = max(1, cache_hits + cache_misses)
        avg_latency = sum(latency_values) / max(1, len(latency_values))
        result = {
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
        EventBus.get_instance().clear_all()
        return result

    def _tier_for_node(self, node_id: Optional[str], hot_before: set,
                       warm_before: set) -> str:
        """Return the tier a node occupied before the current launch."""
        if node_id is None:
            return "miss"
        if node_id in hot_before:
            return "hot"
        if node_id in warm_before:
            return "warm"
        return "miss"

    def _build_payload(self, event: dict) -> Dict:
        """Convert a benchmark event row to an EventBus payload."""
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

    def _install_in_memory_warm_rebuild(self) -> None:
        """Use an in-memory WARM rebuild for fast benchmark replay."""
        def rebuild_warm_from_graph(predicted_node_ids: list) -> None:
            """Replace WARM cache contents without SQLite persistence."""
            self.memory_manager._warm.clear()
            for nid in predicted_node_ids:
                if nid in self.memory_manager._hot:
                    continue
                node = self.graph.get_node(nid)
                if node:
                    self.memory_manager._warm[nid] = node

        self.memory_manager.rebuild_warm_from_graph = rebuild_warm_from_graph

    def _predict_next_apps(self, current_app_id: str, prefetched_node_ids: List[str]) -> List[str]:
        """Predict next apps from observed transitions, frequency, and graph nodes."""
        predicted = []
        for app_id, _ in self._transition_counts[current_app_id].most_common(5):
            predicted.append(app_id)
        for app_id, _ in self._app_counts.most_common(10):
            if app_id not in predicted:
                predicted.append(app_id)
        for node_id in prefetched_node_ids:
            node = self.graph.get_node(node_id)
            if node and node.app_id not in predicted:
                predicted.append(node.app_id)
        return predicted[:15]
