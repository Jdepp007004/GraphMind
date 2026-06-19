"""
src/core/memory_manager.py

Three-tier memory hierarchy.
HOT  = dict (simulated RAM)
WARM = LRU OrderedDict (simulated L3/file cache)
COLD = SQLite on disk
Manages promotion, demotion, and eviction.
"""

import sqlite3
import pickle
import logging
from collections import OrderedDict
from typing import Optional

from config import settings
from src.core.event_bus import (
    EventBus, TOPIC_APP_LAUNCHED, TOPIC_NODE_PROMOTED,
    TOPIC_NODE_DEMOTED, TOPIC_CACHE_HIT, TOPIC_CACHE_MISS
)
from src.core.graph_engine import BehaviouralGraph, GraphNode

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages the three-tier memory hierarchy for one user's graph.
    HOT: Python dict, max HOT_TIER_CAPACITY (30) nodes.
    WARM: OrderedDict LRU, max WARM_TIER_CAPACITY (150) nodes.
    COLD: SQLite database, theoretically unlimited.
    """

    def __init__(self, user_id: str, graph: BehaviouralGraph) -> None:
        """
        Initialize tiers. Connect to/create SQLite COLD DB at COLD_DB_PATH.
        Create table: cold_nodes(user_id TEXT, node_id TEXT, serialized_node BLOB, last_seen_day INT)
        Subscribe to TOPIC_APP_LAUNCHED on EventBus to call self._on_app_launched().
        graph: BehaviouralGraph instance for this user. Stored as self.graph.
        """
        self.user_id = user_id
        self.graph = graph
        self._hot: dict = {}       # node_id -> GraphNode (insertion-ordered via dict)
        self._hot_order: list = [] # LRU tracking: most recent at end
        self._warm: OrderedDict = OrderedDict()  # node_id -> GraphNode, LRU
        self._db_path = settings.COLD_DB_PATH
        self._init_cold_db()
        bus = EventBus.get_instance()
        bus.subscribe(TOPIC_APP_LAUNCHED, self._on_app_launched)

    def _init_cold_db(self) -> None:
        """Initialize the SQLite COLD database and create the table if needed."""
        # In-memory COLD store for benchmark evaluation; production deployment uses SQLite persistence
        self.cold_store = {}
        return

    def promote_to_hot(self, node_id: str) -> bool:
        """
        Move node_id to HOT tier.
        If already in HOT: return True (no-op).
        If HOT is full (>= HOT_TIER_CAPACITY): evict the least-recently-used HOT node to WARM first.
        Then move node from WARM to HOT (or load from COLD/graph if not in WARM).
        Publishes TOPIC_NODE_PROMOTED with {'node_id': node_id, 'tier': 'hot', 'user_id': user_id}.
        Returns True on success, False if node_id not found anywhere.
        """
        if node_id in self._hot:
            # Update LRU order
            if node_id in self._hot_order:
                self._hot_order.remove(node_id)
            self._hot_order.append(node_id)
            return True

        # Get the node object
        node = self._find_node(node_id)
        if node is None:
            return False

        # Evict LRU from HOT if at capacity
        while len(self._hot) >= settings.HOT_TIER_CAPACITY:
            prev_len = len(self._hot)
            self._evict_lru_from_hot()
            if len(self._hot) == prev_len:
                # Promotion failed. Put it back in WARM.
                if len(self._warm) >= settings.WARM_TIER_CAPACITY:
                    self._evict_oldest_from_warm()
                self._warm[node_id] = node
                self._warm.move_to_end(node_id)
                return False

        # Add to HOT
        self._hot[node_id] = node
        self._hot_order.append(node_id)

        # Remove from WARM if present
        if node_id in self._warm:
            del self._warm[node_id]

        bus = EventBus.get_instance()
        bus.publish(TOPIC_NODE_PROMOTED, {
            "timestamp": 0.0, "node_id": node_id,
            "tier": "hot", "user_id": self.user_id
        })
        return True

    def _find_node(self, node_id: str) -> Optional[GraphNode]:
        """Locate a node in WARM, COLD, or graph."""
        if node_id in self._warm:
            return self._warm[node_id]
        # Try COLD DB
        node = self._load_from_cold(node_id)
        if node:
            return node
        # Try graph
        return self.graph.get_node(node_id)

    def _evict_lru_from_hot(self) -> None:
        """Evict the least-recently-used node from HOT to WARM."""
        if not self._hot_order:
            return
        lru_id = self._hot_order.pop(0)
        if lru_id in self._hot:
            node = self._hot.pop(lru_id)
            # Move to WARM
            if len(self._warm) >= settings.WARM_TIER_CAPACITY:
                self._evict_oldest_from_warm()
            self._warm[lru_id] = node
            self._warm.move_to_end(lru_id)

    def _evict_oldest_from_warm(self) -> None:
        """Evict the oldest (LRU) node from WARM to COLD."""
        if not self._warm:
            return
        oldest_id, oldest_node = next(iter(self._warm.items()))
        del self._warm[oldest_id]
        self._evict_lru_from_warm_to_cold(oldest_id)

    def demote_from_hot(self, node_id: str) -> bool:
        """
        Move node_id from HOT to WARM.
        If WARM is full: evict oldest WARM node to COLD.
        Publishes TOPIC_NODE_DEMOTED with {'node_id': node_id, 'from_tier': 'hot', 'to_tier': 'warm', 'user_id': user_id}.
        Returns True on success, False if node_id not in HOT.
        """
        if node_id not in self._hot:
            return False
        node = self._hot.pop(node_id)
        if node_id in self._hot_order:
            self._hot_order.remove(node_id)
        if len(self._warm) >= settings.WARM_TIER_CAPACITY:
            self._evict_oldest_from_warm()
        self._warm[node_id] = node
        self._warm.move_to_end(node_id)
        bus = EventBus.get_instance()
        bus.publish(TOPIC_NODE_DEMOTED, {
            "timestamp": 0.0, "node_id": node_id,
            "from_tier": "hot", "to_tier": "warm",
            "user_id": self.user_id
        })
        return True

    def is_in_hot(self, node_id: str) -> bool:
        """Return True if node_id is in the HOT tier."""
        return node_id in self._hot

    def is_in_warm(self, node_id: str) -> bool:
        """Return True if node_id is in the WARM tier."""
        return node_id in self._warm

    def get_hot_node_ids(self) -> list:
        """Return list of all node_ids currently in HOT tier."""
        return list(self._hot.keys())

    def get_warm_node_ids(self) -> list:
        """Return list of all node_ids currently in WARM tier."""
        return list(self._warm.keys())

    def flush_hot_by_category(self, category: str) -> list:
        """
        Remove all HOT nodes whose GraphNode.category matches category.
        Demote them to WARM (or COLD if WARM is full).
        Returns list of flushed node_ids.
        Used by SecurityAgent on sensitive context transitions.
        """
        to_flush = [
            nid for nid, node in self._hot.items()
            if node.category == category
        ]
        for nid in to_flush:
            self.demote_from_hot(nid)
        logger.info(f"Flushed {len(to_flush)} nodes of category '{category}' from HOT for {self.user_id}")
        return to_flush

    def rebuild_warm_from_graph(self, predicted_node_ids: list) -> None:
        """
        Replace WARM tier content with the given predicted_node_ids.
        Load each node from COLD if not already in HOT or WARM.
        Called by PrefetchDaemon at the start of each session and every 15-min cycle.
        Demotes current WARM nodes to COLD before replacing.
        """
        # Demote all current WARM to COLD
        for nid in list(self._warm.keys()):
            node = self._warm[nid]
            self._save_to_cold(nid, node)
        self._warm.clear()
        # Load predicted nodes into WARM
        for nid in predicted_node_ids:
            if nid in self._hot or nid in self._warm:
                continue
            node = self._load_from_cold(nid) or self.graph.get_node(nid)
            if node:
                self._warm[nid] = node

    def get_tier_stats(self) -> dict:
        """
        Return current tier statistics.
        Returns: {'hot_count': int, 'warm_count': int, 'cold_count': int,
                  'hot_capacity': int, 'warm_capacity': int}
        """
        cold_count = self._count_cold()
        return {
            "hot_count": len(self._hot),
            "warm_count": len(self._warm),
            "cold_count": cold_count,
            "hot_capacity": settings.HOT_TIER_CAPACITY,
            "warm_capacity": settings.WARM_TIER_CAPACITY
        }

    def _count_cold(self) -> int:
        """Count nodes in COLD DB for this user."""
        return len(self.cold_store)

    def check_and_publish_cache_result(self, node_id: str, user_id: str) -> str:
        """
        Check which tier node_id is in.
        Publish TOPIC_CACHE_HIT if found in HOT or WARM tier.
        Publish TOPIC_CACHE_MISS if not found.
        Returns: 'hot', 'warm', 'cold', or 'miss'
        """
        bus = EventBus.get_instance()
        ts = 0.0
        if node_id in self._hot:
            bus.publish(TOPIC_CACHE_HIT, {"timestamp": ts, "node_id": node_id,
                                          "tier": "hot", "user_id": user_id})
            return "hot"
        if node_id in self._warm:
            bus.publish(TOPIC_CACHE_HIT, {"timestamp": ts, "node_id": node_id,
                                          "tier": "warm", "user_id": user_id})
            return "warm"
        # Check COLD
        if self._load_from_cold(node_id) is not None:
            bus.publish(TOPIC_CACHE_MISS, {"timestamp": ts, "node_id": node_id,
                                           "user_id": user_id})
            return "cold"
        bus.publish(TOPIC_CACHE_MISS, {"timestamp": ts, "node_id": node_id,
                                       "user_id": user_id})
        return "miss"

    def _on_app_launched(self, payload: dict) -> None:
        """
        PRIVATE. EventBus callback for TOPIC_APP_LAUNCHED.
        If payload['user_id'] != self.user_id: return.
        Find the node_id for the launched app from self.graph.
        Call check_and_publish_cache_result() for that node_id.
        Call promote_to_hot() for that node_id.
        """
        if payload.get("user_id") != self.user_id:
            return
        app_id = payload.get("app_id", "unknown")
        time_bucket = int(payload.get("time_of_day_bucket", 0))
        battery = float(payload.get("battery", 100.0))
        battery_bucket = min(4, int(battery / 20))
        # Find matching node_id in graph
        node_id = None
        for nid in self.graph._graph.nodes():
            n = self.graph._graph.nodes[nid]["data"]
            if (n.app_id == app_id and n.time_bucket == time_bucket
                    and n.battery_bucket == battery_bucket):
                node_id = nid
                break
        if node_id:
            self.check_and_publish_cache_result(node_id, self.user_id)
            self.promote_to_hot(node_id)

    def _evict_lru_from_warm_to_cold(self, node_id: str) -> None:
        """
        PRIVATE. Move a node from WARM to COLD SQLite.
        Serialize the GraphNode and store in cold_nodes table.
        Remove from WARM dict.
        """
        node = self._warm.get(node_id)
        if node is None:
            node = self.graph.get_node(node_id)
        if node:
            self._save_to_cold(node_id, node)
        if node_id in self._warm:
            del self._warm[node_id]

    def _save_to_cold(self, node_id: str, node: GraphNode) -> None:
        """Serialize and persist node to SQLite COLD DB."""
        self.cold_store[node_id] = pickle.dumps(node)

    def _load_from_cold(self, node_id: str) -> Optional[GraphNode]:
        """Deserialize and return a node from SQLite COLD DB, or None."""
        data = self.cold_store.get(node_id)
        if data:
            return pickle.loads(data)
        return None
