"""
src/core/graph_engine.py

Core graph data structure. Nodes are situation embeddings. Edges are 3D weighted
directed connections. Handles all graph CRUD, pruning, eviction, serialization.
"""

import uuid
import pickle
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from config import settings
from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """
    Represents a single situation in the user's behavioural graph.
    """
    node_id: str
    embedding: np.ndarray          # shape (NODE_EMBEDDING_DIM,) = (64,)
    app_id: str                    # e.g. "com.instagram.android"
    time_bucket: int               # 0-47 (30-min buckets in a 24hr day)
    battery_bucket: int            # 0-4 (0=0-20%, 1=20-40%, ..., 4=80-100%)
    context_flags: dict            # {"headphones": bool, "calendar_near": bool, "weekend": bool}
    last_seen_day: int             # simulation day of last access (for eviction)
    access_count: int              # total number of times this node was accessed
    category: str                  # from app_taxonomy: "social", "financial", etc.


@dataclass
class GraphEdge:
    """
    Directed weighted edge between two nodes.
    """
    source_id: str
    target_id: str
    transition_prob: float         # [0.0, 1.0] — probability of going to target from source
    time_sensitivity: float        # [0.0, 1.0] — how time-dependent this transition is
    battery_cost: float            # [0.0, 1.0] — battery penalty for pre-fetching target


class BehaviouralGraph:
    """
    The main directed weighted graph. Wraps NetworkX DiGraph.
    One instance per user.
    """

    def __init__(self, user_id: str) -> None:
        """
        Initialize an empty graph for a user.
        user_id: string identifier, e.g. 'user_00'
        Creates an internal nx.DiGraph().
        Subscribes to TOPIC_APP_LAUNCHED on EventBus to call self._on_app_launched().
        """
        self.user_id = user_id
        self._graph = nx.DiGraph()
        self._previous_node_id: Optional[str] = None
        self._session_day: int = 0
        bus = EventBus.get_instance()
        bus.subscribe(TOPIC_APP_LAUNCHED, self._on_app_launched)
        logger.debug(f"BehaviouralGraph initialized for {user_id}")

    def add_node(self, node: GraphNode) -> None:
        """
        Add a GraphNode to the graph.
        If node_id already exists, update last_seen_day and access_count only.
        Publishes nothing.
        """
        if node.node_id in self._graph:
            existing = self._graph.nodes[node.node_id]["data"]
            existing.last_seen_day = node.last_seen_day
            existing.access_count += 1
        else:
            self._graph.add_node(node.node_id, data=node)

    def add_edge(self, source_id: str, target_id: str,
                 transition_prob: float, time_sensitivity: float,
                 battery_cost: float) -> None:
        """
        Add or update a directed edge between two existing nodes.
        If edge already exists, update all three weight values.
        Raises ValueError if source_id or target_id not in graph.
        """
        if source_id not in self._graph:
            raise ValueError(f"Source node '{source_id}' not in graph")
        if target_id not in self._graph:
            raise ValueError(f"Target node '{target_id}' not in graph")
        self._graph.add_edge(source_id, target_id,
                             transition_prob=transition_prob,
                             time_sensitivity=time_sensitivity,
                             battery_cost=battery_cost)

    def update_edge_weights(self, source_id: str, target_id: str,
                            delta_prob: float, delta_time: float,
                            delta_battery: float) -> None:
        """
        Apply additive delta to edge weights. Clamp all values to [0.0, 1.0] after update.
        Raises ValueError if edge does not exist.
        """
        if not self._graph.has_edge(source_id, target_id):
            raise ValueError(f"Edge '{source_id}' -> '{target_id}' does not exist")
        edge_data = self._graph[source_id][target_id]
        edge_data["transition_prob"] = max(0.0, min(1.0, edge_data["transition_prob"] + delta_prob))
        edge_data["time_sensitivity"] = max(0.0, min(1.0, edge_data["time_sensitivity"] + delta_time))
        edge_data["battery_cost"] = max(0.0, min(1.0, edge_data["battery_cost"] + delta_battery))

    def normalize_outgoing_edges(self, source_id: str) -> None:
        """Normalize outgoing transition weights so they sum to 1.0."""
        if source_id not in self._graph:
            return
        outgoing = list(self._graph.out_edges(source_id, data=True))
        total = sum(max(0.0, d.get("transition_prob", 0.0)) for _, _, d in outgoing)
        if total <= 0:
            return
        for _, _, edge_data in outgoing:
            edge_data["transition_prob"] = edge_data.get("transition_prob", 0.0) / total

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """
        Return the GraphNode for node_id, or None if not found.
        """
        if node_id in self._graph:
            return self._graph.nodes[node_id]["data"]
        return None

    def get_edges_from(self, node_id: str) -> list:
        """
        Return all outgoing edges from node_id as a list of GraphEdge objects.
        Returns empty list if node not found or has no outgoing edges.
        """
        if node_id not in self._graph:
            return []
        edges = []
        for _, target_id, edge_data in self._graph.out_edges(node_id, data=True):
            edges.append(GraphEdge(
                source_id=node_id,
                target_id=target_id,
                transition_prob=edge_data.get("transition_prob", 0.0),
                time_sensitivity=edge_data.get("time_sensitivity", 0.0),
                battery_cost=edge_data.get("battery_cost", 0.0)
            ))
        return edges

    def get_top_k_next_nodes(self, current_node_id: str, k: int,
                              battery_level: float) -> list:
        """
        Return the top-k most likely next node_ids from current_node_id.
        Scoring: score = transition_prob - (battery_cost * (1 - battery_level/100))
        If battery_level < BATTERY_SUPPRESS_THRESHOLD, set k = max(1, k // 2).
        Sort edges by score descending, return top-k target node_ids.
        Returns empty list if current_node_id not in graph.
        """
        if current_node_id not in self._graph:
            return []
        if battery_level < settings.BATTERY_SUPPRESS_THRESHOLD:
            k = max(1, k // 2)
        edges = self.get_edges_from(current_node_id)
        scored = []
        for edge in edges:
            score = edge.transition_prob - (edge.battery_cost * (1.0 - battery_level / 100.0))
            scored.append((score, edge.target_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [nid for _, nid in scored[:k]]

    def prune_weak_edges(self) -> int:
        """
        Delete all edges where transition_prob < EDGE_PRUNE_THRESHOLD (0.05).
        Returns the number of edges deleted.
        Does NOT publish an event.
        """
        to_remove = [
            (u, v) for u, v, d in list(self._graph.edges(data=True))
            if d.get("transition_prob", 0.0) < settings.EDGE_PRUNE_THRESHOLD
        ]
        self._graph.remove_edges_from(to_remove)
        if to_remove:
            logger.debug(f"Pruned {len(to_remove)} weak edges for {self.user_id}")
        return len(to_remove)

    def evict_stale_nodes(self, current_day: int) -> int:
        """
        Delete all nodes where (current_day - last_seen_day) > NODE_EVICTION_DAYS (45).
        Also delete all edges connected to evicted nodes.
        Returns the number of nodes evicted.
        """
        to_evict = []
        for node_id in list(self._graph.nodes()):
            node = self._graph.nodes[node_id]["data"]
            if (current_day - node.last_seen_day) > settings.NODE_EVICTION_DAYS:
                to_evict.append(node_id)
        for node_id in to_evict:
            self._graph.remove_node(node_id)
        if to_evict:
            logger.debug(f"Evicted {len(to_evict)} stale nodes for {self.user_id}")
        return len(to_evict)

    def node_count(self) -> int:
        """Return total number of nodes in the graph."""
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Return total number of edges in the graph."""
        return self._graph.number_of_edges()

    def save_to_disk(self, path: str) -> None:
        """
        Serialize the entire graph to a pickle file at path.
        Creates parent directories if they don't exist.
        Raises IOError on failure.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "user_id": self.user_id,
                    "graph": self._graph,
                    "previous_node_id": self._previous_node_id,
                    "session_day": self._session_day
                }, f)
            logger.debug(f"Graph saved to {path}")
        except Exception as e:
            raise IOError(f"Failed to save graph to {path}: {e}")

    def load_from_disk(self, path: str) -> None:
        """
        Load graph state from pickle file at path. Overwrites current state.
        Raises FileNotFoundError if path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph file not found: {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.user_id = state["user_id"]
        self._graph = state["graph"]
        self._previous_node_id = state.get("previous_node_id")
        self._session_day = state.get("session_day", 0)
        logger.debug(f"Graph loaded from {path}: {self.node_count()} nodes, {self.edge_count()} edges")

    def get_graph_snapshot(self, day: int) -> dict:
        """
        Return a JSON-serializable snapshot of the graph for the dashboard.
        Returns: {
            'day': day,
            'user_id': self.user_id,
            'node_count': int,
            'edge_count': int,
            'nodes': [{'node_id': str, 'app_id': str, 'category': str, 'access_count': int}, ...],
            'edges': [{'source': str, 'target': str, 'prob': float}, ...]
        }
        Truncates to max 200 nodes/500 edges for rendering performance.
        """
        nodes_list = []
        for nid in list(self._graph.nodes())[:200]:
            node = self._graph.nodes[nid]["data"]
            nodes_list.append({
                "node_id": node.node_id,
                "app_id": node.app_id,
                "category": node.category,
                "access_count": int(node.access_count)
            })
        edges_list = []
        for u, v, d in list(self._graph.edges(data=True))[:500]:
            edges_list.append({
                "source": u,
                "target": v,
                "prob": float(d.get("transition_prob", 0.0))
            })
        return {
            "day": day,
            "user_id": self.user_id,
            "node_count": self.node_count(),
            "edge_count": self.edge_count(),
            "nodes": nodes_list,
            "edges": edges_list
        }

    def _on_app_launched(self, payload: dict) -> None:
        """
        PRIVATE. EventBus callback for TOPIC_APP_LAUNCHED.
        payload keys: app_id, user_id, battery, time_of_day_bucket
        If payload['user_id'] != self.user_id: return immediately.
        Find or create a GraphNode for this (app_id, time_of_day_bucket, battery_bucket) tuple.
        Update last_seen_day and access_count.
        If a previous node exists in this session, add/update edge from previous -> current.
        Increment transition_prob by 0.01 on each occurrence (clamped to 1.0).
        """
        if payload.get("user_id") != self.user_id:
            return
        app_id = payload.get("app_id", "unknown")
        time_bucket = int(payload.get("time_of_day_bucket", 0))
        battery = float(payload.get("battery", 100.0))
        battery_bucket = min(4, int(battery / 20))
        day = int(payload.get("day", self._session_day))
        self._session_day = day
        category = payload.get("category", "utility")
        headphones = bool(payload.get("headphones", False))
        calendar_mins = payload.get("calendar_event_in_mins")
        calendar_near = calendar_mins is not None and calendar_mins <= 30
        weekend = bool(payload.get("weekend", False))

        # Find matching node
        current_node_id = None
        for nid in self._graph.nodes():
            n = self._graph.nodes[nid]["data"]
            if (n.app_id == app_id and n.time_bucket == time_bucket
                    and n.battery_bucket == battery_bucket):
                current_node_id = nid
                break

        if current_node_id is None:
            # Create new node
            new_node = GraphNode(
                node_id=str(uuid.uuid4()),
                embedding=np.zeros(settings.NODE_EMBEDDING_DIM),
                app_id=app_id,
                time_bucket=time_bucket,
                battery_bucket=battery_bucket,
                context_flags={"headphones": headphones,
                                "calendar_near": calendar_near,
                                "weekend": weekend},
                last_seen_day=day,
                access_count=1,
                category=category
            )
            self._graph.add_node(new_node.node_id, data=new_node)
            current_node_id = new_node.node_id
        else:
            node = self._graph.nodes[current_node_id]["data"]
            node.last_seen_day = day
            node.access_count += 1

        # Update edge from previous to current
        if self._previous_node_id is not None and self._previous_node_id in self._graph:
            if self._graph.has_edge(self._previous_node_id, current_node_id):
                self.update_edge_weights(self._previous_node_id, current_node_id,
                                         delta_prob=0.01, delta_time=0.0, delta_battery=0.0)
            else:
                self.add_edge(self._previous_node_id, current_node_id, 0.1, 0.5, 0.2)
            self.normalize_outgoing_edges(self._previous_node_id)

        self._previous_node_id = current_node_id
