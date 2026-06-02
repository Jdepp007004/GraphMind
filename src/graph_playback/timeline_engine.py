"""
src/graph_playback/timeline_engine.py

Reconstructs the day-by-day graph evolution timeline from simulation snapshots.
Computes per-day deltas: node growth, edge strengthening, edge pruning,
tier promotions, and drift events.
"""

import logging
from typing import List, Dict, Optional, Tuple

from src.graph_playback.snapshot_manager import SnapshotManager

logger = logging.getLogger(__name__)


class TimelineFrame:
    """
    Represents the state of the graph at a single point in time (one day).
    Includes delta metrics versus the previous day.
    """

    def __init__(self, day: int, snapshot: dict) -> None:
        self.day = day
        self.snapshot = snapshot
        self.node_count: int = snapshot.get("node_count", 0)
        self.edge_count: int = snapshot.get("edge_count", 0)
        self.nodes: List[dict] = snapshot.get("nodes", [])
        self.edges: List[dict] = snapshot.get("edges", [])
        self.tier_stats: dict = snapshot.get("tier_stats", {})
        self.state: dict = snapshot.get("state", {})

        # Delta fields — filled in by TimelineEngine
        self.node_delta: int = 0          # nodes added since previous day
        self.edge_delta: int = 0          # edges added/pruned since previous day
        self.new_node_ids: List[str] = []
        self.pruned_edge_count: int = 0
        self.cache_hit_rate: float = self.state.get("cache_hit_rate", 0.0)
        self.kl_divergence: float = self.state.get("kl_divergence", 0.0)
        self.security_flush_count: int = self.state.get("security_flush_count", 0)
        self.drift_detected: bool = self.kl_divergence > 0.3

    def to_dict(self) -> dict:
        """Serialize the timeline frame to a dashboard-ready dict."""
        return {
            "day": self.day,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "node_delta": self.node_delta,
            "edge_delta": self.edge_delta,
            "new_node_count": len(self.new_node_ids),
            "pruned_edge_count": self.pruned_edge_count,
            "cache_hit_rate": self.cache_hit_rate,
            "kl_divergence": self.kl_divergence,
            "security_flush_count": self.security_flush_count,
            "drift_detected": self.drift_detected,
            "hot_count": self.tier_stats.get("hot_count", 0),
            "warm_count": self.tier_stats.get("warm_count", 0),
            "cold_count": self.tier_stats.get("cold_count", 0),
        }

    def __repr__(self) -> str:
        return (f"TimelineFrame(day={self.day}, nodes={self.node_count}, "
                f"edges={self.edge_count}, hit_rate={self.cache_hit_rate:.2%})")


class TimelineEngine:
    """
    Reconstructs and navigates the graph evolution timeline for a user.
    Provides a scrubber-style API: go to day N, step forward/backward.
    """

    def __init__(self, user_id: str,
                 snapshot_manager: Optional[SnapshotManager] = None) -> None:
        self.user_id = user_id
        self.snapshot_manager = snapshot_manager or SnapshotManager()
        self._frames: List[TimelineFrame] = []
        self._loaded = False

    def load(self) -> bool:
        """
        Load all snapshots from simulation log and build the timeline.
        Returns True if any frames were loaded.
        """
        snapshots = self.snapshot_manager.load_from_simulation_log(self.user_id)
        if not snapshots:
            logger.warning(f"No snapshots available for {self.user_id}")
            self._frames = []
            self._loaded = False
            return False

        self._frames = [TimelineFrame(s.get("day", i), s)
                        for i, s in enumerate(snapshots)]
        self._compute_deltas()
        self._loaded = True
        logger.info(f"TimelineEngine loaded {len(self._frames)} frames for {self.user_id}")
        return True

    def _compute_deltas(self) -> None:
        """Compute day-over-day delta metrics for all frames."""
        if not self._frames:
            return
        prev_node_ids: set = set()
        prev_edge_count: int = 0

        for frame in self._frames:
            current_node_ids = {n.get("node_id", "") for n in frame.nodes}
            frame.new_node_ids = list(current_node_ids - prev_node_ids)
            frame.node_delta = frame.node_count - len(prev_node_ids)
            frame.edge_delta = frame.edge_count - prev_edge_count
            frame.pruned_edge_count = max(0, -frame.edge_delta)
            prev_node_ids = current_node_ids
            prev_edge_count = frame.edge_count

    def get_frame(self, day: int) -> Optional[TimelineFrame]:
        """Return the TimelineFrame for a specific day. None if not found."""
        if not self._loaded:
            self.load()
        for frame in self._frames:
            if frame.day == day:
                return frame
        return None

    def get_all_frames(self) -> List[TimelineFrame]:
        """Return all frames in chronological order."""
        if not self._loaded:
            self.load()
        return list(self._frames)

    def get_frames_dict(self) -> List[dict]:
        """Return all frames as list of dicts (for dashboard/Plotly)."""
        return [f.to_dict() for f in self.get_all_frames()]

    def available_days(self) -> List[int]:
        """Return list of days with available data."""
        if not self._loaded:
            self.load()
        return [f.day for f in self._frames]

    def get_milestone_frames(self) -> List[TimelineFrame]:
        """
        Return milestone frames at days 1, 7, 14, 21, 30 (or nearest available).
        Used by dashboard to show key evolution checkpoints.
        """
        milestones = [1, 7, 14, 21, 29]
        all_frames = self.get_all_frames()
        if not all_frames:
            return []
        result = []
        for target_day in milestones:
            # Find nearest available day
            nearest = min(all_frames,
                          key=lambda f: abs(f.day - target_day),
                          default=None)
            if nearest and nearest not in result:
                result.append(nearest)
        return result

    def get_growth_series(self) -> Dict[str, List]:
        """
        Return time-series data for plotting graph growth.
        Returns: {
          'days': [0, 1, 2, ...],
          'node_counts': [...],
          'edge_counts': [...],
          'cache_hit_rates': [...],
          'kl_divergence': [...]
        }
        """
        frames = self.get_all_frames()
        return {
            "days": [f.day for f in frames],
            "node_counts": [f.node_count for f in frames],
            "edge_counts": [f.edge_count for f in frames],
            "cache_hit_rates": [f.cache_hit_rate for f in frames],
            "kl_divergence": [f.kl_divergence for f in frames],
            "hot_counts": [f.tier_stats.get("hot_count", 0) for f in frames],
            "warm_counts": [f.tier_stats.get("warm_count", 0) for f in frames],
        }

    def get_drift_events(self) -> List[dict]:
        """Return all frames where behavioral drift was detected."""
        return [f.to_dict() for f in self.get_all_frames() if f.drift_detected]
