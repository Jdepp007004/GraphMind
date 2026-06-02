"""
src/graph_playback/snapshot_manager.py

Saves and loads graph snapshots keyed by (user_id, day).
Reads from simulation logs produced by BehaviouralGraph.get_graph_snapshot()
which already exist in results/{user_id}_simulation_log.json.

Does NOT modify any existing core modules.
"""

import json
import logging
import os
import pickle
from typing import Optional, List, Dict

from config import settings

logger = logging.getLogger(__name__)


class SnapshotManager:
    """
    Manages serialized graph snapshots for playback.
    Snapshots are either:
      (a) Loaded from existing simulation log JSON files (results/)
      (b) Saved as lightweight pickle files in results/snapshots/
    """

    SNAPSHOT_DIR = os.path.join(settings.RESULTS_DIR, "snapshots")

    def __init__(self) -> None:
        os.makedirs(self.SNAPSHOT_DIR, exist_ok=True)

    def load_from_simulation_log(self, user_id: str) -> List[dict]:
        """
        Load all day snapshots from the existing simulation log JSON.
        Returns list of graph_snapshot dicts ordered by day.
        Each snapshot has: {day, user_id, node_count, edge_count, nodes, edges}
        Returns empty list if log does not exist.
        """
        log_path = os.path.join(settings.RESULTS_DIR, f"{user_id}_simulation_log.json")
        if not os.path.exists(log_path):
            logger.warning(f"Simulation log not found for {user_id}: {log_path}")
            return []
        try:
            with open(log_path) as f:
                log = json.load(f)
            snapshots = []
            for day_data in log.get("days", []):
                snap = day_data.get("graph_snapshot", {})
                if snap:
                    snap["day"] = day_data.get("day", 0)
                    snap["tier_stats"] = day_data.get("tier_stats", {})
                    snap["state"] = day_data.get("state", {})
                    snapshots.append(snap)
            snapshots.sort(key=lambda s: s.get("day", 0))
            logger.debug(f"Loaded {len(snapshots)} snapshots for {user_id}")
            return snapshots
        except Exception as e:
            logger.error(f"Failed to load simulation log for {user_id}: {e}")
            return []

    def save_snapshot(self, user_id: str, day: int, snapshot: dict) -> str:
        """
        Save a single snapshot to disk as JSON in snapshots directory.
        Returns the path written to.
        """
        path = os.path.join(self.SNAPSHOT_DIR, f"{user_id}_day{day:03d}.json")
        try:
            with open(path, "w") as f:
                json.dump(snapshot, f)
        except Exception as e:
            logger.error(f"Failed to save snapshot for {user_id} day {day}: {e}")
        return path

    def load_snapshot(self, user_id: str, day: int) -> Optional[dict]:
        """Load a single saved snapshot for (user_id, day). Returns None if not found."""
        path = os.path.join(self.SNAPSHOT_DIR, f"{user_id}_day{day:03d}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load snapshot for {user_id} day {day}: {e}")
            return None

    def list_available_days(self, user_id: str) -> List[int]:
        """Return sorted list of days for which snapshots are available."""
        # First try simulation log
        snapshots = self.load_from_simulation_log(user_id)
        if snapshots:
            return [s.get("day", 0) for s in snapshots]
        # Fallback: scan snapshot dir
        days = []
        for fname in os.listdir(self.SNAPSHOT_DIR):
            if fname.startswith(f"{user_id}_day") and fname.endswith(".json"):
                try:
                    day = int(fname.split("_day")[1].replace(".json", ""))
                    days.append(day)
                except Exception:
                    pass
        return sorted(days)

    def verify_integrity(self, user_id: str, day: int) -> bool:
        """
        Verify that a snapshot for (user_id, day) can be loaded and has required fields.
        Returns True if valid.
        """
        snapshots = self.load_from_simulation_log(user_id)
        target = next((s for s in snapshots if s.get("day") == day), None)
        if target is None:
            target = self.load_snapshot(user_id, day)
        if target is None:
            return False
        required = {"node_count", "edge_count"}
        return required.issubset(set(target.keys()))
