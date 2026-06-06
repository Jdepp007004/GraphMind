"""
src/data/event_dataset.py

Abstract EventDataset interface + concrete implementations.

All evaluation code in GraphMind v2 must consume events through this
interface. This ensures that any data source (synthetic, Device Analyzer,
future Samsung logs) can be swapped in without touching evaluation code.

Implementations:
  SyntheticDataset     — wraps the existing DatasetGenerator
  DeviceAnalyzerDataset — wraps DeviceAnalyzerLoader (requires raw data)
  SamsungLogDataset    — stub for future Samsung production logs
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event type alias — all datasets produce dicts matching this schema.
# ---------------------------------------------------------------------------
#
# Required fields (all datasets must populate):
#   timestamp               : float   — Unix epoch seconds
#   app_id                  : str     — Android package ID
#   battery                 : float   — 0.0–100.0
#   time_bucket             : int     — 0–47 (30-min buckets)
#   headphones              : bool
#   calendar_event_in_mins  : int | None
#   weekend                 : bool
#
# Optional fields (filled when available):
#   day                     : int     — simulation day index (0-indexed)
#   category                : str     — from app_taxonomy

GraphMindEvent = Dict[str, object]


class EventDataset(ABC):
    """
    Abstract base class for all GraphMind event data sources.

    Subclasses must implement:
      load()        — parse/load data from disk; idempotent
      iter_events() — yield events in chronological order
      metadata()    — return a JSON-serialisable summary dict

    The interface deliberately does not expose random-access indexing.
    Events should always be consumed in temporal order to prevent
    future-leakage in sequence models.
    """

    @abstractmethod
    def load(self) -> None:
        """Load and parse data from disk. Must be idempotent."""

    @abstractmethod
    def iter_events(
        self, split: str = "train"
    ) -> Iterator[GraphMindEvent]:
        """
        Yield events from the requested split in chronological order.

        Args:
            split: One of "train", "val", "test", or "all".
                   Splits are chronological (train=earliest 80%,
                   val=next 10%, test=last 10%).

        Yields:
            GraphMindEvent dicts in ascending timestamp order.
        """

    @abstractmethod
    def metadata(self) -> dict:
        """
        Return a JSON-serialisable metadata dict describing the dataset.

        Minimum required keys:
          source        : str  — dataset identifier
          total_events  : int
          split_sizes   : dict — {"train": int, "val": int, "test": int}
          loaded        : bool
        """

    def get_splits(self) -> Dict[str, List[GraphMindEvent]]:
        """
        Return all events partitioned into {"train": [...], "val": [...], "test": [...]}.

        Convenience wrapper around iter_events() for callers that need list access.
        """
        return {
            split: list(self.iter_events(split))
            for split in ("train", "val", "test")
        }

    @staticmethod
    def _chronological_split(
        events: List[GraphMindEvent],
        train_ratio: float = settings.DATASET_TRAIN_RATIO,
        val_ratio: float = settings.DATASET_VAL_RATIO,
    ) -> Dict[str, List[GraphMindEvent]]:
        """
        Split a chronologically sorted event list into train/val/test.

        IMPORTANT: The split is done by index position on a pre-sorted list,
        NOT by random sampling. This prevents future behavior from leaking
        into the training set — critical for sequence prediction evaluation.

        Args:
            events:      Sorted list of GraphMindEvent dicts.
            train_ratio: Fraction of events for training (default 0.80).
            val_ratio:   Fraction of events for validation (default 0.10).

        Returns:
            {"train": [...], "val": [...], "test": [...]}
        """
        n = len(events)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return {
            "train": events[:train_end],
            "val": events[train_end:val_end],
            "test": events[val_end:],
        }


# ---------------------------------------------------------------------------
# SyntheticDataset
# ---------------------------------------------------------------------------

class SyntheticDataset(EventDataset):
    """
    Wraps the existing DatasetGenerator to expose the EventDataset interface.

    Uses the 10 synthetic user personas. All existing simulations that
    produce these events remain unmodified — this class is a read-only
    adapter.

    Data source: data/synthetic/users/user_XX.json
    """

    def __init__(
        self,
        users_dir: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            users_dir: Override for settings.USERS_DIR.
            user_ids:  Subset of user IDs to load. None = all 10 users.
        """
        self._users_dir = users_dir or settings.USERS_DIR
        self._user_ids = user_ids
        self._events: List[GraphMindEvent] = []
        self._splits: Dict[str, List[GraphMindEvent]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all synthetic user JSON files and build chronological splits."""
        import json

        if self._loaded:
            return

        from src.data.dataset_generator import USER_PROFILES

        profiles = [
            p for p in USER_PROFILES
            if self._user_ids is None or p["user_id"] in self._user_ids
        ]

        all_events: List[GraphMindEvent] = []
        for profile in profiles:
            uid = profile["user_id"]
            path = os.path.join(self._users_dir, f"{uid}.json")
            if not os.path.exists(path):
                logger.warning(
                    f"SyntheticDataset: missing file for {uid} at {path}. "
                    "Run scripts/generate_dataset.py first."
                )
                continue
            with open(path, encoding="utf-8") as fh:
                events = json.load(fh)
            # Tag each event with user_id for traceability
            for evt in events:
                evt.setdefault("user_id", uid)
            all_events.extend(events)

        # Sort globally by timestamp (then by day as secondary key)
        all_events.sort(key=lambda e: (int(e.get("day", 0)), float(e.get("timestamp", 0.0))))

        self._events = all_events
        self._splits = self._chronological_split(all_events)
        self._loaded = True
        logger.info(
            f"SyntheticDataset loaded: {len(all_events)} events "
            f"({len(profiles)} users) — "
            f"train={len(self._splits['train'])} "
            f"val={len(self._splits['val'])} "
            f"test={len(self._splits['test'])}"
        )

    def iter_events(self, split: str = "train") -> Iterator[GraphMindEvent]:
        """Yield events from the requested split in chronological order."""
        if not self._loaded:
            self.load()
        if split == "all":
            yield from self._events
        else:
            yield from self._splits.get(split, [])

    def metadata(self) -> dict:
        """Return dataset metadata."""
        if not self._loaded:
            self.load()
        return {
            "source": "synthetic",
            "users_dir": self._users_dir,
            "total_events": len(self._events),
            "split_sizes": {k: len(v) for k, v in self._splits.items()},
            "loaded": self._loaded,
        }


# ---------------------------------------------------------------------------
# DeviceAnalyzerDataset
# ---------------------------------------------------------------------------

class DeviceAnalyzerDataset(EventDataset):
    """
    Wraps DeviceAnalyzerLoader to expose the EventDataset interface.

    Requires data to be present under data/device_analyzer/raw/.
    Run `python scripts/setup_device_analyzer.py` first.

    Falls back gracefully to SyntheticDataset when raw data is absent.
    """

    def __init__(self, fallback_to_synthetic: bool = True) -> None:
        """
        Args:
            fallback_to_synthetic: If True and raw data is missing,
                                   delegate to SyntheticDataset silently.
        """
        self._fallback_to_synthetic = fallback_to_synthetic
        self._events: List[GraphMindEvent] = []
        self._splits: Dict[str, List[GraphMindEvent]] = {}
        self._loaded = False
        self._used_fallback = False

    def load(self) -> None:
        """Load Device Analyzer data; fall back to synthetic if absent."""
        if self._loaded:
            return

        # Check for pre-built split files first (fastest path)
        splits_dir = settings.DEVICE_ANALYZER_SPLITS_DIR
        if os.path.isdir(splits_dir) and any(
            f.endswith(".json") for f in os.listdir(splits_dir)
        ):
            self._load_from_splits(splits_dir)
            return

        # Check for raw CSV files
        raw_dir = settings.DEVICE_ANALYZER_RAW_DIR
        has_raw = os.path.isdir(raw_dir) and any(
            f.endswith(".csv") for f in os.listdir(raw_dir)
        )
        if has_raw:
            self._load_from_raw()
            return

        # No data available
        if self._fallback_to_synthetic:
            logger.warning(
                "DeviceAnalyzerDataset: no raw data found. "
                "Falling back to SyntheticDataset. "
                "Run `python scripts/setup_device_analyzer.py` to acquire real data."
            )
            fallback = SyntheticDataset()
            fallback.load()
            self._events = list(fallback.iter_events("all"))
            self._splits = fallback.get_splits()
            self._used_fallback = True
            self._loaded = True
        else:
            raise FileNotFoundError(
                f"Device Analyzer raw data not found at {raw_dir}. "
                "Run `python scripts/setup_device_analyzer.py`."
            )

    def _load_from_splits(self, splits_dir: str) -> None:
        """Load pre-built JSON split files."""
        import json

        all_events: List[GraphMindEvent] = []
        for split_name in ("train", "val", "test"):
            path = os.path.join(splits_dir, f"{split_name}.json")
            if not os.path.exists(path):
                self._splits[split_name] = []
                continue
            with open(path, encoding="utf-8") as fh:
                events = json.load(fh)
            self._splits[split_name] = events
            all_events.extend(events)

        self._events = all_events
        self._loaded = True
        logger.info(
            f"DeviceAnalyzerDataset loaded from splits: {len(all_events)} events — "
            f"train={len(self._splits.get('train', []))} "
            f"val={len(self._splits.get('val', []))} "
            f"test={len(self._splits.get('test', []))}"
        )

    def _load_from_raw(self) -> None:
        """Parse raw CSV files via DeviceAnalyzerLoader."""
        from src.data.device_analyzer_loader import DeviceAnalyzerLoader
        loader = DeviceAnalyzerLoader()
        loader.load()
        self._splits = loader.get_splits()
        self._events = (
            self._splits.get("train", [])
            + self._splits.get("val", [])
            + self._splits.get("test", [])
        )
        self._loaded = True
        logger.info(
            f"DeviceAnalyzerDataset loaded from raw: {len(self._events)} events"
        )

    def iter_events(self, split: str = "train") -> Iterator[GraphMindEvent]:
        """Yield events from the requested split in chronological order."""
        if not self._loaded:
            self.load()
        if split == "all":
            yield from self._events
        else:
            yield from self._splits.get(split, [])

    def metadata(self) -> dict:
        """Return dataset metadata."""
        if not self._loaded:
            self.load()
        return {
            "source": "device_analyzer" if not self._used_fallback else "synthetic_fallback",
            "total_events": len(self._events),
            "split_sizes": {k: len(v) for k, v in self._splits.items()},
            "used_fallback": self._used_fallback,
            "loaded": self._loaded,
        }


# ---------------------------------------------------------------------------
# SamsungLogDataset  (future)
# ---------------------------------------------------------------------------

class SamsungLogDataset(EventDataset):
    """
    Stub for future Samsung production log integration.

    Not implemented. Raises NotImplementedError on all method calls.
    This stub exists to reserve the interface contract so that future
    integration does not require changes to evaluation code.
    """

    def load(self) -> None:
        raise NotImplementedError(
            "SamsungLogDataset is not yet implemented. "
            "This stub reserves the EventDataset interface for future Samsung log integration."
        )

    def iter_events(self, split: str = "train") -> Iterator[GraphMindEvent]:
        raise NotImplementedError("SamsungLogDataset is not yet implemented.")

    def metadata(self) -> dict:
        return {
            "source": "samsung_logs",
            "total_events": 0,
            "split_sizes": {"train": 0, "val": 0, "test": 0},
            "loaded": False,
            "note": "Not yet implemented — stub only.",
        }
