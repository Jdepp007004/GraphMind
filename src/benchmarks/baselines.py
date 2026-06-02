"""
src/benchmarks/baselines.py

Implements 4 baseline policies to compare against GraphMind.
"""

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import List, Dict, Any

from config import settings

logger = logging.getLogger(__name__)


class BaselinePolicy(ABC):
    """Abstract base class for all baselines."""

    @abstractmethod
    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return list of predicted next app_ids (ordered by confidence)."""

    @abstractmethod
    def update(self, event: dict) -> None:
        """Update policy state with a new observed event."""

    def reset(self) -> None:
        """Reset policy to initial state."""

    @abstractmethod
    def get_name(self) -> str:
        """Return BASELINE_* constant name."""


class LMKDReactiveBaseline(BaselinePolicy):
    """
    Simulates Android LMKD behavior: purely reactive, no prediction.
    Keeps the N most-recently-used apps in memory. Evicts LRU on overflow.
    No time-of-day awareness. No transition modelling.
    capacity: HOT_TIER_CAPACITY
    """

    def __init__(self) -> None:
        """Initialize with LRU tracking."""
        self.capacity = settings.HOT_TIER_CAPACITY
        self._lru: OrderedDict = OrderedDict()

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Returns top-5 most recently used apps regardless of context."""
        return list(self._lru.keys())[:5]

    def update(self, event: dict) -> None:
        """Add app_id to front of LRU queue. Evict tail if over capacity."""
        app_id = event.get("app_id", "unknown")
        if app_id in self._lru:
            self._lru.move_to_end(app_id, last=False)
        else:
            self._lru[app_id] = True
            self._lru.move_to_end(app_id, last=False)
        while len(self._lru) > self.capacity:
            self._lru.popitem(last=True)

    def reset(self) -> None:
        """Reset LRU state."""
        self._lru.clear()

    def get_name(self) -> str:
        """Return LMKD baseline name."""
        return settings.BASELINE_LMKD


class ARTStaticProfileBaseline(BaselinePolicy):
    """
    Simulates Android ART Baseline Profile behavior:
    Pre-warms the top-N most frequently launched apps per time-of-day bucket.
    Profile is built from Day 1-7 and then FROZEN (static, no further learning).
    Represents ART's AOT compilation of hot code paths.
    """

    def __init__(self) -> None:
        """Initialize with empty profile."""
        self._profile: Dict[int, List[str]] = {}
        self._profile_built: bool = False

    def build_profile(self, events: List[dict]) -> None:
        """
        Build static frequency profile from first 7 days of events.
        profile[time_bucket] = [app_id ordered by frequency]
        """
        freq: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for evt in events:
            if int(evt.get("day", 0)) < 7:
                bucket = int(evt.get("time_bucket", 0))
                app_id = evt.get("app_id", "unknown")
                freq[bucket][app_id] += 1
        for bucket, app_counts in freq.items():
            sorted_apps = sorted(app_counts.keys(), key=lambda a: app_counts[a], reverse=True)
            self._profile[bucket] = sorted_apps[:10]
        self._profile_built = True

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return profile[context['time_bucket']] top-5."""
        bucket = int(context.get("time_bucket", 0))
        return self._profile.get(bucket, [])[:5]

    def update(self, event: dict) -> None:
        """No-op: ART profile is frozen after Day 7."""
        pass  # Profile is frozen — do not update

    def reset(self) -> None:
        """Reset profile."""
        self._profile = {}
        self._profile_built = False

    def get_name(self) -> str:
        """Return ART baseline name."""
        return settings.BASELINE_ART


class UsageStatsLRUBaseline(BaselinePolicy):
    """
    Simulates Android UsageStatsManager + LRU process cache.
    Keeps recently-used apps warm. Updates continuously but uses recency only.
    No transition modelling (doesn't know that Instagram follows WhatsApp).
    """

    def __init__(self) -> None:
        """Initialize LRU tracker."""
        self._lru: OrderedDict = OrderedDict()

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Returns top-5 most recently used apps, context-agnostic."""
        return list(self._lru.keys())[:5]

    def update(self, event: dict) -> None:
        """Add/move app_id to front of LRU."""
        app_id = event.get("app_id", "unknown")
        if app_id in self._lru:
            self._lru.move_to_end(app_id, last=False)
        else:
            self._lru[app_id] = True
            self._lru.move_to_end(app_id, last=False)

    def reset(self) -> None:
        """Reset LRU."""
        self._lru.clear()

    def get_name(self) -> str:
        """Return LRU baseline name."""
        return settings.BASELINE_LRU


class BixbyFrequencyBaseline(BaselinePolicy):
    """
    Simulates Samsung Bixby Routines / One UI app suggestions.
    Uses frequency counts per (time_bucket, day_of_week) pair.
    Updates continuously but no RL, no graph structure, no transition chains.
    """

    def __init__(self) -> None:
        """Initialize frequency tracker."""
        self._freq: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return top-5 most frequent apps for current (time_bucket, day_of_week)."""
        bucket = int(context.get("time_bucket", 0))
        weekend = bool(context.get("weekend", False))
        key = (bucket, weekend)
        app_freq = self._freq.get(key, {})
        sorted_apps = sorted(app_freq.keys(), key=lambda a: app_freq[a], reverse=True)
        return sorted_apps[:5]

    def update(self, event: dict) -> None:
        """Update frequency for (time_bucket, weekend) key."""
        bucket = int(event.get("time_bucket", 0))
        weekend = bool(event.get("weekend", False))
        app_id = event.get("app_id", "unknown")
        key = (bucket, weekend)
        self._freq[key][app_id] += 1

    def reset(self) -> None:
        """Reset frequency counts."""
        self._freq = defaultdict(lambda: defaultdict(int))

    def get_name(self) -> str:
        """Return Bixby baseline name."""
        return settings.BASELINE_BIXBY
