"""
src/core/five_tier_cache.py

GraphMind V6 -- 5-Tier Memory Hierarchy.

Architecture:
    PIN  (3 slots)  -- permanently pinned top-N most-frequent apps. ~10ms access.
    HOT  (5 slots)  -- LRU dynamic resident apps. ~42ms access.
    WARM (8 slots)  -- Prefetched by confidence scorer. ~190ms access.
    COOL (20 slots) -- Recently evicted from WARM, compressed standby. ~400ms access.
    COLD (unlimited)-- On-disk / evicted from all upper tiers. ~720ms access.

The COOL tier is the key innovation in V6. When apps are evicted from WARM, they
are not immediately dropped to COLD (720ms). Instead they enter COOL (~400ms) for
up to COOL_TIER_CAPACITY slots. This captures short-term re-access patterns where
a user briefly switches away and comes back -- reducing effective cold-start latency
for these re-access events by ~(720-400)/720 = 44%.

Public API mirrors MemoryManager (V5) for drop-in compatibility.
"""

import logging
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)


class FiveTierCache:
    """
    5-tier memory cache for GraphMind V6.

    Tier hierarchy (fastest to slowest):
        PIN  -> HOT  -> WARM  -> COOL  -> COLD

    Promotion path (on access): COLD/COOL -> WARM -> HOT -> (PIN, if top-freq)
    Demotion path (on eviction): HOT -> WARM -> COOL -> COLD
    """

    def __init__(self, user_id: str = "default") -> None:
        self.user_id = user_id

        # PIN: set of permanently pinned app_ids (top-N by frequency)
        self._pin: set = set()

        # HOT: OrderedDict LRU, app_id -> latency_tier tag ("hot")
        self._hot: OrderedDict = OrderedDict()

        # WARM: OrderedDict LRU, app_id -> "warm"
        self._warm: OrderedDict = OrderedDict()

        # COOL: OrderedDict LRU, app_id -> "cool"
        self._cool: OrderedDict = OrderedDict()

        # COLD: set (unlimited, disk-equivalent)
        self._cold: set = set()

        # Frequency counter for PIN determination
        self._freq: Counter = Counter()

        # Stats
        self.total_hits = 0
        self.total_misses = 0
        self.pin_hits = 0
        self.hot_hits = 0
        self.warm_hits = 0
        self.cool_hits = 0

    # -- PIN management -----------------------------------------------------

    def _refresh_pin(self) -> None:
        """Recompute the PIN set from top-N most-frequent apps."""
        top_n = [app for app, _ in self._freq.most_common(settings.PIN_TIER_CAPACITY)]
        self._pin = set(top_n)

    # -- Lookup (read path) -------------------------------------------------

    def lookup(self, app_id: str) -> str:
        """
        Look up app_id. Returns the tier it was found in and updates hit/miss stats.

        Returns: 'pin', 'hot', 'warm', 'cool', or 'cold'
        Updates LRU order on hit.
        """
        self._freq[app_id] += 1
        self._refresh_pin()

        if app_id in self._pin:
            self.pin_hits += 1
            self.total_hits += 1
            return "pin"

        if app_id in self._hot:
            self._hot.move_to_end(app_id)
            self.hot_hits += 1
            self.total_hits += 1
            return "hot"

        if app_id in self._warm:
            self._warm.move_to_end(app_id)
            self.warm_hits += 1
            self.total_hits += 1
            # Promote to HOT on warm hit
            self._promote_warm_to_hot(app_id)
            return "warm"

        if app_id in self._cool:
            self._cool.move_to_end(app_id)
            self.cool_hits += 1
            self.total_hits += 1
            # Promote COOL -> WARM -> maybe HOT
            self._promote_cool_to_warm(app_id)
            return "cool"

        # COLD / miss
        self.total_misses += 1
        self._cold.discard(app_id)
        # Bring into WARM on first access
        self._insert_into_warm(app_id)
        return "cold"

    # -- Insert / prefetch (write path) -------------------------------------

    def prefetch(self, app_ids: List[str]) -> None:
        """
        Prefetch a list of apps into WARM tier.
        Apps already in PIN/HOT are skipped.
        Apps already in WARM are refreshed (LRU bump).
        Apps in COOL are promoted to WARM.
        """
        for app_id in app_ids:
            if app_id in self._pin or app_id in self._hot:
                continue
            if app_id in self._warm:
                self._warm.move_to_end(app_id)
                continue
            if app_id in self._cool:
                self._promote_cool_to_warm(app_id)
                continue
            self._insert_into_warm(app_id)

    # -- Tier promotions ----------------------------------------------------

    def _promote_warm_to_hot(self, app_id: str) -> None:
        if app_id in self._warm:
            del self._warm[app_id]
        self._insert_into_hot(app_id)

    def _promote_cool_to_warm(self, app_id: str) -> None:
        if app_id in self._cool:
            del self._cool[app_id]
        self._insert_into_warm(app_id)

    # -- Tier insertions with overflow handling -----------------------------

    def _insert_into_hot(self, app_id: str) -> None:
        if app_id in self._hot:
            self._hot.move_to_end(app_id)
            return
        # Evict LRU from HOT -> WARM if over capacity
        while len(self._hot) >= settings.HOT_TIER_CAPACITY:
            lru_id, _ = self._hot.popitem(last=False)
            self._insert_into_warm(lru_id)
        self._hot[app_id] = "hot"

    def _insert_into_warm(self, app_id: str) -> None:
        if app_id in self._warm:
            self._warm.move_to_end(app_id)
            return
        # Evict LRU from WARM -> COOL if over capacity
        while len(self._warm) >= settings.WARM_TIER_CAPACITY:
            lru_id, _ = self._warm.popitem(last=False)
            self._insert_into_cool(lru_id)
        self._warm[app_id] = "warm"

    def _insert_into_cool(self, app_id: str) -> None:
        if app_id in self._cool:
            self._cool.move_to_end(app_id)
            return
        # Evict LRU from COOL -> COLD if over capacity
        while len(self._cool) >= settings.COOL_TIER_CAPACITY:
            lru_id, _ = self._cool.popitem(last=False)
            self._cold.add(lru_id)
        self._cool[app_id] = "cool"

    # -- Query helpers -------------------------------------------------------

    def get_all_cached_apps(self) -> set:
        """Return all apps currently in PIN, HOT, WARM, or COOL (not COLD)."""
        return self._pin | set(self._hot.keys()) | set(self._warm.keys()) | set(self._cool.keys())

    def get_tier(self, app_id: str) -> str:
        """Return the current tier of an app without updating LRU or stats."""
        if app_id in self._pin:
            return "pin"
        if app_id in self._hot:
            return "hot"
        if app_id in self._warm:
            return "warm"
        if app_id in self._cool:
            return "cool"
        return "cold"

    def is_cached(self, app_id: str) -> bool:
        """Return True if app is in PIN, HOT, WARM, or COOL."""
        return app_id in self.get_all_cached_apps()

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self.total_hits + self.total_misses
        return {
            "pin_size":   len(self._pin),
            "hot_size":   len(self._hot),
            "warm_size":  len(self._warm),
            "cool_size":  len(self._cool),
            "cold_size":  len(self._cold),
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate":   round(self.total_hits / max(1, total), 4),
            "pin_hits":   self.pin_hits,
            "hot_hits":   self.hot_hits,
            "warm_hits":  self.warm_hits,
            "cool_hits":  self.cool_hits,
        }

    def reset(self) -> None:
        """Reset all tiers and counters."""
        self._pin.clear()
        self._hot.clear()
        self._warm.clear()
        self._cool.clear()
        self._cold.clear()
        self._freq.clear()
        self.total_hits = 0
        self.total_misses = 0
        self.pin_hits = 0
        self.hot_hits = 0
        self.warm_hits = 0
        self.cool_hits = 0
