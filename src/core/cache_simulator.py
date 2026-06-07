"""
GraphMind Cache Simulator Utilities
=====================================
Shared utilities for simulating the HOT/WARM/COLD cache architecture
used in both the production prefetch engine and the dashboard simulator.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


# ── Cache tier constants ──────────────────────────────────────────────────────

HOT_CAPACITY = 5       # Max apps in HOT tier (in RAM, 0ms access)
WARM_CAPACITY = 15     # Max apps in WARM tier (pre-loaded, ~200ms access)

HOT_LATENCY_MS = 0.0
WARM_LATENCY_MS = 200.0
COLD_LATENCY_MS = 1800.0


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """Represents an app in the cache."""
    package_name: str
    confidence_score: float
    inserted_at: float = field(default_factory=time.monotonic)
    access_count: int = 0

    def age_seconds(self) -> float:
        return time.monotonic() - self.inserted_at


@dataclass
class CacheStats:
    """Cumulative statistics for a cache simulation run."""
    hot_hits: int = 0
    warm_hits: int = 0
    cold_hits: int = 0
    total_launches: int = 0
    total_latency_saved_ms: float = 0.0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of launches served by HOT or WARM cache."""
        cache_hits = self.hot_hits + self.warm_hits
        return cache_hits / max(self.total_launches, 1)

    @property
    def avg_latency_saved_ms(self) -> float:
        return self.total_latency_saved_ms / max(self.total_launches, 1)

    def __str__(self) -> str:
        return (
            f"CacheStats("
            f"hit_rate={self.hit_rate:.1%}, "
            f"hot={self.hot_hits}, warm={self.warm_hits}, cold={self.cold_hits}, "
            f"avg_latency_saved={self.avg_latency_saved_ms:.0f}ms, "
            f"evictions={self.evictions})"
        )


# ── Cache implementation ──────────────────────────────────────────────────────

class TwoTierCache:
    """
    Simulates the GraphMind two-tier HOT/WARM cache architecture.

    HOT tier: LRU cache of the 5 most recently accessed apps (0ms latency).
    WARM tier: Confidence-based prefetch cache for up to 15 apps (~200ms latency).
    COLD tier: Everything else (SQLite / filesystem, ~1800ms latency).
    """

    def __init__(
        self,
        hot_capacity: int = HOT_CAPACITY,
        warm_capacity: int = WARM_CAPACITY,
    ):
        self.hot_capacity = hot_capacity
        self.warm_capacity = warm_capacity
        self._hot: OrderedDict[str, CacheEntry] = OrderedDict()   # LRU order
        self._warm: dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    # ── Public API ────────────────────────────────────────────────────────────

    def access(self, package_name: str) -> tuple[str, float]:
        """
        Record an app access. Returns (tier_name, latency_ms).

        Promotes apps from WARM to HOT automatically.
        """
        self.stats.total_launches += 1

        if package_name in self._hot:
            # HOT hit — move to end (most recently used)
            entry = self._hot.pop(package_name)
            entry.access_count += 1
            self._hot[package_name] = entry
            self.stats.hot_hits += 1
            self.stats.total_latency_saved_ms += COLD_LATENCY_MS - HOT_LATENCY_MS
            return "HOT", HOT_LATENCY_MS

        elif package_name in self._warm:
            # WARM hit — promote to HOT
            entry = self._warm.pop(package_name)
            entry.access_count += 1
            self._promote_to_hot(entry)
            self.stats.warm_hits += 1
            self.stats.total_latency_saved_ms += COLD_LATENCY_MS - WARM_LATENCY_MS
            return "WARM", WARM_LATENCY_MS

        else:
            # COLD miss — load from storage, add to HOT
            self._promote_to_hot(CacheEntry(package_name, confidence_score=0.0))
            self.stats.cold_hits += 1
            return "COLD", COLD_LATENCY_MS

    def prefetch(self, package_name: str, confidence_score: float) -> bool:
        """
        Add an app to the WARM prefetch cache.

        Returns True if added, False if already in HOT or WARM.
        """
        if package_name in self._hot or package_name in self._warm:
            return False
        if len(self._warm) >= self.warm_capacity:
            self._evict_warm()
        self._warm[package_name] = CacheEntry(package_name, confidence_score)
        return True

    def state(self) -> dict:
        """Return a snapshot of the current cache state."""
        return {
            "hot": [{"app": k, "access_count": v.access_count} for k, v in self._hot.items()],
            "warm": [{"app": k, "score": v.confidence_score} for k, v in self._warm.items()],
            "hot_utilization": f"{len(self._hot)}/{self.hot_capacity}",
            "warm_utilization": f"{len(self._warm)}/{self.warm_capacity}",
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _promote_to_hot(self, entry: CacheEntry) -> None:
        if len(self._hot) >= self.hot_capacity:
            # Evict LRU (oldest) from HOT
            evicted_key, _ = self._hot.popitem(last=False)
            self.stats.evictions += 1
        self._hot[entry.package_name] = entry

    def _evict_warm(self) -> None:
        """Evict the lowest-confidence app from WARM."""
        if not self._warm:
            return
        lowest = min(self._warm, key=lambda k: self._warm[k].confidence_score)
        del self._warm[lowest]
        self.stats.evictions += 1
