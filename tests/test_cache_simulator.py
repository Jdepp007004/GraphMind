"""
Tests for the TwoTierCache simulator.

Validates correctness of HOT/WARM/COLD tier logic, LRU eviction,
confidence-based WARM eviction, and statistics accumulation.
"""

import pytest
from src.core.cache_simulator import (
    TwoTierCache,
    CacheStats,
    HOT_CAPACITY,
    WARM_CAPACITY,
    HOT_LATENCY_MS,
    WARM_LATENCY_MS,
    COLD_LATENCY_MS,
)


class TestHotTier:
    """Tests for HOT tier LRU behaviour."""

    def test_first_access_is_cold(self):
        cache = TwoTierCache()
        tier, latency = cache.access("com.whatsapp")
        assert tier == "COLD"
        assert latency == COLD_LATENCY_MS

    def test_second_access_is_hot(self):
        cache = TwoTierCache()
        cache.access("com.whatsapp")
        tier, latency = cache.access("com.whatsapp")
        assert tier == "HOT"
        assert latency == HOT_LATENCY_MS

    def test_hot_capacity_lru_eviction(self):
        cache = TwoTierCache(hot_capacity=3)
        apps = ["app_a", "app_b", "app_c"]
        for a in apps:
            cache.access(a)  # All go to COLD, then HOT

        # Adding a 4th app should evict app_a (LRU)
        cache.access("app_d")
        tier, _ = cache.access("app_a")
        assert tier == "COLD", "app_a should have been evicted from HOT"

    def test_hot_tier_updates_on_access(self):
        """Accessing an app in HOT should keep it alive (move to MRU)."""
        cache = TwoTierCache(hot_capacity=3)
        for a in ["app_a", "app_b", "app_c"]:
            cache.access(a)
        # Re-access app_a so it becomes MRU
        cache.access("app_a")
        # Now app_b should be LRU — adding app_d should evict app_b
        cache.access("app_d")
        tier_b, _ = cache.access("app_b")
        tier_a, _ = cache.access("app_a")
        assert tier_b == "COLD"
        assert tier_a == "HOT"


class TestWarmTier:
    """Tests for WARM tier prefetch behaviour."""

    def test_prefetch_and_warm_hit(self):
        cache = TwoTierCache()
        cache.prefetch("com.spotify.music", confidence_score=0.45)
        tier, latency = cache.access("com.spotify.music")
        assert tier == "WARM"
        assert latency == WARM_LATENCY_MS

    def test_warm_promotes_to_hot(self):
        cache = TwoTierCache()
        cache.prefetch("com.spotify.music", confidence_score=0.45)
        cache.access("com.spotify.music")  # WARM hit — promotes to HOT
        tier, latency = cache.access("com.spotify.music")
        assert tier == "HOT"

    def test_warm_capacity_evicts_lowest_confidence(self):
        cache = TwoTierCache(warm_capacity=3)
        cache.prefetch("app_high", confidence_score=0.8)
        cache.prefetch("app_med", confidence_score=0.5)
        cache.prefetch("app_low", confidence_score=0.2)
        # Adding a 4th should evict app_low (lowest confidence)
        cache.prefetch("app_new", confidence_score=0.6)
        tier_low, _ = cache.access("app_low")
        tier_high, _ = cache.access("app_high")
        assert tier_low == "COLD", "Lowest confidence app should have been evicted"
        assert tier_high == "WARM", "Highest confidence app should remain"

    def test_prefetch_ignored_if_in_hot(self):
        cache = TwoTierCache()
        cache.access("com.whatsapp")  # Goes to HOT after COLD miss
        cache.access("com.whatsapp")  # Now in HOT
        result = cache.prefetch("com.whatsapp", confidence_score=0.9)
        assert result is False  # Should not add to WARM since it's in HOT

    def test_prefetch_ignored_if_already_in_warm(self):
        cache = TwoTierCache()
        cache.prefetch("com.whatsapp", confidence_score=0.5)
        result = cache.prefetch("com.whatsapp", confidence_score=0.9)
        assert result is False


class TestCacheStats:
    """Tests for statistics accumulation."""

    def test_hit_rate_all_cold(self):
        cache = TwoTierCache()
        for i in range(5):
            cache.access(f"app_{i}")
        assert cache.stats.hot_hits == 0
        assert cache.stats.warm_hits == 0
        assert cache.stats.cold_hits == 5
        assert cache.stats.hit_rate == 0.0

    def test_hit_rate_mixed(self):
        cache = TwoTierCache()
        cache.access("com.whatsapp")        # COLD
        cache.access("com.whatsapp")        # HOT
        cache.prefetch("com.spotify", 0.5)
        cache.access("com.spotify")         # WARM
        # 2 / 3 launches were cache hits
        assert cache.stats.hit_rate == pytest.approx(2 / 3)

    def test_latency_saved_accumulates(self):
        cache = TwoTierCache()
        cache.access("com.whatsapp")   # COLD — no saving
        cache.access("com.whatsapp")   # HOT — saves COLD - HOT = 1800ms
        assert cache.stats.total_latency_saved_ms == pytest.approx(COLD_LATENCY_MS - HOT_LATENCY_MS)

    def test_evictions_counted(self):
        cache = TwoTierCache(hot_capacity=2)
        for app in ["app_a", "app_b", "app_c"]:
            cache.access(app)
        # Adding app_c evicted app_a from HOT
        assert cache.stats.evictions >= 1

    def test_state_snapshot(self):
        cache = TwoTierCache()
        cache.access("com.whatsapp")
        cache.access("com.whatsapp")
        cache.prefetch("com.spotify", 0.6)
        state = cache.state()
        assert "hot" in state
        assert "warm" in state
        assert len(state["hot"]) == 1
        assert len(state["warm"]) == 1
