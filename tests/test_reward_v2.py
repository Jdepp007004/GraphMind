"""tests/test_reward_v2.py — RewardV2 formula correctness."""
import pytest
import math
from src.rl.reward_v2 import RewardV2, compute_reward_v2
from config import settings


def test_perfect_hit_rate_gives_positive_reward():
    r = compute_reward_v2(
        hit_rate=1.0,
        latency_saved_ms=500.0,
        battery_overhead_pct=0.0,
        false_prefetch_count=0,
        thrash_count=0,
    )
    assert r > 0


def test_zero_hit_rate_with_penalties_gives_negative_reward():
    r = compute_reward_v2(
        hit_rate=0.0,
        latency_saved_ms=0.0,
        battery_overhead_pct=5.0,
        false_prefetch_count=10,
        thrash_count=10,
    )
    assert r < 0


def test_reward_formula_components():
    """Verify formula: R = W_HIT*hit + W_LAT*(lat/max_lat) - penalties."""
    hit_rate = 0.8
    lat_ms = 400.0
    battery_pct = 2.5
    fp = 5
    thrash = 5
    prefetch_total = 10

    expected = (
        settings.REWARD_V2_HIT_RATE_WEIGHT * hit_rate
        + settings.REWARD_V2_LATENCY_SAVED_WEIGHT * (lat_ms / settings.REWARD_V2_MAX_LATENCY_SAVED_MS)
        - settings.REWARD_V2_BATTERY_WEIGHT * min(1.0, battery_pct / settings.REWARD_V2_MAX_BATTERY_OVERHEAD_PCT)
        - settings.REWARD_V2_FALSE_PREFETCH_WEIGHT * min(1.0, fp / prefetch_total)
        - settings.REWARD_V2_THRASH_WEIGHT * min(1.0, thrash / settings.REWARD_V2_MAX_THRASH_PER_STEP)
    )
    actual = compute_reward_v2(
        hit_rate=hit_rate,
        latency_saved_ms=lat_ms,
        battery_overhead_pct=battery_pct,
        false_prefetch_count=fp,
        thrash_count=thrash,
        prefetch_total=prefetch_total,
    )
    assert abs(actual - expected) < 1e-9


def test_latency_normalised_to_max():
    """Latency beyond MAX_LATENCY_SAVED_MS should be clamped to 1.0 contribution."""
    r_within = compute_reward_v2(
        hit_rate=0.5, latency_saved_ms=settings.REWARD_V2_MAX_LATENCY_SAVED_MS,
        battery_overhead_pct=0, false_prefetch_count=0, thrash_count=0
    )
    r_over = compute_reward_v2(
        hit_rate=0.5, latency_saved_ms=settings.REWARD_V2_MAX_LATENCY_SAVED_MS * 2,
        battery_overhead_pct=0, false_prefetch_count=0, thrash_count=0
    )
    assert abs(r_within - r_over) < 1e-9


def test_battery_penalty_clamped():
    """Battery overhead beyond MAX should be clamped."""
    r_max = compute_reward_v2(
        hit_rate=0, latency_saved_ms=0,
        battery_overhead_pct=settings.REWARD_V2_MAX_BATTERY_OVERHEAD_PCT,
        false_prefetch_count=0, thrash_count=0
    )
    r_over = compute_reward_v2(
        hit_rate=0, latency_saved_ms=0,
        battery_overhead_pct=settings.REWARD_V2_MAX_BATTERY_OVERHEAD_PCT * 5,
        false_prefetch_count=0, thrash_count=0
    )
    assert abs(r_max - r_over) < 1e-9


def test_reward_v2_stateful_episode_tracking():
    rw = RewardV2()
    rw.compute(hit_rate=0.8, latency_saved_ms=300, battery_overhead_pct=0,
               false_prefetch_count=0, thrash_count=0)
    rw.compute(hit_rate=0.5, latency_saved_ms=200, battery_overhead_pct=1,
               false_prefetch_count=1, thrash_count=0)
    assert len(rw._step_rewards) == 2


def test_reward_v2_episode_summary_keys():
    rw = RewardV2()
    for i in range(10):
        rw.compute(hit_rate=0.7, latency_saved_ms=300, battery_overhead_pct=0.5,
                   false_prefetch_count=2, thrash_count=1)
    summary = rw.episode_summary()
    assert "mean" in summary
    assert "median" in summary
    assert "std" in summary
    assert "total" in summary
    assert "steps" in summary
    assert summary["steps"] == 10


def test_reward_v2_reset_clears_history():
    rw = RewardV2()
    rw.compute(hit_rate=1.0, latency_saved_ms=500, battery_overhead_pct=0,
               false_prefetch_count=0, thrash_count=0)
    rw.reset()
    assert len(rw._step_rewards) == 0
    summary = rw.episode_summary()
    assert summary["steps"] == 0


def test_zero_prefetch_total_uses_default():
    """prefetch_total=0 should not divide by zero."""
    r = compute_reward_v2(
        hit_rate=0.5, latency_saved_ms=0, battery_overhead_pct=0,
        false_prefetch_count=0, thrash_count=0, prefetch_total=0
    )
    assert math.isfinite(r)
