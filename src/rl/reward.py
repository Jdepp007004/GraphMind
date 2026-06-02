"""
src/rl/reward.py

Computes the RL reward signal from simulation state.
Pure function, no side effects.
"""

import logging
from typing import List

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


def compute_reward(
    cache_hits: int,
    cache_misses: int,
    thrash_events: int,
    battery_consumed: float,
    friction_saved: int,
    step_duration_seconds: float,
    prefetch_fp_count: int = 0
) -> float:
    """
    Compute the scalar reward for one RL step.

    Formula: R = α*cache_hit_rate + β*speed_gain - γ*thrash_rate - δ*battery_cost + ε*friction_saved_rate - ζ*fp_rate

    Where:
        cache_hit_rate = cache_hits / max(1, cache_hits + cache_misses)   [0.0 to 1.0]
        speed_gain = min(1.0, friction_saved / max(1, cache_hits + cache_misses))
        thrash_rate = min(1.0, thrash_events / 10.0)  [normalize: 10 thrashes = max penalty]
        battery_cost = min(1.0, battery_consumed / 5.0)  [normalize: 5% drain = max penalty]
        friction_saved_rate = min(1.0, friction_saved / max(1, cache_hits + cache_misses))
        fp_rate = min(1.0, prefetch_fp_count / 15.0)  [normalize: 15 FPs = max penalty]

    α = REWARD_ALPHA = 1.0
    β = REWARD_BETA = 0.8
    γ = REWARD_GAMMA = 0.5
    δ = REWARD_DELTA = 0.3
    ε = REWARD_EPSILON = 0.4
    ζ = REWARD_ZETA = 0.3

    Returns: float reward value (can be negative if thrash + battery high)
    Logs the breakdown at DEBUG level.
    """
    total = max(1, cache_hits + cache_misses)
    cache_hit_rate = cache_hits / total
    speed_gain = min(1.0, friction_saved / total)
    thrash_rate = min(1.0, thrash_events / 10.0)
    battery_cost = min(1.0, battery_consumed / 5.0)
    friction_saved_rate = min(1.0, friction_saved / total)
    fp_rate = min(1.0, prefetch_fp_count / 15.0)

    reward = (
        settings.REWARD_ALPHA * cache_hit_rate
        + settings.REWARD_BETA * speed_gain
        - settings.REWARD_GAMMA * thrash_rate
        - settings.REWARD_DELTA * battery_cost
        + settings.REWARD_EPSILON * friction_saved_rate
        - settings.REWARD_ZETA * fp_rate
    )
    logger.debug(
        f"Reward breakdown: cache_hit_rate={cache_hit_rate:.3f}, speed_gain={speed_gain:.3f}, "
        f"thrash_rate={thrash_rate:.3f}, battery_cost={battery_cost:.3f}, "
        f"friction_saved_rate={friction_saved_rate:.3f}, fp_rate={fp_rate:.3f} => R={reward:.4f}"
    )
    return float(reward)


def compute_episode_summary(rewards: List[float]) -> dict:
    """
    Compute summary statistics for a training episode.
    rewards: list of per-step reward values.
    Returns: {'mean': float, 'min': float, 'max': float, 'total': float, 'steps': int}
    """
    if not rewards:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "total": 0.0, "steps": 0}
    arr = np.array(rewards, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "total": float(arr.sum()),
        "steps": len(rewards)
    }
