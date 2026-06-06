"""
src/rl/reward_v2.py

Multi-component reward function for the GraphMind ResourceAllocationPolicy (RL V2).

All weights are defined in config/settings.py so they can be tuned without
touching this file. This is critical for ablation studies.

Reward formula:
  R = W_HIT_RATE    * cache_hit_rate
    + W_LATENCY     * (latency_saved_ms / MAX_LATENCY_SAVED_MS)
    - W_BATTERY     * battery_overhead_pct_normalised
    - W_FALSE_PREF  * false_prefetch_rate_normalised
    - W_THRASH      * thrash_rate_normalised

Where:
  cache_hit_rate              = hits / (hits + misses) ∈ [0, 1]
  latency_saved_ms            = cold_start_ms - hot/warm_start_ms (from literature)
  latency_saved_normalised    = latency_saved_ms / MAX_LATENCY_SAVED_MS ∈ [0, 1]
  battery_overhead_pct_norm   = battery_overhead_pct / MAX_BATTERY_OVERHEAD_PCT ∈ [0, 1]
  false_prefetch_rate_norm    = false_prefetch_count / max(1, prefetch_total) ∈ [0, 1]
  thrash_rate_normalised      = thrash_count / MAX_THRASH_PER_STEP ∈ [0, 1]

All weights:
  REWARD_V2_HIT_RATE_WEIGHT          = 2.0  (primary objective, highest weight)
  REWARD_V2_LATENCY_SAVED_WEIGHT     = 1.0
  REWARD_V2_BATTERY_WEIGHT           = 0.5
  REWARD_V2_FALSE_PREFETCH_WEIGHT    = 0.8
  REWARD_V2_THRASH_WEIGHT            = 1.2  (strongest penalty)

Maximum possible reward per step: W_HIT + W_LATENCY = 3.0
Minimum possible reward per step: -(W_BATTERY + W_FALSE_PREF + W_THRASH) = -2.5
"""

import logging
from typing import List

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class RewardV2:
    """
    Stateful reward computer for RL V2.

    Tracks running averages of each reward component for logging and
    episode summaries. All computation methods are pure (deterministic
    given inputs) — the state is only for diagnostics.
    """

    def __init__(self) -> None:
        self._step_rewards: List[float] = []
        self._component_history: List[dict] = []

    def compute(
        self,
        hit_rate: float,
        latency_saved_ms: float,
        battery_overhead_pct: float,
        false_prefetch_count: int,
        thrash_count: int,
        prefetch_total: int = 1,
    ) -> float:
        """
        Compute the scalar reward for one RL step.

        Args:
            hit_rate:             Cache hit rate for this step ∈ [0, 1].
            latency_saved_ms:     Latency saved vs cold start in milliseconds.
            battery_overhead_pct: Battery overhead from prefetch actions ∈ [0, ∞).
            false_prefetch_count: Number of prefetched items not subsequently accessed.
            thrash_count:         Number of HOT evictions followed by re-access.
            prefetch_total:       Total prefetch candidates (denominator for FP rate).

        Returns:
            Scalar reward value. Positive = good, negative = bad.
        """
        # Normalise all components to [0, 1] range
        latency_norm = min(
            1.0,
            max(0.0, latency_saved_ms) / settings.REWARD_V2_MAX_LATENCY_SAVED_MS
        )
        battery_norm = min(
            1.0,
            max(0.0, battery_overhead_pct) / settings.REWARD_V2_MAX_BATTERY_OVERHEAD_PCT
        )
        fp_rate = false_prefetch_count / max(1, prefetch_total)
        fp_norm = min(1.0, max(0.0, fp_rate))
        thrash_norm = min(
            1.0,
            max(0, thrash_count) / settings.REWARD_V2_MAX_THRASH_PER_STEP
        )

        # Apply weights
        reward = (
            settings.REWARD_V2_HIT_RATE_WEIGHT * hit_rate
            + settings.REWARD_V2_LATENCY_SAVED_WEIGHT * latency_norm
            - settings.REWARD_V2_BATTERY_WEIGHT * battery_norm
            - settings.REWARD_V2_FALSE_PREFETCH_WEIGHT * fp_norm
            - settings.REWARD_V2_THRASH_WEIGHT * thrash_norm
        )

        components = {
            "hit_rate": round(hit_rate, 4),
            "latency_norm": round(latency_norm, 4),
            "battery_norm": round(battery_norm, 4),
            "fp_norm": round(fp_norm, 4),
            "thrash_norm": round(thrash_norm, 4),
            "reward": round(float(reward), 4),
        }
        self._step_rewards.append(float(reward))
        self._component_history.append(components)

        logger.debug(
            "RewardV2: hit_rate=%.3f latency=%.3f battery=%.3f "
            "fp=%.3f thrash=%.3f → R=%.4f",
            hit_rate, latency_norm, battery_norm, fp_norm, thrash_norm, reward
        )
        return float(reward)

    def reset(self) -> None:
        """Clear episode history."""
        self._step_rewards.clear()
        self._component_history.clear()

    def episode_summary(self) -> dict:
        """
        Return summary statistics for the current episode.

        Returns:
            dict with mean, median, std, min, max, total reward,
            and per-component averages.
        """
        if not self._step_rewards:
            return {
                "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0, "total": 0.0, "steps": 0,
                "components": {},
            }

        arr = np.array(self._step_rewards, dtype=np.float64)
        summary: dict = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "total": float(arr.sum()),
            "steps": len(arr),
            "components": {},
        }

        # Average each component across steps
        if self._component_history:
            for key in self._component_history[0]:
                if key != "reward":
                    vals = [c[key] for c in self._component_history]
                    summary["components"][key] = round(float(np.mean(vals)), 4)

        return summary


def compute_reward_v2(
    hit_rate: float,
    latency_saved_ms: float,
    battery_overhead_pct: float,
    false_prefetch_count: int,
    thrash_count: int,
    prefetch_total: int = 1,
) -> float:
    """
    Stateless convenience wrapper for single-step reward computation.

    Uses the same formula as RewardV2.compute() but without tracking state.
    Useful for unit tests and one-off calculations.
    """
    return RewardV2().compute(
        hit_rate=hit_rate,
        latency_saved_ms=latency_saved_ms,
        battery_overhead_pct=battery_overhead_pct,
        false_prefetch_count=false_prefetch_count,
        thrash_count=thrash_count,
        prefetch_total=prefetch_total,
    )
