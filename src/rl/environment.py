"""
src/rl/environment.py

Custom Gymnasium environment wrapping the simulator and memory manager.
This is what PPO trains on.
"""

import logging
import os
from collections import deque
from typing import Optional, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import settings
from src.core.event_bus import EventBus, TOPIC_CACHE_HIT, TOPIC_CACHE_MISS
from src.core.memory_manager import MemoryManager
from src.core.graph_engine import BehaviouralGraph
from src.data.event_simulator import EventSimulator
from src.data.context_encoder import ContextEncoder
from src.rl.reward import compute_reward

logger = logging.getLogger(__name__)

OBS_DIM = 35 + settings.HOT_TIER_CAPACITY + 3  # 35 + 30 + 3 = 68


class GraphMindEnv(gym.Env):
    """
    Custom Gymnasium environment for RL training.

    Observation space: Box(shape=(35 + HOT_TIER_CAPACITY + 3,), dtype=float32)
        = context_embedding(35) + hot_tier_occupancy(30) + [battery, time_bucket_norm, cache_hit_rate_recent]
        Total: 68 dimensions

    Action space: Discrete(HOT_TIER_CAPACITY + 1)
        Actions 0 to 28: promote node at hot_tier_index to front (signal to prioritize)
        Action 29: 'no-op / run prune cycle'
        Action 30: 'emergency: demote bottom half of HOT to WARM'

    Episode: one simulated day (all events for one day for one user)
    """

    metadata = {"render_modes": []}

    def __init__(self, user_id: str) -> None:
        """
        Initialize the environment for a specific user.
        Create EventSimulator(user_id).
        Create BehaviouralGraph(user_id) and MemoryManager(user_id, graph).
        Load graph from BASE_GRAPHS_DIR/{user_id}_base.pkl if exists, else start empty.
        Set observation_space and action_space per spec above.
        Initialize counters: self.cache_hits=0, self.cache_misses=0, self.thrash_events=0, self.battery_start=100.0
        Subscribe to TOPIC_CACHE_HIT to increment self.cache_hits.
        Subscribe to TOPIC_CACHE_MISS to increment self.cache_misses.
        """
        super().__init__()
        self.user_id = user_id
        self.simulator = EventSimulator(user_id)
        self.graph = BehaviouralGraph(user_id)
        self.memory_manager = MemoryManager(user_id, self.graph)
        self.encoder = ContextEncoder()

        # Load base graph if exists
        base_path = os.path.join(settings.BASE_GRAPHS_DIR, f"{user_id}_base.pkl")
        if os.path.exists(base_path):
            try:
                self.graph.load_from_disk(base_path)
                logger.debug(f"Loaded base graph for {user_id}")
            except Exception as e:
                logger.warning(f"Could not load base graph: {e}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(settings.HOT_TIER_CAPACITY + 1)  # 31

        # Counters
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.thrash_events: int = 0
        self.battery_start: float = 100.0
        self._current_day: int = -1
        self._step_count: int = 0
        self._last_obs: np.ndarray = np.zeros(OBS_DIM, dtype=np.float32)
        self._last_event: Optional[dict] = None
        self._recent_hits: deque = deque(maxlen=50)
        self._prev_hot_set: set = set()

        # Subscribe for cache tracking
        bus = EventBus.get_instance()
        bus.subscribe(TOPIC_CACHE_HIT, self._on_cache_hit)
        bus.subscribe(TOPIC_CACHE_MISS, self._on_cache_miss)

    def _on_cache_hit(self, payload: dict) -> None:
        """Callback: increment cache hit counter."""
        if payload.get("user_id") == self.user_id:
            self.cache_hits += 1
            self._recent_hits.append(1)

    def _on_cache_miss(self, payload: dict) -> None:
        """Callback: increment cache miss counter."""
        if payload.get("user_id") == self.user_id:
            self.cache_misses += 1
            self._recent_hits.append(0)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset to start of a new day (or day 0 if first call).
        Advance to the next unprocessed day.
        Reset step counters.
        Returns (initial_observation, {})
        """
        super().reset(seed=seed)
        self._current_day += 1
        # If we've run out of days, cycle back
        if self._current_day >= settings.SIMULATION_DAYS:
            self.simulator.reset()
            self._current_day = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.thrash_events = 0
        self._step_count = 0
        self._last_event = None
        self._recent_hits.clear()
        self._prev_hot_set = set(self.memory_manager.get_hot_node_ids())
        obs = self._get_observation()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step: publish one event via simulator.step(), apply the action,
        compute reward.

        Action interpretation:
            0-28: promote corresponding HOT node index to priority front
            29: call graph.prune_weak_edges()
            30: call memory_manager to demote bottom 15 HOT nodes to WARM

        Returns: (observation, reward, terminated, truncated, info)
            terminated = True when day's events are exhausted
            truncated = False always
            info = {'cache_hits': int, 'cache_misses': int, 'day': int}
        """
        # Advance one event in the simulator
        day_events_before = self._step_count
        result = None
        target_day = self._current_day
        # Step through events until we get one for the current day or run out
        while self.simulator.current_event_index < len(self.simulator.events):
            evt = self.simulator.events[self.simulator.current_event_index]
            if int(evt.get("day", 0)) != target_day:
                break
            result = self.simulator.step()
            if result:
                self._last_event = result
                break

        terminated = (result is None or
                      self.simulator.current_event_index >= len(self.simulator.events) or
                      (self.simulator.current_event_index < len(self.simulator.events) and
                       int(self.simulator.events[self.simulator.current_event_index].get("day", 0)) != target_day))

        # Apply action
        battery_consumed = 0.0
        hot_ids = self.memory_manager.get_hot_node_ids()
        if action < 29:
            # Promote node at index 'action' in HOT list
            if action < len(hot_ids):
                self.memory_manager.promote_to_hot(hot_ids[action])
                battery_consumed = 0.1
        elif action == 29:
            # Prune weak edges
            self.graph.prune_weak_edges()
        elif action == 30:
            # Emergency demote bottom half
            n_demote = max(1, len(hot_ids) // 2)
            for nid in hot_ids[-n_demote:]:
                self.memory_manager.demote_from_hot(nid)
                self.thrash_events += 1

        # Thrash detection: nodes that were in HOT and are now missing
        current_hot = set(self.memory_manager.get_hot_node_ids())
        lost = self._prev_hot_set - current_hot
        self.thrash_events += len(lost)
        self._prev_hot_set = current_hot

        # Battery consumed (estimate)
        battery = float(self._last_event.get("battery", 100.0)) if self._last_event else 100.0
        battery_drain = max(0.0, self.battery_start - battery)
        friction_saved = self.cache_hits

        reward = compute_reward(
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            thrash_events=self.thrash_events,
            battery_consumed=battery_consumed,
            friction_saved=friction_saved,
            step_duration_seconds=1.0
        )
        self._step_count += 1
        obs = self._get_observation()
        info = {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "day": self._current_day
        }
        return obs, float(reward), bool(terminated), False, info

    def _get_observation(self) -> np.ndarray:
        """
        PRIVATE. Build the 68-dim observation vector from current state.
        Use zeros for context embedding if no event has been published yet.
        Cache hit rate = cache_hits / max(1, cache_hits + cache_misses) for last 50 steps.
        """
        # Context embedding (35 dims) — derived from last event
        if self._last_event:
            context_35 = self.encoder.encode(self._last_event)[:35]
        else:
            context_35 = np.zeros(35, dtype=np.float32)

        # HOT tier occupancy (30 dims)
        hot_ids = self.memory_manager.get_hot_node_ids()
        hot_occ = np.zeros(settings.HOT_TIER_CAPACITY, dtype=np.float32)
        for i in range(min(len(hot_ids), settings.HOT_TIER_CAPACITY)):
            hot_occ[i] = 1.0

        # 3 state signals
        battery = float(self._last_event.get("battery", 100.0)) if self._last_event else 100.0
        time_bucket = float(self._last_event.get("time_bucket", 0)) if self._last_event else 0.0
        recent_total = len(self._recent_hits)
        recent_hit_rate = sum(self._recent_hits) / max(1, recent_total)
        state_signals = np.array([battery / 100.0, time_bucket / 47.0, recent_hit_rate],
                                  dtype=np.float32)

        obs = np.concatenate([context_35, hot_occ, state_signals]).astype(np.float32)
        self._last_obs = obs
        return obs

    def render(self) -> None:
        """No-op. Required by Gymnasium interface."""
        pass

    def close(self) -> None:
        """Cleanup. Unsubscribe EventBus callbacks."""
        bus = EventBus.get_instance()
        bus.unsubscribe(TOPIC_CACHE_HIT, self._on_cache_hit)
        bus.unsubscribe(TOPIC_CACHE_MISS, self._on_cache_miss)
