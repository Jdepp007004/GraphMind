"""
src/rl/environment_v2.py

GraphMind RL Environment V2 — ResourceAllocationPolicy.

This environment explicitly models the RL agent as a RESOURCE ALLOCATOR,
not an app selector. The distinction is critical for training stability
and interpretability:

  Graph predicts candidates (BehaviouralGraph.get_top_k_next_nodes)
  RL decides:
    [0] HOT budget  — how many candidates to promote to HOT tier
    [1] WARM budget — how many candidates to hold in WARM tier
    [2] Prefetch aggressiveness — confidence threshold level for prefetch

This formulation avoids the combinatorial action space problem: PPO does
not need to select individual apps from a vocabulary of hundreds. Instead
it learns the right resource allocation policy for the current context.

Action Space: MultiDiscrete([5, 5, 5])
  Dimension 0 (hot_budget):  index into RL_V2_HOT_CAPACITY_OPTIONS  = [1, 5, 10, 20, 30]
  Dimension 1 (warm_budget): index into RL_V2_WARM_CAPACITY_OPTIONS = [10, 30, 50, 100, 150]
  Dimension 2 (conf_level):  index into RL_V2_CONF_THRESHOLD_OPTIONS = [0.5, 0.6, 0.7, 0.8, 0.9]

Observation Space: Box(shape=(RL_V2_OBS_DIM,), dtype=float32)
  [0:50]   current app one-hot (app vocabulary index)
  [50:100] previous app one-hot
  [100]    time_bucket normalised to [0,1] (bucket/47)
  [101]    day_of_week normalised to [0,1] (0=Mon, 6=Sun → 0/6)
  [102]    HOT occupancy ratio (current HOT count / HOT_TIER_CAPACITY)
  [103]    WARM occupancy ratio (current WARM count / WARM_TIER_CAPACITY)
  [104:109] recent cache hit/miss binary history (last 5 steps)
  Total: 109 dimensions

NOTE: Battery level is deliberately excluded from the observation space.
  UbiqLog does not contain battery measurements. Using battery=constant
  would add no information and pollute the feature space.
  Replacement: day_of_week provides complementary temporal context.

Episode: one full pass through the test split events for one user.
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import settings
from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED, TOPIC_CACHE_HIT, TOPIC_CACHE_MISS
from src.core.graph_engine import BehaviouralGraph
from src.core.memory_manager import MemoryManager
from src.prefetch.confidence_prefetch import ConfidencePrefetch
from src.rl.reward_v2 import RewardV2

logger = logging.getLogger(__name__)

# Verify observation dimension constant
_EXPECTED_OBS_DIM = (
    settings.RL_V2_APP_VOCAB_SIZE   # current app OHE
    + settings.RL_V2_APP_VOCAB_SIZE  # previous app OHE
    + 1   # battery
    + 1   # time_bucket_norm
    + 1   # HOT occupancy ratio
    + 1   # WARM occupancy ratio
    + settings.RL_V2_HIT_HISTORY_LEN  # cache hit history
)
assert _EXPECTED_OBS_DIM == settings.RL_V2_OBS_DIM, (
    f"OBS_DIM mismatch: computed {_EXPECTED_OBS_DIM}, settings has {settings.RL_V2_OBS_DIM}"
)


class GraphMindEnvV2(gym.Env):
    """
    Gymnasium environment for training a RL ResourceAllocationPolicy.

    The agent learns WHEN to allocate more or fewer resources to HOT/WARM
    tiers based on the current context. The graph always provides the
    candidate app list — the agent only decides the resource budget.

    This design makes the RL problem tractable:
    - Small, structured action space (MultiDiscrete [5,5,5])
    - Observation fully observable from runtime state
    - Reward directly reflects cache quality and resource cost
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        user_id: str,
        events: Optional[List[dict]] = None,
    ) -> None:
        """
        Args:
            user_id: User identifier (used for graph and memory manager).
            events:  Optional pre-loaded event list. If None, the environment
                     expects events to be set via set_events() before reset().
        """
        super().__init__()
        self.user_id = user_id
        self._events: List[dict] = events or []
        self._event_index: int = 0
        self._current_event: Optional[dict] = None

        # Core GraphMind components
        EventBus.get_instance().clear_all()
        self.graph = BehaviouralGraph(user_id)
        self.memory_manager = MemoryManager(user_id, self.graph)
        self.confidence_prefetch = ConfidencePrefetch(self.graph)
        self.reward_fn = RewardV2()

        # App vocabulary for OHE (built incrementally from observed events)
        self._app_vocab: Dict[str, int] = {}
        self._prev_app: Optional[str] = None
        self._current_app: Optional[str] = None

        # Episode state
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._thrash_events: int = 0
        self._false_prefetches: int = 0
        self._prev_hot_set: set = set()
        self._hit_history: deque = deque(
            [0.0] * settings.RL_V2_HIT_HISTORY_LEN,
            maxlen=settings.RL_V2_HIT_HISTORY_LEN
        )
        self._episode_reward: float = 0.0

        # Subscribe to EventBus for cache tracking
        bus = EventBus.get_instance()
        bus.subscribe(TOPIC_CACHE_HIT, self._on_cache_hit)
        bus.subscribe(TOPIC_CACHE_MISS, self._on_cache_miss)

        # Action space: MultiDiscrete([N_HOT, N_WARM, N_CONF])
        self.action_space = spaces.MultiDiscrete([
            settings.RL_V2_N_HOT_LEVELS,
            settings.RL_V2_N_WARM_LEVELS,
            settings.RL_V2_N_CONF_LEVELS,
        ])

        # Observation space: Box(109,)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(settings.RL_V2_OBS_DIM,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def set_events(self, events: List[dict]) -> None:
        """Set the event stream for the next episode. Must call before reset()."""
        self._events = events

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset to the beginning of the event stream.

        Returns:
            (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        self._event_index = 0
        self._current_event = None
        self._cache_hits = 0
        self._cache_misses = 0
        self._thrash_events = 0
        self._false_prefetches = 0
        self._prev_hot_set = set(self.memory_manager.get_hot_node_ids())
        self._hit_history = deque(
            [0.0] * settings.RL_V2_HIT_HISTORY_LEN,
            maxlen=settings.RL_V2_HIT_HISTORY_LEN
        )
        self._prev_app = None
        self._current_app = None
        self._episode_reward = 0.0
        self.confidence_prefetch.reset()
        self.reward_fn.reset()

        obs = self._build_observation()
        return obs, {"event_index": 0, "total_events": len(self._events)}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one RL step.

        Step logic:
          1. Decode action into (hot_n, warm_n, conf_threshold).
          2. Advance one event from the stream.
          3. Publish TOPIC_APP_LAUNCHED to update graph and memory.
          4. Apply the resource allocation action:
             - Run confidence prefetch with the selected threshold.
             - Promote top hot_n candidates to HOT.
             - Keep top warm_n candidates in WARM.
          5. Detect cache hit/miss and thrash.
          6. Compute reward via RewardV2.
          7. Return (observation, reward, terminated, truncated, info).

        Args:
            action: np.ndarray of shape (3,) from MultiDiscrete space.
                    action[0] = HOT budget index
                    action[1] = WARM budget index
                    action[2] = confidence threshold index

        Returns:
            Standard Gymnasium 5-tuple.
        """
        # 1. Decode action
        hot_n = settings.RL_V2_HOT_CAPACITY_OPTIONS[int(action[0])]
        warm_n = settings.RL_V2_WARM_CAPACITY_OPTIONS[int(action[1])]
        conf_threshold = settings.RL_V2_CONF_THRESHOLD_OPTIONS[int(action[2])]

        # 2. Advance event
        if self._event_index >= len(self._events):
            obs = self._build_observation()
            return obs, 0.0, True, False, self._build_info()

        event = self._events[self._event_index]
        self._event_index += 1
        self._current_event = event

        # Update vocabulary
        app_id = event.get("app_id", "unknown")
        if app_id not in self._app_vocab:
            if len(self._app_vocab) < settings.RL_V2_APP_VOCAB_SIZE:
                self._app_vocab[app_id] = len(self._app_vocab)

        # 3. Record pre-event state for hit detection
        before_hot = set(self.memory_manager.get_hot_node_ids())
        before_warm = set(self.memory_manager.get_warm_node_ids())

        # Publish event to update graph
        payload = self._event_to_payload(event)
        EventBus.get_instance().publish(TOPIC_APP_LAUNCHED, payload)

        # 4. Apply resource allocation
        time_bucket  = int(event.get("time_bucket", 0))
        day_of_week  = int(event.get("day_of_week", 0))

        # Update confidence scorer state
        self.confidence_prefetch.observe_event(event)

        # Find current node (no battery dimension)
        current_node_id = self._find_node_id(app_id, time_bucket, day_of_week)

        # Run confidence prefetch with agent-selected threshold
        self.confidence_prefetch._threshold = conf_threshold
        if current_node_id:
            prefetch_ids, scored = self.confidence_prefetch.prefetch(
                current_node_id, time_bucket, 0  # battery removed — UbiqLog has no battery
            )
        else:
            prefetch_ids, scored = [], []

        # Promote top hot_n to HOT tier
        promoted_count = 0
        for node_id in prefetch_ids:
            if promoted_count >= hot_n:
                break
            try:
                self.memory_manager.promote_to_hot(node_id)
                promoted_count += 1
            except Exception:
                pass

        # Rebuild WARM from remaining candidates up to warm_n
        warm_candidates = prefetch_ids[promoted_count:promoted_count + warm_n]
        try:
            if warm_candidates:
                self.memory_manager.rebuild_warm_from_graph(warm_candidates)
        except Exception:
            pass

        # 5. Detect cache hit/miss
        after_hot = set(self.memory_manager.get_hot_node_ids())
        after_warm = set(self.memory_manager.get_warm_node_ids())

        was_hot_hit = current_node_id is not None and current_node_id in before_hot
        was_warm_hit = current_node_id is not None and current_node_id in before_warm
        is_hit = was_hot_hit or was_warm_hit

        if is_hit:
            self._cache_hits += 1
            self._hit_history.append(1.0)
        else:
            self._cache_misses += 1
            self._hit_history.append(0.0)

        # Thrash detection: HOT nodes that disappeared without being accessed
        evicted = self._prev_hot_set - after_hot
        self._thrash_events += len(evicted)
        self._prev_hot_set = after_hot

        # False prefetch count: prefetched nodes never accessed
        if self._event_index > 1 and prefetch_ids:
            fp = max(0, len(prefetch_ids) - (1 if is_hit else 0))
            self._false_prefetches += fp

        # Prefetch overhead proxy: larger budgets cost more IO/memory pressure
        prefetch_overhead = (hot_n + warm_n * 0.1) * 0.002

        # 6. Compute reward
        hit_rate = self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        latency_saved_ms = self._estimate_latency_saved(was_hot_hit, was_warm_hit, app_id)
        reward = self.reward_fn.compute(
            hit_rate=hit_rate,
            latency_saved_ms=latency_saved_ms,
            battery_overhead_pct=prefetch_overhead,
            false_prefetch_count=self._false_prefetches,
            thrash_count=self._thrash_events,
        )
        self._episode_reward += reward

        # Update prev_app tracking
        self._prev_app = self._current_app
        self._current_app = app_id

        terminated = self._event_index >= len(self._events)
        obs = self._build_observation()
        info = self._build_info()
        info.update({
            "action_hot_n": hot_n,
            "action_warm_n": warm_n,
            "action_conf_threshold": conf_threshold,
            "is_hit": is_hit,
            "prefetch_count": len(prefetch_ids),
        })
        return obs, float(reward), terminated, False, info

    def render(self) -> None:
        """No-op. Required by Gymnasium interface."""

    def close(self) -> None:
        """Unsubscribe EventBus callbacks."""
        bus = EventBus.get_instance()
        bus.unsubscribe(TOPIC_CACHE_HIT, self._on_cache_hit)
        bus.unsubscribe(TOPIC_CACHE_MISS, self._on_cache_miss)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_cache_hit(self, payload: dict) -> None:
        if payload.get("user_id") == self.user_id:
            self._cache_hits += 1

    def _on_cache_miss(self, payload: dict) -> None:
        if payload.get("user_id") == self.user_id:
            self._cache_misses += 1

    def _build_observation(self) -> np.ndarray:
        """
        Construct the 109-dimensional observation vector.

        Components:
          [0:50]    current app one-hot
          [50:100]  previous app one-hot
          [100]     time_bucket / 47
          [101]     day_of_week / 6  (0=Monday, 6=Sunday)
          [102]     HOT occupancy ratio
          [103]     WARM occupancy ratio
          [104:109] hit history (binary, float)

        Note: Battery deliberately excluded — not available in UbiqLog dataset.
        """
        obs = np.zeros(settings.RL_V2_OBS_DIM, dtype=np.float32)

        vocab_size = settings.RL_V2_APP_VOCAB_SIZE

        if self._current_app and self._current_app in self._app_vocab:
            obs[self._app_vocab[self._current_app]] = 1.0

        if self._prev_app and self._prev_app in self._app_vocab:
            obs[vocab_size + self._app_vocab[self._prev_app]] = 1.0

        if self._current_event:
            time_bucket = int(self._current_event.get("time_bucket", 0))
            day_of_week = int(self._current_event.get("day_of_week", 0))
            obs[vocab_size * 2]     = time_bucket / 47.0
            obs[vocab_size * 2 + 1] = day_of_week / 6.0

        hot_count  = len(self.memory_manager.get_hot_node_ids())
        warm_count = len(self.memory_manager.get_warm_node_ids())
        obs[vocab_size * 2 + 2] = hot_count  / max(1, settings.HOT_TIER_CAPACITY)
        obs[vocab_size * 2 + 3] = warm_count / max(1, settings.WARM_TIER_CAPACITY)

        hit_start = vocab_size * 2 + 4
        for i, h in enumerate(self._hit_history):
            obs[hit_start + i] = float(h)

        return obs

    def _build_info(self) -> dict:
        """Build info dict for current step."""
        total = max(1, self._cache_hits + self._cache_misses)
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / total,
            "thrash_events": self._thrash_events,
            "episode_reward": self._episode_reward,
            "event_index": self._event_index,
            "total_events": len(self._events),
        }

    def _event_to_payload(self, event: dict) -> dict:
        """Convert a GraphMindEvent dict to an EventBus payload."""
        time_bucket = int(event.get("time_bucket", 0))
        return {
            "timestamp":        float(event.get("timestamp", 0.0)),
            "user_id":          self.user_id,
            "app_id":           event.get("app_id", "unknown"),
            "category":         event.get("category", "utility"),
            "time_of_day_bucket": time_bucket,
            "time_bucket":      time_bucket,
            "day":              int(event.get("day", 0)),
            "day_of_week":      int(event.get("day_of_week", 0)),
            "weekend":          bool(event.get("weekend", False)),
            "headphones":       bool(event.get("headphones", False)),
            "calendar_event_in_mins": event.get("calendar_event_in_mins"),
        }

    def _find_node_id(
        self, app_id: str, time_bucket: int, day_of_week: int = 0
    ) -> Optional[str]:
        """Find the graph node matching the given app/context."""
        for b_bucket in range(5):
            nid = self.graph._node_lookup.get((app_id, time_bucket, b_bucket))
            if nid is not None:
                return nid
        return None

    def _estimate_latency_saved_ms(
        self, was_hot_hit: bool, was_warm_hit: bool, app_id: str
    ) -> float:
        """
        Estimate latency saved vs cold start for this cache hit.

        Uses literature values from settings.LATENCY_COLD_START_MS and
        settings.LATENCY_HOT_START_MS / settings.LATENCY_WARM_START_MS.
        """
        cold_ms = settings.LATENCY_COLD_START_MS.get(
            app_id, settings.LATENCY_COLD_START_MS["default"]
        )
        if was_hot_hit:
            hot_ms = settings.LATENCY_HOT_START_MS.get(
                app_id, settings.LATENCY_HOT_START_MS["default"]
            )
            return max(0.0, cold_ms - hot_ms)
        if was_warm_hit:
            warm_ms = settings.LATENCY_WARM_START_MS.get(
                app_id, settings.LATENCY_WARM_START_MS["default"]
            )
            return max(0.0, cold_ms - warm_ms)
        return 0.0

    def _estimate_latency_saved(
        self, was_hot_hit: bool, was_warm_hit: bool, app_id: str
    ) -> float:
        """Alias with consistent naming for reward computation."""
        return self._estimate_latency_saved_ms(was_hot_hit, was_warm_hit, app_id)
