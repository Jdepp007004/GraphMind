"""
src/benchmarks/baselines_v2.py

Ten research-grade baseline policies for GraphMind v2 evaluation.
All policies extend the BaselinePolicy ABC from baselines.py.
"""

import logging
import math
import random
from abc import ABC
from collections import Counter, defaultdict, OrderedDict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from config import settings
from src.benchmarks.baselines import BaselinePolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. RandomPolicy
# ---------------------------------------------------------------------------

class RandomPolicy(BaselinePolicy):
    """
    Predicts random apps from the observed vocabulary.
    """

    def __init__(self, seed: int = settings.RANDOM_SEED, top_k: int = settings.PREFETCH_TOP_K) -> None:
        self._vocab: List[str] = []
        self._rng = random.Random(seed)
        self.top_k = top_k

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return randomly sampled app IDs from the observed vocabulary."""
        if not self._vocab:
            return []
        k = min(self.top_k, len(self._vocab))
        return self._rng.sample(self._vocab, k)

    def update(self, event: dict) -> None:
        """Add new app IDs to vocabulary."""
        app_id = event.get("app_id", "")
        if app_id and app_id not in self._vocab:
            self._vocab.append(app_id)

    def reset(self) -> None:
        self._vocab.clear()

    def get_name(self) -> str:
        return settings.BASELINE_V2_RANDOM


# ---------------------------------------------------------------------------
# 2. LRUPolicy
# ---------------------------------------------------------------------------

class LRUPolicy(BaselinePolicy):
    """
    Predicts the N most recently used apps.
    """

    def __init__(self, capacity: int = None, top_k: int = settings.PREFETCH_TOP_K) -> None:
        self._lru: OrderedDict = OrderedDict()
        if capacity is None:
            self._capacity = (
                settings.PIN_TIER_CAPACITY +
                settings.HOT_TIER_CAPACITY +
                settings.WARM_TIER_CAPACITY +
                settings.COOL_TIER_CAPACITY
            )
        else:
            self._capacity = capacity
        self.top_k = top_k

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return top-k most recently used apps."""
        return list(self._lru.keys())[:self.top_k]

    def update(self, event: dict) -> None:
        """Move app to front of LRU queue."""
        app_id = event.get("app_id", "")
        if not app_id:
            return
        if app_id in self._lru:
            self._lru.move_to_end(app_id, last=False)
        else:
            self._lru[app_id] = True
            self._lru.move_to_end(app_id, last=False)
        while len(self._lru) > self._capacity:
            self._lru.popitem(last=True)

    def reset(self) -> None:
        self._lru.clear()

    def get_name(self) -> str:
        return settings.BASELINE_V2_LRU


# ---------------------------------------------------------------------------
# 3. LFUPolicy
# ---------------------------------------------------------------------------

class LFUPolicy(BaselinePolicy):
    """
    Predicts apps with the lowest global access frequency (LFU).
    """

    def __init__(self, top_k: int = settings.PREFETCH_TOP_K) -> None:
        self._freq: Counter = Counter()
        self.top_k = top_k

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return top-k most frequently used apps."""
        return [app for app, _ in self._freq.most_common(self.top_k)]

    def update(self, event: dict) -> None:
        app_id = event.get("app_id", "")
        if app_id:
            self._freq[app_id] += 1

    def reset(self) -> None:
        self._freq.clear()

    def get_name(self) -> str:
        return settings.BASELINE_V2_LFU


# ---------------------------------------------------------------------------
# 4. MRUPolicy
# ---------------------------------------------------------------------------

class MRUPolicy(BaselinePolicy):
    """
    Predicts the single most recently used app plus global top-N.
    """

    def __init__(self, top_k: int = settings.PREFETCH_TOP_K) -> None:
        self._last: Optional[str] = None
        self._recent: List[str] = []  # ordered most-recent-first
        self.top_k = top_k

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return most recent app first, then next most recent."""
        return list(self._recent[:self.top_k])

    def update(self, event: dict) -> None:
        app_id = event.get("app_id", "")
        if not app_id:
            return
        if app_id in self._recent:
            self._recent.remove(app_id)
        self._recent.insert(0, app_id)
        self._last = app_id

    def reset(self) -> None:
        self._last = None
        self._recent.clear()

    def get_name(self) -> str:
        return settings.BASELINE_V2_MRU


# ---------------------------------------------------------------------------
# 5. FrequencyPolicy
# ---------------------------------------------------------------------------

class FrequencyPolicy(BaselinePolicy):
    """
    Predicts the globally most frequent apps, stratified by time bucket.
    """

    def __init__(self, top_k: int = settings.PREFETCH_TOP_K) -> None:
        self._freq: Dict[Tuple[int, bool], Counter] = defaultdict(Counter)
        self.top_k = top_k

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return top-k most frequent apps for the current (time_bucket, weekend)."""
        bucket = int(context.get("time_bucket", 0))
        weekend = bool(context.get("weekend", False))
        counter = self._freq.get((bucket, weekend), Counter())
        return [app for app, _ in counter.most_common(self.top_k)]

    def update(self, event: dict) -> None:
        bucket = int(event.get("time_bucket", 0))
        weekend = bool(event.get("weekend", False))
        app_id = event.get("app_id", "")
        if app_id:
            self._freq[(bucket, weekend)][app_id] += 1

    def reset(self) -> None:
        self._freq.clear()

    def get_name(self) -> str:
        return settings.BASELINE_V2_FREQUENCY


# ---------------------------------------------------------------------------
# 6. RecencyFrequencyPolicy
# ---------------------------------------------------------------------------

class RecencyFrequencyPolicy(BaselinePolicy):
    """
    Scores each candidate app as: score = α*recency + β*frequency
    """

    def __init__(
        self,
        alpha: float = settings.BASELINE_RF_ALPHA,
        beta: float = settings.BASELINE_RF_BETA,
        recency_decay: float = settings.BASELINE_RF_RECENCY_DECAY,
        top_k: int = settings.PREFETCH_TOP_K,
    ) -> None:
        if not math.isclose(alpha + beta, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"RecencyFrequencyPolicy: alpha + beta must equal 1.0, "
                f"got alpha={alpha}, beta={beta}"
            )
        self._alpha = alpha
        self._beta = beta
        self._decay = recency_decay
        self._recency: Dict[str, float] = defaultdict(float)
        self._frequency: Counter = Counter()
        self._total_accesses: int = 0
        self.top_k = top_k
        self._last_update: Dict[str, int] = defaultdict(int)
        self._step_index = 0

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Score all known apps and return top-k by combined recency-frequency score."""
        if not self._frequency:
            return []

        # Get lazy-decayed recency values to find max_recency
        decayed_recency = {}
        for app_id in self._frequency:
            last_t = self._last_update[app_id]
            decayed_recency[app_id] = self._recency[app_id] * (self._decay ** (self._step_index - last_t))

        max_recency = max(decayed_recency.values()) if decayed_recency else 1.0
        total = max(1, self._total_accesses)

        scores: Dict[str, float] = {}
        for app_id in self._frequency:
            recency_norm = decayed_recency[app_id] / max(max_recency, 1e-9)
            freq_norm = self._frequency[app_id] / total
            scores[app_id] = self._alpha * recency_norm + self._beta * freq_norm

        top = sorted(scores, key=scores.__getitem__, reverse=True)
        return top[:self.top_k]

    def update(self, event: dict) -> None:
        """Decay all existing recency scores, then increment the launched app."""
        app_id = event.get("app_id", "")
        if not app_id:
            return

        self._step_index += 1
        # Lazy decay for the launched app
        last_t = self._last_update[app_id]
        self._recency[app_id] = self._recency[app_id] * (self._decay ** (self._step_index - last_t))
        self._recency[app_id] += 1.0
        self._last_update[app_id] = self._step_index

        self._frequency[app_id] += 1
        self._total_accesses += 1

    def reset(self) -> None:
        self._recency.clear()
        self._frequency.clear()
        self._total_accesses = 0
        self._last_update.clear()
        self._step_index = 0

    def get_name(self) -> str:
        return settings.BASELINE_V2_RECENCY_FREQUENCY


# ---------------------------------------------------------------------------
# 7. FirstOrderMarkovPolicy
# ---------------------------------------------------------------------------

class FirstOrderMarkovPolicy(BaselinePolicy):
    """
    First-order Markov chain: P(next_app | current_app).
    """

    def __init__(self, top_k: int = settings.PREFETCH_TOP_K) -> None:
        self._counts: Dict[str, Counter] = defaultdict(Counter)
        self._probs: Dict[str, Dict[str, float]] = {}
        self._trained = False
        self._previous_app: Optional[str] = None
        self.top_k = top_k

    def train(self, events: List[dict]) -> None:
        """Build the transition matrix from training events."""
        self._counts.clear()
        prev_app: Optional[str] = None
        for event in events:
            app_id = event.get("app_id", "")
            if not app_id:
                continue
            if prev_app is not None:
                self._counts[prev_app][app_id] += 1
            prev_app = app_id

        # Normalise counts to probabilities
        self._probs = {}
        for from_app, to_counts in self._counts.items():
            total = sum(to_counts.values())
            self._probs[from_app] = {
                to_app: count / total
                for to_app, count in to_counts.items()
            }

        self._trained = True
        n_states = len(self._probs)
        logger.info(
            f"FirstOrderMarkovPolicy trained: {n_states} states from {len(events)} events."
        )

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return top-k most probable next apps given the current app."""
        if not self._trained:
            return []
        transitions = self._probs.get(current_app_id, {})
        if not transitions:
            return []
        sorted_apps = sorted(transitions, key=transitions.__getitem__, reverse=True)
        return sorted_apps[:self.top_k]

    def get_transition_probability(self, from_app: str, to_app: str) -> float:
        """Return P(to_app | from_app) or 0.0 if unseen."""
        return self._probs.get(from_app, {}).get(to_app, 0.0)

    def update(self, event: dict) -> None:
        pass

    def reset(self) -> None:
        self._counts.clear()
        self._probs.clear()
        self._trained = False
        self._previous_app = None

    def get_name(self) -> str:
        return settings.BASELINE_V2_MARKOV

    @property
    def is_trained(self) -> bool:
        return self._trained


# ---------------------------------------------------------------------------
# 8. SecondOrderMarkovPolicy
# ---------------------------------------------------------------------------

class SecondOrderMarkovPolicy(BaselinePolicy):
    """
    Second-order Markov chain: P(next_app | prev_app, current_app).
    """

    def __init__(self, top_k: int = settings.PREFETCH_TOP_K) -> None:
        self._counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self._probs: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._first_order_marginal: Dict[str, Counter] = defaultdict(Counter)
        self._trained = False
        self._prev_app: Optional[str] = None
        self._curr_app: Optional[str] = None
        self.top_k = top_k

    def train(self, events: List[dict]) -> None:
        """Build the second-order transition matrix from training events."""
        self._counts.clear()
        prev: Optional[str] = None
        curr: Optional[str] = None
        for event in events:
            app_id = event.get("app_id", "")
            if not app_id:
                continue
            if prev is not None and curr is not None:
                self._counts[(prev, curr)][app_id] += 1
            prev = curr
            curr = app_id

        self._probs = {}
        for (p, c), to_counts in self._counts.items():
            total = sum(to_counts.values())
            self._probs[(p, c)] = {
                to_app: count / total
                for to_app, count in to_counts.items()
            }

        # Precompute first-order marginal for O(1) fallback
        self._first_order_marginal.clear()
        for (p, c), to_probs in self._probs.items():
            for to_app, prob in to_probs.items():
                self._first_order_marginal[c][to_app] += prob

        self._trained = True
        n_bigrams = len(self._probs)
        logger.info(
            f"SecondOrderMarkovPolicy trained: {n_bigrams} bigrams from {len(events)} events."
        )

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return top-k most probable next apps given (prev_app, current_app)."""
        if not self._trained:
            return []

        # Full second-order lookup
        if self._prev_app is not None:
            key = (self._prev_app, current_app_id)
            transitions = self._probs.get(key, {})
            if transitions:
                sorted_apps = sorted(transitions, key=transitions.__getitem__, reverse=True)
                self._prev_app = current_app_id
                return sorted_apps[:self.top_k]

        # First-order fallback (marginalise over prev)
        self._prev_app = current_app_id
        fallback_counts = self._first_order_marginal.get(current_app_id)
        if fallback_counts:
            return [app for app, _ in fallback_counts.most_common(self.top_k)]
        return []

    def update(self, event: dict) -> None:
        app_id = event.get("app_id", "")
        if app_id:
            self._prev_app = self._curr_app
            self._curr_app = app_id

    def reset(self) -> None:
        self._counts.clear()
        self._probs.clear()
        self._first_order_marginal.clear()
        self._trained = False
        self._prev_app = None
        self._curr_app = None

    def get_name(self) -> str:
        return settings.BASELINE_V2_MARKOV2

    @property
    def is_trained(self) -> bool:
        return self._trained


# ---------------------------------------------------------------------------
# 9. GraphOnlyPolicy
# ---------------------------------------------------------------------------

class GraphOnlyPolicy(BaselinePolicy):
    """
    BehaviouralGraph prediction, no RL.
    """

    def __init__(self, user_id: str = "eval_user", top_k: int = settings.PREFETCH_TOP_K) -> None:
        self._user_id = user_id
        self._top_k = top_k
        self._graph = None
        self._memory_manager = None
        self._current_node_id: Optional[str] = None

    def _ensure_graph(self) -> None:
        if self._graph is None:
            from src.core.event_bus import EventBus
            from src.core.graph_engine import BehaviouralGraph
            from src.core.memory_manager import MemoryManager
            EventBus.get_instance().clear_all()
            self._graph = BehaviouralGraph(self._user_id)
            self._memory_manager = MemoryManager(self._user_id, self._graph)

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return top-k next apps predicted by the graph."""
        self._ensure_graph()
        if self._current_node_id is None:
            return []
        battery = float(context.get("battery", 100.0))
        node_ids = self._graph.get_top_k_next_nodes(
            self._current_node_id, self._top_k, battery
        )
        apps = []
        for nid in node_ids:
            node = self._graph.get_node(nid)
            if node and node.app_id not in apps:
                apps.append(node.app_id)
        return apps[:self._top_k]

    def update(self, event: dict) -> None:
        self._ensure_graph()
        from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED
        payload = {
            "timestamp": float(event.get("timestamp", 0.0)),
            "user_id": self._user_id,
            "app_id": event.get("app_id", "unknown"),
            "category": event.get("category", "utility"),
            "battery": float(event.get("battery", 100.0)),
            "time_of_day_bucket": int(event.get("time_bucket", 0)),
            "time_bucket": int(event.get("time_bucket", 0)),
            "day": int(event.get("day", 0)),
            "weekend": bool(event.get("weekend", False)),
            "headphones": bool(event.get("headphones", False)),
            "calendar_event_in_mins": event.get("calendar_event_in_mins"),
        }
        EventBus.get_instance().publish(TOPIC_APP_LAUNCHED, payload)
        app_id = event.get("app_id", "")
        time_bucket = int(event.get("time_bucket", 0))
        battery = float(event.get("battery", 100.0))
        battery_bucket = min(4, int(battery / 20))
        key = (app_id, time_bucket, battery_bucket)
        self._current_node_id = self._graph._node_lookup.get(key, None)

    def reset(self) -> None:
        from src.core.event_bus import EventBus
        if self._graph is not None:
            EventBus.get_instance().clear_all()
        self._graph = None
        self._memory_manager = None
        self._current_node_id = None

    def get_name(self) -> str:
        return settings.BASELINE_V2_GRAPH_ONLY


# ---------------------------------------------------------------------------
# 10. GraphMindRLPolicy
# ---------------------------------------------------------------------------

class GraphMindRLPolicy(BaselinePolicy):
    """
    Full GraphMind system: Graph + RL ResourceAllocationPolicy + ConfidencePrefetch.
    """

    def __init__(self, user_id: str = "eval_user", top_k: int = 15) -> None:
        self._user_id = user_id
        self._top_k = top_k
        self._last_result: Optional[dict] = None
        self._train_events: List[dict] = []

    def train(self, events: List[dict]) -> None:
        self._train_events = events

    def run_full_evaluation(self, events: List[dict]) -> dict:
        from src.benchmarks.graphmind_policy_runner import GraphMindPolicyRunner
        runner = GraphMindPolicyRunner(self._user_id, top_k=self._top_k)
        if hasattr(self, "_train_events") and self._train_events:
            runner.train(self._train_events)
        self._last_result = runner.run(events)
        return self._last_result

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        logger.warning(
            "GraphMindRLPolicy.predict_next_apps() called without run_full_evaluation(). "
            "Use run_full_evaluation() for accurate GraphMind metrics."
        )
        return []

    def update(self, event: dict) -> None:
        pass

    def reset(self) -> None:
        self._last_result = None

    def get_name(self) -> str:
        return settings.BASELINE_V2_GRAPHMIND_RL
