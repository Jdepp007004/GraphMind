"""
src/benchmarks/baselines_v2.py

Ten research-grade baseline policies for GraphMind v2 evaluation.

All policies extend the existing BaselinePolicy ABC from baselines.py.
None of the existing baselines are modified.

Policies (in benchmark table order):
  1.  RandomPolicy               — random prediction
  2.  LRUPolicy                  — least-recently-used
  3.  LFUPolicy                  — least-frequently-used
  4.  MRUPolicy                  — most-recently-used (strong recency)
  5.  FrequencyPolicy            — global frequency counts
  6.  RecencyFrequencyPolicy     — α*recency + β*frequency (strong classical)
  7.  FirstOrderMarkovPolicy     — P(next | current)
  8.  SecondOrderMarkovPolicy    — P(next | prev, current)
  9.  GraphOnlyPolicy            — BehaviouralGraph prediction, no RL
  10. GraphMindRLPolicy          — Graph + RL + ConfidencePrefetch (full system)

Training protocols:
  Markov policies: trained ONLY on the train split, evaluated on test split.
  All other policies: online learning (update on each observed event).
  GraphOnlyPolicy / GraphMindRLPolicy: replayed through GraphMindPolicyRunner.
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

    This is the absolute baseline — any policy that does not beat random
    prediction has failed. Maintains a vocabulary of seen apps and samples
    uniformly.
    """

    def __init__(self, seed: int = settings.RANDOM_SEED) -> None:
        self._vocab: List[str] = []
        self._rng = random.Random(seed)

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return up to 5 randomly sampled app IDs from the observed vocabulary."""
        if not self._vocab:
            return []
        k = min(5, len(self._vocab))
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

    Standard LRU eviction policy. No transition modelling. No time-of-day
    awareness. Included because it matches Android's UsageStatsManager
    default behavior.
    """

    def __init__(self, capacity: int = settings.HOT_TIER_CAPACITY) -> None:
        self._lru: OrderedDict = OrderedDict()
        self._capacity = capacity

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return top-5 most recently used apps."""
        return list(self._lru.keys())[:5]

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

    LFU captures long-term usage patterns but is slow to adapt to changes.
    Included as a complement to LRU.
    """

    def __init__(self) -> None:
        self._freq: Counter = Counter()

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return top-5 most frequently used apps."""
        return [app for app, _ in self._freq.most_common(5)]

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
    Predicts the single most recently used app plus global top-4.

    MRU is a strong recency-biased predictor. It outperforms LRU in
    workloads where the last-used app is the most likely next app
    (e.g., back-and-forth between messaging and social apps).
    """

    def __init__(self) -> None:
        self._last: Optional[str] = None
        self._recent: List[str] = []  # ordered most-recent-first

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return most recent app first, then next most recent."""
        return list(self._recent[:5])

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

    This is an improved Bixby-style predictor: frequency counts are tracked
    per (time_bucket, weekend) pair, giving light time-of-day awareness without
    any transition modelling.
    """

    def __init__(self) -> None:
        self._freq: Dict[Tuple[int, bool], Counter] = defaultdict(Counter)

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """Return top-5 most frequent apps for the current (time_bucket, weekend)."""
        bucket = int(context.get("time_bucket", 0))
        weekend = bool(context.get("weekend", False))
        counter = self._freq.get((bucket, weekend), Counter())
        return [app for app, _ in counter.most_common(5)]

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

    This classical baseline often performs surprisingly well because it
    captures both short-term (recency) and long-term (frequency) patterns.
    It is typically a stronger baseline than either LRU or LFU alone.

    The benchmark question is whether GraphMind beats this combined signal.

    Scoring formulas:
      recency[app]   = Σ (RECENCY_DECAY^k) for each past access, k steps ago
                       (exponentially decaying — recent accesses weigh more)
      frequency[app] = count[app] / total_accesses
                       (normalised global frequency)
      score[app]     = α * recency_norm[app] + β * frequency[app]

    α = settings.BASELINE_RF_ALPHA  (default 0.6)
    β = settings.BASELINE_RF_BETA   (default 0.4)
    """

    def __init__(
        self,
        alpha: float = settings.BASELINE_RF_ALPHA,
        beta: float = settings.BASELINE_RF_BETA,
        recency_decay: float = settings.BASELINE_RF_RECENCY_DECAY,
    ) -> None:
        """
        Args:
            alpha:        Weight for recency component. alpha + beta should = 1.0.
            beta:         Weight for frequency component.
            recency_decay: Exponential decay applied to recency scores each step.
        """
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

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """
        Score all known apps and return top-5 by combined recency-frequency score.

        recency scores are normalised to [0,1] before weighting.
        """
        if not self._frequency:
            return []

        max_recency = max(self._recency.values()) if self._recency else 1.0
        total = max(1, self._total_accesses)

        scores: Dict[str, float] = {}
        for app_id in self._frequency:
            recency_norm = self._recency[app_id] / max(max_recency, 1e-9)
            freq_norm = self._frequency[app_id] / total
            scores[app_id] = self._alpha * recency_norm + self._beta * freq_norm

        top = sorted(scores, key=scores.__getitem__, reverse=True)
        return top[:5]

    def update(self, event: dict) -> None:
        """
        Decay all existing recency scores, then increment the launched app.
        """
        app_id = event.get("app_id", "")
        if not app_id:
            return

        # Decay all existing recency scores
        for key in list(self._recency.keys()):
            self._recency[key] *= self._decay

        # Increment launched app
        self._recency[app_id] += 1.0
        self._frequency[app_id] += 1
        self._total_accesses += 1

    def reset(self) -> None:
        self._recency.clear()
        self._frequency.clear()
        self._total_accesses = 0

    def get_name(self) -> str:
        return settings.BASELINE_V2_RECENCY_FREQUENCY


# ---------------------------------------------------------------------------
# 7. FirstOrderMarkovPolicy
# ---------------------------------------------------------------------------

class FirstOrderMarkovPolicy(BaselinePolicy):
    """
    First-order Markov chain: P(next_app | current_app).

    Transition matrix is built from training data ONLY. This policy must
    be trained via train() on the training split before predict_next_apps()
    is valid. Calling predict_next_apps() before training returns an empty list.

    This is the primary non-RL baseline. GraphMind must beat this to
    justify the complexity of the graph + RL architecture.

    Metrics exposed:
      precision, recall, f1, hit_rate
    """

    def __init__(self) -> None:
        # Transition counts: counts[from_app][to_app] = count
        self._counts: Dict[str, Counter] = defaultdict(Counter)
        # Normalised transition probabilities (built by train())
        self._probs: Dict[str, Dict[str, float]] = {}
        self._trained = False
        self._previous_app: Optional[str] = None

    def train(self, events: List[dict]) -> None:
        """
        Build the transition matrix from a list of training events.

        Args:
            events: Chronologically sorted list of GraphMindEvent dicts.
                    These must be the TRAIN split only.
        """
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
        """
        Return top-5 most probable next apps given the current app.

        Returns [] if untrained or current_app_id has no observed transitions.
        """
        if not self._trained:
            return []
        transitions = self._probs.get(current_app_id, {})
        if not transitions:
            return []
        sorted_apps = sorted(transitions, key=transitions.__getitem__, reverse=True)
        return sorted_apps[:5]

    def get_transition_probability(self, from_app: str, to_app: str) -> float:
        """Return P(to_app | from_app) or 0.0 if unseen."""
        return self._probs.get(from_app, {}).get(to_app, 0.0)

    def update(self, event: dict) -> None:
        """
        No-op during evaluation. Markov matrix is fixed after train().
        Online updates would create data leakage.
        """

    def reset(self) -> None:
        """Reset training state. Requires re-training."""
        self._counts.clear()
        self._probs.clear()
        self._trained = False
        self._previous_app = None

    def get_name(self) -> str:
        return settings.BASELINE_V2_MARKOV

    @property
    def is_trained(self) -> bool:
        """Return True if the transition matrix has been built."""
        return self._trained


# ---------------------------------------------------------------------------
# 8. SecondOrderMarkovPolicy
# ---------------------------------------------------------------------------

class SecondOrderMarkovPolicy(BaselinePolicy):
    """
    Second-order Markov chain: P(next_app | prev_app, current_app).

    Captures two-step dependencies in app usage sequences. If GraphMind
    beats first-order Markov but loses to second-order Markov, that
    indicates the graph's value comes primarily from bigram-level patterns.

    The transition key is (prev_app, current_app) → Counter of next apps.

    This policy requires train() on training data before evaluation.
    """

    def __init__(self) -> None:
        # counts[(prev, current)][next] = count
        self._counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self._probs: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._trained = False
        self._prev_app: Optional[str] = None
        self._curr_app: Optional[str] = None

    def train(self, events: List[dict]) -> None:
        """
        Build the second-order transition matrix from training events.

        Args:
            events: Chronologically sorted TRAIN split only.
        """
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

        self._trained = True
        n_bigrams = len(self._probs)
        logger.info(
            f"SecondOrderMarkovPolicy trained: {n_bigrams} bigrams from {len(events)} events."
        )

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """
        Return top-5 most probable next apps given (prev_app, current_app).

        Falls back to most frequent next-apps for current_app alone
        when the bigram (prev, current) is unseen.
        """
        if not self._trained:
            return []

        # Full second-order lookup
        if self._prev_app is not None:
            key = (self._prev_app, current_app_id)
            transitions = self._probs.get(key, {})
            if transitions:
                sorted_apps = sorted(transitions, key=transitions.__getitem__, reverse=True)
                self._prev_app = current_app_id
                return sorted_apps[:5]

        # First-order fallback (marginalise over prev)
        fallback_counts: Counter = Counter()
        for (p, c), to_probs in self._probs.items():
            if c == current_app_id:
                for to_app, prob in to_probs.items():
                    fallback_counts[to_app] += prob
        self._prev_app = current_app_id
        if fallback_counts:
            return [app for app, _ in fallback_counts.most_common(5)]
        return []

    def update(self, event: dict) -> None:
        """
        Track the previous app during evaluation. No matrix updates.
        """
        app_id = event.get("app_id", "")
        if app_id:
            self._prev_app = self._curr_app
            self._curr_app = app_id

    def reset(self) -> None:
        """Reset all state. Requires re-training."""
        self._counts.clear()
        self._probs.clear()
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
    BehaviouralGraph prediction without RL and without confidence-based prefetch.

    This policy uses the graph's get_top_k_next_nodes() method directly,
    with a fixed top-k and no RL budget management. It is the critical
    ablation baseline for isolating the contribution of the graph structure.

    Graph → candidates. No RL. No confidence scoring.

    This policy must be compared against:
      - FirstOrderMarkov   (does the graph add beyond transition probs?)
      - GraphMindRL        (does RL improve on pure graph prediction?)
    """

    def __init__(self, user_id: str = "eval_user", top_k: int = settings.PREFETCH_TOP_K) -> None:
        self._user_id = user_id
        self._top_k = top_k
        self._graph = None
        self._memory_manager = None
        self._current_node_id: Optional[str] = None

    def _ensure_graph(self) -> None:
        """Lazily initialise the BehaviouralGraph and MemoryManager."""
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
        return apps[:5]

    def update(self, event: dict) -> None:
        """Publish the event through the graph engine to update state."""
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
        # Find the node that matches the launched app
        app_id = event.get("app_id", "")
        time_bucket = int(event.get("time_bucket", 0))
        battery = float(event.get("battery", 100.0))
        battery_bucket = min(4, int(battery / 20))
        for nid in self._graph._graph.nodes():
            n = self._graph._graph.nodes[nid]["data"]
            if (n.app_id == app_id and n.time_bucket == time_bucket
                    and n.battery_bucket == battery_bucket):
                self._current_node_id = nid
                break

    def reset(self) -> None:
        """Reset the graph and memory state."""
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

    This is the proposed system being evaluated. It uses the existing
    GraphMindPolicyRunner to replay events, which includes:
      - BehaviouralGraph (existing)
      - MemoryManager HOT/WARM/COLD tiers (existing)
      - ConfidencePrefetch (new v2)
      - RL ResourceAllocationPolicy environment (new v2)

    This policy does not implement update() / predict_next_apps() directly —
    it delegates the full evaluation loop to GraphMindPolicyRunner.run().
    Use run_full_evaluation() instead of the incremental interface.
    """

    def __init__(self, user_id: str = "eval_user", top_k: int = 15) -> None:
        self._user_id = user_id
        self._top_k = top_k
        self._last_result: Optional[dict] = None
        self._train_events: List[dict] = []

    def train(self, events: List[dict]) -> None:
        """Store training events to build frequency baselines during full evaluation."""
        self._train_events = events

    def run_full_evaluation(self, events: List[dict]) -> dict:
        """
        Run the full GraphMind pipeline on an event list and return metrics.

        Args:
            events: List of GraphMindEvent dicts (typically the test split).

        Returns:
            dict with cache_hit_rate, prefetch_precision, prefetch_recall,
            prefetch_f1, thrash_rate, battery_overhead_pct, avg_latency_ms, records.
        """
        from src.benchmarks.graphmind_policy_runner import GraphMindPolicyRunner
        runner = GraphMindPolicyRunner(self._user_id, top_k=self._top_k)
        if hasattr(self, "_train_events") and self._train_events:
            runner.train(self._train_events)
        self._last_result = runner.run(events)
        return self._last_result

    def predict_next_apps(self, current_app_id: str, context: dict) -> List[str]:
        """
        Not used in the incremental interface — GraphMindRLPolicy requires
        run_full_evaluation() for a valid evaluation. Returns [] when called
        incrementally without a prior full evaluation.
        """
        logger.warning(
            "GraphMindRLPolicy.predict_next_apps() called without run_full_evaluation(). "
            "Use run_full_evaluation() for accurate GraphMind metrics."
        )
        return []

    def update(self, event: dict) -> None:
        """No-op for incremental interface. Use run_full_evaluation()."""

    def reset(self) -> None:
        self._last_result = None

    def get_name(self) -> str:
        return settings.BASELINE_V2_GRAPHMIND_RL
