"""
src/prefetch/confidence_prefetch.py

Confidence-based prefetch scorer — GraphMindRL_V5 production configuration.

Validated 2026-06-06: F1=0.7745, ΔF1=+0.0321, p=0.0115, 31 users (UbiqLog).

Confidence formula:
  confidence = W_TRANSITION * transition_prob
             + W_RECENCY    * recency_score
             + W_FREQUENCY  * frequency_score
             + W_CONTEXT    * context_score

GraphMindRL_V5 weights (config/settings.py):
  PREFETCH_CONFIDENCE_W_TRANSITION = 0.50  # transition prob (primary signal)
  PREFETCH_CONFIDENCE_W_RECENCY    = 0.10  # was 0.20 — recency overweighted
  PREFETCH_CONFIDENCE_W_FREQUENCY  = 0.40  # was 0.20 — frequency underweighted
  PREFETCH_CONFIDENCE_W_CONTEXT    = 0.00  # zeroed — time context adds noise

  PREFETCH_CONFIDENCE_THRESHOLD    = 0.16  # was 0.70; adaptive ±0.005 on 20-step HR

Component definitions:
  transition_prob : P(candidate | current_app) from the BehaviouralGraph edge.
                    Pulled directly from GraphEdge.transition_prob ∈ [0,1].

  recency_score   : Exponentially decaying score for how recently the candidate
                    was last seen. recency[app] *= RECENCY_DECAY each step,
                    += 1.0 on access. Normalised to [0,1] by dividing by max.

  frequency_score : count[app] / total_events. Normalised global frequency.

  context_score   : 1.0 if the candidate's most common time_bucket matches the
                    current time_bucket, 0.5 if within ±2 buckets, 0.0 otherwise.

All weights are configurable in settings.py to support ablation studies.
Setting W_RECENCY=0, W_FREQUENCY=0, W_CONTEXT=0 gives a graph-only scorer.

Each prediction exposes:
  {
    "app_id"       : str,
    "node_id"      : str,
    "confidence"   : float,   # combined score
    "transition"   : float,   # component: transition probability
    "recency"      : float,   # component: normalised recency
    "frequency"    : float,   # component: normalised frequency
    "context"      : float,   # component: context match
  }
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from config import settings
from src.core.graph_engine import BehaviouralGraph

logger = logging.getLogger(__name__)


class ConfidencePrefetch:
    """
    Pure confidence scorer for next-app prediction.

    This class is intentionally decoupled from PrefetchDaemon and APScheduler.
    It is a stateful scorer that can be called synchronously from both:
      - BenchmarkEvaluatorV2 (batch evaluation)
      - PrefetchDaemon (background daemon, optional integration)

    State tracking (updated via observe_event()):
      - recency scores per app_id
      - frequency counts per app_id
      - time_bucket distribution per app_id

    The graph provides transition probabilities. The scorer combines all
    signals into a single confidence score per candidate.
    """

    def __init__(
        self,
        graph: BehaviouralGraph,
        w_transition: float = settings.PREFETCH_CONFIDENCE_W_TRANSITION,
        w_recency: float = settings.PREFETCH_CONFIDENCE_W_RECENCY,
        w_frequency: float = settings.PREFETCH_CONFIDENCE_W_FREQUENCY,
        w_context: float = settings.PREFETCH_CONFIDENCE_W_CONTEXT,
        confidence_threshold: float = settings.PREFETCH_CONFIDENCE_THRESHOLD,
        recency_decay: float = settings.PREFETCH_RECENCY_DECAY,
    ) -> None:
        """
        Args:
            graph:               BehaviouralGraph instance (read-only).
            w_transition:        Weight for transition probability component.
            w_recency:           Weight for recency score component.
            w_frequency:         Weight for frequency score component.
            w_context:           Weight for context match component.
            confidence_threshold: Only prefetch when confidence >= this value.
            recency_decay:       Exponential decay applied to recency each step.
        """
        total_w = w_transition + w_recency + w_frequency + w_context
        if abs(total_w - 1.0) > 1e-6:
            raise ValueError(
                f"ConfidencePrefetch: weights must sum to 1.0, "
                f"got {total_w:.4f} (w_transition={w_transition}, "
                f"w_recency={w_recency}, w_frequency={w_frequency}, "
                f"w_context={w_context})"
            )
        self._graph = graph
        self._w_t = w_transition
        self._w_r = w_recency
        self._w_f = w_frequency
        self._w_c = w_context
        self._threshold = confidence_threshold
        self._init_threshold = confidence_threshold
        self._decay = recency_decay

        # Per-app recency scores (exponentially decaying)
        self._recency: Dict[str, float] = defaultdict(float)
        # Per-app frequency counts
        self._frequency: Counter = Counter()
        # Per-app time bucket observations (for context scoring)
        self._time_buckets: Dict[str, Counter] = defaultdict(Counter)
        # Total events observed (for frequency normalisation)
        self._total_events: int = 0
        # Rolling hit history for adaptive threshold (last 20 steps)
        self._hit_history: List[float] = []
        self._HIT_HISTORY_LEN = 20
        self._ADAPT_STEP = 0.005
        self._THRESHOLD_MIN = 0.05
        self._THRESHOLD_MAX = 0.25

    def observe_event(self, event: dict, hit: Optional[bool] = None) -> None:
        """
        Update internal state from a new observed event.

        Must be called for every event in chronological order before scoring.
        Equivalent to an online learning update.

        Args:
            event: GraphMindEvent dict with at least 'app_id' and 'time_bucket'.
            hit:   Optional bool — whether the last prediction was a cache hit.
                   If provided, updates the adaptive threshold (GraphMindRL_V5
                   mechanism: threshold ±0.005 based on 20-step rolling hit rate).
        """
        app_id = event.get("app_id", "")
        if not app_id:
            return

        time_bucket = int(event.get("time_bucket", 0))

        # Decay all recency scores before incrementing
        for key in list(self._recency.keys()):
            self._recency[key] *= self._decay

        self._recency[app_id] += 1.0
        self._frequency[app_id] += 1
        self._time_buckets[app_id][time_bucket] += 1
        self._total_events += 1

        # Adaptive threshold update (GraphMindRL_V5)
        if hit is not None:
            self._hit_history.append(1.0 if hit else 0.0)
            if len(self._hit_history) > self._HIT_HISTORY_LEN:
                self._hit_history.pop(0)
            if len(self._hit_history) == self._HIT_HISTORY_LEN:
                rolling_hr = sum(self._hit_history) / self._HIT_HISTORY_LEN
                if rolling_hr < 0.50:
                    self._threshold = max(
                        self._THRESHOLD_MIN,
                        self._threshold - self._ADAPT_STEP
                    )
                elif rolling_hr > 0.80:
                    self._threshold = min(
                        self._THRESHOLD_MAX,
                        self._threshold + self._ADAPT_STEP
                    )

    def score_candidates(
        self,
        current_node_id: str,
        current_time_bucket: int,
        battery: float = 100.0,
        max_candidates: int = 20,
    ) -> List[dict]:
        """
        Score all candidate next nodes from the graph and return those above threshold.

        Args:
            current_node_id:     Node ID of the currently active app.
            current_time_bucket: Current 30-min bucket (0–47).
            battery:             Current battery level (0–100).
            max_candidates:      Maximum number of graph candidates to consider.

        Returns:
            List of candidate dicts sorted by confidence descending.
            Each dict contains: app_id, node_id, confidence, transition,
            recency, frequency, context.
            Only candidates with confidence >= threshold are included.
        """
        # Get candidate node IDs from graph (more than top_k so we can threshold)
        candidate_ids = self._graph.get_top_k_next_nodes(
            current_node_id, max_candidates, battery
        )
        if not candidate_ids:
            return []

        # Precompute normalisation denominators
        max_recency = max(self._recency.values(), default=1.0)
        total_freq = max(self._total_events, 1)

        scored: List[dict] = []
        for node_id in candidate_ids:
            node = self._graph.get_node(node_id)
            if node is None:
                continue

            transition = self._get_transition_prob(current_node_id, node_id)
            recency_norm = self._recency.get(node.app_id, 0.0) / max(max_recency, 1e-9)
            freq_norm = self._frequency.get(node.app_id, 0) / total_freq
            context_score = self._compute_context_score(node.app_id, current_time_bucket)

            confidence = (
                self._w_t * transition
                + self._w_r * recency_norm
                + self._w_f * freq_norm
                + self._w_c * context_score
            )

            if confidence >= self._threshold:
                scored.append({
                    "app_id": node.app_id,
                    "node_id": node_id,
                    "confidence": round(confidence, 4),
                    "transition": round(transition, 4),
                    "recency": round(recency_norm, 4),
                    "frequency": round(freq_norm, 4),
                    "context": round(context_score, 4),
                })

        scored.sort(key=lambda x: x["confidence"], reverse=True)
        logger.debug(
            f"ConfidencePrefetch: {len(scored)} candidates above threshold "
            f"{self._threshold} from node {current_node_id}"
        )
        return scored

    def prefetch(
        self,
        current_node_id: str,
        current_time_bucket: int,
        battery: float = 100.0,
    ) -> Tuple[List[str], List[dict]]:
        """
        Convenience method: score candidates and return (node_id_list, scored_list).

        Returns:
            (prefetch_node_ids, scored_candidates)
            prefetch_node_ids: IDs above the confidence threshold, sorted by confidence.
            scored_candidates: Full scoring detail for each prefetched candidate.
        """
        candidates = self.score_candidates(current_node_id, current_time_bucket, battery)
        return [c["node_id"] for c in candidates], candidates

    def reset(self) -> None:
        """Reset all learned state. Scorer returns to untrained state."""
        self._recency.clear()
        self._frequency.clear()
        self._time_buckets.clear()
        self._total_events = 0
        self._hit_history = []
        self._threshold = self._init_threshold

    def get_stats(self) -> dict:
        """Return summary statistics about the scorer's internal state."""
        return {
            "total_events_observed": self._total_events,
            "unique_apps_tracked": len(self._frequency),
            "confidence_threshold": self._threshold,
            "weights": {
                "transition": self._w_t,
                "recency": self._w_r,
                "frequency": self._w_f,
                "context": self._w_c,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_transition_prob(self, source_id: str, target_id: str) -> float:
        """Look up the edge transition probability from the graph."""
        edges = self._graph.get_edges_from(source_id)
        for edge in edges:
            if edge.target_id == target_id:
                return float(edge.transition_prob)
        return 0.0

    def _compute_context_score(self, app_id: str, current_bucket: int) -> float:
        """
        Score how well the candidate's historical time distribution matches now.

        Returns 1.0 if the most common bucket for this app matches exactly.
        Returns 0.5 if within ±2 buckets (30-min slots = ±1 hour window).
        Returns 0.0 otherwise.
        """
        bucket_counts = self._time_buckets.get(app_id)
        if not bucket_counts:
            return 0.0

        most_common_bucket, _ = bucket_counts.most_common(1)[0]
        diff = abs(most_common_bucket - current_bucket)
        # Handle bucket wraparound (47→0 is 1 bucket away)
        diff = min(diff, 48 - diff)

        if diff == 0:
            return 1.0
        if diff <= 2:
            return 0.5
        return 0.0
