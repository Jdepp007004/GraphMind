"""
src/models/context_markov.py

Context-Aware Markov predictor.

Builds three conditional models:
  1. P(next | current, time_bucket)
  2. P(next | current, weekday)
  3. P(next | current, time_bucket, weekday)

Combines them with weights learned from a validation split.

Time bucket: 0-47  (30-minute intervals across 24 hours)
Weekday:     0-6   (Monday=0, Sunday=6)

Confidence output is the max probability across the combined prediction.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class ContextMarkov:
    """
    Context-Aware Markov model that conditions on temporal context.

    The three sub-models are combined via learned weights:

        P_combined(next | current, ctx) =
            w_tb  * P(next | current, time_bucket) +
            w_wd  * P(next | current, weekday) +
            w_full* P(next | current, time_bucket, weekday) +
            w_base* P(next | current)       ← baseline fallback

    Weights are initialised uniformly and updated on a validation split
    by rewarding each sub-model proportionally to its top-1 accuracy.
    """

    _EPSILON = 1e-9

    def __init__(self, top_k: int = 5, laplace_alpha: float = 0.3) -> None:
        self.top_k = top_k
        self.laplace_alpha = laplace_alpha

        # Sub-model count tables
        # key → {next_app: count}
        self._m_base: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._m_tb:   Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._m_wd:   Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._m_full: Dict[Tuple[str, int, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        self._vocab: set = set()

        # Combination weights [base, tb, wd, full]
        self._weights = [0.25, 0.25, 0.25, 0.25]

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        events: List[str],
        time_buckets: List[int],
        weekdays: List[int],
    ) -> None:
        """
        Train on app sequence with temporal context.

        Args:
            events:       Ordered list of app package names.
            time_buckets: 0-47 time bucket for each event.
            weekdays:     0-6 weekday for each event.
        """
        assert len(events) == len(time_buckets) == len(weekdays), (
            "events, time_buckets, weekdays must have equal length"
        )
        self._clear()

        for app in events:
            self._vocab.add(app)

        for i in range(1, len(events)):
            cur = events[i - 1]
            nxt = events[i]
            tb  = time_buckets[i - 1]
            wd  = weekdays[i - 1]

            self._m_base[cur][nxt] += 1
            self._m_tb[(cur, tb)][nxt] += 1
            self._m_wd[(cur, wd)][nxt] += 1
            self._m_full[(cur, tb, wd)][nxt] += 1

    def fit_weights(
        self,
        val_events: List[str],
        val_time_buckets: List[int],
        val_weekdays: List[int],
        n_epochs: int = 3,
        lr: float = 0.1,
    ) -> None:
        """
        Learn combination weights on a held-out validation split.

        Uses a proportional accuracy reward: each sub-model gets credit
        proportional to the probability it assigns to the correct next app.
        Weights are normalised after each epoch.

        Args:
            val_events:       Validation app sequence.
            val_time_buckets: Validation time buckets.
            val_weekdays:     Validation weekdays.
            n_epochs:         Number of weight update passes.
            lr:               Learning rate for weight update.
        """
        if len(val_events) < 2:
            return

        w = list(self._weights)

        for _ in range(n_epochs):
            scores = [0.0, 0.0, 0.0, 0.0]  # base, tb, wd, full
            n_steps = 0

            for i in range(1, len(val_events)):
                cur = val_events[i - 1]
                nxt = val_events[i]
                tb  = val_time_buckets[i - 1]
                wd  = val_weekdays[i - 1]

                p = [
                    self._prob_base(cur, nxt),
                    self._prob_tb(cur, tb, nxt),
                    self._prob_wd(cur, wd, nxt),
                    self._prob_full(cur, tb, wd, nxt),
                ]
                total_p = sum(p) + self._EPSILON
                for j in range(4):
                    scores[j] += p[j] / total_p
                n_steps += 1

            if n_steps == 0:
                break

            # Update weights proportionally to accuracy scores
            for j in range(4):
                w[j] = w[j] + lr * (scores[j] / n_steps - w[j])

            # Normalise
            total_w = sum(w) + self._EPSILON
            w = [v / total_w for v in w]

        self._weights = w

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(
        self,
        current: str,
        time_bucket: int = 0,
        weekday: int = 0,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Predict top-k next apps with confidence scores.

        Args:
            current:     Current app.
            time_bucket: 0-47 (30-min intervals).
            weekday:     0-6 (Mon-Sun).
            top_k:       Override instance top_k.

        Returns:
            List of (app, score) tuples sorted descending by score.
        """
        k = top_k or self.top_k
        V = len(self._vocab) or 1
        w_base, w_tb, w_wd, w_full = self._weights

        candidates: Dict[str, float] = {}
        for app in self._vocab:
            if app == current:
                continue
            score = (
                w_base * self._prob_base(current, app) +
                w_tb   * self._prob_tb(current, time_bucket, app) +
                w_wd   * self._prob_wd(current, weekday, app) +
                w_full * self._prob_full(current, time_bucket, weekday, app)
            )
            candidates[app] = score

        ranked = sorted(candidates.items(), key=lambda x: -x[1])[:k]
        return ranked

    def predict_apps(
        self,
        current: str,
        time_bucket: int = 0,
        weekday: int = 0,
        top_k: Optional[int] = None,
    ) -> List[str]:
        return [app for app, _ in self.predict(current, time_bucket, weekday, top_k)]

    def confidence(self, current: str, time_bucket: int = 0, weekday: int = 0) -> float:
        results = self.predict(current, time_bucket, weekday, top_k=1)
        return results[0][1] if results else 0.0

    def get_weights(self) -> Dict[str, float]:
        return {
            "base": round(self._weights[0], 4),
            "time_bucket": round(self._weights[1], 4),
            "weekday": round(self._weights[2], 4),
            "full_context": round(self._weights[3], 4),
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _clear(self) -> None:
        self._m_base.clear()
        self._m_tb.clear()
        self._m_wd.clear()
        self._m_full.clear()
        self._vocab.clear()
        self._weights = [0.25, 0.25, 0.25, 0.25]

    def _laplace_prob(self, counts: Dict[str, int], app: str) -> float:
        V = len(self._vocab) or 1
        alpha = self.laplace_alpha
        total = sum(counts.values()) or 0
        return (counts.get(app, 0) + alpha) / (total + alpha * V)

    def _prob_base(self, cur: str, nxt: str) -> float:
        return self._laplace_prob(self._m_base.get(cur, {}), nxt)

    def _prob_tb(self, cur: str, tb: int, nxt: str) -> float:
        return self._laplace_prob(self._m_tb.get((cur, tb), {}), nxt)

    def _prob_wd(self, cur: str, wd: int, nxt: str) -> float:
        return self._laplace_prob(self._m_wd.get((cur, wd), {}), nxt)

    def _prob_full(self, cur: str, tb: int, wd: int, nxt: str) -> float:
        return self._laplace_prob(self._m_full.get((cur, tb, wd), {}), nxt)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ContextMarkov("
            f"vocab={len(self._vocab)}, "
            f"weights={self.get_weights()}, "
            f"top_k={self.top_k})"
        )
