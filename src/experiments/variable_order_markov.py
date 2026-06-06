"""
src/models/variable_order_markov.py

Variable-Order Markov (VOM) predictor for app sequences.

Prediction hierarchy:
  1. P(next | prev, current)   — second-order context
  2. P(next | current)         — first-order fallback
  3. Global frequency          — ultimate fallback

Key design choices:
  - Laplace smoothing to handle sparse bigrams
  - Per-candidate confidence score
  - Top-k output with confidence
  - Train/inference separated cleanly
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class VariableOrderMarkov:
    """
    Variable-Order Markov model for app-transition prediction.

    Trains on an app event sequence and predicts the most likely
    next apps given the current app and optional previous app.

    Attributes:
        laplace_alpha: Laplace smoothing constant (default 0.5 — half-Laplace).
        top_k:         Maximum number of candidates returned per prediction.
    """

    def __init__(self, laplace_alpha: float = 0.5, top_k: int = 5) -> None:
        self.laplace_alpha = laplace_alpha
        self.top_k = top_k

        # Second-order counts: (prev, current) → {next: count}
        self._m2_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # First-order counts: current → {next: count}
        self._m1_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Global frequency: app → count
        self._global_freq: Dict[str, int] = defaultdict(int)
        # Vocabulary size (for Laplace denominator)
        self._vocab: set = set()

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, events: List[str]) -> None:
        """
        Train on a chronological sequence of app identifiers.

        Args:
            events: Ordered list of app package names (train split).
        """
        self._m2_counts.clear()
        self._m1_counts.clear()
        self._global_freq.clear()
        self._vocab.clear()

        for app in events:
            self._vocab.add(app)
            self._global_freq[app] += 1

        for i in range(1, len(events)):
            self._m1_counts[events[i - 1]][events[i]] += 1
            if i >= 2:
                bigram = (events[i - 2], events[i - 1])
                self._m2_counts[bigram][events[i]] += 1

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(
        self,
        current: str,
        prev: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Predict the top-k next apps with confidence scores.

        Args:
            current: The most recently launched app.
            prev:    The app launched before current (optional).
            top_k:   Override instance top_k if provided.

        Returns:
            List of (app, confidence) tuples, sorted descending by confidence.
        """
        k = top_k or self.top_k
        V = len(self._vocab) or 1
        alpha = self.laplace_alpha

        candidates: Dict[str, float] = {}

        # Order-2: use (prev, current) context if available
        if prev is not None:
            bigram = (prev, current)
            if bigram in self._m2_counts:
                m2 = self._m2_counts[bigram]
                total = sum(m2.values())
                for app in self._vocab:
                    cnt = m2.get(app, 0)
                    # Laplace-smoothed probability
                    candidates[app] = (cnt + alpha) / (total + alpha * V)

        # Order-1: blend or fallback
        if not candidates and current in self._m1_counts:
            m1 = self._m1_counts[current]
            total = sum(m1.values())
            for app in self._vocab:
                cnt = m1.get(app, 0)
                candidates[app] = (cnt + alpha) / (total + alpha * V)
        elif current in self._m1_counts:
            # Blend order-2 and order-1 (interpolation weight 0.7 / 0.3)
            m1 = self._m1_counts[current]
            total_m1 = sum(m1.values())
            for app in self._vocab:
                p1 = (m1.get(app, 0) + alpha) / (total_m1 + alpha * V)
                # Interpolate: 70% order-2, 30% order-1
                candidates[app] = 0.7 * candidates.get(app, 0.0) + 0.3 * p1

        # Global frequency fallback
        if not candidates:
            total_global = sum(self._global_freq.values()) or 1
            for app in self._vocab:
                candidates[app] = self._global_freq.get(app, 0) / total_global

        # Remove current app from candidates (don't predict self-loop)
        candidates.pop(current, None)

        # Sort and return top-k
        ranked = sorted(candidates.items(), key=lambda x: -x[1])[:k]
        return ranked

    def predict_apps(
        self,
        current: str,
        prev: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """Return only app names (no scores), for cache prefetch use."""
        return [app for app, _ in self.predict(current, prev, top_k)]

    def confidence(
        self,
        current: str,
        prev: Optional[str] = None,
    ) -> float:
        """
        Return confidence in the top-1 prediction.

        High confidence = model has seen this context many times.
        Low confidence = sparse context, falling back to global freq.
        """
        results = self.predict(current, prev, top_k=1)
        if not results:
            return 0.0
        return results[0][1]

    # ── Serialisation helpers ─────────────────────────────────────────────

    def get_vocab_size(self) -> int:
        return len(self._vocab)

    def get_m2_state_count(self) -> int:
        return len(self._m2_counts)

    def get_m1_state_count(self) -> int:
        return len(self._m1_counts)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"VariableOrderMarkov("
            f"vocab={self.get_vocab_size()}, "
            f"m1_states={self.get_m1_state_count()}, "
            f"m2_states={self.get_m2_state_count()}, "
            f"alpha={self.laplace_alpha})"
        )
