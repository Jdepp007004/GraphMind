"""
src/agents/drift_detector_agent.py

Monitors KL-divergence between recent and historical app transition distributions.
"""

import logging
from collections import deque
from typing import Dict, Any

import numpy as np
from scipy.stats import entropy

from config import settings
from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED, TOPIC_DRIFT_DETECTED

logger = logging.getLogger(__name__)


class DriftDetectorAgent:
    """
    Tracks the distribution of app transitions over time.
    Computes KL divergence between a sliding window and historical baseline.
    Triggers learning rate spike if drift detected.
    """

    def __init__(self, user_id: str) -> None:
        """
        Set self.transition_history = deque(maxlen=DRIFT_WINDOW_SIZE * 2)
        Set self.recent_window = deque(maxlen=DRIFT_WINDOW_SIZE)
        Subscribe to TOPIC_APP_LAUNCHED -> _record_transition().
        """
        self.user_id = user_id
        self.transition_history: deque = deque(maxlen=settings.DRIFT_WINDOW_SIZE * 2)
        self.recent_window: deque = deque(maxlen=settings.DRIFT_WINDOW_SIZE)
        bus = EventBus.get_instance()
        bus.subscribe(TOPIC_APP_LAUNCHED, self._record_transition)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main agent function called by LangGraph.
        Compute KL divergence between recent_window and older half of transition_history.
        Update state['kl_divergence'] = computed KL value.
        If KL > DRIFT_KL_THRESHOLD: publish TOPIC_DRIFT_DETECTED.
        Append detection result to state['messages'].
        Return updated state.
        """
        kl = self.compute_kl_divergence()
        state["kl_divergence"] = kl
        state["last_agent"] = "drift_detector"
        msg = {"agent": "drift_detector", "kl_divergence": kl, "drift_detected": kl > settings.DRIFT_KL_THRESHOLD}
        state["messages"].append(msg)
        if kl > settings.DRIFT_KL_THRESHOLD:
            bus = EventBus.get_instance()
            bus.publish(TOPIC_DRIFT_DETECTED, {
                "timestamp": 0.0,
                "kl_divergence": kl,
                "user_id": self.user_id
            })
            logger.info(f"Drift detected for {self.user_id}: KL={kl:.4f}")
        return state

    def compute_kl_divergence(self) -> float:
        """
        Compute KL divergence between recent and historical transition distributions.
        Convert both windows to probability distributions over app_ids.
        Use scipy.stats.entropy(P, Q) for KL(P||Q).
        Returns 0.0 if insufficient data (< DRIFT_WINDOW_SIZE events).
        Add small epsilon (1e-10) to avoid log(0).
        """
        if len(self.recent_window) < settings.DRIFT_WINDOW_SIZE:
            return 0.0
        if len(self.transition_history) < settings.DRIFT_WINDOW_SIZE:
            return 0.0

        # Build vocabulary from both windows
        all_apps = set(self.recent_window) | set(self.transition_history)
        if not all_apps:
            return 0.0

        vocab = sorted(all_apps)
        eps = 1e-10

        # Historical distribution (from transition_history)
        hist_counts = {app: 0 for app in vocab}
        for app in self.transition_history:
            if app in hist_counts:
                hist_counts[app] += 1
        hist_total = sum(hist_counts.values())
        P = np.array([hist_counts[a] / hist_total + eps for a in vocab], dtype=np.float64)
        P /= P.sum()

        # Recent distribution
        rec_counts = {app: 0 for app in vocab}
        for app in self.recent_window:
            if app in rec_counts:
                rec_counts[app] += 1
        rec_total = sum(rec_counts.values())
        Q = np.array([rec_counts[a] / rec_total + eps for a in vocab], dtype=np.float64)
        Q /= Q.sum()

        kl = float(entropy(Q, P))  # KL(Q||P)
        return max(0.0, kl)

    def _record_transition(self, payload: dict) -> None:
        """PRIVATE. EventBus callback. Record app_id into both deques."""
        if payload.get("user_id") != self.user_id:
            return
        app_id = payload.get("app_id", "unknown")
        self.transition_history.append(app_id)
        self.recent_window.append(app_id)
