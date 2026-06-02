"""
src/agents/drift_visualizer.py

Visualization layer ONLY for the existing drift detection system.
Wraps DriftDetectorAgent to provide dashboard-ready KL history and analytics.
Does NOT reimplement KL divergence logic.
"""

import logging
import time
from collections import deque
from typing import List, Dict, Optional, Deque

from config import settings
from src.core.event_bus import EventBus, TOPIC_DRIFT_DETECTED, TOPIC_APP_LAUNCHED

logger = logging.getLogger(__name__)

# Drift lifecycle state machine
STATE_NORMAL = "Normal"
STATE_DRIFT = "Drift"
STATE_KL_SPIKE = "KL Spike"
STATE_ADAPTATION = "Adaptation"
STATE_CONVERGENCE = "Convergence"


class DriftEvent:
    """Record of a single drift detection event for timeline rendering."""

    def __init__(self, timestamp: float, kl_value: float, state: str,
                 user_id: str) -> None:
        self.timestamp = timestamp
        self.kl_value = kl_value
        self.state = state
        self.user_id = user_id

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "kl_value": self.kl_value,
            "state": self.state,
            "user_id": self.user_id,
        }


class DriftVisualizer:
    """
    Monitors KL divergence history in real time by subscribing to EventBus.
    Wraps an existing DriftDetectorAgent to read its state.
    Provides metrics for the Drift Analytics dashboard tab.

    Does NOT recompute KL divergence — reads from the agent's output.
    """

    def __init__(self, user_id: str, drift_agent=None,
                 max_history: int = 500) -> None:
        """
        user_id: target user.
        drift_agent: existing DriftDetectorAgent instance (optional).
                     If None, only live EventBus events are tracked.
        max_history: max number of KL readings to keep.
        """
        self.user_id = user_id
        self.drift_agent = drift_agent
        self._kl_history: Deque[dict] = deque(maxlen=max_history)
        self._drift_events: List[DriftEvent] = []
        self._current_state: str = STATE_NORMAL
        self._last_kl: float = 0.0
        self._adaptation_start_ts: Optional[float] = None
        self._convergence_ts: Optional[float] = None
        self._recovery_times: List[float] = []
        self._event_counter: int = 0

        bus = EventBus.get_instance()
        bus.subscribe(TOPIC_DRIFT_DETECTED, self._on_drift_detected)
        bus.subscribe(TOPIC_APP_LAUNCHED, self._on_app_launched)

    def _on_app_launched(self, payload: dict) -> None:
        """Sample KL from the drift agent after each app launch event."""
        if payload.get("user_id") != self.user_id:
            return
        self._event_counter += 1
        kl = 0.0
        if self.drift_agent is not None:
            kl = self.drift_agent.compute_kl_divergence()
        ts = float(payload.get("timestamp", time.time()))
        self._last_kl = kl
        self._kl_history.append({
            "timestamp": ts,
            "kl_value": kl,
            "event_number": self._event_counter,
            "state": self._classify_state(kl),
        })

    def _on_drift_detected(self, payload: dict) -> None:
        """Record a drift spike event."""
        if payload.get("user_id") != self.user_id:
            return
        kl = float(payload.get("kl_divergence", 0.0))
        ts = float(payload.get("timestamp", time.time()))
        event = DriftEvent(timestamp=ts, kl_value=kl,
                          state=STATE_KL_SPIKE, user_id=self.user_id)
        self._drift_events.append(event)
        self._current_state = STATE_DRIFT
        self._adaptation_start_ts = ts
        self._convergence_ts = None
        logger.debug(f"DriftVisualizer: KL spike {kl:.4f} for {self.user_id}")

    def _classify_state(self, kl: float) -> str:
        """Classify the current system state from KL value and history."""
        threshold = settings.DRIFT_KL_THRESHOLD
        if kl <= threshold * 0.3:
            if self._current_state in (STATE_ADAPTATION, STATE_DRIFT):
                if self._adaptation_start_ts is not None:
                    if self._convergence_ts is None:
                        self._convergence_ts = time.time()
                        recovery = time.time() - self._adaptation_start_ts
                        self._recovery_times.append(recovery)
                self._current_state = STATE_CONVERGENCE
            else:
                self._current_state = STATE_NORMAL
        elif kl > threshold:
            self._current_state = STATE_KL_SPIKE if kl > threshold * 1.5 else STATE_DRIFT
        elif threshold * 0.3 < kl <= threshold:
            if self._current_state in (STATE_KL_SPIKE, STATE_DRIFT):
                self._current_state = STATE_ADAPTATION
        return self._current_state

    # ── Public API ─────────────────────────────────────────────────────────

    def get_kl_history(self, limit: int = 100) -> List[dict]:
        """Return KL history as list of dicts (newest last) for Plotly."""
        history = list(self._kl_history)
        return history[-limit:]

    def get_drift_events(self) -> List[dict]:
        """Return all recorded drift spike events."""
        return [e.to_dict() for e in self._drift_events]

    def get_current_state(self) -> str:
        """Return the current drift lifecycle state."""
        return self._current_state

    def get_adaptation_metrics(self) -> dict:
        """
        Return adaptation speed and convergence metrics.
        {
          'total_drift_events': int,
          'avg_recovery_time_s': float,
          'min_recovery_time_s': float,
          'max_recovery_time_s': float,
          'current_kl': float,
          'current_state': str,
          'adaptation_half_life_events': int,  # events to halve KL after spike
        }
        """
        drift_count = len(self._drift_events)
        avg_recovery = (sum(self._recovery_times) / len(self._recovery_times)
                        if self._recovery_times else 0.0)
        # Compute adaptation half-life: events from spike to KL/2
        half_life = self._estimate_half_life()
        return {
            "total_drift_events": drift_count,
            "avg_recovery_time_s": round(avg_recovery, 2),
            "min_recovery_time_s": round(min(self._recovery_times, default=0.0), 2),
            "max_recovery_time_s": round(max(self._recovery_times, default=0.0), 2),
            "current_kl": round(self._last_kl, 4),
            "current_state": self._current_state,
            "adaptation_half_life_events": half_life,
        }

    def _estimate_half_life(self) -> int:
        """Estimate number of events to halve KL after a spike."""
        history = list(self._kl_history)
        threshold = settings.DRIFT_KL_THRESHOLD
        in_spike = False
        spike_kl = 0.0
        spike_event = 0
        for entry in history:
            kl = entry["kl_value"]
            if kl > threshold and not in_spike:
                in_spike = True
                spike_kl = kl
                spike_event = entry["event_number"]
            elif in_spike and kl <= spike_kl / 2.0:
                return entry["event_number"] - spike_event
        return 0

    def get_timeline_data(self) -> List[dict]:
        """
        Return the full KL timeline with lifecycle state annotations.
        For the Drift Analytics tab scrubber.
        """
        return self.get_kl_history(limit=500)

    def get_state_transitions(self) -> List[dict]:
        """
        Return list of state transition events for Plotly annotations.
        """
        transitions = []
        prev_state = None
        for entry in self._kl_history:
            state = entry.get("state", STATE_NORMAL)
            if state != prev_state:
                transitions.append({
                    "event_number": entry["event_number"],
                    "timestamp": entry["timestamp"],
                    "from_state": prev_state or STATE_NORMAL,
                    "to_state": state,
                    "kl_value": entry["kl_value"],
                })
                prev_state = state
        return transitions

    def inject_kl_reading(self, kl_value: float, timestamp: Optional[float] = None) -> None:
        """
        Manually inject a KL reading. Used for replay from simulation logs.
        """
        ts = timestamp or time.time()
        self._event_counter += 1
        self._last_kl = kl_value
        state = self._classify_state(kl_value)
        self._kl_history.append({
            "timestamp": ts,
            "kl_value": kl_value,
            "event_number": self._event_counter,
            "state": state,
        })
        if kl_value > settings.DRIFT_KL_THRESHOLD:
            self._drift_events.append(
                DriftEvent(ts, kl_value, STATE_KL_SPIKE, self.user_id)
            )
