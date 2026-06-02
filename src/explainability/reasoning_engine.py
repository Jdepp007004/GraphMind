"""
src/explainability/reasoning_engine.py

Builds human-readable reasoning strings from graph state and event context.
Purely functional: takes data structures, returns reason lists.
Does NOT access EventBus or modify any core state.
"""

import logging
from typing import List, Optional, Dict, Any

from config import settings

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Generates explanation reason lists from GraphMind context data.
    All methods are pure (no side effects).
    """

    # Day-of-week pattern labels
    _TIME_BUCKET_LABELS = {
        range(0, 12): "night hours",
        range(12, 20): "early morning",
        range(20, 28): "morning commute hours",
        range(28, 36): "midday",
        range(36, 44): "afternoon",
        range(44, 48): "evening",
    }

    def _time_label(self, time_bucket: int) -> str:
        """Convert a 30-min time bucket (0-47) to a human-readable label."""
        for r, label in self._TIME_BUCKET_LABELS.items():
            if time_bucket in r:
                return label
        return f"time bucket {time_bucket}"

    def _battery_label(self, battery_pct: float) -> str:
        """Convert battery percentage to a human-readable prefetch label."""
        if battery_pct >= 80:
            return "high battery (allows prefetch)"
        if battery_pct >= 50:
            return "moderate battery"
        if battery_pct >= 20:
            return "low battery (reduced prefetch)"
        return "critical battery (prefetch suppressed)"

    def reasons_for_preload(self, app_id: str,
                             transition_prob: float,
                             battery: float,
                             time_bucket: int,
                             access_count: int,
                             headphones: bool = False,
                             calendar_mins: Optional[int] = None,
                             weekend: bool = False,
                             category: str = "utility") -> List[str]:
        """
        Build reason list for why an app was preloaded into HOT.
        """
        reasons = []
        app_short = app_id.split(".")[-1] if "." in app_id else app_id

        # Transition probability
        if transition_prob > 0:
            reasons.append(f"transition probability {transition_prob:.2f} from previous app")

        # Access count
        if access_count >= 10:
            reasons.append(f"frequently accessed ({access_count} times in session)")
        elif access_count >= 3:
            reasons.append(f"used {access_count} times in recent history")

        # Time pattern
        time_lbl = self._time_label(time_bucket)
        reasons.append(f"{time_lbl} usage pattern detected")

        # Weekend pattern
        if weekend:
            reasons.append("weekend behavioral pattern")
        else:
            reasons.append("weekday behavioral pattern")

        # Headphones
        if headphones and category in ("entertainment", "social"):
            reasons.append("headphones connected (media app affinity)")

        # Calendar proximity
        if calendar_mins is not None and calendar_mins <= 30:
            reasons.append(f"calendar event in {calendar_mins} minutes (productivity context)")

        # Battery
        reasons.append(self._battery_label(battery))

        # Category context
        if category in settings.SENSITIVE_CATEGORIES:
            reasons.append(f"sensitive category ({category}) kept in secure tier")

        return reasons

    def reasons_for_promotion(self, app_id: str, from_tier: str,
                               access_count: int,
                               time_bucket: int,
                               kl_divergence: float = 0.0) -> List[str]:
        """Build reasons for HOT tier promotion."""
        reasons = []
        if from_tier == "warm":
            reasons.append(f"elevated from WARM tier (warm hit confirmed)")
        elif from_tier == "cold":
            reasons.append(f"restored from COLD persistent store")

        if access_count >= 5:
            reasons.append(f"high access frequency ({access_count} recent accesses)")

        reasons.append(f"{self._time_label(time_bucket)} — active usage window")

        if kl_divergence > 0.1:
            reasons.append(f"behavioral drift detected (KL={kl_divergence:.3f}) — adaptive promotion")

        return reasons

    def reasons_for_demotion(self, app_id: str,
                              hot_pressure: float,
                              days_inactive: int = 0) -> List[str]:
        """Build reasons for HOT tier demotion to WARM."""
        reasons = []
        if hot_pressure > 0.8:
            reasons.append(f"HOT tier pressure {hot_pressure:.0%} — evicting LRU")
        if days_inactive > 0:
            reasons.append(f"not accessed for {days_inactive} days")
        reasons.append("demoted to WARM to free HOT capacity")
        return reasons

    def reasons_for_flush(self, from_category: str, to_category: str,
                           flushed_count: int) -> List[str]:
        """Build reasons for security context flush."""
        reasons = [
            f"sensitive context boundary: {from_category} to {to_category} transition",
            f"privacy policy requires clearing {from_category} data",
            f"{flushed_count} node(s) removed from HOT cache",
            "SENSITIVE category isolation enforced",
        ]
        return reasons

    def reasons_for_prediction(self, app_id: str,
                                source_app: str,
                                transition_prob: float,
                                rank: int,
                                battery: float,
                                time_bucket: int) -> List[str]:
        """Build reasons for a next-app prediction."""
        reasons = []
        reasons.append(f"rank #{rank} in transition graph from {source_app.split('.')[-1]}")
        reasons.append(f"transition probability {transition_prob:.2f}")
        reasons.append(f"{self._time_label(time_bucket)} — contextual match")
        reasons.append(self._battery_label(battery))
        return reasons

    def build_summary(self, action: str, app_id: str, reasons: List[str],
                       confidence: float) -> str:
        """Format the full explanation text block."""
        app_short = app_id.split(".")[-1] if "." in app_id else app_id
        lines = [f"{app_short} {action} because:"]
        for r in reasons:
            lines.append(f"  - {r}")
        lines.append(f"  confidence: {confidence:.0%}")
        return "\n".join(lines)
