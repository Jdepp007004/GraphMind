"""
src/security/sensitivity_model.py

4-level numeric sensitivity model for GraphMind v2.

Coexists with the existing ContextBoundaryEnforcer. Does not replace it.
This module adds numeric sensitivity semantics on top of the existing
category-based classification.

Sensitivity levels (from config/settings.py):
  SENSITIVITY_PUBLIC    = 0  (entertainment, gaming, shopping)
  SENSITIVITY_PERSONAL  = 1  (social, productivity, enterprise)
  SENSITIVITY_FINANCIAL = 2  (banking, payment apps)
  SENSITIVITY_HEALTH    = 3  (health, medical apps)

Flush rule:
  When the user transitions from a higher-sensitivity context to a
  lower-sensitivity context (next_level < current_level), the HOT
  and WARM caches must be flushed to prevent sensitive app data from
  being accessible in a lower-trust context.

  Example:
    HEALTH(3) → ENTERTAINMENT(0): FLUSH (level drops 3→0)
    FINANCIAL(2) → SOCIAL(1):     FLUSH (level drops 2→1)
    PERSONAL(1) → FINANCIAL(2):   NO FLUSH (level rises)
    SOCIAL(1) → GAMING(0):        FLUSH (level drops 1→0)

  This is strictly more expressive than the existing ContextBoundaryEnforcer,
  which only distinguishes SENSITIVE vs CONSUMER categories.

Integration with existing code:
  SensitivityModel is used by EvaluatorV2 to measure:
    1. Number of flush events triggered per evaluation run.
    2. Whether the existing ContextBoundaryEnforcer agrees with the numeric model.
    3. Security overhead: how many HOT evictions are security-triggered vs
       capacity-triggered.
"""

import logging
from typing import Dict, List, Optional, Tuple

from config import settings
from src.security.classification_guard import ClassificationGuard

logger = logging.getLogger(__name__)


def _load_taxonomy() -> dict:
    """Load app taxonomy JSON. Returns empty dict on failure."""
    import json
    try:
        with open(settings.APP_TAXONOMY_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning(f"SensitivityModel: could not load taxonomy: {exc}")
        return {}


class SensitivityModel:
    """
    Assigns numeric sensitivity levels to app nodes and enforces
    context transitions via flush rules.

    This model works on app_id strings (not node IDs) for simplicity.
    The same sensitivity level applies to all graph nodes for a given app_id.
    """

    def __init__(self) -> None:
        """Initialise with the default CATEGORY_SENSITIVITY_MAP from settings."""
        # Cache: app_id → sensitivity level (computed on first access)
        self._sensitivity_cache: Dict[str, int] = {}
        # Track current session context
        self._current_level: Optional[int] = None
        self._current_app_id: Optional[str] = None
        # Audit log
        self._flush_events: List[dict] = []
        self._total_transitions: int = 0
        # ClassificationGuard for taxonomy lookup
        self._taxonomy = _load_taxonomy()
        self._guard = ClassificationGuard(self._taxonomy)

    def get_sensitivity(self, app_id: str) -> int:
        """
        Return the numeric sensitivity level for an app_id.

        Lookup order:
          1. In-memory cache (fastest).
          2. settings.CATEGORY_SENSITIVITY_MAP via app_taxonomy category.
          3. Default: SENSITIVITY_FINANCIAL (conservative for unknowns).

        Args:
            app_id: Android package ID string.

        Returns:
            int ∈ {0, 1, 2, 3} (SENSITIVITY_PUBLIC .. SENSITIVITY_HEALTH).
        """
        if app_id in self._sensitivity_cache:
            return self._sensitivity_cache[app_id]

        # Get category from ClassificationGuard (taxonomy lookup)
        category = self._guard.classify(app_id)
        level = settings.CATEGORY_SENSITIVITY_MAP.get(
            category,
            settings.SENSITIVITY_FINANCIAL  # conservative default for unknowns
        )
        self._sensitivity_cache[app_id] = level
        return level

    def get_category(self, app_id: str) -> str:
        """Return the category string for an app_id via the taxonomy."""
        return self._guard.classify(app_id)

    def should_flush(
        self,
        current_app_id: str,
        next_app_id: str,
    ) -> Tuple[bool, str]:
        """
        Determine whether a cache flush is required for the given transition.

        Flush rule:
          flush = (sensitivity(next_app_id) < sensitivity(current_app_id))

        Args:
            current_app_id: App being left.
            next_app_id:    App being launched next.

        Returns:
            (should_flush: bool, reason: str)
            reason is human-readable for audit log purposes.
        """
        current_level = self.get_sensitivity(current_app_id)
        next_level = self.get_sensitivity(next_app_id)

        if next_level < current_level:
            reason = (
                f"Sensitivity drop: {current_app_id} "
                f"(level={current_level}) → {next_app_id} "
                f"(level={next_level})"
            )
            return True, reason

        return False, ""

    def on_app_launched(
        self,
        app_id: str,
        memory_manager=None,
    ) -> dict:
        """
        Process an app launch event. Evaluates flush rule and performs
        flush if required (via MemoryManager if provided).

        Args:
            app_id:          Newly launched app ID.
            memory_manager:  MemoryManager instance. If None, flush is
                             logged but not executed on real memory.

        Returns:
            dict with:
              flushed      : bool  — whether a flush was triggered
              prev_app     : str   — previously active app (or None)
              next_app     : str   — app_id
              prev_level   : int   — sensitivity level of prev app
              next_level   : int   — sensitivity level of next app
              reason       : str   — flush reason (empty if no flush)
        """
        result: dict = {
            "flushed": False,
            "prev_app": self._current_app_id,
            "next_app": app_id,
            "prev_level": (
                self.get_sensitivity(self._current_app_id)
                if self._current_app_id else -1
            ),
            "next_level": self.get_sensitivity(app_id),
            "reason": "",
        }

        if self._current_app_id is not None:
            flush_needed, reason = self.should_flush(self._current_app_id, app_id)
            if flush_needed:
                result["flushed"] = True
                result["reason"] = reason
                self._flush_events.append({
                    "prev_app": self._current_app_id,
                    "prev_level": result["prev_level"],
                    "next_app": app_id,
                    "next_level": result["next_level"],
                    "reason": reason,
                })
                logger.info(f"SensitivityModel: FLUSH triggered — {reason}")

                if memory_manager is not None:
                    try:
                        memory_manager.flush_hot_tier()
                        memory_manager.flush_warm_tier()
                    except AttributeError:
                        logger.debug(
                            "MemoryManager does not implement flush_hot_tier/flush_warm_tier. "
                            "Flush recorded in audit log only."
                        )

        self._current_app_id = app_id
        self._current_level = self.get_sensitivity(app_id)
        self._total_transitions += 1
        return result

    def flush_rate(self) -> float:
        """
        Return the fraction of transitions that triggered a flush.

        flush_rate = len(flush_events) / total_transitions

        Returns:
            float ∈ [0, 1]. 0.0 = no security flushes.
        """
        if self._total_transitions == 0:
            return 0.0
        return len(self._flush_events) / self._total_transitions

    def get_flush_events(self) -> List[dict]:
        """Return the full audit log of all flush events."""
        return list(self._flush_events)

    def reset(self) -> None:
        """Reset session state. Sensitivity cache is preserved."""
        self._current_level = None
        self._current_app_id = None
        self._flush_events.clear()
        self._total_transitions = 0

    def summary(self) -> dict:
        """
        Return a summary of the security model's activity.

        Returns:
            dict with: total_transitions, flush_count, flush_rate,
            unique_apps_seen, sensitivity_distribution.
        """
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for level in self._sensitivity_cache.values():
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            "total_transitions":      self._total_transitions,
            "flush_count":            len(self._flush_events),
            "flush_rate":             round(self.flush_rate(), 4),
            "unique_apps_seen":       len(self._sensitivity_cache),
            "sensitivity_distribution": {
                "PUBLIC(0)":    level_counts.get(0, 0),
                "PERSONAL(1)":  level_counts.get(1, 0),
                "FINANCIAL(2)": level_counts.get(2, 0),
                "HEALTH(3)":    level_counts.get(3, 0),
            },
        }
