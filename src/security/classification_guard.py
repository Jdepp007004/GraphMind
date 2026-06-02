"""
src/security/classification_guard.py

Conservative package classification and retention policy support.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionPolicy:
    hot_retention_events: int = settings.HOT_RETENTION_EVENTS
    warm_retention_events: int = settings.WARM_RETENTION_EVENTS
    cold_retention_days: int = settings.COLD_RETENTION_DAYS
    trace_retention_events: int = settings.TRACE_RETENTION_EVENTS
    graph_retention_days: int = settings.GRAPH_RETENTION_DAYS

    def to_dict(self) -> dict:
        return {
            "hot_retention_events": self.hot_retention_events,
            "warm_retention_events": self.warm_retention_events,
            "cold_retention_days": self.cold_retention_days,
            "trace_retention_events": self.trace_retention_events,
            "graph_retention_days": self.graph_retention_days,
        }


class ClassificationGuard:
    """
    Classifies packages with a conservative unknown-app fallback.

    Unknown packages are isolated as `unknown_sensitive` until the taxonomy is
    updated. This prevents benign defaults such as `utility` from bypassing
    context-boundary flushing.
    """

    def __init__(self, taxonomy: Dict[str, dict],
                 retention_policy: Optional[RetentionPolicy] = None) -> None:
        self.taxonomy = taxonomy
        self.retention_policy = retention_policy or RetentionPolicy()
        self.classification_log: List[dict] = []

    def classify(self, package_name: str, payload_category: Optional[str] = None) -> str:
        if package_name in self.taxonomy:
            category = self.taxonomy[package_name].get("category", settings.UNKNOWN_SENSITIVE_CATEGORY)
            self._log(package_name, category, "taxonomy")
            return category
        category = settings.UNKNOWN_SENSITIVE_CATEGORY
        self._log(package_name, category, "unknown_isolated")
        return category

    def is_sensitive(self, category: str) -> bool:
        return category in settings.SENSITIVE_CATEGORIES

    def retention_summary(self) -> dict:
        return self.retention_policy.to_dict()

    def trim_classification_log(self) -> int:
        limit = self.retention_policy.trace_retention_events
        overflow = max(0, len(self.classification_log) - limit)
        if overflow:
            del self.classification_log[:overflow]
        return overflow

    def _log(self, package_name: str, category: str, source: str) -> None:
        self.classification_log.append({
            "package_name": package_name,
            "category": category,
            "source": source,
        })
        logger.debug("Classified %s as %s via %s", package_name, category, source)
