"""
src/android/telemetry_event_adapter.py

Converts raw ADB/device data into GraphMind EventBus event format.
This is the bridge between real device telemetry and the simulation layer.

Maps Samsung app package names to existing app_taxonomy categories.
Publishes through the existing EventBus without any modification to it.
"""

import json
import logging
import os
import time
from typing import Optional

from config import settings
from src.core.event_bus import (
    EventBus,
    TOPIC_APP_LAUNCHED,
    TOPIC_BATTERY_UPDATED,
    TOPIC_HEADPHONES_CONNECTED,
    TOPIC_CALENDAR_EVENT,
)

logger = logging.getLogger(__name__)

# New topics for real device events (extend existing set without modifying event_bus.py)
TOPIC_SCREEN_STATE_CHANGED = "screen_state_changed"
TOPIC_NETWORK_STATE_CHANGED = "network_state_changed"
TOPIC_DEVICE_CONNECTED = "device_connected"


class TelemetryEventAdapter:
    """
    Converts raw telemetry data from ADB collectors into GraphMind event payloads
    and publishes them through the existing EventBus singleton.

    All payloads conform to the schema expected by existing subscribers
    (BehaviouralGraph, MemoryManager, PrefetchDaemon, ContextBoundaryEnforcer).
    """

    def __init__(self, user_id: str, device_serial: Optional[str] = None) -> None:
        self.user_id = user_id
        self.device_serial = device_serial
        self._taxonomy = self._load_taxonomy()
        self._previous_app: Optional[str] = None
        self._session_day: int = 0
        self._session_start_time: float = time.time()

    def _load_taxonomy(self) -> dict:
        """Load app taxonomy from the existing data/app_taxonomy.json."""
        try:
            with open(settings.APP_TAXONOMY_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load app taxonomy: {e}")
            return {}

    def get_app_category(self, package_name: str) -> str:
        """
        Look up package in taxonomy. Returns category string.
        Falls back to heuristic rules for common packages not in taxonomy.
        """
        # Direct taxonomy lookup
        entry = self._taxonomy.get(package_name, {})
        if entry and "category" in entry:
            return entry["category"]

        # Heuristic fallback by package prefix
        pkg = package_name.lower()
        if any(p in pkg for p in ["bank", "pay", "finance", "insurance", "tax"]):
            return "financial"
        if any(p in pkg for p in ["health", "medical", "fitness", "doctor"]):
            return "health"
        if any(p in pkg for p in ["facebook", "instagram", "twitter", "tiktok",
                                   "snapchat", "whatsapp", "telegram"]):
            return "social"
        if any(p in pkg for p in ["youtube", "netflix", "spotify", "vlc",
                                   "prime", "hotstar"]):
            return "entertainment"
        if any(p in pkg for p in ["gmail", "office", "docs", "calendar",
                                   "slack", "teams", "zoom"]):
            return "productivity"
        if any(p in pkg for p in ["game", "pubg", "free.fire", "clash",
                                   "candy", "chess"]):
            return "gaming"
        if any(p in pkg for p in ["maps", "uber", "ola", "lyft", "navigation"]):
            return "navigation"
        if any(p in pkg for p in ["amazon", "flipkart", "shop", "meesho"]):
            return "shopping"
        if any(p in pkg for p in ["samsung", "google", "android", "settings"]):
            return "utility"
        return "utility"

    def _compute_time_bucket(self) -> int:
        """Convert current hour to 30-min time bucket (0-47)."""
        import datetime
        now = datetime.datetime.now()
        return now.hour * 2 + (1 if now.minute >= 30 else 0)

    def _compute_day_offset(self) -> int:
        """Compute simulation day offset from session start."""
        elapsed = time.time() - self._session_start_time
        return int(elapsed / 86400)  # days since session start

    def _is_weekend(self) -> bool:
        import datetime
        return datetime.datetime.now().weekday() >= 5

    # ── Publish Methods ────────────────────────────────────────────────────

    def publish_app_launched(self, package_name: str, battery: float,
                              headphones: bool = False,
                              calendar_event_in_mins: Optional[int] = None) -> None:
        """
        Convert a real foreground app change into a TOPIC_APP_LAUNCHED event.
        This is the primary integration point — called whenever the foreground
        app changes on the device.
        """
        if package_name == self._previous_app:
            return  # No change, skip

        category = self.get_app_category(package_name)
        payload = {
            "timestamp": time.time(),
            "user_id": self.user_id,
            "app_id": package_name,
            "category": category,
            "battery": battery,
            "time_of_day_bucket": self._compute_time_bucket(),
            "day": self._compute_day_offset(),
            "headphones": headphones,
            "weekend": self._is_weekend(),
            "source": "real_device",
            "device_serial": self.device_serial or "",
        }
        if calendar_event_in_mins is not None:
            payload["calendar_event_in_mins"] = calendar_event_in_mins

        bus = EventBus.get_instance()
        bus.publish(TOPIC_APP_LAUNCHED, payload)
        logger.info(f"TelemetryAdapter: app_launched {package_name} ({category}) battery={battery:.0f}%")
        self._previous_app = package_name

    def publish_battery_updated(self, battery_pct: float, charging: bool,
                                 power_saver: bool) -> None:
        """Publish battery state update."""
        bus = EventBus.get_instance()
        bus.publish(TOPIC_BATTERY_UPDATED, {
            "timestamp": time.time(),
            "user_id": self.user_id,
            "battery": battery_pct,
            "charging": charging,
            "power_saver": power_saver,
            "source": "real_device",
        })

    def publish_headphones_connected(self, wired: bool, bluetooth: bool) -> None:
        """Publish headphone connection event."""
        bus = EventBus.get_instance()
        bus.publish(TOPIC_HEADPHONES_CONNECTED, {
            "timestamp": time.time(),
            "user_id": self.user_id,
            "wired": wired,
            "bluetooth": bluetooth,
            "headphones_connected": True,
            "source": "real_device",
        })

    def publish_calendar_event(self, minutes_until: int, event_title: str = "") -> None:
        """Publish upcoming calendar event proximity."""
        bus = EventBus.get_instance()
        bus.publish(TOPIC_CALENDAR_EVENT, {
            "timestamp": time.time(),
            "user_id": self.user_id,
            "minutes_until_event": minutes_until,
            "event_title": event_title,
            "source": "real_device",
        })

    def publish_screen_state(self, screen_on: bool, screen_locked: bool) -> None:
        """Publish screen state change (extension topic, not in original EventBus)."""
        bus = EventBus.get_instance()
        bus.publish(TOPIC_SCREEN_STATE_CHANGED, {
            "timestamp": time.time(),
            "user_id": self.user_id,
            "screen_on": screen_on,
            "screen_locked": screen_locked,
            "source": "real_device",
        })

    def publish_device_connected(self, device_info: dict) -> None:
        """Publish device connection event with device metadata."""
        bus = EventBus.get_instance()
        bus.publish(TOPIC_DEVICE_CONNECTED, {
            "timestamp": time.time(),
            "user_id": self.user_id,
            "device_info": device_info,
            "source": "real_device",
        })
