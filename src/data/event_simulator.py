"""
src/data/event_simulator.py

Replays a user's saved event log as a real-time stream,
publishing EventBus events. This is the 'Android OS' for the simulation.
"""

import json
import os
import logging
import time
from typing import Optional

from config import settings
from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED

logger = logging.getLogger(__name__)


class EventSimulator:
    """
    Replays the saved synthetic event log for one user.
    Publishes events to the EventBus at each step.
    Tracks current day, time, battery for simulation state.
    """

    def __init__(self, user_id: str) -> None:
        """
        Load event log from USERS_DIR/{user_id}.json.
        Store as self.events: list of event dicts.
        Set self.current_event_index = 0.
        Set self.current_day = 0.
        Set self.bus = EventBus.get_instance().
        Raises FileNotFoundError if user file doesn't exist.
        """
        self.user_id = user_id
        path = os.path.join(settings.USERS_DIR, f"{user_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Event file not found: {path}")
        with open(path) as f:
            self.events: list = json.load(f)
        self.current_event_index: int = 0
        self.current_day: int = 0
        self.bus = EventBus.get_instance()
        self._last_app_id: Optional[str] = None
        self._last_battery: float = 100.0
        logger.debug(f"EventSimulator loaded {len(self.events)} events for {user_id}")

    def step(self) -> Optional[dict]:
        """
        Advance simulation by one event.
        Publish the current event to the EventBus as TOPIC_APP_LAUNCHED.
        Increment self.current_event_index.
        Returns the event dict that was published, or None if simulation is complete.
        """
        if self.current_event_index >= len(self.events):
            return None
        event = self.events[self.current_event_index]
        self.current_event_index += 1
        self._last_app_id = event.get("app_id")
        self._last_battery = event.get("battery", 100.0)
        payload = {
            "timestamp": float(event.get("timestamp", 0.0)),
            "app_id": event.get("app_id", "unknown"),
            "user_id": self.user_id,
            "battery": float(event.get("battery", 100.0)),
            "time_of_day_bucket": int(event.get("time_bucket", 0)),
            "headphones": bool(event.get("headphones", False)),
            "calendar_event_in_mins": event.get("calendar_event_in_mins"),
            "weekend": bool(event.get("weekend", False)),
            "category": event.get("category", "utility"),
            "day": int(event.get("day", self.current_day))
        }
        self.bus.publish(TOPIC_APP_LAUNCHED, payload)
        return event

    def step_day(self) -> list:
        """
        Advance simulation by all events in the next day.
        Calls step() for each event on the current day.
        Increments self.current_day.
        Returns list of all events published for that day.
        """
        day_events = []
        target_day = self.current_day
        while self.current_event_index < len(self.events):
            next_event = self.events[self.current_event_index]
            if int(next_event.get("day", 0)) != target_day:
                break
            result = self.step()
            if result is not None:
                day_events.append(result)
        self.current_day += 1
        return day_events

    def step_all(self) -> None:
        """
        Replay all events in the entire 30-day log.
        Calls step() for each event sequentially.
        Logs progress every 1000 events.
        """
        while self.current_event_index < len(self.events):
            if self.current_event_index % 1000 == 0 and self.current_event_index > 0:
                logger.info(f"EventSimulator: replayed {self.current_event_index}/{len(self.events)} events")
            self.step()

    def reset(self) -> None:
        """
        Reset simulator to day 0, event 0.
        Clears any session state.
        """
        self.current_event_index = 0
        self.current_day = 0
        self._last_app_id = None
        self._last_battery = 100.0

    def get_current_state(self) -> dict:
        """
        Return current simulation state.
        Returns: {'user_id': str, 'current_day': int, 'current_event_index': int,
                  'total_events': int, 'battery': float, 'last_app_id': str | None}
        """
        return {
            "user_id": self.user_id,
            "current_day": self.current_day,
            "current_event_index": self.current_event_index,
            "total_events": len(self.events),
            "battery": self._last_battery,
            "last_app_id": self._last_app_id
        }

    def get_events_for_day(self, day: int) -> list:
        """
        Return all events for a specific day without publishing them.
        Used by benchmarks to get the ground truth sequence.
        """
        return [e for e in self.events if int(e.get("day", -1)) == day]
