"""
src/android/calendar_collector.py

Reads upcoming calendar events via ADB content query.
Calculates proximity in minutes to the nearest event.
"""

import logging
import re
import time
from typing import Optional, List, Dict

from src.android.adb_connector import ADBConnector

logger = logging.getLogger(__name__)

# Android Calendar Events content URI
CALENDAR_URI = "content://com.android.calendar/events"


class CalendarCollector:
    """
    Queries the Android Calendar provider for upcoming events.
    Calculates how many minutes until the next event starts.
    """

    def __init__(self, connector: ADBConnector, serial: Optional[str] = None) -> None:
        self.connector = connector
        self.serial = serial

    def collect(self) -> dict:
        """
        Return dict:
        {
          'has_upcoming_event': bool,
          'minutes_until_next_event': Optional[int],  # None if no event within 2 hours
          'next_event_title': str,
          'calendar_event_in_mins': Optional[int]    # alias for EventBus compatibility
        }
        """
        result = {
            "has_upcoming_event": False,
            "minutes_until_next_event": None,
            "next_event_title": "",
            "calendar_event_in_mins": None
        }

        now_ms = int(time.time() * 1000)
        two_hours_ms = now_ms + 2 * 3600 * 1000

        # Query upcoming events in next 2 hours
        query_cmd = (
            f"content query --uri {CALENDAR_URI} "
            f"--where \"dtstart >= {now_ms} AND dtstart <= {two_hours_ms} AND deleted = 0\" "
            f"--projection _id:title:dtstart "
            f"--sort \"dtstart ASC\" "
            f"--limit 1"
        )
        ok, output = self.connector.shell(query_cmd, serial=self.serial, timeout=10)

        if not ok or "No result found" in output or not output.strip():
            return result

        # Parse the adb content query output
        # Row: 0 _id=N, title=Event Name, dtstart=TIMESTAMP
        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("Row:"):
                continue
            title_m = re.search(r"title=([^,\n]+)", line)
            dtstart_m = re.search(r"dtstart=(\d+)", line)
            if dtstart_m:
                event_start_ms = int(dtstart_m.group(1))
                delta_ms = event_start_ms - now_ms
                if delta_ms >= 0:
                    minutes = int(delta_ms / 60000)
                    result["has_upcoming_event"] = True
                    result["minutes_until_next_event"] = minutes
                    result["calendar_event_in_mins"] = minutes
                    if title_m:
                        result["next_event_title"] = title_m.group(1).strip()
            break  # Only first event

        return result

    def get_events_today(self) -> List[Dict]:
        """
        Return all events for today.
        Returns list of {'title': str, 'dtstart': int, 'minutes_from_now': int}
        """
        now_ms = int(time.time() * 1000)
        end_of_day_ms = now_ms + 24 * 3600 * 1000

        query_cmd = (
            f"content query --uri {CALENDAR_URI} "
            f"--where \"dtstart >= {now_ms} AND dtstart <= {end_of_day_ms} AND deleted = 0\" "
            f"--projection _id:title:dtstart "
            f"--sort \"dtstart ASC\""
        )
        ok, output = self.connector.shell(query_cmd, serial=self.serial, timeout=10)
        events = []
        if not ok:
            return events

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("Row:"):
                continue
            title_m = re.search(r"title=([^,\n]+)", line)
            dtstart_m = re.search(r"dtstart=(\d+)", line)
            if dtstart_m:
                event_start_ms = int(dtstart_m.group(1))
                minutes = max(0, int((event_start_ms - now_ms) / 60000))
                events.append({
                    "title": title_m.group(1).strip() if title_m else "",
                    "dtstart": event_start_ms,
                    "minutes_from_now": minutes
                })
        return events
