"""
src/android/telemetry_collector.py

Orchestrates all individual collectors into a single polling loop.
Runs continuously (or on-demand) and publishes events through TelemetryEventAdapter.
"""

import logging
import threading
import time
from typing import Optional

from src.android.adb_connector import ADBConnector
from src.android.device_detector import DeviceDetector, DeviceInfo
from src.android.battery_collector import BatteryCollector
from src.android.usage_stats_collector import UsageStatsCollector
from src.android.audio_collector import AudioCollector
from src.android.screen_collector import ScreenCollector
from src.android.calendar_collector import CalendarCollector
from src.android.telemetry_event_adapter import TelemetryEventAdapter

logger = logging.getLogger(__name__)


class TelemetryCollector:
    """
    Orchestrates all sensor collectors and drives the polling loop.
    Publishes all events through TelemetryEventAdapter.
    """

    DEFAULT_POLL_INTERVAL = 5  # seconds between polls

    def __init__(self, user_id: str, connector: ADBConnector,
                 device: DeviceInfo, poll_interval: int = DEFAULT_POLL_INTERVAL) -> None:
        self.user_id = user_id
        self.connector = connector
        self.device = device
        self.poll_interval = poll_interval
        self.serial = device.serial

        # Adapter publishes to EventBus
        self.adapter = TelemetryEventAdapter(user_id, device_serial=self.serial)

        # Individual collectors
        self.battery = BatteryCollector(connector, serial=self.serial)
        self.usage = UsageStatsCollector(connector, serial=self.serial)
        self.audio = AudioCollector(connector, serial=self.serial)
        self.screen = ScreenCollector(connector, serial=self.serial)
        self.calendar = CalendarCollector(connector, serial=self.serial)

        # State tracking
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_battery: dict = {}
        self._last_audio: dict = {}
        self._last_screen: dict = {}
        self._last_calendar: dict = {}

    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            logger.warning("TelemetryCollector already running")
            return
        self._running = True
        # Publish device connection event
        self.adapter.publish_device_connected(self.device.to_dict())
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name=f"TelemetryCollector-{self.user_id}"
        )
        self._thread.start()
        logger.info(f"TelemetryCollector started for {self.user_id} on device {self.serial}")

    def stop(self) -> None:
        """Stop the polling loop gracefully."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.poll_interval + 2)
        logger.info(f"TelemetryCollector stopped for {self.user_id}")

    def collect_once(self) -> dict:
        """
        Perform a single collection pass and return all collected data.
        Does NOT publish — for diagnostics and testing only.
        """
        battery_data = self.battery.collect()
        foreground_app = self.usage.get_foreground_app()
        audio_data = self.audio.collect()
        screen_data = self.screen.collect()
        calendar_data = self.calendar.collect()
        return {
            "battery": battery_data,
            "foreground_app": foreground_app,
            "audio": audio_data,
            "screen": screen_data,
            "calendar": calendar_data,
            "device_serial": self.serial,
            "user_id": self.user_id,
            "timestamp": time.time(),
        }

    def _poll_loop(self) -> None:
        """Main background loop. Polls all collectors and publishes events."""
        logger.info(f"Telemetry poll loop started (interval={self.poll_interval}s)")
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Telemetry poll error: {e}")
            time.sleep(self.poll_interval)

    def _tick(self) -> None:
        """One polling tick: collect all sensors and publish changed events."""
        battery_data = self.battery.collect()
        foreground_app = self.usage.get_foreground_app()
        audio_data = self.audio.collect()
        screen_data = self.screen.collect()
        calendar_data = self.calendar.collect()

        bat_pct = battery_data.get("battery_pct", 100.0)
        headphones = audio_data.get("headphones_any", False)
        cal_mins = calendar_data.get("calendar_event_in_mins")

        # Always try to publish app launched (adapter deduplicates)
        if foreground_app:
            self.adapter.publish_app_launched(
                package_name=foreground_app,
                battery=bat_pct,
                headphones=headphones,
                calendar_event_in_mins=cal_mins
            )

        # Battery — publish on any change
        if battery_data != self._last_battery:
            self.adapter.publish_battery_updated(
                battery_pct=bat_pct,
                charging=battery_data.get("charging", False),
                power_saver=battery_data.get("power_saver", False)
            )
            self._last_battery = battery_data

        # Headphones — publish only on state change
        if audio_data.get("headphones_any") != self._last_audio.get("headphones_any"):
            if audio_data.get("headphones_any"):
                self.adapter.publish_headphones_connected(
                    wired=audio_data.get("headphones_wired", False),
                    bluetooth=audio_data.get("headphones_bluetooth", False)
                )
            self._last_audio = audio_data

        # Screen state change
        screen_key = (screen_data.get("screen_on"), screen_data.get("screen_locked"))
        last_screen_key = (self._last_screen.get("screen_on"), self._last_screen.get("screen_locked"))
        if screen_key != last_screen_key:
            self.adapter.publish_screen_state(
                screen_on=screen_data.get("screen_on", True),
                screen_locked=screen_data.get("screen_locked", False)
            )
            self._last_screen = screen_data

        # Calendar proximity
        if (calendar_data.get("has_upcoming_event") and
                cal_mins != self._last_calendar.get("calendar_event_in_mins")):
            self.adapter.publish_calendar_event(
                minutes_until=cal_mins,
                event_title=calendar_data.get("next_event_title", "")
            )
            self._last_calendar = calendar_data
