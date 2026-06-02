"""
src/android/battery_collector.py

Collects battery state from a connected Android device via ADB.
"""

import logging
import re
from typing import Optional

from src.android.adb_connector import ADBConnector

logger = logging.getLogger(__name__)


class BatteryCollector:
    """Reads battery level, charging state, and power saver mode via adb shell."""

    def __init__(self, connector: ADBConnector, serial: Optional[str] = None) -> None:
        self.connector = connector
        self.serial = serial

    def collect(self) -> dict:
        """
        Return dict:
        {
          'battery_pct': float,      # 0-100
          'charging': bool,
          'power_saver': bool,
          'temperature_c': float,
          'health': str              # 'Good', 'Overheat', 'Unknown', etc.
        }
        Returns safe defaults on failure.
        """
        ok, output = self.connector.shell("dumpsys battery", serial=self.serial)
        result = {
            "battery_pct": 100.0,
            "charging": False,
            "power_saver": False,
            "temperature_c": 25.0,
            "health": "Unknown"
        }
        if not ok:
            return result

        for line in output.splitlines():
            line = line.strip()
            if line.startswith("level:"):
                result["battery_pct"] = float(_extract_int(line, 100))
            elif line.startswith("status:"):
                # 2=Charging, 3=Discharging, 5=Full
                status = _extract_int(line, 3)
                result["charging"] = status in (2, 5)
            elif line.startswith("temperature:"):
                temp_raw = _extract_int(line, 250)
                result["temperature_c"] = temp_raw / 10.0
            elif line.startswith("health:"):
                health_map = {1: "Unknown", 2: "Good", 3: "Overheat",
                              4: "Dead", 5: "OverVoltage", 7: "Cold"}
                health_val = _extract_int(line, 1)
                result["health"] = health_map.get(health_val, "Unknown")

        # Power saver check (separate call)
        ok2, ps_out = self.connector.shell(
            "settings get global low_power", serial=self.serial
        )
        if ok2:
            result["power_saver"] = ps_out.strip() == "1"

        return result


def _extract_int(line: str, default: int) -> int:
    """Extract the integer value from 'key: value' line."""
    try:
        return int(line.split(":", 1)[1].strip())
    except Exception:
        return default
