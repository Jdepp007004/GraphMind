"""
src/android/audio_collector.py

Detects headphone and Bluetooth audio device connection state via ADB.
"""

import logging
import re
from typing import Optional

from src.android.adb_connector import ADBConnector

logger = logging.getLogger(__name__)


class AudioCollector:
    """Reads wired and Bluetooth headphone connection state via dumpsys audio."""

    def __init__(self, connector: ADBConnector, serial: Optional[str] = None) -> None:
        self.connector = connector
        self.serial = serial

    def collect(self) -> dict:
        """
        Return dict:
        {
          'headphones_wired': bool,
          'headphones_bluetooth': bool,
          'headphones_any': bool,     # True if either is True
          'output_device': str        # e.g. 'EARPIECE', 'SPEAKER', 'WIRED_HEADSET', 'BT_A2DP'
        }
        """
        result = {
            "headphones_wired": False,
            "headphones_bluetooth": False,
            "headphones_any": False,
            "output_device": "SPEAKER"
        }

        ok, output = self.connector.shell(
            "dumpsys audio", serial=self.serial, timeout=10
        )
        if not ok:
            return result

        lower = output.lower()

        # Wired detection
        if "wired_headset" in lower or "wiredheadset" in lower:
            # Check if it's actually connected
            m = re.search(r"wired_headset.*?connected.*?(\d)", lower)
            if m and m.group(1) == "1":
                result["headphones_wired"] = True
            elif "wiredheadset: true" in lower or "headset is on" in lower:
                result["headphones_wired"] = True

        # Bluetooth detection
        if "bt_a2dp" in lower or "a2dp" in lower:
            m = re.search(r"(bt_a2dp|a2dp).*?(connected|true)", lower)
            if m:
                result["headphones_bluetooth"] = True

        # Also check headphone jack state via sys/class
        ok2, state_out = self.connector.shell(
            "cat /sys/class/switch/h2w/state 2>/dev/null || echo 0",
            serial=self.serial
        )
        if ok2:
            state_val = state_out.strip()
            if state_val in ("1", "2"):  # 1=headset, 2=headphones without mic
                result["headphones_wired"] = True

        result["headphones_any"] = result["headphones_wired"] or result["headphones_bluetooth"]

        if result["headphones_bluetooth"]:
            result["output_device"] = "BT_A2DP"
        elif result["headphones_wired"]:
            result["output_device"] = "WIRED_HEADSET"

        return result
