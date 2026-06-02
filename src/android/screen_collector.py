"""
src/android/screen_collector.py

Detects screen state (on/off/unlock) and network state via ADB.
"""

import logging
import re
from typing import Optional

from src.android.adb_connector import ADBConnector

logger = logging.getLogger(__name__)


class ScreenCollector:
    """Reads screen on/off/locked state and WiFi/network info."""

    def __init__(self, connector: ADBConnector, serial: Optional[str] = None) -> None:
        self.connector = connector
        self.serial = serial

    def collect(self) -> dict:
        """
        Return dict:
        {
          'screen_on': bool,
          'screen_locked': bool,
          'screen_unlocked': bool,
          'wifi_connected': bool,
          'wifi_ssid': str,
          'network_type': str    # 'WIFI', 'MOBILE', 'NONE'
        }
        """
        result = {
            "screen_on": True,
            "screen_locked": False,
            "screen_unlocked": True,
            "wifi_connected": False,
            "wifi_ssid": "",
            "network_type": "NONE"
        }

        # Screen state via dumpsys power
        ok, power_out = self.connector.shell(
            "dumpsys power", serial=self.serial, timeout=8
        )
        if ok:
            lower = power_out.lower()
            # mWakefulness: Awake / Dozing / Asleep
            if "mwakefulness=awake" in lower:
                result["screen_on"] = True
            elif "mwakefulness=asleep" in lower or "mwakefulness=dozing" in lower:
                result["screen_on"] = False
            # Screen interactivity
            if "minteractive=false" in lower:
                result["screen_on"] = False

        # Lock state via dumpsys window
        ok2, win_out = self.connector.shell(
            "dumpsys window policy | grep -i keyguard",
            serial=self.serial,
            timeout=8
        )
        if ok2:
            lower2 = win_out.lower()
            result["screen_locked"] = (
                "mshowing=true" in lower2 or
                "keyguardshowing=true" in lower2 or
                "isshowing=true" in lower2
            )
            result["screen_unlocked"] = not result["screen_locked"]

        # Network / WiFi state
        ok3, net_out = self.connector.shell(
            "dumpsys connectivity | grep -E 'NetworkInfo|type=WIFI|type=MOBILE' | head -5",
            serial=self.serial,
            timeout=8
        )
        if ok3:
            lower3 = net_out.lower()
            if "type: wifi" in lower3 and "connected" in lower3:
                result["wifi_connected"] = True
                result["network_type"] = "WIFI"
            elif "type: mobile" in lower3 and "connected" in lower3:
                result["network_type"] = "MOBILE"

        # WiFi SSID
        if result["wifi_connected"]:
            ok4, ssid_out = self.connector.shell(
                "dumpsys wifi | grep -i 'SSID' | head -2",
                serial=self.serial,
                timeout=5
            )
            if ok4:
                m = re.search(r'SSID: "?([^",\s]+)"?', ssid_out)
                if m:
                    result["wifi_ssid"] = m.group(1)

        return result
