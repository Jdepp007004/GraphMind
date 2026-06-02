"""
src/android/usage_stats_collector.py

Collects foreground app and app launch statistics via ADB.
Uses `adb shell dumpsys usagestats` for recent usage.
"""

import logging
import re
import time
from typing import Optional, List, Dict

from src.android.adb_connector import ADBConnector

logger = logging.getLogger(__name__)


class UsageStatsCollector:
    """
    Reads foreground app and recent app usage from Android's UsageStats service.
    """

    def __init__(self, connector: ADBConnector, serial: Optional[str] = None) -> None:
        self.connector = connector
        self.serial = serial

    def get_foreground_app(self) -> Optional[str]:
        """
        Return the package name of the currently active foreground app.
        Tries multiple methods for compatibility across Android versions.
        Returns None if not determinable.
        """
        # Method 1: dumpsys activity (Android 11+)
        ok, output = self.connector.shell(
            "dumpsys activity activities | grep mResumedActivity",
            serial=self.serial,
            timeout=8
        )
        if ok and output.strip():
            pkg = _parse_package_from_activity(output)
            if pkg:
                return pkg

        # Method 2: window manager (fallback)
        ok2, out2 = self.connector.shell(
            "dumpsys window windows | grep -E 'mCurrentFocus|mFocusedApp'",
            serial=self.serial,
            timeout=8
        )
        if ok2 and out2.strip():
            pkg = _parse_package_from_window(out2)
            if pkg:
                return pkg

        # Method 3: activity manager tasks
        ok3, out3 = self.connector.shell(
            "dumpsys activity recents | grep 'Recent #0'",
            serial=self.serial,
            timeout=8
        )
        if ok3 and out3.strip():
            pkg = _parse_package_from_recents(out3)
            if pkg:
                return pkg

        return None

    def get_recent_apps(self, count: int = 10) -> List[Dict]:
        """
        Return list of recently used apps:
        [{'package': str, 'last_time_used': int, 'total_time_in_foreground': int}, ...]
        Uses usagestats dump which requires PACKAGE_USAGE_STATS permission.
        """
        ok, output = self.connector.shell(
            "dumpsys usagestats",
            serial=self.serial,
            timeout=15
        )
        apps = []
        if not ok:
            return apps

        current_pkg = None
        last_time = 0
        total_time = 0

        for line in output.splitlines():
            line = line.strip()
            pkg_match = re.search(r"package=([a-z][a-zA-Z0-9._]+)", line)
            if pkg_match:
                if current_pkg and last_time > 0:
                    apps.append({
                        "package": current_pkg,
                        "last_time_used": last_time,
                        "total_time_in_foreground": total_time
                    })
                current_pkg = pkg_match.group(1)
                last_time = 0
                total_time = 0

            if "lastTimeUsed" in line or "mLastTimeUsed" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    last_time = int(m.group(1))
            if "totalTimeInForeground" in line or "mTotalTimeInForeground" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    total_time = int(m.group(1))

        if current_pkg and last_time > 0:
            apps.append({
                "package": current_pkg,
                "last_time_used": last_time,
                "total_time_in_foreground": total_time
            })

        # Sort by last_time_used descending
        apps.sort(key=lambda x: x["last_time_used"], reverse=True)
        return apps[:count]


def _parse_package_from_activity(output: str) -> Optional[str]:
    """Parse package name from mResumedActivity dump line."""
    # Pattern: ActivityRecord{... pkg/Activity ...}
    m = re.search(r"mResumedActivity.*?([a-z][a-zA-Z0-9._]+)/", output)
    if m:
        return m.group(1)
    # Alternate: u0 {package}
    m2 = re.search(r"([a-z][a-zA-Z0-9._]+)/[A-Z]", output)
    if m2:
        return m2.group(1)
    return None


def _parse_package_from_window(output: str) -> Optional[str]:
    """Parse package from mCurrentFocus or mFocusedApp dump."""
    m = re.search(r"mCurrentFocus=Window\{[^}]+\s+([a-z][a-zA-Z0-9._]+)/", output)
    if m:
        return m.group(1)
    m2 = re.search(r"mFocusedApp=.*?([a-z][a-zA-Z0-9._]+)/", output)
    if m2:
        return m2.group(1)
    return None


def _parse_package_from_recents(output: str) -> Optional[str]:
    """Parse package from Recent #0 line."""
    m = re.search(r"([a-z][a-zA-Z0-9._]+)/", output)
    if m:
        return m.group(1)
    return None
