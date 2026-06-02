"""
src/cli/device_setup.py

Platform detection, ADB verification, and Samsung developer mode instructions.
Used by the wizard as individual composable steps.
"""

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ADB download URLs per platform
ADB_DOWNLOAD_URLS = {
    "Windows": "https://dl.google.com/android/repository/platform-tools-latest-windows.zip",
    "Darwin": "https://dl.google.com/android/repository/platform-tools-latest-darwin.zip",
    "Linux": "https://dl.google.com/android/repository/platform-tools-latest-linux.zip",
}

SAMSUNG_DEVELOPER_STEPS = """
SAMSUNG DEVELOPER MODE SETUP
==============================
Follow these steps on your Samsung device:

  1. Open Settings
  2. Tap "About Phone" (or "About Tablet")
  3. Tap "Software Information"
  4. Tap "Build Number" 7 times rapidly
     - You will see: "You are now a developer!"
  5. Go back to Settings
  6. Open "Developer Options" (now visible)
  7. Enable "USB Debugging"
  8. Enable "Wireless Debugging" (Android 11+)

CONNECT VIA USB:
  - Connect phone to computer with USB cable
  - Tap "Allow" on the phone screen when prompted

CONNECT WIRELESSLY (Android 11+):
  - In Developer Options, tap "Wireless Debugging"
  - Tap "Pair device with pairing code"
  - Note the IP:PORT and 6-digit pairing code

Press ENTER when ready to continue...
"""

TROUBLESHOOTING_STEPS = """
TROUBLESHOOTING
===============
Device not showing up? Try these steps:

  1. Unplug and re-plug the USB cable
  2. On the phone: tap "Revoke USB debugging authorizations" and re-allow
  3. Try a different USB port or cable
  4. Restart ADB: run 'adb kill-server' then 'adb start-server'
  5. On Windows: check Device Manager for driver issues
  6. Enable "Transfer files" (MTP) mode instead of charging-only

Still not working?
  - Make sure USB Debugging is ON in Developer Options
  - Check that your Samsung device is Android 11+
  - Try wireless ADB pairing instead

Press ENTER to try again or type 'skip' to skip device detection...
"""


def detect_platform() -> str:
    """Return 'Windows', 'Linux', or 'macOS'."""
    system = platform.system()
    if system == "Darwin":
        return "macOS"
    return system  # Windows or Linux


def find_adb() -> Optional[str]:
    """
    Locate adb binary. Checks PATH, common install locations.
    Returns absolute path or None.
    """
    # Check PATH
    found = shutil.which("adb")
    if found:
        return found

    # Common locations
    candidates = []
    system = platform.system()
    if system == "Windows":
        candidates = [
            os.path.expanduser("~/AppData/Local/Android/Sdk/platform-tools/adb.exe"),
            "C:/Android/platform-tools/adb.exe",
            "C:/Program Files/Android/platform-tools/adb.exe",
        ]
    elif system == "Darwin":
        candidates = [
            os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
            "/usr/local/bin/adb",
            "/opt/homebrew/bin/adb",
        ]
    else:  # Linux
        candidates = [
            os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
            "/usr/bin/adb",
            "/usr/local/bin/adb",
        ]

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def get_adb_version(adb_path: str) -> Tuple[bool, str]:
    """
    Get the adb version string.
    Returns (success, version_string).
    """
    try:
        result = subprocess.run(
            [adb_path, "version"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "Android Debug Bridge" in line:
                return True, line.strip()
        return True, result.stdout.strip()[:100]
    except Exception as e:
        return False, str(e)


def get_adb_install_instructions(system: str) -> str:
    """Return platform-specific ADB installation instructions."""
    url = ADB_DOWNLOAD_URLS.get(system, ADB_DOWNLOAD_URLS["Linux"])
    if system == "Windows":
        return (
            f"ADB is not installed. To install:\n"
            f"  1. Download Platform Tools: {url}\n"
            f"  2. Extract the ZIP file (e.g., to C:\\platform-tools)\n"
            f"  3. Add C:\\platform-tools to your PATH environment variable\n"
            f"  4. Restart this terminal and run the wizard again"
        )
    elif system == "macOS":
        return (
            f"ADB is not installed. To install:\n"
            f"  Option A (Homebrew):  brew install android-platform-tools\n"
            f"  Option B (Manual): Download {url}\n"
            f"  Then add the platform-tools folder to your PATH"
        )
    else:  # Linux
        return (
            f"ADB is not installed. To install:\n"
            f"  Ubuntu/Debian:  sudo apt install adb\n"
            f"  Fedora/RHEL:    sudo dnf install android-tools\n"
            f"  Arch:           sudo pacman -S android-tools\n"
            f"  Or download:    {url}"
        )


def verify_device_permissions(adb_path: str, serial: str) -> dict:
    """
    Verify that the device has the required permissions enabled.
    Returns dict of permission check results.
    """
    checks = {
        "usb_debugging": False,
        "wireless_debugging": False,
        "developer_options": False,
    }
    try:
        # Check USB debugging via adb (if we can run adb shell, it's authorized)
        result = subprocess.run(
            [adb_path, "-s", serial, "shell", "echo ok"],
            capture_output=True, text=True, timeout=5
        )
        if "ok" in result.stdout:
            checks["usb_debugging"] = True
            checks["developer_options"] = True

        # Check wireless debugging
        result2 = subprocess.run(
            [adb_path, "-s", serial, "shell", "settings get global adb_wifi_enabled"],
            capture_output=True, text=True, timeout=5
        )
        checks["wireless_debugging"] = result2.stdout.strip() == "1"
    except Exception:
        pass
    return checks


def print_samsung_setup_instructions() -> None:
    """Print the step-by-step Samsung developer mode instructions."""
    print(SAMSUNG_DEVELOPER_STEPS)
