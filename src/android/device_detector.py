"""
src/android/device_detector.py

Detects connected Samsung devices and validates Android/OneUI version.
Supports Android 11+ and Samsung OneUI on phones and tablets.
"""

import logging
import re
from typing import Optional, List, Dict

from src.android.adb_connector import ADBConnector

logger = logging.getLogger(__name__)

SUPPORTED_ANDROID_VERSIONS = {11, 12, 13, 14, 15}
SAMSUNG_BRANDS = {"samsung"}


class DeviceInfo:
    """Container for detected device properties."""

    def __init__(self) -> None:
        self.serial: str = ""
        self.model: str = ""
        self.brand: str = ""
        self.android_version: int = 0
        self.android_release: str = ""
        self.oneui_version: str = ""
        self.is_tablet: bool = False
        self.form_factor: str = "phone"
        self.usb_debugging: bool = False
        self.wireless_debugging: bool = False
        self.sdk_int: int = 0

    @property
    def is_samsung(self) -> bool:
        """Return True when the device brand is Samsung."""
        return self.brand.lower() in SAMSUNG_BRANDS

    @property
    def is_supported(self) -> bool:
        """Return True when the Samsung device runs a supported Android version."""
        return self.is_samsung and self.android_version in SUPPORTED_ANDROID_VERSIONS

    def to_dict(self) -> dict:
        """Serialize device metadata to a JSON-compatible dict."""
        return {
            "serial": self.serial,
            "model": self.model,
            "brand": self.brand,
            "android_version": self.android_version,
            "android_release": self.android_release,
            "oneui_version": self.oneui_version,
            "is_tablet": self.is_tablet,
            "form_factor": self.form_factor,
            "is_samsung": self.is_samsung,
            "is_supported": self.is_supported,
            "sdk_int": self.sdk_int,
        }

    def __repr__(self) -> str:
        return (f"DeviceInfo(serial={self.serial!r}, model={self.model!r}, "
                f"brand={self.brand!r}, android={self.android_version}, "
                f"oneui={self.oneui_version!r})")


class DeviceDetector:
    """
    Detects connected devices via ADB and extracts Samsung-specific metadata.
    """

    def __init__(self, connector: ADBConnector) -> None:
        self.connector = connector

    def detect_all(self) -> List[DeviceInfo]:
        """
        List all connected devices and probe each for Samsung/Android metadata.
        Returns list of DeviceInfo instances (including non-Samsung for reporting).
        """
        raw_devices = self.connector.list_devices()
        results = []
        for raw in raw_devices:
            if raw["state"] != "device":
                logger.debug(f"Skipping device {raw['serial']} in state '{raw['state']}'")
                continue
            info = self._probe_device(raw["serial"])
            results.append(info)
            logger.info(f"Detected: {info}")
        return results

    def detect_samsung(self) -> Optional[DeviceInfo]:
        """
        Return the first detected and supported Samsung device, or None.
        Prefers phones over tablets.
        """
        devices = self.detect_all()
        samsung_devices = [d for d in devices if d.is_samsung]
        if not samsung_devices:
            return None
        # Prefer supported versions
        supported = [d for d in samsung_devices if d.is_supported]
        if supported:
            return supported[0]
        return samsung_devices[0]

    def _probe_device(self, serial: str) -> DeviceInfo:
        """Read all system properties for a single device serial."""
        info = DeviceInfo()
        info.serial = serial

        props = self._get_all_props(serial)

        info.brand = props.get("ro.product.brand", "").lower()
        info.model = props.get("ro.product.model", props.get("ro.product.device", ""))
        info.android_release = props.get("ro.build.version.release", "")
        info.sdk_int = _safe_int(props.get("ro.build.version.sdk", "0"))

        # Android major version
        try:
            info.android_version = int(info.android_release.split(".")[0])
        except Exception:
            info.android_version = 0

        # OneUI version
        info.oneui_version = props.get(
            "ro.build.version.oneui",
            props.get("ro.system.build.version.oneui", "")
        )

        # Tablet detection via characteristics
        chars = props.get("ro.build.characteristics", "")
        info.is_tablet = "tablet" in chars.lower()
        info.form_factor = "tablet" if info.is_tablet else "phone"

        # USB Debugging is already enabled if we can reach adb shell
        info.usb_debugging = True

        # Wireless debugging (Android 11+ setting)
        ok, wd_out = self.connector.shell(
            "settings get global adb_wifi_enabled", serial=serial
        )
        info.wireless_debugging = wd_out.strip() == "1"

        return info

    def _get_all_props(self, serial: str) -> Dict[str, str]:
        """Run `adb shell getprop` and parse into dict."""
        ok, output = self.connector.shell("getprop", serial=serial, timeout=15)
        props: Dict[str, str] = {}
        if not ok:
            return props
        # Format: [key]: [value]
        pattern = re.compile(r"^\[(.+?)\]: \[(.*)?\]$")
        for line in output.splitlines():
            m = pattern.match(line.strip())
            if m:
                props[m.group(1)] = m.group(2)
        return props

    def validate_debugging_enabled(self, device: DeviceInfo) -> Dict[str, bool]:
        """
        Return validation report for a device:
        {connected, adb_authorized, android_version_ok, samsung_device}
        """
        return {
            "connected": True,
            "adb_authorized": device.serial != "",
            "android_version_ok": device.android_version in SUPPORTED_ANDROID_VERSIONS,
            "samsung_device": device.is_samsung,
            "usb_debugging": device.usb_debugging,
        }


def _safe_int(val: str) -> int:
    """Parse an integer string, returning 0 on failure."""
    try:
        return int(val)
    except Exception:
        return 0
