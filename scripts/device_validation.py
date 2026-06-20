"""
scripts/device_validation.py

Samsung device validation and reproducibility report.
"""

import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.android.adb_connector import ADBConnector
from src.android.device_detector import DeviceDetector


def collect_device_report(connector: Optional[ADBConnector] = None) -> dict:
    report = {
        "timestamp": time.time(),
        "adb_status": "unavailable",
        "adb_version": "",
        "device_detected": False,
        "telemetry_status": "not_checked",
        "dashboard_status": "available" if os.path.exists(os.path.join(settings.PROJECT_ROOT, "src", "dashboard", "app.py")) else "missing",
        "device": None,
        "checks": {
            "adb": False,
            "device": False,
            "permissions": False,
            "telemetry": False,
            "dashboard": True,
        },
    }
    try:
        connector = connector or ADBConnector()
    except Exception as e:
        report["error"] = str(e)
        return report

    report["adb_status"] = "available" if connector.is_available() else "unavailable"
    report["adb_version"] = connector.get_version()
    report["checks"]["adb"] = report["adb_status"] == "available"

    detector = DeviceDetector(connector)
    device = detector.detect_samsung()
    if device is None:
        report["telemetry_status"] = "no_device"
        return report

    validation = detector.validate_debugging_enabled(device)
    report["device_detected"] = True
    report["device"] = device.to_dict()
    report["checks"]["device"] = device.is_samsung
    report["checks"]["permissions"] = validation.get("adb_authorized", False)

    ok, battery = connector.shell("dumpsys battery", serial=device.serial, timeout=8)
    report["telemetry_status"] = "ok" if ok else "failed"
    report["checks"]["telemetry"] = ok
    report["battery_probe_sample"] = battery[:500] if ok else ""
    return report


def write_device_report(output_path: str = None, connector: Optional[ADBConnector] = None) -> dict:
    output_path = output_path or os.path.join(settings.RESULTS_DIR, "device_report.json")
    report = collect_device_report(connector)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    report = write_device_report()
    print(json.dumps(report, indent=2))
