#!/usr/bin/env python3
"""
scripts/collect_app_latency.py

ADB-based cold/warm/hot start latency measurement for GraphMind v2.

Measures:
  - Cold start: force-stop app → launch → measure TotalTime
  - Warm start: home button → wait 3s → relaunch → measure TotalTime
  - Hot start:  bring app to foreground (app already in memory) → measure TotalTime

Apps measured:
  Instagram, WhatsApp, YouTube, Spotify, Gmail, Maps, Chrome,
  Netflix, Amazon, Slack, PhonePe, Paytm, Samsung Health

For each app × start_type × trial:
  - Runs N_TRIALS launches (default 5)
  - Collects TotalTime from `adb shell am start -W` output

Statistics exported:
  mean_ms, median_ms, p50_ms, p95_ms, p99_ms

Output:
  data/measured_latency.csv

Requirements:
  - Android device connected via USB with ADB debugging enabled
  - `adb` available in PATH
  - Apps installed on the target device
  - Android 7+ (for TotalTime field in am start -W output)

Target device:
  Samsung Galaxy A23 (or equivalent mid-range Android device)

Usage:
  python scripts/collect_app_latency.py
  python scripts/collect_app_latency.py --trials 10 --output data/my_latency.csv
  python scripts/collect_app_latency.py --apps com.instagram.android com.whatsapp
"""

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import time
from datetime import date
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import settings  # noqa: E402

# ---------------------------------------------------------------------------
# Target apps: package_id → human name + main activity
# ---------------------------------------------------------------------------

TARGET_APPS: Dict[str, dict] = {
    "com.instagram.android": {
        "name": "Instagram",
        "activity": "com.instagram.android/.activity.MainTabActivity",
    },
    "com.whatsapp": {
        "name": "WhatsApp",
        "activity": "com.whatsapp/.HomeActivity",
    },
    "com.google.youtube": {
        "name": "YouTube",
        "activity": "com.google.android.youtube/.HomeActivity",
    },
    "com.spotify.music": {
        "name": "Spotify",
        "activity": "com.spotify.music/.MainActivity",
    },
    "com.google.android.gm": {
        "name": "Gmail",
        "activity": "com.google.android.gm/.ConversationListActivityGmail",
    },
    "com.google.android.maps": {
        "name": "Google Maps",
        "activity": "com.google.android.maps/com.google.android.maps.MapsActivity",
    },
    "com.android.chrome": {
        "name": "Chrome",
        "activity": "com.android.chrome/com.google.android.apps.chrome.Main",
    },
    "com.netflix.mediaclient": {
        "name": "Netflix",
        "activity": "com.netflix.mediaclient/.ui.splash.GuardianActivity",
    },
    "com.amazon.mShop.android": {
        "name": "Amazon",
        "activity": "com.amazon.mShop.android.shopping/.HomeActivity",
    },
    "com.slack.android": {
        "name": "Slack",
        "activity": "com.slack/.ui.HomeActivity",
    },
    "com.phonepe.app": {
        "name": "PhonePe",
        "activity": "com.phonepe.app/.ui.activity.HomeActivity",
    },
    "net.one97.paytm": {
        "name": "Paytm",
        "activity": "net.one97.paytm/.AJRActivity",
    },
    "com.samsung.health": {
        "name": "Samsung Health",
        "activity": "com.samsung.android.app.shealth/.app.SplashActivity",
    },
}

N_TRIALS_DEFAULT = 5
WARM_WAIT_SECONDS = 3.0     # seconds to wait after pressing home for warm start
HOT_WAIT_SECONDS = 0.5      # seconds to wait before hot-start re-launch
_TOTAL_TIME_RE = re.compile(r"TotalTime:\s*(\d+)")
_THIS_TIME_RE = re.compile(r"ThisTime:\s*(\d+)")


# ---------------------------------------------------------------------------
# ADB helpers
# ---------------------------------------------------------------------------

def _check_device() -> Optional[str]:
    """
    Check if an ADB device is connected. Return device serial or None.
    Exits gracefully with instructions if no device found.
    """
    try:
        result = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().splitlines()
        # Lines after header that contain a device (not "offline" or "unauthorized")
        devices = [
            l.split("\t")[0]
            for l in lines[1:]
            if "\tdevice" in l
        ]
        if devices:
            return devices[0]
        logger.error(
            "No ADB device found. Connect a Samsung Galaxy A23 (or equivalent)\n"
            "via USB with 'USB Debugging' enabled in Developer Options."
        )
        return None
    except FileNotFoundError:
        logger.error(
            "'adb' command not found. Install Android Platform Tools:\n"
            "  https://developer.android.com/tools/releases/platform-tools"
        )
        return None
    except subprocess.TimeoutExpired:
        logger.error("ADB device check timed out.")
        return None


def _get_device_info(serial: str) -> dict:
    """Return basic device metadata (model, Android version) via ADB."""
    info = {
        "device_class": "Unknown Android Device",
        "android_version": "Unknown",
    }
    try:
        model = subprocess.run(
            ["adb", "-s", serial, "shell", "getprop", "ro.product.model"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        android = subprocess.run(
            ["adb", "-s", serial, "shell", "getprop", "ro.build.version.release"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if model:
            info["device_class"] = model
        if android:
            info["android_version"] = f"Android {android}"
    except Exception:
        pass
    return info


def _force_stop(serial: str, package: str) -> None:
    """Force-stop a package via ADB."""
    subprocess.run(
        ["adb", "-s", serial, "shell", "am", "force-stop", package],
        capture_output=True, timeout=5
    )


def _press_home(serial: str) -> None:
    """Send HOME key event via ADB."""
    subprocess.run(
        ["adb", "-s", serial, "shell", "input", "keyevent", "KEYCODE_HOME"],
        capture_output=True, timeout=5
    )


def _launch_and_measure(serial: str, activity: str) -> Optional[float]:
    """
    Launch an app via `adb shell am start -W` and return TotalTime in ms.

    Returns None if the launch fails or TotalTime cannot be parsed.
    """
    try:
        result = subprocess.run(
            ["adb", "-s", serial, "shell", "am", "start", "-W", activity],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        match = _TOTAL_TIME_RE.search(output)
        if match:
            return float(match.group(1))
        # Fallback to ThisTime
        match = _THIS_TIME_RE.search(output)
        if match:
            return float(match.group(1))
        logger.debug(f"Could not parse TotalTime from: {output[:200]}")
        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"Launch timed out for {activity}")
        return None
    except Exception as exc:
        logger.warning(f"Launch failed for {activity}: {exc}")
        return None


def _measure_cold(serial: str, package: str, activity: str, n: int) -> List[float]:
    """
    Measure cold start latency N times.

    Protocol: force-stop → wait 0.5s → launch → record TotalTime.
    """
    samples: List[float] = []
    for trial in range(n):
        _force_stop(serial, package)
        time.sleep(0.5)
        ms = _launch_and_measure(serial, activity)
        if ms is not None:
            samples.append(ms)
            logger.info(f"    Cold trial {trial+1}/{n}: {ms:.0f} ms")
        else:
            logger.warning(f"    Cold trial {trial+1}/{n}: FAILED")
    return samples


def _measure_warm(serial: str, package: str, activity: str, n: int) -> List[float]:
    """
    Measure warm start latency N times.

    Protocol: launch once (prime) → home → wait WARM_WAIT_SECONDS → relaunch.
    The first priming launch is not recorded.
    """
    samples: List[float] = []
    # Prime: ensure app is in memory
    _launch_and_measure(serial, activity)
    time.sleep(0.5)

    for trial in range(n):
        _press_home(serial)
        time.sleep(WARM_WAIT_SECONDS)
        ms = _launch_and_measure(serial, activity)
        if ms is not None:
            samples.append(ms)
            logger.info(f"    Warm trial {trial+1}/{n}: {ms:.0f} ms")
        else:
            logger.warning(f"    Warm trial {trial+1}/{n}: FAILED")
    return samples


def _measure_hot(serial: str, package: str, activity: str, n: int) -> List[float]:
    """
    Measure hot start latency N times.

    Protocol: launch app → home → immediately relaunch (app still in foreground memory).
    Wait only HOT_WAIT_SECONDS between home and relaunch.
    """
    samples: List[float] = []
    # Prime
    _launch_and_measure(serial, activity)
    time.sleep(0.3)

    for trial in range(n):
        _press_home(serial)
        time.sleep(HOT_WAIT_SECONDS)
        ms = _launch_and_measure(serial, activity)
        if ms is not None:
            samples.append(ms)
            logger.info(f"    Hot  trial {trial+1}/{n}: {ms:.0f} ms")
        else:
            logger.warning(f"    Hot  trial {trial+1}/{n}: FAILED")
    return samples


def _compute_stats(samples: List[float]) -> dict:
    """Compute mean, median, p50, p95, p99 from a list of measurements."""
    if not samples:
        return {
            "mean_ms": None, "median_ms": None,
            "p50_ms": None, "p95_ms": None, "p99_ms": None,
            "n_trials": 0,
        }
    import numpy as np
    arr = np.array(samples, dtype=float)
    return {
        "mean_ms":   round(float(arr.mean()), 1),
        "median_ms": round(float(np.median(arr)), 1),
        "p50_ms":    round(float(np.percentile(arr, 50)), 1),
        "p95_ms":    round(float(np.percentile(arr, 95)), 1),
        "p99_ms":    round(float(np.percentile(arr, 99)), 1),
        "n_trials":  len(samples),
    }


def _write_csv(rows: List[dict], output_path: str) -> None:
    """Write measurement rows to CSV."""
    if not rows:
        logger.warning("No rows to write.")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Results written to: {output_path}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Measure app launch latency via ADB for GraphMind v2."
    )
    parser.add_argument(
        "--trials", type=int, default=N_TRIALS_DEFAULT,
        help=f"Number of launches per app per start type (default {N_TRIALS_DEFAULT}).",
    )
    parser.add_argument(
        "--output", type=str, default=settings.LATENCY_MEASURED_CSV_PATH,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--apps", nargs="+", default=None,
        help="Subset of package IDs to measure (default: all).",
    )
    args = parser.parse_args()

    serial = _check_device()
    if serial is None:
        sys.exit(1)

    logger.info(f"Device: {serial}")
    device_info = _get_device_info(serial)
    logger.info(f"Model: {device_info['device_class']} | {device_info['android_version']}")
    measurement_date = date.today().isoformat()

    apps_to_measure = (
        {k: v for k, v in TARGET_APPS.items() if k in args.apps}
        if args.apps else TARGET_APPS
    )
    logger.info(f"Measuring {len(apps_to_measure)} app(s), {args.trials} trials each.")

    rows: List[dict] = []
    for package, meta in apps_to_measure.items():
        name = meta["name"]
        activity = meta["activity"]
        logger.info(f"\n[{name}] {package}")

        for start_type, measure_fn in [
            ("cold", _measure_cold),
            ("warm", _measure_warm),
            ("hot", _measure_hot),
        ]:
            logger.info(f"  {start_type.upper()} starts:")
            try:
                samples = measure_fn(serial, package, activity, args.trials)
            except Exception as exc:
                logger.error(f"  Measurement failed: {exc}")
                samples = []

            stats = _compute_stats(samples)
            row = {
                "app_id": package,
                "app_name": name,
                "start_type": start_type,
                "device_class": device_info["device_class"],
                "android_version": device_info["android_version"],
                "app_version": "latest",
                "measurement_date": measurement_date,
                **stats,
            }
            rows.append(row)
            if stats["mean_ms"] is not None:
                logger.info(
                    f"  → mean={stats['mean_ms']:.0f}ms  "
                    f"p95={stats['p95_ms']:.0f}ms  "
                    f"p99={stats['p99_ms']:.0f}ms"
                )
            else:
                logger.warning(f"  → No valid measurements collected.")

    _write_csv(rows, args.output)
    logger.info(
        f"\nDone. {len(rows)} rows written.\n"
        f"Set LATENCY_MEASURED_CSV_PATH in settings.py or use default path:\n"
        f"  {args.output}"
    )


if __name__ == "__main__":
    main()
