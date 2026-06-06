"""
src/data/device_analyzer_loader.py

Parses raw University of Cambridge Device Analyzer CSV files into
GraphMind's standard event format and produces chronological train/val/test splits.

Device Analyzer CSV format (minimum required columns):
  timestamp    — Unix epoch seconds (integer or float)
  package_name — Android package ID string

Optional columns (used when present):
  battery      — battery level 0–100
  screen_on    — boolean (0 or 1)

GraphMind event format produced:
  {
    "timestamp"              : float,
    "app_id"                 : str,
    "battery"                : float,
    "time_bucket"            : int,     # 0–47 (30-min buckets)
    "headphones"             : bool,    # always False (not in dataset)
    "calendar_event_in_mins" : None,    # not available in dataset
    "weekend"                : bool,
    "day"                    : int,     # relative day index from first event
    "category"               : str,    # from app_taxonomy lookup
    "source"                 : str,    # "device_analyzer"
  }

Chronological split:
  Train = earliest 80% of events by timestamp
  Val   = next 10%
  Test  = last 10%
  (Never random. See EventDataset._chronological_split for rationale.)
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

# Packages to exclude from event stream (system internals, launchers, etc.)
# These create noise without contributing to meaningful app-usage signal.
_EXCLUDED_PACKAGES: frozenset = frozenset({
    "android",
    "com.android.systemui",
    "com.android.launcher",
    "com.android.launcher2",
    "com.android.launcher3",
    "com.google.android.launcher",
    "com.sec.android.app.launcher",
    "com.android.phone",
    "com.android.contacts",
    "com.android.settings",
    "com.android.packageinstaller",
    "com.android.providers.downloads",
    "com.android.server.telecom",
})


def _load_taxonomy() -> dict:
    """Load app taxonomy JSON from disk. Returns empty dict on failure."""
    try:
        with open(settings.APP_TAXONOMY_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning(f"Could not load app taxonomy: {exc}")
        return {}


def _timestamp_to_time_bucket(ts: float) -> int:
    """
    Convert a Unix timestamp to a 30-minute time bucket index.

    Buckets: 0 = 00:00–00:30, 1 = 00:30–01:00, ..., 47 = 23:30–00:00
    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    minute_of_day = dt.hour * 60 + dt.minute
    return min(47, minute_of_day // 30)


def _is_weekend(ts: float) -> bool:
    """Return True if the Unix timestamp falls on a weekend (Saturday/Sunday, UTC)."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.weekday() >= 5  # 5=Saturday, 6=Sunday


class DeviceAnalyzerLoader:
    """
    Loads and converts raw Device Analyzer CSV data into GraphMind event format.

    Usage:
        loader = DeviceAnalyzerLoader()
        loader.load()                          # parses all CSV files
        splits = loader.get_splits()           # {"train": [...], "val": [...], "test": [...]}
        meta = loader.metadata()               # summary dict
    """

    def __init__(
        self,
        raw_dir: Optional[str] = None,
        max_events_per_device: Optional[int] = None,
    ) -> None:
        """
        Args:
            raw_dir:                Override for settings.DEVICE_ANALYZER_RAW_DIR.
            max_events_per_device:  Cap events per CSV file (useful for quick testing).
        """
        self._raw_dir = raw_dir or settings.DEVICE_ANALYZER_RAW_DIR
        self._max_events_per_device = max_events_per_device
        self._taxonomy = _load_taxonomy()
        self._all_events: List[dict] = []
        self._splits: Dict[str, List[dict]] = {}
        self._loaded = False
        self._device_count = 0
        self._skipped_count = 0

    def load(self) -> None:
        """
        Parse all CSV files in raw_dir and build chronological splits.

        This method is idempotent — calling it twice is safe and cheap.
        """
        if self._loaded:
            return

        csv_files = self._find_csv_files()
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self._raw_dir}. "
                "Run `python scripts/setup_device_analyzer.py` first."
            )

        all_events: List[dict] = []
        for path in csv_files:
            device_events = self._parse_csv(path)
            all_events.extend(device_events)
            self._device_count += 1

        # Sort globally by timestamp — mandatory for chronological split correctness
        all_events.sort(key=lambda e: float(e["timestamp"]))

        # Assign relative day indices from first event
        if all_events:
            first_ts = float(all_events[0]["timestamp"])
            first_day = datetime.fromtimestamp(first_ts, tz=timezone.utc).date()
            for evt in all_events:
                evt_date = datetime.fromtimestamp(
                    float(evt["timestamp"]), tz=timezone.utc
                ).date()
                evt["day"] = (evt_date - first_day).days

        self._all_events = all_events

        # Build chronological splits
        n = len(all_events)
        train_end = int(n * settings.DATASET_TRAIN_RATIO)
        val_end = int(n * (settings.DATASET_TRAIN_RATIO + settings.DATASET_VAL_RATIO))
        self._splits = {
            "train": all_events[:train_end],
            "val":   all_events[train_end:val_end],
            "test":  all_events[val_end:],
        }

        self._loaded = True
        logger.info(
            f"DeviceAnalyzerLoader: {len(all_events)} events from "
            f"{self._device_count} device(s). Skipped {self._skipped_count} rows. "
            f"Splits — train: {len(self._splits['train'])}, "
            f"val: {len(self._splits['val'])}, "
            f"test: {len(self._splits['test'])}"
        )

    def get_splits(self) -> Dict[str, List[dict]]:
        """Return the chronological splits. Calls load() if not yet loaded."""
        if not self._loaded:
            self.load()
        return self._splits

    def metadata(self) -> dict:
        """Return a JSON-serialisable summary of the loaded dataset."""
        if not self._loaded:
            self.load()
        return {
            "source": "device_analyzer",
            "raw_dir": self._raw_dir,
            "device_count": self._device_count,
            "total_events": len(self._all_events),
            "skipped_rows": self._skipped_count,
            "split_sizes": {k: len(v) for k, v in self._splits.items()},
            "split_ratios": {
                "train": settings.DATASET_TRAIN_RATIO,
                "val":   settings.DATASET_VAL_RATIO,
                "test":  settings.DATASET_TEST_RATIO,
            },
            "split_method": "chronological",
            "loaded": self._loaded,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_csv_files(self) -> List[str]:
        """Return sorted list of CSV paths under raw_dir."""
        if not os.path.isdir(self._raw_dir):
            return []
        return sorted(
            os.path.join(self._raw_dir, f)
            for f in os.listdir(self._raw_dir)
            if f.endswith(".csv")
        )

    def _parse_csv(self, path: str) -> List[dict]:
        """
        Parse a single Device Analyzer CSV file into GraphMind event dicts.

        Rows are filtered for:
          - Valid numeric timestamp
          - Non-empty, non-excluded package name

        Battery level defaults to 80.0 when not present in the CSV.
        Headphones and calendar_event_in_mins are always None/False
        because the Device Analyzer dataset does not capture these signals.
        """
        events: List[dict] = []
        device_id = os.path.splitext(os.path.basename(path))[0]

        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    logger.warning(f"Empty or headerless file: {path}")
                    return events

                # Normalise column names to lowercase
                col_map = {
                    col.strip().lower(): col
                    for col in (reader.fieldnames or [])
                }

                for i, row in enumerate(reader):
                    if (
                        self._max_events_per_device is not None
                        and len(events) >= self._max_events_per_device
                    ):
                        break

                    row_lower = {k.strip().lower(): v.strip() for k, v in row.items() if k}

                    # --- timestamp ---
                    raw_ts = row_lower.get("timestamp", "")
                    try:
                        ts = float(raw_ts)
                    except (ValueError, TypeError):
                        self._skipped_count += 1
                        continue

                    # Sanity check: timestamps must be plausible Unix epoch
                    # Device Analyzer data spans ~2011–2016.
                    if not (1_200_000_000 <= ts <= 2_000_000_000):
                        self._skipped_count += 1
                        continue

                    # --- package name ---
                    pkg = row_lower.get("package_name", "").strip()
                    if not pkg or pkg in _EXCLUDED_PACKAGES:
                        self._skipped_count += 1
                        continue

                    # --- battery ---
                    try:
                        battery = float(row_lower.get("battery", "80.0") or "80.0")
                        battery = max(0.0, min(100.0, battery))
                    except (ValueError, TypeError):
                        battery = 80.0

                    # --- derived fields ---
                    time_bucket = _timestamp_to_time_bucket(ts)
                    weekend = _is_weekend(ts)
                    category = self._taxonomy.get(pkg, {}).get("category", "utility")

                    events.append({
                        "timestamp": ts,
                        "app_id": pkg,
                        "battery": round(battery, 2),
                        "time_bucket": time_bucket,
                        "headphones": False,
                        "calendar_event_in_mins": None,
                        "weekend": weekend,
                        "day": 0,          # filled by load() after global sort
                        "category": category,
                        "source": "device_analyzer",
                        "device_id": device_id,
                    })

        except Exception as exc:
            logger.error(f"Failed to parse {path}: {exc}")

        return events
