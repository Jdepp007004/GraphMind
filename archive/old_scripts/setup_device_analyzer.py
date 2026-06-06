#!/usr/bin/env python3
"""
scripts/setup_device_analyzer.py

Verifies, validates, and prepares the University of Cambridge Device Analyzer
dataset for use with GraphMind. Does NOT attempt any automatic download.

The dataset requires academic registration at:
  https://deviceanalyzer.cl.cam.ac.uk/

Responsibilities:
  1. Verify dataset files are present under data/device_analyzer/raw/.
  2. Validate the CSV schema of raw files.
  3. Display acquisition instructions when data is missing.
  4. Build processed splits (chronological 80/10/10) when data is present.

Usage:
  python scripts/setup_device_analyzer.py [--validate-only]

Options:
  --validate-only  Validate raw files only; skip split building.
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import settings  # noqa: E402

# Expected columns in Device Analyzer CSV exports.
# The exact column set varies by export version; we check for the minimum required set.
REQUIRED_COLUMNS: frozenset = frozenset({"timestamp", "package_name"})
OPTIONAL_COLUMNS: frozenset = frozenset({"device_id", "battery", "screen_on"})


def _show_acquisition_instructions() -> None:
    """Print step-by-step instructions for obtaining Device Analyzer data."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  DEVICE ANALYZER DATASET — ACQUISITION INSTRUCTIONS")
    print(sep)
    print()
    print("The University of Cambridge Device Analyzer dataset requires")
    print("academic registration. Automatic download is not possible.")
    print()
    print("STEPS TO ACQUIRE:")
    print()
    print("  Step 1:  Visit the project website:")
    print(f"           {settings.DEVICE_ANALYZER_URL}")
    print()
    print("  Step 2:  Click 'Get the Data' and submit the academic")
    print("           registration form. A download link is typically")
    print("           emailed within 1–2 business days.")
    print()
    print("  Step 3:  Download the dataset archive (tar.gz format).")
    print("           It contains per-device CSV files. Expected columns:")
    print("             - timestamp    (Unix epoch, seconds)")
    print("             - package_name (Android package ID)")
    print("             - battery      (optional, 0–100)")
    print("             - screen_on    (optional, boolean)")
    print()
    print("  Step 4:  Extract and place the CSV files in:")
    print(f"           {settings.DEVICE_ANALYZER_RAW_DIR}")
    print()
    print("  Step 5:  Re-run this script to validate and build splits:")
    print("           python scripts/setup_device_analyzer.py")
    print()
    print("CITATION (required if you publish results using this dataset):")
    print()
    print("  Haddadi, H. et al. (2014). 'Heterogeneity in Smartphone Usage:")
    print("  Diversity in Calls, Texts and Environments'.")
    print("  HotMobile '13. ACM. DOI: 10.1145/2457152.2457165")
    print()
    print("FALLBACK:")
    print("  GraphMind v2 will automatically use the built-in synthetic")
    print("  dataset when Device Analyzer data is absent. All benchmarks")
    print("  support both sources via the EventDataset interface.")
    print()
    print(f"{sep}\n")


def _find_csv_files() -> List[str]:
    """Return list of CSV file paths under the raw directory."""
    raw_dir = settings.DEVICE_ANALYZER_RAW_DIR
    if not os.path.isdir(raw_dir):
        return []
    return [
        os.path.join(raw_dir, f)
        for f in sorted(os.listdir(raw_dir))
        if f.endswith(".csv")
    ]


def _validate_csv_schema(path: str) -> Tuple[bool, str]:
    """
    Validate that a CSV file has the minimum required columns.

    Args:
        path: Absolute path to the CSV file.

    Returns:
        (valid, message) tuple.
    """
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
        if header is None:
            return False, "Empty file (no header row)."
        columns = frozenset(col.strip().lower() for col in header)
        missing = REQUIRED_COLUMNS - columns
        if missing:
            return False, f"Missing required columns: {sorted(missing)}"
        return True, "OK"
    except Exception as exc:
        return False, f"Read error: {exc}"


def _validate_all(csv_files: List[str]) -> Dict[str, object]:
    """
    Validate all CSV files. Return a summary dict.

    Returns:
        dict with: total, valid, invalid, errors (list), total_size_mb
    """
    summary: Dict[str, object] = {
        "total": len(csv_files),
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "total_size_mb": 0.0,
    }
    for path in csv_files:
        size_bytes = os.path.getsize(path)
        summary["total_size_mb"] = float(summary["total_size_mb"]) + size_bytes / (1024 * 1024)  # type: ignore[operator]
        ok, msg = _validate_csv_schema(path)
        if ok:
            summary["valid"] = int(summary["valid"]) + 1  # type: ignore[operator]
        else:
            summary["invalid"] = int(summary["invalid"]) + 1  # type: ignore[operator]
            summary["errors"].append(f"{os.path.basename(path)}: {msg}")  # type: ignore[union-attr]
    summary["total_size_mb"] = round(float(summary["total_size_mb"]), 2)
    return summary


def _build_splits() -> bool:
    """
    Build chronological train/val/test splits from raw CSV files.
    Delegates to DeviceAnalyzerLoader.

    Returns True on success.
    """
    try:
        from src.data.device_analyzer_loader import DeviceAnalyzerLoader
        loader = DeviceAnalyzerLoader()
        loader.load()
        splits = loader.get_splits()

        os.makedirs(settings.DEVICE_ANALYZER_SPLITS_DIR, exist_ok=True)
        for split_name, events in splits.items():
            out_path = os.path.join(
                settings.DEVICE_ANALYZER_SPLITS_DIR, f"{split_name}.json"
            )
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(events, fh, indent=2)
            logger.info(f"  {split_name}: {len(events)} events → {out_path}")

        meta = loader.metadata()
        meta_path = os.path.join(settings.DEVICE_ANALYZER_DIR, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info(f"Metadata written to {meta_path}")
        return True

    except Exception as exc:
        logger.error(f"Split building failed: {exc}")
        return False


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="GraphMind Device Analyzer dataset setup utility."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate raw files only; skip building splits.",
    )
    args = parser.parse_args()

    os.makedirs(settings.DEVICE_ANALYZER_RAW_DIR, exist_ok=True)
    os.makedirs(settings.DEVICE_ANALYZER_PROCESSED_DIR, exist_ok=True)

    csv_files = _find_csv_files()
    if not csv_files:
        logger.info("No CSV files found in raw directory.")
        _show_acquisition_instructions()
        logger.info(
            "GraphMind will use synthetic data until Device Analyzer data is placed in:\n"
            f"  {settings.DEVICE_ANALYZER_RAW_DIR}"
        )
        sys.exit(0)

    logger.info(f"Found {len(csv_files)} CSV file(s). Validating schema...")
    summary = _validate_all(csv_files)

    print("\n--- Validation Report ---")
    print(f"  Total CSV files : {summary['total']}")
    print(f"  Valid           : {summary['valid']}")
    print(f"  Invalid         : {summary['invalid']}")
    print(f"  Total size      : {summary['total_size_mb']:.2f} MB")
    if summary["errors"]:
        print("  Schema errors:")
        for err in summary["errors"]:  # type: ignore[union-attr]
            print(f"    - {err}")
    print()

    if int(summary["invalid"]) > 0:  # type: ignore[call-overload]
        logger.error("Schema validation failed. Fix the listed files and re-run.")
        sys.exit(1)

    logger.info("All files pass schema validation.")

    if args.validate_only:
        logger.info("--validate-only flag set. Skipping split building.")
        sys.exit(0)

    logger.info("Building chronological train/val/test splits...")
    logger.info(
        f"  Ratios: train={settings.DATASET_TRAIN_RATIO:.0%}  "
        f"val={settings.DATASET_VAL_RATIO:.0%}  "
        f"test={settings.DATASET_TEST_RATIO:.0%}"
    )
    success = _build_splits()
    if success:
        logger.info(
            f"Splits written to: {settings.DEVICE_ANALYZER_SPLITS_DIR}"
        )
    else:
        logger.error("Split building failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
