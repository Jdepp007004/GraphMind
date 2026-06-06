#!/usr/bin/env python3
"""
scripts/ubiqlog_user_analysis.py

Phase 2: User Analysis for UbiqLog dataset.

Computes per-user statistics from all Application events:
  - first_timestamp, last_timestamp, duration_days, active_days
  - application_events, unique_apps, transition_count

Rankings and outputs:
  - reports/user_summary.csv
  - reports/user_ranking.csv
  - reports/top_users.csv
  - data/processed/users.json
"""

import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

UBIQLOG_ROOT = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog", "UbiqLog4UCI")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Minimum thresholds for "usable" user
MIN_APP_EVENTS = 500
MIN_TRANSITIONS = 100

# System services to exclude (background processes, not user-initiated)
SYSTEM_PREFIXES = (
    "com.android.",
    "com.google.android.providers",
    "com.google.android.gms",
    "com.google.android.gsf",
    "com.sec.android.provider",
    "com.samsung.android.provider",
    "com.redbend.",
    "android.",
    "android",
    "com.android.nfc",
    "com.android.phone",
    "com.android.systemui",
    "com.android.keyguard",
    "com.android.settings",
)

SYSTEM_SUFFIXES = (
    ":engine",
    ":client",
    ":daemon",
    ":service",
    ":pushservice",
    ":sync",
)

# Valid date range for filtering anomalies (2011–2016)
MIN_YEAR = 2011
MAX_YEAR = 2016

# Max gap between sessions to count as a transition (seconds)
MAX_TRANSITION_GAP_S = 3600  # 1 hour


def is_system_app(package: str) -> bool:
    """Return True for known background services and system processes."""
    p = package.lower()
    for prefix in SYSTEM_PREFIXES:
        if p.startswith(prefix.lower()):
            return True
    for suffix in SYSTEM_SUFFIXES:
        if p.endswith(suffix):
            return True
    return False


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Parse Application-event timestamp: 'MM-D-YYYY HH:MM:SS'

    Returns None if parsing fails or date is out of valid range.
    """
    try:
        dt = datetime.strptime(ts_str.strip(), "%m-%d-%Y %H:%M:%S")
        if dt.year < MIN_YEAR or dt.year > MAX_YEAR:
            return None
        return dt
    except Exception:
        return None


def extract_app_events(user_dir: str) -> List[dict]:
    """
    Read all daily log files for one user, extract Application events.

    Returns list of dicts: {package, start_dt, end_dt, duration_s, date_str}
    sorted chronologically by start_dt.
    """
    events = []
    for fname in os.listdir(user_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(user_dir, fname)
        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "Application" not in obj:
                            continue
                        app = obj["Application"]
                        package = app.get("ProcessName", "").strip()
                        start_str = app.get("Start", "")
                        end_str = app.get("End", "")
                        if not package or not start_str:
                            continue
                        start_dt = parse_timestamp(start_str)
                        end_dt = parse_timestamp(end_str) if end_str else start_dt
                        if start_dt is None:
                            continue
                        if end_dt is None:
                            end_dt = start_dt
                        duration_s = max(0.0, (end_dt - start_dt).total_seconds())
                        if is_system_app(package):
                            continue
                        events.append({
                            "package": package,
                            "start_dt": start_dt,
                            "end_dt": end_dt,
                            "duration_s": duration_s,
                        })
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning(f"  Failed to read {fpath}: {exc}")

    events.sort(key=lambda e: e["start_dt"])
    return events


def compute_user_stats(user_id: str, events: List[dict]) -> dict:
    """Compute all required statistics for one user from their event list."""
    if not events:
        return {
            "user_id": user_id,
            "usable": False,
            "reason": "no_application_events",
            "application_events": 0,
            "unique_apps": 0,
            "transition_count": 0,
            "duration_days": 0,
            "active_days": 0,
            "first_timestamp": None,
            "last_timestamp": None,
            "score": 0.0,
        }

    first_dt = events[0]["start_dt"]
    last_dt = events[-1]["start_dt"]
    duration_days = max(1, (last_dt - first_dt).days + 1)

    active_days = len(set(e["start_dt"].date() for e in events))
    unique_apps = len(set(e["package"] for e in events))

    # Count transitions
    transition_count = 0
    for i in range(1, len(events)):
        gap = (events[i]["start_dt"] - events[i - 1]["end_dt"]).total_seconds()
        if 0 <= gap <= MAX_TRANSITION_GAP_S:
            transition_count += 1

    app_event_count = len(events)
    usable = (app_event_count >= MIN_APP_EVENTS and transition_count >= MIN_TRANSITIONS)
    reason = "ok" if usable else (
        "too_few_events" if app_event_count < MIN_APP_EVENTS else "too_few_transitions"
    )

    score = duration_days * math.log10(max(1, transition_count))

    return {
        "user_id": user_id,
        "usable": usable,
        "reason": reason,
        "application_events": app_event_count,
        "unique_apps": unique_apps,
        "transition_count": transition_count,
        "duration_days": duration_days,
        "active_days": active_days,
        "first_timestamp": first_dt.isoformat() if first_dt else None,
        "last_timestamp": last_dt.isoformat() if last_dt else None,
        "score": round(score, 3),
    }


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    users = sorted([
        d for d in os.listdir(UBIQLOG_ROOT)
        if os.path.isdir(os.path.join(UBIQLOG_ROOT, d)) and not d.startswith(".")
    ])
    logger.info(f"Found {len(users)} user directories")

    all_stats = []
    for user_id in users:
        user_dir = os.path.join(UBIQLOG_ROOT, user_id)
        logger.info(f"Processing {user_id}...")
        events = extract_app_events(user_dir)
        stats = compute_user_stats(user_id, events)
        all_stats.append(stats)
        logger.info(
            f"  {user_id}: {stats['application_events']} events, "
            f"{stats['transition_count']} transitions, "
            f"{stats['duration_days']} days, "
            f"usable={stats['usable']}"
        )

    # user_summary.csv — all users sorted by user_id
    summary_path = os.path.join(REPORTS_DIR, "user_summary.csv")
    fieldnames = [
        "user_id", "usable", "reason", "application_events", "unique_apps",
        "transition_count", "duration_days", "active_days",
        "first_timestamp", "last_timestamp", "score"
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_stats)
    logger.info(f"Written: {summary_path}")

    # user_ranking.csv — all users sorted by score descending
    ranked = sorted(all_stats, key=lambda x: x["score"], reverse=True)
    for rank, s in enumerate(ranked, 1):
        s["rank"] = rank
    ranking_path = os.path.join(REPORTS_DIR, "user_ranking.csv")
    with open(ranking_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank"] + fieldnames)
        w.writeheader()
        w.writerows(ranked)
    logger.info(f"Written: {ranking_path}")

    # top_users.csv — only usable users, sorted by score
    usable = [s for s in ranked if s["usable"]]
    top_path = os.path.join(REPORTS_DIR, "top_users.csv")
    with open(top_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank"] + fieldnames)
        w.writeheader()
        w.writerows(usable)
    logger.info(f"Written: {top_path} ({len(usable)} usable users)")

    # users.json — machine-readable usable user list
    users_json = {
        "total_users": len(all_stats),
        "usable_users": len(usable),
        "excluded_users": len(all_stats) - len(usable),
        "users": usable,
    }
    users_json_path = os.path.join(PROCESSED_DIR, "users.json")
    with open(users_json_path, "w", encoding="utf-8") as f:
        json.dump(users_json, f, indent=2, ensure_ascii=False)
    logger.info(f"Written: {users_json_path}")

    # Summary to console
    logger.info(f"\n{'='*50}")
    logger.info(f"TOTAL USERS:   {len(all_stats)}")
    logger.info(f"USABLE:        {len(usable)}")
    logger.info(f"EXCLUDED:      {len(all_stats) - len(usable)}")
    logger.info(f"\nTop 5 users by score:")
    for s in usable[:5]:
        logger.info(
            f"  [{s['rank']}] {s['user_id']:8s} "
            f"score={s['score']:.1f} "
            f"events={s['application_events']:,} "
            f"transitions={s['transition_count']:,} "
            f"days={s['duration_days']}"
        )


if __name__ == "__main__":
    main()
