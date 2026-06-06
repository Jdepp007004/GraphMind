#!/usr/bin/env python3
"""
scripts/ubiqlog_transition_pipeline.py

Phase 3: Transition Pipeline for UbiqLog.

Reads all Application events, constructs:
  - transitions.parquet (user_id, from_app, to_app, timestamp, gap_s, time_bucket, day_of_week)
  - Markov-1 matrix per user (pickle)
  - Markov-2 matrix per user (pickle)
  - NetworkX DiGraph per user (pickle)
  - reports/transition_statistics.md

Requires: reports/user_summary.csv to be generated first (Phase 2).
"""

import csv
import json
import logging
import math
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UBIQLOG_ROOT  = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog", "UbiqLog4UCI")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MARKOV_DIR    = os.path.join(PROCESSED_DIR, "markov")
GRAPHS_DIR    = os.path.join(PROCESSED_DIR, "graphs")

MAX_GAP_S    = 3600   # max seconds between sessions to count as transition
MIN_YEAR     = 2011
MAX_YEAR     = 2016

SYSTEM_PREFIXES = (
    "com.android.",
    "com.google.android.providers",
    "com.google.android.gms",
    "com.google.android.gsf",
    "com.sec.android.provider",
    "com.samsung.android.provider",
    "com.redbend.",
    "android.",
)
SYSTEM_SUFFIXES = (":engine", ":client", ":daemon", ":service", ":pushservice", ":sync")


def is_system_app(p: str) -> bool:
    p = p.lower()
    for pfx in SYSTEM_PREFIXES:
        if p.startswith(pfx.lower()):
            return True
    for sfx in SYSTEM_SUFFIXES:
        if p.endswith(sfx):
            return True
    return False


def parse_ts(s: str) -> Optional[datetime]:
    try:
        dt = datetime.strptime(s.strip(), "%m-%d-%Y %H:%M:%S")
        return dt if MIN_YEAR <= dt.year <= MAX_YEAR else None
    except Exception:
        return None


def load_usable_users() -> List[str]:
    """Load usable user IDs from reports/user_summary.csv."""
    summary_path = os.path.join(REPORTS_DIR, "user_summary.csv")
    if not os.path.exists(summary_path):
        logger.warning("user_summary.csv not found — using all users")
        return sorted([
            d for d in os.listdir(UBIQLOG_ROOT)
            if os.path.isdir(os.path.join(UBIQLOG_ROOT, d)) and not d.startswith(".")
        ])
    usable = []
    with open(summary_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("usable", "").lower() == "true":
                usable.append(row["user_id"])
    return usable


def extract_events(user_dir: str) -> List[dict]:
    events = []
    for fname in sorted(os.listdir(user_dir)):
        if not fname.endswith(".txt"):
            continue
        try:
            with open(os.path.join(user_dir, fname), encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "Application" not in obj:
                            continue
                        app = obj["Application"]
                        pkg = app.get("ProcessName", "").strip()
                        if not pkg or is_system_app(pkg):
                            continue
                        start = parse_ts(app.get("Start", ""))
                        end   = parse_ts(app.get("End",   "")) or start
                        if start is None:
                            continue
                        events.append({"pkg": pkg, "start": start, "end": end})
                    except Exception:
                        pass
        except Exception:
            pass
    events.sort(key=lambda e: e["start"])
    return events


def build_transitions(events: List[dict]) -> List[dict]:
    """Build app→app transitions from sorted event list."""
    transitions = []
    for i in range(1, len(events)):
        a, b = events[i - 1], events[i]
        gap = (b["start"] - a["end"]).total_seconds()
        if not (0 <= gap <= MAX_GAP_S):
            continue
        start_dt = b["start"]
        time_bucket = start_dt.hour * 2 + start_dt.minute // 30  # 0–47
        transitions.append({
            "from_app":    a["pkg"],
            "to_app":      b["pkg"],
            "timestamp":   start_dt.isoformat(),
            "gap_s":       round(gap, 1),
            "time_bucket": time_bucket,
            "day_of_week": start_dt.weekday(),  # 0=Mon, 6=Sun
        })
    return transitions


def build_markov1(transitions: List[dict]) -> Dict[str, Dict[str, float]]:
    """Markov-1: P(to | from) from transition list."""
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in transitions:
        counts[t["from_app"]][t["to_app"]] += 1
    markov: Dict[str, Dict[str, float]] = {}
    for src, dests in counts.items():
        total = sum(dests.values())
        markov[src] = {dst: cnt / total for dst, cnt in sorted(dests.items(), key=lambda x: -x[1])}
    return markov


def build_markov2(transitions: List[dict]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Markov-2: P(to | from_prev, from_cur) from bigram history."""
    counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(1, len(transitions)):
        prev = transitions[i - 1]["from_app"]
        cur  = transitions[i - 1]["to_app"]
        nxt  = transitions[i]["to_app"]
        if transitions[i]["from_app"] == cur:  # consecutive
            counts[(prev, cur)][nxt] += 1
    markov2: Dict[Tuple[str, str], Dict[str, float]] = {}
    for bigram, dests in counts.items():
        total = sum(dests.values())
        markov2[bigram] = {dst: cnt / total for dst, cnt in sorted(dests.items(), key=lambda x: -x[1])}
    return markov2


def build_graph(transitions: List[dict]):
    """Build NetworkX DiGraph with edge weights = transition probability."""
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not available — skipping graph construction")
        return None

    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in transitions:
        counts[t["from_app"]][t["to_app"]] += 1

    G = nx.DiGraph()
    for src, dests in counts.items():
        total = sum(dests.values())
        G.add_node(src)
        for dst, cnt in dests.items():
            G.add_edge(src, dst, weight=cnt / total, count=cnt)
    return G


def compute_graph_stats(user_id: str, transitions: List[dict], markov1: dict) -> dict:
    """Compute graph density and top transitions for reporting."""
    unique_apps = set()
    for t in transitions:
        unique_apps.add(t["from_app"])
        unique_apps.add(t["to_app"])
    n = len(unique_apps)
    e = sum(len(v) for v in markov1.values())
    density = e / (n * (n - 1)) if n > 1 else 0.0

    # Top transitions by count
    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for t in transitions:
        edge_counts[(t["from_app"], t["to_app"])] += 1
    top = sorted(edge_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "user_id": user_id,
        "n_transitions": len(transitions),
        "unique_apps": n,
        "unique_edges": e,
        "graph_density": round(density, 4),
        "top_transitions": [
            {"from": k[0], "to": k[1], "count": v} for k, v in top
        ],
    }


def main():
    os.makedirs(MARKOV_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    usable_users = load_usable_users()
    logger.info(f"Processing {len(usable_users)} usable users")

    all_transitions = []  # for parquet
    all_stats = []        # for statistics report

    for user_id in usable_users:
        user_dir = os.path.join(UBIQLOG_ROOT, user_id)
        if not os.path.isdir(user_dir):
            logger.warning(f"Directory not found: {user_dir}")
            continue

        logger.info(f"Building transitions for {user_id}...")
        events = extract_events(user_dir)
        if not events:
            logger.warning(f"  {user_id}: no events after filtering")
            continue

        transitions = build_transitions(events)
        if not transitions:
            logger.warning(f"  {user_id}: no transitions found")
            continue

        # Add user_id to each transition row for parquet
        for t in transitions:
            t["user_id"] = user_id
        all_transitions.extend(transitions)

        # Markov-1
        markov1 = build_markov1(transitions)
        m1_path = os.path.join(MARKOV_DIR, f"markov1_{user_id}.pkl")
        with open(m1_path, "wb") as f:
            pickle.dump(markov1, f)

        # Markov-2
        markov2 = build_markov2(transitions)
        m2_path = os.path.join(MARKOV_DIR, f"markov2_{user_id}.pkl")
        with open(m2_path, "wb") as f:
            pickle.dump(markov2, f)

        # Graph
        G = build_graph(transitions)
        if G is not None:
            g_path = os.path.join(GRAPHS_DIR, f"{user_id}_graph.pkl")
            with open(g_path, "wb") as f:
                pickle.dump(G, f)

        # Stats
        stats = compute_graph_stats(user_id, transitions, markov1)
        all_stats.append(stats)
        logger.info(
            f"  {user_id}: {len(transitions)} transitions, "
            f"{stats['unique_apps']} apps, "
            f"density={stats['graph_density']:.4f}"
        )

    # Write transitions.parquet
    try:
        import pandas as pd
        df = pd.DataFrame(all_transitions)
        parquet_path = os.path.join(PROCESSED_DIR, "transitions.parquet")
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Written: {parquet_path} ({len(df):,} rows)")
    except Exception as exc:
        logger.error(f"Failed to write parquet: {exc}")
        # Fallback: CSV
        csv_path = os.path.join(PROCESSED_DIR, "transitions.csv")
        if all_transitions:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(all_transitions[0].keys()))
                w.writeheader()
                w.writerows(all_transitions)
            logger.info(f"Written fallback CSV: {csv_path}")

    # Compute aggregate stats for report
    if all_stats:
        avg_transitions = sum(s["n_transitions"] for s in all_stats) / len(all_stats)
        avg_unique_apps = sum(s["unique_apps"] for s in all_stats) / len(all_stats)
        avg_density     = sum(s["graph_density"] for s in all_stats) / len(all_stats)

        # Global top transitions across all users
        global_edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for t in all_transitions:
            global_edge_counts[(t["from_app"], t["to_app"])] += 1
        global_top = sorted(global_edge_counts.items(), key=lambda x: -x[1])[:10]

        # Write report
        report_path = os.path.join(REPORTS_DIR, "transition_statistics.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# UbiqLog Transition Statistics\n\n")
            f.write(f"**Users processed:** {len(all_stats)}\n")
            f.write(f"**Total transitions:** {len(all_transitions):,}\n")
            f.write(f"**Average transitions per user:** {avg_transitions:.0f}\n")
            f.write(f"**Average unique apps per user:** {avg_unique_apps:.1f}\n")
            f.write(f"**Average graph density:** {avg_density:.4f}\n\n")
            f.write("---\n\n")
            f.write("## Per-User Statistics\n\n")
            f.write("| User | Transitions | Unique Apps | Graph Density |\n")
            f.write("|------|-------------|-------------|---------------|\n")
            for s in sorted(all_stats, key=lambda x: -x["n_transitions"]):
                f.write(
                    f"| {s['user_id']} | {s['n_transitions']:,} | "
                    f"{s['unique_apps']} | {s['graph_density']:.4f} |\n"
                )
            f.write("\n---\n\n")
            f.write("## Global Top 10 Transitions (All Users)\n\n")
            f.write("| From App | To App | Count |\n")
            f.write("|----------|--------|-------|\n")
            for (src, dst), cnt in global_top:
                f.write(f"| `{src}` | `{dst}` | {cnt:,} |\n")

        logger.info(f"Written: {report_path}")


if __name__ == "__main__":
    main()
