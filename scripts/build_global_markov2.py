#!/usr/bin/env python3
"""
scripts/build_global_markov2.py

Phase 2: Build a GlobalMarkov2 baseline.

GlobalMarkov2 trains a single second-order Markov chain using the training
splits of ALL usable users combined. This tests whether a population-level
model can match personalized per-user models.

Architecture:
  - Load training events from all 31 usable users (80% split each)
  - Build joint second-order Markov: P(C | A→B) from combined corpus
  - Also build Markov-1 fallback from combined corpus
  - Save to data/processed/markov/global_markov2.pkl

The GlobalMarkov2Policy in run_benchmarks_v2.py is initialized with this
pre-trained matrix (no per-user training needed during evaluation).
"""

import csv
import json
import logging
import os
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UBIQLOG_ROOT  = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog", "UbiqLog4UCI")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MARKOV_DIR    = os.path.join(PROCESSED_DIR, "markov")

MIN_YEAR, MAX_YEAR = 2011, 2016
TRAIN_RATIO        = 0.80

SYSTEM_PREFIXES = (
    "com.android.", "com.google.android.providers",
    "com.google.android.gms", "com.google.android.gsf",
    "com.sec.android.provider", "com.samsung.android.provider",
    "com.redbend.", "android.",
)
SYSTEM_SUFFIXES = (":engine", ":client", ":daemon", ":service", ":pushservice", ":sync")


def is_system_app(p: str) -> bool:
    p = p.lower()
    for pfx in SYSTEM_PREFIXES:
        if p.startswith(pfx): return True
    for sfx in SYSTEM_SUFFIXES:
        if p.endswith(sfx): return True
    return False


def parse_ts(s: str) -> Optional[datetime]:
    try:
        dt = datetime.strptime(s.strip(), "%m-%d-%Y %H:%M:%S")
        return dt if MIN_YEAR <= dt.year <= MAX_YEAR else None
    except Exception:
        return None


def load_train_events(user_id: str) -> List[str]:
    """Load training split (80%) of sorted app sequence for one user."""
    user_dir = os.path.join(UBIQLOG_ROOT, user_id)
    events = []
    for fname in os.listdir(user_dir):
        if not fname.endswith(".txt"): continue
        try:
            with open(os.path.join(user_dir, fname), encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        if "Application" not in obj: continue
                        app = obj["Application"]
                        pkg = app.get("ProcessName", "").strip()
                        if not pkg or is_system_app(pkg): continue
                        start = parse_ts(app.get("Start", ""))
                        if start is None: continue
                        events.append((start, pkg))
                    except Exception:
                        pass
        except Exception:
            pass
    events.sort(key=lambda x: x[0])
    seq = [pkg for _, pkg in events]
    n = len(seq)
    train_end = int(n * TRAIN_RATIO)
    return seq[:train_end]


def build_global_markov1(all_seqs: List[List[str]]) -> Dict[str, Dict[str, float]]:
    """Build Markov-1 from all training sequences combined."""
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seq in all_seqs:
        for i in range(1, len(seq)):
            counts[seq[i-1]][seq[i]] += 1
    m = {}
    for src, dests in counts.items():
        total = sum(dests.values())
        m[src] = {d: c/total for d, c in sorted(dests.items(), key=lambda x: -x[1])}
    return m


def build_global_markov2(
    all_seqs: List[List[str]],
    fallback_m1: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Build Markov-2 from all training sequences combined."""
    counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seq in all_seqs:
        for i in range(2, len(seq)):
            bigram = (seq[i-2], seq[i-1])
            counts[bigram][seq[i]] += 1
    m2 = {}
    for bigram, dests in counts.items():
        total = sum(dests.values())
        m2[bigram] = {d: c/total for d, c in sorted(dests.items(), key=lambda x: -x[1])}
    return m2


def main():
    os.makedirs(MARKOV_DIR, exist_ok=True)

    # Load usable users
    users_path = os.path.join(PROCESSED_DIR, "users.json")
    with open(users_path, encoding="utf-8") as f:
        usable_users = [u["user_id"] for u in json.load(f)["users"]]
    logger.info(f"Loading training sequences for {len(usable_users)} users...")

    all_seqs = []
    total_events = 0
    for user_id in usable_users:
        seq = load_train_events(user_id)
        if len(seq) >= 50:
            all_seqs.append(seq)
            total_events += len(seq)

    logger.info(f"Total training events: {total_events:,} from {len(all_seqs)} users")

    # Build GlobalMarkov-1
    logger.info("Building GlobalMarkov-1...")
    gm1 = build_global_markov1(all_seqs)
    logger.info(f"  GlobalMarkov-1: {len(gm1):,} source states")

    # Build GlobalMarkov-2 with M1 fallback
    logger.info("Building GlobalMarkov-2...")
    gm2 = build_global_markov2(all_seqs, fallback_m1=gm1)
    logger.info(f"  GlobalMarkov-2: {len(gm2):,} bigram states")

    # Save
    gm1_path = os.path.join(MARKOV_DIR, "global_markov1.pkl")
    gm2_path = os.path.join(MARKOV_DIR, "global_markov2.pkl")

    with open(gm1_path, "wb") as f:
        pickle.dump(gm1, f)
    with open(gm2_path, "wb") as f:
        pickle.dump({"markov2": gm2, "fallback_m1": gm1}, f)

    logger.info(f"Written: {gm1_path}")
    logger.info(f"Written: {gm2_path}")

    # Stats
    bigram_counts = [len(v) for v in gm2.values()]
    m1_counts = [len(v) for v in gm1.values()]
    logger.info(f"\n=== GlobalMarkov-2 Stats ===")
    logger.info(f"  Bigram states:   {len(gm2):,}")
    logger.info(f"  Avg successors:  {np.mean(bigram_counts):.2f}")
    logger.info(f"  M1 states:       {len(gm1):,}")
    logger.info(f"  M1 avg succs:    {np.mean(m1_counts):.2f}")
    logger.info(f"  Total vocab:     {len(set(k[0] for k in gm2) | set(k[1] for k in gm2)):,} apps")


if __name__ == "__main__":
    main()
