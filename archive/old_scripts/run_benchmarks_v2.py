#!/usr/bin/env python3
"""
scripts/run_benchmarks_v2.py

Phase 5: Run all 11 policies on all 31 usable UbiqLog users.

Policies:
  1  Random
  2  LRU
  3  LFU
  4  MRU
  5  Frequency
  6  RecencyFrequency
  7  Markov-1
  8  Markov-2
  9  GraphOnly
  10 Graph+Confidence
  11 GraphMindRL

Per-user per-policy output:
  - Hit Rate, Precision, Recall, F1
  - Latency Saved (ms), Latency Saved %
  - False Prefetch Rate, Thrash Rate
  - Memory Usage (MB), Prediction Latency (ms)

Aggregated over all users:
  - mean, median, std, P50, P90, P95, P99

Outputs:
  - results/benchmark_results_v2.csv     (per-user per-policy)
  - results/advanced_metrics_v2.csv      (aggregate across users)
"""

import csv
import json
import logging
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MARKOV_DIR    = os.path.join(PROCESSED_DIR, "markov")
UBIQLOG_ROOT  = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog", "UbiqLog4UCI")

LATENCY_CSV = os.path.join(PROJECT_ROOT, "datasets", "app_launch_latency.csv")

# Cache sizes
HOT_SIZE  = 5
WARM_SIZE = 15
COLD_SIZE = 50

# Evaluation: chronological 80/10/10 split
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10  (remainder)

MAX_GAP_S = 3600
MIN_YEAR  = 2011
MAX_YEAR  = 2016

SYSTEM_PREFIXES = (
    "com.android.", "com.google.android.providers",
    "com.google.android.gms", "com.google.android.gsf",
    "com.sec.android.provider", "com.samsung.android.provider",
    "com.redbend.", "android.",
)
SYSTEM_SUFFIXES = (":engine", ":client", ":daemon", ":service", ":pushservice", ":sync")


# ── Latency Model ────────────────────────────────────────────────────────────

class MeasuredLatencyModel:
    """Load measured Galaxy A23 latency. No literature fallback."""

    def __init__(self, csv_path: str):
        self._cold: Dict[str, float] = {}
        self._warm: Dict[str, float] = {}
        self._hot:  Dict[str, float] = {}
        self._pkg_to_app: Dict[str, str] = {}
        self._default_cold = 2763.0
        self._default_warm = 1301.0
        self._default_hot  =  274.0
        self._load(csv_path)

    def _load(self, csv_path: str):
        if not os.path.exists(csv_path):
            logger.warning(f"Latency CSV not found: {csv_path}")
            return
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                app_id  = row["app_id"]
                pkg     = row["package_name"]
                st      = row["start_type"]
                mean_ms = float(row["total_time_ms"])
                self._pkg_to_app[pkg] = app_id
                if st == "cold":
                    self._cold.setdefault(app_id, []).append(mean_ms)   # type: ignore[arg-type]
                elif st == "warm":
                    self._warm.setdefault(app_id, []).append(mean_ms)   # type: ignore[arg-type]
                elif st == "hot":
                    self._hot.setdefault(app_id, []).append(mean_ms)    # type: ignore[arg-type]
        # Average per app
        for d in (self._cold, self._warm, self._hot):
            for k in list(d.keys()):
                d[k] = float(np.mean(d[k]))   # type: ignore[arg-type]
        logger.info(f"Loaded latency for {len(self._cold)} apps")

    def _app_key(self, package: str) -> Optional[str]:
        # Try direct match, then package lookup
        if package in self._cold:
            return package
        return self._pkg_to_app.get(package)

    def cold_ms(self, package: str) -> float:
        k = self._app_key(package)
        return self._cold.get(k, self._default_cold) if k else self._default_cold

    def warm_ms(self, package: str) -> float:
        k = self._app_key(package)
        return self._warm.get(k, self._default_warm) if k else self._default_warm

    def hot_ms(self, package: str) -> float:
        k = self._app_key(package)
        return self._hot.get(k, self._default_hot) if k else self._default_hot

    def saved_ms(self, package: str, tier: str) -> float:
        cold = self.cold_ms(package)
        if tier == "hot":
            return cold - self.hot_ms(package)
        if tier == "warm":
            return cold - self.warm_ms(package)
        return 0.0


# ── Cache Simulator ──────────────────────────────────────────────────────────

class CacheSimulator:
    """Three-tier cache: HOT (list), WARM (list), COLD (set)."""

    def __init__(self, hot_n: int = HOT_SIZE, warm_n: int = WARM_SIZE):
        self.hot_n  = hot_n
        self.warm_n = warm_n
        self._hot:  List[str] = []
        self._warm: List[str] = []
        self._cold: set = set()

    def lookup(self, app: str) -> str:
        """Return 'hot', 'warm', 'cold', or 'miss'."""
        if app in self._hot:
            return "hot"
        if app in self._warm:
            return "warm"
        if app in self._cold:
            return "cold"
        return "miss"

    def access(self, app: str) -> str:
        """Record an access. Promote tiers. Return tier before access."""
        tier = self.lookup(app)
        # Promote to HOT
        if app in self._hot:
            self._hot.remove(app)
        elif app in self._warm:
            self._warm.remove(app)
        self._hot.insert(0, app)
        # Evict HOT overflow to WARM
        while len(self._hot) > self.hot_n:
            ev = self._hot.pop()
            self._warm.insert(0, ev)
        # Evict WARM overflow to COLD
        while len(self._warm) > self.warm_n:
            ev = self._warm.pop()
            self._cold.add(ev)
        return tier

    def prefetch(self, apps: List[str]):
        """Pre-warm WARM tier with predicted apps."""
        for app in apps:
            if app not in self._hot and app not in self._warm:
                self._warm.insert(0, app)
                while len(self._warm) > self.warm_n:
                    ev = self._warm.pop()
                    self._cold.add(ev)

    def memory_mb(self) -> float:
        n_hot  = len(self._hot)
        n_warm = len(self._warm)
        n_cold = len(self._cold)
        return (n_hot + n_warm + n_cold) * 8192 / (1024 * 1024)

    def reset(self):
        self._hot = []; self._warm = []; self._cold = set()


# ── Policies ─────────────────────────────────────────────────────────────────

class Policy:
    name = "Base"

    def train(self, events: List[str]):
        pass

    def predict(self, current: str, history: List[str]) -> List[str]:
        return []

    def update(self, event: str):
        pass

    def reset(self):
        pass


class RandomPolicy(Policy):
    name = "Random"

    def __init__(self, k: int = HOT_SIZE):
        self._vocab: List[str] = []
        self._k = k

    def train(self, events: List[str]):
        self._vocab = list(set(events))

    def predict(self, current: str, history: List[str]) -> List[str]:
        if not self._vocab:
            return []
        return random.sample(self._vocab, min(self._k, len(self._vocab)))


class LRUPolicy(Policy):
    name = "LRU"

    def __init__(self, k: int = HOT_SIZE):
        self._k = k
        self._queue: deque = deque()
        self._seen: set = set()

    def update(self, event: str):
        if event in self._seen:
            self._queue.remove(event)
        else:
            self._seen.add(event)
        self._queue.appendleft(event)

    def predict(self, current: str, history: List[str]) -> List[str]:
        return [x for x in list(self._queue) if x != current][: self._k]

    def reset(self):
        self._queue.clear(); self._seen.clear()


class LFUPolicy(Policy):
    name = "LFU"

    def __init__(self, k: int = HOT_SIZE):
        self._k = k
        self._freq: Dict[str, int] = defaultdict(int)

    def update(self, event: str):
        self._freq[event] += 1

    def predict(self, current: str, history: List[str]) -> List[str]:
        sorted_apps = sorted(self._freq.keys(), key=lambda a: -self._freq[a])
        return [a for a in sorted_apps if a != current][: self._k]

    def reset(self):
        self._freq.clear()


class MRUPolicy(Policy):
    name = "MRU"

    def __init__(self, k: int = HOT_SIZE):
        self._k = k
        self._stack: List[str] = []

    def update(self, event: str):
        if event in self._stack:
            self._stack.remove(event)
        self._stack.append(event)

    def predict(self, current: str, history: List[str]) -> List[str]:
        return [x for x in reversed(self._stack) if x != current][: self._k]

    def reset(self):
        self._stack.clear()


class FrequencyPolicy(Policy):
    name = "Frequency"

    def __init__(self, k: int = HOT_SIZE):
        self._k = k
        self._freq: Dict[str, int] = defaultdict(int)

    def train(self, events: List[str]):
        for e in events:
            self._freq[e] += 1

    def update(self, event: str):
        self._freq[event] += 1

    def predict(self, current: str, history: List[str]) -> List[str]:
        top = sorted(self._freq.keys(), key=lambda a: -self._freq[a])
        return [a for a in top if a != current][: self._k]

    def reset(self):
        self._freq.clear()


class RecencyFrequencyPolicy(Policy):
    name = "RecencyFrequency"

    def __init__(self, k: int = HOT_SIZE, alpha: float = 0.5, beta: float = 0.5, decay: float = 0.95):
        self._k = k
        self._alpha = alpha
        self._beta  = beta
        self._decay = decay
        self._freq:    Dict[str, float] = defaultdict(float)
        self._recency: Dict[str, float] = defaultdict(float)
        self._total: float = 0.0

    def update(self, event: str):
        for k in self._recency:
            self._recency[k] *= self._decay
        self._recency[event] = 1.0
        self._freq[event] += 1
        self._total += 1

    def predict(self, current: str, history: List[str]) -> List[str]:
        tot = self._total or 1.0
        scores = {
            a: self._alpha * self._recency[a] + self._beta * (self._freq[a] / tot)
            for a in self._freq
        }
        top = sorted(scores.keys(), key=lambda a: -scores[a])
        return [a for a in top if a != current][: self._k]

    def reset(self):
        self._freq.clear(); self._recency.clear(); self._total = 0.0


class Markov1Policy(Policy):
    name = "Markov-1"

    def __init__(self, k: int = HOT_SIZE):
        self._k = k
        self._matrix: Dict[str, Dict[str, float]] = {}

    def train(self, events: List[str]):
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(events)):
            counts[events[i - 1]][events[i]] += 1
        for src, dests in counts.items():
            total = sum(dests.values())
            self._matrix[src] = {d: c / total for d, c in sorted(dests.items(), key=lambda x: -x[1])}

    def predict(self, current: str, history: List[str]) -> List[str]:
        if current not in self._matrix:
            return []
        return list(self._matrix[current].keys())[: self._k]


class Markov2Policy(Policy):
    name = "Markov-2"

    def __init__(self, k: int = HOT_SIZE, fallback: Optional["Markov1Policy"] = None):
        self._k = k
        self._matrix: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._fallback = fallback

    def train(self, events: List[str]):
        counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(2, len(events)):
            bigram = (events[i - 2], events[i - 1])
            counts[bigram][events[i]] += 1
        for bigram, dests in counts.items():
            total = sum(dests.values())
            self._matrix[bigram] = {d: c / total for d, c in sorted(dests.items(), key=lambda x: -x[1])}

    def predict(self, current: str, history: List[str]) -> List[str]:
        if len(history) >= 1:
            bigram = (history[-1], current)
            if bigram in self._matrix:
                return list(self._matrix[bigram].keys())[: self._k]
        if self._fallback:
            return self._fallback.predict(current, history)
        return []


class GraphOnlyPolicy(Policy):
    name = "GraphOnly"

    def __init__(self, k: int = HOT_SIZE):
        self._k = k
        self._graph: Dict[str, Dict[str, float]] = {}
        self._freq:  Dict[str, int] = defaultdict(int)

    def train(self, events: List[str]):
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(events)):
            counts[events[i - 1]][events[i]] += 1
            self._freq[events[i - 1]] += 1
        for src, dests in counts.items():
            total = sum(dests.values())
            self._graph[src] = {d: c / total for d, c in sorted(dests.items(), key=lambda x: -x[1])}

    def predict(self, current: str, history: List[str]) -> List[str]:
        if current not in self._graph:
            return []
        return list(self._graph[current].keys())[: self._k]


class GraphConfidencePolicy(Policy):
    """Graph + recency/frequency confidence re-ranking."""
    name = "Graph+Confidence"

    def __init__(self, k: int = HOT_SIZE, threshold: float = 0.05):
        self._k = k
        self._threshold = threshold
        self._graph: Dict[str, Dict[str, float]] = {}
        self._recency: Dict[str, float] = defaultdict(float)
        self._freq:    Dict[str, float] = defaultdict(float)
        self._total: float = 0.0

    def train(self, events: List[str]):
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(events)):
            counts[events[i - 1]][events[i]] += 1
        for src, dests in counts.items():
            total = sum(dests.values())
            self._graph[src] = {d: c / total for d, c in sorted(dests.items(), key=lambda x: -x[1])}

    def update(self, event: str):
        for k in self._recency:
            self._recency[k] *= 0.95
        self._recency[event] = 1.0
        self._freq[event] += 1
        self._total += 1

    def predict(self, current: str, history: List[str]) -> List[str]:
        if current not in self._graph:
            return []
        tot = self._total or 1.0
        candidates = {}
        for app, trans_prob in self._graph[current].items():
            rec = self._recency.get(app, 0.0)
            freq = self._freq.get(app, 0.0) / tot
            conf = 0.5 * trans_prob + 0.3 * rec + 0.2 * freq
            if conf >= self._threshold:
                candidates[app] = conf
        top = sorted(candidates.keys(), key=lambda a: -candidates[a])
        return top[: self._k]


class GraphMindRLPolicy(Policy):
    """GraphMind RL: graph predictions + PPO-trained budget allocation.
    
    For evaluation without a trained PPO model, we use a heuristic that
    dynamically adjusts HOT budget based on hit rate history.
    """
    name = "GraphMindRL"

    def __init__(self, k: int = HOT_SIZE):
        self._k = k
        self._graph: Dict[str, Dict[str, float]] = {}
        self._recency: Dict[str, float] = defaultdict(float)
        self._freq:    Dict[str, float] = defaultdict(float)
        self._total: float = 0.0
        self._hit_history: deque = deque(maxlen=20)
        # RL state: dynamic budget
        self._hot_budget = HOT_SIZE
        self._conf_threshold = 0.05

    def train(self, events: List[str]):
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(events)):
            counts[events[i - 1]][events[i]] += 1
        for src, dests in counts.items():
            total = sum(dests.values())
            self._graph[src] = {d: c / total for d, c in sorted(dests.items(), key=lambda x: -x[1])}

    def update(self, event: str, hit: bool = False):
        for k in self._recency:
            self._recency[k] *= 0.95
        self._recency[event] = 1.0
        self._freq[event] += 1
        self._total += 1
        self._hit_history.append(1.0 if hit else 0.0)
        # RL heuristic: adjust budget based on recent hit rate
        if len(self._hit_history) == 20:
            recent_hr = sum(self._hit_history) / 20
            if recent_hr < 0.3:
                self._hot_budget = min(HOT_SIZE + 2, 8)
                self._conf_threshold = 0.03
            elif recent_hr > 0.7:
                self._hot_budget = max(HOT_SIZE - 1, 3)
                self._conf_threshold = 0.08
            else:
                self._hot_budget = HOT_SIZE
                self._conf_threshold = 0.05

    def predict(self, current: str, history: List[str]) -> List[str]:
        if current not in self._graph:
            return []
        tot = self._total or 1.0
        candidates = {}
        for app, trans_prob in self._graph[current].items():
            rec  = self._recency.get(app, 0.0)
            freq = self._freq.get(app, 0.0) / tot
            conf = 0.5 * trans_prob + 0.3 * rec + 0.2 * freq
            if conf >= self._conf_threshold:
                candidates[app] = conf
        top = sorted(candidates.keys(), key=lambda a: -candidates[a])
        return top[: self._hot_budget]


# ── Evaluation ───────────────────────────────────────────────────────────────

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


def load_user_events(user_id: str) -> List[str]:
    """Load chronological app sequence for one user."""
    user_dir = os.path.join(UBIQLOG_ROOT, user_id)
    events = []
    for fname in sorted(os.listdir(user_dir)):
        if not fname.endswith(".txt"):
            continue
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
    return [pkg for _, pkg in events]


def evaluate_policy(policy: Policy, train_events: List[str], test_events: List[str],
                    latency_model: MeasuredLatencyModel) -> dict:
    """Run a policy on the test split, return all 10 metrics."""
    # Train
    t0 = time.perf_counter()
    policy.train(train_events)
    train_time = time.perf_counter() - t0

    policy.reset()  # reset online state for test

    cache = CacheSimulator()
    hits = misses = tp = fp = fn = thrash = prefetched = 0
    latency_saved_total = 0.0
    pred_times = []
    history: List[str] = []

    # Warm up cache with a few train events
    for pkg in train_events[-20:]:
        cache.access(pkg)

    for i, pkg in enumerate(test_events):
        # Predict BEFORE access
        t_pred = time.perf_counter()
        preds = policy.predict(pkg, history[-3:])
        pred_time_ms = (time.perf_counter() - t_pred) * 1000
        pred_times.append(pred_time_ms)

        # Prefetch
        if preds:
            cache.prefetch(preds[:HOT_SIZE])
            prefetched += len(preds[:HOT_SIZE])

        # Check hit/miss before access
        tier_before = cache.lookup(pkg)
        is_hit = tier_before in ("hot", "warm")

        if is_hit:
            hits += 1
            tp += 1
            latency_saved_total += latency_model.saved_ms(pkg, tier_before)
        else:
            misses += 1

        # Compute false prefetches: predicted but not the next actual app
        if preds:
            correct = 1 if (i + 1 < len(test_events) and test_events[i + 1] in preds) else 0
            fp += len(preds) - correct
            tp_pref = correct
            fn += 1 if (i + 1 < len(test_events) and test_events[i + 1] not in preds) else 0

        # Check for thrash: did we evict something we just accessed?
        hot_before = set(cache._hot)
        cache.access(pkg)
        hot_after = set(cache._hot)
        evicted = hot_before - hot_after
        if any(e in train_events[-50:] for e in evicted):
            thrash += 1

        # Update policy
        if hasattr(policy, "update"):
            if policy.name == "GraphMindRL":
                policy.update(pkg, hit=is_hit)
            else:
                policy.update(pkg)

        history.append(pkg)

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    lat_pct = (latency_saved_total / (total * latency_model.cold_ms("instagram") or 1)) * 100
    fpr  = fp / prefetched if prefetched > 0 else 0.0
    thrash_rate = thrash / total if total > 0 else 0.0
    mem_mb = cache.memory_mb()
    pred_lat = float(np.mean(pred_times)) if pred_times else 0.0

    return {
        "hit_rate":           round(hit_rate, 4),
        "precision":          round(prec, 4),
        "recall":             round(rec, 4),
        "f1":                 round(f1, 4),
        "latency_saved_ms":   round(latency_saved_total / total if total > 0 else 0.0, 2),
        "latency_saved_pct":  round(lat_pct, 2),
        "false_prefetch_rate":round(fpr, 4),
        "thrash_rate":        round(thrash_rate, 4),
        "memory_usage_mb":    round(mem_mb, 3),
        "prediction_latency_ms": round(pred_lat, 4),
    }


def make_policies():
    m1 = Markov1Policy()
    m2 = Markov2Policy(fallback=m1)
    return [
        RandomPolicy(),
        LRUPolicy(),
        LFUPolicy(),
        MRUPolicy(),
        FrequencyPolicy(),
        RecencyFrequencyPolicy(),
        m1,
        m2,
        GraphOnlyPolicy(),
        GraphConfidencePolicy(),
        GraphMindRLPolicy(),
    ]


def percentile(values, p):
    return float(np.percentile(values, p)) if values else 0.0


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load usable users
    users_json_path = os.path.join(PROCESSED_DIR, "users.json")
    with open(users_json_path, encoding="utf-8") as f:
        users_data = json.load(f)
    usable_users = [u["user_id"] for u in users_data["users"]]
    logger.info(f"Benchmarking {len(usable_users)} users with 11 policies")

    latency_model = MeasuredLatencyModel(LATENCY_CSV)

    METRICS = [
        "hit_rate", "precision", "recall", "f1",
        "latency_saved_ms", "latency_saved_pct",
        "false_prefetch_rate", "thrash_rate",
        "memory_usage_mb", "prediction_latency_ms",
    ]

    all_rows = []  # per-user per-policy rows
    # policy_name → metric → list of values across users
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for user_id in usable_users:
        logger.info(f"User {user_id}...")
        events = load_user_events(user_id)
        if len(events) < 200:
            logger.warning(f"  {user_id}: only {len(events)} events, skipping")
            continue

        n = len(events)
        train_end = int(n * TRAIN_RATIO)
        val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
        train_events = events[:train_end]
        test_events  = events[val_end:]

        if len(test_events) < 10:
            logger.warning(f"  {user_id}: test split too small ({len(test_events)}), skipping")
            continue

        policies = make_policies()
        for policy in policies:
            try:
                metrics = evaluate_policy(policy, train_events, test_events, latency_model)
                row = {"user_id": user_id, "policy": policy.name}
                row.update(metrics)
                all_rows.append(row)
                for m, v in metrics.items():
                    agg[policy.name][m].append(v)
                logger.info(
                    f"  {policy.name:20s}: HR={metrics['hit_rate']:.3f} "
                    f"F1={metrics['f1']:.3f} "
                    f"LatSaved={metrics['latency_saved_ms']:.0f}ms"
                )
            except Exception as exc:
                logger.error(f"  {user_id}/{policy.name}: {exc}")

    # Write per-user results
    results_path = os.path.join(RESULTS_DIR, "benchmark_results_v2.csv")
    if all_rows:
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    logger.info(f"Written: {results_path} ({len(all_rows)} rows)")

    # Write aggregated advanced metrics
    adv_rows = []
    for policy_name in [p.name for p in make_policies()]:
        if policy_name not in agg:
            continue
        row = {"policy": policy_name}
        for m in METRICS:
            vals = agg[policy_name][m]
            if vals:
                row[f"{m}_mean"]   = round(float(np.mean(vals)), 4)
                row[f"{m}_median"] = round(float(np.median(vals)), 4)
                row[f"{m}_std"]    = round(float(np.std(vals)), 4)
                row[f"{m}_p50"]    = round(percentile(vals, 50), 4)
                row[f"{m}_p90"]    = round(percentile(vals, 90), 4)
                row[f"{m}_p95"]    = round(percentile(vals, 95), 4)
                row[f"{m}_p99"]    = round(percentile(vals, 99), 4)
        adv_rows.append(row)

    adv_path = os.path.join(RESULTS_DIR, "advanced_metrics_v2.csv")
    if adv_rows:
        with open(adv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(adv_rows[0].keys()))
            w.writeheader()
            w.writerows(adv_rows)
    logger.info(f"Written: {adv_path}")

    # Console summary
    logger.info("\n=== BENCHMARK SUMMARY (mean over users) ===")
    logger.info(f"{'Policy':22s} {'HitRate':>8} {'F1':>8} {'LatSaved':>10} {'FalseP':>8}")
    logger.info("-" * 62)
    for row in sorted(adv_rows, key=lambda r: -r.get("hit_rate_mean", 0)):
        logger.info(
            f"{row['policy']:22s} "
            f"{row.get('hit_rate_mean', 0):8.3f} "
            f"{row.get('f1_mean', 0):8.3f} "
            f"{row.get('latency_saved_ms_mean', 0):10.1f} "
            f"{row.get('false_prefetch_rate_mean', 0):8.3f}"
        )


if __name__ == "__main__":
    main()
