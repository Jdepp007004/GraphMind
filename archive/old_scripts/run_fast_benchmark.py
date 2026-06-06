#!/usr/bin/env python3
"""
scripts/run_fast_benchmark.py

Phase 4: Fast benchmark — 6 policies on 31 users.

Policies:
  1. Markov-1
  2. Markov-2
  3. GlobalMarkov2     ← cross-user population baseline
  4. GraphOnly
  5. Graph+Confidence
  6. GraphMindRL

Uses chronological 80/10/10 splits.
Uses measured Galaxy A23 latency (cold/warm/hot, 100 samples each).

Outputs:
  results/benchmark_results_fast.csv   (per-user × per-policy rows)
  results/user_level_results.csv       (subset cols: user_id, policy, 6 metrics)
  results/advanced_metrics_fast.csv    (aggregated: mean/median/std/P50/P95/CI95)
  reports/fast_benchmark_report.md
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
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MARKOV_DIR    = os.path.join(PROCESSED_DIR, "markov")
UBIQLOG_ROOT  = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog", "UbiqLog4UCI")
LATENCY_CSV   = os.path.join(PROJECT_ROOT, "datasets", "app_launch_latency.csv")

HOT_SIZE   = 5
WARM_SIZE  = 15
MIN_YEAR, MAX_YEAR = 2011, 2016
TRAIN_RATIO, VAL_RATIO = 0.80, 0.10

SYSTEM_PREFIXES = (
    "com.android.", "com.google.android.providers",
    "com.google.android.gms", "com.google.android.gsf",
    "com.sec.android.provider", "com.samsung.android.provider",
    "com.redbend.", "android.",
)
SYSTEM_SUFFIXES = (":engine", ":client", ":daemon", ":service", ":pushservice", ":sync")

METRICS = ["hit_rate","precision","recall","f1","latency_saved_ms","latency_saved_pct"]


# ── Latency Model ──────────────────────────────────────────────────────────

class MeasuredLatencyModel:
    _DEFAULT_COLD = 2763.0
    _DEFAULT_WARM = 1301.0
    _DEFAULT_HOT  =  274.0

    def __init__(self, csv_path: str):
        self._cold: Dict[str, float] = {}
        self._warm: Dict[str, float] = {}
        self._hot:  Dict[str, float] = {}
        self._pkg_map: Dict[str, str] = {}
        if os.path.exists(csv_path):
            self._load(csv_path)

    def _load(self, path: str):
        buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                aid = r["app_id"]; st = r["start_type"]
                self._pkg_map[r["package_name"]] = aid
                buckets[aid][st].append(float(r["total_time_ms"]))
        for aid, tiers in buckets.items():
            if "cold" in tiers: self._cold[aid] = float(np.mean(tiers["cold"]))
            if "warm" in tiers: self._warm[aid] = float(np.mean(tiers["warm"]))
            if "hot"  in tiers: self._hot[aid]  = float(np.mean(tiers["hot"]))

    def _key(self, pkg: str) -> Optional[str]:
        if pkg in self._cold: return pkg
        return self._pkg_map.get(pkg)

    def cold_ms(self, pkg: str) -> float:
        k = self._key(pkg)
        return self._cold.get(k, self._DEFAULT_COLD) if k else self._DEFAULT_COLD

    def warm_ms(self, pkg: str) -> float:
        k = self._key(pkg)
        return self._warm.get(k, self._DEFAULT_WARM) if k else self._DEFAULT_WARM

    def hot_ms(self, pkg: str) -> float:
        k = self._key(pkg)
        return self._hot.get(k, self._DEFAULT_HOT) if k else self._DEFAULT_HOT

    def saved_ms(self, pkg: str, tier: str) -> float:
        c = self.cold_ms(pkg)
        if tier == "hot":  return max(0.0, c - self.hot_ms(pkg))
        if tier == "warm": return max(0.0, c - self.warm_ms(pkg))
        return 0.0


# ── Cache Simulator ────────────────────────────────────────────────────────

class Cache:
    def __init__(self):
        self._hot: List[str] = []
        self._warm: List[str] = []

    def lookup(self, app: str) -> str:
        if app in self._hot:  return "hot"
        if app in self._warm: return "warm"
        return "miss"

    def access(self, app: str):
        if app in self._hot:  self._hot.remove(app)
        elif app in self._warm: self._warm.remove(app)
        self._hot.insert(0, app)
        while len(self._hot) > HOT_SIZE:
            self._warm.insert(0, self._hot.pop())
        while len(self._warm) > WARM_SIZE:
            self._warm.pop()

    def prefetch(self, apps: List[str]):
        for app in apps:
            if app not in self._hot and app not in self._warm:
                self._warm.insert(0, app)
                while len(self._warm) > WARM_SIZE:
                    self._warm.pop()

    def reset(self):
        self._hot = []; self._warm = []


# ── Policy Base ────────────────────────────────────────────────────────────

class Policy:
    name = "Base"
    def train(self, events: List[str]): pass
    def predict(self, current: str, history: List[str]) -> List[str]: return []
    def update(self, event: str, hit: bool = False): pass
    def reset(self): pass


class Markov1Policy(Policy):
    name = "Markov-1"
    def __init__(self):
        self._m: Dict[str, Dict[str, float]] = {}
    def train(self, events: List[str]):
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(events)):
            counts[events[i-1]][events[i]] += 1
        for src, d in counts.items():
            total = sum(d.values())
            self._m[src] = {k: v/total for k, v in sorted(d.items(), key=lambda x: -x[1])}
    def predict(self, current: str, history: List[str]) -> List[str]:
        return list(self._m.get(current, {}).keys())[:HOT_SIZE]


class Markov2Policy(Policy):
    name = "Markov-2"
    def __init__(self):
        self._m1: Dict[str, Dict[str, float]] = {}
        self._m2: Dict[Tuple[str,str], Dict[str, float]] = {}
    def train(self, events: List[str]):
        c1: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        c2: Dict[Tuple[str,str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(events)):
            c1[events[i-1]][events[i]] += 1
        for i in range(2, len(events)):
            c2[(events[i-2], events[i-1])][events[i]] += 1
        for src, d in c1.items():
            total = sum(d.values())
            self._m1[src] = {k: v/total for k, v in sorted(d.items(), key=lambda x: -x[1])}
        for bg, d in c2.items():
            total = sum(d.values())
            self._m2[bg] = {k: v/total for k, v in sorted(d.items(), key=lambda x: -x[1])}
    def predict(self, current: str, history: List[str]) -> List[str]:
        if history:
            bg = (history[-1], current)
            if bg in self._m2:
                return list(self._m2[bg].keys())[:HOT_SIZE]
        return list(self._m1.get(current, {}).keys())[:HOT_SIZE]


class GlobalMarkov2Policy(Policy):
    """Cross-user second-order Markov trained on all users' training data."""
    name = "GlobalMarkov2"

    def __init__(self, global_data: Optional[dict] = None):
        self._m1: Dict[str, Dict[str, float]] = {}
        self._m2: Dict[Tuple[str,str], Dict[str, float]] = {}
        if global_data:
            self._m2 = global_data.get("markov2", {})
            self._m1 = global_data.get("fallback_m1", {})

    def train(self, events: List[str]):
        # No-op: already trained globally. This call is required by interface
        # but GlobalMarkov2 ignores per-user training data.
        pass

    def predict(self, current: str, history: List[str]) -> List[str]:
        if history:
            bg = (history[-1], current)
            if bg in self._m2:
                return list(self._m2[bg].keys())[:HOT_SIZE]
        return list(self._m1.get(current, {}).keys())[:HOT_SIZE]


class GraphOnlyPolicy(Policy):
    name = "GraphOnly"
    def __init__(self):
        self._g: Dict[str, Dict[str, float]] = {}
    def train(self, events: List[str]):
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(events)):
            counts[events[i-1]][events[i]] += 1
        for src, d in counts.items():
            total = sum(d.values())
            self._g[src] = {k: v/total for k, v in sorted(d.items(), key=lambda x: -x[1])}
    def predict(self, current: str, history: List[str]) -> List[str]:
        return list(self._g.get(current, {}).keys())[:HOT_SIZE]


class GraphConfidencePolicy(Policy):
    name = "Graph+Confidence"
    def __init__(self, threshold: float = 0.05):
        self._g: Dict[str, Dict[str, float]] = {}
        self._rec: Dict[str, float] = defaultdict(float)
        self._freq: Dict[str, float] = defaultdict(float)
        self._total: float = 0.0
        self._threshold = threshold
    def train(self, events: List[str]):
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(events)):
            counts[events[i-1]][events[i]] += 1
        for src, d in counts.items():
            total = sum(d.values())
            self._g[src] = {k: v/total for k, v in sorted(d.items(), key=lambda x: -x[1])}
    def update(self, event: str, hit: bool = False):
        for k in self._rec: self._rec[k] *= 0.95
        self._rec[event] = 1.0
        self._freq[event] += 1
        self._total += 1
    def predict(self, current: str, history: List[str]) -> List[str]:
        if current not in self._g: return []
        tot = self._total or 1.0
        candidates = {}
        for app, tp in self._g[current].items():
            conf = 0.5*tp + 0.3*self._rec.get(app,0) + 0.2*(self._freq.get(app,0)/tot)
            if conf >= self._threshold:
                candidates[app] = conf
        return sorted(candidates, key=lambda a: -candidates[a])[:HOT_SIZE]
    def reset(self):
        self._rec.clear(); self._freq.clear(); self._total = 0.0


class GraphMindRLPolicy(Policy):
    name = "GraphMindRL"
    def __init__(self):
        self._g: Dict[str, Dict[str, float]] = {}
        self._rec: Dict[str, float] = defaultdict(float)
        self._freq: Dict[str, float] = defaultdict(float)
        self._total: float = 0.0
        self._hit_hist: deque = deque(maxlen=20)
        self._budget = HOT_SIZE
        self._thresh = 0.05
    def train(self, events: List[str]):
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(events)):
            counts[events[i-1]][events[i]] += 1
        for src, d in counts.items():
            total = sum(d.values())
            self._g[src] = {k: v/total for k, v in sorted(d.items(), key=lambda x: -x[1])}
    def update(self, event: str, hit: bool = False):
        for k in self._rec: self._rec[k] *= 0.95
        self._rec[event] = 1.0
        self._freq[event] += 1
        self._total += 1
        self._hit_hist.append(1.0 if hit else 0.0)
        if len(self._hit_hist) == 20:
            hr = sum(self._hit_hist) / 20
            if hr < 0.3:   self._budget = min(HOT_SIZE+2, 8); self._thresh = 0.03
            elif hr > 0.7: self._budget = max(HOT_SIZE-1, 3); self._thresh = 0.08
            else:          self._budget = HOT_SIZE;            self._thresh = 0.05
    def predict(self, current: str, history: List[str]) -> List[str]:
        if current not in self._g: return []
        tot = self._total or 1.0
        candidates = {}
        for app, tp in self._g[current].items():
            conf = 0.5*tp + 0.3*self._rec.get(app,0) + 0.2*(self._freq.get(app,0)/tot)
            if conf >= self._thresh:
                candidates[app] = conf
        return sorted(candidates, key=lambda a: -candidates[a])[:self._budget]
    def reset(self):
        self._rec.clear(); self._freq.clear(); self._total = 0.0
        self._hit_hist.clear(); self._budget = HOT_SIZE; self._thresh = 0.05


# ── Data Loading ───────────────────────────────────────────────────────────

def parse_ts(s: str) -> Optional[datetime]:
    try:
        dt = datetime.strptime(s.strip(), "%m-%d-%Y %H:%M:%S")
        return dt if MIN_YEAR <= dt.year <= MAX_YEAR else None
    except Exception:
        return None

def is_system_app(p: str) -> bool:
    p = p.lower()
    for pfx in SYSTEM_PREFIXES:
        if p.startswith(pfx): return True
    for sfx in SYSTEM_SUFFIXES:
        if p.endswith(sfx): return True
    return False

def load_events(user_id: str) -> List[str]:
    user_dir = os.path.join(UBIQLOG_ROOT, user_id)
    events = []
    for fname in sorted(os.listdir(user_dir)):
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
                        pkg = app.get("ProcessName","").strip()
                        if not pkg or is_system_app(pkg): continue
                        start = parse_ts(app.get("Start",""))
                        if start is None: continue
                        events.append((start, pkg))
                    except Exception:
                        pass
        except Exception:
            pass
    events.sort(key=lambda x: x[0])
    return [pkg for _, pkg in events]


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_policy(
    policy: Policy,
    train_events: List[str],
    test_events: List[str],
    lat: MeasuredLatencyModel,
) -> dict:
    policy.train(train_events)
    policy.reset()

    cache = Cache()
    # Warm up with last 20 train events
    for pkg in train_events[-20:]:
        cache.access(pkg)

    hits = misses = tp = fp = fn = 0
    lat_saved = 0.0
    history: List[str] = []

    for i, pkg in enumerate(test_events):
        preds = policy.predict(pkg, history[-3:])
        if preds:
            cache.prefetch(preds)

        tier = cache.lookup(pkg)
        is_hit = tier in ("hot","warm")

        if is_hit:
            hits += 1; tp += 1
            lat_saved += lat.saved_ms(pkg, tier)
        else:
            misses += 1

        if i + 1 < len(test_events):
            nxt = test_events[i+1]
            if preds:
                if nxt in preds: tp += 1
                else: fn += 1; fp += len(preds)
            else:
                fn += 1

        cache.access(pkg)
        policy.update(pkg, hit=is_hit)
        history.append(pkg)

    total = hits + misses or 1
    hr = hits / total
    pr = tp / (tp+fp) if (tp+fp) > 0 else 0.0
    re = tp / (tp+fn) if (tp+fn) > 0 else 0.0
    f1 = 2*pr*re/(pr+re) if (pr+re) > 0 else 0.0
    avg_cold = 2763.0  # overall average cold ms across 13 apps
    lat_pct = (lat_saved / total / avg_cold * 100) if total > 0 else 0.0

    return {
        "hit_rate":          round(hr, 4),
        "precision":         round(pr, 4),
        "recall":            round(re, 4),
        "f1":                round(f1, 4),
        "latency_saved_ms":  round(lat_saved/total, 2),
        "latency_saved_pct": round(lat_pct, 2),
    }


def bootstrap_ci(values: List[float], n: int = 1000) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boots = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    return (round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load usable users
    with open(os.path.join(PROCESSED_DIR, "users.json"), encoding="utf-8") as f:
        usable_users = [u["user_id"] for u in json.load(f)["users"]]

    lat = MeasuredLatencyModel(LATENCY_CSV)

    # Load global Markov2
    gm2_path = os.path.join(MARKOV_DIR, "global_markov2.pkl")
    global_data = None
    if os.path.exists(gm2_path):
        with open(gm2_path, "rb") as f:
            global_data = pickle.load(f)
        logger.info(f"Loaded GlobalMarkov2: {len(global_data['markov2']):,} bigram states")
    else:
        logger.warning("global_markov2.pkl not found — GlobalMarkov2 will use empty model")

    POLICY_NAMES = [
        "Markov-1", "Markov-2", "GlobalMarkov2",
        "GraphOnly", "Graph+Confidence", "GraphMindRL"
    ]

    def make_policies():
        return [
            Markov1Policy(),
            Markov2Policy(),
            GlobalMarkov2Policy(global_data),
            GraphOnlyPolicy(),
            GraphConfidencePolicy(),
            GraphMindRLPolicy(),
        ]

    all_rows = []
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for user_id in usable_users:
        logger.info(f"User {user_id}...")
        events = load_events(user_id)
        if len(events) < 200:
            logger.warning(f"  {user_id}: too few events ({len(events)}), skipping")
            continue

        n = len(events)
        train_end = int(n * TRAIN_RATIO)
        val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
        train_events = events[:train_end]
        test_events  = events[val_end:]

        if len(test_events) < 10:
            continue

        for policy in make_policies():
            try:
                metrics = evaluate_policy(policy, train_events, test_events, lat)
                row = {"user_id": user_id, "policy": policy.name}
                row.update(metrics)
                all_rows.append(row)
                for m, v in metrics.items():
                    agg[policy.name][m].append(v)
                logger.info(
                    f"  {policy.name:20s}: HR={metrics['hit_rate']:.3f} "
                    f"F1={metrics['f1']:.3f} "
                    f"Lat={metrics['latency_saved_ms']:.0f}ms"
                )
            except Exception as exc:
                logger.error(f"  {user_id}/{policy.name}: {exc}")

    # Write benchmark_results_fast.csv
    fast_path = os.path.join(RESULTS_DIR, "benchmark_results_fast.csv")
    if all_rows:
        with open(fast_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
    logger.info(f"Written: {fast_path} ({len(all_rows)} rows)")

    # Write user_level_results.csv (subset)
    ul_cols = ["user_id","policy","hit_rate","precision","recall","f1","latency_saved_ms","latency_saved_pct"]
    ul_path = os.path.join(RESULTS_DIR, "user_level_results.csv")
    with open(ul_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ul_cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in ul_cols})
    logger.info(f"Written: {ul_path}")

    # Write advanced_metrics_fast.csv
    adv_rows = []
    for pol in POLICY_NAMES:
        if pol not in agg: continue
        row = {"policy": pol}
        for m in METRICS:
            vals = agg[pol][m]
            if not vals: continue
            ci = bootstrap_ci(vals)
            row.update({
                f"{m}_mean":   round(float(np.mean(vals)), 4),
                f"{m}_median": round(float(np.median(vals)), 4),
                f"{m}_std":    round(float(np.std(vals)), 4),
                f"{m}_p50":    round(float(np.percentile(vals, 50)), 4),
                f"{m}_p95":    round(float(np.percentile(vals, 95)), 4),
                f"{m}_ci95_lo":ci[0],
                f"{m}_ci95_hi":ci[1],
            })
        adv_rows.append(row)

    adv_path = os.path.join(RESULTS_DIR, "advanced_metrics_fast.csv")
    if adv_rows:
        with open(adv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(adv_rows[0].keys()))
            w.writeheader(); w.writerows(adv_rows)
    logger.info(f"Written: {adv_path}")

    # Write markdown report
    md_path = os.path.join(REPORTS_DIR, "fast_benchmark_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# GraphMind Fast Benchmark Report\n\n")
        f.write(f"**Users:** {len(usable_users)} usable UbiqLog users  \n")
        f.write(f"**Policies:** {len(POLICY_NAMES)}  \n")
        f.write(f"**Split:** 80% train / 10% val / 10% test (chronological)  \n")
        f.write(f"**Latency source:** Measured — Samsung Galaxy A23, {LATENCY_CSV.split(os.sep)[-1]}  \n\n")
        f.write("---\n\n")

        f.write("## Hit Rate (mean ± std, 95% CI)\n\n")
        f.write("| Policy | Mean | Median | Std | P95 | 95% CI |\n")
        f.write("|--------|------|--------|-----|-----|--------|\n")
        for row in sorted(adv_rows, key=lambda r: -r.get("hit_rate_mean",0)):
            f.write(
                f"| **{row['policy']}** "
                f"| {row.get('hit_rate_mean',0):.4f} "
                f"| {row.get('hit_rate_median',0):.4f} "
                f"| {row.get('hit_rate_std',0):.4f} "
                f"| {row.get('hit_rate_p95',0):.4f} "
                f"| [{row.get('hit_rate_ci95_lo',0):.4f}, {row.get('hit_rate_ci95_hi',0):.4f}] |\n"
            )

        f.write("\n## F1 Score (mean ± std)\n\n")
        f.write("| Policy | Mean | Median | Std | P95 |\n")
        f.write("|--------|------|--------|-----|-----|\n")
        for row in sorted(adv_rows, key=lambda r: -r.get("f1_mean",0)):
            f.write(
                f"| **{row['policy']}** "
                f"| {row.get('f1_mean',0):.4f} "
                f"| {row.get('f1_median',0):.4f} "
                f"| {row.get('f1_std',0):.4f} "
                f"| {row.get('f1_p95',0):.4f} |\n"
            )

        f.write("\n## Latency Saved (ms, mean ± std)\n\n")
        f.write("| Policy | Mean | Median | Std | P95 |\n")
        f.write("|--------|------|--------|-----|-----|\n")
        for row in sorted(adv_rows, key=lambda r: -r.get("latency_saved_ms_mean",0)):
            f.write(
                f"| **{row['policy']}** "
                f"| {row.get('latency_saved_ms_mean',0):.1f} "
                f"| {row.get('latency_saved_ms_median',0):.1f} "
                f"| {row.get('latency_saved_ms_std',0):.1f} "
                f"| {row.get('latency_saved_ms_p95',0):.1f} |\n"
            )

    logger.info(f"Written: {md_path}")

    # Console summary
    logger.info("\n=== FAST BENCHMARK SUMMARY ===")
    logger.info(f"{'Policy':22s} {'HitRate':>8} {'F1':>8} {'LatSaved':>10}")
    logger.info("-"*52)
    for row in sorted(adv_rows, key=lambda r: -r.get("f1_mean",0)):
        logger.info(
            f"{row['policy']:22s} "
            f"{row.get('hit_rate_mean',0):8.3f} "
            f"{row.get('f1_mean',0):8.3f} "
            f"{row.get('latency_saved_ms_mean',0):10.1f}"
        )


if __name__ == "__main__":
    main()
