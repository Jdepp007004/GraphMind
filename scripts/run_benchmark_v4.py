#!/usr/bin/env python3
"""
scripts/run_benchmark_v4.py

GraphMind V4 — Full Benchmark Suite.

Policies (15):
  Classical baselines:
    1.  Random
    2.  LRU
    3.  LFU
    4.  Frequency
    5.  RecencyFrequency

  Markov baselines:
    6.  Markov-1            (personal, first-order)
    7.  Markov-2            (personal, second-order)
    8.  VariableOrderMarkov (personal, order 1+2 + Laplace)
    9.  ContextMarkov       (personal, time+weekday conditioned)
    10. ClusterMarkov       (personal → cluster → global)
    11. GlobalMarkov2       (population-level, cross-user)

  GraphMind stack:
    12. GraphOnly           (transition graph, top-k)
    13. Graph+Confidence    (graph + recency/freq confidence)
    14. GraphMindRL         (V3 cache allocator RL)
    15. RLAdaptiveEnsemble  (V4 REINFORCE predictor weights)

All policies are evaluated on the same 31 users, same 80/10/10 splits.

Outputs:
  results/benchmark_results_v4.csv
  results/user_level_results_v4.csv
  results/advanced_metrics_v4.csv
  reports/benchmark_v4_report.md
"""

import csv
import json
import logging
import math
import os
import pickle
import random
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path for model imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.variable_order_markov import VariableOrderMarkov
from src.models.context_markov import ContextMarkov
from src.models.cluster_markov import ClusterMarkov
from src.rl.adaptive_ensemble_env import AdaptiveEnsembleController

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MARKOV_DIR    = os.path.join(PROCESSED_DIR, "markov")
UBIQLOG_ROOT  = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog", "UbiqLog4UCI")
LATENCY_CSV   = os.path.join(PROJECT_ROOT, "datasets", "app_launch_latency.csv")

HOT_SIZE   = 5
WARM_SIZE  = 15
TRAIN_RATIO, VAL_RATIO = 0.80, 0.10
MIN_YEAR, MAX_YEAR = 2011, 2016

METRICS = ["hit_rate", "precision", "recall", "f1",
           "latency_saved_ms", "latency_saved_pct"]

SYSTEM_PREFIXES = (
    "com.android.", "com.google.android.providers",
    "com.google.android.gms", "com.google.android.gsf",
    "com.sec.android.provider", "com.samsung.android.provider",
    "com.redbend.", "android.",
)
SYSTEM_SUFFIXES = (":engine", ":client", ":daemon", ":service",
                   ":pushservice", ":sync")


# ── Latency Model ──────────────────────────────────────────────────────────

class MeasuredLatencyModel:
    _DEFAULT_COLD = 2763.0
    _DEFAULT_WARM = 1301.0
    _DEFAULT_HOT  =  274.0

    def __init__(self, path: str):
        self._cold: Dict[str, float] = {}
        self._warm: Dict[str, float] = {}
        self._hot:  Dict[str, float] = {}
        self._pkg:  Dict[str, str]   = {}
        if os.path.exists(path):
            b = defaultdict(lambda: defaultdict(list))
            with open(path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    b[r["app_id"]][r["start_type"]].append(float(r["total_time_ms"]))
                    self._pkg[r["package_name"]] = r["app_id"]
            for aid, tiers in b.items():
                if "cold" in tiers: self._cold[aid] = float(np.mean(tiers["cold"]))
                if "warm" in tiers: self._warm[aid] = float(np.mean(tiers["warm"]))
                if "hot"  in tiers: self._hot[aid]  = float(np.mean(tiers["hot"]))

    def _k(self, pkg: str) -> Optional[str]:
        if pkg in self._cold: return pkg
        return self._pkg.get(pkg)

    def saved(self, pkg: str, tier: str) -> float:
        k = self._k(pkg)
        cold = self._cold.get(k, self._DEFAULT_COLD) if k else self._DEFAULT_COLD
        if tier == "hot":
            hot = self._hot.get(k, self._DEFAULT_HOT) if k else self._DEFAULT_HOT
            return max(0.0, cold - hot)
        if tier == "warm":
            warm = self._warm.get(k, self._DEFAULT_WARM) if k else self._DEFAULT_WARM
            return max(0.0, cold - warm)
        return 0.0


# ── Cache Simulator ────────────────────────────────────────────────────────

class Cache:
    def __init__(self):
        self._hot:  List[str] = []
        self._warm: List[str] = []

    def lookup(self, app: str) -> str:
        if app in self._hot:  return "hot"
        if app in self._warm: return "warm"
        return "miss"

    def access(self, app: str):
        if app in self._hot:   self._hot.remove(app)
        elif app in self._warm: self._warm.remove(app)
        self._hot.insert(0, app)
        while len(self._hot) > HOT_SIZE:
            self._warm.insert(0, self._hot.pop())
        while len(self._warm) > WARM_SIZE:
            self._warm.pop()

    def prefetch(self, apps: List[str]):
        for a in apps:
            if a not in self._hot and a not in self._warm:
                self._warm.insert(0, a)
                while len(self._warm) > WARM_SIZE:
                    self._warm.pop()

    def reset(self):
        self._hot = []; self._warm = []


# ── Data Loading ───────────────────────────────────────────────────────────

def _is_system(p: str) -> bool:
    p = p.lower()
    for pfx in SYSTEM_PREFIXES:
        if p.startswith(pfx): return True
    for sfx in SYSTEM_SUFFIXES:
        if p.endswith(sfx): return True
    return False


def _parse_ts(s: str) -> Optional[datetime]:
    try:
        dt = datetime.strptime(s.strip(), "%m-%d-%Y %H:%M:%S")
        return dt if MIN_YEAR <= dt.year <= MAX_YEAR else None
    except Exception:
        return None


def load_events_with_context(user_id: str) -> Tuple[List[str], List[int], List[int]]:
    """Returns (apps, time_buckets, weekdays), sorted chronologically."""
    user_dir = os.path.join(UBIQLOG_ROOT, user_id)
    raw = []
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
                        pkg = app.get("ProcessName", "").strip()
                        if not pkg or _is_system(pkg): continue
                        dt = _parse_ts(app.get("Start", ""))
                        if dt is None: continue
                        # time_bucket: 30-min intervals (0-47)
                        tb = dt.hour * 2 + (1 if dt.minute >= 30 else 0)
                        wd = dt.weekday()  # 0=Mon, 6=Sun
                        raw.append((dt, pkg, tb, wd))
                    except Exception:
                        pass
        except Exception:
            pass
    raw.sort(key=lambda x: x[0])
    apps = [r[1] for r in raw]
    tbs  = [r[2] for r in raw]
    wds  = [r[3] for r in raw]
    return apps, tbs, wds


# ── Policy Base ────────────────────────────────────────────────────────────

class Policy:
    name = "Base"
    def train(self, apps, tbs=None, wds=None, val_apps=None, val_tbs=None, val_wds=None): pass
    def predict(self, cur: str, prev: Optional[str] = None,
                tb: int = 0, wd: int = 0) -> List[str]: return []
    def update(self, app: str, hit: bool = False): pass
    def reset(self): pass


# ── Classical Baselines ────────────────────────────────────────────────────

class RandomPolicy(Policy):
    name = "Random"
    def __init__(self):
        self._vocab: List[str] = []
        self._rng = np.random.default_rng(42)
    def train(self, apps, **kw):
        self._vocab = list(set(apps))
    def predict(self, cur, prev=None, tb=0, wd=0):
        if not self._vocab: return []
        idx = self._rng.choice(len(self._vocab), size=min(HOT_SIZE, len(self._vocab)), replace=False)
        return [self._vocab[i] for i in idx]


class LRUPolicy(Policy):
    name = "LRU"
    def __init__(self):
        self._recent: deque = deque(maxlen=WARM_SIZE)
    def train(self, apps, **kw):
        for a in apps[-20:]: self._recent.append(a)
    def predict(self, cur, prev=None, tb=0, wd=0):
        seen, out = set(), []
        for a in reversed(list(self._recent)):
            if a != cur and a not in seen:
                seen.add(a); out.append(a)
            if len(out) >= HOT_SIZE: break
        return out
    def update(self, app, hit=False):
        self._recent.append(app)
    def reset(self):
        self._recent.clear()


class LFUPolicy(Policy):
    name = "LFU"
    def __init__(self):
        self._freq: Dict[str, int] = defaultdict(int)
    def train(self, apps, **kw):
        for a in apps: self._freq[a] += 1
    def predict(self, cur, prev=None, tb=0, wd=0):
        cands = sorted(self._freq, key=lambda a: -self._freq[a])
        return [a for a in cands if a != cur][:HOT_SIZE]
    def update(self, app, hit=False):
        self._freq[app] += 1
    def reset(self):
        self._freq.clear()


class FrequencyPolicy(Policy):
    name = "Frequency"
    def __init__(self):
        self._freq: Dict[str, int] = defaultdict(int)
    def train(self, apps, **kw):
        for a in apps: self._freq[a] += 1
    def predict(self, cur, prev=None, tb=0, wd=0):
        return [a for a in sorted(self._freq, key=lambda x: -self._freq[x]) if a != cur][:HOT_SIZE]
    def update(self, app, hit=False):
        self._freq[app] += 1


class RecencyFrequencyPolicy(Policy):
    name = "RecencyFrequency"
    def __init__(self):
        self._freq:    Dict[str, float] = defaultdict(float)
        self._recency: Dict[str, float] = defaultdict(float)
        self._t: int = 0
    def train(self, apps, **kw):
        for a in apps:
            self._freq[a] += 1
            self._t += 1
            self._recency[a] = self._t
    def predict(self, cur, prev=None, tb=0, wd=0):
        t = self._t or 1
        max_f = max(self._freq.values()) if self._freq else 1
        cands = {a: 0.5*(self._freq[a]/max_f) + 0.5*(self._recency.get(a,0)/t)
                 for a in self._freq if a != cur}
        return sorted(cands, key=lambda a: -cands[a])[:HOT_SIZE]
    def update(self, app, hit=False):
        self._t += 1
        self._freq[app] += 1
        self._recency[app] = self._t


# ── Markov Policies ────────────────────────────────────────────────────────

class Markov1Policy(Policy):
    name = "Markov-1"
    def __init__(self):
        self._m: Dict[str, Dict[str, float]] = {}
    def train(self, apps, **kw):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c[apps[i-1]][apps[i]] += 1
        for s, d in c.items():
            t = sum(d.values())
            self._m[s] = dict(sorted({k: v/t for k,v in d.items()}.items(), key=lambda x:-x[1]))
    def predict(self, cur, prev=None, tb=0, wd=0):
        return list(self._m.get(cur, {}).keys())[:HOT_SIZE]


class Markov2Policy(Policy):
    name = "Markov-2"
    def __init__(self):
        self._m1: Dict[str, Dict[str, float]] = {}
        self._m2: Dict[Tuple[str,str], Dict[str, float]] = {}
    def train(self, apps, **kw):
        c1 = defaultdict(lambda: defaultdict(int))
        c2 = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c1[apps[i-1]][apps[i]] += 1
        for i in range(2, len(apps)):
            c2[(apps[i-2], apps[i-1])][apps[i]] += 1
        for s, d in c1.items():
            t = sum(d.values())
            self._m1[s] = dict(sorted({k: v/t for k,v in d.items()}.items(), key=lambda x:-x[1]))
        for bg, d in c2.items():
            t = sum(d.values())
            self._m2[bg] = dict(sorted({k: v/t for k,v in d.items()}.items(), key=lambda x:-x[1]))
    def predict(self, cur, prev=None, tb=0, wd=0):
        if prev:
            bg = (prev, cur)
            if bg in self._m2:
                return list(self._m2[bg].keys())[:HOT_SIZE]
        return list(self._m1.get(cur, {}).keys())[:HOT_SIZE]


class VOMPolicy(Policy):
    """Variable-Order Markov with Laplace smoothing."""
    name = "VariableOrderMarkov"
    def __init__(self):
        self._vom = VariableOrderMarkov(laplace_alpha=0.5, top_k=HOT_SIZE)
    def train(self, apps, **kw):
        self._vom.train(apps)
    def predict(self, cur, prev=None, tb=0, wd=0):
        return self._vom.predict_apps(cur, prev)


class ContextMarkovPolicy(Policy):
    """Context-aware Markov conditioned on time_bucket + weekday."""
    name = "ContextMarkov"
    def __init__(self):
        self._ctx = ContextMarkov(top_k=HOT_SIZE, laplace_alpha=0.3)
    def train(self, apps, tbs=None, wds=None, val_apps=None, val_tbs=None, val_wds=None, **kw):
        tbs = tbs or [0]*len(apps)
        wds = wds or [0]*len(apps)
        self._ctx.train(apps, tbs, wds)
        # Learn weights on val split if provided
        if val_apps and len(val_apps) > 5:
            self._ctx.fit_weights(
                val_apps,
                val_tbs or [0]*len(val_apps),
                val_wds or [0]*len(val_apps),
            )
    def predict(self, cur, prev=None, tb=0, wd=0):
        return self._ctx.predict_apps(cur, tb, wd)


class ClusterMarkovPolicy(Policy):
    """Cluster-level Markov: personal → cluster → global."""
    name = "ClusterMarkov"
    _cluster_model: Optional[ClusterMarkov] = None  # shared across instances

    def __init__(self):
        self._user_id: Optional[str] = None
        self._local_seq: List[str] = []

    @classmethod
    def fit_clusters(cls, user_sequences: Dict[str, List[str]]) -> None:
        """Call once before the benchmark loop to fit the shared cluster model."""
        cls._cluster_model = ClusterMarkov(n_clusters=4, top_k=HOT_SIZE)
        cls._cluster_model.fit(user_sequences)

    def train(self, apps, user_id: str = "unknown", **kw):
        self._user_id = user_id
        self._local_seq = apps
        if self._cluster_model is not None:
            self._cluster_model.train_user(user_id, apps)

    def predict(self, cur, prev=None, tb=0, wd=0):
        uid = self._user_id or "unknown"
        if self._cluster_model is not None:
            return self._cluster_model.predict_apps(uid, cur, prev)
        return []


class GlobalMarkov2Policy(Policy):
    """Population-level Markov-2 trained on all users."""
    name = "GlobalMarkov2"
    def __init__(self, global_data: Optional[dict] = None):
        self._m1: Dict[str, Dict[str, float]] = {}
        self._m2: Dict[Tuple[str,str], Dict[str, float]] = {}
        if global_data:
            self._m2 = global_data.get("markov2", {})
            self._m1 = global_data.get("fallback_m1", {})
    def train(self, apps, **kw): pass  # global model, no per-user training
    def predict(self, cur, prev=None, tb=0, wd=0):
        if prev:
            bg = (prev, cur)
            if bg in self._m2:
                return list(self._m2[bg].keys())[:HOT_SIZE]
        return list(self._m1.get(cur, {}).keys())[:HOT_SIZE]


# ── GraphMind Policies ─────────────────────────────────────────────────────

class GraphOnlyPolicy(Policy):
    name = "GraphOnly"
    def __init__(self):
        self._g: Dict[str, Dict[str, float]] = {}
    def train(self, apps, **kw):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c[apps[i-1]][apps[i]] += 1
        for s, d in c.items():
            t = sum(d.values())
            self._g[s] = dict(sorted({k: v/t for k,v in d.items()}.items(), key=lambda x:-x[1]))
    def predict(self, cur, prev=None, tb=0, wd=0):
        return list(self._g.get(cur, {}).keys())[:HOT_SIZE]


class GraphConfidencePolicy(Policy):
    name = "Graph+Confidence"
    def __init__(self):
        self._g:    Dict[str, Dict[str, float]] = {}
        self._rec:  Dict[str, float] = defaultdict(float)
        self._freq: Dict[str, float] = defaultdict(float)
        self._total = 0.0
    def train(self, apps, **kw):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c[apps[i-1]][apps[i]] += 1
        for s, d in c.items():
            t = sum(d.values())
            self._g[s] = dict(sorted({k: v/t for k,v in d.items()}.items(), key=lambda x:-x[1]))
    def update(self, app, hit=False):
        for k in self._rec: self._rec[k] *= 0.95
        self._rec[app] = 1.0; self._freq[app] += 1; self._total += 1
    def predict(self, cur, prev=None, tb=0, wd=0):
        if cur not in self._g: return []
        tot = self._total or 1.0
        cands = {}
        for app, p in self._g[cur].items():
            conf = 0.5*p + 0.3*self._rec.get(app,0) + 0.2*(self._freq.get(app,0)/tot)
            if conf >= 0.05: cands[app] = conf
        return sorted(cands, key=lambda a: -cands[a])[:HOT_SIZE]
    def reset(self):
        self._rec.clear(); self._freq.clear(); self._total = 0.0


class GraphMindRLPolicy(Policy):
    """V3 cache-allocator RL (kept for comparison)."""
    name = "GraphMindRL"
    def __init__(self):
        self._g:    Dict[str, Dict[str, float]] = {}
        self._rec:  Dict[str, float] = defaultdict(float)
        self._freq: Dict[str, float] = defaultdict(float)
        self._total = 0.0
        self._hit_hist: deque = deque(maxlen=20)
        self._budget = HOT_SIZE
        self._thresh = 0.05
    def train(self, apps, **kw):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c[apps[i-1]][apps[i]] += 1
        for s, d in c.items():
            t = sum(d.values())
            self._g[s] = dict(sorted({k: v/t for k,v in d.items()}.items(), key=lambda x:-x[1]))
    def update(self, app, hit=False):
        for k in self._rec: self._rec[k] *= 0.95
        self._rec[app] = 1.0; self._freq[app] += 1; self._total += 1
        self._hit_hist.append(1.0 if hit else 0.0)
        if len(self._hit_hist) == 20:
            hr = sum(self._hit_hist)/20
            if hr < 0.3:   self._budget = min(HOT_SIZE+2, 8); self._thresh = 0.03
            elif hr > 0.7: self._budget = max(HOT_SIZE-1, 3); self._thresh = 0.08
            else:          self._budget = HOT_SIZE;            self._thresh = 0.05
    def predict(self, cur, prev=None, tb=0, wd=0):
        if cur not in self._g: return []
        tot = self._total or 1.0
        cands = {}
        for app, p in self._g[cur].items():
            conf = 0.5*p + 0.3*self._rec.get(app,0) + 0.2*(self._freq.get(app,0)/tot)
            if conf >= self._thresh: cands[app] = conf
        return sorted(cands, key=lambda a: -cands[a])[:self._budget]
    def reset(self):
        self._rec.clear(); self._freq.clear(); self._total = 0.0
        self._hit_hist.clear(); self._budget = HOT_SIZE; self._thresh = 0.05


class RLAdaptiveEnsemblePolicy(Policy):
    """V4 REINFORCE ensemble controller."""
    name = "RLAdaptiveEnsemble"
    def __init__(self):
        self._agent = AdaptiveEnsembleController(lr=0.05, entropy_penalty=0.02)
        self._vom = VariableOrderMarkov(laplace_alpha=0.5, top_k=HOT_SIZE)
        self._ctx = ContextMarkov(top_k=HOT_SIZE, laplace_alpha=0.3)
        self._prev: Optional[str] = None

    def train(self, apps, tbs=None, wds=None, val_apps=None, val_tbs=None, val_wds=None, **kw):
        tbs = tbs or [0]*len(apps)
        wds = wds or [0]*len(apps)
        self._vom.train(apps)
        self._ctx.train(apps, tbs, wds)
        if val_apps and len(val_apps) > 5:
            self._ctx.fit_weights(
                val_apps,
                val_tbs or [0]*len(val_apps),
                val_wds or [0]*len(val_apps),
            )
        self._agent.set_vom_model(self._vom)
        self._agent.set_ctx_model(self._ctx)
        self._agent.train(apps, tbs, wds, n_passes=3)

    def predict(self, cur, prev=None, tb=0, wd=0):
        return self._agent.predict(cur, tb, wd)

    def update(self, app, hit=False):
        self._agent.update_state(app, hit)
        self._prev = app

    def reset(self):
        self._agent.reset()
        self._prev = None


# ── Evaluation Engine ──────────────────────────────────────────────────────

def evaluate_policy(
    policy: Policy,
    train_apps: List[str],
    val_apps: List[str],
    test_apps: List[str],
    train_tbs: List[int],
    val_tbs:   List[int],
    test_tbs:  List[int],
    train_wds: List[int],
    val_wds:   List[int],
    test_wds:  List[int],
    lat: MeasuredLatencyModel,
    user_id: str = "unknown",
) -> dict:
    """Evaluate a single policy on one user's test split."""
    # Train — pass extra context for policies that use it
    if isinstance(policy, ClusterMarkovPolicy):
        policy.train(train_apps, user_id=user_id, tbs=train_tbs, wds=train_wds,
                     val_apps=val_apps, val_tbs=val_tbs, val_wds=val_wds)
    else:
        policy.train(train_apps, tbs=train_tbs, wds=train_wds,
                     val_apps=val_apps, val_tbs=val_tbs, val_wds=val_wds)
    policy.reset()

    cache = Cache()
    for app in train_apps[-20:]:  # warm up cache
        cache.access(app)

    hits = misses = tp = fp = fn = 0
    lat_saved = 0.0
    prev: Optional[str] = None

    for i, cur in enumerate(test_apps):
        tb = test_tbs[i] if test_tbs else 0
        wd = test_wds[i] if test_wds else 0
        preds = policy.predict(cur, prev, tb, wd)
        if preds:
            cache.prefetch(preds)

        tier = cache.lookup(cur)
        is_hit = tier in ("hot", "warm")

        if is_hit:
            hits += 1; tp += 1
            lat_saved += lat.saved(cur, tier)
        else:
            misses += 1

        if i + 1 < len(test_apps):
            nxt = test_apps[i+1]
            if preds:
                if nxt in preds: tp += 1
                else: fn += 1; fp += len(preds)
            else:
                fn += 1

        cache.access(cur)
        policy.update(cur, hit=is_hit)
        prev = cur

    total = hits + misses or 1
    hr = hits / total
    pr = tp / (tp+fp) if (tp+fp) > 0 else 0.0
    re = tp / (tp+fn) if (tp+fn) > 0 else 0.0
    f1 = 2*pr*re/(pr+re) if (pr+re) > 0 else 0.0
    avg_cold = 2763.0
    lat_pct  = (lat_saved / total / avg_cold * 100) if total > 0 else 0.0

    return {
        "hit_rate":          round(hr, 4),
        "precision":         round(pr, 4),
        "recall":            round(re, 4),
        "f1":                round(f1, 4),
        "latency_saved_ms":  round(lat_saved / total, 2),
        "latency_saved_pct": round(lat_pct, 2),
    }


def bootstrap_ci(vals: List[float], n: int = 1000) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    arr = np.array(vals)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n)]
    return (round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4))


def pct(x, p):
    return round(float(np.percentile(x, p)), 4)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load usable users
    with open(os.path.join(PROCESSED_DIR, "users.json"), encoding="utf-8") as f:
        usable_users = [u["user_id"] for u in json.load(f)["users"]]
    logger.info(f"Benchmark V4: {len(usable_users)} users")

    lat = MeasuredLatencyModel(LATENCY_CSV)

    # Load GlobalMarkov2
    gm2_path = os.path.join(MARKOV_DIR, "global_markov2.pkl")
    global_data = None
    if os.path.exists(gm2_path):
        with open(gm2_path, "rb") as f:
            global_data = pickle.load(f)
        logger.info(f"GlobalMarkov2: {len(global_data['markov2']):,} bigram states")

    # Pre-load all training sequences for ClusterMarkov
    logger.info("Pre-loading train sequences for ClusterMarkov...")
    all_train_seqs: Dict[str, List[str]] = {}
    user_data_cache: Dict[str, tuple] = {}

    for uid in usable_users:
        apps, tbs, wds = load_events_with_context(uid)
        if len(apps) < 200: continue
        n = len(apps)
        te = int(n * TRAIN_RATIO)
        all_train_seqs[uid] = apps[:te]
        user_data_cache[uid] = (apps, tbs, wds)

    logger.info(f"Fitting ClusterMarkov on {len(all_train_seqs)} users...")
    ClusterMarkovPolicy.fit_clusters(all_train_seqs)
    logger.info(f"Cluster sizes: {ClusterMarkovPolicy._cluster_model.get_cluster_sizes()}")

    POLICY_NAMES = [
        "Random", "LRU", "LFU", "Frequency", "RecencyFrequency",
        "Markov-1", "Markov-2", "VariableOrderMarkov", "ContextMarkov",
        "ClusterMarkov", "GlobalMarkov2",
        "GraphOnly", "Graph+Confidence", "GraphMindRL", "RLAdaptiveEnsemble",
    ]

    def make_policies():
        return [
            RandomPolicy(),
            LRUPolicy(),
            LFUPolicy(),
            FrequencyPolicy(),
            RecencyFrequencyPolicy(),
            Markov1Policy(),
            Markov2Policy(),
            VOMPolicy(),
            ContextMarkovPolicy(),
            ClusterMarkovPolicy(),
            GlobalMarkov2Policy(global_data),
            GraphOnlyPolicy(),
            GraphConfidencePolicy(),
            GraphMindRLPolicy(),
            RLAdaptiveEnsemblePolicy(),
        ]

    all_rows = []
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for uid in usable_users:
        if uid not in user_data_cache: continue
        apps, tbs, wds = user_data_cache[uid]
        n = len(apps)
        te = int(n * TRAIN_RATIO)
        ve = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_apps = apps[:te]; val_apps = apps[te:ve]; test_apps = apps[ve:]
        train_tbs  = tbs[:te];  val_tbs  = tbs[te:ve];  test_tbs  = tbs[ve:]
        train_wds  = wds[:te];  val_wds  = wds[te:ve];  test_wds  = wds[ve:]

        if len(test_apps) < 10: continue

        logger.info(f"User {uid} ({len(apps)} events, test={len(test_apps)})...")
        for policy in make_policies():
            try:
                metrics = evaluate_policy(
                    policy,
                    train_apps, val_apps, test_apps,
                    train_tbs, val_tbs, test_tbs,
                    train_wds, val_wds, test_wds,
                    lat, uid,
                )
                row = {"user_id": uid, "policy": policy.name}
                row.update(metrics)
                all_rows.append(row)
                for m, v in metrics.items():
                    agg[policy.name][m].append(v)
                logger.info(
                    f"  {policy.name:22s}: HR={metrics['hit_rate']:.3f} "
                    f"F1={metrics['f1']:.3f} Lat={metrics['latency_saved_ms']:.0f}ms"
                )
            except Exception as exc:
                logger.error(f"  {uid}/{policy.name}: {exc}", exc_info=True)

    # Write CSVs
    v4_path = os.path.join(RESULTS_DIR, "benchmark_results_v4.csv")
    if all_rows:
        with open(v4_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
    logger.info(f"Written: {v4_path} ({len(all_rows)} rows)")

    ul_path = os.path.join(RESULTS_DIR, "user_level_results_v4.csv")
    ul_cols = ["user_id","policy","hit_rate","precision","recall","f1",
               "latency_saved_ms","latency_saved_pct"]
    with open(ul_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ul_cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in ul_cols})

    # Advanced metrics
    adv_rows = []
    for pol in POLICY_NAMES:
        if pol not in agg: continue
        row = {"policy": pol}
        for m in METRICS:
            vals = agg[pol][m]
            if not vals: continue
            ci = bootstrap_ci(vals)
            row.update({
                f"{m}_mean":    round(float(np.mean(vals)), 4),
                f"{m}_median":  round(float(np.median(vals)), 4),
                f"{m}_std":     round(float(np.std(vals)), 4),
                f"{m}_p50":     pct(vals, 50),
                f"{m}_p90":     pct(vals, 90),
                f"{m}_p95":     pct(vals, 95),
                f"{m}_p99":     pct(vals, 99),
                f"{m}_ci95_lo": ci[0],
                f"{m}_ci95_hi": ci[1],
            })
        adv_rows.append(row)

    adv_path = os.path.join(RESULTS_DIR, "advanced_metrics_v4.csv")
    if adv_rows:
        with open(adv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(adv_rows[0].keys()))
            w.writeheader(); w.writerows(adv_rows)

    # Markdown report
    md_path = os.path.join(REPORTS_DIR, "benchmark_v4_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# GraphMind V4 — Full Benchmark Report\n\n")
        f.write(f"**Users:** {len(usable_users)} usable UbiqLog users  \n")
        f.write(f"**Policies:** {len(POLICY_NAMES)}  \n")
        f.write(f"**Split:** 80% train / 10% val / 10% test (chronological)  \n\n")
        f.write("---\n\n")
        f.write("## F1 Score Ranking (mean ± std)\n\n")
        f.write("| Rank | Policy | F1 Mean | F1 Median | Std | P95 | 95% CI |\n")
        f.write("|------|--------|---------|-----------|-----|-----|--------|\n")
        sorted_rows = sorted(adv_rows, key=lambda r: -r.get("f1_mean", 0))
        for rank, row in enumerate(sorted_rows, 1):
            ci = f"[{row.get('f1_ci95_lo',0):.4f}, {row.get('f1_ci95_hi',0):.4f}]"
            f.write(
                f"| {rank} | **{row['policy']}** "
                f"| {row.get('f1_mean',0):.4f} "
                f"| {row.get('f1_median',0):.4f} "
                f"| {row.get('f1_std',0):.4f} "
                f"| {row.get('f1_p95',0):.4f} "
                f"| {ci} |\n"
            )
        f.write("\n## Hit Rate Ranking\n\n")
        f.write("| Rank | Policy | Mean | Median | Std | P95 | 95% CI |\n")
        f.write("|------|--------|------|--------|-----|-----|--------|\n")
        for rank, row in enumerate(sorted(adv_rows, key=lambda r: -r.get("hit_rate_mean",0)), 1):
            ci = f"[{row.get('hit_rate_ci95_lo',0):.4f}, {row.get('hit_rate_ci95_hi',0):.4f}]"
            f.write(
                f"| {rank} | **{row['policy']}** "
                f"| {row.get('hit_rate_mean',0):.4f} "
                f"| {row.get('hit_rate_median',0):.4f} "
                f"| {row.get('hit_rate_std',0):.4f} "
                f"| {row.get('hit_rate_p95',0):.4f} "
                f"| {ci} |\n"
            )
        f.write("\n## Latency Saved (ms)\n\n")
        f.write("| Rank | Policy | Mean | Median | Std | P95 |\n")
        f.write("|------|--------|------|--------|-----|-----|\n")
        for rank, row in enumerate(sorted(adv_rows, key=lambda r: -r.get("latency_saved_ms_mean",0)), 1):
            f.write(
                f"| {rank} | **{row['policy']}** "
                f"| {row.get('latency_saved_ms_mean',0):.1f} "
                f"| {row.get('latency_saved_ms_median',0):.1f} "
                f"| {row.get('latency_saved_ms_std',0):.1f} "
                f"| {row.get('latency_saved_ms_p95',0):.1f} |\n"
            )

    logger.info(f"Written: {md_path}")
    logger.info("\n=== V4 BENCHMARK SUMMARY (by F1) ===")
    logger.info(f"{'Policy':25s} {'HitRate':>8} {'F1':>8} {'LatSaved':>10}")
    logger.info("-"*55)
    for row in sorted(adv_rows, key=lambda r: -r.get("f1_mean", 0)):
        logger.info(
            f"{row['policy']:25s} "
            f"{row.get('hit_rate_mean',0):8.3f} "
            f"{row.get('f1_mean',0):8.3f} "
            f"{row.get('latency_saved_ms_mean',0):10.1f}"
        )


if __name__ == "__main__":
    main()
