#!/usr/bin/env python3
"""
scripts/run_v5_validation.py

GraphMind V5 Validation Study — Phases 3–8.

Isolated experimental benchmark.
DO NOT MODIFY production GraphMind.

Experimental policies (26 total):
  Phase 3 — Time-Aware M1:
    TimeAwareM1_6Band   — P(next|app, 6 coarse buckets)
    TimeAwareM1_12Band  — P(next|app, 12 bands)
    TimeAwareM1_24Hour  — P(next|app, hour)
    TimeAwareM1_48Bucket — P(next|app, 30-min bucket)

  Phase 4 — Order Analysis:
    M2_Naive             — raw P(C|A,B), fallback=M1
    M2_Laplace           — P(C|A,B) with Laplace α (tuned)
    M2_Backoff           — P(C|A,B) if count>=3 else P(C|B)
    M2_JM                — Jelinek-Mercer interpolation
    M2_JM_K3             — JM with K=3
    M2_JM_K10            — JM with K=10

  Phase 5 — Combined Context:
    JM_6Band             — JM-M2 + time_6band fallback
    JM_12Band            — JM-M2 + time_12band
    JM_24Hour            — JM-M2 + time_24hour
    JM_48Bucket          — JM-M2 + time_48bucket

  Phase 6 — Graph Representation:
    Graph_NodeApp        — Node=app (same as Markov-1)
    Graph_NodeAppTime6   — Node=(app,time_6band) [graph topology]
    Graph_NodeAppTime12  — Node=(app,time_12band)
    Graph_Bigram         — Node=(prev_app,app) [bigram graph]

  Phase 7 — RL Reward Variants:
    RL_Threshold         — adaptive threshold controller (no PPO)
    RL_F1Reward          — F1-proxy reward
    RL_PrecisionFocus    — high precision threshold (conservative)
    RL_RecallFocus       — low precision threshold (aggressive)

  Phase 8 — Temporal Decay:
    Decay_7d             — half-life 7 days
    Decay_14d            — half-life 14 days
    Decay_30d            — half-life 30 days
    Decay_60d            — half-life 60 days (near-baseline)

Outputs:
  results/v5_time_context.csv
  results/v5_order_analysis.csv
  results/v5_combined_context.csv
  results/v5_graph_study.csv
  results/v5_rl_ablation.csv
  results/v5_temporal_decay.csv
  results/v5_all_experiments.csv
"""

import csv
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
UBIQLOG_ROOT  = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog", "UbiqLog4UCI")
LATENCY_CSV   = os.path.join(PROJECT_ROOT, "datasets", "app_launch_latency.csv")

HOT_SIZE   = 5
WARM_SIZE  = 15
TRAIN_RATIO, VAL_RATIO = 0.80, 0.10
MIN_YEAR, MAX_YEAR = 2011, 2016

SYSTEM_PREFIXES = (
    "com.android.", "com.google.android.providers",
    "com.google.android.gms", "com.google.android.gsf",
    "com.sec.android.provider", "com.samsung.android.provider",
    "com.redbend.", "android.",
)
SYSTEM_SUFFIXES = (":engine",":client",":daemon",":service",":pushservice",":sync")


# ── Latency model (same as V4) ─────────────────────────────────────────────

class MeasuredLatencyModel:
    _DEFAULT_COLD = 2763.0
    _DEFAULT_WARM = 1301.0
    _DEFAULT_HOT  =  274.0
    def __init__(self, path):
        self._cold, self._warm, self._hot, self._pkg = {}, {}, {}, {}
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
    def _k(self, pkg):
        if pkg in self._cold: return pkg
        return self._pkg.get(pkg)
    def saved(self, pkg, tier):
        k = self._k(pkg)
        cold = self._cold.get(k, self._DEFAULT_COLD) if k else self._DEFAULT_COLD
        if tier == "hot":
            hot = self._hot.get(k, self._DEFAULT_HOT) if k else self._DEFAULT_HOT
            return max(0.0, cold - hot)
        if tier == "warm":
            warm = self._warm.get(k, self._DEFAULT_WARM) if k else self._DEFAULT_WARM
            return max(0.0, cold - warm)
        return 0.0


# ── Cache simulator (same as V4) ───────────────────────────────────────────

class Cache:
    def __init__(self):
        self._hot: List[str] = []
        self._warm: List[str] = []
    def lookup(self, app):
        if app in self._hot:  return "hot"
        if app in self._warm: return "warm"
        return "miss"
    def access(self, app):
        if app in self._hot:   self._hot.remove(app)
        elif app in self._warm: self._warm.remove(app)
        self._hot.insert(0, app)
        while len(self._hot) > HOT_SIZE:
            self._warm.insert(0, self._hot.pop())
        while len(self._warm) > WARM_SIZE:
            self._warm.pop()
    def prefetch(self, apps):
        for a in apps:
            if a not in self._hot and a not in self._warm:
                self._warm.insert(0, a)
                while len(self._warm) > WARM_SIZE:
                    self._warm.pop()
    def reset(self):
        self._hot = []; self._warm = []


# ── Data loader (same as V4) ────────────────────────────────────────────────

def _is_system(p):
    p = p.lower()
    for pfx in SYSTEM_PREFIXES:
        if p.startswith(pfx): return True
    for sfx in SYSTEM_SUFFIXES:
        if p.endswith(sfx): return True
    return False

def _parse_ts(s):
    try:
        dt = datetime.strptime(s.strip(), "%m-%d-%Y %H:%M:%S")
        return dt if MIN_YEAR <= dt.year <= MAX_YEAR else None
    except Exception:
        return None

def load_user_data(user_id):
    """Load (apps, tbs, wds, timestamps) for a user."""
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
                        tb = dt.hour * 2 + (1 if dt.minute >= 30 else 0)
                        wd = dt.weekday()
                        raw.append((dt, pkg, tb, wd))
                    except Exception:
                        pass
        except Exception:
            pass
    raw.sort(key=lambda x: x[0])
    apps = [r[1] for r in raw]
    tbs  = [r[2] for r in raw]
    wds  = [r[3] for r in raw]
    dts  = [r[0] for r in raw]
    return apps, tbs, wds, dts


# ── Evaluation engine (same logic as V4) ───────────────────────────────────

def evaluate_policy(policy, train_apps, val_apps, test_apps,
                    train_tbs, val_tbs, test_tbs,
                    train_wds, val_wds, test_wds,
                    lat, user_id="x",
                    train_dts=None, test_dts=None):
    try:
        policy.train(train_apps, tbs=train_tbs, wds=train_wds,
                     val_apps=val_apps, val_tbs=val_tbs, val_wds=val_wds,
                     user_id=user_id, train_dts=train_dts)
    except TypeError:
        policy.train(train_apps, tbs=train_tbs, wds=train_wds,
                     val_apps=val_apps, val_tbs=val_tbs, val_wds=val_wds)
    policy.reset()

    cache = Cache()
    for app in train_apps[-20:]:
        cache.access(app)

    hits = misses = tp = fp = fn = 0
    lat_saved = 0.0
    prev = None

    for i, cur in enumerate(test_apps):
        tb = test_tbs[i] if test_tbs else 0
        wd = test_wds[i] if test_wds else 0
        preds = policy.predict(cur, prev=prev, tb=tb, wd=wd)
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


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3 — Time-Aware M1 Policies
# ══════════════════════════════════════════════════════════════════════════

class _BasePolicy:
    name = "Base"
    def train(self, apps, tbs=None, wds=None, val_apps=None, val_tbs=None,
              val_wds=None, user_id="x", train_dts=None): pass
    def predict(self, cur, prev=None, tb=0, wd=0): return []
    def update(self, app, hit=False): pass
    def reset(self): pass


class TimeAwareM1(_BasePolicy):
    """P(next | app, coarse_bucket) with M1 fallback."""
    def __init__(self, n_bands: int, name: str):
        self.name = name
        self.n_bands = n_bands          # number of time bands
        self._ctx: Dict = {}            # (app, band) → {next: prob}
        self._m1:  Dict = {}            # app → {next: prob}

    def _band(self, tb: int) -> int:
        """Map time_bucket (0-47) to coarse band index."""
        return (tb * self.n_bands) // 48

    def train(self, apps, tbs=None, wds=None, **kw):
        tbs = tbs or [0]*len(apps)
        # Build time-conditioned counts
        cc = defaultdict(lambda: defaultdict(int))
        c1 = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c1[apps[i-1]][apps[i]] += 1
            band = self._band(tbs[i-1])
            cc[(apps[i-1], band)][apps[i]] += 1
        # Normalize M1
        self._m1 = {s: dict(sorted({k: v/sum(d.values()) for k,v in d.items()}.items(), key=lambda x:-x[1]))
                    for s, d in c1.items()}
        # Normalize context-M1
        self._ctx = {k: dict(sorted({a: v/sum(d.values()) for a,v in d.items()}.items(), key=lambda x:-x[1]))
                     for k, d in cc.items()}

    def predict(self, cur, prev=None, tb=0, wd=0):
        band = self._band(tb)
        key  = (cur, band)
        if key in self._ctx and sum(self._ctx[key].values()) > 0:
            return list(self._ctx[key].keys())[:HOT_SIZE]
        return list(self._m1.get(cur, {}).keys())[:HOT_SIZE]


def make_time_aware_policies():
    return [
        TimeAwareM1(6,  "TimeAwareM1_6Band"),
        TimeAwareM1(12, "TimeAwareM1_12Band"),
        TimeAwareM1(24, "TimeAwareM1_24Hour"),
        TimeAwareM1(48, "TimeAwareM1_48Bucket"),
    ]


# ══════════════════════════════════════════════════════════════════════════
# PHASE 4 — Order Analysis Policies
# ══════════════════════════════════════════════════════════════════════════

class M2Naive(_BasePolicy):
    """Naive M2: P(C|A,B) raw, M1 fallback. No smoothing."""
    name = "M2_Naive"
    def __init__(self):
        self._m1, self._m2 = {}, {}
    def train(self, apps, **kw):
        c1 = defaultdict(lambda: defaultdict(int))
        c2 = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c1[apps[i-1]][apps[i]] += 1
        for i in range(2, len(apps)):
            c2[(apps[i-2],apps[i-1])][apps[i]] += 1
        self._m1 = {s: {k: v/sum(d.values()) for k,v in d.items()} for s,d in c1.items()}
        self._m2 = {bg: {k: v/sum(d.values()) for k,v in d.items()} for bg,d in c2.items()}
    def predict(self, cur, prev=None, tb=0, wd=0):
        if prev and (prev,cur) in self._m2:
            return sorted(self._m2[(prev,cur)], key=lambda x: -self._m2[(prev,cur)][x])[:HOT_SIZE]
        return sorted(self._m1.get(cur,{}), key=lambda x: -self._m1.get(cur,{}).get(x,0))[:HOT_SIZE]


class M2Laplace(_BasePolicy):
    """M2 with Laplace smoothing α."""
    def __init__(self, alpha: float = 0.1, name: str = "M2_Laplace"):
        self.name = name
        self.alpha = alpha
        self._m1, self._m2, self._vocab = {}, {}, set()
    def train(self, apps, **kw):
        self._vocab = set(apps)
        V = max(len(self._vocab), 1)
        c1 = defaultdict(lambda: defaultdict(float))
        c2 = defaultdict(lambda: defaultdict(float))
        for i in range(1, len(apps)):
            c1[apps[i-1]][apps[i]] += 1
        for i in range(2, len(apps)):
            c2[(apps[i-2],apps[i-1])][apps[i]] += 1
        for s, d in c1.items():
            total = sum(d.values()) + self.alpha * V
            self._m1[s] = {k: (v + self.alpha)/total for k,v in d.items()}
        for bg, d in c2.items():
            total = sum(d.values()) + self.alpha * V
            self._m2[bg] = {k: (v + self.alpha)/total for k,v in d.items()}
    def predict(self, cur, prev=None, tb=0, wd=0):
        if prev and (prev,cur) in self._m2:
            return sorted(self._m2[(prev,cur)], key=lambda x: -self._m2[(prev,cur)][x])[:HOT_SIZE]
        return sorted(self._m1.get(cur,{}), key=lambda x: -self._m1.get(cur,{}).get(x,0))[:HOT_SIZE]


class M2Backoff(_BasePolicy):
    """M2 with count-threshold backoff: use M2 only if bigram count >= threshold."""
    def __init__(self, min_count: int = 3, name: str = "M2_Backoff"):
        self.name = name
        self.min_count = min_count
        self._m1, self._m2, self._m2c = {}, {}, {}
    def train(self, apps, **kw):
        c1 = defaultdict(lambda: defaultdict(int))
        c2 = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c1[apps[i-1]][apps[i]] += 1
        for i in range(2, len(apps)):
            c2[(apps[i-2],apps[i-1])][apps[i]] += 1
        self._m1 = {s: {k: v/sum(d.values()) for k,v in d.items()} for s,d in c1.items()}
        for bg, d in c2.items():
            total = sum(d.values())
            if total >= self.min_count:
                self._m2[bg]  = {k: v/total for k,v in d.items()}
                self._m2c[bg] = total
    def predict(self, cur, prev=None, tb=0, wd=0):
        if prev and (prev,cur) in self._m2:
            return sorted(self._m2[(prev,cur)], key=lambda x: -self._m2[(prev,cur)][x])[:HOT_SIZE]
        return sorted(self._m1.get(cur,{}), key=lambda x: -self._m1.get(cur,{}).get(x,0))[:HOT_SIZE]


class M2JM(_BasePolicy):
    """Jelinek-Mercer interpolated M2.
    P_JM(C|A,B) = λ₂(A,B) × P(C|A,B) + λ₁(B) × P(C|B) + λ₀ × P(C)
    λ₂ = n(A,B) / (n(A,B) + K)
    λ₁ = (1-λ₂) × n(B) / (n(B) + K)
    λ₀ = 1 - λ₂ - λ₁
    """
    def __init__(self, K: float = 5.0, name: str = "M2_JM"):
        self.name = name
        self.K = K
        self._m1, self._m2 = {}, {}
        self._cnt_bigram: Dict = {}    # count(A,B)
        self._cnt_unigram: Dict = {}   # count(B)
        self._global_freq: Dict = {}
    def train(self, apps, **kw):
        c1  = defaultdict(lambda: defaultdict(int))
        c2  = defaultdict(lambda: defaultdict(int))
        cb  = defaultdict(int)
        cu  = defaultdict(int)
        cg  = defaultdict(int)
        for i in range(1, len(apps)):
            c1[apps[i-1]][apps[i]] += 1
            cu[apps[i-1]] += 1
        for i in range(2, len(apps)):
            c2[(apps[i-2],apps[i-1])][apps[i]] += 1
            cb[(apps[i-2],apps[i-1])] += 1
        for app in apps:
            cg[app] += 1
        total = max(sum(cg.values()), 1)
        self._m1 = {s: {k: v/sum(d.values()) for k,v in d.items()} for s,d in c1.items()}
        self._m2 = {bg: {k: v/sum(d.values()) for k,v in d.items()} for bg,d in c2.items()}
        self._cnt_bigram  = dict(cb)
        self._cnt_unigram = dict(cu)
        self._global_freq = {k: v/total for k,v in cg.items()}

    def predict(self, cur, prev=None, tb=0, wd=0):
        K  = self.K
        n2 = self._cnt_bigram.get((prev, cur), 0) if prev else 0
        n1 = self._cnt_unigram.get(cur, 0)
        lam2 = n2 / (n2 + K) if n2 > 0 else 0.0
        lam1 = (1 - lam2) * n1 / (n1 + K) if n1 > 0 else 0.0
        lam0 = 1.0 - lam2 - lam1

        scores: Dict[str, float] = defaultdict(float)
        if lam2 > 0 and prev and (prev, cur) in self._m2:
            for app, p in self._m2[(prev, cur)].items():
                scores[app] += lam2 * p
        if lam1 > 0 and cur in self._m1:
            for app, p in self._m1[cur].items():
                scores[app] += lam1 * p
        if lam0 > 0:
            for app, p in self._global_freq.items():
                scores[app] += lam0 * p

        return sorted(scores, key=lambda a: -scores[a])[:HOT_SIZE]


def make_order_policies():
    return [
        M2Naive(),
        M2Laplace(alpha=0.01, name="M2_Laplace_001"),
        M2Laplace(alpha=0.10, name="M2_Laplace_010"),
        M2Laplace(alpha=0.50, name="M2_Laplace_050"),
        M2Backoff(min_count=3,  name="M2_Backoff_3"),
        M2Backoff(min_count=5,  name="M2_Backoff_5"),
        M2Backoff(min_count=10, name="M2_Backoff_10"),
        M2JM(K=3,  name="M2_JM_K3"),
        M2JM(K=5,  name="M2_JM_K5"),
        M2JM(K=10, name="M2_JM_K10"),
    ]


# ══════════════════════════════════════════════════════════════════════════
# PHASE 5 — Combined Context Policies
# ══════════════════════════════════════════════════════════════════════════

class JM_TimeAware(_BasePolicy):
    """JM-M2 + time-conditioned M1 fallback."""
    def __init__(self, n_bands: int, K: float = 5.0, name: str = "JM_6Band"):
        self.name = name
        self.n_bands = n_bands
        self.K = K
        self._m2jm  = M2JM(K=K)
        self._ctx_m1 = TimeAwareM1(n_bands, f"_ctx_{n_bands}")

    def _band(self, tb):
        return (tb * self.n_bands) // 48

    def train(self, apps, tbs=None, wds=None, **kw):
        tbs = tbs or [0]*len(apps)
        self._m2jm.train(apps, tbs=tbs)
        self._ctx_m1.train(apps, tbs=tbs)

    def predict(self, cur, prev=None, tb=0, wd=0):
        # Try JM-M2 first
        preds = self._m2jm.predict(cur, prev=prev, tb=tb, wd=wd)
        if preds:
            return preds
        # Fall back to time-aware M1
        return self._ctx_m1.predict(cur, prev=prev, tb=tb, wd=wd)

    def update(self, app, hit=False): pass
    def reset(self): pass


def make_combined_policies():
    return [
        JM_TimeAware(6,  K=5, name="JM_6Band"),
        JM_TimeAware(12, K=5, name="JM_12Band"),
        JM_TimeAware(24, K=5, name="JM_24Hour"),
        JM_TimeAware(48, K=5, name="JM_48Bucket"),
    ]


# ══════════════════════════════════════════════════════════════════════════
# PHASE 6 — Graph Representation Policies
# ══════════════════════════════════════════════════════════════════════════

class GraphNodeApp(_BasePolicy):
    """Graph where node = app (baseline, identical to M1 by construction)."""
    name = "Graph_NodeApp"
    def __init__(self):
        self._g = {}
    def train(self, apps, **kw):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c[apps[i-1]][apps[i]] += 1
        self._g = {s: dict(sorted({k: v/sum(d.values()) for k,v in d.items()}.items(), key=lambda x:-x[1]))
                   for s, d in c.items()}
    def predict(self, cur, prev=None, tb=0, wd=0):
        return list(self._g.get(cur, {}).keys())[:HOT_SIZE]


class GraphNodeAppTime(_BasePolicy):
    """Graph where node = (app, time_band). Edge = (node_src → node_tgt)."""
    def __init__(self, n_bands: int, name: str):
        self.name = name
        self.n_bands = n_bands
        self._g: Dict = {}  # (app, band) → {(next_app, next_band): prob}
        self._fallback: Dict = {}  # app → {next_app: prob}

    def _band(self, tb):
        return (tb * self.n_bands) // 48

    def train(self, apps, tbs=None, **kw):
        tbs = tbs or [0]*len(apps)
        c_ctx = defaultdict(lambda: defaultdict(int))
        c_fb  = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            src = (apps[i-1], self._band(tbs[i-1]))
            tgt = (apps[i],   self._band(tbs[i]))
            c_ctx[src][tgt] += 1
            c_fb[apps[i-1]][apps[i]] += 1
        self._g = {s: dict(sorted({k: v/sum(d.values()) for k,v in d.items()}.items(), key=lambda x:-x[1]))
                   for s, d in c_ctx.items()}
        self._fallback = {s: dict(sorted({k: v/sum(d.values()) for k,v in d.items()}.items(), key=lambda x:-x[1]))
                          for s, d in c_fb.items()}

    def predict(self, cur, prev=None, tb=0, wd=0):
        key = (cur, self._band(tb))
        if key in self._g:
            # Return the target app_id from (app, band) tuples
            targets = list(self._g[key].keys())[:HOT_SIZE]
            return [t[0] for t in targets]  # extract app_id
        return list(self._fallback.get(cur, {}).keys())[:HOT_SIZE]


class GraphBigram(_BasePolicy):
    """Graph where node = (prev_app, app). Captures second-order structure."""
    name = "Graph_Bigram"
    def __init__(self):
        self._g: Dict = {}   # (prev,cur) → {next: prob}
        self._fallback = {}

    def train(self, apps, **kw):
        c2 = defaultdict(lambda: defaultdict(int))
        c1 = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c1[apps[i-1]][apps[i]] += 1
        for i in range(2, len(apps)):
            c2[(apps[i-2],apps[i-1])][apps[i]] += 1
        self._g = {bg: dict(sorted({k: v/sum(d.values()) for k,v in d.items()}.items(), key=lambda x:-x[1]))
                   for bg, d in c2.items()}
        self._fallback = {s: dict(sorted({k: v/sum(d.values()) for k,v in d.items()}.items(), key=lambda x:-x[1]))
                          for s, d in c1.items()}

    def predict(self, cur, prev=None, tb=0, wd=0):
        if prev and (prev, cur) in self._g:
            return list(self._g[(prev, cur)].keys())[:HOT_SIZE]
        return list(self._fallback.get(cur, {}).keys())[:HOT_SIZE]


def make_graph_policies():
    return [
        GraphNodeApp(),
        GraphNodeAppTime(6,  "Graph_NodeAppTime6"),
        GraphNodeAppTime(12, "Graph_NodeAppTime12"),
        GraphBigram(),
    ]


# ══════════════════════════════════════════════════════════════════════════
# PHASE 7 — RL Reward Variant Policies
# ══════════════════════════════════════════════════════════════════════════

class _GraphMindRLBase(_BasePolicy):
    """Base GraphMind policy (replicates V3 GraphMindRL exactly)."""
    name = "RL_Base"
    def __init__(self, init_thresh=0.05, thresh_lo=0.03, thresh_hi=0.08):
        self._g = {}
        self._rec  = defaultdict(float)
        self._freq = defaultdict(float)
        self._total = 0.0
        self._hit_hist: deque = deque(maxlen=20)
        self._budget = HOT_SIZE
        self._thresh = init_thresh
        self._thresh_lo = thresh_lo
        self._thresh_hi = thresh_hi
        # Stats for F1/precision/recall-based variants
        self._tp = self._fp = self._fn = 0
        self._last_preds: List[str] = []

    def train(self, apps, **kw):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c[apps[i-1]][apps[i]] += 1
        self._g = {s: dict(sorted({k: v/sum(d.values()) for k,v in d.items()}.items(), key=lambda x:-x[1]))
                   for s, d in c.items()}

    def predict(self, cur, prev=None, tb=0, wd=0):
        if cur not in self._g: return []
        tot = self._total or 1.0
        cands = {}
        for app, p in self._g[cur].items():
            conf = 0.5*p + 0.3*self._rec.get(app,0) + 0.2*(self._freq.get(app,0)/tot)
            if conf >= self._thresh:
                cands[app] = conf
        self._last_preds = sorted(cands, key=lambda a: -cands[a])[:self._budget]
        return self._last_preds

    def update(self, app, hit=False):
        for k in self._rec: self._rec[k] *= 0.95
        self._rec[app] = 1.0; self._freq[app] += 1; self._total += 1
        self._hit_hist.append(1.0 if hit else 0.0)
        # subclass-specific update
        self._adapt(app, hit)

    def _adapt(self, app, hit):
        pass  # override in subclasses

    def reset(self):
        self._rec.clear(); self._freq.clear(); self._total = 0.0
        self._hit_hist.clear()
        self._budget = HOT_SIZE; self._thresh = 0.05
        self._tp = self._fp = self._fn = 0
        self._last_preds = []


class RL_Threshold(_GraphMindRLBase):
    """Adaptive threshold controller using running hit-rate."""
    name = "RL_Threshold"
    def _adapt(self, app, hit):
        if len(self._hit_hist) == 20:
            hr = sum(self._hit_hist) / 20
            # Threshold adapts to hit rate signal
            if hr < 0.30:
                self._thresh = max(0.02, self._thresh - 0.005)
                self._budget = min(8, self._budget + 1)
            elif hr > 0.65:
                self._thresh = min(0.20, self._thresh + 0.005)
                self._budget = max(3, self._budget - 1)


class RL_PrecisionFocus(_GraphMindRLBase):
    """Conservative: high precision, low recall."""
    name = "RL_PrecisionFocus"
    def __init__(self):
        super().__init__(init_thresh=0.15, thresh_lo=0.10, thresh_hi=0.20)
        self._budget = 3  # fewer predictions → higher precision


class RL_RecallFocus(_GraphMindRLBase):
    """Aggressive: low precision threshold, high recall."""
    name = "RL_RecallFocus"
    def __init__(self):
        super().__init__(init_thresh=0.02, thresh_lo=0.01, thresh_hi=0.05)
        self._budget = WARM_SIZE  # more predictions → higher recall


class RL_F1Reward(_GraphMindRLBase):
    """Threshold adapts to F1 proxy = 2PR/(P+R) over a window."""
    name = "RL_F1Reward"
    def __init__(self):
        super().__init__(init_thresh=0.05)
        self._window_tp = deque(maxlen=50)
        self._window_fp = deque(maxlen=50)
        self._window_fn = deque(maxlen=50)
        self._prev_next: Optional[str] = None

    def update(self, app, hit=False):
        # Update TP/FP/FN for previous prediction
        if self._last_preds and self._prev_next is not None:
            if self._prev_next in self._last_preds:
                self._window_tp.append(1); self._window_fp.append(max(0, len(self._last_preds)-1)); self._window_fn.append(0)
            else:
                self._window_tp.append(0); self._window_fp.append(len(self._last_preds)); self._window_fn.append(1)
        self._prev_next = app
        super().update(app, hit)

    def _adapt(self, app, hit):
        if len(self._window_tp) < 10: return
        tp = sum(self._window_tp); fp = sum(self._window_fp); fn = sum(self._window_fn)
        P = tp / max(tp+fp, 1); R = tp / max(tp+fn, 1)
        f1 = 2*P*R / max(P+R, 1e-9)
        # Adjust threshold to improve F1
        if P > R + 0.15:  # precision too high → reduce threshold
            self._thresh = max(0.02, self._thresh - 0.003)
            self._budget = min(HOT_SIZE+2, self._budget + 1)
        elif R > P + 0.15:  # recall too high → increase threshold
            self._thresh = min(0.20, self._thresh + 0.003)
            self._budget = max(3, self._budget - 1)


class RL_LatencyFocus(_GraphMindRLBase):
    """Threshold adapts to maximize latency saved (prefer HOT hits)."""
    name = "RL_LatencyFocus"
    def __init__(self):
        super().__init__(init_thresh=0.10)  # conservative = HOT is likely
        self._budget = HOT_SIZE  # stick to small prefetch set

    def _adapt(self, app, hit):
        if len(self._hit_hist) == 20:
            hr = sum(self._hit_hist) / 20
            # Stay conservative: better to be right with small set
            if hr < 0.5:
                self._thresh = max(0.05, self._thresh - 0.005)
            elif hr > 0.8:
                self._thresh = min(0.25, self._thresh + 0.005)


def make_rl_policies():
    return [
        RL_Threshold(),
        RL_PrecisionFocus(),
        RL_RecallFocus(),
        RL_F1Reward(),
        RL_LatencyFocus(),
    ]


# ══════════════════════════════════════════════════════════════════════════
# PHASE 8 — Temporal Decay Policies
# ══════════════════════════════════════════════════════════════════════════

class TemporalDecayM1(_BasePolicy):
    """M1 with exponential edge weight decay by recency of transition."""
    def __init__(self, halflife_days: float, name: str):
        self.name = name
        self.halflife_days = halflife_days
        self._edges: Dict = {}  # (from, to) → list of datetimes
        self._m1: Dict = {}    # fallback

    def train(self, apps, tbs=None, wds=None, train_dts=None, **kw):
        if train_dts is None or len(train_dts) < 2:
            # No timestamps: fall back to plain M1
            c = defaultdict(lambda: defaultdict(int))
            for i in range(1, len(apps)):
                c[apps[i-1]][apps[i]] += 1
            self._m1 = {s: {k: v/sum(d.values()) for k,v in d.items()} for s,d in c.items()}
            self._edges = {}
            return

        last_dt = train_dts[-1]
        # Store timestamps for each edge
        edge_ts: Dict = defaultdict(list)
        for i in range(1, len(apps)):
            edge_ts[(apps[i-1], apps[i])].append(train_dts[i])

        # Compute decayed weight: w = sum(0.5 ^ (days_ago / halflife))
        lam = math.log(2) / (self.halflife_days * 86400)  # decay per second

        scored: Dict = defaultdict(lambda: defaultdict(float))
        for (src, dst), timestamps in edge_ts.items():
            weight = sum(math.exp(-lam * (last_dt - ts).total_seconds()) for ts in timestamps)
            scored[src][dst] = weight

        # Normalize
        self._m1 = {}
        for src, dsts in scored.items():
            total = sum(dsts.values())
            self._m1[src] = dict(sorted({k: v/total for k,v in dsts.items()}.items(), key=lambda x:-x[1]))

    def predict(self, cur, prev=None, tb=0, wd=0):
        return list(self._m1.get(cur, {}).keys())[:HOT_SIZE]


def make_decay_policies():
    return [
        TemporalDecayM1(halflife_days=7,  name="Decay_7d"),
        TemporalDecayM1(halflife_days=14, name="Decay_14d"),
        TemporalDecayM1(halflife_days=30, name="Decay_30d"),
        TemporalDecayM1(halflife_days=60, name="Decay_60d"),
    ]


# ══════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK LOOP
# ══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lat = MeasuredLatencyModel(LATENCY_CSV)

    with open(os.path.join(PROCESSED_DIR, "users.json"), encoding="utf-8") as f:
        usable_users = [u["user_id"] for u in json.load(f)["users"]]
    logger.info(f"Users: {len(usable_users)}")

    # Define experiment groups (phase → list of policy factories)
    EXPERIMENTS = {
        "phase3_time":    make_time_aware_policies,
        "phase4_order":   make_order_policies,
        "phase5_combined":make_combined_policies,
        "phase6_graph":   make_graph_policies,
        "phase7_rl":      make_rl_policies,
        "phase8_decay":   make_decay_policies,
    }

    # All results
    all_rows: List[dict] = []
    phase_rows: Dict[str, List[dict]] = {k: [] for k in EXPERIMENTS}

    # Load all user data once
    logger.info("Pre-loading user data...")
    user_cache = {}
    for uid in usable_users:
        try:
            apps, tbs, wds, dts = load_user_data(uid)
            if len(apps) >= 200:
                user_cache[uid] = (apps, tbs, wds, dts)
        except Exception as e:
            logger.warning(f"Skip {uid}: {e}")

    logger.info(f"Loaded {len(user_cache)} users")

    for phase_name, factory in EXPERIMENTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {phase_name}")
        logger.info(f"{'='*60}")

        for uid in usable_users:
            if uid not in user_cache: continue
            apps, tbs, wds, dts = user_cache[uid]
            n = len(apps)
            te = int(n * TRAIN_RATIO)
            ve = int(n * (TRAIN_RATIO + VAL_RATIO))

            train_apps = apps[:te]; val_apps = apps[te:ve]; test_apps = apps[ve:]
            train_tbs  = tbs[:te];  val_tbs  = tbs[te:ve];  test_tbs  = tbs[ve:]
            train_wds  = wds[:te];  val_wds  = wds[te:ve];  test_wds  = wds[ve:]
            train_dts  = dts[:te]

            if len(test_apps) < 10: continue

            # Fresh policies for this user
            policies = factory()

            for policy in policies:
                try:
                    metrics = evaluate_policy(
                        policy,
                        train_apps, val_apps, test_apps,
                        train_tbs,  val_tbs,  test_tbs,
                        train_wds,  val_wds,  test_wds,
                        lat, uid,
                        train_dts=train_dts, test_dts=None,
                    )
                    row = {"phase": phase_name, "user_id": uid, "policy": policy.name}
                    row.update(metrics)
                    all_rows.append(row)
                    phase_rows[phase_name].append(row)
                    logger.info(
                        f"  {uid:8s} {policy.name:25s}: "
                        f"HR={metrics['hit_rate']:.3f} F1={metrics['f1']:.3f} "
                        f"Lat={metrics['latency_saved_ms']:.0f}ms"
                    )
                except Exception as exc:
                    logger.error(f"  {uid}/{policy.name}: {exc}", exc_info=False)

    # Write master CSV
    master_path = os.path.join(RESULTS_DIR, "v5_all_experiments.csv")
    if all_rows:
        with open(master_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
    logger.info(f"\nWritten master: {master_path} ({len(all_rows)} rows)")

    # Write per-phase CSVs
    phase_file_map = {
        "phase3_time":     "v5_time_context.csv",
        "phase4_order":    "v5_order_analysis.csv",
        "phase5_combined": "v5_combined_context.csv",
        "phase6_graph":    "v5_graph_study.csv",
        "phase7_rl":       "v5_rl_ablation.csv",
        "phase8_decay":    "v5_temporal_decay.csv",
    }
    for phase, fname in phase_file_map.items():
        rows = phase_rows.get(phase, [])
        if not rows: continue
        path = os.path.join(RESULTS_DIR, fname)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        logger.info(f"Written: {path} ({len(rows)} rows)")

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("V5 VALIDATION SUMMARY (mean F1 vs baseline GraphMindRL=0.7424)")
    logger.info("="*70)
    logger.info(f"{'Policy':28s} {'Phase':20s} {'ΔF1':>7} {'F1':>7} {'HR':>7}")
    logger.info("-"*70)
    baseline_f1 = 0.7424

    agg_rows = []
    for phase, rows in phase_rows.items():
        if not rows: continue
        by_policy = defaultdict(list)
        for r in rows:
            by_policy[r["policy"]].append(r)
        for pol, prows in sorted(by_policy.items()):
            mean_f1 = float(np.mean([float(r["f1"]) for r in prows]))
            mean_hr = float(np.mean([float(r["hit_rate"]) for r in prows]))
            delta   = mean_f1 - baseline_f1
            agg_rows.append({"phase": phase, "policy": pol, "f1": round(mean_f1,4),
                              "hit_rate": round(mean_hr,4), "delta_f1": round(delta,4),
                              "n": len(prows)})
            marker = "✅" if delta >= 0.02 else ("📈" if delta > 0 else "❌")
            logger.info(f"  {pol:28s} {phase:20s} {delta:+7.4f} {mean_f1:7.4f} {mean_hr:7.4f} {marker}")

    # Write aggregated summary
    agg_path = os.path.join(RESULTS_DIR, "v5_summary.csv")
    if agg_rows:
        with open(agg_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader(); w.writerows(agg_rows)
    logger.info(f"\nSummary: {agg_path}")


if __name__ == "__main__":
    main()
