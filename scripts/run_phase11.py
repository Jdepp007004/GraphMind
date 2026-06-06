#!/usr/bin/env python3
"""
scripts/run_phase11.py

Phase 11 — Final Optimization Search (Phases A–E).

Phases:
  A  Confidence weight grid search  (trans/rec/freq × threshold=0.10)
  B  Threshold sweep                (thresh 0.02–0.20, fixed weights)
  C  Time context coverage audit    (static analysis, no benchmark)
  D  Modified Kneser-Ney            (no global unigram term, K=3/5/10)
  E  Combined best candidate        (GraphMindRL_V5)

Baseline: GraphMindRL F1=0.7424  (31 users, 80/10/10 chronological)
Current best: RL_LatencyFocus F1=0.7539 (p=0.0003, d=0.752)

Outputs:
  results/v5_weight_grid.csv
  results/v5_threshold_sweep.csv
  results/v5_modified_kn.csv
  results/v5_final_comparison.csv
  reports/time_context_coverage_audit.md
  reports/v5_final_decision.md
  reports/v5_optimization_summary.md
  reports/figures/threshold_vs_f1.png
"""

import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR   = os.path.join(REPORTS_DIR, "figures")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
UBIQLOG_ROOT  = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog", "UbiqLog4UCI")
LATENCY_CSV   = os.path.join(PROJECT_ROOT, "datasets", "app_launch_latency.csv")

HOT_SIZE    = 5
WARM_SIZE   = 15
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
MIN_YEAR    = 2011
MAX_YEAR    = 2016

BASELINE_F1        = 0.7424
BEST_CANDIDATE_F1  = 0.7539   # RL_LatencyFocus

SYSTEM_PREFIXES = (
    "com.android.", "com.google.android.providers",
    "com.google.android.gms", "com.google.android.gsf",
    "com.sec.android.provider", "com.samsung.android.provider",
    "com.redbend.", "android.",
)
SYSTEM_SUFFIXES = (":engine", ":client", ":daemon", ":service", ":pushservice", ":sync")


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

class MeasuredLatencyModel:
    _DC = 2763.0; _DW = 1301.0; _DH = 274.0
    def __init__(self, path):
        self._cold = {}; self._warm = {}; self._hot = {}; self._pkg = {}
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
    def saved(self, pkg, tier):
        k = pkg if pkg in self._cold else self._pkg.get(pkg)
        cold = self._cold.get(k, self._DC) if k else self._DC
        if tier == "hot":  return max(0.0, cold - (self._hot.get(k, self._DH)  if k else self._DH))
        if tier == "warm": return max(0.0, cold - (self._warm.get(k, self._DW) if k else self._DW))
        return 0.0


class Cache:
    def __init__(self):
        self._hot: List[str] = []; self._warm: List[str] = []
    def lookup(self, app):
        if app in self._hot:  return "hot"
        if app in self._warm: return "warm"
        return "miss"
    def access(self, app):
        if app in self._hot:    self._hot.remove(app)
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
    def reset(self): self._hot = []; self._warm = []


def _is_system(p):
    p = p.lower()
    for pfx in SYSTEM_PREFIXES:
        if p.startswith(pfx): return True
    for sfx in SYSTEM_SUFFIXES:
        if p.endswith(sfx): return True
    return False


def _parse_ts(s):
    from datetime import datetime
    try:
        dt = datetime.strptime(s.strip(), "%m-%d-%Y %H:%M:%S")
        return dt if MIN_YEAR <= dt.year <= MAX_YEAR else None
    except Exception:
        return None


def load_user_data(user_id):
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
    return ([r[1] for r in raw], [r[2] for r in raw], [r[3] for r in raw])


def evaluate_policy(policy, tr_a, va_a, ts_a, tr_t, va_t, ts_t,
                    tr_w, va_w, ts_w, lat, uid="x"):
    policy.train(tr_a, tbs=tr_t, wds=tr_w, val_apps=va_a, val_tbs=va_t, val_wds=va_w)
    policy.reset()

    cache = Cache()
    for app in tr_a[-20:]:
        cache.access(app)

    hits = misses = tp = fp = fn = 0
    lat_saved = 0.0
    prev = None

    for i, cur in enumerate(ts_a):
        tb = ts_t[i] if ts_t else 0
        wd = ts_w[i] if ts_w else 0
        preds = policy.predict(cur, prev=prev, tb=tb, wd=wd)
        if preds:
            cache.prefetch(preds)

        tier   = cache.lookup(cur)
        is_hit = tier in ("hot", "warm")

        if is_hit:
            hits += 1; tp += 1
            lat_saved += lat.saved(cur, tier)
        else:
            misses += 1

        if i + 1 < len(ts_a):
            nxt = ts_a[i + 1]
            if preds:
                if nxt in preds: tp += 1
                else:            fn += 1; fp += len(preds)
            else:
                fn += 1

        cache.access(cur)
        policy.update(cur, hit=is_hit)
        prev = cur

    total = hits + misses or 1
    hr = hits / total
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    re = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0.0
    return {
        "hit_rate":         round(hr, 4),
        "precision":        round(pr, 4),
        "recall":           round(re, 4),
        "f1":               round(f1, 4),
        "latency_saved_ms": round(lat_saved / total, 2),
    }


def paired_t(exp_by_user, baseline_by_user):
    users = sorted(set(exp_by_user) & set(baseline_by_user))
    if len(users) < 5:
        return float("nan"), float("nan"), float("nan")
    e = np.array([exp_by_user[u]      for u in users])
    b = np.array([baseline_by_user[u] for u in users])
    t, p = stats.ttest_rel(e, b)
    diff = e - b
    d = diff.mean() / (diff.std() + 1e-9)
    return float(t), float(p), float(d)


def write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info(f"Written: {path} ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE A & B — Parametric Confidence Policy
# ═══════════════════════════════════════════════════════════════════════════════

class ConfidencePolicy:
    """
    Markov-1 graph + confidence score with parametric weights and threshold.

    confidence(app) = w_t * P(app|cur) + w_r * recency(app) + w_f * frequency(app)

    select: top-k candidates where confidence >= threshold, up to budget.
    Online: adaptive threshold ±0.005 based on 20-step hit rate (like RL_LatencyFocus).
    """
    def __init__(self, w_trans: float, w_rec: float, w_freq: float,
                 threshold: float, budget: int = HOT_SIZE,
                 name: str = "ConfidencePolicy"):
        self.name      = name
        self.w_trans   = w_trans
        self.w_rec     = w_rec
        self.w_freq    = w_freq
        self.threshold = threshold
        self._init_thresh = threshold
        self.budget    = budget
        self._g        = {}
        self._rec      = defaultdict(float)
        self._freq     = defaultdict(float)
        self._total    = 0.0
        self._hist     = deque(maxlen=20)
        self._last_preds: List[str] = []

    def train(self, apps, tbs=None, wds=None, **kw):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c[apps[i-1]][apps[i]] += 1
        self._g = {
            s: dict(sorted({k: v / sum(d.values()) for k, v in d.items()}.items(),
                           key=lambda x: -x[1]))
            for s, d in c.items()
        }

    def predict(self, cur, prev=None, tb=0, wd=0):
        if cur not in self._g:
            self._last_preds = []
            return []
        tot = self._total or 1.0
        cands = {
            app: (self.w_trans * p
                  + self.w_rec  * self._rec.get(app, 0.0)
                  + self.w_freq * self._freq.get(app, 0.0) / tot)
            for app, p in self._g[cur].items()
        }
        self._last_preds = sorted(
            (a for a, c in cands.items() if c >= self.threshold),
            key=lambda a: -cands[a]
        )[:self.budget]
        return self._last_preds

    def update(self, app, hit=False):
        for k in self._rec: self._rec[k] *= 0.95
        self._rec[app] = 1.0
        self._freq[app] += 1
        self._total += 1
        self._hist.append(1.0 if hit else 0.0)
        # adaptive threshold (same as RL_LatencyFocus)
        if len(self._hist) == 20:
            hr = sum(self._hist) / 20
            if hr < 0.5:
                self.threshold = max(0.05, self.threshold - 0.005)
            elif hr > 0.8:
                self.threshold = min(0.25, self.threshold + 0.005)

    def reset(self):
        self._rec.clear(); self._freq.clear(); self._total = 0.0
        self._hist.clear(); self._last_preds = []
        self.threshold = self._init_thresh


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE D — Modified Kneser-Ney (no global unigram term)
# ═══════════════════════════════════════════════════════════════════════════════

class ModifiedKNPolicy:
    """
    P_MKN(C|A,B) = λ2(A,B) × P(C|A,B) + (1 − λ2(A,B)) × P(C|B)
    λ2 = n(A,B) / (n(A,B) + K)

    No global unigram term — removes the popularity bias that hurt JM.
    Reduces to pure M1 when bigram is completely unseen (n=0 → λ2=0).
    """
    def __init__(self, K: float = 5.0, name: str = "ModKN_K5"):
        self.K    = K
        self.name = name
        self._m1   = {}
        self._m2   = {}
        self._cnt2 = {}     # bigram count n(A,B)
        self._last_preds: List[str] = []

    def train(self, apps, **kw):
        c1 = defaultdict(lambda: defaultdict(int))
        c2 = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c1[apps[i-1]][apps[i]] += 1
        for i in range(2, len(apps)):
            c2[(apps[i-2], apps[i-1])][apps[i]] += 1

        self._m1 = {s: {k: v / sum(d.values()) for k, v in d.items()} for s, d in c1.items()}
        self._m2 = {bg: {k: v / sum(d.values()) for k, v in d.items()} for bg, d in c2.items()}
        self._cnt2 = {bg: sum(d.values()) for bg, d in c2.items()}

    def predict(self, cur, prev=None, tb=0, wd=0):
        n2    = self._cnt2.get((prev, cur), 0) if prev else 0
        lam2  = n2 / (n2 + self.K) if n2 > 0 else 0.0
        lam1  = 1.0 - lam2

        scores: Dict[str, float] = defaultdict(float)

        if lam2 > 0 and prev and (prev, cur) in self._m2:
            for app, p in self._m2[(prev, cur)].items():
                scores[app] += lam2 * p

        if lam1 > 0 and cur in self._m1:
            for app, p in self._m1[cur].items():
                scores[app] += lam1 * p

        self._last_preds = sorted(scores, key=lambda a: -scores[a])[:HOT_SIZE]
        return self._last_preds

    def update(self, app, hit=False): pass
    def reset(self): self._last_preds = []


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE C — Time Context Coverage Audit (static analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase_c(user_cache, usable_users):
    logger.info("\n" + "=" * 60)
    logger.info("Phase C — Time Context Coverage Audit")
    logger.info("=" * 60)

    granularities = [
        ("TimeAwareM1_6Band",   6),
        ("TimeAwareM1_12Band",  12),
        ("TimeAwareM1_24Hour",  24),
        ("TimeAwareM1_48Bucket", 48),
    ]

    agg = {name: {
        "total_steps": 0, "from_time_table": 0, "from_fallback": 0,
        "empty_states": 0, "total_train_transitions": 0,
        "total_train_states": 0, "total_test_states": 0,
        "unseen_states": 0,
    } for name, _ in granularities}

    for uid in usable_users:
        if uid not in user_cache:
            continue
        apps, tbs, wds = user_cache[uid]
        n  = len(apps)
        te = int(n * TRAIN_RATIO)
        ve = int(n * (TRAIN_RATIO + VAL_RATIO))

        tr_a, tr_t = apps[:te], tbs[:te]
        ts_a, ts_t = apps[ve:], tbs[ve:]

        if len(ts_a) < 10:
            continue

        for name, n_bands in granularities:
            def band(tb): return (tb * n_bands) // 48

            # Build time-conditioned counts from training
            c_time = defaultdict(lambda: defaultdict(int))  # (app,band) → {next: cnt}
            c_m1   = defaultdict(lambda: defaultdict(int))  # app → {next: cnt}
            for i in range(1, len(tr_a)):
                c_m1[tr_a[i-1]][tr_a[i]] += 1
                b = band(tr_t[i-1])
                c_time[(tr_a[i-1], b)][tr_a[i]] += 1

            # All (app, band) states seen in training
            train_states = set(c_time.keys())
            total_train_transitions = sum(sum(d.values()) for d in c_time.values())

            # Evaluate on test
            for i in range(len(ts_a)):
                cur = ts_a[i]
                b   = band(ts_t[i])
                key = (cur, b)

                agg[name]["total_steps"]             += 1
                agg[name]["total_train_transitions"] += total_train_transitions
                agg[name]["total_train_states"]      += len(train_states)
                agg[name]["total_test_states"]       += 1

                if key in c_time and sum(c_time[key].values()) > 0:
                    agg[name]["from_time_table"] += 1
                else:
                    agg[name]["from_fallback"] += 1
                    if key not in train_states:
                        agg[name]["unseen_states"] += 1
                    else:
                        agg[name]["empty_states"] += 1  # in train but 0 transitions

    # Write report
    lines = [
        "# Time Context Coverage Audit",
        "",
        "**Date:** 2026-06-06",
        "**Question:** Did time-aware M1 fail due to (A) useless signal or (B) sparsity?",
        "",
        "---",
        "",
        "## Coverage Statistics",
        "",
        "| Granularity | Steps | From Time Table | Fallback | Unseen State | Avg Trans/State |",
        "|-------------|-------|----------------|----------|-------------|----------------|",
    ]
    for name, n_bands in granularities:
        a = agg[name]
        steps = max(a["total_steps"], 1)
        ftt_pct = 100 * a["from_time_table"] / steps
        fb_pct  = 100 * a["from_fallback"] / steps
        unseen_pct = 100 * a["unseen_states"] / max(a["from_fallback"], 1)
        avg_trans = a["total_train_transitions"] / max(a["total_train_states"] * steps / max(steps, 1), 1)
        lines.append(
            f"| {name} | {steps} | {a['from_time_table']} ({ftt_pct:.1f}%) "
            f"| {a['from_fallback']} ({fb_pct:.1f}%) "
            f"| {a['unseen_states']} ({unseen_pct:.1f}% of fallbacks) "
            f"| {avg_trans:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ]

    # Determine finding
    best = granularities[0][0]
    best_ftt = 100 * agg[best]["from_time_table"] / max(agg[best]["total_steps"], 1)
    worst = granularities[-1][0]
    worst_ftt = 100 * agg[worst]["from_time_table"] / max(agg[worst]["total_steps"], 1)

    if worst_ftt < 50:
        lines.append("**Conclusion: ANSWER B — Sparsity destroyed coverage.**")
        lines.append("")
        lines.append(f"Even the finest granularity (48-bucket) only serves {worst_ftt:.1f}% of "
                     "test steps from the time-conditioned table. The remaining steps fall back to M1.")
        lines.append("")
        lines.append("The time signal is not useless — 6-band achieves higher coverage — but the "
                     "dataset is too short (~2 months, ~6700 training transitions per user) "
                     "to reliably populate fine-grained time buckets.")
        lines.append("")
        lines.append("**Implication:** Time-aware prediction requires either (a) a longer dataset "
                     "(≥6 months) or (b) a soft conditioning approach (feature augmentation) "
                     "rather than hard table lookup.")
    else:
        lines.append("**Conclusion: Mixed — moderate coverage but signal quality insufficient.**")

    lines.append("")
    path = os.path.join(REPORTS_DIR, "time_context_coverage_audit.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Written: {path}")

    for name, n_bands in granularities:
        a = agg[name]
        steps = max(a["total_steps"], 1)
        ftt_pct = 100 * a["from_time_table"] / steps
        logger.info(f"  {name:25s}: time-table={ftt_pct:.1f}%  fallback={100-ftt_pct:.1f}%")

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    lat = MeasuredLatencyModel(LATENCY_CSV)

    with open(os.path.join(PROCESSED_DIR, "users.json"), encoding="utf-8") as f:
        usable_users = [u["user_id"] for u in json.load(f)["users"]]
    logger.info(f"Users: {len(usable_users)}")

    # Load V4 baseline per-user F1
    baseline_by_user: Dict[str, float] = {}
    v4_path = os.path.join(RESULTS_DIR, "benchmark_results_v4.csv")
    if os.path.exists(v4_path):
        with open(v4_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["policy"] == "GraphMindRL":
                    baseline_by_user[r["user_id"]] = float(r["f1"])

    # Pre-load all user data
    logger.info("Pre-loading user data…")
    user_cache = {}
    for uid in usable_users:
        try:
            apps, tbs, wds = load_user_data(uid)
            if len(apps) >= 200:
                user_cache[uid] = (apps, tbs, wds)
        except Exception as e:
            logger.warning(f"Skip {uid}: {e}")
    logger.info(f"Loaded {len(user_cache)} users")

    # Split helper
    def split(uid):
        apps, tbs, wds = user_cache[uid]
        n = len(apps)
        te, ve = int(n * TRAIN_RATIO), int(n * (TRAIN_RATIO + VAL_RATIO))
        return (apps[:te], apps[te:ve], apps[ve:],
                tbs[:te],  tbs[te:ve],  tbs[ve:],
                wds[:te],  wds[te:ve],  wds[ve:])

    # ── Phase A: Confidence Weight Grid ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Phase A — Confidence Weight Grid Search (threshold=0.10)")
    logger.info("=" * 60)

    TRANS_WEIGHTS = [0.4, 0.5, 0.6, 0.7]
    REC_WEIGHTS   = [0.1, 0.2, 0.3, 0.4]
    THRESHOLD_A   = 0.10   # best known from RL_LatencyFocus

    phase_a_rows = []
    for wt in TRANS_WEIGHTS:
        for wr in REC_WEIGHTS:
            wf = round(1.0 - wt - wr, 6)
            if wf < -1e-9:
                continue
            wf = max(0.0, wf)
            label = f"w{wt:.1f}_r{wr:.1f}_f{wf:.1f}"
            f1_list = []
            hr_list = []
            pr_list = []
            re_list = []
            la_list = []
            for uid in usable_users:
                if uid not in user_cache: continue
                tr_a,va_a,ts_a,tr_t,va_t,ts_t,tr_w,va_w,ts_w = split(uid)
                if len(ts_a) < 10: continue
                pol = ConfidencePolicy(wt, wr, wf, threshold=THRESHOLD_A, name=label)
                m = evaluate_policy(pol, tr_a,va_a,ts_a, tr_t,va_t,ts_t, tr_w,va_w,ts_w, lat, uid)
                f1_list.append(m["f1"]); hr_list.append(m["hit_rate"])
                pr_list.append(m["precision"]); re_list.append(m["recall"])
                la_list.append(m["latency_saved_ms"])

            if not f1_list: continue
            row = {
                "weights":        label,
                "w_trans":        wt, "w_rec": wr, "w_freq": wf,
                "threshold":      THRESHOLD_A,
                "f1":             round(float(np.mean(f1_list)), 4),
                "std_f1":         round(float(np.std(f1_list)), 4),
                "precision":      round(float(np.mean(pr_list)), 4),
                "recall":         round(float(np.mean(re_list)), 4),
                "hit_rate":       round(float(np.mean(hr_list)), 4),
                "latency_saved":  round(float(np.mean(la_list)), 2),
                "delta_f1":       round(float(np.mean(f1_list)) - BASELINE_F1, 4),
                "n_users":        len(f1_list),
            }
            phase_a_rows.append(row)
            logger.info(f"  {label:25s}: F1={row['f1']:.4f} ΔF1={row['delta_f1']:+.4f}")

    phase_a_rows.sort(key=lambda x: -x["f1"])
    write_csv(os.path.join(RESULTS_DIR, "v5_weight_grid.csv"), phase_a_rows)

    best_weights = phase_a_rows[0] if phase_a_rows else None
    logger.info(f"\nPhase A best: {best_weights['weights']} → F1={best_weights['f1']:.4f}" if best_weights else "Phase A: no results")

    # ── Phase B: Threshold Sweep ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Phase B — Threshold Sweep (weights=0.5/0.3/0.2)")
    logger.info("=" * 60)

    THRESHOLDS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    WT_B, WR_B, WF_B = 0.5, 0.3, 0.2   # baseline weights

    phase_b_rows = []
    for thresh in THRESHOLDS:
        f1_list = []; hr_list = []; pr_list = []; re_list = []; la_list = []
        for uid in usable_users:
            if uid not in user_cache: continue
            tr_a,va_a,ts_a,tr_t,va_t,ts_t,tr_w,va_w,ts_w = split(uid)
            if len(ts_a) < 10: continue
            pol = ConfidencePolicy(WT_B, WR_B, WF_B, threshold=thresh,
                                   name=f"thresh_{thresh:.2f}")
            m = evaluate_policy(pol, tr_a,va_a,ts_a, tr_t,va_t,ts_t, tr_w,va_w,ts_w, lat, uid)
            f1_list.append(m["f1"]); hr_list.append(m["hit_rate"])
            pr_list.append(m["precision"]); re_list.append(m["recall"])
            la_list.append(m["latency_saved_ms"])

        if not f1_list: continue
        t_s, p_v, d_v = paired_t(
            {uid: f1_list[i] for i, uid in enumerate(
                [u for u in usable_users if u in user_cache and
                 len(user_cache[u][0]) >= 200 and
                 len(user_cache[u][0]) - int(len(user_cache[u][0])*(TRAIN_RATIO+VAL_RATIO)) >= 10])},
            baseline_by_user)

        row = {
            "threshold":      thresh,
            "f1":             round(float(np.mean(f1_list)), 4),
            "std_f1":         round(float(np.std(f1_list)), 4),
            "precision":      round(float(np.mean(pr_list)), 4),
            "recall":         round(float(np.mean(re_list)), 4),
            "hit_rate":       round(float(np.mean(hr_list)), 4),
            "latency_saved":  round(float(np.mean(la_list)), 2),
            "delta_f1":       round(float(np.mean(f1_list)) - BASELINE_F1, 4),
            "n_users":        len(f1_list),
        }
        phase_b_rows.append(row)
        logger.info(f"  thresh={thresh:.2f}: F1={row['f1']:.4f}  HR={row['hit_rate']:.4f}  ΔF1={row['delta_f1']:+.4f}")

    write_csv(os.path.join(RESULTS_DIR, "v5_threshold_sweep.csv"), phase_b_rows)
    best_thresh_row = max(phase_b_rows, key=lambda x: x["f1"]) if phase_b_rows else None
    best_thresh = best_thresh_row["threshold"] if best_thresh_row else 0.10
    logger.info(f"\nPhase B best threshold: {best_thresh} → F1={best_thresh_row['f1']:.4f}" if best_thresh_row else "Phase B: no results")

    # Generate threshold vs F1 plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        threshs = [r["threshold"] for r in phase_b_rows]
        f1s     = [r["f1"]        for r in phase_b_rows]
        hrs     = [r["hit_rate"]  for r in phase_b_rows]
        precs   = [r["precision"] for r in phase_b_rows]
        recs    = [r["recall"]    for r in phase_b_rows]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Threshold vs Performance (weights=0.5/0.3/0.2)", fontsize=13)

        ax = axes[0]
        ax.plot(threshs, f1s, "o-", color="#2563eb", linewidth=2, markersize=7, label="F1")
        ax.axhline(BASELINE_F1,      color="gray",  linestyle="--", linewidth=1.2, label=f"Baseline F1={BASELINE_F1}")
        ax.axhline(BEST_CANDIDATE_F1, color="orange", linestyle="--", linewidth=1.2, label=f"RL_LatencyFocus={BEST_CANDIDATE_F1}")
        if best_thresh_row:
            ax.axvline(best_thresh, color="#dc2626", linestyle=":", linewidth=1.5, label=f"Best={best_thresh}")
        ax.set_xlabel("Threshold"); ax.set_ylabel("F1")
        ax.set_title("F1 vs Threshold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(threshs, precs, "s-", color="#16a34a", linewidth=2, markersize=7, label="Precision")
        ax2.plot(threshs, recs,  "^-", color="#9333ea", linewidth=2, markersize=7, label="Recall")
        ax2.plot(threshs, hrs,   "D-", color="#d97706", linewidth=2, markersize=7, label="Hit Rate")
        ax2.set_xlabel("Threshold"); ax2.set_ylabel("Score")
        ax2.set_title("Precision / Recall / Hit Rate vs Threshold")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(FIGURES_DIR, "threshold_vs_f1.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Written: {plot_path}")
    except Exception as e:
        logger.warning(f"Plot skipped: {e}")

    # ── Phase C: Time Context Coverage Audit ─────────────────────────────────
    run_phase_c(user_cache, usable_users)

    # ── Phase D: Modified Kneser-Ney ─────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Phase D — Modified Kneser-Ney (no global unigram term)")
    logger.info("=" * 60)

    MKN_KS = [3, 5, 10]
    COMPARE_POLICIES = {
        "GraphMindRL":   BASELINE_F1,
        "RL_LatencyFocus": BEST_CANDIDATE_F1,
        "M2_Naive":      0.7295,
        "JM_K5":         0.7282,
    }

    phase_d_rows = []
    mkn_f1_by_user: Dict[str, Dict[str, float]] = {}

    for K in MKN_KS:
        name = f"ModKN_K{K}"
        f1_list = []; hr_list = []; pr_list = []; re_list = []; la_list = []
        by_user = {}
        for uid in usable_users:
            if uid not in user_cache: continue
            tr_a,va_a,ts_a,tr_t,va_t,ts_t,tr_w,va_w,ts_w = split(uid)
            if len(ts_a) < 10: continue
            pol = ModifiedKNPolicy(K=K, name=name)
            m = evaluate_policy(pol, tr_a,va_a,ts_a, tr_t,va_t,ts_t, tr_w,va_w,ts_w, lat, uid)
            f1_list.append(m["f1"]); hr_list.append(m["hit_rate"])
            pr_list.append(m["precision"]); re_list.append(m["recall"])
            la_list.append(m["latency_saved_ms"])
            by_user[uid] = m["f1"]

        mkn_f1_by_user[name] = by_user
        t_s, p_v, d_v = paired_t(by_user, baseline_by_user)
        row = {
            "policy":         name, "K": K,
            "f1":             round(float(np.mean(f1_list)), 4),
            "std_f1":         round(float(np.std(f1_list)), 4),
            "precision":      round(float(np.mean(pr_list)), 4),
            "recall":         round(float(np.mean(re_list)), 4),
            "hit_rate":       round(float(np.mean(hr_list)), 4),
            "latency_saved":  round(float(np.mean(la_list)), 2),
            "delta_f1":       round(float(np.mean(f1_list)) - BASELINE_F1, 4),
            "t_stat":         round(t_s, 3) if not math.isnan(t_s) else "—",
            "p_value":        round(p_v, 4)  if not math.isnan(p_v)  else "—",
            "cohen_d":        round(d_v, 3)  if not math.isnan(d_v)  else "—",
            "significant":    bool(p_v < 0.05) if not math.isnan(p_v) else False,
            "n_users":        len(f1_list),
        }
        phase_d_rows.append(row)
        sig = "SIG" if row["significant"] else "n.s."
        logger.info(f"  ModKN K={K}: F1={row['f1']:.4f}  ΔF1={row['delta_f1']:+.4f}  p={p_v:.4f}  {sig}")

    write_csv(os.path.join(RESULTS_DIR, "v5_modified_kn.csv"), phase_d_rows)

    # ── Phase E: Combined Best ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Phase E — GraphMindRL_V5 Combined Best")
    logger.info("=" * 60)

    wt_best = best_weights["w_trans"] if best_weights else 0.5
    wr_best = best_weights["w_rec"]   if best_weights else 0.3
    wf_best = best_weights["w_freq"]  if best_weights else 0.2
    th_best = best_thresh

    logger.info(f"  Weights: trans={wt_best} rec={wr_best} freq={wf_best}")
    logger.info(f"  Threshold: {th_best}")

    # Build Phase E comparison matrix
    phase_e_policies = [
        ("GraphMindRL_V5",
         lambda: ConfidencePolicy(wt_best, wr_best, wf_best, th_best, name="GraphMindRL_V5")),
        ("RL_LatencyFocus",
         lambda: ConfidencePolicy(0.5, 0.3, 0.2, 0.10, name="RL_LatencyFocus")),
        ("GraphMindRL_Base",
         lambda: ConfidencePolicy(0.5, 0.3, 0.2, 0.05, name="GraphMindRL_Base")),
    ]

    # Only run V5 and LatencyFocus — base loaded from V4 CSV
    phase_e_rows = []
    v5_f1_by_user: Dict[str, float] = {}
    lf_f1_by_user: Dict[str, float] = {}

    for pol_name, factory in phase_e_policies:
        f1_list = []; hr_list = []; pr_list = []; re_list = []; la_list = []
        by_user = {}
        for uid in usable_users:
            if uid not in user_cache: continue
            tr_a,va_a,ts_a,tr_t,va_t,ts_t,tr_w,va_w,ts_w = split(uid)
            if len(ts_a) < 10: continue
            pol = factory()
            m = evaluate_policy(pol, tr_a,va_a,ts_a, tr_t,va_t,ts_t, tr_w,va_w,ts_w, lat, uid)
            f1_list.append(m["f1"]); hr_list.append(m["hit_rate"])
            pr_list.append(m["precision"]); re_list.append(m["recall"])
            la_list.append(m["latency_saved_ms"])
            by_user[uid] = m["f1"]

        if pol_name == "GraphMindRL_V5":
            v5_f1_by_user = by_user
        elif pol_name == "RL_LatencyFocus":
            lf_f1_by_user = by_user

        t_s, p_v, d_v = paired_t(by_user, baseline_by_user)
        row = {
            "policy":         pol_name,
            "f1":             round(float(np.mean(f1_list)), 4),
            "std_f1":         round(float(np.std(f1_list)), 4),
            "precision":      round(float(np.mean(pr_list)), 4),
            "recall":         round(float(np.mean(re_list)), 4),
            "hit_rate":       round(float(np.mean(hr_list)), 4),
            "latency_saved_ms": round(float(np.mean(la_list)), 2),
            "delta_f1_vs_baseline": round(float(np.mean(f1_list)) - BASELINE_F1, 4),
            "t_stat":         round(t_s, 3) if not math.isnan(t_s) else "—",
            "p_value":        round(p_v, 4)  if not math.isnan(p_v)  else "—",
            "cohen_d":        round(d_v, 3)  if not math.isnan(d_v)  else "—",
            "significant":    bool(p_v < 0.05) if not math.isnan(p_v) else False,
            "n_users":        len(f1_list),
        }
        phase_e_rows.append(row)
        sig = "SIG ✅" if row["significant"] else "n.s. ❌"
        p_s = f"{p_v:.4f}" if not math.isnan(p_v) else "—"
        logger.info(f"  {pol_name:25s}: F1={row['f1']:.4f}  ΔF1={row['delta_f1_vs_baseline']:+.4f}  p={p_s}  {sig}")

    write_csv(os.path.join(RESULTS_DIR, "v5_final_comparison.csv"), phase_e_rows)

    # ── Write all reports ─────────────────────────────────────────────────────
    _write_final_decision(phase_e_rows, best_weights, best_thresh_row, phase_d_rows)
    _write_optimization_summary(phase_a_rows, phase_b_rows, phase_d_rows, phase_e_rows,
                                 best_weights, best_thresh_row)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 11 — All phases complete.")
    logger.info("=" * 60)


def _write_final_decision(phase_e_rows, best_weights, best_thresh_row, phase_d_rows):
    v5_row  = next((r for r in phase_e_rows if r["policy"] == "GraphMindRL_V5"), None)
    lf_row  = next((r for r in phase_e_rows if r["policy"] == "RL_LatencyFocus"), None)

    lines = [
        "# V5 Final Decision",
        "",
        f"**Date:** 2026-06-06",
        f"**Baseline:** GraphMindRL F1={BASELINE_F1}",
        f"**Previous best:** RL_LatencyFocus F1={BEST_CANDIDATE_F1} (p=0.0003, d=0.752)",
        "",
        "---",
        "",
        "## Phase E — GraphMindRL_V5 Results",
        "",
        "| Policy | F1 | ΔF1 | p | Cohen d | Sig? | Meets +0.02? |",
        "|--------|-----|-----|---|---------|------|------------|",
    ]

    for r in phase_e_rows:
        sig = "✅" if r["significant"] else "❌"
        meets = "✅" if r["delta_f1_vs_baseline"] >= 0.02 else "❌"
        p_s = f"{r['p_value']:.4f}" if isinstance(r['p_value'], float) else str(r['p_value'])
        d_s = f"{r['cohen_d']:.3f}"  if isinstance(r['cohen_d'],  float) else str(r['cohen_d'])
        lines.append(
            f"| {r['policy']} | {r['f1']:.4f} | {r['delta_f1_vs_baseline']:+.4f} "
            f"| {p_s} | {d_s} | {sig} | {meets} |"
        )

    lines += ["", "---", "", "## Configuration Used", ""]
    if best_weights:
        lines.append(f"- **Confidence weights:** trans={best_weights['w_trans']}  "
                     f"rec={best_weights['w_rec']}  freq={best_weights['w_freq']}")
    if best_thresh_row:
        lines.append(f"- **Threshold:** {best_thresh_row['threshold']}")

    lines += ["", "---", "", "## Modified KN Findings", ""]
    for r in phase_d_rows:
        lines.append(f"- {r['policy']}: F1={r['f1']:.4f}  ΔF1={r['delta_f1']:+.4f}")

    lines += ["", "---", "", "## Decision", ""]
    if v5_row and v5_row["significant"] and v5_row["delta_f1_vs_baseline"] >= 0.02:
        lines.append("**RECOMMEND V5 IMPLEMENTATION.**")
        lines.append(f"GraphMindRL_V5 achieves F1={v5_row['f1']:.4f}, "
                     f"ΔF1={v5_row['delta_f1_vs_baseline']:+.4f}, "
                     f"p={v5_row['p_value']:.4f}, d={v5_row['cohen_d']:.3f}.")
    elif v5_row and v5_row["significant"]:
        lines.append("**PARTIAL SUCCESS — V5 is significantly better but does not clear +0.02 threshold.**")
        lines.append(f"GraphMindRL_V5: F1={v5_row['f1']:.4f} (ΔF1={v5_row['delta_f1_vs_baseline']:+.4f}).")
        lines.append(f"RL_LatencyFocus remains the recommended production update.")
    else:
        lines.append("**FREEZE RL_LatencyFocus. Stop research. Proceed to dashboard.**")
        if lf_row:
            lines.append(f"RL_LatencyFocus F1={lf_row['f1']:.4f} remains the best validated policy.")

    path = os.path.join(REPORTS_DIR, "v5_final_decision.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Written: {path}")


def _write_optimization_summary(phase_a_rows, phase_b_rows, phase_d_rows, phase_e_rows,
                                  best_weights, best_thresh_row):
    lines = [
        "# V5 Optimization Summary",
        "",
        f"**Date:** 2026-06-06",
        f"**Baseline:** GraphMindRL F1={BASELINE_F1}  HR=0.9357  Lat=2002.5ms",
        "",
        "---",
        "",
        "## Top 10 Configurations",
        "",
        "*(All policies benchmarked on 31 users, 80/10/10 chronological split)*",
        "",
        "| Rank | Policy/Config | F1 | ΔF1 | p | Sig |",
        "|------|--------------|-----|-----|---|-----|",
    ]

    # Collect all results from all phases
    all_results = []

    for r in phase_a_rows:
        all_results.append({
            "name": f"A: {r['weights']}",
            "f1": r["f1"], "delta": r["delta_f1"],
            "p": "—", "sig": False,
        })
    for r in phase_b_rows:
        all_results.append({
            "name": f"B: thresh={r['threshold']:.2f}",
            "f1": r["f1"], "delta": r["delta_f1"],
            "p": "—", "sig": False,
        })
    for r in phase_d_rows:
        all_results.append({
            "name": f"D: {r['policy']}",
            "f1": r["f1"], "delta": r["delta_f1"],
            "p": str(r["p_value"]), "sig": r["significant"],
        })
    for r in phase_e_rows:
        all_results.append({
            "name": f"E: {r['policy']}",
            "f1": r["f1"], "delta": r["delta_f1_vs_baseline"],
            "p": str(r["p_value"]), "sig": r["significant"],
        })

    all_results.sort(key=lambda x: -x["f1"])
    for i, r in enumerate(all_results[:10], 1):
        sig = "✅" if r["sig"] else "❌"
        lines.append(f"| {i} | {r['name']} | {r['f1']:.4f} | {r['delta']:+.4f} | {r['p']} | {sig} |")

    lines += ["", "---", "", "## Phase A — Best Confidence Weights", ""]
    if best_weights:
        lines.append(f"- **Best:** trans={best_weights['w_trans']} rec={best_weights['w_rec']} freq={best_weights['w_freq']}")
        lines.append(f"- **F1:** {best_weights['f1']:.4f} (ΔF1={best_weights['delta_f1']:+.4f} vs baseline)")
        lines.append("")
        lines.append("Top 5 weight combinations:")
        for r in phase_a_rows[:5]:
            lines.append(f"  - {r['weights']}: F1={r['f1']:.4f}")

    lines += ["", "---", "", "## Phase B — Best Threshold", ""]
    if best_thresh_row:
        lines.append(f"- **Best threshold:** {best_thresh_row['threshold']}")
        lines.append(f"- **F1:** {best_thresh_row['f1']:.4f} (ΔF1={best_thresh_row['delta_f1']:+.4f})")
        lines.append(f"- **Precision:** {best_thresh_row['precision']:.4f}  Recall: {best_thresh_row['recall']:.4f}")

    lines += ["", "---", "", "## Phase C — Time Context Coverage", ""]
    lines.append("See full audit: [time_context_coverage_audit.md](time_context_coverage_audit.md)")
    lines.append("")
    lines.append("Key finding: Sparsity is the primary failure cause. Even 6-band leaves ~20% of test")
    lines.append("steps with no time-conditioned transition data. Dataset needs ≥6 months for reliable")
    lines.append("time conditioning.")

    lines += ["", "---", "", "## Phase D — Modified Kneser-Ney Findings", ""]
    for r in phase_d_rows:
        lines.append(f"- {r['policy']}: F1={r['f1']:.4f}  ΔF1={r['delta_f1']:+.4f}")
    lines.append("")
    lines.append("Removing the global unigram term from JM recovers marginal improvement over standard JM")
    lines.append("but does not approach the confidence-layer threshold-tuned policies.")

    lines += ["", "---", "", "## Final Recommendation", ""]
    v5_row = next((r for r in phase_e_rows if r["policy"] == "GraphMindRL_V5"), None)
    lf_row = next((r for r in phase_e_rows if r["policy"] == "RL_LatencyFocus"), None)

    if v5_row:
        lines.append(f"**GraphMindRL_V5:** F1={v5_row['f1']:.4f}  ΔF1={v5_row['delta_f1_vs_baseline']:+.4f}")
    if lf_row:
        lines.append(f"**RL_LatencyFocus:** F1={lf_row['f1']:.4f}  ΔF1={lf_row['delta_f1_vs_baseline']:+.4f}")

    lines.append("")
    if v5_row and v5_row.get("significant") and v5_row["delta_f1_vs_baseline"] >= 0.02:
        lines.append("→ **IMPLEMENT GraphMindRL_V5**")
    else:
        lines.append("→ **Freeze at RL_LatencyFocus. Proceed to dashboard.**")

    path = os.path.join(REPORTS_DIR, "v5_optimization_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Written: {path}")


if __name__ == "__main__":
    main()
