#!/usr/bin/env python3
"""
scripts/run_v5_rl_graph.py

GraphMind V5 — RL-on-Graph Benchmark.

Two mechanisms combined:
  #1 RL Per-Candidate Edge Scorer
     Graph proposes top-15 candidates.
     RL learns weight vector W (6-dim) over edge features:
       features = [trans_prob, log_count_norm, out_degree_norm,
                   in_degree_norm, recency_norm, frequency_norm]
     score(c|a) = W · features(a→c)   (linear, no sigmoid at select time)
     REINFORCE: F1-proxy reward, per-candidate gradient.

  #5 RL Joint Threshold + Budget Controller
     Context state S (8-dim) → [threshold_logit, budget_logit]
     via learned W_ctx (8×2) + b_ctx (2).
     threshold = 0.05 + 0.20 × sigmoid(W_ctx[:,0]·S + b_ctx[0])
     budget    = 3    + int(7 × sigmoid(W_ctx[:,1]·S + b_ctx[1]))

Policies tested (each vs baseline GraphMindRL F1=0.7424):
  GraphRL_EdgeScorer          — full combined policy (3 REINFORCE passes)
  GraphRL_EdgeScorer_5pass    — 5 training passes
  GraphRL_EdgeScorer_OnlineRL — light online RL updates during test
  GraphRL_CtxOnly             — context controller only, fixed M1 scoring
  GraphRL_EdgeOnly            — edge scorer only, fixed threshold=0.10
  RL_LatencyFocus             — Phase 7 best baseline (threshold=0.10)

Outputs:
  results/v5_rl_graph.csv          — per-user per-policy results
  results/v5_rl_graph_summary.csv  — aggregated means + significance
  reports/rl_graph_results.md      — human-readable report
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
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
UBIQLOG_ROOT  = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog", "UbiqLog4UCI")
LATENCY_CSV   = os.path.join(PROJECT_ROOT, "datasets", "app_launch_latency.csv")

HOT_SIZE        = 5
WARM_SIZE       = 15
TRAIN_RATIO     = 0.80
VAL_RATIO       = 0.10
MIN_YEAR        = 2011
MAX_YEAR        = 2016
BASELINE_F1     = 0.7424   # GraphMindRL V3

SYSTEM_PREFIXES = (
    "com.android.", "com.google.android.providers",
    "com.google.android.gms", "com.google.android.gsf",
    "com.sec.android.provider", "com.samsung.android.provider",
    "com.redbend.", "android.",
)
SYSTEM_SUFFIXES = (":engine", ":client", ":daemon", ":service", ":pushservice", ":sync")


# ── Math helpers ──────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))

def _clip(x, lo, hi):
    return max(lo, min(hi, x))


# ── Latency model ─────────────────────────────────────────────────────────────

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
        if tier == "hot":
            return max(0.0, cold - (self._hot.get(k, self._DH) if k else self._DH))
        if tier == "warm":
            return max(0.0, cold - (self._warm.get(k, self._DW) if k else self._DW))
        return 0.0


# ── Cache simulator ───────────────────────────────────────────────────────────

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
    def reset(self):
        self._hot = []; self._warm = []


# ── Data loader ───────────────────────────────────────────────────────────────

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


# ── Evaluation engine (same logic as V4) ─────────────────────────────────────

def evaluate_policy(policy, train_apps, val_apps, test_apps,
                    train_tbs, val_tbs, test_tbs,
                    train_wds, val_wds, test_wds, lat, user_id="x"):
    policy.train(train_apps, tbs=train_tbs, wds=train_wds,
                 val_apps=val_apps, val_tbs=val_tbs, val_wds=val_wds)
    policy.reset_state()

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

        tier    = cache.lookup(cur)
        is_hit  = tier in ("hot", "warm")

        if is_hit:
            hits += 1; tp += 1
            lat_saved += lat.saved(cur, tier)
        else:
            misses += 1

        if i + 1 < len(test_apps):
            nxt = test_apps[i + 1]
            if preds:
                if nxt in preds: tp += 1
                else:            fn += 1; fp += len(preds)
            else:
                fn += 1

        cache.access(cur)
        policy.update(cur, hit=is_hit)
        prev = cur

    total = hits + misses or 1
    hr  = hits / total
    pr  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    re  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1  = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0.0
    avg_lat_saved = lat_saved / total

    return {
        "hit_rate":         round(hr,  4),
        "precision":        round(pr,  4),
        "recall":           round(re,  4),
        "f1":               round(f1,  4),
        "latency_saved_ms": round(avg_lat_saved, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CORE: GraphRL_EdgeScorer
#  Combines:
#    #1 — RL per-candidate edge feature scorer (W, 6-dim)
#    #5 — RL joint threshold + budget controller (W_ctx 8×2, b_ctx 2)
# ══════════════════════════════════════════════════════════════════════════════

class GraphRL_EdgeScorer:
    """
    RL-on-graph policy combining edge scoring + context-adaptive threshold/budget.

    GRAPH
    -----
    Built from training sequence.  Per-edge statistics stored:
      trans_prob  = P(dst | src) normalised from counts
      raw_count   = int occurrence count
    Per-node statistics:
      out_degree  = total transitions leaving src
      in_degree   = total transitions arriving at dst

    EDGE FEATURES  (6-dim, fed to RL scorer)
    ------------
      f[0] trans_prob           in [0, 1]
      f[1] log_count_norm       log(count+1) / log(max_count+1)  in [0, 1]
      f[2] out_degree_norm      out_deg(src) / max_out_deg        in [0, 1]
      f[3] in_degree_norm       in_deg(dst)  / max_in_deg         in [0, 1]
      f[4] recency_norm         exp-decay recency of dst          in [0, 1]
      f[5] frequency_norm       freq(dst) / total × 20, clipped   in [0, 1]

    RL EDGE SCORER  (W, 6-dim)
    ----------------
    score(dst | src) = W · features(src→dst)
    Candidates: top-POOL_SIZE from graph, then sorted by RL score.
    Selected:   those with score ≥ threshold_raw, up to budget.
    Gradient:   REINFORCE on per-candidate binary outcomes.
      TP candidate: +advantage × sigmoid_grad × feat
      FP candidate: -advantage × sigmoid_grad × feat  (weighted 0.8)
      FN candidate: +advantage × sigmoid_grad × feat  (weighted 0.5)

    RL CONTEXT CONTROLLER  (W_ctx 8×2, b_ctx 2)
    ----------------------
    State S (8-dim):
      s[0] out_degree_norm(src)          graph branching factor
      s[1] tanh(top1_rl_score)           how confident the best candidate is
      s[2] tanh(top1−top2 margin)        score separation (peakedness)
      s[3] recent_hit_rate               20-step rolling window
      s[4] recent_precision              50-step TP/(TP+FP)
      s[5] recent_fp_rate                50-step FP/(TP+FP)
      s[6] time_bucket / 47              normalised time of day
      s[7] weekday / 6                   normalised day of week

    threshold = 0.05 + 0.20 × sigmoid(W_ctx[:,0]·S + b_ctx[0])   ∈ [0.05, 0.25]
    budget    = 3    + int(7 × sigmoid(W_ctx[:,1]·S + b_ctx[1]))   ∈ [3, 10]
    threshold_raw = logit(threshold)  used for score comparison

    REINFORCE UPDATE  (both W and W_ctx jointly)
    -----------------
    Reward = F1-proxy = 2PR/(P+R) per step  ∈ [0, 1]
    Baseline: EMA with decay BASELINE_DECAY
    Advantage = reward - baseline
    W update:   W ← W + LR_W × advantage × grad_W
    Ctx update: W_ctx ← W_ctx + LR_CTX × advantage × grad_ctx (outer product)

    TRAINING
    --------
    Pass 1: build graph and statistics
    Passes 2..N_PASSES: REINFORCE through training sequence (LR decayed each pass)
    After training: W, W_ctx frozen for test (online_update=False)
                    or lightly updated with 0.05× LR (online_update=True)
    """

    # Hyperparameters (class-level so subclasses can override)
    N_EDGE_FEATS  = 6
    N_STATE_DIMS  = 8
    POOL_SIZE     = 15      # candidates from graph before RL scoring
    N_PASSES      = 3       # REINFORCE training passes
    LR_W          = 0.05    # edge scorer learning rate
    LR_CTX        = 0.03    # context controller learning rate
    BASELINE_DECAY = 0.95
    FP_PENALTY    = 0.8     # relative weight of FP penalty vs TP reward
    FN_REWARD     = 0.5     # relative weight of FN recovery push
    ONLINE_LR_SCALE = 0.0   # online LR during test (0 = frozen)

    def __init__(self, name: str = "GraphRL_EdgeScorer"):
        self.name = name
        self._reset_all()

    def _reset_all(self):
        # Graph structures
        self._graph:   Dict[str, Dict[str, float]] = {}
        self._counts:  Dict[str, Dict[str, int]]   = {}
        self._out_deg: Dict[str, int] = {}
        self._in_deg:  Dict[str, int] = {}
        self._max_count   = 1.0
        self._max_out_deg = 1.0
        self._max_in_deg  = 1.0
        self._total_trans = 1

        # RL parameters (initialised to small noise so symmetry is broken)
        rng = np.random.default_rng(42)
        self._W     = rng.normal(0, 0.01, self.N_EDGE_FEATS)
        self._W_ctx = rng.normal(0, 0.01, (self.N_STATE_DIMS, 2))
        self._b_ctx = np.zeros(2)
        self._baseline = 0.0

        # Online state (recency, frequency, windows)
        self._recency: Dict[str, float] = defaultdict(float)
        self._freq:    Dict[str, int]   = defaultdict(int)
        self._total_obs = 0
        self._hit_hist: deque = deque(maxlen=20)
        self._tp_win:   deque = deque(maxlen=50)
        self._fp_win:   deque = deque(maxlen=50)
        self._fn_win:   deque = deque(maxlen=50)
        self._last_preds: List[str] = []

    # ── Feature extraction ────────────────────────────────────────────────────

    def _edge_feat(self, src: str, dst: str) -> np.ndarray:
        prob  = self._graph.get(src, {}).get(dst, 0.0)
        cnt   = self._counts.get(src, {}).get(dst, 0)
        log_c = math.log1p(cnt) / math.log1p(self._max_count + 1)
        out_n = self._out_deg.get(src, 0) / (self._max_out_deg + 1e-9)
        in_n  = self._in_deg.get(dst, 0)  / (self._max_in_deg  + 1e-9)
        tot   = max(self._total_obs, 1)
        rec_n = self._recency.get(dst, 0.0)                           # already [0,1]
        freq_n = _clip(self._freq.get(dst, 0) / tot * 20, 0.0, 1.0)  # scale so 5% → 1.0
        return np.array([prob, log_c, out_n, in_n, rec_n, freq_n], dtype=np.float64)

    def _build_state(self, src: str, tb: int, wd: int,
                     scored_cands: Optional[List[Tuple]] = None) -> np.ndarray:
        # out-degree (branching factor)
        n_cands = len(self._graph.get(src, {}))
        out_n   = _clip(n_cands / 20.0, 0.0, 1.0)

        # top-1 score and margin from the scored candidates
        if scored_cands and len(scored_cands) >= 1:
            top1_score = scored_cands[0][1]
            margin     = (scored_cands[0][1] - scored_cands[1][1]) if len(scored_cands) > 1 else top1_score
        else:
            top1_score = 0.0
            margin     = 0.0

        # rolling windows
        hr  = sum(self._hit_hist) / max(len(self._hit_hist), 1)
        tp  = sum(self._tp_win); fp = sum(self._fp_win); fn = sum(self._fn_win)
        prec   = tp / max(tp + fp, 1)
        fp_rate = fp / max(tp + fp + 1, 1)

        return np.array([
            out_n,
            math.tanh(top1_score),    # normalise raw score with tanh
            math.tanh(margin),
            hr,
            prec,
            fp_rate,
            tb / 47.0,
            wd / 6.0,
        ], dtype=np.float64)

    # ── Threshold / budget from context controller ────────────────────────────

    def _get_ctrl(self, state: np.ndarray) -> Tuple[float, int, float]:
        """Returns (threshold, budget, threshold_raw_score)."""
        logits    = state @ self._W_ctx + self._b_ctx          # (2,)
        threshold = 0.05 + 0.20 * _sigmoid(logits[0])         # [0.05, 0.25]
        budget    = 3 + int(7 * _sigmoid(logits[1]))           # [3, 10]
        # Convert threshold probability to raw score cutoff
        # score(c) ≥ thresh_raw  ↔  sigmoid(score) ≥ threshold  (if we used sigmoid)
        # We use raw linear score, so map threshold to its logit for comparison
        thresh_raw = math.log(_clip(threshold, 1e-6, 1-1e-6) /
                               (1 - _clip(threshold, 1e-6, 1-1e-6)))
        return threshold, budget, thresh_raw

    # ── Score and select ──────────────────────────────────────────────────────

    def _score_and_select(self, src: str, tb: int, wd: int):
        """
        Returns (selected_apps, debug_info).
        debug_info = (state, scored_cands, threshold, budget, thresh_raw)
        scored_cands sorted descending by RL score.
        """
        # Get top-POOL_SIZE apps from graph (by transition prob)
        pool = list(self._graph.get(src, {}).keys())[:self.POOL_SIZE]
        if not pool:
            return [], None

        # Compute RL score for each candidate
        scored: List[Tuple[str, float, np.ndarray]] = []
        for dst in pool:
            feat  = self._edge_feat(src, dst)
            score = float(np.dot(self._W, feat))
            scored.append((dst, score, feat))

        # Sort by RL score descending
        scored.sort(key=lambda x: -x[1])

        # Build state using scored information
        state = self._build_state(src, tb, wd, scored)
        threshold, budget, thresh_raw = self._get_ctrl(state)

        # Select candidates above threshold cutoff, up to budget
        selected = [dst for dst, sc, _ in scored if sc >= thresh_raw][:budget]

        return selected, (state, scored, threshold, budget, thresh_raw)

    # ── REINFORCE gradient update ─────────────────────────────────────────────

    def _reinforce_update(self,
                          state: np.ndarray,
                          scored: List[Tuple],
                          selected: List[str],
                          actual_next: str,
                          thresh_raw: float,
                          budget: int,
                          lr_scale: float = 1.0):

        if not scored:
            return

        # ── Reward: F1-proxy ──────────────────────────────────────────────────
        tp_s = 1 if actual_next in selected else 0
        fp_s = max(0, len(selected) - tp_s)
        fn_s = 1 - tp_s
        P = tp_s / max(len(selected), 1)
        R = tp_s / max(tp_s + fn_s, 1)
        reward = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

        # ── Advantage ─────────────────────────────────────────────────────────
        advantage = reward - self._baseline
        self._baseline = (self.BASELINE_DECAY * self._baseline
                          + (1 - self.BASELINE_DECAY) * reward)

        # ── Gradient for W (edge scorer) ─────────────────────────────────────
        # We treat selection as a binary classifier per candidate:
        # positive class = actual_next,  negative = everything else selected
        grad_W = np.zeros(self.N_EDGE_FEATS, dtype=np.float64)

        selected_set = set(selected)
        for dst, sc, feat in scored:
            sig = _sigmoid(sc)           # sigmoid of raw score ∈ (0,1)
            g   = sig * (1.0 - sig)      # sigmoid derivative

            if dst == actual_next:
                if dst in selected_set:
                    # TP: reinforce (push score up → keep selecting this)
                    grad_W += g * feat * advantage
                else:
                    # FN: push score up with partial credit
                    grad_W += g * feat * abs(advantage) * self.FN_REWARD
            elif dst in selected_set:
                # FP: penalise (push score down)
                grad_W -= g * feat * abs(advantage) * self.FP_PENALTY

        self._W += self.LR_W * lr_scale * grad_W

        # ── Gradient for W_ctx (context controller) ───────────────────────────
        logits = state @ self._W_ctx + self._b_ctx   # (2,)

        # Gradient wrt threshold logit:
        #   if FN (missed): lower threshold → increase sigmoid → neg gradient
        #   if too many FPs: raise threshold → positive gradient
        sig_t = _sigmoid(logits[0])
        d_thresh = 0.0
        if actual_next not in selected_set:
            # Missed — lower threshold (more inclusive)
            d_thresh = -sig_t * (1.0 - sig_t)
        elif fp_s > 2:
            # Too many FPs — raise threshold
            d_thresh = sig_t * (1.0 - sig_t) * advantage

        # Gradient wrt budget logit:
        #   reward aligns with right budget: advantage drives both directions
        sig_b   = _sigmoid(logits[1])
        d_budget = sig_b * (1.0 - sig_b) * advantage * 0.5

        # Outer-product update: (N_STATE,) × (2,) → (N_STATE, 2)
        d_ctx         = np.outer(state, np.array([d_thresh, d_budget]))
        self._W_ctx  += self.LR_CTX * lr_scale * d_ctx
        self._b_ctx  += self.LR_CTX * lr_scale * np.array([d_thresh, d_budget]) * 0.1

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, apps: List[str], tbs=None, wds=None,
              val_apps=None, val_tbs=None, val_wds=None, **kw):
        tbs = tbs or [0] * len(apps)
        wds = wds or [0] * len(apps)

        # ── Pass 0: build graph statistics ────────────────────────────────────
        c_cnt: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c_cnt[apps[i-1]][apps[i]] += 1

        self._counts = {s: dict(d) for s, d in c_cnt.items()}

        # Normalise to get transition probs
        self._graph = {}
        for s, d in self._counts.items():
            tot = sum(d.values())
            self._graph[s] = dict(sorted(
                {k: v / tot for k, v in d.items()}.items(), key=lambda x: -x[1]
            ))

        # Degree statistics
        self._out_deg = {s: sum(d.values()) for s, d in self._counts.items()}
        in_deg_raw: Dict[str, int] = defaultdict(int)
        for s, d in self._counts.items():
            for t, cnt in d.items():
                in_deg_raw[t] += cnt
        self._in_deg      = dict(in_deg_raw)
        self._max_count   = float(max(
            (max(d.values()) for d in self._counts.values()), default=1))
        self._max_out_deg = float(max(self._out_deg.values(), default=1))
        self._max_in_deg  = float(max(self._in_deg.values(),  default=1))
        self._total_trans = sum(self._out_deg.values())

        # Build recency/frequency from training data (warm-up)
        for i, app in enumerate(apps):
            for k in list(self._recency.keys()):
                self._recency[k] *= 0.99
            self._recency[app] = 1.0
            self._freq[app]   += 1
            self._total_obs   += 1

        # Normalise recency to [0,1]
        max_rec = max(self._recency.values(), default=1.0)
        for k in self._recency:
            self._recency[k] /= max(max_rec, 1e-9)

        # Snapshot online state for restoration after each pass
        rec_snap  = dict(self._recency)
        freq_snap = dict(self._freq)
        tot_snap  = self._total_obs

        # ── Passes 1..N_PASSES: REINFORCE ─────────────────────────────────────
        for pass_idx in range(self.N_PASSES):
            lr_scale = 1.0 / (1.0 + pass_idx * 0.5)

            # Reset online state to training snapshot at each pass start
            self._recency   = defaultdict(float, rec_snap)
            self._freq      = defaultdict(int, freq_snap)
            self._total_obs = tot_snap
            self._hit_hist.clear()
            self._tp_win.clear(); self._fp_win.clear(); self._fn_win.clear()
            self._baseline = 0.0

            for i in range(len(apps) - 1):
                cur = apps[i]
                nxt = apps[i + 1]
                tb  = tbs[i]
                wd  = wds[i]

                selected, info = self._score_and_select(cur, tb, wd)
                if info is not None:
                    state, scored, threshold, budget, thresh_raw = info
                    self._reinforce_update(
                        state, scored, selected, nxt, thresh_raw, budget, lr_scale)

                # Update online state for next step in this pass
                for k in list(self._recency.keys()):
                    self._recency[k] *= 0.95
                self._recency[cur] = 1.0

                # Update TP/FP/FN for context state
                if selected:
                    is_hit_pred = nxt in selected
                    self._tp_win.append(1 if is_hit_pred else 0)
                    self._fp_win.append(0 if is_hit_pred else len(selected))
                    self._fn_win.append(0 if is_hit_pred else 1)
                self._hit_hist.append(1.0)  # approximate during training

        # Restore clean online state for test evaluation
        self._recency   = defaultdict(float, rec_snap)
        self._freq      = defaultdict(int, freq_snap)
        self._total_obs = tot_snap

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, cur: str, prev=None, tb: int = 0, wd: int = 0) -> List[str]:
        selected, info = self._score_and_select(cur, tb, wd)
        self._last_preds = selected
        if info is not None and self.ONLINE_LR_SCALE > 0:
            # Store info for deferred online update in update()
            self._pending_info = info
        else:
            self._pending_info = None
        return selected

    def update(self, app: str, hit: bool = False):
        # Recency / frequency update
        for k in list(self._recency.keys()):
            self._recency[k] *= 0.95
        self._recency[app] = 1.0
        self._freq[app]   += 1
        self._total_obs   += 1
        self._hit_hist.append(1.0 if hit else 0.0)

        # Update prediction quality windows
        if self._last_preds:
            is_in = app in self._last_preds
            self._tp_win.append(1 if is_in else 0)
            self._fp_win.append(0 if is_in else len(self._last_preds))
            self._fn_win.append(0 if is_in else 1)

        # Online RL update (light, only if enabled)
        if (self.ONLINE_LR_SCALE > 0
                and hasattr(self, "_pending_info")
                and self._pending_info is not None):
            state, scored, threshold, budget, thresh_raw = self._pending_info
            self._reinforce_update(
                state, scored, self._last_preds, app,
                thresh_raw, budget, self.ONLINE_LR_SCALE)
            self._pending_info = None

    def reset_state(self):
        """Called between train and eval — clears transient hit/window state."""
        self._hit_hist.clear()
        self._tp_win.clear(); self._fp_win.clear(); self._fn_win.clear()
        self._last_preds = []
        self._pending_info = None


# ══════════════════════════════════════════════════════════════════════════════
#  VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

class GraphRL_EdgeScorer_5pass(GraphRL_EdgeScorer):
    N_PASSES = 5
    def __init__(self):
        super().__init__(name="GraphRL_EdgeScorer_5pass")


class GraphRL_EdgeScorer_OnlineRL(GraphRL_EdgeScorer):
    """Full edge scorer with light online RL update during test."""
    ONLINE_LR_SCALE = 0.05
    def __init__(self):
        super().__init__(name="GraphRL_EdgeScorer_OnlineRL")


class GraphRL_CtxOnly(GraphRL_EdgeScorer):
    """Context controller only: W stays zero, fixed M1 scores candidates."""
    N_PASSES = 3
    def __init__(self):
        super().__init__(name="GraphRL_CtxOnly")

    def train(self, apps, tbs=None, wds=None, **kw):
        # Build graph and statistics as normal
        super().train(apps, tbs=tbs, wds=wds, **kw)
        # Zero out W so scoring reduces to graph transition prob order
        self._W = np.zeros(self.N_EDGE_FEATS)

    def predict(self, cur, prev=None, tb=0, wd=0):
        # Score using trans_prob only (feature[0]) via W_ctx threshold control
        pool = list(self._graph.get(cur, {}).keys())[:self.POOL_SIZE]
        if not pool:
            self._last_preds = []
            return []
        # Build a minimal scored list using graph prob order (W is zero so raw scores all 0)
        # Use transition probs as scores instead
        scored = [(dst, self._graph[cur][dst], self._edge_feat(cur, dst))
                  for dst in pool]
        scored.sort(key=lambda x: -x[1])
        state = self._build_state(cur, tb, wd, scored)
        _, budget, _ = self._get_ctrl(state)
        self._last_preds = [d for d, _, _ in scored][:budget]
        return self._last_preds


class GraphRL_EdgeOnly(GraphRL_EdgeScorer):
    """Edge scorer only: W_ctx fixed to give threshold=0.10, budget=5."""
    N_PASSES = 3
    def __init__(self):
        super().__init__(name="GraphRL_EdgeOnly")

    def _get_ctrl(self, state):
        # Fixed threshold=0.10, budget=5 (replicate RL_LatencyFocus)
        threshold  = 0.10
        budget     = 5
        thresh_raw = math.log(threshold / (1 - threshold))
        return threshold, budget, thresh_raw


class RL_LatencyFocus:
    """Phase 7 best (threshold=0.10, budget=5, adaptive ±0.005). Reference."""
    name = "RL_LatencyFocus"

    def __init__(self):
        self._g     = {}
        self._rec   = defaultdict(float)
        self._freq  = defaultdict(float)
        self._total = 0.0
        self._hist  = deque(maxlen=20)
        self._thresh = 0.10
        self._budget = HOT_SIZE
        self._last_preds = []

    def train(self, apps, tbs=None, wds=None, **kw):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c[apps[i-1]][apps[i]] += 1
        self._g = {s: dict(sorted({k: v/sum(d.values()) for k,v in d.items()}.items(),
                                   key=lambda x: -x[1]))
                   for s, d in c.items()}

    def predict(self, cur, prev=None, tb=0, wd=0):
        if cur not in self._g:
            self._last_preds = []
            return []
        tot = self._total or 1.0
        cands = {app: 0.5*p + 0.3*self._rec.get(app,0) + 0.2*(self._freq.get(app,0)/tot)
                 for app, p in self._g[cur].items()}
        self._last_preds = sorted(
            (a for a, c in cands.items() if c >= self._thresh),
            key=lambda a: -cands[a])[:self._budget]
        return self._last_preds

    def update(self, app, hit=False):
        for k in self._rec: self._rec[k] *= 0.95
        self._rec[app] = 1.0; self._freq[app] += 1; self._total += 1
        self._hist.append(1.0 if hit else 0.0)
        if len(self._hist) == 20:
            hr = sum(self._hist) / 20
            if hr < 0.5:
                self._thresh = max(0.05, self._thresh - 0.005)
            elif hr > 0.8:
                self._thresh = min(0.25, self._thresh + 0.005)

    def reset_state(self):
        self._hist.clear(); self._last_preds = []
        self._thresh = 0.10; self._budget = HOT_SIZE


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    lat = MeasuredLatencyModel(LATENCY_CSV)

    with open(os.path.join(PROCESSED_DIR, "users.json"), encoding="utf-8") as f:
        usable_users = [u["user_id"] for u in json.load(f)["users"]]
    logger.info(f"Users: {len(usable_users)}")

    # Policy factories — called fresh per user
    POLICY_FACTORIES = [
        lambda: GraphRL_EdgeScorer(),
        lambda: GraphRL_EdgeScorer_5pass(),
        lambda: GraphRL_EdgeScorer_OnlineRL(),
        lambda: GraphRL_CtxOnly(),
        lambda: GraphRL_EdgeOnly(),
        lambda: RL_LatencyFocus(),
    ]

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

    all_rows: List[dict] = []

    for uid in usable_users:
        if uid not in user_cache:
            continue
        apps, tbs, wds = user_cache[uid]
        n  = len(apps)
        te = int(n * TRAIN_RATIO)
        ve = int(n * (TRAIN_RATIO + VAL_RATIO))

        tr_a, va_a, ts_a = apps[:te], apps[te:ve], apps[ve:]
        tr_t, va_t, ts_t = tbs[:te],  tbs[te:ve],  tbs[ve:]
        tr_w, va_w, ts_w = wds[:te],  wds[te:ve],  wds[ve:]

        if len(ts_a) < 10:
            continue

        for factory in POLICY_FACTORIES:
            policy = factory()
            try:
                metrics = evaluate_policy(
                    policy,
                    tr_a, va_a, ts_a,
                    tr_t, va_t, ts_t,
                    tr_w, va_w, ts_w,
                    lat, uid,
                )
                row = {"user_id": uid, "policy": policy.name}
                row.update(metrics)
                all_rows.append(row)
                logger.info(
                    f"  {uid:8s} {policy.name:35s}: "
                    f"HR={metrics['hit_rate']:.3f}  "
                    f"F1={metrics['f1']:.4f}  "
                    f"Lat={metrics['latency_saved_ms']:.0f}ms"
                )
            except Exception as exc:
                logger.error(f"  {uid}/{policy.name}: {exc}", exc_info=True)

    if not all_rows:
        logger.error("No results produced.")
        return

    # ── Write per-user CSV ────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "v5_rl_graph.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    logger.info(f"\nWritten: {out_path} ({len(all_rows)} rows)")

    # ── Aggregate + significance test ─────────────────────────────────────────
    from collections import defaultdict as dd
    by_policy: Dict[str, List[dict]] = dd(list)
    for r in all_rows:
        by_policy[r["policy"]].append(r)

    # Load V4 baseline per-user F1 for paired t-test
    baseline_f1_per_user: Dict[str, float] = {}
    v4_path = os.path.join(RESULTS_DIR, "benchmark_results_v4.csv")
    if os.path.exists(v4_path):
        with open(v4_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["policy"] == "GraphMindRL":
                    baseline_f1_per_user[r["user_id"]] = float(r["f1"])

    summary_rows = []
    for pol_name in sorted(by_policy.keys()):
        rows = by_policy[pol_name]
        f1s  = np.array([float(r["f1"])               for r in rows])
        hrs  = np.array([float(r["hit_rate"])          for r in rows])
        lats = np.array([float(r["latency_saved_ms"])  for r in rows])
        precs = np.array([float(r["precision"])        for r in rows])
        recs  = np.array([float(r["recall"])           for r in rows])

        mean_f1 = float(f1s.mean())
        delta   = mean_f1 - BASELINE_F1

        # Paired t-test vs GraphMindRL baseline
        common = [u for u in [r["user_id"] for r in rows] if u in baseline_f1_per_user]
        p_val  = float("nan")
        t_stat = float("nan")
        cohen_d = float("nan")
        if len(common) >= 5:
            exp_f1  = np.array([float(r["f1"]) for r in rows
                                 if r["user_id"] in baseline_f1_per_user])
            base_f1 = np.array([baseline_f1_per_user[r["user_id"]] for r in rows
                                  if r["user_id"] in baseline_f1_per_user])
            if len(exp_f1) >= 5:
                t_stat, p_val = stats.ttest_rel(exp_f1, base_f1)
                diff = exp_f1 - base_f1
                cohen_d = diff.mean() / (diff.std() + 1e-9)

        sig = p_val < 0.05 if not math.isnan(p_val) else False
        summary_rows.append({
            "policy":       pol_name,
            "n_users":      len(rows),
            "mean_f1":      round(mean_f1, 4),
            "std_f1":       round(float(f1s.std()), 4),
            "mean_hr":      round(float(hrs.mean()), 4),
            "mean_prec":    round(float(precs.mean()), 4),
            "mean_rec":     round(float(recs.mean()), 4),
            "mean_lat_ms":  round(float(lats.mean()), 1),
            "delta_f1":     round(delta, 4),
            "t_stat":       round(float(t_stat), 3) if not math.isnan(t_stat) else "—",
            "p_value":      round(float(p_val), 4)  if not math.isnan(p_val)  else "—",
            "cohen_d":      round(float(cohen_d), 3) if not math.isnan(cohen_d) else "—",
            "significant":  sig,
            "meets_02":     delta >= 0.02,
        })

    summary_rows.sort(key=lambda x: -x["mean_f1"])

    sum_path = os.path.join(RESULTS_DIR, "v5_rl_graph_summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    logger.info(f"Written: {sum_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"RL-ON-GRAPH RESULTS  (baseline GraphMindRL F1={BASELINE_F1})")
    logger.info("=" * 80)
    logger.info(f"{'Policy':35s} {'F1':>7} {'ΔF1':>7} {'p':>8} {'d':>7} {'Sig':>5} {'≥.02':>5}")
    logger.info("-" * 80)
    for r in summary_rows:
        p_str = f"{r['p_value']:.4f}" if isinstance(r['p_value'], float) else r['p_value']
        d_str = f"{r['cohen_d']:.3f}" if isinstance(r['cohen_d'], float) else r['cohen_d']
        sig_m = "✅" if r["significant"] else "❌"
        thr_m = "✅" if r["meets_02"]   else "❌"
        logger.info(
            f"  {r['policy']:35s} {r['mean_f1']:7.4f} {r['delta_f1']:+7.4f} "
            f"{p_str:>8} {d_str:>7}  {sig_m}  {thr_m}"
        )

    # ── Write markdown report ─────────────────────────────────────────────────
    _write_report(summary_rows)


def _write_report(summary_rows: List[dict]):
    lines = [
        "# RL-on-Graph Results",
        "",
        f"**Baseline:** GraphMindRL F1={BASELINE_F1}  (same 31 users, 80/10/10 split)",
        "**New policies:** GraphRL_EdgeScorer variants + RL_LatencyFocus reference",
        "**Significance:** paired t-test vs baseline per user",
        "",
        "---",
        "",
        "## Results",
        "",
        "| Policy | F1 | ΔF1 | HR | Prec | Rec | Lat ms | p | d | Sig | ≥+0.02 |",
        "|--------|-----|-----|-----|------|-----|--------|---|---|-----|--------|",
    ]
    for r in summary_rows:
        p_s = f"{r['p_value']:.4f}" if isinstance(r['p_value'], float) else str(r['p_value'])
        d_s = f"{r['cohen_d']:.3f}" if isinstance(r['cohen_d'], float) else str(r['cohen_d'])
        sig = "✅" if r["significant"] else "❌"
        thr = "✅" if r["meets_02"]   else "❌"
        lines.append(
            f"| {r['policy']} | {r['mean_f1']:.4f} | {r['delta_f1']:+.4f} "
            f"| {r['mean_hr']:.4f} | {r['mean_prec']:.4f} | {r['mean_rec']:.4f} "
            f"| {r['mean_lat_ms']:.1f} | {p_s} | {d_s} | {sig} | {thr} |"
        )
    lines += [
        "",
        "---",
        "",
        "## Edge Scorer Interpretation",
        "",
        "The RL agent learns a 6-dim weight vector **W** scoring each graph candidate:",
        "```",
        "score(dst|src) = W[0]×trans_prob + W[1]×log_count_norm",
        "              + W[2]×out_degree_norm + W[3]×in_degree_norm",
        "              + W[4]×recency_norm + W[5]×frequency_norm",
        "```",
        "Selection: candidates with score ≥ threshold_raw are prefetched (up to budget).",
        "",
        "The context controller **W_ctx** (8×2) adapts threshold and budget per step",
        "based on graph branching factor, score peakedness, recent precision/FP-rate,",
        "time of day, and weekday.",
        "",
    ]
    path = os.path.join(REPORTS_DIR, "rl_graph_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Written: {path}")


if __name__ == "__main__":
    main()
