"""
src/rl/adaptive_ensemble_env.py

GraphMind RL Redesign — Adaptive Ensemble Controller.

MOTIVATION
----------
The previous RL design (ResourceAllocationPolicy) acted as a cache allocator:
it adjusted HOT/WARM tier budgets but did NOT control which prediction model
to trust. This meant the RL contribution was marginal.

The new design promotes RL to be the PRIMARY decision layer:

  Graph, VOM, ContextMarkov → produce candidate lists + confidence scores
  RL agent → learns WHICH predictor to trust in which context (adaptive weights)

ARCHITECTURE
------------
State (9 + n_predictors dimensions):
  [0]   current_app_hash  (normalised vocabulary index)
  [1]   prev_app_hash
  [2]   time_bucket_norm  (0-47 → 0.0-1.0)
  [3]   weekday_norm      (0-6  → 0.0-1.0)
  [4]   transition_entropy (running estimate, normalised)
  [5]   confidence_m1     (Markov-1 top-1 confidence)
  [6]   confidence_m2     (Markov-2 top-1 confidence)
  [7]   confidence_vom    (VOM top-1 confidence)
  [8]   confidence_ctx    (ContextMarkov top-1 confidence)
  [9]   confidence_graph  (Graph transition probability)
  [10-14] recent hit history (last 5 steps, binary)

Total: 15 dimensions

Action (5 continuous weights, softmax-normalised):
  [w_m1, w_m2, w_vom, w_ctx, w_graph]

Each predictor's candidate list is scored. Candidates are merged weighted
by these weights. Top-k from merged ranking = prefetch targets.

RL ALGORITHM
------------
Policy-gradient (REINFORCE-style) with:
  - Softmax weight output (Dirichlet-like)
  - Reward: +1.0 hit, 0.0 miss, -0.05 × weight_entropy_penalty
  - Baseline: exponential moving average of returns
  - Parameter update: gradient ascent on log(π(a|s)) × advantage

This is genuine RL (policy gradient), kept lightweight for CPU training.
Training loop is external (train on train split, evaluate on test split).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Softmax helper ───────────────────────────────────────────────────────

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) / max(temperature, 1e-8)
    x -= x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


# ── Predictor interface (lightweight wrappers) ───────────────────────────

class _Markov1:
    """Inline Markov-1 predictor."""
    def __init__(self):
        self._m: Dict[str, Dict[str, float]] = {}
    def train(self, seq: List[str]):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(seq)):
            c[seq[i-1]][seq[i]] += 1
        for s, d in c.items():
            t = sum(d.values())
            self._m[s] = dict(sorted({k: v/t for k, v in d.items()}.items(), key=lambda x: -x[1]))
    def top_k(self, cur: str, k: int) -> List[Tuple[str, float]]:
        return list(self._m.get(cur, {}).items())[:k]
    def confidence(self, cur: str) -> float:
        items = list(self._m.get(cur, {}).items())
        return items[0][1] if items else 0.0


class _Markov2:
    """Inline Markov-2 predictor with M1 fallback."""
    def __init__(self):
        self._m1: Dict[str, Dict[str, float]] = {}
        self._m2: Dict[Tuple[str,str], Dict[str, float]] = {}
    def train(self, seq: List[str]):
        c1 = defaultdict(lambda: defaultdict(int))
        c2 = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(seq)):
            c1[seq[i-1]][seq[i]] += 1
        for i in range(2, len(seq)):
            c2[(seq[i-2], seq[i-1])][seq[i]] += 1
        for s, d in c1.items():
            t = sum(d.values())
            self._m1[s] = dict(sorted({k: v/t for k, v in d.items()}.items(), key=lambda x: -x[1]))
        for bg, d in c2.items():
            t = sum(d.values())
            self._m2[bg] = dict(sorted({k: v/t for k, v in d.items()}.items(), key=lambda x: -x[1]))
    def top_k(self, cur: str, prev: Optional[str], k: int) -> List[Tuple[str, float]]:
        if prev:
            bg = (prev, cur)
            if bg in self._m2:
                return list(self._m2[bg].items())[:k]
        return list(self._m1.get(cur, {}).items())[:k]
    def confidence(self, cur: str, prev: Optional[str]) -> float:
        items = self.top_k(cur, prev, 1)
        return items[0][1] if items else 0.0


class _GraphPredictor:
    """Graph-based predictor (same as Markov-1 structurally but named separately)."""
    def __init__(self):
        self._g: Dict[str, Dict[str, float]] = {}
    def train(self, seq: List[str]):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(seq)):
            c[seq[i-1]][seq[i]] += 1
        for s, d in c.items():
            t = sum(d.values())
            self._g[s] = dict(sorted({k: v/t for k, v in d.items()}.items(), key=lambda x: -x[1]))
    def top_k(self, cur: str, k: int) -> List[Tuple[str, float]]:
        return list(self._g.get(cur, {}).items())[:k]
    def confidence(self, cur: str) -> float:
        items = list(self._g.get(cur, {}).items())
        return items[0][1] if items else 0.0


# ── Adaptive Ensemble Controller ──────────────────────────────────────────

class AdaptiveEnsembleController:
    """
    RL-based adaptive predictor weighting.

    The agent learns per-context weights over 5 predictors:
      [Markov-1, Markov-2, VariableOrderMarkov, ContextMarkov, Graph]

    Uses REINFORCE with baseline (policy gradient), keeping only linear
    feature→weight mapping for interpretability and speed.

    This replaces the previous cache-allocator RL role with a genuine
    PREDICTION CONTROLLER role.
    """

    N_PREDICTORS = 5
    OBS_DIM      = 15   # see module docstring
    HIT_HIST_LEN = 5
    TOP_K        = 5

    def __init__(
        self,
        lr: float = 0.05,
        baseline_decay: float = 0.95,
        entropy_penalty: float = 0.02,
        temperature: float = 1.0,
        rng_seed: int = 42,
    ) -> None:
        self.lr              = lr
        self.baseline_decay  = baseline_decay
        self.entropy_penalty = entropy_penalty
        self.temperature     = temperature

        self._rng = np.random.default_rng(rng_seed)

        # Policy: linear map from OBS_DIM → N_PREDICTORS logits
        self._W = np.zeros((self.OBS_DIM, self.N_PREDICTORS), dtype=np.float64)
        self._b = np.zeros(self.N_PREDICTORS, dtype=np.float64)

        # Running baseline for advantage
        self._baseline: float = 0.0

        # Online state
        self._m1  = _Markov1()
        self._m2  = _Markov2()
        self._graph = _GraphPredictor()

        # VOM and ContextMarkov are optional external models
        self._vom = None    # set via set_vom_model()
        self._ctx = None    # set via set_ctx_model()

        # Episodic state
        self._prev_app: Optional[str] = None
        self._hit_history: deque = deque([0.0]*self.HIT_HIST_LEN, maxlen=self.HIT_HIST_LEN)
        self._transition_entropy: float = 0.5
        self._entropy_window: deque = deque(maxlen=50)

        # Training episode buffer
        self._trajectory: List[dict] = []
        self._app_vocab: Dict[str, int] = {}

    # ── Model setters ────────────────────────────────────────────────────

    def set_vom_model(self, vom) -> None:
        """Inject a trained VariableOrderMarkov model."""
        self._vom = vom

    def set_ctx_model(self, ctx) -> None:
        """Inject a trained ContextMarkov model."""
        self._ctx = ctx

    # ── Training ─────────────────────────────────────────────────────────

    def train(
        self,
        events: List[str],
        time_buckets: Optional[List[int]] = None,
        weekdays: Optional[List[int]] = None,
        n_passes: int = 3,
    ) -> Dict[str, float]:
        """
        Train all predictors and learn ensemble weights via REINFORCE.

        Args:
            events:       Training app sequence.
            time_buckets: 0-47 time bucket per event.
            weekdays:     0-6 weekday per event.
            n_passes:     Number of passes through the training data.

        Returns:
            Training stats dict.
        """
        tbs = time_buckets or [0] * len(events)
        wds = weekdays or [0] * len(events)

        # Build vocabulary
        for app in events:
            if app not in self._app_vocab and len(self._app_vocab) < 200:
                self._app_vocab[app] = len(self._app_vocab)

        # Train base predictors
        self._m1.train(events)
        self._m2.train(events)
        self._graph.train(events)

        # Train VOM if available
        if self._vom is not None:
            self._vom.train(events)

        # Train ContextMarkov if available
        if self._ctx is not None:
            self._ctx.train(events, tbs, wds)

        # REINFORCE training passes
        total_hits = 0
        total_steps = 0

        for _pass in range(n_passes):
            self._trajectory.clear()
            self._reset_episode_state()

            for i in range(len(events) - 1):
                cur = events[i]
                nxt = events[i + 1]
                tb  = tbs[i]
                wd  = wds[i]

                obs      = self._build_obs(cur, tb, wd)
                logits   = obs @ self._W + self._b
                weights  = _softmax(logits, self.temperature)

                # Get merged predictions
                preds    = self._merge_predictions(cur, weights, self._prev_app, tb, wd)
                is_hit   = nxt in preds

                # Reward
                w_entropy = float(-np.sum(weights * np.log(weights + 1e-12)))
                reward    = (1.0 if is_hit else 0.0) - self.entropy_penalty * w_entropy

                self._trajectory.append({
                    "obs":     obs,
                    "weights": weights,
                    "reward":  reward,
                    "is_hit":  is_hit,
                })

                # Update online state
                self._hit_history.append(1.0 if is_hit else 0.0)
                self._update_entropy_estimate(cur, nxt)
                self._prev_app = cur
                total_hits += int(is_hit)
                total_steps += 1

            # Policy gradient update (REINFORCE with baseline)
            self._update_policy()

        hit_rate = total_hits / max(1, total_steps)
        return {
            "hit_rate": round(hit_rate, 4),
            "n_steps":  total_steps,
            "n_passes": n_passes,
            "baseline": round(self._baseline, 4),
        }

    def _update_policy(self) -> None:
        """REINFORCE gradient update on stored trajectory."""
        if not self._trajectory:
            return

        # Compute discounted returns (no discount — episodic)
        returns = [t["reward"] for t in self._trajectory]

        for i, step in enumerate(self._trajectory):
            obs     = step["obs"]
            weights = step["weights"]
            G       = returns[i]

            # Advantage
            advantage = G - self._baseline
            self._baseline = (
                self.baseline_decay * self._baseline
                + (1 - self.baseline_decay) * G
            )

            # Policy gradient: ∇log π(a|s) = ∇log softmax
            # For softmax: ∂log(w_j)/∂logit_k = δ_jk - w_k
            # We use the max-weight action
            action_idx = int(np.argmax(weights))
            grad_logits = -weights.copy()
            grad_logits[action_idx] += 1.0  # ∂log π / ∂logit

            # Chain rule through linear layer
            grad_W = np.outer(obs, grad_logits)
            grad_b = grad_logits

            self._W += self.lr * advantage * grad_W
            self._b += self.lr * advantage * grad_b

    # ── Inference ────────────────────────────────────────────────────────

    def predict(
        self,
        current: str,
        time_bucket: int = 0,
        weekday: int = 0,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Predict top-k next apps using the learned ensemble weights.

        Args:
            current:     Current app.
            time_bucket: 0-47 time bucket.
            weekday:     0-6 weekday.
            top_k:       Override instance top_k.

        Returns:
            List of predicted app names (no scores).
        """
        k = top_k or self.TOP_K
        obs     = self._build_obs(current, time_bucket, weekday)
        logits  = obs @ self._W + self._b
        weights = _softmax(logits, self.temperature)
        return self._merge_predictions(current, weights, self._prev_app, time_bucket, weekday, k)

    def update_state(self, app: str, hit: bool) -> None:
        """Update rolling state after observing a launch event."""
        self._hit_history.append(1.0 if hit else 0.0)
        if self._prev_app is not None:
            self._update_entropy_estimate(self._prev_app, app)
        self._prev_app = app

    def reset(self) -> None:
        self._reset_episode_state()

    def get_weights(self) -> Dict[str, float]:
        """Return current learned predictor weights (dummy forward pass)."""
        obs     = np.zeros(self.OBS_DIM, dtype=np.float64)
        logits  = obs @ self._W + self._b
        weights = _softmax(logits, self.temperature)
        names   = ["M1", "M2", "VOM", "ContextMarkov", "Graph"]
        return {n: round(float(w), 4) for n, w in zip(names, weights)}

    # ── Private helpers ───────────────────────────────────────────────────

    def _reset_episode_state(self) -> None:
        self._prev_app = None
        self._hit_history = deque([0.0]*self.HIT_HIST_LEN, maxlen=self.HIT_HIST_LEN)
        self._transition_entropy = 0.5
        self._entropy_window.clear()

    def _build_obs(self, current: str, time_bucket: int, weekday: int) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float64)

        # App hashes
        vocab_size = max(len(self._app_vocab), 1)
        if current in self._app_vocab:
            obs[0] = self._app_vocab[current] / vocab_size
        if self._prev_app and self._prev_app in self._app_vocab:
            obs[1] = self._app_vocab[self._prev_app] / vocab_size

        obs[2] = time_bucket / 47.0
        obs[3] = weekday / 6.0
        obs[4] = min(1.0, self._transition_entropy / 4.0)  # normalise to [0,1]

        # Predictor confidences
        obs[5] = self._m1.confidence(current)
        obs[6] = self._m2.confidence(current, self._prev_app)
        obs[7] = (self._vom.confidence(current, self._prev_app) if self._vom else 0.0)
        obs[8] = (self._ctx.confidence(current, time_bucket, weekday) if self._ctx else 0.0)
        obs[9] = self._graph.confidence(current)

        # Hit history
        for j, h in enumerate(self._hit_history):
            obs[10 + j] = float(h)

        return obs

    def _merge_predictions(
        self,
        current: str,
        weights: np.ndarray,
        prev: Optional[str],
        time_bucket: int,
        weekday: int,
        k: Optional[int] = None,
    ) -> List[str]:
        """Merge candidate lists from all predictors using learned weights."""
        top_k = k or self.TOP_K
        scores: Dict[str, float] = defaultdict(float)

        # Markov-1
        w0 = float(weights[0])
        for app, p in self._m1.top_k(current, top_k * 2):
            if app != current:
                scores[app] += w0 * p

        # Markov-2
        w1 = float(weights[1])
        for app, p in self._m2.top_k(current, prev, top_k * 2):
            if app != current:
                scores[app] += w1 * p

        # VOM
        w2 = float(weights[2])
        if self._vom is not None:
            for app, p in self._vom.predict(current, prev, top_k=top_k * 2):
                if app != current:
                    scores[app] += w2 * p

        # ContextMarkov
        w3 = float(weights[3])
        if self._ctx is not None:
            for app, p in self._ctx.predict(current, time_bucket, weekday, top_k=top_k * 2):
                if app != current:
                    scores[app] += w3 * p

        # Graph
        w4 = float(weights[4])
        for app, p in self._graph.top_k(current, top_k * 2):
            if app != current:
                scores[app] += w4 * p

        if not scores:
            return []

        ranked = sorted(scores.keys(), key=lambda a: -scores[a])
        return ranked[:top_k]

    def _update_entropy_estimate(self, prev: str, current: str) -> None:
        """Update running transition entropy estimate."""
        self._entropy_window.append((prev, current))
        if len(self._entropy_window) < 5:
            return
        from collections import Counter
        src_counts: Dict[str, Counter] = defaultdict(Counter)
        for a, b in self._entropy_window:
            src_counts[a][b] += 1
        entropies = []
        for src, dst in src_counts.items():
            total = sum(dst.values())
            H = -sum((c/total) * math.log2(c/total + 1e-9) for c in dst.values())
            entropies.append(H)
        if entropies:
            self._transition_entropy = float(np.mean(entropies))
