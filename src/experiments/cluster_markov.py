"""
src/models/cluster_markov.py

Cluster-Level Markov predictor.

Clusters users based on behavioural features:
  - Transition entropy (how predictable the user's transitions are)
  - Active hour distribution (morning/afternoon/evening/night ratio)
  - Top app concentration (Herfindahl index of top-5 app frequencies)
  - Transition density (edges / vocab^2)

Uses 3-5 clusters (selected by within-cluster sum of squares elbow, pure numpy).

Prediction chain per user:
  Personal Markov-2 → Cluster Markov-2 → Global Markov-2

This provides graceful degradation for sparse sequences.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


class ClusterMarkov:
    """
    Cluster-Aware Markov predictor.

    Training:
      1. Fit user-level behavioural feature vectors.
      2. Cluster users into N groups (k-means, pure numpy).
      3. Train per-cluster Markov-2 on pooled cluster members' data.
      4. Also train personal Markov-2 per user.

    Prediction (for a given user):
      1. Try personal Markov-2.
      2. Fallback to cluster Markov-2.
      3. Final fallback to global Markov-2.
    """

    def __init__(self, n_clusters: int = 4, top_k: int = 5, rng_seed: int = 42) -> None:
        self.n_clusters = n_clusters
        self.top_k = top_k
        self._rng = np.random.default_rng(rng_seed)

        # Per-user personal Markov-2
        self._personal_m2: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {}
        self._personal_m1: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Cluster Markov-2: cluster_id → bigram → {next: prob}
        self._cluster_m2: Dict[int, Dict[Tuple[str, str], Dict[str, float]]] = {}
        self._cluster_m1: Dict[int, Dict[str, Dict[str, float]]] = {}

        # Global Markov-2 fallback
        self._global_m2: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._global_m1: Dict[str, Dict[str, float]] = {}

        # User → cluster assignment
        self._user_cluster: Dict[str, int] = {}

        # Feature normalisation params
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std:  Optional[np.ndarray] = None

        # Cluster centroids (in normalised feature space)
        self._centroids: Optional[np.ndarray] = None

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        user_sequences: Dict[str, List[str]],
        user_time_buckets: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        """
        Fit the cluster model across all users.

        Args:
            user_sequences:   {user_id: [app, app, ...]} (train splits only)
            user_time_buckets:{user_id: [bucket, ...]}   (optional, same length as sequence)
        """
        if not user_sequences:
            return

        user_ids = list(user_sequences.keys())

        # 1. Extract features per user
        features = []
        for uid in user_ids:
            seq  = user_sequences[uid]
            tbs  = user_time_buckets.get(uid, []) if user_time_buckets else []
            feat = self._user_features(seq, tbs)
            features.append(feat)
        X = np.array(features, dtype=np.float32)

        # 2. Normalise features
        self._feat_mean = X.mean(axis=0)
        self._feat_std  = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        X_norm = (X - self._feat_mean) / self._feat_std

        # 3. K-means clustering (pure numpy)
        n_clust = min(self.n_clusters, len(user_ids))
        labels, centroids = self._kmeans(X_norm, n_clust)
        self._centroids = centroids

        for uid, label in zip(user_ids, labels):
            self._user_cluster[uid] = int(label)

        # 4. Pool sequences per cluster → train cluster Markov-2
        cluster_seqs: Dict[int, List[str]] = defaultdict(list)
        global_seq: List[str] = []
        for uid in user_ids:
            c = self._user_cluster[uid]
            cluster_seqs[c].extend(user_sequences[uid])
            global_seq.extend(user_sequences[uid])

        for cid, cseq in cluster_seqs.items():
            m2, m1 = self._build_markov2(cseq)
            self._cluster_m2[cid] = m2
            self._cluster_m1[cid] = m1

        gm2, gm1 = self._build_markov2(global_seq)
        self._global_m2 = gm2
        self._global_m1 = gm1

        # 5. Train personal Markov-2 per user
        for uid in user_ids:
            m2, m1 = self._build_markov2(user_sequences[uid])
            self._personal_m2[uid] = m2
            self._personal_m1[uid] = m1

    def train_user(self, user_id: str, events: List[str]) -> None:
        """
        Update personal Markov-2 for a single user (used in benchmark loop).
        Cluster assignment is inferred from feature similarity to centroids.
        """
        m2, m1 = self._build_markov2(events)
        self._personal_m2[user_id] = m2
        self._personal_m1[user_id] = m1

        if self._centroids is not None and self._feat_mean is not None:
            feat = self._user_features(events, [])
            feat_norm = (np.array(feat) - self._feat_mean) / self._feat_std
            dists = np.linalg.norm(self._centroids - feat_norm, axis=1)
            self._user_cluster[user_id] = int(np.argmin(dists))

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(
        self,
        user_id: str,
        current: str,
        prev: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Predict next apps using the Personal → Cluster → Global chain.

        Args:
            user_id: Target user.
            current: Current app.
            prev:    Previous app (enables second-order).
            top_k:   Max candidates returned.

        Returns:
            List of (app, probability) sorted descending.
        """
        k = top_k or self.top_k
        results: Dict[str, float] = {}

        # --- Personal Markov-2 ---
        personal_found = False
        if user_id in self._personal_m2 and prev is not None:
            bg = (prev, current)
            if bg in self._personal_m2[user_id]:
                for app, p in self._personal_m2[user_id][bg].items():
                    if app != current:
                        results[app] = max(results.get(app, 0), 0.8 * p)
                personal_found = True
        if not personal_found and user_id in self._personal_m1:
            if current in self._personal_m1[user_id]:
                for app, p in self._personal_m1[user_id][current].items():
                    if app != current:
                        results[app] = max(results.get(app, 0), 0.6 * p)
                personal_found = True

        # --- Cluster Markov-2 (fill gaps) ---
        cid = self._user_cluster.get(user_id)
        if cid is not None and (not personal_found or len(results) < k):
            if prev is not None and cid in self._cluster_m2:
                bg = (prev, current)
                if bg in self._cluster_m2[cid]:
                    for app, p in self._cluster_m2[cid][bg].items():
                        if app != current and app not in results:
                            results[app] = 0.4 * p
            if cid in self._cluster_m1 and current in self._cluster_m1[cid]:
                for app, p in self._cluster_m1[cid][current].items():
                    if app != current and app not in results:
                        results[app] = 0.3 * p

        # --- Global Markov-2 (final fallback) ---
        if len(results) < k:
            if prev is not None:
                bg = (prev, current)
                if bg in self._global_m2:
                    for app, p in self._global_m2[bg].items():
                        if app != current and app not in results:
                            results[app] = 0.2 * p
            if current in self._global_m1:
                for app, p in self._global_m1[current].items():
                    if app != current and app not in results:
                        results[app] = 0.1 * p

        return sorted(results.items(), key=lambda x: -x[1])[:k]

    def predict_apps(
        self,
        user_id: str,
        current: str,
        prev: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[str]:
        return [app for app, _ in self.predict(user_id, current, prev, top_k)]

    def get_cluster_assignment(self, user_id: str) -> Optional[int]:
        return self._user_cluster.get(user_id)

    def get_cluster_sizes(self) -> Dict[int, int]:
        sizes: Dict[int, int] = defaultdict(int)
        for cid in self._user_cluster.values():
            sizes[cid] += 1
        return dict(sizes)

    # ── Private: Feature extraction ────────────────────────────────────────

    def _user_features(self, seq: List[str], time_buckets: List[int]) -> List[float]:
        """Extract 4 behavioural features from an app sequence."""
        if not seq:
            return [0.0, 0.0, 0.0, 0.0]

        # Feature 1: Transition entropy (predictability)
        from collections import Counter
        counts: Dict[str, Counter] = defaultdict(Counter)
        for i in range(1, len(seq)):
            counts[seq[i-1]][seq[i]] += 1
        entropies = []
        for src, dst_counts in counts.items():
            total = sum(dst_counts.values())
            H = -sum((c/total) * math.log2(c/total + 1e-9) for c in dst_counts.values())
            entropies.append(H)
        entropy = float(np.mean(entropies)) if entropies else 0.0

        # Feature 2: Active hour concentration (0 = uniform, 1 = concentrated)
        if time_buckets and len(time_buckets) == len(seq):
            hour_counts = np.zeros(4)  # 4 quarters of day
            for tb in time_buckets:
                hour_counts[min(3, tb // 12)] += 1
            total = hour_counts.sum() or 1
            hour_counts /= total
            # Gini coefficient
            hour_gini = float(np.std(hour_counts) / (np.mean(hour_counts) + 1e-9))
        else:
            hour_gini = 0.5

        # Feature 3: Top-5 app concentration (Herfindahl index)
        app_counts = Counter(seq)
        total_apps = len(seq)
        top5 = [c for _, c in app_counts.most_common(5)]
        herfindahl = sum((c / total_apps) ** 2 for c in top5)

        # Feature 4: Transition density
        vocab = set(seq)
        n_vocab = len(vocab)
        n_edges = sum(len(v) for v in counts.values())
        density = n_edges / (n_vocab * (n_vocab - 1) + 1)

        return [entropy, hour_gini, herfindahl, density]

    # ── Private: K-means (pure numpy) ─────────────────────────────────────

    def _kmeans(
        self,
        X: np.ndarray,
        k: int,
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vanilla k-means with k-means++ initialisation."""
        n = len(X)
        if k >= n:
            return np.arange(n), X.copy()

        # k-means++ init
        centroids = [X[self._rng.integers(n)].copy()]
        for _ in range(k - 1):
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            probs = dists / (dists.sum() + 1e-12)
            idx = self._rng.choice(n, p=probs)
            centroids.append(X[idx].copy())
        centroids = np.array(centroids)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.array([
                X[new_labels == j].mean(axis=0) if (new_labels == j).any() else centroids[j]
                for j in range(k)
            ])

            if np.allclose(centroids, new_centroids, atol=tol):
                labels = new_labels
                centroids = new_centroids
                break

            labels = new_labels
            centroids = new_centroids

        return labels, centroids

    # ── Private: Markov builder ────────────────────────────────────────────

    @staticmethod
    def _build_markov2(
        seq: List[str],
    ) -> Tuple[Dict[Tuple[str, str], Dict[str, float]],
               Dict[str, Dict[str, float]]]:
        """Build normalised Markov-2 and Markov-1 from a sequence."""
        c1: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        c2: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for i in range(1, len(seq)):
            c1[seq[i-1]][seq[i]] += 1
        for i in range(2, len(seq)):
            c2[(seq[i-2], seq[i-1])][seq[i]] += 1

        m1 = {s: {d: n/sum(dc.values()) for d, n in dc.items()}
              for s, dc in c1.items()}
        m2 = {bg: {d: n/sum(dc.values()) for d, n in dc.items()}
              for bg, dc in c2.items()}

        # Sort each dict descending by probability
        m1 = {s: dict(sorted(d.items(), key=lambda x: -x[1])) for s, d in m1.items()}
        m2 = {bg: dict(sorted(d.items(), key=lambda x: -x[1])) for bg, d in m2.items()}

        return m2, m1

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ClusterMarkov("
            f"n_clusters={self.n_clusters}, "
            f"n_users={len(self._personal_m2)}, "
            f"cluster_sizes={self.get_cluster_sizes()})"
        )
