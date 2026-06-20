"""
src/models/v6_pipeline.py

GraphMind V6 Pipeline -- V5 + EmbeddingTransformerReranker + FiveTierCache.

V6 is built on top of V5 (unchanged) with two additive layers:
  1. FiveTierCache: PIN/HOT/WARM/COOL/COLD hierarchy
  2. EmbeddingTransformerReranker: per-user reranker that reranks V5 top-K
     candidates to improve Hit@1 — trained only on each user's own events.

Per-user reranker strategy:
    For multi-user datasets (e.g., UbiqLog with 31 users), one small reranker
    model is trained per user.  This avoids gradient conflicts between users
    with very different app usage patterns, enabling fast convergence (5-10
    epochs per user on ~3,000-8,000 samples per user vs a single bloated
    model on 258,000 mixed samples).

    For single-user datasets (synthetic), a single reranker is trained on all
    events (legacy behaviour preserved).
"""

import logging
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from config import settings
from src.benchmarks.baselines_v2 import GraphMindRLPolicy
from src.core.five_tier_cache import FiveTierCache
from src.models.transformer_reranker import (
    # v2 embedding-based (used for real multi-user datasets)
    EmbeddingRerankerTrainer,
    build_candidate_indices,
    # v1 legacy (used for synthetic single-user dataset)
    RerankerTrainer,
    build_candidate_tensor,
)
from src.benchmarks.graphmind_policy_runner import GraphMindPolicyRunner
from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED

logger = logging.getLogger(__name__)

_SAVED_MODELS_DIR = os.path.join(settings.PROJECT_ROOT, "models", "saved")
_FORCE_RETRAIN = False

# Maximum samples collected per user for reranker training.
# Keeps per-user training fast while still capturing all pattern diversity.
_MAX_SAMPLES_PER_USER = 6_000


def set_force_retrain(val: bool) -> None:
    global _FORCE_RETRAIN
    _FORCE_RETRAIN = val


def _dataset_tag(n_events: int) -> str:
    if n_events > 400_000:
        return "ubiqlog"
    if n_events < 50_000:
        return "synthetic"
    return f"custom_{n_events}"


def _try_tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


# ---------------------------------------------------------------------------
# V6 Runner -- used by run_full_evaluation
# ---------------------------------------------------------------------------

class GraphMindV6PolicyRunner(GraphMindPolicyRunner):
    """
    V6 runner that uses FiveTierCache for memory management and either:
      - A dict of per-user EmbeddingRerankerTrainer (multi-user datasets), or
      - A single RerankerTrainer (synthetic single-user dataset)
    to re-order the prefetched apps.
    """

    def __init__(
        self,
        user_id: str,
        top_k: int = 8,
        # v2 per-user rerankers (multi-user)
        per_user_rerankers: Optional[Dict[str, EmbeddingRerankerTrainer]] = None,
        app_to_idx: Optional[Dict[str, int]] = None,
        # v1 single reranker (synthetic)
        reranker=None,
        reranker_ready: bool = False,
        app_vocab: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(user_id, top_k=top_k)
        self.cache_v6 = FiveTierCache(user_id=user_id)

        # v2
        self.per_user_rerankers: Dict[str, EmbeddingRerankerTrainer] = per_user_rerankers or {}
        self.app_to_idx: Dict[str, int] = app_to_idx or {}

        # v1 fallback
        self.reranker = reranker
        self.reranker_ready = reranker_ready
        self.app_vocab = app_vocab or []

        self.device = device

    def _rerank_candidates(
        self,
        predicted_apps: List[str],
        event: dict,
    ) -> List[str]:
        """Rerank predicted_apps using the appropriate reranker for this event's user."""
        if not predicted_apps:
            return predicted_apps

        time_norm = event.get("time_bucket", 0) / 47.0
        confidences = [1.0 / (i + 1) for i in range(len(predicted_apps))]

        # --- v2: per-user embedding reranker ---
        uid = event.get("user_id", "default")
        if self.per_user_rerankers and uid in self.per_user_rerankers:
            try:
                return self.per_user_rerankers[uid].rerank(
                    candidates=predicted_apps,
                    app_to_idx=self.app_to_idx,
                    confidences=confidences,
                    time_norm=time_norm,
                )
            except Exception as e:
                logger.debug(f"V6 per-user reranker failed for {uid}: {e}")
                return predicted_apps

        # --- v1: single legacy reranker fallback ---
        if self.reranker_ready and self.reranker and self.app_vocab:
            try:
                tensor = build_candidate_tensor(
                    candidates=predicted_apps,
                    confidences=confidences,
                    time_norm=time_norm,
                    app_vocab=self.app_vocab,
                    top_k=self.top_k,
                ).unsqueeze(0).to(self.device).float()

                self.reranker.model.eval()
                with torch.no_grad():
                    scores = self.reranker.model(tensor).squeeze(0)
                    ranked_indices = torch.argsort(scores, descending=True).tolist()

                reranked = []
                for idx in ranked_indices:
                    if idx < len(predicted_apps):
                        reranked.append(predicted_apps[idx])
                for c in predicted_apps:
                    if c not in reranked:
                        reranked.append(c)
                return reranked
            except Exception as e:
                logger.debug(f"V6 reranker inference failed: {e}")

        return predicted_apps

    def run(self, events: List[dict]) -> dict:
        """Replay events using V6 cache + appropriate reranker."""
        cache_hits = 0
        cache_misses = 0
        evictions = 0
        prefetched_total = 0
        latency_values: List[float] = []
        true_thrash_events = 0
        raw_evictions = 0

        prev_hot: set = set()
        actual_apps: List[str] = [e.get("app_id", "unknown") for e in events]

        for current_event_index, event in enumerate(
            _try_tqdm(events, desc="Evaluating GraphMind_V6", leave=False)
        ):
            app_id = event.get("app_id", "unknown")
            self._current_event_index = current_event_index

            # Snapshot cache state BEFORE lookup so hit detection correctly
            # answers "was this app already prefetched/cached before I needed it?"
            # (lookup() itself inserts the app into WARM, so checking after is circular)
            # This matches V5 runner which captures before_hot/before_warm before
            # calling EventBus.publish().
            cached_apps_before: set = self.cache_v6.get_all_cached_apps() | set(self._prefetched_apps)

            # 5-tier cache lookup — drives latency simulation, eviction bookkeeping,
            # and tier promotion. The FiveTierCache remains fully active.
            tier = self.cache_v6.lookup(app_id)

            # Hit detection: 5-event lookahead window (matches V5 runner exactly).
            # A prefetch is a "hit" if any of the next 5 apps launched was already
            # in any non-COLD tier (PIN/HOT/WARM/COOL) OR in the last prefetch list
            # BEFORE the current event's lookup triggered any cache update.
            HIT_LOOKAHEAD_WINDOW = 5
            lookahead_end = min(len(actual_apps), current_event_index + HIT_LOOKAHEAD_WINDOW)
            lookahead_window = actual_apps[current_event_index:lookahead_end]
            is_cache_hit = any(a in cached_apps_before for a in lookahead_window)

            if is_cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1

            # Simulate latency based on tier
            if tier == "pin":
                lat = 10.0
            elif tier == "hot":
                lat = 42.0
            elif tier == "warm":
                lat = 190.0
            elif tier == "cool":
                lat = settings.LATENCY_COOL_START_MS.get(
                    app_id, settings.LATENCY_COOL_START_MS.get("default", 400.0)
                )
            else:
                lat = settings.LATENCY_COLD_START_MS.get(
                    app_id, settings.LATENCY_COLD_START_MS.get("default", 720.0)
                )

            latency = lat * random.gauss(1.0, 0.08)
            latency = max(10.0, latency)
            latency_values.append(latency)

            # Publish event so graph updates its state
            payload = self._build_payload(event)
            EventBus.get_instance().publish(TOPIC_APP_LAUNCHED, payload)

            # Thrashing detection
            current_hot = set(self.cache_v6._hot.keys())
            newly_evicted = prev_hot - current_hot
            raw_evictions += len(newly_evicted)
            for nid in newly_evicted:
                self._eviction_index[nid] = current_event_index

            current_node_id = self.prefetch.current_node_id
            if current_node_id is not None and current_node_id in self._eviction_index:
                if current_event_index - self._eviction_index[current_node_id] <= 5:
                    true_thrash_events += 1

            # F1 / exact prediction tracking
            if current_event_index > 0 and app_id != "unknown":
                top1_predicted = next(
                    (a for a in self._prefetched_apps if a != "unknown"),
                    None,
                )
                if top1_predicted is not None:
                    if top1_predicted == app_id:
                        self.prefetch_tp += 1
                    else:
                        self.prefetch_fp += 1
                        self.prefetch_fn += 1

            # Update transition table
            if self._previous_app_id is not None:
                self._transition_counts[self._previous_app_id][app_id] += 1
            self._app_counts[app_id] += 1
            self._previous_app_id = app_id

            # Predict next apps using V5 base logic
            prefetched = self.prefetch.run_prefetch_cycle()
            predicted_apps = self._predict_next_apps(app_id, prefetched)

            # Rerank with the appropriate V6 reranker
            predicted_apps = self._rerank_candidates(predicted_apps, event)

            self._prefetched_apps = predicted_apps
            prefetched_total += len(prefetched)

            # Prefetch into V6 cache
            self.cache_v6.prefetch(predicted_apps)
            prev_hot = current_hot

            self.records.append(
                {
                    "user_id": self.user_id,
                    "day": int(event.get("day", 0)),
                    "app_id": event.get("app_id", "unknown"),
                    "node_id": current_node_id,
                    "tier": tier,
                    "cache_hit": is_cache_hit,
                    "latency_ms": latency,
                    "prefetched_ids": prefetched,
                    "prefetched_apps": predicted_apps,
                    "hot_count": len(self.cache_v6._hot),
                    "warm_count": len(self.cache_v6._warm),
                    "cold_count": len(self.cache_v6._cold),
                }
            )

        total = max(1, cache_hits + cache_misses)
        avg_latency = sum(latency_values) / max(1, len(latency_values))

        p_denom = self.prefetch_tp + self.prefetch_fp
        r_denom = self.prefetch_tp + self.prefetch_fn
        prefetch_precision = self.prefetch_tp / p_denom if p_denom > 0 else 0.0
        prefetch_recall = self.prefetch_tp / r_denom if r_denom > 0 else 0.0
        f1_denom = prefetch_precision + prefetch_recall
        prefetch_f1 = (
            2 * prefetch_precision * prefetch_recall / f1_denom
            if f1_denom > 0
            else 0.0
        )

        fp_rate = (
            self.prefetch_fp / (self.prefetch_tp + self.prefetch_fp)
            if (self.prefetch_tp + self.prefetch_fp) > 0
            else 0.0
        )

        result = {
            "cache_hit_rate": cache_hits / total,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "thrash_rate": true_thrash_events / total,
            "raw_evictions": raw_evictions,
            "battery_overhead_pct": min(5.0, prefetched_total * 0.001),
            "avg_latency_ms": avg_latency,
            "latency_saved_ms": max(0.0, (850.0 - avg_latency)),
            "graph_node_count": self.graph.node_count(),
            "graph_edge_count": self.graph.edge_count(),
            "precision": round(prefetch_precision, 4),
            "recall": round(prefetch_recall, 4),
            "f1": round(prefetch_f1, 4),
            "false_prefetch_rate": round(fp_rate, 4),
            "prefetch_precision": prefetch_precision,
            "prefetch_recall": prefetch_recall,
            "prefetch_f1": prefetch_f1,
            "prefetch_tp": self.prefetch_tp,
            "prefetch_fp": self.prefetch_fp,
            "prefetch_fn": self.prefetch_fn,
            "records": self.records,
        }
        EventBus.get_instance().clear_all()
        return result


# ---------------------------------------------------------------------------
# V6 Policy -- top-level interface used by evaluator_v2
# ---------------------------------------------------------------------------

class GraphMindV6Policy:
    """
    GraphMind V6 Policy.

    Trains either:
      - Per-user EmbeddingRerankerTrainer instances (multi-user real datasets),
      - A single legacy RerankerTrainer (synthetic single-user dataset).

    Provides the same interface as V5 so it plugs directly into
    BenchmarkEvaluatorV2.
    """

    def __init__(
        self,
        user_id: str = "v6_user",
        top_k: int = 8,
        reranker_epochs: int = 10,
        device: str = "cpu",
    ) -> None:
        self.user_id = user_id
        self.top_k = top_k
        self.device = device
        self._reranker_epochs = reranker_epochs

        self._v5 = GraphMindRLPolicy(user_id=f"{user_id}_v5", top_k=top_k)
        self._cache = FiveTierCache(user_id=user_id)

        # v2: per-user rerankers (used when >1 user detected)
        self._per_user_rerankers: Dict[str, EmbeddingRerankerTrainer] = {}
        self._app_to_idx: Dict[str, int] = {}   # global vocab, shared across users
        self._is_multi_user: bool = False

        # v1: legacy single reranker (synthetic single-user)
        self._reranker_trainer: Optional[RerankerTrainer] = None
        self._reranker_ready: bool = False
        self._app_vocab: List[str] = []

        self._train_events: List[dict] = []

    def get_name(self) -> str:
        return "GraphMind_V6"

    def reset(self) -> None:
        self._v5.reset()
        self._cache.reset()
        self._reranker_ready = False
        self._per_user_rerankers = {}
        self._train_events = []

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(self, events: list) -> None:
        self._train_events = events
        self._v5.train(events)

        # Build global app vocabulary from all events
        all_apps = sorted({e.get("app_id", "") for e in events if e.get("app_id")})
        self._app_vocab = all_apps
        self._app_to_idx = {a: i + 1 for i, a in enumerate(all_apps)}  # 0 = PAD
        n_apps = len(all_apps)

        # Detect multi-user vs single-user dataset
        users = {e.get("user_id", "default") for e in events if e.get("app_id")}
        self._is_multi_user = len(users) > 1

        tag = _dataset_tag(len(events))

        if self._is_multi_user:
            logger.info(
                f"V6: Multi-user dataset detected ({len(users)} users). "
                "Using per-user embedding rerankers."
            )
            self._train_per_user_rerankers(events, users, n_apps, tag)
        else:
            logger.info("V6: Single-user dataset. Using global embedding reranker.")
            self._train_single_reranker(events, n_apps, tag)

    def _train_per_user_rerankers(
        self,
        events: List[dict],
        users: set,
        n_apps: int,
        tag: str,
    ) -> None:
        """Train one EmbeddingRerankerTrainer per user."""
        # Group events by user
        user_events: Dict[str, List[dict]] = defaultdict(list)
        for event in events:
            uid = event.get("user_id", "default")
            user_events[uid].append(event)

        trained_count = 0
        total_users = len(users)

        for user_id in _try_tqdm(sorted(user_events.keys()), desc="V6: Training per-user rerankers"):
            u_events = sorted(
                user_events[user_id],
                key=lambda e: float(e.get("timestamp", 0)),
            )

            # --- Cache check ---
            pt_path = os.path.join(
                _SAVED_MODELS_DIR, f"v6_reranker_{tag}_{user_id}.pt"
            )
            meta_path = os.path.join(
                _SAVED_MODELS_DIR, f"v6_reranker_{tag}_{user_id}_meta.pkl"
            )

            if not _FORCE_RETRAIN and os.path.exists(pt_path) and os.path.exists(meta_path):
                try:
                    trainer = EmbeddingRerankerTrainer(
                        n_apps=n_apps,
                        top_k=self.top_k,
                        n_epochs=self._reranker_epochs,
                        device=self.device,
                    )
                    trainer.load(pt_path)
                    self._per_user_rerankers[user_id] = trainer
                    logger.info(f"V6: Loaded cached reranker for user {user_id}.")
                    trained_count += 1
                    continue
                except Exception as exc:
                    logger.warning(
                        f"V6: Cache load failed for user {user_id} ({exc}), retraining."
                    )

            # --- Run V5 on this user's events to build training samples ---
            v5_runner = GraphMindPolicyRunner(user_id, top_k=self.top_k)
            v5_runner.train(u_events)
            v5_result = v5_runner.run(u_events)
            records = v5_result["records"]

            samples = self._collect_samples(records, u_events)

            if len(samples) < 20:
                logger.info(
                    f"V6: User {user_id}: too few samples ({len(samples)}), skipping."
                )
                continue

            # --- Train per-user reranker ---
            trainer = EmbeddingRerankerTrainer(
                n_apps=n_apps,
                top_k=self.top_k,
                n_epochs=self._reranker_epochs,
                device=self.device,
            )
            trainer.train(samples, user_label=user_id)
            self._per_user_rerankers[user_id] = trainer
            trained_count += 1

            # --- Save to cache ---
            try:
                os.makedirs(_SAVED_MODELS_DIR, exist_ok=True)
                trainer.save(pt_path)
                with open(meta_path, "wb") as fh:
                    pickle.dump(
                        {"n_apps": n_apps, "app_vocab": self._app_vocab},
                        fh,
                        protocol=4,
                    )
                logger.info(f"V6: Saved reranker for user {user_id} -> {pt_path}")
            except Exception as exc:
                logger.warning(f"V6: Save failed for user {user_id}: {exc}")

        self._reranker_ready = trained_count > 0
        logger.info(
            f"V6: Per-user rerankers done — "
            f"{trained_count}/{total_users} users trained."
        )

    def _train_single_reranker(
        self,
        events: List[dict],
        n_apps: int,
        tag: str,
    ) -> None:
        """Train a single EmbeddingRerankerTrainer on all events (single-user)."""
        pt_path = os.path.join(_SAVED_MODELS_DIR, f"v6_reranker_{tag}.pt")
        meta_path = os.path.join(_SAVED_MODELS_DIR, f"v6_reranker_{tag}_meta.pkl")

        # --- Cache check (embedding-based) ---
        emb_pt_path = os.path.join(_SAVED_MODELS_DIR, f"v6_reranker_emb_{tag}.pt")
        emb_meta_path = os.path.join(_SAVED_MODELS_DIR, f"v6_reranker_emb_{tag}_meta.pkl")

        if not _FORCE_RETRAIN and os.path.exists(emb_pt_path) and os.path.exists(emb_meta_path):
            try:
                with open(emb_meta_path, "rb") as fh:
                    meta = pickle.load(fh)
                self._app_vocab = meta["app_vocab"]
                self._app_to_idx = {a: i + 1 for i, a in enumerate(self._app_vocab)}
                trainer = EmbeddingRerankerTrainer(
                    n_apps=len(self._app_vocab),
                    top_k=self.top_k,
                    n_epochs=self._reranker_epochs,
                    device=self.device,
                )
                trainer.load(emb_pt_path)
                # Store as "default" user reranker for single-user mode
                self._per_user_rerankers["default"] = trainer
                self._reranker_ready = True
                logger.info(f"V6: Embedding reranker loaded from cache ({emb_pt_path}).")
                return
            except Exception as exc:
                logger.warning(f"V6: Emb cache load failed ({exc}), retraining.")

        # --- Collect samples from V5 dry run ---
        logger.info("V6: Running V5 dry run to collect training samples...")
        v5_runner = GraphMindPolicyRunner(self.user_id, top_k=self.top_k)
        v5_runner.train(events)
        v5_result = v5_runner.run(events)
        records = v5_result["records"]

        logger.info(f"V6: Building reranker training data from {len(records)} records...")
        samples = self._collect_samples(records, events)
        logger.info(f"V6: Collected {len(samples)} reranker training samples.")

        if len(samples) < 20:
            logger.warning("V6: Too few reranker samples (<20). Skipping reranker training.")
            self._reranker_ready = False
            return

        trainer = EmbeddingRerankerTrainer(
            n_apps=n_apps,
            top_k=self.top_k,
            n_epochs=self._reranker_epochs,
            device=self.device,
        )
        trainer.train(samples, user_label="global")
        self._per_user_rerankers["default"] = trainer
        self._reranker_ready = True
        logger.info("V6: Single embedding reranker training complete.")

        # --- Save to cache ---
        try:
            os.makedirs(_SAVED_MODELS_DIR, exist_ok=True)
            trainer.save(emb_pt_path)
            with open(emb_meta_path, "wb") as fh:
                pickle.dump({"app_vocab": self._app_vocab}, fh, protocol=4)
            logger.info(f"V6: Saved embedding reranker -> {emb_pt_path}")
        except Exception as exc:
            logger.warning(f"V6: Cache save failed: {exc}")

    def _collect_samples(
        self,
        records: List[dict],
        events: List[dict],
    ) -> List:
        """
        Build (app_indices, extra_features, label) training samples from V5
        dry-run records.  Caps at _MAX_SAMPLES_PER_USER to keep training fast.
        """
        samples = []
        for t in range(len(records) - 1):
            actual_next = records[t + 1]["app_id"]
            candidates = records[t]["prefetched_apps"]

            if actual_next not in candidates or actual_next == "unknown":
                continue

            label = candidates.index(actual_next)
            confidences = [1.0 / (i + 1) for i in range(len(candidates))]
            time_norm = 0.0
            if t + 1 < len(events):
                time_norm = events[t + 1].get("time_bucket", 0) / 47.0

            app_indices, extra_features = build_candidate_indices(
                candidates=candidates,
                confidences=confidences,
                time_norm=time_norm,
                app_to_idx=self._app_to_idx,
                top_k=self.top_k,
            )
            samples.append((app_indices, extra_features, label))

            if len(samples) >= _MAX_SAMPLES_PER_USER:
                break

        return samples

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def predict_next_apps(self, current_app: str, context: dict) -> List[str]:
        candidates = self._v5.predict_next_apps(current_app, context)

        if not self._reranker_ready or not candidates:
            return candidates

        time_norm = context.get("time_bucket", 0) / 47.0
        confidences = [1.0 / (i + 1) for i in range(len(candidates))]

        # Determine which reranker to use
        uid = context.get("user_id", "default")
        reranker = self._per_user_rerankers.get(uid) or self._per_user_rerankers.get("default")

        if reranker is None:
            return candidates

        try:
            return reranker.rerank(
                candidates=candidates,
                app_to_idx=self._app_to_idx,
                confidences=confidences,
                time_norm=time_norm,
            )
        except Exception:
            return candidates

    def update(self, event: dict) -> None:
        self._v5.update(event)
        app_id = event.get("app_id", "")
        if app_id:
            self._cache.lookup(app_id)

    def run_full_evaluation(self, test_events: list) -> dict:
        """
        Evaluate V6 on test events.

        Multi-user mode (UbiqLog, is_multi_user=True):
            Creates one GraphMindV6PolicyRunner per user, trains each only on
            that user's training events, runs each on that user's test events,
            then aggregates the results.  This avoids the 31-user gradient
            conflict that kills performance when all events are mixed into one
            shared graph.

        Single-user mode (synthetic):
            Falls back to a single runner trained on all events (original
            behaviour).
        """
        if self._is_multi_user and self._train_events:
            return self._run_per_user_evaluation(test_events)

        # ── Single-user fallback (synthetic dataset) ──────────────────────
        runner = GraphMindV6PolicyRunner(
            user_id=self.user_id,
            top_k=self.top_k,
            per_user_rerankers=self._per_user_rerankers,
            app_to_idx=self._app_to_idx,
            reranker=self._reranker_trainer,
            reranker_ready=self._reranker_ready,
            app_vocab=self._app_vocab,
            device=self.device,
        )
        if self._train_events:
            runner.train(self._train_events)
        return runner.run(test_events)

    def _run_per_user_evaluation(self, test_events: list) -> dict:
        """
        Per-user evaluation for multi-user datasets (e.g. UbiqLog).

        Each user gets their own isolated GraphMindV6PolicyRunner — their own
        graph, memory manager, and 5-tier cache — so no user's transitions
        pollute another's.  Results are aggregated via weighted average.
        """
        from collections import defaultdict

        # Split training events by user
        train_by_user: Dict[str, List[dict]] = defaultdict(list)
        for e in self._train_events:
            uid = e.get("user_id", "default")
            train_by_user[uid].append(e)

        # Split test events by user
        test_by_user: Dict[str, List[dict]] = defaultdict(list)
        for e in test_events:
            uid = e.get("user_id", "default")
            test_by_user[uid].append(e)

        all_records = []
        agg = {
            "cache_hits": 0, "cache_misses": 0, "raw_evictions": 0,
            "prefetch_tp": 0, "prefetch_fp": 0, "prefetch_fn": 0,
            "thrash_count": 0, "prefetched_total": 0, "latency_sum": 0.0,
            "latency_n": 0, "graph_nodes": 0, "graph_edges": 0,
            "battery_sum": 0.0,
        }

        users_evaluated = 0
        for uid in _try_tqdm(sorted(test_by_user.keys()), desc="V6: Per-user evaluation"):
            u_train = sorted(train_by_user.get(uid, []), key=lambda e: float(e.get("timestamp", 0)))
            u_test  = sorted(test_by_user[uid],         key=lambda e: float(e.get("timestamp", 0)))

            if not u_test:
                continue

            runner = GraphMindV6PolicyRunner(
                user_id=uid,
                top_k=self.top_k,
                per_user_rerankers=self._per_user_rerankers,
                app_to_idx=self._app_to_idx,
                reranker=self._reranker_trainer,
                reranker_ready=self._reranker_ready,
                app_vocab=self._app_vocab,
                device=self.device,
            )
            if u_train:
                runner.train(u_train)

            try:
                res = runner.run(u_test)
            except Exception as exc:
                logger.warning(f"V6 per-user runner failed for {uid}: {exc}")
                continue

            # Accumulate
            agg["cache_hits"]      += res.get("cache_hits", 0)
            agg["cache_misses"]    += res.get("cache_misses", 0)
            agg["raw_evictions"]   += res.get("raw_evictions", 0)
            agg["prefetch_tp"]     += res.get("prefetch_tp", 0)
            agg["prefetch_fp"]     += res.get("prefetch_fp", 0)
            agg["prefetch_fn"]     += res.get("prefetch_fn", 0)
            n_test = len(u_test)
            agg["thrash_count"]    += int(res.get("thrash_rate", 0.0) * n_test)
            agg["prefetched_total"]+= int(res.get("battery_overhead_pct", 0.0) / 0.001)
            lat = res.get("avg_latency_ms", 0.0)
            agg["latency_sum"]     += lat * n_test
            agg["latency_n"]       += n_test
            agg["graph_nodes"]     += res.get("graph_node_count", 0)
            agg["graph_edges"]     += res.get("graph_edge_count", 0)
            all_records.extend(res.get("records", []))
            users_evaluated += 1

        if users_evaluated == 0:
            logger.warning("V6 per-user evaluation: no users completed successfully.")
            return {"cache_hit_rate": 0.0, "f1": 0.0, "precision": 0.0,
                    "recall": 0.0, "records": []}

        total  = max(1, agg["cache_hits"] + agg["cache_misses"])
        p_den  = agg["prefetch_tp"] + agg["prefetch_fp"]
        r_den  = agg["prefetch_tp"] + agg["prefetch_fn"]
        prec   = agg["prefetch_tp"] / p_den if p_den > 0 else 0.0
        rec    = agg["prefetch_tp"] / r_den if r_den > 0 else 0.0
        f1_den = prec + rec
        f1     = 2 * prec * rec / f1_den if f1_den > 0 else 0.0
        fp_r   = agg["prefetch_fp"] / p_den if p_den > 0 else 0.0
        avg_lat = agg["latency_sum"] / max(1, agg["latency_n"])

        logger.info(
            f"V6 per-user evaluation done ({users_evaluated} users) — "
            f"hit_rate={agg['cache_hits']/total:.4f}  f1={f1:.4f}"
        )

        return {
            "cache_hit_rate":       agg["cache_hits"] / total,
            "cache_hits":           agg["cache_hits"],
            "cache_misses":         agg["cache_misses"],
            "thrash_rate":          agg["thrash_count"] / total,
            "raw_evictions":        agg["raw_evictions"],
            "battery_overhead_pct": min(5.0, agg["prefetched_total"] * 0.001),
            "avg_latency_ms":       avg_lat,
            "latency_saved_ms":     max(0.0, 850.0 - avg_lat),
            "graph_node_count":     agg["graph_nodes"],
            "graph_edge_count":     agg["graph_edges"],
            "precision":            round(prec, 4),
            "recall":               round(rec, 4),
            "f1":                   round(f1, 4),
            "false_prefetch_rate":  round(fp_r, 4),
            "prefetch_precision":   prec,
            "prefetch_recall":      rec,
            "prefetch_f1":          f1,
            "prefetch_tp":          agg["prefetch_tp"],
            "prefetch_fp":          agg["prefetch_fp"],
            "prefetch_fn":          agg["prefetch_fn"],
            "records":              all_records,
        }

    def get_cache_stats(self) -> dict:
        return self._cache.stats()

    # -----------------------------------------------------------------------
    # Reranker evaluation helper (used by evaluator_v2.py)
    # -----------------------------------------------------------------------

    def evaluate_reranker(
        self,
        test_events: List[dict],
    ) -> dict:
        """
        Evaluate the trained reranker(s) on test events.

        Returns aggregate Hit@1, Hit@3 across all users.
        """
        if not self._reranker_ready or not self._per_user_rerankers:
            return {"hit_at_1": 0.0, "hit_at_3": 0.0, "n_samples": 0}

        all_samples = []
        prev_event = None

        for event in test_events:
            if prev_event is not None:
                prev_app = prev_event.get("app_id", "")
                actual = event.get("app_id", "")
                uid = event.get("user_id", "default")
                ctx = {
                    "time_bucket": event.get("time_bucket", 0),
                    "battery": event.get("battery", 100.0),
                    "weekend": event.get("weekend", False),
                    "user_id": uid,
                }
                try:
                    candidates = self._v5.predict_next_apps(prev_app, ctx)
                    if actual in candidates:
                        label = candidates.index(actual)
                        time_norm = event.get("time_bucket", 0) / 47.0
                        confidences = [1.0 / (i + 1) for i in range(len(candidates))]
                        app_indices, extra_features = build_candidate_indices(
                            candidates=candidates,
                            confidences=confidences,
                            time_norm=time_norm,
                            app_to_idx=self._app_to_idx,
                            top_k=self.top_k,
                        )
                        # Tag with user_id for per-user reranker lookup
                        all_samples.append((app_indices, extra_features, label, uid))
                except Exception:
                    pass
            prev_event = event

        if not all_samples:
            return {"hit_at_1": 0.0, "hit_at_3": 0.0, "n_samples": 0}

        hit1 = hit3 = 0
        for app_indices, extra_features, label, uid in all_samples:
            reranker = self._per_user_rerankers.get(uid) or self._per_user_rerankers.get("default")
            if reranker is None:
                continue
            reranker.model.eval()
            with torch.no_grad():
                app_idx_t = app_indices.unsqueeze(0).to(self.device)
                extra_t = extra_features.unsqueeze(0).to(self.device)
                scores = reranker.model(app_idx_t, extra_t).squeeze(0)
                top3 = torch.topk(scores, min(3, scores.shape[0])).indices.tolist()
                if label == top3[0]:
                    hit1 += 1
                if label in top3:
                    hit3 += 1

        n = len(all_samples)
        return {
            "hit_at_1": round(hit1 / n * 100, 2),
            "hit_at_3": round(hit3 / n * 100, 2),
            "n_samples": n,
        }
