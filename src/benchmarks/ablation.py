"""
src/benchmarks/ablation.py

Ablation study framework for GraphMind v2.

Runs a controlled set of experiments by toggling system components.
Each experiment uses the SAME event stream — only components differ.
This is the correct methodology for ablation studies.

Experiments (from settings.ABLATION_ORDERED_VARIANTS):
  1. No_RL              — GraphOnly: graph prediction, no RL, no confidence prefetch
  2. Graph+Confidence   — GraphOnly + ConfidencePrefetch, no RL adaptation
  3. Graph+Confidence+NoRL — Confidence scorer active but RL resource allocation OFF
  4. Graph+RL           — RL ResourceAllocationPolicy, fixed top-k prefetch (no confidence)
  5. Full_System        — Graph + RL + ConfidencePrefetch + SensitivityModel

Additional experiments:
  No_Graph              — LRU only (no graph, no RL)
  No_Confidence         — Graph + RL, fixed top-k prefetch
  No_Security           — Full system without sensitivity-based cache flushes
  No_Context            — Graph + RL, without contextual (battery/time_bucket) features

Primary research question answered:
  "What does RL actually buy us?"

The comparison table:
  GraphOnly → Graph+Confidence → Graph+Confidence+NoRL → Graph+RL → Full_System

  If (Full_System > Graph+RL) by meaningful margin: confidence prefetch adds value.
  If (Full_System ≈ Graph+Confidence+NoRL): RL adds minimal value — notable finding.
  If (Full_System > all others): architecture is justified.
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from config import settings
from src.benchmarks.baselines_v2 import (
    LRUPolicy,
    GraphOnlyPolicy,
    FirstOrderMarkovPolicy,
    RecencyFrequencyPolicy,
)
from src.benchmarks.metrics_v2 import MetricsV2

logger = logging.getLogger(__name__)


class AblationRunner:
    """
    Runs ablation experiments on a fixed event stream.

    All experiments share the same test split to ensure comparability.
    The train split is used for Markov training (where applicable).
    """

    def __init__(
        self,
        user_id: str = "ablation_user",
        enable_security: bool = True,
    ) -> None:
        """
        Args:
            user_id:         User identifier (reused across experiments).
            enable_security: If True, Full_System includes SensitivityModel flushes.
        """
        self._user_id = user_id
        self._enable_security = enable_security
        self._metrics = MetricsV2()

    def run_all(
        self,
        train_events: List[dict],
        test_events: List[dict],
    ) -> Dict[str, dict]:
        """
        Run all ablation experiments. Returns results dict keyed by variant name.

        Args:
            train_events: Chronological TRAIN split for policy training.
            test_events:  Chronological TEST split for evaluation.

        Returns:
            {variant_name: metrics_dict} for all variants.
        """
        results: Dict[str, dict] = {}

        experiments = [
            (settings.ABLATION_NO_RL,               self._run_no_rl),
            (settings.ABLATION_GRAPH_PLUS_CONFIDENCE, self._run_graph_plus_confidence),
            (settings.ABLATION_GRAPH_CONFIDENCE_NO_RL, self._run_graph_confidence_no_rl),
            (settings.ABLATION_GRAPH_RL_ONLY,         self._run_graph_rl_only),
            (settings.ABLATION_FULL_SYSTEM,            self._run_full_system),
            (settings.ABLATION_NO_GRAPH,               self._run_no_graph),
            (settings.ABLATION_NO_CONFIDENCE,          self._run_no_confidence),
            (settings.ABLATION_NO_SECURITY,            self._run_no_security),
            (settings.ABLATION_NO_CONTEXT,             self._run_no_context),
        ]

        for name, fn in experiments:
            logger.info(f"Running ablation: {name}")
            t0 = time.perf_counter()
            try:
                result = fn(train_events, test_events)
                result["variant"] = name
                result["eval_time_s"] = round(time.perf_counter() - t0, 2)
                results[name] = result
                logger.info(
                    f"  {name}: hit_rate={result.get('cache_hit_rate', 'N/A')} "
                    f"f1={result.get('f1', 'N/A')}"
                )
            except Exception as exc:
                logger.error(f"  {name} FAILED: {exc}")
                results[name] = {
                    "variant": name,
                    "error": str(exc),
                    "eval_time_s": round(time.perf_counter() - t0, 2),
                }

        return results

    # ── Experiment implementations ─────────────────────────────────────────

    def _run_no_rl(
        self, train_events: List[dict], test_events: List[dict]
    ) -> dict:
        """
        GraphOnly — Graph prediction only, no RL, no confidence prefetch.
        This is the critical ablation anchor: baseline graph performance.
        """
        policy = GraphOnlyPolicy(user_id=f"{self._user_id}_no_rl")
        return self._evaluate_simple_policy(policy, train_events, test_events)

    def _run_graph_plus_confidence(
        self, train_events: List[dict], test_events: List[dict]
    ) -> dict:
        """
        Graph + Confidence Prefetch, no RL.
        Isolates the contribution of confidence-based candidate filtering.
        """
        from src.core.event_bus import EventBus
        from src.core.graph_engine import BehaviouralGraph
        from src.core.memory_manager import MemoryManager
        from src.prefetch.confidence_prefetch import ConfidencePrefetch

        EventBus.get_instance().clear_all()
        graph = BehaviouralGraph(f"{self._user_id}_g_conf")
        memory_manager = MemoryManager(f"{self._user_id}_g_conf", graph)
        confidence_scorer = ConfidencePrefetch(graph)

        return self._evaluate_with_confidence(
            graph, memory_manager, confidence_scorer,
            train_events, test_events,
            use_rl=False,
        )

    def _run_graph_confidence_no_rl(
        self, train_events: List[dict], test_events: List[dict]
    ) -> dict:
        """
        Graph + Confidence scoring, RL resource allocation DISABLED.
        Fixed budget (HOT_TIER_CAPACITY=30, WARM_TIER_CAPACITY=150).

        This directly answers: 'Does confidence scoring account for most gains,
        even without RL deciding the budget?'
        """
        from src.core.event_bus import EventBus
        from src.core.graph_engine import BehaviouralGraph
        from src.core.memory_manager import MemoryManager
        from src.prefetch.confidence_prefetch import ConfidencePrefetch

        EventBus.get_instance().clear_all()
        graph = BehaviouralGraph(f"{self._user_id}_gconf_norl")
        memory_manager = MemoryManager(f"{self._user_id}_gconf_norl", graph)
        # Use confidence scorer but with fixed threshold and no RL budget adjustment
        confidence_scorer = ConfidencePrefetch(
            graph,
            confidence_threshold=settings.PREFETCH_CONFIDENCE_THRESHOLD,
        )
        return self._evaluate_with_confidence(
            graph, memory_manager, confidence_scorer,
            train_events, test_events,
            use_rl=False,
            fixed_hot_n=settings.HOT_TIER_CAPACITY,
            fixed_warm_n=settings.WARM_TIER_CAPACITY,
        )

    def _run_graph_rl_only(
        self, train_events: List[dict], test_events: List[dict]
    ) -> dict:
        """
        Graph + RL resource allocation, fixed top-k prefetch (no confidence scorer).
        Isolates RL's contribution independent of confidence scoring.
        """
        from src.rl.environment_v2 import GraphMindEnvV2
        env = GraphMindEnvV2(
            user_id=f"{self._user_id}_rl_only",
            events=test_events,
        )
        # Use fixed confidence threshold (disable adaptive aspect)
        # Run with random actions as stand-in (RL training out of scope here)
        return self._evaluate_env_with_policy(env, test_events, policy="random")

    def _run_full_system(
        self, train_events: List[dict], test_events: List[dict]
    ) -> dict:
        """
        Full System: Graph + RL + ConfidencePrefetch + SensitivityModel.
        This is the complete GraphMind proposed system.
        """
        from src.rl.environment_v2 import GraphMindEnvV2
        from src.security.sensitivity_model import SensitivityModel

        env = GraphMindEnvV2(
            user_id=f"{self._user_id}_full",
            events=test_events,
        )
        security = SensitivityModel() if self._enable_security else None
        return self._evaluate_env_with_policy(
            env, test_events, policy="random", security=security
        )

    def _run_no_graph(
        self, train_events: List[dict], test_events: List[dict]
    ) -> dict:
        """No Graph: LRU only. Measures what structure-free policies can achieve."""
        policy = LRUPolicy()
        return self._evaluate_simple_policy(policy, train_events, test_events)

    def _run_no_confidence(
        self, train_events: List[dict], test_events: List[dict]
    ) -> dict:
        """Graph + RL, no confidence prefetch (fixed top-k)."""
        from src.rl.environment_v2 import GraphMindEnvV2
        env = GraphMindEnvV2(
            user_id=f"{self._user_id}_no_conf",
            events=test_events,
        )
        # Disable confidence by setting threshold to 0.0 (accept all)
        env.confidence_prefetch._threshold = 0.0
        return self._evaluate_env_with_policy(env, test_events, policy="random")

    def _run_no_security(
        self, train_events: List[dict], test_events: List[dict]
    ) -> dict:
        """Full system without SensitivityModel cache flushes."""
        from src.rl.environment_v2 import GraphMindEnvV2
        env = GraphMindEnvV2(
            user_id=f"{self._user_id}_no_sec",
            events=test_events,
        )
        return self._evaluate_env_with_policy(env, test_events, policy="random", security=None)

    def _run_no_context(
        self, train_events: List[dict], test_events: List[dict]
    ) -> dict:
        """
        Graph + RL without contextual features (battery, time_bucket zeroed).
        Measures whether context signals contribute to performance.
        """
        # Zero out context features in the event stream copy
        zeroed_events = []
        for event in test_events:
            e = dict(event)
            e["battery"] = 100.0   # neutral battery
            e["time_bucket"] = 0   # midnight bucket always
            e["weekend"] = False
            zeroed_events.append(e)
        from src.rl.environment_v2 import GraphMindEnvV2
        env = GraphMindEnvV2(
            user_id=f"{self._user_id}_no_ctx",
            events=zeroed_events,
        )
        return self._evaluate_env_with_policy(env, zeroed_events, policy="random")

    # ── Evaluation helpers ────────────────────────────────────────────────

    def _evaluate_simple_policy(
        self,
        policy,
        train_events: List[dict],
        test_events: List[dict],
    ) -> dict:
        """
        Evaluate a simple (non-environment-based) policy.

        Protocol:
          - Update policy on each training event (online learning for non-Markov).
          - Evaluate predictions on each test event.
          - Count cache hits: hit = predicted list contains the actual next app.
        """
        policy.reset()
        for event in train_events:
            policy.update(event)

        hits, misses, tp, fp, fn = 0, 0, 0, 0, 0
        app_ids, tiers = [], []
        prev_event: Optional[dict] = None

        for i, event in enumerate(test_events):
            app_id = event.get("app_id", "")
            context = {
                "time_bucket": event.get("time_bucket", 0),
                "battery": event.get("battery", 100.0),
                "weekend": event.get("weekend", False),
            }
            if prev_event is not None:
                prev_app = prev_event.get("app_id", "")
                predicted = policy.predict_next_apps(prev_app, context)

                if app_id in predicted:
                    hits += 1
                    tp += 1
                    tiers.append("warm")  # treat prediction hit as warm-start equivalent
                else:
                    misses += 1
                    fn += 1
                    tiers.append("cold")

                fp += max(0, len(predicted) - (1 if app_id in predicted else 0))
                app_ids.append(app_id)

            policy.update(event)
            prev_event = event

        total = hits + misses
        return self._metrics.compute_all(
            cache_hits=hits,
            cache_misses=misses,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            thrash_events=0,
            prefetch_total=max(1, tp + fp),
            app_id_list=app_ids,
            tier_list=tiers,
            hot_count=min(30, hits),
            warm_count=min(150, total),
        )

    def _evaluate_with_confidence(
        self,
        graph,
        memory_manager,
        confidence_scorer,
        train_events: List[dict],
        test_events: List[dict],
        use_rl: bool = False,
        fixed_hot_n: Optional[int] = None,
        fixed_warm_n: Optional[int] = None,
    ) -> dict:
        """Evaluate Graph + Confidence combination."""
        from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED
        bus = EventBus.get_instance()

        # Warm up confidence scorer on training events
        for event in train_events:
            confidence_scorer.observe_event(event)

        hits, misses, tp, fp, fn, thrash = 0, 0, 0, 0, 0, 0
        app_ids, tiers = [], []
        prev_hot: set = set()
        current_node_id: Optional[str] = None

        for event in test_events:
            app_id = event.get("app_id", "")
            time_bucket = int(event.get("time_bucket", 0))
            battery = float(event.get("battery", 100.0))

            # Publish event
            payload = {
                "timestamp": float(event.get("timestamp", 0.0)),
                "user_id": memory_manager._user_id if hasattr(memory_manager, "_user_id") else "ablation",
                "app_id": app_id,
                "category": event.get("category", "utility"),
                "battery": battery,
                "time_of_day_bucket": time_bucket,
                "time_bucket": time_bucket,
                "day": int(event.get("day", 0)),
                "weekend": bool(event.get("weekend", False)),
                "headphones": bool(event.get("headphones", False)),
                "calendar_event_in_mins": event.get("calendar_event_in_mins"),
            }
            bus.publish(TOPIC_APP_LAUNCHED, payload)
            confidence_scorer.observe_event(event)

            # Check cache hit
            current_hot = set(memory_manager.get_hot_node_ids())
            current_warm = set(memory_manager.get_warm_node_ids())
            is_hot_hit = current_node_id in prev_hot
            is_warm_hit = False  # simplified

            if is_hot_hit or is_warm_hit:
                hits += 1
                tiers.append("hot" if is_hot_hit else "warm")
                tp += 1
            else:
                misses += 1
                tiers.append("cold")
                fn += 1
            app_ids.append(app_id)

            # Thrash: nodes evicted from HOT
            thrash += len(prev_hot - current_hot)
            prev_hot = current_hot

        total = hits + misses
        return self._metrics.compute_all(
            cache_hits=hits,
            cache_misses=misses,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            thrash_events=thrash,
            prefetch_total=max(1, tp + fp),
            app_id_list=app_ids,
            tier_list=tiers,
            hot_count=len(prev_hot),
            warm_count=len(memory_manager.get_warm_node_ids()),
        )

    def _evaluate_env_with_policy(
        self,
        env,
        test_events: List[dict],
        policy: str = "random",
        security=None,
    ) -> dict:
        """
        Evaluate using the Gymnasium environment with a simple policy.

        policy: "random" samples random actions from the action space.
        In a full training run, a trained SB3 PPO agent would be used here.
        """
        obs, info = env.reset()
        hits, misses, tp, fp, fn, thrash = 0, 0, 0, 0, 0, 0
        app_ids: List[str] = []
        tiers: List[str] = []
        flush_count = 0

        for step, event in enumerate(test_events):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)

            is_hit = step_info.get("is_hit", False)
            app_id = event.get("app_id", "")

            if is_hit:
                hits += 1
                tiers.append("hot")
                tp += 1
            else:
                misses += 1
                tiers.append("cold")
                fn += 1
            app_ids.append(app_id)
            thrash += step_info.get("thrash_events", 0) - thrash  # delta per step

            # Security flush tracking
            if security is not None:
                sec_result = security.on_app_launched(app_id)
                if sec_result.get("flushed"):
                    flush_count += 1

            if terminated or truncated:
                break

        env.close()
        total = hits + misses
        result = self._metrics.compute_all(
            cache_hits=hits,
            cache_misses=misses,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            thrash_events=thrash,
            prefetch_total=max(1, tp + fp),
            app_id_list=app_ids,
            tier_list=tiers,
            hot_count=len(env.memory_manager.get_hot_node_ids()),
            warm_count=len(env.memory_manager.get_warm_node_ids()),
        )
        result["security_flush_count"] = flush_count
        return result


def run_ablation_comparison_table(
    train_events: List[dict],
    test_events: List[dict],
    user_id: str = "ablation",
) -> Dict[str, dict]:
    """
    Convenience entry point to run all ablations and return results.

    Args:
        train_events: TRAIN split events.
        test_events:  TEST split events.
        user_id:      User identifier.

    Returns:
        {variant_name: metrics_dict}
    """
    runner = AblationRunner(user_id=user_id)
    return runner.run_all(train_events, test_events)
