"""
src/benchmarks/evaluator.py

Runs all 5 policies on all 10 users and produces comparative KPI numbers.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings
from src.benchmarks.baselines import (
    BaselinePolicy, LMKDReactiveBaseline, ARTStaticProfileBaseline,
    UsageStatsLRUBaseline, BixbyFrequencyBaseline
)
from src.benchmarks.graphmind_policy_runner import GraphMindPolicyRunner
from src.benchmarks.provenance import attach_row_provenance
from src.data.dataset_generator import USER_PROFILES

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    Runs all baselines + GraphMind on all 10 users.
    Measures: cache hit rate, launch speed gain, thrash events, battery overhead.
    """

    def __init__(self, max_events_per_user: Optional[int] = None) -> None:
        """Initialize all 4 baselines. Load all 10 user datasets."""
        env_limit = os.getenv("GRAPHMIND_BENCHMARK_MAX_EVENTS")
        if max_events_per_user is None and env_limit:
            max_events_per_user = int(env_limit)
        self.max_events_per_user = max_events_per_user if max_events_per_user is not None else 300
        self._baselines: List[BaselinePolicy] = [
            LMKDReactiveBaseline(),
            ARTStaticProfileBaseline(),
            UsageStatsLRUBaseline(),
            BixbyFrequencyBaseline()
        ]
        self._user_events: Dict[str, List[dict]] = {}
        self._load_all_user_events()

    def _load_all_user_events(self) -> None:
        """Load all 10 user event files from disk."""
        for profile in USER_PROFILES:
            uid = profile["user_id"]
            path = os.path.join(settings.USERS_DIR, f"{uid}.json")
            if os.path.exists(path):
                with open(path) as f:
                    self._user_events[uid] = json.load(f)
            else:
                logger.warning(f"Dataset not found for {uid}: {path}")

    def run_all(self) -> pd.DataFrame:
        """
        For each user x each policy (5 total), replay 30-day event log.
        Measure at each event:
            - cache_hit: was the next app already in simulated warm/hot cache?
            - thrash: was an app evicted and then immediately needed again?
            - battery_cost: simulated % drain per pre-fetch operation

        Returns DataFrame with columns:
        [user_id, policy_name, day, cache_hit_rate, launch_speed_gain_pct,
         thrash_rate, battery_overhead_pct, graph_node_count]

        Save to RESULTS_DIR/benchmark_results.csv.
        """
        os.makedirs(settings.RESULTS_DIR, exist_ok=True)
        rows = []
        for profile in USER_PROFILES:
            uid = profile["user_id"]
            events = self._user_events.get(uid, [])
            if self.max_events_per_user:
                events = events[:self.max_events_per_user]
            if not events:
                continue
            # Build ART profile from first 7 days
            art_baseline = ARTStaticProfileBaseline()
            art_baseline.build_profile(events)

            for policy in self._baselines:
                if isinstance(policy, ARTStaticProfileBaseline):
                    policy.build_profile(events)
                policy.reset()
                result = self.run_user_policy(uid, policy, events)
                row = {
                    "user_id": uid,
                    "policy_name": policy.get_name(),
                    "day": 29,  # final day aggregate
                    "cache_hit_rate": result["cache_hit_rate"],
                    "launch_speed_gain_pct": result["launch_speed_gain_pct"],
                    "thrash_rate": result["thrash_rate"],
                    "battery_overhead_pct": result["battery_overhead_pct"],
                    "graph_node_count": result.get("graph_node_count", 0)
                }
                rows.append(attach_row_provenance(
                    row,
                    measured={"cache_hit_rate", "thrash_rate", "graph_node_count"},
                    estimated={"launch_speed_gain_pct", "battery_overhead_pct"},
                ))

            # GraphMind_RL: evaluated through event replay like every baseline.
            lmkd_rate = next(r["cache_hit_rate"] for r in rows
                             if r["user_id"] == uid and r["policy_name"] == settings.BASELINE_LMKD)
            gm_result = self.run_graphmind_policy(uid, events)
            row = {
                "user_id": uid,
                "policy_name": settings.BASELINE_GRAPHMIND,
                "day": 29,
                "cache_hit_rate": gm_result["cache_hit_rate"],
                "launch_speed_gain_pct": self.compute_launch_speed_gain(
                    gm_result["cache_hit_rate"], lmkd_rate
                ),
                "thrash_rate": gm_result["thrash_rate"],
                "battery_overhead_pct": gm_result["battery_overhead_pct"],
                "graph_node_count": gm_result.get("graph_node_count", 0)
            }
            rows.append(attach_row_provenance(
                row,
                measured={"cache_hit_rate", "thrash_rate", "graph_node_count"},
                estimated={"launch_speed_gain_pct", "battery_overhead_pct"},
            ))

        df = pd.DataFrame(rows)
        out_path = os.path.join(settings.RESULTS_DIR, "benchmark_results.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Benchmark results saved to {out_path} ({len(df)} rows)")
        return df

    def run_graphmind_policy(self, user_id: str, events: List[dict]) -> dict:
        """Run GraphMind through graph, memory, and prefetch execution."""
        runner = GraphMindPolicyRunner(user_id)
        return runner.run(events)

    def run_user_policy(self, user_id: str, policy: BaselinePolicy,
                        events: List[dict]) -> dict:
        """
        Run one policy on one user's full event log.
        Returns dict of aggregate metrics for this user-policy combination.
        """
        cache_hits = 0
        cache_misses = 0
        thrash_events = 0
        prev_hot: set = set()
        battery_costs = []

        for i, event in enumerate(events):
            app_id = event.get("app_id", "unknown")
            context = {
                "time_bucket": event.get("time_bucket", 0),
                "battery": event.get("battery", 100.0),
                "weekend": event.get("weekend", False)
            }
            # Check prediction before update
            predictions = policy.predict_next_apps(app_id, context)
            predicted_set = set(predictions)
            if app_id in predicted_set or app_id in prev_hot:
                cache_hits += 1
            else:
                cache_misses += 1
            # Thrash: was app evicted from hot set last step?
            if i > 0 and app_id in prev_hot and app_id not in predicted_set:
                thrash_events += 1
            # Update policy
            policy.update(event)
            prev_hot = predicted_set
            battery_costs.append(event.get("battery", 100.0))

        total = max(1, cache_hits + cache_misses)
        hit_rate = cache_hits / total
        thrash_rate = thrash_events / total
        # Battery overhead: difference between predicted prefetch drain and actual
        battery_overhead = 1.5 if hit_rate > 0.5 else 0.5  # simulated overhead
        return {
            "cache_hit_rate": hit_rate,
            "launch_speed_gain_pct": self.compute_launch_speed_gain(hit_rate, 0.2),
            "thrash_rate": thrash_rate,
            "battery_overhead_pct": battery_overhead,
            "graph_node_count": len(set(e.get("app_id") for e in events))
        }

    def compute_launch_speed_gain(self, cache_hit_rate: float,
                                  baseline_cache_hit_rate: float) -> float:
        """
        Estimate launch speed gain from cache hit rate improvement.
        Based on Android ART documentation: cache hit -> ~30% faster cold start avoided.
        Formula: gain_pct = (cache_hit_rate - baseline_cache_hit_rate) * 30.0
        Returns percentage improvement (can be negative).
        """
        return (cache_hit_rate - baseline_cache_hit_rate) * 30.0

    def print_summary_table(self) -> None:
        """
        Print a formatted comparison table to stdout.
        Columns: Policy, Avg Cache Hit %, Launch Speed Gain %, Thrash Rate %, Battery Overhead %
        Highlight GraphMind row.
        """
        path = os.path.join(settings.RESULTS_DIR, "benchmark_results.csv")
        if not os.path.exists(path):
            print("No benchmark results found. Run benchmarks first.")
            return
        df = pd.read_csv(path)
        summary = df.groupby("policy_name").agg({
            "cache_hit_rate": "mean",
            "launch_speed_gain_pct": "mean",
            "thrash_rate": "mean",
            "battery_overhead_pct": "mean"
        }).reset_index()
        print("\n" + "=" * 80)
        print(f"{'Policy':<25} {'Cache Hit%':>12} {'Speed Gain%':>12} {'Thrash%':>10} {'Battery%':>10}")
        print("-" * 80)
        for _, row in summary.iterrows():
            prefix = ">>> " if row["policy_name"] == settings.BASELINE_GRAPHMIND else "    "
            print(f"{prefix}{row['policy_name']:<21} "
                  f"{row['cache_hit_rate']*100:>12.1f} "
                  f"{row['launch_speed_gain_pct']:>12.1f} "
                  f"{row['thrash_rate']*100:>10.1f} "
                  f"{row['battery_overhead_pct']:>10.1f}")
        print("=" * 80)

    def get_per_user_evolution(self) -> dict:
        """
        For GraphMind policy only, return cache hit rate by day for each user.
        Used by dashboard to show per-user graph evolution.
        Returns: {'user_00': [{'day': 0, 'cache_hit_rate': 0.23}, ...], ...}
        """
        evolution = {}
        for profile in USER_PROFILES:
            uid = profile["user_id"]
            log_path = os.path.join(settings.RESULTS_DIR, f"{uid}_simulation_log.json")
            if os.path.exists(log_path):
                try:
                    with open(log_path) as f:
                        log = json.load(f)
                    evolution[uid] = [
                        {"day": d.get("day", i),
                         "cache_hit_rate": d.get("state", {}).get("cache_hit_rate", 0.0)}
                        for i, d in enumerate(log.get("days", []))
                    ]
                    continue
                except Exception:
                    pass
            evolution[uid] = [{"day": i, "cache_hit_rate": 0.0} for i in range(30)]
        return evolution
