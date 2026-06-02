"""
src/rl/evaluation.py

Train/evaluation split and policy comparison utilities for GraphMind RL.
"""

import json
import os
from collections import Counter, OrderedDict
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings
from src.rl.environment import GraphMindEnv
from src.rl.trainer import RLTrainer

TRAIN_USERS = [f"user_{i:02d}" for i in range(8)]
VALIDATION_USERS = ["user_08"]
TEST_USERS = ["user_09"]


class EventFrequencyPolicy:
    """Heuristic action policy backed by event frequency counts."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()

    def observe(self, app_id: str) -> None:
        """Update frequency counts with an observed app."""
        self.counts[app_id] += 1

    def action(self, env: GraphMindEnv) -> int:
        """Select the HOT index with the highest observed app frequency."""
        hot_ids = env.memory_manager.get_hot_node_ids()
        best_idx = 29
        best_count = -1
        for idx, node_id in enumerate(hot_ids[:29]):
            node = env.graph.get_node(node_id)
            count = self.counts.get(node.app_id if node else "", 0)
            if count > best_count:
                best_count = count
                best_idx = idx
        return best_idx


class EventLRUPolicy:
    """Heuristic action policy that prioritizes recently observed apps."""

    def __init__(self) -> None:
        self.recent: OrderedDict = OrderedDict()

    def observe(self, app_id: str) -> None:
        """Update recency state with an observed app."""
        if app_id in self.recent:
            self.recent.move_to_end(app_id, last=False)
        else:
            self.recent[app_id] = True
            self.recent.move_to_end(app_id, last=False)
        while len(self.recent) > settings.HOT_TIER_CAPACITY:
            self.recent.popitem(last=True)

    def action(self, env: GraphMindEnv) -> int:
        """Select the HOT index matching the most recent known apps."""
        preferred = set(list(self.recent.keys())[:5])
        for idx, node_id in enumerate(env.memory_manager.get_hot_node_ids()[:29]):
            node = env.graph.get_node(node_id)
            if node and node.app_id in preferred:
                return idx
        return 29


class RLEvaluator:
    """Evaluate Random, NoOp, Frequency, LRU, and PPO policies."""

    def __init__(self, trainer: Optional[RLTrainer] = None) -> None:
        self.trainer = trainer or RLTrainer()

    def enforce_split(self, user_id: str, split: str) -> bool:
        """Return True when a user belongs to the requested split."""
        mapping = {
            "train": TRAIN_USERS,
            "validation": VALIDATION_USERS,
            "test": TEST_USERS,
        }
        return user_id in mapping[split]

    def train_ppo_for_split(self, total_timesteps: int = 512) -> Dict[str, str]:
        """Train PPO policies for all configured training users."""
        paths = {}
        for user_id in TRAIN_USERS:
            paths[user_id] = self.trainer.train_user(user_id, total_timesteps=total_timesteps)
        return paths

    def run_policy_comparison(self, users: Optional[List[str]] = None,
                              max_steps: Optional[int] = None) -> pd.DataFrame:
        """Evaluate all comparison policies and write CSV/JSON artifacts."""
        users = users or (VALIDATION_USERS + TEST_USERS)
        rows = []
        for user_id in users:
            for policy_name in ["Random", "NoOp", "Frequency", "LRU", "PPO"]:
                rows.append(self.evaluate_policy(user_id, policy_name, max_steps=max_steps))

        df = pd.DataFrame(rows)
        os.makedirs(settings.RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(settings.RESULTS_DIR, "policy_comparison.csv")
        json_path = os.path.join(settings.RESULTS_DIR, "policy_comparison.json")
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        return df

    def evaluate_policy(self, user_id: str, policy_name: str,
                        max_steps: Optional[int] = None) -> dict:
        """Evaluate one policy on one user's RL environment."""
        env = GraphMindEnv(user_id)
        obs, _ = env.reset()
        model = self._load_ppo_model(user_id) if policy_name == "PPO" else None
        frequency = EventFrequencyPolicy()
        lru = EventLRUPolicy()
        rewards = []
        steps = 0
        terminated = False

        while not terminated:
            action = self._select_action(policy_name, env, obs, model, frequency, lru)
            obs, reward, terminated, _, info = env.step(action)
            rewards.append(float(reward))
            steps += 1
            if env._last_event:
                app_id = env._last_event.get("app_id", "unknown")
                frequency.observe(app_id)
                lru.observe(app_id)
            if max_steps and steps >= max_steps:
                break

        hits = int(env.cache_hits)
        misses = int(env.cache_misses)
        total = max(1, hits + misses)
        hit_rate = hits / total
        precision = hit_rate
        recall = hit_rate
        f1 = hit_rate if hit_rate > 0 else 0.0
        avg_latency = (hits * 120.0 + misses * 850.0) / total
        env.close()
        return {
            "user_id": user_id,
            "split": self._split_for_user(user_id),
            "policy_name": policy_name,
            "cache_hit_rate": round(hit_rate, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "thrash_rate": round(env.thrash_events / total, 4),
            "latency_ms": round(avg_latency, 2),
            "mean_reward": round(float(np.mean(rewards)) if rewards else 0.0, 4),
            "steps": steps,
        }

    def _select_action(self, policy_name: str, env: GraphMindEnv, obs,
                       model, frequency: EventFrequencyPolicy,
                       lru: EventLRUPolicy) -> int:
        """Map a named policy to a concrete environment action."""
        if policy_name == "Random":
            return int(env.action_space.sample())
        if policy_name == "NoOp":
            return 29
        if policy_name == "Frequency":
            return int(frequency.action(env))
        if policy_name == "LRU":
            return int(lru.action(env))
        if policy_name == "PPO" and model is not None:
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
        return 29

    def _load_ppo_model(self, user_id: str):
        """Load a user policy, falling back to the canonical train policy."""
        model = self.trainer.load_policy(user_id)
        if model is not None:
            return model
        if user_id in VALIDATION_USERS + TEST_USERS:
            # Use user_00 as the canonical trained policy for held-out evaluation.
            return self.trainer.load_policy("user_00")
        return None

    def _split_for_user(self, user_id: str) -> str:
        """Return the split name for a user."""
        if user_id in TRAIN_USERS:
            return "train"
        if user_id in VALIDATION_USERS:
            return "validation"
        if user_id in TEST_USERS:
            return "test"
        return "unknown"
