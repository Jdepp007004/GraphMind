"""
Tests for RL train/eval split and policy comparison outputs.
"""

import os

import pandas as pd

from config import settings
from src.rl.evaluation import (
    RLEvaluator, TRAIN_USERS, VALIDATION_USERS, TEST_USERS
)
from src.rl.trainer import RLTrainer


def test_train_validation_test_split_enforced():
    evaluator = RLEvaluator()
    assert TRAIN_USERS == [f"user_{i:02d}" for i in range(8)]
    assert VALIDATION_USERS == ["user_08"]
    assert TEST_USERS == ["user_09"]
    assert evaluator.enforce_split("user_00", "train")
    assert evaluator.enforce_split("user_08", "validation")
    assert evaluator.enforce_split("user_09", "test")
    assert not evaluator.enforce_split("user_09", "train")


def test_policy_evaluation_generates_metrics():
    evaluator = RLEvaluator()
    row = evaluator.evaluate_policy("user_08", "NoOp", max_steps=8)
    required = {
        "user_id", "split", "policy_name", "cache_hit_rate", "precision",
        "recall", "f1", "thrash_rate", "latency_ms", "mean_reward", "steps"
    }
    assert required.issubset(set(row.keys()))
    assert row["split"] == "validation"
    assert row["steps"] > 0
    assert 0.0 <= row["cache_hit_rate"] <= 1.0


def test_policy_comparison_outputs_csv_and_json():
    evaluator = RLEvaluator()
    df = evaluator.run_policy_comparison(users=["user_08"], max_steps=6)
    assert isinstance(df, pd.DataFrame)
    assert set(df["policy_name"]) == {"Random", "NoOp", "Frequency", "LRU", "PPO"}
    assert os.path.exists(os.path.join(settings.RESULTS_DIR, "policy_comparison.csv"))
    assert os.path.exists(os.path.join(settings.RESULTS_DIR, "policy_comparison.json"))


def test_ppo_trains_and_writes_real_metrics():
    trainer = RLTrainer()
    path = trainer.train_user("user_00", total_timesteps=16)
    metrics_path = os.path.join(settings.RESULTS_DIR, "ppo_training_metrics.csv")
    assert os.path.exists(path)
    assert os.path.exists(metrics_path)
    df = pd.read_csv(metrics_path)
    assert {"user_id", "step", "episode_reward", "policy_loss",
            "value_loss", "entropy"}.issubset(set(df.columns))
    assert (df["user_id"] == "user_00").any()
