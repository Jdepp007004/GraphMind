"""
src/rl/trainer.py

Runs PPO training for all 10 users. Saves trained policy to disk.
"""

import os
import logging
import json
import time
from typing import Optional, List

from config import settings
from src.rl.environment import GraphMindEnv
from src.data.dataset_generator import USER_PROFILES

logger = logging.getLogger(__name__)


class TrainingMetricsCallback:
    """Collect real SB3 training metrics observed during learning."""

    def __init__(self, user_id: str) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: "TrainingMetricsCallback") -> None:
                super().__init__()
                self.outer = outer
                self.episode_reward = 0.0

            def _on_step(self) -> bool:
                """Collect reward and logger values for the current training step."""
                rewards = self.locals.get("rewards", [0.0])
                reward = float(rewards[0]) if len(rewards) else 0.0
                self.episode_reward += reward
                dones = self.locals.get("dones", [False])
                done = bool(dones[0]) if len(dones) else False
                logger_values = getattr(self.model.logger, "name_to_value", {})
                self.outer.records.append({
                    "user_id": self.outer.user_id,
                    "step": int(self.num_timesteps),
                    "episode_reward": self.episode_reward,
                    "policy_loss": _to_float(logger_values.get("train/policy_gradient_loss")),
                    "value_loss": _to_float(logger_values.get("train/value_loss")),
                    "entropy": _to_float(logger_values.get("train/entropy_loss")),
                })
                if done:
                    self.episode_reward = 0.0
                return True

        self.user_id = user_id
        self.records: List[dict] = []
        self.callback = _Callback(self)


def _to_float(value):
    """Convert logger values to floats while preserving missing values."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


class RLTrainer:
    """
    Manages PPO training for GraphMind across all users.
    """

    def __init__(self) -> None:
        """
        Create MODELS_DIR/rl_policies/ directory if needed.
        Initialize W&B run if WANDB_API_KEY is set, else log offline.
        Set self.trained_users = {} (dict of user_id -> model path)
        """
        os.makedirs(settings.RL_MODELS_DIR, exist_ok=True)
        os.makedirs(settings.RESULTS_DIR, exist_ok=True)
        self.trained_users: dict = {}
        self._wandb_enabled = False
        wandb_key = os.getenv("WANDB_API_KEY", "")
        if wandb_key and wandb_key.lower() not in ("", "disabled", "none"):
            try:
                import wandb
                wandb.init(project="graphmind", mode=os.getenv("WANDB_MODE", "offline"))
                self._wandb_enabled = True
            except Exception as e:
                logger.warning(f"W&B init failed: {e}. Logging offline.")

    def train_user(self, user_id: str,
                   total_timesteps: int = settings.PPO_TOTAL_TIMESTEPS) -> str:
        """
        Train a PPO agent for one user.
        Creates GraphMindEnv(user_id).
        Wraps with stable_baselines3.PPO using MlpPolicy.
        Training hyperparams from settings.py (PPO_LEARNING_RATE, PPO_N_STEPS, etc.)
        Uses WandbCallback if W&B is active.
        Saves model to RL_MODELS_DIR/{user_id}_ppo.zip.
        Returns the save path.
        Logs training start and completion.
        """
        from stable_baselines3 import PPO

        logger.info(f"Training PPO for {user_id}, timesteps={total_timesteps}")
        start_time = time.time()

        env = GraphMindEnv(user_id)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=settings.PPO_LEARNING_RATE,
            n_steps=min(settings.PPO_N_STEPS, 512),  # reduced for short episodes
            batch_size=settings.PPO_BATCH_SIZE,
            n_epochs=settings.PPO_N_EPOCHS,
            gamma=settings.PPO_GAMMA,
            verbose=0,
            seed=settings.RANDOM_SEED
        )

        callbacks = []
        metrics_callback = TrainingMetricsCallback(user_id)
        callbacks.append(metrics_callback.callback)
        if self._wandb_enabled:
            try:
                from wandb.integration.sb3 import WandbCallback
                callbacks.append(WandbCallback(model_save_freq=1000,
                                               model_save_path=settings.RL_MODELS_DIR))
            except Exception:
                pass

        model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None)

        save_path = os.path.join(settings.RL_MODELS_DIR, f"{user_id}_ppo")
        model.save(save_path)
        self.trained_users[user_id] = save_path + ".zip"
        elapsed = time.time() - start_time
        logger.info(f"PPO training complete for {user_id} in {elapsed:.1f}s. Saved to {save_path}.zip")

        self._save_training_metrics(metrics_callback.records)
        env.close()
        return save_path + ".zip"

    def _save_training_metrics(self, records: List[dict]) -> None:
        """Persist callback-collected PPO training metrics."""
        try:
            import pandas as pd

            out_path = os.path.join(settings.RESULTS_DIR, "ppo_training_metrics.csv")
            if os.path.exists(out_path):
                existing = pd.read_csv(out_path)
                df = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
            else:
                df = pd.DataFrame(records)
            df.to_csv(out_path, index=False)
        except Exception as e:
            logger.warning(f"Could not save PPO training metrics: {e}")

    def train_all_users(self) -> dict:
        """
        Train PPO for all 10 users in USER_PROFILES order.
        Calls train_user() for each.
        Returns dict: {user_id: model_path}
        Logs total training time at completion.
        """
        start = time.time()
        for profile in USER_PROFILES:
            uid = profile["user_id"]
            try:
                path = self.train_user(uid)
                self.trained_users[uid] = path
            except Exception as e:
                logger.error(f"Training failed for {uid}: {e}")
        elapsed = time.time() - start
        logger.info(f"All users trained in {elapsed:.1f}s")
        return self.trained_users

    def load_policy(self, user_id: str):
        """
        Load a saved PPO policy from RL_MODELS_DIR/{user_id}_ppo.zip.
        Returns the PPO model object.
        Returns None if file doesn't exist.
        """
        from stable_baselines3 import PPO
        path = os.path.join(settings.RL_MODELS_DIR, f"{user_id}_ppo.zip")
        if not os.path.exists(path):
            return None
        try:
            model = PPO.load(path)
            logger.debug(f"Loaded PPO policy from {path}")
            return model
        except Exception as e:
            logger.error(f"Could not load policy from {path}: {e}")
            return None

    def get_training_curves(self) -> dict:
        """
        Return training curve data for dashboard rendering.
        Reads from W&B local logs or from saved CSV in RESULTS_DIR/training_curves.csv.
        Returns: {'user_id': [{'step': int, 'reward': float}, ...], ...}
        Returns empty dict if no training data found.
        """
        metrics_path = os.path.join(settings.RESULTS_DIR, "ppo_training_metrics.csv")
        if os.path.exists(metrics_path):
            try:
                import pandas as pd

                df = pd.read_csv(metrics_path)
                curves = {}
                for user_id, group in df.groupby("user_id"):
                    curves[user_id] = [
                        {"step": int(r["step"]), "reward": float(r["episode_reward"])}
                        for _, r in group.iterrows()
                    ]
                return curves
            except Exception:
                pass
        return {}
