"""
src/agents/rl_trainer_agent.py

LangGraph agent: PPO training and weight updates.
"""

import logging
import os
from typing import Dict, Any, Optional

from config import settings
from src.rl.trainer import RLTrainer
from src.core.event_bus import EventBus, TOPIC_RL_WEIGHT_UPDATED

logger = logging.getLogger(__name__)


class RLTrainerAgent:
    """
    LangGraph agent that triggers additional PPO training when drift is detected.
    """

    def __init__(self, user_id: str) -> None:
        """Store user_id. Load PPO policy from disk if exists."""
        self.user_id = user_id
        self.trainer = RLTrainer()
        self._model = self.trainer.load_policy(user_id)
        if self._model:
            logger.debug(f"RLTrainerAgent: loaded policy for {user_id}")
        else:
            logger.debug(f"RLTrainerAgent: no policy found for {user_id}")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        If drift was detected (state['kl_divergence'] > DRIFT_KL_THRESHOLD):
            Spike learning rate: multiply current LR by DRIFT_LR_SPIKE_MULTIPLIER.
            Run 1000 additional PPO timesteps.
        Else: no-op (training runs in background via scripts/train_rl.py).
        Update state['last_agent'] = 'rl_trainer'.
        Return state.
        """
        kl = state.get("kl_divergence", 0.0)
        triggered = False
        if kl > settings.DRIFT_KL_THRESHOLD and self._model is not None:
            try:
                old_lr = self._model.learning_rate
                new_lr = old_lr * settings.DRIFT_LR_SPIKE_MULTIPLIER
                self._model.learning_rate = new_lr
                # Run short fine-tuning session (1000 steps)
                from src.rl.environment import GraphMindEnv
                env = GraphMindEnv(self.user_id)
                self._model.set_env(env)
                self._model.learn(total_timesteps=1000, reset_num_timesteps=False)
                self._model.learning_rate = old_lr  # restore
                env.close()
                triggered = True
                logger.info(f"RLTrainerAgent: LR spike + 1000 extra steps for {self.user_id}")
                # Publish RL weight updated
                bus = EventBus.get_instance()
                bus.publish(TOPIC_RL_WEIGHT_UPDATED, {
                    "timestamp": 0.0, "user_id": self.user_id,
                    "reason": "drift", "kl_divergence": kl
                })
            except Exception as e:
                logger.warning(f"RLTrainerAgent fine-tune failed: {e}")
        state["last_agent"] = "rl_trainer"
        state["messages"].append({
            "agent": "rl_trainer",
            "drift_triggered_training": triggered,
            "kl_divergence": kl
        })
        return state
