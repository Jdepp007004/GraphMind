"""
tests/test_phase3_rl.py

Phase 3 gate tests: RL environment, reward function, and PPO training.
"""

import os
import pytest
import numpy as np

from src.core.event_bus import EventBus
from src.rl.reward import compute_reward, compute_episode_summary
from src.rl.environment import GraphMindEnv
from src.rl.trainer import RLTrainer
from config import settings


def test_reward_compute_positive():
    """Good performance must yield positive reward."""
    r = compute_reward(9, 1, 0, 0.5, 8, 1.0)
    assert isinstance(r, float)
    assert r > 0


def test_reward_penalizes_thrash():
    """High thrash must reduce reward compared to no thrash."""
    no_thrash = compute_reward(5, 5, 0, 1.0, 3, 1.0)
    high_thrash = compute_reward(5, 5, 10, 1.0, 3, 1.0)
    assert high_thrash < no_thrash


def test_reward_penalizes_battery():
    """High battery consumed must reduce reward."""
    low_battery = compute_reward(5, 5, 0, 0.0, 3, 1.0)
    high_battery = compute_reward(5, 5, 0, 5.0, 3, 1.0)
    assert high_battery < low_battery


def test_reward_episode_summary():
    """compute_episode_summary must return correct statistics."""
    rewards = [1.0, 2.0, 3.0, 0.5, -0.5]
    s = compute_episode_summary(rewards)
    assert set(s.keys()) == {"mean", "min", "max", "total", "steps"}
    assert s["steps"] == 5
    assert abs(s["mean"] - 1.2) < 0.001
    assert s["min"] == -0.5
    assert s["max"] == 3.0


def test_rl_env_instantiate():
    """GraphMindEnv must instantiate without error."""
    EventBus.get_instance().clear_all()
    env = GraphMindEnv("user_00")
    assert env is not None
    EventBus.get_instance().clear_all()


def test_rl_env_observation_space():
    """Observation space must be Box of shape (68,)."""
    import gymnasium
    EventBus.get_instance().clear_all()
    env = GraphMindEnv("user_00")
    assert isinstance(env.observation_space, gymnasium.spaces.Box)
    assert env.observation_space.shape == (68,)
    EventBus.get_instance().clear_all()


def test_rl_env_action_space():
    """Action space must be Discrete(31)."""
    import gymnasium
    EventBus.get_instance().clear_all()
    env = GraphMindEnv("user_00")
    assert isinstance(env.action_space, gymnasium.spaces.Discrete)
    assert env.action_space.n == 31
    EventBus.get_instance().clear_all()


def test_rl_env_reset():
    """reset() must return (obs, info) with obs of shape (68,) and dtype float32."""
    EventBus.get_instance().clear_all()
    env = GraphMindEnv("user_00")
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (68,)
    assert obs.dtype == np.float32
    EventBus.get_instance().clear_all()


def test_rl_env_step():
    """step() must return correct 5-tuple."""
    EventBus.get_instance().clear_all()
    env = GraphMindEnv("user_00")
    env.reset()
    result = env.step(29)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (68,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert "cache_hits" in info
    env.close()
    EventBus.get_instance().clear_all()


def test_rl_trainer_import():
    """RLTrainer must instantiate without error."""
    trainer = RLTrainer()
    assert trainer is not None


def test_rl_model_exists():
    """PPO model file must exist for user_00."""
    path = os.path.join(settings.RL_MODELS_DIR, "user_00_ppo.zip")
    assert os.path.exists(path), f"Model not found: {path}"


def test_rl_model_loadable():
    """Loaded PPO must predict valid actions."""
    trainer = RLTrainer()
    model = trainer.load_policy("user_00")
    assert model is not None
    obs = np.zeros((68,), dtype=np.float32)
    action, _ = model.predict(obs)
    assert 0 <= int(action) <= 30
