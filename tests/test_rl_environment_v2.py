"""tests/test_rl_environment_v2.py — GraphMindEnvV2 ResourceAllocationPolicy."""
import pytest
import numpy as np
from src.rl.environment_v2 import GraphMindEnvV2
from config import settings


def _make_events(n=20):
    apps = ["com.instagram.android", "com.whatsapp", "com.android.chrome"]
    return [
        {
            "app_id": apps[i % len(apps)],
            "battery": 80.0,
            "time_bucket": i % 48,
            "weekend": False,
            "headphones": False,
            "calendar_event_in_mins": None,
            "timestamp": float(i),
            "day": i // 10,
            "category": "social",
        }
        for i in range(n)
    ]


@pytest.fixture
def env():
    events = _make_events(30)
    e = GraphMindEnvV2(user_id="test_user", events=events)
    yield e
    e.close()


def test_env_observation_space_shape(env):
    assert env.observation_space.shape == (settings.RL_V2_OBS_DIM,)
    assert settings.RL_V2_OBS_DIM == 109


def test_env_action_space_shape(env):
    assert env.action_space.nvec.tolist() == [
        settings.RL_V2_N_HOT_LEVELS,
        settings.RL_V2_N_WARM_LEVELS,
        settings.RL_V2_N_CONF_LEVELS,
    ]


def test_env_reset_returns_correct_obs_shape(env):
    obs, info = env.reset()
    assert obs.shape == (settings.RL_V2_OBS_DIM,)
    assert obs.dtype == np.float32


def test_env_reset_obs_in_valid_range(env):
    obs, _ = env.reset()
    assert (obs >= 0.0).all()
    assert (obs <= 1.0).all()


def test_env_step_returns_5_tuple(env):
    env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_step_obs_shape(env):
    env.reset()
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    assert obs.shape == (settings.RL_V2_OBS_DIM,)


def test_env_runs_full_episode(env):
    obs, _ = env.reset()
    terminated = False
    steps = 0
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if truncated:
            break
    assert steps > 0


def test_env_action_hot_n_decoded_correctly(env):
    """Verify that action[0]=0 maps to smallest HOT capacity."""
    env.reset()
    # Use action with index 0 for hot tier (should pick smallest = 1)
    action = np.array([0, 0, 0])
    obs, reward, _, _, info = env.step(action)
    assert info["action_hot_n"] == settings.RL_V2_HOT_CAPACITY_OPTIONS[0]


def test_env_action_conf_threshold_decoded(env):
    """Verify that action[2]=4 maps to highest confidence threshold."""
    env.reset()
    action = np.array([0, 0, 4])
    obs, reward, _, _, info = env.step(action)
    assert info["action_conf_threshold"] == settings.RL_V2_CONF_THRESHOLD_OPTIONS[4]


def test_env_terminates_at_end_of_events(env):
    env.reset()
    terminated = False
    events_count = len(env._events)
    for _ in range(events_count + 5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    assert terminated


def test_env_hit_history_length(env):
    obs, _ = env.reset()
    # Hit history occupies last HIT_HISTORY_LEN positions
    hit_start = 2 * settings.RL_V2_APP_VOCAB_SIZE + 4
    hit_history = obs[hit_start: hit_start + settings.RL_V2_HIT_HISTORY_LEN]
    assert len(hit_history) == settings.RL_V2_HIT_HISTORY_LEN
