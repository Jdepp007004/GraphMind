"""tests/test_baselines_v2.py — All 10 baseline policies."""
import pytest
from src.benchmarks.baselines_v2 import (
    RandomPolicy, LRUPolicy, LFUPolicy, MRUPolicy,
    FrequencyPolicy, RecencyFrequencyPolicy,
    FirstOrderMarkovPolicy, SecondOrderMarkovPolicy,
    GraphOnlyPolicy, GraphMindRLPolicy,
)
from config import settings


def _make_events(n=20):
    apps = ["com.instagram.android", "com.whatsapp", "com.android.chrome",
            "com.spotify.music", "com.google.youtube"]
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


def test_random_policy():
    policy = RandomPolicy()
    events = _make_events()
    for e in events:
        policy.update(e)
    preds = policy.predict_next_apps("com.whatsapp", {"time_bucket": 0, "battery": 80})
    assert isinstance(preds, list)
    assert all(isinstance(p, str) for p in preds)


def test_lru_policy():
    policy = LRUPolicy()
    events = _make_events()
    for e in events:
        policy.update(e)
    preds = policy.predict_next_apps("", {})
    assert isinstance(preds, list)
    assert len(preds) <= 5


def test_lfu_policy():
    policy = LFUPolicy()
    events = _make_events()
    for e in events:
        policy.update(e)
    preds = policy.predict_next_apps("", {})
    assert len(preds) <= 5


def test_mru_policy():
    policy = MRUPolicy()
    events = _make_events()
    for e in events:
        policy.update(e)
    preds = policy.predict_next_apps("", {})
    assert isinstance(preds, list)


def test_frequency_policy():
    policy = FrequencyPolicy()
    events = _make_events()
    for e in events:
        policy.update(e)
    preds = policy.predict_next_apps("", {"time_bucket": 0, "weekend": False})
    assert isinstance(preds, list)


def test_recency_frequency_policy():
    policy = RecencyFrequencyPolicy()
    events = _make_events()
    for e in events:
        policy.update(e)
    preds = policy.predict_next_apps("", {"time_bucket": 0, "battery": 80})
    assert isinstance(preds, list)
    assert len(preds) <= 5


def test_recency_frequency_weights_sum_to_one():
    """Constructor must raise when alpha + beta != 1.0"""
    with pytest.raises(ValueError, match="must equal 1.0"):
        RecencyFrequencyPolicy(alpha=0.7, beta=0.7)


def test_recency_frequency_score_ordering():
    """Most recently accessed app should rank highest when alpha=1.0 (pure recency)."""
    policy = RecencyFrequencyPolicy(alpha=1.0, beta=0.0)
    # Access old_app once, then recent_app many times
    policy.update({"app_id": "com.old.app"})
    # recent_app accessed 5 times in a row — dominates both recency and freq
    for _ in range(5):
        policy.update({"app_id": "com.recent.app"})
    preds = policy.predict_next_apps("", {})
    assert preds[0] == "com.recent.app"


def test_first_order_markov_trains():
    policy = FirstOrderMarkovPolicy()
    events = _make_events(50)
    policy.train(events)
    assert policy.is_trained


def test_first_order_markov_predicts_after_training():
    policy = FirstOrderMarkovPolicy()
    events = _make_events(50)
    policy.train(events)
    preds = policy.predict_next_apps("com.instagram.android", {})
    assert isinstance(preds, list)
    assert len(preds) <= 5


def test_first_order_markov_empty_before_training():
    policy = FirstOrderMarkovPolicy()
    preds = policy.predict_next_apps("com.anything", {})
    assert preds == []


def test_second_order_markov_trains():
    policy = SecondOrderMarkovPolicy()
    events = _make_events(50)
    policy.train(events)
    assert policy.is_trained


def test_second_order_markov_predicts():
    policy = SecondOrderMarkovPolicy()
    events = _make_events(50)
    policy.train(events)
    # Simulate prev/curr tracking via update
    policy.update({"app_id": "com.instagram.android"})
    preds = policy.predict_next_apps("com.whatsapp", {})
    assert isinstance(preds, list)


def test_second_order_markov_falls_back_without_prev():
    policy = SecondOrderMarkovPolicy()
    events = _make_events(50)
    policy.train(events)
    # No update called — no prev_app
    preds = policy.predict_next_apps("com.instagram.android", {})
    assert isinstance(preds, list)


def test_all_policies_have_get_name():
    policies = [
        RandomPolicy(), LRUPolicy(), LFUPolicy(), MRUPolicy(),
        FrequencyPolicy(), RecencyFrequencyPolicy(),
        FirstOrderMarkovPolicy(), SecondOrderMarkovPolicy(),
        GraphMindRLPolicy(),
    ]
    for p in policies:
        assert isinstance(p.get_name(), str)
        assert len(p.get_name()) > 0


def test_all_policies_reset_without_crash():
    policies = [
        RandomPolicy(), LRUPolicy(), LFUPolicy(), MRUPolicy(),
        FrequencyPolicy(), RecencyFrequencyPolicy(),
        FirstOrderMarkovPolicy(), SecondOrderMarkovPolicy(),
        GraphMindRLPolicy(),
    ]
    for p in policies:
        p.reset()  # Should not raise


def test_markov1_transition_probability():
    policy = FirstOrderMarkovPolicy()
    events = [
        {"app_id": "A"}, {"app_id": "B"}, {"app_id": "A"},
        {"app_id": "B"}, {"app_id": "A"}, {"app_id": "C"},
    ]
    policy.train(events)
    # A→B twice, A→C once: P(B|A) = 2/3 ≈ 0.667
    prob_ab = policy.get_transition_probability("A", "B")
    prob_ac = policy.get_transition_probability("A", "C")
    assert abs(prob_ab - 2/3) < 1e-6
    assert abs(prob_ac - 1/3) < 1e-6
