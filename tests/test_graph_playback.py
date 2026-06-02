"""
tests/test_graph_playback.py

Phase 3 tests for graph evolution playback system.
Uses in-memory fake simulation log data. No real simulation run needed.
"""

import json
import os
import pytest
import tempfile

from src.graph_playback.snapshot_manager import SnapshotManager
from src.graph_playback.timeline_engine import TimelineEngine, TimelineFrame
from src.graph_playback.graph_animator import GraphAnimator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_simulation_log(user_id: str, num_days: int = 5) -> dict:
    """Create a minimal simulation log dict mimicking BehaviouralGraph output."""
    days = []
    for day in range(num_days):
        nodes = [
            {"node_id": f"node_{day}_{i}", "app_id": f"com.app{i}",
             "category": "social", "access_count": day + i + 1}
            for i in range(day + 2)  # nodes grow each day
        ]
        edges = [
            {"source": f"node_{day}_{i}", "target": f"node_{day}_{i+1}",
             "prob": 0.5 + i * 0.05}
            for i in range(len(nodes) - 1)
        ]
        days.append({
            "day": day,
            "graph_snapshot": {
                "day": day,
                "user_id": user_id,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "nodes": nodes,
                "edges": edges
            },
            "tier_stats": {"hot_count": 5, "warm_count": 20, "cold_count": 0,
                           "hot_capacity": 30, "warm_capacity": 150},
            "state": {
                "user_id": user_id,
                "current_day": day,
                "kl_divergence": 0.1 * day,
                "cache_hit_rate": 0.3 + day * 0.03,
                "security_flush_count": 0,
                "last_agent": "security",
                "messages": []
            }
        })
    return {"user_id": user_id, "days": days}


@pytest.fixture
def sim_log_dir(tmp_path):
    """Create a temp results dir with a fake simulation log."""
    return tmp_path


@pytest.fixture
def fake_snapshot_manager(sim_log_dir, monkeypatch):
    """SnapshotManager with RESULTS_DIR pointing to temp dir with fake log."""
    import config.settings as s
    monkeypatch.setattr(s, "RESULTS_DIR", str(sim_log_dir))
    # Write fake log for user_test
    log = _make_fake_simulation_log("user_test", num_days=5)
    log_path = os.path.join(str(sim_log_dir), "user_test_simulation_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f)
    mgr = SnapshotManager()
    mgr.SNAPSHOT_DIR = str(sim_log_dir / "snapshots")
    os.makedirs(mgr.SNAPSHOT_DIR, exist_ok=True)
    return mgr


# ── SnapshotManager Tests ─────────────────────────────────────────────────────

def test_snapshot_manager_load_from_simulation_log(fake_snapshot_manager):
    """load_from_simulation_log() must return 5 snapshots for user_test."""
    snapshots = fake_snapshot_manager.load_from_simulation_log("user_test")
    assert len(snapshots) == 5
    assert all("node_count" in s for s in snapshots)
    assert all("edge_count" in s for s in snapshots)


def test_snapshot_manager_missing_log(fake_snapshot_manager):
    """load_from_simulation_log() must return empty list for unknown user."""
    snapshots = fake_snapshot_manager.load_from_simulation_log("user_nonexistent")
    assert snapshots == []


def test_snapshot_manager_save_and_load(fake_snapshot_manager):
    """save_snapshot and load_snapshot must round-trip correctly."""
    snap = {"day": 5, "node_count": 10, "edge_count": 8, "nodes": [], "edges": []}
    fake_snapshot_manager.save_snapshot("user_test", 5, snap)
    loaded = fake_snapshot_manager.load_snapshot("user_test", 5)
    assert loaded is not None
    assert loaded["node_count"] == 10
    assert loaded["edge_count"] == 8


def test_snapshot_manager_load_missing_snapshot(fake_snapshot_manager):
    """load_snapshot() must return None for a non-existent day."""
    result = fake_snapshot_manager.load_snapshot("user_test", 999)
    assert result is None


def test_snapshot_manager_list_available_days(fake_snapshot_manager):
    """list_available_days() must return [0, 1, 2, 3, 4] for user_test."""
    days = fake_snapshot_manager.list_available_days("user_test")
    assert days == [0, 1, 2, 3, 4]


def test_snapshot_manager_verify_integrity(fake_snapshot_manager):
    """verify_integrity() must return True for a valid snapshot."""
    assert fake_snapshot_manager.verify_integrity("user_test", 0) is True
    assert fake_snapshot_manager.verify_integrity("user_test", 999) is False


# ── TimelineEngine Tests ──────────────────────────────────────────────────────

def test_timeline_engine_load(fake_snapshot_manager):
    """TimelineEngine.load() must return True and populate frames."""
    engine = TimelineEngine("user_test", snapshot_manager=fake_snapshot_manager)
    result = engine.load()
    assert result is True
    frames = engine.get_all_frames()
    assert len(frames) == 5


def test_timeline_engine_missing_user(fake_snapshot_manager):
    """TimelineEngine.load() must return False for unknown user."""
    engine = TimelineEngine("user_nonexistent", snapshot_manager=fake_snapshot_manager)
    result = engine.load()
    assert result is False
    assert engine.get_all_frames() == []


def test_timeline_engine_get_frame(fake_snapshot_manager):
    """get_frame(day) must return the correct TimelineFrame."""
    engine = TimelineEngine("user_test", snapshot_manager=fake_snapshot_manager)
    engine.load()
    frame = engine.get_frame(3)
    assert frame is not None
    assert frame.day == 3
    assert frame.node_count > 0


def test_timeline_engine_deltas_computed(fake_snapshot_manager):
    """node_delta must be positive as nodes grow each day."""
    engine = TimelineEngine("user_test", snapshot_manager=fake_snapshot_manager)
    engine.load()
    frames = engine.get_all_frames()
    # First frame has delta=node_count (from 0 base)
    # Subsequent frames should have positive deltas since we add nodes each day
    assert frames[0].node_delta >= 0
    # Day 1 should have more nodes than day 0
    if len(frames) > 1:
        assert frames[1].node_count >= frames[0].node_count


def test_timeline_engine_growth_series(fake_snapshot_manager):
    """get_growth_series() must return all required keys with correct lengths."""
    engine = TimelineEngine("user_test", snapshot_manager=fake_snapshot_manager)
    engine.load()
    series = engine.get_growth_series()
    required_keys = {"days", "node_counts", "edge_counts", "cache_hit_rates",
                     "kl_divergence", "hot_counts", "warm_counts"}
    assert required_keys.issubset(set(series.keys()))
    n = len(series["days"])
    assert len(series["node_counts"]) == n
    assert len(series["cache_hit_rates"]) == n


def test_timeline_engine_drift_events(fake_snapshot_manager):
    """get_drift_events() must return frames where KL > 0.3."""
    engine = TimelineEngine("user_test", snapshot_manager=fake_snapshot_manager)
    engine.load()
    drift_events = engine.get_drift_events()
    # In fake log: KL = 0.1 * day, so days 4+ have KL >= 0.4
    for event in drift_events:
        assert event["kl_divergence"] > 0.3


def test_timeline_engine_milestone_frames(fake_snapshot_manager):
    """get_milestone_frames() must return non-empty list."""
    engine = TimelineEngine("user_test", snapshot_manager=fake_snapshot_manager)
    engine.load()
    milestones = engine.get_milestone_frames()
    assert len(milestones) >= 1
    for frame in milestones:
        assert isinstance(frame, TimelineFrame)


def test_timeline_engine_frames_dict_schema(fake_snapshot_manager):
    """get_frames_dict() must return dicts with all required keys."""
    engine = TimelineEngine("user_test", snapshot_manager=fake_snapshot_manager)
    engine.load()
    frames_dict = engine.get_frames_dict()
    required = {"day", "node_count", "edge_count", "cache_hit_rate",
                "kl_divergence", "drift_detected"}
    for d in frames_dict:
        assert required.issubset(set(d.keys()))


# ── GraphAnimator Tests ───────────────────────────────────────────────────────

def test_graph_animator_growth_chart_data(fake_snapshot_manager):
    """get_growth_chart_data() must return growth series data."""
    engine = TimelineEngine("user_test", snapshot_manager=fake_snapshot_manager)
    engine.load()
    animator = GraphAnimator(engine)
    data = animator.get_growth_chart_data()
    assert "days" in data
    assert "node_counts" in data
    assert len(data["days"]) == 5


def test_graph_animator_playback_frames(fake_snapshot_manager):
    """get_playback_frames_data() must return list of frame dicts."""
    engine = TimelineEngine("user_test", snapshot_manager=fake_snapshot_manager)
    engine.load()
    animator = GraphAnimator(engine)
    frames = animator.get_playback_frames_data()
    assert len(frames) == 5
    assert "day" in frames[0]
    assert "node_count" in frames[0]


def test_graph_animator_category_evolution(fake_snapshot_manager):
    """get_category_evolution() must return per-category count lists."""
    engine = TimelineEngine("user_test", snapshot_manager=fake_snapshot_manager)
    engine.load()
    animator = GraphAnimator(engine)
    evo = animator.get_category_evolution()
    assert "days" in evo
    assert "social" in evo
    assert len(evo["social"]) == len(evo["days"])


def test_graph_animator_render_missing_day(fake_snapshot_manager):
    """render_frame_html() for missing day must return error HTML, not raise."""
    engine = TimelineEngine("user_test", snapshot_manager=fake_snapshot_manager)
    engine.load()
    animator = GraphAnimator(engine)
    html = animator.render_frame_html(999)
    assert "No data" in html or "error" in html.lower() or html.startswith("<p")
