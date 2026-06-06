"""tests/test_sensitivity_model.py — SensitivityModel flush rule correctness."""
import pytest
from unittest.mock import MagicMock, patch
from src.security.sensitivity_model import SensitivityModel
from config import settings


def _make_model():
    model = SensitivityModel()
    # Patch ClassificationGuard to return deterministic categories
    _CATEGORY_MAP = {
        "com.gaming.app": "gaming",
        "com.social.app": "social",
        "com.bank.app":   "financial",
        "com.health.app": "health",
        "unknown.app":    "utility",
    }
    model._guard.classify = lambda app_id, **kw: _CATEGORY_MAP.get(app_id, "utility")
    # Clear cache so patched classify is used
    model._sensitivity_cache.clear()
    return model



def test_sensitivity_level_public():
    model = _make_model()
    assert model.get_sensitivity("com.gaming.app") == settings.SENSITIVITY_PUBLIC


def test_sensitivity_level_personal():
    model = _make_model()
    assert model.get_sensitivity("com.social.app") == settings.SENSITIVITY_PERSONAL


def test_sensitivity_level_financial():
    model = _make_model()
    assert model.get_sensitivity("com.bank.app") == settings.SENSITIVITY_FINANCIAL


def test_sensitivity_level_health():
    model = _make_model()
    assert model.get_sensitivity("com.health.app") == settings.SENSITIVITY_HEALTH


def test_should_flush_on_level_drop():
    """HEALTH → GAMING: flush required."""
    model = _make_model()
    flush, reason = model.should_flush("com.health.app", "com.gaming.app")
    assert flush is True
    assert "Sensitivity drop" in reason


def test_should_flush_on_level_rise_no_flush():
    """GAMING → FINANCIAL: no flush (level rises)."""
    model = _make_model()
    flush, reason = model.should_flush("com.gaming.app", "com.bank.app")
    assert flush is False
    assert reason == ""


def test_should_flush_same_level_no_flush():
    """FINANCIAL → FINANCIAL: no flush."""
    model = _make_model()
    flush, reason = model.should_flush("com.bank.app", "com.bank.app")
    assert flush is False


def test_on_app_launched_first_event_no_flush():
    """First event never triggers flush (no previous context)."""
    model = _make_model()
    result = model.on_app_launched("com.bank.app")
    assert result["flushed"] is False
    assert result["prev_app"] is None


def test_on_app_launched_flush_triggered():
    model = _make_model()
    model.on_app_launched("com.health.app")   # set context to HEALTH
    result = model.on_app_launched("com.gaming.app")  # drop to PUBLIC
    assert result["flushed"] is True


def test_on_app_launched_no_flush_on_rise():
    model = _make_model()
    model.on_app_launched("com.gaming.app")   # PUBLIC
    result = model.on_app_launched("com.bank.app")  # rise to FINANCIAL
    assert result["flushed"] is False


def test_flush_rate_calculation():
    model = _make_model()
    model.on_app_launched("com.health.app")   # transition 1 (no prev app, no flush)
    model.on_app_launched("com.gaming.app")   # transition 2: flush (HEALTH→PUBLIC)
    model.on_app_launched("com.social.app")   # transition 3: no flush (PUBLIC→PERSONAL)
    # 1 flush out of 3 transitions = 1/3 ≈ 0.333
    assert model.flush_rate() == pytest.approx(1/3, abs=1e-6)


def test_flush_rate_zero_transitions():
    model = _make_model()
    assert model.flush_rate() == 0.0


def test_get_flush_events_audit_log():
    model = _make_model()
    model.on_app_launched("com.health.app")
    model.on_app_launched("com.gaming.app")
    events = model.get_flush_events()
    assert len(events) == 1
    assert events[0]["prev_app"] == "com.health.app"
    assert events[0]["next_app"] == "com.gaming.app"


def test_sensitivity_cache_is_populated():
    model = _make_model()
    model.get_sensitivity("com.bank.app")
    assert "com.bank.app" in model._sensitivity_cache


def test_reset_clears_session_state():
    model = _make_model()
    model.on_app_launched("com.health.app")
    model.on_app_launched("com.gaming.app")
    model.reset()
    assert model._total_transitions == 0
    assert len(model._flush_events) == 0
    assert model._current_app_id is None


def test_reset_preserves_sensitivity_cache():
    """Sensitivity cache survives reset (no reason to recompute)."""
    model = _make_model()
    model.get_sensitivity("com.bank.app")
    model.reset()
    assert "com.bank.app" in model._sensitivity_cache


def test_summary_structure():
    model = _make_model()
    model.on_app_launched("com.health.app")
    model.on_app_launched("com.gaming.app")
    summary = model.summary()
    assert "total_transitions" in summary
    assert "flush_count" in summary
    assert "flush_rate" in summary
    assert "sensitivity_distribution" in summary
    assert "PUBLIC(0)" in summary["sensitivity_distribution"]


def test_memory_manager_flush_called_on_flush():
    from unittest.mock import MagicMock
    model = _make_model()
    mock_mm = MagicMock()
    model.on_app_launched("com.health.app")
    model.on_app_launched("com.gaming.app", memory_manager=mock_mm)
    mock_mm.flush_hot_tier.assert_called_once()
    mock_mm.flush_warm_tier.assert_called_once()
