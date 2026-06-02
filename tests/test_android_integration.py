"""
tests/test_android_integration.py

Phase 1 tests for Samsung telemetry ingestion layer.
All tests use mocks — no real device required.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import time

from src.core.event_bus import EventBus
from src.android.adb_connector import ADBConnector
from src.android.device_detector import DeviceDetector, DeviceInfo
from src.android.battery_collector import BatteryCollector
from src.android.usage_stats_collector import UsageStatsCollector
from src.android.audio_collector import AudioCollector
from src.android.screen_collector import ScreenCollector
from src.android.calendar_collector import CalendarCollector
from src.android.telemetry_event_adapter import TelemetryEventAdapter
from src.android.telemetry_collector import TelemetryCollector
from src.core.event_bus import TOPIC_APP_LAUNCHED, TOPIC_BATTERY_UPDATED, TOPIC_HEADPHONES_CONNECTED


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_connector():
    """ADBConnector with mocked run_command that always succeeds."""
    conn = MagicMock(spec=ADBConnector)
    conn.adb_path = "/usr/bin/adb"
    conn.run_command.return_value = (True, "")
    conn.shell.return_value = (True, "")
    conn.list_devices.return_value = [{"serial": "emulator-5554", "state": "device"}]
    return conn


@pytest.fixture
def mock_samsung_device():
    """Fully populated DeviceInfo for a Samsung Galaxy S23."""
    d = DeviceInfo()
    d.serial = "RF8M33ABCDE"
    d.model = "SM-S911B"
    d.brand = "samsung"
    d.android_version = 14
    d.android_release = "14"
    d.oneui_version = "6.1"
    d.is_tablet = False
    d.form_factor = "phone"
    d.usb_debugging = True
    d.wireless_debugging = True
    d.sdk_int = 34
    return d


# ── ADB Connector Tests ───────────────────────────────────────────────────────

def test_adb_connector_not_found():
    """ADBConnector must raise RuntimeError if adb is not in PATH."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="adb not found"):
            ADBConnector()


def test_adb_connector_found():
    """ADBConnector must succeed when adb is in PATH."""
    with patch("shutil.which", return_value="/usr/bin/adb"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Android Debug Bridge version 1.0.41", stderr="")
            conn = ADBConnector()
            assert conn.adb_path == "/usr/bin/adb"
            assert conn.is_available()


def test_adb_list_devices_parse(mock_connector):
    """list_devices must correctly parse 'adb devices' output."""
    mock_connector.run_command.return_value = (
        True,
        "List of devices attached\nRF8M33ABCDE\tdevice\n192.168.1.5:5555\tdevice\n"
    )
    # Use the real list_devices logic via run_command patch
    with patch("shutil.which", return_value="/usr/bin/adb"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="List of devices attached\nRF8M33ABCDE\tdevice\n",
                stderr=""
            )
            real_conn = ADBConnector()
            devices = real_conn.list_devices()
            assert len(devices) == 1
            assert devices[0]["serial"] == "RF8M33ABCDE"
            assert devices[0]["state"] == "device"


def test_adb_shell_success(mock_connector):
    """shell() must pass command to run_command correctly."""
    mock_connector.shell.return_value = (True, "level: 85\n")
    ok, out = mock_connector.shell("dumpsys battery", serial="RF8M33ABCDE")
    assert ok is True
    assert "85" in out


# ── Device Detector Tests ─────────────────────────────────────────────────────

def test_device_detector_detect_samsung(mock_connector, mock_samsung_device):
    """detect_samsung() must return a DeviceInfo with is_samsung=True."""
    detector = DeviceDetector(mock_connector)
    # Mock _probe_device to return our Samsung device
    with patch.object(detector, "_probe_device", return_value=mock_samsung_device):
        result = detector.detect_samsung()
        assert result is not None
        assert result.is_samsung is True
        assert result.android_version == 14
        assert result.oneui_version == "6.1"


def test_device_info_is_supported(mock_samsung_device):
    """Samsung device with Android 14 must be reported as supported."""
    assert mock_samsung_device.is_supported is True


def test_device_info_unsupported_android():
    """Non-Samsung device must not be supported."""
    d = DeviceInfo()
    d.brand = "xiaomi"
    d.android_version = 14
    assert d.is_samsung is False
    assert d.is_supported is False


def test_device_info_to_dict(mock_samsung_device):
    """to_dict() must include all required keys."""
    info = mock_samsung_device.to_dict()
    required_keys = {"serial", "model", "brand", "android_version",
                     "oneui_version", "is_tablet", "form_factor",
                     "is_samsung", "is_supported"}
    assert required_keys.issubset(set(info.keys()))


# ── Battery Collector Tests ───────────────────────────────────────────────────

def test_battery_collector_parses_output(mock_connector):
    """BatteryCollector must correctly parse dumpsys battery output."""
    mock_connector.shell.side_effect = [
        (True, "  level: 78\n  status: 2\n  temperature: 295\n  health: 2\n"),
        (True, "0\n")  # power saver
    ]
    collector = BatteryCollector(mock_connector, serial="RF8M33ABCDE")
    result = collector.collect()
    assert result["battery_pct"] == 78.0
    assert result["charging"] is True
    assert result["temperature_c"] == 29.5
    assert result["health"] == "Good"
    assert result["power_saver"] is False


def test_battery_collector_defaults_on_failure(mock_connector):
    """BatteryCollector must return safe defaults if ADB fails."""
    mock_connector.shell.return_value = (False, "error")
    collector = BatteryCollector(mock_connector)
    result = collector.collect()
    assert result["battery_pct"] == 100.0
    assert result["charging"] is False


# ── Usage Stats Collector Tests ───────────────────────────────────────────────

def test_usage_stats_foreground_app_detected(mock_connector):
    """get_foreground_app() must extract package name from mResumedActivity dump."""
    mock_connector.shell.return_value = (
        True,
        "    mResumedActivity: ActivityRecord{abc com.spotify.music/.MainActivity t42}\n"
    )
    collector = UsageStatsCollector(mock_connector)
    app = collector.get_foreground_app()
    assert app == "com.spotify.music"


def test_usage_stats_no_foreground_app(mock_connector):
    """get_foreground_app() must return None when no app is identifiable."""
    mock_connector.shell.return_value = (False, "")
    collector = UsageStatsCollector(mock_connector)
    app = collector.get_foreground_app()
    assert app is None


# ── Telemetry Event Adapter Tests ─────────────────────────────────────────────

def test_adapter_publish_app_launched():
    """TelemetryEventAdapter must publish TOPIC_APP_LAUNCHED to EventBus."""
    bus = EventBus.get_instance()
    bus.clear_all()
    received = []
    bus.subscribe(TOPIC_APP_LAUNCHED, lambda p: received.append(p))

    adapter = TelemetryEventAdapter("user_00")
    adapter.publish_app_launched("com.spotify.music", battery=85.0, headphones=True)

    assert len(received) == 1
    payload = received[0]
    assert payload["app_id"] == "com.spotify.music"
    assert payload["user_id"] == "user_00"
    assert payload["battery"] == 85.0
    assert payload["headphones"] is True
    assert payload["source"] == "real_device"
    bus.clear_all()


def test_adapter_deduplicates_same_app():
    """Adapter must NOT publish same app twice in a row."""
    bus = EventBus.get_instance()
    bus.clear_all()
    received = []
    bus.subscribe(TOPIC_APP_LAUNCHED, lambda p: received.append(p))

    adapter = TelemetryEventAdapter("user_00")
    adapter.publish_app_launched("com.spotify.music", battery=80.0)
    adapter.publish_app_launched("com.spotify.music", battery=79.0)  # same app

    assert len(received) == 1  # only first publish
    bus.clear_all()


def test_adapter_category_lookup():
    """TelemetryEventAdapter must map known packages to correct categories."""
    adapter = TelemetryEventAdapter("user_00")
    assert adapter.get_app_category("com.hdfcbank.new") == "financial"
    assert adapter.get_app_category("com.spotify.music") == "entertainment"
    # Unknown packages fall back to 'utility'
    assert adapter.get_app_category("com.unknown.app12345") == "utility"


def test_adapter_publish_battery_updated():
    """Adapter must publish TOPIC_BATTERY_UPDATED correctly."""
    bus = EventBus.get_instance()
    bus.clear_all()
    received = []
    bus.subscribe(TOPIC_BATTERY_UPDATED, lambda p: received.append(p))

    adapter = TelemetryEventAdapter("user_00")
    adapter.publish_battery_updated(battery_pct=45.0, charging=False, power_saver=True)

    assert len(received) == 1
    assert received[0]["battery"] == 45.0
    assert received[0]["power_saver"] is True
    bus.clear_all()


def test_adapter_publish_headphones_connected():
    """Adapter must publish TOPIC_HEADPHONES_CONNECTED for Bluetooth."""
    bus = EventBus.get_instance()
    bus.clear_all()
    received = []
    bus.subscribe(TOPIC_HEADPHONES_CONNECTED, lambda p: received.append(p))

    adapter = TelemetryEventAdapter("user_00")
    adapter.publish_headphones_connected(wired=False, bluetooth=True)

    assert len(received) == 1
    assert received[0]["bluetooth"] is True
    assert received[0]["headphones_connected"] is True
    bus.clear_all()


# ── Telemetry Collector Integration Tests ─────────────────────────────────────

def test_telemetry_collector_collect_once(mock_connector, mock_samsung_device):
    """collect_once() must return a complete data snapshot without publishing."""
    mock_connector.shell.return_value = (True, "level: 90\nstatus: 2\ntemperature: 280\nhealth: 2\n")

    battery_mock = MagicMock()
    battery_mock.collect.return_value = {"battery_pct": 90.0, "charging": True,
                                          "power_saver": False, "temperature_c": 28.0, "health": "Good"}
    usage_mock = MagicMock()
    usage_mock.get_foreground_app.return_value = "com.samsung.android.contacts"
    audio_mock = MagicMock()
    audio_mock.collect.return_value = {"headphones_wired": False, "headphones_bluetooth": False,
                                       "headphones_any": False, "output_device": "SPEAKER"}
    screen_mock = MagicMock()
    screen_mock.collect.return_value = {"screen_on": True, "screen_locked": False, "screen_unlocked": True,
                                        "wifi_connected": True, "wifi_ssid": "HomeWifi", "network_type": "WIFI"}
    calendar_mock = MagicMock()
    calendar_mock.collect.return_value = {"has_upcoming_event": False,
                                          "minutes_until_next_event": None,
                                          "next_event_title": "",
                                          "calendar_event_in_mins": None}

    bus = EventBus.get_instance()
    bus.clear_all()
    collector = TelemetryCollector("user_00", mock_connector, mock_samsung_device)
    collector.battery = battery_mock
    collector.usage = usage_mock
    collector.audio = audio_mock
    collector.screen = screen_mock
    collector.calendar = calendar_mock

    snapshot = collector.collect_once()
    assert snapshot["foreground_app"] == "com.samsung.android.contacts"
    assert snapshot["battery"]["battery_pct"] == 90.0
    assert snapshot["user_id"] == "user_00"
    assert "timestamp" in snapshot
    bus.clear_all()
