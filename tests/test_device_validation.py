"""
Tests for device validation report and doctor CLI.
"""

from unittest.mock import MagicMock

from scripts.device_validation import collect_device_report, write_device_report
from src.android.device_detector import DeviceInfo
from src.cli.connect_samsung import parse_args


class FakeConnector:
    def is_available(self):
        return True

    def get_version(self):
        return "Android Debug Bridge version 1.0.41"

    def list_devices(self):
        return [{"serial": "RF8M33ABCDE", "state": "device"}]

    def shell(self, command, serial=None, timeout=10):
        if command == "getprop":
            return True, "\n".join([
                "[ro.product.brand]: [samsung]",
                "[ro.product.model]: [Galaxy S24]",
                "[ro.build.version.release]: [14]",
                "[ro.build.version.sdk]: [34]",
                "[ro.build.version.oneui]: [6.1]",
            ])
        if "adb_wifi_enabled" in command:
            return True, "0"
        if "dumpsys battery" in command:
            return True, "level: 88"
        return True, ""


def test_device_report_with_fake_samsung_connector():
    report = collect_device_report(FakeConnector())
    assert report["adb_status"] == "available"
    assert report["device_detected"] is True
    assert report["device"]["brand"] == "samsung"
    assert report["device"]["model"] == "Galaxy S24"
    assert report["telemetry_status"] == "ok"
    assert report["checks"]["telemetry"] is True


def test_write_device_report(tmp_path):
    out = tmp_path / "device_report.json"
    report = write_device_report(str(out), FakeConnector())
    assert out.exists()
    assert report["checks"]["adb"] is True


def test_cli_doctor_arg(monkeypatch):
    monkeypatch.setattr("sys.argv", ["connect_samsung", "--doctor"])
    args = parse_args()
    assert args.doctor is True
