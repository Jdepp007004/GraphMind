"""
src/cli/wizard.py

Step-by-step guided Samsung connection wizard.
9 interactive steps guiding the user from zero to live telemetry stream.
"""

import logging
import os
import sys
import time
from typing import Optional

from src.cli.device_setup import (
    detect_platform, find_adb, get_adb_version, get_adb_install_instructions,
    print_samsung_setup_instructions, TROUBLESHOOTING_STEPS,
    verify_device_permissions
)

logger = logging.getLogger(__name__)

GRAPHMIND_BANNER = """
 ╔═══════════════════════════════════════════════════════════╗
 ║         GraphMind — Samsung Device Connection Wizard       ║
 ║         Iteration 2 — Real Device Telemetry Mode           ║
 ╚═══════════════════════════════════════════════════════════╝
"""

STEP_HEADER = "\n{'=' * 60}\n"


def _print_banner() -> None:
    print(GRAPHMIND_BANNER)


def _step_header(step_num: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'=' * 60}")


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _warn(msg: str) -> None:
    print(f"  [!!] {msg}")


def _info(msg: str) -> None:
    print(f"       {msg}")


def _prompt(msg: str) -> str:
    """Print a prompt and return user input (stripped)."""
    print(f"\n  > {msg} ", end="", flush=True)
    try:
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        return ""


class WizardResult:
    """Contains the outcome of a completed wizard run."""

    def __init__(self) -> None:
        self.success: bool = False
        self.platform: str = ""
        self.adb_path: Optional[str] = None
        self.adb_version: str = ""
        self.device_serial: Optional[str] = None
        self.device_model: str = ""
        self.android_version: str = ""
        self.oneui_version: str = ""
        self.battery_pct: float = 0.0
        self.foreground_app: str = ""
        self.permissions: dict = {}
        self.error: Optional[str] = None

    def __repr__(self) -> str:
        return (f"WizardResult(success={self.success}, device={self.device_model}, "
                f"android={self.android_version}, serial={self.device_serial})")


class SamsungConnectionWizard:
    """
    Interactive 9-step wizard that guides the user through:
      1. Platform detection
      2. ADB verification
      3. Samsung developer mode setup instructions
      4. Device detection via adb devices
      5. Troubleshooting (if device not found)
      6. Wireless pairing guidance
      7. Permission verification
      8. Live telemetry stream startup
      9. Dashboard launch
    """

    def __init__(self, non_interactive: bool = False,
                 user_id: str = "user_00") -> None:
        """
        non_interactive: if True, skip all input() calls (for testing).
        user_id: GraphMind user ID to attach telemetry to.
        """
        self.non_interactive = non_interactive
        self.user_id = user_id
        self.result = WizardResult()

    def run(self) -> WizardResult:
        """Run all 9 wizard steps. Returns WizardResult."""
        _print_banner()
        print("  This wizard will connect your Samsung device to GraphMind.")
        print("  Estimated time: 3-5 minutes\n")

        try:
            # Step 1: Platform
            self._step1_detect_platform()
            # Step 2: ADB
            if not self._step2_verify_adb():
                return self.result
            # Step 3: Samsung setup
            self._step3_samsung_instructions()
            # Step 4: Detect device
            device_found = self._step4_detect_device()
            # Step 5: Troubleshoot if not found
            if not device_found:
                device_found = self._step5_troubleshoot()
            if not device_found:
                # Step 6: Wireless pairing
                device_found = self._step6_wireless_pairing()
            if not device_found:
                _warn("Could not detect device. Proceeding in simulation mode.")
                self.result.error = "device_not_found"
                self.result.success = False
                # Still proceed with steps 7-9 in simulation mode
            # Step 7: Permissions
            self._step7_verify_permissions()
            # Step 8: Start telemetry
            self._step8_start_telemetry()
            # Step 9: Launch dashboard
            self._step9_launch_dashboard()

        except KeyboardInterrupt:
            print("\n\n  Wizard interrupted. Goodbye!")
            self.result.error = "interrupted"
            return self.result

        return self.result

    # ── Step Implementations ────────────────────────────────────────────────

    def _step1_detect_platform(self) -> None:
        _step_header(1, "Platform Detection")
        platform_name = detect_platform()
        self.result.platform = platform_name
        _ok(f"Detected platform: {platform_name}")
        _info("GraphMind supports Windows, Linux, and macOS.")

    def _step2_verify_adb(self) -> bool:
        _step_header(2, "ADB Verification")
        adb_path = find_adb()
        if not adb_path:
            _warn("ADB (Android Debug Bridge) not found in PATH.")
            print()
            print(get_adb_install_instructions(self.result.platform))
            print()
            if not self.non_interactive:
                _prompt("Press ENTER after installing ADB, or Ctrl+C to exit")
            # Re-check after user installs
            adb_path = find_adb()
            if not adb_path:
                _warn("ADB still not found. Please install it and restart the wizard.")
                self.result.error = "adb_not_found"
                return False

        self.result.adb_path = adb_path
        ok, version = get_adb_version(adb_path)
        self.result.adb_version = version
        _ok(f"ADB found: {adb_path}")
        _ok(f"Version: {version}")
        return True

    def _step3_samsung_instructions(self) -> None:
        _step_header(3, "Samsung Developer Mode Setup")
        print_samsung_setup_instructions()
        if not self.non_interactive:
            _prompt("Press ENTER when Developer Options, USB Debugging, and "
                    "Wireless Debugging are enabled on your device")

    def _step4_detect_device(self) -> bool:
        _step_header(4, "Device Detection")
        _info("Running: adb devices")
        print()
        try:
            from src.android.adb_connector import ADBConnector
            from src.android.device_detector import DeviceDetector
            connector = ADBConnector(self.result.adb_path)
            detector = DeviceDetector(connector)
            devices = connector.list_devices()
            if not devices:
                _warn("No devices detected. Make sure USB cable is connected.")
                return False
            print(f"  Found {len(devices)} device(s):")
            for d in devices:
                print(f"    - {d['serial']} ({d['state']})")
            authorized = [d for d in devices if d["state"] == "device"]
            if not authorized:
                _warn("Device found but not authorized. Check device screen for 'Allow USB Debugging' prompt.")
                return False
            # Probe Samsung info
            device_info = detector.detect_samsung()
            if device_info:
                self.result.device_serial = device_info.serial
                self.result.device_model = device_info.model
                self.result.android_version = device_info.android_release
                self.result.oneui_version = device_info.oneui_version
                _ok(f"Samsung device detected: {device_info.model}")
                _ok(f"Android {device_info.android_release} | OneUI {device_info.oneui_version or 'N/A'}")
                _ok(f"Serial: {device_info.serial}")
                if not device_info.is_supported:
                    _warn("Device runs Android <11. Some features may not be available.")
            else:
                # Use first authorized device
                self.result.device_serial = authorized[0]["serial"]
                _warn("Non-Samsung device detected. GraphMind works best on Samsung Galaxy devices.")
                _ok(f"Using device: {self.result.device_serial}")
            return True
        except RuntimeError as e:
            _warn(f"ADB error: {e}")
            return False
        except Exception as e:
            logger.error(f"Device detection error: {e}")
            _warn(f"Device detection failed: {e}")
            return False

    def _step5_troubleshoot(self) -> bool:
        _step_header(5, "Troubleshooting")
        print(TROUBLESHOOTING_STEPS)
        if self.non_interactive:
            return False
        response = _prompt("Press ENTER to retry device detection, or type 'skip' to proceed to wireless pairing")
        if response.lower() == "skip":
            return False
        return self._step4_detect_device()

    def _step6_wireless_pairing(self) -> bool:
        _step_header(6, "Wireless ADB Pairing")
        print("""
  On your Samsung device:
    1. Settings > Developer Options > Wireless Debugging
    2. Tap "Pair device with pairing code"
    3. Note the IP address, port, and 6-digit pairing code shown
""")
        if self.non_interactive:
            return False
        address = _prompt("Enter pairing address (e.g., 192.168.1.5:37123) or ENTER to skip")
        if not address:
            return False
        code = _prompt("Enter 6-digit pairing code")
        if not code:
            return False
        try:
            from src.android.adb_connector import ADBConnector
            connector = ADBConnector(self.result.adb_path)
            ok, msg = connector.pair_device(address, code)
            if ok:
                _ok(f"Paired: {msg}")
                # Now connect
                connect_addr = _prompt("Enter connection address (e.g., 192.168.1.5:5555)")
                if connect_addr:
                    ok2, msg2 = connector.connect_device(connect_addr)
                    if ok2:
                        _ok(f"Connected: {msg2}")
                        self.result.device_serial = connect_addr
                        return True
                    else:
                        _warn(f"Connection failed: {msg2}")
            else:
                _warn(f"Pairing failed: {msg}")
        except Exception as e:
            _warn(f"Wireless pairing error: {e}")
        return False

    def _step7_verify_permissions(self) -> None:
        _step_header(7, "Permission Verification")
        if not self.result.device_serial or not self.result.adb_path:
            _info("Skipping permission check (no device connected).")
            return
        checks = verify_device_permissions(self.result.adb_path, self.result.device_serial)
        self.result.permissions = checks
        for name, status in checks.items():
            label = name.replace("_", " ").title()
            if status:
                _ok(f"{label}: Enabled")
            else:
                _warn(f"{label}: Not detected (may still work)")

    def _step8_start_telemetry(self) -> None:
        _step_header(8, "Starting Live Telemetry Stream")
        if not self.result.device_serial:
            _info("No device connected. GraphMind will run in simulation mode.")
            return

        print(f"  Initializing telemetry for {self.user_id} on {self.result.device_serial}...")
        try:
            from src.android.adb_connector import ADBConnector
            from src.android.device_detector import DeviceDetector
            from src.android.battery_collector import BatteryCollector
            from src.android.usage_stats_collector import UsageStatsCollector
            connector = ADBConnector(self.result.adb_path)

            # Collect initial snapshot
            battery_col = BatteryCollector(connector, serial=self.result.device_serial)
            usage_col = UsageStatsCollector(connector, serial=self.result.device_serial)

            battery_data = battery_col.collect()
            foreground = usage_col.get_foreground_app()

            self.result.battery_pct = battery_data.get("battery_pct", 0.0)
            self.result.foreground_app = foreground or "unknown"

            _ok(f"Battery: {self.result.battery_pct:.0f}%")
            _ok(f"Foreground App: {self.result.foreground_app}")
            _ok("Telemetry stream initialized.")
            self.result.success = True
        except Exception as e:
            _warn(f"Telemetry startup error: {e}")
            _info("Falling back to simulation mode.")

    def _step9_launch_dashboard(self) -> None:
        _step_header(9, "Dashboard Launch")
        print("""
  GraphMind Dashboard is ready to launch.

  Connected Device:  """ + (self.result.device_model or "Simulation Mode") + """
  Android Version:   """ + (self.result.android_version or "N/A") + """
  OneUI Version:     """ + (self.result.oneui_version or "N/A") + """
  Battery:           """ + (f"{self.result.battery_pct:.0f}%" if self.result.battery_pct else "N/A") + """
  Foreground App:    """ + (self.result.foreground_app or "N/A") + """

  Dashboard URL: http://localhost:8501

  Run the dashboard with:
    streamlit run src/dashboard/app.py
""")

        if not self.non_interactive:
            launch = _prompt("Launch dashboard now? (y/N)")
            if launch.lower() in ("y", "yes"):
                try:
                    import subprocess
                    subprocess.Popen(
                        [sys.executable, "-m", "streamlit", "run",
                         "src/dashboard/app.py"],
                        cwd=os.getcwd()
                    )
                    _ok("Dashboard starting at http://localhost:8501")
                    time.sleep(1)
                except Exception as e:
                    _warn(f"Could not auto-launch dashboard: {e}")
                    _info("Run manually: streamlit run src/dashboard/app.py")
        else:
            _info("Non-interactive mode: skipping dashboard auto-launch.")
