"""
src/android/adb_connector.py

Low-level ADB subprocess wrapper. All ADB commands go through here.
Supports USB and Wireless ADB (adb pair / adb connect).
"""

import logging
import subprocess
import shutil
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class ADBConnector:
    """
    Wraps the adb CLI. All methods return (success: bool, output: str).
    Never raises — callers check the bool.
    """

    def __init__(self, adb_path: Optional[str] = None) -> None:
        """
        adb_path: explicit path to adb binary. If None, locate via shutil.which.
        Raises RuntimeError if adb is not found.
        """
        if adb_path:
            self.adb_path = adb_path
        else:
            found = shutil.which("adb")
            if not found:
                raise RuntimeError(
                    "adb not found in PATH. Install Android Platform Tools:\n"
                    "  Windows: https://developer.android.com/studio/releases/platform-tools\n"
                    "  Linux:   sudo apt install adb\n"
                    "  macOS:   brew install android-platform-tools"
                )
            self.adb_path = found
        logger.debug(f"ADBConnector using adb at: {self.adb_path}")

    def is_available(self) -> bool:
        """Return True if adb binary is accessible."""
        ok, _ = self.run_command(["version"])
        return ok

    def get_version(self) -> str:
        """Return adb version string, or empty string on failure."""
        ok, out = self.run_command(["version"])
        if ok:
            for line in out.splitlines():
                if "Android Debug Bridge" in line:
                    return line.strip()
        return ""

    def list_devices(self) -> List[dict]:
        """
        Run `adb devices` and parse output.
        Returns list of dicts: [{'serial': str, 'state': str}, ...]
        States: 'device' (authorized), 'unauthorized', 'offline'
        """
        ok, output = self.run_command(["devices"])
        if not ok:
            return []
        devices = []
        lines = output.strip().splitlines()
        for line in lines[1:]:  # skip header
            line = line.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                devices.append({"serial": parts[0].strip(), "state": parts[1].strip()})
        return devices

    def pair_device(self, address: str, pairing_code: str) -> Tuple[bool, str]:
        """
        Run `adb pair <address> <code>` for wireless ADB pairing (Android 11+).
        address: 'ip:port', pairing_code: 6-digit code shown on device.
        Returns (success, message).
        """
        ok, out = self.run_command(["pair", address, pairing_code], timeout=30)
        if ok and ("Successfully" in out or "paired" in out.lower()):
            return True, f"Paired with {address}"
        return False, out.strip() or "Pairing failed"

    def connect_device(self, address: str) -> Tuple[bool, str]:
        """
        Run `adb connect <address>` for wireless ADB.
        address: 'ip:port'
        Returns (success, message).
        """
        ok, out = self.run_command(["connect", address], timeout=15)
        if ok and ("connected" in out.lower() or "already connected" in out.lower()):
            return True, out.strip()
        return False, out.strip() or "Connection failed"

    def shell(self, command: str, serial: Optional[str] = None,
              timeout: int = 10) -> Tuple[bool, str]:
        """
        Run `adb [-s serial] shell <command>`.
        Returns (success, stdout_output).
        """
        args = []
        if serial:
            args += ["-s", serial]
        args += ["shell", command]
        return self.run_command(args, timeout=timeout)

    def run_command(self, args: List[str], timeout: int = 10) -> Tuple[bool, str]:
        """
        Run `adb <args>` and return (success, stdout).
        Catches all exceptions — never raises.
        """
        cmd = [self.adb_path] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout + result.stderr
            success = result.returncode == 0
            if not success:
                logger.debug(f"ADB command {cmd} returned code {result.returncode}: {output.strip()[:200]}")
            return success, output
        except subprocess.TimeoutExpired:
            logger.warning(f"ADB command timed out: {cmd}")
            return False, "timeout"
        except FileNotFoundError:
            logger.error(f"ADB binary not found at {self.adb_path}")
            return False, "adb not found"
        except Exception as e:
            logger.error(f"ADB command error: {e}")
            return False, str(e)
