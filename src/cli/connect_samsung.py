"""
src/cli/connect_samsung.py

Entry point for the Samsung connection wizard.

Usage:
    python -m src.cli.connect_samsung
    python -m src.cli.connect_samsung --non-interactive
    python -m src.cli.connect_samsung --user user_01
"""

import argparse
import logging
import sys

from src.cli.wizard import SamsungConnectionWizard


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s | %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.connect_samsung",
        description="GraphMind Samsung Device Connection Wizard",
        epilog="Connects a real Samsung Galaxy device to GraphMind's live telemetry pipeline."
    )
    parser.add_argument(
        "--non-interactive", "-n",
        action="store_true",
        default=False,
        help="Skip all interactive prompts (for testing/CI)"
    )
    parser.add_argument(
        "--user", "-u",
        default="user_00",
        help="GraphMind user ID to attach telemetry to (default: user_00)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        default=False,
        help="Run Samsung/ADB diagnostics and write results/device_report.json"
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point. Returns exit code."""
    args = parse_args()
    setup_logging(args.verbose)

    if args.doctor:
        from scripts.device_validation import write_device_report
        report = write_device_report()
        print("GraphMind Samsung Doctor")
        print(f"ADB: {report.get('adb_status')}")
        print(f"Device detected: {report.get('device_detected')}")
        print(f"Telemetry: {report.get('telemetry_status')}")
        print(f"Dashboard: {report.get('dashboard_status')}")
        print("Report: results/device_report.json")
        return 0 if report.get("checks", {}).get("adb") else 1

    wizard = SamsungConnectionWizard(
        non_interactive=args.non_interactive,
        user_id=args.user
    )
    result = wizard.run()

    print()
    if result.success:
        print("=" * 60)
        print("ITERATION 2 COMPLETE")
        print()
        print("Run:")
        print()
        print("  python -m src.cli.connect_samsung")
        print()
        print("The wizard will guide Samsung device setup and launch")
        print("GraphMind with live telemetry.")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        if result.error == "adb_not_found":
            print("  Setup incomplete: ADB not found.")
            print("  Install Android Platform Tools and restart the wizard.")
        elif result.error == "device_not_found":
            print("  No Samsung device detected.")
            print("  GraphMind will run in simulation mode.")
            print("  Run: streamlit run src/dashboard/app.py")
        elif result.error == "interrupted":
            print("  Wizard interrupted.")
        else:
            print("  Setup completed with warnings.")
            print("  Run: streamlit run src/dashboard/app.py")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
