"""
scripts/run_iteration3_validation.py

Runs Iteration 3 credibility and validation checks.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


COMMANDS = [
    ["python", "-m", "pytest", "-q",
     "tests/test_benchmark_fairness.py",
     "tests/test_benchmark_provenance.py",
     "tests/test_rl_evaluation.py",
     "tests/test_scale.py",
     "tests/test_security_hardening.py",
     "tests/test_event_validation.py",
     "tests/test_device_validation.py"],
    ["python", "scripts/run_scale_test.py"],
    ["python", "scripts/device_validation.py"],
]


def run_validation() -> int:
    for command in COMMANDS:
        print("RUN", " ".join(command))
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            print("FAILED", " ".join(command))
            return result.returncode
    print()
    print("ITERATION 3 VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_validation())
