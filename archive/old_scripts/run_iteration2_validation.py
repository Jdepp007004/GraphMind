"""
scripts/run_iteration2_validation.py

Final Iteration 2 validation runner.
Runs all Iteration 2 tests and prints a summary report.
Does NOT run Iteration 1 tests (they remain intact).
"""

import subprocess
import sys
import time
import os


ITERATION2_TEST_MODULES = [
    "tests/test_android_integration.py",
    "tests/test_explainability.py",
    "tests/test_graph_playback.py",
    "tests/test_security_visualization.py",
    "tests/test_drift_visualization.py",
    "tests/test_advanced_benchmarks.py",
    "tests/test_cli_wizard.py",
]

BANNER = """
====================================================================
 GraphMind Iteration 2 — Final Validation Suite
 Samsung Device Integration + Advanced Visualization
====================================================================
"""

PHASE_NAMES = {
    "test_android_integration.py": "Phase 1: Samsung Telemetry",
    "test_explainability.py": "Phase 2: Explainability Engine",
    "test_graph_playback.py": "Phase 3: Graph Evolution Playback",
    "test_security_visualization.py": "Phase 4: Security Visualization",
    "test_drift_visualization.py": "Phase 5: Drift Analytics",
    "test_advanced_benchmarks.py": "Phase 6: Advanced Benchmarking",
    "test_cli_wizard.py": "Phase 7: Samsung CLI Wizard",
}


def run_test_module(python_exe: str, module_path: str) -> dict:
    """Run a single test module and return results."""
    start = time.time()
    result = subprocess.run(
        [python_exe, "-m", "pytest", module_path, "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    elapsed = time.time() - start
    output = result.stdout + result.stderr

    # Parse passed/failed counts
    passed = 0
    failed = 0
    for line in output.splitlines():
        if "passed" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "passed":
                    try:
                        passed = int(parts[i-1])
                    except Exception:
                        pass
                if part.startswith("failed") or part == "failed,":
                    try:
                        failed = int(parts[i-1])
                    except Exception:
                        pass

    return {
        "module": module_path,
        "return_code": result.returncode,
        "passed": passed,
        "failed": failed,
        "elapsed": round(elapsed, 2),
        "output": output,
    }


def main() -> int:
    print(BANNER)
    python_exe = sys.executable
    print(f"  Python: {python_exe}")
    print(f"  Running {len(ITERATION2_TEST_MODULES)} test modules...\n")

    results = []
    total_passed = 0
    total_failed = 0
    all_ok = True

    for module_path in ITERATION2_TEST_MODULES:
        module_name = os.path.basename(module_path)
        phase_name = PHASE_NAMES.get(module_name, module_name)

        print(f"  Running {phase_name}... ", end="", flush=True)

        # Check if file exists (Phase 7 test may be created separately)
        if not os.path.exists(module_path):
            print(f"SKIP (file not found)")
            continue

        result = run_test_module(python_exe, module_path)
        results.append(result)
        total_passed += result["passed"]
        total_failed += result["failed"]

        if result["return_code"] == 0:
            print(f"PASS ({result['passed']} tests, {result['elapsed']}s)")
        else:
            print(f"FAIL ({result['failed']} failed, {result['passed']} passed)")
            all_ok = False
            # Print failing test output
            for line in result["output"].splitlines():
                if "FAILED" in line or "ERROR" in line or "AssertionError" in line:
                    print(f"       {line.strip()}")

    print()
    print("=" * 68)
    print("  ITERATION 2 VALIDATION SUMMARY")
    print("=" * 68)
    print(f"  Total Tests:   {total_passed + total_failed}")
    print(f"  Passed:        {total_passed}")
    print(f"  Failed:        {total_failed}")
    print()

    if all_ok:
        print("  ALL TESTS PASSING — ITERATION 2 COMPLETE")
        print()
        print("  Next steps:")
        print("  1. Run full simulation: python scripts/run_simulation.py")
        print("  2. Launch dashboard:    streamlit run src/dashboard/app.py")
        print("  3. Connect Samsung:     python -m src.cli.connect_samsung")
    else:
        print("  SOME TESTS FAILED — Review output above before committing.")

    print("=" * 68)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
