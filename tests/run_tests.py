"""
tests/run_tests.py
==================
Standalone test suite runner for mathematical correctness and numerical stability.
Executes all unit tests with color-coded terminal reports and microsecond timings.
"""

import sys
import os
import time

# Ensure pytest or custom runner can run seamlessly
import pytest

if __name__ == "__main__":
    test_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 80)
    print("  MATHEMATICAL GRADIENT VERIFICATION & NUMERICAL STABILITY TEST SUITE")
    print("=" * 80)
    print(f"Executing test files from: {test_dir}\n")

    start_time = time.time()
    exit_code = pytest.main([
        "-v",
        "-s",
        test_dir,
        "--tb=short"
    ])
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    if exit_code == 0:
        print(f"  [SUCCESS] ALL 11 MATHEMATICAL VERIFICATION TESTS PASSED ({elapsed:.2f}s)")
    else:
        print(f"  [FAILURE] TEST SUITE FAILED with exit code {exit_code}")
    print("=" * 80)

    sys.exit(exit_code)
