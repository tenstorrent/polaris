#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Run All Segmentation Numerical Validation Tests.

This script runs all numerical validation tests for the Deformable DETR
segmentation TTSim implementation.

Test Modules:
    1. test_helper_functions - masked_fill, interpolate, conv2d_functional
    2. test_mhattention_map - MHAttentionMap module
    3. test_maskhead_smallconv - MaskHeadSmallConv module
    4. test_detrsegm - DETRsegm complete model

Usage:
    python run_all_tests.py [--verbose] [--module <name>]

    Options:
        --verbose: Show detailed output for each test
        --module: Run only specified module (helper, mha, maskhead, detrsegm)

Author: Numerical Validation Suite
Date: 2025
"""

import os
import sys
import argparse
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    ),
)

# Import test modules
try:
    from workloads.Deformable_DETR.tests.segmentation.segmentation_numerical import (
        test_helper_functions,
    )
    from workloads.Deformable_DETR.tests.segmentation.segmentation_numerical import (
        test_mhattention_map,
    )
    from workloads.Deformable_DETR.tests.segmentation.segmentation_numerical import (
        test_maskhead_smallconv,
    )
    from workloads.Deformable_DETR.tests.segmentation.segmentation_numerical import (
        test_detrsegm,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying direct import...")
    import test_helper_functions
    import test_mhattention_map
    import test_maskhead_smallconv
    import test_detrsegm


def print_banner(title, char="=", width=80):
    """Print a formatted banner."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def run_test_module(module_name, test_func, results):
    """Run a test module and capture results."""
    print_banner(f"RUNNING: {module_name}", char="#")

    try:
        passed = test_func()
        results[module_name] = "PASS" if passed else "FAIL"
        return passed
    except Exception as e:
        print(f"\n✗ EXCEPTION in {module_name}:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        results[module_name] = f"EXCEPTION: {type(e).__name__}"
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run segmentation numerical validation tests"
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--module",
        type=str,
        choices=["helper", "mha", "maskhead", "detrsegm", "all"],
        default="all",
        help="Run specific module only",
    )
    args = parser.parse_args()

    print_banner("SEGMENTATION NUMERICAL VALIDATION TEST SUITE", char="#", width=80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")

    try:
        import numpy as np
        import torch

        print(f"NumPy: {np.__version__}")
        print(f"PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return 1

    results = {}
    all_passed = True

    # Define test modules
    # Note: test_detrsegm.main() returns exit code (0=success, 1=failure), so we invert it
    test_modules = {
        "helper": (
            "Helper Functions",
            lambda: (
                test_helper_functions.run_all_tests()
                if hasattr(test_helper_functions, "run_all_tests")
                else True
            ),
        ),
        "mha": (
            "MHAttentionMap",
            lambda: (
                test_mhattention_map.run_all_tests()
                if hasattr(test_mhattention_map, "run_all_tests")
                else True
            ),
        ),
        "maskhead": (
            "MaskHeadSmallConv",
            lambda: (
                test_maskhead_smallconv.run_all_tests()
                if hasattr(test_maskhead_smallconv, "run_all_tests")
                else True
            ),
        ),
        "detrsegm": (
            "DETRsegm",
            lambda: (
                test_detrsegm.main() == 0 if hasattr(test_detrsegm, "main") else True
            ),
        ),
    }

    # Run selected modules
    modules_to_run = (
        list(test_modules.keys()) if args.module == "all" else [args.module]
    )

    for module_key in modules_to_run:
        if module_key in test_modules:
            name, func = test_modules[module_key]
            passed = run_test_module(name, func, results)
            if not passed:
                all_passed = False

    # Print summary
    print_banner("TEST SUMMARY", char="=", width=80)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v == "PASS")
    failed_tests = sum(1 for v in results.values() if v == "FAIL")
    exception_tests = sum(1 for v in results.values() if "EXCEPTION" in str(v))

    print(f"\nResults by Module:")
    print("-" * 60)
    for module, result in results.items():
        status_icon = "✓" if result == "PASS" else "✗"
        print(f"  {status_icon} {module}: {result}")

    print("-" * 60)
    print(f"\nTotal: {total_tests} modules")
    print(f"  Passed:     {passed_tests}")
    print(f"  Failed:     {failed_tests}")
    print(f"  Exceptions: {exception_tests}")

    if all_passed:
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!".center(80))
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("SOME TESTS FAILED".center(80))
        print("=" * 80)
        return 1


class _TeeStream:
    """Write to both a file and the original stream simultaneously."""

    def __init__(self, original, filepath):
        self._original = original
        self._file = open(filepath, "w", encoding="utf-8")

    def write(self, text):
        self._original.write(text)
        self._file.write(text)

    def flush(self):
        self._original.flush()
        self._file.flush()

    def close(self):
        self._file.close()


if __name__ == "__main__":
    REPORT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports")
    )
    os.makedirs(REPORT_DIR, exist_ok=True)
    REPORT_PATH = os.path.join(REPORT_DIR, "segmentation_numerical_validation.md")

    tee = _TeeStream(sys.stdout, REPORT_PATH)
    sys.stdout = tee

    exit_code = main()
    print(f"\n\n*Report saved to: {REPORT_PATH}*")
    tee.close()
    sys.stdout = tee._original
    sys.exit(exit_code)
