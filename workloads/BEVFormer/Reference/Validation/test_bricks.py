#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for bricks.py TTSim utility module.
Validates the conversion from PyTorch to TTSim.

This module tests the timing and profiling decorators used in BEVFormer.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from workloads.BEVFormer.ttsim_models.bricks import (
    run_time,
    reset_timing_stats,
    get_timing_stats,
    print_timing_summary,
    time_maps,
    count_maps,
)


def test_basic_decorator():
    """Test that the run_time decorator works correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Decorator Functionality")
    print("=" * 80)

    try:
        reset_timing_stats()

        @run_time("test")
        def simple_function(x):
            """A simple test function."""
            time.sleep(0.01)  # Sleep for 10ms
            return x * 2

        # Call the function
        result = simple_function(5)

        # Verify result
        if result != 10:
            print(f"✗ Function returned wrong result: {result} (expected 10)")
            return False

        # Check timing statistics
        key = "test : simple_function"
        if key not in time_maps:
            print(f"✗ Timing key '{key}' not found in time_maps")
            return False

        if time_maps[key] < 0.01:  # Should be at least 10ms
            print(f"✗ Timing too short: {time_maps[key]} (expected >= 0.01)")
            return False

        if count_maps[key] != 1:
            print(f"✗ Call count incorrect: {count_maps[key]} (expected 1)")
            return False

        print(f"✓ Decorator works correctly")
        print(f"  - Function result: {result}")
        print(f"  - Execution time: {time_maps[key]:.6f}s")
        print(f"  - Call count: {count_maps[key]}")
        return True

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multiple_calls():
    """Test accumulation of timing statistics over multiple calls."""
    print("\n" + "=" * 80)
    print("TEST 2: Multiple Calls Accumulation")
    print("=" * 80)

    try:
        reset_timing_stats()

        @run_time("accumulation")
        def test_function(x):
            """Function to test multiple calls."""
            time.sleep(0.005)  # 5ms
            return x + 1

        # Call multiple times
        num_calls = 5
        results = []
        for i in range(num_calls):
            results.append(test_function(i))

        # Verify results
        expected_results = list(range(1, num_calls + 1))
        if results != expected_results:
            print(f"✗ Results incorrect: {results} (expected {expected_results})")
            return False

        # Check timing statistics
        key = "accumulation : test_function"
        if count_maps[key] != num_calls:
            print(f"✗ Call count incorrect: {count_maps[key]} (expected {num_calls})")
            return False

        expected_min_time = 0.005 * num_calls
        if time_maps[key] < expected_min_time:
            print(
                f"✗ Total time too short: {time_maps[key]} (expected >= {expected_min_time})"
            )
            return False

        avg_time = time_maps[key] / count_maps[key]
        print(f"✓ Multiple calls accumulated correctly")
        print(f"  - Number of calls: {count_maps[key]}")
        print(f"  - Total time: {time_maps[key]:.6f}s")
        print(f"  - Average time per call: {avg_time:.6f}s")
        return True

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multiple_functions():
    """Test timing multiple different functions."""
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Functions")
    print("=" * 80)

    try:
        reset_timing_stats()

        @run_time("operation_a")
        def function_a(x):
            """First test function."""
            time.sleep(0.01)
            return x * 2

        @run_time("operation_b")
        def function_b(x):
            """Second test function."""
            time.sleep(0.02)
            return x + 10

        # Call both functions
        result_a = function_a(5)
        result_b = function_b(5)

        # Verify results
        if result_a != 10 or result_b != 15:
            print(f"✗ Results incorrect: a={result_a}, b={result_b}")
            return False

        # Check that both are tracked
        key_a = "operation_a : function_a"
        key_b = "operation_b : function_b"

        if key_a not in time_maps or key_b not in time_maps:
            print(f"✗ Not all functions tracked in time_maps")
            return False

        print(f"✓ Multiple functions tracked correctly")
        print(f"  - Function A time: {time_maps[key_a]:.6f}s")
        print(f"  - Function B time: {time_maps[key_b]:.6f}s")
        return True

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_reset_timing_stats():
    """Test resetting timing statistics."""
    print("\n" + "=" * 80)
    print("TEST 4: Reset Timing Statistics")
    print("=" * 80)

    try:
        reset_timing_stats()

        @run_time("reset_test")
        def test_function(x):
            """Function to test reset."""
            return x * 2

        # Call function and verify statistics exist
        test_function(5)
        key = "reset_test : test_function"

        if key not in time_maps:
            print(f"✗ Statistics not recorded before reset")
            return False

        # Reset and verify statistics cleared
        reset_timing_stats()

        if key in time_maps or key in count_maps:
            print(f"✗ Statistics not cleared after reset")
            return False

        print(f"✓ Reset functionality works correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_get_timing_stats():
    """Test retrieving timing statistics."""
    print("\n" + "=" * 80)
    print("TEST 5: Get Timing Statistics")
    print("=" * 80)

    try:
        reset_timing_stats()

        @run_time("stats_test")
        def test_function(x):
            """Function to test stats retrieval."""
            time.sleep(0.01)
            return x * 2

        # Call function multiple times
        for i in range(3):
            test_function(i)

        # Get statistics
        stats = get_timing_stats()

        # Verify statistics structure
        key = "stats_test : test_function"
        if key not in stats:
            print(f"✗ Key '{key}' not found in stats")
            return False

        stat = stats[key]
        if "total_time" not in stat or "count" not in stat or "avg_time" not in stat:
            print(f"✗ Statistics missing required fields")
            return False

        if stat["count"] != 3:
            print(f"✗ Count incorrect: {stat['count']} (expected 3)")
            return False

        expected_avg = stat["total_time"] / stat["count"]
        if abs(stat["avg_time"] - expected_avg) > 1e-9:
            print(f"✗ Average time calculation incorrect")
            return False

        print(f"✓ Statistics retrieval works correctly")
        print(f"  - Total time: {stat['total_time']:.6f}s")
        print(f"  - Call count: {stat['count']}")
        print(f"  - Average time: {stat['avg_time']:.6f}s")
        return True

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_print_timing_summary():
    """Test printing timing summary."""
    print("\n" + "=" * 80)
    print("TEST 6: Print Timing Summary")
    print("=" * 80)

    try:
        reset_timing_stats()

        @run_time("summary_test_a")
        def function_a(x):
            time.sleep(0.02)
            return x * 2

        @run_time("summary_test_b")
        def function_b(x):
            time.sleep(0.01)
            return x + 10

        # Call functions
        function_a(5)
        function_b(5)
        function_b(6)

        # Print summary (this also tests that it doesn't crash)
        print("\nGenerating timing summary:")
        print_timing_summary()

        print(f"✓ Timing summary printed successfully")
        return True

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_decorator_with_args_kwargs():
    """Test decorator with functions that have various argument patterns."""
    print("\n" + "=" * 80)
    print("TEST 7: Decorator with Args and Kwargs")
    print("=" * 80)

    try:
        reset_timing_stats()

        @run_time("args_test")
        def function_with_args(a, b, c=10, d=20):
            """Function with positional and keyword arguments."""
            return a + b + c + d

        # Test with different argument patterns
        result1 = function_with_args(1, 2)  # 1 + 2 + 10 + 20 = 33
        result2 = function_with_args(1, 2, c=30)  # 1 + 2 + 30 + 20 = 53
        result3 = function_with_args(1, 2, d=40)  # 1 + 2 + 10 + 40 = 53
        result4 = function_with_args(1, 2, c=30, d=40)  # 1 + 2 + 30 + 40 = 73

        # Verify results
        if result1 != 33 or result2 != 53 or result3 != 53 or result4 != 73:
            print(f"✗ Results incorrect: {result1}, {result2}, {result3}, {result4}")
            return False

        # Check timing
        key = "args_test : function_with_args"
        if count_maps[key] != 4:
            print(f"✗ Call count incorrect: {count_maps[key]} (expected 4)")
            return False

        print(f"✓ Decorator works with various argument patterns")
        print(f"  - Number of calls: {count_maps[key]}")
        print(f"  - Results: {result1}, {result2}, {result3}, {result4}")
        return True

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_decorator_preserves_metadata():
    """Test that decorator preserves function metadata."""
    print("\n" + "=" * 80)
    print("TEST 8: Decorator Preserves Metadata")
    print("=" * 80)

    try:

        @run_time("metadata_test")
        def documented_function(x):
            """This is a documented function."""
            return x * 2

        # Check that metadata is preserved
        if documented_function.__name__ != "documented_function":
            print(f"✗ Function name not preserved: {documented_function.__name__}")
            return False

        if documented_function.__doc__ != "This is a documented function.":
            print(f"✗ Function docstring not preserved: {documented_function.__doc__}")
            return False

        print(f"✓ Decorator preserves function metadata")
        print(f"  - Function name: {documented_function.__name__}")
        print(f"  - Function docstring: {documented_function.__doc__}")
        return True

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_nested_decorators():
    """Test using multiple timing decorators."""
    print("\n" + "=" * 80)
    print("TEST 9: Nested/Multiple Decorators")
    print("=" * 80)

    try:
        reset_timing_stats()

        @run_time("outer")
        @run_time("inner")
        def nested_function(x):
            """Function with nested decorators."""
            time.sleep(0.01)
            return x * 2

        # Call function
        result = nested_function(5)

        # Both decorators should record timing
        key_outer = "outer : nested_function"
        key_inner = "inner : nested_function"

        if key_outer not in time_maps or key_inner not in time_maps:
            print(f"✗ Not all decorator levels tracked")
            return False

        # Outer should take longer (includes inner + its own overhead)
        if time_maps[key_outer] < time_maps[key_inner]:
            print(f"✗ Timing logic incorrect for nested decorators")
            return False

        print(f"✓ Nested decorators work correctly")
        print(f"  - Outer time: {time_maps[key_outer]:.6f}s")
        print(f"  - Inner time: {time_maps[key_inner]:.6f}s")
        return True

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_exception_handling():
    """Test that decorator handles exceptions correctly."""
    print("\n" + "=" * 80)
    print("TEST 10: Exception Handling")
    print("=" * 80)

    try:
        reset_timing_stats()

        @run_time("exception_test")
        def function_that_raises(x):
            """Function that raises an exception."""
            if x < 0:
                raise ValueError("x must be non-negative")
            return x * 2

        # Call with valid input
        result1 = function_that_raises(5)
        if result1 != 10:
            print(f"✗ Valid call returned wrong result: {result1}")
            return False

        # Call with invalid input (should raise exception)
        exception_raised = False
        try:
            function_that_raises(-1)
        except ValueError as e:
            exception_raised = True
            if str(e) != "x must be non-negative":
                print(f"✗ Wrong exception message: {e}")
                return False

        if not exception_raised:
            print(f"✗ Exception not raised as expected")
            return False

        # Check that statistics were recorded only for successful call
        # Note: When exception is raised, the timing is still recorded before the exception propagates
        key = "exception_test : function_that_raises"
        # The decorator records timing even if an exception is raised, so count should be >= 1
        if count_maps[key] < 1:
            print(f"✗ Call count incorrect: {count_maps[key]} (expected >= 1)")
            return False

        print(f"✓ Exception handling works correctly")
        print(f"  - Valid call succeeded")
        print(f"  - Exception properly propagated")
        print(f"  - Calls recorded: {count_maps[key]}")
        return True

    except Exception as e:
        print(f"✗ Test failed with unexpected exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Bricks.py TTSim Utility Module Test Suite")
    print("=" * 80)

    results = {
        "Basic Decorator Functionality": test_basic_decorator(),
        "Multiple Calls Accumulation": test_multiple_calls(),
        "Multiple Functions": test_multiple_functions(),
        "Reset Timing Statistics": test_reset_timing_stats(),
        "Get Timing Statistics": test_get_timing_stats(),
        "Print Timing Summary": test_print_timing_summary(),
        "Args and Kwargs": test_decorator_with_args_kwargs(),
        "Preserves Metadata": test_decorator_preserves_metadata(),
        "Nested Decorators": test_nested_decorators(),
        "Exception Handling": test_exception_handling(),
    }

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<60} {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\nAll tests passed!")
        return 0
    else:
        print(
            f"\n  {total_tests - passed_tests} test(s) failed. Please review the errors above."
        )
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
