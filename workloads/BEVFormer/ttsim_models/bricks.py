#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim version of bricks.py utility module.

This module provides timing and profiling utilities for BEVFormer model operations.
Converted from PyTorch implementation with PyTorch-specific dependencies removed.

Original file: projects/mmdet3d_plugin/models/utils/bricks.py
"""

import functools
import time
from collections import defaultdict
from loguru import logger

# Global dictionaries to track timing statistics
time_maps = defaultdict(lambda: 0.0)
count_maps = defaultdict(lambda: 0.0)


def run_time(name):
    """
    Decorator for timing function execution.

    This decorator measures the execution time of functions and accumulates statistics.
    It's useful for profiling model operations during development and debugging.

    Args:
        name (str): A descriptive name for the operation being timed.
                   This will be used as a prefix in the timing output.

    Returns:
        decorator: A decorator function that can be applied to any function.

    Example:
        @run_time("forward_pass")
        def my_function(x):
            return x * 2

        # Output: "forward_pass : my_function takes up 0.000123"

    Notes:
        - Unlike the PyTorch version, this TTSim version does not call cuda.synchronize()
          since TTSim operations are CPU-based or use different acceleration mechanisms.
        - Timing statistics are accumulated across multiple calls to the same function.
        - The printed time is the average execution time per call.
    """

    def middle(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Record start time
            start = time.time()

            # Execute the function
            res = fn(*args, **kwargs)

            # Record end time and update statistics
            elapsed = time.time() - start
            key = "%s : %s" % (name, fn.__name__)
            time_maps[key] += elapsed
            count_maps[key] += 1

            # Print average execution time
            avg_time = time_maps[key] / count_maps[key]
            logger.debug("%s : %s takes up %f " % (name, fn.__name__, avg_time))

            return res

        return wrapper

    return middle


def reset_timing_stats():
    """
    Reset all accumulated timing statistics.

    This is useful when you want to start fresh timing measurements,
    for example at the beginning of a new benchmark run.
    """
    time_maps.clear()
    count_maps.clear()


def get_timing_stats():
    """
    Get a copy of current timing statistics.

    Returns:
        dict: A dictionary containing timing statistics with the following structure:
              {
                  'operation_name': {
                      'total_time': float,  # Total accumulated time in seconds
                      'count': int,         # Number of calls
                      'avg_time': float     # Average time per call in seconds
                  }
              }
    """
    stats = {}
    for key in time_maps.keys():
        stats[key] = {
            "total_time": time_maps[key],
            "count": count_maps[key],
            "avg_time": time_maps[key] / count_maps[key],
        }
    return stats


def print_timing_summary():
    """
    Print a summary of all accumulated timing statistics.

    This provides a nice overview of all timed operations, sorted by total time.
    """
    logger.info("\n" + "=" * 80)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 80)

    if not time_maps:
        logger.info("No timing data collected.")
        return

    # Sort by total time (descending)
    sorted_keys = sorted(time_maps.keys(), key=lambda k: time_maps[k], reverse=True)

    logger.info(f"{'Operation':<50} {'Calls':>8} {'Total (s)':>12} {'Avg (s)':>12}")
    logger.info("-" * 80)

    for key in sorted_keys:
        total_time = time_maps[key]
        count = count_maps[key]
        avg_time = total_time / count
        logger.debug(
            f"{key:<50} {count:>8} {total_time:>12.6f} {avg_time:>12.6f}"
        )

    logger.info("=" * 80 + "\n")
