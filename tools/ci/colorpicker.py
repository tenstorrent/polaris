#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Script to pick a color based on value and thresholds, or based on conclusion/exit code.

Usage:
    colorpicker.py --value NUM --highcolor COLOR threshold1 color1 threshold2 color2 ...
    colorpicker.py --conclusion {success,failure,cancelled,skipped}
    colorpicker.py --exitcode NUM
"""

import argparse
import sys
from typing import List, Tuple


def validate_mutually_exclusive(args) -> None:
    """Validate that exactly one of value, conclusion, or exitcode is provided."""
    provided_args = []
    if args.value is not None:
        provided_args.append('--value')
    if args.conclusion is not None:
        provided_args.append('--conclusion')
    if args.exitcode is not None:
        provided_args.append('--exitcode')
    
    if len(provided_args) == 0:
        print("error: exactly one of --value, --conclusion, or --exitcode must be provided", file=sys.stderr)
        sys.exit(1)
    elif len(provided_args) > 1:
        print(f"error: only one of --value, --conclusion, or --exitcode can be provided, got: {', '.join(provided_args)}", file=sys.stderr)
        sys.exit(1)


def validate_value_mode_args(args, threshold_color_pairs: List[str]) -> None:
    """Validate arguments for --value mode."""
    if args.highcolor is None:
        print("error: --highcolor is required when using --value", file=sys.stderr)
        sys.exit(1)
    
    if len(threshold_color_pairs) == 0:
        print("error: at least one threshold/color pair is required when using --value", file=sys.stderr)
        sys.exit(1)
    
    if len(threshold_color_pairs) % 2 != 0:
        print("error: threshold/color pairs must be even in number", file=sys.stderr)
        sys.exit(1)


def validate_simple_mode_args(args, threshold_color_pairs: List[str]) -> None:
    """Validate arguments for --conclusion or --exitcode modes."""
    if args.highcolor is not None:
        print("error: --highcolor cannot be used with --conclusion or --exitcode", file=sys.stderr)
        sys.exit(1)
    
    if len(threshold_color_pairs) > 0:
        print("error: threshold/color pairs cannot be used with --conclusion or --exitcode", file=sys.stderr)
        sys.exit(1)


def parse_threshold_color_pairs(pairs: List[str]) -> List[Tuple[float, str]]:
    """Parse and validate threshold/color pairs."""
    result = []
    for i in range(0, len(pairs), 2):
        threshold_str = pairs[i]
        color = pairs[i + 1]
        
        try:
            threshold = float(threshold_str)
        except ValueError:
            print(f"error: threshold '{threshold_str}' must be a number", file=sys.stderr)
            sys.exit(1)
        
        if not color.strip():
            print(f"error: color cannot be empty (threshold: {threshold})", file=sys.stderr)
            sys.exit(1)
        
        result.append((threshold, color))
    
    return result


def get_color_for_value(value: float, highcolor: str, threshold_pairs: List[Tuple[float, str]]) -> str:
    """Get color based on value and thresholds."""
    for threshold, color in threshold_pairs:
        if value < threshold:
            return color
    return highcolor


def get_color_for_conclusion(conclusion: str) -> str:
    """Get color based on conclusion value."""
    if conclusion == 'success':
        return 'brightgreen'
    else:  # failure, cancelled, skipped
        return 'red'


def get_color_for_exitcode(exitcode: int) -> str:
    """Get color based on exit code."""
    if exitcode == 0:
        return 'brightgreen'
    else:
        return 'red'


def main():
    parser = argparse.ArgumentParser(
        description='Pick a color based on value and thresholds, or based on conclusion/exit code.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --value 75 --highcolor green 50 red 85 yellow
  %(prog)s --conclusion success
  %(prog)s --exitcode 0
        """
    )
    
    # Mutually exclusive main arguments
    parser.add_argument('--value', type=float, help='Numeric value to compare against thresholds')
    parser.add_argument('--conclusion', choices=['success', 'failure', 'cancelled', 'skipped'],
                        help='Conclusion status (success=brightgreen, others=red)')
    parser.add_argument('--exitcode', type=int, help='Exit code (0=brightgreen, others=red)')
    
    # Arguments for --value mode
    parser.add_argument('--highcolor', type=str, help='Color to use if value exceeds all thresholds (required with --value)')
    
    # Threshold/color pairs (only for --value mode)
    parser.add_argument('threshold_color_pairs', nargs='*', 
                        help='Threshold/color pairs: threshold1 color1 threshold2 color2 ...')
    
    args = parser.parse_args()
    
    # Validate mutually exclusive arguments
    validate_mutually_exclusive(args)
    
    # Handle different modes
    if args.value is not None:
        # Value mode - requires highcolor and threshold pairs
        validate_value_mode_args(args, args.threshold_color_pairs)
        threshold_pairs = parse_threshold_color_pairs(args.threshold_color_pairs)
        color = get_color_for_value(args.value, args.highcolor, threshold_pairs)
        print(color)
        
    elif args.conclusion is not None:
        # Conclusion mode - no additional arguments allowed
        validate_simple_mode_args(args, args.threshold_color_pairs)
        color = get_color_for_conclusion(args.conclusion)
        print(color)
        
    elif args.exitcode is not None:
        # Exit code mode - no additional arguments allowed
        validate_simple_mode_args(args, args.threshold_color_pairs)
        color = get_color_for_exitcode(args.exitcode)
        print(color)


if __name__ == '__main__':
    main()
