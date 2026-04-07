#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare Polaris and Profiler CSV layer sequences."""

import sys
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

# Maximum distance to search forward for matching operations
DEFAULT_MAX_SEARCH_DISTANCE = 10

# Operation type normalization mappings
OPTYPE_NORMALIZATION = {
    'layernormalization': 'layernorm',
    'reshapeview': 'reshape',
}

# Import layer extraction functions
try:
    from show_layers_polaris import layers_polaris
    from show_layers_profiler import layers_profiler
except ImportError:
    # Try relative imports
    from .show_layers_polaris import layers_polaris  # type: ignore
    from .show_layers_profiler import layers_profiler  # type: ignore


@dataclass
class ComparisonStats:
    """Statistics for layer comparison."""
    total_matches: int = 0
    name_mismatches: int = 0
    shape_mismatches: int = 0
    input_shape_mismatches: int = 0
    output_shape_mismatches: int = 0
    unmatched_polaris: int = 0
    unmatched_profiler: int = 0
    ambiguous: int = 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare Polaris and Profiler CSV layer sequences'
    )
    parser.add_argument('file1', type=str, help='First CSV file')
    parser.add_argument('file2', type=str, help='Second CSV file')
    parser.add_argument(
        '--max-search-distance',
        type=int,
        default=DEFAULT_MAX_SEARCH_DISTANCE,
        help=f'Maximum distance to search forward for matching operations (default: {DEFAULT_MAX_SEARCH_DISTANCE})'
    )
    parser.add_argument(
        '--strip-leading-ones',
        action='store_true',
        help='Strip all leading 1s from shapes instead of collapsing to single 1 (more lenient matching for broadcast dimensions)'
    )
    parser.add_argument(
        '--filter-optype',
        type=str,
        default=None,
        help='Filter to only compare layers with this operation type (case-insensitive)'
    )
    return parser.parse_args()


def normalize_optype(optype: str) -> str:
    """
    Normalize operation type name to canonical form.
    
    Handles semantic equivalents like:
    - layernormalization -> layernorm
    - reshapeview -> reshape
    
    Args:
        optype: Original operation type name
    
    Returns:
        Normalized operation type name
    """
    optype_lower = optype.lower().strip()
    return OPTYPE_NORMALIZATION.get(optype_lower, optype_lower)


def detect_file_type(filepath: str) -> Optional[str]:
    """
    Detect if file is polaris or profiler CSV.
    
    Returns:
        'polaris' if archname column found
        'profiler' if OP CODE column found
        None if neither found
    """
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            if 'archname' in headers:
                return 'polaris'
            elif 'OP CODE' in headers:
                return 'profiler'
            return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None


def normalize_shape(shape_list: List[int], strip_leading_ones: bool = False) -> List[int]:
    """
    Normalize shape by handling leading 1s.
    
    Args:
        shape_list: List of dimension sizes
        strip_leading_ones: If True, strip all leading 1s; if False, collapse to single 1
    
    Examples (strip_leading_ones=False, default):
        [1, 1, 1, 224, 224] -> [1, 224, 224]
        [2, 3, 1, 1, 5] -> [2, 3, 1, 1, 5]
        [1] -> [1]
    
    Examples (strip_leading_ones=True):
        [1, 1, 1024, 768] -> [1024, 768]
        [1024, 768] -> [1024, 768]
        [1, 1, 1] -> [1]
    """
    if not shape_list:
        return []
    
    # Find how many leading 1s we have
    leading_ones = 0
    for dim in shape_list:
        if dim == 1:
            leading_ones += 1
        else:
            break
    
    # If all are 1s, keep just one
    if leading_ones == len(shape_list):
        return [1]
    
    # Handle leading 1s based on strategy
    if leading_ones > 0:
        if strip_leading_ones:
            # Strip all leading 1s
            return shape_list[leading_ones:]
        elif leading_ones > 1:
            # Collapse multiple leading 1s to a single 1
            return [1] + shape_list[leading_ones:]
    
    return shape_list


def parse_shape(shape_str: str) -> List[int]:
    """
    Parse shape string like '1x224x224' into list of integers.
    
    Returns:
        List of dimension sizes, or empty list if parsing fails
    """
    if not shape_str or shape_str.strip() == '':
        return []
    
    try:
        parts = shape_str.split('x')
        return [int(p.strip()) for p in parts if p.strip()]
    except ValueError:
        return []


def validate_mul_compatibility(
    polaris_inputs: List[str],
    profiler_inputs: List[str],
    strip_leading_ones: bool = False
) -> Tuple[bool, str]:
    """
    Special validation for mul operations with different input representations.
    
    If polaris has more inputs than profiler, ignore additional polaris inputs
    (they may be constants or parameters not tracked in profiler).
    
    Args:
        polaris_inputs: Polaris input tensor shapes
        profiler_inputs: Profiler input tensor shapes
        strip_leading_ones: Whether to strip leading 1s
    
    Returns:
        (valid: bool, details: str)
    """
    # If lengths match, use standard comparison
    if len(polaris_inputs) == len(profiler_inputs):
        return compare_tensor_shapes(polaris_inputs, profiler_inputs, strip_leading_ones)
    
    # If polaris has more inputs, compare only the first profiler_count inputs
    if len(polaris_inputs) > len(profiler_inputs):
        polaris_subset = polaris_inputs[:len(profiler_inputs)]
        match, details = compare_tensor_shapes(polaris_subset, profiler_inputs, strip_leading_ones)
        if match:
            return True, f"mul compatible: using first {len(profiler_inputs)} of {len(polaris_inputs)} polaris inputs"
        else:
            return False, details
    
    # If profiler has more inputs, this is unexpected
    return False, f"profiler has more inputs ({len(profiler_inputs)}) than polaris ({len(polaris_inputs)})"


def validate_reshape_compatibility(
    polaris_inputs: List[str],
    polaris_outputs: List[str],
    profiler_outputs: List[str],
    strip_leading_ones: bool = False
) -> Tuple[bool, str]:
    """
    Special validation for reshape operations with different representations.
    
    Allows matching when:
    - Polaris has 2 input tensors (data + target_shape)
    - Second polaris input is 1-D (contains target shape)
    - Profiler output is 2-D after trimming leading 1s
    - First dimension of profiler output = product of first 3 dims of polaris output
    
    Args:
        polaris_inputs: Polaris input tensor shapes
        polaris_outputs: Polaris output tensor shapes
        profiler_outputs: Profiler output tensor shapes
        strip_leading_ones: Whether to strip leading 1s
    
    Returns:
        (valid: bool, details: str)
    """
    # Check if polaris has 2 inputs
    if len(polaris_inputs) != 2:
        return False, "polaris doesn't have 2 inputs"
    
    # Parse shapes
    pol_out_parsed = parse_shape(polaris_outputs[0]) if polaris_outputs else []
    prof_out_parsed = parse_shape(profiler_outputs[0]) if profiler_outputs else []
    pol_input2_parsed = parse_shape(polaris_inputs[1]) if len(polaris_inputs) > 1 else []
    
    # Normalize profiler output
    prof_out_normalized = normalize_shape(prof_out_parsed, strip_leading_ones)
    
    # Check if second polaris input is 1-D
    if len(pol_input2_parsed) != 1:
        return False, f"polaris second input not 1-D: {pol_input2_parsed}"
    
    # Check if profiler output is 2-D after normalization
    if len(prof_out_normalized) != 2:
        return False, f"profiler output not 2-D after normalization: {prof_out_normalized}"
    
    # Check if polaris output has at least 3 dimensions
    if len(pol_out_parsed) < 3:
        return False, f"polaris output has < 3 dims: {pol_out_parsed}"
    
    # Calculate product of first 3 dimensions of polaris output
    pol_first_three_product = pol_out_parsed[0] * pol_out_parsed[1] * pol_out_parsed[2]
    prof_first_dim = prof_out_normalized[0]
    
    if pol_first_three_product == prof_first_dim:
        return True, f"reshape compatible: pol {pol_out_parsed[:3]} product={pol_first_three_product} matches prof first dim={prof_first_dim}"
    else:
        return False, f"product mismatch: pol {pol_out_parsed[:3]} product={pol_first_three_product} vs prof first dim={prof_first_dim}"


def compare_tensor_shapes(
    polaris_shapes: List[str],
    profiler_shapes: List[str],
    strip_leading_ones: bool = False,
    optype: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Compare two lists of tensor shapes.
    
    Args:
        polaris_shapes: List of shape strings from polaris
        profiler_shapes: List of shape strings from profiler
        strip_leading_ones: If True, strip all leading 1s for more lenient matching
        optype: Operation type for special case handling (e.g., reshape)
    
    Returns:
        (match: bool, details: str) - True if shapes match, details about mismatch
    """
    # Parse and normalize all shapes
    polaris_normalized = [
        normalize_shape(parse_shape(s), strip_leading_ones) for s in polaris_shapes
    ]
    profiler_normalized = [
        normalize_shape(parse_shape(s), strip_leading_ones) for s in profiler_shapes
    ]
    
    # Check counts
    if len(polaris_normalized) != len(profiler_normalized):
        # Special case for reshape: polaris may have 2 inputs (data + target_shape)
        # while profiler just shows the output
        if optype and normalize_optype(optype) == 'reshape':
            # This is handled in output comparison below
            pass
        else:
            return False, f"count mismatch: {len(polaris_normalized)} vs {len(profiler_normalized)}"
    
    # Compare each tensor
    for i, (p_shape, f_shape) in enumerate(zip(polaris_normalized, profiler_normalized)):
        if p_shape != f_shape:
            p_str = 'x'.join(map(str, p_shape)) if p_shape else 'empty'
            f_str = 'x'.join(map(str, f_shape)) if f_shape else 'empty'
            return False, f"tensor {i}: {p_str} vs {f_str}"
    
    return True, ""


def find_next_match(
    layers: List[Dict[str, Any]],
    start_idx: int,
    target_optype: str,
    max_distance: Optional[int] = None
) -> Optional[int]:
    """
    Find the next occurrence of target_optype in layers starting from start_idx.
    
    Args:
        layers: List of layer dictionaries
        start_idx: Index to start searching from
        target_optype: Normalized operation type to search for
        max_distance: Maximum number of operations to search ahead (None = unlimited)
    
    Returns:
        Index of match, or None if not found within max_distance
    """
    end_idx = len(layers)
    if max_distance is not None:
        end_idx = min(end_idx, start_idx + max_distance)
    
    for i in range(start_idx, end_idx):
        if normalize_optype(layers[i]['optype']) == target_optype:
            return i
    return None


def format_shapes(shapes: List[str]) -> str:
    """Format list of shape strings for display."""
    if not shapes:
        return "[]"
    return "[" + ", ".join(shapes) + "]"


def compare_layers(
    polaris_layers: List[Dict[str, Any]],
    profiler_layers: List[Dict[str, Any]],
    max_search_distance: int = DEFAULT_MAX_SEARCH_DISTANCE,
    strip_leading_ones: bool = False
) -> ComparisonStats:
    """
    Compare two layer sequences and print results.
    
    Returns:
        ComparisonStats object with statistics
    """
    stats = ComparisonStats()
    ndx_polaris = 0
    ndx_profiler = 0
    
    while ndx_polaris < len(polaris_layers) or ndx_profiler < len(profiler_layers):
        # Case 3: One sequence exhausted
        if ndx_polaris >= len(polaris_layers):
            # Profiler has remaining entries
            layer = profiler_layers[ndx_profiler]
            print(f"⊘ [F:{layer['seqno']}] {layer['optype']} (skipped in polaris)")
            stats.unmatched_profiler += 1
            ndx_profiler += 1
            continue
        
        if ndx_profiler >= len(profiler_layers):
            # Polaris has remaining entries
            layer = polaris_layers[ndx_polaris]
            print(f"⊘ [P:{layer['seqno']}] {layer['optype']} (skipped in profiler)")
            stats.unmatched_polaris += 1
            ndx_polaris += 1
            continue
        
        # Get current layers
        p_layer = polaris_layers[ndx_polaris]
        f_layer = profiler_layers[ndx_profiler]
        
        # Normalize optypes for comparison
        p_optype_norm = normalize_optype(p_layer['optype'])
        f_optype_norm = normalize_optype(f_layer['optype'])
        
        # Case 1: optypes match (after normalization)
        if p_optype_norm == f_optype_norm:
            # Compare shapes
            input_match, input_details = compare_tensor_shapes(
                p_layer.get('input_tensors', []),
                f_layer.get('input_tensors', []),
                strip_leading_ones,
                p_layer['optype']
            )
            output_match, output_details = compare_tensor_shapes(
                p_layer.get('output_tensors', []),
                f_layer.get('output_tensors', []),
                strip_leading_ones,
                p_layer['optype']
            )
            
            # Special handling for mul if input counts don't match
            if not input_match and normalize_optype(p_layer['optype']) == 'mul':
                mul_valid, mul_details = validate_mul_compatibility(
                    p_layer.get('input_tensors', []),
                    f_layer.get('input_tensors', []),
                    strip_leading_ones
                )
                if mul_valid:
                    input_match = True
                    input_details = mul_details
            
            # Special handling for reshape if standard comparison fails
            if normalize_optype(p_layer['optype']) == 'reshape' and (not input_match or not output_match):
                reshape_valid, reshape_details = validate_reshape_compatibility(
                    p_layer.get('input_tensors', []),
                    p_layer.get('output_tensors', []),
                    f_layer.get('output_tensors', []),
                    strip_leading_ones
                )
                if reshape_valid:
                    # Accept reshape as valid - inputs may differ but outputs are compatible
                    input_match = True
                    output_match = True
                    output_details = reshape_details
            
            if input_match and output_match:
                # Perfect match
                print(f"✓ [P:{p_layer['seqno']}] [F:{f_layer['seqno']}] {p_layer['optype']}  "
                      f"in: {format_shapes(p_layer.get('input_tensors', []))} | "
                      f"out: {format_shapes(p_layer.get('output_tensors', []))}")
                stats.total_matches += 1
            else:
                # Shape mismatch
                print(f"✗ shape [P:{p_layer['seqno']}] [F:{f_layer['seqno']}] {p_layer['optype']}")
                if not input_match:
                    print(f"  input: polaris={format_shapes(p_layer.get('input_tensors', []))} "
                          f"profiler={format_shapes(f_layer.get('input_tensors', []))} ({input_details})")
                    stats.input_shape_mismatches += 1
                if not output_match:
                    print(f"  output: polaris={format_shapes(p_layer.get('output_tensors', []))} "
                          f"profiler={format_shapes(f_layer.get('output_tensors', []))} ({output_details})")
                    stats.output_shape_mismatches += 1
                stats.shape_mismatches += 1
            
            ndx_polaris += 1
            ndx_profiler += 1
            continue
        
        # Case 2: optypes don't match - search forward in profiler only (polaris is pivot)
        profiler_match_idx = find_next_match(
            profiler_layers, ndx_profiler + 1, p_optype_norm, max_search_distance
        )
        
        # If found in profiler, skip profiler entries to get there
        if profiler_match_idx is not None:
            for i in range(ndx_profiler, profiler_match_idx):
                layer = profiler_layers[i]
                print(f"⊘ [F:{layer['seqno']}] {layer['optype']} (skipped in polaris)")
                stats.unmatched_profiler += 1
            
            # Move profiler to matched position, polaris stays to compare
            ndx_profiler = profiler_match_idx
            # Continue to compare at this position (will be handled in next iteration)
        else:
            # Polaris entry not found in profiler - mark and advance polaris only
            print(f"✗ name [P:{p_layer['seqno']}] --- {p_layer['optype']} (not in profiler)")
            stats.name_mismatches += 1
            ndx_polaris += 1
    
    return stats


def print_summary(stats: ComparisonStats) -> None:
    """Print summary statistics."""
    print("\n=== Summary ===")
    print(f"Total matches: {stats.total_matches}")
    print(f"Name mismatches: {stats.name_mismatches}")
    print(f"Shape mismatches: {stats.shape_mismatches} "
          f"({stats.input_shape_mismatches} input, {stats.output_shape_mismatches} output)")
    print(f"Unmatched entries: {stats.unmatched_polaris + stats.unmatched_profiler} "
          f"({stats.unmatched_polaris} polaris, {stats.unmatched_profiler} profiler)")
    print(f"Ambiguous: {stats.ambiguous}")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if not Path(args.file1).exists():
        print(f"Error: File not found: {args.file1}", file=sys.stderr)
        return 1
    
    if not Path(args.file2).exists():
        print(f"Error: File not found: {args.file2}", file=sys.stderr)
        return 1
    
    # Detect file types
    type1 = detect_file_type(args.file1)
    type2 = detect_file_type(args.file2)
    
    if type1 is None or type2 is None:
        print("Error: Could not determine file types. Expected CSV files with "
              "'archname' (polaris) or 'OP CODE' (profiler) columns.", file=sys.stderr)
        return 1
    
    if type1 == type2:
        print(f"Error: Both files appear to be {type1} CSVs. "
              f"Expected one polaris and one profiler CSV.", file=sys.stderr)
        return 1
    
    # Assign files based on type
    polaris_file = args.file1 if type1 == 'polaris' else args.file2
    profiler_file = args.file1 if type1 == 'profiler' else args.file2
    
    print(f"Polaris CSV: {polaris_file}")
    print(f"Profiler CSV: {profiler_file}")
    print()
    
    # Extract layers
    try:
        polaris_layers = layers_polaris(polaris_file)
        profiler_layers = layers_profiler(profiler_file)
    except Exception as e:
        print(f"Error extracting layers: {e}", file=sys.stderr)
        return 1
    
    print(f"Loaded {len(polaris_layers)} polaris layers, {len(profiler_layers)} profiler layers")
    
    # Filter by optype if requested
    if args.filter_optype:
        filter_optype_norm = normalize_optype(args.filter_optype)
        polaris_layers_orig = polaris_layers
        profiler_layers_orig = profiler_layers
        
        polaris_layers = [
            layer for layer in polaris_layers
            if normalize_optype(layer['optype']) == filter_optype_norm
        ]
        profiler_layers = [
            layer for layer in profiler_layers
            if normalize_optype(layer['optype']) == filter_optype_norm
        ]
        
        print(f"Filtered to {len(polaris_layers)} polaris layers, {len(profiler_layers)} profiler layers with optype='{args.filter_optype}'")
        
        if len(polaris_layers) == 0 and len(profiler_layers) == 0:
            print(f"Warning: No layers found with optype='{args.filter_optype}'")
            return 0
    
    print()
    
    # Compare layers
    stats = compare_layers(polaris_layers, profiler_layers, args.max_search_distance, args.strip_leading_ones)
    
    # Print summary
    print_summary(stats)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
