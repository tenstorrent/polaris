#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare Polaris and Profiler CSV layer sequences."""

import sys
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from op_canonical import normalize_polaris_optype, to_comparison_group
    from shape_canonical import (
        parse_shape_string as parse_shape,
        normalize_shape,
        compare_tensor_shapes,
        validate_binary_compatibility,
        validate_reshape_compatibility,
        compare_tensor_attributes,
        LAYOUT_NORMALIZATION,
        DTYPE_NORMALIZATION,
    )
except ImportError:
    from .op_canonical import normalize_polaris_optype, to_comparison_group  # type: ignore
    from .shape_canonical import (  # type: ignore
        parse_shape_string as parse_shape,
        normalize_shape,
        compare_tensor_shapes,
        validate_binary_compatibility,
        validate_reshape_compatibility,
        compare_tensor_attributes,
        LAYOUT_NORMALIZATION,
        DTYPE_NORMALIZATION,
    )

# Maximum distance to search forward for matching operations
DEFAULT_MAX_SEARCH_DISTANCE = 10

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
    attr_mismatches: int = 0
    unmatched_polaris: int = 0
    unmatched_profiler: int = 0
    ambiguous: int = 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare Polaris and Profiler CSV layer sequences'
    )
    parser.add_argument('file1', type=str, help='First CSV file (polaris or profiler)')
    parser.add_argument('file2', type=str, nargs='?', default=None,
                        help='Second CSV file (optional; required for shape comparison)')
    parser.add_argument(
        '--perf',
        action='store_true',
        help='Enable performance matching: show network-total and layer-type-wise '
             'duration comparison (ms). With two files, shows gap w.r.t. profiler. '
             'With one file, shows standalone breakdown.'
    )
    parser.add_argument(
        '--max-search-distance',
        type=int,
        default=DEFAULT_MAX_SEARCH_DISTANCE,
        help=f'Maximum distance to search forward for matching operations (default: {DEFAULT_MAX_SEARCH_DISTANCE})'
    )
    parser.add_argument(
        '--strip-leading-ones',
        action='store_true',
        default=True,
        help='Strip all leading 1s from shapes (default: enabled). '
             'Leading 1s are a batch-dimension convention difference between Polaris and HW. '
             'Use --no-strip-leading-ones for strict matching.'
    )
    parser.add_argument(
        '--no-strip-leading-ones',
        action='store_false',
        dest='strip_leading_ones',
        help='Disable stripping of leading 1s for strict shape matching'
    )
    # NOTE: --strip-singleton-dims is currently required for fused head ops
    # (CreateQKVHeads, ConcatHeads) because HW uses 4D shapes with a
    # seq_groups=1 singleton dim (e.g. [B, 1, S, H]) while Polaris emits 3D
    # shapes (e.g. [B, S, H]). Additionally, HW implicitly reinterprets the
    # output of ConcatHeads from [B, 1, S, H] to [1, B, S, H] for
    # downstream ops without an explicit reshape. Future work: update the
    # Polaris shim to emit 4D shapes and model this implicit view change,
    # which would allow removing this flag for those ops.
    parser.add_argument(
        '--strip-singleton-dims',
        action='store_true',
        help='Strip all singleton (=1) dimensions from shapes regardless of position '
             '(handles HW seq_groups=1 convention)'
    )
    parser.add_argument(
        '--filter-optype',
        type=str,
        default=None,
        help='Filter to only compare layers with this operation type (case-insensitive)'
    )
    parser.add_argument(
        '--ignore-attrs',
        action='store_true',
        help='Skip tensor attribute comparison (dtype, layout, memory); compare shapes only'
    )
    return parser.parse_args()


def normalize_optype(optype: str) -> str:
    """Normalize operation type name to canonical form then coarsen for
    sequence matching (e.g. add/mul/sub → binary).

    Delegates to :mod:`op_canonical` for the canonical form and comparison
    group mapping.
    """
    return to_comparison_group(normalize_polaris_optype(optype))


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
    strip_leading_ones: bool = False,
    strip_singleton_dims: bool = False,
    ignore_attrs: bool = False,
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
                p_layer['optype'],
                strip_singleton_dims=strip_singleton_dims,
            )
            output_match, output_details = compare_tensor_shapes(
                p_layer.get('output_tensors', []),
                f_layer.get('output_tensors', []),
                strip_leading_ones,
                p_layer['optype'],
                strip_singleton_dims=strip_singleton_dims,
            )

            # Special handling for binary ops (add/mul/sub) if input counts
            # or shapes don't match — one side may use scalar/untracked operands
            p_canonical = normalize_polaris_optype(p_layer['optype'])
            if not input_match and to_comparison_group(p_canonical) == 'binary':
                bin_valid, bin_details = validate_binary_compatibility(
                    p_layer.get('input_tensors', []),
                    f_layer.get('input_tensors', []),
                    strip_leading_ones,
                    strip_singleton_dims=strip_singleton_dims,
                )
                if bin_valid:
                    input_match = True
                    input_details = bin_details

            # Special handling for reshape if standard comparison fails
            if p_canonical == 'reshape' and (not input_match or not output_match):
                reshape_valid, reshape_details = validate_reshape_compatibility(
                    p_layer.get('input_tensors', []),
                    p_layer.get('output_tensors', []),
                    f_layer.get('output_tensors', []),
                    strip_leading_ones,
                    strip_singleton_dims=strip_singleton_dims,
                )
                if reshape_valid:
                    # Accept reshape as valid - inputs may differ but outputs are compatible
                    input_match = True
                    output_match = True
                    output_details = reshape_details

            # Attribute comparison (dtype, layout, memory)
            attr_ok = True
            attr_details_parts = []
            if input_match and output_match and not ignore_attrs:
                in_attr_ok, in_attr_det = compare_tensor_attributes(p_layer, f_layer, 'input')
                out_attr_ok, out_attr_det = compare_tensor_attributes(p_layer, f_layer, 'output')
                if not in_attr_ok:
                    attr_ok = False
                    attr_details_parts.append(f"input attrs: {in_attr_det}")
                if not out_attr_ok:
                    attr_ok = False
                    attr_details_parts.append(f"output attrs: {out_attr_det}")

            if input_match and output_match and attr_ok:
                print(f"✓ [P:{p_layer['seqno']}] [F:{f_layer['seqno']}] {p_layer['optype']}  "
                      f"in: {format_shapes(p_layer.get('input_tensors', []))} | "
                      f"out: {format_shapes(p_layer.get('output_tensors', []))}")
                stats.total_matches += 1
            elif input_match and output_match and not attr_ok:
                print(f"✗ attr [P:{p_layer['seqno']}] [F:{f_layer['seqno']}] {p_layer['optype']}")
                for part in attr_details_parts:
                    print(f"  {part}")
                stats.attr_mismatches += 1
            else:
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
    print(f"Attribute mismatches: {stats.attr_mismatches}")
    print(f"Unmatched entries: {stats.unmatched_polaris + stats.unmatched_profiler} "
          f"({stats.unmatched_polaris} polaris, {stats.unmatched_profiler} profiler)")
    print(f"Ambiguous: {stats.ambiguous}")


# ---------------------------------------------------------------------------
# Performance summary helpers
# ---------------------------------------------------------------------------

def _aggregate_duration_by_optype(
    layers: List[Dict[str, Any]],
) -> Dict[str, Tuple[int, float, int]]:
    """Group layers by fine-grained canonical optype and sum durations.

    Returns ``{optype: (count, total_ms, lut_hits)}`` sorted by descending
    total_ms.  Layers whose ``duration_ms`` is None are counted but
    contribute 0 ms.  ``lut_hits`` counts layers where
    ``uses_perf_lookup`` is True (Polaris-only; always 0 for profiler).
    """
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    lut_hits: Dict[str, int] = defaultdict(int)
    for layer in layers:
        optype = layer['optype']
        counts[optype] += 1
        dur = layer.get('duration_ms')
        if dur is not None:
            totals[optype] += dur
        if layer.get('uses_perf_lookup'):
            lut_hits[optype] += 1
    all_optypes = set(counts) | set(totals)
    return {
        op: (counts[op], totals.get(op, 0.0), lut_hits.get(op, 0))
        for op in sorted(all_optypes, key=lambda o: totals.get(o, 0.0), reverse=True)
    }


def _pct_gap(reference: float, other: float) -> str:
    """Format percentage gap of *other* w.r.t. *reference*.

    Positive means *other* is larger (slower).
    """
    if reference == 0.0:
        return "N/A"
    gap = (other - reference) / reference * 100.0
    sign = "+" if gap >= 0 else ""
    return f"{sign}{gap:.2f}%"


def _print_perf_standalone(
    layers: List[Dict[str, Any]],
    source_label: str,
) -> None:
    """Print a standalone performance breakdown for a single source."""
    by_optype = _aggregate_duration_by_optype(layers)
    total_ms = sum(ms for _, ms, _ in by_optype.values())
    total_count = sum(cnt for cnt, _, _ in by_optype.values())
    total_lut = sum(lut for _, _, lut in by_optype.values())
    has_lut = total_lut > 0

    print(f"\n{'=' * 60}")
    print(f"  Performance Summary ({source_label})")
    print(f"{'=' * 60}")
    print(f"\n  Network total: {total_ms:.4f} ms")
    if has_lut:
        print(f"  LUT hits: {total_lut}/{total_count}")
    print()

    hdr_type = "Layer Type"
    hdr_cnt = "Count"
    hdr_dur = "Duration (ms)"
    hdr_lut = "LUT"
    col_w_type = max(len(hdr_type), max((len(op) for op in by_optype), default=10))
    col_w_cnt = max(len(hdr_cnt), 6)
    col_w_dur = max(len(hdr_dur), 14)
    col_w_lut = max(len(hdr_lut), 8)

    header = f"  {hdr_type:<{col_w_type}}  {hdr_cnt:>{col_w_cnt}}  {hdr_dur:>{col_w_dur}}"
    if has_lut:
        header += f"  {hdr_lut:>{col_w_lut}}"
    print(header)
    rule_len = col_w_type + col_w_cnt + col_w_dur + 4 + (col_w_lut + 2 if has_lut else 0)
    print(f"  {'─' * rule_len}")

    for op, (cnt, ms, lut) in by_optype.items():
        line = f"  {op:<{col_w_type}}  {cnt:>{col_w_cnt}}  {ms:>{col_w_dur}.4f}"
        if has_lut:
            line += f"  {f'{lut}/{cnt}':>{col_w_lut}}"
        print(line)

    print(f"  {'─' * rule_len}")
    line = f"  {'TOTAL':<{col_w_type}}  {total_count:>{col_w_cnt}}  {total_ms:>{col_w_dur}.4f}"
    if has_lut:
        line += f"  {f'{total_lut}/{total_count}':>{col_w_lut}}"
    print(line)
    print()


def _print_perf_comparison(
    profiler_layers: List[Dict[str, Any]],
    polaris_layers: List[Dict[str, Any]],
) -> None:
    """Print side-by-side performance comparison with gap w.r.t. profiler."""
    prof_by_op = _aggregate_duration_by_optype(profiler_layers)
    pol_by_op = _aggregate_duration_by_optype(polaris_layers)

    prof_total_ms = sum(ms for _, ms, _ in prof_by_op.values())
    pol_total_ms = sum(ms for _, ms, _ in pol_by_op.values())
    prof_total_cnt = sum(cnt for cnt, _, _ in prof_by_op.values())
    pol_total_cnt = sum(cnt for cnt, _, _ in pol_by_op.values())
    pol_total_lut = sum(lut for _, _, lut in pol_by_op.values())

    print(f"\n{'=' * 82}")
    print("  Performance Summary")
    print(f"{'=' * 82}")

    print(f"\n  Network total:")
    print(f"    Profiler:  {prof_total_ms:.4f} ms")
    print(f"    Polaris:   {pol_total_ms:.4f} ms")
    print(f"    Gap:       {_pct_gap(prof_total_ms, pol_total_ms)} (w.r.t. profiler)")
    print(f"    Polaris LUT hits: {pol_total_lut}/{pol_total_cnt}")
    print()

    all_ops = list(dict.fromkeys(
        list(prof_by_op.keys()) + list(pol_by_op.keys())
    ))

    # Re-sort by profiler duration descending (missing → 0)
    all_ops.sort(key=lambda o: prof_by_op.get(o, (0, 0.0, 0))[1], reverse=True)

    col_w_type = max(10, max((len(op) for op in all_ops), default=10))
    hdr = (
        f"  {'Layer Type':<{col_w_type}}"
        f"  {'#Prof':>6}  {'Profiler(ms)':>13}"
        f"  {'#Pol':>6}  {'Polaris(ms)':>13}"
        f"  {'LUT':>8}"
        f"  {'Abs Gap(ms)':>12}"
        f"  {'Gap%':>9}"
    )
    print(hdr)
    rule_len = col_w_type + 6 + 13 + 6 + 13 + 8 + 12 + 9 + 14
    print(f"  {'─' * rule_len}")

    for op in all_ops:
        p_cnt, p_ms, _ = prof_by_op.get(op, (0, 0.0, 0))
        s_cnt, s_ms, s_lut = pol_by_op.get(op, (0, 0.0, 0))
        gap_pct = _pct_gap(p_ms, s_ms)
        abs_gap = s_ms - p_ms
        abs_gap_s = f"{abs_gap:+.4f}" if (p_cnt and s_cnt) else "—"
        p_cnt_s = str(p_cnt) if p_cnt else "—"
        s_cnt_s = str(s_cnt) if s_cnt else "—"
        p_ms_s = f"{p_ms:.4f}" if p_cnt else "—"
        s_ms_s = f"{s_ms:.4f}" if s_cnt else "—"
        lut_s = f"{s_lut}/{s_cnt}" if s_cnt else "—"
        print(
            f"  {op:<{col_w_type}}"
            f"  {p_cnt_s:>6}  {p_ms_s:>13}"
            f"  {s_cnt_s:>6}  {s_ms_s:>13}"
            f"  {lut_s:>8}"
            f"  {abs_gap_s:>12}"
            f"  {gap_pct:>9}"
        )

    abs_total = pol_total_ms - prof_total_ms
    print(f"  {'─' * rule_len}")
    print(
        f"  {'TOTAL':<{col_w_type}}"
        f"  {prof_total_cnt:>6}  {prof_total_ms:>13.4f}"
        f"  {pol_total_cnt:>6}  {pol_total_ms:>13.4f}"
        f"  {f'{pol_total_lut}/{pol_total_cnt}':>8}"
        f"  {abs_total:>+12.4f}"
        f"  {_pct_gap(prof_total_ms, pol_total_ms):>9}"
    )
    print()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if not Path(args.file1).exists():
        print(f"Error: File not found: {args.file1}", file=sys.stderr)
        return 1

    if args.file2 is not None and not Path(args.file2).exists():
        print(f"Error: File not found: {args.file2}", file=sys.stderr)
        return 1

    # --- Standalone mode (single file + --perf) ---
    if args.file2 is None:
        if not args.perf:
            print("Error: Two files are required for shape comparison. "
                  "Use --perf with a single file for standalone performance breakdown.",
                  file=sys.stderr)
            return 1

        ftype = detect_file_type(args.file1)
        if ftype is None:
            print("Error: Could not determine file type. Expected CSV with "
                  "'archname' (polaris) or 'OP CODE' (profiler) columns.",
                  file=sys.stderr)
            return 1

        try:
            if ftype == 'polaris':
                layers = layers_polaris(args.file1)
                label = "Polaris"
            else:
                layers = layers_profiler(args.file1)
                label = "Profiler"
        except Exception as e:
            print(f"Error extracting layers: {e}", file=sys.stderr)
            return 1

        print(f"{label} CSV: {args.file1}")
        print(f"Loaded {len(layers)} {label.lower()} layers")

        if args.filter_optype:
            filter_norm = normalize_optype(args.filter_optype)
            layers = [l for l in layers if normalize_optype(l['optype']) == filter_norm]
            print(f"Filtered to {len(layers)} layers with optype='{args.filter_optype}'")

        _print_perf_standalone(layers, label)
        return 0

    # --- Two-file mode ---
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

    # Compare layers (shape / attribute matching)
    stats = compare_layers(polaris_layers, profiler_layers, args.max_search_distance,
                           args.strip_leading_ones, args.strip_singleton_dims,
                           ignore_attrs=args.ignore_attrs)

    # Print shape-comparison summary
    print_summary(stats)

    # Performance comparison (when --perf is enabled)
    if args.perf:
        _print_perf_comparison(profiler_layers, polaris_layers)

    return 0


if __name__ == '__main__':
    sys.exit(main())
