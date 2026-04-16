#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Profiler to Polaris CSV Converter

Converts TTNN profiler CSV output to Polaris STATS format; column and tensor
mapping rules are implemented in this module (``map_optype_to_polaris``,
``format_tensors``, etc.).
"""

import csv
import sys
import yaml

import argparse
import importlib
import traceback
import math
from types import ModuleType
from typing import Optional

from loguru import logger


def _load_op_canonical() -> ModuleType:
    """Load sibling ``op_canonical`` when run as ``-m`` package or as a script."""
    if __package__:
        return importlib.import_module(".op_canonical", __package__)
    return importlib.import_module("op_canonical")


_op_canonical = _load_op_canonical()
normalize_profiler_opcode = _op_canonical.normalize_profiler_opcode
canonical_to_stats_display = _op_canonical.canonical_to_stats_display


def extract_dimension_from_bracket(value: str) -> str:
    """Extract the value inside brackets, e.g., '1[1]' -> '1'."""
    if '[' in value and ']' in value:
        start = value.index('[') + 1
        end = value.index(']')
        return value[start:end]
    return value


def get_datatype_bytes(datatype: str) -> int:
    """Get bytes per element for a given datatype."""
    datatype_upper = datatype.upper()
    if 'BFLOAT16' in datatype_upper or 'BF16' in datatype_upper:
        return 2
    elif 'FLOAT16' in datatype_upper or 'FP16' in datatype_upper:
        return 2
    elif 'FLOAT32' in datatype_upper or 'FP32' in datatype_upper:
        return 4
    elif 'INT32' in datatype_upper:
        return 4
    elif 'UINT32' in datatype_upper:
        return 4
    else:
        return 2  # Default to 2 bytes


def normalize_datatype_short(datatype: str) -> str:
    """Convert datatype to short form for precision column."""
    datatype_upper = datatype.upper()
    if 'BFLOAT16' in datatype_upper:
        return 'BF16'
    elif 'FLOAT32' in datatype_upper:
        return 'FP32'
    elif 'FLOAT16' in datatype_upper:
        return 'FP16'
    elif 'INT32' in datatype_upper:
        return 'INT32'
    elif 'UINT32' in datatype_upper:
        return 'UINT32'
    else:
        return datatype


def normalize_datatype_long(datatype: str) -> str:
    """Convert datatype to long form for tensor format strings."""
    datatype_upper = datatype.upper()
    if 'BFLOAT16' in datatype_upper or 'BF16' in datatype_upper:
        return 'float16'
    elif 'FLOAT32' in datatype_upper or 'FP32' in datatype_upper:
        return 'float32'
    elif 'FLOAT16' in datatype_upper or 'FP16' in datatype_upper:
        return 'float16'
    elif 'INT32' in datatype_upper:
        return 'INT32'
    elif 'UINT32' in datatype_upper:
        return 'UINT32'
    else:
        return 'float16'  # Default


def parse_tensor_dimensions(
    row: dict,
    prefix: str,
    idx: int,
    *,
    pad_extent: str = 'LOGICAL',
) -> Optional[dict]:
    """
    Parse tensor dimensions from profiler row.

    ``pad_extent`` is the bracket suffix on ``*_W_PAD[...]`` columns, e.g. ``LOGICAL`` or
    ``PADDED`` (when the profiler export includes tile-padded physical extents).

    Returns dict with 'dims' (list of dimension values), 'datatype', and 'elements' count.
    Returns None if tensor data is incomplete.
    """
    w_col = f'{prefix}_{idx}_W_PAD[{pad_extent}]'
    z_col = f'{prefix}_{idx}_Z_PAD[{pad_extent}]'
    y_col = f'{prefix}_{idx}_Y_PAD[{pad_extent}]'
    x_col = f'{prefix}_{idx}_X_PAD[{pad_extent}]'
    dtype_col = f'{prefix}_{idx}_DATATYPE'
    
    # Check if all dimensions exist and are non-blank
    dims = {}
    for dim_name, col in [('W', w_col), ('Z', z_col), ('Y', y_col), ('X', x_col)]:
        if col in row and row[col] and row[col].strip():
            dims[dim_name] = int(extract_dimension_from_bracket(row[col].strip()))
        else:
            return None  # Incomplete tensor data
    
    # Get datatype
    if dtype_col not in row or not row[dtype_col] or not row[dtype_col].strip():
        return None
    
    datatype = row[dtype_col].strip()
    
    # Calculate elements (broadcast result for inputs)
    w, z, y, x = dims['W'], dims['Z'], dims['Y'], dims['X']
    elements = w * z * y * x
    
    # Build dimension list (collapse W=1)
    if w == 1:
        dim_list = [z, y, x]
    else:
        dim_list = [w, z, y, x]
    
    return {
        'dims': dim_list,
        'datatype': datatype,
        'elements': elements
    }


def format_tensor_string(name: str, dims: list, datatype: str) -> str:
    """Format a tensor as: name[dim1xdim2x...]:datatype"""
    shape_str = 'x'.join(str(d) for d in dims)
    dtype_long = normalize_datatype_long(datatype)
    return f'{name}[{shape_str}]:{dtype_long}'


def map_optype_to_polaris(opcode: str, attrs: dict) -> str:
    """Map profiler operation types to Polaris STATS format (PascalCase).

    Delegates to :func:`~tools.profiling.op_canonical.normalize_profiler_opcode`
    for canonical resolution, then maps to the PascalCase STATS display name.
    """
    canonical = normalize_profiler_opcode(opcode, attrs)
    return canonical_to_stats_display(canonical)


def get_polaris_column_order() -> list[str]:
    """Return the column order for Polaris STATS output."""
    return [
        'archname', 'devname', 'freq_MHz', 'pipe', 'precision', 'wlgroup', 'wlname', 'wlinstance',
        'batch', 'opnum', 'opname', 'is_input_node', 'is_output_node', 'optype', 'op_rpt_count',
        'attrs', 'inList', 'outList', 'input_tensors', 'output_tensors', 'weight_tensors',
        'domain', 'opclass', 'removed', 'fused', 'fused_with_op', 'inElems', 'outElems',
        'inBytes', 'outBytes', 'instrs', 'inParamCount', 'inActCount', 'outActCount',
        'instr_count', 'compute_cycles', 'mem_rd_cycles', 'mem_wr_cycles', 'ramp_penalty',
        'rsrc_bnck', 'ideal_cycles', 'ideal_msecs', 'cycles', 'matrix_cycles', 'vector_cycles',
        'msecs', 'matrix_pipe_util', 'vector_pipe_util', 'mem_rd_util', 'mem_wr_util'
    ]


def convert_profiler_row_to_polaris(row: dict, freq_mhz: float, row_idx: int) -> dict:
    """
    Convert a single profiler row to Polaris STATS format.
    
    Args:
        row: Dictionary containing profiler CSV row data
        freq_mhz: Frequency in MHz for cycle calculations
        row_idx: Row index for opnum calculation
        
    Returns:
        Dictionary with Polaris STATS column names and values
    """
    # Initialize polaris row with NA for all columns
    polaris_row = {col: 'NA' for col in get_polaris_column_order()}
    
    # Parse attributes
    attrs = {}
    if 'ATTRIBUTES' in row and row['ATTRIBUTES']:
        try:
            parsed_attrs = yaml.safe_load(row['ATTRIBUTES'].replace(";", ","))
            if parsed_attrs and isinstance(parsed_attrs, dict):
                attrs = parsed_attrs
        except Exception as e:
            op_code = row.get('OP CODE', 'unknown')
            global_call_count = row.get('GLOBAL CALL COUNT', 'unknown')
            logger.warning(f"Row {row_idx} (OP CODE: {op_code}, GLOBAL CALL COUNT: {global_call_count}): Could not parse ATTRIBUTES column: {e}")
    
    # 1. Basic operation info
    opcode = row.get('OP CODE', '')
    polaris_row['optype'] = map_optype_to_polaris(opcode, attrs)
    polaris_row['opnum'] = str(int(row.get('GLOBAL CALL COUNT', 1024)) // 1024 - 1) if row.get('GLOBAL CALL COUNT') else str(row_idx)
    polaris_row['opname'] = f"{polaris_row['optype']}_{polaris_row['opnum']}"
    polaris_row['attrs'] = str(attrs) if attrs else '{}'
    polaris_row['op_rpt_count'] = '1'
    
    # 2. Configuration (leave as NA per spec)
    polaris_row['archname'] = 'NA'
    polaris_row['devname'] = 'NA'
    polaris_row['freq_MHz'] = str(freq_mhz)
    polaris_row['wlgroup'] = 'NA'
    polaris_row['wlname'] = 'NA'
    polaris_row['wlinstance'] = 'NA'
    polaris_row['batch'] = 'NA'
    polaris_row['pipe'] = ''  # Leave blank per spec
    
    # 3. Precision - from INPUT_0_DATATYPE
    input_0_datatype = row.get('INPUT_0_DATATYPE', '').strip()
    if input_0_datatype:
        polaris_row['precision'] = normalize_datatype_short(input_0_datatype)
    
    # 4. Parse input tensors
    input_tensors = []
    total_input_elems = 0
    total_input_bytes = 0
    
    for idx in [0, 1]:
        tensor_info = parse_tensor_dimensions(row, 'INPUT', idx)
        if tensor_info:
            tensor_str = format_tensor_string(f'input_{idx}', tensor_info['dims'], tensor_info['datatype'])
            input_tensors.append(tensor_str)
            total_input_elems += tensor_info['elements']
            total_input_bytes += tensor_info['elements'] * get_datatype_bytes(tensor_info['datatype'])
    
    polaris_row['input_tensors'] = ';'.join(input_tensors) if input_tensors else ''
    polaris_row['inElems'] = str(total_input_elems) if total_input_elems > 0 else 'NA'
    polaris_row['inBytes'] = str(total_input_bytes) if total_input_bytes > 0 else 'NA'
    polaris_row['inActCount'] = polaris_row['inElems']
    
    # 5. Parse output tensors
    output_tensors = []
    total_output_elems = 0
    total_output_bytes = 0
    
    tensor_info = parse_tensor_dimensions(row, 'OUTPUT', 0)
    if tensor_info:
        tensor_str = format_tensor_string('output_0', tensor_info['dims'], tensor_info['datatype'])
        output_tensors.append(tensor_str)
        total_output_elems += tensor_info['elements']
        total_output_bytes += tensor_info['elements'] * get_datatype_bytes(tensor_info['datatype'])
    
    polaris_row['output_tensors'] = ';'.join(output_tensors) if output_tensors else ''
    polaris_row['outElems'] = str(total_output_elems) if total_output_elems > 0 else 'NA'
    polaris_row['outBytes'] = str(total_output_bytes) if total_output_bytes > 0 else 'NA'
    polaris_row['outActCount'] = polaris_row['outElems']
    
    # 6. Timing - msecs from DEVICE KERNEL DURATION
    if 'DEVICE KERNEL DURATION [ns]' in row and row['DEVICE KERNEL DURATION [ns]']:
        try:
            duration_ns = float(row['DEVICE KERNEL DURATION [ns]'])
            msecs = duration_ns / 1_000_000.0
            polaris_row['msecs'] = str(msecs)
            
            # Calculate cycles: cycles = (duration_ns / 1000) * freq_MHz
            cycles = (duration_ns / 1000.0) * freq_mhz
            polaris_row['cycles'] = str(int(cycles))
        except (ValueError, TypeError):
            polaris_row['msecs'] = 'NA'
            polaris_row['cycles'] = 'NA'
    
    # 7. Utilization and pipe cycles
    fpu_util = 0.0
    sfpu_util = 0.0
    
    if 'Avg FPU util on full grid (%)' in row and row['Avg FPU util on full grid (%)']:
        try:
            val = row['Avg FPU util on full grid (%)'].strip().lower()
            # Check for string 'nan' or empty value
            if val != 'nan' and val != '':
                fpu_util_parsed = float(row['Avg FPU util on full grid (%)'])
                # Check for NaN float value
                if not math.isnan(fpu_util_parsed):
                    fpu_util = fpu_util_parsed
        except (ValueError, TypeError):
            pass
    
    if 'Avg SFPU util on full grid (%)' in row and row['Avg SFPU util on full grid (%)']:
        try:
            val = row['Avg SFPU util on full grid (%)'].strip().lower()
            # Check for string 'nan' or empty value
            if val != 'nan' and val != '':
                sfpu_util_parsed = float(row['Avg SFPU util on full grid (%)'])
                # Check for NaN float value
                if not math.isnan(sfpu_util_parsed):
                    sfpu_util = sfpu_util_parsed
        except (ValueError, TypeError):
            pass
    
    # Calculate matrix_cycles and vector_cycles
    if polaris_row['cycles'] != 'NA':
        cycles = float(polaris_row['cycles'])
        # matrix_cycles = fpu_utilization * cycles (utilization as percentage)
        matrix_cycles = (fpu_util / 100.0) * cycles
        vector_cycles = (sfpu_util / 100.0) * cycles
        
        polaris_row['matrix_cycles'] = str(int(matrix_cycles))
        polaris_row['vector_cycles'] = str(int(vector_cycles))
        
        # Pipe utilization (already in percentage form)
        polaris_row['matrix_pipe_util'] = str(fpu_util) if fpu_util > 0 else '0.0'
        polaris_row['vector_pipe_util'] = str(sfpu_util) if sfpu_util > 0 else '0.0'
        
        # compute_cycles = max(matrix_cycles, vector_cycles)
        compute_cycles = max(matrix_cycles, vector_cycles)
        polaris_row['compute_cycles'] = str(int(compute_cycles))
    
    # 8. Memory cycles from PM BANDWIDTH
    if 'PM BANDWIDTH [ns]' in row and row['PM BANDWIDTH [ns]']:
        try:
            pm_bandwidth_ns = float(row['PM BANDWIDTH [ns]'])
            # memory_cycles = (PM BANDWIDTH [ns] / 1000) * freq_MHz
            memory_cycles = (pm_bandwidth_ns / 1000.0) * freq_mhz
            
            # Split into rd and wr cycles (half each)
            polaris_row['mem_rd_cycles'] = str(int(memory_cycles / 2))
            polaris_row['mem_wr_cycles'] = str(int(memory_cycles / 2))
            
            # Determine resource bottleneck
            if polaris_row['compute_cycles'] != 'NA':
                comp_cycles = float(polaris_row['compute_cycles'])
                if memory_cycles > comp_cycles:
                    polaris_row['rsrc_bnck'] = 'MEM'
                else:
                    polaris_row['rsrc_bnck'] = 'COMP'
        except (ValueError, TypeError):
            pass
    
    # 9. Memory utilization from DRAM BW UTIL
    if 'DRAM BW UTIL (%)' in row and row['DRAM BW UTIL (%)']:
        try:
            dram_util = float(row['DRAM BW UTIL (%)'])
            # Use same value for both read and write
            polaris_row['mem_rd_util'] = str(dram_util)
            polaris_row['mem_wr_util'] = str(dram_util)
        except (ValueError, TypeError):
            pass
    
    # 10. Fixed/default values
    polaris_row['weight_tensors'] = ''
    polaris_row['domain'] = 'None'
    polaris_row['opclass'] = 'None'
    polaris_row['removed'] = 'False'
    polaris_row['fused'] = 'False'
    polaris_row['fused_with_op'] = 'NA'
    polaris_row['instrs'] = '{}'
    polaris_row['inParamCount'] = '0'
    polaris_row['instr_count'] = 'NA'
    polaris_row['ramp_penalty'] = '50.0'
    polaris_row['ideal_cycles'] = ''  # Leave blank per spec
    polaris_row['ideal_msecs'] = ''  # Leave blank per spec
    
    # 11. Graph topology (will be updated in second pass)
    polaris_row['is_input_node'] = 'False'
    polaris_row['is_output_node'] = 'False'
    polaris_row['inList'] = ''  # Leave blank per spec
    polaris_row['outList'] = ''  # Leave blank per spec
    
    return polaris_row


def analyze_graph_topology(polaris_rows: list[dict]) -> None:
    """
    Analyze graph topology to determine is_input_node and is_output_node.
    
    Modifies polaris_rows in place.
    
    is_input_node: True if operation's inputs are external (not outputs of other ops)
    is_output_node: True if operation's outputs are not inputs to any other ops
    """
    # Build mapping of operation outputs
    # For simplicity, we'll use opname as identifier
    op_outputs = set()
    for row in polaris_rows:
        op_outputs.add(row['opname'])
    
    # Check each operation
    for i, row in enumerate(polaris_rows):
        # is_input_node: Check if this is one of the first operations
        # (since profiler doesn't track tensor names, we use position as heuristic)
        # First few operations are likely input nodes
        if i < 3:  # Simple heuristic: first 3 ops are likely input nodes
            row['is_input_node'] = 'True'
        
        # is_output_node: Check if this is one of the last operations
        # Last few operations are likely output nodes
        if i >= len(polaris_rows) - 3:  # Simple heuristic: last 3 ops are likely output nodes
            row['is_output_node'] = 'True'


def convert_profiler_to_polaris(input_file: str, output_file: str, freq_mhz: float) -> None:
    """
    Convert TTNN profiler CSV to Polaris STATS CSV format.
    
    Output includes all original profiler columns PLUS Polaris columns with 'polaris_' prefix.
    
    Args:
        input_file: Path to input profiler CSV
        output_file: Path to output Polaris CSV
        freq_mhz: Frequency in MHz for cycle calculations
    """
    # Read profiler CSV
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        input_fieldnames = reader.fieldnames or []
        profiler_rows = list(reader)
    
    logger.info(f"Read {len(profiler_rows)} rows from {input_file}")
    
    # Convert each row
    output_rows = []
    polaris_columns = get_polaris_column_order()
    
    for idx, profiler_row in enumerate(profiler_rows):
        try:
            # Get Polaris mapping
            polaris_row = convert_profiler_row_to_polaris(profiler_row, freq_mhz, idx)
            
            # Create output row with all original columns
            output_row = profiler_row.copy()
            
            # Add Polaris columns with 'polaris_' prefix
            for col in polaris_columns:
                output_row[f'polaris_{col}'] = polaris_row.get(col, '')
            
            output_rows.append((output_row, polaris_row))
        except Exception as e:
            op_code = profiler_row.get('OP CODE', 'unknown')
            global_call_count = profiler_row.get('GLOBAL CALL COUNT', 'unknown')
            logger.error(f"Failed to convert row {idx} (OP CODE: {op_code}, GLOBAL CALL COUNT: {global_call_count}): {e}")
            raise
    
    # Analyze graph topology on the Polaris rows
    polaris_rows_for_topology = [row[1] for row in output_rows]
    analyze_graph_topology(polaris_rows_for_topology)
    
    # Update the Polaris columns in output rows with topology info
    for i, (output_row, polaris_row) in enumerate(output_rows):
        output_row['polaris_is_input_node'] = polaris_rows_for_topology[i]['is_input_node']
        output_row['polaris_is_output_node'] = polaris_rows_for_topology[i]['is_output_node']
    
    # Build output fieldnames: all original + prefixed Polaris columns
    polaris_prefixed = [f'polaris_{col}' for col in polaris_columns]
    output_fieldnames = list(input_fieldnames) + polaris_prefixed
    
    # Write output CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for output_row, _ in output_rows:
            writer.writerow(output_row)
    
    logger.info(f"Wrote {len(output_rows)} rows to {output_file}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert TTNN Profiler CSV to Polaris STATS CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage (from repository root):
  python -m tools.profiling.profiler_to_polaris_converter --input profiler.csv --output polaris.csv --freq-mhz 1000.0

Mapping details: implemented in this module (see ``map_optype_to_polaris``,
``format_tensors``, and related helpers).
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input profiler CSV file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output Polaris CSV file')
    parser.add_argument('--freq-mhz', type=float, required=True,
                        help='Frequency in MHz for cycle calculations (mandatory)')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        convert_profiler_to_polaris(args.input, args.output, args.freq_mhz)
        logger.success("Conversion completed successfully!")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Missing required column {e} in input CSV")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
