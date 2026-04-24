#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import argparse
import csv
import re
import yaml
from typing import Any, Dict, List, Optional, Tuple

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Show layers from profiler CSV')
    parser.add_argument('input', type=str, help='Input CSV file')
    return parser.parse_args()

COLUMNS_OF_INTEREST = ['GLOBAL CALL COUNT', 'OP CODE']


def expand_tensor_string(row: Dict[str, Any], input_index: int, in_or_out: str='INPUT') -> str:
    normalized_fields = []
    for tmp in ['W', 'Z', 'Y', 'X']:
        tmpvalue = row.get(f'{in_or_out}_{input_index}_{tmp}_PAD[LOGICAL]', '')
        if tmpvalue == '':
            continue
        value = tmpvalue.split('[')[1].replace(']', '')
        normalized_fields.append(value)
    return 'x'.join(normalized_fields)


def expand_tensor_dims(
    row: Dict[str, Any], input_index: int, in_or_out: str = 'INPUT',
) -> List[Tuple[int, int]]:
    """Return ``[(pad, logical), ...]`` for each non-empty dimension of a tensor slot."""
    dims: List[Tuple[int, int]] = []
    for d in ['W', 'Z', 'Y', 'X']:
        cell = row.get(f'{in_or_out}_{input_index}_{d}_PAD[LOGICAL]', '')
        if cell == '':
            continue
        pad_str, rest = cell.split('[', 1)
        logical_str = rest.rstrip(']')
        dims.append((int(pad_str), int(logical_str)))
    return dims


def expand_tensor_attrs(row: Dict[str, Any], input_index: int, in_or_out: str = 'INPUT') -> Optional[Dict[str, str]]:
    """Extract layout, datatype, and memory for a single tensor slot.

    Returns None when the slot is empty (no shape data).
    """
    prefix = f'{in_or_out}_{input_index}'
    layout = (row.get(f'{prefix}_LAYOUT') or '').strip()
    dtype = (row.get(f'{prefix}_DATATYPE') or '').strip()
    memory_raw = (row.get(f'{prefix}_MEMORY') or '').strip()
    if not layout and not dtype and not memory_raw:
        return None
    # Strip device prefix (e.g. "DEV_1_L1_INTERLEAVED" -> "L1_INTERLEAVED")
    memory = re.sub(r'^DEV_\d+_', '', memory_raw) if memory_raw else ''
    return {'layout': layout, 'dtype': dtype, 'memory': memory}


def normalize_tensor_string(col: str,tensor_string: str) -> str:
    if 'tensors' not in col:
        return tensor_string
    fields = tensor_string.split(';')
    normalized_fields = []
    for field in fields:
        tmp = field.split(':')[0].split('[')[1].replace(']', '')
        normalized_fields.append(tmp)
    return ';'.join(normalized_fields)


def layers_profiler(input_file: str) -> List[Dict[str, Any]]:
    rows = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['OP CODE'] = row['OP CODE'].replace("DeviceOperation", "")
            if 'BinaryNg' in row['OP CODE']:
                attrs_cell = (row.get('ATTRIBUTES') or '').strip()
                attributes = None
                if attrs_cell:
                    try:
                        attributes = yaml.safe_load(attrs_cell.replace(";", ","))
                    except yaml.YAMLError:
                        attributes = None
                if isinstance(attributes, dict) and 'binary_op_type' in attributes:
                    bot = attributes['binary_op_type']
                    if isinstance(bot, str):
                        row['OP CODE'] = bot.replace("BinaryOpType::", "")
            if 'Unary' in row['OP CODE']:
                attrs_cell = (row.get('ATTRIBUTES') or '').strip()
                attributes = None
                if attrs_cell:
                    try:
                        attributes = yaml.safe_load(attrs_cell.replace(";", ","))
                    except yaml.YAMLError:
                        attributes = None
                if isinstance(attributes, dict):
                    # Try direct unary_op_type field first
                    if 'unary_op_type' in attributes:
                        uot = attributes['unary_op_type']
                        if isinstance(uot, str):
                            row['OP CODE'] = uot.replace("UnaryOpType::", "")
                    # Otherwise, try to extract from op_chain field
                    elif 'op_chain' in attributes:
                        op_chain = str(attributes['op_chain'])
                        match = re.search(r'UnaryOpType::(\w+)', op_chain)
                        if match:
                            row['OP CODE'] = match.group(1)
            # Attribute lists (dtypes, layouts, memories) must stay parallel
            # with their tensor lists so that positional indexing in
            # compare_tensor_attributes (compare_layers.py) compares the
            # correct tensor slot.  When attrs are missing for a slot that
            # has a tensor, we append None to preserve alignment.
            a0_in = expand_tensor_attrs(row, 0, 'INPUT')
            a0_out = expand_tensor_attrs(row, 0, 'OUTPUT')
            filtered_row: Dict[str, Any] = {
                'seqno': int(row['GLOBAL CALL COUNT']),
                'optype': row['OP CODE'].lower(),
                'input_tensors': [expand_tensor_string(row, 0, 'INPUT')],
                'output_tensors': [expand_tensor_string(row, 0, 'OUTPUT')],
                'input_pad_logical': [expand_tensor_dims(row, 0, 'INPUT')],
                'output_pad_logical': [expand_tensor_dims(row, 0, 'OUTPUT')],
                'input_dtypes': [a0_in['dtype'] if a0_in else None],
                'input_layouts': [a0_in['layout'] if a0_in else None],
                'input_memories': [a0_in['memory'] if a0_in else None],
                'output_dtypes': [a0_out['dtype'] if a0_out else None],
                'output_layouts': [a0_out['layout'] if a0_out else None],
                'output_memories': [a0_out['memory'] if a0_out else None],
            }
            for idx in (1, 2):
                s = expand_tensor_string(row, idx, 'INPUT')
                if s:
                    filtered_row['input_tensors'].append(s)
                    filtered_row['input_pad_logical'].append(
                        expand_tensor_dims(row, idx, 'INPUT'))
                    a = expand_tensor_attrs(row, idx, 'INPUT')
                    filtered_row['input_dtypes'].append(a['dtype'] if a else None)
                    filtered_row['input_layouts'].append(a['layout'] if a else None)
                    filtered_row['input_memories'].append(a['memory'] if a else None)
                s = expand_tensor_string(row, idx, 'OUTPUT')
                if s:
                    filtered_row['output_tensors'].append(s)
                    filtered_row['output_pad_logical'].append(
                        expand_tensor_dims(row, idx, 'OUTPUT'))
                    a = expand_tensor_attrs(row, idx, 'OUTPUT')
                    filtered_row['output_dtypes'].append(a['dtype'] if a else None)
                    filtered_row['output_layouts'].append(a['layout'] if a else None)
                    filtered_row['output_memories'].append(a['memory'] if a else None)
            rows.append(filtered_row)

    # Post-process: correct ops where the profiler conflates PAD and LOGICAL
    # shapes (PAD == LOGICAL) by inheriting the true LOGICAL value from the
    # predecessor's output when available.
    for i in range(1, len(rows)):
        layer = rows[i]
        prev = rows[i - 1]
        for io in ('input', 'output'):
            pl_key = f'{io}_pad_logical'
            tens_key = f'{io}_tensors'
            for slot_idx, dims in enumerate(layer[pl_key]):
                corrected = False
                for dim_idx, (pad, logical) in enumerate(dims):
                    if pad != logical or pad == 0:
                        continue
                    # Search predecessor output slot 0 for a dimension with
                    # the same PAD but a different (correct) LOGICAL value.
                    # Check same position first, then any position (handles
                    # transposed outputs like K^T in CreateQKVHeads).
                    prev_out = prev['output_pad_logical']
                    if not prev_out:
                        continue
                    prev_dims = prev_out[0]
                    match_logical: Optional[int] = None
                    if dim_idx < len(prev_dims):
                        pp, pl = prev_dims[dim_idx]
                        if pp == pad and pl != logical:
                            match_logical = pl
                    if match_logical is None:
                        for pp, pl in prev_dims:
                            if pp == pad and pl != logical:
                                match_logical = pl
                                break
                    if match_logical is not None:
                        dims[dim_idx] = (pad, match_logical)
                        corrected = True
                if corrected:
                    layer[tens_key][slot_idx] = 'x'.join(
                        str(lg) for _, lg in dims
                    )

    return rows


def show_layers_profiler(input_file: str) -> None:
    for row in layers_profiler(input_file):
        print(row)


def main() -> int:
    args = parse_args()
    show_layers_profiler(input_file=args.input)
    return 0


if __name__ == '__main__':
    sys.exit(main())
