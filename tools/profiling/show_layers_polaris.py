#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import argparse
import csv
import json
from typing import Any, Dict, List

try:
    from op_canonical import normalize_polaris_optype  # type: ignore[import-not-found]
except ImportError:
    from .op_canonical import normalize_polaris_optype  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Show layers from Polaris CSV')
    parser.add_argument('input', type=str, help='Input CSV file')
    return parser.parse_args()

COLUMNS_OF_INTEREST = ['opnum', 'optype', 'input_tensors', 'output_tensors']


def _parse_tensor_fields(tensor_string: str):
    """Parse a semicolon-separated tensor string into parallel lists of shapes and dtypes.

    Each segment has the format ``name[dim1xdim2]:dtype``.
    Returns (shapes: list[str], dtypes: list[str]).
    """
    shapes = []
    dtypes = []
    for field in tensor_string.split(';'):
        if not field.strip():
            continue
        before_colon, _, dtype_part = field.rpartition(':')
        shape_part = before_colon.split('[')[1].replace(']', '') if '[' in before_colon else ''
        shapes.append(shape_part)
        dtypes.append(dtype_part.strip())
    return shapes, dtypes


def normalize_tensor_string(col: str, tensor_string: str) -> List[str]:
    if 'tensors' not in col:
        return [tensor_string]
    shapes, _ = _parse_tensor_fields(tensor_string)
    return shapes


def layers_polaris(input_file: str) -> List[Dict[str, Any]]:
    rows = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        has_tensor_attributes = None
        for row in reader:
            if has_tensor_attributes is None:
                has_tensor_attributes = 'tensor_attributes' in row

            filtered_row: Dict[str, Any] = {}
            for col in COLUMNS_OF_INTEREST:
                s = normalize_tensor_string(col, row[col])
                if col == 'opnum':
                    filtered_row['seqno'] = int(s[0])
                elif col == 'optype':
                    filtered_row['optype'] = normalize_polaris_optype(s[0])
                else:
                    if s is not None:
                        filtered_row[col] = s

            # Extract dtypes from existing tensor strings
            _, in_dtypes = _parse_tensor_fields(row['input_tensors'])
            _, out_dtypes = _parse_tensor_fields(row['output_tensors'])
            filtered_row['input_dtypes'] = in_dtypes
            filtered_row['output_dtypes'] = out_dtypes

            # Extract layout and memory from tensor_attributes column (if present)
            if has_tensor_attributes:
                ta_raw = row.get('tensor_attributes', '{}')
                try:
                    ta = json.loads(ta_raw)
                except (json.JSONDecodeError, TypeError):
                    ta = {}
                in_attrs = ta.get('inputs', [])
                out_attrs = ta.get('outputs', [])
                # Ensure attribute lists match tensor list lengths by padding with None
                n_in = len(filtered_row.get('input_tensors', []))
                n_out = len(filtered_row.get('output_tensors', []))
                input_layouts = [a.get('layout') for a in in_attrs]
                input_memories = [a.get('memory') for a in in_attrs]
                output_layouts = [a.get('layout') for a in out_attrs]
                output_memories = [a.get('memory') for a in out_attrs]
                # Pad or truncate to match tensor counts
                filtered_row['input_layouts'] = (input_layouts + [None] * n_in)[:n_in]
                filtered_row['input_memories'] = (input_memories + [None] * n_in)[:n_in]
                filtered_row['output_layouts'] = (output_layouts + [None] * n_out)[:n_out]
                filtered_row['output_memories'] = (output_memories + [None] * n_out)[:n_out]
            else:
                n_in = len(filtered_row.get('input_tensors', []))
                n_out = len(filtered_row.get('output_tensors', []))
                filtered_row['input_layouts'] = [None] * n_in
                filtered_row['input_memories'] = [None] * n_in
                filtered_row['output_layouts'] = [None] * n_out
                filtered_row['output_memories'] = [None] * n_out

            # Duration (msecs column → duration_ms)
            msecs_raw = row.get('msecs', '').strip()
            if msecs_raw and msecs_raw.upper() != 'NA':
                try:
                    filtered_row['duration_ms'] = float(msecs_raw)
                except ValueError:
                    filtered_row['duration_ms'] = None
            else:
                filtered_row['duration_ms'] = None

            # LUT hit flag (uses_perf_lookup column)
            lut_raw = row.get('uses_perf_lookup', '').strip().lower()
            filtered_row['uses_perf_lookup'] = lut_raw in ('true', '1', 'yes')

            rows.append(filtered_row)
    return rows

def show_layers_polaris(input_file: str) -> None:
    for row in layers_polaris(input_file):
        print(row)

def main() -> int:
    args = parse_args()
    show_layers_polaris(input_file=args.input)
    return 0


if __name__ == '__main__':
    sys.exit(main())
