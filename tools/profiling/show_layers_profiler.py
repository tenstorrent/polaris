#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import argparse
import csv
import re
import yaml
from typing import Any, Dict, List, Optional

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Show layers from profiler CSV')
    parser.add_argument('input', type=str, help='Input CSV file')
    return parser.parse_args()

COLUMNS_OF_INTEREST = ['GLOBAL CALL COUNT', 'OP CODE']


def expand_tensor_string(row: Dict[str, Any], input_index: int, in_or_out: str='INPUT') -> str:
    normalized_fields = []
    for tmp in ['W', 'Z', 'Y', 'X']:
        tmpvalue = row[f'{in_or_out}_{input_index}_{tmp}_PAD[LOGICAL]']
        if tmpvalue == '':
            continue
        value = tmpvalue.split('[')[1].replace(']', '')
        normalized_fields.append(value)
    return 'x'.join(normalized_fields)


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
            filtered_row = {
                'seqno': int(row['GLOBAL CALL COUNT']),
                'optype': row['OP CODE'].lower(),
                'input_tensors': [expand_tensor_string(row, 0, 'INPUT')],
                'output_tensors': [expand_tensor_string(row, 0, 'OUTPUT')]
            }
            for idx in (1, 2):
                s = expand_tensor_string(row, idx, 'INPUT')
                if s:
                    filtered_row['input_tensors'].append(s)
                s = expand_tensor_string(row, idx, 'OUTPUT')
                if s:
                    filtered_row['output_tensors'].append(s)
            rows.append(filtered_row)
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
