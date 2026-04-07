#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import argparse
import csv
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Show layers from Polaris CSV')
    parser.add_argument('input', type=str, help='Input CSV file')
    return parser.parse_args()

COLUMNS_OF_INTEREST = ['opnum', 'optype', 'input_tensors', 'output_tensors']


def normalize_tensor_string(col: str, tensor_string: str) -> List[str]:
    if 'tensors' not in col:
        return [tensor_string]
    fields = tensor_string.split(';')
    normalized_fields = []
    for field in fields:
        tmp = field.split(':')[0].split('[')[1].replace(']', '')
        normalized_fields.append(tmp)
    return normalized_fields


def layers_polaris(input_file: str) -> List[Dict[str, Any]]:
    rows = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filtered_row = {}
            for col in COLUMNS_OF_INTEREST:
                s = normalize_tensor_string(col, row[col])
                if col == 'opnum':
                    filtered_row['seqno'] = int(s[0])
                elif col == 'optype':
                    filtered_row['optype'] = s[0].lower()
                else:
                    if s is not None:
                        filtered_row[col] = s
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
