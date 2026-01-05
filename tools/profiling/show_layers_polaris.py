#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser(description='Show layers from Polaris CSV')
    parser.add_argument('input', type=str, help='Input CSV file')
    return parser.parse_args()

COLUMNS_OF_INTEREST = ['opnum', 'optype', 'input_tensors', 'output_tensors']


def normalize_tensor_string(col: str,tensor_string: str) -> str:
    if 'tensors' not in col:
        return tensor_string
    fields = tensor_string.split(';')
    normalized_fields = []
    for field in fields:
        tmp = field.split(':')[0].split('[')[1].replace(']', '')
        normalized_fields.append(tmp)
    return ';'.join(normalized_fields)


def show_layers_polaris(input_file: str):
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filtered_row = [normalize_tensor_string(col, row[col]) for col in COLUMNS_OF_INTEREST]
            print(filtered_row)


def main():
    args = parse_args()
    show_layers_polaris(input_file=args.input)
    return 0


if __name__ == '__main__':
    sys.exit(main())
