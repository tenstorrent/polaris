#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Print operator type and input tensor shapes for each row of a Polaris *opstats* CSV."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def _shapes_from_input_tensors_field(field: str) -> list[str]:
    """
    Parse ``input_tensors`` cell:
    ``input_0[8x224x768]:BFLOAT16;input_1[1x768x768]:BFLOAT16`` → ``['8x224x768', '1x768x768']``.
    """
    out: list[str] = []
    for chunk in field.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.search(r'\[([^\]]+)\]', chunk)
        if not m:
            continue
        inner = m.group(1).replace('×', 'x')
        out.append(inner)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv_path', type=Path, help='Polaris opstats CSV (e.g. *opstats*.csv)')
    args = ap.parse_args()
    path = args.csv_path
    if not path.is_file():
        print(f'error: not a file: {path}', file=sys.stderr)
        return 1

    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print('error: empty CSV', file=sys.stderr)
            return 1
        if 'optype' not in reader.fieldnames:
            print('error: CSV missing required column "optype"', file=sys.stderr)
            return 1
        if 'input_tensors' not in reader.fieldnames:
            print('error: CSV missing required column "input_tensors"', file=sys.stderr)
            return 1

        print('row\toperator_type\tinput_shapes')
        for i, row in enumerate(reader):
            row = {str(k).strip(): (v or '').strip() if v is not None else '' for k, v in row.items()}
            optype = row.get('optype', '').strip()
            shapes = _shapes_from_input_tensors_field(row.get('input_tensors', ''))
            print(f'{i + 1}\t{optype}\t{"; ".join(shapes)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
