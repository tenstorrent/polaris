#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Print operator type and input tensor shapes for each row of a TTNN ops perf (profiler) CSV."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import yaml
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.profiling.profiler_to_polaris_converter import parse_tensor_dimensions  # noqa: E402

import tools.profiling.profiler_polaris_opname_mapping as _ppm  # noqa: E402


def _parse_profiler_attributes(raw: str | None) -> dict[str, Any]:
    if not raw or not str(raw).strip():
        return {}
    try:
        parsed = yaml.safe_load(str(raw).replace(';', ','))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _input_indices_from_row(row: dict[str, str]) -> list[int]:
    seen: set[int] = set()
    for k in row:
        m = re.match(r'^INPUT_(\d+)_W_PAD\[', k)
        if m:
            seen.add(int(m.group(1)))
    return sorted(seen)


def _shape_str(dims: list[int]) -> str:
    return 'x'.join(str(d) for d in dims)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv_path', type=Path, help='TTNN ops_perf_results_*.csv (profiler export)')
    ap.add_argument(
        '--op-type',
        choices=('polaris', 'op_code'),
        default='polaris',
        help='polaris: mapped optype (matches Polaris opstats); op_code: raw OP CODE column',
    )
    ap.add_argument(
        '--pad-extent',
        choices=('LOGICAL', 'PADDED'),
        default='LOGICAL',
        help='Which INPUT_*_W_PAD[...] bracket suffix the CSV uses',
    )
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
        print('row\toperator_type\tinput_shapes')
        for i, row in enumerate(reader):
            row = {str(k).strip(): (v or '').strip() if v is not None else '' for k, v in row.items()}
            op_code = row.get('OP CODE', '').strip()
            attrs = _parse_profiler_attributes(row.get('ATTRIBUTES'))
            if args.op_type == 'polaris':
                op_t = _ppm._map_profiler_opcode_to_polaris_optype(op_code, attrs)
            else:
                op_t = op_code

            shapes: list[str] = []
            for idx in _input_indices_from_row(row):
                info = parse_tensor_dimensions(row, 'INPUT', idx, pad_extent=args.pad_extent)
                if info:
                    shapes.append(_shape_str(info['dims']))

            print(f'{i + 1}\t{op_t}\t{"; ".join(shapes)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
