#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Sum Polaris opstats durations by operator type.

Defaults for Polaris STATS opstats CSV:
- Operator column: "optype"
- Duration column: "msecs" (already milliseconds)

Header matching is case-insensitive and whitespace-tolerant.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _normalize_header(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


def _parse_duration(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_column_name(headers: list[str], desired_name: str) -> str:
    normalized_to_actual = {_normalize_header(h): h for h in headers}
    wanted = _normalize_header(desired_name)
    if wanted not in normalized_to_actual:
        raise KeyError(f"Column '{desired_name}' not found in file headers.")
    return normalized_to_actual[wanted]


def _load_rows(input_csv: Path) -> list[dict[str, Any]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_opstats_duration_ms(
    rows: list[dict[str, Any]],
    operator_column: str,
    duration_column: str,
    duration_unit: str,
) -> dict[str, tuple[int, float]]:
    if not rows:
        return {}

    headers = [str(h) for h in rows[0].keys() if h is not None]
    operator_col_actual = _resolve_column_name(headers, operator_column)
    duration_col_actual = _resolve_column_name(headers, duration_column)

    unit = duration_unit.strip().lower()
    if unit not in {"ms", "ns"}:
        raise ValueError("duration_unit must be one of: ms, ns")

    totals_ms: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        operator_type = str(row.get(operator_col_actual, "")).strip()
        if not operator_type:
            continue
        duration_val = _parse_duration(row.get(duration_col_actual))
        if duration_val is None:
            continue
        duration_ms = duration_val / 1_000_000.0 if unit == "ns" else duration_val
        totals_ms[operator_type] += duration_ms
        counts[operator_type] += 1
    return {op: (counts[op], total_ms) for op, total_ms in totals_ms.items()}


def _write_csv(output_path: Path, totals_ms: dict[str, tuple[int, float]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["operator_type", "entry_count", "duration_ms"])
        for operator_type, (count, total_ms) in sorted(
            totals_ms.items(), key=lambda kv: kv[1][1], reverse=True
        ):
            writer.writerow([operator_type, count, f"{total_ms:.6f}"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Sum Polaris opstats duration grouped by operator type.")
    parser.add_argument("input_csv", type=Path, help="Polaris opstats CSV path")
    parser.add_argument(
        "--operator-column",
        type=str,
        default="optype",
        help="Operator type column name (default: optype).",
    )
    parser.add_argument(
        "--duration-column",
        type=str,
        default="msecs",
        help="Duration column name (default: msecs).",
    )
    parser.add_argument(
        "--duration-unit",
        type=str,
        default="ms",
        choices=["ms", "ns"],
        help="Unit for --duration-column values; converted to ms in output.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.input_csv)
    totals_ms = summarize_opstats_duration_ms(
        rows=rows,
        operator_column=args.operator_column,
        duration_column=args.duration_column,
        duration_unit=args.duration_unit,
    )
    grand_total_ms = sum(total_ms for _, total_ms in totals_ms.values())
    grand_total_count = sum(count for count, _ in totals_ms.values())

    if args.output_csv is not None:
        _write_csv(args.output_csv, totals_ms)
        print(f"Wrote {len(totals_ms)} rows to {args.output_csv}")
        print(f"total_entries,{grand_total_count}")
        print(f"total_duration_ms,{grand_total_ms:.6f}")
        return 0

    print("operator_type,entry_count,duration_ms")
    for operator_type, (count, total_ms) in sorted(totals_ms.items(), key=lambda kv: kv[1][1], reverse=True):
        print(f"{operator_type},{count},{total_ms:.6f}")
    print(f"total_entries,{grand_total_count}")
    print(f"total_duration_ms,{grand_total_ms:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
