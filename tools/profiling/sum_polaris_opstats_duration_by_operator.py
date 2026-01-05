#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Sum Polaris opstats durations by operator type.

Defaults for Polaris STATS opstats CSV:
- Operator column: "optype"
- Duration column: "msecs" (already milliseconds)
- Optional: "uses_perf_lookup" — when the column is present (and not disabled via CLI),
  also emits LUT hit/miss rollups as separate rows (see LUT_MATCHES_ROW / LUT_MISSES_ROW).

Header matching is case-insensitive and whitespace-tolerant.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# CSV / stdout labels for perf-lookup rollups (not Polaris optypes; kept out of grand totals).
LUT_MATCHES_ROW = "LUT-Matches"
LUT_MISSES_ROW = "LUT-Mismatches"


@dataclass(frozen=True)
class DurationBucket:
    """entry_count rows contributing; total duration in milliseconds."""

    entry_count: int
    duration_ms: float


@dataclass(frozen=True)
class OpstatsDurationSummary:
    """Per-optype totals plus optional LUT hit/miss splits."""

    by_operator: dict[str, DurationBucket]
    lut_matches: DurationBucket | None
    lut_mismatches: DurationBucket | None


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


def _try_resolve_column_name(headers: list[str], desired_name: str) -> str | None:
    try:
        return _resolve_column_name(headers, desired_name)
    except KeyError:
        return None


def _cell_truthy(raw: Any) -> bool:
    """Interpret CSV / bool cell as LUT hit (matches Polaris bool written as True/False strings)."""
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, float) and math.isnan(raw):
        return False
    text = str(raw).strip().lower()
    return text in ("true", "1", "yes")


def _load_rows(input_csv: Path) -> list[dict[str, Any]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_opstats_duration_ms(
    rows: list[dict[str, Any]],
    operator_column: str,
    duration_column: str,
    duration_unit: str,
    uses_lookup_column: str,
) -> OpstatsDurationSummary:
    if not rows:
        return OpstatsDurationSummary(by_operator={}, lut_matches=None, lut_mismatches=None)

    headers = [str(h) for h in rows[0].keys() if h is not None]
    operator_col_actual = _resolve_column_name(headers, operator_column)
    duration_col_actual = _resolve_column_name(headers, duration_column)
    uses_key = uses_lookup_column.strip()
    uses_col_actual = _try_resolve_column_name(headers, uses_key) if uses_key else None

    unit = duration_unit.strip().lower()
    if unit not in {"ms", "ns"}:
        raise ValueError("duration_unit must be one of: ms, ns")

    totals_ms: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    lut_match_ms = 0.0
    lut_match_n = 0
    lut_miss_ms = 0.0
    lut_miss_n = 0

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
        if uses_col_actual is not None:
            if _cell_truthy(row.get(uses_col_actual)):
                lut_match_ms += duration_ms
                lut_match_n += 1
            else:
                lut_miss_ms += duration_ms
                lut_miss_n += 1

    by_operator = {
        op: DurationBucket(counts[op], ms) for op, ms in totals_ms.items()
    }
    if uses_col_actual is None:
        return OpstatsDurationSummary(
            by_operator=by_operator, lut_matches=None, lut_mismatches=None
        )
    return OpstatsDurationSummary(
        by_operator=by_operator,
        lut_matches=DurationBucket(lut_match_n, lut_match_ms),
        lut_mismatches=DurationBucket(lut_miss_n, lut_miss_ms),
    )


def _iter_output_rows(
    summary: OpstatsDurationSummary,
) -> list[tuple[str, int, float]]:
    """Sorted operator rows, then optional LUT rows (fixed order)."""
    rows: list[tuple[str, int, float]] = [
        (op, b.entry_count, b.duration_ms)
        for op, b in sorted(
            summary.by_operator.items(), key=lambda kv: kv[1].duration_ms, reverse=True
        )
    ]
    if summary.lut_matches is not None and summary.lut_mismatches is not None:
        b, c = summary.lut_matches, summary.lut_mismatches
        rows.append((LUT_MATCHES_ROW, b.entry_count, b.duration_ms))
        rows.append((LUT_MISSES_ROW, c.entry_count, c.duration_ms))
    return rows


def _write_csv(output_path: Path, summary: OpstatsDurationSummary) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["operator_type", "entry_count", "duration_ms"])
        for operator_type, count, total_ms in _iter_output_rows(summary):
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
        "--uses-lookup-column",
        type=str,
        default="uses_perf_lookup",
        help=(
            "Column for tt-perf LUT hit (default: uses_perf_lookup). "
            "Empty string disables LUT hit/miss rollups even if the file has that column."
        ),
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
    summary = summarize_opstats_duration_ms(
        rows=rows,
        operator_column=args.operator_column,
        duration_column=args.duration_column,
        duration_unit=args.duration_unit,
        uses_lookup_column=args.uses_lookup_column,
    )
    grand_total_ms = sum(b.duration_ms for b in summary.by_operator.values())
    grand_total_count = sum(b.entry_count for b in summary.by_operator.values())
    n_out = len(summary.by_operator) + (
        2 if summary.lut_matches is not None else 0
    )

    if args.output_csv is not None:
        _write_csv(args.output_csv, summary)
        print(f"Wrote {n_out} rows to {args.output_csv}")
        print(f"total_entries,{grand_total_count}")
        print(f"total_duration_ms,{grand_total_ms:.6f}")
        return 0

    print("operator_type,entry_count,duration_ms")
    for operator_type, count, total_ms in _iter_output_rows(summary):
        print(f"{operator_type},{count},{total_ms:.6f}")
    print(f"total_entries,{grand_total_count}")
    print(f"total_duration_ms,{grand_total_ms:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
