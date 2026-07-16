#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Merge three TT-Metal ops perf CSVs from a **non-iterative trace+replay** capture.

Specification: doc/SPEC_ops_perf_trace_replay_merge.md

This is the trace+replay sibling of ``ops_perf_three_csv_merge.py`` (the iterative,
VGG-style merge).  It reuses that module's classify / header-check / join / output
machinery verbatim and only swaps the *row-reduction* step: instead of detecting
and selecting one iteration, it selects one **replay session** and drops the
compile/warmup pass, so the join keys are unique.

Why a separate reducer
----------------------
A trace+replay capture (ViT, llama3 decode/prefill) is *not* the concatenation of
equal-size iterations that the iterative reducer expects.  It is one warmup/compile
pass followed by N replays of a captured trace.  TT-Metal tags each op with
``METAL TRACE REPLAY SESSION ID``:

* ``''`` (blank)  -> the compile/warmup pass (ops dispatched individually before the
  trace is captured); its op set overlaps the replay op set, so keeping it makes the
  ``(GLOBAL CALL COUNT, OP CODE, OP TYPE)`` join key non-unique.
* ``'1'``, ``'2'``, ...  -> successive replays of the captured trace; within any one
  replay session the join keys are unique and identical across the raw/perf/trace
  variants.

The iterative reducer's first-op marker heuristic mis-fires here (the marker recurs
intra-pass, giving unequal chunks), and the downstream join then aborts on the
duplicate key.  This reducer instead keeps exactly one replay session — steady-state,
compile cost excluded — and hands unique keys to the shared join.

Composition (trace+replay+iterative)
------------------------------------
If a future capture is trace+replay **and** iterative (a replay session that itself
contains several concatenated iterations), the selected session's join keys will not
be unique.  In that case (unless ``--no-compose-iteration``) the iterative reducer is
applied *within* the selected session, composing the two reductions.  For the common
one-inference-per-session case the keys are already unique and no iteration reduction
runs (so the iterative marker heuristic — and its warning — is never triggered).

Uses stdlib **csv**; **loguru** for warnings/info.  Exit 0 on success, 1 on error.
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

# Reuse the shared machinery from the iterative merge tool verbatim.  Only the
# row-reduction strategy differs (replay-session selection vs iteration selection).
# Dual import so this works both as a package module under pytest / `mypy ./`
# (imported as tools.si_profiling_helpers.*) and as an on-device script, where the
# rsynced bundle is a flat directory of scripts with no package path (the bare name
# resolves because the script's own directory is sys.path[0]).  The package import
# is listed first so `mypy ./` (run from the repo root) resolves the module.
try:
    from tools.si_profiling_helpers import ops_perf_three_csv_merge as _merge
except ModuleNotFoundError:  # on-device flat script bundle: no package path
    import ops_perf_three_csv_merge as _merge  # type: ignore[import-not-found, no-redef]

COL_DEVICE_KERNEL_NS = _merge.COL_DEVICE_KERNEL_NS
COL_GLOBAL_CALL_COUNT = _merge.COL_GLOBAL_CALL_COUNT
COL_OP_CODE = _merge.COL_OP_CODE
COL_OP_TYPE = _merge.COL_OP_TYPE
MergeError = _merge.MergeError
parse_finite_float = _merge.parse_finite_float
reduce_iterative = _merge.reduce_iterative
run_merge = _merge.run_merge
_strip_cell = _merge._strip_cell

# TT-Metal tags each op with the replay session it belongs to; the compile/warmup
# pass carries a blank id.  (METAL TRACE ID also exists but is not needed for
# reduction — the replay session id alone discriminates warmup from each replay.)
COL_REPLAY_SESSION_ID = "METAL TRACE REPLAY SESSION ID"

SESSION_SELECT_CHOICES: Tuple[str, ...] = ("median", "min", "max", "first", "last")


def _session_id(row: Dict[str, str]) -> str:
    """Stripped replay-session id for a row ('' = compile/warmup pass)."""
    return _strip_cell(row.get(COL_REPLAY_SESSION_ID))


def replay_session_ids(rows: Sequence[Dict[str, str]]) -> List[str]:
    """Distinct non-blank replay-session ids, in first-appearance (capture) order."""
    seen: set = set()
    out: List[str] = []
    for r in rows:
        sid = _session_id(r)
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _session_total_kdur_ns(rows: Sequence[Dict[str, str]], sid: str) -> float:
    """Sum of DEVICE KERNEL DURATION [ns] over the rows of one replay session."""
    total = 0.0
    for r in rows:
        if _session_id(r) == sid:
            v = parse_finite_float(r.get(COL_DEVICE_KERNEL_NS))
            if v is not None:
                total += v
    return total


def select_replay_session(rows_van: Sequence[Dict[str, str]], select: str) -> str:
    """Return the replay-session id to keep, chosen from the vanilla CSV.

    Selection vocabulary matches the iterative tool's ``--select-iteration``:
      * ``median`` — lower-middle by total kernel ns (stable; the default).
      * ``min`` / ``max`` — least / greatest total kernel ns.
      * ``first`` / ``last`` — earliest / latest replay session in capture order.

    The compile/warmup pass (blank session id) is never a candidate.
    """
    sessions = replay_session_ids(rows_van)
    if not sessions:
        raise MergeError(
            f"No populated {COL_REPLAY_SESSION_ID!r} values found — this does not look like a "
            "trace+replay capture. Use ops_perf_three_csv_merge.py for iterative (VGG-style) runs."
        )
    if select in ("first", "last"):
        chosen = sessions[0] if select == "first" else sessions[-1]
    else:
        totals = [(sid, _session_total_kdur_ns(rows_van, sid)) for sid in sessions]
        if select == "min":
            chosen = min(totals, key=lambda x: x[1])[0]
        elif select == "max":
            chosen = max(totals, key=lambda x: x[1])[0]
        elif select == "median":
            ordered = sorted(totals, key=lambda x: x[1])
            chosen = ordered[(len(ordered) - 1) // 2][0]  # lower-middle for even counts
        else:
            raise MergeError(f"Unknown --select-session value: {select!r}")
    return chosen


def _keys_unique(rows: Sequence[Dict[str, str]]) -> bool:
    """True if every (GLOBAL CALL COUNT, OP CODE, OP TYPE) join key in rows is distinct."""
    seen: set = set()
    for r in rows:
        k = (
            _strip_cell(r.get(COL_GLOBAL_CALL_COUNT)),
            _strip_cell(r.get(COL_OP_CODE)),
            _strip_cell(r.get(COL_OP_TYPE)),
        )
        if k in seen:
            return False
        seen.add(k)
    return True


def reduce_trace_replay(
    rows_noc: List[Dict[str, str]],
    rows_fpu: List[Dict[str, str]],
    rows_van: List[Dict[str, str]],
    *,
    select_session: str = "median",
    compose_iteration: bool = True,
    cli_ops_per_iteration: Optional[int] = None,
    cli_measured_indices: Optional[List[int]] = None,
    cli_select_iteration: str = "median",
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Reduce a trace+replay capture to a single replay session.

    Steps: choose one replay session from the vanilla CSV, then filter all three
    variants to that session (dropping the compile/warmup pass and the other
    sessions).  If the kept session still has duplicate join keys (a trace+replay
    *and* iterative capture) and ``compose_iteration`` is set, apply the iterative
    reducer within the session to collapse it to one iteration.
    """
    if COL_REPLAY_SESSION_ID not in (rows_van[0] if rows_van else {}):
        raise MergeError(
            f"Vanilla CSV has no {COL_REPLAY_SESSION_ID!r} column — not a trace+replay capture. "
            "Use ops_perf_three_csv_merge.py for iterative (VGG-style) runs."
        )

    sessions = replay_session_ids(rows_van)
    chosen = select_replay_session(rows_van, select_session)

    def _keep(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return [r for r in rows if _session_id(r) == chosen]

    n_compile = sum(1 for r in rows_van if not _session_id(r))
    rows_noc, rows_fpu, rows_van = _keep(rows_noc), _keep(rows_fpu), _keep(rows_van)
    logger.info(
        "Replay-session filter: {} session(s) {}; dropped {} compile/warmup op(s); "
        "selected session {!r} (select={!r}) -> {} op(s) per variant.",
        len(sessions),
        sessions,
        n_compile,
        chosen,
        select_session,
        len(rows_van),
    )

    if not _keys_unique(rows_van):
        if compose_iteration:
            logger.info(
                "Selected replay session {!r} has duplicate join keys "
                "(trace+replay+iterative); composing the iteration reducer within the session.",
                chosen,
            )
            rows_noc, rows_fpu, rows_van = reduce_iterative(
                rows_noc,
                rows_fpu,
                rows_van,
                cli_ops_per_iteration=cli_ops_per_iteration,
                cli_measured_indices=cli_measured_indices,
                cli_select_iteration=cli_select_iteration,
            )
        else:
            raise MergeError(
                f"Selected replay session {chosen!r} has duplicate join keys and "
                "--no-compose-iteration was given; the join requires unique keys. "
                "Re-run without --no-compose-iteration to reduce iterations within the session."
            )
    return rows_noc, rows_fpu, rows_van


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Merge three TT-Metal ops perf CSVs from a non-iterative trace+replay capture "
            "(ViT, llama3) under --input-dir into one CSV, keeping a single replay session."
        ),
        epilog=(
            "Default --output is <input-dir>/merged_ops.csv. For iterative (VGG-style) runs use "
            "ops_perf_three_csv_merge.py instead. See doc/SPEC_ops_perf_trace_replay_merge.md."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Root directory; recursively discovers exactly three ops_perf_results_*.csv files "
            "(falls back to *.csv when none match), classified by content."
        ),
    )
    p.add_argument(
        "--dram-peak-bw-gbps",
        type=float,
        required=True,
        help="Peak DRAM bandwidth in GB/s (used with DRAM BW UTIL (%%) for MemTraffic).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input-dir>/merged_ops.csv).",
    )
    p.add_argument(
        "--duration-rel-tol",
        type=float,
        default=0.05,
        help="Relative tolerance comparing DEVICE KERNEL DURATION [ns] between fpu and vanilla rows.",
    )
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output CSV files.",
    )
    p.add_argument(
        "--select-session",
        choices=SESSION_SELECT_CHOICES,
        default="median",
        help=(
            "How to choose the replay session to keep: 'median' (lower-middle by total kernel "
            "ns), 'min', 'max', 'first', or 'last' (capture order). The compile/warmup pass "
            "(blank session id) is always dropped."
        ),
    )
    p.add_argument(
        "--no-compose-iteration",
        action="store_true",
        help=(
            "Disable the iteration reducer that otherwise runs when the selected replay session "
            "still has duplicate join keys (a trace+replay+iterative capture). With this flag such "
            "a capture is a hard error instead."
        ),
    )
    # Forwarded to the composed iteration reducer; only take effect for a
    # trace+replay+iterative capture (a replay session with duplicate keys).
    p.add_argument(
        "--ops-per-iteration",
        type=int,
        default=None,
        help="Ops per iteration for the composed iteration reducer (trace+replay+iterative only).",
    )
    p.add_argument(
        "--measured-iteration-indices",
        type=str,
        default=None,
        help="Comma-separated 0-based iteration indices for the composed iteration reducer.",
    )
    p.add_argument(
        "--select-iteration",
        choices=("median", "min", "max", "first", "last"),
        default="median",
        help="Iteration selection for the composed iteration reducer (trace+replay+iterative only).",
    )
    return p


def _parse_measured_indices(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError as e:
            raise MergeError(f"--measured-iteration-indices: cannot parse {part!r} as int") from e
    return out if out else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    input_dir = args.input_dir.resolve()
    out = args.output
    out = input_dir / "merged_ops.csv" if out is None else Path(out).resolve()
    try:
        measured = _parse_measured_indices(args.measured_iteration_indices)
        reducer = partial(
            reduce_trace_replay,
            select_session=args.select_session,
            compose_iteration=not args.no_compose_iteration,
            cli_ops_per_iteration=args.ops_per_iteration,
            cli_measured_indices=measured,
            cli_select_iteration=args.select_iteration,
        )
        run_merge(
            input_dir=input_dir,
            output_path=out,
            dram_peak_bw_gbps=float(args.dram_peak_bw_gbps),
            duration_rel_tol=float(args.duration_rel_tol),
            encoding=str(args.encoding),
            reducer=reducer,
        )
    except MergeError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
