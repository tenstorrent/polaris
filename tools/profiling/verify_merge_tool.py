#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify ops_perf_three_csv_merge.py output matches golden merged_ops.csv for all 4 combos.

Runs the merge tool on the raw golden inputs, writes results to a temp directory,
then diffs each output against the corresponding reference merged_ops.csv.
Exits 0 if all combos pass, 1 if any fail.

Safety guarantee: this script NEVER writes to any input directory or overwrites
any golden file. All merge-tool output goes to a temporary directory that is
deleted on exit regardless of outcome.

Typical usage (run from the main worktree that contains __vggoutput / __vitoutput):

    python /path/to/vgg-polaris-442/tools/profiling/verify_merge_tool.py

Or from anywhere with an explicit base dir:

    python verify_merge_tool.py --base-dir /path/to/vgg-polaris
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
# The merge tool was relocated to tools/si_profiling_helpers/ so it ships with the
# rsynced preset bundle and can run on a hardware node. verify_merge_tool stays here.
MERGE_TOOL = SCRIPT_DIR.parent / 'si_profiling_helpers' / 'ops_perf_three_csv_merge.py'

# Each combo: name, input_dir relative to base, reference CSV relative to base,
# DRAM peak BW in GB/s, extra args for the merge tool.
# ops-per-iteration is required for VGG BH because row 0 OP CODE
# (ReshardDeviceOperation) recurs within a single iteration, defeating auto-detection.
# VGG WH does not have this recurrence so auto-detection works there.
_COMBO_DEFS = [
    {
        'name': 'VGG BH',
        'input_dir':  '__vggoutput/golden-blackhole',
        'reference':  '__vggoutput/golden-blackhole/merged_ops.csv',
        'dram_bw':    448.0,
        'extra_args': ['--ops-per-iteration', '107'],
    },
    {
        'name': 'VGG WH',
        'input_dir':  '__vggoutput/golden-wormhole',
        'reference':  '__vggoutput/golden-wormhole/merged_ops.csv',
        'dram_bw':    288.0,
        'extra_args': [],  # auto-detection works for WH (no intra-iteration Reshard recurrence)
        # NOTE: this combo will diff on 14 FPU-util columns (e.g. 'FPU Util Median (%)')
        # because the WH perf CSV was regenerated with lower float precision (9 vs 16 sig-figs)
        # after the golden was captured. The iteration selected, row count, and all non-FPU
        # columns match exactly. This is a source-data precision difference, not a tool bug.
    },
    {
        'name': 'ViT BH',
        'input_dir':  '__vitoutput/golden-blackhole/reports',
        'reference':  '__vitoutput/golden-blackhole/reports/merged_ops.csv',
        'dram_bw':    448.0,
        'extra_args': ['--ops-per-iteration', '195'],  # first op recurs; auto-detect fails
    },
    {
        'name': 'ViT WH',
        'input_dir':  '__vitoutput/golden-wormhole/reports',
        'reference':  '__vitoutput/golden-wormhole/reports/merged_ops.csv',
        'dram_bw':    288.0,
        'extra_args': ['--ops-per-iteration', '206'],  # first op recurs; auto-detect fails
    },
]


def _run_combo(name: str, input_dir: Path, reference: Path,
               dram_bw: float, extra_args: list[str], out_dir: Path) -> tuple[bool, str]:
    out_csv = out_dir / f'{name.replace(" ", "_")}_merged_ops.csv'

    cmd = [
        sys.executable, str(MERGE_TOOL),
        '--input-dir', str(input_dir),
        '--dram-peak-bw-gbps', str(dram_bw),
        '--output', str(out_csv),
        *extra_args,
    ]
    logger.debug('Running: {}', ' '.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip()[:600] or proc.stdout.strip()[:600]
        return False, f'merge tool exited {proc.returncode}:\n{detail}'

    if not out_csv.exists():
        return False, 'merge tool reported success but output file was not created'

    diff = subprocess.run(
        ['diff', str(reference), str(out_csv)],
        capture_output=True, text=True,
    )
    if diff.returncode == 0:
        return True, 'exact match'
    if diff.returncode == 1:
        lines = diff.stdout.splitlines()
        preview = '\n'.join(lines[:40])
        return False, f'{len(lines)} diff line(s) (first 40 shown):\n{preview}'
    # returncode >= 2: diff command itself failed (bad args, missing file, IO error)
    error_detail = (diff.stderr.strip() or diff.stdout.strip())[:300]
    return False, f'diff command failed (exit {diff.returncode}): {error_detail}'


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path.cwd(),
        help='Root directory containing __vggoutput and __vitoutput (default: CWD)',
    )
    args = parser.parse_args()
    base = args.base_dir.resolve()

    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format='<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>',
    )

    if not MERGE_TOOL.exists():
        logger.error('merge tool not found: {}', MERGE_TOOL)
        return 1

    combos = [
        {**d, 'input_dir': base / d['input_dir'], 'reference': base / d['reference']}
        for d in _COMBO_DEFS
    ]

    results: list[tuple[str, bool, str]] = []

    with tempfile.TemporaryDirectory(prefix='verify_merge_') as tmpdir:
        out_dir = Path(tmpdir)
        for c in combos:
            name = c['name']
            if not c['input_dir'].exists():
                logger.error('{}: input_dir not found: {}', name, c['input_dir'])
                results.append((name, False, 'input_dir not found'))
                continue
            if not c['reference'].exists():
                logger.error('{}: reference not found: {}', name, c['reference'])
                results.append((name, False, 'reference not found'))
                continue

            logger.info('{}: running ...', name)
            ok, detail = _run_combo(
                name=name,
                input_dir=c['input_dir'],
                reference=c['reference'],
                dram_bw=c['dram_bw'],
                extra_args=c['extra_args'],
                out_dir=out_dir,
            )
            if ok:
                logger.success('{}: PASS — {}', name, detail)
            else:
                logger.error('{}: FAIL\n{}', name, detail)
            results.append((name, ok, detail))

    width = max(len(r[0]) for r in results)
    print()
    print('─' * (width + 12))
    failures = 0
    for name, ok, _ in results:
        tag = 'PASS' if ok else 'FAIL'
        print(f'  {tag}  {name}')
        if not ok:
            failures += 1
    print('─' * (width + 12))
    print()
    if failures:
        logger.error('{}/{} combo(s) failed', failures, len(results))
        return 1
    logger.success('All {} combos passed', len(results))
    return 0


if __name__ == '__main__':
    sys.exit(main())
