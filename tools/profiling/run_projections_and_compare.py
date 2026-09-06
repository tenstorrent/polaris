#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Drive polaris projections + compare_layers XLSX comparisons over a fixed (workload × device) matrix.

v1 hardcodes the combo list.  Future iterations: auto-discover from
``config/all_workloads.yaml`` + ``config/tt_{wh,bh}.yaml``.

HW-reference CSVs are not kept in the tree.  Each combo names its reference by the path it occupies
under ``hlm-refrun/`` on LFC, and ``ttsim.utils.lfc.resolve_lfc_path`` fetches it on first use into
``__ext/hlm-refrun/`` (re-checked weekly, falling back to the cached copy when LFC is unreachable) —
the same mechanism the arch configs use for ``operator_lookup_file``.

Usage::

    python tools/profiling/run_projections_and_compare.py --run-tag my_run

Outputs (under ``__output/<run-tag>/``):
  - ``<combo>/STATS/...-opstats.csv``       (polaris)
  - ``<combo>/<combo>_comparison.xlsx``     (compare_layers)
  - ``<combo>/polaris.log``                 (polaris stdout/stderr)
  - ``<combo>/compare.log``                 (compare_layers stdout/stderr)
  - ``summary.md``                          (the aggregated gap / LUT-hit table, also printed)
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttsim.utils.lfc import resolve_lfc_path  # noqa: E402

# LFC subtree holding the HW-reference merged-ops CSVs; see its README for the layout rules.
REFRUN_LFC_ROOT = 'hlm-refrun'


@dataclass
class Combo:
    """One (workload × device) verification cell."""
    label: str               # short tag for output dir + summary row
    archfile: str            # path to arch YAML (relative to REPO_ROOT)
    filterarch: str          # --filterarch value (e.g. 'n150', 'p100a')
    filterwli: str           # --filterwli value (workload instance name)
    workload_yaml_name: str  # the wlname inside config/all_workloads.yaml's group
    instance_name: str       # the wli inside the group (usually == filterwli)
    bs_runtime: int          # batch size to use (--batchsize override; matches HW refrun)
    refrun_rel: str          # HW refrun CSV, as <workload>/<arch_family>/<sku>/<file> under
    #                          hlm-refrun/ on LFC


COMBOS: list[Combo] = [
    Combo(
        label='vit_wh_n150',
        archfile='config/tt_wh.yaml',
        filterarch='n150',
        filterwli='vitoptim_wh_b16_device_perf',
        workload_yaml_name='vitoptim_wh_device_perf',
        instance_name='vitoptim_wh_b16_device_perf',
        bs_runtime=8,
        refrun_rel='vit/wh/n150/merged_ops_opt_sharded_260415.csv',
    ),
    Combo(
        label='vit_bh_p100a',
        archfile='config/tt_bh.yaml',
        filterarch='p100a',
        filterwli='vitoptim_bh_b16_device_perf',
        workload_yaml_name='vitoptim_bh_device_perf',
        instance_name='vitoptim_bh_b16_device_perf',
        bs_runtime=10,
        refrun_rel='vit/bh/p100a/merged_ops_opt_sharded_260426.csv',
    ),
    Combo(
        label='vgg_wh_n150',
        archfile='config/tt_wh.yaml',
        filterarch='n150',
        filterwli='vgg_unet_wh_b1_device_perf',
        workload_yaml_name='vgg_unet_wh_device_perf',
        instance_name='vgg_unet_wh_b1_device_perf',
        bs_runtime=1,
        refrun_rel='vgg_unet/wh/n150/merged_ops_dualref_260515.csv',
    ),
    Combo(
        label='vgg_bh_p100a',
        archfile='config/tt_bh.yaml',
        filterarch='p100a',
        filterwli='vgg_unet_bh_b1_device_perf',
        workload_yaml_name='vgg_unet_bh_device_perf',
        instance_name='vgg_unet_bh_b1_device_perf',
        bs_runtime=1,
        refrun_rel='vgg_unet/bh/p100a/merged_ops_dualref_260519.csv',
    ),
    # llama3 dual-mode decode (BH p100a). Its arch config + workload landed on main 2026-08-07, so
    # this combo is live here. bs=1 / seq_len=1024 (set on the all_workloads instance) match the
    # batch-1 reference. Verified 100% (749/749) at +0.53% gap 2026-08-10. Combos whose arch config
    # or refrun CSV is absent on a given branch are auto-skipped by validate_combos().
    Combo(
        label='llama3_decode_bh_p100a',
        archfile='config/tt_bh_llama3.yaml',
        filterarch='p100a',
        filterwli='llama3_8b_dualmode_decode_b1',
        workload_yaml_name='llama3_dualmode_decode',
        instance_name='llama3_8b_dualmode_decode_b1',
        bs_runtime=1,
        refrun_rel='llama3/bh/p100a/merged_ops_decode_260703.csv',
    ),
]


def opstats_path_for(run_tag: str, combo: Combo) -> Path:
    return (
        REPO_ROOT / '__output' / run_tag / combo.label / 'STATS' /
        f'{combo.filterarch}-TTNN-{combo.workload_yaml_name}-{combo.instance_name}'
        f'-b{combo.bs_runtime}-opstats.csv'
    )


def xlsx_path_for(run_tag: str, combo: Combo) -> Path:
    return REPO_ROOT / '__output' / run_tag / combo.label / f'{combo.label}_comparison.xlsx'


def log_path_for(run_tag: str, combo: Combo, name: str) -> Path:
    """One construction site for the per-combo log paths, so clearing cannot drift from writing."""
    return REPO_ROOT / '__output' / run_tag / combo.label / name


def summary_path_for(run_tag: str) -> Path:
    """The run tag's top-level report. One construction site, for the same reason as the logs."""
    return REPO_ROOT / '__output' / run_tag / 'summary.md'


def run_polaris(combo: Combo, run_tag: str) -> Path:
    """Run polaris.py for this combo. Returns the opstats CSV path."""
    study = f'{run_tag}/{combo.label}'
    cmd = [
        sys.executable, 'polaris.py',
        '-w', 'config/all_workloads.yaml',
        '-a', combo.archfile,
        '-m', 'config/wl2archmapping.yaml',
        '--filterarch', combo.filterarch,
        '--filterwli', combo.filterwli,
        '--study', study,
        '-o', '__output/',
        '--dump_stats_csv',
    ]
    if combo.bs_runtime != 1:
        cmd.extend(['--batchsize', '1', str(combo.bs_runtime), str(combo.bs_runtime)])
    log_path = log_path_for(run_tag, combo, 'polaris.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The caller has already cleared any CSV left by an earlier run under this tag (see
    # clear_stale_outputs), so the check below tests what *this* run produced -- otherwise a
    # surviving stale file would be compared as though it described the current run.
    # This used to carry a second job: polaris exited 0 with "completed with 0 experiments" when
    # --filterarch/--filterwli selected nothing, so check=True could not catch a bad filter. It
    # now rejects an unmatched filter entry and a zero-experiment selection itself (issue #517),
    # so check=True is sufficient for that case and only the staleness concern remains here.
    opstats = opstats_path_for(run_tag, combo)
    with log_path.open('w') as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, check=True)
    if not opstats.exists():
        raise FileNotFoundError(
            f'polaris ran but opstats CSV not found: {opstats}\n'
            f'see log: {log_path}'
        )
    return opstats


def clear_stale_outputs(run_tag: str, combo: Combo, *, keep_opstats: bool) -> None:
    """Delete this combo's outputs from any earlier run under the same tag, before anything runs.

    Called for every *selected* combo before validation, not per producing step and not per runnable
    combo. A combo can drop out at validation, at polaris, at compare, or at parsing, and in every
    one of those cases it is correctly left out of the summary while a file from a previous run stays
    in the output directory looking current. Clearing next to a producing step only covers failures
    after that step; clearing inside the run loop misses combos that never reach it.

    The logs are cleared here too. ``open('w')`` truncates them, but only for a combo that actually
    reaches a subprocess -- a combo dropped at validation opens neither file, so an earlier run's
    logs would sit beside the new report. Every output named in the module docstring is covered here,
    which is the point: no producing step is trusted to clean up after itself.

    It clears the whole combo directory rather than a list of known outputs, because enumerating
    them kept falling behind what the subprocesses actually write. Beyond the four files named in the
    module docstring, one run also leaves ``CONFIG/<dev>.json``, a ``-opstats.json`` per simulated
    batch size, ``SUMMARY/study-summary.{json,csv}``, and -- for the ViT cells, which sweep
    ``--batchsize 1 <bs> <bs>`` -- a ``-b1-`` CSV that is never consumed. The directory is wholly
    owned by this tool for a given run tag, so emptying it is both simpler and complete, and it stays
    correct if polaris grows another output.

    ``keep_opstats`` is set when ``--skip-polaris`` is reusing an existing CSV -- that one file is
    this run's input rather than a leftover, so it is the single exception.

    The run tag's top-level ``summary.md`` is deliberately not this function's business: it is one
    file for the whole matrix, so the caller clears it once before the per-combo loop.
    """
    combo_dir = REPO_ROOT / '__output' / run_tag / combo.label
    if not combo_dir.is_dir():
        return
    keep = opstats_path_for(run_tag, combo) if keep_opstats else None
    protected = {keep, *keep.parents} if keep is not None else set()
    # Deepest first, so a directory is only removed once its contents are gone.
    for path in sorted(combo_dir.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path in protected:
            continue
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            with contextlib.suppress(OSError):   # non-empty because it holds the reused CSV
                path.rmdir()


def resolve_refrun(combo: Combo) -> Path:
    """Return the local path to this combo's HW-reference CSV, fetching it from LFC on first use.

    ``resolve_lfc_path`` returns a path relative to the working directory and validates the cache
    location against a relative ``__ext``, so pin the CWD to the repo root — this script is
    runnable from anywhere.
    """
    with contextlib.chdir(REPO_ROOT):
        return REPO_ROOT / resolve_lfc_path(f'lfc://{REFRUN_LFC_ROOT}/{combo.refrun_rel}')


def compare_env() -> dict[str, str]:
    """Environment for the compare_layers subprocess.

    compare_layers validates every input path against ``POLARIS_BASE_DIR``, falling back to
    ``HOME``, and refuses anything outside it. Both of our inputs live under the repo (``__output/``
    for opstats, ``__ext/`` for the LFC-cached reference), so point the base at the repo: a checkout
    outside the user's home directory would otherwise be rejected for every combo.
    """
    return {**os.environ, 'POLARIS_BASE_DIR': str(REPO_ROOT)}


def run_compare(combo: Combo, run_tag: str, opstats: Path, refrun: Path) -> tuple[Path, Path]:
    """Run compare_layers.py. Returns (xlsx_path, compare_log_path).

    The caller clears any stale workbook before the combo starts -- see ``clear_stale_outputs``.
    """
    xlsx_path = xlsx_path_for(run_tag, combo)
    log_path = log_path_for(run_tag, combo, 'compare.log')
    cmd = [
        sys.executable, 'tools/profiling/compare_layers.py',
        '--perf', '--by-lut-key',
        '--xlsx', str(xlsx_path),
        str(opstats),
        str(refrun),
    ]
    with log_path.open('w') as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, check=True,
                       env=compare_env())
    return xlsx_path, log_path


# The LUT-hits line and a numeric gap are both optional: compare_layers.py only prints
# "<label> LUT hits" when at least one side has a hit (so a 0-hit baseline run omits it), and prints
# "Gap:  N/A" when the reference total is 0. Neither should cost us the row.
_SUMMARY_RE = re.compile(
    r'Network total:\s*\n'
    r'\s*Polaris:\s*([\d.]+)\s*ms\s*\n'
    r'\s*Profiler:\s*([\d.]+)\s*ms\s*\n'
    r'\s*Gap:\s*(N/A|[+-]?[\d.]+)%?[^\n]*\n'
    r'(?:\s*Polaris LUT hits:\s*(\d+)/(\d+))?'
)


def extract_summary(compare_log: Path) -> dict | None:
    """Pull the network totals out of a compare_layers log.

    ``--perf --by-lut-key`` prints *two* ``Network total`` blocks: the LUT-key rollup first, then the
    full performance summary. Take the **last** one. The rollup aggregates only layers that have a
    LUT key -- ``_aggregate_by_lut_key`` skips a layer whose ``lut_key``/``lut_key_resolved`` is
    None -- so on a workload with any keyless op its total and hit denominator are short of the real
    network. The performance summary covers every layer.
    """
    text = compare_log.read_text()
    matches = list(_SUMMARY_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    gap, hits, total = m.group(3), m.group(4), m.group(5)
    return {
        'polaris_ms': float(m.group(1)),
        'profiler_ms': float(m.group(2)),
        'gap_pct': None if gap == 'N/A' else float(gap),
        'lut_hits': None if hits is None else int(hits),
        'lut_total': None if total is None else int(total),
    }


def validate_combos(combos: list[Combo]) -> tuple[list[tuple[Combo, Path]], list[tuple[str, str]]]:
    """Pair each combo with its resolved refrun CSV, splitting off the ones that can't run.

    A combo is dropped when its arch config is absent from this branch (letting combos for
    workloads that live on other branches ship here dormant and auto-activate once those land on
    main) or when its refrun CSV is neither cached nor fetchable from LFC. Resolving up front means
    a missing reference is reported before a multi-minute simulation rather than after it. Drops are
    reported and make the run exit nonzero, so a partial matrix is never mistaken for a clean one.
    """
    ok: list[tuple[Combo, Path]] = []
    skipped: list[tuple[str, str]] = []
    for c in combos:
        if not (REPO_ROOT / c.archfile).exists():
            reason = f'arch config not on this branch: {c.archfile}'
        else:
            try:
                ok.append((c, resolve_refrun(c)))
                continue
            except (RuntimeError, ValueError, OSError) as e:
                reason = f'refrun {c.refrun_rel} unavailable (local cache and LFC): {e}'
        print(f'WARN: skipping {c.label!r} — {reason}', file=sys.stderr)
        skipped.append((c.label, reason))
    return ok, skipped


def emit_report(run_tag: str, summaries: list[dict], problems: list[tuple[str, str]],
                n_requested: int) -> None:
    """Print the aggregated table and write the same content to ``summary.md``.

    Called on every path, including the ones where nothing ran: when the whole matrix fails, the
    problem list *is* the report, and a caller looking for ``summary.md`` afterwards should find the
    reasons rather than an absent file.
    """
    labels = [f'{s["label"]} (bs={s["bs"]})' for s in summaries]
    col_label = max([len('Combo')] + [len(x) for x in labels])
    header = (f'| {"Combo":<{col_label}} | {"Polaris (ms)":>13} | {"Profiler (ms)":>14} '
              f'| {"Gap":>8} | {"LUT hits":>11} |')
    sep = f'|{"-"*(col_label+2)}|{"-"*15}|{"-"*16}|{"-"*10}|{"-"*13}|'
    rows = []
    for s, label in zip(summaries, labels):
        gap = 'N/A' if s['gap_pct'] is None else f'{s["gap_pct"]:+.2f}%'
        hits = 'n/a' if s['lut_hits'] is None else f'{s["lut_hits"]}/{s["lut_total"]}'
        rows.append(f'| {label:<{col_label}} | {s["polaris_ms"]:>13.4f} '
                    f'| {s["profiler_ms"]:>14.4f} | {gap:>8} | {hits:>11} |')
    problem_lines = [f'- `{label}`: {reason}' for label, reason in problems]
    no_rows_note = 'No combo produced a summary row.'

    print()
    print('=== Summary ===')
    if rows:
        print(header)
        print(sep)
        for r in rows:
            print(r)
    else:
        print(no_rows_note)
    if problem_lines:
        print()
        print(f'=== Not reported ({len(problems)} of {n_requested} requested) ===')
        for line in problem_lines:
            print(line)

    md_lines = [
        f'# Projection + compare summary — run_tag `{run_tag}`',
        '',
        f'Generated: {datetime.now().isoformat(timespec="seconds")}',
        f'Combos reported: {len(summaries)} of {n_requested} requested ({len(COMBOS)} known)',
        '',
    ]
    md_lines += [header, sep, *rows, ''] if rows else [f'_{no_rows_note}_', '']
    if problem_lines:
        md_lines += ['## Not reported', '', *problem_lines, '']
    summary_path = summary_path_for(run_tag)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text('\n'.join(md_lines))
    print()
    print(f'Summary saved: {summary_path.relative_to(REPO_ROOT)}')


# fullmatch, not match: with `$` a trailing newline is still a match, so `match()` would accept
# "run1\n" despite it being outside the documented character set. The first character also may not be
# `-`: the tag is interpolated into polaris's `--study <value>`, and argparse in the child reads a
# leading-hyphen value as an option, so every subprocess would exit before running.
_SAFE_RUN_TAG = re.compile(r'[A-Za-z0-9._][A-Za-z0-9._-]*')


def run_tag_arg(value: str) -> str:
    """Accept only a single relative directory name.

    The tag is both a path component under ``__output/`` and polaris's ``--study`` path, so a value
    containing a separator or ``..`` would write outside the output tree, and an absolute one would
    make ``Path.__truediv__`` discard the ``__output`` prefix entirely and then trip the
    ``relative_to(REPO_ROOT)`` calls used for display.
    """
    if value in {'.', '..'} or not _SAFE_RUN_TAG.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f'must be a single relative directory name matching [A-Za-z0-9._][A-Za-z0-9._-]*, '
            f'got {value!r}')
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-tag', type=run_tag_arg,
                        help='Output subdirectory tag, e.g. "session_2026-05-25". A single relative '
                             'directory name. Required unless --list is used.')
    parser.add_argument('--skip-polaris', action='store_true',
                        help='Reuse an existing opstats CSV instead of simulating. Combos with no '
                             'reusable CSV are still simulated, with a note on stderr, so an '
                             'interrupted matrix can be topped up in one re-run.')
    parser.add_argument('--only', metavar='COMBO_LABEL', action='append',
                        help='Run only the named combo(s) (default: all of them). Repeatable.')
    parser.add_argument('--list', action='store_true', help='List available combo labels and exit')
    args = parser.parse_args()

    if args.list:
        width = max(len(c.label) for c in COMBOS)
        for c in COMBOS:
            print(f'{c.label:<{width}}  arch={c.filterarch:<5}  bs={c.bs_runtime}  '
                  f'wli={c.filterwli}')
        return 0

    if not args.run_tag:
        parser.error('--run-tag is required (unless using --list)')

    # Reject unknown labels rather than filtering them away: a silently-dropped typo would not be
    # counted in n_requested, so it could not turn the exit status nonzero and would leave a
    # partial run looking clean.
    known = {c.label for c in COMBOS}
    if args.only:
        unknown = sorted(set(args.only) - known)
        if unknown:
            parser.error(f'unknown --only label(s): {", ".join(unknown)}. '
                         f'Known labels: {", ".join(sorted(known))}')

    selected = COMBOS if not args.only else [c for c in COMBOS if c.label in args.only]
    n_requested = len(selected)

    # Clear stale outputs for every *selected* combo, before validation rather than inside the run
    # loop. A combo dropped by validate_combos (arch config absent, reference unfetchable) never
    # enters that loop, so clearing there left its previous workbook and CSV sitting next to a report
    # saying the combo was not reported. Deciding reuse per combo here keeps --skip-polaris's input.
    reuse = {c.label: args.skip_polaris and opstats_path_for(args.run_tag, c).exists()
             for c in selected}
    # The top-level report is cleared here too, not only the per-combo directories. It is written
    # last, so an interrupted or aborted run would otherwise leave the previous run's summary.md
    # sitting at the top of this run tag's output, where a caller reads it as the current result.
    summary_path_for(args.run_tag).unlink(missing_ok=True)
    problems: list[tuple[str, str]] = []
    cleared: list[Combo] = []
    for combo in selected:
        # A combo whose directory cannot be cleared is this combo's failure, not the matrix's: an
        # unwritable directory used to abort every remaining combo before emit_report() ran, which
        # both lost the aggregate and (with the unlink above) left no report at all.
        try:
            clear_stale_outputs(args.run_tag, combo, keep_opstats=reuse[combo.label])
        except OSError as e:
            reason = f'could not clear stale outputs: {e}'
            print(f'  FAILED: {combo.label}: {reason}', file=sys.stderr)
            problems.append((combo.label, reason))
            continue
        cleared.append(combo)

    runnable, validation_problems = validate_combos(cleared)
    problems += validation_problems
    if not runnable:
        # Not only "arch config or reference missing" any more: a combo can also drop out here
        # because its output directory could not be cleared, so point at the reasons themselves.
        print('No runnable combos — see "Not reported" below for the reason on each.',
              file=sys.stderr)
        emit_report(args.run_tag, [], problems, n_requested)
        return 1

    summaries: list[dict] = []
    for combo, refrun in runnable:
        print(f'=== {combo.label} ===', flush=True)
        print(f'  refrun:  {refrun.relative_to(REPO_ROOT)}')
        # A failure here must not abort the rest of the matrix: this tool exists to aggregate all
        # combos, so a broken one is recorded and the remaining combos still run.
        try:
            opstats = opstats_path_for(args.run_tag, combo)
            reusing = reuse[combo.label]
            if reusing:
                print(f'  reusing opstats: {opstats.relative_to(REPO_ROOT)}')
            else:
                if args.skip_polaris:
                    # Falling through deliberately, so an interrupted matrix can be topped up with
                    # one --skip-polaris re-run. Say so: the flag implies no simulation, and a
                    # silent multi-minute polaris run here would be a surprise.
                    print('  NOTE: --skip-polaris given but no opstats to reuse; simulating anyway',
                          file=sys.stderr)
                print(f'  running polaris (study={args.run_tag}/{combo.label})...', flush=True)
                opstats = run_polaris(combo, args.run_tag)
                print(f'  opstats: {opstats.relative_to(REPO_ROOT)}')
            print('  running compare_layers...', flush=True)
            xlsx, compare_log = run_compare(combo, args.run_tag, opstats, refrun)
            print(f'  xlsx:    {xlsx.relative_to(REPO_ROOT)}')
            # Reading the summary belongs inside this boundary: it opens, decodes and converts, so
            # a bad log has to cost this combo its row and nothing more.
            s = extract_summary(compare_log)
        except subprocess.CalledProcessError as e:
            reason = f'{Path(e.cmd[1]).name} exited {e.returncode} — see __output/{args.run_tag}/{combo.label}/'
            print(f'  FAILED: {reason}', file=sys.stderr)
            problems.append((combo.label, reason))
            continue
        except OSError as e:
            print(f'  FAILED: {e}', file=sys.stderr)
            problems.append((combo.label, str(e)))
            continue
        except ValueError as e:
            reason = f'unreadable numbers in the compare summary: {e}'
            print(f'  FAILED: {reason}', file=sys.stderr)
            problems.append((combo.label, reason))
            continue
        if s is None:
            reason = f'could not parse summary from {compare_log.relative_to(REPO_ROOT)}'
            print(f'  WARN: {reason}', file=sys.stderr)
            problems.append((combo.label, reason))
            continue
        s['label'] = combo.label
        s['bs'] = combo.bs_runtime
        summaries.append(s)

    emit_report(args.run_tag, summaries, problems, n_requested)
    if not summaries:
        print('No combo produced a parseable summary.', file=sys.stderr)
        return 1
    # Exit nonzero if any requested combo is missing from the table, so a partial matrix can't be
    # mistaken for a clean one by a caller that only checks the status.
    if problems:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
