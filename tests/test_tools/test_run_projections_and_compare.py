#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools.profiling.run_projections_and_compare.

The summary parser reads compare_layers.py's ``Network total`` block, so these tests pin the shape
of that contract: field ordering (the labels are printed Polaris-then-Profiler), and the two cases
where compare_layers legitimately omits information — no LUT-hits line when neither side has a hit,
and ``Gap: N/A`` when the reference total is zero.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

import pytest
import tools.profiling.run_projections_and_compare as rpc
from tools.profiling.run_projections_and_compare import (
    COMBOS,
    REPO_ROOT,
    compare_env,
    emit_report,
    extract_summary,
    run_tag_arg,
)

# Exactly as compare_layers.py --perf --by-lut-key prints it (two-space indents, label order,
# the "(w.r.t. Polaris)" suffix on the gap line).
NORMAL_LOG = """
  LUT key comparison: Polaris vs Profiler

  Network total:
    Polaris:  3.2675 ms
    Profiler:  3.2911 ms
    Gap:  +0.72% (w.r.t. Polaris)
    Polaris LUT hits: 102/102
    Profiler LUT hits: 102/102

  op                 #1
"""

# has_lut is false in compare_layers when neither side has a hit, so both LUT lines are absent.
ZERO_HIT_LOG = """
  Network total:
    Polaris:  3.2675 ms
    Profiler:  3.2911 ms
    Gap:  -9.05% (w.r.t. Polaris)

  op                 #1
"""

# _pct_gap returns "N/A" when the reference total is 0.0.
NA_GAP_LOG = """
  Network total:
    Polaris:  0.0000 ms
    Profiler:  3.2911 ms
    Gap:  N/A (w.r.t. Polaris)
"""


def _write(tmp_path, text):
    p = tmp_path / 'compare.log'
    p.write_text(text)
    return p


# ── extract_summary ────────────────────────────────────────────────────


@pytest.mark.unit
def test_extract_summary_normal(tmp_path):
    """A complete block yields every field, with the gap signed."""
    s = extract_summary(_write(tmp_path, NORMAL_LOG))
    assert s == {
        'polaris_ms': 3.2675,
        'profiler_ms': 3.2911,
        'gap_pct': 0.72,
        'lut_hits': 102,
        'lut_total': 102,
    }


@pytest.mark.unit
def test_extract_summary_negative_gap(tmp_path):
    """A negative gap keeps its sign rather than being dropped by the pattern."""
    s = extract_summary(_write(tmp_path, ZERO_HIT_LOG))
    assert s is not None
    assert s['gap_pct'] == -9.05


@pytest.mark.unit
def test_extract_summary_zero_hits_keeps_the_row(tmp_path):
    """No LUT-hits line must not cost the row: the timing fields still parse.

    A 0-hit run is the normal starting point of LUT-hit work, so dropping it would hide the very
    baseline being measured.
    """
    s = extract_summary(_write(tmp_path, ZERO_HIT_LOG))
    assert s is not None
    assert s['polaris_ms'] == 3.2675
    assert s['profiler_ms'] == 3.2911
    assert s['lut_hits'] is None
    assert s['lut_total'] is None


@pytest.mark.unit
def test_extract_summary_na_gap_keeps_the_row(tmp_path):
    """``Gap: N/A`` parses to a None gap rather than failing the whole match."""
    s = extract_summary(_write(tmp_path, NA_GAP_LOG))
    assert s is not None
    assert s['gap_pct'] is None
    assert s['profiler_ms'] == 3.2911


@pytest.mark.unit
def test_extract_summary_returns_none_when_absent(tmp_path):
    """A log with no Network total block is reported as unparseable, not as zeroes."""
    assert extract_summary(_write(tmp_path, 'nothing useful here\n')) is None


@pytest.mark.unit
def test_extract_summary_requires_polaris_before_profiler(tmp_path):
    """Reversed labels must not silently swap the two timings.

    compare_layers prints file1 first, and the tool always passes the polaris CSV as file1; an
    inverted block means an assumption broke, so failing to parse is the honest outcome.
    """
    swapped = NORMAL_LOG.replace('Polaris:  3.2675', 'XXX').replace(
        'Profiler:  3.2911 ms', 'Polaris:  3.2911 ms').replace('XXX', 'Profiler:  3.2675')
    assert extract_summary(_write(tmp_path, swapped)) is None


# --perf --by-lut-key prints the LUT-key rollup first, then the full performance summary. The two
# totals differ here on purpose: the rollup skips layers with no LUT key, so only the second block
# describes the whole network. Numbers are deliberately distinct so a first-match parser fails.
TWO_BLOCK_LOG = """
==================================================
  LUT key comparison: Polaris vs Profiler
==================================================

  Network total:
    Polaris:  2.0000 ms
    Profiler:  2.1000 ms
    Gap:  +5.00% (w.r.t. Polaris)
    Polaris LUT hits: 90/90
    Profiler LUT hits: 0/90

  op                 #1

==================================================
  Performance comparison
==================================================

  Network total:
    Polaris:  3.2675 ms
    Profiler:  3.2911 ms
    Gap:  +0.72% (w.r.t. Polaris)
    Polaris LUT hits: 102/102

  Layer Type   #1
"""


@pytest.mark.unit
def test_extract_summary_takes_the_last_network_total_block(tmp_path):
    """The full performance summary, not the LUT-key rollup, is the network total.

    ``_aggregate_by_lut_key`` drops any layer whose ``lut_key``/``lut_key_resolved`` is None, so on a
    workload with a keyless op the rollup understates both the timing and the hit denominator. That
    is exactly the early-bring-up case this tool is used for, so reading the first block would report
    a quietly wrong number rather than failing.
    """
    s = extract_summary(_write(tmp_path, TWO_BLOCK_LOG))
    assert s is not None
    assert s['polaris_ms'] == 3.2675
    assert s['profiler_ms'] == 3.2911
    assert s['gap_pct'] == 0.72
    assert s['lut_hits'] == 102
    assert s['lut_total'] == 102


# ── compare_env ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_compare_env_pins_base_dir_to_the_repo():
    """compare_layers rejects inputs outside POLARIS_BASE_DIR (default HOME).

    Without this override a checkout outside the user's home directory fails every combo with
    "Access denied", since the wrapper hands compare_layers absolute repo-contained paths.
    """
    env = compare_env()
    assert env['POLARIS_BASE_DIR'] == str(REPO_ROOT)


@pytest.mark.unit
def test_compare_env_preserves_the_inherited_environment(monkeypatch):
    """The LFC downloader and conda both rely on inherited vars, so this must not be a bare dict."""
    monkeypatch.setenv('A_CANARY_VAR', 'kept')
    env = compare_env()
    assert env.get('A_CANARY_VAR') == 'kept'
    assert len(env) > 1


# ── run_tag_arg ────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize('tag', ['run1', 'session_2026-05-25', 'pr454.final', 'A-b_1.2'])
def test_run_tag_arg_accepts_plain_names(tag):
    assert run_tag_arg(tag) == tag


@pytest.mark.unit
@pytest.mark.parametrize('tag', ['-trial', '-', '--x', '-1'])
def test_run_tag_arg_rejects_a_leading_hyphen(tag):
    """A leading hyphen would break every polaris subprocess.

    The tag is interpolated into `--study <value>`, and argparse in the child reads a value starting
    with `-` as an option: verified against the real CLI, which fails with
    "argument --study/-s: expected one argument".
    """
    with pytest.raises(argparse.ArgumentTypeError):
        run_tag_arg(tag)


@pytest.mark.unit
@pytest.mark.parametrize('tag', ['a-b', 'run-1', 'x_y.z-2'])
def test_run_tag_arg_still_allows_interior_hyphens(tag):
    """Only the leading position is a problem — dated tags like `session-260817` must keep working."""
    assert run_tag_arg(tag) == tag


@pytest.mark.unit
@pytest.mark.parametrize('tag', ['run1\n', 'run1\n\n', '\nrun1', 'run1\t'])
def test_run_tag_arg_rejects_trailing_whitespace(tag):
    """`$` matches before a final newline, so `match()` accepted "run1\\n".

    The validator uses `fullmatch` for exactly this reason: a value outside the documented character
    set must not slip through just because the offending character is last.
    """
    with pytest.raises(argparse.ArgumentTypeError):
        run_tag_arg(tag)


@pytest.mark.unit
@pytest.mark.parametrize('tag', ['..', '.', '', 'a/b', '../escaped', '/tmp/absolute', 'a b', 'x;y'])
def test_run_tag_arg_rejects_paths_and_traversal(tag):
    """The tag is a path component under __output/ and polaris's --study value.

    A separator or ``..`` would write outside the output tree, and an absolute value would make
    Path joining discard the __output prefix entirely.
    """
    with pytest.raises(argparse.ArgumentTypeError):
        run_tag_arg(tag)


# ── emit_report ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_emit_report_writes_summary_when_nothing_ran(tmp_path, monkeypatch, capsys):
    """An all-failed matrix still leaves a summary.md holding the reasons.

    This is the case where the report matters most, so it must not be skipped by an early return.
    """
    monkeypatch.setattr('tools.profiling.run_projections_and_compare.REPO_ROOT', tmp_path)
    emit_report('tag1', [], [('vgg_wh_n150', 'refrun unavailable')], 1)

    written = (tmp_path / '__output' / 'tag1' / 'summary.md').read_text()
    assert 'No combo produced a summary row.' in written
    assert '## Not reported' in written
    assert 'vgg_wh_n150' in written and 'refrun unavailable' in written
    assert 'Combos reported: 0 of 1 requested' in written

    out = capsys.readouterr().out
    assert 'Not reported (1 of 1 requested)' in out


@pytest.mark.unit
def test_emit_report_lists_both_rows_and_problems(tmp_path, monkeypatch):
    """A partial matrix reports the rows it got *and* what is missing."""
    monkeypatch.setattr('tools.profiling.run_projections_and_compare.REPO_ROOT', tmp_path)
    summaries = [{'label': 'vgg_wh_n150', 'bs': 1, 'polaris_ms': 3.2675,
                  'profiler_ms': 3.2911, 'gap_pct': 0.72, 'lut_hits': 102, 'lut_total': 102}]
    emit_report('tag2', summaries, [('vit_wh_n150', 'polaris.py exited 1')], 2)

    written = (tmp_path / '__output' / 'tag2' / 'summary.md').read_text()
    assert '| vgg_wh_n150 (bs=1) |' in written
    assert '+0.72%' in written and '102/102' in written
    assert 'vit_wh_n150' in written
    assert 'Combos reported: 1 of 2 requested' in written


@pytest.mark.unit
def test_emit_report_renders_missing_gap_and_hits(tmp_path, monkeypatch):
    """None gap / None hits render as placeholders rather than raising on format."""
    monkeypatch.setattr('tools.profiling.run_projections_and_compare.REPO_ROOT', tmp_path)
    summaries = [{'label': 'x', 'bs': 1, 'polaris_ms': 1.0, 'profiler_ms': 2.0,
                  'gap_pct': None, 'lut_hits': None, 'lut_total': None}]
    emit_report('tag3', summaries, [], 1)

    written = (tmp_path / '__output' / 'tag3' / 'summary.md').read_text()
    assert 'N/A' in written
    assert 'n/a' in written


# ── main(): the keep-going contract ────────────────────────────────────
#
# These drive main() end to end with the two subprocesses faked, because the properties they pin are
# the ones the tool exists for and they live in main()'s loop, not in a helper: one bad combo must
# not stop the matrix, the aggregate report must still be written, and a partial matrix must not
# report success. Everything below the fake is real code -- run_polaris/run_compare do their own file
# handling, and the summary is parsed out of a log the fake actually wrote.


def _stage_refrun(tmp_path, monkeypatch):
    """Fake resolve_refrun so it behaves like the real one: it returns a CSV that is *there*.

    `resolve_lfc_path` downloads the reference when the cache misses, so in production the path it
    returns always exists as a regular file -- which is what `validate_combos` now requires, since
    `compare_layers` refuses anything else. A fixture handing back a path that was never created
    was modelling a state production cannot reach.
    """
    def fake(combo):
        path = tmp_path / '__ext' / combo.refrun_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text('archname\n')
        return path
    monkeypatch.setattr(rpc, 'resolve_refrun', fake)


def _stage_main(tmp_path, monkeypatch, failing_labels):
    """Point the module at tmp_path and fake both subprocesses. Returns the invocation log."""
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    # validate_combos needs the arch config present, and must not touch LFC.
    for c in COMBOS:
        arch = tmp_path / c.archfile
        arch.parent.mkdir(parents=True, exist_ok=True)
        arch.write_text('# stub\n')
    _stage_refrun(tmp_path, monkeypatch)

    invoked = []

    def fake_run(cmd, **kwargs):
        script = Path(cmd[1]).name
        study = cmd[cmd.index('--study') + 1] if '--study' in cmd else ''
        invoked.append((script, study))
        if script == 'polaris.py':
            label = study.split('/')[-1]
            if label in failing_labels:
                raise subprocess.CalledProcessError(1, cmd)
            combo = next(c for c in COMBOS if c.label == label)
            ops = rpc.opstats_path_for(study.split('/')[0], combo)
            ops.parent.mkdir(parents=True, exist_ok=True)
            ops.write_text('archname\n')
        else:
            # compare_layers writes to the log handle it was given; do the same.
            kwargs['stdout'].write(NORMAL_LOG)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rpc.subprocess, 'run', fake_run)
    return invoked


@pytest.mark.unit
def test_main_keeps_going_after_a_failed_combo(tmp_path, monkeypatch):
    """A failing combo is recorded, later combos still run, and the exit status is nonzero."""
    invoked = _stage_main(tmp_path, monkeypatch, failing_labels={'vit_wh_n150'})
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vit_wh_n150',
                                     '--only', 'vgg_wh_n150'])
    rc = rpc.main()

    polaris_runs = [s for s, _ in invoked if s == 'polaris.py']
    assert len(polaris_runs) == 2, 'the matrix stopped at the failing combo'
    assert rc == 1, 'a partial matrix must not report success'

    written = (tmp_path / '__output' / 'tag' / 'summary.md').read_text()
    assert '| vgg_wh_n150 (bs=1) |' in written, 'the combo that succeeded is missing its row'
    assert 'Combos reported: 1 of 2 requested' in written
    assert '## Not reported' in written and 'vit_wh_n150' in written
    assert 'polaris.py exited 1' in written


@pytest.mark.unit
def test_main_returns_zero_only_when_the_matrix_is_complete(tmp_path, monkeypatch):
    """With every requested combo reported, the run is clean and no problems are listed."""
    _stage_main(tmp_path, monkeypatch, failing_labels=set())
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150'])
    assert rpc.main() == 0

    written = (tmp_path / '__output' / 'tag' / 'summary.md').read_text()
    assert 'Combos reported: 1 of 1 requested' in written
    assert '## Not reported' not in written


@pytest.mark.unit
def test_main_writes_the_report_when_every_combo_fails(tmp_path, monkeypatch):
    """The all-failed matrix is the case where the reasons are the entire report."""
    _stage_main(tmp_path, monkeypatch, failing_labels={'vgg_wh_n150', 'vgg_bh_p100a'})
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150',
                                     '--only', 'vgg_bh_p100a'])
    assert rpc.main() == 1

    written = (tmp_path / '__output' / 'tag' / 'summary.md').read_text()
    assert 'No combo produced a summary row.' in written
    assert written.count('polaris.py exited 1') == 2


@pytest.mark.unit
def test_main_rejects_a_stale_opstats_csv_under_a_reused_run_tag(tmp_path, monkeypatch):
    """A leftover CSV must never be compared as though it described the current run.

    An existence check cannot by itself tell this run's output from a file left by an earlier
    invocation under the same tag, so the path is cleared first and the check then means what it
    says. Otherwise a plausible-looking row describing stale data survives, which is worse than a
    reported failure.

    This originally also compensated for polaris exiting 0 with "completed with 0 experiments" when
    the filters selected nothing. It now rejects that itself (issue #517), so only the staleness
    concern is load-bearing here -- but a run can still produce no CSV for other reasons, so the
    clear-then-check order still matters.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    for c in COMBOS:
        arch = tmp_path / c.archfile
        arch.parent.mkdir(parents=True, exist_ok=True)
        arch.write_text('# stub\n')
    _stage_refrun(tmp_path, monkeypatch)

    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    stale = rpc.opstats_path_for('tag', combo)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('archname\nstale-row-from-a-previous-run\n')

    def fake_run(cmd, **kwargs):
        # polaris "succeeds" but produces no opstats -- an empty selection is rejected outright
        # since #517, so this stands in for the other ways a clean exit can leave no CSV.
        if Path(cmd[1]).name == 'compare_layers.py':
            kwargs['stdout'].write(NORMAL_LOG)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rpc.subprocess, 'run', fake_run)
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150'])

    assert rpc.main() == 1, 'a run that produced nothing must not report success'
    written = (tmp_path / '__output' / 'tag' / 'summary.md').read_text()
    assert 'No combo produced a summary row.' in written
    assert 'opstats CSV not found' in written
    assert '+0.72%' not in written, 'stale data was compared and reported as a result'


@pytest.mark.unit
def test_main_clears_a_stale_workbook_when_compare_fails(tmp_path, monkeypatch):
    """A failed compare must not leave an earlier run's workbook in the output directory.

    The summary correctly omits the combo, so no wrong number is reported — but the workbook is an
    advertised output, and a leftover one can be opened as though it described this run. It is the
    only advertised output that can survive a failure: both logs are truncated by open('w') and
    opstats is unlinked before polaris runs.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    for c in COMBOS:
        arch = tmp_path / c.archfile
        arch.parent.mkdir(parents=True, exist_ok=True)
        arch.write_text('# stub\n')
    _stage_refrun(tmp_path, monkeypatch)

    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    stale_xlsx = rpc.xlsx_path_for('tag', combo)
    stale_xlsx.parent.mkdir(parents=True, exist_ok=True)
    stale_xlsx.write_bytes(b'stale workbook from a previous run')

    def fake_run(cmd, **kwargs):
        script = Path(cmd[1]).name
        if script == 'polaris.py':
            ops = rpc.opstats_path_for('tag', combo)
            ops.parent.mkdir(parents=True, exist_ok=True)
            ops.write_text('archname\n')
            return subprocess.CompletedProcess(cmd, 0)
        raise subprocess.CalledProcessError(1, cmd)   # compare fails before writing the workbook

    monkeypatch.setattr(rpc.subprocess, 'run', fake_run)
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150'])

    assert rpc.main() == 1
    assert not stale_xlsx.exists(), 'a previous run\'s workbook survived a failed compare'
    written = (tmp_path / '__output' / 'tag' / 'summary.md').read_text()
    assert 'compare_layers.py exited 1' in written


@pytest.mark.unit
def test_main_clears_a_stale_workbook_when_polaris_fails(tmp_path, monkeypatch):
    """The stale-output guard must cover a failure at *any* step, not just at compare.

    Clearing the workbook inside run_compare only covered the compare-fails path: if polaris fails
    first, run_compare is never reached, and an earlier run's workbook survives — the same defect
    the cleanup exists to prevent, arrived at from a different direction.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    for c in COMBOS:
        arch = tmp_path / c.archfile
        arch.parent.mkdir(parents=True, exist_ok=True)
        arch.write_text('# stub\n')
    _stage_refrun(tmp_path, monkeypatch)

    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    stale_xlsx = rpc.xlsx_path_for('tag', combo)
    stale_xlsx.parent.mkdir(parents=True, exist_ok=True)
    stale_xlsx.write_bytes(b'stale workbook from a previous run')

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)   # polaris fails; compare is never reached

    monkeypatch.setattr(rpc.subprocess, 'run', fake_run)
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150'])

    assert rpc.main() == 1
    assert not stale_xlsx.exists(), "a previous run's workbook survived a failed polaris run"
    written = (tmp_path / '__output' / 'tag' / 'summary.md').read_text()
    assert 'polaris.py exited 1' in written


@pytest.mark.unit
def test_main_keeps_the_reused_opstats_when_skip_polaris_is_used(tmp_path, monkeypatch):
    """Clearing stale outputs must not delete the CSV that --skip-polaris is reusing.

    That file is this run's input, not a leftover. The CSV is staged with its completion marker
    because reuse now turns on the marker rather than on the CSV's existence -- a CSV on its own
    is what an interrupted run leaves behind.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    for c in COMBOS:
        arch = tmp_path / c.archfile
        arch.parent.mkdir(parents=True, exist_ok=True)
        arch.write_text('# stub\n')
    _stage_refrun(tmp_path, monkeypatch)

    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    ops = rpc.opstats_path_for('tag', combo)
    ops.parent.mkdir(parents=True, exist_ok=True)
    ops.write_text('archname\nreusable\n')
    rpc.completion_marker_for('tag', combo).write_text('2026-09-07T00:00:00\n')

    invoked = []

    def fake_run(cmd, **kwargs):
        invoked.append(Path(cmd[1]).name)
        if Path(cmd[1]).name == 'compare_layers.py':
            kwargs['stdout'].write(NORMAL_LOG)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rpc.subprocess, 'run', fake_run)
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150',
                                     '--skip-polaris'])

    assert rpc.main() == 0
    assert 'polaris.py' not in invoked, 'the reusable CSV was deleted, forcing a re-simulation'
    assert ops.read_text() == 'archname\nreusable\n'


@pytest.mark.unit
def test_main_clears_stale_outputs_for_a_combo_dropped_at_validation(tmp_path, monkeypatch):
    """A combo that never reaches the run loop still needs its old outputs cleared.

    validate_combos drops combos whose arch config or reference is unavailable, so clearing inside
    the loop missed them entirely: the summary said "not reported" while the previous run's workbook
    and CSV sat in the output directory looking current.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    # Arch config deliberately absent -> validate_combos drops this combo.
    _stage_refrun(tmp_path, monkeypatch)

    stale_xlsx = rpc.xlsx_path_for('tag', combo)
    stale_xlsx.parent.mkdir(parents=True, exist_ok=True)
    stale_xlsx.write_bytes(b'stale workbook')
    stale_ops = rpc.opstats_path_for('tag', combo)
    stale_ops.parent.mkdir(parents=True, exist_ok=True)
    stale_ops.write_text('archname\nstale\n')

    def fake_run(cmd, **kwargs):
        raise AssertionError('no subprocess should run for a combo dropped at validation')

    monkeypatch.setattr(rpc.subprocess, 'run', fake_run)
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150'])

    stale_logs = [rpc.log_path_for('tag', combo, n) for n in ('polaris.log', 'compare.log')]
    for lg in stale_logs:
        lg.write_text('log text from a previous run\n')

    assert rpc.main() == 1
    assert not stale_xlsx.exists(), "a dropped combo's workbook survived"
    assert not stale_ops.exists(), "a dropped combo's opstats CSV survived"
    for lg in stale_logs:
        # open('w') truncates these, but only for a combo that reaches a subprocess. A combo dropped
        # at validation opens neither, so they have to be cleared explicitly.
        assert not lg.exists(), f"a dropped combo's {lg.name} survived"
    written = (tmp_path / '__output' / 'tag' / 'summary.md').read_text()
    assert 'arch config not on this branch' in written


@pytest.mark.unit
def test_clear_stale_outputs_empties_the_whole_combo_directory(tmp_path, monkeypatch):
    """Everything a previous run left under the combo directory goes, not a known-outputs list.

    One run writes more than the four files the module docstring advertises: `CONFIG/<dev>.json`, a
    `-opstats.json` per simulated batch size, and `SUMMARY/study-summary.{json,csv}`. Enumerating
    them kept falling behind what the subprocesses write, so the contract is the directory.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    combo = next(c for c in COMBOS if c.label == 'vit_wh_n150')
    base = tmp_path / '__output' / 'tag' / combo.label
    planted = [
        base / 'polaris.log', base / 'compare.log',
        base / f'{combo.label}_comparison.xlsx',
        base / 'CONFIG' / 'n150.json',
        base / 'STATS' / 'n150-TTNN-x-y-b1-opstats.csv',
        base / 'STATS' / 'n150-TTNN-x-y-b1-opstats.json',
        base / 'STATS' / 'n150-TTNN-x-y-b8-opstats.json',
        base / 'SUMMARY' / 'study-summary.json',
        base / 'SUMMARY' / 'study-summary.csv',
    ]
    for p in planted:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('stale\n')

    rpc.clear_stale_outputs('tag', combo, keep_opstats=False)
    survivors = [p for p in planted if p.exists()]
    assert survivors == [], f'stale files survived: {[p.name for p in survivors]}'


@pytest.mark.unit
def test_clear_stale_outputs_is_a_noop_when_the_directory_is_absent(tmp_path, monkeypatch):
    """A first run under a fresh tag has nothing to clear and must not raise."""
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    rpc.clear_stale_outputs('brand_new_tag', combo, keep_opstats=False)
    rpc.clear_stale_outputs('brand_new_tag', combo, keep_opstats=True)


@pytest.mark.unit
def test_clear_stale_outputs_removes_every_opstats_csv_not_just_the_consumed_one(tmp_path, monkeypatch):
    """A combo can leave more than one CSV behind, and all of them are stale.

    The ViT cells run `--batchsize 1 <bs> <bs>`, so polaris writes `-b1-` as well as `-b<bs>-` and
    only the latter is consumed. Naming the consumed path left the other sitting in the output
    directory. Clearing the whole directory also covers names this tool never constructs, e.g.
    polaris's `f<freq>` segment under `--frequency`.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    combo = next(c for c in COMBOS if c.label == 'vit_wh_n150')
    consumed = rpc.opstats_path_for('tag', combo)
    consumed.parent.mkdir(parents=True, exist_ok=True)
    consumed.write_text('archname\n')
    extra = consumed.parent / consumed.name.replace(f'-b{combo.bs_runtime}-', '-b1-')
    extra.write_text('archname\n')
    freqvar = consumed.parent / consumed.name.replace('n150-', 'n150-f800-')
    freqvar.write_text('archname\n')
    assert extra != consumed

    rpc.clear_stale_outputs('tag', combo, keep_opstats=False)
    for p in (consumed, extra, freqvar):
        assert not p.exists(), f'{p.name} survived the clear'


@pytest.mark.unit
def test_clear_stale_outputs_keeps_only_the_reused_csv(tmp_path, monkeypatch):
    """Under --skip-polaris the consumed CSV is this run's input; its siblings are still stale."""
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    combo = next(c for c in COMBOS if c.label == 'vit_wh_n150')
    consumed = rpc.opstats_path_for('tag', combo)
    consumed.parent.mkdir(parents=True, exist_ok=True)
    consumed.write_text('archname\nreusable\n')
    extra = consumed.parent / consumed.name.replace(f'-b{combo.bs_runtime}-', '-b1-')
    extra.write_text('archname\nstale\n')

    rpc.clear_stale_outputs('tag', combo, keep_opstats=True)
    assert consumed.read_text() == 'archname\nreusable\n', 'the reused CSV was deleted'
    assert not extra.exists(), 'a stale sibling CSV survived'


@pytest.mark.unit
def test_main_rejects_an_unknown_only_label(tmp_path, monkeypatch):
    """A typo must not be silently dropped: it would not count as requested, so a partial run
    could exit 0."""
    _stage_main(tmp_path, monkeypatch, failing_labels=set())
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150',
                                     '--only', 'typo'])
    with pytest.raises(SystemExit) as e:
        rpc.main()
    assert e.value.code == 2


# ── a run that dies must not leave the previous run's report behind ────


@pytest.mark.unit
def test_main_removes_a_stale_top_level_summary_before_the_matrix_runs(tmp_path, monkeypatch):
    """A previous run's summary.md must not survive a run that dies before writing its own.

    The report is written last, so an interrupted or aborted run would otherwise leave the earlier
    summary sitting at the top of this run tag's output directory, where the next reader takes it
    for the current result. Clearing the per-combo directories alone does not cover it: it is one
    file for the whole matrix.
    """
    _stage_main(tmp_path, monkeypatch, failing_labels=set())
    stale = rpc.summary_path_for('tag')
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('# report from a previous run\n')

    def die(*args, **kwargs):
        raise RuntimeError('aborted mid-matrix')

    monkeypatch.setattr(rpc, 'run_compare', die)
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150'])

    with pytest.raises(RuntimeError):
        rpc.main()
    assert not stale.exists(), "the previous run's report survived a run that never wrote one"


@pytest.mark.unit
def test_main_records_a_failed_clear_as_that_combos_problem(tmp_path, monkeypatch):
    """An unclearable combo directory costs that combo only, and the report is still written.

    The clearing loop runs before validation and outside the per-combo boundary, so an OSError here
    used to abort every remaining combo before the report was written — losing the aggregate the
    tool exists to produce.
    """
    invoked = _stage_main(tmp_path, monkeypatch, failing_labels=set())
    real_clear = rpc.clear_stale_outputs

    def clear(run_tag, combo, *, keep_opstats):
        if combo.label == 'vit_wh_n150':
            raise PermissionError('read-only output directory')
        return real_clear(run_tag, combo, keep_opstats=keep_opstats)

    monkeypatch.setattr(rpc, 'clear_stale_outputs', clear)
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vit_wh_n150',
                                     '--only', 'vgg_wh_n150'])
    rc = rpc.main()

    assert rc == 1, 'a partial matrix must not report success'
    assert [st for s, st in invoked if s == 'polaris.py'] == ['tag/vgg_wh_n150'], \
        'the clearable combo did not run'
    written = rpc.summary_path_for('tag').read_text()
    assert 'Combos reported: 1 of 2 requested' in written
    assert 'could not clear stale outputs' in written and 'vit_wh_n150' in written


@pytest.mark.unit
def test_main_records_an_unreadable_summary_as_that_combos_problem(tmp_path, monkeypatch):
    """Summary extraction reads and converts, so its failure must cost one row, not the matrix.

    ``extract_summary`` returns None on a parse miss, but a numeric conversion can still raise, and
    it used to sit outside the per-combo exception boundary.
    """
    invoked = _stage_main(tmp_path, monkeypatch, failing_labels=set())
    real_extract = rpc.extract_summary

    def extract(log_path):
        if 'vit_wh_n150' in str(log_path):
            raise ValueError("could not convert string to float: '1.2.3'")
        return real_extract(log_path)

    monkeypatch.setattr(rpc, 'extract_summary', extract)
    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vit_wh_n150',
                                     '--only', 'vgg_wh_n150'])
    rc = rpc.main()

    assert rc == 1, 'a partial matrix must not report success'
    assert len([s for s, _ in invoked if s == 'polaris.py']) == 2, \
        'the matrix stopped at the unreadable combo'
    written = rpc.summary_path_for('tag').read_text()
    assert '| vgg_wh_n150 (bs=1) |' in written, 'the readable combo is missing its row'
    assert 'unreadable numbers in the compare summary' in written


# ── the tool must own the directory it deletes inside ──────────────────


@pytest.mark.unit
def test_main_refuses_a_run_tag_directory_that_is_a_symlink(tmp_path, monkeypatch):
    """The run tag directory is deleted inside, so a symlink there would redirect the deletions.

    `--run-tag` already cannot traverse, but validating the *value* says nothing about what is
    on disk at that path when the run starts.
    """
    _stage_main(tmp_path, monkeypatch, failing_labels=set())
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    bystander = elsewhere / 'summary.md'
    bystander.write_text('# not ours to delete\n')
    out = tmp_path / '__output'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'tag').symlink_to(elsewhere, target_is_directory=True)

    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150'])
    assert rpc.main() == 1, 'a symlinked run directory must not be run into'
    assert bystander.read_text() == '# not ours to delete\n', 'the symlink target was written through'


@pytest.mark.unit
def test_main_records_a_symlinked_combo_directory_as_that_combos_problem(tmp_path, monkeypatch):
    """`is_dir()` is true through a symlink and rglob follows it, so clearing must refuse.

    This is the per-combo half of the same hazard: the run directory can be genuine while one
    combo inside it is a symlink.
    """
    invoked = _stage_main(tmp_path, monkeypatch, failing_labels=set())
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    bystander = elsewhere / 'keep-me.txt'
    bystander.write_text('untouched\n')
    combo_dir = tmp_path / '__output' / 'tag' / 'vit_wh_n150'
    combo_dir.parent.mkdir(parents=True, exist_ok=True)
    combo_dir.symlink_to(elsewhere, target_is_directory=True)

    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vit_wh_n150',
                                     '--only', 'vgg_wh_n150'])
    assert rpc.main() == 1, 'a partial matrix must not report success'
    assert bystander.read_text() == 'untouched\n', 'cleared through the symlink'
    assert [st for s, st in invoked if s == 'polaris.py'] == ['tag/vgg_wh_n150'], \
        'the healthy combo did not run'
    written = rpc.summary_path_for('tag').read_text()
    assert 'vit_wh_n150' in written, 'the redirected combo is missing from "Not reported"'
    assert 'does not resolve' in written, 'the report does not say why the combo was dropped'


@pytest.mark.unit
def test_main_refuses_a_symlinked_output_root(tmp_path, monkeypatch):
    """A redirected `__output` is refused up front rather than failing every comparison later.

    Redirecting the output tree to another volume looks like a reasonable disk-quota measure, but
    `compare_layers.sanitize_file_path()` resolves its inputs and requires them under
    `POLARIS_BASE_DIR`, which `compare_env()` pins to the repo. An opstats CSV on another volume
    is therefore refused with "Access denied" and every combo fails. Refusing here costs the
    caller a message; accepting would cost a multi-minute run with nothing to show.
    """
    _stage_main(tmp_path, monkeypatch, failing_labels=set())
    scratch = tmp_path / 'scratch_volume'
    scratch.mkdir()
    (tmp_path / '__output').symlink_to(scratch, target_is_directory=True)

    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150'])
    assert rpc.main() == 1, 'a redirected output root must be refused'
    assert not (scratch / 'tag').exists(), 'the run started anyway and wrote to the target'


@pytest.mark.unit
@pytest.mark.parametrize('where', ['__output', '__output/tag'])
def test_main_refuses_a_non_directory_on_the_output_path(tmp_path, monkeypatch, where):
    """A regular file at either level resolves to the right place, so type needs its own test.

    Both are parametrized deliberately: fixing only the run tag left the root unchecked, and the
    root case is the sneakier one — `run_dir.resolve()` is still lexically correct and
    `run_dir.exists()` is false, so nothing looks wrong until the summary unlink raises
    `NotADirectoryError`, past any per-combo handling or `emit_report()`.
    """
    _stage_main(tmp_path, monkeypatch, failing_labels=set())
    victim = tmp_path / where
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text('stale junk, not a directory\n')

    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150'])
    assert rpc.main() == 1, f'a non-directory at {where} must be refused'


@pytest.mark.unit
def test_clear_stale_outputs_removes_a_nested_directory_symlink(tmp_path, monkeypatch):
    """A nested link must be cleared as a link, or the next run writes through it.

    Both tests in the clearing loop follow a symlink: for a link to a directory `is_file()` is
    false and `is_dir()` is true, and `rmdir()` on the link raises into the suppression — so the
    link survived, and polaris then wrote into the target instead of the combo directory.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    combo_dir = tmp_path / '__output' / 'tag' / combo.label
    combo_dir.mkdir(parents=True)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'precious.csv').write_text('external data\n')
    (combo_dir / 'STATS').symlink_to(outside, target_is_directory=True)

    rpc.clear_stale_outputs('tag', combo, keep_opstats=False)

    assert not (combo_dir / 'STATS').is_symlink(), 'the stale link survived clearing'
    assert (outside / 'precious.csv').read_text() == 'external data\n', 'cleared through the link'


@pytest.mark.unit
def test_clear_stale_outputs_removes_a_stale_fifo(tmp_path, monkeypatch):
    """Clearing must remove anything that is not a directory, not a list of chosen types.

    A FIFO is neither a symlink, a regular file, nor a directory, so a type-by-type clear spared
    it — and a stale FIFO at `polaris.log` or the opstats path makes the next `open('w')` block
    forever waiting for a reader. That is a hung matrix rather than a failed combo, which is worse
    than any wrong number. Verified separately that a non-blocking write-open on such a FIFO
    raises ENXIO, which is the condition a blocking open waits on.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    combo_dir = tmp_path / '__output' / 'tag' / combo.label
    combo_dir.mkdir(parents=True)
    fifo = combo_dir / 'polaris.log'
    os.mkfifo(fifo)

    rpc.clear_stale_outputs('tag', combo, keep_opstats=False)

    assert not fifo.exists(), 'a stale FIFO survived clearing and would block the next open()'


@pytest.mark.unit
def test_clear_stale_outputs_refuses_to_reuse_an_opstats_csv_through_a_symlink(tmp_path, monkeypatch):
    """`--skip-polaris` must not reuse a CSV reached through a link.

    The reused CSV and its parent directories are exempt from clearing, so a link on that chain
    would survive and be read — comparing a file from outside this combo as though this run had
    produced it.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    combo_dir = tmp_path / '__output' / 'tag' / combo.label
    combo_dir.mkdir(parents=True)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (combo_dir / 'STATS').symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match='refusing to reuse'):
        rpc.clear_stale_outputs('tag', combo, keep_opstats=True)


@pytest.mark.unit
def test_main_refuses_a_symlink_loop_on_the_output_path(tmp_path, monkeypatch):
    """The path checks answer questions; they do not stop the filesystem from refusing to answer.

    A loop is the case that separates the two: non-strict `resolve()` returns the lexical path and
    `exists()` is false, so both ownership and type checks pass — and the summary unlink then
    fails with `ELOOP`. Uncaught, that is a traceback where every other malformed output path
    produces a refusal.
    """
    _stage_main(tmp_path, monkeypatch, failing_labels=set())
    (tmp_path / '__output').symlink_to(tmp_path / '__loop')
    (tmp_path / '__loop').symlink_to(tmp_path / '__output')

    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vgg_wh_n150'])
    assert rpc.main() == 1, 'a symlink loop must produce a controlled refusal, not a traceback'


@pytest.mark.unit
def test_a_symlink_loop_at_one_combo_costs_only_that_combo(tmp_path, monkeypatch):
    """A loop inside the matrix must be recorded like any other per-combo failure.

    Documents the boundary that already handles it: non-strict `resolve()` does not raise on a
    loop, `is_dir()` is false so clearing returns early, and the first real write raises `ELOOP`
    — an `OSError`, so the per-combo handler catches it and the remaining combos still run.
    """
    invoked = _stage_main(tmp_path, monkeypatch, failing_labels=set())
    combo_dir = tmp_path / '__output' / 'tag' / 'vit_wh_n150'
    combo_dir.parent.mkdir(parents=True, exist_ok=True)
    combo_dir.symlink_to(tmp_path / '__output' / 'tag' / '__loop')
    (tmp_path / '__output' / 'tag' / '__loop').symlink_to(combo_dir)

    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--only', 'vit_wh_n150',
                                     '--only', 'vgg_wh_n150'])
    assert rpc.main() == 1, 'a partial matrix must not report success'
    assert [st for s, st in invoked if s == 'polaris.py'] == ['tag/vgg_wh_n150'], \
        'the healthy combo did not run'
    written = rpc.summary_path_for('tag').read_text()
    assert 'vit_wh_n150' in written, 'the looping combo is missing from "Not reported"'


@pytest.mark.unit
@pytest.mark.parametrize('flaw, expected', [
    ('outside_repo', 'resolves outside the repo'),
    ('directory', 'is not a regular file'),
])
def test_validate_combos_drops_a_reference_compare_layers_would_refuse(tmp_path, monkeypatch,
                                                                       flaw, expected):
    """Both ways a resolved reference can be unusable must be caught before simulating.

    `compare_layers` resolves its inputs and requires them (a) under `POLARIS_BASE_DIR`, pinned to
    the repo, and (b) to be regular files. `resolve_lfc_path` guards neither: it validates the
    cache location against a *relative* `__ext`, so a redirected one passes, and it treats any
    fresh *existing* path as a cache hit, a directory included. Either mismatch left to
    `compare_layers` costs a multi-minute simulation before the refusal arrives.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    for c in COMBOS:
        arch = tmp_path / c.archfile
        arch.parent.mkdir(parents=True, exist_ok=True)
        arch.write_text('# stub\n')

    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    if flaw == 'outside_repo':
        outside = tmp_path.parent / f'{tmp_path.name}_external_cache'
        outside.mkdir(exist_ok=True)
        monkeypatch.setattr(rpc, 'resolve_refrun', lambda c: outside / c.refrun_rel)
    else:
        inside = tmp_path / '__ext' / combo.refrun_rel          # a *directory* wearing the name
        inside.mkdir(parents=True, exist_ok=True)
        _stage_refrun(tmp_path, monkeypatch)

    ok, skipped = rpc.validate_combos([combo])

    assert ok == [], f'an unusable reference ({flaw}) was accepted'
    assert len(skipped) == 1 and expected in skipped[0][1], f'wrong reason for {flaw}'


# ── a CSV only counts as reusable once a run finished ──────────────────


@pytest.mark.unit
def test_skip_polaris_does_not_reuse_a_csv_from_an_unfinished_run(tmp_path, monkeypatch):
    """An interrupted run leaves a CSV that parses but describes part of the network.

    `print_csv` writes rows into the open file, so a killed polaris leaves a truncated but valid
    CSV. Reusing it would report a plausible network total for an incomplete network — the
    failure this tool exists to prevent. Existence proves a run started, not that one finished,
    so reuse tests the completion marker instead.
    """
    invoked = _stage_main(tmp_path, monkeypatch, failing_labels=set())
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    partial = rpc.opstats_path_for('tag', combo)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text('archname\nhalf-a-network\n')          # no marker beside it
    assert not rpc.completion_marker_for('tag', combo).exists()

    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--skip-polaris',
                                     '--only', 'vgg_wh_n150'])
    assert rpc.main() == 0
    assert [s for s, _ in invoked if s == 'polaris.py'] == ['polaris.py'], \
        'the partial CSV was reused instead of re-simulated'


@pytest.mark.unit
def test_skip_polaris_reuses_a_csv_that_carries_its_completion_marker(tmp_path, monkeypatch):
    """The flag must still do its job: a completed run's CSV is reused, not re-simulated.

    The counterpart to the test above — without this one, refusing to reuse *anything* would
    also pass.
    """
    invoked = _stage_main(tmp_path, monkeypatch, failing_labels=set())
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    done = rpc.opstats_path_for('tag', combo)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text('archname\na-whole-network\n')
    rpc.completion_marker_for('tag', combo).write_text('2026-09-07T00:00:00\n')

    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--skip-polaris',
                                     '--only', 'vgg_wh_n150'])
    assert rpc.main() == 0
    assert [s for s, _ in invoked if s == 'polaris.py'] == [], 'the completed CSV was re-simulated'
    assert done.read_text() == 'archname\na-whole-network\n', 'the reused CSV was cleared'
    assert rpc.completion_marker_for('tag', combo).exists(), 'the marker was cleared from under it'


@pytest.mark.unit
def test_skip_polaris_does_not_reuse_a_marker_whose_csv_is_gone(tmp_path, monkeypatch):
    """A marker on its own must not stand in for the CSV it vouches for.

    Testing the marker alone was the mirror image of testing the CSV alone: if the CSV is removed
    independently, reuse would skip simulation and hand `compare_layers` a path that is not there.
    Both artifacts are required.
    """
    invoked = _stage_main(tmp_path, monkeypatch, failing_labels=set())
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    marker = rpc.completion_marker_for('tag', combo)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('2026-09-07T00:00:00\n')                  # no CSV beside it
    assert not rpc.opstats_path_for('tag', combo).exists()

    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--skip-polaris',
                                     '--only', 'vgg_wh_n150'])
    assert rpc.main() == 0
    assert [s for s, _ in invoked if s == 'polaris.py'] == ['polaris.py'], \
        'an orphaned marker was taken as reusable output'


@pytest.mark.unit
@pytest.mark.parametrize('name', ['csv', 'marker'])
def test_reusable_opstats_rejects_a_directory_wearing_either_name(tmp_path, monkeypatch, name):
    """`is_file()` rather than `exists()`: a directory carrying either name answers for neither."""
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    csv, marker = rpc.opstats_path_for('tag', combo), rpc.completion_marker_for('tag', combo)
    csv.parent.mkdir(parents=True, exist_ok=True)
    if name == 'csv':
        csv.mkdir()
        marker.write_text('2026-09-07T00:00:00\n')
    else:
        csv.write_text('archname\n')
        marker.mkdir()

    assert rpc.reusable_opstats(rpc.opstats_artifacts('tag', combo)) is False


@pytest.mark.unit
@pytest.mark.parametrize('state, expected', [
    ('nothing', 'none present'),
    ('csv_only', 'an earlier run left one but did not finish'),
    ('marker_only', 'a completion marker outlived its CSV'),
])
def test_skip_polaris_note_names_which_case_it_hit(tmp_path, monkeypatch, capsys, state, expected):
    """The note has to be read off state captured *before* clearing, or it can only say one thing.

    Clearing deletes both artifacts for every combo that is not being reused, so a diagnostic
    that asks the filesystem inside the run loop reports "none present" in all three cases —
    including the two the message claims to distinguish. These assertions are what make the other
    two branches reachable rather than decorative.
    """
    _stage_main(tmp_path, monkeypatch, failing_labels=set())
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    csv, marker = rpc.opstats_path_for('tag', combo), rpc.completion_marker_for('tag', combo)
    csv.parent.mkdir(parents=True, exist_ok=True)
    if state == 'csv_only':
        csv.write_text('archname\nhalf-a-network\n')
    elif state == 'marker_only':
        marker.write_text('2026-09-07T00:00:00\n')

    monkeypatch.setattr('sys.argv', ['x', '--run-tag', 'tag', '--skip-polaris',
                                     '--only', 'vgg_wh_n150'])
    assert rpc.main() == 0
    assert expected in capsys.readouterr().err, f'the note did not name the {state} case'


@pytest.mark.unit
def test_run_polaris_writes_the_marker_only_after_the_csv_is_there(tmp_path, monkeypatch):
    """A marker must never vouch for a CSV that was not produced.

    `run_polaris` raises when polaris exits 0 without leaving a CSV; the marker is written after
    that check, so the next `--skip-polaris` run cannot trust a run that produced nothing.
    """
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')

    monkeypatch.setattr(rpc.subprocess, 'run',
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))
    with pytest.raises(FileNotFoundError):
        rpc.run_polaris(combo, 'tag')
    assert not rpc.completion_marker_for('tag', combo).exists(), \
        'a marker was left for a run that produced no CSV'


# ── the two contracts that live outside this file ──────────────────────


@pytest.mark.unit
def test_bs_runtime_1_combos_match_all_workloads_yaml():
    """A `bs_runtime == 1` combo passes no `--batchsize`, so the CSV name depends on the YAML.

    `opstats_path_for` builds `-b{bs_runtime}-opstats.csv`, and polaris names that file from the
    batch size it actually simulated. The ViT cells force agreement with an override; these cells
    are correct only while the instance carries `bs: 1` in `config/all_workloads.yaml`. Bump it
    there and `run_polaris` reports "opstats CSV not found" with nothing pointing at the reason,
    so the dependency is pinned here rather than discovered later.
    """
    spec = yaml.safe_load((REPO_ROOT / 'config' / 'all_workloads.yaml').read_text(encoding='utf-8'))
    declared = {
        name: body.get('bs')
        for group in spec['workloads']
        for name, body in (group.get('instances') or {}).items()
    }
    for combo in COMBOS:
        if combo.bs_runtime != 1:
            continue
        assert combo.instance_name in declared, f'{combo.label}: instance is not in all_workloads'
        assert declared[combo.instance_name] == 1, (
            f'{combo.label}: instance declares bs={declared[combo.instance_name]}, but the combo '
            f'passes no --batchsize override, so polaris will not write -b1-opstats.csv'
        )


@pytest.mark.unit
def test_polaris_argv_carries_the_flags_the_tool_depends_on(tmp_path, monkeypatch):
    """Pin the polaris argv, including the batch-size override only the ViT cells get.

    Nothing else asserts these: the output path is reconstructed from `bs_runtime`, and the
    comparison is only meaningful because of `--dump_stats_csv` and the two filters.
    """
    _stage_refrun(tmp_path, monkeypatch)
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen[Path(cmd[1]).name] = list(cmd)
        ops = rpc.opstats_path_for('tag', next(c for c in COMBOS if c.label == label))
        ops.parent.mkdir(parents=True, exist_ok=True)
        ops.write_text('archname\n')
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rpc.subprocess, 'run', fake_run)

    for label, expect_override in [('vit_wh_n150', True), ('vgg_wh_n150', False)]:
        seen.clear()
        combo = next(c for c in COMBOS if c.label == label)
        rpc.run_polaris(combo, 'tag')
        cmd = seen['polaris.py']
        assert cmd[:2] == [sys.executable, 'polaris.py']
        for flag, value in [('-w', 'config/all_workloads.yaml'), ('-a', combo.archfile),
                            ('-m', 'config/wl2archmapping.yaml'),
                            ('--filterarch', combo.filterarch),
                            ('--filterwli', combo.filterwli),
                            ('--study', f'tag/{combo.label}'), ('-o', '__output/')]:
            assert cmd[cmd.index(flag) + 1] == value, f'{label}: {flag} carries the wrong value'
        assert '--dump_stats_csv' in cmd, f'{label}: without this there is no opstats CSV to read'
        if expect_override:
            i = cmd.index('--batchsize')
            assert cmd[i + 1:i + 4] == ['1', str(combo.bs_runtime), str(combo.bs_runtime)], \
                'the leading 1 is forced by RangeArgument (start != end)'
        else:
            assert '--batchsize' not in cmd, f'{label}: bs comes from all_workloads.yaml'


@pytest.mark.unit
def test_compare_argv_passes_polaris_first_then_the_reference(tmp_path, monkeypatch):
    """The parser's Polaris-before-Profiler contract rests on this argument order.

    `test_extract_summary_requires_polaris_before_profiler` pins that a reversed label block must
    fail to parse, and its rationale is that this tool always passes the polaris CSV as file1 --
    which was the one property nothing checked. Swap these two and every combo would be reported
    as unparseable with the suite still green.
    """
    _stage_refrun(tmp_path, monkeypatch)
    monkeypatch.setattr(rpc, 'REPO_ROOT', tmp_path)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen['cmd'] = list(cmd)
        kwargs['stdout'].write(NORMAL_LOG)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rpc.subprocess, 'run', fake_run)
    combo = next(c for c in COMBOS if c.label == 'vgg_wh_n150')
    opstats, refrun = rpc.opstats_path_for('tag', combo), tmp_path / '__ext' / combo.refrun_rel
    # run_polaris creates the combo directory in the real flow; run_compare only writes into it.
    rpc.log_path_for('tag', combo, 'compare.log').parent.mkdir(parents=True, exist_ok=True)
    rpc.run_compare(combo, 'tag', opstats, refrun)

    cmd = seen['cmd']
    assert Path(cmd[1]).name == 'compare_layers.py'
    for flag in ['--perf', '--by-lut-key']:
        assert flag in cmd, f'{flag} shapes the output the parser reads'
    assert cmd[cmd.index('--xlsx') + 1] == str(rpc.xlsx_path_for('tag', combo))
    assert cmd[-2:] == [str(opstats), str(refrun)], 'polaris CSV must be file1, reference file2'


# ── combo table sanity ─────────────────────────────────────────────────


@pytest.mark.unit
def test_combo_labels_are_unique():
    """--only selects by label, so duplicates would make a selection ambiguous."""
    labels = [c.label for c in COMBOS]
    assert len(labels) == len(set(labels))


@pytest.mark.unit
def test_combo_refrun_paths_are_relative_and_csv():
    """Refruns are resolved under hlm-refrun/ on LFC; an absolute or traversing value would escape."""
    for c in COMBOS:
        assert not c.refrun_rel.startswith('/')
        assert '..' not in c.refrun_rel.split('/')
        assert c.refrun_rel.endswith('.csv')
