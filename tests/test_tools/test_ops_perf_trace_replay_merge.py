# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the trace+replay merge sibling (ops_perf_trace_replay_merge.py).

Focus is the reduction step that differs from the iterative tool: dropping the
compile/warmup pass and selecting one replay session so join keys are unique.
The shared classify/join/output machinery is covered by
test_ops_perf_three_csv_merge.py and is exercised here only end-to-end.
"""

import csv

import pytest

from tools.si_profiling_helpers.ops_perf_trace_replay_merge import (
    COL_REPLAY_SESSION_ID,
    MergeError,
    main,
    reduce_trace_replay,
    replay_session_ids,
    select_replay_session,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TR_COLS = [
    'GLOBAL CALL COUNT', 'OP CODE', 'OP TYPE', 'DEVICE KERNEL DURATION [ns]',
    'DRAM BW UTIL (%)', 'FPU Util Median (%)', 'SFPU Util Median (%)',
    'NOC UTIL (%)', 'MULTICAST NOC UTIL (%)', 'ETH BW UTIL (%)', 'NPE CONG IMPACT (%)',
    COL_REPLAY_SESSION_ID,
]


def _tr_row(gcc, op, ns, rsid, dram='', fpu=''):
    return {
        'GLOBAL CALL COUNT': str(gcc), 'OP CODE': op, 'OP TYPE': 'op',
        'DEVICE KERNEL DURATION [ns]': str(ns), 'DRAM BW UTIL (%)': dram,
        'FPU Util Median (%)': fpu, 'SFPU Util Median (%)': '',
        'NOC UTIL (%)': '', 'MULTICAST NOC UTIL (%)': '', 'ETH BW UTIL (%)': '',
        'NPE CONG IMPACT (%)': '', COL_REPLAY_SESSION_ID: rsid,
    }


def _capture_rows(ns_by_session, ops=(('1024', 'Matmul'), ('2048', 'Add')), dram='', fpu=''):
    """Build a trace+replay capture: a compile pass (rsid='') plus one block per
    replay session, every block carrying the same ops/join keys.

    ``ns_by_session`` maps session id ('' for compile) -> per-op kernel ns, so a
    session's total (and thus median/min/max selection) is controllable.
    """
    rows = []
    for sid, ns in ns_by_session.items():
        for gcc, op in ops:
            rows.append(_tr_row(gcc, op, ns, sid, dram=dram, fpu=fpu))
    return rows


# ---------------------------------------------------------------------------
# replay_session_ids / select_replay_session
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_replay_session_ids_distinct_in_capture_order():
    rows = _capture_rows({'': 999, '1': 100, '2': 200})
    assert replay_session_ids(rows) == ['1', '2']  # blank excluded, first-appearance order


@pytest.mark.unit
def test_select_replay_session_median_lower_middle():
    # session '1' total < session '2' total -> median (lower-middle of 2) picks '1'.
    rows = _capture_rows({'': 999, '1': 100, '2': 200})
    assert select_replay_session(rows, 'median') == '1'


@pytest.mark.unit
def test_select_replay_session_min_max():
    rows = _capture_rows({'': 999, '1': 100, '2': 200})
    assert select_replay_session(rows, 'min') == '1'
    assert select_replay_session(rows, 'max') == '2'


@pytest.mark.unit
def test_select_replay_session_first_last_capture_order():
    rows = _capture_rows({'': 999, '2': 200, '1': 100})  # session 2 appears first
    assert select_replay_session(rows, 'first') == '2'
    assert select_replay_session(rows, 'last') == '1'


@pytest.mark.unit
def test_select_replay_session_no_sessions_raises():
    rows = _capture_rows({'': 999})  # compile pass only, no replay session
    with pytest.raises(MergeError, match='trace\\+replay'):
        select_replay_session(rows, 'median')


# ---------------------------------------------------------------------------
# reduce_trace_replay
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reduce_trace_replay_drops_compile_and_selects_one_session():
    rows = _capture_rows({'': 999, '1': 100, '2': 200})
    noc, fpu, van = reduce_trace_replay(list(rows), list(rows), list(rows))
    # 2 ops kept (one session), compile pass and the other session dropped.
    assert len(van) == 2
    # median picks session '1' (lower total) -> its distinguishing ns=100 survives.
    assert {r['DEVICE KERNEL DURATION [ns]'] for r in van} == {'100'}
    # join keys are now unique.
    keys = [(r['GLOBAL CALL COUNT'], r['OP CODE'], r['OP TYPE']) for r in van]
    assert len(keys) == len(set(keys))
    assert len(noc) == len(fpu) == len(van)


@pytest.mark.unit
def test_reduce_trace_replay_requires_rsid_column():
    row = {'GLOBAL CALL COUNT': '1', 'OP CODE': 'Matmul', 'OP TYPE': 'op',
           'DEVICE KERNEL DURATION [ns]': '10'}  # no replay-session column
    with pytest.raises(MergeError, match='trace\\+replay'):
        reduce_trace_replay([row], [row], [row])


@pytest.mark.unit
def test_reduce_trace_replay_compose_iteration_within_session():
    # A single replay session that itself holds two iterations (op sequence AB
    # repeated with per-position join keys) -> keys not unique -> the iteration
    # reducer composes to collapse it to one iteration.
    seq = [('1024', 'A'), ('2048', 'B')]
    rows = []
    for _ in range(2):  # two iterations within session '1'
        for gcc, op in seq:
            rows.append(_tr_row(gcc, op, 100, '1'))
    noc, fpu, van = reduce_trace_replay(list(rows), list(rows), list(rows))
    assert len(van) == 2  # collapsed to a single iteration
    keys = [(r['GLOBAL CALL COUNT'], r['OP CODE'], r['OP TYPE']) for r in van]
    assert len(keys) == len(set(keys))


@pytest.mark.unit
def test_reduce_trace_replay_no_compose_raises_on_duplicate_keys():
    seq = [('1024', 'A'), ('2048', 'B')]
    rows = []
    for _ in range(2):
        for gcc, op in seq:
            rows.append(_tr_row(gcc, op, 100, '1'))
    with pytest.raises(MergeError, match='no-compose-iteration'):
        reduce_trace_replay(list(rows), list(rows), list(rows), compose_iteration=False)


# ---------------------------------------------------------------------------
# End-to-end via main(): reproduces the duplicate-key failure and its fix
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=_TR_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in _TR_COLS})


@pytest.mark.unit
def test_main_merges_trace_replay_capture(tmp_path):
    # Compile pass + two replay sessions, all sharing join keys -> the iterative
    # tool would abort on a duplicate key; the trace+replay tool keeps one session.
    _write_csv(tmp_path / 'trace.csv',
               _capture_rows({'': 999, '1': 100, '2': 200}, dram='45.2'))
    _write_csv(tmp_path / 'perf.csv',
               _capture_rows({'': 999, '1': 100, '2': 200}, fpu='62.1'))
    _write_csv(tmp_path / 'raw.csv',
               _capture_rows({'': 999, '1': 100, '2': 200}))

    out = tmp_path / 'merged_ops.csv'
    rc = main([
        '--input-dir', str(tmp_path), '--dram-peak-bw-gbps', '288.0',
        '--output', str(out), '--duration-rel-tol', '1e9',
    ])
    assert rc == 0

    with out.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2  # one replay session, compile pass dropped
    keys = [(r['GLOBAL CALL COUNT'], r['OP CODE'], r['OP TYPE']) for r in rows]
    assert len(keys) == len(set(keys))  # unique -> join did not abort
    # median selected session '1' (lower total); its vanilla ns is canonical.
    assert {r['DEVICE KERNEL DURATION [ns]'] for r in rows} == {'100'}


def _tr_signpost(name):
    # Tracy signpost marker as TT-Metal emits it: OP TYPE 'signpost', empty GLOBAL CALL
    # COUNT and no replay session.
    r = {c: '' for c in _TR_COLS}
    r['OP CODE'] = name
    r['OP TYPE'] = 'signpost'
    return r


@pytest.mark.unit
def test_main_drops_signpost_rows_in_trace_replay(tmp_path):
    # A trace+replay capture whose passes are wrapped in start/stop Tracy signposts
    # (empty GLOBAL CALL COUNT). The signpost drop lives in the shared run_merge, so it
    # must cover the trace+replay variant too: the merge must succeed and the marker
    # rows must never reach the output. (Same root cause as the ResNet50 3-CSV failure.)
    def wrapped(**kw):
        capture = _capture_rows({'': 999, '1': 100, '2': 200}, **kw)
        return [_tr_signpost('start'), *capture, _tr_signpost('stop')]

    _write_csv(tmp_path / 'trace.csv', wrapped(dram='45.2'))
    _write_csv(tmp_path / 'perf.csv', wrapped(fpu='62.1'))
    _write_csv(tmp_path / 'raw.csv', wrapped())

    out = tmp_path / 'merged_ops.csv'
    rc = main([
        '--input-dir', str(tmp_path), '--dram-peak-bw-gbps', '288.0',
        '--output', str(out), '--duration-rel-tol', '1e9',
    ])
    assert rc == 0

    with out.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert {r['OP CODE'] for r in rows} == {'Matmul', 'Add'}
    assert all(r['OP TYPE'] != 'signpost' for r in rows)


@pytest.mark.unit
def test_main_errors_on_non_trace_replay_capture(tmp_path):
    # No populated replay-session id -> not a trace+replay capture -> exit 1.
    _write_csv(tmp_path / 'trace.csv', _capture_rows({'': 100}, dram='45.2'))
    _write_csv(tmp_path / 'perf.csv', _capture_rows({'': 100}, fpu='62.1'))
    _write_csv(tmp_path / 'raw.csv', _capture_rows({'': 100}))
    rc = main(['--input-dir', str(tmp_path), '--dram-peak-bw-gbps', '288.0'])
    assert rc == 1
