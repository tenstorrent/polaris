# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.profiling.ops_perf_three_csv_merge import (
    MergeError,
    classify_file,
    detect_iteration_boundaries,
    iteration_summary,
    pick_median_iteration,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows(op_codes, kdurs=None, drams=None, fpus=None):
    """Build minimal row dicts for testing."""
    n = len(op_codes)
    kdurs = kdurs or ['100'] * n
    drams = drams or [''] * n
    fpus = fpus or [''] * n
    return [
        {
            'OP CODE': op_codes[i],
            'OP TYPE': 'op',
            'GLOBAL CALL COUNT': str(i),
            'DEVICE KERNEL DURATION [ns]': kdurs[i],
            'DRAM BW UTIL (%)': drams[i],
            'FPU Util Median (%)': fpus[i],
            'SFPU Util Median (%)': '',
            'NOC UTIL (%)': '',
            'MULTICAST NOC UTIL (%)': '',
            'ETH BW UTIL (%)': '',
            'NPE CONG IMPACT (%)': '',
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# detect_iteration_boundaries
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_detect_iteration_boundaries_marker_basic():
    rows = _rows(['conv', 'relu', 'pool', 'conv', 'relu', 'pool'])
    iters = detect_iteration_boundaries(rows)
    assert iters == [(0, 3), (3, 6)]


@pytest.mark.unit
def test_detect_iteration_boundaries_single_iteration():
    rows = _rows(['conv', 'relu', 'pool'])
    iters = detect_iteration_boundaries(rows)
    assert iters == [(0, 3)]


@pytest.mark.unit
def test_detect_iteration_boundaries_fixed_size():
    rows = _rows(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'])
    iters = detect_iteration_boundaries(rows, ops_per_iteration=3)
    assert iters == [(0, 3), (3, 6), (6, 9)]


@pytest.mark.unit
def test_detect_iteration_boundaries_fixed_size_not_divisible():
    rows = _rows(['a', 'b', 'c', 'a', 'b'])
    with pytest.raises(MergeError, match='does not evenly divide'):
        detect_iteration_boundaries(rows, ops_per_iteration=3)


@pytest.mark.unit
def test_detect_iteration_boundaries_unequal_sizes():
    # 'a' recurs at positions 0 and 2 (size 2), then at 4 (size 3)
    rows = _rows(['a', 'b', 'a', 'b', 'a', 'b', 'c'])
    with pytest.raises(MergeError, match='unequal sizes'):
        detect_iteration_boundaries(rows)


# ---------------------------------------------------------------------------
# iteration_summary
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_iteration_summary_complete():
    rows = _rows(['a'] * 4, kdurs=['100', '200', '150', '50'])
    iters = [(0, 4)]
    summary = iteration_summary(rows, iters)
    assert len(summary) == 1
    start, end, is_complete, total = summary[0]
    assert (start, end) == (0, 4)
    assert is_complete is True
    assert total == pytest.approx(500.0)


@pytest.mark.unit
def test_iteration_summary_float_duration_counted():
    # '123.0' must be counted — the fix ensures parse_finite_float is used, not int()
    rows = _rows(['a'] * 3, kdurs=['100', '123.0', '50'])
    iters = [(0, 3)]
    summary = iteration_summary(rows, iters)
    _, _, is_complete, total = summary[0]
    assert is_complete is True
    assert total == pytest.approx(273.0)


@pytest.mark.unit
def test_iteration_summary_sparse_incomplete():
    # Only 1/4 rows have a duration — below the 95% threshold
    rows = _rows(['a'] * 4, kdurs=['100', '', '', ''])
    iters = [(0, 4)]
    summary = iteration_summary(rows, iters)
    _, _, is_complete, _ = summary[0]
    assert is_complete is False


@pytest.mark.unit
def test_iteration_summary_dash_treated_as_empty():
    # '-' is non-parseable and must not count toward completeness
    rows = _rows(['a'] * 4, kdurs=['100', '-', '-', '-'])
    iters = [(0, 4)]
    summary = iteration_summary(rows, iters)
    _, _, is_complete, _ = summary[0]
    assert is_complete is False


@pytest.mark.unit
def test_iteration_summary_multiple_iters():
    rows = _rows(['a'] * 6, kdurs=['10', '20', '', '30', '40', '50'])
    iters = [(0, 3), (3, 6)]
    summary = iteration_summary(rows, iters)
    # First iter: 2/3 rows — below 95%
    assert summary[0][2] is False
    # Second iter: 3/3 rows — complete
    assert summary[1][2] is True
    assert summary[1][3] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# pick_median_iteration
# ---------------------------------------------------------------------------

def _make_summary(totals, complete=None):
    if complete is None:
        complete = [True] * len(totals)
    return [(i * 10, (i + 1) * 10, complete[i], float(totals[i])) for i in range(len(totals))]


@pytest.mark.unit
def test_pick_median_odd():
    # 3 complete iters with totals 30, 10, 20 → sorted [10, 20, 30] → median = 20
    summary = _make_summary([30, 10, 20])
    start, end, idx = pick_median_iteration(summary)
    assert summary[idx][3] == pytest.approx(20.0)


@pytest.mark.unit
def test_pick_median_even_lower_middle():
    # 4 complete iters, totals 10, 40, 20, 30 → sorted [10, 20, 30, 40]
    # lower-middle index = (4-1)//2 = 1 → value 20
    summary = _make_summary([10, 40, 20, 30])
    start, end, idx = pick_median_iteration(summary)
    assert summary[idx][3] == pytest.approx(20.0)


@pytest.mark.unit
def test_pick_min():
    summary = _make_summary([30, 10, 20])
    _, _, idx = pick_median_iteration(summary, select='min')
    assert summary[idx][3] == pytest.approx(10.0)


@pytest.mark.unit
def test_pick_max():
    summary = _make_summary([30, 10, 20])
    _, _, idx = pick_median_iteration(summary, select='max')
    assert summary[idx][3] == pytest.approx(30.0)


@pytest.mark.unit
def test_pick_first():
    summary = _make_summary([30, 10, 20])
    _, _, idx = pick_median_iteration(summary, select='first')
    assert idx == 0


@pytest.mark.unit
def test_pick_last():
    summary = _make_summary([30, 10, 20])
    _, _, idx = pick_median_iteration(summary, select='last')
    assert idx == 2


@pytest.mark.unit
def test_pick_skips_incomplete():
    # Only iter 1 is complete
    summary = _make_summary([30, 10, 20], complete=[False, True, False])
    _, _, idx = pick_median_iteration(summary)
    assert idx == 1


@pytest.mark.unit
def test_pick_no_complete_raises():
    summary = _make_summary([30, 10, 20], complete=[False, False, False])
    with pytest.raises(MergeError, match='No iterations available'):
        pick_median_iteration(summary)


@pytest.mark.unit
def test_pick_measured_indices_override():
    # measured_indices=[0, 2] → candidates are iters 0 and 2 regardless of completeness
    summary = _make_summary([30, 10, 20], complete=[False, True, False])
    _, _, idx = pick_median_iteration(summary, measured_indices=[0, 2], select='min')
    assert summary[idx][3] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# classify_file
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_classify_noctrace():
    rows = _rows(['conv'], drams=['45.2'])
    assert classify_file(rows) == 'noctrace'


@pytest.mark.unit
def test_classify_fpu():
    rows = _rows(['conv'], drams=[''], fpus=['62.1'])
    assert classify_file(rows) == 'fpu'


@pytest.mark.unit
def test_classify_pm_fpu_alone_is_vanilla():
    # PM FPU UTIL (%) is present in all three CSV types and must not be used as
    # a classification signal. A file with only PM FPU UTIL (non-zero) and no
    # FPU Util Median / SFPU Util Median columns must classify as vanilla.
    rows = [
        {
            'OP CODE': 'conv', 'OP TYPE': 'op', 'GLOBAL CALL COUNT': '0',
            'DEVICE KERNEL DURATION [ns]': '100',
            'DRAM BW UTIL (%)': '', 'FPU Util Median (%)': '', 'SFPU Util Median (%)': '',
            'PM FPU UTIL (%)': '55.0',
            'NOC UTIL (%)': '', 'MULTICAST NOC UTIL (%)': '',
            'ETH BW UTIL (%)': '', 'NPE CONG IMPACT (%)': '',
        }
    ]
    assert classify_file(rows) == 'vanilla'


@pytest.mark.unit
def test_classify_vanilla():
    rows = _rows(['conv'], drams=[''], fpus=[''])
    assert classify_file(rows) == 'vanilla'


@pytest.mark.unit
def test_classify_empty_rows():
    assert classify_file([]) == 'unassigned'
