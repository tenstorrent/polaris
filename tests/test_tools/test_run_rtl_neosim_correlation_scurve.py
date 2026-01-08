# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from tools.run_rtl_neosim_correlation import (
    SCurve,
    SCurveElem,
    SCurveElemIndices,
    SCurveUtils,
)


def make_elem(
    name: str,
    cls: str,
    rtl_pass: bool,
    rtl_cycles: int | None,
    model_pass: bool,
    model_cycles: int | None,
    ratio: float | None,
) -> SCurveElem:
    return SCurveElem(
        (
            name,
            cls,
            rtl_pass,
            rtl_cycles,
            model_pass,
            model_cycles,
            ratio,
        )
    )


def sample_data() -> list[SCurveElem]:
    # Valid ratios: 1.0, 1.1, 0.8, 4.0, 0.1
    return [
        make_elem("t1", "A", True, 100, True, 100, 1.0),
        make_elem("t2", "A", True, 100, True, 110, 1.1),
        make_elem("t3", "B", True, 200, True, 160, 0.8),
        make_elem("t4", "B", True, 120, False, None, None),
        make_elem("t5", "C", False, None, True, 50, None),
        make_elem("t6", "C", True, 100, True, 400, 4.0),
        make_elem("t7", "C", True, 100, True, 10, 0.1),
    ]


def test_scurveelem_str_and_tuple():
    e = make_elem("x", "cls", True, 10, False, None, None)
    s = e.get_s_curve_str(None)
    assert "x" in s and "cls" in s and "RTL: PASS" in s and "Model: FAIL" in s
    t = e.to_tuple()
    assert t[SCurveElemIndices.test_name] == "x"


def test_scurveutils_sort_default_and_custom():
    data = sample_data()
    default_sorted = SCurveUtils.sort(data[:], None, None)
    # Default: model_passed asc, rtl_passed asc, ratio desc, name asc
    # Hence a model-fail appears first (t4)
    assert default_sorted[0].test_name == "t4"

    # Custom sort: by test_name asc
    custom = SCurveUtils.sort(data[:], SCurveElemIndices.test_name, False)
    assert [e.test_name for e in custom] == sorted([e.test_name for e in data])


def test_scurve_basic_stats_and_counts():
    sc = SCurve("tag", sample_data())
    slower = sc.get_num_slower_than_rtl_tests()
    assert slower == 2  # t2 and t6

    gm = sc.geometric_mean_of_model_by_rtl_ratios()
    assert gm is not None and math.isclose(gm, 0.8116, rel_tol=1e-3)


def test_tolerance_band_stats_and_summary():
    sc = SCurve("tag", sample_data())
    n10, tot10, p10 = sc.get_tolerance_band_stats(10.0)
    assert (n10, tot10, round(p10, 2)) == (2, 5, 40.00)

    n20, tot20, p20 = sc.get_tolerance_band_stats(20.0)
    assert (n20, tot20, round(p20, 2)) == (3, 5, 60.00)

    summary = sc.get_tolerance_bands_summary()
    # Be robust to formatting while avoiding non-ASCII expectations
    assert "Tests within" in summary and "of Model/RTL=1.0" in summary
    assert "2/5" in summary and "3/5" in summary

    # Test edge case: negative tolerance should raise ValueError
    with pytest.raises(ValueError, match="tolerance_percent must be in range"):
        sc.get_tolerance_band_stats(-5.0)

    # Test edge case: tolerance > 100 should raise ValueError
    with pytest.raises(ValueError, match="tolerance_percent must be in range"):
        sc.get_tolerance_band_stats(150.0)

    # Test edge case: tolerance = 0 (exact match only)
    n0, tot0, p0 = sc.get_tolerance_band_stats(0.0)
    # Only tests with ratio exactly 1.0 should be counted
    assert n0 == 1 and tot0 == 5  # Only t1 has ratio=1.0

    # Test edge case: tolerance = 100 (all valid tests within band)
    n100, tot100, p100 = sc.get_tolerance_band_stats(100.0)
    # With ±100%, bounds are [0.0, 2.0], which includes ratios: 1.0, 1.1, 0.8, 0.1 but not 4.0
    assert tot100 == 5
    assert n100 == 4  # All except t6 (ratio=4.0)


def test_percentile_and_iqr_statistics_and_outliers():
    sc = SCurve("tag", sample_data())
    # Percentiles from explicit list
    ratios = [0.1, 0.8, 1.0, 1.1, 4.0]
    assert math.isclose(sc.calculate_percentile(ratios, 25.0), 0.8)
    assert math.isclose(sc.calculate_percentile(ratios, 50.0), 1.0)
    assert math.isclose(sc.calculate_percentile(ratios, 75.0), 1.1)
    assert math.isclose(sc.calculate_percentile([], 50.0), 0.0)

    # Test edge case: negative percentile should raise ValueError
    with pytest.raises(ValueError, match="percentile must be in range"):
        sc.calculate_percentile(ratios, -10.0)

    # Test edge case: percentile > 100 should raise ValueError
    with pytest.raises(ValueError, match="percentile must be in range"):
        sc.calculate_percentile(ratios, 150.0)

    # Test edge case: percentile = 0 (minimum value)
    assert math.isclose(sc.calculate_percentile(ratios, 0.0), 0.1)

    # Test edge case: percentile = 100 (maximum value)
    assert math.isclose(sc.calculate_percentile(ratios, 100.0), 4.0)

    iqr = sc.get_iqr_statistics()
    assert iqr is not None
    assert math.isclose(iqr["q1"], 0.8)
    assert math.isclose(iqr["median"], 1.0)
    assert math.isclose(iqr["q3"], 1.1)
    assert math.isclose(iqr["iqr"], 0.3)

    std_out = sc.get_iqr_outliers(False)
    ext_out = sc.get_iqr_outliers(True)
    assert {e.test_name for e in std_out} == {"t6", "t7"}
    assert {e.test_name for e in ext_out} == {"t6"}


def test_iqr_summaries_and_detailed_lists():
    sc = SCurve("tag", sample_data())
    smry = sc.get_iqr_summary()
    assert "IQR Statistics" in smry and "Outliers (1.5*IQR):" in smry
    detailed_std = sc.get_iqr_outliers_detailed(False, 2)
    detailed_ext = sc.get_iqr_outliers_detailed(True, 4)
    assert "+ Standard IQR Outliers" in detailed_std
    assert "+ Extreme IQR Outliers" in detailed_ext


def test_iqr_insufficient_data_returns_none():
    # Test with 3 valid tests (below the 4 minimum required for IQR)
    insufficient_data = [
        make_elem("t1", "A", True, 100, True, 100, 1.0),
        make_elem("t2", "A", True, 100, True, 110, 1.1),
        make_elem("t3", "B", True, 200, True, 160, 0.8),
        make_elem("t4", "B", True, 120, False, None, None),  # Invalid (model failed)
    ]
    sc = SCurve("tag", insufficient_data)

    # Verify get_iqr_statistics returns None
    iqr_stats = sc.get_iqr_statistics()
    assert iqr_stats is None

    # Verify dependent functions handle None gracefully
    assert sc.get_iqr_outliers(extreme=False) == []
    assert sc.get_iqr_outliers(extreme=True) == []

    summary = sc.get_iqr_summary()
    assert "Insufficient data" in summary and "need at least 4" in summary

    detailed = sc.get_iqr_outliers_detailed(extreme=False, num_white_chars_at_start=0)
    assert "No standard outliers detected" in detailed

    # Test with 0 valid tests
    no_valid_data = [
        make_elem("t1", "A", True, 100, False, None, None),
        make_elem("t2", "B", False, None, True, 50, None),
    ]
    sc_empty = SCurve("tag", no_valid_data)
    assert sc_empty.get_iqr_statistics() is None
    assert sc_empty.get_iqr_outliers(extreme=False) == []


def test_absolute_threshold_outliers_and_summary():
    sc = SCurve("tag", sample_data())
    under, over = sc.get_absolute_threshold_outliers(0.5)
    assert {e.test_name for e in under} == {"t7"}
    assert {e.test_name for e in over} == {"t6"}

    msg = sc.get_absolute_threshold_summary()
    assert ">30%" in msg and ">50%" in msg and ">100%" in msg

    # Test edge case: negative threshold should raise ValueError
    with pytest.raises(ValueError, match="ratio_threshold must be non-negative"):
        sc.get_absolute_threshold_outliers(-0.5)

    # Test edge case: zero threshold (all tests are outliers)
    under_zero, over_zero = sc.get_absolute_threshold_outliers(0.0)
    # With threshold=0, lower_bound=1.0, upper_bound=1.0, so anything != 1.0 is an outlier
    assert len(under_zero) + len(over_zero) > 0


def test_top_n_worst_by_ratio_and_cycles():
    sc = SCurve("tag", sample_data())
    msg_ratio = sc.get_top_n_worst_tests_summary(n=3, by_ratio=True)
    assert "t6" in msg_ratio and "t7" in msg_ratio and "t3" in msg_ratio
    msg_diff = sc.get_top_n_worst_tests_summary(n=3, by_ratio=False)
    assert "diff:" in msg_diff and "t6" in msg_diff and "t7" in msg_diff

    # Test edge case: n larger than number of valid tests (5 valid in sample_data)
    worst_10 = sc.get_top_n_worst_tests(n=10, by_ratio=True)
    assert len(worst_10) == 5  # Should return all 5 valid tests, not 10

    msg_all = sc.get_top_n_worst_tests_summary(n=100, by_ratio=True)
    assert "Top 5 Worst Tests" in msg_all  # Should show actual count (5), not requested (100)

    # Test with no valid tests
    no_valid_data = [
        make_elem("t1", "A", True, 100, False, None, None),
        make_elem("t2", "B", False, None, True, 50, None),
    ]
    sc_empty = SCurve("tag", no_valid_data)
    worst_empty = sc_empty.get_top_n_worst_tests(n=10, by_ratio=True)
    assert worst_empty == []
    msg_empty = sc_empty.get_top_n_worst_tests_summary(n=10, by_ratio=True)
    assert "No valid tests" in msg_empty


def test_s_curve_str_formatting():
    sc = SCurve("tag", sample_data())
    s = sc.get_s_curve_str(2)
    assert s.count("[") == len(sc.s_curve) and "RTL:" in s and "Model:" in s
