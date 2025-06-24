#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from tools.run_mlperf_nv_correlation import main as run_mlperf_nv_correlation


def test_run_mlperf_nv_correlation(tmp_path_factory):
    """
    Test MLPerf correlation script.
    """
    tmpdir = tmp_path_factory.mktemp('parse_metal_tensix_results')
    res = run_mlperf_nv_correlation()
    assert res == 0, "run_mlperf_nv_correlation failed"
