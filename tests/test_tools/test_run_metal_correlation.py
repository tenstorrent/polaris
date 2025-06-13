#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from tools.run_metal_tensix_correlation import main as run_metal_tensix_correlation


def test_run_metal_tensix_correlation(tmp_path_factory):
    """
    Test MLPerf correlation script.
    """
    tmpdir = tmp_path_factory.mktemp('parse_metal_tensix_results')
    res = run_metal_tensix_correlation()
    assert res == 0, "run_metal_tensix_correlation failed"
