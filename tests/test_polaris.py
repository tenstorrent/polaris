#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import polaris


def test_polaris():
    assert polaris.main(['--odir', '__dummy', '--study', 'dummy', '--wlspec', 'config/mlperf_inference.yaml',
                         '--archspec', 'config/all_archs.yaml', '--wlmapspec',  'config/wl2archmapping.yaml',
                         '--dryrun']) == 0, "Polaris main function should return 0 on success"
