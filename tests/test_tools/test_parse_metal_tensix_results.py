#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from tools.parse_metal_tensix_results import parse_metal_tensix_results


def test_parse_metal_tensix_results(tmp_path_factory):
    """
    Test the parsing of metal tensix performance numbers.
    """
    tmpdir = tmp_path_factory.mktemp('parse_metal_tensix_results')
    res = parse_metal_tensix_results(['--output-dir', tmpdir.as_posix(),], use_cache=False)
    assert res == 0, "parse_metal_tensix_results failed"
