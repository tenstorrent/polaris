#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
from tools.parse_metal_tensix_results import parse_metal_tensix_results

@pytest.mark.tools_secondary
def test_parse_metal_tensix_results(tmp_path_factory):
    """
    Test the parsing of metal tensix performance numbers.

    Parameters:
        tmp_path_factory (pytest.TempPathFactory): Factory for creating temporary directories for test isolation.

    Expected Behavior:
        The function should create a temporary directory, run the parse_metal_tensix_results tool with the directory as output,
        and assert that the tool returns 0 indicating success.
    """
    tmpdir = tmp_path_factory.mktemp('parse_metal_tensix_results')
    res = parse_metal_tensix_results(['--output-dir', tmpdir.as_posix(), '--no-use-cache'])
    assert res == 0, f"parse_metal_tensix_results failed, got result: {res}"
