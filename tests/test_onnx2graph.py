#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
from ttsim.front.onnx.onnx2nx import onnx2graph

@pytest.mark.unit
def test_read_onnx():
    """
    Test reading an ONNX file and converting it to a graph.
    """
    # NOTE: The 'tests/__models/' directory is used for large model files managed by Large File Cache (LFC) or Git LFS.
    # Ensure that these files are present (e.g., by running 'git lfs pull') before running this test.
    # See project documentation for more details on setting up large file dependencies.
    graph = onnx2graph('temp', 'tests/__models/onnx/inference/gpt_nano.onnx')
    assert graph is not None, "Graph should not be None after conversion"