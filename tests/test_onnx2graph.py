#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
from pathlib import Path
from ttsim.front.onnx.onnx2nx import onnx2graph
from tests.helpers.lfc_helper import require_lfc_file

@pytest.mark.unit
@pytest.mark.tools_secondary  # Mark as secondary since it requires external files
def test_read_onnx():
    """
    Test reading an ONNX file and converting it to a graph.
    
    This test uses the centralized __ext directory structure for LFC files.
    The required ONNX model file will be automatically downloaded if not present.
    """
    # Define the file path in the new __ext structure
    model_file = "__ext/tests/models/onnx/inference/gpt_nano.onnx"
    
    # Check if file exists before trying to download
    if not Path(model_file).exists():
        pytest.skip(f"LFC file not available: {model_file}. Run LFC downloader to get test files.")
    
    # Test the ONNX to graph conversion
    graph = onnx2graph('temp', model_file)
    assert graph is not None, "Graph should not be None after conversion"