#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
from ttsim.front.onnx.onnx2nx import onnx2graph
from tests.helpers.lfc_helper import require_lfc_file

@pytest.mark.unit
def test_read_onnx():
    """
    Test reading an ONNX file and converting it to a graph.
    
    This test uses the centralized __ext directory structure for LFC files.
    The required ONNX model file will be automatically downloaded if not present.
    """
    # Define the file path in the new __ext structure
    model_file = "__ext/tests/models/onnx/inference/gpt_nano.onnx"
    
    # Ensure the file exists, downloading from LFC if necessary
    require_lfc_file(
        model_file, 
        "ext_test_onnx_models.tar.gz",
        "ONNX model for graph conversion test"
    )
    
    # Test the ONNX to graph conversion
    graph = onnx2graph('temp', model_file)
    assert graph is not None, "Graph should not be None after conversion"