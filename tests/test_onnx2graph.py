#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
from ttsim.front.onnx.onnx2nx import onnx2graph

def test_read_onnx():
    graph = onnx2graph('temp', 'tests/models/onnx/inference/gpt_nano.onnx')

