#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
from ttsim.utils.common import parse_yaml, parse_csv
from ttsim.config import create_ipblock, create_package
from ttsim.front.onnx.onnx2nx import onnx2graph

def test_read_onnx():
    graph = onnx2graph('temp', 'tests/models/onnx/inference/gpt_nano.onnx')

