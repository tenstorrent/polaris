#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

@pytest.mark.unit
@pytest.mark.opunit
def test_dequantize_linear_shape_and_dtype():
    """DequantizeLinear should preserve shape and produce fp32 output."""
    op_name = "DequantizeLinearOp"

    x = F._from_shape("X", [2, 3, 4], np_dtype=np.uint8)
    s = F._from_shape("Scale", [], np_dtype=np.float32)
    zp = F._from_shape("ZeroPoint", [], np_dtype=np.uint8)

    y = make_tensor("Y")

    op_info = {
        "name": op_name,
        "optype": "DequantizeLinear",
        "inList": [x.name, s.name, zp.name],
        "outList": [y.name],
    }
    op = SimOp(op_info)

    x.op_in = [op_name]
    s.op_in = [op_name]
    zp.op_in = [op_name]
    y.op_out = [op_name]

    op.get_perf_counts([x, s, zp], [y])

    assert y.shape == x.shape
    assert y.dtype == np.dtype(np.float32)

@pytest.mark.unit
@pytest.mark.opunit
def test_quantize_linear_shape_and_dtype_default_uint8():
    """QuantizeLinear should preserve shape and produce uint8 output by default."""
    op_name = "QuantizeLinearOp"

    x = F._from_shape("X", [2, 3, 4], np_dtype=np.float32)
    s = F._from_shape("Scale", [], np_dtype=np.float32)
    zp = F._from_shape("ZeroPoint", [], np_dtype=np.uint8)

    y = make_tensor("Y")

    op_info = {
        "name": op_name,
        "optype": "QuantizeLinear",
        "inList": [x.name, s.name, zp.name],
        "outList": [y.name],
    }
    op = SimOp(op_info)

    x.op_in = [op_name]
    s.op_in = [op_name]
    zp.op_in = [op_name]
    y.op_out = [op_name]

    op.get_perf_counts([x, s, zp], [y])

    assert y.shape == x.shape
    assert y.dtype == np.dtype(np.uint8)