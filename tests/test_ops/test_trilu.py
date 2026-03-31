#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for TTSim Trilu — DLRM 1-input path + ONNX 2-input path.

- 1-input: legacy DLRM behavior (flattened upper triangle without diagonal).
- 2-input: ONNX behavior (output shape == input shape), k is ignored for shape.
"""

import numpy as np
import pytest

from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.tensor import trilu_sinf
from tests.test_ops.utils import generate_test_data

RTOL = 1e-4
ATOL = 1e-5
SEED = 42

def _seed():
    np.random.seed(SEED)

def make_sim_tensor(data, name="t"):
    return SimTensor(
        {
            "name": name,
            "shape": list(data.shape),
            "data": data.copy(),
            "dtype": np.dtype(np.float32),
        }
    )

class _FakeOp:
    def __init__(self, **attrs):
        self.attrs = attrs
        self.precision = np.dtype(np.float32)
        self.perf_stats = None

@pytest.mark.unit
@pytest.mark.opunit
def test_trilu_dlrm_flatten_upper():
    """
    1-input Trilu — DLRM legacy path via F.TriluX.

    Input:  [B, N, N]
    Output: [B, N*(N-1)/2] upper-triangle (i<j) flattened.
    """
    _seed()
    B, N = 2, 4
    x_data = generate_test_data((B, N, N), "mixed").astype(np.float32)
    x = make_sim_tensor(x_data, "trilu_dlrm_in")

    trilu = F.TriluX("trilu_dlrm")
    out = trilu(x)

    expected_cols = N * (N - 1) // 2
    assert list(out.shape) == [B, expected_cols]

    ref = []
    for b in range(B):
        upper_vals = []
        for i in range(N):
            for j in range(i + 1, N):
                upper_vals.append(x_data[b, i, j])
        ref.append(upper_vals)
    ref = np.asarray(ref, dtype=np.float32)

    assert out.data is not None
    assert np.allclose(out.data, ref, rtol=RTOL, atol=ATOL)

@pytest.mark.unit
@pytest.mark.opunit
def test_trilu_onnx_two_input_shape():
    """
    2-input Trilu — ONNX path via trilu_sinf directly.

    Input:  data [B, H, W], k scalar
    Output: same shape as data.
    """
    _seed()
    B, H, W = 3, 5, 5
    x_data = generate_test_data((B, H, W), "mixed").astype(np.float32)
    x = make_sim_tensor(x_data, "trilu_onnx_in")

    k_data = np.array(0, dtype=np.int64)
    k = SimTensor(
        {
            "name": "trilu_k",
            "shape": [],
            "data": k_data.copy(),
            "dtype": k_data.dtype,
        }
    )

    y_data = np.zeros_like(x_data, dtype=np.float32)
    y = make_sim_tensor(y_data, "trilu_onnx_out")

    op = _FakeOp(upper=1)

    trilu_sinf([x, k], [y], op)

    assert list(y.shape) == [B, H, W]
    assert y.dtype == x.dtype