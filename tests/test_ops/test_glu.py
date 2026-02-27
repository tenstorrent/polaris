#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim GLU — data_compute numerical validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py

GLU splits input in half along last dim: first_half * sigmoid(second_half).
Requires even-sized last dimension.

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.data_compute import compute_glu
from tests.test_ops.utils import generate_test_data

RTOL = 1e-4
ATOL = 1e-5
SEED = 42

edge_types = ['positive', 'negative', 'zeros', 'mixed', 'small', 'large', 'minimum_input']


def _seed():
    np.random.seed(SEED)


def make_sim_tensor(data, name="t"):
    return SimTensor({
        "name": name,
        "shape": list(data.shape),
        "data": data.copy(),
        "dtype": np.dtype(np.float32),
    })


class _FakeOp:
    def __init__(self, **attrs):
        self.attrs = attrs


def ref_glu(x, dim=-1):
    """NumPy reference GLU: first_half * sigmoid(second_half)."""
    half = x.shape[dim] // 2
    sl1 = [slice(None)] * x.ndim
    sl2 = [slice(None)] * x.ndim
    sl1[dim] = slice(0, half)
    sl2[dim] = slice(half, None)
    a = x[tuple(sl1)]
    b = x[tuple(sl2)]
    return a * (1.0 / (1.0 + np.exp(-b)))


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_glu(data_type):
    """GLU numerical validation — splits last dim, applies sigmoid gate."""
    _seed()
    # GLU needs even-sized last dim; minimum_input uses (1, 2)
    shape = (1, 2) if data_type == 'minimum_input' else (2, 4, 8)
    raw = generate_test_data(shape, data_type)
    t = make_sim_tensor(raw, "glu_in")

    tt_out = compute_glu([t], _FakeOp(dim=-1))
    ref_out = ref_glu(raw, dim=-1)

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    num_ok = bool(np.allclose(tt_out, ref_out, rtol=RTOL, atol=ATOL))

    if shape_ok and num_ok:
        print(f"Glu/{data_type}: PASS")
    else:
        max_diff = float(np.abs(tt_out - ref_out).max())
        print(f"Glu/{data_type}: FAIL (max_diff={max_diff:.2e})")

    assert shape_ok, f"Glu/{data_type} shape mismatch"
    assert num_ok, f"Glu/{data_type} numerical mismatch"


@pytest.mark.unit
@pytest.mark.opunit
def test_glu_output_shape():
    """GLU output should have half the size of last dim."""
    _seed()
    raw = generate_test_data((2, 4, 16), 'mixed')
    t = make_sim_tensor(raw, "glu_shape")
    tt_out = compute_glu([t], _FakeOp(dim=-1))
    expected_shape = [2, 4, 8]
    assert list(tt_out.shape) == expected_shape, \
        f"GLU output shape {list(tt_out.shape)} != {expected_shape}"


@pytest.mark.unit
@pytest.mark.opunit
def test_glu_zeros_gate():
    """GLU with zero gate values: first_half * sigmoid(0) = first_half * 0.5."""
    _seed()
    shape = (2, 4, 8)
    raw = np.zeros(shape, dtype=np.float32)
    # Set first half to known values
    raw[:, :, :4] = 2.0  # first half = 2.0
    # second half is 0.0 → sigmoid(0) = 0.5
    t = make_sim_tensor(raw, "glu_zeros")
    tt_out = compute_glu([t], _FakeOp(dim=-1))
    expected = np.full((2, 4, 4), 1.0, dtype=np.float32)  # 2.0 * 0.5 = 1.0
    assert np.allclose(tt_out, expected, rtol=RTOL, atol=ATOL), \
        "GLU zeros-gate test failed"
