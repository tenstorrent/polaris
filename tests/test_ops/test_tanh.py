#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim Tanh — data_compute numerical validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.data_compute import compute_tanh
from tests.test_ops.utils import generate_test_data

RTOL = 1e-4
ATOL = 1e-5
SEED = 42

edge_types = ['positive', 'negative', 'zeros', 'mixed', 'small', 'large', 'minimum_input']
base_shape = (2, 4, 8)


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


def _shape_for(data_type):
    return (1,) if data_type == 'minimum_input' else base_shape


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_tanh(data_type):
    """Tanh numerical validation across edge cases."""
    _seed()
    shape = _shape_for(data_type)
    raw = generate_test_data(shape, data_type)
    t = make_sim_tensor(raw, "tanh_in")

    tt_out = compute_tanh([t], _FakeOp())
    ref_out = np.tanh(raw)

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    num_ok = bool(np.allclose(tt_out, ref_out, rtol=RTOL, atol=ATOL))

    if shape_ok and num_ok:
        print(f"Tanh/{data_type}: PASS")
    else:
        max_diff = float(np.abs(tt_out - ref_out).max())
        print(f"Tanh/{data_type}: FAIL (max_diff={max_diff:.2e})")

    assert shape_ok, f"Tanh/{data_type} shape mismatch"
    assert num_ok, f"Tanh/{data_type} numerical mismatch"


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh_output_range():
    """Tanh output should be in [-1, 1]."""
    _seed()
    raw = generate_test_data((4, 8, 16), 'mixed')
    t = make_sim_tensor(raw, "tanh_range")
    tt_out = compute_tanh([t], _FakeOp())
    assert np.all(tt_out >= -1.0) and np.all(tt_out <= 1.0), \
        "Tanh output outside [-1, 1]"


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh_odd_function():
    """tanh(-x) = -tanh(x) — odd function property."""
    _seed()
    raw = generate_test_data((2, 4, 8), 'mixed')
    t_pos = make_sim_tensor(raw, "tanh_pos")
    t_neg = make_sim_tensor(-raw, "tanh_neg")
    out_pos = compute_tanh([t_pos], _FakeOp())
    out_neg = compute_tanh([t_neg], _FakeOp())
    assert np.allclose(out_neg, -out_pos, rtol=RTOL, atol=ATOL), \
        "Tanh odd-function property violated"


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh_at_zero():
    """tanh(0) = 0."""
    raw = np.zeros((1,), dtype=np.float32)
    t = make_sim_tensor(raw, "tanh_zero")
    tt_out = compute_tanh([t], _FakeOp())
    assert np.allclose(tt_out, 0.0, rtol=RTOL, atol=ATOL), \
        f"tanh(0) = {tt_out}, expected 0.0"
