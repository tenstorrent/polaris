#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim Sigmoid — data_compute numerical validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.data_compute import compute_sigmoid
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


def ref_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_sigmoid(data_type):
    """Sigmoid numerical validation across edge cases."""
    _seed()
    shape = _shape_for(data_type)
    raw = generate_test_data(shape, data_type)
    t = make_sim_tensor(raw, "sigmoid_in")
    fake_op = _FakeOp()

    tt_out = compute_sigmoid([t], fake_op)
    ref_out = ref_sigmoid(raw)

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    num_ok = bool(np.allclose(tt_out, ref_out, rtol=RTOL, atol=ATOL))

    if shape_ok and num_ok:
        print(f"Sigmoid/{data_type}: PASS")
    else:
        max_diff = float(np.abs(tt_out - ref_out).max())
        print(f"Sigmoid/{data_type}: FAIL (max_diff={max_diff:.2e})")

    assert shape_ok, f"Sigmoid/{data_type} shape mismatch"
    assert num_ok, f"Sigmoid/{data_type} numerical mismatch"


@pytest.mark.unit
@pytest.mark.opunit
def test_sigmoid_output_range():
    """Sigmoid output should always be in (0, 1)."""
    _seed()
    raw = generate_test_data((4, 8, 16), 'mixed')
    t = make_sim_tensor(raw, "sigmoid_range")
    tt_out = compute_sigmoid([t], _FakeOp())
    assert np.all(tt_out >= 0.0) and np.all(tt_out <= 1.0), \
        "Sigmoid output outside [0, 1]"


@pytest.mark.unit
@pytest.mark.opunit
def test_sigmoid_symmetry():
    """sigmoid(-x) = 1 - sigmoid(x)."""
    _seed()
    raw = generate_test_data((2, 4, 8), 'mixed')
    t_pos = make_sim_tensor(raw, "sig_pos")
    t_neg = make_sim_tensor(-raw, "sig_neg")
    out_pos = compute_sigmoid([t_pos], _FakeOp())
    out_neg = compute_sigmoid([t_neg], _FakeOp())
    assert np.allclose(out_neg, 1.0 - out_pos, rtol=RTOL, atol=ATOL), \
        "Sigmoid symmetry violated"


@pytest.mark.unit
@pytest.mark.opunit
def test_sigmoid_at_zero():
    """sigmoid(0) = 0.5."""
    raw = np.zeros((1,), dtype=np.float32)
    t = make_sim_tensor(raw, "sig_zero")
    tt_out = compute_sigmoid([t], _FakeOp())
    assert np.allclose(tt_out, 0.5, rtol=RTOL, atol=ATOL), \
        f"sigmoid(0) = {tt_out}, expected 0.5"
