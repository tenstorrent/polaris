#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim Sqrt — data_compute numerical validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py

Edge cases: positive, zeros, small, large, minimum_input
(Sqrt requires non-negative input.)
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.data_compute import compute_sqrt
from tests.test_ops.utils import generate_test_data

RTOL = 1e-4
ATOL = 1e-5
SEED = 42

sqrt_edge_types = ['positive', 'zeros', 'small', 'large', 'minimum_input']
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


def ensure_nonneg(x):
    return np.abs(x).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", sqrt_edge_types, ids=sqrt_edge_types)
def test_sqrt(data_type):
    """Sqrt numerical validation — non-negative input required."""
    _seed()
    shape = _shape_for(data_type)
    raw = ensure_nonneg(generate_test_data(shape, data_type))
    t = make_sim_tensor(raw, "sqrt_in")

    tt_out = compute_sqrt([t], _FakeOp())
    ref_out = np.sqrt(raw)

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    num_ok = bool(np.allclose(tt_out, ref_out, rtol=RTOL, atol=ATOL))

    if shape_ok and num_ok:
        print(f"Sqrt/{data_type}: PASS")
    else:
        max_diff = float(np.abs(tt_out - ref_out).max())
        print(f"Sqrt/{data_type}: FAIL (max_diff={max_diff:.2e})")

    assert shape_ok, f"Sqrt/{data_type} shape mismatch"
    assert num_ok, f"Sqrt/{data_type} numerical mismatch"


@pytest.mark.unit
@pytest.mark.opunit
def test_sqrt_of_squares():
    """sqrt(x^2) = |x| — for positive inputs."""
    _seed()
    raw = generate_test_data((2, 4, 8), 'positive')
    squared = raw ** 2
    t = make_sim_tensor(squared, "sqrt_sq")
    tt_out = compute_sqrt([t], _FakeOp())
    assert np.allclose(tt_out, raw, rtol=RTOL, atol=ATOL), \
        "sqrt(x^2) != x for positive x"


@pytest.mark.unit
@pytest.mark.opunit
def test_sqrt_preserves_zero():
    """sqrt(0) = 0."""
    raw = np.zeros((4,), dtype=np.float32)
    t = make_sim_tensor(raw, "sqrt_zero")
    tt_out = compute_sqrt([t], _FakeOp())
    assert np.allclose(tt_out, 0.0, atol=ATOL), f"sqrt(0) = {tt_out}"
