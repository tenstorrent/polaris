#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim InverseSigmoid — data_compute numerical validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py

Edge cases: positive, small, mixed, minimum_input
(InverseSigmoid requires input in (0, 1).)
"""

import pytest
import numpy as np
from loguru import logger
from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.data_compute import compute_inverse_sigmoid
from tests.test_ops.utils import generate_test_data

RTOL = 1e-4
ATOL = 1e-5
SEED = 42

inverse_sigmoid_edge_types = ['positive', 'small', 'mixed', 'minimum_input']
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


def to_01(x):
    """Map arbitrary data into (eps, 1-eps) via sigmoid."""
    return np.clip(1.0 / (1.0 + np.exp(-x)), 1e-5, 1 - 1e-5).astype(np.float32)


def ref_inverse_sigmoid(x):
    eps = 1e-5
    x_c = np.clip(x, eps, 1 - eps)
    return np.log(x_c / (1 - x_c))


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", inverse_sigmoid_edge_types,
                         ids=inverse_sigmoid_edge_types)
def test_inverse_sigmoid(data_type):
    """InverseSigmoid numerical validation — input clamped to (0, 1)."""
    _seed()
    shape = _shape_for(data_type)
    raw = to_01(generate_test_data(shape, data_type))
    t = make_sim_tensor(raw, "invsig_in")

    tt_out = compute_inverse_sigmoid([t], _FakeOp())
    ref_out = ref_inverse_sigmoid(raw)

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    num_ok = bool(np.allclose(tt_out, ref_out, rtol=RTOL, atol=ATOL))

    if shape_ok and num_ok:
        logger.debug(f"InverseSigmoid/{data_type}: PASS")
    else:
        max_diff = float(np.abs(tt_out - ref_out).max())
        logger.debug(f"InverseSigmoid/{data_type}: FAIL (max_diff={max_diff:.2e})")

    assert shape_ok, f"InverseSigmoid/{data_type} shape mismatch"
    assert num_ok, f"InverseSigmoid/{data_type} numerical mismatch"


@pytest.mark.unit
@pytest.mark.opunit
def test_inverse_sigmoid_roundtrip():
    """inverse_sigmoid(sigmoid(x)) ≈ x for moderate values."""
    _seed()
    raw = generate_test_data((2, 4, 8), 'mixed')
    sig = 1.0 / (1.0 + np.exp(-np.clip(raw, -10, 10)))
    sig_clamped = np.clip(sig, 1e-5, 1 - 1e-5).astype(np.float32)
    t = make_sim_tensor(sig_clamped, "invsig_rt")
    tt_out = compute_inverse_sigmoid([t], _FakeOp())
    clipped_raw = np.clip(raw, -10, 10)
    assert np.allclose(tt_out, clipped_raw, rtol=1e-3, atol=1e-3), \
        "InverseSigmoid roundtrip failed"
