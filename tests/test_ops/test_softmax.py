#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim Softmax — data_compute + SimTensor monkey-patch validation.

Ported from:
  - workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py
  - workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.data_compute import compute_softmax
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as tensor_op  # noqa: F401
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


class DummyModule(SimNN.Module):
    def __init__(self, name="DummyModule"):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x


def make_linked_tensor(data, module, name="t"):
    t = SimTensor({
        "name": name,
        "shape": list(data.shape),
        "data": data.copy(),
        "dtype": np.dtype(np.float32),
    })
    t.link_module = module
    if t.name not in module._tensors:
        module._tensors[t.name] = t
    return t


def _shape_for(data_type):
    return (1,) if data_type == 'minimum_input' else base_shape


def ref_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Section 1 — data_compute Softmax
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_softmax(data_type):
    """Softmax numerical validation via compute_softmax."""
    _seed()
    shape = _shape_for(data_type)
    if data_type == 'minimum_input':
        shape = (1, 4)  # softmax needs >1 element along axis
    raw = generate_test_data(shape, data_type)
    t = make_sim_tensor(raw, "sm_in")

    tt_out = compute_softmax([t], _FakeOp(axis=-1))
    ref_out = ref_softmax(raw, axis=-1)

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    num_ok = bool(np.allclose(tt_out, ref_out, rtol=RTOL, atol=ATOL))

    if shape_ok and num_ok:
        print(f"Softmax/{data_type}: PASS")
    else:
        max_diff = float(np.abs(tt_out - ref_out).max())
        print(f"Softmax/{data_type}: FAIL (max_diff={max_diff:.2e})")

    assert shape_ok, f"Softmax/{data_type} shape mismatch"
    assert num_ok, f"Softmax/{data_type} numerical mismatch"


# ---------------------------------------------------------------------------
# Section 2 — SimTensor.softmax() monkey-patch
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_tensor_softmax(data_type):
    """SimTensor.softmax() — monkey-patched softmax unary op."""
    _seed()
    mod = DummyModule("softmax_test")
    shape = (1, 4) if data_type == 'minimum_input' else (2, 3, 4)
    raw = generate_test_data(shape, data_type)
    t = make_linked_tensor(raw, mod, "sm_in")
    out = t.softmax()

    ref = ref_softmax(raw)

    shape_ok = list(out.shape) == list(ref.shape)
    num_ok = False
    if out.data is not None:
        num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL))

    assert shape_ok, f"tensor_softmax/{data_type} shape failed"
    if out.data is not None:
        assert num_ok, f"tensor_softmax/{data_type} numerical failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_softmax_sums_to_one():
    """Softmax output should sum to 1 along last axis."""
    _seed()
    raw = generate_test_data((2, 4, 8), 'mixed')
    t = make_sim_tensor(raw, "sm_sum")
    tt_out = compute_softmax([t], _FakeOp(axis=-1))
    sums = tt_out.sum(axis=-1)
    assert np.allclose(sums, 1.0, rtol=RTOL, atol=ATOL), \
        "Softmax does not sum to 1"


@pytest.mark.unit
@pytest.mark.opunit
def test_softmax_non_negative():
    """Softmax output should be non-negative."""
    _seed()
    raw = generate_test_data((2, 4, 8), 'mixed')
    t = make_sim_tensor(raw, "sm_nn")
    tt_out = compute_softmax([t], _FakeOp(axis=-1))
    assert np.all(tt_out >= 0.0), "Softmax produced negative values"


@pytest.mark.unit
@pytest.mark.opunit
def test_softmax_uniform_input():
    """Softmax of uniform input → uniform output (1/N)."""
    raw = np.ones((2, 8), dtype=np.float32) * 3.0
    t = make_sim_tensor(raw, "sm_uniform")
    tt_out = compute_softmax([t], _FakeOp(axis=-1))
    expected = np.full_like(raw, 1.0 / 8)
    assert np.allclose(tt_out, expected, rtol=RTOL, atol=ATOL), \
        "Softmax of uniform input not uniform"
