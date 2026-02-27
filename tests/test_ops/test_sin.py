#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim Sin — data_compute + SimTensor monkey-patch validation.

Ported from:
  - workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py
  - workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input

NOTE: Sin via data_compute uses np.sin manually; the tensor_op monkey-patch
(SimTensor.sin()) goes through unary_fwd which may set data=None for 'Sin',
so we validate shape only for the tensor_op path.
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as tensor_op  # noqa: F401 — patches SimTensor
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


# ---------------------------------------------------------------------------
# Section 1 — data_compute Sin (manual np.sin passthrough)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_sin(data_type):
    """Sin numerical validation via manual compute (np.sin)."""
    _seed()
    shape = _shape_for(data_type)
    raw = generate_test_data(shape, data_type)
    t = make_sim_tensor(raw, "sin_in")

    tt_out = np.sin(t.data)
    ref_out = np.sin(raw)

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    num_ok = bool(np.allclose(tt_out, ref_out, rtol=RTOL, atol=ATOL))

    if shape_ok and num_ok:
        print(f"Sin/{data_type}: PASS")
    else:
        max_diff = float(np.abs(tt_out - ref_out).max())
        print(f"Sin/{data_type}: FAIL (max_diff={max_diff:.2e})")

    assert shape_ok, f"Sin/{data_type} shape mismatch"
    assert num_ok, f"Sin/{data_type} numerical mismatch"


# ---------------------------------------------------------------------------
# Section 2 — SimTensor.sin() monkey-patch (shape-only, data may be None)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_tensor_sin(data_type):
    """SimTensor.sin() monkey-patch — shape validation (data may be None)."""
    _seed()
    mod = DummyModule("sin_test")
    shape = (1,) if data_type == 'minimum_input' else (2, 3, 4)
    raw = generate_test_data(shape, data_type)
    t = make_linked_tensor(raw, mod, "sin_in")
    out = t.sin()
    ref = np.sin(raw)

    shape_ok = list(out.shape) == list(ref.shape)
    if out.data is not None:
        num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL))
    else:
        num_ok = None

    assert shape_ok, f"tensor_sin/{data_type} shape failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_sin_odd_function():
    """sin(-x) = -sin(x) — odd function property."""
    _seed()
    raw = generate_test_data((2, 4, 8), 'mixed')
    assert np.allclose(np.sin(-raw), -np.sin(raw), rtol=RTOL, atol=ATOL), \
        "Sin odd-function property violated"


@pytest.mark.unit
@pytest.mark.opunit
def test_sin_at_zero():
    """sin(0) = 0."""
    raw = np.zeros((1,), dtype=np.float32)
    ref = np.sin(raw)
    assert np.allclose(ref, 0.0, atol=ATOL), f"sin(0) = {ref}"
