#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim Sub — data_compute + SimTensor monkey-patch validation.

Ported from:
  - workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py
  - workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.data_compute import compute_sub
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as tensor_op  # noqa: F401
from tests.test_ops.utils import generate_test_data

RTOL = 1e-4
ATOL = 1e-5
SEED = 42

edge_types = ['positive', 'negative', 'zeros', 'mixed', 'small', 'large', 'minimum_input']
binary_shape = (2, 4, 8)


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


# ---------------------------------------------------------------------------
# Section 1 — data_compute Sub
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_sub(data_type):
    """Sub numerical validation via compute_sub."""
    _seed()
    sa = (1,) if data_type == 'minimum_input' else binary_shape
    sb = (1,) if data_type == 'minimum_input' else binary_shape
    raw_a = generate_test_data(sa, data_type)
    raw_b = generate_test_data(sb, data_type)
    ta = make_sim_tensor(raw_a, "sub_a")
    tb = make_sim_tensor(raw_b, "sub_b")

    tt_out = compute_sub([ta, tb], _FakeOp())
    ref_out = raw_a - raw_b

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    num_ok = bool(np.allclose(tt_out, ref_out, rtol=RTOL, atol=ATOL))

    if shape_ok and num_ok:
        print(f"Sub/{data_type}: PASS")
    else:
        max_diff = float(np.abs(tt_out - ref_out).max())
        print(f"Sub/{data_type}: FAIL (max_diff={max_diff:.2e})")

    assert shape_ok, f"Sub/{data_type} shape mismatch"
    assert num_ok, f"Sub/{data_type} numerical mismatch"


# ---------------------------------------------------------------------------
# Section 2 — SimTensor __sub__ monkey-patch
# ---------------------------------------------------------------------------

arith_edge_types = ['positive', 'negative', 'zeros', 'mixed', 'small', 'large', 'minimum_input']


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", arith_edge_types, ids=arith_edge_types)
def test_tensor_sub(data_type):
    """SimTensor.__sub__ monkey-patch — a - b."""
    _seed()
    mod = DummyModule("sub_test")
    sa = (1,) if data_type == 'minimum_input' else (2, 3, 4)
    sb = (1,) if data_type == 'minimum_input' else (2, 3, 4)
    raw_a = generate_test_data(sa, data_type)
    raw_b = generate_test_data(sb, data_type)
    ta = make_linked_tensor(raw_a, mod, "sub_a")
    tb = make_linked_tensor(raw_b, mod, "sub_b")

    out = ta - tb
    ref = raw_a - raw_b

    shape_ok = list(out.shape) == list(ref.shape)
    num_ok = False
    if out.data is not None:
        num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL))

    assert shape_ok, f"tensor_sub/{data_type} shape failed"
    if out.data is not None:
        assert num_ok, f"tensor_sub/{data_type} numerical failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_sub_self_is_zero():
    """a - a = 0."""
    _seed()
    raw = generate_test_data((2, 4, 8), 'mixed')
    ta = make_sim_tensor(raw, "sub_self_a")
    tb = make_sim_tensor(raw, "sub_self_b")
    tt_out = compute_sub([ta, tb], _FakeOp())
    assert np.allclose(tt_out, 0.0, atol=ATOL), "a - a != 0"


@pytest.mark.unit
@pytest.mark.opunit
def test_sub_identity():
    """a - 0 = a."""
    _seed()
    raw = generate_test_data((2, 4, 8), 'positive')
    zeros = np.zeros_like(raw)
    ta = make_sim_tensor(raw, "sub_id_a")
    tb = make_sim_tensor(zeros, "sub_id_b")
    tt_out = compute_sub([ta, tb], _FakeOp())
    assert np.allclose(tt_out, raw, rtol=RTOL, atol=ATOL), "a - 0 != a"
