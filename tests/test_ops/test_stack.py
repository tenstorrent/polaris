#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim stack — tensor_op.stack validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py

tensor_op.stack(simtensor_list, dim) — stacks tensors along a NEW dimension.

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as tensor_op  # noqa: F401
from ttsim.front.functional.tensor_op import stack
from tests.test_ops.utils import generate_test_data

RTOL = 1e-4
ATOL = 1e-5
SEED = 42

edge_types = ['positive', 'negative', 'zeros', 'mixed', 'small', 'large', 'minimum_input']


def _seed():
    np.random.seed(SEED)


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


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_stack(data_type):
    """tensor_op.stack(simtensor_list, dim) — stacks along a NEW dimension."""
    _seed()
    mod = DummyModule("stack_test")
    if data_type == 'minimum_input':
        shape = (1,)
        dim = 0
        n_tensors = 2
    else:
        shape = (2, 3, 4)
        dim = 1
        n_tensors = 3

    raws = [generate_test_data(shape, data_type) for _ in range(n_tensors)]
    ts = [make_linked_tensor(r, mod, f"stk_{i}") for i, r in enumerate(raws)]

    out = stack(ts, dim=dim)
    ref = np.stack(raws, axis=dim)

    shape_ok = list(out.shape) == list(ref.shape)
    num_ok = False
    if out.data is not None:
        num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL))

    if shape_ok and (num_ok or out.data is None):
        print(f"stack/{data_type}: PASS")
    else:
        print(f"stack/{data_type}: FAIL")

    assert shape_ok, f"stack/{data_type} shape failed"
    if out.data is not None:
        assert num_ok, f"stack/{data_type} numerical failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_stack_dim0():
    """Stack along dim=0 (default)."""
    _seed()
    mod = DummyModule("stk_d0")
    raws = [generate_test_data((3, 4), 'positive') for _ in range(2)]
    ts = [make_linked_tensor(r, mod, f"stk0_{i}") for i, r in enumerate(raws)]
    out = stack(ts, dim=0)
    ref = np.stack(raws, axis=0)
    shape_ok = list(out.shape) == list(ref.shape)
    num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL)) if out.data is not None else True
    assert shape_ok and num_ok, "stack dim=0 failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_stack_dim_neg1():
    """Stack along dim=-1 (last)."""
    _seed()
    mod = DummyModule("stk_dn1")
    raws = [generate_test_data((2, 3), 'mixed') for _ in range(4)]
    ts = [make_linked_tensor(r, mod, f"stkn1_{i}") for i, r in enumerate(raws)]
    out = stack(ts, dim=-1)
    ref = np.stack(raws, axis=-1)
    shape_ok = list(out.shape) == list(ref.shape)
    num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL)) if out.data is not None else True
    assert shape_ok and num_ok, "stack dim=-1 failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_stack_empty_raises():
    """stack with empty list should raise ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        stack([], dim=0)


@pytest.mark.unit
@pytest.mark.opunit
def test_stack_shape_mismatch_raises():
    """stack with differing shapes should raise ValueError."""
    mod = DummyModule("stk_err")
    t1 = make_linked_tensor(np.zeros((2, 3), dtype=np.float32), mod, "s1")
    t2 = make_linked_tensor(np.zeros((3, 4), dtype=np.float32), mod, "s2")
    with pytest.raises(ValueError):
        stack([t1, t2], dim=0)
