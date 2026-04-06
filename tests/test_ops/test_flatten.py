#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim flatten — SimTensor.flatten validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from loguru import logger
from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as tensor_op  # noqa: F401
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
def test_flatten(data_type):
    """SimTensor.flatten(start_dim, end_dim) — collapses dimension range."""
    _seed()
    mod = DummyModule("flat_test")
    if data_type == 'minimum_input':
        shape = (1, 1, 1)
        start_dim, end_dim = 0, 2
    else:
        shape = (2, 3, 4)
        start_dim, end_dim = 1, 2

    raw = generate_test_data(shape, data_type)
    t = make_linked_tensor(raw, mod, "flat_in")
    out = t.flatten(start_dim, end_dim)

    # NumPy reference
    flat_size = 1
    for d in shape[start_dim:end_dim + 1]:
        flat_size *= d
    ref_shape = list(shape[:start_dim]) + [flat_size] + list(shape[end_dim + 1:])
    ref = raw.reshape(ref_shape)

    shape_ok = list(out.shape) == list(ref.shape)
    num_ok = False
    if out.data is not None:
        num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL))

    if shape_ok and (num_ok or out.data is None):
        logger.debug(f"flatten/{data_type}: PASS")
    else:
        logger.debug(f"flatten/{data_type}: FAIL")

    assert shape_ok, f"flatten/{data_type} shape failed"
    if out.data is not None:
        assert num_ok, f"flatten/{data_type} numerical failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_flatten_default_end():
    """flatten(start_dim=1) defaults end_dim=-1 (last)."""
    _seed()
    mod = DummyModule("flat_def")
    raw = generate_test_data((2, 3, 4), 'mixed')
    t = make_linked_tensor(raw, mod, "flat_def_in")
    out = t.flatten(1)
    ref = raw.reshape(2, -1)
    shape_ok = list(out.shape) == list(ref.shape)
    assert shape_ok, "flatten default end shape failed"
    if out.data is not None:
        num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL))
        assert num_ok, "flatten default end numerical failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_flatten_invalid_range_raises():
    """flatten with start_dim > end_dim should raise ValueError."""
    mod = DummyModule("fl_err")
    raw = generate_test_data((2, 3, 4), 'positive')
    t = make_linked_tensor(raw, mod, "flerr_in")
    with pytest.raises(ValueError, match="Invalid"):
        t.flatten(2, 0)
