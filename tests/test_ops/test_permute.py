#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim permute — SimTensor.permute validation.

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
def test_permute(data_type):
    """SimTensor.permute(perm) — arbitrary dimension ordering."""
    _seed()
    mod = DummyModule("permute_test")
    if data_type == 'minimum_input':
        shape = (1, 1, 1)
        perm = [2, 0, 1]
    else:
        shape = (2, 3, 4)
        perm = [2, 0, 1]

    raw = generate_test_data(shape, data_type)
    t = make_linked_tensor(raw, mod, "perm_in")
    out = t.permute(perm)

    ref = np.transpose(raw, axes=perm)
    shape_ok = list(out.shape) == list(ref.shape)
    num_ok = False
    if out.data is not None:
        num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL))

    if shape_ok and (num_ok or out.data is None):
        logger.info(f"permute/{data_type}: PASS")
    else:
        logger.debug(f"permute/{data_type}: FAIL")

    assert shape_ok, f"permute/{data_type} shape failed"
    if out.data is not None:
        assert num_ok, f"permute/{data_type} numerical failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_permute_4d():
    """Permute a 4D tensor (NCHW → NHWC)."""
    _seed()
    mod = DummyModule("perm4d")
    raw = generate_test_data((2, 3, 4, 5), 'mixed')
    t = make_linked_tensor(raw, mod, "perm4d_in")
    perm = [0, 2, 3, 1]  # NCHW → NHWC
    out = t.permute(perm)
    ref = np.transpose(raw, axes=perm)
    shape_ok = list(out.shape) == list(ref.shape)
    assert shape_ok, "permute 4D shape failed"
    if out.data is not None:
        assert np.allclose(out.data, ref, rtol=RTOL, atol=ATOL), \
            "permute 4D numerical failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_permute_identity():
    """Permute with identity perm [0,1,2] should be no-op."""
    _seed()
    mod = DummyModule("perm_id")
    raw = generate_test_data((2, 3, 4), 'positive')
    t = make_linked_tensor(raw, mod, "perm_id_in")
    out = t.permute([0, 1, 2])
    shape_ok = list(out.shape) == list(raw.shape)
    assert shape_ok, "permute identity shape failed"
    if out.data is not None:
        assert np.allclose(out.data, raw, rtol=RTOL, atol=ATOL)
