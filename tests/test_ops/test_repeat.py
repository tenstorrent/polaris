#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim repeat — SimTensor.repeat validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py

SimTensor.repeat(*sizes) tiles tensor (like torch.Tensor.repeat / np.tile).

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
def test_repeat(data_type):
    """SimTensor.repeat(*sizes) — tiles tensor."""
    _seed()
    mod = DummyModule("repeat_test")
    if data_type == 'minimum_input':
        shape = (1,)
        sizes = (3,)
    else:
        shape = (2, 3)
        sizes = (1, 2)

    raw = generate_test_data(shape, data_type)
    t = make_linked_tensor(raw, mod, "rep_in")
    out = t.repeat(*sizes)

    ref = np.tile(raw, sizes)
    shape_ok = list(out.shape) == list(ref.shape)
    num_ok = False
    if out.data is not None:
        num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL))

    if shape_ok and (num_ok or out.data is None):
        logger.info(f"repeat/{data_type}: PASS")
    else:
        logger.debug(f"repeat/{data_type}: FAIL")

    assert shape_ok, f"repeat/{data_type} shape failed"
    if out.data is not None:
        assert num_ok, f"repeat/{data_type} numerical failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_repeat_wrong_ndim_raises():
    """repeat with wrong number of repeat dims should raise ValueError."""
    mod = DummyModule("rep_err")
    raw = generate_test_data((2, 3), 'positive')
    t = make_linked_tensor(raw, mod, "reperr_in")
    with pytest.raises(ValueError, match="expects"):
        t.repeat(1, 2, 3)  # 3 dims for 2D tensor
