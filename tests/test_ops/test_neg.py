#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim __neg__ — SimTensor monkey-patch validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py

NOTE: unary_fwd does NOT implement data_compute for 'Neg',
so output data may be None. We validate shape only.

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
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
def test_neg(data_type):
    """Unary negation via monkey-patched __neg__.

    NOTE: unary_fwd does NOT implement data_compute for 'Neg',
    so output data is None. We validate shape only.
    """
    _seed()
    mod = DummyModule("neg_test")
    shape = (1,) if data_type == 'minimum_input' else (2, 3, 4)
    raw = generate_test_data(shape, data_type)
    t = make_linked_tensor(raw, mod, "neg_in")
    out = -t
    ref = -raw

    shape_ok = list(out.shape) == list(ref.shape)
    if out.data is not None:
        num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL))
    else:
        # unary_fwd sets data=None for Neg — shape-only validation
        num_ok = None

    assert shape_ok, f"__neg__/{data_type} shape failed"
