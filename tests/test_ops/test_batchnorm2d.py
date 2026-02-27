#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim batchnorm2d (v1) — F.BatchNorm2d composite pipeline (shape-only).

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py

F.BatchNorm2d(name, C): 4 params (scale, bias, input_mean, input_var).
  Shape-only validation (data can be None).

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.op as F
from tests.test_ops.utils import generate_test_data

SEED = 42

edge_types = ['positive', 'negative', 'zeros', 'mixed', 'small', 'large', 'minimum_input']


def _seed():
    np.random.seed(SEED)


def make_sim_tensor(data, name="t"):
    return SimTensor({
        "name": name,
        "shape": list(data.shape),
        "data": data.copy(),
        "dtype": np.dtype(np.float32),
    })


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_batchnorm2d_shape(data_type):
    """F.BatchNorm2d — output should preserve input shape."""
    _seed()
    C = 8

    if data_type == 'minimum_input':
        x_shape = (1, C, 1, 1)
    else:
        x_shape = (2, C, 4, 4)

    x_data = generate_test_data(x_shape, data_type)
    x_sim = make_sim_tensor(x_data, "bn_x")

    handle = F.BatchNorm2d("test_bn", C)
    out = handle(x_sim)

    assert list(out.shape) == list(x_shape), f"batchnorm2d/{data_type} shape"


@pytest.mark.unit
@pytest.mark.opunit
def test_batchnorm2d_param_count():
    """F.BatchNorm2d should have 4 params (scale, bias, input_mean, input_var)."""
    C = 16
    handle = F.BatchNorm2d("bn_pc", C)
    assert len(handle.params) == 4, "BatchNorm2d should have 4 params"


@pytest.mark.unit
@pytest.mark.opunit
def test_batchnorm2d_param_shapes():
    """Each BatchNorm2d param should be shaped (C,)."""
    C = 8
    handle = F.BatchNorm2d("bn_ps", C)
    for pname, ptensor in handle.params:
        assert list(ptensor.shape) == [C], f"BatchNorm2d param '{pname}' shape"
