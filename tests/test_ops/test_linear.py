#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim Linear — SimNN.Linear module numerical validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py

Tests SimNN.Linear forward: y = x @ W^T + b

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.sim_nn as SimNN
from tests.test_ops.utils import generate_test_data

RTOL = 1e-4
ATOL = 1e-5
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
def test_linear(data_type):
    """SimNN.Linear forward: y = x @ W^T + b — numerical validation."""
    _seed()
    if data_type == 'minimum_input':
        batch, in_f, out_f = 1, 1, 1
    else:
        batch, in_f, out_f = 2, 8, 4

    x_data = generate_test_data((batch, in_f), data_type)
    w_data = generate_test_data((out_f, in_f), data_type) * 0.01
    b_data = generate_test_data((out_f,), data_type) * 0.01

    linear = SimNN.Linear("test_linear", in_f, out_f, bias=True)
    linear.param.data = w_data.copy()
    linear.bias.data = b_data.copy()

    x_sim = make_sim_tensor(x_data, "lin_x")
    x_sim.set_module(linear)
    linear._tensors[x_sim.name] = x_sim
    tt_out = linear(x_sim)

    # Reference: y = x @ W^T + b
    ref_out = x_data @ w_data.T + b_data

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    if tt_out.data is not None:
        num_ok = bool(np.allclose(tt_out.data, ref_out, rtol=RTOL, atol=ATOL))
    else:
        num_ok = None

    if shape_ok and num_ok:
        print(f"Linear/{data_type}: PASS")
    else:
        print(f"Linear/{data_type}: shape_ok={shape_ok}, num_ok={num_ok}")

    assert shape_ok, f"Linear/{data_type} shape mismatch"
    if num_ok is not None:
        assert num_ok, f"Linear/{data_type} numerical mismatch"


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_linear_no_bias(data_type):
    """SimNN.Linear without bias: y = x @ W^T."""
    _seed()
    if data_type == 'minimum_input':
        batch, in_f, out_f = 1, 1, 1
    else:
        batch, in_f, out_f = 2, 8, 4

    x_data = generate_test_data((batch, in_f), data_type)
    w_data = generate_test_data((out_f, in_f), data_type) * 0.01

    linear = SimNN.Linear("test_linear_nb", in_f, out_f, bias=False)
    linear.param.data = w_data.copy()

    x_sim = make_sim_tensor(x_data, "lin_nb_x")
    x_sim.set_module(linear)
    linear._tensors[x_sim.name] = x_sim
    tt_out = linear(x_sim)

    ref_out = x_data @ w_data.T

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    assert shape_ok, f"Linear_no_bias/{data_type} shape mismatch"
    if tt_out.data is not None:
        num_ok = bool(np.allclose(tt_out.data, ref_out, rtol=RTOL, atol=ATOL))
        assert num_ok, f"Linear_no_bias/{data_type} numerical mismatch"
