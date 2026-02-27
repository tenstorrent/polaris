#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim GroupNorm — data_compute, SimOpHandle, and SimNN module.

Ported from:
  - workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py
  - workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py
  - workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.data_compute import compute_groupnorm
import ttsim.front.functional.op as F
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


class _FakeOp:
    def __init__(self, **attrs):
        self.attrs = attrs


def ref_groupnorm(x, scale, bias, num_groups, eps=1e-5):
    """NumPy reference GroupNorm."""
    N, C, H, W = x.shape
    G = num_groups
    C_per_g = C // G
    x_g = x.reshape(N, G, C_per_g, H, W)
    axes = tuple(range(2, x_g.ndim))
    mean = x_g.mean(axis=axes, keepdims=True)
    var = x_g.var(axis=axes, keepdims=True)
    x_norm = (x_g - mean) / np.sqrt(var + eps)
    return x_norm.reshape(N, C, H, W) * scale.reshape(1, C, 1, 1) + bias.reshape(1, C, 1, 1)


# ---------------------------------------------------------------------------
# Section 1 — data_compute GroupNorm
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_groupnorm(data_type):
    """GroupNorm via compute_groupnorm — numerical validation."""
    _seed()
    if data_type == 'minimum_input':
        N, C, H, W = 1, 4, 1, 1
    else:
        N, C, H, W = 1, 8, 4, 4
    num_groups = 4 if C >= 4 else 1

    raw = generate_test_data((N, C, H, W), data_type)
    scale = np.ones(C, dtype=np.float32)
    bias = np.zeros(C, dtype=np.float32)
    t_x = make_sim_tensor(raw, "gn_x")
    t_s = make_sim_tensor(scale, "gn_s")
    t_b = make_sim_tensor(bias, "gn_b")

    tt_out = compute_groupnorm([t_x, t_s, t_b], _FakeOp(num_groups=num_groups, epsilon=1e-5))
    ref_out = ref_groupnorm(raw, scale, bias, num_groups)

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    num_ok = bool(np.allclose(tt_out, ref_out, rtol=RTOL, atol=ATOL))

    if shape_ok and num_ok:
        print(f"GroupNorm/{data_type}: PASS")
    else:
        max_diff = float(np.abs(tt_out - ref_out).max())
        print(f"GroupNorm/{data_type}: FAIL (max_diff={max_diff:.2e})")

    assert shape_ok, f"GroupNorm/{data_type} shape mismatch"
    assert num_ok, f"GroupNorm/{data_type} numerical mismatch"


# ---------------------------------------------------------------------------
# Section 2 — SimOpHandle GroupNorm (F.GroupNormalization)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_groupnorm_pipeline(data_type):
    """F.GroupNormalization through SimOpHandle — shape + numerical."""
    _seed()
    if data_type == 'minimum_input':
        N, C, H, W = 1, 4, 1, 1
    else:
        N, C, H, W = 1, 8, 4, 4
    num_groups = 4 if C >= 4 else 1

    gn_op = F.GroupNormalization(f"test_gn_{data_type}",
                                num_groups=num_groups, epsilon=1e-5)

    x_data = generate_test_data((N, C, H, W), data_type)
    scale = np.ones(C, dtype=np.float32)
    bias = np.zeros(C, dtype=np.float32)

    x_tensor = make_sim_tensor(x_data, f"gn_x_{data_type}")
    s_tensor = make_sim_tensor(scale, f"gn_s_{data_type}")
    b_tensor = make_sim_tensor(bias, f"gn_b_{data_type}")

    out = gn_op(x_tensor, s_tensor, b_tensor)
    ref = ref_groupnorm(x_data, scale, bias, num_groups)

    shape_ok = list(out.shape) == list(ref.shape)
    assert shape_ok, f"GroupNorm pipeline/{data_type} shape failed"
    if out.data is not None:
        num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL))
        assert num_ok, f"GroupNorm pipeline/{data_type} numerical failed"


# ---------------------------------------------------------------------------
# Section 3 — SimNN.GroupNorm module
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_groupnorm_module(data_type):
    """SimNN.GroupNorm module — numerical validation."""
    _seed()
    if data_type == 'minimum_input':
        N, C, H, W = 1, 4, 1, 1
    else:
        N, C, H, W = 1, 8, 4, 4
    num_groups = 4 if C >= 4 else 1

    x_data = generate_test_data((N, C, H, W), data_type)
    scale = np.ones(C, dtype=np.float32)
    bias = np.zeros(C, dtype=np.float32)

    gn = SimNN.GroupNorm("test_gn", num_groups=num_groups, num_channels=C)
    gn.weight.data = scale.copy()
    gn.bias.data = bias.copy()

    x_sim = make_sim_tensor(x_data, "gn_x")
    x_sim.set_module(gn)
    gn._tensors[x_sim.name] = x_sim
    tt_out = gn(x_sim)

    ref_out = ref_groupnorm(x_data, scale, bias, num_groups)

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    assert shape_ok, f"GroupNorm module/{data_type} shape failed"
    if tt_out.data is not None:
        num_ok = bool(np.allclose(tt_out.data, ref_out, rtol=RTOL, atol=ATOL))
        assert num_ok, f"GroupNorm module/{data_type} numerical failed"
