#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim Resize — SimOpHandle pipeline validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py

F.Resize → resize_sinf → compute_resize (nearest neighbor upsampling).

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.op as F
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


def ref_resize_nearest(x, scale):
    """NumPy nearest-neighbor upsampling reference."""
    N, C, H, W = x.shape
    if isinstance(scale, (list, tuple)):
        scale_h, scale_w = scale[0], scale[1]
    else:
        scale_h = scale_w = float(scale)
    H_out = int(H * scale_h)
    W_out = int(W * scale_w)
    ref = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    for h in range(H_out):
        for w in range(W_out):
            src_h = min(int(h / scale_h), H - 1)
            src_w = min(int(w / scale_w), W - 1)
            ref[:, :, h, w] = x[:, :, src_h, src_w]
    return ref


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_resize(data_type):
    """F.Resize through SimOpHandle — shape + numerical (nearest neighbor)."""
    _seed()
    if data_type == 'minimum_input':
        N, C, H, W = 1, 1, 2, 2
        scale = 2.0
    else:
        N, C, H, W = 1, 2, 4, 4
        scale = 2.0

    resize_op = F.Resize(f"test_resize_{data_type}", scale_factor=scale)

    x_data = generate_test_data((N, C, H, W), data_type)
    x_tensor = make_sim_tensor(x_data, f"resize_x_{data_type}")
    out = resize_op(x_tensor)

    ref = ref_resize_nearest(x_data, scale)
    ref_shape = list(ref.shape)

    shape_ok = list(out.shape) == ref_shape
    assert shape_ok, f"Resize/{data_type} shape failed"
    if out.data is not None:
        num_ok = bool(np.allclose(out.data, ref, rtol=RTOL, atol=ATOL))
        assert num_ok, f"Resize/{data_type} numerical failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_resize_scale_list():
    """Resize with scale_factor=[2.0, 3.0] — asymmetric scaling."""
    _seed()
    N, C, H, W = 1, 2, 3, 4
    scale_h, scale_w = 2.0, 3.0
    resize_op = F.Resize("test_resize_list", scale_factor=[scale_h, scale_w])

    x_data = generate_test_data((N, C, H, W), 'positive')
    x_tensor = make_sim_tensor(x_data, "resize_list_x")
    out = resize_op(x_tensor)

    ref_shape = [N, C, int(H * scale_h), int(W * scale_w)]
    shape_ok = list(out.shape) == ref_shape
    assert shape_ok, "Resize/scale_list shape failed"


@pytest.mark.unit
@pytest.mark.opunit
def test_resize_scale_1x():
    """Resize with scale=1.0 — identity."""
    _seed()
    N, C, H, W = 1, 2, 4, 4
    resize_op = F.Resize("test_resize_1x", scale_factor=1.0)
    x_data = generate_test_data((N, C, H, W), 'positive')
    x_tensor = make_sim_tensor(x_data, "resize_1x_x")
    out = resize_op(x_tensor)
    shape_ok = list(out.shape) == [N, C, H, W]
    assert shape_ok, "Resize/1x shape failed"
    if out.data is not None:
        assert np.allclose(out.data, x_data, rtol=RTOL, atol=ATOL), \
            "Resize/1x not identity"
