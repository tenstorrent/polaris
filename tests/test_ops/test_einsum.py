#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TTSim Einsum — data_compute numerical validation.

Ported from: workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py

Tests the subscript 'bqnc,bnchw->bqnhw' used in MSDeformAttn.

Edge cases: positive, negative, zeros, mixed, small, large, minimum_input
"""

import pytest
import numpy as np
from loguru import logger
from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.data_compute import compute_einsum
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


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("data_type", edge_types, ids=edge_types)
def test_einsum(data_type):
    """Einsum 'bqnc,bnchw->bqnhw' — used in MSDeformAttn."""
    _seed()
    if data_type == 'minimum_input':
        B, Q, N, C, H, W = 1, 1, 1, 1, 1, 1
    else:
        B, Q, N, C, H, W = 1, 4, 2, 3, 3, 3

    raw_a = generate_test_data((B, Q, N, C), data_type)
    raw_b = generate_test_data((B, N, C, H, W), data_type)
    ta = make_sim_tensor(raw_a, "ein_a")
    tb = make_sim_tensor(raw_b, "ein_b")

    tt_out = compute_einsum([ta, tb], _FakeOp(subscripts='bqnc,bnchw->bqnhw'))
    ref_out = np.einsum('bqnc,bnchw->bqnhw', raw_a, raw_b)

    shape_ok = list(tt_out.shape) == list(ref_out.shape)
    num_ok = bool(np.allclose(tt_out, ref_out, rtol=RTOL, atol=ATOL))

    if shape_ok and num_ok:
        logger.debug(f"Einsum/{data_type}: PASS")
    else:
        max_diff = float(np.abs(tt_out - ref_out).max())
        logger.debug(f"Einsum/{data_type}: FAIL (max_diff={max_diff:.2e})")

    assert shape_ok, f"Einsum/{data_type} shape mismatch"
    assert num_ok, f"Einsum/{data_type} numerical mismatch"


@pytest.mark.unit
@pytest.mark.opunit
def test_einsum_matmul_equivalence():
    """Einsum 'ij,jk->ik' is equivalent to matmul for 2D matrices."""
    _seed()
    A = generate_test_data((4, 8), 'mixed')
    B = generate_test_data((8, 3), 'mixed')
    ta = make_sim_tensor(A, "ein_mm_a")
    tb = make_sim_tensor(B, "ein_mm_b")
    tt_out = compute_einsum([ta, tb], _FakeOp(subscripts='ij,jk->ik'))
    ref_out = A @ B
    assert np.allclose(tt_out, ref_out, rtol=RTOL, atol=ATOL), \
        "Einsum matmul equivalence failed"
