#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for voxel_encoder_utils TTSim module.

Three test categories:
  1. Shape Validation  – output shapes for VFELayer (all max_out/cat_max
                         combinations) and DynamicVFELayer.
  2. Edge Case Creation – single voxel, large M, C_in==C_out, deep channels.
  3. Data Validation   – get_paddings_indicator values, VFELayer data vs
                         numpy Conv1d+BN+ReLU, DynamicVFELayer data.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_voxel_encoder_utils.py -v
"""

import os
import sys
import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.voxel_encoder_utils import (
    get_paddings_indicator, VFELayer, DynamicVFELayer
)

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _ll; _ll.remove(); _ll.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(7)

def _rw(*s): return (rng.randn(*s) * 0.1).astype(np.float32)


# ===========================================================================
# Category 1 – Shape Validation
# ===========================================================================

_VFE_SHAPE_CASES = [
    # (name, N, M, C_in, C_out, max_out, cat_max, expected_last_dim)
    ("no_max",              8, 10, 5, 16, False, False, 16),
    ("max_no_cat",          8, 10, 5, 16, True,  False, 16),
    ("max_cat",             8, 10, 5, 16, True,  True,  32),
    ("large_channels",      4, 32, 64, 128, False, False, 128),
    ("single_point",        1, 1,  4,  8,  False, False, 8),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_voxel_encoder_utils_shape_validation():
    """Category 1 – Shape Validation."""
    all_passed = True
    for i, (name, N, M, C_in, C_out, mo, cm, exp_C) in enumerate(_VFE_SHAPE_CASES):
        try:
            vfe = VFELayer(f"vfe_sv{i}", C_in, C_out, max_out=mo, cat_max=cm)
            x_s = _from_shape(f"vfe_sv{i}_in", [N, M, C_in])
            out = vfe(x_s)
            if mo is False:
                ok = list(out.shape) == [N, M, exp_C]
            elif cm is False:
                ok = len(out.shape) == 3 and out.shape[2] == exp_C
            else:
                ok = len(out.shape) == 3 and out.shape[2] == exp_C
            print(f"  [{i:02d}] {name:25s} {'PASS' if ok else 'FAIL'}  shape={out.shape}")
            if not ok:
                all_passed = False
        except Exception as exc:
            print(f"  [{i:02d}] {name:25s} ERROR: {exc}")
            all_passed = False

    # DynamicVFELayer shapes
    dyns = [(100, 10, 32), (1, 3, 8), (500, 64, 128)]
    for i, (M, C_in, C_out) in enumerate(dyns):
        try:
            dvfe = DynamicVFELayer(f"dvfe_sv{i}", C_in, C_out)
            out  = dvfe(_from_shape(f"dvfe_sv{i}_in", [M, C_in]))
            ok   = list(out.shape) == [M, C_out]
            print(f"  [D{i:02d}] DynamicVFE M={M} {'PASS' if ok else 'FAIL'}  shape={out.shape}")
            if not ok:
                all_passed = False
        except Exception as exc:
            print(f"  [D{i:02d}] DynamicVFE M={M} ERROR: {exc}")
            all_passed = False

    assert all_passed


# ===========================================================================
# Category 2 – Edge Cases
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_voxel_encoder_utils_edge_cases():
    """Category 2 – Edge Cases."""
    all_passed = True

    # 1. get_paddings_indicator: all zeros → all False
    mask = get_paddings_indicator(np.zeros(4, dtype=np.int32), max_num=5)
    ok = not mask.any()
    print(f"  [00] all-zero actual_num → all False: {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # 2. get_paddings_indicator: actual >= max → all True
    mask2 = get_paddings_indicator(np.array([5, 10], dtype=np.int32), max_num=5)
    ok2 = mask2.all()
    print(f"  [01] actual>=max → all True: {'PASS' if ok2 else 'FAIL'}")
    if not ok2: all_passed = False

    # 3. VFELayer C_in == C_out
    vfe = VFELayer("vfe_ec0", 8, 8, max_out=False)
    out = vfe(_from_shape("vfe_ec0_in", [4, 12, 8]))
    ok3 = list(out.shape) == [4, 12, 8]
    print(f"  [02] VFE C_in==C_out: {'PASS' if ok3 else 'FAIL'}  shape={out.shape}")
    if not ok3: all_passed = False

    # 4. DynamicVFELayer single-point batch
    dvfe = DynamicVFELayer("dvfe_ec0", 4, 8)
    out  = dvfe(_from_shape("dvfe_ec0_in", [1, 4]))
    ok4  = list(out.shape) == [1, 8]
    print(f"  [03] DynamicVFE single point: {'PASS' if ok4 else 'FAIL'}  shape={out.shape}")
    if not ok4: all_passed = False

    # 5. VFELayer param count > 0
    vfe2 = VFELayer("vfe_ec1", 5, 16, max_out=False)
    pc   = vfe2.conv_module.analytical_param_count()
    ok5  = pc > 0
    print(f"  [04] VFE param_count > 0: {'PASS' if ok5 else 'FAIL'}  pc={pc}")
    if not ok5: all_passed = False

    assert all_passed


# ===========================================================================
# Category 3 – Data Validation
# ===========================================================================

def _pt_conv1d_bn_relu(x_np, w, b, gm, bs, rm, rv, eps=1e-3):
    m = nn.Conv1d(w.shape[1], w.shape[0], 1, bias=True)
    m.weight.data, m.bias.data = torch.tensor(w), torch.tensor(b)
    bn = nn.BatchNorm1d(gm.shape[0], eps=eps, momentum=0.01)
    bn.weight.data, bn.bias.data = torch.tensor(gm), torch.tensor(bs)
    bn.running_mean.data, bn.running_var.data = torch.tensor(rm), torch.tensor(rv)
    bn.eval(); m.eval()
    with torch.no_grad():
        return torch.relu(bn(m(torch.tensor(x_np)))).numpy()


@pytest.mark.unit
@pytest.mark.opunit
def test_voxel_encoder_utils_data_validation():
    """Category 3 – Data Validation."""
    all_passed = True

    # 1. get_paddings_indicator correctness
    actual = np.array([0, 3, 5, 1], dtype=np.int32)
    mask   = get_paddings_indicator(actual, max_num=5)
    expected = np.array([
        [False, False, False, False, False],
        [True,  True,  True,  False, False],
        [True,  True,  True,  True,  True ],
        [True,  False, False, False, False],
    ])
    ok = np.array_equal(mask, expected)
    print(f"  [00] paddings_indicator values: {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # 2. VFELayer no-max data
    N, M, C_in, C_out = 3, 6, 5, 8
    vfe = VFELayer("vfe_dv0", C_in, C_out, max_out=False)
    w = _rw(C_out, C_in, 1); b = _rw(C_out)
    gm = np.ones(C_out, dtype=np.float32); bs = np.zeros(C_out, dtype=np.float32)
    rm = np.zeros(C_out, dtype=np.float32); rv = np.ones(C_out, dtype=np.float32)
    vfe.conv_module.conv_weight.data = w;  vfe.conv_module.conv_bias.data = b
    vfe._bn.scale.data = gm;   vfe._bn.bias_bn.data = bs
    vfe._bn.running_mean.data = rm; vfe._bn.running_var.data = rv

    x_np = _rw(N, M, C_in)
    out  = vfe(_from_data("vfe_dv0_in", x_np))
    assert out.data is not None, "VFE data is None"

    ref = _pt_conv1d_bn_relu(x_np.transpose(0, 2, 1), w, b, gm, bs, rm, rv)  # [N, C_out, M]
    ref_t = ref.transpose(0, 2, 1)  # [N, M, C_out]
    ok2 = np.allclose(ref_t, out.data, atol=1e-4)
    print(f"  [01] VFE no-max data vs torch: {'PASS' if ok2 else 'FAIL'}  max_diff={np.max(np.abs(ref_t-out.data)):.3e}")
    if not ok2: all_passed = False

    # 3. DynamicVFELayer data
    M_d, C_in_d, C_out_d = 20, 6, 12
    dvfe = DynamicVFELayer("dvfe_dv0", C_in_d, C_out_d)
    wd = _rw(C_out_d, C_in_d, 1); bd = _rw(C_out_d)
    gmd = np.ones(C_out_d, dtype=np.float32); bsd = np.zeros(C_out_d, dtype=np.float32)
    rmd = np.zeros(C_out_d, dtype=np.float32); rvd = np.ones(C_out_d, dtype=np.float32)
    dvfe.conv_module.conv_weight.data = wd; dvfe.conv_module.conv_bias.data = bd
    dvfe._bn.scale.data = gmd; dvfe._bn.bias_bn.data = bsd
    dvfe._bn.running_mean.data = rmd; dvfe._bn.running_var.data = rvd

    x_np_d = _rw(M_d, C_in_d)
    out_d  = dvfe(_from_data("dvfe_dv0_in", x_np_d))
    assert out_d.data is not None, "DynamicVFE data is None"

    ref_d = _pt_conv1d_bn_relu(x_np_d.T.reshape(1, C_in_d, M_d),
                                wd, bd, gmd, bsd, rmd, rvd)[0].T
    ok3 = np.allclose(ref_d, out_d.data, atol=1e-4)
    print(f"  [02] DynamicVFE data vs torch: {'PASS' if ok3 else 'FAIL'}  max_diff={np.max(np.abs(ref_d-out_d.data)):.3e}")
    if not ok3: all_passed = False

    assert all_passed
