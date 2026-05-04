#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for transfusion_bbox_coder TTSim module.

Three test categories:
  1. Shape Validation  – encode_numpy [N, code_size], decode boxes/scores.
  2. Edge Case Creation – single box, N=0, code_size=10 with velocity,
                         large batch, zero rotation.
  3. Data Validation   – encode formulas (x,y,z,log_dims,sin/cos),
                         decode exp(log_dim), center rescaling.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_transfusion_bbox_coder.py -v
"""

import os
import sys
import logging

import numpy as np
import pytest

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.transfusion_bbox_coder import TransFusionBBoxCoder

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _ll; _ll.remove(); _ll.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

PC   = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
OSF  = 8
VSZ  = [0.075, 0.075, 0.2]
rng  = np.random.RandomState(77)


def _coder(code_size=8):
    return TransFusionBBoxCoder("tfc_ut", PC, OSF, VSZ, code_size=code_size)


def _boxes(N=10):
    b = rng.randn(N, 7).astype(np.float32)
    b[:, 3:6] = np.abs(b[:, 3:6]) + 0.5
    return b


# ===========================================================================
# Category 1 – Shape Validation
# ===========================================================================

_SHAPES = [
    # (name, N, code_size, B, num_cls, P)
    ("N=1  code8",   1,  8, 1, 3,   50),
    ("N=20 code8",  20,  8, 2, 10, 200),
    ("N=50 code10", 50, 10, 1, 5,  100),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_transfusion_bbox_coder_shape_validation():
    """Category 1 – Shape Validation."""
    all_passed = True
    for i, (name, N, cs, B, nc, P) in enumerate(_SHAPES):
        try:
            coder = _coder(cs)
            # encode
            tgt = coder.encode_numpy(_boxes(N))
            ok_enc = tgt.shape == (N, cs)
            # decode
            hm   = _from_shape(f"tfc_hm{i}",  [B, nc, P])
            rot  = _from_shape(f"tfc_rot{i}", [B, 2, P])
            dim  = _from_shape(f"tfc_dim{i}", [B, 3, P])
            ctr  = _from_shape(f"tfc_ctr{i}", [B, 2, P])
            hgt  = _from_shape(f"tfc_hgt{i}", [B, 1, P])
            vel  = _from_shape(f"tfc_vel{i}", [B, 2, P]) if cs == 10 else None
            boxes, scores = coder.decode(hm, rot, dim, ctr, hgt, vel)
            exp_C = 9 if (cs == 10 and vel is not None) else 7
            ok_dec = list(boxes.shape) == [B, P, exp_C] and list(scores.shape) == [B, P]
            ok = ok_enc and ok_dec
            print(f"  [{i:02d}] {name:15s} enc={tgt.shape}  dec_boxes={boxes.shape} scores={scores.shape}  {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i:02d}] {name:15s} ERROR: {exc}")
            all_passed = False
    assert all_passed


# ===========================================================================
# Category 2 – Edge Cases
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_transfusion_bbox_coder_edge_cases():
    """Category 2 – Edge Cases."""
    all_passed = True

    # 1. Single box encode
    try:
        coder = _coder()
        tgt = coder.encode_numpy(_boxes(1))
        ok = tgt.shape == (1, 8)
        print(f"  [00] single box encode: {'PASS' if ok else 'FAIL'}  shape={tgt.shape}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  [00] single box encode ERROR: {exc}"); all_passed = False

    # 2. Zero yaw → sin=0, cos=1
    try:
        coder = _coder()
        box = np.array([[0., 0., 0., 1., 1., 1., 0.]], dtype=np.float32)  # yaw=0
        tgt = coder.encode_numpy(box)
        ok2 = np.isclose(tgt[0, 6], 0.0, atol=1e-5) and np.isclose(tgt[0, 7], 1.0, atol=1e-5)
        print(f"  [01] yaw=0 → sin≈0 cos≈1: {'PASS' if ok2 else 'FAIL'}  sin={tgt[0,6]:.4f} cos={tgt[0,7]:.4f}")
        if not ok2: all_passed = False
    except Exception as exc:
        print(f"  [01] ERROR: {exc}"); all_passed = False

    # 3. code_size=10 with velocity decode
    try:
        coder10 = _coder(10)
        B, nc, P = 1, 5, 30
        hm  = _from_shape("tfc_ec2_hm",  [B, nc, P])
        rot = _from_shape("tfc_ec2_rot", [B, 2, P])
        dim = _from_shape("tfc_ec2_dim", [B, 3, P])
        ctr = _from_shape("tfc_ec2_ctr", [B, 2, P])
        hgt = _from_shape("tfc_ec2_hgt", [B, 1, P])
        vel = _from_shape("tfc_ec2_vel", [B, 2, P])
        boxes, scores = coder10.decode(hm, rot, dim, ctr, hgt, vel)
        ok3 = boxes.shape[2] == 9
        print(f"  [02] code_size=10 with vel: {'PASS' if ok3 else 'FAIL'}  boxes_C={boxes.shape[2]}")
        if not ok3: all_passed = False
    except Exception as exc:
        print(f"  [02] ERROR: {exc}"); all_passed = False

    # 4. param count = 0 (no trainable parameters)
    try:
        coder = _coder()
        pc = coder.analytical_param_count()
        ok4 = pc == 0
        print(f"  [03] param_count=0: {'PASS' if ok4 else 'FAIL'}  pc={pc}")
        if not ok4: all_passed = False
    except Exception as exc:
        print(f"  [03] ERROR: {exc}"); all_passed = False

    assert all_passed


# ===========================================================================
# Category 3 – Data Validation
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_transfusion_bbox_coder_data_validation():
    """Category 3 – Data Validation."""
    all_passed = True

    coder = _coder()
    vx, vy = VSZ[0], VSZ[1]
    osf = float(OSF)

    # 1. encode_numpy formulas
    boxes = _boxes(5)
    tgt   = coder.encode_numpy(boxes)

    exp_cx   = (boxes[:, 0] - PC[0]) / (osf * vx)
    exp_cy   = (boxes[:, 1] - PC[1]) / (osf * vy)
    exp_cz   = boxes[:, 2] + boxes[:, 5] * 0.5
    exp_logw = np.log(boxes[:, 3])
    exp_logl = np.log(boxes[:, 4])
    exp_logh = np.log(boxes[:, 5])
    exp_sin  = np.sin(boxes[:, 6])
    exp_cos  = np.cos(boxes[:, 6])

    checks = [
        ("cx",   exp_cx,   tgt[:, 0]),
        ("cy",   exp_cy,   tgt[:, 1]),
        ("cz",   exp_cz,   tgt[:, 2]),
        ("logw", exp_logw, tgt[:, 3]),
        ("logl", exp_logl, tgt[:, 4]),
        ("logh", exp_logh, tgt[:, 5]),
        ("sin",  exp_sin,  tgt[:, 6]),
        ("cos",  exp_cos,  tgt[:, 7]),
    ]
    for colname, ref_col, out_col in checks:
        ok = np.allclose(ref_col, out_col, atol=1e-5)
        print(f"  [ENC-{colname:5s}] {'PASS' if ok else 'FAIL'}  max_diff={np.max(np.abs(ref_col-out_col)):.3e}")
        if not ok: all_passed = False

    # 2. decode dim = exp of log
    B, P = 1, 4
    wlh = np.array([[2.0, 3.0, 1.5]] * P, dtype=np.float32)
    log_dim = np.log(wlh).T.reshape(1, 3, P)  # [1, 3, P]
    hm  = _from_shape("tfc_dv_hm",  [B, 3, P])
    rot = _from_shape("tfc_dv_rot", [B, 2, P])
    dim = _from_data("tfc_dv_dim",   log_dim)
    ctr = _from_shape("tfc_dv_ctr", [B, 2, P])
    hgt = _from_shape("tfc_dv_hgt", [B, 1, P])
    boxes_out, _ = coder.decode(hm, rot, dim, ctr, hgt)
    if boxes_out.data is not None:
        ok_dim = np.allclose(wlh, boxes_out.data[0, :, 3:6], atol=1e-4)
        print(f"  [DEC-dim_exp] {'PASS' if ok_dim else 'FAIL'}  max_diff={np.max(np.abs(wlh-boxes_out.data[0,:,3:6])):.3e}")
        if not ok_dim: all_passed = False

    # 3. decode center rescaling
    cx_g = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    cy_g = np.array([5.0, 15.0, 25.0, 35.0],  dtype=np.float32)
    ctr_np = np.zeros((1, 2, 4), dtype=np.float32)
    ctr_np[0, 0, :] = cx_g; ctr_np[0, 1, :] = cy_g
    boxes_ctr, _ = coder.decode(
        _from_shape("tfc_cr_hm", [1, 3, 4]),
        _from_shape("tfc_cr_rot", [1, 2, 4]),
        _from_shape("tfc_cr_dim", [1, 3, 4]),
        _from_data("tfc_cr_ctr", ctr_np),
        _from_shape("tfc_cr_hgt", [1, 1, 4]),
    )
    if boxes_ctr.data is not None:
        exp_x = cx_g * osf * vx + PC[0]
        exp_y = cy_g * osf * vy + PC[1]
        ok_cx = np.allclose(exp_x, boxes_ctr.data[0, :, 0], atol=1e-4)
        ok_cy = np.allclose(exp_y, boxes_ctr.data[0, :, 1], atol=1e-4)
        print(f"  [DEC-cx_rescale] {'PASS' if ok_cx else 'FAIL'}")
        print(f"  [DEC-cy_rescale] {'PASS' if ok_cy else 'FAIL'}")
        if not (ok_cx and ok_cy): all_passed = False

    assert all_passed
