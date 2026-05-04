#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for FFN detection head TTSim module.

Three test categories:
  1. Shape Validation  – each head output is [B, num_classes, P].
  2. Edge Case Creation – single head, deep head (num_conv=3), various head sizes.
  3. Data Validation   – param count per-head formula and data-carrying inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_ffn.py -v
"""

import os, sys, logging
import numpy as np
import pytest

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)
_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.ffn import FFN

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(23)


def _head_params(in_ch, head_conv, num_classes, num_conv):
    p, c_in = 0, in_ch
    for _ in range(num_conv - 1):
        p += c_in * head_conv + 2 * head_conv
        c_in = head_conv
    p += head_conv * num_classes + num_classes
    return p


# TransFusion-like full head config
_TRANSFUSION_HEADS = {
    "center":  (2,  2),
    "height":  (1,  2),
    "dim":     (3,  2),
    "rot":     (2,  2),
    "vel":     (2,  2),
    "heatmap": (10, 2),
}


@pytest.mark.unit
@pytest.mark.opunit
def test_ffn_shape_validation():
    """Category 1 – each head produces [B, num_classes, P]."""
    all_passed = True
    ffn = FFN("ffn_sv", in_channels=128, heads=_TRANSFUSION_HEADS, head_conv=64)
    cases = [
        ([2, 128, 200], "B=2 P=200"),
        ([1, 128, 100], "B=1 P=100"),
        ([4, 128,  50], "B=4 P=50"),
    ]
    for i, (inp_shape, label) in enumerate(cases):
        try:
            B, _, P = inp_shape
            x    = _from_shape(f"ffn_sv_{i}_in", inp_shape)
            outs = ffn(x)
            ok   = all(
                list(outs[hn].shape) == [B, nc, P]
                for hn, (nc, _) in _TRANSFUSION_HEADS.items()
            )
            print(f"  [{i}] {label:15s} {'PASS' if ok else 'FAIL'}  "
                  f"shapes={{'center':{list(outs['center'].shape)}, ...}}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i}] {label:15s} ERROR: {exc}")
            all_passed = False
    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_ffn_edge_cases():
    """Category 2 – edge cases."""
    all_passed = True

    # Edge 1: single head
    ffn1 = FFN("ffn_ec1", in_channels=64, heads={"cls": (5, 2)}, head_conv=32)
    x1   = _from_shape("ffn_ec1_in", [2, 64, 100])
    outs1 = ffn1(x1)
    ok1   = isinstance(outs1, dict) and set(outs1.keys()) == {"cls"}
    ok1   = ok1 and list(outs1["cls"].shape) == [2, 5, 100]
    print(f"  Single head 'cls': {list(outs1['cls'].shape)}  {'PASS' if ok1 else 'FAIL'}")
    all_passed = all_passed and ok1

    # Edge 2: num_conv=3 (deep)
    ffn2  = FFN("ffn_ec2", in_channels=128, heads={"deep": (4, 3)}, head_conv=64)
    x2    = _from_shape("ffn_ec2_in", [1, 128, 50])
    outs2 = ffn2(x2)
    ok2   = list(outs2["deep"].shape) == [1, 4, 50]
    print(f"  num_conv=3: {list(outs2['deep'].shape)}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    # Edge 3: many heads with different num_classes
    multi_heads = {f"h{k}": (k + 1, 2) for k in range(5)}
    ffn3  = FFN("ffn_ec3", in_channels=64, heads=multi_heads, head_conv=32)
    x3    = _from_shape("ffn_ec3_in", [2, 64, 100])
    outs3 = ffn3(x3)
    ok3   = all(list(outs3[f"h{k}"].shape) == [2, k + 1, 100] for k in range(5))
    print(f"  5 heads correct: {'PASS' if ok3 else 'FAIL'}")
    all_passed = all_passed and ok3

    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_ffn_data_validation():
    """Category 3 – param count formula and data-carrying inputs."""
    all_passed = True

    # Param count per head
    in_ch, head_conv = 128, 64
    ffn = FFN("ffn_dv_pc", in_channels=in_ch, heads=_TRANSFUSION_HEADS, head_conv=head_conv)
    expected = sum(_head_params(in_ch, head_conv, nc, nv)
                   for nc, nv in _TRANSFUSION_HEADS.values())
    got   = ffn.analytical_param_count()
    ok    = got == expected
    print(f"  TransFusion-like params: got={got:,} exp={expected:,}  {'PASS' if ok else 'FAIL'}")
    all_passed = all_passed and ok

    # Data input
    ffn2   = FFN("ffn_dv_data", in_channels=64, heads={"center": (2, 2)}, head_conv=32)
    x_np   = rng.randn(2, 64, 100).astype(np.float32)
    outs   = ffn2(_from_data("ffn_dv_in", x_np))
    ok2    = list(outs["center"].shape) == [2, 2, 100]
    print(f"  Data input shape: {list(outs['center'].shape)}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    assert all_passed
