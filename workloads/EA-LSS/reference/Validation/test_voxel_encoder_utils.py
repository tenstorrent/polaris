#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation tests for voxel_encoder_utils TTSim module.

Validates the TTSim conversion of
mmdet3d/models/voxel_encoders/utils.py.

Test Coverage:
  1.  get_paddings_indicator  – mask shape and values
  2.  VFELayer shape          – output shapes for max_out/cat_max combinations
  3.  VFELayer data (no-max)  – pointwise Linear+BN+ReLU vs numpy reference
  4.  DynamicVFELayer shape   – output [M, C_out]
  5.  DynamicVFELayer data    – Linear+BN+ReLU vs numpy reference
"""

import os
import sys

_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_val_dir      = os.path.dirname(__file__)
for p in (_polaris_root, _ealss_root, _val_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn
from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.voxel_encoder_utils import (
    get_paddings_indicator, VFELayer, DynamicVFELayer
)
from ttsim_utils import compare_arrays, print_header, print_test


# ============================================================================
# Helpers
# ============================================================================

rng = np.random.RandomState(42)

def _rw(shape):
    return (rng.randn(*shape) * 0.1).astype(np.float32)


def _inject_vfe_weights(vfe, C_in, C_out):
    """Inject known weights into VFELayer for data validation."""
    vfe.conv_module.conv_weight.data  = _rw([C_out, C_in, 1])
    vfe.conv_module.conv_bias.data    = _rw([C_out])
    vfe._bn.scale.data        = np.ones(C_out, dtype=np.float32)
    vfe._bn.bias_bn.data      = np.zeros(C_out, dtype=np.float32)
    vfe._bn.running_mean.data = np.zeros(C_out, dtype=np.float32)
    vfe._bn.running_var.data  = np.ones(C_out, dtype=np.float32)


def _pt_conv_bn_relu(x_np, conv_w, conv_b, bn_scale, bn_bias, bn_mean, bn_var):
    """PyTorch Conv1d + BN1d + ReLU reference."""
    m_conv = nn.Conv1d(conv_w.shape[1], conv_w.shape[0], kernel_size=1, bias=True)
    m_conv.weight.data = torch.tensor(conv_w)
    m_conv.bias.data   = torch.tensor(conv_b)
    m_bn = nn.BatchNorm1d(bn_scale.shape[0], eps=1e-3, momentum=0.01)
    m_bn.weight.data       = torch.tensor(bn_scale)
    m_bn.bias.data         = torch.tensor(bn_bias)
    m_bn.running_mean.data = torch.tensor(bn_mean)
    m_bn.running_var.data  = torch.tensor(bn_var)
    m_bn.eval()
    m_conv.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_np)         # [N, C_in, M]
        out  = torch.relu(m_bn(m_conv(x_t)))
    return out.numpy()


# ============================================================================
# Tests
# ============================================================================

def test_paddings_indicator():
    print_header("TEST 1: get_paddings_indicator")
    actual = np.array([3, 5, 0, 1], dtype=np.int32)
    mask = get_paddings_indicator(actual, max_num=5)
    assert mask.shape == (4, 5), f"Expected (4,5), got {mask.shape}"
    assert mask[0].tolist() == [True, True, True, False, False]
    assert mask[1].tolist() == [True] * 5
    assert mask[2].tolist() == [False] * 5
    assert mask[3].tolist() == [True, False, False, False, False]
    print_test("PASS", f"shape={mask.shape} values correct")


def test_vfelayer_shapes():
    print_header("TEST 2: VFELayer output shapes")
    shapes_cfg = [
        # (max_out, cat_max, expected_dim)
        (False, False, 16),
        (True,  False, 16),
        (True,  True,  32),
    ]
    N, M, C_in, C_out = 10, 8, 5, 16
    for (mo, cm, expected_C) in shapes_cfg:
        vfe = VFELayer(f"vfe_mo{mo}_cm{cm}", C_in, C_out,
                       max_out=mo, cat_max=cm)
        x = _from_shape(f"vfe_in_mo{mo}", [N, M, C_in])
        out = vfe(x)
        if mo is False:
            assert out.shape == [N, M, C_out], f"shape={out.shape}"
        elif cm is False:
            assert out.shape[2] == C_out, f"agg shape dim2={out.shape[2]}"
        else:
            assert out.shape[2] in (int(C_out * 2), C_out * 2), \
                f"cat shape dim2={out.shape[2]}"
        print_test("PASS", f"max_out={mo} cat_max={cm} → {out.shape}")


def test_vfelayer_data_no_max():
    print_header("TEST 3: VFELayer data (max_out=False)")
    N, M, C_in, C_out = 4, 6, 8, 16
    vfe = VFELayer("vfe_ptw", C_in, C_out, max_out=False, cat_max=False)
    _inject_vfe_weights(vfe, C_in, C_out)

    pts = _rw([N, M, C_in])
    x = _from_data("vfe_ptw_in", pts)
    out = vfe(x)
    assert out.data is not None, "VFELayer data is None"

    # Reference: transpose → Conv1d+BN+ReLU → transpose back
    pts_t = pts.transpose(0, 2, 1)  # [N, C_in, M]
    ref = _pt_conv_bn_relu(
        pts_t,
        vfe.conv_module.conv_weight.data,
        vfe.conv_module.conv_bias.data,
        vfe._bn.scale.data, vfe._bn.bias_bn.data,
        vfe._bn.running_mean.data, vfe._bn.running_var.data,
    )  # [N, C_out, M]
    ref_t = ref.transpose(0, 2, 1)  # [N, M, C_out]
    compare_arrays(ref_t, out.data, "VFELayer ptw vs torch", rtol=1e-4, atol=1e-4)


def test_dynamic_vfelayer_shapes():
    print_header("TEST 4: DynamicVFELayer shapes")
    M, C_in, C_out = 500, 10, 32
    dvfe = DynamicVFELayer("dvfe", C_in, C_out)
    out = dvfe(_from_shape("dvfe_in", [M, C_in]))
    assert out.shape == [M, C_out], f"Expected [{M},{C_out}], got {out.shape}"
    print_test("PASS", f"shape={out.shape}")


def test_dynamic_vfelayer_data():
    print_header("TEST 5: DynamicVFELayer data")
    M, C_in, C_out = 20, 8, 16
    dvfe = DynamicVFELayer("dvfe_d", C_in, C_out)
    # Inject weights
    dvfe.conv_module.conv_weight.data  = _rw([C_out, C_in, 1])
    dvfe.conv_module.conv_bias.data    = _rw([C_out])
    dvfe._bn.scale.data        = np.ones(C_out, dtype=np.float32)
    dvfe._bn.bias_bn.data      = np.zeros(C_out, dtype=np.float32)
    dvfe._bn.running_mean.data = np.zeros(C_out, dtype=np.float32)
    dvfe._bn.running_var.data  = np.ones(C_out, dtype=np.float32)

    pts = _rw([M, C_in])
    out = dvfe(_from_data("dvfe_d_in", pts))
    assert out.data is not None, "DynamicVFELayer data is None"

    # Reference: treat input as [1, C_in, M] → Conv1d+BN+ReLU
    pts_3d = pts.T.reshape(1, C_in, M)  # [1, C_in, M]
    ref = _pt_conv_bn_relu(
        pts_3d,
        dvfe.conv_module.conv_weight.data,
        dvfe.conv_module.conv_bias.data,
        dvfe._bn.scale.data, dvfe._bn.bias_bn.data,
        dvfe._bn.running_mean.data, dvfe._bn.running_var.data,
    )  # [1, C_out, M]
    ref_2d = ref[0].T   # [M, C_out]
    compare_arrays(ref_2d, out.data, "DynamicVFE vs torch", rtol=1e-4, atol=1e-4)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("paddings_indicator",       test_paddings_indicator),
        ("vfe_shapes",               test_vfelayer_shapes),
        ("vfe_data_no_max",          test_vfelayer_data_no_max),
        ("dynamic_vfe_shapes",       test_dynamic_vfelayer_shapes),
        ("dynamic_vfe_data",         test_dynamic_vfelayer_data),
    ]:
        try:
            fn()
            results[name] = "PASS"
        except Exception as exc:
            results[name] = f"FAIL: {exc}"
            import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v == "PASS")
    for k, v in results.items():
        print(f"  {'✓' if v=='PASS' else '✗'} {k}: {v}")
    print(f"\n  {passed}/{len(results)} tests passed")
    import sys; sys.exit(0 if passed == len(results) else 1)
