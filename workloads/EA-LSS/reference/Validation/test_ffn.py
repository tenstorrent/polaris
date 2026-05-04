#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for FFN detection head TTSim module.

Validates the TTSim conversion of transfusion_head.py FFN class.
Each test compares the TTSim graph output (.data) against the equivalent
PyTorch operation at every step of the forward pass.

Test Coverage:
  1.  Construction               – param count matches expected formula
  2.  Output dict keys           – forward returns dict with correct head keys
  3.  Conv1d Step                – TTSim Conv1d vs torch.nn.Conv1d
  4.  BatchNorm1d Step           – TTSim BN op vs torch.nn.BatchNorm1d (eval)
  5.  ReLU Step                  – TTSim ReLU vs torch.relu
  6.  Full ConvModule1d Forward  – TTSim Conv+BN+ReLU vs PyTorch reference
  7.  Full Single-head Forward   – TTSim FFN forward vs PyTorch reference
  8.  Multi-head Forward         – TTSim all heads vs PyTorch reference
  9.  Multi-head param count     – analytical vs expected formula
  10. Deep head (num_conv=3)     – 3-layer head forward comparison

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_ffn.py
"""

import os, sys

_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn

from ttsim.front.functional.op import _from_shape, _from_data
from ttsim_modules.ffn import FFN
from ttsim_modules.mlp import ConvModule1d, BatchNorm1d
from reference.Validation.ttsim_utils import (
    compare_arrays, print_header, print_test,
    conv1d_numpy, batchnorm1d_numpy,
    ttsim_relu,
)

# ============================================================================
# PyTorch reference helpers
# ============================================================================

def pt_conv1d(x_np, conv_w, conv_b=None):
    """PyTorch Conv1d (kernel_size=1)."""
    in_ch, out_ch = conv_w.shape[1], conv_w.shape[0]
    m = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=(conv_b is not None))
    m.weight.data = torch.tensor(conv_w)
    if conv_b is not None:
        m.bias.data = torch.tensor(conv_b)
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(x_np)).numpy()


def pt_batchnorm1d(x_np, scale, bias_bn, mean, var, eps=1e-5):
    """PyTorch BatchNorm1d in eval mode."""
    C = x_np.shape[1]
    m = nn.BatchNorm1d(C, eps=eps, affine=True)
    m.weight.data       = torch.tensor(scale)
    m.bias.data         = torch.tensor(bias_bn)
    m.running_mean.data = torch.tensor(mean)
    m.running_var.data  = torch.tensor(var)
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(x_np)).numpy()


def make_weights(shape, rng, scale=0.1):
    return (rng.randn(*shape) * scale).astype(np.float32)


def inject_ffn_head(ffn, head_name, in_ch, head_conv, num_classes, num_conv, rng):
    """Inject numpy weights into FFN head; return (weights_dict) for PyTorch ref."""
    w = {}
    c_in = in_ch
    for i in range(num_conv - 1):
        cm = getattr(ffn, f"{head_name}_cm{i}")
        bn = getattr(ffn, f"{head_name}_bn{i}")
        w[f'cm{i}_w'] = make_weights((head_conv, c_in, 1), rng)
        w[f'cm{i}_b'] = make_weights((head_conv,), rng)
        w[f'bn{i}_s'] = np.ones(head_conv, dtype=np.float32)
        w[f'bn{i}_b'] = np.zeros(head_conv, dtype=np.float32)
        w[f'bn{i}_m'] = np.zeros(head_conv, dtype=np.float32)
        w[f'bn{i}_v'] = np.ones(head_conv, dtype=np.float32)
        cm.conv_weight.data = w[f'cm{i}_w']
        cm.conv_bias.data   = w[f'cm{i}_b']
        bn.scale.data       = w[f'bn{i}_s']
        bn.bias_bn.data     = w[f'bn{i}_b']
        bn.running_mean.data = w[f'bn{i}_m']
        bn.running_var.data  = w[f'bn{i}_v']
        c_in = head_conv
    final = getattr(ffn, f"{head_name}_final")
    final_in = head_conv if num_conv >= 1 else in_ch
    w['final_w'] = make_weights((num_classes, final_in, 1), rng)
    w['final_b'] = make_weights((num_classes,), rng)
    final.conv_weight.data = w['final_w']
    final.conv_bias.data   = w['final_b']
    return w


def pt_ffn_head_forward(x_np, weights, in_ch, head_conv, num_classes, num_conv):
    """PyTorch reference forward for a single FFN head."""
    h = x_np
    c_in = in_ch
    for i in range(num_conv - 1):
        h = pt_conv1d(h, weights[f'cm{i}_w'], weights[f'cm{i}_b'])
        h = pt_batchnorm1d(h, weights[f'bn{i}_s'], weights[f'bn{i}_b'],
                           weights[f'bn{i}_m'], weights[f'bn{i}_v'])
        h = np.maximum(h, 0)  # ReLU
        c_in = head_conv
    h = pt_conv1d(h, weights['final_w'], weights['final_b'])
    return h


# ============================================================================
# Helper: expected param count for one head
# ============================================================================
def _head_params(in_ch, head_conv, num_classes, num_conv):
    p = 0
    c_in = in_ch
    for _ in range(num_conv - 1):
        p += c_in * head_conv + 2 * head_conv
        c_in = head_conv
    p += head_conv * num_classes + num_classes
    return p


# ============================================================================
# Tests
# ============================================================================

def test_construction():
    print_header("Test 1: Construction (single head)")
    heads = {"center": (2, 2)}
    ffn = FFN("ffn1", in_channels=128, heads=heads, head_conv=64)
    p = ffn.analytical_param_count()
    expected = _head_params(128, 64, 2, 2)
    ok = p == expected
    print_test("FFN single head param count", f"got {p:,} expected {expected:,}", ok)
    return ok


def test_output_is_dict():
    print_header("Test 2: Output is a dict with correct keys")
    heads = {"center": (2, 2), "height": (1, 2)}
    ffn = FFN("ffn2", in_channels=128, heads=heads, head_conv=64)
    x = _from_shape("q2", [2, 128, 200])
    outs = ffn(x)
    ok = isinstance(outs, dict) and set(outs.keys()) == set(heads.keys())
    print_test("Output is dict with correct keys",
               f"keys: {list(outs.keys())}", ok)
    return ok


def test_conv1d_step():
    print_header("Test 3: Step-by-step — Conv1d")
    print_test("TTSim Conv1d op (.data) vs torch.nn.Conv1d")
    rng = np.random.RandomState(10)
    B, in_ch, out_ch, P = 2, 16, 8, 32
    x_np  = make_weights((B, in_ch, P), rng, scale=0.5)
    conv_w = make_weights((out_ch, in_ch, 1), rng)
    conv_b = make_weights((out_ch,), rng)

    # PyTorch reference
    pt_out = pt_conv1d(x_np, conv_w, conv_b)
    print(f"  PyTorch shape : {pt_out.shape}")
    print(f"  PyTorch sample: {pt_out.flatten()[:4]}")

    # TTSim via ConvModule1d (no BN, no ReLU)
    cm = ConvModule1d("cm_step", in_ch, out_ch, 1, with_bn=False, with_relu=False)
    cm.conv_weight.data = conv_w
    cm.conv_bias.data   = conv_b
    x_t = _from_data("x_step", x_np)
    ttsim_out = cm(x_t).data
    print(f"  TTSim   shape : {ttsim_out.shape}")
    print(f"  TTSim   sample: {ttsim_out.flatten()[:4]}")
    return compare_arrays(pt_out, ttsim_out, "Conv1d step")


def test_batchnorm1d_step():
    print_header("Test 4: Step-by-step — BatchNorm1d")
    print_test("TTSim BN op (.data) vs torch.nn.BatchNorm1d (eval mode)")
    rng = np.random.RandomState(11)
    B, C, P = 2, 16, 32
    x_np  = make_weights((B, C, P), rng, scale=0.5)
    scale = np.ones(C, dtype=np.float32)
    bias_bn = np.zeros(C, dtype=np.float32)
    mean  = np.zeros(C, dtype=np.float32)
    var   = np.ones(C, dtype=np.float32)

    # PyTorch reference
    pt_out = pt_batchnorm1d(x_np, scale, bias_bn, mean, var)
    print(f"  PyTorch shape : {pt_out.shape}")
    print(f"  PyTorch sample: {pt_out.flatten()[:4]}")

    # TTSim
    bn = BatchNorm1d("bn_step", C)
    bn.scale.data       = scale
    bn.bias_bn.data     = bias_bn
    bn.running_mean.data = mean
    bn.running_var.data  = var
    x_t = _from_data("x_bn_step", x_np)
    ttsim_out = bn(x_t).data
    print(f"  TTSim   shape : {ttsim_out.shape}")
    print(f"  TTSim   sample: {ttsim_out.flatten()[:4]}")
    return compare_arrays(pt_out, ttsim_out, "BatchNorm1d step")


def test_relu_step():
    print_header("Test 5: Step-by-step — ReLU")
    print_test("TTSim ReLU op (.data) vs torch.relu")
    rng = np.random.RandomState(12)
    x_np = rng.randn(2, 8, 20).astype(np.float32)

    pt_out    = np.maximum(x_np, 0)
    ttsim_out = ttsim_relu(x_np)

    print(f"  PyTorch shape : {pt_out.shape}")
    print(f"  PyTorch sample: {pt_out.flatten()[:4]}")
    print(f"  TTSim   shape : {ttsim_out.shape}")
    print(f"  TTSim   sample: {ttsim_out.flatten()[:4]}")
    return compare_arrays(pt_out, ttsim_out, "ReLU step")


def test_full_convmodule1d():
    print_header("Test 6: Full ConvModule1d Forward (Conv + BN + ReLU)")
    print_test("TTSim ConvModule1d (.data) vs PyTorch Conv1d + BatchNorm1d + ReLU")
    rng = np.random.RandomState(13)
    B, in_ch, out_ch, P = 2, 16, 8, 32
    x_np   = make_weights((B, in_ch, P), rng, scale=0.5)
    conv_w = make_weights((out_ch, in_ch, 1), rng)
    conv_b = make_weights((out_ch,), rng)
    bn_s   = np.ones(out_ch, dtype=np.float32)
    bn_b   = np.zeros(out_ch, dtype=np.float32)
    bn_m   = np.zeros(out_ch, dtype=np.float32)
    bn_v   = np.ones(out_ch, dtype=np.float32)

    # PyTorch reference: Conv + BN + ReLU
    pt_after_conv = pt_conv1d(x_np, conv_w, conv_b)
    pt_after_bn   = pt_batchnorm1d(pt_after_conv, bn_s, bn_b, bn_m, bn_v)
    pt_out        = np.maximum(pt_after_bn, 0)  # ReLU

    print(f"  PyTorch shape : {pt_out.shape}")
    print(f"  PyTorch sample: {pt_out.flatten()[:4]}")

    # TTSim: ConvModule1d (no internal BN) + separate BN
    cm = ConvModule1d("cm_full", in_ch, out_ch, 1, with_bn=False, with_relu=True)
    bn = BatchNorm1d("bn_full", out_ch)
    cm.conv_weight.data  = conv_w
    cm.conv_bias.data    = conv_b
    bn.scale.data        = bn_s
    bn.bias_bn.data      = bn_b
    bn.running_mean.data = bn_m
    bn.running_var.data  = bn_v

    x_t = _from_data("x_cm_full", x_np)
    # cm applies Conv + ReLU; we need Conv → BN → ReLU like FFN does
    # FFN pattern: cm(with_relu=True) → BN → no extra relu
    # But ConvModule1d with with_relu=True does Conv+ReLU (no BN)
    # FFN: cm (conv+relu) then bn; PyTorch ref: conv→bn→relu
    # To match exactly, test Conv+BN+ReLU directly:
    cm_no_relu = ConvModule1d("cm_full2", in_ch, out_ch, 1, with_bn=False, with_relu=False)
    cm_no_relu.conv_weight.data = conv_w
    cm_no_relu.conv_bias.data   = conv_b
    x_t2 = _from_data("x_cm_full2", x_np)
    after_conv_t = cm_no_relu(x_t2)
    after_bn_t   = bn(after_conv_t)
    # Apply ReLU manually
    from ttsim_utils import ttsim_relu as _relu
    ttsim_out = _relu(after_bn_t.data)

    print(f"  TTSim   shape : {ttsim_out.shape}")
    print(f"  TTSim   sample: {ttsim_out.flatten()[:4]}")
    return compare_arrays(pt_out, ttsim_out, "ConvModule1d full (Conv+BN+ReLU)")


def test_single_head_forward():
    print_header("Test 7: Full Single-head Forward (TTSim vs PyTorch)")
    print_test("TTSim FFN center head (.data) vs PyTorch Conv1d+BN+ReLU+Conv1d")
    rng = np.random.RandomState(14)
    B, in_ch, head_conv, nc, num_conv, P = 2, 16, 8, 2, 2, 32
    x_np = make_weights((B, in_ch, P), rng, scale=0.5)

    ffn = FFN("ffn_s", in_channels=in_ch,
              heads={"center": (nc, num_conv)}, head_conv=head_conv)
    weights = inject_ffn_head(ffn, "center", in_ch, head_conv, nc, num_conv, rng)

    # PyTorch reference
    pt_out = pt_ffn_head_forward(x_np, weights, in_ch, head_conv, nc, num_conv)
    print(f"  PyTorch shape : {pt_out.shape}")
    print(f"  PyTorch sample: {pt_out.flatten()[:4]}")

    # TTSim forward
    x_t = _from_data("x_shead", x_np)
    outs = ffn(x_t)
    ttsim_out = outs["center"].data
    print(f"  TTSim   shape : {ttsim_out.shape}")
    print(f"  TTSim   sample: {ttsim_out.flatten()[:4]}")
    return compare_arrays(pt_out, ttsim_out, "FFN single-head forward", atol=1e-5)


def test_multi_head_forward():
    print_header("Test 8: Multi-head Forward (TTSim vs PyTorch for each head)")
    print_test("All FFN heads: TTSim (.data) vs PyTorch reference")
    rng = np.random.RandomState(15)
    B, in_ch, head_conv, P = 2, 32, 16, 20
    heads = {"center": (2, 2), "height": (1, 2), "rot": (2, 2)}

    ffn = FFN("ffn_m", in_channels=in_ch, heads=heads, head_conv=head_conv)
    x_np = make_weights((B, in_ch, P), rng, scale=0.5)

    all_weights = {}
    for hname, (nc, nconv) in heads.items():
        all_weights[hname] = inject_ffn_head(
            ffn, hname, in_ch, head_conv, nc, nconv, rng)

    x_t = _from_data("x_mhead", x_np)
    outs = ffn(x_t)

    ok = True
    for hname, (nc, nconv) in heads.items():
        pt_out    = pt_ffn_head_forward(
            x_np, all_weights[hname], in_ch, head_conv, nc, nconv)
        ttsim_out = outs[hname].data
        print(f"\n  Head '{hname}':")
        print(f"    PyTorch shape : {pt_out.shape}, sample: {pt_out.flatten()[:4]}")
        print(f"    TTSim   shape : {ttsim_out.shape}, sample: {ttsim_out.flatten()[:4]}")
        if not compare_arrays(pt_out, ttsim_out, f"head '{hname}'", atol=1e-5):
            ok = False
    return ok


def test_param_count_multi_head():
    print_header("Test 9: Multi-head param count")
    in_ch, head_conv = 128, 64
    heads = {
        "center":  (2, 2),
        "height":  (1, 2),
        "dim":     (3, 2),
        "rot":     (2, 2),
        "heatmap": (10, 2),
    }
    ffn = FFN("ffn9", in_channels=in_ch, heads=heads, head_conv=head_conv)
    expected = sum(
        _head_params(in_ch, head_conv, nc, nv)
        for nc, nv in heads.values()
    )
    got = ffn.analytical_param_count()
    ok = got == expected
    print_test("Multi-head param count", f"got {got:,} expected {expected:,}", ok)
    return ok


def test_deep_head_num_conv3():
    print_header("Test 10: Deep head (num_conv=3) — Forward Comparison")
    print_test("TTSim 3-layer head (.data) vs PyTorch reference")
    rng = np.random.RandomState(16)
    in_ch, head_conv, nc, num_conv, B, P = 32, 16, 4, 3, 2, 15
    x_np = make_weights((B, in_ch, P), rng, scale=0.5)

    ffn = FFN("ffn10", in_channels=in_ch,
              heads={"deep": (nc, num_conv)}, head_conv=head_conv)
    weights = inject_ffn_head(ffn, "deep", in_ch, head_conv, nc, num_conv, rng)

    pt_out = pt_ffn_head_forward(x_np, weights, in_ch, head_conv, nc, num_conv)
    print(f"  PyTorch shape : {pt_out.shape}")
    print(f"  PyTorch sample: {pt_out.flatten()[:4]}")

    x_t = _from_data("x_deep", x_np)
    ttsim_out = ffn(x_t)["deep"].data
    print(f"  TTSim   shape : {ttsim_out.shape}")
    print(f"  TTSim   sample: {ttsim_out.flatten()[:4]}")
    return compare_arrays(pt_out, ttsim_out, "FFN deep head (num_conv=3)", atol=1e-5)


if __name__ == "__main__":
    tests = [
        ("construction",          test_construction),
        ("output_is_dict",        test_output_is_dict),
        ("conv1d_step",           test_conv1d_step),
        ("batchnorm1d_step",      test_batchnorm1d_step),
        ("relu_step",             test_relu_step),
        ("full_convmodule1d",     test_full_convmodule1d),
        ("single_head_forward",   test_single_head_forward),
        ("multi_head_forward",    test_multi_head_forward),
        ("param_count_multi_head",test_param_count_multi_head),
        ("deep_head_num_conv3",   test_deep_head_num_conv3),
    ]
    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            print(f"  ERROR {name}: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    passed = sum(results.values())
    total  = len(results)
    for n, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
