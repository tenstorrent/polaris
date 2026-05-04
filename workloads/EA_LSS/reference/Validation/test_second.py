#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for SECOND backbone TTSim module.
Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_second.py

Weight injection:
  - Conv2d: stage._tensors['{name}.conv{i}.param'].data = w  [out,in,3,3]
  - BN:     stage._tensors['{name}.bn{i}.scale/bias/input_mean/input_var'].data
  - BN eps = 1e-3 (SECOND default); PyTorch: nn.BatchNorm2d(out_ch, eps=1e-3)
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn

_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ttsim.front.functional.op import _from_shape, _from_data
from ttsim_modules.second import SECOND, SECONDStage
from reference.Validation.ttsim_utils import (
    print_header, print_test, compare_arrays, ttsim_conv2d, ttsim_relu,
    batchnorm1d_numpy,
)

BN_EPS = 1e-3  # SECOND default BN epsilon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def inject_stage_weights(stage, name, in_ch, out_ch, layer_num, rng):
    """Inject known random weights into all convs and BN params of a SECONDStage."""
    weights = {}
    # conv0 (strided)
    w0 = (rng.randn(out_ch, in_ch, 3, 3) * 0.1).astype(np.float32)
    weights['conv0'] = w0
    stage._tensors[f'{name}.conv0.param'].data = w0
    weights[f'bn0_scale'] = np.ones(out_ch, dtype=np.float32)
    weights[f'bn0_bias']  = np.zeros(out_ch, dtype=np.float32)
    weights[f'bn0_mean']  = np.zeros(out_ch, dtype=np.float32)
    weights[f'bn0_var']   = np.ones(out_ch, dtype=np.float32)
    stage._tensors[f'{name}.bn0.scale'].data      = weights['bn0_scale']
    stage._tensors[f'{name}.bn0.bias'].data       = weights['bn0_bias']
    stage._tensors[f'{name}.bn0.input_mean'].data = weights['bn0_mean']
    stage._tensors[f'{name}.bn0.input_var'].data  = weights['bn0_var']
    # extra layers
    for j in range(layer_num):
        wj = (rng.randn(out_ch, out_ch, 3, 3) * 0.1).astype(np.float32)
        weights[f'conv{j+1}'] = wj
        stage._tensors[f'{name}.conv{j+1}.param'].data = wj
        weights[f'bn{j+1}_scale'] = np.ones(out_ch, dtype=np.float32)
        weights[f'bn{j+1}_bias']  = np.zeros(out_ch, dtype=np.float32)
        weights[f'bn{j+1}_mean']  = np.zeros(out_ch, dtype=np.float32)
        weights[f'bn{j+1}_var']   = np.ones(out_ch, dtype=np.float32)
        stage._tensors[f'{name}.bn{j+1}.scale'].data      = weights[f'bn{j+1}_scale']
        stage._tensors[f'{name}.bn{j+1}.bias'].data       = weights[f'bn{j+1}_bias']
        stage._tensors[f'{name}.bn{j+1}.input_mean'].data = weights[f'bn{j+1}_mean']
        stage._tensors[f'{name}.bn{j+1}.input_var'].data  = weights[f'bn{j+1}_var']
    return weights


def pt_stage_forward(x_np, in_ch, out_ch, stride, layer_num, weights):
    """PyTorch reference for SECONDStage: strided Conv→BN→ReLU then layer_num repeats."""
    x = torch.tensor(x_np)
    # conv0 (strided)
    conv0 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
    conv0.weight.data = torch.tensor(weights['conv0'])
    bn0 = nn.BatchNorm2d(out_ch, eps=BN_EPS, momentum=0.01)
    bn0.weight.data = torch.tensor(weights['bn0_scale'])
    bn0.bias.data   = torch.tensor(weights['bn0_bias'])
    bn0.running_mean.data = torch.tensor(weights['bn0_mean'])
    bn0.running_var.data  = torch.tensor(weights['bn0_var'])
    bn0.eval()
    with torch.no_grad():
        x = torch.relu(bn0(conv0(x)))
    # extra layers
    for j in range(layer_num):
        conv_j = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        conv_j.weight.data = torch.tensor(weights[f'conv{j+1}'])
        bn_j = nn.BatchNorm2d(out_ch, eps=BN_EPS, momentum=0.01)
        bn_j.weight.data = torch.tensor(weights[f'bn{j+1}_scale'])
        bn_j.bias.data   = torch.tensor(weights[f'bn{j+1}_bias'])
        bn_j.running_mean.data = torch.tensor(weights[f'bn{j+1}_mean'])
        bn_j.running_var.data  = torch.tensor(weights[f'bn{j+1}_var'])
        bn_j.eval()
        with torch.no_grad():
            x = torch.relu(bn_j(conv_j(x)))
    return x.numpy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_construction():
    print_header("Test 1: Construction (default config)")
    second = SECOND("sec1", in_channels=128, out_channels=[128,128,256],
                    layer_nums=[3,5,5], layer_strides=[2,2,2])
    p = second.analytical_param_count()
    ok = p > 0
    print_test("SECOND default params > 0", f"got {p:,}")
    return ok


def test_conv2d_step():
    """Single Conv2d step comparison: TTSim ttsim_conv2d vs PyTorch Conv2d."""
    print_header("Test 2: Conv2d step — PyTorch vs TTSim")
    rng = np.random.RandomState(10)
    in_ch, out_ch = 16, 32
    x_np = (rng.randn(1, in_ch, 8, 8) * 0.5).astype(np.float32)
    conv_w = (rng.randn(out_ch, in_ch, 3, 3) * 0.1).astype(np.float32)

    # PyTorch
    conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
    conv.weight.data = torch.tensor(conv_w)
    with torch.no_grad():
        pt_conv_out = conv(torch.tensor(x_np)).numpy()
    print(f"  PyTorch Conv2d: shape={list(pt_conv_out.shape)}, "
          f"sample={pt_conv_out.flatten()[:4]}")

    # TTSim
    ts_conv_out = ttsim_conv2d(x_np, conv_w, bias=None, stride=2, padding=1)
    print(f"  TTSim  Conv2d: shape={list(ts_conv_out.shape)}, "
          f"sample={ts_conv_out.flatten()[:4]}")

    ok = compare_arrays(pt_conv_out, ts_conv_out, "Conv2d step")
    return ok


def test_bn_step():
    """BatchNorm2d step comparison: PyTorch vs TTSim batchnorm."""
    print_header("Test 3: BatchNorm2d step — PyTorch vs TTSim")
    rng = np.random.RandomState(11)
    C = 32
    x_np = (rng.randn(1, C, 4, 4) * 0.5).astype(np.float32)
    scale = (rng.randn(C) * 0.1 + 1.0).astype(np.float32)
    bias_bn = (rng.randn(C) * 0.1).astype(np.float32)
    rm = (rng.randn(C) * 0.1).astype(np.float32)
    rv = (np.abs(rng.randn(C)) + 0.5).astype(np.float32)

    # PyTorch (inference mode)
    bn = nn.BatchNorm2d(C, eps=BN_EPS, momentum=0.01)
    bn.weight.data = torch.tensor(scale)
    bn.bias.data   = torch.tensor(bias_bn)
    bn.running_mean.data = torch.tensor(rm)
    bn.running_var.data  = torch.tensor(rv)
    bn.eval()
    with torch.no_grad():
        pt_bn_out = bn(torch.tensor(x_np)).numpy()
    print(f"  PyTorch BN2d: shape={list(pt_bn_out.shape)}, "
          f"sample={pt_bn_out.flatten()[:4]}")

    # TTSim: BN2d inference is the same formula as BN1d for 4-D input
    # Use numpy formula: (x - mean) / sqrt(var + eps) * scale + bias
    mu    = rm[np.newaxis, :, np.newaxis, np.newaxis]
    sigma = np.sqrt(rv[np.newaxis, :, np.newaxis, np.newaxis] + BN_EPS)
    gamma = scale[np.newaxis, :, np.newaxis, np.newaxis]
    beta  = bias_bn[np.newaxis, :, np.newaxis, np.newaxis]
    ts_bn_out = (x_np - mu) / sigma * gamma + beta
    print(f"  TTSim  BN2d: shape={list(ts_bn_out.shape)}, "
          f"sample={ts_bn_out.flatten()[:4]}")

    ok = compare_arrays(pt_bn_out, ts_bn_out, "BN2d step", atol=1e-5)
    return ok


def test_stage_conv_bn_relu():
    """Full Conv2d→BN→ReLU in one SECONDStage (layer_num=0)."""
    print_header("Test 4: SECONDStage (Conv→BN→ReLU) — PyTorch vs TTSim")
    rng = np.random.RandomState(12)
    in_ch, out_ch = 16, 32
    x_np = (rng.randn(1, in_ch, 8, 8) * 0.5).astype(np.float32)

    stage = SECONDStage('s4', in_ch, out_ch, stride=2, layer_num=0, bias=False)
    weights = inject_stage_weights(stage, 's4', in_ch, out_ch, layer_num=0, rng=rng)

    # PyTorch reference
    pt_out = pt_stage_forward(x_np, in_ch, out_ch, stride=2, layer_num=0, weights=weights)
    print(f"  PyTorch stage: shape={list(pt_out.shape)}, "
          f"sample={pt_out.flatten()[:4]}")

    # TTSim
    x_t = _from_data('x4', x_np)
    ts_out = stage(x_t)
    print(f"  TTSim  stage: shape={list(ts_out.shape)}, "
          f"sample={ts_out.data.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out.data, "SECONDStage Conv→BN→ReLU", atol=1e-5)
    return ok


def test_stage_with_extra_layers():
    """SECONDStage with layer_num=2 (3 conv blocks total)."""
    print_header("Test 5: SECONDStage (layer_num=2) — PyTorch vs TTSim")
    rng = np.random.RandomState(13)
    in_ch, out_ch = 16, 32
    x_np = (rng.randn(1, in_ch, 16, 16) * 0.5).astype(np.float32)

    stage = SECONDStage('s5', in_ch, out_ch, stride=2, layer_num=2, bias=False)
    weights = inject_stage_weights(stage, 's5', in_ch, out_ch, layer_num=2, rng=rng)

    pt_out = pt_stage_forward(x_np, in_ch, out_ch, stride=2, layer_num=2, weights=weights)
    print(f"  PyTorch stage(ln=2): shape={list(pt_out.shape)}, "
          f"sample={pt_out.flatten()[:4]}")

    x_t = _from_data('x5', x_np)
    ts_out = stage(x_t)
    print(f"  TTSim  stage(ln=2): shape={list(ts_out.shape)}, "
          f"sample={ts_out.data.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out.data, "SECONDStage ln=2", atol=1e-5)
    return ok


def test_output_shapes_default():
    print_header("Test 6: Output shapes (default SECOND config)")
    second = SECOND("sec6", in_channels=128, out_channels=[128,128,256],
                    layer_nums=[3,5,5], layer_strides=[2,2,2])
    x = _from_shape("x6", [2, 128, 256, 256])
    outs = second(x)
    expected = [[2,128,128,128],[2,128,64,64],[2,256,32,32]]
    ok = True
    for i, (o, e) in enumerate(zip(outs, expected)):
        match = list(o.shape) == e
        print_test(f"Stage {i} shape", f"got {list(o.shape)} expected {e}")
        if not match: ok = False
    return ok


def test_param_count_manual():
    print_header("Test 7: Parameter count verification")
    # layer_num=0: 1 Conv(3×3,no bias) + 1 BN(scale+bias)
    stage = SECONDStage("ts7", 32, 64, stride=2, layer_num=0, bias=False)
    expected = 9*32*64 + 2*64  # conv weights + BN params
    got = stage.analytical_param_count()
    ok = got == expected
    print_test("SECONDStage(32→64, ln=0)", f"got {got:,}, expected {expected:,}")
    return ok

if __name__ == "__main__":
    tests = [
        ("construction",              test_construction),
        ("conv2d_step",               test_conv2d_step),
        ("bn_step",                   test_bn_step),
        ("stage_conv_bn_relu",        test_stage_conv_bn_relu),
        ("stage_with_extra_layers",   test_stage_with_extra_layers),
        ("output_shapes_default",     test_output_shapes_default),
        ("param_count_manual",        test_param_count_manual),
    ]
    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    passed = sum(results.values())
    total  = len(results)
    for n, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
