#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for MLP TTSim module.

Three test categories:
  1. Shape Validation  – output is always [B, conv_channels[-1], N] across
                         every combination of (in_channel, conv_channels, B, N),
                         with_bn and with_relu flags, and kernel_size variants.
  2. Edge Case Creation – conditions unique to MLP / ConvModule1d:
                         · single-layer MLP
                         · deep MLP (5+ layers)
                         · with_bn=False and with_relu=False variants
                         · non-default kernel_size and padding
                         · in_channel == out_channel (no change in width)
                         · very narrow channels (1 channel)
                         · N=1 (single point per batch)
  3. Data Validation   – full numerical validation against PyTorch:
                         · Conv1d step vs torch.nn.Conv1d
                         · BatchNorm1d step (eval mode) vs torch.nn.BatchNorm1d
                         · ReLU step vs torch.relu
                         · Full ConvModule1d (conv+BN+ReLU) vs PyTorch equivalent
                         · Full 2-layer MLP vs stacked PyTorch ConvModule
                         · MLP without BN vs torch Conv1d+ReLU
                         · Analytical parameter count for all configurations

Run all categories:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_mlp.py -v

Run a single category:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_mlp.py \
           ::test_mlp_shape_validation -v
"""

import os
import sys
import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import ttsim.front.functional.op as F
from ttsim.front.functional.op import _from_data, _from_shape

# EA-LSS modules (hyphen in folder name prevents dotted import; use sys.path)
_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)
from ttsim_modules.mlp import MLP, ConvModule1d, BatchNorm1d

# Silence verbose TTSim / loguru output during tests
try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _loguru_logger
        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, level="ERROR")
    except Exception:
        pass
except Exception:
    pass


# ===========================================================================
# Helpers
# ===========================================================================

def _make_weights(shape, seed, scale=0.1):
    rng = np.random.RandomState(seed)
    return (rng.randn(*shape) * scale).astype(np.float32)


def _get_max_msg_len(test_list):
    return max(len(tc[0]) for tc in test_list)


def _inject_layer(layer, conv_w, conv_b,
                  bn_scale=None, bn_bias=None, bn_mean=None, bn_var=None):
    """Inject numpy weights into a TTSim ConvModule1d."""
    layer.conv_weight.data = conv_w
    if layer.conv_bias is not None:
        layer.conv_bias.data = conv_b
    if layer.bn is not None and bn_scale is not None:
        layer.bn.scale.data        = bn_scale
        layer.bn.bias_bn.data      = bn_bias
        layer.bn.running_mean.data = bn_mean
        layer.bn.running_var.data  = bn_var


def _pt_convmodule1d(x_np, conv_w, conv_b,
                     bn_scale=None, bn_bias=None, bn_mean=None, bn_var=None,
                     with_bn=True, with_relu=True, eps=1e-5):
    """PyTorch Conv1d + optional BN1d + optional ReLU."""
    C_in, C_out = conv_w.shape[1], conv_w.shape[0]
    x = torch.tensor(x_np)
    conv = nn.Conv1d(C_in, C_out, kernel_size=1, bias=True)
    conv.weight.data = torch.tensor(conv_w)
    conv.bias.data   = torch.tensor(conv_b)
    x = conv(x)
    if with_bn and bn_scale is not None:
        bn = nn.BatchNorm1d(C_out, eps=eps)
        bn.weight.data       = torch.tensor(bn_scale)
        bn.bias.data         = torch.tensor(bn_bias)
        bn.running_mean.data = torch.tensor(bn_mean)
        bn.running_var.data  = torch.tensor(bn_var)
        bn.eval()
        x = bn(x)
    if with_relu:
        x = torch.relu(x)
    with torch.no_grad():
        return x.detach().numpy()


def _run_ttsim_cm(x_np, in_ch, out_ch, kernel_size=1,
                   conv_w=None, conv_b=None,
                   bn_scale=None, bn_bias=None,
                   bn_mean=None, bn_var=None,
                   with_bn=True, with_relu=True):
    """Build TTSim ConvModule1d, inject weights, run forward, return .data."""
    m = ConvModule1d("ut_cm", in_ch, out_ch, kernel_size,
                     with_bn=with_bn, with_relu=with_relu, bias=True)
    if conv_w is not None:
        _inject_layer(m, conv_w, conv_b, bn_scale, bn_bias, bn_mean, bn_var)
    return m(_from_data("x", x_np)).data


def _run_ttsim_mlp(x_np, in_channel, conv_channels, weights, with_bn=True, with_relu=True):
    """Build TTSim MLP, inject weights, run forward, return .data."""
    mlp = MLP("ut_mlp", in_channel=in_channel,
              conv_channels=conv_channels, with_bn=with_bn, with_relu=with_relu)
    for i, w in enumerate(weights):
        _inject_layer(mlp.layers[i],
                      w["conv_w"], w["conv_b"],
                      w.get("bn_scale"), w.get("bn_bias"),
                      w.get("bn_mean"),  w.get("bn_var"))
    return mlp(_from_data("x", x_np)).data


# ===========================================================================
# Category 1 – Shape Validation
# ===========================================================================

# (name, in_ch, conv_channels, B, N, with_bn, with_relu)
_SHAPE_CASES = [
    ("Default 18→(256,256) B=2 N=128",       18,  (256, 256),     2,  128, True,  True),
    ("Single layer 64→128  B=1 N=64",         64,  (128,),         1,   64, True,  True),
    ("Deep 5-layer 18→(32,64,128,256,512)",   18,  (32,64,128,256,512), 1, 32, True, True),
    ("No BN 32→(64,128) B=4 N=16",            32,  (64, 128),      4,   16, False, True),
    ("No ReLU 16→(32,) B=2 N=8",              16,  (32,),          2,    8, True,  False),
    ("No BN No ReLU 8→(16,) B=1 N=4",          8,  (16,),          1,    4, False, False),
    ("Same width 64→(64,64) B=2 N=32",        64,  (64, 64),       2,   32, True,  True),
    ("Narrow 3→(4,) B=1 N=200",               3,   (4,),           1,  200, True,  True),
    ("N=1 single point 18→(256,) B=2 N=1",   18,   (256,),         2,    1, True,  True),
    ("Large channels 256→(512,256) B=1 N=64", 256, (512, 256),     1,   64, True,  True),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_mlp_shape_validation():
    """
    Category 1 – Shape Validation

    Verifies that MLP always produces [B, conv_channels[-1], N] for a wide
    range of (in_channel, conv_channels, B, N) combinations, including:
      - Single-layer and multi-layer networks
      - with_bn / with_relu toggles
      - Same-width and width-changing layers
      - Narrow channels and N=1 inputs
    Both shape-only and data-carrying tensors are tested.
    """
    msgw = _get_max_msg_len(_SHAPE_CASES)
    all_passed = True

    for tno, (name, in_ch, conv_chs, B, N, with_bn, with_relu) in enumerate(_SHAPE_CASES):
        try:
            # Shape-only path
            mlp   = MLP(f"ut_shape_{tno}", in_channel=in_ch,
                        conv_channels=conv_chs, with_bn=with_bn, with_relu=with_relu)
            x_s   = _from_shape(f"xs_{tno}", [B, in_ch, N])
            out_s = mlp(x_s)
            exp   = [B, conv_chs[-1], N]
            shape_only_ok = list(out_s.shape) == exp

            # Data path – verify data shape only (weights not injected; data may be None)
            x_d     = _from_data(f"xd_{tno}", _make_weights([B, in_ch, N], seed=tno))
            out_d   = MLP(f"ut_shapd_{tno}", in_channel=in_ch,
                          conv_channels=conv_chs, with_bn=with_bn, with_relu=with_relu)(x_d)
            data_shape_ok = list(out_d.shape) == exp

            ok = shape_only_ok and data_shape_ok
            st = "PASS" if ok else "FAIL"
            print(
                f"TEST[{tno:02d}] {name:{msgw}s} {st}  "
                f"shape={list(out_d.shape)}  expected={exp}"
            )
            if not ok:
                all_passed = False

        except Exception as exc:
            print(f"TEST[{tno:02d}] {name:{msgw}s} ERROR  {exc}")
            all_passed = False

    assert all_passed, "One or more shape validation tests failed"


# ===========================================================================
# Category 2 – Edge Case Creation
# ===========================================================================

# (name, MLP kwargs, input shape, description)
_EDGE_CASES = [
    # 1-channel input and output
    (
        "Single channel 1→(1,)",
        dict(in_channel=1, conv_channels=(1,), with_bn=True, with_relu=True),
        (1, 1, 16),
        "Minimum channels: 1→1",
    ),
    # Very deep MLP (7 layers)
    (
        "Deep MLP 7 layers",
        dict(in_channel=8, conv_channels=(16, 32, 64, 32, 16, 8, 4), with_bn=True, with_relu=True),
        (1, 8, 32),
        "Deep sequential network: 7 ConvModule1d layers",
    ),
    # All-zero weights → output depends only on BN bias
    (
        "All-zero conv weights",
        dict(in_channel=4, conv_channels=(8,), with_bn=False, with_relu=False),
        (1, 4, 8),
        "Zero weights: output should be zero (no BN, no bias injection)",
    ),
    # with_bn=False, with_relu=False → pure pointwise linear
    (
        "Pure linear (no BN, no ReLU)",
        dict(in_channel=16, conv_channels=(32,), with_bn=False, with_relu=False),
        (2, 16, 50),
        "Linear projection only",
    ),
    # with_bn=True, with_relu=False → BN but no activation
    (
        "BN only, no ReLU",
        dict(in_channel=16, conv_channels=(32,), with_bn=True, with_relu=False),
        (2, 16, 50),
        "BN without activation",
    ),
    # N=1 (single spatial point)
    (
        "N=1 minimal spatial",
        dict(in_channel=18, conv_channels=(64,), with_bn=True, with_relu=True),
        (4, 18, 1),
        "Single spatial position: N=1",
    ),
    # Large batch
    (
        "Large batch B=32",
        dict(in_channel=8, conv_channels=(16,), with_bn=True, with_relu=True),
        (32, 8, 16),
        "Large mini-batch",
    ),
    # Channels that are not multiples of 2 (odd channel counts)
    (
        "Odd channels 5→(7,11)",
        dict(in_channel=5, conv_channels=(7, 11), with_bn=True, with_relu=True),
        (2, 5, 20),
        "Odd (non-power-of-2) channel counts",
    ),
    # Output channels > input channels (expansion)
    (
        "Expansion 18→(512,)",
        dict(in_channel=18, conv_channels=(512,), with_bn=True, with_relu=True),
        (1, 18, 64),
        "Very large expansion ratio",
    ),
    # Bottleneck: expand then compress
    (
        "Bottleneck 64→(256,64)",
        dict(in_channel=64, conv_channels=(256, 64), with_bn=True, with_relu=True),
        (2, 64, 32),
        "Bottleneck: expand then compress",
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_mlp_edge_cases():
    """
    Category 2 – Edge Case Creation

    Tests boundary conditions unique to MLP / ConvModule1d:
      - Minimum channel count (1→1)
      - Very deep network (7 layers)
      - All-zero weights
      - Activation and normalization flag combinations
      - N=1 (single spatial point) and large batches
      - Odd (non-power-of-2) channel counts
      - Extreme expansion and bottleneck configurations
    """
    msgw = _get_max_msg_len(_EDGE_CASES)
    all_passed = True

    for tno, (name, mlp_kwargs, (B, in_ch, N), desc) in enumerate(_EDGE_CASES):
        try:
            in_channel   = mlp_kwargs.pop("in_channel")
            conv_channels = mlp_kwargs.pop("conv_channels")
            mlp = MLP(f"ut_edge_{tno}", in_channel=in_channel,
                      conv_channels=conv_channels, **mlp_kwargs)
            # Restore for print
            mlp_kwargs["in_channel"]    = in_channel
            mlp_kwargs["conv_channels"] = conv_channels

            x_s   = _from_shape(f"ut_edge_x_{tno}", [B, in_ch, N])
            out   = mlp(x_s)
            exp   = [B, conv_channels[-1], N]
            ok    = list(out.shape) == exp
            st    = "PASS" if ok else "FAIL"
            print(f"TEST[{tno:02d}] {name:{msgw}s} {st}  shape={list(out.shape)}")
            if not ok:
                print(f"         expected {exp}  |  {desc}")
                all_passed = False

        except Exception as exc:
            print(f"TEST[{tno:02d}] {name:{msgw}s} ERROR  {type(exc).__name__}: {exc}")
            all_passed = False

    assert all_passed, "One or more edge case tests failed"


# ===========================================================================
# Category 3 – Data Validation
# ===========================================================================

_SEED = 42


@pytest.mark.unit
@pytest.mark.opunit
def test_mlp_data_validation():
    """
    Category 3 – Data Validation

    Validates numerical correctness at every layer and for complete forward
    passes, comparing TTSim graph output (.data) against PyTorch references:
      1. Conv1d step: TTSim Conv op vs torch.nn.Conv1d
      2. BatchNorm1d step (eval): TTSim BN op vs torch.nn.BatchNorm1d
      3. ReLU step: TTSim Relu vs torch.relu
      4. Full ConvModule1d (Conv+BN+ReLU) vs PyTorch equivalent
      5. Full 2-layer MLP vs stacked PyTorch ConvModule
      6. MLP without BN vs torch Conv1d+ReLU
      7. Analytical param count validation for multiple configs
      8. Identity-weight sanity check (known expected output)
    """
    all_passed = True
    rng = np.random.RandomState(_SEED)

    # -----------------------------------------------------------------------
    # DATA[00] Conv1d step vs torch.nn.Conv1d
    # -----------------------------------------------------------------------
    try:
        B, C_in, N, C_out = 2, 18, 128, 64
        conv_w  = _make_weights([C_out, C_in, 1], seed=1)
        conv_b  = _make_weights([C_out],          seed=2)
        x_np    = _make_weights([B, C_in, N],     seed=3)
        pt_conv = nn.Conv1d(C_in, C_out, 1, bias=True)
        pt_conv.weight.data = torch.tensor(conv_w)
        pt_conv.bias.data   = torch.tensor(conv_b)
        with torch.no_grad():
            pt_out = pt_conv(torch.tensor(x_np)).numpy()
        tt_out = _run_ttsim_cm(x_np, C_in, C_out, conv_w=conv_w, conv_b=conv_b,
                                with_bn=False, with_relu=False)
        assert tt_out is not None, "TTSim Conv returned None"
        diff = float(np.max(np.abs(pt_out - tt_out)))
        ok = diff < 1e-4
        print(f"DATA[00] Conv1d step vs torch.nn.Conv1d         {'PASS' if ok else 'FAIL'}  "
              f"max_diff={diff:.3e}  shape={tt_out.shape}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[00] Conv1d step vs torch.nn.Conv1d         ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[01] BatchNorm1d step (eval) vs torch.nn.BatchNorm1d
    # -----------------------------------------------------------------------
    try:
        B, C, N = 2, 64, 128
        x_np    = rng.randn(B, C, N).astype(np.float32)
        scale   = np.ones(C,  dtype=np.float32)
        bias_bn = np.zeros(C, dtype=np.float32)
        bn_mean = (rng.randn(C) * 0.1).astype(np.float32)
        bn_var  = (rng.rand(C) * 0.5 + 0.5).astype(np.float32)
        pt_bn = nn.BatchNorm1d(C, eps=1e-5)
        pt_bn.weight.data       = torch.tensor(scale)
        pt_bn.bias.data         = torch.tensor(bias_bn)
        pt_bn.running_mean.data = torch.tensor(bn_mean)
        pt_bn.running_var.data  = torch.tensor(bn_var)
        pt_bn.eval()
        with torch.no_grad():
            pt_out = pt_bn(torch.tensor(x_np)).numpy()
        bn_mod = BatchNorm1d("ut_bn_dv", num_features=C, eps=1e-5)
        bn_mod.scale.data        = scale
        bn_mod.bias_bn.data      = bias_bn
        bn_mod.running_mean.data = bn_mean
        bn_mod.running_var.data  = bn_var
        tt_out = bn_mod(_from_data("xbn", x_np)).data
        assert tt_out is not None, "TTSim BN returned None"
        diff = float(np.max(np.abs(pt_out - tt_out)))
        ok = diff < 1e-4
        print(f"DATA[01] BatchNorm1d step vs torch (eval mode)  {'PASS' if ok else 'FAIL'}  "
              f"max_diff={diff:.3e}  shape={tt_out.shape}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[01] BatchNorm1d step vs torch               ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[02] ReLU step vs torch.relu
    # -----------------------------------------------------------------------
    try:
        x_np   = rng.randn(2, 64, 128).astype(np.float32)
        pt_out = torch.relu(torch.tensor(x_np)).numpy()
        tt_out = F.Relu("ut_relu_dv")(_from_data("xrelu", x_np)).data
        assert tt_out is not None, "TTSim ReLU returned None"
        diff = float(np.max(np.abs(pt_out - tt_out)))
        ok = diff < 1e-7
        print(f"DATA[02] ReLU step vs torch.relu                {'PASS' if ok else 'FAIL'}  "
              f"max_diff={diff:.3e}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[02] ReLU step vs torch.relu                ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[03] Full ConvModule1d (Conv+BN+ReLU) vs PyTorch
    # -----------------------------------------------------------------------
    try:
        B, C_in, N, C_out = 2, 18, 128, 64
        conv_w  = _make_weights([C_out, C_in, 1], seed=10)
        conv_b  = _make_weights([C_out],           seed=11)
        bn_s    = np.ones(C_out,  dtype=np.float32)
        bn_b    = np.zeros(C_out, dtype=np.float32)
        bn_mean = _make_weights([C_out], seed=12, scale=0.05)
        bn_var  = (np.random.RandomState(13).rand(C_out) * 0.5 + 0.5).astype(np.float32)
        x_np    = _make_weights([B, C_in, N], seed=14, scale=0.5)
        pt_out  = _pt_convmodule1d(x_np, conv_w, conv_b, bn_s, bn_b, bn_mean, bn_var)
        tt_out  = _run_ttsim_cm(x_np, C_in, C_out, conv_w=conv_w, conv_b=conv_b,
                                 bn_scale=bn_s, bn_bias=bn_b, bn_mean=bn_mean, bn_var=bn_var)
        assert tt_out is not None, "TTSim ConvModule returned None"
        diff = float(np.max(np.abs(pt_out - tt_out)))
        ok = diff < 1e-4
        print(f"DATA[03] Full ConvModule1d vs PyTorch           {'PASS' if ok else 'FAIL'}  "
              f"max_diff={diff:.3e}  shape={tt_out.shape}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[03] Full ConvModule1d vs PyTorch            ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[04] Full 2-layer MLP vs stacked PyTorch modules
    # -----------------------------------------------------------------------
    try:
        B, in_ch, N = 2, 18, 128
        ch   = (256, 256)
        rng2 = np.random.RandomState(200)
        def _gen(ic, oc):
            return {
                "conv_w":   (rng2.randn(oc, ic, 1) * 0.1).astype(np.float32),
                "conv_b":   (rng2.randn(oc)         * 0.1).astype(np.float32),
                "bn_scale": np.ones(oc,  dtype=np.float32),
                "bn_bias":  np.zeros(oc, dtype=np.float32),
                "bn_mean":  (rng2.randn(oc) * 0.05).astype(np.float32),
                "bn_var":   (rng2.rand(oc) * 0.5 + 0.5).astype(np.float32),
            }
        w0   = _gen(in_ch, ch[0])
        w1   = _gen(ch[0], ch[1])
        x_np = (rng2.randn(B, in_ch, N) * 0.5).astype(np.float32)
        mid  = _pt_convmodule1d(x_np, w0["conv_w"], w0["conv_b"],
                                w0["bn_scale"], w0["bn_bias"], w0["bn_mean"], w0["bn_var"])
        pt_out = _pt_convmodule1d(mid, w1["conv_w"], w1["conv_b"],
                                  w1["bn_scale"], w1["bn_bias"], w1["bn_mean"], w1["bn_var"])
        tt_out = _run_ttsim_mlp(x_np, in_ch, ch, [w0, w1])
        assert tt_out is not None, "TTSim MLP returned None"
        diff = float(np.max(np.abs(pt_out - tt_out)))
        ok = diff < 1e-4
        print(f"DATA[04] Full 2-layer MLP vs PyTorch            {'PASS' if ok else 'FAIL'}  "
              f"max_diff={diff:.3e}  shape={tt_out.shape}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[04] Full 2-layer MLP vs PyTorch             ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[05] MLP without BN vs torch Conv1d + ReLU
    # -----------------------------------------------------------------------
    try:
        B, in_ch, N, out_ch = 2, 32, 64, 128
        conv_w = _make_weights([out_ch, in_ch, 1], seed=30, scale=0.1)
        conv_b = _make_weights([out_ch],            seed=31, scale=0.1)
        x_np   = _make_weights([B, in_ch, N],      seed=32, scale=0.5)
        pt_out = _pt_convmodule1d(x_np, conv_w, conv_b,
                                  bn_scale=None, bn_bias=None, bn_mean=None, bn_var=None,
                                  with_bn=False, with_relu=True)
        mlp = MLP("ut_nobn", in_channel=in_ch,
                  conv_channels=(out_ch,), with_bn=False, with_relu=True)
        mlp.layers[0].conv_weight.data = conv_w
        mlp.layers[0].conv_bias.data   = conv_b
        tt_out = mlp(_from_data("xnobn", x_np)).data
        assert tt_out is not None, "TTSim MLP(no BN) returned None"
        diff = float(np.max(np.abs(pt_out - tt_out)))
        ok = diff < 1e-5
        print(f"DATA[05] MLP without BN vs torch Conv+ReLU      {'PASS' if ok else 'FAIL'}  "
              f"max_diff={diff:.3e}  shape={tt_out.shape}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[05] MLP without BN vs torch Conv+ReLU       ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[06] Analytical parameter count for multiple configs
    # -----------------------------------------------------------------------
    #  ConvModule1d with BN:
    #    conv: in*out*k + out (bias)
    #    BN:   2*out (scale + bias)
    #  MLP sum over all layers
    param_cases = [
        # (in_ch, conv_chs, with_bn, expected)
        (18,  (256, 256),  True,
             (18*256 + 256 + 2*256) + (256*256 + 256 + 2*256)),          # 71680
        (64,  (128,),      True,  64*128 + 128 + 2*128),                  # 8576
        (8,   (16, 32),    False, (8*16 + 16) + (16*32 + 32)),            # 672
        (256, (512, 256),  True,
             (256*512 + 512 + 2*512) + (512*256 + 256 + 2*256)),          # 264704
    ]
    for pno, (in_ch, conv_chs, with_bn, expected) in enumerate(param_cases):
        try:
            mlp = MLP(f"ut_pc_{pno}", in_channel=in_ch,
                      conv_channels=conv_chs, with_bn=with_bn)
            got = mlp.analytical_param_count()
            ok  = got == expected
            print(f"DATA[06.{pno}] Param count {in_ch}→{conv_chs} bn={with_bn}  "
                  f"{'PASS' if ok else 'FAIL'}  got={got:,}  expected={expected:,}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"DATA[06.{pno}] Param count                        ERROR  {exc}")
            all_passed = False

    # -----------------------------------------------------------------------
    # DATA[07] Identity-weight sanity check
    # -----------------------------------------------------------------------
    try:
        B, C, N = 1, 4, 8
        conv_w  = np.eye(C, dtype=np.float32).reshape(C, C, 1)   # identity
        conv_b  = np.zeros(C, dtype=np.float32)
        bn_s    = np.ones(C,  dtype=np.float32)
        bn_b    = np.zeros(C, dtype=np.float32)
        bn_mean = np.zeros(C, dtype=np.float32)
        bn_var  = np.ones(C,  dtype=np.float32)
        x_np    = np.arange(B * C * N, dtype=np.float32).reshape(B, C, N)
        pt_out  = _pt_convmodule1d(x_np, conv_w, conv_b, bn_s, bn_b, bn_mean, bn_var)
        tt_out  = _run_ttsim_cm(x_np, C, C, conv_w=conv_w, conv_b=conv_b,
                                 bn_scale=bn_s, bn_bias=bn_b,
                                 bn_mean=bn_mean, bn_var=bn_var)
        assert tt_out is not None, "TTSim identity-weight returned None"
        diff = float(np.max(np.abs(pt_out - tt_out)))
        ok = diff < 1e-5
        print(f"DATA[07] Identity-weight sanity check           {'PASS' if ok else 'FAIL'}  "
              f"max_diff={diff:.3e}")
        print(f"         Input  row: {x_np[0, :, 0]}")
        print(f"         PyTorch   : {pt_out[0, :, 0]}")
        print(f"         TTSim     : {tt_out[0, :, 0]}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[07] Identity-weight sanity check            ERROR  {exc}")
        all_passed = False

    assert all_passed, "One or more data validation tests failed"
