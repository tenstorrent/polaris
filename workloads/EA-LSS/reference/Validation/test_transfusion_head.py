#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for TransformerDecoderLayer and TransFusionHead TTSim modules.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_transfusion_head.py

Note: TDL uses LayerNorm which does not propagate TTSim data.
      Numerical comparisons cover individual ops (linear, layernorm, MHA)
      using PyTorch references and NumPy utilities.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn

_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ttsim.front.functional.op import _from_shape, _from_data
from ttsim_modules.transfusion_head import TransformerDecoderLayer, TransFusionHead
from ttsim_modules.multihead_attention import MultiheadAttention
from reference.Validation.ttsim_utils import (
    print_header, print_test, compare_arrays,
    ttsim_matmul, ttsim_layernorm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def inject_mha_weights(mha_mod, in_proj_w_pt, in_proj_b, out_proj_w, out_proj_b):
    """Inject weights into a TTSim MultiheadAttention module."""
    mha_mod._mha._tensors['in_proj_weight'].data = in_proj_w_pt.T
    mha_mod._mha._tensors['in_proj_bias'].data   = in_proj_b
    mha_mod._mha._submodules['out_proj']._tensors['param'].data = out_proj_w.T
    mha_mod._mha._submodules['out_proj']._tensors['bias'].data  = out_proj_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_linear_step():
    """Linear (FFN) step: PyTorch nn.Linear vs NumPy (TTSim equivalent)."""
    print_header("Test 1: FFN Linear step — PyTorch vs NumPy")
    rng = np.random.RandomState(60)
    B, E, ffn = 4, 16, 32

    x_np = (rng.randn(B, E) * 0.5).astype(np.float32)
    w_pt = (rng.randn(ffn, E) * 0.1).astype(np.float32)   # [out, in] PyTorch convention
    b    = (rng.randn(ffn)     * 0.1).astype(np.float32)

    lin = nn.Linear(E, ffn)
    lin.weight.data = torch.tensor(w_pt)
    lin.bias.data   = torch.tensor(b)
    with torch.no_grad():
        pt_out = lin(torch.tensor(x_np)).numpy()
    print(f"  PyTorch Linear: shape={list(pt_out.shape)}, sample={pt_out.flatten()[:4]}")

    # NumPy equivalent: x @ W.T + b
    ts_out = x_np @ w_pt.T + b
    print(f"  NumPy   Linear: shape={list(ts_out.shape)}, sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "FFN linear step")
    return ok


def test_layernorm_step():
    """LayerNorm step: PyTorch nn.LayerNorm vs ttsim_layernorm."""
    print_header("Test 2: LayerNorm step — PyTorch vs ttsim_layernorm")
    rng = np.random.RandomState(61)
    B, E = 4, 16
    x_np = (rng.randn(B, E) * 0.5).astype(np.float32)

    ln = nn.LayerNorm(E, eps=1e-5)
    ln.weight.data = torch.ones(E)
    ln.bias.data   = torch.zeros(E)
    with torch.no_grad():
        pt_out = ln(torch.tensor(x_np)).numpy()
    print(f"  PyTorch LN: shape={list(pt_out.shape)}, sample={pt_out.flatten()[:4]}")

    ts_out = ttsim_layernorm(x_np, dim=-1)
    print(f"  TTSim   LN: shape={list(ts_out.shape)}, sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "LayerNorm step", atol=1e-5)
    return ok


def test_mha_in_tdl():
    """Self-attention step: PyTorch MHA vs TTSim MultiheadAttention."""
    print_header("Test 3: TDL Self-attention — PyTorch vs TTSim MHA")
    rng = np.random.RandomState(62)
    E, N, L = 16, 2, 5

    in_proj_w  = (rng.randn(3*E, E) * 0.1).astype(np.float32)
    in_proj_b  = (rng.randn(3*E)    * 0.1).astype(np.float32)
    out_proj_w = (rng.randn(E, E)   * 0.1).astype(np.float32)
    out_proj_b = (rng.randn(E)      * 0.1).astype(np.float32)
    x_np = (rng.randn(N, L, E) * 0.1).astype(np.float32)

    pt_mha = nn.MultiheadAttention(E, 2, bias=True, batch_first=True)
    pt_mha.in_proj_weight.data  = torch.tensor(in_proj_w)
    pt_mha.in_proj_bias.data    = torch.tensor(in_proj_b)
    pt_mha.out_proj.weight.data = torch.tensor(out_proj_w)
    pt_mha.out_proj.bias.data   = torch.tensor(out_proj_b)
    pt_mha.eval()
    with torch.no_grad():
        pt_out, _ = pt_mha(torch.tensor(x_np), torch.tensor(x_np), torch.tensor(x_np))
    pt_out = pt_out.numpy()
    print(f"  PyTorch MHA: shape={list(pt_out.shape)}, sample={pt_out.flatten()[:4]}")

    mha = MultiheadAttention('mha3', embed_dim=E, num_heads=2)
    inject_mha_weights(mha, in_proj_w, in_proj_b, out_proj_w, out_proj_b)
    x_t = _from_data('x3', x_np)
    ts_out, _ = mha(x_t, x_t, x_t)
    print(f"  TTSim   MHA: shape={list(ts_out.shape)}, sample={ts_out.data.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out.data, "TDL self-attn", atol=1e-6)
    return ok


def test_tdl_params():
    print_header("Test 4: TransformerDecoderLayer param count (expected 233088)")
    tdl = TransformerDecoderLayer("tdl_p", d_model=128, nhead=8, dim_feedforward=256)
    p   = tdl.analytical_param_count()
    ok  = p == 233088
    print_test("TDL params == 233088", f"got {p:,}")
    return ok


def test_tdl_shape():
    print_header("Test 5: TransformerDecoderLayer output shape")
    tdl  = TransformerDecoderLayer("tdl_s", d_model=128, nhead=8, dim_feedforward=256)
    q    = _from_shape("tdl_q",  [1, 128, 200])
    k    = _from_shape("tdl_k",  [1, 128, 500])
    qpos = _from_shape("tdl_qp", [1, 200, 2])
    kpos = _from_shape("tdl_kp", [1, 500, 2])
    out  = tdl(q, k, qpos, kpos)
    ok   = list(out.shape) == [1, 128, 200]
    print_test("TDL output shape [1,128,200]", f"got {list(out.shape)}")
    return ok


def test_tdl_cross_only():
    print_header("Test 6: TransformerDecoderLayer cross_only=True param count")
    tdl = TransformerDecoderLayer("tdl_co", d_model=128, nhead=8, dim_feedforward=256, cross_only=True)
    p   = tdl.analytical_param_count()
    # cross_only removes self_attn (66048) + norm1 (256)
    expected_cross = 233088 - 66048 - 256
    ok = p == expected_cross
    print_test("TDL cross_only params", f"got {p:,}  expected {expected_cross:,}")
    return ok


def test_tfh_ealss_config():
    print_header("Test 7: TransFusionHead (EA-LSS config)")
    tfh = TransFusionHead(
        "tfh_v",
        in_channels=1024,
        hidden_channel=128,
        num_classes=10,
        num_proposals=200,
        num_decoder_layers=1,
        num_heads=8,
        ffn_channel=256,
        initialize_by_heatmap=True,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
    )
    x    = _from_shape("tfh_x", [1, 1024, 128, 128])
    pred = tfh(x)
    ok_keys = set(pred.keys()) == {"center", "height", "dim", "rot", "vel", "heatmap"}
    print_test("TransFusionHead output keys", f"keys={set(pred.keys())}")
    ok_p = tfh.analytical_param_count() > 0
    print_test("TransFusionHead params > 0", f"params={tfh.analytical_param_count():,}")
    return ok_keys and ok_p


if __name__ == "__main__":
    tests = [
        ("linear_step",      test_linear_step),
        ("layernorm_step",   test_layernorm_step),
        ("mha_in_tdl",       test_mha_in_tdl),
        ("tdl_params",       test_tdl_params),
        ("tdl_shape",        test_tdl_shape),
        ("tdl_cross_only",   test_tdl_cross_only),
        ("tfh_ealss_config", test_tfh_ealss_config),
    ]
    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            results[name] = False

    print("\n" + "="*60)
    passed = sum(results.values())
    total  = len(results)
    for n, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
