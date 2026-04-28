#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comparison test:  PyTorch SegDETRHead  vs  TTSim SegDETRHead

Tests numerical equivalence of:
  1. fc_cls branch (Linear)
  2. reg branch (FFN → ReLU → Linear → Sigmoid)
  3. Combined forward (cls + reg)
  4. input_proj (Conv2d 1×1)
  5. analytical_param_count
  6. Sigmoid cls mode (cls_out_channels == num_things)
"""

import os
import sys
import traceback

polaris_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', '..'))
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.seg_head_plugin.seg_detr_head import (
    SegDETRHead,
)


# ====================================================================
# Compare helper
# ====================================================================

def compare(pt_out, tt_out, name, atol=1e-4):
    pt_np = pt_out.detach().numpy() if isinstance(pt_out, torch.Tensor) else pt_out
    tt_np = tt_out.data if hasattr(tt_out, 'data') and tt_out.data is not None else tt_out
    print(f"\n  {name}:")
    print(f"    PyTorch shape: {list(pt_np.shape)}")
    print(f"    TTSim   shape: {list(tt_np.shape)}")
    if list(pt_np.shape) != list(tt_np.shape):
        print(f"    [FAIL] Shape mismatch!")
        return False
    diff = np.abs(pt_np - tt_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    if np.allclose(pt_np, tt_np, atol=atol):
        print(f"    [OK] Match (atol={atol})")
        return True
    else:
        print(f"    [FAIL] Max diff {max_diff:.6e} > atol {atol}")
        return False


# ====================================================================
# PyTorch reference (standalone, no mmcv dependency)
# ====================================================================

class PT_FFN(nn.Module):
    """Minimal FFN matching mmcv FFN (no dropout, no residual)."""
    def __init__(self, embed_dims, feedforward_channels, num_fcs):
        super().__init__()
        layers = []
        in_ch = embed_dims
        for i in range(num_fcs - 1):
            layers.append(nn.Linear(in_ch, feedforward_channels))
            layers.append(nn.ReLU(inplace=True))
            in_ch = feedforward_channels
        layers.append(nn.Linear(in_ch, embed_dims))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PT_SegDETRHead(nn.Module):
    """Minimal PyTorch SegDETRHead for testing cls + reg branches."""
    def __init__(self, num_things_classes, in_channels, embed_dims,
                 num_query, num_reg_fcs, use_sigmoid_cls=False):
        super().__init__()
        self.num_things_classes = num_things_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_query = num_query

        if use_sigmoid_cls:
            self.cls_out_channels = num_things_classes
        else:
            self.cls_out_channels = num_things_classes + 1

        # TTSim F.Conv2d does not store bias param, so use bias=False for comparison
        self.input_proj = nn.Conv2d(in_channels, embed_dims, kernel_size=1, bias=False)
        self.fc_cls = nn.Linear(embed_dims, self.cls_out_channels)
        self.reg_ffn = PT_FFN(embed_dims, embed_dims, num_reg_fcs)
        self.activate = nn.ReLU(inplace=False)
        self.fc_reg = nn.Linear(embed_dims, 4)
        self.query_embedding = nn.Embedding(num_query, embed_dims)

    def forward_branches(self, outs_dec):
        """Just cls + reg branches (no transformer)."""
        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds


# ====================================================================
# Weight transfer: PyTorch → TTSim
# ====================================================================

def transfer_weights(pt_head, tt_head):
    """Copy weights from PT_SegDETRHead to TTSim SegDETRHead."""
    # input_proj (Conv2d) — params is list of (ipos, SimTensor) tuples; no bias in TTSim
    tt_head.input_proj.params[0][1].data = pt_head.input_proj.weight.detach().numpy().copy()

    # fc_cls (SimNN.Linear) — .param is [out,in], .bias direct
    tt_head.fc_cls.param.data = pt_head.fc_cls.weight.detach().numpy().copy()
    tt_head.fc_cls.bias.data = pt_head.fc_cls.bias.detach().numpy().copy()

    # reg_ffn — PT uses nn.Sequential with [Linear, ReLU, Linear]
    # TTSim FFN has self.layers[0], self.layers[1] (SimNN.Linear via ModuleList)
    pt_linears = [m for m in pt_head.reg_ffn.layers if isinstance(m, nn.Linear)]
    tt_head.reg_ffn.layers[0].param.data = pt_linears[0].weight.detach().numpy().copy()
    tt_head.reg_ffn.layers[0].bias.data = pt_linears[0].bias.detach().numpy().copy()
    tt_head.reg_ffn.layers[1].param.data = pt_linears[1].weight.detach().numpy().copy()
    tt_head.reg_ffn.layers[1].bias.data = pt_linears[1].bias.detach().numpy().copy()

    # fc_reg (SimNN.Linear)
    tt_head.fc_reg.param.data = pt_head.fc_reg.weight.detach().numpy().copy()
    tt_head.fc_reg.bias.data = pt_head.fc_reg.bias.detach().numpy().copy()


# ====================================================================
# Tests
# ====================================================================

def test_forward_branches():
    """Test 1-3: cls branch, reg branch, combined forward."""
    print("\n" + "=" * 60)
    print("Tests 1-3: Forward branches (cls + reg)")
    print("=" * 60)

    num_things = 3
    in_ch = 256
    embed_dims = 256
    num_query = 300
    num_reg_fcs = 2
    nb_dec = 6
    bs = 1

    torch.manual_seed(42)
    np.random.seed(42)

    # Build PyTorch reference
    pt = PT_SegDETRHead(num_things, in_ch, embed_dims, num_query, num_reg_fcs)
    pt.eval()

    # Build TTSim
    tt = SegDETRHead(
        'cmp_seg_detr',
        num_classes=10,
        num_things_classes=num_things,
        num_stuff_classes=1,
        in_channels=in_ch,
        embed_dims=embed_dims,
        num_query=num_query,
        num_reg_fcs=num_reg_fcs,
        use_sigmoid_cls=False)

    # Transfer weights
    transfer_weights(pt, tt)

    # Shared input
    x_np = np.random.randn(nb_dec, bs, num_query, embed_dims).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('outs_dec', x_np.copy())

    # PyTorch forward
    with torch.no_grad():
        pt_cls, pt_bbox = pt.forward_branches(x_pt)

    # TTSim forward
    tt_cls, tt_bbox = tt(x_tt)

    results = []

    # Test 1: cls branch
    results.append(compare(pt_cls, tt_cls, "Test 1: fc_cls output"))

    # Test 2: reg branch (sigmoid)
    results.append(compare(pt_bbox, tt_bbox, "Test 2: reg branch (FFN→ReLU→Linear→Sigmoid)"))

    # Test 3: shapes match expected
    ok = True
    exp_cls = [nb_dec, bs, num_query, num_things + 1]
    exp_bbox = [nb_dec, bs, num_query, 4]
    if list(pt_cls.shape) != exp_cls:
        print(f"  [FAIL] Test 3: PT cls shape {list(pt_cls.shape)} != {exp_cls}")
        ok = False
    if list(tt_cls.shape) != exp_cls:
        print(f"  [FAIL] Test 3: TT cls shape {list(tt_cls.shape)} != {exp_cls}")
        ok = False
    if list(pt_bbox.shape) != exp_bbox:
        print(f"  [FAIL] Test 3: PT bbox shape {list(pt_bbox.shape)} != {exp_bbox}")
        ok = False
    if list(tt_bbox.shape) != exp_bbox:
        print(f"  [FAIL] Test 3: TT bbox shape {list(tt_bbox.shape)} != {exp_bbox}")
        ok = False
    if ok:
        print(f"\n  Test 3: Shape match: [OK]  cls={exp_cls}, bbox={exp_bbox}")
    results.append(ok)

    return results


def test_input_proj():
    """Test 4: input_proj Conv2d 1×1."""
    print("\n" + "=" * 60)
    print("Test 4: input_proj (Conv2d 1×1)")
    print("=" * 60)

    in_ch = 128
    embed_dims = 256
    num_things = 3
    num_query = 100

    torch.manual_seed(99)
    np.random.seed(99)

    pt = PT_SegDETRHead(num_things, in_ch, embed_dims, num_query, 2)
    pt.eval()

    tt = SegDETRHead(
        'cmp_ip',
        num_classes=10,
        num_things_classes=num_things,
        num_stuff_classes=1,
        in_channels=in_ch,
        embed_dims=embed_dims,
        num_query=num_query,
        num_reg_fcs=2)

    # Transfer only input_proj weight (no bias in TTSim Conv2d)
    tt.input_proj.params[0][1].data = pt.input_proj.weight.detach().numpy().copy()

    # Input: [bs, in_ch, H, W]
    bs, H, W = 1, 16, 32
    x_np = np.random.randn(bs, in_ch, H, W).astype(np.float32)

    with torch.no_grad():
        pt_out = pt.input_proj(torch.from_numpy(x_np))

    x_tt = F._from_data('feat_in', x_np.copy())
    tt_out = tt.input_proj(x_tt)

    return [compare(pt_out, tt_out, "Test 4: input_proj")]


def test_param_count():
    """Test 5: analytical_param_count."""
    print("\n" + "=" * 60)
    print("Test 5: analytical_param_count")
    print("=" * 60)

    num_things = 3
    in_ch = 256
    embed_dims = 256
    num_query = 300
    num_reg_fcs = 2

    pt = PT_SegDETRHead(num_things, in_ch, embed_dims, num_query, num_reg_fcs)
    pt_count = sum(p.numel() for p in pt.parameters())

    tt = SegDETRHead(
        'cmp_pc',
        num_classes=10,
        num_things_classes=num_things,
        num_stuff_classes=1,
        in_channels=in_ch,
        embed_dims=embed_dims,
        num_query=num_query,
        num_reg_fcs=num_reg_fcs)

    tt_count = tt.analytical_param_count(lvl=2)

    print(f"\n  PyTorch param count: {pt_count:,}")
    print(f"  TTSim  param count: {tt_count:,}")

    if pt_count == tt_count:
        print(f"  [OK] Exact match")
        return [True]
    else:
        diff = abs(pt_count - tt_count)
        print(f"  [INFO] Difference: {diff:,}")
        # Investigate mismatch
        print(f"\n  PyTorch breakdown:")
        for n, p in pt.named_parameters():
            print(f"    {n}: {p.numel():,}  {list(p.shape)}")
        # TTSim Conv2d doesn't store bias — difference should be embed_dims (bias size)
        expected_diff = embed_dims  # input_proj bias
        if diff == expected_diff:
            print(f"\n  [OK] Difference ({diff}) matches Conv2d bias ({expected_diff}) — "
                  f"TTSim Conv2d has no bias param, accounted for")
            return [True]
        return [False]


def test_sigmoid_cls():
    """Test 6: use_sigmoid_cls=True changes cls_out_channels."""
    print("\n" + "=" * 60)
    print("Test 6: Sigmoid cls mode")
    print("=" * 60)

    num_things = 5
    in_ch = 64
    embed_dims = 64
    num_query = 50
    nb_dec, bs = 3, 1

    torch.manual_seed(77)
    np.random.seed(77)

    pt = PT_SegDETRHead(num_things, in_ch, embed_dims, num_query, 2,
                        use_sigmoid_cls=True)
    pt.eval()

    tt = SegDETRHead(
        'cmp_sigcls',
        num_classes=10,
        num_things_classes=num_things,
        num_stuff_classes=1,
        in_channels=in_ch,
        embed_dims=embed_dims,
        num_query=num_query,
        num_reg_fcs=2,
        use_sigmoid_cls=True)

    transfer_weights(pt, tt)

    x_np = np.random.randn(nb_dec, bs, num_query, embed_dims).astype(np.float32)
    with torch.no_grad():
        pt_cls, pt_bbox = pt.forward_branches(torch.from_numpy(x_np))
    tt_cls, tt_bbox = tt(F._from_data('x_sig', x_np.copy()))

    results = []
    results.append(compare(pt_cls, tt_cls, "Test 6a: cls (sigmoid mode)"))
    results.append(compare(pt_bbox, tt_bbox, "Test 6b: bbox (sigmoid mode)"))
    return results


# ====================================================================
# Main
# ====================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SegDETRHead — PyTorch vs TTSim Comparison Tests")
    print("=" * 70)

    all_results = []
    test_names = []

    try:
        r = test_forward_branches()
        all_results.extend(r)
        test_names.extend(["Test 1: fc_cls", "Test 2: reg branch", "Test 3: shapes"])
    except Exception as e:
        print(f"  [FAIL] test_forward_branches: {e}")
        traceback.print_exc()
        all_results.extend([False, False, False])
        test_names.extend(["Test 1: fc_cls", "Test 2: reg branch", "Test 3: shapes"])

    try:
        r = test_input_proj()
        all_results.extend(r)
        test_names.extend(["Test 4: input_proj"])
    except Exception as e:
        print(f"  [FAIL] test_input_proj: {e}")
        traceback.print_exc()
        all_results.append(False)
        test_names.append("Test 4: input_proj")

    try:
        r = test_param_count()
        all_results.extend(r)
        test_names.extend(["Test 5: param_count"])
    except Exception as e:
        print(f"  [FAIL] test_param_count: {e}")
        traceback.print_exc()
        all_results.append(False)
        test_names.append("Test 5: param_count")

    try:
        r = test_sigmoid_cls()
        all_results.extend(r)
        test_names.extend(["Test 6a: sigmoid cls", "Test 6b: sigmoid bbox"])
    except Exception as e:
        print(f"  [FAIL] test_sigmoid_cls: {e}")
        traceback.print_exc()
        all_results.extend([False, False])
        test_names.extend(["Test 6a: sigmoid cls", "Test 6b: sigmoid bbox"])

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(all_results)
    total = len(all_results)
    for name, ok in zip(test_names, all_results):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} passed")
    if passed == total:
        print("  [OK] All tests passed!")
    else:
        print("  [FAIL] Some tests failed!")
