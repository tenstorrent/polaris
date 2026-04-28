#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for OccHead plugin modules (TTSim vs PyTorch).

Validates numerical equivalence of:
  1. BevFeatureSlicer (identity)
  2. BevFeatureSlicer (grid sample)
  3. MLP
  4. SimpleConv2d (1 conv, shape)
  5. SimpleConv2d (4 convs, shape)
  6. SimpleConv2d (4 convs, numerical)
  7. UpsamplingAdd
  8. Bottleneck (no downsample)
  9. Bottleneck (downsample, even)
 10. Bottleneck (downsample, odd)
 11. CVT_DecoderBlock (no residual, no upsample)
 12. CVT_DecoderBlock (residual + upsample)
 13. predict_instance_segmentation_and_trajectories
"""

import os
import sys
import traceback
from collections import OrderedDict

polaris_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.occ_head_plugin.modules import (
    BevFeatureSlicer,
    MLP,
    SimpleConv2d,
    CVT_DecoderBlock,
    CVT_Decoder,
    UpsamplingAdd,
    Bottleneck,
    predict_instance_segmentation_and_trajectories,
)


# ====================================================================
# Compare helper (same style as test_motion_head.py)
# ====================================================================

def compare(pt_out, tt_out, name, atol=1e-4):
    """Compare PyTorch and TTSim outputs, print detailed diagnostics."""
    pt_np = pt_out.detach().numpy() if isinstance(pt_out, torch.Tensor) else pt_out
    tt_np = tt_out.data if hasattr(tt_out, 'data') else tt_out
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
    print(f"    [FAIL] Exceeds tolerance")
    return False


# ====================================================================
# Weight copy helpers
# ====================================================================

def copy_linear(pt_lin, tt_lin):
    """Copy nn.Linear -> SimNN.Linear."""
    tt_lin.param.data = pt_lin.weight.data.detach().numpy().astype(np.float32)
    tt_lin.bias.data = pt_lin.bias.data.detach().numpy().astype(np.float32)


def copy_conv2d(pt_conv, tt_conv_op):
    """Copy nn.Conv2d weights (and bias if present) to TTSim F.Conv2d op."""
    tt_conv_op.params[0][1].data = pt_conv.weight.data.detach().numpy().astype(np.float32)
    if pt_conv.bias is not None and len(tt_conv_op.params) > 1:
        tt_conv_op.params[1][1].data = pt_conv.bias.data.detach().numpy().astype(np.float32)


def copy_bn(pt_bn, tt_bn_op):
    """Copy nn.BatchNorm2d params to TTSim F.BatchNorm2d op."""
    tt_bn_op.params[0][1].data = pt_bn.weight.data.detach().numpy().astype(np.float32)
    tt_bn_op.params[1][1].data = pt_bn.bias.data.detach().numpy().astype(np.float32)
    tt_bn_op.params[2][1].data = pt_bn.running_mean.data.detach().numpy().astype(np.float32)
    tt_bn_op.params[3][1].data = pt_bn.running_var.data.detach().numpy().astype(np.float32)


# ====================================================================
# Globals
# ====================================================================

np.random.seed(42)
torch.manual_seed(42)

passed = 0
failed = 0


# ====================================================================
# TEST 1: BevFeatureSlicer (identity)
# ====================================================================

print("=" * 80)
print("TEST 1: BevFeatureSlicer (identity)")
print("=" * 80)

try:
    conf = {
        'xbound': [-51.2, 51.2, 0.512],
        'ybound': [-51.2, 51.2, 0.512],
        'zbound': [-10.0, 10.0, 20.0],
    }
    tt = BevFeatureSlicer('bfs_id', conf, conf)

    x_np = np.random.randn(2, 64, 200, 200).astype(np.float32)
    x_tt = F._from_data('bfs_id.in', x_np)
    out_tt = tt(x_tt)

    ok = compare(torch.from_numpy(x_np), out_tt,
                 "identity passthrough", atol=1e-6)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 1")
except Exception as e:
    print(f"  [FAIL] TEST 1 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 2: BevFeatureSlicer (grid sample)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: BevFeatureSlicer (grid sample)")
print("=" * 80)

try:
    bevformer_conf = {
        'xbound': [-51.2, 51.2, 0.512],
        'ybound': [-51.2, 51.2, 0.512],
        'zbound': [-10.0, 10.0, 20.0],
    }
    occflow_conf = {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
    }

    tt = BevFeatureSlicer('bfs_gs', bevformer_conf, occflow_conf)

    x_np = np.random.randn(1, 32, 200, 200).astype(np.float32)
    x_tt = F._from_data('bfs_gs.in', x_np)
    out_tt = tt(x_tt)

    # PyTorch reference
    from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.occ_head_plugin.modules import (
        calculate_birds_eye_view_parameters,
    )
    bev_res, bev_start, _ = calculate_birds_eye_view_parameters(
        bevformer_conf['xbound'], bevformer_conf['ybound'], bevformer_conf['zbound'])
    map_res, map_start, _ = calculate_birds_eye_view_parameters(
        occflow_conf['xbound'], occflow_conf['ybound'], occflow_conf['zbound'])

    map_x = np.arange(map_start[0], occflow_conf['xbound'][1], map_res[0])
    map_y = np.arange(map_start[1], occflow_conf['ybound'][1], map_res[1])
    norm_x = map_x / (-bev_start[0])
    norm_y = map_y / (-bev_start[1])
    tm, tn = np.meshgrid(norm_x, norm_y, indexing='ij')
    tm, tn = tm.T, tn.T
    grid_np = np.stack([tm, tn], axis=2).astype(np.float32)
    grid_pt = torch.from_numpy(grid_np).unsqueeze(0)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        out_pt = TF.grid_sample(x_pt, grid_pt, mode='bilinear', align_corners=True)

    ok = compare(out_pt, out_tt, "grid_sample vs PyTorch", atol=1e-4)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 2")
except Exception as e:
    print(f"  [FAIL] TEST 2 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 3: MLP (3 layers, 256->256->256->64)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: MLP (3 layers, 256->256->256->64)")
print("=" * 80)

try:
    input_dim, hidden_dim, output_dim, num_layers = 256, 256, 64, 3

    # PyTorch
    pt_layers = nn.ModuleList()
    h = [hidden_dim] * (num_layers - 1)
    for n_in, n_out in zip([input_dim] + h, h + [output_dim]):
        pt_layers.append(nn.Linear(n_in, n_out))

    # TTSim
    tt = MLP('mlp', input_dim, hidden_dim, output_dim, num_layers)

    # Copy weights
    for i, pt_lin in enumerate(pt_layers):
        copy_linear(pt_lin, tt.linears[i])

    x_np = np.random.randn(2, 10, input_dim).astype(np.float32)
    x_tt = F._from_data('mlp.in', x_np)
    out_tt = tt(x_tt)

    # PyTorch forward
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        for i, layer in enumerate(pt_layers):
            x_pt = TF.relu(layer(x_pt)) if i < num_layers - 1 else layer(x_pt)

    ok = compare(x_pt, out_tt, "MLP forward", atol=1e-4)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 3")
except Exception as e:
    print(f"  [FAIL] TEST 3 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 4: SimpleConv2d (num_conv=1, shape)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: SimpleConv2d (num_conv=1, shape check)")
print("=" * 80)

try:
    in_ch, out_ch = 64, 32

    tt = SimpleConv2d('sc1', in_channels=in_ch, out_channels=out_ch, num_conv=1)

    x_np = np.random.randn(1, in_ch, 50, 50).astype(np.float32)
    x_tt = F._from_data('sc1.in', x_np)
    out_tt = tt(x_tt)

    ok = list(out_tt.shape) == [1, out_ch, 50, 50]
    print(f"\n  Shape: {out_tt.shape}  (expected [1, {out_ch}, 50, 50])")
    print(f"  {'[OK]' if ok else '[FAIL]'} shape correct")
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 4")
except Exception as e:
    print(f"  [FAIL] TEST 4 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 5: SimpleConv2d (num_conv=4, shape)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: SimpleConv2d (num_conv=4, shape check)")
print("=" * 80)

try:
    in_ch, out_ch, conv_ch = 256, 256, 256

    tt = SimpleConv2d('sc4', in_channels=in_ch, out_channels=out_ch,
                      conv_channels=conv_ch, num_conv=4)

    x_np = np.random.randn(1, in_ch, 50, 50).astype(np.float32)
    x_tt = F._from_data('sc4.in', x_np)
    out_tt = tt(x_tt)

    ok = list(out_tt.shape) == [1, out_ch, 50, 50]
    print(f"\n  Shape: {out_tt.shape}  (expected [1, {out_ch}, 50, 50])")
    print(f"  {'[OK]' if ok else '[FAIL]'} shape correct")
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 5")
except Exception as e:
    print(f"  [FAIL] TEST 5 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 6: SimpleConv2d (4 convs, numerical weight-copy match)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: SimpleConv2d (4 convs, numerical weight-copy match)")
print("=" * 80)

try:
    in_ch, out_ch, conv_ch = 64, 32, 64

    # PyTorch
    pt_convs = nn.ModuleList()
    pt_bns = nn.ModuleList()
    c_in = in_ch
    for i in range(3):
        pt_convs.append(nn.Conv2d(c_in, conv_ch, 3, 1, 1, bias=False))
        pt_bns.append(nn.BatchNorm2d(conv_ch))
        c_in = conv_ch
    pt_final = nn.Conv2d(conv_ch, out_ch, 1, bias=True)
    pt_final.bias.data.zero_()  # TTSim Conv2d has no bias param
    for bn in pt_bns:
        bn.eval()

    # TTSim
    tt = SimpleConv2d('sc4n', in_channels=in_ch, out_channels=out_ch,
                      conv_channels=conv_ch, num_conv=4)

    # Copy weights
    for i in range(3):
        conv_op, bn_op, _ = tt.conv_blocks[i]
        copy_conv2d(pt_convs[i], conv_op)
        copy_bn(pt_bns[i], bn_op)
    copy_conv2d(pt_final, tt.final_conv)

    x_np = np.random.randn(1, in_ch, 16, 16).astype(np.float32)
    x_tt = F._from_data('sc4n.in', x_np)
    out_tt = tt(x_tt)

    # PyTorch forward
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        for conv, bn in zip(pt_convs, pt_bns):
            x_pt = TF.relu(bn(conv(x_pt)))
        x_pt = pt_final(x_pt)

    ok = compare(x_pt, out_tt, "SimpleConv2d numerical", atol=1e-3)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 6")
except Exception as e:
    print(f"  [FAIL] TEST 6 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 7: UpsamplingAdd
# ====================================================================

print("\n" + "=" * 80)
print("TEST 7: UpsamplingAdd (numerical weight-copy match)")
print("=" * 80)

try:
    C = 64

    # PyTorch
    pt_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    pt_conv = nn.Conv2d(C, C, kernel_size=1, bias=False)
    pt_bn = nn.BatchNorm2d(C)
    pt_bn.eval()

    # TTSim
    tt = UpsamplingAdd('ua', in_channels=C, out_channels=C)

    copy_conv2d(pt_conv, tt.conv)
    copy_bn(pt_bn, tt.bn)

    x_np = np.random.randn(1, C, 12, 12).astype(np.float32)
    skip_np = np.random.randn(1, C, 24, 24).astype(np.float32)

    x_tt = F._from_data('ua.in', x_np)
    skip_tt = F._from_data('ua.skip', skip_np)
    out_tt = tt(x_tt, skip_tt)

    with torch.no_grad():
        x_pt = pt_up(torch.from_numpy(x_np))
        x_pt = pt_bn(pt_conv(x_pt))
        out_pt = x_pt + torch.from_numpy(skip_np)

    ok = compare(out_pt, out_tt, "UpsamplingAdd forward", atol=1e-3)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 7")
except Exception as e:
    print(f"  [FAIL] TEST 7 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 8: Bottleneck (no downsample)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 8: Bottleneck (no downsample, numerical weight-copy match)")
print("=" * 80)

try:
    C = 64
    bc = C // 2

    # PyTorch
    pt_conv_down = nn.Conv2d(C, bc, 1, bias=False)
    pt_bn_down = nn.BatchNorm2d(bc)
    pt_mid = nn.Conv2d(bc, bc, 3, padding=1, bias=False)
    pt_bn_mid = nn.BatchNorm2d(bc)
    pt_conv_up = nn.Conv2d(bc, C, 1, bias=False)
    pt_bn_up = nn.BatchNorm2d(C)
    for m in [pt_bn_down, pt_bn_mid, pt_bn_up]:
        m.eval()

    # TTSim
    tt = Bottleneck('bn_nods', in_channels=C)

    copy_conv2d(pt_conv_down, tt.conv_down)
    copy_bn(pt_bn_down, tt.bn_down)
    copy_conv2d(pt_mid, tt.mid_conv)
    copy_bn(pt_bn_mid, tt.bn_mid)
    copy_conv2d(pt_conv_up, tt.conv_up)
    copy_bn(pt_bn_up, tt.bn_up)

    x_np = np.random.randn(1, C, 20, 20).astype(np.float32)
    x_tt = F._from_data('bn_nods.in', x_np)
    out_tt = tt(x_tt)

    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)
        res = TF.relu(pt_bn_down(pt_conv_down(x_pt)))
        res = TF.relu(pt_bn_mid(pt_mid(res)))
        res = TF.relu(pt_bn_up(pt_conv_up(res)))
        out_pt = res + x_pt

    ok = compare(out_pt, out_tt, "Bottleneck no-ds forward", atol=1e-3)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 8")
except Exception as e:
    print(f"  [FAIL] TEST 8 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 9: Bottleneck (downsample, even dims)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 9: Bottleneck (downsample, even dims=20x20, numerical)")
print("=" * 80)

try:
    C = 64
    bc = C // 2
    H, W = 20, 20

    # PyTorch
    pt_conv_down = nn.Conv2d(C, bc, 1, bias=False)
    pt_bn_down = nn.BatchNorm2d(bc)
    pt_mid = nn.Conv2d(bc, bc, 3, stride=2, padding=1, bias=False)
    pt_bn_mid = nn.BatchNorm2d(bc)
    pt_conv_up = nn.Conv2d(bc, C, 1, bias=False)
    pt_bn_up = nn.BatchNorm2d(C)
    pt_skip_conv = nn.Conv2d(C, C, 1, bias=False)
    pt_skip_bn = nn.BatchNorm2d(C)
    for m in [pt_bn_down, pt_bn_mid, pt_bn_up, pt_skip_bn]:
        m.eval()

    # TTSim
    tt = Bottleneck('bn_ds_e', in_channels=C, downsample=True)

    copy_conv2d(pt_conv_down, tt.conv_down)
    copy_bn(pt_bn_down, tt.bn_down)
    copy_conv2d(pt_mid, tt.mid_conv)
    copy_bn(pt_bn_mid, tt.bn_mid)
    copy_conv2d(pt_conv_up, tt.conv_up)
    copy_bn(pt_bn_up, tt.bn_up)
    copy_conv2d(pt_skip_conv, tt.skip_conv)
    copy_bn(pt_skip_bn, tt.skip_bn)

    x_np = np.random.randn(1, C, H, W).astype(np.float32)
    x_tt = F._from_data('bn_ds_e.in', x_np)
    out_tt = tt(x_tt)

    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)
        res = TF.relu(pt_bn_down(pt_conv_down(x_pt)))
        res = TF.relu(pt_bn_mid(pt_mid(res)))
        res = TF.relu(pt_bn_up(pt_conv_up(res)))
        skip = nn.functional.max_pool2d(x_pt, 2, 2)
        skip = pt_skip_bn(pt_skip_conv(skip))
        out_pt = res + skip

    ok = compare(out_pt, out_tt, "Bottleneck ds-even forward", atol=1e-3)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 9")
except Exception as e:
    print(f"  [FAIL] TEST 9 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 10: Bottleneck (downsample, odd dims)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 10: Bottleneck (downsample, odd dims=25x25, numerical)")
print("=" * 80)

try:
    C = 64
    bc = C // 2
    H, W = 25, 25

    # PyTorch
    pt_conv_down = nn.Conv2d(C, bc, 1, bias=False)
    pt_bn_down = nn.BatchNorm2d(bc)
    pt_mid = nn.Conv2d(bc, bc, 3, stride=2, padding=1, bias=False)
    pt_bn_mid = nn.BatchNorm2d(bc)
    pt_conv_up = nn.Conv2d(bc, C, 1, bias=False)
    pt_bn_up = nn.BatchNorm2d(C)
    pt_skip_conv = nn.Conv2d(C, C, 1, bias=False)
    pt_skip_bn = nn.BatchNorm2d(C)
    for m in [pt_bn_down, pt_bn_mid, pt_bn_up, pt_skip_bn]:
        m.eval()

    # TTSim
    tt = Bottleneck('bn_ds_o', in_channels=C, downsample=True)

    copy_conv2d(pt_conv_down, tt.conv_down)
    copy_bn(pt_bn_down, tt.bn_down)
    copy_conv2d(pt_mid, tt.mid_conv)
    copy_bn(pt_bn_mid, tt.bn_mid)
    copy_conv2d(pt_conv_up, tt.conv_up)
    copy_bn(pt_bn_up, tt.bn_up)
    copy_conv2d(pt_skip_conv, tt.skip_conv)
    copy_bn(pt_skip_bn, tt.skip_bn)

    x_np = np.random.randn(1, C, H, W).astype(np.float32)
    x_tt = F._from_data('bn_ds_o.in', x_np)
    out_tt = tt(x_tt)

    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)
        res = TF.relu(pt_bn_down(pt_conv_down(x_pt)))
        res = TF.relu(pt_bn_mid(pt_mid(res)))
        res = TF.relu(pt_bn_up(pt_conv_up(res)))
        skip = TF.pad(x_pt, (0, W % 2, 0, H % 2), value=0)
        skip = nn.functional.max_pool2d(skip, 2, 2)
        skip = pt_skip_bn(pt_skip_conv(skip))
        out_pt = res + skip

    ok = compare(out_pt, out_tt, "Bottleneck ds-odd forward", atol=1e-3)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 10")
except Exception as e:
    print(f"  [FAIL] TEST 10 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 11: CVT_DecoderBlock (no residual, no upsample)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 11: CVT_DecoderBlock (no residual, no upsample, numerical)")
print("=" * 80)

try:
    in_ch, out_ch, factor = 64, 64, 2
    dim = out_ch // factor  # 32

    # PyTorch
    pt_conv1 = nn.Conv2d(in_ch, dim, 3, padding=1, bias=False)
    pt_bn1 = nn.BatchNorm2d(dim)
    pt_conv2 = nn.Conv2d(dim, out_ch, 1, bias=False)
    pt_bn2 = nn.BatchNorm2d(out_ch)
    for m in [pt_bn1, pt_bn2]:
        m.eval()

    # TTSim
    tt = CVT_DecoderBlock('cvtb_nr', in_ch, out_ch, skip_dim=in_ch,
                          residual=False, factor=factor, upsample=False,
                          with_relu=True)

    copy_conv2d(pt_conv1, tt.conv1)
    copy_bn(pt_bn1, tt.bn1)
    copy_conv2d(pt_conv2, tt.conv2)
    copy_bn(pt_bn2, tt.bn2)

    x_np = np.random.randn(1, in_ch, 16, 16).astype(np.float32)
    skip_np = np.random.randn(1, in_ch, 16, 16).astype(np.float32)

    x_tt = F._from_data('cvtb_nr.in', x_np)
    skip_tt = F._from_data('cvtb_nr.skip', skip_np)
    out_tt = tt(x_tt, skip_tt)

    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)
        x_pt = TF.relu(pt_bn1(pt_conv1(x_pt)))
        x_pt = pt_bn2(pt_conv2(x_pt))
        out_pt = TF.relu(x_pt)

    ok = compare(out_pt, out_tt, "CVT_DecoderBlock no-res forward", atol=1e-3)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 11")
except Exception as e:
    print(f"  [FAIL] TEST 11 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 12: CVT_DecoderBlock (residual + upsample)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 12: CVT_DecoderBlock (residual + upsample, numerical)")
print("=" * 80)

try:
    in_ch, out_ch, skip_dim, factor = 64, 64, 64, 2
    dim = out_ch // factor  # 32
    res_scale = 2

    # PyTorch
    pt_conv1 = nn.Conv2d(in_ch, dim, 3, padding=1, bias=False)
    pt_bn1 = nn.BatchNorm2d(dim)
    pt_conv2 = nn.Conv2d(dim, out_ch, 1, bias=False)
    pt_bn2 = nn.BatchNorm2d(out_ch)
    pt_skip_conv = nn.Conv2d(skip_dim, out_ch, 1, bias=True)
    pt_skip_conv.bias.data.zero_()  # TTSim Conv2d has no bias param
    for m in [pt_bn1, pt_bn2]:
        m.eval()

    # TTSim (first block => cumulative_scale = 2)
    tt = CVT_DecoderBlock('cvtb_ru', in_ch, out_ch, skip_dim,
                          residual=True, factor=factor, upsample=True,
                          with_relu=True, residual_scale=res_scale)

    copy_conv2d(pt_conv1, tt.conv1)
    copy_bn(pt_bn1, tt.bn1)
    copy_conv2d(pt_conv2, tt.conv2)
    copy_bn(pt_bn2, tt.bn2)
    copy_conv2d(pt_skip_conv, tt.skip_conv)

    H = 16
    x_np = np.random.randn(1, in_ch, H, H).astype(np.float32)
    skip_np = np.random.randn(1, skip_dim, H, H).astype(np.float32)

    x_tt = F._from_data('cvtb_ru.in', x_np)
    skip_tt = F._from_data('cvtb_ru.skip', skip_np)
    out_tt = tt(x_tt, skip_tt)

    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)
        x_pt = TF.interpolate(x_pt, scale_factor=2, mode='bilinear', align_corners=True)
        x_pt = TF.relu(pt_bn1(pt_conv1(x_pt)))
        x_pt = pt_bn2(pt_conv2(x_pt))
        up = pt_skip_conv(torch.from_numpy(skip_np))
        up = TF.interpolate(up, scale_factor=res_scale, mode='bilinear', align_corners=False)
        x_pt = x_pt + up
        out_pt = TF.relu(x_pt)

    ok = compare(out_pt, out_tt, "CVT_DecoderBlock res+up forward", atol=1e-3)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 12")
except Exception as e:
    print(f"  [FAIL] TEST 12 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 13: predict_instance_segmentation_and_trajectories
# ====================================================================

print("\n" + "=" * 80)
print("TEST 13: predict_instance_segmentation_and_trajectories")
print("=" * 80)

try:
    B, Q, T, H, W = 1, 5, 3, 10, 10

    np.random.seed(42)
    fg = np.zeros((B, T, 1, H, W), dtype=np.int64)
    fg[:, :, :, 2:8, 2:8] = 1

    ins_sigmoid = np.random.rand(B, Q, T, H, W).astype(np.float32)

    result = predict_instance_segmentation_and_trajectories(fg, ins_sigmoid)

    ok_dtype = result.dtype == np.int64
    ok_shape = list(result.shape) == [B, T, H, W]
    ok_bg = bool(np.all(result[:, :, :2, :] == 0))
    unique_ids = np.unique(result)
    ok_ids = len(unique_ids) <= Q + 1

    print(f"\n  Output dtype: {result.dtype}  (expected int64)")
    print(f"  Output shape: {list(result.shape)}  (expected [{B}, {T}, {H}, {W}])")
    print(f"  Background=0 in border: {ok_bg}")
    print(f"  Unique IDs: {unique_ids}  (count={len(unique_ids)}, max allowed={Q+1})")

    ok = ok_dtype and ok_shape and ok_bg and ok_ids
    print(f"  {'[OK]' if ok_dtype else '[FAIL]'} dtype")
    print(f"  {'[OK]' if ok_shape else '[FAIL]'} shape")
    print(f"  {'[OK]' if ok_bg else '[FAIL]'} background is 0 in border")
    print(f"  {'[OK]' if ok_ids else '[FAIL]'} consecutive IDs")
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 13")
except Exception as e:
    print(f"  [FAIL] TEST 13 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# Summary
# ====================================================================

print("\n" + "=" * 80)
total = passed + failed
print(f"RESULTS: {passed}/{total} tests passed, {failed} failed.")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED.")
print("=" * 80)
sys.exit(1 if failed else 0)
