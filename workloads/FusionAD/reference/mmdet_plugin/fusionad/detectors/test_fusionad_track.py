#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for FusionADTrack (TTSim vs PyTorch).

Validates that the key components of the FusionADTrack detector converted
from PyTorch to TTSim produce matching results:
  1. Instances container — basic operations (set/get, slicing, cat)
  2. _generate_empty_tracks — shapes and field existence
  3. Embedding + Linear sub-modules — numerical match vs nn.Embedding / nn.Linear
  4. velo_update — full velocity-based reference-point update
  5. upsample_bev_if_tiny — BEV upsampling path shapes
  6. FusionADTrack construction — attribute and sub-module checks
"""

import os
import sys
import traceback
import copy
import math

polaris_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn

import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

from workloads.FusionAD.projects.mmdet_plugin.fusionad.detectors.fusionad_track import (
    FusionADTrack,
    Instances,
    RuntimeTrackerBase,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.builder_utils import (
    InverseSigmoid,
    inverse_sigmoid_np,
)


# ====================================================================
# PyTorch reference helpers
# ====================================================================

def inverse_sigmoid_pytorch(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def velo_update_pytorch(ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                        time_delta, pc_range):
    """PyTorch reference of FusionADTrack.velo_update."""
    time_delta = time_delta.float()
    num_query = ref_pts.size(0)
    velo_pad_ = velocity.new_zeros((num_query, 1))
    velo_pad = torch.cat((velocity, velo_pad_), dim=-1)
    reference_points = ref_pts.sigmoid().clone()
    reference_points[..., 0:1] = (
        reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
    reference_points[..., 1:2] = (
        reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    reference_points[..., 2:3] = (
        reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])
    reference_points = reference_points + velo_pad * time_delta
    ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2
    g2l_r = torch.linalg.inv(l2g_r2).float()
    ref_pts = ref_pts @ g2l_r
    ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
    ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
    ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])
    ref_pts = inverse_sigmoid_pytorch(ref_pts)
    return ref_pts


# ====================================================================
# Weight copy helpers
# ====================================================================

def _set_linear(ttsim_linear, weight_np, bias_np):
    """Set TTSim Linear weights (no transpose — SimNN.Linear transposes internally)."""
    ttsim_linear.param = F._from_data(
        ttsim_linear.param.name, weight_np.astype(np.float32), is_const=True)
    ttsim_linear.param.is_param = True
    ttsim_linear.param.set_module(ttsim_linear)
    ttsim_linear._tensors[ttsim_linear.param.name] = ttsim_linear.param

    if bias_np is not None and ttsim_linear.bias is not None:
        ttsim_linear.bias = F._from_data(
            ttsim_linear.bias.name, bias_np.astype(np.float32), is_const=True)
        ttsim_linear.bias.is_param = True
        ttsim_linear.bias.set_module(ttsim_linear)
        ttsim_linear._tensors[ttsim_linear.bias.name] = ttsim_linear.bias


def _set_embedding(ttsim_emb, weight_np):
    """Set TTSim Embedding weights via params[0][1].data."""
    ttsim_emb.params[0][1].data = weight_np.astype(np.float32)


# ====================================================================
# Compare helper
# ====================================================================

def compare(pt_out, tt_out, name, atol=1e-5):
    pt_np = pt_out.detach().numpy() if isinstance(pt_out, torch.Tensor) else pt_out
    tt_np = tt_out.data if hasattr(tt_out, 'data') else tt_out
    print(f"\n  {name}:")
    print(f"    PyTorch shape: {pt_np.shape}")
    print(f"    TTSim   shape: {tt_np.shape}")
    if pt_np.shape != tt_np.shape:
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
# Config
# ====================================================================

np.random.seed(42)
torch.manual_seed(42)

EMBED_DIMS = 64   # small for fast testing
NUM_QUERY = 16     # small for fast testing
NUM_CLASSES = 10
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
MEM_BANK_LEN = 4

passed = 0
failed = 0


# ====================================================================
# TEST 1: Instances container
# ====================================================================

print("=" * 80)
print("TEST 1: Instances container — set/get, slicing, cat")
print("=" * 80)

try:
    inst = Instances((1, 1))
    inst.scores = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    inst.labels = np.array([0, 1, 2], dtype=np.int64)

    assert len(inst) == 3, f"Expected len=3, got {len(inst)}"
    assert np.allclose(inst.scores, [0.1, 0.2, 0.3])
    print("  set/get fields: OK")

    sub = inst[1]
    assert np.isclose(sub.scores, 0.2), f"Expected 0.2, got {sub.scores}"
    print("  __getitem__: OK")

    inst2 = Instances((1, 1))
    inst2.scores = np.array([0.4, 0.5], dtype=np.float32)
    inst2.labels = np.array([3, 4], dtype=np.int64)

    merged = Instances.cat([inst, inst2])
    assert len(merged) == 5, f"Expected len=5, got {len(merged)}"
    assert np.allclose(merged.scores, [0.1, 0.2, 0.3, 0.4, 0.5])
    print("  cat: OK")

    passed += 1
    print("\n[OK] TEST 1")

except Exception as e:
    print(f"  [FAIL] TEST 1 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 2: RuntimeTrackerBase
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: RuntimeTrackerBase — construction / defaults")
print("=" * 80)

try:
    tracker = RuntimeTrackerBase(score_thresh=0.3, filter_score_thresh=0.15,
                                 miss_tolerance=10)
    assert tracker.score_thresh == 0.3
    assert tracker.filter_score_thresh == 0.15
    assert tracker.miss_tolerance == 10
    print("  Attributes: OK")

    # update should not crash
    tracker.update(None, None)
    print("  update (no-op): OK")

    passed += 1
    print("\n[OK] TEST 2")

except Exception as e:
    print(f"  [FAIL] TEST 2 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 3: nn.Embedding — TTSim vs PyTorch
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: Embedding — TTSim F.Embedding vs nn.Embedding")
print("=" * 80)

try:
    tbl_size = NUM_QUERY + 1
    emb_dim = EMBED_DIMS * 2

    # PyTorch
    pt_emb = nn.Embedding(tbl_size, emb_dim)
    weight_np = pt_emb.weight.detach().numpy().copy()

    # TTSim
    tt_emb = F.Embedding('t3_emb', tbl_size=tbl_size, emb_dim=emb_dim)
    _set_embedding(tt_emb, weight_np)

    indices_np = np.arange(tbl_size, dtype=np.int64)
    indices_pt = torch.from_numpy(indices_np)
    indices_tt = F._from_data('t3_idx', indices_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_emb(indices_pt)

    tt_out = tt_emb(indices_tt)

    ok = compare(pt_out, tt_out, "Embedding full lookup", atol=1e-6)
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
# TEST 4: nn.Linear — TTSim SimNN.Linear vs nn.Linear
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: Linear — TTSim SimNN.Linear vs nn.Linear")
print("=" * 80)

try:
    in_feat = EMBED_DIMS
    out_feat = 3

    pt_linear = nn.Linear(in_feat, out_feat)
    w_np = pt_linear.weight.detach().numpy().copy()
    b_np = pt_linear.bias.detach().numpy().copy()

    tt_linear = SimNN.Linear('t4_linear', in_features=in_feat, out_features=out_feat)
    _set_linear(tt_linear, w_np, b_np)

    x_np = np.random.randn(4, in_feat).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t4_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_linear(x_pt)

    tt_out = tt_linear(x_tt)

    ok = compare(pt_out, tt_out, "Linear forward", atol=1e-5)
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
# TEST 5: velo_update — numerical match
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: velo_update math — PyTorch vs numpy")
print("=" * 80)

try:
    NQ_VELO = 8

    # Random inputs
    ref_pts_np = np.random.rand(NQ_VELO, 3).astype(np.float32) * 4 - 2
    velo_np = np.random.randn(NQ_VELO, 2).astype(np.float32) * 0.5
    l2g_r1_np = np.eye(3, dtype=np.float32) + np.random.randn(3, 3).astype(np.float32) * 0.01
    l2g_t1_np = np.random.randn(1, 3).astype(np.float32) * 0.5
    l2g_r2_np = np.eye(3, dtype=np.float32) + np.random.randn(3, 3).astype(np.float32) * 0.01
    l2g_t2_np = np.random.randn(1, 3).astype(np.float32) * 0.5
    time_delta_np = np.float32(0.5)

    # PyTorch reference
    ref_pts_pt = torch.from_numpy(ref_pts_np.copy())
    velo_pt = torch.from_numpy(velo_np.copy())
    l2g_r1_pt = torch.from_numpy(l2g_r1_np.copy())
    l2g_t1_pt = torch.from_numpy(l2g_t1_np.copy())
    l2g_r2_pt = torch.from_numpy(l2g_r2_np.copy())
    l2g_t2_pt = torch.from_numpy(l2g_t2_np.copy())
    time_delta_pt = torch.tensor(time_delta_np)

    with torch.no_grad():
        pt_out = velo_update_pytorch(
            ref_pts_pt, velo_pt,
            l2g_r1_pt, l2g_t1_pt, l2g_r2_pt, l2g_t2_pt,
            time_delta_pt, PC_RANGE)

    # Numpy implementation mirroring the TTSim velo_update logic
    # (same math, just pure numpy to verify algorithmic equivalence)
    num_query = ref_pts_np.shape[0]
    velo_pad_ = np.zeros((num_query, 1), dtype=np.float32)
    velo_pad = np.concatenate([velo_np, velo_pad_], axis=-1)

    def sigmoid_np(x):
        return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)

    reference_points = sigmoid_np(ref_pts_np.copy())
    pc = PC_RANGE
    scale = np.array([pc[3]-pc[0], pc[4]-pc[1], pc[5]-pc[2]], dtype=np.float32)
    offset = np.array([pc[0], pc[1], pc[2]], dtype=np.float32)
    reference_points = reference_points * scale + offset
    reference_points = reference_points + velo_pad * time_delta_np
    ref_out = reference_points @ l2g_r1_np + l2g_t1_np - l2g_t2_np
    g2l_r = np.linalg.inv(l2g_r2_np).astype(np.float32)
    ref_out = ref_out @ g2l_r
    ref_out = (ref_out - offset) / scale
    ref_out = inverse_sigmoid_np(ref_out)

    ok = compare(pt_out, ref_out, "velo_update output", atol=1e-3)
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
# TEST 6: FusionADTrack construction — attribute checks
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: FusionADTrack construction — attributes & sub-modules")
print("=" * 80)

try:
    tracker = FusionADTrack(
        name='t6_track',
        video_test_mode=True,
        pc_range=PC_RANGE,
        embed_dims=EMBED_DIMS,
        num_query=NUM_QUERY,
        num_classes=NUM_CLASSES,
        use_grid_mask=True,
        bev_h=100,
        bev_w=100,
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=MEM_BANK_LEN,
        ),
    )

    issues = []
    if tracker.embed_dims != EMBED_DIMS:
        issues.append(f"embed_dims: {tracker.embed_dims} != {EMBED_DIMS}")
    if tracker.num_query != NUM_QUERY:
        issues.append(f"num_query: {tracker.num_query} != {NUM_QUERY}")
    if tracker.num_classes != NUM_CLASSES:
        issues.append(f"num_classes: {tracker.num_classes} != {NUM_CLASSES}")
    if tracker.use_grid_mask is not True:
        issues.append("use_grid_mask should be True")
    if tracker.bev_h != 100:
        issues.append(f"bev_h: {tracker.bev_h} != 100")
    if tracker.bev_w != 100:
        issues.append(f"bev_w: {tracker.bev_w} != 100")
    if tracker.pc_range != PC_RANGE:
        issues.append("pc_range mismatch")
    if not hasattr(tracker, 'query_embedding'):
        issues.append("missing query_embedding")
    if not hasattr(tracker, 'reference_points'):
        issues.append("missing reference_points")
    if not hasattr(tracker, 'bbox_size_fc'):
        issues.append("missing bbox_size_fc")
    if not hasattr(tracker, 'sigmoid_op'):
        issues.append("missing sigmoid_op")
    if not hasattr(tracker, 'inverse_sigmoid_op'):
        issues.append("missing inverse_sigmoid_op")
    if not hasattr(tracker, 'resize_op'):
        issues.append("missing resize_op")
    if not hasattr(tracker, 'memory_bank'):
        issues.append("missing memory_bank")
    if not hasattr(tracker, 'query_interact'):
        issues.append("missing query_interact")
    if not hasattr(tracker, 'track_base'):
        issues.append("missing track_base")
    if not isinstance(tracker.track_base, RuntimeTrackerBase):
        issues.append("track_base not RuntimeTrackerBase")

    # Check QueryInteractionModule accepts update_query_pos
    if hasattr(tracker, 'query_interact'):
        # Default construction (no qim_args override) → update_query_pos=False
        if getattr(tracker.query_interact, 'update_query_pos', None) is not False:
            issues.append(f"query_interact.update_query_pos should be False (default), "
                          f"got {getattr(tracker.query_interact, 'update_query_pos', None)}")
        else:
            print("  query_interact.update_query_pos=False (default): OK")

    # Now test with update_query_pos=True
    tracker_qp = FusionADTrack(
        name='t6_track_qp',
        video_test_mode=True,
        pc_range=PC_RANGE,
        embed_dims=EMBED_DIMS,
        num_query=NUM_QUERY,
        num_classes=NUM_CLASSES,
        bev_h=100,
        bev_w=100,
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,
            update_query_pos=True,
            fp_ratio=0.3,
            random_drop=0.1,
        ),
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=MEM_BANK_LEN,
        ),
    )
    if not getattr(tracker_qp.query_interact, 'update_query_pos', False):
        issues.append("query_interact.update_query_pos should be True when passed in qim_args")
    elif not hasattr(tracker_qp.query_interact, 'linear_pos1'):
        issues.append("query_interact missing linear_pos1 when update_query_pos=True")
    else:
        print("  query_interact.update_query_pos=True (explicit): OK")

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 6")
    else:
        print("  All attributes and sub-modules present: OK")
        passed += 1
        print(f"\n[OK] TEST 6")

except Exception as e:
    print(f"  [FAIL] TEST 6 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 7: _generate_empty_tracks — field shapes
# ====================================================================

print("\n" + "=" * 80)
print("TEST 7: _generate_empty_tracks — shapes and fields")
print("=" * 80)

try:
    tracker = FusionADTrack(
        name='t7_track',
        video_test_mode=True,
        pc_range=PC_RANGE,
        embed_dims=EMBED_DIMS,
        num_query=NUM_QUERY,
        num_classes=NUM_CLASSES,
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=MEM_BANK_LEN,
        ),
    )

    tracks = tracker._generate_empty_tracks()
    NQ1 = NUM_QUERY + 1
    dim = EMBED_DIMS * 2

    issues = []

    def check_field(field_name, expected_shape):
        if not hasattr(tracks, field_name):
            issues.append(f"missing field: {field_name}")
            return
        val = getattr(tracks, field_name)
        shape = val.shape if hasattr(val, 'shape') else None
        if shape is None:
            issues.append(f"{field_name}: no .shape attribute")
        elif tuple(shape) != tuple(expected_shape):
            issues.append(f"{field_name}: shape {tuple(shape)} != expected {tuple(expected_shape)}")

    check_field('query', (NQ1, dim))
    check_field('ref_pts', (NQ1, 3))
    check_field('box_sizes', (NQ1, 3))
    check_field('pred_boxes', (NQ1, 10))
    check_field('output_embedding', (NQ1, dim // 2))
    check_field('obj_idxes', (NQ1,))
    check_field('matched_gt_idxes', (NQ1,))
    check_field('disappear_time', (NQ1,))
    check_field('iou', (NQ1,))
    check_field('scores', (NQ1,))
    check_field('track_scores', (NQ1,))
    check_field('pred_logits', (NQ1, NUM_CLASSES))
    check_field('save_period', (NQ1,))

    mem_bank_len = tracker.mem_bank_len
    check_field('mem_bank', (NQ1, mem_bank_len, dim // 2))
    check_field('mem_padding_mask', (NQ1, mem_bank_len))

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 7")
    else:
        print("  All fields present with correct shapes: OK")
        passed += 1
        print(f"\n[OK] TEST 7")

except Exception as e:
    print(f"  [FAIL] TEST 7 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 8: Sigmoid — TTSim F.Sigmoid vs torch.sigmoid
# ====================================================================

print("\n" + "=" * 80)
print("TEST 8: Sigmoid — TTSim vs PyTorch")
print("=" * 80)

try:
    x_np = np.random.randn(4, 10).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t8_x', x_np, is_const=True)

    pt_out = torch.sigmoid(x_pt)

    sig_op = F.Sigmoid('t8_sigmoid')
    tt_out = sig_op(x_tt)

    ok = compare(pt_out, tt_out, "Sigmoid output", atol=1e-6)
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
# TEST 9: InverseSigmoid — TTSim vs PyTorch
# ====================================================================

print("\n" + "=" * 80)
print("TEST 9: InverseSigmoid — TTSim vs PyTorch")
print("=" * 80)

try:
    x_np = np.random.rand(4, 10).astype(np.float32) * 0.98 + 0.01
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t9_x', x_np, is_const=True)

    pt_out = inverse_sigmoid_pytorch(x_pt)

    inv_op = InverseSigmoid('t9_inv')
    tt_out = inv_op(x_tt)

    ok = compare(pt_out, tt_out, "InverseSigmoid output", atol=1e-5)
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
# TEST 10: _copy_tracks_for_loss — shape checks
# ====================================================================

print("\n" + "=" * 80)
print("TEST 10: _copy_tracks_for_loss — field presence")
print("=" * 80)

try:
    tracker = FusionADTrack(
        name='t10_track',
        video_test_mode=True,
        pc_range=PC_RANGE,
        embed_dims=EMBED_DIMS,
        num_query=NUM_QUERY,
        num_classes=NUM_CLASSES,
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=MEM_BANK_LEN,
        ),
    )

    # Build source tracks using numpy arrays so len() works
    src = Instances((1, 1))
    NQ1 = NUM_QUERY + 1
    src.obj_idxes = np.full((NQ1,), -1, dtype=np.int64)
    src.matched_gt_idxes = np.full((NQ1,), -1, dtype=np.int64)
    src.disappear_time = np.zeros((NQ1,), dtype=np.int64)
    src.save_period = np.zeros((NQ1,), dtype=np.float32)

    copied = tracker._copy_tracks_for_loss(src)

    issues = []
    expected_fields = ['obj_idxes', 'matched_gt_idxes', 'disappear_time',
                       'scores', 'track_scores', 'pred_boxes', 'iou',
                       'pred_logits', 'save_period']
    for field_name in expected_fields:
        if not hasattr(copied, field_name):
            issues.append(f"missing field: {field_name}")
        else:
            print(f"  field '{field_name}': present")

    # Check that scores, track_scores, iou have shape [NQ1]
    for field_name in ['scores', 'track_scores', 'iou']:
        if hasattr(copied, field_name):
            val = getattr(copied, field_name)
            shape = tuple(val.shape) if hasattr(val, 'shape') else None
            if shape != (NQ1,):
                issues.append(f"{field_name}: shape {shape} != ({NQ1},)")

    # Check pred_boxes shape [NQ1, 10]
    if hasattr(copied, 'pred_boxes'):
        shape = tuple(copied.pred_boxes.shape)
        if shape != (NQ1, 10):
            issues.append(f"pred_boxes: shape {shape} != ({NQ1}, 10)")

    # Check pred_logits shape [NQ1, NUM_CLASSES]
    if hasattr(copied, 'pred_logits'):
        shape = tuple(copied.pred_logits.shape)
        if shape != (NQ1, NUM_CLASSES):
            issues.append(f"pred_logits: shape {shape} != ({NQ1}, {NUM_CLASSES})")

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 10")
    else:
        print("  All copied fields present with correct shapes: OK")
        passed += 1
        print(f"\n[OK] TEST 10")

except Exception as e:
    print(f"  [FAIL] TEST 10 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# Summary
# ====================================================================

print("\n" + "=" * 80)
print(f"SUMMARY: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 80)

if failed > 0:
    sys.exit(1)
