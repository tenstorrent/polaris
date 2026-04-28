#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for FusionAD end-to-end (TTSim vs PyTorch).

Validates the key components of the FusionAD detector converted
from PyTorch to TTSim with numerical + shape comparison:

  1. pop_elem_in_result - numerical equivalence (TTSim vs PyTorch ref)
  2. pop_elem_in_result - edge cases (empty, no-match, multi-suffix)
  3. FusionAD construction - attributes, sub-modules, inherited shapes
  4. FusionAD properties - with_*_head toggling (None vs dummy)
  5. FusionAD inheritance - MRO, inherited methods
  6. __call__ input validation - TypeError on non-list img_metas
  7. ego_info tensor - F._from_data vs torch.from_numpy shape + value match
  8. can_bus delta - numerical match PyTorch vs TTSim (2-frame sequence)
  9. Post-processing pipeline - dict assembly numerical match
 10. Embedding + Linear - forward pass numerical match (PT vs TT)
 11. Polaris-mode construction (use_lidar=True) - backbone + BEV encoder + LiDAR
 12. create_input_tensors - img + voxels shapes (use_lidar=True)
 13. Polaris-mode construction (use_lidar=False) - no voxels, no pts_backbone
"""

import os
import sys
import traceback
import copy

polaris_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

from workloads.FusionAD.projects.mmdet_plugin.fusionad.detectors.fusionad_e2e import (
    FusionAD,
    pop_elem_in_result,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.detectors.fusionad_track import (
    FusionADTrack,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.encoder import (
    BEVFormerEncoder,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.resnet import (
    ResNetBackbone,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.fpn import FPN
from workloads.FusionAD.projects.mmdet_plugin.models.backbones.sparse_encoder_hd import (
    SparseEncoderHD,
)


# ====================================================================
# PyTorch reference helpers
# ====================================================================

def pop_elem_in_result_pytorch(task_result, pop_list=None):
    """PyTorch reference - identical logic to original torch version."""
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result


def ego_info_pytorch(can_bus_np):
    """PyTorch reference for ego_info creation (from forward_test)."""
    ego = torch.from_numpy(can_bus_np).float()
    ego[:7] = 0
    return ego


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
    """Compare PyTorch and TTSim outputs - shape + numerical."""
    pt_np = pt_out.detach().numpy() if isinstance(pt_out, torch.Tensor) else pt_out
    tt_np = tt_out.data if hasattr(tt_out, 'data') and isinstance(tt_out.data, np.ndarray) else tt_out
    print(f"\n  {name}:")
    print(f"    PyTorch shape: {pt_np.shape}")
    print(f"    TTSim   shape: {tt_np.shape}")
    if pt_np.shape != tt_np.shape:
        print(f"    [FAIL] Shape mismatch!")
        return False
    diff = np.abs(pt_np.astype(np.float64) - tt_np.astype(np.float64))
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

EMBED_DIMS = 64
NUM_QUERY = 16
NUM_CLASSES = 10
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
MEM_BANK_LEN = 4

passed = 0
failed = 0


# ====================================================================
# TEST 1: pop_elem_in_result - TTSim vs PyTorch numerical match
# ====================================================================

print("=" * 80)
print("TEST 1: pop_elem_in_result - TTSim vs PyTorch numerical match")
print("=" * 80)

try:
    d_common = {
        'track_query': np.random.randn(5, 64).astype(np.float32),
        'track_query_pos': np.random.randn(5, 64).astype(np.float32),
        'sdc_embedding': np.random.randn(1, 64).astype(np.float32),
        'bev_embed': np.random.randn(100, 256).astype(np.float32),
        'bev_pos': np.random.randn(100, 256).astype(np.float32),
        'track_query_embeddings': np.random.randn(5, 128).astype(np.float32),
        'scores': np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32),
        'pred_boxes': np.random.randn(5, 10).astype(np.float32),
        'labels': np.array([1, 2, 3, 4, 5], dtype=np.int64),
    }

    pop_list = ['bev_embed', 'bev_pos']
    d_ttsim = copy.deepcopy(d_common)
    d_pt = copy.deepcopy(d_common)

    result_ttsim = pop_elem_in_result(d_ttsim, pop_list=pop_list)
    result_pt = pop_elem_in_result_pytorch(d_pt, pop_list=pop_list)

    issues = []
    if set(result_ttsim.keys()) != set(result_pt.keys()):
        issues.append(f"Key mismatch: ttsim={sorted(result_ttsim.keys())}, pt={sorted(result_pt.keys())}")
    else:
        print(f"  Remaining keys: {sorted(result_ttsim.keys())}")

    for k in result_ttsim:
        tt_val = result_ttsim[k]
        pt_val = result_pt[k]
        if isinstance(tt_val, np.ndarray) and isinstance(pt_val, np.ndarray):
            if tt_val.shape != pt_val.shape:
                issues.append(f"'{k}' shape mismatch: {tt_val.shape} vs {pt_val.shape}")
            elif not np.allclose(tt_val, pt_val, atol=1e-7):
                issues.append(f"'{k}' value mismatch")
            else:
                print(f"  '{k}': shape {tt_val.shape} - match OK")

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 1")
    else:
        passed += 1
        print(f"\n[OK] TEST 1")

except Exception as e:
    print(f"  [FAIL] TEST 1 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 2: pop_elem_in_result - edge cases
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: pop_elem_in_result - edge cases (empty, no-match, multi-suffix)")
print("=" * 80)

try:
    issues = []

    # Case A: empty dict
    result_a = pop_elem_in_result({})
    if len(result_a) != 0:
        issues.append(f"Empty dict: expected len 0, got {len(result_a)}")
    else:
        print("  Empty dict -> empty result: OK")

    # Case B: no matching query/embedding keys, no pop_list
    d_b = {'score': 1.0, 'label': 2, 'pred_box': np.zeros(10)}
    result_b = pop_elem_in_result(copy.deepcopy(d_b))
    if set(result_b.keys()) != set(d_b.keys()):
        issues.append(f"No-match case: keys should be unchanged, got {sorted(result_b.keys())}")
    else:
        print("  No matching keys -> all keys survived: OK")

    # Case C: pop_list with keys that don't exist (should not error)
    d_c = {'score': 1.0, 'track_query': np.zeros(3)}
    result_c = pop_elem_in_result(copy.deepcopy(d_c), pop_list=['nonexistent_key'])
    if 'track_query' in result_c:
        issues.append("track_query should have been removed")
    if 'score' not in result_c:
        issues.append("score should have survived")
    if 'nonexistent_key' in result_c:
        issues.append("nonexistent_key should not appear")
    else:
        print("  Pop non-existent key (no error): OK")

    # Case D: multiple query/embedding suffixes
    d_d = {
        'track_query': 1, 'det_query': 2, 'seg_query_pos': 3,
        'obj_embedding': 4, 'result': 5,
    }
    result_d = pop_elem_in_result(copy.deepcopy(d_d))
    expected_keys = {'result'}
    if set(result_d.keys()) != expected_keys:
        issues.append(f"Multi-suffix: expected {expected_keys}, got {set(result_d.keys())}")
    else:
        print("  Multiple query/embedding suffixes removed: OK")

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 2")
    else:
        passed += 1
        print(f"\n[OK] TEST 2")

except Exception as e:
    print(f"  [FAIL] TEST 2 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 3: FusionAD construction - attributes, sub-modules, shapes
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: FusionAD construction - attributes, sub-modules, inherited shapes")
print("=" * 80)

try:
    model = FusionAD(
        name='t3_e2e',
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
        seg_head_module=None,
        motion_head_module=None,
        occ_head_module=None,
        planning_head_module=None,
    )

    issues = []
    # Inherited attributes from FusionADTrack
    if model.embed_dims != EMBED_DIMS:
        issues.append(f"embed_dims: {model.embed_dims} != {EMBED_DIMS}")
    if model.num_query != NUM_QUERY:
        issues.append(f"num_query: {model.num_query} != {NUM_QUERY}")
    if model.num_classes != NUM_CLASSES:
        issues.append(f"num_classes: {model.num_classes} != {NUM_CLASSES}")
    if model.bev_h != 100:
        issues.append(f"bev_h: {model.bev_h} != 100")
    if model.bev_w != 100:
        issues.append(f"bev_w: {model.bev_w} != 100")
    if model.pc_range != PC_RANGE:
        issues.append("pc_range mismatch")

    # FusionAD-specific attributes (None when not provided)
    if model.seg_head is not None:
        issues.append("seg_head should be None")
    if model.motion_head is not None:
        issues.append("motion_head should be None")
    if model.occ_head is not None:
        issues.append("occ_head should be None")
    if model.planning_head is not None:
        issues.append("planning_head should be None")

    # Inherited sub-modules from FusionADTrack
    if not hasattr(model, 'query_embedding'):
        issues.append("missing query_embedding (inherited)")
    if not hasattr(model, 'reference_points'):
        issues.append("missing reference_points (inherited)")
    if not hasattr(model, 'memory_bank'):
        issues.append("missing memory_bank (inherited)")

    # Verify query_embedding and reference_points are callable (shape tested in TEST 10)
    if not callable(model.query_embedding):
        issues.append("query_embedding is not callable")
    else:
        print(f"  query_embedding: callable OK (type={type(model.query_embedding).__name__})")
    if not callable(model.reference_points):
        issues.append("reference_points is not callable")
    else:
        print(f"  reference_points: callable OK (type={type(model.reference_points).__name__})")

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 3")
    else:
        print("  All attributes and sub-modules present: OK")
        passed += 1
        print(f"\n[OK] TEST 3")

except Exception as e:
    print(f"  [FAIL] TEST 3 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 4: FusionAD properties - with_*_head toggling (None vs dummy)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: FusionAD properties - with_*_head toggling (None vs dummy)")
print("=" * 80)

try:
    # Create a minimal dummy head (just a SimNN.Module that is not None)
    class DummyHead(SimNN.Module):
        def __init__(self, head_name):
            super().__init__()
            self.name = head_name
        def __call__(self, *args, **kwargs):
            return {}
        def analytical_param_count(self):
            return 0

    # Model with no heads
    model_none = FusionAD(
        name='t4a_e2e',
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

    # Model with seg + motion heads
    dummy_seg = DummyHead('dummy_seg')
    dummy_motion = DummyHead('dummy_motion')

    model_heads = FusionAD(
        name='t4b_e2e',
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
        seg_head_module=dummy_seg,
        motion_head_module=dummy_motion,
        occ_head_module=None,
        planning_head_module=None,
    )

    issues = []
    # All-None model
    if model_none.with_seg_head is not False:
        issues.append(f"with_seg_head (none): expected False, got {model_none.with_seg_head}")
    if model_none.with_motion_head is not False:
        issues.append(f"with_motion_head (none): expected False, got {model_none.with_motion_head}")
    if model_none.with_occ_head is not False:
        issues.append(f"with_occ_head (none): expected False, got {model_none.with_occ_head}")
    if model_none.with_planning_head is not False:
        issues.append(f"with_planning_head (none): expected False, got {model_none.with_planning_head}")

    # Heads model
    if model_heads.with_seg_head is not True:
        issues.append(f"with_seg_head (heads): expected True, got {model_heads.with_seg_head}")
    if model_heads.with_motion_head is not True:
        issues.append(f"with_motion_head (heads): expected True, got {model_heads.with_motion_head}")
    if model_heads.with_occ_head is not False:
        issues.append(f"with_occ_head (heads): expected False, got {model_heads.with_occ_head}")
    if model_heads.with_planning_head is not False:
        issues.append(f"with_planning_head (heads): expected False, got {model_heads.with_planning_head}")

    # Verify head references
    if model_heads.seg_head is not dummy_seg:
        issues.append("seg_head reference mismatch")
    if model_heads.motion_head is not dummy_motion:
        issues.append("motion_head reference mismatch")

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 4")
    else:
        print("  All-None: with_seg=False, with_motion=False, with_occ=False, with_plan=False: OK")
        print("  Dummy heads: with_seg=True, with_motion=True, with_occ=False, with_plan=False: OK")
        print("  Head references correct: OK")
        passed += 1
        print(f"\n[OK] TEST 4")

except Exception as e:
    print(f"  [FAIL] TEST 4 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 5: FusionAD inheritance - MRO, inherited methods
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: FusionAD inheritance - MRO, inherited methods")
print("=" * 80)

try:
    issues = []

    # Check MRO contains FusionADTrack (by name, due to dual import paths)
    bases = [b.__name__ for b in FusionAD.__mro__]
    if 'FusionADTrack' not in bases:
        issues.append("FusionAD MRO does not contain FusionADTrack")

    model = FusionAD(
        name='t5_e2e',
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

    # Check instance MRO
    instance_class_names = [c.__name__ for c in type(model).__mro__]
    if 'FusionADTrack' not in instance_class_names:
        issues.append("FusionAD instance MRO does not contain FusionADTrack")
    if 'Module' not in instance_class_names:
        issues.append("FusionAD instance MRO does not contain Module (SimNN.Module)")

    # Verify inherited methods exist
    for method_name in ['_generate_empty_tracks', 'velo_update',
                        '_copy_tracks_for_loss', 'upsample_bev_if_tiny',
                        'extract_img_feat', 'extract_feat',
                        'get_history_bev', 'get_bevs']:
        if not hasattr(model, method_name):
            issues.append(f"missing inherited method: {method_name}")

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 5")
    else:
        print("  FusionADTrack in MRO: OK")
        print("  instance MRO contains FusionADTrack: OK")
        print("  instance MRO contains Module: OK")
        print("  All inherited methods present: OK")
        passed += 1
        print(f"\n[OK] TEST 5")

except Exception as e:
    print(f"  [FAIL] TEST 5 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 6: __call__ input validation - TypeError on non-list img_metas
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: __call__ input validation - TypeError on non-list img_metas")
print("=" * 80)

try:
    model = FusionAD(
        name='t6_e2e',
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

    caught = False
    try:
        model(img=None, img_metas="not_a_list", points=[[None]])
    except TypeError as te:
        caught = True
        print(f"  Caught expected TypeError: {te}")

    if caught:
        passed += 1
        print(f"\n[OK] TEST 6")
    else:
        print("  [FAIL] Expected TypeError was not raised")
        failed += 1
        print(f"\n[FAIL] TEST 6")

except Exception as e:
    print(f"  [FAIL] TEST 6 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 7: ego_info tensor - F._from_data vs torch shape + value match
# ====================================================================

print("\n" + "=" * 80)
print("TEST 7: ego_info tensor - F._from_data vs torch.from_numpy comparison")
print("=" * 80)

try:
    can_bus_np = np.random.randn(18).astype(np.float32)

    # PyTorch reference
    pt_ego = ego_info_pytorch(can_bus_np.copy())

    # TTSim version (same logic as in FusionAD.__call__ planning stage)
    tt_ego_np = can_bus_np.copy()
    tt_ego_np[:7] = 0
    tt_ego = F._from_data('test_ego_info', tt_ego_np, is_const=True)

    ok = compare(pt_ego, tt_ego, "ego_info", atol=1e-7)

    if ok:
        passed += 1
        print(f"\n[OK] TEST 7")
    else:
        failed += 1
        print(f"\n[FAIL] TEST 7")

except Exception as e:
    print(f"  [FAIL] TEST 7 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 8: can_bus delta - 2-frame PyTorch vs TTSim numerical comparison
# ====================================================================

print("\n" + "=" * 80)
print("TEST 8: can_bus delta - 2-frame numerical comparison")
print("=" * 80)

try:
    model = FusionAD(
        name='t8_e2e',
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

    # --- TTSim path (uses model.prev_frame_info) ---
    can_bus_f1 = np.array(
        [10.0, 20.0, 30.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45.0],
        dtype=np.float64)
    can_bus_f2 = np.array(
        [15.0, 25.0, 32.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50.0],
        dtype=np.float64)

    pfi = model.prev_frame_info  # {'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}

    # Frame 1
    tt_cb1 = can_bus_f1.copy()
    if 'scene_A' != pfi['scene_token']:
        pfi['prev_bev'] = None
    pfi['scene_token'] = 'scene_A'
    tmp_pos1 = tt_cb1[:3].copy()
    tmp_angle1 = tt_cb1[-1].copy()
    if pfi['scene_token'] is None:
        tt_cb1[:3] = 0
        tt_cb1[-1] = 0
    else:
        tt_cb1[:3] -= pfi['prev_pos']
        tt_cb1[-1] -= pfi['prev_angle']
    pfi['prev_pos'] = tmp_pos1
    pfi['prev_angle'] = tmp_angle1

    # Frame 2
    tt_cb2 = can_bus_f2.copy()
    if 'scene_A' != pfi['scene_token']:
        pfi['prev_bev'] = None
    pfi['scene_token'] = 'scene_A'
    tmp_pos2 = tt_cb2[:3].copy()
    tmp_angle2 = tt_cb2[-1].copy()
    tt_cb2[:3] -= pfi['prev_pos']
    tt_cb2[-1] -= pfi['prev_angle']
    pfi['prev_pos'] = tmp_pos2
    pfi['prev_angle'] = tmp_angle2

    # --- PyTorch reference (same logic as original torch code) ---
    pt_prev_pos = 0
    pt_prev_angle = 0
    pt_scene_token = None

    # PT frame 1
    pt_cb1 = can_bus_f1.copy()
    if 'scene_A' != pt_scene_token:
        pass  # prev_bev = None
    pt_scene_token = 'scene_A'
    pt_tmp_pos1 = pt_cb1[:3].copy()
    pt_tmp_angle1 = pt_cb1[-1].copy()
    if pt_scene_token is None:
        pt_cb1[:3] = 0
        pt_cb1[-1] = 0
    else:
        pt_cb1[:3] -= pt_prev_pos
        pt_cb1[-1] -= pt_prev_angle
    pt_prev_pos = pt_tmp_pos1
    pt_prev_angle = pt_tmp_angle1

    # PT frame 2
    pt_cb2 = can_bus_f2.copy()
    pt_cb2[:3] -= pt_prev_pos
    pt_cb2[-1] -= pt_prev_angle

    # Compare frame 1
    ok1 = compare(
        torch.from_numpy(pt_cb1[:3].astype(np.float32)),
        F._from_data('f1_pos', tt_cb1[:3].astype(np.float32), is_const=True),
        "Frame 1 delta pos", atol=1e-7)

    ok2 = compare(
        torch.from_numpy(pt_cb2[:3].astype(np.float32)),
        F._from_data('f2_pos', tt_cb2[:3].astype(np.float32), is_const=True),
        "Frame 2 delta pos", atol=1e-7)

    ok3 = np.isclose(tt_cb1[-1], pt_cb1[-1], atol=1e-7)
    ok4 = np.isclose(tt_cb2[-1], pt_cb2[-1], atol=1e-7)
    print(f"\n  Frame 1 delta angle: TTSim={tt_cb1[-1]}, PT={pt_cb1[-1]} - {'OK' if ok3 else 'FAIL'}")
    print(f"  Frame 2 delta angle: TTSim={tt_cb2[-1]}, PT={pt_cb2[-1]} - {'OK' if ok4 else 'FAIL'}")

    if ok1 and ok2 and ok3 and ok4:
        passed += 1
        print(f"\n[OK] TEST 8")
    else:
        failed += 1
        print(f"\n[FAIL] TEST 8")

except Exception as e:
    print(f"  [FAIL] TEST 8 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 9: Post-processing pipeline - dict assembly numerical match
# ====================================================================

print("\n" + "=" * 80)
print("TEST 9: Post-processing pipeline - dict assembly numerical match")
print("=" * 80)

try:
    # Simulate a track result dict with typical keys
    track_result = {
        'scores': np.random.randn(5).astype(np.float32),
        'pred_boxes': np.random.randn(5, 10).astype(np.float32),
        'labels': np.array([1, 2, 3, 4, 5], dtype=np.int64),
        'track_query': np.random.randn(5, 64).astype(np.float32),
        'track_query_pos': np.random.randn(5, 64).astype(np.float32),
        'sdc_embedding': np.random.randn(1, 64).astype(np.float32),
        'track_query_embeddings': np.random.randn(5, 128).astype(np.float32),
        'prev_bev': np.random.randn(100, 256).astype(np.float32),
        'bev_pos': np.random.randn(100, 256).astype(np.float32),
        'bev_embed': np.random.randn(100, 256).astype(np.float32),
    }

    pop_track_list = [
        'prev_bev', 'bev_pos', 'bev_embed',
        'track_query_embeddings', 'sdc_embedding',
    ]

    # TTSim
    tt_result = copy.deepcopy(track_result)
    tt_result = pop_elem_in_result(tt_result, pop_track_list)

    # PyTorch reference
    pt_result = copy.deepcopy(track_result)
    pt_result = pop_elem_in_result_pytorch(pt_result, pop_track_list)

    issues = []
    if set(tt_result.keys()) != set(pt_result.keys()):
        issues.append(f"Key mismatch: tt={sorted(tt_result.keys())}, pt={sorted(pt_result.keys())}")
    else:
        print(f"  Remaining keys after post-processing: {sorted(tt_result.keys())}")

    for k in tt_result:
        tt_v = tt_result[k]
        pt_v = pt_result[k]
        if isinstance(tt_v, np.ndarray) and isinstance(pt_v, np.ndarray):
            if tt_v.shape != pt_v.shape:
                issues.append(f"'{k}' shape mismatch: {tt_v.shape} vs {pt_v.shape}")
            elif not np.allclose(tt_v, pt_v, atol=1e-7):
                issues.append(f"'{k}' value mismatch")
            else:
                print(f"  '{k}': shape {tt_v.shape} - match OK")

    # Simulate final result assembly (as done in __call__)
    result = [dict()]
    result[0]['token'] = 'sample_idx_0'
    result[0].update(tt_result)

    if 'token' not in result[0]:
        issues.append("token missing from assembled result")
    if 'scores' not in result[0]:
        issues.append("scores missing from assembled result")
    if 'track_query' in result[0]:
        issues.append("track_query should have been removed by pop_elem_in_result")

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 9")
    else:
        print("  Result assembly with token: OK")
        passed += 1
        print(f"\n[OK] TEST 9")

except Exception as e:
    print(f"  [FAIL] TEST 9 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 10: Embedding + Linear forward pass - PT vs TT weight copy
# ====================================================================

print("\n" + "=" * 80)
print("TEST 10: Embedding + Linear forward pass - PT vs TT numerical match")
print("=" * 80)

try:
    # ---- PyTorch reference ----
    pt_emb = nn.Embedding(NUM_QUERY + 1, EMBED_DIMS * 2)
    pt_linear = nn.Linear(EMBED_DIMS * 2, 3, bias=True)

    # ---- TTSim modules ----
    tt_emb = F.Embedding('t10_emb', tbl_size=NUM_QUERY + 1, emb_dim=EMBED_DIMS * 2)
    tt_linear = SimNN.Linear('t10_linear', EMBED_DIMS * 2, 3, bias=True)

    # ---- Copy weights PT -> TT ----
    _set_embedding(tt_emb, pt_emb.weight.detach().numpy())
    _set_linear(tt_linear,
                pt_linear.weight.detach().numpy(),
                pt_linear.bias.detach().numpy())

    # ---- Forward pass ----
    idx = np.array([0, 3, NUM_QUERY], dtype=np.int64)

    # PyTorch
    pt_emb_out = pt_emb(torch.from_numpy(idx))
    pt_lin_out = pt_linear(pt_emb_out)

    # TTSim
    tt_idx = F._from_data('t10_idx', idx, is_const=True)
    tt_emb_out = tt_emb(tt_idx)
    tt_lin_out = tt_linear(tt_emb_out)

    ok_emb = compare(pt_emb_out, tt_emb_out, "Embedding output", atol=1e-5)
    ok_lin = compare(pt_lin_out, tt_lin_out, "Linear output", atol=1e-4)

    if ok_emb and ok_lin:
        passed += 1
        print(f"\n[OK] TEST 10")
    else:
        failed += 1
        print(f"\n[FAIL] TEST 10")

except Exception as e:
    print(f"  [FAIL] TEST 10 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 11: Polaris-mode construction (use_lidar=True)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 11: Polaris-mode construction (use_lidar=True) - backbone + BEV encoder + LiDAR")
print("=" * 80)

try:
    cfg = dict(
        embed_dims=EMBED_DIMS,
        num_query=NUM_QUERY,
        num_classes=NUM_CLASSES,
        num_cams=6,
        bev_h=50,
        bev_w=50,
        pc_range=PC_RANGE,
        video_test_mode=True,
        queue_length=3,
        predict_steps=12,
        predict_modes=6,
        planning_steps=6,
        occ_n_future=4,
        bs=1,
        img_height=224,
        img_width=320,
        backbone_layers=[3, 4, 23, 3],
        backbone_out_indices=(1, 2, 3),
        num_bev_encoder_layers=2,
        # LiDAR settings
        use_lidar=True,
        lidar_in_channels=5,
        lidar_sparse_shape=[41, 64, 64],
    )

    model = FusionAD('t11_e2e', cfg)

    issues = []

    # Check image backbone exists and is ResNetBackbone
    if not hasattr(model, 'img_backbone') or model.img_backbone is None:
        issues.append("img_backbone is None or missing")
    elif type(model.img_backbone).__name__ != 'ResNetBackbone':
        issues.append(f"img_backbone type: {type(model.img_backbone).__name__}, expected ResNetBackbone")
    else:
        print(f"  img_backbone: {type(model.img_backbone).__name__} OK")

    # Check FPN neck
    if not hasattr(model, 'img_neck') or model.img_neck is None:
        issues.append("img_neck is None or missing")
    elif type(model.img_neck).__name__ != 'FPN':
        issues.append(f"img_neck type: {type(model.img_neck).__name__}, expected FPN")
    else:
        print(f"  img_neck: {type(model.img_neck).__name__} OK")

    # Check LiDAR backbone (SparseEncoderHD)
    if not hasattr(model, 'pts_backbone') or model.pts_backbone is None:
        issues.append("pts_backbone is None or missing (use_lidar=True)")
    elif type(model.pts_backbone).__name__ != 'SparseEncoderHD':
        issues.append(f"pts_backbone type: {type(model.pts_backbone).__name__}, expected SparseEncoderHD")
    else:
        print(f"  pts_backbone: {type(model.pts_backbone).__name__} OK")
        if model.pts_backbone.sparse_shape != [41, 64, 64]:
            issues.append(f"pts_backbone sparse_shape: {model.pts_backbone.sparse_shape}, expected [41, 64, 64]")
        if model.pts_backbone.output_channels != EMBED_DIMS:
            issues.append(f"pts_backbone output_channels: {model.pts_backbone.output_channels}, expected {EMBED_DIMS}")

    # Check BEV encoder (inside PerceptionTransformer)
    bev_enc = getattr(getattr(getattr(model, 'pts_bbox_head', None), 'transformer', None), 'encoder', None)
    if bev_enc is None:
        issues.append("bev_encoder is None or missing (pts_bbox_head.transformer.encoder)")
    elif type(bev_enc).__name__ != 'BEVFormerEncoder':
        issues.append(f"bev_encoder type: {type(bev_enc).__name__}, expected BEVFormerEncoder")
    else:
        print(f"  bev_encoder: {type(bev_enc).__name__} OK")
        if bev_enc.num_layers != 2:
            issues.append(f"bev_encoder num_layers: {bev_enc.num_layers}, expected 2")
        # Fusion mode: each layer should be BEVFormerFusionLayer with pts_cross_attn
        for i, layer in enumerate(bev_enc.layers):
            ltype = type(layer).__name__
            if ltype != 'BEVFormerFusionLayer':
                issues.append(f"bev_encoder.layer.{i} type: {ltype}, expected BEVFormerFusionLayer")
            elif 'pts_cross_attn' not in layer.operation_order:
                issues.append(f"bev_encoder.layer.{i} operation_order missing pts_cross_attn")

    # Check stored config
    if not model._use_lidar:
        issues.append("_use_lidar should be True")
    if model._lidar_in_channels != 5:
        issues.append(f"_lidar_in_channels: {model._lidar_in_channels}, expected 5")
    if model._lidar_sparse_shape != [41, 64, 64]:
        issues.append(f"_lidar_sparse_shape: {model._lidar_sparse_shape}, expected [41, 64, 64]")

    # Check task heads are created (polaris mode builds all heads)
    if not model.with_seg_head:
        issues.append("seg_head not created in polaris mode")
    if not model.with_motion_head:
        issues.append("motion_head not created in polaris mode")
    if not model.with_occ_head:
        issues.append("occ_head not created in polaris mode")
    if not model.with_planning_head:
        issues.append("planning_head not created in polaris mode")

    # Check DCN in ResNet backbone (stage_with_dcn=(False,False,True,True))
    if hasattr(model, 'img_backbone') and model.img_backbone is not None:
        for si, stage_attr in enumerate(['stage1', 'stage2', 'stage3', 'stage4']):
            stage = getattr(model.img_backbone, stage_attr, None)
            if stage is not None and len(stage) > 0:
                first_blk = stage[0]
                has_dcn = getattr(first_blk, 'use_dcn', False)
                expect_dcn = si >= 2  # stages 2,3 should have DCN
                if has_dcn != expect_dcn:
                    issues.append(f"backbone {stage_attr}[0].use_dcn={has_dcn}, expected {expect_dcn}")
                elif expect_dcn and not hasattr(first_blk, 'conv_offset'):
                    issues.append(f"backbone {stage_attr}[0] missing conv_offset")
        print("  ResNet DCN (stage_with_dcn): checked")

    # Check score_thresh and filter_score_thresh (from fusionad_e2e → fusionad_track)
    if hasattr(model, 'track_base'):
        if model.track_base.score_thresh != 0.4:
            issues.append(f"score_thresh={model.track_base.score_thresh}, expected 0.4")
        if model.track_base.filter_score_thresh != 0.35:
            issues.append(f"filter_score_thresh={model.track_base.filter_score_thresh}, expected 0.35")
        print(f"  score_thresh={model.track_base.score_thresh}, filter_score_thresh={model.track_base.filter_score_thresh}")

    # Check update_query_pos wired into QueryInteractionModule
    if hasattr(model, 'query_interact'):
        if not getattr(model.query_interact, 'update_query_pos', False):
            issues.append("query_interact.update_query_pos should be True")
        elif not hasattr(model.query_interact, 'linear_pos1'):
            issues.append("query_interact missing linear_pos1 (update_query_pos=True)")
        else:
            print("  query_interact.update_query_pos=True: OK")

    # Check planning head BEV adapter
    if model.with_planning_head and hasattr(model, 'planning_head'):
        ph = model.planning_head
        if not hasattr(ph, 'adapter_blocks') or len(ph.adapter_blocks) != 3:
            issues.append(f"planning_head adapter_blocks: expected 3, got {len(getattr(ph, 'adapter_blocks', []))}")
        else:
            print("  planning_head BEV adapter (3 blocks): OK")

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 11")
    else:
        print("  All backbones, BEV encoder, task heads, DCN, QIM, adapter present: OK")
        passed += 1
        print(f"\n[OK] TEST 11")

except Exception as e:
    print(f"  [FAIL] TEST 11 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 12: create_input_tensors - img + voxels shapes (use_lidar=True)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 12: create_input_tensors — img + voxels shapes (use_lidar=True)")
print("=" * 80)

try:
    cfg = dict(
        embed_dims=EMBED_DIMS,
        num_query=NUM_QUERY,
        num_classes=NUM_CLASSES,
        num_cams=6,
        bev_h=50,
        bev_w=50,
        pc_range=PC_RANGE,
        video_test_mode=True,
        bs=2,
        img_height=128,
        img_width=256,
        num_bev_encoder_layers=1,
        use_lidar=True,
        lidar_in_channels=5,
        lidar_sparse_shape=[41, 32, 32],
    )
    model = FusionAD('t12_e2e', cfg)
    model.create_input_tensors()

    issues = []

    # Check img tensor
    if 'img' not in model.input_tensors:
        issues.append("'img' not in input_tensors")
    else:
        img_t = model.input_tensors['img']
        expected_img_shape = (2 * 6, 3, 128, 256)
        if tuple(img_t.shape) != expected_img_shape:
            issues.append(f"img shape: {img_t.shape}, expected {expected_img_shape}")
        else:
            print(f"  img shape: {img_t.shape} OK")

    # Check voxels tensor
    if 'voxels' not in model.input_tensors:
        issues.append("'voxels' not in input_tensors (use_lidar=True)")
    else:
        vox_t = model.input_tensors['voxels']
        expected_vox_shape = (2, 5, 41, 32, 32)
        if tuple(vox_t.shape) != expected_vox_shape:
            issues.append(f"voxels shape: {vox_t.shape}, expected {expected_vox_shape}")
        else:
            print(f"  voxels shape: {vox_t.shape} OK")

    # Both should have set_module called (link_module attribute set)
    if 'img' in model.input_tensors:
        if getattr(model.input_tensors['img'], 'link_module', None) is None:
            issues.append("img tensor has no link_module after set_module")
        else:
            print("  img tensor set_module OK")
    if 'voxels' in model.input_tensors:
        if getattr(model.input_tensors['voxels'], 'link_module', None) is None:
            issues.append("voxels tensor has no link_module after set_module")
        else:
            print("  voxels tensor set_module OK")

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 12")
    else:
        passed += 1
        print(f"\n[OK] TEST 12")

except Exception as e:
    print(f"  [FAIL] TEST 12 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 13: Polaris-mode construction (use_lidar=False) - no voxels
# ====================================================================

print("\n" + "=" * 80)
print("TEST 13: Polaris-mode construction (use_lidar=False) — no voxels, no pts_backbone")
print("=" * 80)

try:
    cfg = dict(
        embed_dims=EMBED_DIMS,
        num_query=NUM_QUERY,
        num_classes=NUM_CLASSES,
        num_cams=6,
        bev_h=50,
        bev_w=50,
        pc_range=PC_RANGE,
        video_test_mode=True,
        bs=1,
        img_height=128,
        img_width=256,
        num_bev_encoder_layers=1,
        use_lidar=False,
    )
    model = FusionAD('t13_e2e', cfg)
    model.create_input_tensors()

    issues = []

    # pts_backbone should be None
    if model.pts_backbone is not None:
        issues.append(f"pts_backbone should be None, got {type(model.pts_backbone).__name__}")
    else:
        print("  pts_backbone: None OK")

    # BEV encoder layers should be BEVFormerLayer (no pts_cross_attn)
    bev_enc = model.pts_bbox_head.transformer.encoder
    for i, layer in enumerate(bev_enc.layers):
        ltype = type(layer).__name__
        if ltype != 'BEVFormerLayer':
            issues.append(f"bev_encoder.layer.{i} type: {ltype}, expected BEVFormerLayer")
        elif 'pts_cross_attn' in getattr(layer, 'operation_order', ()):
            issues.append(f"bev_encoder.layer.{i} should NOT have pts_cross_attn")
    print("  bev_encoder layers: BEVFormerLayer (no LiDAR) OK")

    # _use_lidar config
    if model._use_lidar:
        issues.append("_use_lidar should be False")

    # input_tensors should NOT have voxels
    if 'voxels' in model.input_tensors:
        issues.append("'voxels' should NOT be in input_tensors when use_lidar=False")
    else:
        print("  No voxels in input_tensors: OK")

    # img should still exist
    if 'img' not in model.input_tensors:
        issues.append("'img' should still be in input_tensors")
    else:
        print(f"  img shape: {model.input_tensors['img'].shape} OK")

    # img_backbone / img_neck should still exist
    if model.img_backbone is None:
        issues.append("img_backbone should still exist")
    if model.img_neck is None:
        issues.append("img_neck should still exist")

    if issues:
        for iss in issues:
            print(f"  [FAIL] {iss}")
        failed += 1
        print(f"\n[FAIL] TEST 13")
    else:
        passed += 1
        print(f"\n[OK] TEST 13")

except Exception as e:
    print(f"  [FAIL] TEST 13 EXCEPTION: {e}")
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
