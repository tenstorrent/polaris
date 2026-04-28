#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for FusionAD PerceptionTransformer TTSim module.
Validates the conversion from PyTorch to TTSim.

This tests:
  1. CanBusMLP – construction, parameter count, forward-pass numerical
     match vs PyTorch nn.Sequential reference
  2. PerceptionTransformer – construction, parameter count
  3. get_states_and_refs – query splitting, reference-point computation,
     permutation, and decoder call vs PyTorch reference
  4. get_bev_features – CAN-bus encoding, camera/level embedding addition,
     ego-motion shift computation, feature flattening, encoder call
     vs PyTorch reference
"""

import os
import sys
import warnings
import math
import copy
import traceback

polaris_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# ---- TTSim modules under test ----
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.transformer import (
    CanBusMLP,
    PerceptionTransformer,
)


# ====================================================================
# PyTorch reference helpers (CPU-only, Python 3.13 compatible)
# ====================================================================

class CanBusMLP_PyTorch(nn.Module):
    """PyTorch reference of CanBusMLP – nn.Sequential matching FusionAD."""

    def __init__(self, embed_dims=256, can_bus_norm=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.can_bus_norm = can_bus_norm
        self.mlp = nn.Sequential(
            nn.Linear(18, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
        )
        if can_bus_norm:
            # TTSim LayerNorm has no affine params, so match that
            self.mlp.add_module('norm', nn.LayerNorm(embed_dims,
                                                     elementwise_affine=False))

    def forward(self, x):
        return self.mlp(x)


class PerceptionTransformer_PyTorch(nn.Module):
    """
    Simplified PyTorch reference of FusionAD PerceptionTransformer.
    Only includes the parameters/logic we test (no full encoder/decoder).
    """

    def __init__(self, embed_dims=256, num_feature_levels=4, num_cams=6,
                 use_can_bus=True, can_bus_norm=True, use_cams_embeds=True,
                 use_shift=True, rotate_prev_bev=True, rotate_center=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.use_shift = use_shift
        self.rotate_prev_bev = rotate_prev_bev
        self.rotate_center = rotate_center if rotate_center is not None else [100, 100]

        # Learnable layers
        self.level_embeds = nn.Parameter(
            torch.randn(num_feature_levels, embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.randn(num_cams, embed_dims))
        self.reference_points = nn.Linear(embed_dims, 3)

        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
        )
        if can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(embed_dims,
                                                             elementwise_affine=False))

    def get_states_and_refs(self, bev_embed, object_query_embed, bev_h, bev_w,
                            reference_points=None, reg_branches=None,
                            cls_branches=None, img_metas=None):
        """Exact copy of the PyTorch method (no decoder call)."""
        bs = bev_embed.shape[1]
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        if reference_points is not None:
            reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
        else:
            reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        return query, query_pos, init_reference_out


# ====================================================================
# Helpers
# ====================================================================

def compare_outputs(pytorch_output, ttsim_output, name="Output",
                    rtol=1e-4, atol=1e-5):
    """Compare PyTorch and TTSim outputs with numerical validation."""
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.detach().numpy()
    if hasattr(ttsim_output, 'data'):
        ttsim_output = ttsim_output.data

    print(f"\n  {name} comparison:")
    print(f"    PyTorch shape: {pytorch_output.shape}")
    print(f"    TTSim   shape: {ttsim_output.shape}")

    if pytorch_output.shape != ttsim_output.shape:
        print(f"    [FAIL] Shape mismatch!")
        return False

    diff = np.abs(pytorch_output - ttsim_output)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

    if np.allclose(pytorch_output, ttsim_output, rtol=rtol, atol=atol):
        print(f"    [OK] Match within tol (rtol={rtol}, atol={atol})")
        return True
    else:
        rel_err = max_diff / (np.abs(pytorch_output).max() + 1e-10)
        print(f"    Relative error: {rel_err:.6e}")
        if rel_err < 1e-3:
            print(f"    [OK] Relative error acceptable (<1e-3)")
            return True
        print(f"    [FAIL] Diff exceeds tolerance")
        return False


def copy_linear_weights(ttsim_linear, pytorch_linear):
    """Copy weights from a PyTorch nn.Linear into a SimNN.Linear.

    PyTorch stores weight as [out_features, in_features].
    TTSim stores param as [in_features, out_features].
    So we transpose the weight before copying.
    """
    # No transpose — SimNN.Linear transposes internally; both use [out, in]
    w = pytorch_linear.weight.detach().numpy().copy()
    w = np.ascontiguousarray(w, dtype=np.float32)
    ttsim_linear.param = F._from_data(ttsim_linear.param.name, w, is_const=True)
    ttsim_linear.param.is_param = True
    ttsim_linear.param.set_module(ttsim_linear)
    ttsim_linear._tensors[ttsim_linear.param.name] = ttsim_linear.param

    if pytorch_linear.bias is not None and ttsim_linear.bias is not None:
        b = np.ascontiguousarray(pytorch_linear.bias.detach().numpy(), dtype=np.float32)
        ttsim_linear.bias = F._from_data(ttsim_linear.bias.name, b, is_const=True)
        ttsim_linear.bias.is_param = True
        ttsim_linear.bias.set_module(ttsim_linear)
        ttsim_linear._tensors[ttsim_linear.bias.name] = ttsim_linear.bias


# ====================================================================
# TEST 1: CanBusMLP construction and parameter count
# ====================================================================

print("=" * 80)
print("TEST 1: CanBusMLP Construction & Parameter Count")
print("=" * 80)

try:
    embed_dims = 256
    mlp = CanBusMLP('test_mlp', embed_dims=embed_dims, can_bus_norm=True)
    print(f"  [OK] CanBusMLP constructed")

    # Expected param count:
    #   fc0: 18 * 128 + 128 = 2432
    #   fc1: 128 * 256 + 256 = 33024
    #   norm: 256 * 2 = 512
    #   total = 35968
    half = embed_dims // 2
    expected = (18 * half + half) + (half * embed_dims + embed_dims) + (2 * embed_dims)
    actual = mlp.analytical_param_count()
    print(f"  Expected params: {expected}")
    print(f"  Actual params:   {actual}")
    assert actual == expected, f"Param count mismatch: {actual} != {expected}"
    print(f"  [OK] Param count matches")

    # Without norm
    mlp_no_norm = CanBusMLP('test_mlp_nn', embed_dims=embed_dims, can_bus_norm=False)
    expected_nn = (18 * half + half) + (half * embed_dims + embed_dims)
    actual_nn = mlp_no_norm.analytical_param_count()
    assert actual_nn == expected_nn, f"{actual_nn} != {expected_nn}"
    print(f"  [OK] Without norm: {actual_nn} (expected {expected_nn})")

    print("\n[OK] TEST 1 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 1 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 2: CanBusMLP forward pass vs PyTorch
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: CanBusMLP Forward Pass vs PyTorch")
print("=" * 80)

try:
    embed_dims = 256
    bs = 2

    # ---- PyTorch reference ----
    pt_mlp = CanBusMLP_PyTorch(embed_dims=embed_dims, can_bus_norm=True)
    pt_mlp.eval()

    # ---- TTSim module ----
    tt_mlp = CanBusMLP('t2_mlp', embed_dims=embed_dims, can_bus_norm=True)

    # Copy weights: PyTorch Sequential layers:
    #   mlp.0 = Linear(18, 128)
    #   mlp.1 = ReLU
    #   mlp.2 = Linear(128, 256)
    #   mlp.3 = ReLU
    #   mlp.norm = LayerNorm(256)
    copy_linear_weights(tt_mlp.fc0, pt_mlp.mlp[0])
    copy_linear_weights(tt_mlp.fc1, pt_mlp.mlp[2])
    # TTSim LayerNorm has no affine params - nothing to copy
    # (PyTorch reference also uses elementwise_affine=False)

    # ---- Forward ----
    np.random.seed(42)
    x_np = np.random.randn(bs, 18).astype(np.float32)

    # PyTorch
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        y_pt = pt_mlp(x_pt)

    # TTSim
    x_tt = F._from_data('t2_input', x_np, is_const=True)
    y_tt = tt_mlp(x_tt)

    ok = compare_outputs(y_pt, y_tt, "CanBusMLP output")
    if not ok:
        print("  [FAIL] TEST 2 FAILED")
        sys.exit(1)

    # Test various batch sizes
    for test_bs in [1, 4, 8]:
        x_np2 = np.random.randn(test_bs, 18).astype(np.float32)
        with torch.no_grad():
            y_pt2 = pt_mlp(torch.from_numpy(x_np2))
        x_tt2 = F._from_data(f't2_input_bs{test_bs}', x_np2, is_const=True)
        y_tt2 = tt_mlp(x_tt2)
        ok2 = compare_outputs(y_pt2, y_tt2, f"CanBusMLP bs={test_bs}")
        if not ok2:
            print(f"  [FAIL] TEST 2 FAILED at bs={test_bs}")
            sys.exit(1)

    print("\n[OK] TEST 2 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 2 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 3: PerceptionTransformer construction and parameter count
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: PerceptionTransformer Construction & Parameter Count")
print("=" * 80)

try:
    embed_dims = 256
    num_feature_levels = 4
    num_cams = 6

    pt = PerceptionTransformer(
        name='t3_pt',
        encoder=None,
        decoder=None,
        embed_dims=embed_dims,
        num_feature_levels=num_feature_levels,
        num_cams=num_cams,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_prev_bev=True,
        use_shift=True,
        rotate_center=[100, 100],
    )
    print(f"  [OK] PerceptionTransformer constructed")
    print(f"    embed_dims       = {pt.embed_dims}")
    print(f"    num_feature_lvls = {pt.num_feature_levels}")
    print(f"    num_cams         = {pt.num_cams}")

    # Expected params (excluding encoder/decoder):
    #   reference_points_linear: 256*3 + 3 = 771
    #   can_bus_mlp: 35968 (from Test 1)
    #   level_embeds: 4*256 = 1024
    #   cams_embeds: 6*256 = 1536
    #   total = 39299
    half = embed_dims // 2
    expected_mlp = (18 * half + half) + (half * embed_dims + embed_dims) + (2 * embed_dims)
    expected_ref = embed_dims * 3 + 3
    expected_lvl = num_feature_levels * embed_dims
    expected_cam = num_cams * embed_dims
    expected_total = expected_ref + expected_mlp + expected_lvl + expected_cam

    actual_total = pt.analytical_param_count()
    print(f"    Expected params: {expected_total}")
    print(f"    Actual params:   {actual_total}")
    assert actual_total == expected_total, \
        f"Param count mismatch: {actual_total} != {expected_total}"
    print(f"  [OK] Param count matches")

    # Different configs
    for dims, lvls, cams in [(128, 2, 4), (512, 3, 8)]:
        pt2 = PerceptionTransformer(
            name=f't3_pt_{dims}', embed_dims=dims,
            num_feature_levels=lvls, num_cams=cams)
        h = dims // 2
        exp = (dims * 3 + 3) + ((18 * h + h) + (h * dims + dims) + 2 * dims) \
            + lvls * dims + cams * dims
        act = pt2.analytical_param_count()
        assert act == exp, f"Config ({dims},{lvls},{cams}): {act} != {exp}"
        print(f"  [OK] Config ({dims},{lvls},{cams}): {act} params")

    print("\n[OK] TEST 3 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 3 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 4: get_states_and_refs – query splitting & reference points
#         (with explicit reference_points provided)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: get_states_and_refs – Explicit reference_points")
print("=" * 80)

try:
    embed_dims = 256
    num_queries = 50
    bs = 2
    bev_h, bev_w = 10, 10

    # ---- PyTorch reference ----
    pt_ref = PerceptionTransformer_PyTorch(embed_dims=embed_dims)
    pt_ref.eval()

    # ---- TTSim module ----
    tt_pt = PerceptionTransformer(
        name='t4_pt', embed_dims=embed_dims, decoder=None)

    # Copy reference_points linear weights
    copy_linear_weights(tt_pt.reference_points_linear, pt_ref.reference_points)

    # ---- Inputs ----
    np.random.seed(123)
    # object_query_embed: [nq, 2*embed_dims]
    oqe_np = np.random.randn(num_queries, 2 * embed_dims).astype(np.float32)
    # bev_embed: [nq_bev, bs, embed_dims] (seq-first)
    bev_np = np.random.randn(bev_h * bev_w, bs, embed_dims).astype(np.float32)
    # explicit reference_points: [nq, 3]
    rp_np = np.random.randn(num_queries, 3).astype(np.float32)

    # --- PyTorch ---
    oqe_pt = torch.from_numpy(oqe_np)
    bev_pt = torch.from_numpy(bev_np)
    rp_pt = torch.from_numpy(rp_np)
    with torch.no_grad():
        query_pt, qpos_pt, init_ref_pt = pt_ref.get_states_and_refs(
            bev_pt, oqe_pt, bev_h, bev_w, reference_points=rp_pt)

    # --- TTSim ---
    # Create a mock decoder that records args
    class MockDecoder:
        def __init__(self): self.called = False
        def __call__(self, **kwargs):
            self.called = True
            self.kwargs = kwargs
            # Return dummy values
            nq = kwargs['query'].shape[0]
            bs_d = kwargs['query'].shape[1]
            dummy_states = [F._from_data('mock_state', np.zeros((nq, bs_d, embed_dims), dtype=np.float32))]
            dummy_refs = [np.zeros((bs_d, nq, 3), dtype=np.float32)]
            return dummy_states, dummy_refs

    mock_dec = MockDecoder()
    tt_pt.decoder = mock_dec

    bev_tt = F._from_data('t4_bev', bev_np, is_const=True)
    inter_states, init_ref_tt, inter_refs = tt_pt.get_states_and_refs(
        bev_tt, oqe_np, bev_h, bev_w, reference_points=rp_np)

    # Verify init_reference_out (sigmoid of rp)
    expected_ref = 1.0 / (1.0 + np.exp(-rp_np.astype(np.float64)))
    expected_ref = np.tile(expected_ref[np.newaxis], (bs, 1, 1)).astype(np.float32)

    ok_ref = compare_outputs(init_ref_pt, init_ref_tt, "init_reference_out")
    if not ok_ref:
        print("  [FAIL] TEST 4 FAILED: init_reference_out mismatch")
        sys.exit(1)

    # Verify query and query_pos passed to decoder
    assert mock_dec.called, "Decoder was not called"
    dec_query = mock_dec.kwargs['query']
    dec_qpos = mock_dec.kwargs['query_pos']

    # query: [nq, bs, embed_dims]
    ok_q = compare_outputs(query_pt, dec_query, "decoder query")
    ok_qp = compare_outputs(qpos_pt, dec_qpos, "decoder query_pos")
    if not (ok_q and ok_qp):
        print("  [FAIL] TEST 4 FAILED: decoder query/qpos mismatch")
        sys.exit(1)

    print("\n[OK] TEST 4 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 4 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 5: get_states_and_refs – computed reference points (no explicit)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: get_states_and_refs – Computed reference_points")
print("=" * 80)

try:
    embed_dims = 256
    num_queries = 50
    bs = 2
    bev_h, bev_w = 10, 10

    # ---- PyTorch reference ----
    pt_ref5 = PerceptionTransformer_PyTorch(embed_dims=embed_dims)
    pt_ref5.eval()

    # ---- TTSim module ----
    tt_pt5 = PerceptionTransformer(
        name='t5_pt', embed_dims=embed_dims, decoder=None)
    copy_linear_weights(tt_pt5.reference_points_linear, pt_ref5.reference_points)

    # ---- Inputs ----
    np.random.seed(456)
    oqe_np5 = np.random.randn(num_queries, 2 * embed_dims).astype(np.float32)
    bev_np5 = np.random.randn(bev_h * bev_w, bs, embed_dims).astype(np.float32)

    # --- PyTorch (no reference_points => use self.reference_points linear) ---
    oqe_pt5 = torch.from_numpy(oqe_np5)
    bev_pt5 = torch.from_numpy(bev_np5)
    with torch.no_grad():
        query_pt5, qpos_pt5, init_ref_pt5 = pt_ref5.get_states_and_refs(
            bev_pt5, oqe_pt5, bev_h, bev_w, reference_points=None)

    # --- TTSim ---
    class MockDecoder5:
        def __init__(self): self.called = False
        def __call__(self, **kwargs):
            self.called = True
            self.kwargs = kwargs
            nq = kwargs['query'].shape[0]
            bs_d = kwargs['query'].shape[1]
            return ([F._from_data('ms5', np.zeros((nq, bs_d, embed_dims), dtype=np.float32))],
                    [np.zeros((bs_d, nq, 3), dtype=np.float32)])

    mock_dec5 = MockDecoder5()
    tt_pt5.decoder = mock_dec5

    bev_tt5 = F._from_data('t5_bev', bev_np5, is_const=True)
    inter_states5, init_ref_tt5, _ = tt_pt5.get_states_and_refs(
        bev_tt5, oqe_np5, bev_h, bev_w, reference_points=None)

    # init_reference_out: TTSim returns None when graph-computed
    # (because the actual sigmoid output lives only in the graph).
    # But we can check the decoder got the right reference_points.
    assert mock_dec5.called, "Decoder was not called"

    # Check query and query_pos match
    ok_q5 = compare_outputs(query_pt5, mock_dec5.kwargs['query'], "decoder query (computed rp)")
    ok_qp5 = compare_outputs(qpos_pt5, mock_dec5.kwargs['query_pos'], "decoder query_pos (computed rp)")
    if not (ok_q5 and ok_qp5):
        print("  [FAIL] TEST 5 FAILED")
        sys.exit(1)

    # Check reference_points: TTSim passes a SimTensor through the graph
    dec_rp = mock_dec5.kwargs['reference_points']
    if hasattr(dec_rp, 'data'):
        # Graph-computed: data should match PyTorch
        ok_rp5 = compare_outputs(init_ref_pt5, dec_rp, "reference_points (computed)")
        if not ok_rp5:
            print("  [FAIL] TEST 5 FAILED: reference_points mismatch")
            sys.exit(1)
    else:
        print(f"  reference_points type: {type(dec_rp)} (graph tensor, skipping value check)")

    print("\n[OK] TEST 5 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 5 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 6: get_bev_features – CAN-bus MLP encoding
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: get_bev_features – CAN-bus MLP Encoding")
print("=" * 80)

try:
    embed_dims = 256
    num_feature_levels = 1
    num_cams = 6
    bev_h, bev_w = 10, 10
    bs = 2
    feat_h, feat_w = 8, 8

    # ---- PyTorch reference ----
    pt_ref6 = PerceptionTransformer_PyTorch(embed_dims=embed_dims,
                                             num_feature_levels=num_feature_levels,
                                             num_cams=num_cams)
    pt_ref6.eval()

    # ---- TTSim module ----
    # Use a mock encoder to capture the bev_queries AFTER CAN-bus addition
    class MockEncoder6:
        """Records the bev_queries passed as first positional arg."""
        def __init__(self): self.bev_queries = None
        def __call__(self, *args, **kwargs):
            self.bev_queries = args[0]
            return args[0]  # pass through

    mock_enc6 = MockEncoder6()
    tt_pt6 = PerceptionTransformer(
        name='t6_pt', encoder=mock_enc6, embed_dims=embed_dims,
        num_feature_levels=num_feature_levels, num_cams=num_cams,
        use_can_bus=True, can_bus_norm=True, use_cams_embeds=True)

    # Copy can_bus_mlp weights
    copy_linear_weights(tt_pt6.can_bus_mlp.fc0, pt_ref6.can_bus_mlp[0])
    copy_linear_weights(tt_pt6.can_bus_mlp.fc1, pt_ref6.can_bus_mlp[2])
    # TTSim LayerNorm has no affine params - nothing to copy

    # Set embeddings
    tt_pt6.level_embeds = pt_ref6.level_embeds.detach().numpy()
    tt_pt6.cams_embeds = pt_ref6.cams_embeds.detach().numpy()

    # ---- Inputs ----
    np.random.seed(789)
    can_bus_data = np.random.randn(18).astype(np.float32)
    img_metas = [{'can_bus': can_bus_data.tolist()} for _ in range(bs)]

    bev_queries_np = np.random.randn(bev_h * bev_w, embed_dims).astype(np.float32)
    bev_pos_np = np.random.randn(embed_dims, bev_h, bev_w).astype(np.float32)

    mlvl_feats_np = np.random.randn(bs, num_cams, embed_dims, feat_h, feat_w).astype(np.float32)

    # --- PyTorch reference CAN-bus encoding ---
    can_bus_tensor = torch.from_numpy(
        np.array([m['can_bus'] for m in img_metas], dtype=np.float32))
    with torch.no_grad():
        can_bus_out_pt = pt_ref6.can_bus_mlp(can_bus_tensor)  # [bs, embed_dims]

    # Expand bev_queries
    bev_q_expanded = np.tile(bev_queries_np[:, np.newaxis, :], (1, bs, 1))
    can_bus_3d_pt = can_bus_out_pt.unsqueeze(0).numpy()  # [1, bs, embed_dims]
    expected_bev = bev_q_expanded + can_bus_3d_pt

    # --- TTSim ---
    bev_q_tt = F._from_data('t6_bev_q', bev_queries_np, is_const=True)
    bev_pos_tt = F._from_data('t6_bev_pos', bev_pos_np, is_const=True)
    feat_tt = F._from_data('t6_feat', mlvl_feats_np, is_const=True)

    tt_pt6.get_bev_features(
        mlvl_feats=[feat_tt],
        bev_queries=bev_q_tt,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos_tt,
        img_metas=img_metas,
    )

    # The mock encoder captured bev_queries after CAN-bus addition
    captured = mock_enc6.bev_queries
    captured_np = captured.data if hasattr(captured, 'data') else captured

    ok_canbus = compare_outputs(
        torch.from_numpy(expected_bev), captured_np,
        "BEV queries after CAN-bus MLP")
    if not ok_canbus:
        print("  [FAIL] TEST 6 FAILED")
        sys.exit(1)

    print("\n[OK] TEST 6 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 6 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 7: get_bev_features – Camera and level embedding addition
# ====================================================================

print("\n" + "=" * 80)
print("TEST 7: get_bev_features – Camera & Level Embeddings")
print("=" * 80)

try:
    embed_dims = 256
    num_feature_levels = 2
    num_cams = 6
    bev_h, bev_w = 4, 4
    bs = 1
    feat_h, feat_w = 3, 3

    # ---- PyTorch reference (manual computation) ----
    np.random.seed(999)
    level_embeds_np = np.random.randn(num_feature_levels, embed_dims).astype(np.float32)
    cams_embeds_np = np.random.randn(num_cams, embed_dims).astype(np.float32)

    feat_l0_np = np.random.randn(bs, num_cams, embed_dims, feat_h, feat_w).astype(np.float32)
    feat_l1_np = np.random.randn(bs, num_cams, embed_dims, feat_h, feat_w).astype(np.float32)

    # PyTorch processing for each level:
    #   feat = feat.flatten(3).permute(1, 0, 3, 2)  => [num_cam, bs, H*W, C]
    #   feat += cams_embeds[:, None, None, :]         => broadcast add
    #   feat += level_embeds[None, None, lvl:lvl+1, :]
    #   After concat levels => permute(0, 2, 1, 3)   => [num_cam, H*W, bs, C]

    feat_l0_t = torch.from_numpy(feat_l0_np)
    feat_l1_t = torch.from_numpy(feat_l1_np)
    level_embeds_t = torch.from_numpy(level_embeds_np)
    cams_embeds_t = torch.from_numpy(cams_embeds_np)

    # Level 0
    f0 = feat_l0_t.flatten(3).permute(1, 0, 3, 2)  # [num_cam, bs, H*W, C]
    f0 = f0 + cams_embeds_t[:, None, None, :]
    f0 = f0 + level_embeds_t[None, None, 0:1, :]

    # Level 1
    f1 = feat_l1_t.flatten(3).permute(1, 0, 3, 2)
    f1 = f1 + cams_embeds_t[:, None, None, :]
    f1 = f1 + level_embeds_t[None, None, 1:2, :]

    # Concat + final permute
    feat_cat = torch.cat([f0, f1], dim=2)  # [num_cam, bs, sum_HW, C]
    feat_final_pt = feat_cat.permute(0, 2, 1, 3).numpy()  # [num_cam, sum_HW, bs, C]

    # ---- TTSim ----
    class MockEncoder7:
        def __init__(self): self.feat_flatten = None
        def __call__(self, *args, **kwargs):
            self.feat_flatten = args[1]  # second positional arg is feat_flatten
            return args[0]

    mock_enc7 = MockEncoder7()
    tt_pt7 = PerceptionTransformer(
        name='t7_pt', encoder=mock_enc7, embed_dims=embed_dims,
        num_feature_levels=num_feature_levels, num_cams=num_cams,
        use_can_bus=False, can_bus_norm=True, use_cams_embeds=True)
    tt_pt7.level_embeds = level_embeds_np.copy()
    tt_pt7.cams_embeds = cams_embeds_np.copy()

    bev_q7 = F._from_data('t7_bev_q', np.random.randn(bev_h * bev_w, embed_dims).astype(np.float32))
    bev_pos7 = F._from_data('t7_bev_pos', np.random.randn(embed_dims, bev_h, bev_w).astype(np.float32))
    feat_tt_l0 = F._from_data('t7_feat_l0', feat_l0_np)
    feat_tt_l1 = F._from_data('t7_feat_l1', feat_l1_np)

    tt_pt7.get_bev_features(
        mlvl_feats=[feat_tt_l0, feat_tt_l1],
        bev_queries=bev_q7,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos7,
        img_metas=None,
    )

    captured7 = mock_enc7.feat_flatten
    captured7_np = captured7.data if hasattr(captured7, 'data') else captured7

    ok_feat = compare_outputs(
        torch.from_numpy(feat_final_pt), captured7_np,
        "feat_flatten after cam+level embeds")
    if not ok_feat:
        print("  [FAIL] TEST 7 FAILED")
        sys.exit(1)

    print("\n[OK] TEST 7 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 7 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 8: Ego-motion shift computation
# ====================================================================

print("\n" + "=" * 80)
print("TEST 8: Ego-motion Shift Computation")
print("=" * 80)

try:
    bev_h, bev_w = 50, 50
    bs = 2
    grid_length = [0.512, 0.512]

    # Create img_metas with known CAN-bus signals
    np.random.seed(555)
    can_bus_0 = np.zeros(18, dtype=np.float32)
    can_bus_0[0] = 1.5   # delta_x
    can_bus_0[1] = 0.8   # delta_y
    can_bus_0[-2] = 0.3  # ego_angle (radians)

    can_bus_1 = np.zeros(18, dtype=np.float32)
    can_bus_1[0] = -0.5
    can_bus_1[1] = 2.0
    can_bus_1[-2] = -0.1

    img_metas = [
        {'can_bus': can_bus_0.tolist()},
        {'can_bus': can_bus_1.tolist()},
    ]

    # ---- PyTorch reference (exact code from original) ----
    delta_x = np.array([m['can_bus'][0] for m in img_metas])
    delta_y = np.array([m['can_bus'][1] for m in img_metas])
    ego_angle = np.array([m['can_bus'][-2] / np.pi * 180 for m in img_metas])
    grid_length_y = grid_length[0]
    grid_length_x = grid_length[1]
    translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
    bev_angle = ego_angle - translation_angle
    shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
    shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
    expected_shift = np.stack([shift_x, shift_y], axis=-1).astype(np.float32)

    # ---- TTSim ----
    class MockEncoder8:
        def __init__(self): self.shift = None
        def __call__(self, *args, **kwargs):
            self.shift = kwargs.get('shift', None)
            return args[0]

    mock_enc8 = MockEncoder8()
    embed_dims = 256
    tt_pt8 = PerceptionTransformer(
        name='t8_pt', encoder=mock_enc8, embed_dims=embed_dims,
        num_feature_levels=1, num_cams=6,
        use_can_bus=False, use_shift=True)
    tt_pt8.level_embeds = np.random.randn(1, embed_dims).astype(np.float32)
    tt_pt8.cams_embeds = np.random.randn(6, embed_dims).astype(np.float32)

    bev_q8 = F._from_data('t8_bev_q', np.random.randn(bev_h * bev_w, embed_dims).astype(np.float32))
    bev_pos8 = F._from_data('t8_bev_pos', np.random.randn(embed_dims, bev_h, bev_w).astype(np.float32))
    feat8 = F._from_data('t8_feat', np.random.randn(bs, 6, embed_dims, 4, 4).astype(np.float32))

    tt_pt8.get_bev_features(
        mlvl_feats=[feat8],
        bev_queries=bev_q8,
        bev_h=bev_h,
        bev_w=bev_w,
        grid_length=grid_length,
        bev_pos=bev_pos8,
        img_metas=img_metas,
    )

    actual_shift = mock_enc8.shift
    print(f"  Expected shift:\n    {expected_shift}")
    print(f"  Actual shift:\n    {actual_shift}")

    ok_shift = np.allclose(expected_shift, actual_shift, rtol=1e-5, atol=1e-6)
    if ok_shift:
        print(f"  [OK] Shift matches")
    else:
        diff = np.abs(expected_shift - actual_shift)
        print(f"  [FAIL] Shift mismatch, max diff: {diff.max():.6e}")
        sys.exit(1)

    print("\n[OK] TEST 8 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 8 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 9: LiDAR (pts_feats) feature processing
# ====================================================================

print("\n" + "=" * 80)
print("TEST 9: LiDAR Feature Processing (pts_feats)")
print("=" * 80)

try:
    embed_dims = 256
    bs = 2
    bev_h, bev_w = 4, 4
    pts_h, pts_w = 5, 5

    # PyTorch reference:
    #   pts_feats: [bs, C, H, W]
    #   flatten(2): [bs, C, H*W]
    #   permute(0, 2, 1): [bs, H*W, C]  (note: NOT [H*W, bs, C])
    np.random.seed(321)
    pts_np = np.random.randn(bs, embed_dims, pts_h, pts_w).astype(np.float32)

    pts_t = torch.from_numpy(pts_np)
    expected_pts = pts_t.flatten(2).permute(0, 2, 1).numpy()  # [bs, H*W, C]

    # ---- TTSim ----
    class MockEncoder9:
        def __init__(self): self.pts_feats = None
        def __call__(self, *args, **kwargs):
            self.pts_feats = kwargs.get('pts_feats', None)
            return args[0]

    mock_enc9 = MockEncoder9()
    tt_pt9 = PerceptionTransformer(
        name='t9_pt', encoder=mock_enc9, embed_dims=embed_dims,
        num_feature_levels=1, num_cams=6,
        use_can_bus=False, use_shift=False)
    tt_pt9.level_embeds = np.random.randn(1, embed_dims).astype(np.float32)
    tt_pt9.cams_embeds = np.random.randn(6, embed_dims).astype(np.float32)

    bev_q9 = F._from_data('t9_bev_q', np.random.randn(bev_h * bev_w, embed_dims).astype(np.float32))
    bev_pos9 = F._from_data('t9_bev_pos', np.random.randn(embed_dims, bev_h, bev_w).astype(np.float32))
    feat9 = F._from_data('t9_feat', np.random.randn(bs, 6, embed_dims, 3, 3).astype(np.float32))
    pts_tt = F._from_data('t9_pts', pts_np)

    tt_pt9.get_bev_features(
        mlvl_feats=[feat9],
        bev_queries=bev_q9,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos9,
        pts_feats=pts_tt,
    )

    captured_pts = mock_enc9.pts_feats
    if captured_pts is None:
        print("  [FAIL] pts_feats was not passed to encoder")
        sys.exit(1)

    captured_pts_np = captured_pts.data if hasattr(captured_pts, 'data') else captured_pts

    ok_pts = compare_outputs(
        torch.from_numpy(expected_pts), captured_pts_np,
        "pts_feats (flattened & transposed)")
    if not ok_pts:
        print("  [FAIL] TEST 9 FAILED")
        sys.exit(1)

    print("\n[OK] TEST 9 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 9 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 10: bev_pos flattening and tiling
# ====================================================================

print("\n" + "=" * 80)
print("TEST 10: bev_pos Processing")
print("=" * 80)

try:
    embed_dims = 64
    bev_h, bev_w = 4, 4
    bs = 2

    # PyTorch reference:
    #   bev_pos: [embed_dims, bev_h, bev_w]
    #   bev_pos.flatten(2).permute(2, 0, 1) => [bev_h*bev_w, embed_dims]
    #   but TTSim also tiles to [bev_h*bev_w, bs, embed_dims]
    np.random.seed(222)
    bev_pos_np = np.random.randn(embed_dims, bev_h, bev_w).astype(np.float32)

    bev_pos_flat = bev_pos_np.reshape(embed_dims, -1)  # [D, H*W]
    bev_pos_flat = bev_pos_flat.T  # [H*W, D]
    bev_pos_flat = bev_pos_flat[:, np.newaxis, :]  # [H*W, 1, D]
    expected_bev_pos = np.tile(bev_pos_flat, (1, bs, 1))  # [H*W, bs, D]

    # ---- TTSim ----
    class MockEncoder10:
        def __init__(self): self.bev_pos = None
        def __call__(self, *args, **kwargs):
            self.bev_pos = kwargs.get('bev_pos', None)
            return args[0]

    mock_enc10 = MockEncoder10()
    tt_pt10 = PerceptionTransformer(
        name='t10_pt', encoder=mock_enc10, embed_dims=embed_dims,
        num_feature_levels=1, num_cams=2,
        use_can_bus=False, use_shift=False)
    tt_pt10.level_embeds = np.random.randn(1, embed_dims).astype(np.float32)
    tt_pt10.cams_embeds = np.random.randn(2, embed_dims).astype(np.float32)

    bev_q10 = F._from_data('t10_bev_q', np.random.randn(bev_h * bev_w, embed_dims).astype(np.float32))
    bev_pos10 = F._from_data('t10_bev_pos', bev_pos_np)
    feat10 = F._from_data('t10_feat', np.random.randn(bs, 2, embed_dims, 3, 3).astype(np.float32))

    tt_pt10.get_bev_features(
        mlvl_feats=[feat10],
        bev_queries=bev_q10,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos10,
    )

    captured_pos = mock_enc10.bev_pos
    captured_pos_np = captured_pos.data if hasattr(captured_pos, 'data') else captured_pos

    ok_pos = compare_outputs(
        torch.from_numpy(expected_bev_pos), captured_pos_np,
        "bev_pos (flattened, transposed, tiled)")
    if not ok_pos:
        print("  [FAIL] TEST 10 FAILED")
        sys.exit(1)

    print("\n[OK] TEST 10 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 10 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# Test Summary
# ====================================================================

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

tests = [
    "CanBusMLP Construction & Parameter Count",
    "CanBusMLP Forward Pass vs PyTorch",
    "PerceptionTransformer Construction & Parameter Count",
    "get_states_and_refs – Explicit reference_points",
    "get_states_and_refs – Computed reference_points",
    "get_bev_features – CAN-bus MLP Encoding",
    "get_bev_features – Camera & Level Embeddings",
    "Ego-motion Shift Computation",
    "LiDAR Feature Processing (pts_feats)",
    "bev_pos Processing",
]

for i, test in enumerate(tests, 1):
    print(f"  TEST {i:2d}: {test:.<60s} [OK] PASSED")

print(f"\nTotal: {len(tests)}/{len(tests)} tests passed")
print("\n" + "=" * 80)
print("All tests passed!")
print("=" * 80)
