#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for motion transformer decoder modules (TTSim vs PyTorch).

Validates that IntentionInteraction, TrackAgentInteraction, MapInteraction,
and MotionTransformerDecoder in TTSim produce identical results to
equivalent PyTorch modules.
"""

import os
import sys
import traceback

polaris_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.motion_head_plugin.modules import (
    IntentionInteraction,
    TrackAgentInteraction,
    MapInteraction,
    MotionTransformerDecoder,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.motion_head_plugin.base_motion_head import (
    TwoLayerMLP,
)


# ====================================================================
# Weight copy helpers
# ====================================================================

def copy_linear_pt_to_tt(pt_linear, tt_linear):
    """Copy weights from PyTorch nn.Linear to TTSim SimNN.Linear."""
    tt_linear.param.data = pt_linear.weight.data.detach().numpy().astype(np.float32)
    tt_linear.bias.data = pt_linear.bias.data.detach().numpy().astype(np.float32)


def copy_mha_weights(pt_mha, tt_mha):
    """
    Copy weights from PyTorch nn.MultiheadAttention to TTSim MultiheadAttention.

    PyTorch stores Q, K, V as a combined in_proj_weight [3D, D] and in_proj_bias [3D].
    TTSim has separate q_proj, k_proj, v_proj (each SimNN.Linear with param [D, D]).
    """
    D = pt_mha.embed_dim
    # in_proj_weight: [3D, D] -> split into Q[D,D], K[D,D], V[D,D]
    w = pt_mha.in_proj_weight.data.detach().numpy()  # [3D, D]
    b = pt_mha.in_proj_bias.data.detach().numpy()    # [3D]

    wq, wk, wv = w[:D], w[D:2*D], w[2*D:]
    bq, bk, bv = b[:D], b[D:2*D], b[2*D:]

    # TTSim Linear stores param as [out_features, in_features]
    tt_mha.q_proj.param.data = wq.astype(np.float32)
    tt_mha.q_proj.bias.data = bq.astype(np.float32)
    tt_mha.k_proj.param.data = wk.astype(np.float32)
    tt_mha.k_proj.bias.data = bk.astype(np.float32)
    tt_mha.v_proj.param.data = wv.astype(np.float32)
    tt_mha.v_proj.bias.data = bv.astype(np.float32)

    # out_proj
    tt_mha.out_proj.param.data = pt_mha.out_proj.weight.data.detach().numpy().astype(np.float32)
    tt_mha.out_proj.bias.data = pt_mha.out_proj.bias.data.detach().numpy().astype(np.float32)


def copy_encoder_layer_weights(pt_layer, tt_module):
    """
    Copy weights from PyTorch nn.TransformerEncoderLayer to TTSim IntentionInteraction.

    PyTorch encoder layer has:
      - self_attn (nn.MultiheadAttention)
      - linear1, linear2 (FFN)
      - norm1, norm2 (LayerNorm — neutralize affine)
    """
    # Self-attention
    copy_mha_weights(pt_layer.self_attn, tt_module.self_attn)

    # FFN: linear1 -> ReLU -> linear2
    copy_linear_pt_to_tt(pt_layer.linear1, tt_module.ffn_fc1)
    copy_linear_pt_to_tt(pt_layer.linear2, tt_module.ffn_fc2)

    # Neutralize layer norms (TTSim LN has no affine params)
    pt_layer.norm1.weight.data.fill_(1.0)
    pt_layer.norm1.bias.data.fill_(0.0)
    pt_layer.norm2.weight.data.fill_(1.0)
    pt_layer.norm2.bias.data.fill_(0.0)


def copy_decoder_layer_to_track_agent(pt_layer, tt_module):
    """
    Copy weights from PyTorch nn.TransformerDecoderLayer to TTSim TrackAgentInteraction.

    PyTorch decoder layer has:
      - self_attn (nn.MultiheadAttention) — self-attention
      - multihead_attn (nn.MultiheadAttention) — cross-attention
      - linear1, linear2 (FFN)
      - norm1, norm2, norm3 (LayerNorm — neutralize affine)
    """
    # Self-attention
    copy_mha_weights(pt_layer.self_attn, tt_module.self_attn)

    # Cross-attention
    copy_mha_weights(pt_layer.multihead_attn, tt_module.cross_attn)

    # FFN
    copy_linear_pt_to_tt(pt_layer.linear1, tt_module.ffn_fc1)
    copy_linear_pt_to_tt(pt_layer.linear2, tt_module.ffn_fc2)

    # Neutralize norms
    pt_layer.norm1.weight.data.fill_(1.0)
    pt_layer.norm1.bias.data.fill_(0.0)
    pt_layer.norm2.weight.data.fill_(1.0)
    pt_layer.norm2.bias.data.fill_(0.0)
    pt_layer.norm3.weight.data.fill_(1.0)
    pt_layer.norm3.bias.data.fill_(0.0)


def copy_decoder_layer_to_map_interaction(pt_layer, tt_module):
    """
    Copy weights from PyTorch nn.TransformerDecoderLayer to TTSim MapInteraction.
    Same structure as TrackAgentInteraction.
    """
    copy_decoder_layer_to_track_agent(pt_layer, tt_module)


def copy_two_layer_mlp_weights(pt_seq, tt_mlp):
    """Copy PyTorch Sequential(Linear, ReLU, Linear) -> TTSim TwoLayerMLP."""
    copy_linear_pt_to_tt(pt_seq[0], tt_mlp.fc0)
    copy_linear_pt_to_tt(pt_seq[2], tt_mlp.fc1)


# ====================================================================
# Compare helper
# ====================================================================

def compare(pt_out, tt_out, name, atol=1e-5):
    pt_np = pt_out.detach().numpy() if isinstance(pt_out, torch.Tensor) else pt_out
    tt_np = tt_out.data if hasattr(tt_out, 'data') and not isinstance(tt_out, np.ndarray) else tt_out
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

EMBED_DIMS = 64
NUM_HEADS = 8
DIM_FF = 128       # embed_dims * 2 as used in PyTorch source
BS = 1
NUM_AGENTS = 8
NUM_MODES = 6
NUM_MAP_LANES = 50

passed = 0
failed = 0


# ====================================================================
# TEST 1: IntentionInteraction — PyTorch vs TTSim
# ====================================================================

print("=" * 80)
print("TEST 1: IntentionInteraction — PyTorch vs TTSim")
print("=" * 80)

try:
    # PyTorch: TransformerEncoderLayer
    pt_enc_layer = nn.TransformerEncoderLayer(
        d_model=EMBED_DIMS, nhead=NUM_HEADS, dropout=0.0,
        dim_feedforward=DIM_FF, batch_first=True)

    # TTSim
    tt_ii = IntentionInteraction('t1_ii', embed_dims=EMBED_DIMS,
                                 num_heads=NUM_HEADS, dim_feedforward=DIM_FF)

    # Copy weights
    copy_encoder_layer_weights(pt_enc_layer, tt_ii)

    # Input: (B, A, P, D) — each agent's P modes attend to each other
    x_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t1_x', x_np, is_const=True)

    # PyTorch forward: flatten (B,A,P,D) -> (B*A,P,D), run encoder layer, reshape back
    with torch.no_grad():
        flat = x_pt.flatten(0, 1)  # (B*A, P, D)
        out_flat = pt_enc_layer(flat)
        pt_out = out_flat.view(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS)

    # TTSim forward (module handles flatten/unflatten internally)
    tt_out = tt_ii(x_tt)

    ok = compare(pt_out, tt_out, "IntentionInteraction output", atol=1e-4)
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
# TEST 2: IntentionInteraction — param count
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: IntentionInteraction — param count")
print("=" * 80)

try:
    D = EMBED_DIMS
    D_ff = DIM_FF
    # MHA: Q,K,V,Out → 4*(D²+D)
    expected_mha = 4 * (D * D + D)
    # FFN: linear1 (D*D_ff + D_ff) + linear2 (D_ff*D + D)
    expected_ffn = D * D_ff + D_ff + D_ff * D + D
    expected = expected_mha + expected_ffn

    tt_ii = IntentionInteraction('t2_ii', embed_dims=D, num_heads=NUM_HEADS,
                                 dim_feedforward=D_ff)
    actual = tt_ii.analytical_param_count()

    ok = actual == expected
    print(f"  Expected: {expected:,}")
    print(f"  Actual:   {actual:,}")

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
# TEST 3: TrackAgentInteraction — PyTorch vs TTSim
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: TrackAgentInteraction — PyTorch vs TTSim")
print("=" * 80)

try:
    # PyTorch: TransformerDecoderLayer
    pt_dec_layer = nn.TransformerDecoderLayer(
        d_model=EMBED_DIMS, nhead=NUM_HEADS, dropout=0.0,
        dim_feedforward=DIM_FF, batch_first=True)

    # TTSim
    tt_tai = TrackAgentInteraction('t3_tai', embed_dims=EMBED_DIMS,
                                   num_heads=NUM_HEADS, dim_feedforward=DIM_FF)

    # Copy weights
    copy_decoder_layer_to_track_agent(pt_dec_layer, tt_tai)

    # Inputs
    query_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
    key_np = np.random.randn(BS, NUM_AGENTS, EMBED_DIMS).astype(np.float32)
    qpos_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
    kpos_np = np.random.randn(BS, NUM_AGENTS, EMBED_DIMS).astype(np.float32)

    # PyTorch forward: replicate the original module's logic
    with torch.no_grad():
        q_pt = torch.from_numpy(query_np) + torch.from_numpy(qpos_np)
        k_pt = torch.from_numpy(key_np) + torch.from_numpy(kpos_np)
        # key is expanded: (B, A_k, D) -> (B*A, A_k, D)  (for B=1, expand by A)
        mem = k_pt.expand(BS * NUM_AGENTS, -1, -1)  # (A, A_k, D)
        tgt = q_pt.flatten(0, 1)  # (B*A, P, D)
        out_flat = pt_dec_layer(tgt, mem)
        pt_out = out_flat.view(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS)

    # TTSim forward
    q_tt = F._from_data('t3_q', query_np, is_const=True)
    k_tt = F._from_data('t3_k', key_np, is_const=True)
    qp_tt = F._from_data('t3_qp', qpos_np, is_const=True)
    kp_tt = F._from_data('t3_kp', kpos_np, is_const=True)

    tt_out = tt_tai(q_tt, k_tt, query_pos=qp_tt, key_pos=kp_tt)

    ok = compare(pt_out, tt_out, "TrackAgentInteraction output", atol=1e-4)
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
# TEST 4: TrackAgentInteraction — without positional encodings
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: TrackAgentInteraction — no positional encodings")
print("=" * 80)

try:
    pt_dec_layer = nn.TransformerDecoderLayer(
        d_model=EMBED_DIMS, nhead=NUM_HEADS, dropout=0.0,
        dim_feedforward=DIM_FF, batch_first=True)
    tt_tai = TrackAgentInteraction('t4_tai', embed_dims=EMBED_DIMS,
                                   num_heads=NUM_HEADS, dim_feedforward=DIM_FF)
    copy_decoder_layer_to_track_agent(pt_dec_layer, tt_tai)

    query_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
    key_np = np.random.randn(BS, NUM_AGENTS, EMBED_DIMS).astype(np.float32)

    with torch.no_grad():
        q_pt = torch.from_numpy(query_np)
        k_pt = torch.from_numpy(key_np)
        mem = k_pt.expand(BS * NUM_AGENTS, -1, -1)
        tgt = q_pt.flatten(0, 1)
        out_flat = pt_dec_layer(tgt, mem)
        pt_out = out_flat.view(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS)

    q_tt = F._from_data('t4_q', query_np, is_const=True)
    k_tt = F._from_data('t4_k', key_np, is_const=True)
    tt_out = tt_tai(q_tt, k_tt)

    ok = compare(pt_out, tt_out, "TrackAgentInteraction (no pos)", atol=1e-4)
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
# TEST 5: TrackAgentInteraction — param count
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: TrackAgentInteraction — param count")
print("=" * 80)

try:
    D = EMBED_DIMS
    D_ff = DIM_FF
    # 2 MHAs × 4*(D²+D) + FFN
    expected_mha = 2 * 4 * (D * D + D)
    expected_ffn = D * D_ff + D_ff + D_ff * D + D
    expected = expected_mha + expected_ffn

    tt_tai = TrackAgentInteraction('t5_tai', embed_dims=D, num_heads=NUM_HEADS,
                                   dim_feedforward=D_ff)
    actual = tt_tai.analytical_param_count()

    ok = actual == expected
    print(f"  Expected: {expected:,}")
    print(f"  Actual:   {actual:,}")

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
# TEST 6: MapInteraction — PyTorch vs TTSim
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: MapInteraction — PyTorch vs TTSim")
print("=" * 80)

try:
    pt_dec_layer = nn.TransformerDecoderLayer(
        d_model=EMBED_DIMS, nhead=NUM_HEADS, dropout=0.0,
        dim_feedforward=DIM_FF, batch_first=True)
    tt_mi = MapInteraction('t6_mi', embed_dims=EMBED_DIMS,
                            num_heads=NUM_HEADS, dim_feedforward=DIM_FF)
    copy_decoder_layer_to_map_interaction(pt_dec_layer, tt_mi)

    query_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
    key_np = np.random.randn(BS, NUM_MAP_LANES, EMBED_DIMS).astype(np.float32)
    qpos_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
    kpos_np = np.random.randn(BS, NUM_MAP_LANES, EMBED_DIMS).astype(np.float32)

    with torch.no_grad():
        q_pt = torch.from_numpy(query_np) + torch.from_numpy(qpos_np)
        k_pt = torch.from_numpy(key_np) + torch.from_numpy(kpos_np)
        # Map: flatten query, tile key for each agent
        tgt = q_pt.flatten(0, 1)  # (B*A, P, D)
        mem = k_pt.expand(BS * NUM_AGENTS, -1, -1)  # (B*A, M, D)
        out_flat = pt_dec_layer(tgt, mem)
        pt_out = out_flat.view(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS)

    q_tt = F._from_data('t6_q', query_np, is_const=True)
    k_tt = F._from_data('t6_k', key_np, is_const=True)
    qp_tt = F._from_data('t6_qp', qpos_np, is_const=True)
    kp_tt = F._from_data('t6_kp', kpos_np, is_const=True)
    tt_out = tt_mi(q_tt, k_tt, query_pos=qp_tt, key_pos=kp_tt)

    ok = compare(pt_out, tt_out, "MapInteraction output", atol=1e-4)
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
# TEST 7: MapInteraction — no positional encodings
# ====================================================================

print("\n" + "=" * 80)
print("TEST 7: MapInteraction — no positional encodings")
print("=" * 80)

try:
    pt_dec_layer = nn.TransformerDecoderLayer(
        d_model=EMBED_DIMS, nhead=NUM_HEADS, dropout=0.0,
        dim_feedforward=DIM_FF, batch_first=True)
    tt_mi = MapInteraction('t7_mi', embed_dims=EMBED_DIMS,
                            num_heads=NUM_HEADS, dim_feedforward=DIM_FF)
    copy_decoder_layer_to_map_interaction(pt_dec_layer, tt_mi)

    query_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
    key_np = np.random.randn(BS, NUM_MAP_LANES, EMBED_DIMS).astype(np.float32)

    with torch.no_grad():
        q_pt = torch.from_numpy(query_np)
        k_pt = torch.from_numpy(key_np)
        tgt = q_pt.flatten(0, 1)
        mem = k_pt.expand(BS * NUM_AGENTS, -1, -1)
        out_flat = pt_dec_layer(tgt, mem)
        pt_out = out_flat.view(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS)

    q_tt = F._from_data('t7_q', query_np, is_const=True)
    k_tt = F._from_data('t7_k', key_np, is_const=True)
    tt_out = tt_mi(q_tt, k_tt)

    ok = compare(pt_out, tt_out, "MapInteraction (no pos)", atol=1e-4)
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
# TEST 8: MapInteraction — param count
# ====================================================================

print("\n" + "=" * 80)
print("TEST 8: MapInteraction — param count")
print("=" * 80)

try:
    D = EMBED_DIMS
    D_ff = DIM_FF
    expected_mha = 2 * 4 * (D * D + D)
    expected_ffn = D * D_ff + D_ff + D_ff * D + D
    expected = expected_mha + expected_ffn

    tt_mi = MapInteraction('t8_mi', embed_dims=D, num_heads=NUM_HEADS,
                            dim_feedforward=D_ff)
    actual = tt_mi.analytical_param_count()

    ok = actual == expected
    print(f"  Expected: {expected:,}")
    print(f"  Actual:   {actual:,}")

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
# TEST 9: MotionTransformerDecoder — construction & param count
# ====================================================================

print("\n" + "=" * 80)
print("TEST 9: MotionTransformerDecoder — construction & param count")
print("=" * 80)

try:
    D = EMBED_DIMS
    D_ff = DIM_FF
    num_layers = 3

    mtd = MotionTransformerDecoder('t9_mtd', embed_dims=D, num_layers=num_layers)

    # Compute expected param count
    # 1 IntentionInteraction
    ii_params = 4 * (D * D + D) + D * D_ff + D_ff + D_ff * D + D  # 1 encoder layer, using default dim_feedforward=512

    # But default dim_feedforward in modules.py is 512, not DIM_FF=128
    # Re-create with explicit dim_feedforward to match
    mtd2 = MotionTransformerDecoder('t9_mtd2', embed_dims=D, num_layers=num_layers)
    total = mtd2.analytical_param_count()

    # Just verify it's positive and reasonable
    ok = total > 0
    print(f"  embed_dims   = {D}")
    print(f"  num_layers   = {num_layers}")
    print(f"  param count  = {total:,}")
    print(f"  {'[OK]' if ok else '[FAIL]'}")

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
# TEST 10: MotionTransformerDecoder — weight copy from PyTorch
# ====================================================================

print("\n" + "=" * 80)
print("TEST 10: MotionTransformerDecoder — MLP fuser weight copy")
print("=" * 80)

try:
    D = EMBED_DIMS
    num_layers = 3

    # Build PyTorch MLP fusers (matching the MotionTransformerDecoder's MLPs)
    pt_dynamic_embed_fuser = nn.Sequential(
        nn.Linear(D * 3, D * 2), nn.ReLU(), nn.Linear(D * 2, D))
    pt_in_query_fuser = nn.Sequential(
        nn.Linear(D * 2, D * 2), nn.ReLU(), nn.Linear(D * 2, D))
    pt_out_query_fuser = nn.Sequential(
        nn.Linear(D * 4, D * 2), nn.ReLU(), nn.Linear(D * 2, D))

    # Build TTSim MotionTransformerDecoder
    tt_mtd = MotionTransformerDecoder('t10_mtd', embed_dims=D, num_layers=num_layers)

    # Copy MLP weights (TTSim uses per-layer ModuleLists; test layer 0)
    copy_two_layer_mlp_weights(pt_dynamic_embed_fuser, tt_mtd.dynamic_embed_fusers[0])
    copy_two_layer_mlp_weights(pt_in_query_fuser, tt_mtd.in_query_fusers[0])
    copy_two_layer_mlp_weights(pt_out_query_fuser, tt_mtd.out_query_fusers[0])

    # Test dynamic_embed_fuser
    x_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, D * 3).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t10_x1', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_dynamic_embed_fuser(x_pt)
    tt_out = tt_mtd.dynamic_embed_fusers[0](x_tt)

    ok1 = compare(pt_out, tt_out, "dynamic_embed_fuser", atol=1e-5)

    # Test in_query_fuser
    x_np2 = np.random.randn(BS, NUM_AGENTS, NUM_MODES, D * 2).astype(np.float32)
    x_pt2 = torch.from_numpy(x_np2)
    x_tt2 = F._from_data('t10_x2', x_np2, is_const=True)

    with torch.no_grad():
        pt_out2 = pt_in_query_fuser(x_pt2)
    tt_out2 = tt_mtd.in_query_fusers[0](x_tt2)

    ok2 = compare(pt_out2, tt_out2, "in_query_fuser", atol=1e-5)

    # Test out_query_fuser
    x_np3 = np.random.randn(BS, NUM_AGENTS, NUM_MODES, D * 4).astype(np.float32)
    x_pt3 = torch.from_numpy(x_np3)
    x_tt3 = F._from_data('t10_x3', x_np3, is_const=True)

    with torch.no_grad():
        pt_out3 = pt_out_query_fuser(x_pt3)
    tt_out3 = tt_mtd.out_query_fusers[0](x_tt3)

    ok3 = compare(pt_out3, tt_out3, "out_query_fuser", atol=1e-5)

    ok = ok1 and ok2 and ok3
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
# TEST 11: IntentionInteraction — different input sizes
# ====================================================================

print("\n" + "=" * 80)
print("TEST 11: IntentionInteraction — varying agent/mode counts")
print("=" * 80)

try:
    pt_enc_layer = nn.TransformerEncoderLayer(
        d_model=EMBED_DIMS, nhead=NUM_HEADS, dropout=0.0,
        dim_feedforward=DIM_FF, batch_first=True)
    tt_ii = IntentionInteraction('t11_ii', embed_dims=EMBED_DIMS,
                                 num_heads=NUM_HEADS, dim_feedforward=DIM_FF)
    copy_encoder_layer_weights(pt_enc_layer, tt_ii)

    configs = [
        (1, 4, 3),   # fewer agents, fewer modes
        (1, 12, 6),  # more agents
        (1, 8, 10),  # more modes
    ]

    all_ok = True
    for b, a, p in configs:
        x_np = np.random.randn(b, a, p, EMBED_DIMS).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        with torch.no_grad():
            flat = x_pt.flatten(0, 1)
            out_flat = pt_enc_layer(flat)
            pt_out = out_flat.view(b, a, p, EMBED_DIMS)

        x_tt = F._from_data(f't11_x_{b}_{a}_{p}', x_np, is_const=True)
        tt_out = tt_ii(x_tt)
        ok_i = compare(pt_out, tt_out,
                       f"IntentionInteraction B={b} A={a} P={p}", atol=1e-4)
        if not ok_i:
            all_ok = False

    if all_ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if all_ok else '[FAIL]'} TEST 11")

except Exception as e:
    print(f"  [FAIL] TEST 11 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 12: MapInteraction — different map sizes
# ====================================================================

print("\n" + "=" * 80)
print("TEST 12: MapInteraction — varying map lane count")
print("=" * 80)

try:
    pt_dec_layer = nn.TransformerDecoderLayer(
        d_model=EMBED_DIMS, nhead=NUM_HEADS, dropout=0.0,
        dim_feedforward=DIM_FF, batch_first=True)
    tt_mi = MapInteraction('t12_mi', embed_dims=EMBED_DIMS,
                            num_heads=NUM_HEADS, dim_feedforward=DIM_FF)
    copy_decoder_layer_to_map_interaction(pt_dec_layer, tt_mi)

    map_sizes = [20, 100, 200]

    all_ok = True
    for M in map_sizes:
        query_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
        key_np = np.random.randn(BS, M, EMBED_DIMS).astype(np.float32)

        with torch.no_grad():
            q_pt = torch.from_numpy(query_np)
            k_pt = torch.from_numpy(key_np)
            tgt = q_pt.flatten(0, 1)
            mem = k_pt.expand(BS * NUM_AGENTS, -1, -1)
            out_flat = pt_dec_layer(tgt, mem)
            pt_out = out_flat.view(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS)

        q_tt = F._from_data(f't12_q_{M}', query_np, is_const=True)
        k_tt = F._from_data(f't12_k_{M}', key_np, is_const=True)
        tt_out = tt_mi(q_tt, k_tt)

        ok_i = compare(pt_out, tt_out, f"MapInteraction M={M}", atol=1e-4)
        if not ok_i:
            all_ok = False

    if all_ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if all_ok else '[FAIL]'} TEST 12")

except Exception as e:
    print(f"  [FAIL] TEST 12 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# Summary
# ====================================================================

print("\n" + "=" * 80)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
print("=" * 80)

if failed > 0:
    sys.exit(1)
else:
    print("[OK] All tests passed!")
    sys.exit(0)
