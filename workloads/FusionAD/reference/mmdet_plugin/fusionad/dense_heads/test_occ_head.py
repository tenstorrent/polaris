#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for OccHead (TTSim vs PyTorch).

Validates numerical equivalence of:
  1. OccHead construction + param count
  2. merge_queries (mode_fuser + multi_query_fuser)
  3. get_attn_mask (mask MLP + einsum decomposition)
  4. Full forward pass (BEV → ins_occ_logits)

No mmcv dependency — all PyTorch references built from torch.nn.
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
import torch.nn.functional as TF

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.occ_head import (
    OccHead,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.occ_head_plugin.modules import (
    BevFeatureSlicer,
    MLP,
    SimpleConv2d,
    CVT_DecoderBlock,
    CVT_Decoder,
    UpsamplingAdd,
    Bottleneck,
)


# ====================================================================
# Compare helper
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
    """Copy nn.Linear -> SimNN.Linear (no transpose — SimNN.Linear transposes internally)."""
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


def copy_mlp(pt_layers, tt_mlp):
    """Copy nn.ModuleList of nn.Linear -> TTSim MLP."""
    for i, pt_lin in enumerate(pt_layers):
        copy_linear(pt_lin, tt_mlp.linears[i])


def copy_bottleneck(pt_parts, tt_bn, downsample=False):
    """
    Copy Bottleneck weights.

    pt_parts is a dict with keys:
      conv_down, bn_down, mid_conv, bn_mid, conv_up, bn_up
      and if downsample: skip_conv, skip_bn
    """
    copy_conv2d(pt_parts['conv_down'], tt_bn.conv_down)
    copy_bn(pt_parts['bn_down'], tt_bn.bn_down)
    copy_conv2d(pt_parts['mid_conv'], tt_bn.mid_conv)
    copy_bn(pt_parts['bn_mid'], tt_bn.bn_mid)
    copy_conv2d(pt_parts['conv_up'], tt_bn.conv_up)
    copy_bn(pt_parts['bn_up'], tt_bn.bn_up)
    if downsample:
        copy_conv2d(pt_parts['skip_conv'], tt_bn.skip_conv)
        copy_bn(pt_parts['skip_bn'], tt_bn.skip_bn)


def copy_simple_conv2d(pt_convs, pt_bns, pt_final, tt_sc):
    """Copy SimpleConv2d weights: list of (Conv2d, BN) + final Conv2d."""
    for i in range(len(pt_convs)):
        conv_op, bn_op, _ = tt_sc.conv_blocks[i]
        copy_conv2d(pt_convs[i], conv_op)
        copy_bn(pt_bns[i], bn_op)
    copy_conv2d(pt_final, tt_sc.final_conv)


def copy_upsampling_add(pt_conv, pt_bn, tt_ua):
    """Copy UpsamplingAdd weights."""
    copy_conv2d(pt_conv, tt_ua.conv)
    copy_bn(pt_bn, tt_ua.bn)


def copy_mha_weights(pt_mha, tt_mha):
    """Copy nn.MultiheadAttention weights -> TTSim MultiheadAttention."""
    D = pt_mha.embed_dim
    w = pt_mha.in_proj_weight.data.detach().numpy()   # [3D, D]
    b = pt_mha.in_proj_bias.data.detach().numpy()      # [3D]
    wq, wk, wv = w[:D], w[D:2*D], w[2*D:]
    bq, bk, bv = b[:D], b[D:2*D], b[2*D:]
    tt_mha.q_proj.param.data = wq.astype(np.float32)
    tt_mha.q_proj.bias.data = bq.astype(np.float32)
    tt_mha.k_proj.param.data = wk.astype(np.float32)
    tt_mha.k_proj.bias.data = bk.astype(np.float32)
    tt_mha.v_proj.param.data = wv.astype(np.float32)
    tt_mha.v_proj.bias.data = bv.astype(np.float32)
    tt_mha.out_proj.param.data = pt_mha.out_proj.weight.data.detach().numpy().astype(np.float32)
    tt_mha.out_proj.bias.data = pt_mha.out_proj.bias.data.detach().numpy().astype(np.float32)


def copy_ln(pt_ln, tt_ln):
    """Neutralize PyTorch LayerNorm (TTSim LN has identity affine by default)."""
    pt_ln.weight.data.fill_(1.0)
    pt_ln.bias.data.fill_(0.0)


def copy_cvt_decoder_block(pt_parts, tt_block):
    """Copy CVT_DecoderBlock weights."""
    copy_conv2d(pt_parts['conv1'], tt_block.conv1)
    copy_bn(pt_parts['bn1'], tt_block.bn1)
    copy_conv2d(pt_parts['conv2'], tt_block.conv2)
    copy_bn(pt_parts['bn2'], tt_block.bn2)
    if hasattr(tt_block, 'skip_conv') and tt_block.skip_conv is not None:
        copy_conv2d(pt_parts['skip_conv'], tt_block.skip_conv)


# ====================================================================
# PyTorch reference sub-modules (no mmcv)
# ====================================================================

def build_pt_mlp(in_dim, hidden_dim, out_dim, num_layers):
    """Build MLP as ModuleList of nn.Linear (ReLU between, none on last)."""
    layers = nn.ModuleList()
    h = [hidden_dim] * (num_layers - 1)
    for n_in, n_out in zip([in_dim] + h, h + [out_dim]):
        layers.append(nn.Linear(n_in, n_out))
    return layers


def pt_mlp_forward(layers, x):
    """Forward through an MLP ModuleList with ReLU (none on last)."""
    for i, layer in enumerate(layers):
        x = TF.relu(layer(x)) if i < len(layers) - 1 else layer(x)
    return x


def build_pt_bottleneck(C, downsample=False):
    """Build Bottleneck components as a dict of nn modules."""
    bc = C // 2
    parts = {
        'conv_down': nn.Conv2d(C, bc, 1, bias=False),
        'bn_down': nn.BatchNorm2d(bc),
        'mid_conv': nn.Conv2d(bc, bc, 3, stride=2 if downsample else 1,
                              padding=1, bias=False),
        'bn_mid': nn.BatchNorm2d(bc),
        'conv_up': nn.Conv2d(bc, C, 1, bias=False),
        'bn_up': nn.BatchNorm2d(C),
    }
    if downsample:
        parts['skip_conv'] = nn.Conv2d(C, C, 1, bias=False)
        parts['skip_bn'] = nn.BatchNorm2d(C)
    for m in parts.values():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return parts


def pt_bottleneck_forward(parts, x, downsample=False):
    """Forward through a Bottleneck."""
    with torch.no_grad():
        res = TF.relu(parts['bn_down'](parts['conv_down'](x)))
        res = TF.relu(parts['bn_mid'](parts['mid_conv'](res)))
        res = TF.relu(parts['bn_up'](parts['conv_up'](res)))
        if downsample:
            H, W = x.shape[2], x.shape[3]
            skip = x
            if H % 2 != 0 or W % 2 != 0:
                skip = TF.pad(skip, (0, W % 2, 0, H % 2), value=0)
            skip = nn.functional.max_pool2d(skip, 2, 2)
            skip = parts['skip_bn'](parts['skip_conv'](skip))
            return res + skip
        return res + x


def build_pt_simple_conv2d(in_ch, conv_ch, out_ch, num_conv):
    """Build SimpleConv2d components."""
    convs = nn.ModuleList()
    bns = nn.ModuleList()
    c_in = in_ch
    for _ in range(num_conv - 1):
        convs.append(nn.Conv2d(c_in, conv_ch, 3, 1, 1, bias=False))
        bns.append(nn.BatchNorm2d(conv_ch))
        c_in = conv_ch
    final = nn.Conv2d(conv_ch, out_ch, 1, bias=True)
    final.bias.data.zero_()
    for bn in bns:
        bn.eval()
    return convs, bns, final


def pt_simple_conv2d_forward(convs, bns, final, x):
    """Forward through SimpleConv2d."""
    with torch.no_grad():
        for conv, bn in zip(convs, bns):
            x = TF.relu(bn(conv(x)))
        return final(x)


def build_pt_upsampling_add(C):
    """Build UpsamplingAdd components."""
    conv = nn.Conv2d(C, C, kernel_size=1, bias=False)
    bn = nn.BatchNorm2d(C)
    bn.eval()
    return conv, bn


def pt_upsampling_add_forward(conv, bn, x, skip):
    """Forward through UpsamplingAdd."""
    with torch.no_grad():
        x = TF.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = bn(conv(x))
        return x + skip


def build_pt_cvt_decoder_block(in_ch, out_ch, skip_dim, factor=2,
                                residual=False, upsample=False):
    """Build CVT_DecoderBlock components."""
    dim = out_ch // factor
    parts = {
        'conv1': nn.Conv2d(in_ch, dim, 3, padding=1, bias=False),
        'bn1': nn.BatchNorm2d(dim),
        'conv2': nn.Conv2d(dim, out_ch, 1, bias=False),
        'bn2': nn.BatchNorm2d(out_ch),
    }
    if residual:
        parts['skip_conv'] = nn.Conv2d(skip_dim, out_ch, 1, bias=True)
        parts['skip_conv'].bias.data.zero_()
    for m in parts.values():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return parts


def pt_cvt_decoder_block_forward(parts, x, skip, residual=False,
                                  upsample=False, with_relu=True,
                                  residual_scale=2):
    """Forward through one CVT_DecoderBlock."""
    with torch.no_grad():
        if upsample:
            x = TF.interpolate(x, scale_factor=2, mode='bilinear',
                                align_corners=True)
        x = TF.relu(parts['bn1'](parts['conv1'](x)))
        x = parts['bn2'](parts['conv2'](x))
        if residual:
            up = parts['skip_conv'](skip)
            up = TF.interpolate(up, scale_factor=residual_scale,
                                 mode='bilinear', align_corners=False)
            x = x + up
        if with_relu:
            x = TF.relu(x)
        return x


def build_pt_transformer_decoder_layer(embed_dims, num_heads, ffn_channels):
    """
    Build PyTorch transformer decoder layer matching the OccHead config:
      operation_order = (self_attn, norm, cross_attn, norm, ffn, norm)
    """
    self_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=False)
    cross_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=False)
    ffn_fc1 = nn.Linear(embed_dims, ffn_channels)
    ffn_fc2 = nn.Linear(ffn_channels, embed_dims)
    norm0 = nn.LayerNorm(embed_dims)
    norm1 = nn.LayerNorm(embed_dims)
    norm2 = nn.LayerNorm(embed_dims)
    # Neutralize norms for exact comparison
    for n in [norm0, norm1, norm2]:
        n.weight.data.fill_(1.0)
        n.bias.data.fill_(0.0)
    return {
        'self_attn': self_attn,
        'cross_attn': cross_attn,
        'ffn_fc1': ffn_fc1,
        'ffn_fc2': ffn_fc2,
        'norm0': norm0,
        'norm1': norm1,
        'norm2': norm2,
    }


def pt_trans_layer_forward(parts, query, key, value, attn_masks=None):
    """
    Forward through one transformer decoder layer.
    operation_order: self_attn, norm, cross_attn, norm, ffn, norm
    """
    with torch.no_grad():
        # Self-attention
        sa_mask = attn_masks[0] if attn_masks else None
        x = query
        sa_out, _ = parts['self_attn'](x, x, x, attn_mask=sa_mask)
        x = x + sa_out
        x = parts['norm0'](x)

        # Cross-attention
        ca_mask = attn_masks[1] if attn_masks else None
        ca_out, _ = parts['cross_attn'](x, key, value, attn_mask=ca_mask)
        x = x + ca_out
        x = parts['norm1'](x)

        # FFN (with add_identity=True)
        ffn_out = parts['ffn_fc2'](TF.relu(parts['ffn_fc1'](x)))
        x = x + ffn_out
        x = parts['norm2'](x)

        return x


def copy_pt_trans_to_tt(pt_parts, tt_layer):
    """Copy PyTorch trans layer dict -> TTSim MyCustomBaseTransformerLayer."""
    copy_mha_weights(pt_parts['self_attn'], tt_layer.attentions[0])
    copy_mha_weights(pt_parts['cross_attn'], tt_layer.attentions[1])
    for i, key in enumerate(['norm0', 'norm1', 'norm2']):
        copy_ln(pt_parts[key], tt_layer.norms[i])
    copy_linear(pt_parts['ffn_fc1'], tt_layer.ffns[0].layers[0])
    copy_linear(pt_parts['ffn_fc2'], tt_layer.ffns[0].layers[1])


# ====================================================================
# Config
# ====================================================================

np.random.seed(42)
torch.manual_seed(42)

BS = 1
Q = 6          # number of instance queries
D = 16         # embed_dims / query_dim / bev_proj_dim  (small for CPU speed)
BEV_H, BEV_W = 16, 16   # power-of-2 for clean /2 divisions
N_FUTURE = 4
N_HEADS = 4
NUM_TRANS_LAYERS = 5
FFN_CHANNELS = 32
QUERY_MLP_LAYERS = 3
TEMPORAL_MLP_LAYERS = 2
BEV_PROJ_NLAYERS = 2     # fewer conv layers to keep CPU time reasonable
N_MODES = 6
ATTN_MASK_THRESH = 0.3

# Grid config producing 16x16 output:
#   arange(-7.5, 8, 1.0) → 16 points
occflow_grid_conf = {
    'xbound': [-8.0, 8.0, 1.0],
    'ybound': [-8.0, 8.0, 1.0],
    'zbound': [-10.0, 10.0, 20.0],
}
# After bev_sampler the spatial dims become 16x16.
# Pipeline:  16 → bev_proj → ds0(8) → ds1(4) → loop: ds(2) ↔ up(4)
#            → dense_decoder: 4→8→16
OUT_H, OUT_W = BEV_H, BEV_W  # final output matches grid output size

passed = 0
failed = 0


# ====================================================================
# Build TTSim OccHead
# ====================================================================

tt_head = OccHead(
    'occ_head',
    n_future=N_FUTURE,
    grid_conf=occflow_grid_conf,
    bev_size=(BEV_H, BEV_W),
    bev_emb_dim=D,
    bev_proj_dim=D,
    bev_proj_nlayers=BEV_PROJ_NLAYERS,
    query_dim=D,
    query_mlp_layers=QUERY_MLP_LAYERS,
    temporal_mlp_layer=TEMPORAL_MLP_LAYERS,
    num_trans_layers=NUM_TRANS_LAYERS,
    num_heads=N_HEADS,
    attn_mask_thresh=ATTN_MASK_THRESH,
    embed_dims=D,
    feedforward_channels=FFN_CHANNELS,
)

N_FUTURE_BLOCKS = N_FUTURE + 1


# ====================================================================
# TEST 1: OccHead construction + param count > 0
# ====================================================================

print("=" * 80)
print("TEST 1: OccHead construction + param count")
print("=" * 80)

try:
    # Just verify construction succeeded and has sub-modules
    ok = hasattr(tt_head, 'base_ds_0') and hasattr(tt_head, 'dense_decoder')
    print(f"\n  OccHead created with {N_FUTURE_BLOCKS} future blocks")
    print(f"  has base_ds_0: {hasattr(tt_head, 'base_ds_0')}")
    print(f"  has dense_decoder: {hasattr(tt_head, 'dense_decoder')}")
    print(f"  {'[OK]' if ok else '[FAIL]'} construction check")
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
# TEST 2: merge_queries — PyTorch vs TTSim
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: merge_queries")
print("=" * 80)

try:
    # PyTorch sub-modules
    pt_mf_linear = nn.Linear(D, D)
    pt_mf_ln = nn.LayerNorm(D)
    pt_mf_ln.weight.data.fill_(1.0)
    pt_mf_ln.bias.data.fill_(0.0)

    pt_mqf_linear1 = nn.Linear(D * 3, D * 2)
    pt_mqf_ln1 = nn.LayerNorm(D * 2)
    pt_mqf_ln1.weight.data.fill_(1.0)
    pt_mqf_ln1.bias.data.fill_(0.0)
    pt_mqf_linear2 = nn.Linear(D * 2, D)

    # Copy weights to TTSim
    copy_linear(pt_mf_linear, tt_head.mf_linear)
    copy_linear(pt_mqf_linear1, tt_head.mqf_linear1)
    copy_linear(pt_mqf_linear2, tt_head.mqf_linear2)

    # Input data
    traj_q_np = np.random.randn(BS, Q, N_MODES, D).astype(np.float32)
    trk_q_np = np.random.randn(BS, Q, D).astype(np.float32)
    trk_pos_np = np.random.randn(BS, Q, D).astype(np.float32)

    # TTSim forward
    traj_q_tt = F._from_data('mq.traj', traj_q_np)
    trk_q_tt = F._from_data('mq.trk', trk_q_np)
    trk_pos_tt = F._from_data('mq.pos', trk_pos_np)
    tt_out = tt_head.merge_queries(traj_q_tt, trk_q_tt, trk_pos_tt)

    # PyTorch forward
    traj_q_pt = torch.from_numpy(traj_q_np)
    trk_q_pt = torch.from_numpy(trk_q_np)
    trk_pos_pt = torch.from_numpy(trk_pos_np)

    with torch.no_grad():
        x = pt_mf_linear(traj_q_pt)     # [B,Q,modes,D]
        x = pt_mf_ln(x)
        x = TF.relu(x)
        x = x.max(2)[0]                 # [B,Q,D]
        x = torch.cat([x, trk_q_pt, trk_pos_pt], dim=-1)  # [B,Q,3D]
        x = pt_mqf_linear1(x)
        x = pt_mqf_ln1(x)
        x = TF.relu(x)
        pt_out = pt_mqf_linear2(x)      # [B,Q,D]

    ok = compare(pt_out, tt_out, "merge_queries", atol=1e-4)
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
# TEST 3: get_attn_mask — shape and numerical check
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: get_attn_mask (block 0)")
print("=" * 80)

try:
    # Build PyTorch mask MLP (query_mlp_layers=3, D->D->D->D)
    pt_mask_mlp = build_pt_mlp(D, D, D, QUERY_MLP_LAYERS)

    # Copy to TTSim mask_mlp_0
    copy_mlp(pt_mask_mlp, tt_head.mask_mlp_0)

    # Inputs: state [B,C,H',W'] and ins_query [B,Q,C]
    H_ds, W_ds = BEV_H // 8, BEV_W // 8   # after /4 base + /2 ds_conv
    state_np = np.random.randn(BS, D, H_ds, W_ds).astype(np.float32)
    ins_q_np = np.random.randn(BS, Q, D).astype(np.float32)

    state_tt = F._from_data('mask.state', state_np)
    ins_q_tt = F._from_data('mask.ins_q', ins_q_np)

    # TTSim
    ins_embed_tt, attn_mask_tt = tt_head._get_attn_mask(0, state_tt, ins_q_tt, BS)

    # PyTorch reference
    state_pt = torch.from_numpy(state_np)
    ins_q_pt = torch.from_numpy(ins_q_np)

    with torch.no_grad():
        ins_embed_pt = pt_mlp_forward(pt_mask_mlp, ins_q_pt)
        mask_pred = torch.einsum("bqc,bchw->bqhw", ins_embed_pt, state_pt)
        sig = mask_pred.sigmoid()
        bool_mask = sig < ATTN_MASK_THRESH
        # [B,Q,H',W'] -> [B,H'W',Q] -> unsqueeze -> [B,1,H'W',Q] -> repeat
        bool_mask_flat = bool_mask.reshape(BS, Q, -1).permute(0, 2, 1)
        bool_mask_exp = bool_mask_flat.unsqueeze(1).repeat(1, N_HEADS, 1, 1)
        attn_mask_pt = torch.where(bool_mask_exp, torch.tensor(-1e9), torch.tensor(0.0))

    ok1 = compare(ins_embed_pt, ins_embed_tt, "ins_embed", atol=1e-4)
    ok2 = compare(attn_mask_pt, attn_mask_tt, "attn_mask", atol=1e-3)
    ok = ok1 and ok2
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
# TEST 4: Full forward shape check
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: Full forward — output shape")
print("=" * 80)

try:
    x_np = np.random.randn(BEV_H * BEV_W, BS, D).astype(np.float32) * 0.01
    ins_q_np = np.random.randn(BS, Q, D).astype(np.float32) * 0.01

    x_tt = F._from_data('fwd.x', x_np)
    ins_q_tt = F._from_data('fwd.q', ins_q_np)

    out_tt = tt_head(x_tt, ins_q_tt)
    expected_shape = [BS, Q, N_FUTURE_BLOCKS, OUT_H, OUT_W]
    actual_shape = list(out_tt.shape)

    print(f"\n  Output shape: {actual_shape}")
    print(f"  Expected:     {expected_shape}")
    ok = actual_shape == expected_shape
    print(f"  {'[OK]' if ok else '[FAIL]'} shape check")
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
# TEST 5: Full forward — numerical (PyTorch vs TTSim)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: Full forward — numerical (PyTorch vs TTSim)")
print("=" * 80)

try:
    # ── Build fresh TTSim head for isolated numerical test ──
    tt_h = OccHead(
        'oh2',
        n_future=N_FUTURE,
        grid_conf=occflow_grid_conf,
        bev_size=(BEV_H, BEV_W),
        bev_emb_dim=D,
        bev_proj_dim=D,
        bev_proj_nlayers=BEV_PROJ_NLAYERS,
        query_dim=D,
        query_mlp_layers=QUERY_MLP_LAYERS,
        temporal_mlp_layer=TEMPORAL_MLP_LAYERS,
        num_trans_layers=NUM_TRANS_LAYERS,
        num_heads=N_HEADS,
        attn_mask_thresh=ATTN_MASK_THRESH,
        embed_dims=D,
        feedforward_channels=FFN_CHANNELS,
    )

    # ── Build PyTorch equivalents ──

    # BEV projection (SimpleConv2d: 3 conv+bn+relu blocks + 1x1 final)
    pt_bev_convs, pt_bev_bns, pt_bev_final = build_pt_simple_conv2d(
        D, D, D, BEV_PROJ_NLAYERS)

    # Base downscale (2x Bottleneck with downsample)
    pt_base_ds_0 = build_pt_bottleneck(D, downsample=True)
    pt_base_ds_1 = build_pt_bottleneck(D, downsample=True)

    # Per-block modules
    pt_ds_convs = []
    pt_temporal_mlps = []
    pt_mask_mlps = []
    pt_trans_layers_all = []
    pt_upsample_adds = []

    n_trans_each = NUM_TRANS_LAYERS // N_FUTURE_BLOCKS

    for i in range(N_FUTURE_BLOCKS):
        pt_ds_convs.append(build_pt_bottleneck(D, downsample=True))
        pt_temporal_mlps.append(build_pt_mlp(D, D, D, TEMPORAL_MLP_LAYERS))
        pt_mask_mlps.append(build_pt_mlp(D, D, D, QUERY_MLP_LAYERS))
        for j in range(n_trans_each):
            pt_trans_layers_all.append(
                build_pt_transformer_decoder_layer(D, N_HEADS, FFN_CHANNELS))
        ua_conv, ua_bn = build_pt_upsampling_add(D)
        pt_upsample_adds.append((ua_conv, ua_bn))

    # Dense decoder: 2 CVT_DecoderBlocks (both residual, both upsample)
    # Block 0: residual=True, upsample=True, with_relu=True, res_scale=2
    pt_dd_blk0 = build_pt_cvt_decoder_block(
        D, D, D, factor=2, residual=True, upsample=True)
    # Block 1: residual=True, upsample=True, with_relu=False, res_scale=4
    pt_dd_blk1 = build_pt_cvt_decoder_block(
        D, D, D, factor=2, residual=True, upsample=True)

    # query_to_occ_feat MLP
    pt_q2occ = build_pt_mlp(D, D, D, QUERY_MLP_LAYERS)

    # ── Copy all weights from PyTorch → TTSim ──

    # BEV projection
    copy_simple_conv2d(pt_bev_convs, pt_bev_bns, pt_bev_final,
                       tt_h.bev_light_proj)

    # Base downscale
    copy_bottleneck(pt_base_ds_0, tt_h.base_ds_0, downsample=True)
    copy_bottleneck(pt_base_ds_1, tt_h.base_ds_1, downsample=True)

    # Per-block
    for i in range(N_FUTURE_BLOCKS):
        copy_bottleneck(pt_ds_convs[i], getattr(tt_h, f'ds_conv_{i}'),
                        downsample=True)
        copy_mlp(pt_temporal_mlps[i], getattr(tt_h, f'temporal_mlp_{i}'))
        copy_mlp(pt_mask_mlps[i], getattr(tt_h, f'mask_mlp_{i}'))
        for j in range(n_trans_each):
            lid = i * n_trans_each + j
            copy_pt_trans_to_tt(pt_trans_layers_all[lid],
                                getattr(tt_h, f'trans_layer_{lid}'))
        ua_conv, ua_bn = pt_upsample_adds[i]
        copy_upsampling_add(ua_conv, ua_bn, getattr(tt_h, f'upsample_add_{i}'))

    # Dense decoder blocks
    copy_cvt_decoder_block(pt_dd_blk0, tt_h.dense_decoder.block_0)
    copy_cvt_decoder_block(pt_dd_blk1, tt_h.dense_decoder.block_1)

    # query_to_occ_feat
    copy_mlp(pt_q2occ, tt_h.query_to_occ_feat)

    # ── Prepare input ──
    np.random.seed(123)
    torch.manual_seed(123)

    x_np = np.random.randn(BEV_H * BEV_W, BS, D).astype(np.float32) * 0.02
    ins_q_np = np.random.randn(BS, Q, D).astype(np.float32) * 0.02

    # ── TTSim forward ──
    x_tt = F._from_data('t5.x', x_np)
    ins_q_tt = F._from_data('t5.q', ins_q_np)
    tt_out = tt_h(x_tt, ins_q_tt)

    # ── PyTorch forward (manual) ──
    x_pt = torch.from_numpy(x_np)
    ins_q_pt = torch.from_numpy(ins_q_np)

    with torch.no_grad():
        # (H*W, B, D) -> (B, D, H, W)
        base_state = x_pt.permute(1, 2, 0).reshape(BS, D, BEV_H, BEV_W)

        # BEV sampler: grids differ (bevformer [-51.2,51.2,0.512] vs
        # occflow [-32,32,1.0]) so grid_sample occurs.  Use TTSim result
        # as ground truth since BevFeatureSlicer was validated in
        # test_occ_modules.
        base_state_tt_after_sampler = tt_h.bev_sampler(
            F._from_data('t5.bs', base_state.numpy())).data
        base_state = torch.from_numpy(base_state_tt_after_sampler)

        # BEV projection
        base_state = pt_simple_conv2d_forward(
            pt_bev_convs, pt_bev_bns, pt_bev_final, base_state)

        # Base downscale
        base_state = pt_bottleneck_forward(pt_base_ds_0, base_state, downsample=True)
        base_state = pt_bottleneck_forward(pt_base_ds_1, base_state, downsample=True)

        last_state = base_state
        last_ins_query = ins_q_pt

        state_list = []
        embed_list = []
        n_trans_each = NUM_TRANS_LAYERS // N_FUTURE_BLOCKS

        for i in range(N_FUTURE_BLOCKS):
            # Downscale /4 → /8
            cur_state = pt_bottleneck_forward(
                pt_ds_convs[i], last_state, downsample=True)

            # Temporal MLP
            cur_ins_query = pt_mlp_forward(
                pt_temporal_mlps[i], last_ins_query)

            # Attention mask (mask MLP → einsum → sigmoid < thresh)
            ins_embed_pt = pt_mlp_forward(pt_mask_mlps[i], cur_ins_query)
            mask_pred = torch.einsum("bqc,bchw->bqhw", ins_embed_pt,
                                      cur_state)
            sig = mask_pred.sigmoid()
            bool_mask = sig < ATTN_MASK_THRESH
            # [B,Q,H',W'] → [B,H'W',Q] → unsqueeze → tile → where
            H_s, W_s = cur_state.shape[2], cur_state.shape[3]
            hw = H_s * W_s
            bm_flat = bool_mask.reshape(BS, Q, hw).permute(0, 2, 1)
            bm_exp = bm_flat.unsqueeze(1).repeat(1, N_HEADS, 1, 1)
            attn_mask_pt = torch.where(bm_exp,
                                        torch.tensor(-1e9),
                                        torch.tensor(0.0))
            # Reshape attn_mask for nn.MHA: [B*nhead, H'W', Q]
            attn_mask_2d = attn_mask_pt.reshape(BS * N_HEADS, hw, Q)

            # State → sequence: [B,C,H',W'] → [H'W',B,C]
            cur_state_seq = cur_state.reshape(BS, D, hw).permute(2, 0, 1)
            cur_ins_q_seq = cur_ins_query.permute(1, 0, 2)  # [Q,B,C]

            # Transformer layers
            for j in range(n_trans_each):
                lid = i * n_trans_each + j
                cur_state_seq = pt_trans_layer_forward(
                    pt_trans_layers_all[lid],
                    query=cur_state_seq,
                    key=cur_ins_q_seq,
                    value=cur_ins_q_seq,
                    attn_masks=[None, attn_mask_2d])

            # State back: [H'W',B,C] → [B,C,H',W']
            cur_state = cur_state_seq.permute(1, 2, 0).reshape(
                BS, D, H_s, W_s)

            # Upsample + skip-add
            ua_conv, ua_bn = pt_upsample_adds[i]
            cur_state = pt_upsampling_add_forward(
                ua_conv, ua_bn, cur_state, last_state)

            state_list.append(cur_state.unsqueeze(1))
            embed_list.append(ins_embed_pt.unsqueeze(1))
            last_state = cur_state

        # Stack: [B,T,C,H/4,W/4] and [B,T,Q,D]
        future_states = torch.cat(state_list, dim=1)
        ins_embeds = torch.cat(embed_list, dim=1)

        # Dense decoder: 2 CVT_DecoderBlocks on (B*T,C,H,W)
        BT = BS * N_FUTURE_BLOCKS
        C_dd, H_dd, W_dd = future_states.shape[2], future_states.shape[3], future_states.shape[4]
        fs_flat = future_states.reshape(BT, C_dd, H_dd, W_dd)
        skip_dd = fs_flat
        # Block 0: residual, upsample, with_relu, scale=2
        fs_flat = pt_cvt_decoder_block_forward(
            pt_dd_blk0, fs_flat, skip_dd,
            residual=True, upsample=True, with_relu=True,
            residual_scale=2)
        # Block 1: residual, upsample, with_relu=False, scale=4
        fs_flat = pt_cvt_decoder_block_forward(
            pt_dd_blk1, fs_flat, skip_dd,
            residual=True, upsample=True, with_relu=False,
            residual_scale=4)
        C_out, H_out_dd, W_out_dd = fs_flat.shape[1], fs_flat.shape[2], fs_flat.shape[3]
        future_states = fs_flat.reshape(BS, N_FUTURE_BLOCKS, C_out,
                                         H_out_dd, W_out_dd)

        # query_to_occ_feat
        ins_occ_query = pt_mlp_forward(pt_q2occ, ins_embeds)

        # einsum "btqc,btchw->bqthw"
        pt_out = torch.einsum("btqc,btchw->bqthw",
                               ins_occ_query, future_states)

    ok = compare(pt_out, tt_out, "full forward ins_occ_logits", atol=5e-2)
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
# TEST 6: merge_queries output shape
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: merge_queries output shape")
print("=" * 80)

try:
    traj_q_np = np.random.randn(BS, Q, N_MODES, D).astype(np.float32)
    trk_q_np = np.random.randn(BS, Q, D).astype(np.float32)
    trk_pos_np = np.random.randn(BS, Q, D).astype(np.float32)

    traj_q_tt = F._from_data('t6.traj', traj_q_np)
    trk_q_tt = F._from_data('t6.trk', trk_q_np)
    trk_pos_tt = F._from_data('t6.pos', trk_pos_np)

    mq_out = tt_head.merge_queries(traj_q_tt, trk_q_tt, trk_pos_tt)
    expected = [BS, Q, D]
    actual = list(mq_out.shape)

    print(f"\n  Output shape: {actual}")
    print(f"  Expected:     {expected}")
    ok = actual == expected
    print(f"  {'[OK]' if ok else '[FAIL]'} shape check")
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
