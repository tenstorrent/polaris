#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Block-by-Block Decoder Debugger
================================

Isolates the exact point of divergence in the multi-layer decoder by:
  1. Comparing reference_points expansion (decoder wrapper)
  2. Running each sub-block of each layer independently
  3. Feeding identical intermediate values to pinpoint accumulation vs single-step errors

Blocks per decoder layer:
  A. q = k = tgt + query_pos           (pos embedding)
  B. tgt2 = self_attn(q, k, tgt)       (self-attention)
  C. tgt = norm(tgt + dropout(tgt2))    (residual + norm after self-attn)
  D. tgt2 = cross_attn(tgt+pos, ref, src)  (cross-attention)
  E. tgt = norm(tgt + dropout(tgt2))    (residual + norm after cross-attn)
  F. tgt = forward_ffn(tgt)             (FFN block)
"""

import os, sys
import torch
import torch.nn as nn
import numpy as np

# Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from workloads.Deformable_DETR.reference.deformable_transformer import (
    DeformableTransformerDecoderLayer as DecoderLayerPT,
    DeformableTransformerDecoder as DecoderPT,
)
from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
    DeformableTransformerDecoderLayer as DecoderLayerTT,
    DeformableTransformerDecoder as DecoderTT,
)
from ttsim.ops.tensor import SimTensor

# ============================================================================
# Helpers
# ============================================================================


def to_sim(t, name="t", module=None):
    """Torch tensor → SimTensor with data + link_module."""
    d = t.detach().cpu().numpy().copy()
    s = SimTensor({"name": name, "shape": list(d.shape), "data": d, "dtype": d.dtype})
    if module is not None:
        s.link_module = module
        module._tensors[name] = s
    return s


def compare(label, pt_val, tt_val, atol=1e-4):
    """Compare numpy arrays, return (ok, max_diff)."""
    pt = (
        pt_val.detach().cpu().numpy()
        if isinstance(pt_val, torch.Tensor)
        else np.asarray(pt_val)
    )
    tt = tt_val.data if isinstance(tt_val, SimTensor) else np.asarray(tt_val)
    if pt.shape != tt.shape:
        print(f"  ✗ {label}: SHAPE MISMATCH PT={pt.shape} TT={tt.shape}")
        return False, float("inf")
    diff = np.abs(pt - tt)
    mx = float(diff.max())
    mn = float(diff.mean())
    ok = mx < atol
    tag = "✓" if ok else "✗"
    print(f"  {tag} {label}: max_diff={mx:.2e}  mean_diff={mn:.2e}  shape={pt.shape}")
    if not ok:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"      worst at {idx}: PT={pt[idx]:.6f}  TT={tt[idx]:.6f}")
        print(f"      PT[:8]={pt.flatten()[:8]}")
        print(f"      TT[:8]={tt.flatten()[:8]}")
    return ok, mx


def sync_layer_weights(pt_layer, tt_layer):
    """Copy PT layer weights → TT layer weights, with norm swap."""
    # Self-attention (MHA)
    tt_layer.self_attn.in_proj_weight.data = (
        pt_layer.self_attn.in_proj_weight.detach().numpy().T.copy()
    )
    tt_layer.self_attn.in_proj_bias.data = (
        pt_layer.self_attn.in_proj_bias.detach().numpy().copy()
    )
    tt_layer.self_attn.out_proj.param.data = (
        pt_layer.self_attn.out_proj.weight.detach().numpy().T.copy()
    )
    tt_layer.self_attn.out_proj.bias.data = (
        pt_layer.self_attn.out_proj.bias.detach().numpy().copy()
    )

    # Cross-attention (MSDeformAttn — no transpose)
    for pn in ["sampling_offsets", "attention_weights", "value_proj", "output_proj"]:
        pt_p = getattr(pt_layer.cross_attn, pn)
        tt_p = getattr(tt_layer.cross_attn, pn)
        tt_p.weight = pt_p.weight.detach().numpy().copy()
        tt_p.bias = pt_p.bias.detach().numpy().copy()

    # FFN (SimNN.Linear — no transpose needed)
    tt_layer.linear1.param.data = pt_layer.linear1.weight.detach().numpy().copy()
    tt_layer.linear1.bias.data = pt_layer.linear1.bias.detach().numpy().copy()
    tt_layer.linear2.param.data = pt_layer.linear2.weight.detach().numpy().copy()
    tt_layer.linear2.bias.data = pt_layer.linear2.bias.detach().numpy().copy()

    # LayerNorm (SWAPPED: PT norm2 → TT norm1, PT norm1 → TT norm2)
    tt_layer.norm1.params[0][1].data = pt_layer.norm2.weight.detach().numpy().copy()
    tt_layer.norm1.params[1][1].data = pt_layer.norm2.bias.detach().numpy().copy()
    tt_layer.norm2.params[0][1].data = pt_layer.norm1.weight.detach().numpy().copy()
    tt_layer.norm2.params[1][1].data = pt_layer.norm1.bias.detach().numpy().copy()
    tt_layer.norm3.params[0][1].data = pt_layer.norm3.weight.detach().numpy().copy()
    tt_layer.norm3.params[1][1].data = pt_layer.norm3.bias.detach().numpy().copy()


# ============================================================================
# DEBUG 1: Reference Points Expansion
# ============================================================================


def debug_ref_points_expansion():
    """Verify reference_points expansion matches PyTorch exactly."""
    print("\n" + "=" * 80)
    print("DEBUG 1: Reference Points Expansion")
    print("=" * 80)

    B, Q, L = 1, 10, 2
    np.random.seed(42)
    ref_np = np.random.rand(B, Q, 2).astype(np.float32)
    vr_np = np.ones((B, L, 2), dtype=np.float32)

    # PyTorch expansion (from DecoderPT.forward)
    ref_torch = torch.from_numpy(ref_np)
    vr_torch = torch.from_numpy(vr_np)
    pt_expanded = ref_torch[:, :, None] * vr_torch[:, None]
    print(f"  PT expanded shape: {pt_expanded.shape}")

    # TTSim expansion (from DecoderTT.__call__)
    ref_unsqueezed = ref_np[:, :, None, :]
    vr_unsqueezed = vr_np[:, None, :, :]
    tt_expanded = ref_unsqueezed * vr_unsqueezed
    print(f"  TT expanded shape: {tt_expanded.shape}")

    ok, _ = compare("ref_points expansion", pt_expanded, tt_expanded, atol=1e-7)

    # Also test with non-trivial valid_ratios
    vr_np2 = np.array([[[0.9, 0.8], [0.7, 0.6]]], dtype=np.float32)
    vr_torch2 = torch.from_numpy(vr_np2)
    pt_exp2 = ref_torch[:, :, None] * vr_torch2[:, None]
    tt_exp2 = ref_np[:, :, None, :] * vr_np2[:, None, :, :]
    ok2, _ = compare(
        "ref_points expansion (non-trivial ratios)", pt_exp2, tt_exp2, atol=1e-7
    )

    return ok and ok2


# ============================================================================
# DEBUG 2: Single Layer Block-by-Block (through Decoder wrapper)
# ============================================================================


def debug_layer_blocks():
    """Run each sub-block of a decoder layer and compare PT vs TT."""
    print("\n" + "=" * 80)
    print("DEBUG 2: Single Decoder Layer — Block-by-Block")
    print("=" * 80)

    B, Q, S, D = 1, 10, 50, 64
    n_levels, n_heads, n_points, d_ffn = 2, 4, 4, 128
    ss_list = [[5, 5], [5, 5]]
    lsi_list = [0, 25]

    torch.manual_seed(42)
    np.random.seed(42)

    tgt_np = np.random.randn(B, Q, D).astype(np.float32) * 0.1
    qpos_np = np.random.randn(B, Q, D).astype(np.float32) * 0.1
    ref_np = np.random.rand(B, Q, n_levels, 2).astype(np.float32)  # already expanded
    src_np = np.random.randn(B, S, D).astype(np.float32) * 0.1

    # Create layers
    pt_layer = DecoderLayerPT(
        d_model=D,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    pt_layer.eval()

    tt_layer = DecoderLayerTT(
        name="dbg_layer",
        d_model=D,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    sync_layer_weights(pt_layer, tt_layer)

    # Convert inputs
    tgt_t = torch.from_numpy(tgt_np)
    qpos_t = torch.from_numpy(qpos_np)
    ref_t = torch.from_numpy(ref_np)
    src_t = torch.from_numpy(src_np)
    ss_t = torch.tensor(ss_list, dtype=torch.long)
    lsi_t = torch.tensor(lsi_list, dtype=torch.long)

    tgt_s = to_sim(tgt_t, "tgt", tt_layer)
    qpos_s = to_sim(qpos_t, "qpos", tt_layer)
    ref_s = to_sim(ref_t, "ref", tt_layer)
    src_s = to_sim(src_t, "src", tt_layer)
    ss_s = to_sim(ss_t.float(), "ss", tt_layer)
    lsi_s = to_sim(lsi_t.float(), "lsi", tt_layer)

    all_ok = True

    # ── Block A: pos embedding ────────────────────────────────────────────
    print("\n  --- Block A: q = k = tgt + query_pos ---")
    with torch.no_grad():
        q_pt = k_pt = pt_layer.with_pos_embed(tgt_t, qpos_t)
    q_tt = k_tt = tt_layer.with_pos_embed(tgt_s, qpos_s)
    ok, _ = compare("q (=k)", q_pt, q_tt, atol=1e-6)
    all_ok &= ok

    # ── Block B: self-attention ───────────────────────────────────────────
    print("\n  --- Block B: self-attention ---")
    with torch.no_grad():
        # PyTorch: transpose for nn.MultiheadAttention API
        tgt2_pt = pt_layer.self_attn(
            q_pt.transpose(0, 1), k_pt.transpose(0, 1), tgt_t.transpose(0, 1)
        )[0].transpose(0, 1)
    tgt2_tt = tt_layer.self_attn(q_tt, k_tt, tgt_s, None, None, False)
    ok, _ = compare("self_attn output", tgt2_pt, tgt2_tt, atol=1e-4)
    all_ok &= ok

    # ── Block C: residual + norm (self-attn) ──────────────────────────────
    print("\n  --- Block C: residual + norm (self-attn) ---")
    with torch.no_grad():
        # PT: dropout2 → norm2
        tgt_pt_c = tgt_t + pt_layer.dropout2(tgt2_pt)
        tgt_pt_c = pt_layer.norm2(tgt_pt_c)
    # TT: dropout1 → norm1
    dropout1_out = tt_layer.dropout1(tgt2_tt)
    tgt_tt_c = tgt_s + dropout1_out
    tgt_tt_c = tt_layer.norm1(tgt_tt_c)
    ok, _ = compare("after self-attn norm", tgt_pt_c, tgt_tt_c, atol=1e-4)
    all_ok &= ok

    # ── Block D: cross-attention ──────────────────────────────────────────
    print("\n  --- Block D: cross-attention ---")
    with torch.no_grad():
        q_cross_pt = pt_layer.with_pos_embed(tgt_pt_c, qpos_t)
        tgt2_pt_d = pt_layer.cross_attn(q_cross_pt, ref_t, src_t, ss_t, lsi_t, None)
    q_cross_tt = tt_layer.with_pos_embed(tgt_tt_c, qpos_s)
    tgt2_tt_d = tt_layer.cross_attn(q_cross_tt, ref_s, src_s, ss_s, lsi_s, None)
    ok, _ = compare("cross_attn output", tgt2_pt_d, tgt2_tt_d, atol=1e-4)
    all_ok &= ok

    # ── Block E: residual + norm (cross-attn) ────────────────────────────
    print("\n  --- Block E: residual + norm (cross-attn) ---")
    with torch.no_grad():
        # PT: dropout1 → norm1
        tgt_pt_e = tgt_pt_c + pt_layer.dropout1(tgt2_pt_d)
        tgt_pt_e = pt_layer.norm1(tgt_pt_e)
    # TT: dropout2 → norm2
    dropout2_out = tt_layer.dropout2(tgt2_tt_d)
    tgt_tt_e = tgt_tt_c + dropout2_out
    tgt_tt_e = tt_layer.norm2(tgt_tt_e)
    ok, _ = compare("after cross-attn norm", tgt_pt_e, tgt_tt_e, atol=1e-4)
    all_ok &= ok

    # ── Block F: FFN ──────────────────────────────────────────────────────
    print("\n  --- Block F: FFN ---")
    with torch.no_grad():
        tgt_pt_f = pt_layer.forward_ffn(tgt_pt_e)
    tgt_tt_f = tt_layer.forward_ffn(tgt_tt_e)
    ok, _ = compare("after FFN", tgt_pt_f, tgt_tt_f, atol=1e-4)
    all_ok &= ok

    # ── Full layer output comparison ──────────────────────────────────────
    print("\n  --- Full layer output (end-to-end) ---")
    with torch.no_grad():
        full_pt = pt_layer(tgt_t, qpos_t, ref_t, src_t, ss_t, lsi_t, None)
    full_tt = tt_layer(
        to_sim(tgt_t, "tgt2", tt_layer), qpos_s, ref_s, src_s, ss_s, lsi_s, None
    )
    ok, _ = compare("full layer output", full_pt, full_tt, atol=1e-4)
    all_ok &= ok

    return all_ok


# ============================================================================
# DEBUG 3: Two-Layer Decoder — Layer-by-Layer
# ============================================================================


def debug_decoder_layerwise():
    """Run the 2-layer decoder layer-by-layer, comparing after each layer."""
    print("\n" + "=" * 80)
    print("DEBUG 3: Two-Layer Decoder — Layer-by-Layer Comparison")
    print("=" * 80)

    B, Q, S, D = 1, 10, 50, 64
    n_levels, n_heads, n_points, d_ffn = 2, 4, 4, 128
    n_layers = 2
    ss_list = [[5, 5], [5, 5]]
    lsi_list = [0, 25]

    torch.manual_seed(42)
    np.random.seed(42)

    tgt_np = np.random.randn(B, Q, D).astype(np.float32) * 0.1
    qpos_np = np.random.randn(B, Q, D).astype(np.float32) * 0.1
    ref_np = np.random.rand(B, Q, 2).astype(np.float32)  # [B, Q, 2] — decoder expands
    src_np = np.random.randn(B, S, D).astype(np.float32) * 0.1
    vr_np = np.ones((B, n_levels, 2), dtype=np.float32)

    # ── PT Decoder ───────────────────────────────────────────────────────
    pt_template = DecoderLayerPT(
        d_model=D,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    pt_decoder = DecoderPT(pt_template, n_layers, return_intermediate=True)
    pt_decoder.eval()

    tgt_t = torch.from_numpy(tgt_np)
    qpos_t = torch.from_numpy(qpos_np)
    ref_t = torch.from_numpy(ref_np)
    src_t = torch.from_numpy(src_np)
    ss_t = torch.tensor(ss_list, dtype=torch.long)
    lsi_t = torch.tensor(lsi_list, dtype=torch.long)
    vr_t = torch.from_numpy(vr_np)

    with torch.no_grad():
        out_pt, ref_out_pt = pt_decoder(tgt_t, ref_t, src_t, ss_t, lsi_t, vr_t, qpos_t)

    print(f"\n  PT output shape (stacked intermediates): {out_pt.shape}")
    for i in range(n_layers):
        print(f"    Layer {i} output[:8]: {out_pt[i].flatten()[:8].numpy()}")

    # ── TT Decoder ───────────────────────────────────────────────────────
    tt_template = DecoderLayerTT(
        name="dbg_dec_layer",
        d_model=D,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    tt_decoder = DecoderTT(
        name="dbg_decoder",
        decoder_layer=tt_template,
        num_layers=n_layers,
        return_intermediate=True,
    )

    # Sync weights
    for i, (pt_l, tt_l) in enumerate(zip(pt_decoder.layers, tt_decoder.layers)):
        sync_layer_weights(pt_l, tt_l)
        print(f"  Synced layer {i}")

    tgt_s = to_sim(tgt_t, "tgt", tt_decoder)
    qpos_s = to_sim(qpos_t, "qpos", tt_decoder)
    ref_s = to_sim(ref_t, "ref", tt_decoder)
    src_s = to_sim(src_t, "src", tt_decoder)
    ss_s = to_sim(ss_t.float(), "ss", tt_decoder)
    lsi_s = to_sim(lsi_t.float(), "lsi", tt_decoder)
    vr_s = to_sim(vr_t, "vr", tt_decoder)

    out_tt, ref_out_tt = tt_decoder(tgt_s, ref_s, src_s, ss_s, lsi_s, vr_s, qpos_s)

    print(f"\n  TT output shape (stacked intermediates): {out_tt.shape}")
    for i in range(n_layers):
        print(f"    Layer {i} output[:8]: {out_tt.data[i].flatten()[:8]}")

    # ── Compare each layer's intermediate output ─────────────────────────
    all_ok = True
    for i in range(n_layers):
        ok, mx = compare(
            f"Layer {i} intermediate output", out_pt[i], out_tt.data[i], atol=1e-3
        )
        all_ok &= ok

    return all_ok


# ============================================================================
# DEBUG 4: Manual Decoder Loop — feed layer outputs step-by-step
# ============================================================================


def debug_manual_decoder_loop():
    """
    Manually replicate the decoder loop for BOTH PyTorch and TTSim,
    feeding each layer independently with the SAME expanded ref_points.
    This isolates whether the problem is in:
      (a) reference_points expansion
      (b) layer computation
      (c) output flowing between layers
    """
    print("\n" + "=" * 80)
    print("DEBUG 4: Manual Decoder Loop — Step-by-Step")
    print("=" * 80)

    B, Q, S, D = 1, 10, 50, 64
    n_levels, n_heads, n_points, d_ffn = 2, 4, 4, 128
    n_layers = 2
    ss_list = [[5, 5], [5, 5]]
    lsi_list = [0, 25]

    torch.manual_seed(42)
    np.random.seed(42)

    tgt_np = np.random.randn(B, Q, D).astype(np.float32) * 0.1
    qpos_np = np.random.randn(B, Q, D).astype(np.float32) * 0.1
    ref_np = np.random.rand(B, Q, 2).astype(np.float32)
    src_np = np.random.randn(B, S, D).astype(np.float32) * 0.1
    vr_np = np.ones((B, n_levels, 2), dtype=np.float32)

    # ── Create and sync ──────────────────────────────────────────────────
    pt_template = DecoderLayerPT(
        d_model=D,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    pt_decoder = DecoderPT(pt_template, n_layers, return_intermediate=False)
    pt_decoder.eval()

    tt_template = DecoderLayerTT(
        name="man_dec_layer",
        d_model=D,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    tt_decoder = DecoderTT(
        name="man_decoder",
        decoder_layer=tt_template,
        num_layers=n_layers,
        return_intermediate=False,
    )

    for i, (pt_l, tt_l) in enumerate(zip(pt_decoder.layers, tt_decoder.layers)):
        sync_layer_weights(pt_l, tt_l)
    print("  Weights synced for all layers")

    # ── Expand reference_points manually ─────────────────────────────────
    ref_t = torch.from_numpy(ref_np)
    vr_t = torch.from_numpy(vr_np)
    ref_expanded_pt = ref_t[:, :, None] * vr_t[:, None]  # [B, Q, L, 2]
    ref_expanded_np = ref_np[:, :, None, :] * vr_np[:, None, :, :]

    ok, _ = compare(
        "Manual ref_points expansion", ref_expanded_pt, ref_expanded_np, atol=1e-7
    )
    if not ok:
        print("  !! Reference points expansion already diverges!")
        return False

    # ── Run each layer manually ──────────────────────────────────────────
    tgt_t = torch.from_numpy(tgt_np)
    qpos_t = torch.from_numpy(qpos_np)
    src_t = torch.from_numpy(src_np)
    ss_t = torch.tensor(ss_list, dtype=torch.long)
    lsi_t = torch.tensor(lsi_list, dtype=torch.long)

    output_pt = tgt_t.clone()
    output_np = tgt_np.copy()

    all_ok = True

    for lid in range(n_layers):
        print(f"\n  ── Layer {lid} ──")

        # Compare inputs to this layer
        ok_in, _ = compare(
            f"  layer {lid} input (tgt)", output_pt, output_np, atol=1e-4
        )
        all_ok &= ok_in

        # PT: run layer
        pt_l = pt_decoder.layers[lid]
        with torch.no_grad():
            output_pt = pt_l(
                output_pt, qpos_t, ref_expanded_pt, src_t, ss_t, lsi_t, None
            )

        # TT: run layer
        tt_l = tt_decoder.layers[lid]
        # Create fresh SimTensors for this layer's input
        tgt_sim = SimTensor(
            {
                "name": f"tgt_l{lid}",
                "shape": list(output_np.shape),
                "data": output_np.copy(),
                "dtype": np.float32,
            }
        )
        tgt_sim.link_module = tt_l
        tt_l._tensors[tgt_sim.name] = tgt_sim

        qpos_sim = SimTensor(
            {
                "name": f"qpos_l{lid}",
                "shape": list(qpos_np.shape),
                "data": qpos_np.copy(),
                "dtype": np.float32,
            }
        )
        qpos_sim.link_module = tt_l
        tt_l._tensors[qpos_sim.name] = qpos_sim

        ref_sim = SimTensor(
            {
                "name": f"ref_l{lid}",
                "shape": list(ref_expanded_np.shape),
                "data": ref_expanded_np.copy(),
                "dtype": np.float32,
            }
        )
        ref_sim.link_module = tt_l
        tt_l._tensors[ref_sim.name] = ref_sim

        src_sim = SimTensor(
            {
                "name": f"src_l{lid}",
                "shape": list(src_np.shape),
                "data": src_np.copy(),
                "dtype": np.float32,
            }
        )
        src_sim.link_module = tt_l
        tt_l._tensors[src_sim.name] = src_sim

        ss_sim = SimTensor(
            {
                "name": f"ss_l{lid}",
                "shape": list(np.array(ss_list).shape),
                "data": np.array(ss_list, dtype=np.float32),
                "dtype": np.float32,
            }
        )
        ss_sim.link_module = tt_l
        tt_l._tensors[ss_sim.name] = ss_sim

        lsi_sim = SimTensor(
            {
                "name": f"lsi_l{lid}",
                "shape": list(np.array(lsi_list).shape),
                "data": np.array(lsi_list, dtype=np.float32),
                "dtype": np.float32,
            }
        )
        lsi_sim.link_module = tt_l
        tt_l._tensors[lsi_sim.name] = lsi_sim

        output_tt = tt_l(tgt_sim, qpos_sim, ref_sim, src_sim, ss_sim, lsi_sim, None)

        # Compare layer output
        ok_out, mx = compare(f"  layer {lid} output", output_pt, output_tt, atol=1e-3)
        all_ok &= ok_out

        # Update TT output for next layer
        output_np = output_tt.data.copy() if output_tt.data is not None else output_np

    return all_ok


# ============================================================================
# DEBUG 5: Decoder wrapper vs manual loop
# ============================================================================


def debug_wrapper_vs_manual():
    """
    Compare: (a) running through the Decoder wrapper vs (b) manually calling layers.
    If the wrapper gives different results from manual, the issue is in the wrapper.
    """
    print("\n" + "=" * 80)
    print("DEBUG 5: Decoder Wrapper vs Manual Layer Calls")
    print("=" * 80)

    B, Q, S, D = 1, 10, 50, 64
    n_levels, n_heads, n_points, d_ffn = 2, 4, 4, 128
    n_layers = 2
    ss_list = [[5, 5], [5, 5]]
    lsi_list = [0, 25]

    torch.manual_seed(42)
    np.random.seed(42)

    tgt_np = np.random.randn(B, Q, D).astype(np.float32) * 0.1
    qpos_np = np.random.randn(B, Q, D).astype(np.float32) * 0.1
    ref_np = np.random.rand(B, Q, 2).astype(np.float32)
    src_np = np.random.randn(B, S, D).astype(np.float32) * 0.1
    vr_np = np.ones((B, n_levels, 2), dtype=np.float32)

    # Create TT decoder
    tt_template = DecoderLayerTT(
        name="wrap_layer",
        d_model=D,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    tt_decoder = DecoderTT(
        name="wrap_decoder",
        decoder_layer=tt_template,
        num_layers=n_layers,
        return_intermediate=False,
    )

    # Create matching PT decoder to get weights
    pt_template = DecoderLayerPT(
        d_model=D,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    pt_decoder = DecoderPT(pt_template, n_layers, return_intermediate=False)
    pt_decoder.eval()

    for i, (pt_l, tt_l) in enumerate(zip(pt_decoder.layers, tt_decoder.layers)):
        sync_layer_weights(pt_l, tt_l)

    # ── Run through wrapper ──────────────────────────────────────────────
    tgt_s = to_sim(torch.from_numpy(tgt_np), "tgt", tt_decoder)
    qpos_s = to_sim(torch.from_numpy(qpos_np), "qpos", tt_decoder)
    ref_s = to_sim(torch.from_numpy(ref_np), "ref", tt_decoder)
    src_s = to_sim(torch.from_numpy(src_np), "src", tt_decoder)
    ss_s = to_sim(torch.tensor(ss_list, dtype=torch.float32), "ss", tt_decoder)
    lsi_s = to_sim(torch.tensor(lsi_list, dtype=torch.float32), "lsi", tt_decoder)
    vr_s = to_sim(torch.from_numpy(vr_np), "vr", tt_decoder)

    wrapper_out, _ = tt_decoder(tgt_s, ref_s, src_s, ss_s, lsi_s, vr_s, qpos_s)

    # ── Run manually through same layers ─────────────────────────────────
    ref_expanded = ref_np[:, :, None, :] * vr_np[:, None, :, :]
    output_np = tgt_np.copy()

    for lid in range(n_layers):
        tt_l = tt_decoder.layers[lid]

        tgt_sim = SimTensor(
            {
                "name": f"man_tgt_l{lid}",
                "shape": list(output_np.shape),
                "data": output_np.copy(),
                "dtype": np.float32,
            }
        )
        tgt_sim.link_module = tt_l
        tt_l._tensors[tgt_sim.name] = tgt_sim

        qpos_sim = SimTensor(
            {
                "name": f"man_qpos_l{lid}",
                "shape": list(qpos_np.shape),
                "data": qpos_np.copy(),
                "dtype": np.float32,
            }
        )
        qpos_sim.link_module = tt_l
        tt_l._tensors[qpos_sim.name] = qpos_sim

        ref_sim = SimTensor(
            {
                "name": f"man_ref_l{lid}",
                "shape": list(ref_expanded.shape),
                "data": ref_expanded.copy(),
                "dtype": np.float32,
            }
        )
        ref_sim.link_module = tt_l
        tt_l._tensors[ref_sim.name] = ref_sim

        src_sim = SimTensor(
            {
                "name": f"man_src_l{lid}",
                "shape": list(src_np.shape),
                "data": src_np.copy(),
                "dtype": np.float32,
            }
        )
        src_sim.link_module = tt_l
        tt_l._tensors[src_sim.name] = src_sim

        ss_sim = SimTensor(
            {
                "name": f"man_ss_l{lid}",
                "shape": list(np.array(ss_list).shape),
                "data": np.array(ss_list, dtype=np.float32),
                "dtype": np.float32,
            }
        )
        ss_sim.link_module = tt_l
        tt_l._tensors[ss_sim.name] = ss_sim

        lsi_sim = SimTensor(
            {
                "name": f"man_lsi_l{lid}",
                "shape": list(np.array(lsi_list).shape),
                "data": np.array(lsi_list, dtype=np.float32),
                "dtype": np.float32,
            }
        )
        lsi_sim.link_module = tt_l
        tt_l._tensors[lsi_sim.name] = lsi_sim

        output_tt = tt_l(tgt_sim, qpos_sim, ref_sim, src_sim, ss_sim, lsi_sim, None)
        output_np = output_tt.data.copy()

    # Compare wrapper output vs manual output
    ok, _ = compare(
        "Wrapper vs Manual (same TT layers)", wrapper_out, output_np, atol=1e-6
    )

    # Also compare vs PyTorch
    tgt_t = torch.from_numpy(tgt_np)
    qpos_t = torch.from_numpy(qpos_np)
    ref_t = torch.from_numpy(ref_np)
    src_t = torch.from_numpy(src_np)
    ss_t = torch.tensor(ss_list, dtype=torch.long)
    lsi_t = torch.tensor(lsi_list, dtype=torch.long)
    vr_t = torch.from_numpy(vr_np)

    with torch.no_grad():
        out_pt, _ = pt_decoder(tgt_t, ref_t, src_t, ss_t, lsi_t, vr_t, qpos_t)

    ok_pt_wrap, _ = compare("PT vs TT Wrapper", out_pt, wrapper_out, atol=1e-3)
    ok_pt_man, _ = compare("PT vs TT Manual", out_pt, output_np, atol=1e-3)

    return ok and ok_pt_wrap and ok_pt_man


# ============================================================================
# DEBUG 6: Weight verification — are decoder wrapper layers getting synced?
# ============================================================================


def debug_weight_sync():
    """Verify that synced weights match between PT and TT decoder layers."""
    print("\n" + "=" * 80)
    print("DEBUG 6: Weight Sync Verification")
    print("=" * 80)

    D, d_ffn = 64, 128
    n_levels, n_heads, n_points = 2, 4, 4
    n_layers = 2

    torch.manual_seed(42)

    pt_template = DecoderLayerPT(
        d_model=D,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    pt_decoder = DecoderPT(pt_template, n_layers, return_intermediate=False)

    tt_template = DecoderLayerTT(
        name="ws_layer",
        d_model=D,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    tt_decoder = DecoderTT(
        name="ws_decoder",
        decoder_layer=tt_template,
        num_layers=n_layers,
        return_intermediate=False,
    )

    for i, (pt_l, tt_l) in enumerate(zip(pt_decoder.layers, tt_decoder.layers)):
        sync_layer_weights(pt_l, tt_l)

    all_ok = True
    for lid in range(n_layers):
        pt_l = pt_decoder.layers[lid]
        tt_l = tt_decoder.layers[lid]
        print(f"\n  Layer {lid}:")

        # Self-attn in_proj_weight
        pt_w = pt_l.self_attn.in_proj_weight.detach().numpy()  # [3E, E]
        tt_w = tt_l.self_attn.in_proj_weight.data  # [E, 3E] (transposed)
        ok, _ = compare(f"  L{lid} self_attn.in_proj_weight", pt_w.T, tt_w, atol=1e-7)
        all_ok &= ok

        # Self-attn out_proj
        pt_w = pt_l.self_attn.out_proj.weight.detach().numpy()  # [E, E]
        tt_w = tt_l.self_attn.out_proj.param.data  # [E, E] (transposed)
        ok, _ = compare(f"  L{lid} self_attn.out_proj.weight", pt_w.T, tt_w, atol=1e-7)
        all_ok &= ok

        # Cross-attn value_proj
        pt_w = pt_l.cross_attn.value_proj.weight.detach().numpy()
        tt_w = tt_l.cross_attn.value_proj.weight
        ok, _ = compare(f"  L{lid} cross_attn.value_proj.weight", pt_w, tt_w, atol=1e-7)
        all_ok &= ok

        # Cross-attn output_proj
        pt_w = pt_l.cross_attn.output_proj.weight.detach().numpy()
        tt_w = tt_l.cross_attn.output_proj.weight
        ok, _ = compare(
            f"  L{lid} cross_attn.output_proj.weight", pt_w, tt_w, atol=1e-7
        )
        all_ok &= ok

        # FFN linear1
        pt_w = pt_l.linear1.weight.detach().numpy()
        tt_w = tt_l.linear1.param.data
        ok, _ = compare(f"  L{lid} linear1.weight", pt_w.T, tt_w, atol=1e-7)
        all_ok &= ok

        # FFN linear2
        pt_w = pt_l.linear2.weight.detach().numpy()
        tt_w = tt_l.linear2.param.data
        ok, _ = compare(f"  L{lid} linear2.weight", pt_w.T, tt_w, atol=1e-7)
        all_ok &= ok

        # LayerNorm (swapped)
        pt_norm2_w = pt_l.norm2.weight.detach().numpy()  # self-attn norm
        tt_norm1_w = tt_l.norm1.params[0][1].data
        ok, _ = compare(
            f"  L{lid} norm (self-attn: PT.norm2→TT.norm1)",
            pt_norm2_w,
            tt_norm1_w,
            atol=1e-7,
        )
        all_ok &= ok

        pt_norm1_w = pt_l.norm1.weight.detach().numpy()  # cross-attn norm
        tt_norm2_w = tt_l.norm2.params[0][1].data
        ok, _ = compare(
            f"  L{lid} norm (cross-attn: PT.norm1→TT.norm2)",
            pt_norm1_w,
            tt_norm2_w,
            atol=1e-7,
        )
        all_ok &= ok

        pt_norm3_w = pt_l.norm3.weight.detach().numpy()  # FFN norm
        tt_norm3_w = tt_l.norm3.params[0][1].data
        ok, _ = compare(
            f"  L{lid} norm (FFN: PT.norm3→TT.norm3)", pt_norm3_w, tt_norm3_w, atol=1e-7
        )
        all_ok &= ok

        # Check if PT layers 0 and 1 have same weights (they should — cloned)
        if lid == 1:
            pt_l0 = pt_decoder.layers[0]
            same = torch.equal(pt_l0.linear1.weight, pt_l.linear1.weight)
            print(f"  PT layer 0 == layer 1 linear1 weights: {same}")

    return all_ok


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DECODER BLOCK-BY-BLOCK DEBUGGER")
    print("=" * 80)

    results = {}

    try:
        results["ref_expansion"] = debug_ref_points_expansion()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback

        traceback.print_exc()
        results["ref_expansion"] = False

    try:
        results["layer_blocks"] = debug_layer_blocks()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback

        traceback.print_exc()
        results["layer_blocks"] = False

    try:
        results["weight_sync"] = debug_weight_sync()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback

        traceback.print_exc()
        results["weight_sync"] = False

    try:
        results["decoder_layerwise"] = debug_decoder_layerwise()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback

        traceback.print_exc()
        results["decoder_layerwise"] = False

    try:
        results["manual_loop"] = debug_manual_decoder_loop()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback

        traceback.print_exc()
        results["manual_loop"] = False

    try:
        results["wrapper_vs_manual"] = debug_wrapper_vs_manual()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback

        traceback.print_exc()
        results["wrapper_vs_manual"] = False

    # Summary
    print("\n" + "=" * 80)
    print("DEBUGGER SUMMARY")
    print("=" * 80)
    for name, ok in results.items():
        tag = "✓" if ok else "✗"
        print(f"  {tag}  {name}")

    all_ok = all(results.values())
    print(f"\n  {'ALL PASS' if all_ok else 'ISSUES FOUND'}")
    sys.exit(0 if all_ok else 1)
