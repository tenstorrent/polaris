#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Numerical Computation Validation for DeformableTransformer (full encoder-decoder).

Compares PyTorch vs TTSim outputs with synced weights on small random inputs.
Tests the complete one-stage pipeline:
  1. Multi-scale feature flatten + level_embed addition
  2. Encoder (stacked MSDeformAttn encoder layers)
  3. Query embed split into tgt + query_pos
  4. Reference points via Linear + sigmoid
  5. Decoder (stacked MHA + MSDeformAttn decoder layers)

Weight sync covers:
  - level_embed (learnable parameter)
  - reference_points linear (SimNN.Linear, transposed)
  - Encoder layers (MSDeformAttn + FFN + LayerNorm)
  - Decoder layers (MHA + MSDeformAttn + FFN + LayerNorm with norm swap)
"""

import os
import sys
import torch
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
    DeformableTransformer as TransformerPyTorch,
)
from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
    DeformableTransformer as TransformerTTSim,
)
from ttsim.ops.tensor import SimTensor

# ============================================================================
# Utility Functions
# ============================================================================


def torch_to_simtensor(torch_tensor, name="tensor", module=None):
    """Convert PyTorch tensor to SimTensor with data attached and link_module set."""
    data = torch_tensor.detach().cpu().numpy().copy()
    tensor = SimTensor(
        {
            "name": name,
            "shape": list(torch_tensor.shape),
            "data": data,
            "dtype": data.dtype,
        }
    )
    if module is not None:
        tensor.link_module = module
        module._tensors[name] = tensor
    return tensor


def compare_numerical(pytorch_out, ttsim_out, name="Output", rtol=1e-2, atol=1e-3):
    """Compare PyTorch and TTSim outputs numerically."""
    if isinstance(pytorch_out, torch.Tensor):
        pt_np = pytorch_out.detach().cpu().numpy()
    else:
        pt_np = np.asarray(pytorch_out)

    if isinstance(ttsim_out, SimTensor):
        if ttsim_out.data is None:
            print(f"\n  ✗ FAIL - {name}: TTSim output has no data!")
            return False
        tt_np = ttsim_out.data
    else:
        tt_np = np.asarray(ttsim_out)

    if pt_np.shape != tt_np.shape:
        print(
            f"\n  ✗ FAIL - {name}: Shape mismatch PyTorch={pt_np.shape} TTSim={tt_np.shape}"
        )
        return False

    diff = np.abs(pt_np - tt_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    is_close = np.allclose(pt_np, tt_np, rtol=rtol, atol=atol, equal_nan=True)

    if is_close:
        print(f"\n  ✓ PASS - {name}")
        print(
            f"    Shape: {pt_np.shape}  Max diff: {max_diff:.2e}  Mean diff: {mean_diff:.2e}"
        )
        print(f"    PyTorch (first 8): {pt_np.flatten()[:8]}")
        print(f"    TTSim   (first 8): {tt_np.flatten()[:8]}")
    else:
        print(f"\n  ✗ FAIL - {name}")
        print(
            f"    Shape: {pt_np.shape}  Max diff: {max_diff:.2e}  Mean diff: {mean_diff:.2e}"
        )
        print(f"    PyTorch (first 8): {pt_np.flatten()[:8]}")
        print(f"    TTSim   (first 8): {tt_np.flatten()[:8]}")
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(
            f"    Max diff at {max_idx}: PyTorch={pt_np[max_idx]:.6f} TTSim={tt_np[max_idx]:.6f}"
        )
        print(
            f"    PyTorch stats: min={pt_np.min():.4f} max={pt_np.max():.4f} mean={pt_np.mean():.4f}"
        )
        print(
            f"    TTSim   stats: min={tt_np.min():.4f} max={tt_np.max():.4f} mean={tt_np.mean():.4f}"
        )

    return is_close


# ============================================================================
# Weight Sync
# ============================================================================


def sync_encoder_layer_weights(pt_layer, tt_layer):
    """Copy weights from PyTorch encoder layer to TTSim encoder layer."""
    # MSDeformAttn (custom Linear — no transpose)
    for proj_name in [
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
    ]:
        pt_proj = getattr(pt_layer.self_attn, proj_name)
        tt_proj = getattr(tt_layer.self_attn, proj_name)
        tt_proj.param.data = pt_proj.weight.detach().numpy().copy()
        tt_proj.bias.data = pt_proj.bias.detach().numpy().copy()

    # FFN Linear layers (SimNN.Linear — no transpose needed)
    tt_layer.linear1.param.data = pt_layer.linear1.weight.detach().numpy().copy()
    tt_layer.linear1.bias.data = pt_layer.linear1.bias.detach().numpy().copy()
    tt_layer.linear2.param.data = pt_layer.linear2.weight.detach().numpy().copy()
    tt_layer.linear2.bias.data = pt_layer.linear2.bias.detach().numpy().copy()

    # LayerNorm (no swap for encoder)
    tt_layer.norm1.params[0][1].data = pt_layer.norm1.weight.detach().numpy().copy()
    tt_layer.norm1.params[1][1].data = pt_layer.norm1.bias.detach().numpy().copy()
    tt_layer.norm2.params[0][1].data = pt_layer.norm2.weight.detach().numpy().copy()
    tt_layer.norm2.params[1][1].data = pt_layer.norm2.bias.detach().numpy().copy()


def sync_decoder_layer_weights(pt_layer, tt_layer):
    """Copy weights from PyTorch decoder layer to TTSim decoder layer (with norm swap)."""
    # Self-attention (MultiheadAttention — transposed)
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

    # Cross-attention (MSDeformAttn — custom Linear, no transpose)
    for proj_name in [
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
    ]:
        pt_proj = getattr(pt_layer.cross_attn, proj_name)
        tt_proj = getattr(tt_layer.cross_attn, proj_name)
        tt_proj.param.data = pt_proj.weight.detach().numpy().copy()
        tt_proj.bias.data = pt_proj.bias.detach().numpy().copy()

    # FFN Linear (no transpose needed)
    tt_layer.linear1.param.data = pt_layer.linear1.weight.detach().numpy().copy()
    tt_layer.linear1.bias.data = pt_layer.linear1.bias.detach().numpy().copy()
    tt_layer.linear2.param.data = pt_layer.linear2.weight.detach().numpy().copy()
    tt_layer.linear2.bias.data = pt_layer.linear2.bias.detach().numpy().copy()

    # LayerNorm (SWAPPED naming for decoder)
    tt_layer.norm1.params[0][1].data = (
        pt_layer.norm2.weight.detach().numpy().copy()
    )  # self-attn
    tt_layer.norm1.params[1][1].data = pt_layer.norm2.bias.detach().numpy().copy()
    tt_layer.norm2.params[0][1].data = (
        pt_layer.norm1.weight.detach().numpy().copy()
    )  # cross-attn
    tt_layer.norm2.params[1][1].data = pt_layer.norm1.bias.detach().numpy().copy()
    tt_layer.norm3.params[0][1].data = (
        pt_layer.norm3.weight.detach().numpy().copy()
    )  # FFN
    tt_layer.norm3.params[1][1].data = pt_layer.norm3.bias.detach().numpy().copy()


def sync_full_transformer_weights(pt_transformer, tt_transformer):
    """
    Sync all weights from PyTorch DeformableTransformer to TTSim DeformableTransformer.

    Components:
      - level_embed: Learnable parameter [n_levels, d_model]
      - reference_points: SimNN.Linear (one-stage, transposed)
      - encoder.layers[i]: encoder layer weights
      - decoder.layers[i]: decoder layer weights (with norm swap)
    """
    # === Level embedding ===
    tt_transformer.level_embed.data = pt_transformer.level_embed.detach().numpy().copy()
    print("  Synced: level_embed")

    # === Reference points linear (one-stage only, SimNN.Linear — no transpose needed) ===
    if not pt_transformer.two_stage:
        tt_transformer.reference_points.param.data = (
            pt_transformer.reference_points.weight.detach().numpy().copy()
        )
        tt_transformer.reference_points.bias.data = (
            pt_transformer.reference_points.bias.detach().numpy().copy()
        )
        print("  Synced: reference_points linear")

    # === Encoder layers ===
    for i, (pt_layer, tt_layer) in enumerate(
        zip(pt_transformer.encoder.layers, tt_transformer.encoder.layers)
    ):
        sync_encoder_layer_weights(pt_layer, tt_layer)
    print(f"  Synced: {len(pt_transformer.encoder.layers)} encoder layers")

    # === Decoder layers ===
    for i, (pt_layer, tt_layer) in enumerate(
        zip(pt_transformer.decoder.layers, tt_transformer.decoder.layers)
    ):
        sync_decoder_layer_weights(pt_layer, tt_layer)
    print(
        f"  Synced: {len(pt_transformer.decoder.layers)} decoder layers (with norm swap)"
    )


# ============================================================================
# Tests
# ============================================================================


def test_full_transformer_numerical():
    """Test full DeformableTransformer with synced weights and numerical comparison."""

    print("\n" + "=" * 80)
    print("TEST: DeformableTransformer (Full) — Numerical Validation")
    print("=" * 80)

    all_passed = True

    # ── Small config (to keep computation manageable) ──
    batch_size = 1
    num_queries = 10
    d_model = 64
    n_levels = 2
    n_heads = 4
    n_enc_layers = 1
    n_dec_layers = 1
    d_ffn = 128
    n_points = 4

    # Small spatial dimensions for 2 feature levels
    spatial_dims = [(5, 5), (3, 3)]  # 25 + 9 = 34 total tokens

    print(
        f"\n  Config: B={batch_size}, queries={num_queries}, d_model={d_model}, "
        f"levels={n_levels}, heads={n_heads}"
    )
    print(f"  Encoder layers: {n_enc_layers}, Decoder layers: {n_dec_layers}")
    print(f"  Spatial dims: {spatial_dims}")

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Multi-scale feature inputs ──
    srcs_torch = []
    masks_torch = []
    pos_embeds_torch = []

    for lvl, (h, w) in enumerate(spatial_dims):
        src = torch.randn(batch_size, d_model, h, w) * 0.1
        mask = torch.zeros(batch_size, h, w, dtype=torch.bool)
        pos = torch.randn(batch_size, d_model, h, w) * 0.1
        srcs_torch.append(src)
        masks_torch.append(mask)
        pos_embeds_torch.append(pos)

    query_embed_torch = torch.randn(num_queries, d_model * 2) * 0.1

    print(f"\n  Multi-scale features:")
    for i, src in enumerate(srcs_torch):
        print(f"    Level {i}: {list(src.shape)}")
    print(f"  Query embed: {list(query_embed_torch.shape)}")

    # ── PyTorch forward ──
    print("\n" + "-" * 60)
    print("  PyTorch forward pass")
    print("-" * 60)

    pt_transformer = TransformerPyTorch(
        d_model=d_model,
        nhead=n_heads,
        num_encoder_layers=n_enc_layers,
        num_decoder_layers=n_dec_layers,
        dim_feedforward=d_ffn,
        dropout=0.0,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=n_levels,
        dec_n_points=n_points,
        enc_n_points=n_points,
        two_stage=False,
    )
    pt_transformer.eval()

    with torch.no_grad():
        hs_pt, init_ref_pt, inter_ref_pt, _, _ = pt_transformer(
            srcs_torch, masks_torch, pos_embeds_torch, query_embed_torch
        )

    pt_data = hs_pt.detach().numpy()
    print(f"  hs output shape: {list(hs_pt.shape)}")
    print(f"  Stats: mean={pt_data.mean():.6f} std={pt_data.std():.6f}")
    print(f"  Values (first 8): {pt_data.flatten()[:8]}")
    has_nan_pt = np.isnan(pt_data).any()
    print(f"  NaN: {'YES ✗' if has_nan_pt else 'NO ✓'}")

    # ── TTSim forward with synced weights ──
    print("\n" + "-" * 60)
    print("  TTSim forward pass (with weight sync)")
    print("-" * 60)

    tt_transformer = TransformerTTSim(
        name="transformer_num",
        d_model=d_model,
        nhead=n_heads,
        num_encoder_layers=n_enc_layers,
        num_decoder_layers=n_dec_layers,
        dim_feedforward=d_ffn,
        dropout=0.0,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=n_levels,
        dec_n_points=n_points,
        enc_n_points=n_points,
        two_stage=False,
    )

    sync_full_transformer_weights(pt_transformer, tt_transformer)

    # Convert inputs to SimTensors with link_module set
    srcs_sim = []
    for i, src in enumerate(srcs_torch):
        sim = torch_to_simtensor(src, f"src_{i}", tt_transformer)
        srcs_sim.append(sim)

    masks_sim = []
    for i, mask in enumerate(masks_torch):
        sim = torch_to_simtensor(mask.float(), f"mask_{i}", tt_transformer)
        masks_sim.append(sim)

    pos_sim = []
    for i, pos in enumerate(pos_embeds_torch):
        sim = torch_to_simtensor(pos, f"pos_{i}", tt_transformer)
        pos_sim.append(sim)

    query_sim = torch_to_simtensor(query_embed_torch, "query_embed", tt_transformer)

    hs_tt, init_ref_tt, inter_ref_tt, _, _ = tt_transformer(
        srcs_sim, masks_sim, pos_sim, query_sim
    )

    print(f"  hs output shape: {hs_tt.shape}")
    if hs_tt.data is not None:
        tt_data = hs_tt.data
        print(f"  Stats: mean={tt_data.mean():.6f} std={tt_data.std():.6f}")
        print(f"  Values (first 8): {tt_data.flatten()[:8]}")
        has_nan_tt = np.isnan(tt_data).any()
        print(f"  NaN: {'YES ✗' if has_nan_tt else 'NO ✓'}")
    else:
        print(f"  Data: None (shape inference only)")

    # ── Comparison ──
    print("\n" + "-" * 60)
    print("  Numerical Comparison")
    print("-" * 60)

    if not compare_numerical(
        hs_pt, hs_tt, "Full transformer hs output", rtol=0.1, atol=0.05
    ):
        all_passed = False

    # Shape validation — with return_intermediate_dec=False, PyTorch still wraps in stack
    # PyTorch returns hs.shape = [1, B, Q, d_model] (stacked single layer output)
    # or [B, Q, d_model] depending on return_intermediate
    pt_shape = list(hs_pt.shape)
    tt_shape = hs_tt.shape
    print(f"\n  Shape validation: PyTorch={pt_shape}  TTSim={tt_shape}")
    if pt_shape == tt_shape:
        print(f"    ✓ PASS")
    else:
        print(f"    ✗ FAIL — shapes differ")
        all_passed = False

    return all_passed


if __name__ == "__main__":
    try:
        success = test_full_transformer_numerical()
        print("\n" + "=" * 80)
        print(f"  OVERALL: {'PASSED ✓' if success else 'FAILED ✗'}")
        print("=" * 80)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
