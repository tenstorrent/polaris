#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Numerical Computation Validation for DeformableTransformerDecoderLayer.

Compares PyTorch vs TTSim outputs with synced weights on small random inputs.
Tests self-attention (MHA) + cross-attention (MSDeformAttn) + FFN + LayerNorm.

NOTE: PyTorch and TTSim have SWAPPED norm numbering for decoder layer:
  PyTorch: norm2 = self-attn,  norm1 = cross-attn, norm3 = FFN
  TTSim:   norm1 = self-attn,  norm2 = cross-attn, norm3 = FFN
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
    DeformableTransformerDecoderLayer as DecoderLayerPyTorch,
)
from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
    DeformableTransformerDecoderLayer as DecoderLayerTTSim,
)
from ttsim.ops.tensor import SimTensor

# ============================================================================
# Utility Functions
# ============================================================================


def torch_to_simtensor(torch_tensor, name="tensor", module=None):
    """Convert PyTorch tensor to SimTensor with data attached."""
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
        tensor.set_module(module)
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


def sync_decoder_layer_weights(pt_layer, tt_layer):
    """
    Copy weights from PyTorch DecoderLayer to TTSim DecoderLayer.

    Components:
      - self_attn: PyTorch nn.MultiheadAttention → TTSim SimNN.MultiheadAttention
        IMPORTANT: in_proj_weight transposed, out_proj weight transposed
      - cross_attn: MSDeformAttn (custom Linear, no transpose)
      - linear1/linear2: SimNN.Linear (transposed)
      - norm1/norm2/norm3: F.LayerNorm (via params list)
        SWAP: PyTorch norm2 → TTSim norm1 (self-attn), PyTorch norm1 → TTSim norm2 (cross-attn)
    """
    # === Self-attention (MultiheadAttention) ===
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

    # === Cross-attention (MSDeformAttn — custom Linear, no transpose) ===
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

    # === FFN Linear layers (SimNN.Linear — no transpose needed) ===
    tt_layer.linear1.param.data = pt_layer.linear1.weight.detach().numpy().copy()
    tt_layer.linear1.bias.data = pt_layer.linear1.bias.detach().numpy().copy()
    tt_layer.linear2.param.data = pt_layer.linear2.weight.detach().numpy().copy()
    tt_layer.linear2.bias.data = pt_layer.linear2.bias.detach().numpy().copy()

    # === LayerNorm (SWAPPED naming) ===
    tt_layer.norm1.params[0][1].data = pt_layer.norm2.weight.detach().numpy().copy()
    tt_layer.norm1.params[1][1].data = pt_layer.norm2.bias.detach().numpy().copy()
    tt_layer.norm2.params[0][1].data = pt_layer.norm1.weight.detach().numpy().copy()
    tt_layer.norm2.params[1][1].data = pt_layer.norm1.bias.detach().numpy().copy()
    tt_layer.norm3.params[0][1].data = pt_layer.norm3.weight.detach().numpy().copy()
    tt_layer.norm3.params[1][1].data = pt_layer.norm3.bias.detach().numpy().copy()


# ============================================================================
# Tests
# ============================================================================


def test_decoder_layer_numerical():
    """Test DecoderLayer with synced weights and numerical comparison."""

    print("\n" + "=" * 80)
    print("TEST: DeformableTransformerDecoderLayer — Numerical Validation")
    print("=" * 80)

    all_passed = True

    # ── Small config ──
    batch_size = 1
    num_queries = 10
    src_seq_len = 50
    d_model = 64
    n_levels = 2
    n_heads = 4
    n_points = 4
    d_ffn = 128

    src_spatial_shapes = [[5, 5], [5, 5]]  # 25+25=50
    src_level_start_indices = [0, 25]

    print(
        f"\n  Config: B={batch_size}, queries={num_queries}, memory_len={src_seq_len}, "
        f"d_model={d_model}, levels={n_levels}, heads={n_heads}"
    )

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Shared inputs ──
    tgt_np = np.random.randn(batch_size, num_queries, d_model).astype(np.float32) * 0.1
    query_pos_np = (
        np.random.randn(batch_size, num_queries, d_model).astype(np.float32) * 0.1
    )
    ref_np = np.random.rand(batch_size, num_queries, n_levels, 2).astype(np.float32)
    src_np = np.random.randn(batch_size, src_seq_len, d_model).astype(np.float32) * 0.1

    tgt_torch = torch.from_numpy(tgt_np)
    query_pos_torch = torch.from_numpy(query_pos_np)
    ref_torch = torch.from_numpy(ref_np)
    src_torch = torch.from_numpy(src_np)
    ss_torch = torch.tensor(src_spatial_shapes, dtype=torch.long)
    lsi_torch = torch.tensor(src_level_start_indices, dtype=torch.long)

    # ── PyTorch forward ──
    print("\n" + "-" * 60)
    print("  PyTorch forward pass")
    print("-" * 60)

    pt_layer = DecoderLayerPyTorch(
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    pt_layer.eval()

    with torch.no_grad():
        out_pt = pt_layer(
            tgt_torch, query_pos_torch, ref_torch, src_torch, ss_torch, lsi_torch
        )

    pt_data = out_pt.detach().numpy()
    print(f"  Output shape: {list(out_pt.shape)}")
    print(f"  Stats: mean={pt_data.mean():.6f} std={pt_data.std():.6f}")
    print(f"  Values (first 8): {pt_data.flatten()[:8]}")

    # ── TTSim forward with synced weights ──
    print("\n" + "-" * 60)
    print("  TTSim forward pass (with weight sync)")
    print("-" * 60)

    tt_layer = DecoderLayerTTSim(
        name="dec_layer_num",
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )

    sync_decoder_layer_weights(pt_layer, tt_layer)
    print("  Weights synced: PyTorch → TTSim (with norm swap)")

    # Convert to SimTensors with link_module set
    tgt_sim = torch_to_simtensor(tgt_torch, "tgt", tt_layer)
    qpos_sim = torch_to_simtensor(query_pos_torch, "query_pos", tt_layer)
    ref_sim = torch_to_simtensor(ref_torch, "ref_points", tt_layer)
    src_sim = torch_to_simtensor(src_torch, "src", tt_layer)
    ss_sim = torch_to_simtensor(ss_torch.float(), "spatial_shapes", tt_layer)
    lsi_sim = torch_to_simtensor(lsi_torch.float(), "level_start_idx", tt_layer)

    out_tt = tt_layer(tgt_sim, qpos_sim, ref_sim, src_sim, ss_sim, lsi_sim)

    print(f"  Output shape: {out_tt.shape}")
    if out_tt.data is not None:
        tt_data = out_tt.data
        print(f"  Stats: mean={tt_data.mean():.6f} std={tt_data.std():.6f}")
        print(f"  Values (first 8): {tt_data.flatten()[:8]}")
    else:
        print(f"  Data: None (shape inference only)")

    # ── Comparison ──
    print("\n" + "-" * 60)
    print("  Numerical Comparison")
    print("-" * 60)

    if not compare_numerical(
        out_pt, out_tt, "DecoderLayer output", rtol=0.1, atol=0.05
    ):
        all_passed = False

    expected = [batch_size, num_queries, d_model]
    shape_ok = list(out_pt.shape) == expected and out_tt.shape == expected
    print(f"\n  Shape validation: {'✓ PASS' if shape_ok else '✗ FAIL'}")
    if not shape_ok:
        all_passed = False

    return all_passed


if __name__ == "__main__":
    try:
        success = test_decoder_layer_numerical()
        print("\n" + "=" * 80)
        print(f"  OVERALL: {'PASSED ✓' if success else 'FAILED ✗'}")
        print("=" * 80)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
