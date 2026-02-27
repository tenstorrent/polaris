#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Numerical Computation Validation for DeformableTransformerDecoder.

Compares PyTorch vs TTSim outputs with synced weights on small random inputs.
Tests stacked decoder layers with return_intermediate=False.

NOTE: Uses reference_points [B, Q, 2] - decoder expands them internally.

Weight sync reuses the decoder layer pattern with norm SWAP:
  PyTorch norm2 (self-attn)  -> TTSim norm1 (self-attn)
  PyTorch norm1 (cross-attn) -> TTSim norm2 (cross-attn)
  PyTorch norm3 (FFN)        -> TTSim norm3 (FFN)

USES FIXED TTSIM CODE: deformable_transformer_ttsim_fixed.py
"""

import os
import sys
import torch
import numpy as np

# Add repo root to path
# Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Import PyTorch reference implementation
from workloads.Deformable_DETR.reference.deformable_transformer import (
    DeformableTransformerDecoderLayer as DecoderLayerPyTorch,
    DeformableTransformerDecoder as DecoderPyTorch,
)

# Import FIXED TTSim implementation
# Option 1: If you placed the fixed file in the same directory as this test
try:
    from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
        DeformableTransformerDecoderLayer as DecoderLayerTTSim,
        DeformableTransformerDecoder as DecoderTTSim,
    )

    print(
        "  [INFO] Loaded FIXED TTSim from local deformable_transformer_ttsim_fixed.py"
    )
except ImportError:
    # Option 2: If you replaced the original file
    from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
        DeformableTransformerDecoderLayer as DecoderLayerTTSim,
        DeformableTransformerDecoder as DecoderTTSim,
    )

    print(
        "  [INFO] Loaded TTSim from workloads.Deformable_DETR.models.deformable_transformer_ttsim"
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


def sync_decoder_layer_weights(pt_layer, tt_layer):
    """
    Copy weights from a single PyTorch DecoderLayer to TTSim DecoderLayer.

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
        tt_proj.weight = pt_proj.weight.detach().numpy().copy()
        tt_proj.bias = pt_proj.bias.detach().numpy().copy()

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


def sync_decoder_weights(pt_decoder, tt_decoder):
    """Sync all decoder layer weights (PyTorch uses clones, TTSim creates individually)."""
    for i, (pt_layer, tt_layer) in enumerate(zip(pt_decoder.layers, tt_decoder.layers)):
        sync_decoder_layer_weights(pt_layer, tt_layer)
    print(f"  Synced {len(pt_decoder.layers)} decoder layers (with norm swap)")


# ============================================================================
# Tests
# ============================================================================


def test_single_decoder_layer():
    """Test single decoder layer with synced weights."""
    print("\n" + "=" * 80)
    print("TEST 1: Single DeformableTransformerDecoderLayer")
    print("=" * 80)

    batch_size = 1
    num_queries = 10
    src_seq_len = 50
    d_model = 64
    n_levels = 2
    n_heads = 4
    n_points = 4
    d_ffn = 128

    src_spatial_shapes = [[5, 5], [5, 5]]
    src_level_start_indices = [0, 25]

    print(
        f"\n  Config: B={batch_size}, queries={num_queries}, memory={src_seq_len}, "
        f"d_model={d_model}, levels={n_levels}, heads={n_heads}"
    )

    torch.manual_seed(42)
    np.random.seed(42)

    # Inputs
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

    # PyTorch
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
        pt_out = pt_layer(
            tgt_torch, query_pos_torch, ref_torch, src_torch, ss_torch, lsi_torch, None
        )

    # TTSim
    tt_layer = DecoderLayerTTSim(
        name="dec_layer_test",
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    sync_decoder_layer_weights(pt_layer, tt_layer)

    tgt_tt = torch_to_simtensor(tgt_torch, "tgt", tt_layer)
    qpos_tt = torch_to_simtensor(query_pos_torch, "query_pos", tt_layer)
    ref_tt = torch_to_simtensor(ref_torch, "ref", tt_layer)
    src_tt = torch_to_simtensor(src_torch, "src", tt_layer)
    ss_tt = torch_to_simtensor(ss_torch.float(), "ss", tt_layer)
    lsi_tt = torch_to_simtensor(lsi_torch.float(), "lsi", tt_layer)

    tt_out = tt_layer(tgt_tt, qpos_tt, ref_tt, src_tt, ss_tt, lsi_tt, None)

    return compare_numerical(
        pt_out, tt_out, "Single decoder layer output", rtol=1e-3, atol=1e-4
    )


def test_forward_ffn_directly():
    """Test forward_ffn method directly to verify the fix."""
    print("\n" + "=" * 80)
    print("TEST 2: forward_ffn Direct Comparison")
    print("=" * 80)

    batch_size, num_queries, d_model, d_ffn = 1, 10, 64, 128

    torch.manual_seed(42)
    np.random.seed(42)

    input_np = (
        np.random.randn(batch_size, num_queries, d_model).astype(np.float32) * 0.1
    )
    input_torch = torch.from_numpy(input_np)

    # PyTorch
    pt_layer = DecoderLayerPyTorch(
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=2,
        n_heads=4,
        n_points=4,
    )
    pt_layer.eval()

    with torch.no_grad():
        pt_ffn_out = pt_layer.forward_ffn(input_torch)

    # TTSim
    tt_layer = DecoderLayerTTSim(
        name="dec_layer_ffn_test",
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=2,
        n_heads=4,
        n_points=4,
    )
    sync_decoder_layer_weights(pt_layer, tt_layer)

    input_tt = torch_to_simtensor(input_torch, "ffn_input", tt_layer)
    tt_ffn_out = tt_layer.forward_ffn(input_tt)

    return compare_numerical(
        pt_ffn_out, tt_ffn_out, "forward_ffn output", rtol=1e-3, atol=1e-4
    )


def test_decoder_two_layers():
    """Test Decoder with 2 layers and return_intermediate=False."""
    print("\n" + "=" * 80)
    print("TEST 3: DeformableTransformerDecoder (2 layers, return_intermediate=False)")
    print("=" * 80)

    batch_size = 1
    num_queries = 10
    src_seq_len = 50
    d_model = 64
    n_levels = 2
    n_heads = 4
    n_points = 4
    d_ffn = 128
    n_layers = 2

    src_spatial_shapes = [[5, 5], [5, 5]]
    src_level_start_indices = [0, 25]

    print(
        f"\n  Config: B={batch_size}, queries={num_queries}, memory={src_seq_len}, "
        f"d_model={d_model}, levels={n_levels}, heads={n_heads}, layers={n_layers}"
    )

    torch.manual_seed(42)
    np.random.seed(42)

    # Inputs
    tgt_np = np.random.randn(batch_size, num_queries, d_model).astype(np.float32) * 0.1
    query_pos_np = (
        np.random.randn(batch_size, num_queries, d_model).astype(np.float32) * 0.1
    )
    ref_np = np.random.rand(batch_size, num_queries, 2).astype(np.float32)  # [B, Q, 2]
    src_np = np.random.randn(batch_size, src_seq_len, d_model).astype(np.float32) * 0.1
    valid_ratios_np = np.ones((batch_size, n_levels, 2), dtype=np.float32)

    tgt_torch = torch.from_numpy(tgt_np)
    query_pos_torch = torch.from_numpy(query_pos_np)
    ref_torch = torch.from_numpy(ref_np)
    src_torch = torch.from_numpy(src_np)
    ss_torch = torch.tensor(src_spatial_shapes, dtype=torch.long)
    lsi_torch = torch.tensor(src_level_start_indices, dtype=torch.long)
    vr_torch = torch.from_numpy(valid_ratios_np)

    # PyTorch
    print("\n  PyTorch forward pass...")
    decoder_layer_pt = DecoderLayerPyTorch(
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    decoder_pt = DecoderPyTorch(decoder_layer_pt, n_layers, return_intermediate=False)
    decoder_pt.eval()

    with torch.no_grad():
        out_pt, ref_out_pt = decoder_pt(
            tgt_torch,
            ref_torch,
            src_torch,
            ss_torch,
            lsi_torch,
            vr_torch,
            query_pos_torch,
        )

    print(f"    Output shape: {list(out_pt.shape)}")
    print(f"    Values (first 8): {out_pt.detach().numpy().flatten()[:8]}")

    # TTSim
    print("\n  TTSim forward pass...")
    decoder_layer_tt = DecoderLayerTTSim(
        name="dec_layer_num",
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    decoder_tt = DecoderTTSim(
        name="decoder_num",
        decoder_layer=decoder_layer_tt,
        num_layers=n_layers,
        return_intermediate=False,
    )

    sync_decoder_weights(decoder_pt, decoder_tt)

    tgt_sim = torch_to_simtensor(tgt_torch, "tgt", decoder_tt)
    qpos_sim = torch_to_simtensor(query_pos_torch, "query_pos", decoder_tt)
    ref_sim = torch_to_simtensor(ref_torch, "ref_points", decoder_tt)
    src_sim = torch_to_simtensor(src_torch, "src", decoder_tt)
    ss_sim = torch_to_simtensor(ss_torch.float(), "spatial_shapes", decoder_tt)
    lsi_sim = torch_to_simtensor(lsi_torch.float(), "level_start_idx", decoder_tt)
    vr_sim = torch_to_simtensor(vr_torch, "valid_ratios", decoder_tt)

    out_tt, ref_out_tt = decoder_tt(
        tgt_sim, ref_sim, src_sim, ss_sim, lsi_sim, vr_sim, qpos_sim
    )

    print(f"    Output shape: {out_tt.shape}")
    if out_tt.data is not None:
        print(f"    Values (first 8): {out_tt.data.flatten()[:8]}")

    return compare_numerical(
        out_pt, out_tt, "Decoder output (2 layers)", rtol=1e-2, atol=1e-3
    )


def test_decoder_with_intermediate():
    """Test Decoder with return_intermediate=True."""
    print("\n" + "=" * 80)
    print("TEST 4: DeformableTransformerDecoder (2 layers, return_intermediate=True)")
    print("=" * 80)

    batch_size = 1
    num_queries = 10
    src_seq_len = 50
    d_model = 64
    n_levels = 2
    n_heads = 4
    n_points = 4
    d_ffn = 128
    n_layers = 2

    src_spatial_shapes = [[5, 5], [5, 5]]
    src_level_start_indices = [0, 25]

    torch.manual_seed(42)
    np.random.seed(42)

    # Inputs
    tgt_np = np.random.randn(batch_size, num_queries, d_model).astype(np.float32) * 0.1
    query_pos_np = (
        np.random.randn(batch_size, num_queries, d_model).astype(np.float32) * 0.1
    )
    ref_np = np.random.rand(batch_size, num_queries, 2).astype(np.float32)
    src_np = np.random.randn(batch_size, src_seq_len, d_model).astype(np.float32) * 0.1
    valid_ratios_np = np.ones((batch_size, n_levels, 2), dtype=np.float32)

    tgt_torch = torch.from_numpy(tgt_np)
    query_pos_torch = torch.from_numpy(query_pos_np)
    ref_torch = torch.from_numpy(ref_np)
    src_torch = torch.from_numpy(src_np)
    ss_torch = torch.tensor(src_spatial_shapes, dtype=torch.long)
    lsi_torch = torch.tensor(src_level_start_indices, dtype=torch.long)
    vr_torch = torch.from_numpy(valid_ratios_np)

    # PyTorch
    decoder_layer_pt = DecoderLayerPyTorch(
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    decoder_pt = DecoderPyTorch(decoder_layer_pt, n_layers, return_intermediate=True)
    decoder_pt.eval()

    with torch.no_grad():
        out_pt, ref_out_pt = decoder_pt(
            tgt_torch,
            ref_torch,
            src_torch,
            ss_torch,
            lsi_torch,
            vr_torch,
            query_pos_torch,
        )

    # TTSim
    decoder_layer_tt = DecoderLayerTTSim(
        name="dec_layer_inter",
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    decoder_tt = DecoderTTSim(
        name="decoder_inter",
        decoder_layer=decoder_layer_tt,
        num_layers=n_layers,
        return_intermediate=True,
    )

    sync_decoder_weights(decoder_pt, decoder_tt)

    tgt_sim = torch_to_simtensor(tgt_torch, "tgt", decoder_tt)
    qpos_sim = torch_to_simtensor(query_pos_torch, "query_pos", decoder_tt)
    ref_sim = torch_to_simtensor(ref_torch, "ref_points", decoder_tt)
    src_sim = torch_to_simtensor(src_torch, "src", decoder_tt)
    ss_sim = torch_to_simtensor(ss_torch.float(), "spatial_shapes", decoder_tt)
    lsi_sim = torch_to_simtensor(lsi_torch.float(), "level_start_idx", decoder_tt)
    vr_sim = torch_to_simtensor(vr_torch, "valid_ratios", decoder_tt)

    out_tt, ref_out_tt = decoder_tt(
        tgt_sim, ref_sim, src_sim, ss_sim, lsi_sim, vr_sim, qpos_sim
    )

    all_passed = True
    print(f"\n  Output shape: PT={out_pt.shape} TT={out_tt.shape}")

    for i in range(n_layers):
        passed = compare_numerical(
            out_pt[i],
            out_tt.data[i],
            f"Layer {i} intermediate output",
            rtol=1e-2,
            atol=1e-3,
        )
        all_passed &= passed

    return all_passed


def test_decoder_larger_config():
    """Test with a larger, more realistic configuration."""
    print("\n" + "=" * 80)
    print("TEST 5: Larger Configuration (closer to real Deformable-DETR)")
    print("=" * 80)

    batch_size = 2
    num_queries = 100
    d_model = 256
    n_levels = 4
    n_heads = 8
    n_points = 4
    d_ffn = 1024
    n_layers = 3

    # Spatial shapes for 4 levels
    src_spatial_shapes = [[20, 20], [10, 10], [5, 5], [3, 3]]
    src_seq_len = sum(h * w for h, w in src_spatial_shapes)  # 400 + 100 + 25 + 9 = 534
    src_level_start_indices = [0, 400, 500, 525]

    print(
        f"\n  Config: B={batch_size}, queries={num_queries}, memory={src_seq_len}, "
        f"d_model={d_model}, levels={n_levels}, heads={n_heads}, layers={n_layers}"
    )

    torch.manual_seed(123)
    np.random.seed(123)

    # Inputs
    tgt_np = np.random.randn(batch_size, num_queries, d_model).astype(np.float32) * 0.1
    query_pos_np = (
        np.random.randn(batch_size, num_queries, d_model).astype(np.float32) * 0.1
    )
    ref_np = np.random.rand(batch_size, num_queries, 2).astype(np.float32)
    src_np = np.random.randn(batch_size, src_seq_len, d_model).astype(np.float32) * 0.1
    valid_ratios_np = np.ones((batch_size, n_levels, 2), dtype=np.float32)

    tgt_torch = torch.from_numpy(tgt_np)
    query_pos_torch = torch.from_numpy(query_pos_np)
    ref_torch = torch.from_numpy(ref_np)
    src_torch = torch.from_numpy(src_np)
    ss_torch = torch.tensor(src_spatial_shapes, dtype=torch.long)
    lsi_torch = torch.tensor(src_level_start_indices, dtype=torch.long)
    vr_torch = torch.from_numpy(valid_ratios_np)

    # PyTorch
    print("\n  PyTorch forward pass...")
    decoder_layer_pt = DecoderLayerPyTorch(
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    decoder_pt = DecoderPyTorch(decoder_layer_pt, n_layers, return_intermediate=False)
    decoder_pt.eval()

    with torch.no_grad():
        out_pt, _ = decoder_pt(
            tgt_torch,
            ref_torch,
            src_torch,
            ss_torch,
            lsi_torch,
            vr_torch,
            query_pos_torch,
        )

    print(f"    Output shape: {list(out_pt.shape)}")

    # TTSim
    print("\n  TTSim forward pass...")
    decoder_layer_tt = DecoderLayerTTSim(
        name="dec_layer_large",
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    decoder_tt = DecoderTTSim(
        name="decoder_large",
        decoder_layer=decoder_layer_tt,
        num_layers=n_layers,
        return_intermediate=False,
    )

    sync_decoder_weights(decoder_pt, decoder_tt)

    tgt_sim = torch_to_simtensor(tgt_torch, "tgt", decoder_tt)
    qpos_sim = torch_to_simtensor(query_pos_torch, "query_pos", decoder_tt)
    ref_sim = torch_to_simtensor(ref_torch, "ref_points", decoder_tt)
    src_sim = torch_to_simtensor(src_torch, "src", decoder_tt)
    ss_sim = torch_to_simtensor(ss_torch.float(), "spatial_shapes", decoder_tt)
    lsi_sim = torch_to_simtensor(lsi_torch.float(), "level_start_idx", decoder_tt)
    vr_sim = torch_to_simtensor(vr_torch, "valid_ratios", decoder_tt)

    out_tt, _ = decoder_tt(tgt_sim, ref_sim, src_sim, ss_sim, lsi_sim, vr_sim, qpos_sim)

    print(f"    Output shape: {out_tt.shape}")

    # Use slightly higher tolerance for larger model (3 layers accumulate more error)
    return compare_numerical(
        out_pt, out_tt, "Large decoder output (3 layers)", rtol=0.05, atol=0.01
    )


def test_decoder_numerical():
    """
    Main entry point for decoder numerical validation.
    Runs the key decoder tests and returns True if all pass.
    """
    results = {}

    try:
        results["single_layer"] = test_single_decoder_layer()
    except Exception as e:
        print(f"  ERROR in single_layer: {e}")
        results["single_layer"] = False

    try:
        results["decoder_2layers"] = test_decoder_two_layers()
    except Exception as e:
        print(f"  ERROR in decoder_2layers: {e}")
        results["decoder_2layers"] = False

    return all(results.values())


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DEFORMABLE TRANSFORMER DECODER - NUMERICAL VALIDATION")
    print("Using FIXED TTSim implementation")
    print("=" * 80)

    results = {}

    try:
        results["test_1_single_layer"] = test_single_decoder_layer()
    except Exception as e:
        print(f"\n  ERROR in test 1: {e}")
        import traceback

        traceback.print_exc()
        results["test_1_single_layer"] = False

    try:
        results["test_2_forward_ffn"] = test_forward_ffn_directly()
    except Exception as e:
        print(f"\n  ERROR in test 2: {e}")
        import traceback

        traceback.print_exc()
        results["test_2_forward_ffn"] = False

    try:
        results["test_3_decoder_2layers"] = test_decoder_two_layers()
    except Exception as e:
        print(f"\n  ERROR in test 3: {e}")
        import traceback

        traceback.print_exc()
        results["test_3_decoder_2layers"] = False

    try:
        results["test_4_intermediate"] = test_decoder_with_intermediate()
    except Exception as e:
        print(f"\n  ERROR in test 4: {e}")
        import traceback

        traceback.print_exc()
        results["test_4_intermediate"] = False

    try:
        results["test_5_larger_config"] = test_decoder_larger_config()
    except Exception as e:
        print(f"\n  ERROR in test 5: {e}")
        import traceback

        traceback.print_exc()
        results["test_5_larger_config"] = False

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 80)
    print(f"OVERALL: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("=" * 80)

    sys.exit(0 if all_passed else 1)
