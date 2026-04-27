#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Numerical Computation Validation for DeformableTransformerEncoder.

Compares PyTorch vs TTSim outputs with synced weights on small random inputs.
Tests stacked encoder layers with shared weight sync.
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
    DeformableTransformerEncoderLayer as EncoderLayerPyTorch,
    DeformableTransformerEncoder as EncoderPyTorch,
)
from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
    DeformableTransformerEncoderLayer as EncoderLayerTTSim,
    DeformableTransformerEncoder as EncoderTTSim,
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


def sync_encoder_layer_weights(pt_layer, tt_layer):
    """Copy all weights from a PyTorch encoder layer to TTSim encoder layer."""
    # MSDeformAttn (custom Linear — no transpose, raw numpy)
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

    # LayerNorm
    tt_layer.norm1.params[0][1].data = pt_layer.norm1.weight.detach().numpy().copy()
    tt_layer.norm1.params[1][1].data = pt_layer.norm1.bias.detach().numpy().copy()
    tt_layer.norm2.params[0][1].data = pt_layer.norm2.weight.detach().numpy().copy()
    tt_layer.norm2.params[1][1].data = pt_layer.norm2.bias.detach().numpy().copy()


def sync_encoder_weights(pt_encoder, tt_encoder):
    """Sync all encoder layer weights (PyTorch uses clones, TTSim creates individually)."""
    for i, (pt_layer, tt_layer) in enumerate(zip(pt_encoder.layers, tt_encoder.layers)):
        sync_encoder_layer_weights(pt_layer, tt_layer)
    print(f"  Synced {len(pt_encoder.layers)} encoder layers")


# ============================================================================
# Tests
# ============================================================================


def test_encoder_numerical():
    """Test Encoder (stacked layers) with synced weights and numerical comparison."""

    print("\n" + "=" * 80)
    print("TEST: DeformableTransformerEncoder — Numerical Validation")
    print("=" * 80)

    all_passed = True

    # ── Small config ──
    batch_size = 1
    seq_len = 50
    d_model = 64
    n_levels = 2
    n_heads = 4
    n_points = 4
    d_ffn = 128
    n_layers = 2

    spatial_shapes = [[5, 5], [5, 5]]  # 25+25=50
    level_start_indices = [0, 25]

    print(
        f"\n  Config: B={batch_size}, seq={seq_len}, d_model={d_model}, "
        f"levels={n_levels}, heads={n_heads}, layers={n_layers}"
    )

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Shared inputs ──
    src_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 0.1
    pos_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 0.1
    valid_ratios_np = np.ones((batch_size, n_levels, 2), dtype=np.float32)

    src_torch = torch.from_numpy(src_np)
    pos_torch = torch.from_numpy(pos_np)
    spatial_shapes_torch = torch.tensor(spatial_shapes, dtype=torch.long)
    level_start_torch = torch.tensor(level_start_indices, dtype=torch.long)
    valid_ratios_torch = torch.from_numpy(valid_ratios_np)

    # ── PyTorch forward ──
    print("\n" + "-" * 60)
    print("  PyTorch forward pass")
    print("-" * 60)

    encoder_layer_pt = EncoderLayerPyTorch(
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    encoder_pt = EncoderPyTorch(encoder_layer_pt, n_layers)
    encoder_pt.eval()

    with torch.no_grad():
        out_pt = encoder_pt(
            src_torch,
            spatial_shapes_torch,
            level_start_torch,
            valid_ratios_torch,
            pos_torch,
        )

    pt_data = out_pt.detach().numpy()
    print(f"  Output shape: {list(out_pt.shape)}")
    print(f"  Stats: mean={pt_data.mean():.6f} std={pt_data.std():.6f}")
    print(f"  Values (first 8): {pt_data.flatten()[:8]}")

    # ── TTSim forward with synced weights ──
    print("\n" + "-" * 60)
    print("  TTSim forward pass (with weight sync)")
    print("-" * 60)

    encoder_layer_tt = EncoderLayerTTSim(
        name="enc_layer_num",
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    encoder_tt = EncoderTTSim(
        name="encoder_num", encoder_layer=encoder_layer_tt, num_layers=n_layers
    )

    sync_encoder_weights(encoder_pt, encoder_tt)

    # Convert to SimTensors with link_module set
    src_sim = torch_to_simtensor(src_torch, "src", encoder_tt)
    pos_sim = torch_to_simtensor(pos_torch, "pos", encoder_tt)
    ss_sim = torch_to_simtensor(
        spatial_shapes_torch.float(), "spatial_shapes", encoder_tt
    )
    lsi_sim = torch_to_simtensor(
        level_start_torch.float(), "level_start_idx", encoder_tt
    )
    vr_sim = torch_to_simtensor(valid_ratios_torch, "valid_ratios", encoder_tt)

    out_tt = encoder_tt(src_sim, ss_sim, lsi_sim, vr_sim, pos_sim)

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
        out_pt, out_tt, "Encoder output (2 layers)", rtol=0.1, atol=0.05
    ):
        all_passed = False

    expected = [batch_size, seq_len, d_model]
    shape_ok = list(out_pt.shape) == expected and out_tt.shape == expected
    print(f"\n  Shape validation: {'✓ PASS' if shape_ok else '✗ FAIL'}")
    if not shape_ok:
        all_passed = False

    return all_passed


if __name__ == "__main__":
    try:
        success = test_encoder_numerical()
        print("\n" + "=" * 80)
        print(f"  OVERALL: {'PASSED ✓' if success else 'FAILED ✗'}")
        print("=" * 80)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
