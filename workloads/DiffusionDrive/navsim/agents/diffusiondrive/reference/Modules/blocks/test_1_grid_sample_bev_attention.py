#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for GridSampleCrossBEVAttention module.
Tests shape inference and numerical equivalence between PyTorch and TTSIM.
"""

import os
import sys

# Add DiffusionDrive to path for navsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
# Add TT-Polaris to path for ttsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../.."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.diffusiondrive.modules.blocks import GridSampleCrossBEVAttention_TTSIM
from navsim.agents.diffusiondrive.reference.torch_code.navsim.agents.diffusiondrive.modules.blocks import GridSampleCrossBEVAttention
import ttsim.front.functional.op as F


# ---------------------------------------------------------------------------
# Lightweight config mock matching TransfuserConfig defaults
# ---------------------------------------------------------------------------
class MockConfig:
    lidar_max_x: float = 32.0
    lidar_max_y: float = 32.0


# ---------------------------------------------------------------------------
# Weight injection helpers
# ---------------------------------------------------------------------------
def inject_linear(ttsim_linear_op, ttsim_bias_op, pt_linear):
    """Inject nn.Linear weight/bias into TTSIM Linear + Bias ops."""
    w = pt_linear.weight.detach().cpu().numpy().astype(np.float32)
    b = pt_linear.bias.detach().cpu().numpy().astype(np.float32)

    # F.Linear stores weight in params[0][1] — needs transpose
    ttsim_linear_op.params[0][1].data = w.T
    ttsim_bias_op.params[0][1].data = b


def inject_conv2d(ttsim_conv_op, ttsim_bias_op, pt_conv):
    """Inject nn.Conv2d weight/bias into TTSIM Conv2d + Bias ops."""
    w = pt_conv.weight.detach().cpu().numpy().astype(np.float32)
    ttsim_conv_op.params[0][1].data = w

    if pt_conv.bias is not None:
        b = pt_conv.bias.detach().cpu().numpy().astype(np.float32)
        ttsim_bias_op.params[0][1].data = b.reshape(1, -1, 1, 1)


def inject_weights(ttsim_model, pt_model):
    """Inject all weights from PyTorch GridSampleCrossBEVAttention into TTSIM."""
    # attention_weights: nn.Linear(embed_dims, num_points)
    inject_linear(
        ttsim_model.attention_weights_linear,
        ttsim_model.attention_weights_bias,
        pt_model.attention_weights,
    )

    # output_proj: nn.Linear(embed_dims, embed_dims)
    inject_linear(
        ttsim_model.output_proj_linear,
        ttsim_model.output_proj_bias,
        pt_model.output_proj,
    )

    # value_proj: nn.Sequential(Conv2d, ReLU)
    pt_conv = pt_model.value_proj[0]  # Conv2d
    inject_conv2d(
        ttsim_model.value_proj_conv,
        ttsim_model.value_proj_bias,
        pt_conv,
    )


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("GridSampleCrossBEVAttention Validation: PyTorch vs TTSIM")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    embed_dims = 256
    num_heads = 8
    num_levels = 1
    in_bev_dims = 64
    num_points = 8
    config = MockConfig()

    batch_size = 2
    num_queries = 6
    bev_h, bev_w = 16, 16
    spatial_shape = (bev_h, bev_w)

    # --- PyTorch ---
    print("\n--- PyTorch GridSampleCrossBEVAttention ---")
    model_pt = GridSampleCrossBEVAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        in_bev_dims=in_bev_dims,
        num_points=num_points,
        config=config,
    )
    model_pt.eval()

    # Create test inputs
    queries_data = np.random.randn(batch_size, num_queries, embed_dims).astype(
        np.float32
    )
    # traj_points normalised to [-lidar_max, lidar_max] range
    traj_data = (np.random.randn(batch_size, num_queries, num_points, 2) * 10).astype(
        np.float32
    )
    bev_data = np.random.randn(batch_size, in_bev_dims, bev_h, bev_w).astype(np.float32)

    queries_pt = torch.from_numpy(queries_data)
    traj_pt = torch.from_numpy(traj_data)
    bev_pt = torch.from_numpy(bev_data)

    with torch.no_grad():
        out_pt = model_pt(queries_pt, traj_pt, bev_pt, spatial_shape)
    out_pt_np = out_pt.numpy()

    print(f"  queries shape:    {queries_data.shape}")
    print(f"  traj_points shape:{traj_data.shape}")
    print(f"  bev_feature shape:{bev_data.shape}")
    print(f"  Output shape:     {out_pt_np.shape}")
    print(
        f"  Output stats: min={out_pt_np.min():.6f}, max={out_pt_np.max():.6f}, mean={out_pt_np.mean():.6f}"
    )

    # --- TTSIM ---
    print("\n--- TTSIM GridSampleCrossBEVAttention ---")
    model_ttsim = GridSampleCrossBEVAttention_TTSIM(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        in_bev_dims=in_bev_dims,
        num_points=num_points,
        config=config,
    )

    print("  Injecting weights...")
    inject_weights(model_ttsim, model_pt)

    # Create TTSIM tensors
    queries_ts = F._from_data("queries", queries_data)
    traj_ts = F._from_data("traj_points", traj_data)
    bev_ts = F._from_data("bev_feature", bev_data)

    queries_ts.link_module = model_ttsim

    out_ttsim = model_ttsim(queries_ts, traj_ts, bev_ts, spatial_shape)

    print(f"  Output shape: {out_ttsim.shape}")

    # --- Comparison ---
    if out_ttsim.data is not None:
        print(
            f"  Output stats: min={out_ttsim.data.min():.6f}, max={out_ttsim.data.max():.6f}, mean={out_ttsim.data.mean():.6f}"
        )

        print("\n--- Numerical Comparison ---")
        atol, rtol = 1e-4, 1e-4
        print(f"  Tolerance: atol={atol}, rtol={rtol}")

        diff = np.abs(out_pt_np - out_ttsim.data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  Max absolute difference:  {max_diff:.10f}")
        print(f"  Mean absolute difference: {mean_diff:.10f}")

        shape_match = list(out_pt_np.shape) == list(out_ttsim.shape)
        print(
            f"\n  Shape match: {'[PASS]' if shape_match else '[FAIL]'}  PT={out_pt_np.shape}  TTSIM={out_ttsim.shape}"
        )

        is_close = np.allclose(out_pt_np, out_ttsim.data, atol=atol, rtol=rtol)

        print("\n" + "=" * 70)
        if is_close and shape_match:
            print("OVERALL: [PASS] PASS - TTSIM matches PyTorch")
        else:
            print("OVERALL: [FAIL] FAIL - Differences exceed tolerance")
        print("=" * 70)
    else:
        print("  [WARN] SKIPPED: No TTSIM data available")

    print()


if __name__ == "__main__":
    main()
