#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for ModulationLayer.
Tests shape inference and numerical equivalence between PyTorch and TTSIM.
"""

import os
import sys

# Add DiffusionDrive to path for navsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
# Add TT-Polaris to path for ttsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.diffusiondrive.reference.torch_code.navsim.agents.diffusiondrive.transfuser_model_v2 import (
    ModulationLayer as ModulationLayer_PyTorch,
)
from navsim.agents.diffusiondrive.transfuser_model_v2_ttsim import (
    ModulationLayer as ModulationLayer_TTSIM,
)

import ttsim.front.functional.op as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def inject_weights(ttsim_layer, pytorch_layer):
    """Inject PyTorch weights into TTSIM ModulationLayer."""
    # Scale shift MLP
    ttsim_layer.scale_shift_linear.params[0][1].data = pytorch_layer.scale_shift_mlp[
        1
    ].weight.data.T.numpy()
    ttsim_layer.scale_shift_bias.params[0][1].data = pytorch_layer.scale_shift_mlp[
        1
    ].bias.data.numpy()


def main():
    print("=" * 70)
    print("ModulationLayer Validation: PyTorch vs TTSIM")
    print("=" * 70)

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    embed_dims = 256
    condition_dims = 256
    batch_size = 2
    seq_len = 20

    # Create PyTorch model
    print("\n--- PyTorch ModulationLayer ---")
    model_pt = ModulationLayer_PyTorch(
        embed_dims=embed_dims, condition_dims=condition_dims
    )
    model_pt.eval()

    # Generate random inputs
    traj_feature_data = np.random.randn(batch_size, seq_len, embed_dims).astype(
        np.float32
    )
    time_embed_data = np.random.randn(batch_size, 1, condition_dims).astype(np.float32)

    traj_feature_pt = torch.from_numpy(traj_feature_data)
    time_embed_pt = torch.from_numpy(time_embed_data)

    # Forward pass
    with torch.no_grad():
        output_pt = model_pt(
            traj_feature_pt, time_embed_pt, global_cond=None, global_img=None
        )

    output_pt_np = output_pt.numpy()

    print(f"Trajectory feature shape: {traj_feature_pt.shape}")
    print(f"Time embed shape: {time_embed_pt.shape}")
    print(f"Output shape: {output_pt_np.shape}")
    print(
        f"Output stats: min={output_pt_np.min():.6f}, max={output_pt_np.max():.6f}, mean={output_pt_np.mean():.6f}"
    )

    # TTSIM model
    print("\n--- TTSIM ModulationLayer ---")
    model_ttsim = ModulationLayer_TTSIM(
        embed_dims=embed_dims, condition_dims=condition_dims
    )

    # Inject weights
    print("Injecting weights...")
    inject_weights(model_ttsim, model_pt)

    # Forward pass
    traj_feature_ttsim = F._from_data("traj_feature", traj_feature_data)
    time_embed_ttsim = F._from_data("time_embed", time_embed_data)
    traj_feature_ttsim.link_module = model_ttsim
    time_embed_ttsim.link_module = model_ttsim

    output_ttsim = model_ttsim(
        traj_feature_ttsim, time_embed_ttsim, global_cond=None, global_img=None
    )

    print(f"Trajectory feature shape: {traj_feature_ttsim.shape}")
    print(f"Time embed shape: {time_embed_ttsim.shape}")
    print(f"Output shape: {output_ttsim.shape}")

    # Check if data is available
    if output_ttsim.data is not None:
        print(
            f"Output stats: min={output_ttsim.data.min():.6f}, max={output_ttsim.data.max():.6f}, mean={output_ttsim.data.mean():.6f}"
        )

        # Numerical comparison
        print("\n--- Numerical Comparison: PyTorch vs TTSIM ---")

        atol = 1e-4
        rtol = 1e-4

        print(f"Tolerance: atol={atol}, rtol={rtol}")

        diff = np.abs(output_pt_np - output_ttsim.data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"  Max absolute difference: {max_diff:.10f}")
        print(f"  Mean absolute difference: {mean_diff:.10f}")

        is_close = np.allclose(output_pt_np, output_ttsim.data, atol=atol, rtol=rtol)

        print("\n" + "=" * 70)
        if is_close:
            print("OVERALL: PASS - TTSIM matches PyTorch")
        else:
            print("OVERALL: FAIL - Differences exceed tolerance")
        print("=" * 70)
    else:
        print("  WARN: SKIPPED: No TTSIM data available")

    print()


if __name__ == "__main__":
    main()
