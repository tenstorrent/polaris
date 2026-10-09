#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for Conv1dBlock module.
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

from navsim.agents.diffusiondrive.reference.torch_code.navsim.agents.diffusiondrive.modules.conditional_unet1d import (
    Conv1dBlock,
)
from navsim.agents.diffusiondrive.modules.conditional_unet1d_ttsim import (
    Conv1dBlock_TTSIM,
)

import ttsim.front.functional.op as F


def inject_conv1d_as_conv2d(ttsim_conv2d, pytorch_conv1d):
    """
    Inject Conv1d weights [C_out, C_in, K] into Conv2d weight [C_out, C_in, K, K].

    On (N,C,1,L) input with symmetric padding K//2, only kernel row K//2
    touches actual data. So we zero-fill the 2D weight and place the 1D
    weights in row K//2.
    """
    w1d = pytorch_conv1d.weight.detach().cpu().numpy()  # [C_out, C_in, K]
    K = w1d.shape[2]
    C_out, C_in = w1d.shape[0], w1d.shape[1]
    w2d = np.zeros((C_out, C_in, K, K), dtype=np.float32)
    w2d[:, :, K // 2, :] = w1d
    ttsim_conv2d.params[0][1].data = w2d

    # Bias (if any on the Conv1d)
    if pytorch_conv1d.bias is not None:
        b = pytorch_conv1d.bias.detach().cpu().numpy().astype(np.float32)
        # Bias goes into the F.Bias op, not the Conv2d
        return b
    return None


def inject_weights(ttsim_block, pytorch_block):
    """Inject all Conv1dBlock weights: Conv1d → Conv2d, GroupNorm weight/bias."""
    # Conv1d → Conv2d
    pt_conv = pytorch_block.block[0]  # nn.Conv1d
    bias_data = inject_conv1d_as_conv2d(ttsim_block.conv, pt_conv)

    # Bias (Conv1d bias → F.Bias)
    if bias_data is not None:
        ttsim_block.bias.params[0][1].data = bias_data.reshape(1, -1, 1, 1)

    # GroupNorm weight and bias
    pt_gn = pytorch_block.block[1]  # nn.GroupNorm
    ttsim_block.group_norm.weight.data = (
        pt_gn.weight.detach().cpu().numpy().astype(np.float32)
    )
    ttsim_block.group_norm.bias.data = (
        pt_gn.bias.detach().cpu().numpy().astype(np.float32)
    )


def main():
    print("=" * 70)
    print("Conv1dBlock Validation: PyTorch vs TTSIM")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    inp_channels = 64
    out_channels = 128
    kernel_size = 3
    n_groups = 8
    batch_size = 2
    seq_len = 16

    # --- PyTorch ---
    print("\n--- PyTorch Conv1dBlock ---")
    model_pt = Conv1dBlock(inp_channels, out_channels, kernel_size, n_groups)
    model_pt.eval()

    x_data = np.random.randn(batch_size, inp_channels, seq_len).astype(np.float32)
    x_pt = torch.from_numpy(x_data)

    with torch.no_grad():
        out_pt = model_pt(x_pt)
    out_pt_np = out_pt.numpy()

    print(f"Input shape:  {x_data.shape}")
    print(f"Output shape: {out_pt_np.shape}")
    print(
        f"Output stats: min={out_pt_np.min():.6f}, max={out_pt_np.max():.6f}, mean={out_pt_np.mean():.6f}"
    )

    # --- TTSIM ---
    print("\n--- TTSIM Conv1dBlock ---")
    model_ttsim = Conv1dBlock_TTSIM(
        "conv1dblk", inp_channels, out_channels, kernel_size, n_groups
    )

    print("Injecting weights...")
    inject_weights(model_ttsim, model_pt)

    x_ttsim = F._from_data("input", x_data)
    x_ttsim.link_module = model_ttsim

    out_ttsim = model_ttsim(x_ttsim)

    print(f"Input shape:  {x_ttsim.shape}")
    print(f"Output shape: {out_ttsim.shape}")

    # --- Comparison ---
    if out_ttsim.data is not None:
        print(
            f"Output stats: min={out_ttsim.data.min():.6f}, max={out_ttsim.data.max():.6f}, mean={out_ttsim.data.mean():.6f}"
        )

        print("\n--- Numerical Comparison ---")
        atol, rtol = 1e-4, 1e-4
        print(f"Tolerance: atol={atol}, rtol={rtol}")

        diff = np.abs(out_pt_np - out_ttsim.data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  Max absolute difference:  {max_diff:.10f}")
        print(f"  Mean absolute difference: {mean_diff:.10f}")

        # Shape check
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
