#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comparison test for UpsampleBlock: ttsim vs PyTorch
"""

import os, sys

polaris_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, polaris_path)

# Fix for OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import numpy as np
from workloads.MapTracker.plugin.models.backbones.bevformer_backbone import (
    UpsampleBlock,
)


class UpsampleBlockPyTorch(nn.Module):
    """
    PyTorch reference implementation of UpsampleBlock
    Standalone version without mmcv dependencies
    """

    def __init__(self, ins, outs):
        super().__init__()
        self.conv = nn.Conv2d(ins, outs, kernel_size=3, stride=1, padding=1)
        self.gn = nn.GroupNorm(32, outs)

        # Initialize weights with Xavier uniform (matches original)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input [batch, ins, h, w]

        Returns:
            out: Output [batch, outs, h*2, w*2]
        """
        # 1. Conv2d
        x = self.conv(x)

        # 2. GroupNorm
        x = self.gn(x)

        # 3. ReLU
        x = F_torch.relu(x)

        # 4. Upsample 2x (bilinear interpolation)
        x = F_torch.interpolate(
            x, scale_factor=2.0, mode="bilinear", align_corners=True
        )

        return x


def test_upsample_block():
    """Test UpsampleBlock: ttsim vs PyTorch"""

    # Set environment variable for OpenMP compatibility
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    print("=" * 80)
    print("Testing UpsampleBlock: ttsim vs PyTorch")
    print("=" * 80)

    # Test parameters
    batch = 2
    ins = 64
    outs = 128
    h, w = 8, 10

    # Create input
    np.random.seed(42)
    torch.manual_seed(42)

    input_np = np.random.randn(batch, ins, h, w).astype(np.float32)
    input_torch = torch.from_numpy(input_np)

    print(f"\nInput shape: {input_np.shape}")
    print(f"ins={ins}, outs={outs}, h={h}, w={w}")

    # ========== PyTorch Implementation ==========
    print("\n" + "-" * 80)
    print("PyTorch Implementation")
    print("-" * 80)

    model_pytorch = UpsampleBlockPyTorch(ins, outs)
    model_pytorch.eval()

    with torch.no_grad():
        output_pytorch = model_pytorch(input_torch)

    print(f"PyTorch output shape: {output_pytorch.shape}")
    print(
        f"PyTorch output range: [{output_pytorch.min().item():.6f}, {output_pytorch.max().item():.6f}]"
    )
    print(output_pytorch)
    # ========== ttsim Implementation ==========
    print("\n" + "-" * 80)
    print("ttsim Implementation")
    print("-" * 80)

    model_ttsim = UpsampleBlock("upsample_block", ins, outs)

    # Inject PyTorch weights into ttsim
    print("\nInjecting PyTorch weights into ttsim...")

    # Conv2d weights and bias
    conv_weight = model_pytorch.conv.weight.data.numpy()  # [outs, ins, 3, 3]
    conv_bias = model_pytorch.conv.bias.data.numpy()  # [outs]

    # ttsim Conv2d expects [outs, ins, kh, kw] - same as PyTorch
    model_ttsim.conv_weight.params[0][1].data = conv_weight
    model_ttsim.conv_bias_param.data = conv_bias

    # GroupNorm weight and bias
    gn_weight = model_pytorch.gn.weight.data.numpy()  # [outs]
    gn_bias = model_pytorch.gn.bias.data.numpy()  # [outs]

    model_ttsim.gn.params[0][1].data = gn_weight
    model_ttsim.gn.params[1][1].data = gn_bias

    print("Weight injection complete")

    # Create input tensor
    import ttsim.front.functional.op as F

    input_ttsim = F._from_data("input", input_np, is_const=True)

    # Forward pass
    output_ttsim = model_ttsim(input_ttsim)

    if output_ttsim.data is None:
        print("ERROR: ttsim output.data is None!")
        print("Data computation not working properly")
        return

    print(f"ttsim output shape: {output_ttsim.data.shape}")
    print(
        f"ttsim output range: [{output_ttsim.data.min():.6f}, {output_ttsim.data.max():.6f}]"
    )
    print(output_ttsim.data)
    # ========== Comparison ==========
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)

    output_pytorch_np = output_pytorch.numpy()
    diff = np.abs(output_ttsim.data - output_pytorch_np)

    print(f"\nMax absolute difference: {diff.max():.10f}")
    print(f"Mean absolute difference: {diff.mean():.10f}")
    print(f"Std absolute difference: {diff.std():.10f}")

    # Check if outputs match
    atol = 1e-5  # Tolerance for upsample operation
    rtol = 1e-5

    if np.allclose(output_ttsim.data, output_pytorch_np, atol=atol, rtol=rtol):
        print(
            f"\n[PASS] PASS: Outputs match within tolerance (atol={atol}, rtol={rtol})"
        )
    else:
        print(
            f"\n[FAIL] FAIL: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})"
        )

        # Show some differing values
        max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"\nLargest difference at index {max_diff_idx}:")
        print(f"  ttsim:   {output_ttsim.data[max_diff_idx]:.10f}")
        print(f"  PyTorch: {output_pytorch_np[max_diff_idx]:.10f}")
        print(f"  diff:    {diff[max_diff_idx]:.10f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_upsample_block()
