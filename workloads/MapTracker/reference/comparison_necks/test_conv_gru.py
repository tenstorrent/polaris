#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comparison script: ConvGRU PyTorch vs ttsim
Tests Convolutional GRU for temporal feature fusion
"""

import os, sys

# Add polaris directory to path
polaris_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if polaris_path not in sys.path:
    sys.path.insert(0, polaris_path)

import torch
import torch.nn as nn
import numpy as np

import ttsim.front.functional.op as F

# Import ttsim implementation
from workloads.MapTracker.plugin.models.necks.gru import ConvGRU as ConvGRUTtsim

# ============================================================
# PyTorch Reference Implementation
# ============================================================


def kaiming_init_weights(module):
    """
    Apply Kaiming initialization to Conv2d layers
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ConvGRUPyTorch(nn.Module):
    """
    Convolutional GRU for temporal feature fusion (PyTorch reference)
    Standalone version without mmdet dependency
    """

    def __init__(self, out_channels):
        super(ConvGRUPyTorch, self).__init__()
        kernel_size = 1
        padding = kernel_size // 2
        self.convz = nn.Conv2d(
            2 * out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.convr = nn.Conv2d(
            2 * out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.convq = nn.Conv2d(
            2 * out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.ln = nn.LayerNorm(out_channels)
        self.zero_out = nn.Conv2d(out_channels, out_channels, 1, 1, bias=True)

    def init_weights(self):
        kaiming_init_weights(self)
        # Zero-initialize the output conv
        nn.init.zeros_(self.zero_out.weight)
        nn.init.zeros_(self.zero_out.bias)

    def forward(self, h, x):
        if len(h.shape) == 3:
            h = h.unsqueeze(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        hx = torch.cat([h, x], dim=1)  # [B, 2C, H, W]
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        new_x = torch.cat([r * h, x], dim=1)  # [B, 2C, H, W]
        q = self.convq(new_x)

        out = (1 - z) * h + z * q  # [B, C, H, W]
        out = self.ln(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        out = self.zero_out(out)
        out = out + x
        out = out.squeeze(0)

        return out


# ============================================================
# Comparison Tests
# ============================================================


print("=" * 80)
print("ConvGRU Comparison: PyTorch vs ttsim")
print("=" * 80)

# Configuration
out_channels = 64
batch_size = 2
height = 12
width = 12

print(f"\nConfiguration:")
print(f"  out_channels={out_channels}")
print(f"  Input shape: [{batch_size}, {out_channels}, {height}, {width}]")

# Create random inputs (hidden state h and new input x)
h_np = np.random.randn(batch_size, out_channels, height, width).astype(np.float32)
x_np = np.random.randn(batch_size, out_channels, height, width).astype(np.float32)

print(
    f"\nInput h stats: min={h_np.min():.4f}, max={h_np.max():.4f}, mean={h_np.mean():.4f}"
)
print(
    f"Input x stats: min={x_np.min():.4f}, max={x_np.max():.4f}, mean={x_np.mean():.4f}"
)


# ============================================================
# TEST 1: PyTorch ConvGRU
# ============================================================
print("\n" + "=" * 80)
print("TEST 1: PyTorch ConvGRU")
print("-" * 80)

gru_pytorch = ConvGRUPyTorch(out_channels=out_channels)
gru_pytorch.eval()

# Apply Kaiming initialization
gru_pytorch.init_weights()
print("  Applied Kaiming initialization to Conv2d layers")

h_torch = torch.from_numpy(h_np)
x_torch = torch.from_numpy(x_np)

with torch.no_grad():
    output_pytorch = gru_pytorch(h_torch, x_torch)

output_pytorch_np = output_pytorch.numpy()

print(f"  Hidden state shape: {h_torch.shape}")
print(f"  Input shape: {x_torch.shape}")
print(f"  Output shape: {output_pytorch.shape}")
print(f"\n  Output values:\n{output_pytorch_np}")
print(f"  Output stats:")
print(f"    Min:  {output_pytorch_np.min():.6f}")
print(f"    Max:  {output_pytorch_np.max():.6f}")
print(f"    Mean: {output_pytorch_np.mean():.6f}")
print(f"    Std:  {output_pytorch_np.std():.6f}")


# ============================================================
# TEST 2: ttsim ConvGRU
# ============================================================
print("\n" + "=" * 80)
print("TEST 2: ttsim ConvGRU")
print("-" * 80)

gru_ttsim = ConvGRUTtsim(out_channels=out_channels)

# Inject PyTorch weights into ttsim operators
print("  Injecting PyTorch weights into ttsim model...")

# Conv gates (z, r, q): weight only (no bias)
gru_ttsim.convz_weight.params[0][1].data = gru_pytorch.convz.weight.data.numpy()
gru_ttsim.convr_weight.params[0][1].data = gru_pytorch.convr.weight.data.numpy()
gru_ttsim.convq_weight.params[0][1].data = gru_pytorch.convq.weight.data.numpy()

# LayerNorm: access internal parameters via _tensors dict
gru_ttsim._tensors["layer_norm.scale"].data = gru_pytorch.ln.weight.data.numpy()
gru_ttsim._tensors["layer_norm.bias"].data = gru_pytorch.ln.bias.data.numpy()

# Zero-out conv: weight and bias
gru_ttsim.zero_out_weight.params[0][1].data = gru_pytorch.zero_out.weight.data.numpy()
gru_ttsim.zero_out_bias_param.data = gru_pytorch.zero_out.bias.data.numpy()

# Create input SimTensors with data
h_simtensor = F._from_data("h_input", data=h_np, is_const=False)
x_simtensor = F._from_data("x_input", data=x_np, is_const=False)


output_ttsim = gru_ttsim(h_simtensor, x_simtensor)

print(f"\n  Hidden state shape: {h_simtensor.shape}")
print(f"  Input shape: {x_simtensor.shape}")
print(f"  Output shape: {output_ttsim.shape}")
print(f"  Output .data is None? {output_ttsim.data is None}")

# Check intermediate tensors in the module
print("\nDEBUG: Checking intermediate operations:")
if hasattr(gru_ttsim, "_tensors"):
    for name, tensor in gru_ttsim._tensors.items():
        if hasattr(tensor, "data"):
            print(f"  {name}: data is None? {tensor.data is None}")

if output_ttsim.data is not None:
    print(output_ttsim.data)
    print(f"  Output stats:")
    print(f"    Min:  {output_ttsim.data.min():.6f}")
    print(f"    Max:  {output_ttsim.data.max():.6f}")
    print(f"    Mean: {output_ttsim.data.mean():.6f}")
    print(f"    Std:  {output_ttsim.data.std():.6f}")
else:
    print("  WARNING: output.data is None (shape inference only)")


# ============================================================
# TEST 3: Numerical Comparison
# ============================================================
print("\n" + "=" * 80)
print("TEST 3: Numerical Comparison")
print("-" * 80)

if output_ttsim.data is not None:
    ttsim_output = output_ttsim.data

    # Sanity check: ensure we're comparing different arrays
    print(f"  Sanity checks:")
    print(
        f"    PyTorch output is same object as ttsim? {output_pytorch_np is ttsim_output}"
    )
    print(f"    PyTorch output shape: {output_pytorch_np.shape}")
    print(f"    ttsim output shape: {ttsim_output.shape}")
    print(f"    PyTorch sample values: {output_pytorch_np[0, 0, 0, :5]}")
    print(f"    ttsim sample values:   {ttsim_output[0, 0, 0, :5]}")

    # Compute difference
    diff = np.abs(output_pytorch_np - ttsim_output)

    print(f"  Absolute difference:")
    print(f"    Min:  {diff.min():.20e}")
    print(f"    Max:  {diff.max():.20e}")
    print(f"    Mean: {diff.mean():.20e}")
    print(f"    Std:  {diff.std():.20e}")

    # Check if outputs match within tolerance
    atol = 1e-5
    rtol = 1e-5
    matches = np.allclose(output_pytorch_np, ttsim_output, atol=atol, rtol=rtol)

    if matches:
        print(
            f"\n  [PASS] [PASS] Outputs match within tolerance (atol={atol}, rtol={rtol})"
        )
    else:
        print(
            f"\n  [FAIL] [FAIL] Outputs differ beyond tolerance (atol={atol}, rtol={rtol})"
        )

        # Show first mismatch
        diff_mask = ~np.isclose(output_pytorch_np, ttsim_output, atol=atol, rtol=rtol)
        if diff_mask.any():
            mismatch_idx = np.where(diff_mask)
            b, c, h, w = (
                mismatch_idx[0][0],
                mismatch_idx[1][0],
                mismatch_idx[2][0],
                mismatch_idx[3][0],
            )
            print(f"\n  First mismatch at [batch={b}, channel={c}, h={h}, w={w}]:")
            print(f"    PyTorch: {output_pytorch_np[b, c, h, w]:.10f}")
            print(f"    ttsim:   {ttsim_output[b, c, h, w]:.10f}")
            print(f"    Diff:    {diff[b, c, h, w]:.10e}")
else:
    print("  [WARN]  [SKIP] Cannot compare - ttsim output.data is None")


# ============================================================
# TEST 4: Test with 3D inputs (squeeze test)
# ============================================================
print("\n" + "=" * 80)
print("TEST 4: 3D Input Test (no batch dimension)")
print("-" * 80)

# Create 3D inputs [C, H, W]
h_3d_np = h_np[0]  # Take first batch element
x_3d_np = x_np[0]

print(f"  3D Input shape: [{out_channels}, {height}, {width}]")

# PyTorch
h_3d_torch = torch.from_numpy(h_3d_np)
x_3d_torch = torch.from_numpy(x_3d_np)

with torch.no_grad():
    output_3d_pytorch = gru_pytorch(h_3d_torch, x_3d_torch)

output_3d_pytorch_np = output_3d_pytorch.numpy()
print(f"  PyTorch 3D output shape: {output_3d_pytorch.shape}")

# ttsim
h_3d_simtensor = F._from_data("h_3d_input", data=h_3d_np, is_const=False)
x_3d_simtensor = F._from_data("x_3d_input", data=x_3d_np, is_const=False)

output_3d_ttsim = gru_ttsim(h_3d_simtensor, x_3d_simtensor)
print(f"  ttsim 3D output shape: {output_3d_ttsim.shape}")

if output_3d_ttsim.data is not None:
    diff_3d = np.abs(output_3d_pytorch_np - output_3d_ttsim.data)
    matches_3d = np.allclose(
        output_3d_pytorch_np, output_3d_ttsim.data, atol=1e-5, rtol=1e-5
    )

    if matches_3d:
        print(f"  [PASS] [PASS] 3D outputs match")
    else:
        print(f"  [FAIL] [FAIL] 3D outputs differ")
        print(f"    Max diff: {diff_3d.max():.10e}")
else:
    print("  [WARN]  [SKIP] Cannot compare - ttsim 3D output.data is None")


print()
print("=" * 80)
print("Test Complete!")
print("=" * 80)
