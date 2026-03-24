#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
MotionMLP module validation: ttsim vs PyTorch comparison
"""

import os
import sys

polaris_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, polaris_path)

import numpy as np
import torch
import torch.nn as nn

# Fix for OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import ttsim.front.functional.op as F
from ttsim.ops import SimTensor

# Import ttsim MotionMLP
from workloads.MapTracker.plugin.models.utils.query_update import (
    MotionMLP as MotionMLPTtsim,
)


# PyTorch implementation (to avoid maptracker dependencies)
class EmbedderPyTorch(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs, include_input=True):
        super().__init__()
        self.input_dim = input_dim
        self.max_freq_log2 = max_freq_log2
        self.N_freqs = N_freqs
        self.include_input = include_input

        # Compute frequency bands
        freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, N_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        # x: [batch, input_dim]
        # Original MapTracker order: for each freq: sin(x*freq), cos(x*freq)
        # This processes all input dimensions together

        embed_list = []

        # Include raw input if specified
        if self.include_input:
            embed_list.append(x)

        # For each frequency: append sin(x*freq), then cos(x*freq)
        for i in range(self.N_freqs):
            freq = self.freq_bands[i]
            x_scaled = x * freq  # [batch, input_dim]

            # Apply sin
            sin_out = torch.sin(x_scaled)  # [batch, input_dim]
            embed_list.append(sin_out)

            # Apply cos
            cos_out = torch.cos(x_scaled)  # [batch, input_dim]
            embed_list.append(cos_out)

        # Concatenate all: [batch, input_dim + input_dim*N_freqs*2]
        out = torch.cat(embed_list, dim=-1)
        return out


class MotionMLPPyTorch(nn.Module):
    def __init__(self, c_dim, f_dim, identity=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.identity = identity

        # Embedder for positional encoding
        self.embedder = EmbedderPyTorch(
            input_dim=c_dim, max_freq_log2=9, N_freqs=10, include_input=True
        )
        # Match ttsim Embedder output: input_dim + input_dim * N_freqs * 2
        embed_dim = c_dim + c_dim * 10 * 2

        # MLP: Linear -> LayerNorm -> ReLU -> Linear
        self.fc = nn.Sequential(
            nn.Linear(f_dim + embed_dim, f_dim * 2),
            nn.LayerNorm(f_dim * 2),
            nn.ReLU(),
            nn.Linear(f_dim * 2, f_dim),
        )

    def forward(self, input_feature, pose_feature):
        # Embed pose information
        pose_embed = self.embedder(pose_feature)

        # Concatenate input and pose embedding
        x = torch.cat([input_feature, pose_embed], dim=-1)

        # Apply MLP
        output = self.fc(x)

        # Add residual connection if identity=True
        if self.identity:
            output = output + input_feature

        return output


print("=" * 70)
print("MotionMLP Comparison: PyTorch vs ttsim")
print("=" * 70)

# ----------------------------------------------------------------------
# Test configuration
# ----------------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

batch_size = 2
c_dim = 7  # Pose info dimension
f_dim = 64  # Feature dimension (smaller for testing)
identity = True

print(f"\nConfiguration:")
print(f"  c_dim (pose dimension): {c_dim}")
print(f"  f_dim (feature dimension): {f_dim}")
print(f"  identity (residual): {identity}")
print(f"  Input shape: [{batch_size}, {f_dim}]")
print(f"  Pose shape: [{batch_size}, {c_dim}]")

# Create test inputs
input_np = np.random.randn(batch_size, f_dim).astype(np.float32)
pose_np = np.random.randn(batch_size, c_dim).astype(np.float32)

print(f"\nInput stats:")
print(
    f"  Features - Min: {input_np.min():.6f}, Max: {input_np.max():.6f}, Mean: {input_np.mean():.6f}"
)
print(
    f"  Pose     - Min: {pose_np.min():.6f}, Max: {pose_np.max():.6f}, Mean: {pose_np.mean():.6f}"
)

# Common tolerances
atol = 1e-6
rtol = 1e-5

# ============================================================
# TEST 1: PyTorch MotionMLP
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: PyTorch MotionMLP")
print("-" * 70)

motionmlp_pytorch = MotionMLPPyTorch(c_dim=c_dim, f_dim=f_dim, identity=identity)
motionmlp_pytorch.eval()

with torch.no_grad():
    input_torch = torch.from_numpy(input_np)
    pose_torch = torch.from_numpy(pose_np)
    output_pytorch = motionmlp_pytorch(input_torch, pose_torch)
    output_pytorch_np = output_pytorch.cpu().numpy()

print(f"  Input shape: {input_torch.shape}")
print(f"  Pose shape: {pose_torch.shape}")
print(f"  Output shape: {output_pytorch.shape}")
print(f"  Output values:\n{output_pytorch_np}")
print(f"  Output stats:")
print(f"    Min:  {output_pytorch_np.min():.6f}")
print(f"    Max:  {output_pytorch_np.max():.6f}")
print(f"    Mean: {output_pytorch_np.mean():.6f}")
print(f"    Std:  {output_pytorch_np.std():.6f}")

# ============================================================
# TEST 2: ttsim MotionMLP
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: ttsim MotionMLP")
print("-" * 70)

motionmlp_ttsim = MotionMLPTtsim(
    name="motionmlp_test", c_dim=c_dim, f_dim=f_dim, identity=identity
)

# Inject PyTorch weights into ttsim operators
# FC1: Linear layer (weight transposed, bias is separate)
motionmlp_ttsim.fc1.params[0][1].data = motionmlp_pytorch.fc[0].weight.data.numpy().T
motionmlp_ttsim.fc1_bias.data = motionmlp_pytorch.fc[0].bias.data.numpy()

# LN1: LayerNorm (weight, bias)
motionmlp_ttsim.ln1.params[0][1].data = motionmlp_pytorch.fc[1].weight.data.numpy()
motionmlp_ttsim.ln1.params[1][1].data = motionmlp_pytorch.fc[1].bias.data.numpy()

# FC2: Linear layer (weight transposed, bias is separate)
motionmlp_ttsim.fc2.params[0][1].data = motionmlp_pytorch.fc[3].weight.data.numpy().T
motionmlp_ttsim.fc2_bias.data = motionmlp_pytorch.fc[3].bias.data.numpy()

# Run forward pass with ttsim
input_simtensor = F._from_data("input", data=input_np, is_const=False)
pose_simtensor = F._from_data("pose_info", data=pose_np, is_const=False)
output_ttsim = motionmlp_ttsim(input_simtensor, pose_simtensor)

print(f"  Input shape: {input_simtensor.shape}")
print(f"  Pose shape: {pose_simtensor.shape}")
print(f"  Output shape: {output_ttsim.shape}")
print(f"  Output .data is None? {output_ttsim.data is None}")
print(f"  Output values:\n{output_ttsim.data}")

if output_ttsim.data is not None:
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
print("\n" + "=" * 70)
print("TEST 3: Numerical Comparison")
print("-" * 70)

if output_ttsim.data is not None:
    ttsim_output = output_ttsim.data

    # Compute difference
    diff = np.abs(output_pytorch_np - ttsim_output)

    print(f"  Absolute difference:")
    print(f"    Min:  {diff.min():.10e}")
    print(f"    Max:  {diff.max():.10e}")
    print(f"    Mean: {diff.mean():.10e}")
    print(f"    Std:  {diff.std():.10e}")

    # Check if outputs match
    is_close = np.allclose(ttsim_output, output_pytorch_np, atol=atol, rtol=rtol)

    if is_close:
        print(
            f"\n  [PASS] [PASS] Outputs match within tolerance (atol={atol}, rtol={rtol})"
        )
    else:
        print(f"\n  [FAIL] [FAIL] Outputs differ beyond tolerance")
        print(f"    Expected max diff < {atol}, got {diff.max():.10e}")
else:
    print("  [WARNING]  [SKIP] Cannot compare - ttsim output.data is None")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
