#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for SelfAttention module.
Tests shape inference and numerical equivalence between PyTorch and TTSIM.
"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys

# Add DiffusionDrive to path for navsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
# Add TT-Polaris to path for ttsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone_ttsim import (
    SelfAttention as SelfAttention_TTSIM,
)

import ttsim.front.functional.op as F


class SelfAttention_PyTorch(nn.Module):
    """PyTorch reference implementation."""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_embd = n_embd
        self.n_head = n_head

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        b, t, c = x.size()

        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))

        return y


def inject_weights(ttsim_module, pytorch_module):
    """Inject PyTorch weights into TTSIM module."""
    # Key
    ttsim_module.key.params[0][1].data = pytorch_module.key.weight.data.T.numpy()
    ttsim_module.key_bias.params[0][1].data = pytorch_module.key.bias.data.numpy()

    # Query
    ttsim_module.query.params[0][1].data = pytorch_module.query.weight.data.T.numpy()
    ttsim_module.query_bias.params[0][1].data = pytorch_module.query.bias.data.numpy()

    # Value
    ttsim_module.value.params[0][1].data = pytorch_module.value.weight.data.T.numpy()
    ttsim_module.value_bias.params[0][1].data = pytorch_module.value.bias.data.numpy()

    # Projection
    ttsim_module.proj.params[0][1].data = pytorch_module.proj.weight.data.T.numpy()
    ttsim_module.proj_bias.params[0][1].data = pytorch_module.proj.bias.data.numpy()


def main():
    print("=" * 70)
    print("TEST: SelfAttention - Multi-head Self-Attention")
    print("=" * 70)

    # Configuration
    n_embd = 128
    n_head = 4
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    batch_size = 2
    seq_len = 64

    # Create test input
    input_data = np.random.randn(batch_size, seq_len, n_embd).astype(np.float32)

    # PyTorch model
    print("\n--- PyTorch SelfAttention ---")
    model_pt = SelfAttention_PyTorch(n_embd, n_head, attn_pdrop, resid_pdrop)
    model_pt.eval()

    with torch.no_grad():
        x_pt = torch.from_numpy(input_data)
        output_pt = model_pt(x_pt)

    print(f"Input shape: {x_pt.shape}")
    print(f"Output shape: {output_pt.shape}")
    print(
        f"Output stats: min={output_pt.numpy().min():.6f}, max={output_pt.numpy().max():.6f}, mean={output_pt.numpy().mean():.6f}"
    )

    # TTSIM model
    print("\n--- TTSIM SelfAttention ---")
    model_ttsim = SelfAttention_TTSIM(
        "test_attn", n_embd, n_head, attn_pdrop, resid_pdrop
    )

    # Inject weights
    print("Injecting weights...")
    inject_weights(model_ttsim, model_pt)

    # Forward pass
    x_ttsim = F._from_data("input", input_data)
    x_ttsim.link_module = model_ttsim
    output_ttsim = model_ttsim(x_ttsim)

    print(f"Input shape: {x_ttsim.shape}")
    print(f"Output shape: {output_ttsim.shape}")

    if output_ttsim.data is not None:
        print(
            f"Output stats: min={output_ttsim.data.min():.6f}, max={output_ttsim.data.max():.6f}, mean={output_ttsim.data.mean():.6f}"
        )
    else:
        print("FAIL: FAIL: No data computed")
        return

    # Numerical comparison
    print("\n--- Numerical Comparison ---")
    diff = np.abs(output_pt.numpy() - output_ttsim.data)
    print(f"Max difference: {diff.max():.10f}")
    print(f"Mean difference: {diff.mean():.10f}")
    print(f"Std difference: {diff.std():.10f}")

    # Validation thresholds
    if diff.max() < 1e-4:
        print("PASS: PASS: SelfAttention matches PyTorch (max diff < 1e-4)")
    elif diff.max() < 1e-3:
        print("WARN: WARN: SelfAttention close to PyTorch (max diff < 1e-3)")
    else:
        print(f"FAIL: FAIL: Large differences detected (max diff = {diff.max():.6f})")


if __name__ == "__main__":
    main()
