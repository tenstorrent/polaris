#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for Block module (Transformer Block).
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

from navsim.agents.diffusiondrive.transfuser_backbone_ttsim import Block as Block_TTSIM

import ttsim.front.functional.op as F


class SelfAttention_PyTorch(nn.Module):
    """PyTorch SelfAttention for Block."""

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


class Block_PyTorch(nn.Module):
    """PyTorch Block (Transformer Block)."""

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention_PyTorch(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def inject_weights(ttsim_block, pytorch_block):
    """Inject PyTorch weights into TTSIM Block."""
    # LayerNorm 1
    ttsim_block.ln1.params[0][1].data = pytorch_block.ln1.weight.data.numpy()
    ttsim_block.ln1.params[1][1].data = pytorch_block.ln1.bias.data.numpy()

    # LayerNorm 2
    ttsim_block.ln2.params[0][1].data = pytorch_block.ln2.weight.data.numpy()
    ttsim_block.ln2.params[1][1].data = pytorch_block.ln2.bias.data.numpy()

    # Attention
    ttsim_block.attn.key.params[0][
        1
    ].data = pytorch_block.attn.key.weight.data.T.numpy()
    ttsim_block.attn.key_bias.params[0][
        1
    ].data = pytorch_block.attn.key.bias.data.numpy()
    ttsim_block.attn.query.params[0][
        1
    ].data = pytorch_block.attn.query.weight.data.T.numpy()
    ttsim_block.attn.query_bias.params[0][
        1
    ].data = pytorch_block.attn.query.bias.data.numpy()
    ttsim_block.attn.value.params[0][
        1
    ].data = pytorch_block.attn.value.weight.data.T.numpy()
    ttsim_block.attn.value_bias.params[0][
        1
    ].data = pytorch_block.attn.value.bias.data.numpy()
    ttsim_block.attn.proj.params[0][
        1
    ].data = pytorch_block.attn.proj.weight.data.T.numpy()
    ttsim_block.attn.proj_bias.params[0][
        1
    ].data = pytorch_block.attn.proj.bias.data.numpy()

    # MLP
    ttsim_block.mlp_fc1.params[0][1].data = pytorch_block.mlp[0].weight.data.T.numpy()
    ttsim_block.mlp_fc1_bias.params[0][1].data = pytorch_block.mlp[0].bias.data.numpy()
    ttsim_block.mlp_fc2.params[0][1].data = pytorch_block.mlp[2].weight.data.T.numpy()
    ttsim_block.mlp_fc2_bias.params[0][1].data = pytorch_block.mlp[2].bias.data.numpy()


def main():
    print("=" * 70)
    print("TEST: Block - Transformer Block (Attention + MLP)")
    print("=" * 70)

    # Configuration
    n_embd = 128
    n_head = 4
    block_exp = 4
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    batch_size = 2
    seq_len = 64

    # Create test input
    input_data = np.random.randn(batch_size, seq_len, n_embd).astype(np.float32)

    # PyTorch model
    print("\n--- PyTorch Block ---")
    model_pt = Block_PyTorch(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop)
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
    print("\n--- TTSIM Block ---")
    model_ttsim = Block_TTSIM(
        "test_block", n_embd, n_head, block_exp, attn_pdrop, resid_pdrop
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
        print("PASS: PASS: Block matches PyTorch (max diff < 1e-4)")
    elif diff.max() < 1e-3:
        print("WARN: WARN: Block close to PyTorch (max diff < 1e-3)")
    else:
        print(f"FAIL: FAIL: Large differences detected (max diff = {diff.max():.6f})")


if __name__ == "__main__":
    main()
