#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for MultiheadAttentionWithAttention module.
Tests shape inference and numerical equivalence between PyTorch and TTSIM.
"""

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

from navsim.agents.diffusiondrive.transfuser_backbone_ttsim import (
    MultiheadAttentionWithAttention as MHA_TTSIM,
)

import ttsim.front.functional.op as F
import warnings

warnings.filterwarnings("ignore")


class MultiheadAttentionWithAttention_PyTorch(nn.Module):
    """PyTorch reference implementation."""

    def __init__(self, n_embd, n_head, pdrop):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, q_in, k_in, v_in):
        b, t, c = q_in.shape
        t_mem = k_in.shape[1]

        # Calculate query, key, values
        q = self.query(q_in)
        k = self.key(k_in)
        v = self.value(v_in)

        # Reshape for multi-head
        head_size = c // self.n_head
        q = q.view(b, t, self.n_head, head_size).transpose(1, 2)
        k = k.view(b, t_mem, self.n_head, head_size).transpose(1, 2)
        v = v.view(b, t_mem, self.n_head, head_size).transpose(1, 2)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(head_size))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))

        # Average attention over heads
        attention = att.mean(dim=1)

        return y, attention


def inject_weights(ttsim_module, pytorch_module):
    """Inject PyTorch weights into TTSIM module."""
    # Key, Query, Value
    ttsim_module.key.params[0][1].data = pytorch_module.key.weight.data.T.numpy()
    ttsim_module.key_bias.params[0][1].data = pytorch_module.key.bias.data.numpy()
    ttsim_module.query.params[0][1].data = pytorch_module.query.weight.data.T.numpy()
    ttsim_module.query_bias.params[0][1].data = pytorch_module.query.bias.data.numpy()
    ttsim_module.value.params[0][1].data = pytorch_module.value.weight.data.T.numpy()
    ttsim_module.value_bias.params[0][1].data = pytorch_module.value.bias.data.numpy()

    # Projection
    ttsim_module.proj.params[0][1].data = pytorch_module.proj.weight.data.T.numpy()
    ttsim_module.proj_bias.params[0][1].data = pytorch_module.proj.bias.data.numpy()


def main():
    print("=" * 70)
    print("TEST: MultiheadAttentionWithAttention")
    print("=" * 70)

    # Configuration
    n_embd = 256
    n_head = 8
    pdrop = 0.1
    batch_size = 2
    seq_len_q = 32
    seq_len_kv = 64

    # Create test inputs
    query_data = np.random.randn(batch_size, seq_len_q, n_embd).astype(np.float32)
    key_data = np.random.randn(batch_size, seq_len_kv, n_embd).astype(np.float32)
    value_data = np.random.randn(batch_size, seq_len_kv, n_embd).astype(np.float32)

    # PyTorch model
    print("\n--- PyTorch MultiheadAttentionWithAttention ---")
    model_pt = MultiheadAttentionWithAttention_PyTorch(n_embd, n_head, pdrop)
    model_pt.eval()

    with torch.no_grad():
        q_pt = torch.from_numpy(query_data)
        k_pt = torch.from_numpy(key_data)
        v_pt = torch.from_numpy(value_data)
        output_pt, attention_pt = model_pt(q_pt, k_pt, v_pt)

    print(f"Query shape: {q_pt.shape}")
    print(f"Key shape: {k_pt.shape}")
    print(f"Value shape: {v_pt.shape}")
    print(f"Output shape: {output_pt.shape}")
    print(f"Attention shape: {attention_pt.shape}")
    print(
        f"Output stats: min={output_pt.numpy().min():.6f}, max={output_pt.numpy().max():.6f}"
    )

    # TTSIM model
    print("\n--- TTSIM MultiheadAttentionWithAttention ---")
    model_ttsim = MHA_TTSIM("test_mha", n_embd, n_head, pdrop)

    # Inject weights
    print("Injecting weights...")
    inject_weights(model_ttsim, model_pt)

    # Forward pass
    q_ttsim = F._from_data("query", query_data)
    k_ttsim = F._from_data("key", key_data)
    v_ttsim = F._from_data("value", value_data)
    q_ttsim.link_module = model_ttsim
    k_ttsim.link_module = model_ttsim
    v_ttsim.link_module = model_ttsim

    output_ttsim, attention_ttsim = model_ttsim(q_ttsim, k_ttsim, v_ttsim)

    print(f"Query shape: {q_ttsim.shape}")
    print(f"Key shape: {k_ttsim.shape}")
    print(f"Value shape: {v_ttsim.shape}")
    print(f"Output shape: {output_ttsim.shape}")
    print(f"Attention shape: {attention_ttsim.shape}")

    if output_ttsim.data is not None and attention_ttsim.data is not None:
        print(
            f"Output stats: min={output_ttsim.data.min():.6f}, max={output_ttsim.data.max():.6f}"
        )
    else:
        print("FAIL: FAIL: No data computed")
        return

    # Numerical comparison
    print("\n--- Numerical Comparison ---")
    output_diff = np.abs(output_pt.numpy() - output_ttsim.data)
    attention_diff = np.abs(attention_pt.numpy() - attention_ttsim.data)

    print(
        f"Output - Max diff: {output_diff.max():.10f}, Mean diff: {output_diff.mean():.10f}"
    )
    print(
        f"Attention - Max diff: {attention_diff.max():.10f}, Mean diff: {attention_diff.mean():.10f}"
    )

    # Validation using np.allclose
    atol = 1e-4
    rtol = 1e-4
    output_close = np.allclose(
        output_pt.numpy(), output_ttsim.data, atol=atol, rtol=rtol
    )
    attention_close = np.allclose(
        attention_pt.numpy(), attention_ttsim.data, atol=atol, rtol=rtol
    )

    print(f"\n(atol={atol}, rtol={rtol}):")
    print(f"  Output:    {'PASS' if output_close else 'FAIL'}")
    print(f"  Attention: {'PASS' if attention_close else 'FAIL'}")

    print("\n" + "=" * 70)
    if output_close and attention_close:
        print(f"OVERALL: PASS: PASS - (atol={atol}, rtol={rtol})")
    else:
        print(f"OVERALL: FAIL: FAIL - (atol={atol}, rtol={rtol}) returned False")
        max_diff = max(output_diff.max(), attention_diff.max())
        print(f"  Max abs diff: {max_diff:.10f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
