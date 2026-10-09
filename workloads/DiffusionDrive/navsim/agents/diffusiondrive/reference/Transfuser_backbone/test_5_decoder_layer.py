#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for TransformerDecoderLayerWithAttention module.
Tests shape inference and numerical equivalence between PyTorch and TTSIM.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add DiffusionDrive to path for navsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
# Add TT-Polaris to path for ttsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.diffusiondrive.transfuser_backbone_ttsim import (
    TransformerDecoderLayerWithAttention as DecoderLayer_TTSIM,
)

import ttsim.front.functional.op as F


class MultiheadAttentionWithAttention_PyTorch(nn.Module):
    """PyTorch MultiheadAttention with attention returns."""

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

        q = self.query(q_in)
        k = self.key(k_in)
        v = self.value(v_in)

        head_size = c // self.n_head
        q = q.view(b, t, self.n_head, head_size).transpose(1, 2)
        k = k.view(b, t_mem, self.n_head, head_size).transpose(1, 2)
        v = v.view(b, t_mem, self.n_head, head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(head_size))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))

        attention = att.mean(dim=1)

        return y, attention


class TransformerDecoderLayerWithAttention_PyTorch(nn.Module):
    """PyTorch decoder layer."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward

        self.self_attn = MultiheadAttentionWithAttention_PyTorch(
            d_model, nhead, dropout
        )
        self.multihead_attn = MultiheadAttentionWithAttention_PyTorch(
            d_model, nhead, dropout
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory):
        x = tgt

        # Self-attention
        tmp, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(tmp))

        # Cross-attention
        tmp, attention = self.multihead_attn(x, memory, memory)
        x = self.norm2(x + self.dropout2(tmp))

        # Feedforward
        tmp = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(tmp))

        return x, attention


def inject_weights(ttsim_layer, pytorch_layer):
    """Inject PyTorch weights into TTSIM layer."""
    # Self-attention
    ttsim_layer.self_attn.key.params[0][
        1
    ].data = pytorch_layer.self_attn.key.weight.data.T.numpy()
    ttsim_layer.self_attn.key_bias.params[0][
        1
    ].data = pytorch_layer.self_attn.key.bias.data.numpy()
    ttsim_layer.self_attn.query.params[0][
        1
    ].data = pytorch_layer.self_attn.query.weight.data.T.numpy()
    ttsim_layer.self_attn.query_bias.params[0][
        1
    ].data = pytorch_layer.self_attn.query.bias.data.numpy()
    ttsim_layer.self_attn.value.params[0][
        1
    ].data = pytorch_layer.self_attn.value.weight.data.T.numpy()
    ttsim_layer.self_attn.value_bias.params[0][
        1
    ].data = pytorch_layer.self_attn.value.bias.data.numpy()
    ttsim_layer.self_attn.proj.params[0][
        1
    ].data = pytorch_layer.self_attn.proj.weight.data.T.numpy()
    ttsim_layer.self_attn.proj_bias.params[0][
        1
    ].data = pytorch_layer.self_attn.proj.bias.data.numpy()

    # Cross-attention
    ttsim_layer.multihead_attn.key.params[0][
        1
    ].data = pytorch_layer.multihead_attn.key.weight.data.T.numpy()
    ttsim_layer.multihead_attn.key_bias.params[0][
        1
    ].data = pytorch_layer.multihead_attn.key.bias.data.numpy()
    ttsim_layer.multihead_attn.query.params[0][
        1
    ].data = pytorch_layer.multihead_attn.query.weight.data.T.numpy()
    ttsim_layer.multihead_attn.query_bias.params[0][
        1
    ].data = pytorch_layer.multihead_attn.query.bias.data.numpy()
    ttsim_layer.multihead_attn.value.params[0][
        1
    ].data = pytorch_layer.multihead_attn.value.weight.data.T.numpy()
    ttsim_layer.multihead_attn.value_bias.params[0][
        1
    ].data = pytorch_layer.multihead_attn.value.bias.data.numpy()
    ttsim_layer.multihead_attn.proj.params[0][
        1
    ].data = pytorch_layer.multihead_attn.proj.weight.data.T.numpy()
    ttsim_layer.multihead_attn.proj_bias.params[0][
        1
    ].data = pytorch_layer.multihead_attn.proj.bias.data.numpy()

    # Feedforward
    ttsim_layer.linear1.params[0][1].data = pytorch_layer.linear1.weight.data.T.numpy()
    ttsim_layer.linear1_bias.params[0][1].data = pytorch_layer.linear1.bias.data.numpy()
    ttsim_layer.linear2.params[0][1].data = pytorch_layer.linear2.weight.data.T.numpy()
    ttsim_layer.linear2_bias.params[0][1].data = pytorch_layer.linear2.bias.data.numpy()

    # LayerNorms
    ttsim_layer.norm1.params[0][1].data = pytorch_layer.norm1.weight.data.numpy()
    ttsim_layer.norm1.params[1][1].data = pytorch_layer.norm1.bias.data.numpy()
    ttsim_layer.norm2.params[0][1].data = pytorch_layer.norm2.weight.data.numpy()
    ttsim_layer.norm2.params[1][1].data = pytorch_layer.norm2.bias.data.numpy()
    ttsim_layer.norm3.params[0][1].data = pytorch_layer.norm3.weight.data.numpy()
    ttsim_layer.norm3.params[1][1].data = pytorch_layer.norm3.bias.data.numpy()


def main():
    print("=" * 70)
    print("TEST: TransformerDecoderLayerWithAttention")
    print("=" * 70)

    # Configuration
    d_model = 256
    nhead = 8
    dim_feedforward = 1024
    dropout = 0.1
    batch_size = 2
    seq_len_tgt = 32
    seq_len_memory = 64

    # Create test inputs
    tgt_data = np.random.randn(batch_size, seq_len_tgt, d_model).astype(np.float32)
    memory_data = np.random.randn(batch_size, seq_len_memory, d_model).astype(
        np.float32
    )

    # PyTorch model
    print("\n--- PyTorch TransformerDecoderLayerWithAttention ---")
    model_pt = TransformerDecoderLayerWithAttention_PyTorch(
        d_model, nhead, dim_feedforward, dropout
    )
    model_pt.eval()

    with torch.no_grad():
        tgt_pt = torch.from_numpy(tgt_data)
        memory_pt = torch.from_numpy(memory_data)
        output_pt, attention_pt = model_pt(tgt_pt, memory_pt)

    print(f"Target shape: {tgt_pt.shape}")
    print(f"Memory shape: {memory_pt.shape}")
    print(f"Output shape: {output_pt.shape}")
    print(f"Attention shape: {attention_pt.shape}")
    print(
        f"Output stats: min={output_pt.numpy().min():.6f}, max={output_pt.numpy().max():.6f}"
    )

    # TTSIM model
    print("\n--- TTSIM TransformerDecoderLayerWithAttention ---")
    model_ttsim = DecoderLayer_TTSIM(
        "test_decoder_layer", d_model, nhead, dim_feedforward, dropout
    )

    # Inject weights
    print("Injecting weights...")
    inject_weights(model_ttsim, model_pt)

    # Forward pass
    tgt_ttsim = F._from_data("tgt", tgt_data)
    memory_ttsim = F._from_data("memory", memory_data)
    tgt_ttsim.link_module = model_ttsim
    memory_ttsim.link_module = model_ttsim

    output_ttsim, attention_ttsim = model_ttsim(tgt_ttsim, memory_ttsim)

    print(f"Target shape: {tgt_ttsim.shape}")
    print(f"Memory shape: {memory_ttsim.shape}")
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

    # Validation
    max_diff = max(output_diff.max(), attention_diff.max())
    if max_diff < 1e-4:
        print(
            "PASS: PASS: TransformerDecoderLayerWithAttention matches PyTorch (max diff < 1e-4)"
        )
    elif max_diff < 1e-3:
        print(
            "WARN: WARN: TransformerDecoderLayerWithAttention close to PyTorch (max diff < 1e-3)"
        )
    else:
        print(f"FAIL: FAIL: Large differences detected (max diff = {max_diff:.6f})")


if __name__ == "__main__":
    main()
