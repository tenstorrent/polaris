#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for TransformerDecoderWithAttention module.
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
    TransformerDecoderWithAttention as Decoder_TTSIM,
)

import ttsim.front.functional.op as F


class MultiheadAttentionWithAttention_PyTorch(nn.Module):
    """PyTorch MultiheadAttention."""

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

        tmp, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(tmp))

        tmp, attention = self.multihead_attn(x, memory, memory)
        x = self.norm2(x + self.dropout2(tmp))

        tmp = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(tmp))

        return x, attention


class TransformerDecoderWithAttention_PyTorch(nn.Module):
    """PyTorch transformer decoder."""

    def __init__(self, layer_template, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayerWithAttention_PyTorch(
                    layer_template.d_model,
                    layer_template.self_attn.n_head,
                    layer_template.dim_feedforward,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(layer_template.d_model) if norm is not None else None

    def forward(self, queries, memory):
        output = queries
        attentions = []

        for layer in self.layers:
            output, attention = layer(output, memory)
            attentions.append(attention)

        if self.norm is not None:
            output = self.norm(output)

        # Average over layers
        stacked_attentions = torch.stack(attentions, dim=0)
        avg_attention = stacked_attentions.mean(dim=0)

        return output, avg_attention


def inject_decoder_weights(ttsim_decoder, pytorch_decoder):
    """Inject PyTorch decoder weights into TTSIM decoder."""
    for i, (ttsim_layer, pt_layer) in enumerate(
        zip(ttsim_decoder.layers, pytorch_decoder.layers)
    ):
        # Self-attention
        ttsim_layer.self_attn.key.params[0][
            1
        ].data = pt_layer.self_attn.key.weight.data.T.numpy()
        ttsim_layer.self_attn.key_bias.params[0][
            1
        ].data = pt_layer.self_attn.key.bias.data.numpy()
        ttsim_layer.self_attn.query.params[0][
            1
        ].data = pt_layer.self_attn.query.weight.data.T.numpy()
        ttsim_layer.self_attn.query_bias.params[0][
            1
        ].data = pt_layer.self_attn.query.bias.data.numpy()
        ttsim_layer.self_attn.value.params[0][
            1
        ].data = pt_layer.self_attn.value.weight.data.T.numpy()
        ttsim_layer.self_attn.value_bias.params[0][
            1
        ].data = pt_layer.self_attn.value.bias.data.numpy()
        ttsim_layer.self_attn.proj.params[0][
            1
        ].data = pt_layer.self_attn.proj.weight.data.T.numpy()
        ttsim_layer.self_attn.proj_bias.params[0][
            1
        ].data = pt_layer.self_attn.proj.bias.data.numpy()

        # Cross-attention
        ttsim_layer.multihead_attn.key.params[0][
            1
        ].data = pt_layer.multihead_attn.key.weight.data.T.numpy()
        ttsim_layer.multihead_attn.key_bias.params[0][
            1
        ].data = pt_layer.multihead_attn.key.bias.data.numpy()
        ttsim_layer.multihead_attn.query.params[0][
            1
        ].data = pt_layer.multihead_attn.query.weight.data.T.numpy()
        ttsim_layer.multihead_attn.query_bias.params[0][
            1
        ].data = pt_layer.multihead_attn.query.bias.data.numpy()
        ttsim_layer.multihead_attn.value.params[0][
            1
        ].data = pt_layer.multihead_attn.value.weight.data.T.numpy()
        ttsim_layer.multihead_attn.value_bias.params[0][
            1
        ].data = pt_layer.multihead_attn.value.bias.data.numpy()
        ttsim_layer.multihead_attn.proj.params[0][
            1
        ].data = pt_layer.multihead_attn.proj.weight.data.T.numpy()
        ttsim_layer.multihead_attn.proj_bias.params[0][
            1
        ].data = pt_layer.multihead_attn.proj.bias.data.numpy()

        # Feedforward
        ttsim_layer.linear1.params[0][1].data = pt_layer.linear1.weight.data.T.numpy()
        ttsim_layer.linear1_bias.params[0][1].data = pt_layer.linear1.bias.data.numpy()
        ttsim_layer.linear2.params[0][1].data = pt_layer.linear2.weight.data.T.numpy()
        ttsim_layer.linear2_bias.params[0][1].data = pt_layer.linear2.bias.data.numpy()

        # LayerNorms
        ttsim_layer.norm1.params[0][1].data = pt_layer.norm1.weight.data.numpy()
        ttsim_layer.norm1.params[1][1].data = pt_layer.norm1.bias.data.numpy()
        ttsim_layer.norm2.params[0][1].data = pt_layer.norm2.weight.data.numpy()
        ttsim_layer.norm2.params[1][1].data = pt_layer.norm2.bias.data.numpy()
        ttsim_layer.norm3.params[0][1].data = pt_layer.norm3.weight.data.numpy()
        ttsim_layer.norm3.params[1][1].data = pt_layer.norm3.bias.data.numpy()

    # Final norm
    if pytorch_decoder.norm is not None:
        ttsim_decoder.norm.params[0][1].data = pytorch_decoder.norm.weight.data.numpy()
        ttsim_decoder.norm.params[1][1].data = pytorch_decoder.norm.bias.data.numpy()


def main():
    print("=" * 70)
    print("TEST: TransformerDecoderWithAttention")
    print("=" * 70)

    # Configuration
    d_model = 256
    nhead = 8
    dim_feedforward = 1024
    dropout = 0.1
    num_layers = 3
    batch_size = 2
    seq_len_queries = 32
    seq_len_memory = 64

    # Create test inputs
    queries_data = np.random.randn(batch_size, seq_len_queries, d_model).astype(
        np.float32
    )
    memory_data = np.random.randn(batch_size, seq_len_memory, d_model).astype(
        np.float32
    )

    # PyTorch model
    print("\n--- PyTorch TransformerDecoderWithAttention ---")
    layer_template = TransformerDecoderLayerWithAttention_PyTorch(
        d_model, nhead, dim_feedforward, dropout
    )
    model_pt = TransformerDecoderWithAttention_PyTorch(
        layer_template, num_layers, norm=True
    )
    model_pt.eval()

    with torch.no_grad():
        queries_pt = torch.from_numpy(queries_data)
        memory_pt = torch.from_numpy(memory_data)
        output_pt, attention_pt = model_pt(queries_pt, memory_pt)

    print(f"Queries shape: {queries_pt.shape}")
    print(f"Memory shape: {memory_pt.shape}")
    print(f"Output shape: {output_pt.shape}")
    print(f"Attention shape: {attention_pt.shape}")
    print(
        f"Output stats: min={output_pt.numpy().min():.6f}, max={output_pt.numpy().max():.6f}"
    )

    # TTSIM model
    print("\n--- TTSIM TransformerDecoderWithAttention ---")
    layer_template_ttsim = DecoderLayer_TTSIM(
        "template", d_model, nhead, dim_feedforward, dropout
    )
    model_ttsim = Decoder_TTSIM(
        "test_decoder", layer_template_ttsim, num_layers, norm=True
    )

    # Inject weights
    print("Injecting weights...")
    inject_decoder_weights(model_ttsim, model_pt)

    # Forward pass
    queries_ttsim = F._from_data("queries", queries_data)
    memory_ttsim = F._from_data("memory", memory_data)
    queries_ttsim.link_module = model_ttsim
    memory_ttsim.link_module = model_ttsim

    output_ttsim, attention_ttsim = model_ttsim(queries_ttsim, memory_ttsim)

    print(f"Queries shape: {queries_ttsim.shape}")
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
