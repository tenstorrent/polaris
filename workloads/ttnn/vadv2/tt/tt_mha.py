#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
import ttsim.front.functional.sim_nn as SimNN
import math
from workloads.ttnn.vadv2.tt.tt_utils import DictAsAttr


class TtMultiheadAttention(SimNN.Module):
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        init_cfg=None,
        batch_first=False,
    ):
        super().__init__()
        self.name = "TtMultiheadAttention"
        self.params = params
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.device = device
        self.batch_first = batch_first
        params = DictAsAttr(params, depth=1)
        self.attn_in_proj__weight = params.in_proj.weight
        self.attn_in_proj__bias = params.in_proj.bias
        self.attn_in_proj__weight_permute = ttnn.permute(self.attn_in_proj__weight, (1, 0))
        self.attn_in_proj__bias_squeeze = ttnn.squeeze(self.attn_in_proj__bias, 0)
        self.attn_out_proj_weight = params.out_proj.weight
        self.attn_out_proj_bias = params.out_proj.bias

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        batch_first=False,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos

        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if batch_first:
            query = ttnn.permute(query, (1, 0))
            key = ttnn.permute(key, (1, 0))
            value = ttnn.permute(value, (1, 0))

        in_proj_bias = self.attn_in_proj__bias_squeeze

        in_proj_weight = self.attn_in_proj__weight_permute

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        in_proj_weight.set_module(self)
        q_weight = in_proj_weight[: self.embed_dims, :]  # Query weights
        k_weight = in_proj_weight[self.embed_dims : 2 * self.embed_dims, :]  # Key weights
        v_weight = in_proj_weight[2 * self.embed_dims :, :]  # Value weights

        in_proj_bias = ttnn.unsqueeze(in_proj_bias, -1)
        in_proj_bias.set_module(self)
        q_bias = in_proj_bias[: self.embed_dims, :].squeeze(-1)  # Query biases
        k_bias = in_proj_bias[self.embed_dims : 2 * self.embed_dims, :].squeeze(-1)  # Key biases
        v_bias = in_proj_bias[2 * self.embed_dims :, :].squeeze(-1)  # Value biases

        q_batch_size, q_sequence_size, q_hidden_size = query.shape
        q_head_size = q_hidden_size // self.num_heads

        k_batch_size, k_sequence_size, k_hidden_size = key.shape
        # k_head_size = k_hidden_size // self.num_heads

        v_batch_size, v_sequence_size, v_hidden_size = value.shape
        # v_head_size = v_hidden_size // self.num_heads

        q_weight = ttnn.Tensor(shape=q_weight.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, data=q_weight.data)
        k_weight = ttnn.Tensor(shape=k_weight.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, data=k_weight.data)
        v_weight = ttnn.Tensor(shape=v_weight.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, data=v_weight.data)
        q_weight = ttnn.permute(q_weight, [1, 0])
        k_weight = ttnn.permute(k_weight, [1, 0])
        v_weight = ttnn.permute(v_weight, [1, 0])

        q_bias = ttnn.Tensor(shape=q_bias.shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, data=q_bias.data)
        query = ttnn.linear(query, q_weight, bias=q_bias)

        k_bias = ttnn.Tensor(shape=k_bias.shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, data=k_bias.data)
        key = ttnn.linear(key, k_weight, bias=k_bias)

        v_bias = ttnn.Tensor(shape=v_bias.shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, data=v_bias.data)
        value = ttnn.linear(value, v_weight, bias=v_bias)

        query = ttnn.reshape(query, (tgt_len, bsz * self.num_heads, q_head_size))
        query = ttnn.permute(query, (1, 0, 2))

        key = ttnn.reshape(key, (k_batch_size, bsz * self.num_heads, q_head_size))
        key = ttnn.permute(key, (1, 0, 2))

        value = ttnn.reshape(value, (v_batch_size, bsz * self.num_heads, q_head_size))
        value = ttnn.permute(value, (1, 0, 2))

        src_len = key.shape[1]

        B, Nt, E = query.shape
        q_scaled = query * math.sqrt(1.0 / float(E))
        key_transposed = ttnn.permute(key, (0, 2, 1))

        if attn_mask is not None:
            attn_output_weights = ttnn.matmul(q_scaled, key_transposed)
            attn_output_weights = attn_output_weights + attn_mask
        else:
            attn_output_weights = ttnn.matmul(q_scaled, key_transposed)

        attn_output_weights = ttnn.softmax(attn_output_weights, dim=-1)

        attn_output = ttnn.matmul(attn_output_weights, value)

        attn_output = ttnn.permute(attn_output, (1, 0, 2))
        attn_output = ttnn.reshape(attn_output, (tgt_len * bsz, embed_dim))

        attn_output = ttnn.linear(attn_output, self.attn_out_proj_weight, bias=self.attn_out_proj_bias)
        attn_output = ttnn.reshape(attn_output, (tgt_len, bsz, attn_output.shape[1]))
        attn_output_weights = ttnn.reshape(attn_output_weights, (bsz, self.num_heads, tgt_len, src_len))
        attn_output_weights = ttnn.to_layout(attn_output_weights, ttnn.ROW_MAJOR_LAYOUT)
        attn_output_weights = ttnn.mean(attn_output_weights, dim=1)
        identity = ttnn.to_layout(identity, ttnn.TILE_LAYOUT)

        return attn_output + identity
