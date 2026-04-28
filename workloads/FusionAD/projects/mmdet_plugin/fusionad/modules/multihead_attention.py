#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of MultiheadAttention for FusionAD decoder.

This module implements standard multi-head self-attention and cross-attention
mechanisms used in transformer architectures.

Architecture:
    1. Project Q, K, V with separate linear layers
    2. Split into multiple heads
    3. Scaled dot-product attention per head
    4. Concatenate heads
    5. Output projection
    6. Optional dropout

Formula:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
    where head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
"""

import sys
import os
from loguru import logger
import math

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class MultiheadAttention(SimNN.Module):
    """
    TTSim implementation of Multi-Head Attention.

    Allows the model to jointly attend to information from different representation
    subspaces. Implements the architecture from "Attention Is All You Need".

    Can be used for both self-attention (query=key=value) and cross-attention
    (different query and key/value).

    Args:
        name (str): Module name for TTSim graph
        embed_dims (int): Total dimension of the model (must be divisible by num_heads)
        num_heads (int): Number of parallel attention heads. Default: 8
                        Note: embed_dims will be split across num_heads
                        (i.e., each head will have dimension embed_dims // num_heads)
        attn_drop (float): Dropout probability on attention weights. Default: 0.0
        proj_drop (float): Dropout probability on output projection. Default: 0.0
        batch_first (bool): If True, input/output tensors are (batch, seq, feature).
                           If False, input/output tensors are (seq, batch, feature).
                           Default: False
        bias (bool): If True, adds bias to input/output projection layers. Default: True

    Shape:
        - Input (when batch_first=False):
            - query: (L, N, E) where L is target sequence length, N is batch size, E is embed_dims
            - key: (S, N, E) where S is source sequence length
            - value: (S, N, E)
        - Input (when batch_first=True):
            - query: (N, L, E)
            - key: (N, S, E)
            - value: (N, S, E)
        - Output: Same shape as query input
        - Attention weights (if need_weights=True): (N, L, S) or (N, num_heads, L, S)

    Examples:
        >>> attn = MultiheadAttention('attn', embed_dims=256, num_heads=8)
        >>> query = F._from_data('q', np.random.randn(50, 2, 256))
        >>> output, attn_weights = attn(query, need_weights=False)
    """

    def __init__(self,
                 name,
                 embed_dims=256,
                 num_heads=8,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 batch_first=False,
                 bias=True):
        super().__init__()

        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                           f'but got {embed_dims} and {num_heads}')

        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop
        self.batch_first = batch_first
        self.use_bias = bias
        self.scale = self.head_dim ** -0.5

        # Q, K, V projection layers
        self.q_proj = SimNN.Linear(
            name + '.q_proj',
            in_features=embed_dims,
            out_features=embed_dims,
            bias=bias
        )

        self.k_proj = SimNN.Linear(
            name + '.k_proj',
            in_features=embed_dims,
            out_features=embed_dims,
            bias=bias
        )

        self.v_proj = SimNN.Linear(
            name + '.v_proj',
            in_features=embed_dims,
            out_features=embed_dims,
            bias=bias
        )

        # Output projection
        self.out_proj = SimNN.Linear(
            name + '.out_proj',
            in_features=embed_dims,
            out_features=embed_dims,
            bias=bias
        )

        # Store dropout rates (applied conditionally in forward)
        if attn_drop > 0:
            self.attn_dropout = F.Dropout(name + '.attn_dropout', attn_drop, False)
        else:
            self.attn_dropout = None

        if proj_drop > 0:
            self.proj_dropout = F.Dropout(name + '.proj_dropout', proj_drop, False)
        else:
            self.proj_dropout = None

        # Pre-create all ops used in __call__
        self.query_add_pos = F.Add(f"{name}_query_add_pos")
        self.key_add_pos = F.Add(f"{name}_key_add_pos")
        self.query_transpose_in = F.Transpose(f"{name}_query_transpose_in", perm=[1, 0, 2])
        self.key_transpose_in = F.Transpose(f"{name}_key_transpose_in", perm=[1, 0, 2])
        self.value_transpose_in = F.Transpose(f"{name}_value_transpose_in", perm=[1, 0, 2])

        self.q_reshape = F.Reshape(f"{name}_q_reshape")
        self.q_transpose = F.Transpose(f"{name}_q_transpose", perm=[0, 2, 1, 3])
        self.k_reshape = F.Reshape(f"{name}_k_reshape")
        self.k_transpose = F.Transpose(f"{name}_k_transpose", perm=[0, 2, 1, 3])
        self.v_reshape = F.Reshape(f"{name}_v_reshape")
        self.v_transpose = F.Transpose(f"{name}_v_transpose", perm=[0, 2, 1, 3])

        self.k_t = F.Transpose(f"{name}_k_t", perm=[0, 1, 3, 2])
        self.qk_matmul = F.MatMul(f"{name}_qk")
        self.scale_scores = F.Mul(f"{name}_scale_scores")
        self.add_attn_mask = F.Add(f"{name}_add_attn_mask")
        self.softmax = F.Softmax(f"{name}_softmax", axis=-1)
        self.attn_v_matmul = F.MatMul(f"{name}_attn_v")

        # key_padding_mask ops: (N, S) -> (N, 1, 1, S) * -1e9 -> add to scores
        self.kpm_reshape = F.Reshape(f"{name}_kpm_reshape")
        self.kpm_mul = F.Mul(f"{name}_kpm_mul")
        self.kpm_add = F.Add(f"{name}_kpm_add")
        self.kpm_neg_inf = F._from_data(
            f"{name}_kpm_neg_inf",
            np.array([-1e9], dtype=np.float32),
            is_const=True)

        self.output_transpose = F.Transpose(f"{name}_output_transpose", perm=[0, 2, 1, 3])
        self.output_reshape = F.Reshape(f"{name}_output_reshape")

        self.identity_transpose = F.Transpose(f"{name}_identity_transpose", perm=[1, 0, 2])
        self.residual_add = F.Add(f"{name}_residual")
        self.output_transpose_out = F.Transpose(f"{name}_output_transpose_out", perm=[1, 0, 2])
        self.attn_avg_heads = F.ReduceMean(f"{name}_attn_avg_heads", axes=[1], keep_dims=False)

        # Pre-create scale constant (doesn't depend on runtime dims)
        self.scale_const = F._from_data(
            f"{name}_scale",
            np.array([self.scale], dtype=np.float32),
            is_const=True)

        super().link_op2module()

    def analytical_param_count(self, lvl=0):
        """
        Calculate parameter count for this module.

        Args:
            lvl (int): Verbosity level (0=silent, 1=summary, 2=detailed)

        Returns:
            int: Total parameter count
        """
        indent = "  " * lvl
        total_params = 0

        if lvl >= 2:
            logger.debug(f"{indent}MultiheadAttention '{self.name}':")

        # Q projection
        q_params = self.embed_dims * self.embed_dims
        if self.use_bias:
            q_params += self.embed_dims
        total_params += q_params
        if lvl >= 2:
            logger.debug(f"{indent}  q_proj: {q_params:,}")

        # K projection
        k_params = self.embed_dims * self.embed_dims
        if self.use_bias:
            k_params += self.embed_dims
        total_params += k_params
        if lvl >= 2:
            logger.debug(f"{indent}  k_proj: {k_params:,}")

        # V projection
        v_params = self.embed_dims * self.embed_dims
        if self.use_bias:
            v_params += self.embed_dims
        total_params += v_params
        if lvl >= 2:
            logger.debug(f"{indent}  v_proj: {v_params:,}")

        # Output projection
        out_params = self.embed_dims * self.embed_dims
        if self.use_bias:
            out_params += self.embed_dims
        total_params += out_params
        if lvl >= 2:
            logger.debug(f"{indent}  out_proj: {out_params:,}")

        if lvl >= 1:
            logger.debug(f"{indent}Total MultiheadAttention params: {total_params:,}")

        return total_params

    def __call__(self,
                 query,
                 key=None,
                 value=None,
                 identity=None,
                 query_pos=None,
                 key_pos=None,
                 attn_mask=None,
                 key_padding_mask=None,
                 need_weights=False,
                 average_attn_weights=True,
                 **kwargs):
        """
        Forward pass through multi-head attention.

        This method combines PyTorch nn.MultiheadAttention interface with MMCV extensions
        (identity, query_pos, key_pos) for transformer decoder compatibility.

        Args:
            query: Query embeddings.
                   If batch_first=False: (L, N, E)
                   If batch_first=True: (N, L, E)
            key: Key embeddings (if None, uses query for self-attention).
            value: Value embeddings (if None, uses key).
            identity: Identity tensor for residual connection (optional).
            query_pos: Positional encoding for queries (optional).
            key_pos: Positional encoding for keys (optional).
            attn_mask: Attention mask (optional).
            key_padding_mask: Padding mask for keys (optional).
            need_weights: If True, returns attention weights. Default: False
            average_attn_weights: If True, average weights across heads. Default: True

        Returns:
            output or (output, attn_weights)
        """
        # Follow PyTorch MMCV MultiheadAttention flow exactly:
        # 1. Set defaults first
        if key is None:
            key = query
        if value is None:
            value = key

        # 2. key_pos fallback: if key_pos not provided but query_pos
        #    matches key shape, use query_pos for key_pos
        if key_pos is None:
            if query_pos is not None:
                # In TTSim we can't easily compare shapes at graph-build time,
                # but for self-attention (key was set to query above) they
                # always match, so we can safely use query_pos as key_pos.
                key_pos = query_pos

        # 3. Add positional encodings separately
        if query_pos is not None:
            query = self.query_add_pos(query, query_pos)
        if key_pos is not None:
            key = self.key_add_pos(key, key_pos)

        # Handle batch_first conversion
        if not self.batch_first:
            query = self.query_transpose_in(query)
            key = self.key_transpose_in(key)
            value = self.value_transpose_in(value)

        # Get dimensions
        if isinstance(query.shape, tuple):
            bs, seq_q, embed_dims = query.shape
            _, seq_k, _ = key.shape
        else:
            bs = query.shape[0]
            seq_q = query.shape[1]
            embed_dims = query.shape[2]
            seq_k = key.shape[1]

        # Project Q, K, V
        Q = self.q_proj(query)    # [bs, seq_q, embed_dims]
        K = self.k_proj(key)      # [bs, seq_k, embed_dims]
        V = self.v_proj(value)    # [bs, seq_k, embed_dims]

        # Reshape to multi-head format and transpose
        self.q_shape = F._from_data(
            f"{self.name}_q_shape",
            np.array([bs, seq_q, self.num_heads, self.head_dim], dtype=np.int64),
            is_const=True)
        Q = self.q_transpose(self.q_reshape(Q, self.q_shape))

        self.k_shape = F._from_data(
            f"{self.name}_k_shape",
            np.array([bs, seq_k, self.num_heads, self.head_dim], dtype=np.int64),
            is_const=True)
        K = self.k_transpose(self.k_reshape(K, self.k_shape))

        self.v_shape = F._from_data(
            f"{self.name}_v_shape",
            np.array([bs, seq_k, self.num_heads, self.head_dim], dtype=np.int64),
            is_const=True)
        V = self.v_transpose(self.v_reshape(V, self.v_shape))

        # Scaled dot-product attention
        K_T = self.k_t(K)
        scores = self.qk_matmul(Q, K_T)
        scores = self.scale_scores(scores, self.scale_const)

        # Apply attention mask if provided
        if attn_mask is not None:
            scores = self.add_attn_mask(scores, attn_mask)

        # Apply key_padding_mask: (N, S) -> (N, 1, 1, S) * -1e9 added to scores
        if key_padding_mask is not None:
            self.kpm_shape_tensor = F._from_data(
                f"{self.name}_kpm_shape",
                np.array([bs, 1, 1, seq_k], dtype=np.int64),
                is_const=True)
            kpm_4d = self.kpm_reshape(key_padding_mask, self.kpm_shape_tensor)
            kpm_mask = self.kpm_mul(kpm_4d, self.kpm_neg_inf)
            scores = self.kpm_add(scores, kpm_mask)

        # Softmax over last dimension (seq_k)
        attn_weights = self.softmax(scores)

        # Store attention weights before dropout for optional return
        attn_weights_out = attn_weights if need_weights else None

        # Apply attention dropout if specified
        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        output = self.attn_v_matmul(attn_weights, V)

        # Concatenate heads
        output = self.output_transpose(output)
        self.output_shape_tensor = F._from_data(
            f"{self.name}_output_shape",
            np.array([bs, seq_q, self.embed_dims], dtype=np.int64),
            is_const=True)
        output = self.output_reshape(output, self.output_shape_tensor)

        # Output projection
        output = self.out_proj(output)

        # Apply projection dropout if specified
        if self.proj_dropout is not None:
            output = self.proj_dropout(output)

        # Add residual connection if identity is provided
        if identity is not None:
            if not self.batch_first:
                identity = self.identity_transpose(identity)
            output = self.residual_add(output, identity)

        # Convert back if not batch_first
        if not self.batch_first:
            output = self.output_transpose_out(output)

        # Process attention weights for output if requested
        if need_weights and attn_weights_out is not None:
            if average_attn_weights:
                attn_weights_out = self.attn_avg_heads(attn_weights_out)
            return output, attn_weights_out
        else:
            return output


if __name__ == '__main__':
    logger.info("MultiheadAttention TTSim Module (FusionAD)")
    logger.info("=" * 80)

    try:
        attn = MultiheadAttention(
            'test_mha',
            embed_dims=256,
            num_heads=8,
            attn_drop=0.1,
            batch_first=False,
            bias=True
        )
        logger.debug("[OK] MultiheadAttention constructed")
        logger.debug(f"  embed_dims  = {attn.embed_dims}")
        logger.debug(f"  num_heads   = {attn.num_heads}")
        logger.debug(f"  head_dim    = {attn.head_dim}")
        logger.debug(f"  batch_first = {attn.batch_first}")
        logger.debug(f"  params      = {attn.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"[X] MultiheadAttention failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("=" * 80)
