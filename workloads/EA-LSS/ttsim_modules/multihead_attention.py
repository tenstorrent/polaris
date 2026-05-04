#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of MultiheadAttention from TransFusion head.

Original file: mmdet3d/models/dense_heads/transfusion_head.py (class at ~line 80)

This is a custom multi-head attention implementation (not the standard
torch.nn.MultiheadAttention) that uses a combined QKV projection weight
of shape [3*embed_dim, embed_dim].  At inference it is functionally
equivalent to PyTorch's MultiheadAttention with default settings.

Parameters:
    in_proj_weight:  [3*E, E]  (combined Q, K, V projection)
    in_proj_bias:    [3*E]     (combined Q, K, V bias)
    out_proj.weight: [E, E]
    out_proj.bias:   [E]
    Total: 4*E^2 + 4*E

This module wraps the existing SimNN.MultiheadAttention which already
models the same parameter layout.

No torch / mmcv imports.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import SimOpHandle, _from_shape, _from_data


class MultiheadAttention(SimNN.Module):
    """
    TTSim MultiheadAttention — wraps SimNN.MultiheadAttention with the
    same interface as the custom implementation in transfusion_head.py.

    The original class uses:
        in_proj_weight [3*E, E] + in_proj_bias [3*E]
        out_proj: nn.Linear(E, E, bias=True)

    SimNN.MultiheadAttention models exactly the same parameter layout.

    Call signature (same as the source class):
        attn_output, attn_weights = mha(query, key, value,
                                        key_padding_mask=None,
                                        need_weights=True,
                                        attn_mask=None)

    Args:
        name (str): Unique module name prefix.
        embed_dim (int): Total model dimension (E).
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability (applied to attention weights).
            Default: 0.0.
        bias (bool): Add bias to all projections. Default: True.
        kdim (int | None): Key feature dimension. Default: None (= embed_dim).
        vdim (int | None): Value feature dimension. Default: None (= embed_dim).

    Shape:
        - query:  (L, N, E)
        - key:    (S, N, E)
        - value:  (S, N, E)
        - output: (L, N, E)  [plus optional attn_weights (N, L, S)]
    """

    def __init__(
        self,
        name: str,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: int = None,
        vdim: int = None,
    ):
        super().__init__()
        self.name = name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias

        # Delegate to the TTSim built-in MultiheadAttention module
        self._mha = SimNN.MultiheadAttention(
            name + ".mha",
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )
        self._submodules[self._mha.name] = self._mha

        super().link_op2module()

    def __call__(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
    ):
        """
        Forward pass.

        Args:
            query  (SimTensor): [L, N, E]
            key    (SimTensor): [S, N, E]
            value  (SimTensor): [S, N, E]
            key_padding_mask: optional [N, S] boolean mask
            need_weights (bool): Whether to return attention weights.
            attn_mask: optional [L, S] additive mask.

        Returns:
            attn_output (SimTensor): [L, N, E]
            attn_weights (SimTensor | None): [N, L, S] if need_weights, else None
        """
        result = self._mha(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )
        if isinstance(result, tuple):
            return result
        return result, None

    def analytical_param_count(self, lvl: int = 0) -> int:
        """
        in_proj: 3*E^2 + (3*E if bias)
        out_proj: E^2  + (E   if bias)
        Total: 4*E^2 + 4*E (with bias=True)
        """
        E = self.embed_dim
        params = 3 * E * E + E * E          # weights
        if self.bias:
            params += 3 * E + E             # biases
        return params
