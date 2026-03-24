#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Embedder module - Positional encoding with sin/cos functions
"""

# -------------------------------PyTorch--------------------------------

# import math
# import torch
# import torch.nn as nn
# import numpy as np
# from mmcv.cnn import bias_init_with_prob, xavier_init
#
#
# class Embedder:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.create_embedding_fn()
#
#     def create_embedding_fn(self):
#         embed_fns = []
#         d = self.kwargs['input_dims']
#         out_dim = 0
#         if self.kwargs['include_input']:
#             embed_fns.append(lambda x : x)
#             out_dim += d
#
#         max_freq = self.kwargs['max_freq_log2']
#         N_freqs = self.kwargs['num_freqs']
#
#         if self.kwargs['log_sampling']:
#             freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
#         else:
#             freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
#
#         for freq in freq_bands:
#             for p_fn in self.kwargs['periodic_fns']:
#                 embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
#                 out_dim += d
#
#         self.embed_fns = embed_fns
#         self.out_dim = out_dim
#
#     def embed(self, inputs):
#         return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
#
#
# class MotionMLP(nn.Module):
#     '''
#     Args:
#         c_dim (int): dimension of latent code c
#         f_dim (int): feature dimension
#     '''
#
#     def __init__(self, c_dim, f_dim=512, identity=True):
#         super().__init__()
#         self.c_dim = c_dim
#         self.f_dim = f_dim
#         self.identity = identity
#
#         multires = 10
#         embed_kwargs = {
#                 'include_input' : True,
#                 'input_dims' : c_dim,
#                 'max_freq_log2' : multires-1,
#                 'num_freqs' : multires,
#                 'log_sampling' : True,
#                 'periodic_fns' : [torch.sin, torch.cos],
#         }
#         self.pos_embedder = Embedder(**embed_kwargs)
#
#         self.fc = nn.Sequential(
#             nn.Linear(f_dim + self.pos_embedder.out_dim, 2*f_dim),
#             nn.LayerNorm(2*f_dim),
#             nn.ReLU(),
#             nn.Linear(2*f_dim, f_dim)
#         )
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.fc:
#             for param in m.parameters():
#                 if param.dim() > 1:
#                     nn.init.xavier_uniform_(param)
#
#
#     def forward(self, x, pose_info):
#         pose_embed = self.pos_embedder.embed(pose_info)
#         xc = torch.cat([x, pose_embed], dim=-1)
#         out = self.fc(xc)
#
#         if self.identity:
#             out = out + x
#
#         return out

# -------------------------------TTSIM-----------------------------------


import os, sys
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import numpy as np


class Embedder(SimNN.Module):
    """Positional encoding using sin/cos at multiple frequencies"""

    def __init__(
        self,
        name,
        include_input=True,
        input_dims=7,
        max_freq_log2=9,
        num_freqs=10,
        log_sampling=True,
        periodic_fns=["sin", "cos"],
    ):
        super().__init__()
        self.name = name
        self.include_input = include_input
        self.input_dims = input_dims

        # Calculate output dimension
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dims
        self.out_dim += input_dims * len(periodic_fns) * num_freqs

        # Pre-compute frequency bands
        if log_sampling:
            freq_bands = 2.0 ** np.linspace(0.0, max_freq_log2, num_freqs)
        else:
            freq_bands = np.linspace(2.0**0, 2.0**max_freq_log2, num_freqs)

        # Pre-create all operators and constant tensors in __init__
        self.freq_scalars: list[Any] = []
        self.mul_ops: Any = []
        self.sin_ops: Any = []
        self.cos_ops: Any = []
        _mul_ops = []
        _sin_ops = []
        _cos_ops = []

        for i, freq in enumerate(freq_bands):
            # Create frequency scalar constant
            freq_scalar = F._from_data(
                f"{self.name}.freq_{i}",
                np.array(freq, dtype=np.float32).reshape(1),
                is_const=True,
            )
            setattr(self, freq_scalar.name, freq_scalar)
            self.freq_scalars.append(freq_scalar)

            # Create multiplication operator
            mul_op = F.Mul(f"{self.name}.mul_freq_{i}")
            _mul_ops.append(mul_op)

            # Create sin/cos operators based on periodic_fns
            if "sin" in periodic_fns:
                sin_op = F.Sin(f"{self.name}.sin_freq_{i}")
                _sin_ops.append(sin_op)
            if "cos" in periodic_fns:
                cos_op = F.Cos(f"{self.name}.cos_freq_{i}")
                _cos_ops.append(cos_op)

        # Register op lists with the module (SimOpHandleList triggers __setattr__ registration)
        if _mul_ops:
            self.mul_ops = F.SimOpHandleList(_mul_ops)
        if _sin_ops:
            self.sin_ops = F.SimOpHandleList(_sin_ops)
        if _cos_ops:
            self.cos_ops = F.SimOpHandleList(_cos_ops)

        # Concatenation operator
        self.concat = F.ConcatX(self.name + ".concat", axis=-1)

        super().link_op2module()

    def __call__(self, x):
        embed_list = []

        # Include raw input if specified
        if self.include_input:
            embed_list.append(x)

        # Apply periodic functions at each frequency
        # Original MapTracker order: for each freq: sin(x*freq), cos(x*freq)
        for i in range(len(self.freq_scalars)):
            # x * freq
            x_scaled = self.mul_ops[i](x, self.freq_scalars[i])
            setattr(self, x_scaled.name, x_scaled)

            # Apply sin and cos
            if self.sin_ops:
                sin_out = self.sin_ops[i](x_scaled)
                embed_list.append(sin_out)
                setattr(self, sin_out.name, sin_out)

            if self.cos_ops:
                cos_out = self.cos_ops[i](x_scaled)
                embed_list.append(cos_out)
                setattr(self, cos_out.name, cos_out)

        # Concatenate all embeddings
        if len(embed_list) == 1:
            return embed_list[0]
        else:
            result = self.concat(*embed_list)
            setattr(self, result.name, result)
            return result


class MotionMLP(SimNN.Module):
    """
    Motion MLP with positional encoding

    Args:
        name: Module name
        c_dim: Dimension of pose info (latent code c)
        f_dim: Feature dimension (default 512)
        identity: Whether to add residual connection (default True)

    Architecture:
        1. Positional encoding of pose_info using Embedder
        2. Concatenate features x with pose embeddings
        3. Linear(f_dim + pos_embed_dim, 2*f_dim)
        4. LayerNorm(2*f_dim)
        5. ReLU
        6. Linear(2*f_dim, f_dim)
        7. [Optional] Add residual connection
    """

    def __init__(self, name, c_dim, f_dim=512, identity=True):
        super().__init__()
        self.name = name
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.identity = identity

        # Create positional embedder
        multires = 10
        self.pos_embedder = Embedder(
            name=name + ".pos_embedder",
            include_input=True,
            input_dims=c_dim,
            max_freq_log2=multires - 1,
            num_freqs=multires,
            log_sampling=True,
            periodic_fns=["sin", "cos"],
        )

        # Calculate dimensions
        pos_embed_dim = self.pos_embedder.out_dim
        input_dim = f_dim + pos_embed_dim
        hidden_dim = 2 * f_dim
        output_dim = f_dim

        # Pre-create all operators
        # Concatenation for [x, pose_embed]
        self.concat = F.ConcatX(name + ".concat", axis=-1)

        # FC layer 1: Linear + bias + LayerNorm + ReLU
        # ttsim Linear: input @ weight, so [nrow, ncol] = [in_features, out_features]
        self.fc1 = F.Linear(name + ".fc1", nrow=input_dim, ncol=hidden_dim, module=self)
        self.fc1_bias = F._from_shape(name + ".fc1.bias", [hidden_dim], is_param=True)
        self.fc1_add = F.Add(name + ".fc1_add")

        self.ln1 = F.LayerNorm(name + ".ln1", hidden_dim)
        self.relu = F.Relu(name + ".relu")

        # FC layer 2: Linear + bias
        self.fc2 = F.Linear(
            name + ".fc2", nrow=hidden_dim, ncol=output_dim, module=self
        )
        self.fc2_bias = F._from_shape(name + ".fc2.bias", [output_dim], is_param=True)
        self.fc2_add = F.Add(name + ".fc2_add")

        # Residual add if identity
        if self.identity:
            self.add = F.Add(name + ".add")

        super().link_op2module()

    def __call__(self, x, pose_info):
        """
        Forward pass

        Args:
            x: Features [batch, ..., f_dim]
            pose_info: Pose information [batch, ..., c_dim]

        Returns:
            out: Motion-encoded features [batch, ..., f_dim]
        """
        # 1. Positional encoding of pose_info
        pose_embed = self.pos_embedder(pose_info)
        setattr(self, pose_embed.name, pose_embed)

        # 2. Concatenate features with pose embeddings
        xc = self.concat(x, pose_embed)
        setattr(self, xc.name, xc)

        # 3. FC layer 1: Linear + bias
        fc1_out = self.fc1(xc)
        setattr(self, fc1_out.name, fc1_out)
        fc1_out = self.fc1_add(fc1_out, self.fc1_bias)
        setattr(self, fc1_out.name, fc1_out)

        # 4. LayerNorm
        ln1_out = self.ln1(fc1_out)
        setattr(self, ln1_out.name, ln1_out)

        # 5. ReLU
        relu_out = self.relu(ln1_out)
        setattr(self, relu_out.name, relu_out)

        # 6. FC layer 2: Linear + bias
        fc2_out = self.fc2(relu_out)
        setattr(self, fc2_out.name, fc2_out)
        fc2_out = self.fc2_add(fc2_out, self.fc2_bias)
        setattr(self, fc2_out.name, fc2_out)

        # 7. Optional residual connection
        if self.identity:
            out = self.add(fc2_out, x)
            setattr(self, out.name, out)
        else:
            out = fc2_out

        return out

    def analytical_param_count(self, lvl):
        """Calculate parameter count"""
        pos_embed_dim = self.pos_embedder.out_dim
        input_dim = self.f_dim + pos_embed_dim
        hidden_dim = 2 * self.f_dim

        # FC1: weight + bias
        fc1_params = input_dim * hidden_dim + hidden_dim
        # LayerNorm: weight + bias
        ln1_params = hidden_dim + hidden_dim
        # FC2: weight + bias
        fc2_params = hidden_dim * self.f_dim + self.f_dim

        total = fc1_params + ln1_params + fc2_params
        return total
