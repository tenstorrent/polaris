#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim port of titans_pytorch/memory_models.py
- MemoryMLP : the small 2-layer MLP whose weights ARE the neural memory
- ResidualNorm : LayerNorm(model(x)) + x wrapper (TTT-paper convention)
Only MemoryMLP is wired into NeuralMemory by default (matches train_mac.py).
Other variants (GatedResidualMemoryMLP, FactorizedMemoryMLP, MemorySwiGluMLP,
MemoryAttention) can be added later with the same SimNN.Module pattern.
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

class ResidualNorm(SimNN.Module):
    """LayerNorm(model(x)) + x.  Used by NeuralMemory when mem_model_norm_add_residual=True."""
    def __init__(self, name, dim, model):
        super().__init__()
        self.name  = name
        self.dim   = dim
        self.model = model
        self.norm  = F.LayerNorm(f'{name}.ln', dim)
        self.add   = F.Add(f'{name}.add_res')
        super().link_op2module()

    def __call__(self, x):
        out = self.model(x)
        return self.add(self.norm(out), x)

    def analytical_param_count(self, lvl=0):
        return 2 * self.dim + self.model.analytical_param_count(lvl + 1)

class MemoryMLP(SimNN.Module):
    """
    Reference: titans_pytorch.memory_models.MemoryMLP

    depth-layer MLP, GELU between layers, no bias.
    Input/output dim == dim (i.e. dim_head of NeuralMemory).
    """
    def __init__(self, name, dim, depth, expansion_factor=2.0):
        super().__init__()
        self.name  = name
        self.dim   = dim
        self.depth = depth
        dim_hidden = int(dim * expansion_factor)
        dims       = [dim] + [dim_hidden] * (depth - 1) + [dim]
        self.dims  = dims

        self.matmuls = []
        self.gelus   = []
        for i in range(depth):
            mm = F.MatMul(f'{name}.W{i}')
            w  = F._from_shape(f'{name}.W{i}.param', shape=[dims[i], dims[i + 1]], is_param=True)
            setattr(self, f'_W{i}', w)
            setattr(self, f'mm{i}', mm)
            self.matmuls.append((mm, w))
            if i < depth - 1:
                g = F.Gelu(f'{name}.gelu{i}')
                setattr(self, f'gelu{i}', g)
                self.gelus.append(g)
        super().link_op2module()

    def __call__(self, x):
        out = x
        for i, (mm, w) in enumerate(self.matmuls):
            if i > 0:
                out = self.gelus[i - 1](out)
            out = mm(out, w)
        return out

    def analytical_param_count(self, lvl=0):
        return sum(self.dims[i] * self.dims[i + 1] for i in range(self.depth))
