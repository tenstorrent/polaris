#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as ttsimF
import ttsim.front.functional.sim_nn as SimNN

class TimestepEmbedding(SimNN.Module):
    def __init__(
        self,
        objname: str,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,    # type: ignore[assignment]
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()
        self.name = objname
        self.linear_1 = ttsimF.Linear(f'{self.name}_linear_1', in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = ttsimF.Linear(f'{self.name}_cond_proj', cond_proj_dim, in_channels)
        else:
            self.cond_proj = None

        self.act = SimNN.Silu(f'{self.name}_actfn')

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim # type: ignore[unreachable]
        self.linear_2 = ttsimF.Linear(f'{self.name}_linear_2', time_embed_dim, time_embed_dim_out)

        super().link_op2module()

    def __call__(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        return sample


class Timesteps(SimNN.Module):
    def __init__(self, objname: str, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.name = objname
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

        self.expop = ttsimF.Exp(f'{self.name}_exp')
        self.sinop = ttsimF.Sin(f'{self.name}_sin')
        self.cosop = ttsimF.Cos(f'{self.name}_cos')
        self.catop = ttsimF.ConcatX(f'{self.name}_cat', axis=1)
        super().link_op2module()

    def get_timestep_embedding(
        self,
        timesteps: SimNN.SimTensor,
        embedding_dim: int,
    ):
        half_dim = embedding_dim // 2
        exponent = ttsimF._from_shape(f'timestep_embedding_exponent', shape=[half_dim])
        self._tensors[exponent.name] = exponent
        emb = self.expop(exponent)
        s1 = self.sinop(emb).unsqueeze(0)
        c1 = self.cosop(emb).unsqueeze(0)
        emb = self.catop(s1, c1)
        return emb

    def __call__(self, timesteps: SimNN.SimTensor) -> SimNN.SimTensor:
        t_emb = self.get_timestep_embedding(
            timesteps,
            self.num_channels,
        )
        return t_emb
