#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as T
from ttsim.ops import SimTensor

import math
import numpy as np
from typing import Optional, Callable, Tuple

# Optional: Rotary Embeddings implementation (minimal)
class RotaryEmbedding(SimNN.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        #no operators in __init__, following SimNN convention
        super().link_op2module()

    def __call__(self, x):
        assert x.rank() == 4, f"{self.name} Input Assumption [bs, nH, nW, dH] not true: {x.shape}"
        _, _, seq_len, head_dim = x.shape
        seq_len         = x.shape[-2]
        half_dim        = head_dim // 2
        positions       = F._from_shape('positions', [seq_len])
        freqs           = F._from_shape('freq',      [half_dim])
        half_dim_tensor = F._from_data ('half_dim', data=np.array([half_dim], dtype=np.int64), is_const=True)
        ten_thousand    = F._from_data ('10000',    data=np.array([10000],    dtype=np.int64), is_const=True)

        for t in [positions, freqs, half_dim_tensor, ten_thousand]:
            self._tensors[t.name] = t
            t.set_module(self)

        positions = positions.unsqueeze(0).transpose(0,1) #type: ignore
        freqs     = (ten_thousand ** (-freqs / half_dim_tensor)).unsqueeze(0) #type: ignore
        angles    = T.matmul(positions, freqs)
        cos       = angles.cos()[None, None, :, :]
        sin       = angles.sin()[None, None, :, :]
        x1, x2    = x[..., :half_dim], x[..., half_dim:]
        x_rotated = T.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

        return x_rotated

