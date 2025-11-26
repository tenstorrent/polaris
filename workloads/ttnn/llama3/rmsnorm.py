#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
TILE = 32
SHARD_HEIGHT = TILE  # Current ttnn.rms_norm implementation requires shard height to be a single tile
class RMSNorm():
    def __init__(self, device=None,
        dim=None,
        eps=0.00001,
        state_dict=None,
        weight_cache_path=None,
        state_dict_prefix="",
        weight_dtype=ttnn.bfloat16,
        weight_key="ffn_norm",
        is_distributed=False,
        add_unit_offset=False,
        sharded_program_config=None,
        sharded_output_config=None,
        ccl_topology=None):
        self.device = device
        self.dim = dim
        self.eps = eps
        self.state_dict = state_dict
        self.weight_cache_path = weight_cache_path
        self.state_dict_prefix = state_dict_prefix
        self.weight_dtype = weight_dtype
        self.weight_key = weight_key
        self.is_distributed = is_distributed
        self.add_unit_offset = add_unit_offset
        self.sharded_program_config = sharded_program_config
        self.sharded_output_config = sharded_output_config
        self.ccl_topology = ccl_topology
        self.compute_kernel_config_hifi2 = ttnn.MathFidelity.HiFi2
        if dim == 3072:
            self.weight = ttnn._rand(shape=(1, 1, 96, 32), device=device, dtype=self.weight_dtype)
        elif dim == 4096:
            self.weight = ttnn._rand(shape=(1, 1, 128, 32), device=device, dtype=self.weight_dtype)
        elif dim == 2048:
            self.weight = ttnn._rand(shape=(1, 1, 64, 32), device=device, dtype=self.weight_dtype)
        elif dim == 8192:
            self.weight = ttnn._rand(shape=(1, 1, 256, 32), device=device, dtype=self.weight_dtype)

    def __call__(self, x, mode="decode"):
        #print(f'shape of weight is {self.weight.shape} and x is {x.shape}')
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight_tensor=self.weight,
            memory_config=None,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dim=self.dim
        )
