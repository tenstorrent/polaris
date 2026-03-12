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
        args=None,
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
        self.args = args
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
        self.weight = ttnn._rand(shape=(1, 1, 32, 128), device=device, dtype=self.weight_dtype)
        self.bias = None

    def __call__(self, x, mode="decode"):
        rms = ttnn.layer_norm(x, weight=self.weight, epsilon=self.eps, axis=-1)
        normalized = ttnn.div(x, rms)
        normalized = ttnn.repeat(normalized, (1, 1, 1, self.args.num_experts))
        weight_tensor = self.weight
        weight_tensor = ttnn.reshape(weight_tensor, (1, 1, 1, self.dim))
        normalized = ttnn.multiply(normalized, weight_tensor)

        if self.bias is not None:
            normalized = ttnn.add(normalized, self.bias)
        return normalized
