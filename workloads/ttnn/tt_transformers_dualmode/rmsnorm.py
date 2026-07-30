#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode RMSNorm for the llama3 tt_transformers port.

Basis: shim-only workloads/ttnn/tt_transformers/rmsnorm.py, made dual-mode.
The gamma weight shape (1, 1, dim//TILE, TILE) matches the HW LayerNormDeviceOperation
in1 (e.g. dim=4096 -> 1x1x128x32, ROW_MAJOR/bf16). Audit vs the shim-only base: the
per-dim if/elif weight shapes are replaced by the generic (1,1,dim//TILE,TILE) (same
values, no hardcoded dim list). Dummy weights (config-only); shim path uses ttnn._rand,
HW path a torch dummy.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]

TILE = 32
SHARD_HEIGHT = TILE  # ttnn.rms_norm requires shard height = a single tile


class RMSNorm:
    def __init__(self, device=None, dim=None, eps=1e-5, state_dict=None, weight_cache_path=None,
                 state_dict_prefix='', weight_dtype=ttnn.bfloat16, weight_key='ffn_norm',
                 is_distributed=False, add_unit_offset=False, sharded_program_config=None,
                 sharded_output_config=None, ccl_topology=None):
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

        # gamma weight: (1, 1, dim//TILE, TILE), ROW_MAJOR — matches HW LayerNorm in1
        wshape = (1, 1, dim // TILE, TILE)
        if IS_POLARIS:
            self.weight = ttnn._rand(shape=wshape, device=device, dtype=weight_dtype)
        else:
            import torch  # type: ignore[import-not-found]
            self.weight = ttnn.from_torch(
                torch.randn(*wshape), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
            )

    def __call__(self, x, mode='decode'):
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight_tensor=self.weight,
            memory_config=None,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dim=self.dim,
        )
