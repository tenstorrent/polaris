#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn

class TtMixtralMLP():
    def __init__(self, mesh_device, state_dict, args, layer_num, dtypes):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.dtypes = dtypes
        self.model_args = args
        self.dim = args.dim

        self.w1 = ttnn.Tensor(shape=(self.dim, int(3.5 * self.dim)), device=mesh_device, dtype=ttnn.bfloat16)
        self.w2 = ttnn.Tensor(shape=(int(3.5 * self.dim), self.dim), device=mesh_device, dtype=ttnn.bfloat16)
        self.w3 = ttnn.Tensor(shape=(self.dim, int(3.5 * self.dim)), device=mesh_device, dtype=ttnn.bfloat16)

    def forward(self, x: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        if mode == "prefill":
            original_shape = x.shape

            w1_out = ttnn.linear(
                x,
                self.w1,
                compute_kernel_config=None,
                core_grid=None,
                dtype=ttnn.bfloat16,
                activation="silu",
                program_config=None,
            )

            w3_out = ttnn.linear(
                x,
                self.w3,
                compute_kernel_config=None,
                core_grid=None,
                dtype=ttnn.bfloat16,
                program_config=None,
            )

            ttnn.deallocate(x)

            w2_in = ttnn.multiply(w1_out, w3_out, dtype=ttnn.bfloat16, memory_config=None)

            ttnn.deallocate(w3_out)
            ttnn.deallocate(w1_out)

            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=None,
                core_grid=None,
                dtype=ttnn.bfloat8_b,
                program_config=None,
            )

            ttnn.deallocate(w2_in)

            w2_out = ttnn.reshape(w2_out, original_shape)

        else:  # Decode
            w1_out = ttnn.matmul(
                x,
                self.w1,
                program_config=None,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=None,
                dtype=ttnn.bfloat8_b,
            )
            w3_out = ttnn.matmul(
                x,
                self.w3,
                program_config=None,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=None,
                dtype=ttnn.bfloat8_b,
            )

            w2_in = ttnn.multiply(w1_out, w3_out)

            w2_out = ttnn.matmul(
                w2_in,
                self.w2,
                program_config=None,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=None,
                dtype=ttnn.bfloat8_b,
            )

        return w2_out

    def __call__(self, x: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        return self.forward(x, mode)
