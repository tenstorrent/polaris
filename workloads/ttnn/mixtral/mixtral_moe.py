#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn

class TtMoeLayer():
    def __init__(self, mesh_device, state_dict, experts, args, layer_num: int, dtype, tt_ccl):
        super().__init__()
        self.mesh_device = mesh_device
        self.experts = experts
        self.args = args
        self.dtype = dtype
        self.tile_size = args.tile_size
        assert self.tile_size == 32, "tile size must be 32"
        self.num_devices = self.args.num_devices
        assert self.num_devices == 8, "num devices must be 8 for Mixtral MoE"
        self.tt_ccl = tt_ccl
        self.gates_H8 = ttnn._rand(shape=[1, 1, 4096, 64], device=mesh_device, dtype=ttnn.bfloat16)
        self.top8_mask_11B_64 = ttnn.full(shape=[1, 1, 1, 64], fill_value=1.0, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.top2_mask_11BB = ttnn.full(shape=[1, 1, 1, 32], fill_value=1.0, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.reduce_mask = ttnn.zeros(shape=[1, 1, self.tile_size, self.tile_size * 8, 1], device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)


    def forward(self, inputs, mode):
        """
        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        S : seq len
        """
        input_i_1SBH = inputs
        expert_i_HH = self.experts
        # get logits for the experts
        gate_logits_1SB8 = ttnn.matmul(
            input_i_1SBH,
            self.gates_H8,
            memory_config=None,
            compute_kernel_config=None,
            core_grid=None,
            dtype=ttnn.bfloat16,
        )
        # get weights for top-2 experts -- masking out everything except the 8 experts (needed because top-k works with a min input of size 64)
        gate_logits_1SB8 = ttnn.add(gate_logits_1SB8, self.top8_mask_11B_64)

        k_val = 32
        k_tensor = ttnn.full(shape=[1], fill_value=k_val, device=self.mesh_device, dtype=ttnn.int64, layout=ttnn.TILE_LAYOUT)

        if mode == "decode":
            weights_1SB1 = ttnn.moe(gate_logits_1SB8, self.top8_mask_11B_64, self.top2_mask_11BB, k_val, k_tensor)
        else: # prefill
            topk_values, topk_indices = ttnn.topk(gate_logits_1SB8, k_tensor)
            topk_values = ttnn.add(topk_values, self.top2_mask_11BB)
            mask_B2 = ttnn.eqz(topk_indices)
            mask_B2 = ttnn.typecast(mask_B2, dtype=ttnn.bfloat16)
            weights_1SB1 = ttnn.sum(ttnn.softmax(topk_values, dim=-1) * mask_B2, dim=3, keepdim=True)
            weights_1SB1 = ttnn.unsqueeze(weights_1SB1, -1)

        # MLP and masking
        weights = expert_i_HH(input_i_1SBH, mode=mode)
        results_11BH = ttnn.multiply(weights, weights_1SB1)
        original_shape = results_11BH.shape
        output_1SBH = ttnn._rand(shape=[original_shape[0], original_shape[1], original_shape[2],
                                        int(original_shape[3])//int(self.args.num_experts)],
                                        device=self.mesh_device, dtype=self.dtype) # MoE CCL
        seq_len = results_11BH.shape[-2]

        if seq_len >= 2048 and mode == "decode":  # Reshape back to intended shape
            results_11BH = ttnn.reshape(results_11BH, [1, 1, seq_len, self.args.dim])

        output = ttnn.reshape(
            output_1SBH, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2],
                          int(original_shape[-1])//int(self.args.num_experts))
        )

        return output


    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)