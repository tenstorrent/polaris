#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.vadv2.tt.common import TtConv2D

class TtBottleneck:
    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        is_downsample=False,
        blk_sharded=False,
        activation_dtype=ttnn.bfloat16,
        conv3_blk_sharded=False,
    ):
        self.is_downsample = is_downsample
        self.activation_dtype = activation_dtype

        self.conv1 = TtConv2D(
            conv_args.conv1, conv_pth.conv1, device=device, activation='relu'
        )
        self.conv2 = TtConv2D(
            conv_args.conv2,
            conv_pth.conv2,
            device=device,
            activation='relu', # ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h=32,
        )
        self.conv3 = TtConv2D(conv_args.conv3, conv_pth.conv3, device=device, activation=None, is_blk=conv3_blk_sharded)

        if is_downsample:
            self.downsample = TtConv2D(
                conv_args.downsample[0],
                conv_pth.downsample,
                device=device,
                activation=None,
                is_blk=blk_sharded,
                activation_dtype=activation_dtype,
            )

    def __call__(self, x_identity):
        x = self.conv1(x_identity)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.conv2(x)
        x = self.conv3(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        if self.is_downsample:
            x_identity = self.downsample(x_identity)
        x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG)
        
        x = ttnn.add(x, x_identity)
        x = ttnn.relu(x)

        ttnn.deallocate(x_identity)
        return x
