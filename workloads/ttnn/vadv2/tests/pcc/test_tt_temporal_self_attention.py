#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.vadv2.tt import tt_temporal_self_attention
import numpy as np
from loguru import logger

mesh_device = ttnn.open_device(device_id=0)

class LayerParams:
    def __init__(self, in_dim, out_dim, device):
        self.weight = ttnn._rand((in_dim, out_dim), dtype=ttnn.bfloat16, device=device)
        self.bias = ttnn._rand((out_dim,), dtype=ttnn.bfloat16, device=device)

class TemporalSelfAttentionParams:
    def __init__(self, device):
        self.sampling_offsets = LayerParams(512, 128, device)
        self.attention_weights = LayerParams(512, 64, device)
        self.value_proj = LayerParams(256, 256, device)
        self.output_proj = LayerParams(256, 256, device)

class TemporalSelfAttentionTestParams:
    def __init__(self, device):
        self.temporal_self_attention = TemporalSelfAttentionParams(device)

def test_vadv2_tsa():
    query = ttnn._rand([1, 10000, 256], dtype=ttnn.bfloat16, device=mesh_device)
    query_pos = ttnn._rand([1, 10000, 256], dtype=ttnn.bfloat16, device=mesh_device)
    reference_points = ttnn._rand([2, 10000, 1, 2], dtype=ttnn.bfloat16, device=mesh_device)
    spatial_shapes = ttnn.Tensor(shape=[1, 2], dtype=ttnn.bfloat16, device=mesh_device, data=np.array([100, 100]))
    # spatial_shapes = [(100, 100)]
    level_start_index = ttnn._rand([1, 2], dtype=ttnn.bfloat16, device=mesh_device)

    parameter = TemporalSelfAttentionTestParams(mesh_device)
    ttnn_model = tt_temporal_self_attention.TtTemporalSelfAttention(
        params=parameter.temporal_self_attention, device=mesh_device, embed_dims=256, num_levels=1
    )

    logger.debug(f'Input query shape: {query.shape}')
    ttnn_output = ttnn_model(
        query,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )
    logger.debug(f"TSA output shape: {ttnn_output.shape}")
    if query.shape != ttnn_output.shape:
        raise AssertionError("Output shape does not match input query shape.")
    else:
        logger.debug('TT Temporal Self Attention Test Passed!')
        return 0


if __name__ == "__main__":
    test_vadv2_tsa()
