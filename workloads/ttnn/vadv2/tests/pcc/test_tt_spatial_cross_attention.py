#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.vadv2.tt import tt_spatial_cross_attention
import numpy as np
from loguru import logger


mesh_device = ttnn.open_device(device_id=0)

class LayerParams:
    def __init__(self, in_dim, out_dim, device):
        self.weight = ttnn._rand((in_dim, out_dim), dtype=ttnn.bfloat16, device=device)
        self.bias = ttnn._rand((out_dim,), dtype=ttnn.bfloat16, device=device)

class SpatialCrossAttentionParams:
    def __init__(self, device):
        self.sampling_offsets = LayerParams(256, 128, device)
        self.attention_weights = LayerParams(256, 64, device)
        self.value_proj = LayerParams(256, 256, device)
        self.output_proj = LayerParams(256, 256, device)

class SpatialCrossAttentionTestParams:
    def __init__(self, device):
        self.spatial_cross_attention = SpatialCrossAttentionParams(device)

def test_vadv2_sca():
    parameter = SpatialCrossAttentionTestParams(mesh_device)
    query = ttnn._rand([1, 10000, 256], dtype=ttnn.bfloat16, device=mesh_device)
    key = ttnn._rand([6, 240, 1, 256], dtype=ttnn.bfloat16, device=mesh_device)
    value = ttnn._rand([6, 240, 1, 256], dtype=ttnn.bfloat16, device=mesh_device)
    reference_points = ttnn._rand([1, 4, 10000, 3], dtype=ttnn.bfloat16, device=mesh_device)
    spatial_shapes = ttnn.Tensor(shape=[1,2], dtype=ttnn.int32, device=mesh_device, data=np.array([12, 20]))
    reference_points_cam = ttnn._rand([6, 1, 10000, 4, 2], dtype=ttnn.bfloat16, device=mesh_device)
    bev_mask = ttnn._rand([6, 1, 10000, 4], dtype=ttnn.bfloat16, device=mesh_device)
    level_start_index = [0]

    tt_model = tt_spatial_cross_attention.TtSpatialCrossAttention(
        device=mesh_device,
        params=parameter.spatial_cross_attention,
        embed_dims=256,
    )

    tt_output = tt_model(
        query,
        key,
        value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask,
        level_start_index=level_start_index,
    )
    if query.shape != tt_output.shape:
        raise AssertionError("Output shape does not match input query shape.")
    else:
        logger.debug('TT Spatial Cross Attention Test Passed!')
        logger.debug(f'Input query shape: {query.shape}')
        logger.debug(f'TT output shape: {tt_output.shape}')
        return 0

if __name__ == "__main__":
    test_vadv2_sca()
