#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
 
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn

class Embedding():
    def __init__(
        self,
        mesh_device,
        args,
        weight_cache_path,
        state_dict,
        dtype,
        dim = 3072,
    ):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        weight = ttnn._rand(shape=(128256, dim), device=mesh_device, dtype=dtype)
        torch_weight = weight #.unsqueeze(0).unsqueeze(0)
        self.weights = ttnn.as_tensor(
            torch_weight,
            dtype=dtype,
            device=self.mesh_device,
            mesh_mapper=None, #ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=None, #args.get_model_config()["EMB_WEIGHTS_MEMCFG"],
            cache_file_name=None, #cache_name,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    def __call__(self, x):
        return self.forward(x)
