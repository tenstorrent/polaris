#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from typing import Any, Callable, Optional, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.common.gemma_Lightweightmodule import LightweightModule  # type: ignore[import-untyped]

TILE = 32
SHARD_HEIGHT = TILE


def to_numpy(tensor):
    """Convert any tensor type to numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


class SimpleGrid:
    """Simple grid class for simulator compatibility."""
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"SimpleGrid(x={self.x}, y={self.y})"


def create_core_grid(grid_x: int, grid_y: int) -> Any:
    """Create a CoreGrid or SimpleGrid depending on what's available."""
    if hasattr(ttnn, 'CoreGrid'):
        try:
            return ttnn.CoreGrid(x=grid_x, y=grid_y)  # type: ignore[call-arg]
        except (TypeError, AttributeError):
            pass
        try:
            return ttnn.CoreGrid(grid_x, grid_y) # type: ignore[call-arg, arg-type,misc]
        except (TypeError, AttributeError):
            pass
    return SimpleGrid(grid_x, grid_y)


class TtLayerNorm(LightweightModule):

    sharded_input_config: Any
    sharded_program_config: Any
    sharded_output_config: Any

    def __init__(
        self,
        device: Any,
        dim: int,
        state_dict: Optional[Dict[str, Any]],
        state_dict_prefix: str,
        weight_cache_path: Any = None,
        weight_memory_config: Any = None,
        weight_dtype: Any = None,
        model_config: Optional[Dict[str, Any]] = None,
        eps: float = 1e-05,
    ) -> None:
        super().__init__()
        self.device = device
        self.eps = eps

        if weight_memory_config is None:
            weight_memory_config = ttnn.DRAM_MEMORY_CONFIG
        if weight_dtype is None:
            weight_dtype = ttnn.bfloat8_b

        # Load weight
        weight_key = f"{state_dict_prefix}weight"
        if state_dict is not None and weight_key in state_dict:
            weight_data = state_dict[weight_key]
            np_weight = to_numpy(weight_data)
            np_weight = np_weight.reshape(1, 1, dim)
            np_weight = np.broadcast_to(np_weight, (1, SHARD_HEIGHT, dim)).copy()
        else:
            np_weight = np.ones((1, SHARD_HEIGHT, dim), dtype=np.float32)

        # Load bias
        bias_key = f"{state_dict_prefix}bias"
        if state_dict is not None and bias_key in state_dict:
            bias_data = state_dict[bias_key]
            np_bias = to_numpy(bias_data)
            np_bias = np_bias.reshape(1, 1, dim)
            np_bias = np.broadcast_to(np_bias, (1, SHARD_HEIGHT, dim)).copy()
        else:
            np_bias = np.zeros((1, SHARD_HEIGHT, dim), dtype=np.float32)

        cache_name: Callable[..., Any]
        if weight_cache_path is None:
            cache_name = lambda *_: None
        else:
            cache_name = lambda suffix: weight_cache_path / (state_dict_prefix + f"{suffix}")

        is_mesh_device = device.__class__.__name__ == "MeshDevice" if device is not None else False

        # Convert to ttnn tensors
        self.weight = ttnn.as_tensor(
            np_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name("weight"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.bias = ttnn.as_tensor(
            np_bias,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name("bias"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        # Setup sharded configurations
        if model_config:
            self.sharded_input_config = model_config["SHARDED_NORM_INPUT_MEMCFG"]
            self.sharded_program_config = model_config["SHARDED_NORM_PRGM_CFG"]
            self.sharded_output_config = model_config["SHARDED_NORM_OUTPUT_MEMCFG"]
        else:
            assert dim % SHARD_HEIGHT == 0, f"dim ({dim}) must be divisible by SHARD_HEIGHT ({SHARD_HEIGHT})"
            shard_width_hidden_dim_across_32_cores = dim // SHARD_HEIGHT
            grid_x = 8
            grid_y = SHARD_HEIGHT // 8
            core_grid = create_core_grid(grid_x, grid_y)

            self.sharded_input_config = ttnn.create_sharded_memory_config(
                shape=(SHARD_HEIGHT, shard_width_hidden_dim_across_32_cores),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            if hasattr(ttnn, 'LayerNormShardedMultiCoreProgramConfig'):
                self.sharded_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[grid_x, grid_y],
                    subblock_w=shard_width_hidden_dim_across_32_cores // TILE,
                    block_h=SHARD_HEIGHT // TILE,
                    block_w=shard_width_hidden_dim_across_32_cores // TILE,
                    inplace=False,
                )
            else:
                self.sharded_program_config = {
                    "compute_with_storage_grid_size": [grid_x, grid_y],
                    "subblock_w": shard_width_hidden_dim_across_32_cores // TILE,
                    "block_h": SHARD_HEIGHT // TILE,
                    "block_w": shard_width_hidden_dim_across_32_cores // TILE,
                    "inplace": False,
                }

            self.sharded_output_config = self.sharded_input_config

    def forward(self, x: ttnn.Tensor, in_sharded: bool = False, out_sharded: bool = False) -> ttnn.Tensor:
        if in_sharded:
            x = ttnn.layer_norm(
                x,
                epsilon=self.eps,
                weight=self.weight,
                bias=self.bias,
                program_config=self.sharded_program_config,
                memory_config=self.sharded_output_config,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4
                ),
            )
            if out_sharded:
                return x
            x_interleaved = ttnn.sharded_to_interleaved(x)
            if hasattr(x, 'deallocate'):
                x.deallocate(True)
            else:
                del x
            return x_interleaved
        else:
            assert not out_sharded, "Non-sharded LayerNorm cannot output sharded tensor"
            x = ttnn.layer_norm(
                x,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.eps,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                ),
            )
            return x