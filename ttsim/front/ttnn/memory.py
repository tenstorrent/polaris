#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto

class MemoryConfig(Enum):
    DRAM = auto()
    L1   = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return MemoryConfig[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()

#placeholders
def create_sharded_memory_config(shape,
                                 core_grid,
                                 strategy,#: ShardStrategy,
                                 orientation,#: ShardOrientation= None,
                                 use_height_and_width_as_shard_shape: bool = False) -> MemoryConfig:
    """
    shape       (ttnn.Shape | Tuple[int, ...] | List[int])
    core_grid   (ttnn.CoreGrid | ttnn.CoreRangeSet) – the core_grid on which to distribute the sharded tensor on (writes to the cores L1s)
    strategy    (ttnn.ShardStrategy) – the sharding strategy of either height, width or block
    orientation (ttnn.ShardOrientation, optional) – the order in which to traverse the cores when reading/writing shards. Defaults to None
    use_height_and_width_as_shard_shape (bool, optional)
        if True, the height and width of the tensor will be used as the shard shape
        if False, the shard shape will be calculated based on the core_grid and the tensor shape where tensor shape = [math.prod(dims), width]
        Defaults to False
    """
    return MemoryConfig.L1

def create_sharded_memory_config_(shape, grid, mem_layout, orientation, tile_layout):
    return MemoryConfig.L1

def to_memory_config(input_tensor, memory_config=None):
    return input_tensor  # No actual conversion, just returning the input tensor

def get_memory_config(x):
    return MemoryConfig.L1

