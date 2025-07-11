#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto

class TensorMemoryLayout(Enum):
    INTERLEAVED    = auto()
    HEIGHT_SHARDED = auto()
    WIDTH_SHARDED  = auto()
    BLOCK_SHARDED  = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return TensorMemoryLayout[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()

class ShardOrientation(Enum):
    ROW_MAJOR = auto()
    COL_MAJOR = auto()

class ShardDistributionStrategy(Enum):
    ROUND_ROBIN_1D = auto() # Distribute each shard to each of the cores in a linearized list in a round-robin manner.
    GRID_2D        = auto() # Distribute a 2D grid of shards to a 2D grid of cores with one to one mapping.

class ShardMode(Enum):
    PHYSICAL = auto() #TODO: Deprecate this option to treat shard shape as physical
    LOGICAL  = auto()

class BufferType(Enum):
    DRAM          = auto()
    L1            = auto()
    SYSTEM_MEMORY = auto()
    L1_SMALL      = auto()
    TRACE         = auto()

