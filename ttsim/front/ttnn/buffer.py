#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from enum import Enum, auto
from typing import Tuple

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


class ShardSpec:
    """Minimal ShardSpec for simulation - preserves sharding metadata.

    Mirrors tt-metal's ShardSpec structure with grid, shape, and orientation.
    For Polaris simulation, this is primarily metadata tracking - actual shard
    calculations are not performed.

    The grid can be any grid-like object (CoreGrid, CoreRangeSet, tuple, etc.)
    and is stored as-is for later extraction by call sites.
    """
    def __init__(self, grid, shape: Tuple[int, int],
                 orientation: ShardOrientation = ShardOrientation.ROW_MAJOR):
        """Initialize ShardSpec.

        Args:
            grid: Core grid specification (CoreGrid, CoreRangeSet, or grid-like object)
            shape: (height, width) shard dimensions
            orientation: ROW_MAJOR or COL_MAJOR core traversal order
        """
        self.grid = grid
        self.shape = shape
        self.orientation = orientation

    def __repr__(self):
        return f"ShardSpec(grid={self.grid}, shape={self.shape}, orientation={self.orientation})"

    def __eq__(self, other):
        if not isinstance(other, ShardSpec):
            return False
        return (self.grid == other.grid and
                self.shape == other.shape and
                self.orientation == other.orientation)

