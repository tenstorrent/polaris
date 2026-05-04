#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

from .buffer import BufferType, TensorMemoryLayout, ShardSpec, ShardOrientation

# Matches profiler / tools.profiling.shape_canonical canonical memory tags.
_CANONICAL_LAYOUT_SUFFIX: dict[str, str] = {
    "INTERLEAVED": "INTERLEAVED",
    "HEIGHT_SHARDED": "HEIGHT_SHARDED",
    "BLOCK_SHARDED": "BLOCK_SHARDED",
    "WIDTH_SHARDED": "WIDTH_SHARDED",
}


class MemoryConfig:
    """Mirrors tt-metal's ``ttnn.MemoryConfig``.

    Constructor signature matches real TTNN::

        MemoryConfig(memory_layout, buffer_type, shard_spec=None)

    Positional order: memory_layout, buffer_type, shard_spec (matching
    ``ttnn.MemoryConfig(TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1, spec)``).
    """

    DRAM: ClassVar[MemoryConfig]
    L1: ClassVar[MemoryConfig]

    def __init__(self, memory_layout=TensorMemoryLayout.INTERLEAVED,
                 buffer_type=BufferType.DRAM, shard_spec=None):
        self.memory_layout = memory_layout
        self.buffer_type = buffer_type
        self.shard_spec = shard_spec

    def is_sharded(self):
        return self.memory_layout != TensorMemoryLayout.INTERLEAVED

    def __eq__(self, other):
        if not isinstance(other, MemoryConfig):
            return False
        return (self.memory_layout == other.memory_layout
                and self.buffer_type == other.buffer_type
                and self.shard_spec == other.shard_spec)

    def __repr__(self):
        return (f"MemoryConfig(memory_layout={self.memory_layout!r}, "
                f"buffer_type={self.buffer_type!r})")

    def to_canonical_memory_tag(self) -> str:
        """Short form matching HW profiler memory strings (e.g. ``L1_BLOCK_SHARDED``)."""
        buf = self.buffer_type.name.upper()
        lay = self.memory_layout.name.upper()
        suffix = _CANONICAL_LAYOUT_SUFFIX.get(lay, "INTERLEAVED")
        return f"{buf}_{suffix}"

    def __str__(self) -> str:
        return self.to_canonical_memory_tag()


# Singleton defaults matching real TTNN's ttnn.DRAM_MEMORY_CONFIG / L1_MEMORY_CONFIG.
MemoryConfig.DRAM = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM)
MemoryConfig.L1 = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1)

_SHARD_STRATEGY_TO_LAYOUT = {
    "HEIGHT": TensorMemoryLayout.HEIGHT_SHARDED,
    "WIDTH": TensorMemoryLayout.WIDTH_SHARDED,
    "BLOCK": TensorMemoryLayout.BLOCK_SHARDED,
}


def create_sharded_memory_config(shape, core_grid, strategy, orientation=None,
                                 use_height_and_width_as_shard_shape=False):
    """Build an L1-sharded ``MemoryConfig`` from a ``ShardStrategy``.

    Maps ``ShardStrategy.{HEIGHT,BLOCK,WIDTH}`` to the corresponding
    ``TensorMemoryLayout`` -- the same mapping real tt-metal performs.
    """
    strategy_name = getattr(strategy, "name", str(strategy)).upper()
    mem_layout = _SHARD_STRATEGY_TO_LAYOUT.get(
        strategy_name, TensorMemoryLayout.INTERLEAVED,
    )
    return MemoryConfig(mem_layout, BufferType.L1)


def create_sharded_memory_config_(shape, grid, mem_layout, orientation, tile_layout):
    """Create a sharded MemoryConfig with ShardSpec metadata.

    This is a minimal implementation for Polaris simulation that preserves
    sharding metadata (mem_layout, grid, orientation) without performing
    actual shard shape calculations.

    Args:
        shape: Tensor shape (unused in this stub, preserved for API compatibility)
        grid: CoreGrid, CoreRangeSet, or grid-like object specifying core layout
        mem_layout: TensorMemoryLayout (HEIGHT_SHARDED, BLOCK_SHARDED, etc.)
        orientation: ShardOrientation (ROW_MAJOR or COL_MAJOR)
        tile_layout: Whether to use tiled layout (unused in this stub)

    Returns:
        MemoryConfig with the specified memory_layout and a minimal ShardSpec.
    """
    # Normalize grid - convert list to tuple for consistency
    # Otherwise store as-is (could be tuple, CoreRangeSet, or any grid-like object)
    core_grid = tuple(grid) if isinstance(grid, list) else grid

    # Normalize orientation
    if isinstance(orientation, ShardOrientation):
        shard_orientation = orientation
    elif hasattr(orientation, 'name'):
        # Enum-like object, convert by name
        shard_orientation = ShardOrientation[orientation.name]
    else:
        # Default fallback
        shard_orientation = ShardOrientation.ROW_MAJOR

    # Create ShardSpec with placeholder shape (1, 1)
    # In full implementation, this would calculate actual shard dimensions
    # based on tensor shape, grid size, and sharding strategy
    shard_spec = ShardSpec(
        grid=core_grid,
        shape=(1, 1),  # Placeholder - not used in simulation
        orientation=shard_orientation
    )

    # Return MemoryConfig with the specified memory_layout and shard_spec
    return MemoryConfig(mem_layout, BufferType.L1, shard_spec=shard_spec)


def to_memory_config(input_tensor, memory_config=None):
    if memory_config is not None and hasattr(input_tensor, '_memory_config'):
        input_tensor._memory_config = memory_config
    return input_tensor


def get_memory_config(x):
    mc = getattr(x, '_memory_config', None)
    if mc is not None:
        return mc
    return MemoryConfig.L1
