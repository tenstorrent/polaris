#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Shim Functions for Simulation

This module provides drop-in replacements for ttnn operations that track
operations without actually executing them. These shims are designed for
simulation and performance modeling purposes.

All functions match the exact Python API of their ttnn counterparts and
track compute and memory operations that would be required.
"""

# Constants matching TTNN

from loguru import logger
import numpy as np

from ttsim.ops.op import SimOp

from .tensor import DataType, Layout, Shape
from .types import TILE_HEIGHT, TILE_WIDTH, TILE_HW
from .tensor import Tensor
from .op import generate_new_op_name

# # Layout constants
# class Layout:
#     ROW_MAJOR = 0
#     TILE = 1
#     ROW_MAJOR_LAYOUT = ROW_MAJOR
#     TILE_LAYOUT = TILE
#
# # DataType constants
# class DataType:
#     BFLOAT16 = 0
#     FLOAT32 = 1
#     UINT32 = 2
#     INT32 = 3
#     UINT16 = 4
#     BFLOAT8_B = 5
#     BFLOAT4_B = 6
#     INVALID = 7

# Memory config placeholder
class MemoryConfig:
    def __init__(self, memory_layout=None, buffer_type=None, shard_spec=None, nd_shard_spec=None):
        self.memory_layout = memory_layout
        self.buffer_type = buffer_type
        self.shard_spec = shard_spec
        self.nd_shard_spec = nd_shard_spec

    def is_sharded(self):
        return self.shard_spec is not None or self.nd_shard_spec is not None

    def __eq__(self, other):
        if not isinstance(other, MemoryConfig):
            return False
        return (self.memory_layout == other.memory_layout and
                self.buffer_type == other.buffer_type and
                self.shard_spec == other.shard_spec and
                self.nd_shard_spec == other.nd_shard_spec)

# Default memory configs
DRAM_MEMORY_CONFIG = MemoryConfig()
L1_MEMORY_CONFIG = MemoryConfig()

# Execution mode control
class ExecutionMode:
    """Execution mode for shim functions."""
    TRACK_ONLY = "track_only"      # Only track operations, don't execute
    EXECUTE = "execute"             # Actually perform operations
    EXECUTE_AND_TRACK = "execute_and_track"  # Both execute and track

# Global execution mode (default: track only for backward compatibility)
_execution_mode = ExecutionMode.TRACK_ONLY

def set_execution_mode(mode):
    """Set global execution mode.
    Args:
        mode: ExecutionMode.TRACK_ONLY, ExecutionMode.EXECUTE, or ExecutionMode.EXECUTE_AND_TRACK
    """
    global _execution_mode
    if mode not in [ExecutionMode.TRACK_ONLY, ExecutionMode.EXECUTE, ExecutionMode.EXECUTE_AND_TRACK]:
        raise ValueError(f"Invalid execution mode: {mode}")
    _execution_mode = mode

def get_execution_mode():
    """Get current global execution mode."""
    return _execution_mode

# Operation tracking
class OperationTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.tilize_count = 0
        self.untilize_count = 0
        self.tilize_with_val_padding_count = 0
        self.untilize_with_unpadding_count = 0
        self.to_layout_count = 0
        self.reshape_count = 0
        self.pad_count = 0
        self.memory_operations = []
        # Detailed internal operation tracking
        self.compute_operations = []  # Individual compute operations
        self.memory_reads = []  # Memory read operations
        self.memory_writes = []  # Memory write operations
        self.circular_buffer_ops = []  # Circular buffer operations
        self.kernel_invocations = []  # Kernel execution tracking
        self.data_movements = []  # Data movement between buffers

    def track_tilize(self, tensor_shape, num_tiles, num_tiles_per_row=None, num_tiles_per_col=None,
                     use_multicore=True, element_size=2):
        """Track tilize operation with internal operation details."""
        self.tilize_count += 1

        # Calculate internal operations
        if num_tiles_per_row is None or num_tiles_per_col is None:
            # Estimate from shape
            if len(tensor_shape) >= 2:
                width = tensor_shape[-1]
                height = tensor_shape[-2] if len(tensor_shape) >= 2 else 1
                num_tiles_per_row = (width + TILE_WIDTH - 1) // TILE_WIDTH
                num_tiles_per_col = (height + TILE_HEIGHT - 1) // TILE_HEIGHT
            else:
                num_tiles_per_row = 1
                num_tiles_per_col = 1
        # Calculate memory operations
        tile_size_bytes = TILE_HW * element_size
        total_input_bytes = num_tiles * tile_size_bytes  # Input in row-major
        total_output_bytes = num_tiles * tile_size_bytes  # Output in tile layout
        # Track memory reads (reading row-major input)
        self.memory_reads.append({
            'op': 'tilize_read',
            'bytes': total_input_bytes,
            'num_tiles': num_tiles,
            'tiles_per_row': num_tiles_per_row,
            'tiles_per_col': num_tiles_per_col
        })
        # Track memory writes (writing tile layout output)
        self.memory_writes.append({
            'op': 'tilize_write',
            'bytes': total_output_bytes,
            'num_tiles': num_tiles,
            'tiles_per_row': num_tiles_per_row,
            'tiles_per_col': num_tiles_per_col
        })
        # Track compute operations (one per tile)
        compute_ops = num_tiles
        self.compute_operations.append({
            'op': 'tilize_compute',
            'count': compute_ops,
            'type': 'datacopy_tilize',
            'num_tiles': num_tiles,
            'use_multicore': use_multicore
        })
        # Track circular buffer operations
        # Tilize typically uses: input CB -> unpack -> compute -> pack -> output CB
        self.circular_buffer_ops.append({
            'op': 'tilize_cb',
            'input_cb_ops': num_tiles,  # Reads from input CB
            'output_cb_ops': num_tiles,  # Writes to output CB
            'num_tiles': num_tiles
        })
        # Track kernel invocations
        if use_multicore:
            # Multicore: one kernel per core, estimate cores from tile distribution
            estimated_cores = min(num_tiles, 128)  # Reasonable upper bound
        else:
            estimated_cores = 1
        self.kernel_invocations.append({
            'op': 'tilize_kernel',
            'count': estimated_cores,
            'num_tiles_per_core': num_tiles // max(estimated_cores, 1),
            'use_multicore': use_multicore
        })
        # Track data movement
        self.data_movements.append({
            'op': 'tilize_movement',
            'from': 'row_major_buffer',
            'to': 'tile_layout_buffer',
            'bytes': total_input_bytes,
            'num_tiles': num_tiles
        })
        # High-level operation tracking
        self.memory_operations.append({
            'op': 'tilize',
            'shape': tensor_shape,
            'num_tiles': num_tiles,
            'num_tiles_per_row': num_tiles_per_row,
            'num_tiles_per_col': num_tiles_per_col,
            'total_bytes_read': total_input_bytes,
            'total_bytes_written': total_output_bytes,
            'compute_operations': compute_ops,
            'kernel_invocations': estimated_cores
        })
    def track_untilize(self, tensor_shape, num_tiles, num_tiles_per_row=None, num_tiles_per_col=None,
                       use_multicore=True, use_pack_untilize=True, element_size=2):
        """Track untilize operation with internal operation details."""
        self.untilize_count += 1
        # Calculate internal operations
        if num_tiles_per_row is None or num_tiles_per_col is None:
            # Estimate from shape
            if len(tensor_shape) >= 2:
                width = tensor_shape[-1]
                height = tensor_shape[-2] if len(tensor_shape) >= 2 else 1
                num_tiles_per_row = (width + TILE_WIDTH - 1) // TILE_WIDTH
                num_tiles_per_col = (height + TILE_HEIGHT - 1) // TILE_HEIGHT
            else:
                num_tiles_per_row = 1
                num_tiles_per_col = 1
        # Calculate memory operations
        tile_size_bytes = TILE_HW * element_size
        total_input_bytes = num_tiles * tile_size_bytes  # Input in tile layout
        total_output_bytes = num_tiles * tile_size_bytes  # Output in row-major
        # Track memory reads (reading tile layout input)
        self.memory_reads.append({
            'op': 'untilize_read',
            'bytes': total_input_bytes,
            'num_tiles': num_tiles,
            'tiles_per_row': num_tiles_per_row,
            'tiles_per_col': num_tiles_per_col
        })
        # Track memory writes (writing row-major output)
        self.memory_writes.append({
            'op': 'untilize_write',
            'bytes': total_output_bytes,
            'num_tiles': num_tiles,
            'tiles_per_row': num_tiles_per_row,
            'tiles_per_col': num_tiles_per_col
        })
        # Track compute operations
        # Untilize: unpack -> compute (datacopy) -> pack
        compute_ops = num_tiles
        self.compute_operations.append({
            'op': 'untilize_compute',
            'count': compute_ops,
            'type': 'datacopy_untilize',
            'num_tiles': num_tiles,
            'use_pack_untilize': use_pack_untilize,
            'use_multicore': use_multicore
        })
        # Track circular buffer operations
        self.circular_buffer_ops.append({
            'op': 'untilize_cb',
            'input_cb_ops': num_tiles,  # Reads from input CB (tile layout)
            'output_cb_ops': num_tiles,  # Writes to output CB (row-major)
            'num_tiles': num_tiles,
            'use_pack_untilize': use_pack_untilize
        })
        # Track kernel invocations
        if use_multicore:
            estimated_cores = min(num_tiles, 128)
        else:
            estimated_cores = 1
        self.kernel_invocations.append({
            'op': 'untilize_kernel',
            'count': estimated_cores,
            'num_tiles_per_core': num_tiles // max(estimated_cores, 1),
            'use_multicore': use_multicore,
            'use_pack_untilize': use_pack_untilize
        })
        # Track data movement
        self.data_movements.append({
            'op': 'untilize_movement',
            'from': 'tile_layout_buffer',
            'to': 'row_major_buffer',
            'bytes': total_input_bytes,
            'num_tiles': num_tiles
        })
        # High-level operation tracking
        self.memory_operations.append({
            'op': 'untilize',
            'shape': tensor_shape,
            'num_tiles': num_tiles,
            'num_tiles_per_row': num_tiles_per_row,
            'num_tiles_per_col': num_tiles_per_col,
            'total_bytes_read': total_input_bytes,
            'total_bytes_written': total_output_bytes,
            'compute_operations': compute_ops,
            'kernel_invocations': estimated_cores,
            'use_pack_untilize': use_pack_untilize
        })
    def track_tilize_with_val_padding(self, input_shape, output_shape, num_tiles):
        self.tilize_with_val_padding_count += 1
        self.memory_operations.append({
            'op': 'tilize_with_val_padding',
            'input_shape': input_shape,
            'output_shape': output_shape,
            'num_tiles': num_tiles
        })
    def track_untilize_with_unpadding(self, input_shape, output_shape, num_tiles):
        self.untilize_with_unpadding_count += 1
        self.memory_operations.append({
            'op': 'untilize_with_unpadding',
            'input_shape': input_shape,
            'output_shape': output_shape,
            'num_tiles': num_tiles
        })
    def track_to_layout(self, from_layout, to_layout):
        self.to_layout_count += 1
        self.memory_operations.append({
            'op': 'to_layout',
            'from_layout': from_layout,
            'to_layout': to_layout
        })
    def track_reshape(self, from_shape, to_shape):
        self.reshape_count += 1
        self.memory_operations.append({
            'op': 'reshape',
            'from_shape': from_shape,
            'to_shape': to_shape
        })
    def track_pad(self, from_shape, to_shape):
        self.pad_count += 1
        self.memory_operations.append({
            'op': 'pad',
            'from_shape': from_shape,
            'to_shape': to_shape
        })
    def get_summary(self):
        """Get summary of all tracked operations."""
        total_memory_read_bytes = sum(op.get('bytes', 0) for op in self.memory_reads)
        total_memory_write_bytes = sum(op.get('bytes', 0) for op in self.memory_writes)
        total_compute_ops = sum(op.get('count', 0) for op in self.compute_operations)
        total_kernel_invocations = sum(op.get('count', 0) for op in self.kernel_invocations)
        total_cb_ops = sum(op.get('input_cb_ops', 0) + op.get('output_cb_ops', 0) for op in self.circular_buffer_ops)
        return {
            'tilize_count': self.tilize_count,
            'untilize_count': self.untilize_count,
            'tilize_with_val_padding_count': self.tilize_with_val_padding_count,
            'untilize_with_unpadding_count': self.untilize_with_unpadding_count,
            'to_layout_count': self.to_layout_count,
            'reshape_count': self.reshape_count,
            'pad_count': self.pad_count,
            'total_operations': len(self.memory_operations),
            # Internal operation tracking
            'total_memory_read_bytes': total_memory_read_bytes,
            'total_memory_write_bytes': total_memory_write_bytes,
            'total_compute_operations': total_compute_ops,
            'total_kernel_invocations': total_kernel_invocations,
            'total_circular_buffer_ops': total_cb_ops,
            'memory_read_count': len(self.memory_reads),
            'memory_write_count': len(self.memory_writes),
            'compute_operation_count': len(self.compute_operations),
            'kernel_invocation_count': len(self.kernel_invocations),
            'circular_buffer_op_count': len(self.circular_buffer_ops),
            'data_movement_count': len(self.data_movements)
        }
    def get_detailed_summary(self):
        """Get detailed summary with all internal operations."""
        return {
            'high_level_summary': self.get_summary(),
            'memory_reads': self.memory_reads,
            'memory_writes': self.memory_writes,
            'compute_operations': self.compute_operations,
            'kernel_invocations': self.kernel_invocations,
            'circular_buffer_ops': self.circular_buffer_ops,
            'data_movements': self.data_movements,
            'all_operations': self.memory_operations
        }

# Global operation tracker
_tracker = OperationTracker()

def get_tracker():
    """Get the global operation tracker."""
    return _tracker

def reset_tracker():
    """Reset the global operation tracker."""
    _tracker.reset()

#
# class Shape:
#     """Shape class that mimics ttnn.Shape behavior."""
#     def __init__(self, shape):
#         if isinstance(shape, (list, tuple)):
#             self._shape = list(shape)
#         elif isinstance(shape, Shape):
#             self._shape = list(shape._shape)
#         else:
#             raise TypeError(f"Invalid shape type: {type(shape)}")
#
#     def __getitem__(self, index):
#         return self._shape[index]
#
#     def __setitem__(self, index, value):
#         self._shape[index] = value
#
#     def __len__(self):
#         return len(self._shape)
#
#     def __iter__(self):
#         return iter(self._shape)
#
#     def __repr__(self):
#         return f"Shape({self._shape})"
#
#     def rank(self):
#         return len(self._shape)
#
#     def volume(self):
#         result = 1
#         for dim in self._shape:
#             result *= dim
#         return result
#
#     def to_rank(self, rank):
#         """Convert shape to specified rank by padding with 1s or truncating."""
#         current_rank = len(self._shape)
#         if current_rank == rank:
#             return Shape(self._shape)
#         elif current_rank < rank:
#             # Pad with 1s at the beginning
#             new_shape = [1] * (rank - current_rank) + self._shape
#             return Shape(new_shape)
#         else:
#             # Truncate from the beginning
#             new_shape = self._shape[-rank:]
#             return Shape(new_shape)
#
#     def view(self):
#         """Return a view of the shape as a list."""
#         return list(self._shape)

#
# class TensorProxy:
#     """Proxy class for ttnn.Tensor that tracks state and optionally stores actual data."""
#
#     def __init__(self, shape, dtype=DataType.BFLOAT16, layout=Layout.ROW_MAJOR_LAYOUT,
#                  memory_config=None, padded_shape=None, device=None, data=None):
#         # Handle shape input
#         if isinstance(shape, (list, tuple)):
#             self._logical_shape = Shape(shape)
#         elif isinstance(shape, Shape):
#             self._logical_shape = Shape(shape)
#         else:
#             raise TypeError(f"Invalid shape type: {type(shape)}")
#
#         # Calculate padded shape if not provided
#         if padded_shape is None:
#             self._padded_shape = self._calculate_padded_shape(self._logical_shape, layout)
#         else:
#             if isinstance(padded_shape, (list, tuple)):
#                 self._padded_shape = Shape(padded_shape)
#             elif isinstance(padded_shape, Shape):
#                 self._padded_shape = Shape(padded_shape)
#             else:
#                 raise TypeError(f"Invalid padded_shape type: {type(padded_shape)}")
#
#         self._dtype = dtype
#         self._layout = layout
#         self._memory_config = memory_config or DRAM_MEMORY_CONFIG
#         self._device = device
#         self._storage_type = "DEVICE" if device is not None else "HOST"
#         self._buffer = None if device is None else "buffer_placeholder"
#
#         # Store actual data if provided (for execute mode)
#         self._data = data  # List or array of actual values
#
#     def _calculate_padded_shape(self, logical_shape, layout):
#         """Calculate padded shape based on logical shape and layout."""
#         if len(logical_shape) == 0:
#             return Shape([])
#
#         padded = list(logical_shape._shape)
#         if layout == Layout.TILE_LAYOUT and len(padded) >= 2:
#             # Pad last two dimensions to tile boundaries
#             padded[-2] = ((padded[-2] + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
#             padded[-1] = ((padded[-1] + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH
#         return Shape(padded)
#
#     @property
#     def shape(self):
#         return self._logical_shape
#
#     def logical_shape(self):
#         return self._logical_shape
#
#     def padded_shape(self):
#         return self._padded_shape
#
#     def dtype(self):
#         return self._dtype
#
#     def layout(self):
#         return self._layout
#
#     def memory_config(self):
#         return self._memory_config
#
#     def device(self):
#         return self._device
#
#     def storage_type(self):
#         return self._storage_type
#
#     def buffer(self):
#         return self._buffer
#
#     def is_sharded(self):
#         return self._memory_config.is_sharded()
#
#     def physical_volume(self):
#         return self._padded_shape.volume()
#
#     def element_size(self):
#         """Return element size in bytes based on dtype."""
#         dtype_sizes = {
#             DataType.BFLOAT16: 2,
#             DataType.FLOAT32: 4,
#             DataType.UINT32: 4,
#             DataType.INT32: 4,
#             DataType.UINT16: 2,
#         }
#         return dtype_sizes.get(self._dtype, 2)
#
#     def tensor_spec(self):
#         """Return a mock tensor spec."""
#         class TensorSpec:
#             def __init__(self, shape, layout, dtype, tile):
#                 self._shape = shape
#                 self._layout = layout
#                 self._dtype = dtype
#                 self._tile = tile
#
#             def tile(self):
#                 return self._tile
#
#         class Tile:
#             def __init__(self):
#                 self._width = TILE_WIDTH
#                 self._height = TILE_HEIGHT
#
#             def get_width(self):
#                 return self._width
#
#             def get_height(self):
#                 return self._height
#
#         return TensorSpec(self._padded_shape, self._layout, self._dtype, Tile())
#
#     def get_data(self):
#         """Get actual tensor data (if available)."""
#         return self._data
#
#     def set_data(self, data):
#         """Set actual tensor data."""
#         self._data = data
#
#     def has_data(self):
#         """Check if tensor has actual data stored."""
#         return self._data is not None
#
#     def __repr__(self):
#         has_data_str = f", has_data={self.has_data()}" if self.has_data() else ""
#         return f"TensorProxy(shape={self._logical_shape}, dtype={self._dtype}, layout={self._layout}{has_data_str})"
#

TensorProxy = Tensor

def round_up(value, multiple):
    """Round up value to nearest multiple."""
    return ((value + multiple - 1) // multiple) * multiple


def pad_to_tile_shape(shape):
    """Pad shape to tile boundaries."""
    if len(shape) < 2:
        return Shape(shape)
    padded = list(shape)
    padded[-2] = round_up(padded[-2], TILE_HEIGHT)
    padded[-1] = round_up(padded[-1], TILE_WIDTH)
    return Shape(padded)


# ============================================================================
# Pure Python Layout Conversion Algorithms (for execute mode)
# ============================================================================

def _get_dtype_python_type(dtype):
    """Convert DataType to Python type."""
    dtype_map = {
        DataType.BFLOAT16: float,
        DataType.FLOAT32: float,
        DataType.UINT32: int,
        DataType.INT32: int,
        DataType.UINT16: int,
    }
    return dtype_map.get(dtype, float)


def _create_empty_data(shape, dtype, fill_value=0.0):
    """Create empty data array with given shape and dtype."""
    python_type = _get_dtype_python_type(dtype)
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    return [python_type(fill_value) for _ in range(total_elements)]


def _flatten_shape(shape):
    """Flatten shape to 2D (H, W) for layout conversion."""
    if len(shape) == 0:
        return (1, 1)
    elif len(shape) == 1:
        return (1, shape[0])
    elif len(shape) == 2:
        return tuple(shape)
    else:
        # Flatten all leading dimensions
        H = 1
        for i in range(len(shape) - 2):
            H *= shape[i]
        H *= shape[-2]
        W = shape[-1]
        return (H, W)


def _unflatten_data(data, flat_shape, original_shape):
    """Reshape flattened data back to original shape."""
    if len(original_shape) <= 2:
        return data
    # Calculate total elements
    total_elements = 1
    for dim in original_shape:
        total_elements *= dim

    # If shapes match, return as-is
    if len(data) == total_elements:
        return data

    # Otherwise, we need to reshape
    # For simplicity, just return the data (actual reshaping would be more complex)
    return data[:total_elements]


def _row_major_to_tile_layout_swizzled(data, H, W):
    """
    Convert row-major data to tile layout (swizzled format).
    This implements a simplified version of the tilize operation:
    - Divides data into 32x32 tiles
    - Each tile is stored in swizzled format (4 faces of 16x16)
    - Face order: face0, face1, face2, face3 (row-major within tile)
    Args:
        data: List of values in row-major order
        H: Height dimension
        W: Width dimension
    Returns:
        List of values in tile layout
    """
    if len(data) != H * W:
        raise ValueError(f"Data size {len(data)} doesn't match shape {H}x{W}")

    num_tiles_h = (H + TILE_HEIGHT - 1) // TILE_HEIGHT
    num_tiles_w = (W + TILE_WIDTH - 1) // TILE_WIDTH
    output = []

    for tile_h in range(num_tiles_h):
        for tile_w in range(num_tiles_w):
            # Process one 32x32 tile
            for face_row in range(2):  # 2 rows of faces
                for face_col in range(2):  # 2 cols of faces
                    # Process one 16x16 face
                    for row_in_face in range(16):
                        for col_in_face in range(16):
                            # Calculate position in original row-major data
                            global_row = tile_h * TILE_HEIGHT + face_row * 16 + row_in_face
                            global_col = tile_w * TILE_WIDTH + face_col * 16 + col_in_face
                            if global_row < H and global_col < W:
                                idx = global_row * W + global_col
                                output.append(data[idx])
                            else:
                                # Padding - use 0
                                output.append(0.0)
    return output


def _tile_layout_swizzled_to_row_major(data, H, W):
    """
    Convert tile layout (swizzled format) back to row-major.

    This is the inverse of _row_major_to_tile_layout_swizzled.

    Args:
        data: List of values in tile layout
        H: Height dimension (logical)
        W: Width dimension (logical)
    Returns:
        List of values in row-major order
    """
    num_tiles_h = (H + TILE_HEIGHT - 1) // TILE_HEIGHT
    num_tiles_w = (W + TILE_WIDTH - 1) // TILE_WIDTH
    output = [0.0] * (H * W)
    data_idx = 0
    for tile_h in range(num_tiles_h):
        for tile_w in range(num_tiles_w):
            # Process one 32x32 tile
            for face_row in range(2):  # 2 rows of faces
                for face_col in range(2):  # 2 cols of faces
                    # Process one 16x16 face
                    for row_in_face in range(16):
                        for col_in_face in range(16):
                            # Calculate position in output row-major data
                            global_row = tile_h * TILE_HEIGHT + face_row * 16 + row_in_face
                            global_col = tile_w * TILE_WIDTH + face_col * 16 + col_in_face
                            if global_row < H and global_col < W and data_idx < len(data):
                                idx = global_row * W + global_col
                                output[idx] = data[data_idx]
                            data_idx += 1
    return output


def _perform_tilize_operation(input_data, input_shape, output_padded_shape, pad_value=0.0):
    """
    Perform actual tilize operation on data.
    Args:
        input_data: Input data in row-major format
        input_shape: Logical input shape
        output_padded_shape: Padded output shape
        pad_value: Value to use for padding
    Returns:
        Output data in tile layout format
    """
    # Flatten to 2D for processing
    H_in, W_in = _flatten_shape(input_shape)
    H_out, W_out = _flatten_shape(output_padded_shape)
    # Pad input data to output shape if needed
    if H_in < H_out or W_in < W_out:
        padded_data = _create_empty_data([H_out, W_out], DataType.FLOAT32, pad_value)
        # Copy input data
        for h in range(H_in):
            for w in range(W_in):
                idx_in = h * W_in + w
                idx_out = h * W_out + w
                if idx_in < len(input_data):
                    padded_data[idx_out] = input_data[idx_in]
        input_data = padded_data
        H_in, W_in = H_out, W_out

    # Convert to tile layout
    tiled_data = _row_major_to_tile_layout_swizzled(input_data, H_in, W_in)

    return tiled_data


def _perform_untilize_operation(input_data, input_padded_shape, output_shape):
    """
    Perform actual untilize operation on data.
    Args:
        input_data: Input data in tile layout format
        input_padded_shape: Padded input shape
        output_shape: Logical output shape (unpadded)
    Returns:
        Output data in row-major format
    """
    # Flatten to 2D for processing
    H_in, W_in = _flatten_shape(input_padded_shape)
    H_out, W_out = _flatten_shape(output_shape)

    # Convert from tile layout to row-major
    row_major_data = _tile_layout_swizzled_to_row_major(input_data, H_in, W_in)

    # Extract only the logical shape (unpad)
    if H_out < H_in or W_out < W_in:
        output_data = []
        for h in range(H_out):
            for w in range(W_out):
                idx = h * W_in + w
                if idx < len(row_major_data):
                    output_data.append(row_major_data[idx])
        return output_data
    return row_major_data


def requires_padding_change(tensor, layout):
    """Check if padding change is required for layout conversion."""
    if layout == Layout.ROW_MAJOR_LAYOUT:
        # For row major, there shouldn't be extra padding
        logger.debug(
            "requires_padding_change: row major, logical_shape {} and padded_shape {}, padding change is {}",
            tensor.logical_shape()._shape,
            tensor.padded_shape()._shape,
            tensor.logical_shape()._shape != tensor.padded_shape()._shape,
        )
        return tensor.logical_shape()._shape != tensor.padded_shape()._shape
    else:
        # For tile layout, check if current padding matches tile requirements
        logger.debug(" type of tensor logical shape is {}", type(tensor.logical_shape()))
        tile_spec_padded = pad_to_tile_shape(tensor.logical_shape()._shape)
        return tensor.padded_shape()._shape != tile_spec_padded._shape


def squeeze_from_ND_to_4D(tensor, sub_core_grids=None):
    """Squeeze tensor from N dimensions to 4D."""
    shape = tensor.logical_shape()
    rank = shape.rank()

    if rank < 4:
        raise RuntimeError(f"Tensor has to be of rank >= 4! Instead is {rank}")

    if rank == 4:
        return tensor

    # Handle leading 1s
    if shape[0] == 1:
        squeezed = tensor
        i = 0
        while rank > 4 and shape[i] == 1:
            # Simulate squeeze by removing dimension
            new_shape = list(shape._shape)
            new_shape.pop(0)
            # There are few different Tensor types, and this tensor is created
            # using the same type as the input tensor.
            squeezed = type(tensor)(
                shape=Shape(new_shape),
                dtype=tensor.dtype,
                layout=tensor.get_layout(),
                memory_config=tensor.memory_config(),
                padded_shape=tensor.padded_shape()
            )
            rank = squeezed.logical_shape().rank()
            i += 1
            if rank <= 4:
                return squeezed
    # Reshape to 4D
    squeezed_shape = squeeze_shape_to_4D(shape)
    return reshape(squeezed, squeezed_shape, None, None, None, sub_core_grids)


def squeeze_shape_to_4D(shape):
    """Convert shape to 4D by collapsing leading dimensions."""
    if shape.rank() <= 4:
        return Shape(shape)
    shape_4d = [1, 1, 1, 1]
    # Collapse all leading dimensions into first dimension
    shape_4d[0] = 1
    for i in range(shape.rank() - 4 + 1):
        shape_4d[0] *= shape[i]
    shape_4d[0] -= 1 if shape_4d[0] > 0 else 0
    # Copy last 3 dimensions
    extra_rank = shape.rank() - 4
    shape_4d[1] = shape[1 + extra_rank] if 1 + extra_rank < shape.rank() else 1
    shape_4d[2] = shape[2 + extra_rank] if 2 + extra_rank < shape.rank() else 1
    shape_4d[3] = shape[3 + extra_rank] if 3 + extra_rank < shape.rank() else 1

    return Shape(shape_4d)


def reshape(tensor, shape, padded_shape=None, memory_config=None, pad_value=None,
            reshape_map_mode=None, sub_core_grids=None):
    """Reshape tensor to new shape."""
    if isinstance(shape, (list, tuple)):
        shape = Shape(shape)

    if padded_shape is None:
        padded_shape = pad_to_tile_shape(shape._shape) if tensor.get_layout() == Layout.TILE_LAYOUT else shape

    if isinstance(padded_shape, (list, tuple)):
        padded_shape = Shape(padded_shape)

    # Check execution mode for tracking
    mode = get_execution_mode()
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)

    if should_track:
        _tracker.track_reshape(tensor.logical_shape()._shape, shape._shape)

    # Preserve data if available (reshape doesn't change data, just view)
    output_data = None
    if hasattr(tensor, 'has_data') and tensor.has_data():
        output_data = tensor.get_data()
    elif hasattr(tensor, 'get_data'):
        try:
            output_data = tensor.get_data()
        except:
            output_data = None
    # Use the same tensor type among the different tensor types as the input tensor
    return type(tensor)(
        shape=shape,
        dtype=tensor.dtype,
        layout=tensor.get_layout(),
        memory_config=memory_config or tensor.memory_config(),
        padded_shape=padded_shape,
        device=tensor.device,
        data=output_data
    )


def squeeze(tensor, dim):
    """Squeeze tensor at specified dimension."""
    shape = tensor.logical_shape()
    if dim >= shape.rank():
        raise RuntimeError(f"Dimension {dim} out of range for tensor of rank {shape.rank()}")

    if shape[dim] != 1:
        raise RuntimeError(f"Cannot squeeze dimension {dim} with size {shape[dim]}")

    new_shape = list(shape._shape)
    new_shape.pop(dim)

    new_padded_shape = list(tensor.padded_shape()._shape)
    if dim < len(new_padded_shape):
        new_padded_shape.pop(dim)

    # Preserve data if available (squeeze doesn't change data, just removes dimension)
    output_data = tensor.get_data() if tensor.has_data() else None

    # Use the same tensor type among the different tensor types as the input tensor
    return type(tensor)(
        shape=Shape(new_shape),
        dtype=tensor.dtype,
        layout=tensor.get_layout(),
        memory_config=tensor.memory_config(),
        padded_shape=Shape(new_padded_shape),
        device=tensor.device,
        data=output_data
    )


def pad(tensor, padding, pad_value, output_memory_config=None):
    """Pad tensor with specified padding."""
    logical_shape = tensor.logical_shape()
    padded_shape = tensor.padded_shape()
    # Calculate new padded shape
    new_padded = list(padded_shape._shape)
    if len(padding) >= 2:
        # Last two dimensions
        if len(new_padded) >= 2:
            new_padded[-2] += padding[-2][0] + padding[-2][1]
            new_padded[-1] += padding[-1][0] + padding[-1][1]
    # Check execution mode for tracking
    mode = get_execution_mode()
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)

    if should_track:
        _tracker.track_pad(tensor.padded_shape()._shape, new_padded)

    # Perform actual padding if in execute mode
    output_data = None
    if should_execute and tensor.has_data():
        # Simple padding: create new data array and copy existing data
        input_data = tensor.get_data()
        input_shape = tensor.padded_shape()._shape
        output_shape = new_padded
        # Calculate total elements
        total_output_elements = 1
        for dim in output_shape:
            total_output_elements *= dim

        python_type = _get_dtype_python_type(tensor.dtype)
        output_data = [python_type(pad_value) for _ in range(total_output_elements)]

        # Copy input data to output (simplified - assumes padding at end)
        # This is a simplified implementation
        for i in range(min(len(input_data), len(output_data))):
            output_data[i] = input_data[i]

    # Use the same tensor type among the different tensor types as the input tensor
    return type(tensor)(
        shape=logical_shape,
        dtype=tensor.dtype,
        layout=tensor.get_layout(),
        memory_config=output_memory_config or tensor.memory_config(),
        padded_shape=Shape(new_padded),
        device=tensor.device,
        data=output_data
    )


def tilize(input_tensor, memory_config=None, output_dtype=None, use_multicore=True,
           use_low_perf=False, sub_core_grids=None):
    """
    Tilize operation - converts ROW_MAJOR_LAYOUT to TILE_LAYOUT.

    Args:
        input_tensor: Input tensor in ROW_MAJOR_LAYOUT
        memory_config: Optional output memory configuration
        output_dtype: Optional output data type
        use_multicore: Whether to use multicore (default: True)
        use_low_perf: Whether to use low performance mode (default: False)
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with TILE_LAYOUT
    """
    # Validation
    if input_tensor.storage_type() != "DEVICE":
        raise RuntimeError("Operands to tilize need to be on device!")

    if input_tensor.buffer() is None:
        raise RuntimeError("Operands to tilize need to be allocated in buffers on device!")

    if input_tensor.get_layout() != Layout.ROW_MAJOR_LAYOUT:
        raise RuntimeError("Can only tilize row major data")

    # Check physical volume
    physical_vol = input_tensor.physical_volume()
    if physical_vol % TILE_HW != 0:
        raise RuntimeError(
            f"Input tensor physical volume ({physical_vol}) must be divisible by TILE_HW ({TILE_HW})"
        )

    # Check data type - handle both DataType enums and numpy dtypes
    # (Tensor class converts DataType to numpy dtype in __init__)
    tensor_dtype = input_tensor.dtype
    if isinstance(tensor_dtype, np.dtype):
        # Convert numpy dtype back to DataType for validation
        tensor_dtype = DataType.from_numpy(tensor_dtype)

    valid_dtypes = [DataType.BFLOAT16, DataType.FLOAT32, DataType.UINT32,
                    DataType.INT32, DataType.UINT16]
    if tensor_dtype not in valid_dtypes:
        raise RuntimeError(
            f"Can only tilize bfloat16/float32 or int32/uint32/uint16 tensors"
        )

    # Handle empty tensors
    if physical_vol == 0:
        output_shape = input_tensor.logical_shape()
        output_padded = pad_to_tile_shape(output_shape._shape)
        # Preserve data if available (empty tensor case)
        output_data = input_tensor.get_data() if input_tensor.has_data() else None

        # Use the same tensor type among the different tensor types as the input tensor
        return type(input_tensor)(
            shape=output_shape,
            dtype=output_dtype or input_tensor.dtype,
            layout=Layout.TILE_LAYOUT,
            memory_config=memory_config or input_tensor.memory_config(),
            padded_shape=output_padded,
            device=input_tensor.device,
            data=output_data
        )

    # Handle rank > 4
    if input_tensor.logical_shape().rank() > 4:
        squeezed = squeeze_from_ND_to_4D(input_tensor, sub_core_grids)
        result = tilize(squeezed, memory_config, output_dtype, use_multicore,
                       use_low_perf, sub_core_grids)
        # Unsqueeze back
        return reshape(result, input_tensor.logical_shape(),
                     input_tensor.padded_shape(), memory_config, None, None, sub_core_grids)

    # Calculate number of tiles
    padded_shape = input_tensor.padded_shape()
    num_tiles_per_row = padded_shape[-1] // TILE_WIDTH
    num_tiles_per_col = padded_shape[-2] // TILE_HEIGHT if len(padded_shape) >= 2 else 1
    num_tiles = num_tiles_per_row * num_tiles_per_col

    # Calculate output shape (padded to tile boundaries)
    output_shape = input_tensor.logical_shape()
    output_padded = pad_to_tile_shape(output_shape._shape)

    # Check execution mode
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    # Perform actual tilize operation if in execute mode
    output_data = None
    if should_execute:
        # Get input data
        if input_tensor.has_data():
            input_data = input_tensor.get_data()
        else:
            # Create dummy data for testing (zeros)
            input_data = _create_empty_data(
                input_tensor.padded_shape()._shape,
                input_tensor.dtype,
                0.0
            )
        # Perform tilize
        output_data = _perform_tilize_operation(
            input_data,
            input_tensor.logical_shape()._shape,
            output_padded._shape,
            pad_value=0.0
        )
    # Track operation if needed
    if should_track:
        element_size = input_tensor.element_size()
        _tracker.track_tilize(
            input_tensor.logical_shape()._shape,
            num_tiles,
            num_tiles_per_row,
            num_tiles_per_col,
            use_multicore,
            element_size
        )

    # Use the same tensor type among the different tensor types as the input tensor
    return type(input_tensor)(
        shape=output_shape,
        dtype=output_dtype or input_tensor.dtype,
        layout=Layout.TILE_LAYOUT,
        memory_config=memory_config or input_tensor.memory_config(),
        padded_shape=output_padded,
        device=input_tensor.device,
        data=output_data
    )


def untilize(input_tensor, memory_config=None, use_multicore=True,
             use_pack_untilize=True, sub_core_grids=None):
    """
    Untilize operation - converts TILE_LAYOUT to ROW_MAJOR_LAYOUT.

    Args:
        input_tensor: Input tensor in TILE_LAYOUT
        memory_config: Optional output memory configuration
        use_multicore: Whether to use multicore (default: True)
        use_pack_untilize: Whether to use pack untilize (default: True)
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with ROW_MAJOR_LAYOUT
    """
    # Validation
    if input_tensor.storage_type() != "DEVICE":
        raise RuntimeError("Operands to untilize need to be on device!")

    if input_tensor.buffer() is None:
        raise RuntimeError("Operands to untilize need to be allocated in buffers on device!")

    if input_tensor.get_layout() != Layout.TILE_LAYOUT:
        raise RuntimeError("Can only untilize tile major data")

    padded_shape = input_tensor.padded_shape()
    tensor_width = padded_shape[-1] if len(padded_shape) >= 1 else 1
    tensor_height = input_tensor.physical_volume() // tensor_width if tensor_width > 0 else 1

    # Validate tile alignment
    if tensor_width % TILE_WIDTH != 0:
        raise RuntimeError("Width must be evenly divisible into tiles")
    if tensor_height % TILE_HEIGHT != 0:
        raise RuntimeError("Height must be evenly divisible into tiles")

    # Check data type for pack untilize
    # Handle both DataType enums and numpy dtypes
    tensor_dtype = input_tensor.dtype
    if isinstance(tensor_dtype, np.dtype):
        tensor_dtype = DataType.from_numpy(tensor_dtype)

    if not use_pack_untilize:
        if tensor_dtype in [DataType.UINT32, DataType.INT32]:
            raise RuntimeError("Pack untilize must be enabled to support uint32/int32 data types")

    # Handle empty tensors
    if input_tensor.physical_volume() == 0:
        # Preserve data if available (empty tensor case)
        output_data = input_tensor.get_data() if input_tensor.has_data() else None

        # Use the same tensor type among the different tensor types as the input tensor
        return type(input_tensor)(
            shape=input_tensor.logical_shape(),
            dtype=input_tensor.dtype,
            layout=Layout.ROW_MAJOR_LAYOUT,
            memory_config=memory_config or input_tensor.memory_config(),
            padded_shape=input_tensor.logical_shape(),
            device=input_tensor.device,
            data=output_data
        )
    # Handle rank > 4
    if input_tensor.logical_shape().rank() > 4:
        original_logical = input_tensor.logical_shape()
        original_padded = input_tensor.padded_shape()
        squeezed = squeeze_from_ND_to_4D(input_tensor)
        result = untilize(squeezed, memory_config, use_multicore,
                         use_pack_untilize, sub_core_grids)
        # Unsqueeze back
        return reshape(result, original_logical, original_padded,
                     memory_config, None, None, sub_core_grids)

    # Calculate number of tiles
    num_tiles_per_row = tensor_width // TILE_WIDTH
    num_tiles_per_col = tensor_height // TILE_HEIGHT
    num_tiles = num_tiles_per_row * num_tiles_per_col

    # Output is unpadded (logical shape)
    output_shape = input_tensor.logical_shape()

    # Check execution mode
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    # Perform actual untilize operation if in execute mode
    output_data = None
    if should_execute:
        # Get input data
        if input_tensor.has_data():
            input_data = input_tensor.get_data()
        else:
            # Create dummy data for testing (zeros)
            input_data = _create_empty_data(
                input_tensor.padded_shape()._shape,
                input_tensor.dtype,
                0.0
            )
        # Perform untilize
        output_data = _perform_untilize_operation(
            input_data,
            input_tensor.padded_shape()._shape,
            output_shape._shape
        )
    # Track operation if needed
    if should_track:
        element_size = input_tensor.element_size()
        _tracker.track_untilize(
            input_tensor.logical_shape()._shape,
            num_tiles,
            num_tiles_per_row,
            num_tiles_per_col,
            use_multicore,
            use_pack_untilize,
            element_size
        )
    # Use the same tensor type among the different tensor types as the input tensor
    return type(input_tensor)(
        shape=output_shape,
        dtype=input_tensor.dtype,
        layout=Layout.ROW_MAJOR_LAYOUT,
        memory_config=memory_config or input_tensor.memory_config(),
        padded_shape=output_shape,  # No padding in row major
        device=input_tensor.device,
        data=output_data
    )


def tilize_with_val_padding(input_tensor, output_padded_shape, pad_value,
                            memory_config=None, output_dtype=None,
                            use_multicore=True, sub_core_grids=None):
    """
    Tilize with value padding - converts ROW_MAJOR_LAYOUT to TILE_LAYOUT with padding.

    Args:
        input_tensor: Input tensor in ROW_MAJOR_LAYOUT
        output_padded_shape: Target padded shape for output
        pad_value: Value to use for padding (float or int)
        memory_config: Optional output memory configuration
        output_dtype: Optional output data type
        use_multicore: Whether to use multicore (default: True)
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with TILE_LAYOUT
    """
    # Validation
    if input_tensor.storage_type() != "DEVICE":
        raise RuntimeError("Operands need to be on device!")

    if input_tensor.buffer() is None:
        raise RuntimeError("Operands need to be allocated in buffers on device!")

    if input_tensor.get_layout() != Layout.ROW_MAJOR_LAYOUT:
        raise RuntimeError("Can only tilize row major data")

    # Check data type - handle both DataType enums and numpy dtypes
    tensor_dtype = input_tensor.dtype
    if isinstance(tensor_dtype, np.dtype):
        tensor_dtype = DataType.from_numpy(tensor_dtype)

    valid_dtypes = [DataType.BFLOAT16, DataType.FLOAT32, DataType.UINT32,
                    DataType.INT32, DataType.UINT16]
    if tensor_dtype not in valid_dtypes:
        raise RuntimeError(
            "Can only tilize bfloat16/float32 or int32/uint32/uint16 tensors"
        )

    # Handle shape input
    if isinstance(output_padded_shape, (list, tuple)):
        output_padded_shape = Shape(output_padded_shape)
    elif not isinstance(output_padded_shape, Shape):
        raise TypeError(f"Invalid output_padded_shape type: {type(output_padded_shape)}")
    # Validate rank
    if output_padded_shape.rank() < 1:
        raise RuntimeError(
            f"Input tensor must be of rank >= 1, but its shape is {output_padded_shape}"
        )
    # Handle empty tensors
    if input_tensor.physical_volume() == 0:
        # Preserve data if available (empty tensor case)
        output_data = input_tensor.get_data() if input_tensor.has_data() else None

        # Use the same tensor type among the different tensor types as the input tensor
        return type(input_tensor)(
            shape=input_tensor.logical_shape(),
            dtype=output_dtype or input_tensor.dtype,
            layout=Layout.TILE_LAYOUT,
            memory_config=memory_config or input_tensor.memory_config(),
            padded_shape=output_padded_shape,
            device=input_tensor.device,
            data=output_data
        )
    # Handle rank > 4
    if input_tensor.logical_shape().rank() > 4:
        squeezed = squeeze_from_ND_to_4D(input_tensor, sub_core_grids)
        squeezed_output_shape = squeeze_shape_to_4D(output_padded_shape)
        result = tilize_with_val_padding(squeezed, squeezed_output_shape, pad_value,
                                        memory_config, output_dtype, use_multicore, sub_core_grids)
        # Unsqueeze back
        return reshape(result, input_tensor.logical_shape(), output_padded_shape,
                     memory_config, None, None, sub_core_grids)
    # Calculate number of tiles
    num_tiles_per_row = output_padded_shape[-1] // TILE_WIDTH
    num_tiles_per_col = output_padded_shape[-2] // TILE_HEIGHT if len(output_padded_shape) >= 2 else 1
    num_tiles = num_tiles_per_row * num_tiles_per_col

    # Output logical shape is same as input
    output_logical = input_tensor.logical_shape()

    # Check execution mode
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    # Perform actual tilize operation if in execute mode
    output_data = None
    if should_execute:
        # Get input data
        if input_tensor.has_data():
            input_data = input_tensor.get_data()
        else:
            # Create dummy data for testing (zeros)
            input_data = _create_empty_data(
                input_tensor.padded_shape()._shape,
                input_tensor.dtype,
                0.0
            )
        # Perform tilize with padding
        output_data = _perform_tilize_operation(
            input_data,
            input_tensor.logical_shape()._shape,
            output_padded_shape._shape,
            pad_value=pad_value
        )
    # Track operation if needed
    if should_track:
        _tracker.track_tilize_with_val_padding(
            input_tensor.logical_shape()._shape,
            output_padded_shape._shape,
            num_tiles
        )

    # Use the same tensor type among the different tensor types as the input tensor
    return type(input_tensor)(
        shape=output_logical,
        dtype=output_dtype or input_tensor.dtype,
        layout=Layout.TILE_LAYOUT,
        memory_config=memory_config or input_tensor.memory_config(),
        padded_shape=output_padded_shape,
        device=input_tensor.device,
        data=output_data
    )


def untilize_with_unpadding(input_tensor, output_tensor_end, memory_config=None,
                            use_multicore=True, use_pack_untilize=True,
                            sub_core_grids=None):
    """
    Untilize with unpadding - converts TILE_LAYOUT to ROW_MAJOR_LAYOUT with unpadding.

    Args:
        input_tensor: Input tensor in TILE_LAYOUT
        output_tensor_end: Shape specifying the end indices for output (unpadded)
        memory_config: Optional output memory configuration
        use_multicore: Whether to use multicore (default: True)
        use_pack_untilize: Whether to use pack untilize (default: True)
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with ROW_MAJOR_LAYOUT
    """
    # Validation
    if input_tensor.storage_type() != "DEVICE":
        raise RuntimeError("Operands need to be on device!")

    if input_tensor.buffer() is None:
        raise RuntimeError("Operands need to be allocated in buffers on device!")

    if input_tensor.get_layout() != Layout.TILE_LAYOUT:
        raise RuntimeError("Can only untilize tile major data")

    # Handle shape input
    if isinstance(output_tensor_end, (list, tuple)):
        output_tensor_end = Shape(output_tensor_end)
    elif not isinstance(output_tensor_end, Shape):
        raise TypeError(f"Invalid output_tensor_end type: {type(output_tensor_end)}")
    # Handle empty tensors
    if input_tensor.physical_volume() == 0:
        output_shape = Shape([end + 1 for end in output_tensor_end._shape])
        # Preserve data if available (empty tensor case)
        output_data = input_tensor.get_data() if input_tensor.has_data() else None
        # Use the same tensor type among the different tensor types as the input tensor
        return type(input_tensor)(
            shape=output_shape,
            dtype=input_tensor.dtype,
            layout=Layout.ROW_MAJOR_LAYOUT,
            memory_config=memory_config or input_tensor.memory_config(),
            padded_shape=output_shape,
            device=input_tensor.device,
            data=output_data
        )
    # Handle rank > 4
    if input_tensor.logical_shape().rank() > 4:
        original_logical = input_tensor.logical_shape()
        squeezed = squeeze_from_ND_to_4D(input_tensor, sub_core_grids)
        squeezed_output_end = squeeze_shape_to_4D(output_tensor_end)
        result = untilize_with_unpadding(squeezed, squeezed_output_end, memory_config,
                                        use_multicore, use_pack_untilize, sub_core_grids)
        # Unsqueeze back
        return reshape(result, original_logical, None, memory_config, None, None, sub_core_grids)

    # Calculate output shape from end indices
    output_shape = Shape([end + 1 for end in output_tensor_end._shape])

    # Calculate number of tiles
    padded_shape = input_tensor.padded_shape()
    num_tiles_per_row = padded_shape[-1] // TILE_WIDTH
    num_tiles_per_col = padded_shape[-2] // TILE_HEIGHT if len(padded_shape) >= 2 else 1
    num_tiles = num_tiles_per_row * num_tiles_per_col
    # Check execution mode
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    # Perform actual untilize operation if in execute mode
    output_data = None
    if should_execute:
        # Get input data
        if input_tensor.has_data():
            input_data = input_tensor.get_data()
        else:
            # Create dummy data for testing (zeros)
            input_data = _create_empty_data(
                input_tensor.padded_shape()._shape,
                input_tensor.dtype,
                0.0
            )
        # Perform untilize with unpadding
        output_data = _perform_untilize_operation(
            input_data,
            padded_shape._shape,
            output_shape._shape
        )
    # Track operation if needed
    if should_track:
        _tracker.track_untilize_with_unpadding(
            input_tensor.logical_shape()._shape,
            output_shape._shape,
            num_tiles
        )

    # Use the same tensor type among the different tensor types as the input tensor
    return type(input_tensor)(
        shape=output_shape,
        dtype=input_tensor.dtype,
        layout=Layout.ROW_MAJOR_LAYOUT,
        memory_config=memory_config or input_tensor.memory_config(),
        padded_shape=output_shape,  # No padding in row major
        device=input_tensor.device,
        data=output_data
    )


# =============================================================================
# Layout operator APIs (SimOp-only; no execution, no _tracker)
# =============================================================================

def tilize_op(input_tensor, use_multicore=True, element_size=2, memory_config=None):
    """
    Create a Tilize SimOp and output tensor (tracking-only; no execution).
    Output logical shape = input shape; padded = last two dims rounded up to tile.
    """
    assert input_tensor.device is not None, "tilize_op requires input_tensor on device"
    op_name = generate_new_op_name()
    out_logical = input_tensor.logical_shape()
    out_padded = pad_to_tile_shape(out_logical._shape)
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_logical._shape,
        dtype=input_tensor.dtype,
        layout=Layout.TILE_LAYOUT,
        padded_shape=out_padded._shape,
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'Tilize',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {'use_multicore': use_multicore, 'element_size': element_size},
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    input_tensor.device.add_op(opobj)
    return out_tensor


def untilize_op(input_tensor, use_multicore=True, use_pack_untilize=True, element_size=2, memory_config=None):
    """
    Create an Untilize SimOp and output tensor (tracking-only; no execution).
    Output ROW_MAJOR, same logical shape as input.
    """
    assert input_tensor.device is not None, "untilize_op requires input_tensor on device"
    op_name = generate_new_op_name()
    out_shape = input_tensor.logical_shape()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_shape._shape,
        dtype=input_tensor.dtype,
        layout=Layout.ROW_MAJOR_LAYOUT,
        padded_shape=out_shape._shape,
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'Untilize',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {
            'use_multicore': use_multicore,
            'use_pack_untilize': use_pack_untilize,
            'element_size': element_size,
        },
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    input_tensor.device.add_op(opobj)
    return out_tensor


def tilize_with_val_padding_op(input_tensor, output_padded_shape, pad_value,
                                use_multicore=True, element_size=2, memory_config=None):
    """
    Create a TilizeWithValPadding SimOp and output tensor (tracking-only; no execution).
    output_padded_shape and pad_value are stored in op attrs.
    """
    assert input_tensor.device is not None, "tilize_with_val_padding_op requires input_tensor on device"
    if isinstance(output_padded_shape, Shape):
        output_padded_shape = output_padded_shape._shape
    output_padded_shape = list(output_padded_shape)
    op_name = generate_new_op_name()
    out_logical = input_tensor.logical_shape()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_logical._shape,
        dtype=input_tensor.dtype,
        layout=Layout.TILE_LAYOUT,
        padded_shape=output_padded_shape,
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'TilizeWithValPadding',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {
            'output_padded_shape': output_padded_shape,
            'pad_value': pad_value,
            'use_multicore': use_multicore,
            'element_size': element_size,
        },
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    input_tensor.device.add_op(opobj)
    return out_tensor


def untilize_with_unpadding_op(input_tensor, output_shape,
                                use_multicore=True, use_pack_untilize=True, element_size=2, memory_config=None):
    """
    Create an UntilizeWithValUnpadding SimOp and output tensor (tracking-only; no execution).
    (Python name ``untilize_with_unpadding_op`` is historical; optype matches hardware naming.)
    output_shape is the logical output shape (list or tuple); stored in op attrs.
    """
    assert input_tensor.device is not None, "untilize_with_unpadding_op requires input_tensor on device"
    if isinstance(output_shape, Shape):
        output_shape = output_shape._shape
    output_shape = list(output_shape)
    op_name = generate_new_op_name()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=output_shape,
        dtype=input_tensor.dtype,
        layout=Layout.ROW_MAJOR_LAYOUT,
        padded_shape=output_shape,
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'UntilizeWithValUnpadding',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {
            'output_shape': output_shape,
            'use_multicore': use_multicore,
            'use_pack_untilize': use_pack_untilize,
            'element_size': element_size,
        },
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    input_tensor.device.add_op(opobj)
    return out_tensor


def permute_op(input_tensor, dims, memory_config=None):
    """
    Create a Permute SimOp and output tensor (tracking-only; no execution).

    Parity with ``ttnn.permute(tensor, dims)`` on the data movement pattern, but
    records optype **Permute** on the device graph. The generic front-end
    ``ttnn.permute`` in ``op.py`` still lowers to **Transpose** for other call sites.

    Requires ``input_tensor`` on a device (same contract as ``tilize_op``).
    ``dims`` may be a list or tuple of dimension indices (full permutation).
    """
    assert input_tensor.device is not None, "permute_op requires input_tensor on device"
    perm = list(dims)
    in_shape = list(input_tensor.logical_shape()._shape)
    if len(perm) != len(in_shape):
        raise ValueError(
            f"permute_op: len(dims)={len(perm)} must match input rank {len(in_shape)}"
        )
    out_shape = [in_shape[i] for i in perm]
    op_name = generate_new_op_name()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_shape,
        dtype=input_tensor.dtype,
        layout=input_tensor.layout,
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'Permute',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {'perm': perm},
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    input_tensor.device.add_op(opobj)
    return out_tensor


def to_layout(tensor, layout, dtype=None, memory_config=None, sub_core_grids=None):
    """
    Convert tensor to specified layout.
    Args:
        tensor: Input tensor
        layout: Target layout (ROW_MAJOR_LAYOUT or TILE_LAYOUT)
        dtype: Optional output data type (only for TILE_LAYOUT conversion)
        memory_config: Optional output memory configuration
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with requested layout
    """
    # Check execution mode
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)

    # When not executing, we either use operator-style SimOps (_op) or _tracker; tracking is done per-branch below.

    # If already in requested layout
    if tensor.get_layout() == layout:
        # Check if dtype is specified when converting to ROW_MAJOR_LAYOUT (error case)
        if layout == Layout.ROW_MAJOR_LAYOUT and dtype is not None:
            raise RuntimeError("dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!")

        if dtype is not None and dtype != tensor.dtype:
            import warnings
            warnings.warn(
                "ttnn::to_layout: dtype is specified but the tensor is already in the requested layout! "
                "So, the dtype won't be changed!"
            )
        if memory_config is not None:
            current_mem_config = tensor.memory_config()
            if memory_config != current_mem_config:
                import warnings
                warnings.warn(
                    "ttnn::to_layout: memory_config is specified but the tensor is already in the requested layout! "
                    "So, the memory_config won't be changed!"
                )
        return tensor
    # Validate supported layouts
    supported_layouts = [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT]
    if layout not in supported_layouts:
        raise RuntimeError(
            f"ttnn::to_layout: Unsupported layout conversion from {tensor.get_layout()} to {layout}!"
        )

    # Get output memory config
    output_memory_config = memory_config or tensor.memory_config() or DRAM_MEMORY_CONFIG

    # Handle device tensors
    # Check if tensor has a buffer to distinguish real device tensors from host tensors (SHIM) with device reference.
    # ttsim Tensor sets _buffer to a placeholder when on device and implements buffer(); other types may have a real buffer.
    has_buffer = False
    if hasattr(tensor, 'buffer'):
        try:
            has_buffer = tensor.buffer() is not None
        except Exception:
            has_buffer = False

    is_device_tensor = tensor.storage_type() == "DEVICE" and has_buffer

    if is_device_tensor:
        needs_padding_change = requires_padding_change(tensor, layout)

        logger.debug(
            "to_layout: needs_padding_change {} for tensor {} to layout {}",
            needs_padding_change,
            tensor.name,
            layout,
        )
        if not needs_padding_change:
            # Simple conversion without padding change
            if layout == Layout.ROW_MAJOR_LAYOUT:
                if dtype is not None:
                    raise RuntimeError("dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!")
                if should_execute:
                    return untilize(tensor, output_memory_config, True, True, sub_core_grids)
                elif should_track and tensor.device is not None and isinstance(tensor, Tensor):
                    logger.debug(
                        "to_layout: choosing untilize_op (device tensor, no padding change, TILE->ROW_MAJOR; logical_shape {} == padded_shape {})",
                        tensor.logical_shape()._shape,
                        tensor.padded_shape()._shape,
                    )
                    return untilize_op(tensor, use_multicore=True, use_pack_untilize=True, element_size=2, memory_config=output_memory_config)
                else:
                    # Not using untilize_op: input is not a ttsim Tensor (e.g. TensorProxy) or has no device; preserve API with tracker + manual tensor.
                    if should_track:
                        _tracker.track_to_layout(tensor.get_layout(), layout)
                    return type(tensor)(
                        shape=tensor.logical_shape(),
                        dtype=tensor.dtype,
                        layout=layout,
                        memory_config=output_memory_config,
                        padded_shape=tensor.logical_shape(),
                        device=tensor.device
                    )
            elif layout == Layout.TILE_LAYOUT:
                if should_execute:
                    return tilize(tensor, output_memory_config, dtype, True, False, sub_core_grids)
                elif should_track and tensor.device is not None and isinstance(tensor, Tensor):
                    logger.debug(
                        "to_layout: choosing tilize_op (device tensor, no padding change, ROW_MAJOR->TILE; logical_shape already tile-aligned)"
                    )
                    return tilize_op(tensor, use_multicore=True, element_size=2, memory_config=output_memory_config)
                else:
                    # Not using tilize_op: input is not a ttsim Tensor (e.g. TensorProxy) or has no device; preserve API with tracker + manual tensor.
                    if should_track:
                        _tracker.track_to_layout(tensor.get_layout(), layout)
                    return type(tensor)(
                        shape=tensor.logical_shape(),
                        dtype=dtype or tensor.dtype,
                        layout=layout,
                        memory_config=output_memory_config,
                        padded_shape=pad_to_tile_shape(tensor.logical_shape()._shape),
                        device=tensor.device
                    )
        else:
            # Conversion with padding change
            if layout == Layout.ROW_MAJOR_LAYOUT:
                if dtype is not None and dtype != tensor.dtype:
                    raise RuntimeError(
                        "dtype cannot be different from tensor dtype when converting to ROW_MAJOR_LAYOUT on device!"
                    )

                # Calculate output shape (unpadded)
                output_shape = tensor.logical_shape()
                output_tensor_end = Shape([dim - 1 for dim in output_shape._shape])

                if should_execute:
                    result = untilize_with_unpadding(
                        tensor, output_tensor_end, output_memory_config, True, True, sub_core_grids
                    )
                    return reshape(result, output_shape, None, None, None, sub_core_grids)
                elif should_track and tensor.device is not None and isinstance(tensor, Tensor):
                    logger.debug(
                        "to_layout: choosing untilize_with_unpadding_op (device tensor, padding change, TILE->ROW_MAJOR; padded_shape!=logical_shape, padded_shape={}, logical_shape={}, output_shape={})",
                        tensor.padded_shape()._shape,
                        tensor.logical_shape()._shape,
                        output_shape._shape,
                    )
                    return untilize_with_unpadding_op(
                        tensor, output_shape._shape, use_multicore=True, use_pack_untilize=True, element_size=2, memory_config=output_memory_config
                    )
                else:
                    # Not using untilize_with_unpadding_op: input is not a ttsim Tensor or has no device; preserve API with tracker + manual tensor.
                    if should_track:
                        _tracker.track_to_layout(tensor.get_layout(), layout)
                    return type(tensor)(
                        shape=output_shape,
                        dtype=tensor.dtype,
                        layout=layout,
                        memory_config=output_memory_config,
                        padded_shape=output_shape,
                        device=tensor.device
                    )
            elif layout == Layout.TILE_LAYOUT:
                # Calculate padded output shape
                logical_shape = tensor.logical_shape()
                padded_output_shape = pad_to_tile_shape(logical_shape._shape)
                # Check for height sharded tensors (if memory_layout is set)
                mem_config = tensor.memory_config()
                if hasattr(mem_config, 'memory_layout') and mem_config.memory_layout == "HEIGHT_SHARDED":
                    # Use pad + tilize for height sharded
                    # Not using tilize_op / tilize_with_val_padding_op: this path is pad-then-tilize (two ops); there is no layout Pad op, and we do not create an intermediate padded tensor in track-only mode.
                    padding = [[0, 0], [0, 0]]
                    if len(logical_shape) >= 2:
                        padding.append([0, padded_output_shape[-2] - logical_shape[-2]])
                        padding.append([0, padded_output_shape[-1] - logical_shape[-1]])
                    else:
                        padding = [[0, 0], [0, 0]]
                    if should_execute:
                        padded_tensor = pad(tensor, padding, 0, output_memory_config)
                        return tilize(padded_tensor, output_memory_config, dtype, True, False, sub_core_grids)
                    else:
                        if should_track:
                            _tracker.track_to_layout(tensor.get_layout(), layout)
                        return type(tensor)(
                            shape=logical_shape,
                            dtype=dtype or tensor.dtype,
                            layout=layout,
                            memory_config=output_memory_config,
                            padded_shape=padded_output_shape,
                            device=tensor.device
                        )
                else:
                    # Use tilize_with_val_padding
                    pad_value = 0.0 if tensor.dtype in [DataType.BFLOAT16, DataType.FLOAT32] else 0
                    if should_execute:
                        return tilize_with_val_padding(
                            tensor, padded_output_shape, pad_value, output_memory_config, dtype, True, sub_core_grids
                        )
                    elif should_track and tensor.device is not None and isinstance(tensor, Tensor):
                        logger.debug(
                            "to_layout: choosing tilize_with_val_padding_op (device tensor, padding change, ROW_MAJOR->TILE; logical not tile-aligned, padded_output_shape={}, pad_value={})",
                            padded_output_shape._shape,
                            pad_value,
                        )
                        return tilize_with_val_padding_op(
                            tensor, padded_output_shape._shape, pad_value, use_multicore=True, element_size=2, memory_config=output_memory_config
                        )
                    else:
                        # Not using tilize_with_val_padding_op: input is not a ttsim Tensor or has no device; preserve API with tracker + manual tensor.
                        if should_track:
                            _tracker.track_to_layout(tensor.get_layout(), layout)
                        return type(tensor)(
                            shape=logical_shape,
                            dtype=dtype or tensor.dtype,
                            layout=layout,
                            memory_config=output_memory_config,
                            padded_shape=padded_output_shape,
                            device=tensor.device
                        )
    else:
        # Host tensor conversion: not using tiling _op (tilize_op, untilize_op, etc.); layout SimOps are for device op graph only; host path uses in-process conversion and tracker.
        if dtype is not None:
            raise RuntimeError("dtype cannot be specified when converting layout on host!")

        # Perform actual data conversion if in execute mode
        output_data = None
        if should_execute and tensor.has_data():
            input_data = tensor.get_data()
            input_layout = tensor.get_layout()
            logical_shape = tensor.logical_shape()

            if layout == Layout.ROW_MAJOR_LAYOUT:
                # Convert from tile layout to row major
                if input_layout == Layout.TILE_LAYOUT:
                    # Get padded shape for untilize
                    input_padded_shape = tensor.padded_shape()
                    output_data = _perform_untilize_operation(
                        input_data,
                        input_padded_shape._shape,
                        logical_shape._shape
                    )
                else:
                    # Already in row major, just use the data
                    output_data = input_data
            elif layout == Layout.TILE_LAYOUT:
                # Convert from row major to tile layout
                if input_layout == Layout.ROW_MAJOR_LAYOUT:
                    # Calculate padded output shape
                    padded_output_shape = pad_to_tile_shape(logical_shape._shape)
                    pad_value = 0.0 if tensor.dtype in [DataType.BFLOAT16, DataType.FLOAT32] else 0
                    output_data = _perform_tilize_operation(
                        input_data,
                        logical_shape._shape,
                        padded_output_shape._shape,
                        pad_value=pad_value
                    )
                else:
                    # Already in tile layout, just use the data
                    output_data = input_data

        if not requires_padding_change(tensor, layout):
            # Simple conversion - just change layout attribute (host path; tiling _op not used—see host comment above).
            if should_track:
                _tracker.track_to_layout(tensor.get_layout(), layout)
            return type(tensor)(
                shape=tensor.logical_shape(),
                dtype=tensor.dtype,
                layout=layout,
                memory_config=output_memory_config,
                padded_shape=tensor.padded_shape(),
                device=tensor.device,
                data=output_data
            )
        else:
            if layout == Layout.ROW_MAJOR_LAYOUT:
                # Host path with padding change (tiling _op not used—see host comment above).
                if should_track:
                    _tracker.track_to_layout(tensor.get_layout(), layout)
                # Convert to row major and unpad
                result = type(tensor)(
                    shape=tensor.logical_shape(),
                    dtype=tensor.dtype,
                    layout=layout,
                    memory_config=output_memory_config,
                    padded_shape=tensor.logical_shape(),  # Unpadded
                    device=tensor.device,
                    data=output_data
                )
                return reshape(result, tensor.logical_shape(), None, None, None, sub_core_grids)
            elif layout == Layout.TILE_LAYOUT:
                # Host path with padding change (tiling _op not used—see host comment above).
                if should_track:
                    _tracker.track_to_layout(tensor.get_layout(), layout)
                # Pad and convert to tile
                logical_shape = tensor.logical_shape()
                padded_output_shape = pad_to_tile_shape(logical_shape._shape)

                # Convert to tile layout
                return type(tensor)(
                    shape=logical_shape,
                    dtype=tensor.dtype,
                    layout=layout,
                    memory_config=output_memory_config,
                    padded_shape=padded_output_shape,
                    device=tensor.device,
                    data=output_data
                )

    raise RuntimeError(f"ttnn::to_layout: Unsupported output layout: {layout}!")


# Convenience function for tilize_with_zero_padding
def tilize_with_zero_padding(input_tensor, memory_config=None, output_dtype=None,
                            use_multicore=True, sub_core_grids=None):
    """
    Tilize with zero padding - convenience function that auto-calculates padded shape.
    Args:
        input_tensor: Input tensor in ROW_MAJOR_LAYOUT
        memory_config: Optional output memory configuration
        output_dtype: Optional output data type
        use_multicore: Whether to use multicore (default: True)
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with TILE_LAYOUT
    """
    # Calculate padded shape
    padded_shape = input_tensor.padded_shape()
    padded_shape = pad_to_tile_shape(padded_shape._shape)

    # Determine pad value based on dtype - handle both DataType enums and numpy dtypes
    tensor_dtype = input_tensor.dtype
    if isinstance(tensor_dtype, np.dtype):
        tensor_dtype = DataType.from_numpy(tensor_dtype)

    if tensor_dtype in [DataType.BFLOAT16, DataType.FLOAT32]:
        pad_value = 0.0
    else:
        pad_value = 0

    return tilize_with_val_padding(
        input_tensor, padded_shape, pad_value, memory_config, output_dtype,
        use_multicore, sub_core_grids
    )
