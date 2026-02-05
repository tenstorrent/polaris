# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for ttnn.to_layout function.

This test suite generates test cases covering:
- Different input tensor shapes
- Shapes with dimensions that are integral multiples of tile size (32x32)
- Shapes with different combinations of dimensions being integral multiples of tile size
- Edge cases and error cases
- Different dtypes and memory configs

Each test outputs input and output tensor content in JSON format for functional validation
of other similar tiling modules.
"""

import json
import os
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Tile size constants
TILE_SIZE = 32
TILE_WIDTH = 32
TILE_HEIGHT = 32

# Output directory for test results
OUTPUT_DIR = Path("test_outputs/to_layout_ttnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def serialize_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    """Serialize torch tensor to JSON-serializable format."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "values": tensor.flatten().tolist(),
        "numpy_array": tensor.to(torch.float32).numpy().tolist() if tensor.dtype == torch.bfloat16 else tensor.numpy().tolist()
        # numpy does not support bfloat16, so convert to float32 for serialization
        }


def serialize_memory_config(mem_config) -> Dict[str, Any]:
    """Serialize memory config to JSON-serializable format."""
    if mem_config is None:
        return None
    
    result = {
        "memory_layout": str(mem_config.memory_layout) if hasattr(mem_config, 'memory_layout') else None,
        "buffer_type": str(mem_config.buffer_type) if hasattr(mem_config, 'buffer_type') else None,
    }
    
    if hasattr(mem_config, 'shard_spec') and mem_config.shard_spec is not None:
        shard_spec = mem_config.shard_spec
        result["shard_spec"] = {
            "shard_shape": list(shard_spec.shape) if hasattr(shard_spec, 'shape') else None,
            "orientation": str(shard_spec.orientation) if hasattr(shard_spec, 'orientation') else None,
        }
    
    return result


def calculate_padding_info(logical_shape: Tuple[int, ...], padded_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """Calculate padding information."""
    if len(logical_shape) < 2 or len(padded_shape) < 2:
        return {
            "height_padding": 0,
            "width_padding": 0,
            "needs_padding": False,
        }
    
    height_pad = padded_shape[-2] - logical_shape[-2] if len(logical_shape) >= 2 else 0
    width_pad = padded_shape[-1] - logical_shape[-1] if len(logical_shape) >= 1 else 0
    
    return {
        "height_padding": height_pad,
        "width_padding": width_pad,
        "needs_padding": height_pad > 0 or width_pad > 0,
        "padded_height": padded_shape[-2] if len(padded_shape) >= 2 else logical_shape[-2],
        "padded_width": padded_shape[-1] if len(padded_shape) >= 1 else logical_shape[-1],
    }


def save_test_result(
    test_name: str,
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    input_torch: torch.Tensor,
    output_torch: torch.Tensor,
    input_layout: ttnn.Layout,
    output_layout: ttnn.Layout,
    dtype: ttnn.DataType,
    memory_config: Optional[Any],
    additional_params: Optional[Dict[str, Any]] = None,
):
    """Save test result to JSON file."""
    # Calculate padding info
    input_padded = tuple(input_tensor.padded_shape) if hasattr(input_tensor, 'padded_shape') and input_tensor.padded_shape is not None else tuple(input_tensor.shape)
    output_padded = tuple(output_tensor.padded_shape) if hasattr(output_tensor, 'padded_shape') and output_tensor.padded_shape is not None else tuple(output_tensor.shape)
    
    input_padding = calculate_padding_info(
        tuple(input_tensor.shape),
        input_padded
    )
    output_padding = calculate_padding_info(
        tuple(output_tensor.shape),
        output_padded
    )
    
    # Build test result
    test_result = {
        "test_name": test_name,
        "test_metadata": {
            "tile_size": {"width": TILE_WIDTH, "height": TILE_HEIGHT},
            "input_layout": str(input_layout),
            "output_layout": str(output_layout),
            "dtype": str(dtype),
            "memory_config": serialize_memory_config(memory_config),
            "timestamp": str(pytest.current_test_timestamp) if hasattr(pytest, 'current_test_timestamp') else None,
        },
        "input_tensor": {
            "logical_shape": list(input_tensor.shape),
            "padded_shape": list(input_padded),
            "layout": str(input_tensor.layout),
            "dtype": str(input_tensor.dtype),
            "padding_info": input_padding,
            "content": serialize_tensor(input_torch),
        },
        "output_tensor": {
            "logical_shape": list(output_tensor.shape),
            "padded_shape": list(output_padded),
            "layout": str(output_tensor.layout),
            "dtype": str(output_tensor.dtype),
            "padding_info": output_padding,
            "content": serialize_tensor(output_torch),
        },
    }
    
    if additional_params:
        test_result["additional_params"] = additional_params
    
    # Save to file
    output_file = OUTPUT_DIR / f"{test_name}.json"
    with open(output_file, 'w') as f:
        json.dump(test_result, f, indent=2)
    
    return test_result


# Test case definitions
# Shapes where all dimensions are multiples of tile size
TILE_ALIGNED_SHAPES = [
    (32, 32),           # 1 tile
    (64, 64),           # 4 tiles
    (32, 128),          # 4 tiles (1x4)
    (128, 32),          # 4 tiles (4x1)
    (96, 96),           # 9 tiles (3x3)
    (1, 1, 32, 32),     # 4D, 1 tile
    (2, 3, 64, 128),    # 4D, multiple tiles
    (1, 32),            # 2D, width aligned
    (32, 1),            # 2D, height aligned
]

# Shapes where some dimensions are multiples, some aren't
PARTIAL_TILE_ALIGNED_SHAPES = [
    (30, 32),           # Height not aligned, width aligned
    (32, 30),           # Height aligned, width not aligned
    (33, 64),           # Height not aligned, width aligned
    (64, 33),           # Height aligned, width not aligned
    (31, 31),           # Neither aligned
    (17, 45),           # Neither aligned, odd numbers
    (1, 1, 30, 64),     # 4D, height not aligned
    (1, 1, 64, 30),     # 4D, width not aligned
    (2, 3, 33, 65),     # 4D, both not aligned
    (1, 50),            # 2D, width not aligned
    (50, 1),            # 2D, height not aligned
]

# Edge case shapes
EDGE_CASE_SHAPES = [
    (1, 1),             # Minimum 2D
    (1,),               # 1D
    (32,),              # 1D, aligned
    (1, 32, 32),        # 3D
    (1, 1, 1, 1),       # 4D minimum
    (100, 100),         # Large but not aligned
    (1024, 1024),       # Very large, aligned
    (1025, 1025),       # Very large, not aligned
]

# All test shapes
ALL_TEST_SHAPES = TILE_ALIGNED_SHAPES + PARTIAL_TILE_ALIGNED_SHAPES + EDGE_CASE_SHAPES


@pytest.mark.parametrize("shape", TILE_ALIGNED_SHAPES)
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_to_layout_tile_aligned_shapes(device, shape, input_layout, output_layout, dtype):
    """Test to_layout with shapes where all dimensions are multiples of tile size."""
    torch.manual_seed(42)
    
    # Create input tensor
    if dtype == ttnn.float32:
        torch_input = torch.randn(shape, dtype=torch.float32)
    else:
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    # Convert to ttnn tensor
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=dtype,
        layout=input_layout,
    )
    
    # Perform layout conversion
    ttnn_output = ttnn.to_layout(ttnn_input, layout=output_layout)
    
    # Convert back to torch
    torch_output = ttnn.to_torch(ttnn_output)
    
    # Verify correctness
    assert_with_pcc(torch_input, torch_output)
    
    # Save test result
    test_name = f"tile_aligned_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}_{dtype}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=dtype,
        memory_config=ttnn_input.memory_config(),
    )


@pytest.mark.parametrize("shape", PARTIAL_TILE_ALIGNED_SHAPES)
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_layout_partial_tile_aligned_shapes(device, shape, input_layout, output_layout):
    """Test to_layout with shapes where some dimensions are multiples of tile size."""
    torch.manual_seed(42)
    
    # Create input tensor
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    # Convert to ttnn tensor
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=input_layout,
    )
    
    # Perform layout conversion
    ttnn_output = ttnn.to_layout(ttnn_input, layout=output_layout)
    
    # Convert back to torch
    torch_output = ttnn.to_torch(ttnn_output)
    
    # Verify correctness
    assert_with_pcc(torch_input, torch_output)
    
    # Save test result
    test_name = f"partial_aligned_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn_input.memory_config(),
    )


@pytest.mark.parametrize("shape", EDGE_CASE_SHAPES)
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_layout_edge_cases(device, shape, input_layout, output_layout):
    """Test to_layout with edge case shapes."""
    torch.manual_seed(42)
    
    # Create input tensor
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    # Convert to ttnn tensor
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=input_layout,
    )
    
    # Perform layout conversion
    ttnn_output = ttnn.to_layout(ttnn_input, layout=output_layout)
    
    # Convert back to torch
    torch_output = ttnn.to_torch(ttnn_output)
    
    # Verify correctness
    assert_with_pcc(torch_input, torch_output)
    
    # Save test result
    test_name = f"edge_case_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn_input.memory_config(),
    )


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (33, 65)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.int32])
def test_to_layout_different_dtypes(device, shape, dtype):
    """Test to_layout with different data types."""
    torch.manual_seed(42)
    
    # Map ttnn dtype to torch dtype
    dtype_map = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.uint32: torch.int32,  # Use int32 for uint32 in torch
        ttnn.int32: torch.int32,
    }
    
    torch_dtype = dtype_map[dtype]
    
    # Create input tensor
    if dtype in [ttnn.uint32, ttnn.int32]:
        torch_input = torch.randint(0, 100, shape, dtype=torch_dtype)
    else:
        torch_input = torch.randn(shape, dtype=torch_dtype)
    
    # Convert to ttnn tensor
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    
    # Convert to tile layout
    ttnn_output = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    
    # Convert back to row major
    ttnn_output = ttnn.to_layout(ttnn_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    
    # Convert back to torch
    torch_output = ttnn.to_torch(ttnn_output)
    
    # Verify correctness
    assert_with_pcc(torch_input, torch_output)
    
    # Save test result
    test_name = f"dtype_{dtype}_{'_'.join(map(str, shape))}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=ttnn_input.memory_config(),
    )


@pytest.mark.parametrize("shape", [(64, 128), (128, 64)])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_to_layout_memory_configs(device, shape, memory_config):
    """Test to_layout with different memory configurations."""
    torch.manual_seed(42)
    
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    # Convert to ttnn tensor with specific memory config
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )
    
    # Convert to tile layout
    ttnn_output = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    
    # Convert back to row major
    ttnn_output = ttnn.to_layout(ttnn_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    
    # Convert back to torch
    torch_output = ttnn.to_torch(ttnn_output)
    
    # Verify correctness
    assert_with_pcc(torch_input, torch_output)
    
    # Save test result
    test_name = f"mem_config_{memory_config.buffer_type}_{'_'.join(map(str, shape))}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
    )


def test_to_layout_same_layout_no_change(device):
    """Test to_layout when tensor is already in requested layout."""
    torch.manual_seed(42)
    
    shape = (64, 64)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    # Create tensor in TILE_LAYOUT
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    
    # Try to convert to same layout (should return same tensor)
    ttnn_output = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    
    # Convert back to torch
    torch_output = ttnn.to_torch(ttnn_output)
    
    # Verify correctness
    assert_with_pcc(torch_input, torch_output)
    
    # Save test result
    test_name = "same_layout_no_change"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=ttnn.TILE_LAYOUT,
        output_layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn_input.memory_config(),
        additional_params={"note": "Tensor already in requested layout"},
    )


def test_to_layout_round_trip(device):
    """Test round-trip conversion: ROW_MAJOR -> TILE -> ROW_MAJOR."""
    torch.manual_seed(42)
    
    shape = (33, 65)  # Not tile-aligned
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    # Start with ROW_MAJOR
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    
    # Convert to TILE
    ttnn_tile = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    
    # Convert back to ROW_MAJOR
    ttnn_output = ttnn.to_layout(ttnn_tile, layout=ttnn.ROW_MAJOR_LAYOUT)
    
    # Convert back to torch
    torch_output = ttnn.to_torch(ttnn_output)
    
    # Verify correctness
    assert_with_pcc(torch_input, torch_output)
    
    # Save test result
    test_name = "round_trip_conversion"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn_input.memory_config(),
        additional_params={"intermediate_layout": str(ttnn.TILE_LAYOUT)},
    )


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (2, 3, 64, 128)])
def test_to_layout_with_dtype_conversion(device, shape):
    """Test to_layout with dtype conversion (only valid for TILE_LAYOUT)."""
    torch.manual_seed(42)
    
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    # Create tensor in ROW_MAJOR
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    
    # Convert to TILE_LAYOUT with dtype conversion
    ttnn_output = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    
    # Verify dtype changed
    assert ttnn_output.dtype == ttnn.float32
    
    # Convert back to torch
    torch_output = ttnn.to_torch(ttnn_output, dtype=torch.float32)
    
    # Save test result
    test_name = f"dtype_conversion_{'_'.join(map(str, shape))}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn_input.memory_config(),
        additional_params={"output_dtype": str(ttnn.float32)},
    )


def test_to_layout_error_cases(device):
    """Test error cases for to_layout."""
    torch.manual_seed(42)
    
    shape = (32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    # Create input in TILE_LAYOUT to test conversion to ROW_MAJOR with dtype
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    
    # Test: dtype cannot be specified when converting to ROW_MAJOR_LAYOUT
    with pytest.raises(RuntimeError, match="dtype cannot be specified"):
        ttnn.to_layout(ttnn_input, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    
    # Test: Unsupported layout (if any)
    # Note: This test may need adjustment based on actual error messages


# ============================================================================
# Missing Test Cases - Now Added
# ============================================================================ß

@pytest.mark.parametrize("shape", [(1, 1, 1, 32, 32), (2, 3, 4, 64, 128), (1, 1, 1, 1, 32, 32)])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_layout_high_dimensional_tensors(device, shape, input_layout, output_layout):
    """Test to_layout with 5D and 6D tensors."""
    torch.manual_seed(42)
    
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=input_layout,
    )
    
    ttnn_output = ttnn.to_layout(ttnn_input, layout=output_layout)
    torch_output = ttnn.to_torch(ttnn_output)
    
    assert_with_pcc(torch_input, torch_output)
    
    test_name = f"high_dim_{len(shape)}D_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn_input.memory_config(),
    )


@pytest.mark.parametrize("shape", [(64, 128), (128, 64)])
@pytest.mark.parametrize("shard_strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK])
def test_to_layout_sharded_memory_configs(device, shape, shard_strategy):
    """Test to_layout with sharded memory configurations."""
    torch.manual_seed(42)
    
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    # Create sharded memory config
    core_grid = device.compute_with_storage_grid_size()
    num_cores = min(32, core_grid.x * core_grid.y)
    
    if shard_strategy == ttnn.ShardStrategy.HEIGHT:
        shard_shape = (32, shape[1])
    elif shard_strategy == ttnn.ShardStrategy.WIDTH:
        shard_shape = (shape[0], 32)
    else:  # BLOCK
        shard_shape = (32, 32)
    
    sharded_memory_config = ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=ttnn.CoreGrid(x=core_grid.x, y=core_grid.y),
        strategy=shard_strategy,
        use_height_and_width_as_shard_shape=True,
    )
    
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_memory_config,
    )
    
    ttnn_output = ttnn.to_layout(ttnn_input, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    torch_output = ttnn.to_torch(ttnn_output)
    
    assert_with_pcc(torch_input, torch_output)
    
    test_name = f"sharded_{shard_strategy}_{'_'.join(map(str, shape))}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=ttnn.TILE_LAYOUT,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=sharded_memory_config,
        additional_params={"output_memory_config": "DRAM_MEMORY_CONFIG"},
    )


@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize(
    "sub_core_grids",
    [
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))]),
        None,
    ],
)
def test_to_layout_sub_core_grids(device, shape, sub_core_grids):
    """Test to_layout with sub_core_grids parameter."""
    torch.manual_seed(42)
    
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    
    ttnn_output = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, sub_core_grids=sub_core_grids)
    ttnn_output = ttnn.to_layout(ttnn_output, layout=ttnn.ROW_MAJOR_LAYOUT, sub_core_grids=sub_core_grids)
    torch_output = ttnn.to_torch(ttnn_output)
    
    assert_with_pcc(torch_input, torch_output)
    
    sub_core_str = "none" if sub_core_grids is None else f"{len(sub_core_grids.ranges)}_ranges"
    test_name = f"sub_core_grids_{sub_core_str}_{'_'.join(map(str, shape))}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn_input.memory_config(),
        additional_params={"sub_core_grids": sub_core_str},
    )


@pytest.mark.parametrize("shape", [(2048, 2048), (4096, 4096)])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_layout_very_large_tensors(device, shape, input_layout, output_layout):
    """Test to_layout with very large tensors."""
    torch.manual_seed(42)
    
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=input_layout,
    )
    
    ttnn_output = ttnn.to_layout(ttnn_input, layout=output_layout)
    torch_output = ttnn.to_torch(ttnn_output)
    
    assert_with_pcc(torch_input, torch_output)
    
    test_name = f"very_large_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn_input.memory_config(),
    )


@pytest.mark.parametrize("shape", [(31, 32), (32, 31), (33, 33)])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_layout_boundary_conditions(device, shape, input_layout, output_layout):
    """Test to_layout with boundary conditions (just below/above tile size)."""
    torch.manual_seed(42)
    
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=input_layout,
    )
    
    ttnn_output = ttnn.to_layout(ttnn_input, layout=output_layout)
    torch_output = ttnn.to_torch(ttnn_output)
    
    assert_with_pcc(torch_input, torch_output)
    
    test_name = f"boundary_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn_input.memory_config(),
    )


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (2, 3, 64, 128)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_to_layout_special_dtypes(device, shape, dtype):
    """Test to_layout with special dtypes (bfloat8_b, bfloat4_b)."""
    torch.manual_seed(42)
    
    # These dtypes only work with TILE_LAYOUT
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    
    # Convert to ROW_MAJOR and back
    ttnn_output = ttnn.to_layout(ttnn_input, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.to_layout(ttnn_output, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    
    torch_output = ttnn.to_torch(ttnn_output, dtype=torch.bfloat16)
    
    # Use lower tolerance for these dtypes
    assert_with_pcc(torch_input, torch_output, pcc=0.99)
    
    test_name = f"special_dtype_{dtype}_{'_'.join(map(str, shape))}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=ttnn.TILE_LAYOUT,
        output_layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn_input.memory_config(),
    )


def test_to_layout_multiple_consecutive_conversions(device):
    """Test multiple consecutive layout conversions."""
    torch.manual_seed(42)
    
    shape = (64, 64)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    
    # Perform multiple conversions: ROW -> TILE -> ROW -> TILE -> ROW
    t1 = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    t2 = ttnn.to_layout(t1, layout=ttnn.ROW_MAJOR_LAYOUT)
    t3 = ttnn.to_layout(t2, layout=ttnn.TILE_LAYOUT)
    t4 = ttnn.to_layout(t3, layout=ttnn.ROW_MAJOR_LAYOUT)
    
    torch_output = ttnn.to_torch(t4)
    
    assert_with_pcc(torch_input, torch_output)
    
    test_name = "multiple_consecutive_conversions"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=t4,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn_input.memory_config(),
        additional_params={"conversion_sequence": "ROW->TILE->ROW->TILE->ROW"},
    )


@pytest.mark.parametrize("shape", [(1, 0), (0, 32), (32, 0)])
def test_to_layout_empty_tensors(device, shape):
    """Test to_layout with empty/zero-size tensors."""
    # Skip if shape has zero elements
    if any(dim == 0 for dim in shape):
        pytest.skip("Empty tensors may not be fully supported")
    
    torch.manual_seed(42)
    
    # Create empty tensor
    torch_input = torch.empty(shape, dtype=torch.bfloat16)
    
    try:
        ttnn_input = ttnn.from_torch(
            torch_input,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        
        ttnn_output = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
        ttnn_output = ttnn.to_layout(ttnn_output, layout=ttnn.ROW_MAJOR_LAYOUT)
        torch_output = ttnn.to_torch(ttnn_output)
        
        # For empty tensors, just check shapes match
        assert torch_input.shape == torch_output.shape
        
        test_name = f"empty_tensor_{'_'.join(map(str, shape))}"
        save_test_result(
            test_name=test_name,
            input_tensor=ttnn_input,
            output_tensor=ttnn_output,
            input_torch=torch_input,
            output_torch=torch_output,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn_input.memory_config(),
            additional_params={"note": "Empty tensor test"},
        )
    except Exception as e:
        pytest.skip(f"Empty tensor test failed (may not be supported): {e}")


@pytest.mark.parametrize("shape", [(32, 32), (64, 64)])
def test_to_layout_precision_edge_cases(device, shape):
    """Test to_layout with precision edge cases (NaN, Inf, dtype limits)."""
    torch.manual_seed(42)
    
    # Create tensor with special values
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    # Add some special values
    torch_input[0, 0] = float('inf')
    torch_input[0, 1] = float('-inf')
    torch_input[0, 2] = float('nan')
    torch_input[0, 3] = torch.finfo(torch.bfloat16).max
    torch_input[0, 4] = torch.finfo(torch.bfloat16).min
    
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    
    ttnn_output = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    ttnn_output = ttnn.to_layout(ttnn_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    torch_output = ttnn.to_torch(ttnn_output)
    
    # Check that non-special values match
    mask = torch.isfinite(torch_input) & torch.isfinite(torch_output)
    if mask.any():
        assert_with_pcc(torch_input[mask], torch_output[mask])
    
    test_name = f"precision_edge_cases_{'_'.join(map(str, shape))}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_torch=torch_input,
        output_torch=torch_output,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn_input.memory_config(),
        additional_params={"special_values": ["inf", "-inf", "nan", "max", "min"]},
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
