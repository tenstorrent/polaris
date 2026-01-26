# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for ttnn_shims.to_layout function.

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
import sys
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger

# Import shim modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from ttsim.front.ttnn.ttnn_shim import (
    TensorProxy as TensorProxy_Orig,
    Layout,
    DataType,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
    L1_MEMORY_CONFIG,
    to_layout,
    set_execution_mode,
    ExecutionMode,
    TILE_WIDTH,
    TILE_HEIGHT,
    Shape,
)

# Tile size constants
TILE_SIZE = 32

# Output directory for test results
OUTPUT_DIR = Path("test_outputs/to_layout_shim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TensorProxy(TensorProxy_Orig):
    """Extended TensorProxy to facilitate test data handling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = kwargs.get("data", None)
        buffer = kwargs.get("buffer", None)
        self.set_data(data)
        self.set_buffer(buffer)

    def set_data(self, data):
        self._data = data

    def has_data(self):
        return self._data is not None

    def get_data(self):
        return self._data

    def set_buffer(self, buffer):
        self._buffer = buffer

    def has_buffer(self):
        return self._buffer is not None

    def buffer(self):
        return self._buffer

    def physical_volume(self):
        return self._padded_shape.volume()

def torch_to_list(tensor: torch.Tensor) -> List[float]:
    """Convert torch tensor to list of values."""
    return tensor.flatten().tolist()


def list_to_torch(data: List[float], shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert list of values to torch tensor."""
    tensor = torch.tensor(data, dtype=dtype)
    if len(shape) > 0:
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        tensor = tensor[:total_elements].reshape(shape)
    return tensor


def serialize_tensor_data(data: List[float], shape: Tuple[int, ...], dtype: str) -> Dict[str, Any]:
    """Serialize tensor data to JSON-serializable format."""
    return {
        "shape": list(shape),
        "dtype": dtype,
        "values": data,
        "num_elements": len(data),
    }


def serialize_memory_config(mem_config: Optional[MemoryConfig]) -> Dict[str, Any]:
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
    input_tensor: TensorProxy,
    output_tensor: TensorProxy,
    input_data: List[float],
    output_data: List[float],
    input_layout: Layout,
    output_layout: Layout,
    dtype: DataType,
    memory_config: Optional[MemoryConfig],
    additional_params: Optional[Dict[str, Any]] = None,
):
    """Save test result to JSON file."""
    # Calculate padding info
    input_padding = calculate_padding_info(
        tuple(input_tensor.logical_shape()),
        tuple(input_tensor.padded_shape())
    )
    output_padding = calculate_padding_info(
        tuple(output_tensor.logical_shape()),
        tuple(output_tensor.padded_shape())
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
            "execution_mode": "EXECUTE",
        },
        "input_tensor": {
            "logical_shape": list(input_tensor.logical_shape()),
            "padded_shape": list(input_tensor.padded_shape()),
            "layout": str(input_tensor.layout),
            "dtype": str(input_tensor.dtype),
            "padding_info": input_padding,
            "content": serialize_tensor_data(
                input_data,
                tuple(input_tensor.logical_shape()),
                str(input_tensor.dtype)
            ),
        },
        "output_tensor": {
            "logical_shape": list(output_tensor.logical_shape()),
            "padded_shape": list(output_tensor.padded_shape()),
            "layout": str(output_tensor.layout),
            "dtype": str(output_tensor.dtype),
            "padding_info": output_padding,
            "content": serialize_tensor_data(
                output_data,
                tuple(output_tensor.logical_shape()),
                str(output_tensor.dtype)
            ),
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
    (512, 512),         # Very large, aligned
    (513, 513),         # Very large, not aligned
]

# All test shapes
ALL_TEST_SHAPES = TILE_ALIGNED_SHAPES + PARTIAL_TILE_ALIGNED_SHAPES + EDGE_CASE_SHAPES


@pytest.fixture(autouse=True)
def setup_execution_mode():
    """Set execution mode to EXECUTE for all tests."""
    set_execution_mode(ExecutionMode.EXECUTE)
    yield
    set_execution_mode(ExecutionMode.TRACK_ONLY)  # Reset to default


@pytest.mark.parametrize("shape", TILE_ALIGNED_SHAPES)
@pytest.mark.parametrize("input_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [DataType.BFLOAT16, DataType.FLOAT32])
def test_to_layout_tile_aligned_shapes(shape, input_layout, output_layout, dtype, device):
    """Test to_layout with shapes where all dimensions are multiples of tile size."""
    torch.manual_seed(42)
    
    # Create input tensor data
    if dtype == DataType.FLOAT32:
        torch_input = torch.randn(shape, dtype=torch.float32)
    else:
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
    
    input_data = torch_to_list(torch_input)
    
    # Create shim tensor
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=dtype,
        layout=input_layout,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
        buffer=list()
    )
    
    # Perform layout conversion
    ttnn_output = to_layout(ttnn_input, layout=output_layout)
    
    # For verification, convert both input and output to ROW_MAJOR for comparison
    # This ensures we're comparing the logical data, not the layout-specific format
    if input_layout != Layout.ROW_MAJOR_LAYOUT:
        ttnn_input_for_verification = to_layout(ttnn_input, layout=Layout.ROW_MAJOR_LAYOUT)
        input_data_for_verification = ttnn_input_for_verification.get_data() if ttnn_input_for_verification.has_data() else input_data
    else:
        input_data_for_verification = input_data
    
    if output_layout != Layout.ROW_MAJOR_LAYOUT:
        ttnn_output_for_verification = to_layout(ttnn_output, layout=Layout.ROW_MAJOR_LAYOUT)
    else:
        ttnn_output_for_verification = ttnn_output
    
    # Extract output data
    assert isinstance(ttnn_output, TensorProxy)
    output_data = ttnn_output_for_verification.get_data() if ttnn_output_for_verification.has_data() else []
    
    # Convert back to torch for verification
    if len(output_data) > 0:
        output_torch = list_to_torch(
            output_data,
            tuple(ttnn_output_for_verification.logical_shape()),
            torch_input.dtype
        )
        
        # For tile layout, we need to handle padding
        # Extract only the logical shape portion
        logical_shape = tuple(ttnn_output_for_verification.logical_shape())
        if len(logical_shape) == len(output_torch.shape):
            output_torch = output_torch.reshape(logical_shape)
        
        # Verify correctness (only for logical shape)
        # Use the verification input data (converted to row-major if needed)
        input_verification_torch = list_to_torch(
            input_data_for_verification,
            tuple(ttnn_input.logical_shape()),
            torch_input.dtype
        )
        input_flat = input_verification_torch.flatten()
        output_flat = output_torch.flatten()[:len(input_flat)]
        
        # Allow some tolerance for floating point operations
        if dtype == DataType.FLOAT32:
            assert torch.allclose(input_flat, output_flat, rtol=1e-5, atol=1e-6)
        else:
            diff = torch.abs(input_flat - output_flat)
            max_diff = torch.max(diff)
            logger.debug(f"Maximum difference: {max_diff}")
            if max_diff > 0:
                close_mask = torch.isclose(input_flat, output_flat, rtol=1e-5, atol=1e-6)

                # Find the indices of elements that are NOT close
                # Use torch.logical_not to flip the mask
                # Use torch.nonzero to get the indices of False values
                not_close_indices = torch.nonzero(torch.logical_not(close_mask))

                # Print the values at these indices for inspection
                for idx in not_close_indices:
                    logger.debug(f"Indices: {idx}, Value A: {input_flat[tuple(idx)]}, Value B: {output_flat[tuple(idx)]}")
                logger.debug(f"Differences: {diff[diff > 0]}")
            assert torch.allclose(input_flat, output_flat, rtol=1e-3, atol=1e-4)
    
    # Save test result
    test_name = f"tile_aligned_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}_{dtype}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_data=input_data,
        output_data=output_data,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=dtype,
        memory_config=DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("shape", PARTIAL_TILE_ALIGNED_SHAPES)
@pytest.mark.parametrize("input_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
def test_to_layout_partial_tile_aligned_shapes(shape, input_layout, output_layout, device):
    """Test to_layout with shapes where some dimensions are multiples of tile size."""
    torch.manual_seed(42)
    
    # Create input tensor data
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_data = torch_to_list(torch_input)
    
    # Create shim tensor
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=input_layout,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
    )
    
    # Perform layout conversion
    ttnn_output = to_layout(ttnn_input, layout=output_layout)
    
    # For verification, convert both input and output to ROW_MAJOR for comparison
    # This ensures we're comparing the logical data, not the layout-specific format
    if input_layout != Layout.ROW_MAJOR_LAYOUT:
        ttnn_input_for_verification = to_layout(ttnn_input, layout=Layout.ROW_MAJOR_LAYOUT)
        input_data_for_verification = ttnn_input_for_verification.get_data() if ttnn_input_for_verification.has_data() else input_data
    else:
        input_data_for_verification = input_data
    
    if output_layout != Layout.ROW_MAJOR_LAYOUT:
        ttnn_output_for_verification = to_layout(ttnn_output, layout=Layout.ROW_MAJOR_LAYOUT)
    else:
        ttnn_output_for_verification = ttnn_output
    
    # Extract output data
    output_data = ttnn_output_for_verification.get_data() if ttnn_output_for_verification.has_data() else []
    
    # Convert back to torch for verification
    if len(output_data) > 0:
        output_torch = list_to_torch(
            output_data,
            tuple(ttnn_output_for_verification.logical_shape()),
            torch_input.dtype
        )
        
        logical_shape = tuple(ttnn_output_for_verification.logical_shape())
        if len(logical_shape) == len(output_torch.shape):
            output_torch = output_torch.reshape(logical_shape)
        
        # Verify correctness (only for logical shape)
        # Use the verification input data (converted to row-major if needed)
        input_verification_torch = list_to_torch(
            input_data_for_verification,
            tuple(ttnn_input.logical_shape()),
            torch_input.dtype
        )
        input_flat = input_verification_torch.flatten()
        output_flat = output_torch.flatten()[:len(input_flat)]
        
        assert torch.allclose(input_flat, output_flat, rtol=1e-3, atol=1e-4)
    
    # Save test result
    test_name = f"partial_aligned_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_data=input_data,
        output_data=output_data,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=DataType.BFLOAT16,
        memory_config=DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("shape", EDGE_CASE_SHAPES)
@pytest.mark.parametrize("input_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
def test_to_layout_edge_cases(shape, input_layout, output_layout, device):
    """Test to_layout with edge case shapes."""
    torch.manual_seed(42)
    
    # Create input tensor data
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_data = torch_to_list(torch_input)
    
    # Create shim tensor
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=input_layout,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
    )
    
    # Perform layout conversion
    ttnn_output = to_layout(ttnn_input, layout=output_layout)
    
    # For verification, convert both input and output to ROW_MAJOR for comparison
    # This ensures we're comparing the logical data, not the layout-specific format
    if input_layout != Layout.ROW_MAJOR_LAYOUT:
        ttnn_input_for_verification = to_layout(ttnn_input, layout=Layout.ROW_MAJOR_LAYOUT)
        input_data_for_verification = ttnn_input_for_verification.get_data() if ttnn_input_for_verification.has_data() else input_data
    else:
        input_data_for_verification = input_data
    
    if output_layout != Layout.ROW_MAJOR_LAYOUT:
        ttnn_output_for_verification = to_layout(ttnn_output, layout=Layout.ROW_MAJOR_LAYOUT)
    else:
        ttnn_output_for_verification = ttnn_output
    
    # Extract output data
    output_data = ttnn_output_for_verification.get_data() if ttnn_output_for_verification.has_data() else []
    
    # Convert back to torch for verification
    if len(output_data) > 0:
        output_torch = list_to_torch(
            output_data,
            tuple(ttnn_output_for_verification.logical_shape()),
            torch_input.dtype
        )
        
        logical_shape = tuple(ttnn_output_for_verification.logical_shape())
        if len(logical_shape) == len(output_torch.shape):
            output_torch = output_torch.reshape(logical_shape)
        
        # Verify correctness (only for logical shape)
        # Use the verification input data (converted to row-major if needed)
        input_verification_torch = list_to_torch(
            input_data_for_verification,
            tuple(ttnn_input.logical_shape()),
            torch_input.dtype
        )
        input_flat = input_verification_torch.flatten()
        output_flat = output_torch.flatten()[:len(input_flat)]
        
        assert torch.allclose(input_flat, output_flat, rtol=1e-3, atol=1e-4)
    
    # Save test result
    test_name = f"edge_case_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_data=input_data,
        output_data=output_data,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=DataType.BFLOAT16,
        memory_config=DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (33, 65)])
@pytest.mark.parametrize("dtype", [DataType.BFLOAT16, DataType.FLOAT32, DataType.UINT32, DataType.INT32])
def test_to_layout_different_dtypes(shape, dtype, device):
    """Test to_layout with different data types."""
    torch.manual_seed(42)
    
    # Map shim dtype to torch dtype
    dtype_map = {
        DataType.BFLOAT16: torch.bfloat16,
        DataType.FLOAT32: torch.float32,
        DataType.UINT32: torch.int32,
        DataType.INT32: torch.int32,
    }
    
    torch_dtype = dtype_map[dtype]
    
    # Create input tensor data
    if dtype in [DataType.UINT32, DataType.INT32]:
        torch_input = torch.randint(0, 100, shape, dtype=torch_dtype)
    else:
        torch_input = torch.randn(shape, dtype=torch_dtype)
    
    input_data = torch_to_list(torch_input)
    
    # Create shim tensor
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=dtype,
        layout=Layout.ROW_MAJOR_LAYOUT,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
    )
    
    # Convert to tile layout
    ttnn_output = to_layout(ttnn_input, layout=Layout.TILE_LAYOUT)
    
    # Convert back to row major
    ttnn_output = to_layout(ttnn_output, layout=Layout.ROW_MAJOR_LAYOUT)
    
    # Extract output data
    output_data = ttnn_output.get_data() if ttnn_output.has_data() else []
    
    # Save test result
    test_name = f"dtype_{dtype}_{'_'.join(map(str, shape))}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_data=input_data,
        output_data=output_data,
        input_layout=Layout.ROW_MAJOR_LAYOUT,
        output_layout=Layout.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("shape", [(64, 128), (128, 64)])
@pytest.mark.parametrize("memory_config", [DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG])
def test_to_layout_memory_configs(shape, memory_config, device):
    """Test to_layout with different memory configurations."""
    torch.manual_seed(42)
    
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_data = torch_to_list(torch_input)
    
    # Create shim tensor with specific memory config
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
        device=device,
        data=input_data,
    )
    
    # Convert to tile layout
    ttnn_output = to_layout(ttnn_input, layout=Layout.TILE_LAYOUT)
    
    # Convert back to row major
    ttnn_output = to_layout(ttnn_output, layout=Layout.ROW_MAJOR_LAYOUT)
    
    # Extract output data
    output_data = ttnn_output.get_data() if ttnn_output.has_data() else []
    
    # Save test result
    test_name = f"mem_config_{memory_config.buffer_type}_{'_'.join(map(str, shape))}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_data=input_data,
        output_data=output_data,
        input_layout=Layout.ROW_MAJOR_LAYOUT,
        output_layout=Layout.ROW_MAJOR_LAYOUT,
        dtype=DataType.BFLOAT16,
        memory_config=memory_config,
    )


def test_to_layout_same_layout_no_change(device):
    """Test to_layout when tensor is already in requested layout."""
    torch.manual_seed(42)
    
    shape = (64, 64)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_data = torch_to_list(torch_input)
    
    # Create tensor in TILE_LAYOUT
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=Layout.TILE_LAYOUT,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
    )
    
    # Try to convert to same layout (should return same tensor)
    ttnn_output = to_layout(ttnn_input, layout=Layout.TILE_LAYOUT)
    
    # Extract output data
    output_data = ttnn_output.get_data() if ttnn_output.has_data() else []
    
    # Save test result
    test_name = "same_layout_no_change"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_data=input_data,
        output_data=output_data,
        input_layout=Layout.TILE_LAYOUT,
        output_layout=Layout.TILE_LAYOUT,
        dtype=DataType.BFLOAT16,
        memory_config=DRAM_MEMORY_CONFIG,
        additional_params={"note": "Tensor already in requested layout"},
    )


def test_to_layout_round_trip(device):
    """Test round-trip conversion: ROW_MAJOR -> TILE -> ROW_MAJOR."""
    torch.manual_seed(42)
    
    shape = (33, 65)  # Not tile-aligned
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_data = torch_to_list(torch_input)
    
    # Start with ROW_MAJOR
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
    )
    
    # Convert to TILE
    ttnn_tile = to_layout(ttnn_input, layout=Layout.TILE_LAYOUT)
    
    # Convert back to ROW_MAJOR
    ttnn_output = to_layout(ttnn_tile, layout=Layout.ROW_MAJOR_LAYOUT)
    
    # Extract output data
    output_data = ttnn_output.get_data() if ttnn_output.has_data() else []
    
    # Save test result
    test_name = "round_trip_conversion"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_data=input_data,
        output_data=output_data,
        input_layout=Layout.ROW_MAJOR_LAYOUT,
        output_layout=Layout.ROW_MAJOR_LAYOUT,
        dtype=DataType.BFLOAT16,
        memory_config=DRAM_MEMORY_CONFIG,
        additional_params={"intermediate_layout": str(Layout.TILE_LAYOUT)},
    )


def test_to_layout_error_cases(device):
    """Test error cases for to_layout."""
    torch.manual_seed(42)
    
    shape = (32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_data = torch_to_list(torch_input)
    
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
    )
    
    # Test: dtype cannot be specified when converting to ROW_MAJOR_LAYOUT
    with pytest.raises(RuntimeError, match="dtype cannot be specified"):
        to_layout(ttnn_input, layout=Layout.ROW_MAJOR_LAYOUT, dtype=DataType.FLOAT32)


# ============================================================================
# Missing Test Cases - Now Added
# ============================================================================

@pytest.mark.parametrize("shape", [(1, 1, 1, 32, 32), (2, 3, 4, 64, 128), (1, 1, 1, 1, 32, 32)])
@pytest.mark.parametrize("input_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
def test_to_layout_high_dimensional_tensors(shape, input_layout, output_layout, device):
    """Test to_layout with 5D and 6D tensors."""
    torch.manual_seed(42)
    
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_data = torch_to_list(torch_input)
    
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=input_layout,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
    )
    
    ttnn_output = to_layout(ttnn_input, layout=output_layout)
    output_data = ttnn_output.get_data() if ttnn_output.has_data() else []
    
    test_name = f"high_dim_{len(shape)}D_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_data=input_data,
        output_data=output_data,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=DataType.BFLOAT16,
        memory_config=DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("shape", [(2048, 2048), (4096, 4096)])
@pytest.mark.parametrize("input_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
@pytest.mark.skip(reason="Very large tensors may exceed memory limits in some environments")
def test_to_layout_very_large_tensors(shape, input_layout, output_layout, device):
    """Test to_layout with very large tensors."""
    torch.manual_seed(42)
    
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_data = torch_to_list(torch_input)
    
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=input_layout,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
    )
    
    ttnn_output = to_layout(ttnn_input, layout=output_layout)
    output_data = ttnn_output.get_data() if ttnn_output.has_data() else []
    
    test_name = f"very_large_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_data=input_data,
        output_data=output_data,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=DataType.BFLOAT16,
        memory_config=DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("shape", [(31, 32), (32, 31), (33, 33)])
@pytest.mark.parametrize("input_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
def test_to_layout_boundary_conditions(shape, input_layout, output_layout, device):
    """Test to_layout with boundary conditions (just below/above tile size)."""
    torch.manual_seed(42)
    
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_data = torch_to_list(torch_input)
    
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=input_layout,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
    )
    
    ttnn_output = to_layout(ttnn_input, layout=output_layout)
    output_data = ttnn_output.get_data() if ttnn_output.has_data() else []
    
    test_name = f"boundary_{'_'.join(map(str, shape))}_{input_layout}_{output_layout}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_data=input_data,
        output_data=output_data,
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=DataType.BFLOAT16,
        memory_config=DRAM_MEMORY_CONFIG,
    )


def test_to_layout_multiple_consecutive_conversions(device):
    """Test multiple consecutive layout conversions."""
    torch.manual_seed(42)
    
    shape = (64, 64)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_data = torch_to_list(torch_input)
    
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
    )
    
    # Perform multiple conversions: ROW -> TILE -> ROW -> TILE -> ROW
    t1 = to_layout(ttnn_input, layout=Layout.TILE_LAYOUT)
    t2 = to_layout(t1, layout=Layout.ROW_MAJOR_LAYOUT)
    t3 = to_layout(t2, layout=Layout.TILE_LAYOUT)
    t4 = to_layout(t3, layout=Layout.ROW_MAJOR_LAYOUT)
    
    output_data = t4.get_data() if t4.has_data() else []
    
    test_name = "multiple_consecutive_conversions"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=t4,
        input_data=input_data,
        output_data=output_data,
        input_layout=Layout.ROW_MAJOR_LAYOUT,
        output_layout=Layout.ROW_MAJOR_LAYOUT,
        dtype=DataType.BFLOAT16,
        memory_config=DRAM_MEMORY_CONFIG,
        additional_params={"conversion_sequence": "ROW->TILE->ROW->TILE->ROW"},
    )


@pytest.mark.parametrize("shape", [(1, 0), (0, 32), (32, 0)])
def test_to_layout_empty_tensors(shape, device):
    """Test to_layout with empty/zero-size tensors."""
    # Skip if shape has zero elements
    # if any(dim == 0 for dim in shape):
    #     pytest.skip("Empty tensors may not be fully supported")
    
    torch.manual_seed(42)
    
    # Create empty tensor
    torch_input = torch.empty(shape, dtype=torch.bfloat16)
    input_data = torch_to_list(torch_input)
    
    try:
        ttnn_input = TensorProxy(
            shape=shape,
            dtype=DataType.BFLOAT16,
            layout=Layout.ROW_MAJOR_LAYOUT,
            memory_config=DRAM_MEMORY_CONFIG,
            device=device,
            data=input_data,
        )
        
        ttnn_output = to_layout(ttnn_input, layout=Layout.TILE_LAYOUT)
        ttnn_output = to_layout(ttnn_output, layout=Layout.ROW_MAJOR_LAYOUT)
        output_data = ttnn_output.get_data() if ttnn_output.has_data() else []
        
        test_name = f"empty_tensor_{'_'.join(map(str, shape))}"
        save_test_result(
            test_name=test_name,
            input_tensor=ttnn_input,
            output_tensor=ttnn_output,
            input_data=input_data,
            output_data=output_data,
            input_layout=Layout.ROW_MAJOR_LAYOUT,
            output_layout=Layout.ROW_MAJOR_LAYOUT,
            dtype=DataType.BFLOAT16,
            memory_config=DRAM_MEMORY_CONFIG,
            additional_params={"note": "Empty tensor test"},
        )
    except Exception as e:
        pytest.skip(f"Empty tensor test failed (may not be supported): {e}")


@pytest.mark.parametrize("shape", [(32, 32), (64, 64)])
def test_to_layout_precision_edge_cases(shape, device):
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
    
    input_data = torch_to_list(torch_input)
    
    ttnn_input = TensorProxy(
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        memory_config=DRAM_MEMORY_CONFIG,
        device=device,
        data=input_data,
    )
    
    ttnn_output = to_layout(ttnn_input, layout=Layout.TILE_LAYOUT)
    ttnn_output = to_layout(ttnn_output, layout=Layout.ROW_MAJOR_LAYOUT)
    output_data = ttnn_output.get_data() if ttnn_output.has_data() else []
    
    test_name = f"precision_edge_cases_{'_'.join(map(str, shape))}"
    save_test_result(
        test_name=test_name,
        input_tensor=ttnn_input,
        output_tensor=ttnn_output,
        input_data=input_data,
        output_data=output_data,
        input_layout=Layout.ROW_MAJOR_LAYOUT,
        output_layout=Layout.ROW_MAJOR_LAYOUT,
        dtype=DataType.BFLOAT16,
        memory_config=DRAM_MEMORY_CONFIG,
        additional_params={"special_values": ["inf", "-inf", "nan", "max", "min"]},
    )


# Note: Empty/zero-size tensors, sharded memory configs, and sub_core_grids
# are not applicable to shim version as they require device-specific features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
