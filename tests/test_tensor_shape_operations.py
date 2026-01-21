#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.helpers import (
    bidirectional_broadcast_shape_inference,
    multidirectional_broadcast_shape_inference,
    propagate_shape_and_type
)

@pytest.mark.unit
@pytest.mark.opunit
def test_bidirectional_broadcast_shape_inference():
    """Test bidirectional_broadcast_shape_inference function"""
    test_cases = [
        ([3, 1, 4], [2, 4], [3, 2, 4]),
        ([1, 3], [2, 3], [2, 3]),
        ([1, 0], [1], [1, 0]),  # Zero-sized dimension handling
        ([1, 1, 4], [2, 3, 1], [2, 3, 4]),
        ([], [3, 4], [3, 4]),  # Scalar broadcasting
    ]
    
    for shape1, shape2, expected in test_cases:
        result = bidirectional_broadcast_shape_inference(shape1, shape2)
        assert result == expected, f"Broadcast {shape1} and {shape2} should produce {expected}, got {result}"
    
    # Test incompatible shapes
    with pytest.raises(ValueError):
        bidirectional_broadcast_shape_inference([3, 4], [5, 6])

@pytest.mark.unit
@pytest.mark.opunit
def test_multidirectional_broadcast_shape_inference():
    """Test multidirectional_broadcast_shape_inference function"""
    test_cases = [
        ([[1, 3], [2, 1], [2, 3]], [2, 3]),
        ([[1, 1, 4], [2, 1, 1], [1, 3, 1]], [2, 3, 4]),
        ([[1], [1], [3]], [3]),  # All must be 1 or max, so [1, 1, 3] works
        ([[1], [1], [1]], [1]),  # All ones
        ([[2], [2], [2]], [2]),  # All same
    ]
    
    for shapes, expected in test_cases:
        result = multidirectional_broadcast_shape_inference(shapes)
        assert result == expected, f"Multi-broadcast {shapes} should produce {expected}, got {result}"
    
    # Test incompatible shapes
    with pytest.raises(AssertionError):
        multidirectional_broadcast_shape_inference([[3, 4], [5, 6]])
    
    # Test case that fails: [1, 2, 3] - 2 is neither 1 nor 3
    with pytest.raises(AssertionError):
        multidirectional_broadcast_shape_inference([[1], [2], [3]])

@pytest.mark.unit
@pytest.mark.opunit
def test_propagate_shape_and_type():
    """Test propagate_shape_and_type function"""
    input_tensors = [
        F._from_shape('X0', [2, 3, 4], np_dtype=np.float32),
        F._from_shape('X1', [5, 6], np_dtype=np.float64),
    ]
    output_tensors = [make_tensor('Y0'), make_tensor('Y1')]
    
    # Propagate first input to first output
    propagate_shape_and_type(input_tensors, output_tensors, 0, 0)
    assert output_tensors[0].shape == [2, 3, 4], "Shape should be propagated"
    assert output_tensors[0].dtype == np.float32, "Type should be propagated"
    
    # Propagate second input to second output
    propagate_shape_and_type(input_tensors, output_tensors, 1, 1)
    assert output_tensors[1].shape == [5, 6], "Shape should be propagated"
    assert output_tensors[1].dtype == np.float64, "Type should be propagated"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_arithmetic_operations():
    """Test shape arithmetic operations (as used in operators)"""
    # Shape addition (as in Pad)
    shape = [3, 4]
    pad_before = [1, 1]
    pad_after = [1, 1]
    result = [shape[i] + pad_before[i] + pad_after[i] for i in range(len(shape))]
    assert result == [5, 6], "Shape addition should work"
    
    # Shape multiplication (as in Tile)
    shape = [2, 3]
    repeats = [2, 3]
    result = [shape[i] * repeats[i] for i in range(len(shape))]
    assert result == [4, 9], "Shape multiplication should work"
    
    # Shape summation (as in Concat)
    shapes = [[2, 3], [4, 3], [1, 3]]
    axis = 0
    result = sum(s[axis] for s in shapes)
    assert result == 7, "Shape summation should work"
    
    # Shape division (as in Split)
    shape = [10]
    num_outputs = 2
    result = shape[0] // num_outputs
    assert result == 5, "Shape division should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_slicing_operations():
    """Test shape slicing operations"""
    shape = [2, 3, 4, 5]
    
    # Full slice
    assert shape[:] == [2, 3, 4, 5], "Full slice should work"
    
    # Prefix slice
    assert shape[:2] == [2, 3], "Prefix slice should work"
    
    # Suffix slice
    assert shape[1:] == [3, 4, 5], "Suffix slice should work"
    
    # Middle slice
    assert shape[1:3] == [3, 4], "Middle slice should work"
    
    # Negative indices
    assert shape[:-1] == [2, 3, 4], "Negative slice should work"
    assert shape[-2:] == [4, 5], "Negative suffix slice should work"
    
    # Shape extraction (as in Shape operator)
    start, end = 1, 3
    result = shape[start:end]
    assert result == [3, 4], "Shape extraction should work"
    
    # Shape insertion (as in Gather)
    data_shape = [2, 3, 4]
    indices_shape = [5]
    axis = 1
    result = data_shape[:axis] + indices_shape + data_shape[axis + 1:]
    assert result == [2, 5, 4], "Shape insertion should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_filtering_operations():
    """Test shape filtering operations"""
    # Filter by index (as in Squeeze)
    shape = [1, 3, 1, 4]
    indices_to_remove = [0, 2]
    result = [dim for i, dim in enumerate(shape) if i not in indices_to_remove]
    assert result == [3, 4], "Shape filtering by index should work"
    
    # Filter by condition (as in reduction ops)
    shape = [2, 3, 4]
    dim_to_remove = 1
    result = [s for i, s in enumerate(shape) if i != dim_to_remove]
    assert result == [2, 4], "Shape filtering by condition should work"
    
    # Filter by value
    shape = [1, 3, 1, 4, 1]
    result = [d for d in shape if d > 1]
    assert result == [3, 4], "Shape filtering by value should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_padding_operations():
    """Test shape padding operations"""
    # Prepend dimensions (as in Expand)
    shape = [3]
    n = 2
    result = [1] * n + shape
    assert result == [1, 1, 3], "Shape prepending should work"
    
    # Append dimensions
    shape = [3]
    n = 2
    result = shape + [1] * n
    assert result == [3, 1, 1], "Shape appending should work"
    
    # Padding for broadcasting (reverse and pad)
    shape = [3, 4]
    max_len = 4
    reversed_shape = list(shape[::-1])
    padded = reversed_shape + [1] * (max_len - len(shape))
    assert padded == [4, 3, 1, 1], "Shape padding for broadcasting should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_reversal_operations():
    """Test shape reversal operations"""
    shape = [2, 3, 4, 5]
    reversed_shape = shape[::-1]
    assert reversed_shape == [5, 4, 3, 2], "Shape reversal should work"
    
    # Partial reversal
    shape = [2, 3, 4]
    reversed_suffix = shape[-1::-1]
    assert reversed_suffix == [4, 3, 2], "Partial shape reversal should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_concatenation_operations():
    """Test shape concatenation operations"""
    # Simple concatenation
    shape1 = [2, 3]
    shape2 = [4, 5]
    result = shape1 + shape2
    assert result == [2, 3, 4, 5], "Shape concatenation should work"
    
    # With unpacking
    shape1 = [2, 3]
    shape2 = [4]
    result = [*shape1, *shape2]
    assert result == [2, 3, 4], "Shape concatenation with unpacking should work"
    
    # Prefix/suffix extraction (as in Flatten)
    shape = [2, 3, 4, 5]
    axis = 2
    prefix = shape[:axis]
    suffix = shape[axis:]
    assert prefix == [2, 3], "Prefix extraction should work"
    assert suffix == [4, 5], "Suffix extraction should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_comparison_operations():
    """Test shape comparison operations"""
    shape1 = [2, 3, 4]
    shape2 = [2, 3, 4]
    shape3 = [2, 3, 5]
    
    # Equality
    assert shape1 == shape2, "Identical shapes should be equal"
    assert shape1 != shape3, "Different shapes should not be equal"
    
    # Element-wise comparison
    assert shape1[0] == shape2[0], "First dimensions should be equal"
    assert shape1[0] == shape3[0], "First dimensions should be equal"
    assert shape1[2] != shape3[2], "Last dimensions should differ"
    
    # Rank comparison
    shape4 = [2, 3]
    assert len(shape1) != len(shape4), "Shapes with different ranks should not be equal"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_extraction_utilities():
    """Test shape extraction utilities"""
    # Extracting spatial dimensions
    shape = [2, 3, 4, 5, 6]
    spatial_dims = shape[2:]
    assert spatial_dims == [4, 5, 6], "Spatial dimensions extraction should work"
    
    # Extracting batch/channel
    shape = [2, 3, 4, 5]
    batch_channel = shape[:2]
    assert batch_channel == [2, 3], "Batch/channel extraction should work"
    
    # Extracting specific ranges
    shape = [1, 2, 3, 4, 5, 6]
    middle = shape[2:4]
    assert middle == [3, 4], "Middle range extraction should work"
