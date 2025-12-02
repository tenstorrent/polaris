#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from ttsim.front.ttnn.tensor import Tensor, DataType
from ttsim.front.ttnn.device import Device, ARCH


@pytest.mark.unit
def test_tensor_view_update_tensor_counts():
    """
    Test case for line 202 in ttsim/front/ttnn/tensor.py
    Tests that the view() method correctly calls update_tensor_counts on the operation object.

    The view() method creates a Reshape operation and should:
    1. Create a shape tensor (shapeT)
    2. Create an output tensor (outT)
    3. Update tensor counts via opobj.update_tensor_counts([self, shapeT], [outT])
    """
    # Create a device for the tensors
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0

    # Create an input tensor with shape [2, 3, 4]
    input_shape = [2, 3, 4]
    input_tensor = Tensor(
        name="test_input",
        shape=input_shape,
        dtype=DataType.FLOAT32,
        device=device
    )

    # Call view to reshape to [2, 12]
    new_shape = (2, 12)
    output_tensor = input_tensor.view(*new_shape)

    # Verify the output tensor has the correct shape
    assert output_tensor.shape == list(new_shape), \
        f"Expected output shape {list(new_shape)}, got {output_tensor.shape}"

    # Verify the output tensor is on the same device
    assert output_tensor.device == device, \
        "Output tensor should be on the same device as input tensor"

    # Verify that an operation was added to the device
    assert len(device.ops) > 0, \
        "Operation should be added to device"

    # Get the reshape operation robustly (do not assume operation name format)
    reshape_op = None
    for op in device.ops.values():
        if op.optype == 'Reshape' and input_tensor.name in op.inList and output_tensor.name in op.outList:
            reshape_op = op
            break
    assert reshape_op is not None, "Reshape operation not found in device.ops"

    # Verify the operation type
    assert reshape_op.optype == 'Reshape', \
        f"Expected operation type 'Reshape', got {reshape_op.optype}"

    # Verify that perf_stats contains the tensor counts (set by update_tensor_counts on line 202)
    assert reshape_op.perf_stats is not None, \
        "Operation should have performance statistics"

    assert 'inActCount' in reshape_op.perf_stats, \
        "Performance stats should include inActCount"

    assert 'outActCount' in reshape_op.perf_stats, \
        "Performance stats should include outActCount"

    # Verify tensor counts are correct
    # Input tensor has 2*3*4 = 24 elements (activation)
    # Shape tensor has 2 elements (activation, not parameter) for reshape dimensions
    # Total input activation count = 24 + 2 = 26
    expected_in_act_count = np.prod(input_shape) + len(new_shape)  # input tensor + shape tensor
    expected_out_act_count = np.prod(new_shape)

    assert reshape_op.perf_stats['inActCount'] == expected_in_act_count, \
        f"Expected inActCount={expected_in_act_count}, got {reshape_op.perf_stats['inActCount']}"

    assert reshape_op.perf_stats['outActCount'] == expected_out_act_count, \
        f"Expected outActCount={expected_out_act_count}, got {reshape_op.perf_stats['outActCount']}"

    # Verify the operation has correct input and output lists
    assert len(reshape_op.inList) == 2, \
        "Reshape operation should have 2 inputs (tensor and shape)"

    assert len(reshape_op.outList) == 1, \
        "Reshape operation should have 1 output"

    assert input_tensor.name in reshape_op.inList, \
        "Input tensor should be in operation's input list"

    assert output_tensor.name in reshape_op.outList, \
        "Output tensor should be in operation's output list"


@pytest.mark.unit
def test_tensor_view_multiple_reshapes():
    """
    Test multiple view operations to ensure update_tensor_counts works correctly
    for sequential reshape operations.
    """
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0

    # Create initial tensor [4, 6]
    tensor = Tensor(
        name="test_multi_view",
        shape=[4, 6],
        dtype=DataType.FLOAT32,
        device=device
    )

    # First view: [4, 6] -> [2, 12]
    tensor2 = tensor.view(2, 12)
    assert tensor2.shape == [2, 12]

    # Second view: [2, 12] -> [24]
    tensor3 = tensor2.view(24)
    assert tensor3.shape == [24]

    # Third view: [24] -> [3, 8]
    tensor4 = tensor3.view(3, 8)
    assert tensor4.shape == [3, 8]

    # Verify all operations were added
    assert len(device.ops) == 3, \
        "Should have 3 reshape operations"

    # Verify all operations have tensor counts updated
    for op_name in device.ops:
        op = device.ops[op_name]
        assert op.perf_stats is not None
        assert 'inActCount' in op.perf_stats
        assert 'outActCount' in op.perf_stats
        # All should have same total element count (24)
        assert op.perf_stats['outActCount'] == 24


@pytest.mark.unit  
def test_tensor_view_with_zero_elements():
    """
    Test view operation with tensors containing zero elements.
    """
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0

    # Create tensor with zero elements [0, 5]
    tensor = Tensor(
        name="test_zero_view",
        shape=[0, 5],
        dtype=DataType.FLOAT32,
        device=device
    )

    # View to [0, 1, 5]
    output = tensor.view(0, 1, 5)
    assert output.shape == [0, 1, 5]

    # Verify operation exists and has correct counts

    # Get the reshape operation robustly (do not assume operation name format)
    reshape_op = None
    for op in device.ops.values():
        if op.optype == 'Reshape' and tensor.name in op.inList and output.name in op.outList:
            reshape_op = op
            break
    assert reshape_op is not None, "Reshape operation not found in device.ops"
    # Input tensor has 0 elements but shape tensor has 3 elements (0, 1, 5)
    assert reshape_op.perf_stats['inActCount'] == 3  # 0 (input) + 3 (shape tensor)
    assert reshape_op.perf_stats['outActCount'] == 0
