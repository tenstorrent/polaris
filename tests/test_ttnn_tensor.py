#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import ttsim.front.ttnn.op as ttnn_op
from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Tensor


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


@pytest.mark.unit
def test_bfloat16_dtype_mapping():
    """
    Test for commit 88c2821: In TTNN, map BFLOAT16 -> bfloat16, not float32

    Verifies that DataType.BFLOAT16.to_numpy returns np.float16 (not np.float32).
    """
    # Test that BFLOAT16 maps to float16
    assert DataType.BFLOAT16.to_numpy == np.dtype(np.float16), \
        f"BFLOAT16 should map to np.float16, got {DataType.BFLOAT16.to_numpy}"

    # Test that FLOAT32 still maps to float32
    assert DataType.FLOAT32.to_numpy == np.dtype(np.float32), \
        f"FLOAT32 should map to np.float32, got {DataType.FLOAT32.to_numpy}"

    # Test that we can create a tensor with BFLOAT16 dtype
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0

    bf16_tensor = Tensor(
        name="test_bf16_tensor",
        shape=[2, 3],
        dtype=DataType.BFLOAT16,
        device=device
    )

    # Verify the tensor's dtype is float16
    assert bf16_tensor.dtype == np.dtype(np.float16), \
        f"Tensor dtype should be np.float16, got {bf16_tensor.dtype}"


@pytest.mark.unit
def test_add_mul_sub_retain_input_precision():
    """
    Test for commit 1fe42d4: TTNN add/mul/sub retain input tensor precision

    Verifies that Add, Mul, and Sub operations preserve the input tensor's dtype
    instead of defaulting to FLOAT32.
    """
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0

    # Test with BFLOAT16
    bf16_tensor1 = Tensor(
        name="bf16_t1",
        shape=[2, 3],
        dtype=DataType.BFLOAT16,
        device=device
    )
    bf16_tensor2 = Tensor(
        name="bf16_t2",
        shape=[2, 3],
        dtype=DataType.BFLOAT16,
        device=device
    )

    # Test Add operation
    add_result = ttnn_op.add(bf16_tensor1, bf16_tensor2)
    assert add_result.dtype == np.dtype(np.float16), \
        f"Add result should preserve BFLOAT16 dtype, got {add_result.dtype}"

    # Test Mul operation
    mul_result = ttnn_op.multiply(bf16_tensor1, bf16_tensor2)
    assert mul_result.dtype == np.dtype(np.float16), \
        f"Mul result should preserve BFLOAT16 dtype, got {mul_result.dtype}"

    # Test Sub operation
    sub_result = ttnn_op.subtract(bf16_tensor1, bf16_tensor2)
    assert sub_result.dtype == np.dtype(np.float16), \
        f"Sub result should preserve BFLOAT16 dtype, got {sub_result.dtype}"

    # Test with FLOAT32 to ensure it still works
    f32_tensor1 = Tensor(
        name="f32_t1",
        shape=[2, 3],
        dtype=DataType.FLOAT32,
        device=device
    )
    f32_tensor2 = Tensor(
        name="f32_t2",
        shape=[2, 3],
        dtype=DataType.FLOAT32,
        device=device
    )

    f32_add_result = ttnn_op.add(f32_tensor1, f32_tensor2)
    assert f32_add_result.dtype == np.dtype(np.float32), \
        f"Add result should preserve FLOAT32 dtype, got {f32_add_result.dtype}"


@pytest.mark.unit
def test_add_mul_sub_with_scalar_retain_precision():
    """
    Test for commit 1fe42d4: TTNN add/mul/sub retain input tensor precision with scalar inputs

    Verifies that when a scalar is used with Add/Mul/Sub, the scalar's dtype
    matches the tensor input's dtype.
    """
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0

    # Test with BFLOAT16 tensor and scalar
    bf16_tensor = Tensor(
        name="bf16_t",
        shape=[2, 3],
        dtype=DataType.BFLOAT16,
        device=device
    )

    # Test Add with scalar
    add_result = ttnn_op.add(bf16_tensor, 5.0)
    assert add_result.dtype == np.dtype(np.float16), \
        f"Add with scalar should preserve BFLOAT16 dtype, got {add_result.dtype}"

    # Verify the scalar tensor was created with correct dtype
    # Find the add operation
    add_op = None
    for op in device.ops.values():
        if op.optype == 'Add' and bf16_tensor.name in op.inList:
            add_op = op
            break

    assert add_op is not None, "Add operation not found"
    # The scalar input should be in the inList
    scalar_input_name = [name for name in add_op.inList if name != bf16_tensor.name][0]
    scalar_tensor = device.tensors[scalar_input_name]
    assert scalar_tensor.dtype == np.dtype(np.float16), \
        f"Scalar tensor should have BFLOAT16 dtype, got {scalar_tensor.dtype}"


@pytest.mark.unit
def test_reshape_retains_input_precision():
    """
    Test for commit ebb61ed: TTNN reshape output retains input precision

    Verifies that reshape operations preserve the input tensor's dtype.
    """
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0

    # Test reshape with BFLOAT16
    bf16_tensor = Tensor(
        name="bf16_reshape_input",
        shape=[2, 3, 4],
        dtype=DataType.BFLOAT16,
        device=device
    )

    # Use reshape operation
    reshaped = ttnn_op.reshape(bf16_tensor, [2, 12])
    assert reshaped.dtype == np.dtype(np.float16), \
        f"Reshape result should preserve BFLOAT16 dtype, got {reshaped.dtype}"

    # Test with FLOAT32
    f32_tensor = Tensor(
        name="f32_reshape_input",
        shape=[2, 3, 4],
        dtype=DataType.FLOAT32,
        device=device
    )

    f32_reshaped = ttnn_op.reshape(f32_tensor, [2, 12])
    assert f32_reshaped.dtype == np.dtype(np.float32), \
        f"Reshape result should preserve FLOAT32 dtype, got {f32_reshaped.dtype}"


@pytest.mark.unit
def test_view_retains_input_precision():
    """
    Test for commit ebb61ed: TTNN reshape output retains input precision (via view method)

    Verifies that the view() method (which uses reshape) preserves the input tensor's dtype.
    """
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0

    # Test view with BFLOAT16
    bf16_tensor = Tensor(
        name="bf16_view_input",
        shape=[2, 3, 4],
        dtype=DataType.BFLOAT16,
        device=device
    )

    bf16_viewed = bf16_tensor.view(2, 12)
    assert bf16_viewed.dtype == np.dtype(np.float16), \
        f"View result should preserve BFLOAT16 dtype, got {bf16_viewed.dtype}"

    # Test view with FLOAT32
    f32_tensor = Tensor(
        name="f32_view_input",
        shape=[2, 3, 4],
        dtype=DataType.FLOAT32,
        device=device
    )

    f32_viewed = f32_tensor.view(2, 12)
    assert f32_viewed.dtype == np.dtype(np.float32), \
        f"View result should preserve FLOAT32 dtype, got {f32_viewed.dtype}"

    # Test view with INT32
    int32_tensor = Tensor(
        name="int32_view_input",
        shape=[2, 3, 4],
        dtype=DataType.INT32,
        device=device
    )

    int32_viewed = int32_tensor.view(2, 12)
    assert int32_viewed.dtype == np.dtype(np.int32), \
        f"View result should preserve INT32 dtype, got {int32_viewed.dtype}"
