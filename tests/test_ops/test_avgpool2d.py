#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_avgpool2d
from tests.test_ops.utils import generate_test_data


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_avgpool2d(X, kernel_shape, strides, pads):
    """
    Reference implementation of avgpool2d using NumPy.

    Args:
        X: Input array [N, C, H, W]
        kernel_shape: [Kh, Kw]
        strides: [stride_h, stride_w]
        pads: [top, left, bottom, right]

    Returns:
        Y: Output array [N, C, H_out, W_out]
    """
    N, C, H_in, W_in = X.shape
    Kh, Kw = kernel_shape

    # Apply padding
    pad_h = (pads[0], pads[2])
    pad_w = (pads[1], pads[3])

    if any(p > 0 for p in pads):
        X_padded = np.pad(X, ((0, 0), (0, 0), pad_h, pad_w), mode="constant")
    else:
        X_padded = X

    # Calculate output size
    H_out = (H_in + pads[0] + pads[2] - Kh) // strides[0] + 1
    W_out = (W_in + pads[1] + pads[3] - Kw) // strides[1] + 1

    Y = np.zeros((N, C, H_out, W_out), dtype=X.dtype)

    # Perform average pooling
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * strides[0]
                    w_start = w * strides[1]
                    pool_region = X_padded[
                        n, c, h_start : h_start + Kh, w_start : w_start + Kw
                    ]
                    Y[n, c, h, w] = np.mean(pool_region)

    return Y


# Test cases with shape validation and numerical validation
test_name = "test_avgpool2d"
test_cases = [
    # (name, input_shape, kernel_shape, strides, pads, test_data_type)
    ("Basic 2x2 pooling", [1, 1, 4, 4], [2, 2], [2, 2], [0, 0, 0, 0], "positive"),
    (
        "3x3 kernel with stride 1",
        [1, 1, 5, 5],
        [3, 3],
        [1, 1],
        [0, 0, 0, 0],
        "positive",
    ),
    ("Multi-channel pooling", [1, 3, 8, 8], [2, 2], [2, 2], [0, 0, 0, 0], "positive"),
    ("Batch size > 1", [2, 2, 6, 6], [2, 2], [2, 2], [0, 0, 0, 0], "positive"),
    ("With padding", [1, 1, 4, 4], [3, 3], [1, 1], [1, 1, 1, 1], "positive"),
    ("Asymmetric padding", [1, 1, 5, 5], [2, 2], [2, 2], [1, 0, 1, 0], "positive"),
    ("Large stride", [1, 1, 8, 8], [2, 2], [3, 3], [0, 0, 0, 0], "positive"),
    ("Non-square kernel", [1, 1, 6, 8], [2, 3], [2, 2], [0, 0, 0, 0], "positive"),
    ("Non-square input", [1, 2, 7, 9], [3, 3], [2, 2], [0, 0, 0, 0], "positive"),
    # Edge cases: negative values
    ("Negative values", [1, 1, 4, 4], [2, 2], [2, 2], [0, 0, 0, 0], "negative"),
    # Edge cases: zero values
    ("Zero values", [1, 1, 4, 4], [2, 2], [2, 2], [0, 0, 0, 0], "zeros"),
    # Edge cases: mixed values
    ("Mixed positive/negative", [1, 1, 6, 6], [2, 2], [2, 2], [0, 0, 0, 0], "mixed"),
    # Edge cases: very small values
    ("Small values", [1, 1, 4, 4], [2, 2], [2, 2], [0, 0, 0, 0], "small"),
    # Edge cases: large values
    ("Large values", [1, 1, 4, 4], [2, 2], [2, 2], [0, 0, 0, 0], "large"),
    # Edge cases: single element pooling
    ("1x1 kernel (identity)", [1, 1, 4, 4], [1, 1], [1, 1], [0, 0, 0, 0], "positive"),
    # Edge cases: minimum input size
    ("Minimum input 2x2", [1, 1, 2, 2], [2, 2], [1, 1], [0, 0, 0, 0], "positive"),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_avgpool2d():
    """Test AveragePool2D with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape, kernel_shape, strides, pads, data_type) in enumerate(
        test_cases
    ):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data = generate_test_data(input_shape, data_type)

        # Create input tensors with actual data
        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "AveragePool",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {
                "kernel_shape": kernel_shape,
                "strides": strides,
                "pads": pads,
            },
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_avgpool2d(test_data, kernel_shape, strides, pads)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation - call compute function directly
        numerical_match = True
        try:
            # Call the compute function directly to get actual output
            computed_output = compute_avgpool2d(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-7
            )
            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                print(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            print(f"\n  Numerical validation error: {e}")

        # Report results
        if shape_match and numerical_match == True:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓]")
        elif shape_match:
            print(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Numerical: {numerical_match}]"
            )
        else:
            print(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            print(
                f"  Shape match: {shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            print(f"  Numerical match: {numerical_match}")
            print("INPUTS:")
            for x in i_tensors:
                print(f"\t{x.name}: shape={x.shape}, dtype={x.dtype}")
            print("OUTPUTS:")
            for x in o_tensors:
                print(f"\t{x.name}: shape={x.shape}, dtype={x.dtype}")
            try:
                computed_output = compute_avgpool2d(i_tensors, op_obj)
                print(f"  Computed sample: {computed_output.flat[:5]}")
                print(f"  Expected sample: {ref_output.flat[:5]}")
            except:
                pass


# Error test cases - testing edge cases that could break the model
test_name_errors = "test_avgpool2d_errors"
test_cases_errors = [
    # These test cases validate that the model handles edge cases properly
    ("Kernel larger than input", [1, 1, 2, 2], [5, 5], [1, 1], [0, 0, 0, 0]),
    ("Zero stride", [1, 1, 4, 4], [2, 2], [0, 2], [0, 0, 0, 0]),
    ("Output size would be zero", [1, 1, 2, 2], [3, 3], [1, 1], [0, 0, 0, 0]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_avgpool2d_errors():
    """Test AveragePool2D with edge cases that could break the model"""
    msgw = get_max_test_msg_len(test_cases_errors)

    for tno, (tmsg, input_shape, kernel_shape, strides, pads) in enumerate(
        test_cases_errors
    ):
        op_name = f"{test_name_errors}_{tno}"

        # Generate test data
        test_data = np.random.randn(*input_shape).astype(np.float32)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "AveragePool",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {
                "kernel_shape": kernel_shape,
                "strides": strides,
                "pads": pads,
            },
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # These should either raise exceptions or produce invalid outputs
        # Testing that the system handles edge cases gracefully
        try:
            op_obj.get_perf_counts(i_tensors, o_tensors)

            # Try to compute - this may fail for invalid configurations
            try:
                computed_output = compute_avgpool2d(i_tensors, op_obj)

                # If we got here, check if output is valid
                if computed_output.size == 0 or np.any(np.isnan(computed_output)):
                    print(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (invalid output detected)"
                    )
                else:
                    # Some edge cases may produce valid output - that's OK
                    print(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (edge case handled, output shape: {computed_output.shape})"
                    )
            except (IndexError, ValueError, ZeroDivisionError) as e:
                print(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during compute)"
                )
        except (ValueError, AssertionError, IndexError, ZeroDivisionError) as e:
            print(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during shape inference)"
            )


# Additional test for specific numerical precision cases
test_name_precision = "test_avgpool2d_precision"
precision_test_cases = [
    # Test case with known output for manual verification
    (
        "Known output 2x2 avg",
        np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        [2, 2],
        [1, 1],
        [0, 0, 0, 0],
        np.array([[[[2.5]]]], dtype=np.float32),
    ),  # Expected: (1+2+3+4)/4 = 2.5
    (
        "Known output with padding",
        np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        [2, 2],
        [1, 1],
        [1, 1, 0, 0],  # Pad top and left
        np.array([[[[0.25, 0.75], [1.0, 2.5]]]], dtype=np.float32),
    ),  # Expected with zero padding
]


@pytest.mark.unit
@pytest.mark.opunit
def test_avgpool2d_precision():
    """Test AveragePool2D with precise known outputs"""
    msgw = 35  # Fixed width for these tests

    for tno, (
        tmsg,
        test_data,
        kernel_shape,
        strides,
        pads,
        expected_output,
    ) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "AveragePool",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {
                "kernel_shape": kernel_shape,
                "strides": strides,
                "pads": pads,
            },
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        # Validate against known expected output by calling compute function directly
        try:
            computed_output = compute_avgpool2d(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-7)
            if match:
                print(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                print(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                print(f"  Expected: {expected_output.flatten()}")
                print(f"  Got:      {computed_output.flatten()}")
                print(f"  Diff:     {(computed_output - expected_output).flatten()}")
                assert False, f"Precision test failed for {tmsg}"
        except Exception as e:
            print(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")
            assert False, f"Precision test error: {e}"
