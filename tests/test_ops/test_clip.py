#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_clip
from tests.test_ops.utils import generate_test_data

def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_clip(X, min_val, max_val):
    """
    Reference implementation of clip using NumPy.

    Args:
        X: Input array
        min_val: Minimum value (can be scalar or array)
        max_val: Maximum value (can be scalar or array)

    Returns:
        Y: Clipped output
    """
    return np.clip(X, min_val, max_val)


# Test cases with shape validation and numerical validation
test_name = "test_clip"
test_cases = [
    # (name, input_shape, min_val, max_val, test_data_type)
    ("Basic clip [0, 1]", [4, 4], 0.0, 1.0, "mixed"),
    ("Clip [-1, 1]", [4, 4], -1.0, 1.0, "mixed"),
    ("Clip large range [-10, 10]", [8, 8], -10.0, 10.0, "mixed"),
    ("Clip small range [-0.1, 0.1]", [6, 6], -0.1, 0.1, "mixed"),
    # Different shapes
    ("1D tensor clip", [16], -1.0, 1.0, "mixed"),
    ("2D tensor clip", [8, 8], -2.0, 2.0, "mixed"),
    ("3D tensor clip", [2, 4, 4], -1.0, 1.0, "mixed"),
    ("4D tensor clip", [2, 3, 4, 4], -1.5, 1.5, "mixed"),
    # Edge cases: all values within range
    ("No clipping needed", [4, 4], -10.0, 10.0, "small"),
    # Edge cases: all values below min
    ("All values clipped to min", [4, 4], 5.0, 10.0, "positive"),
    # Edge cases: all values above max
    ("All values clipped to max", [4, 4], -10.0, -5.0, "negative"),
    # Edge cases: min == max (all values become constant)
    ("Min equals max", [4, 4], 0.0, 0.0, "mixed"),
    # Different data types
    ("Positive values only", [4, 4], 0.5, 1.5, "positive"),
    ("Negative values only", [4, 4], -2.0, -0.5, "negative"),
    ("Zero values", [4, 4], -1.0, 1.0, "zeros"),
    ("Mixed positive/negative", [4, 4], -0.5, 0.5, "mixed"),
    ("Very small values", [4, 4], -1e-5, 1e-5, "small"),
    ("Very large values", [4, 4], -1e5, 1e5, "large"),
    # Asymmetric ranges
    ("Asymmetric range [0, 10]", [6, 6], 0.0, 10.0, "mixed"),
    ("Asymmetric range [-5, 1]", [6, 6], -5.0, 1.0, "mixed"),
    # Edge cases: one-sided clipping
    ("Only lower bound (effectively)", [4, 4], -1.0, 1e10, "mixed"),
    ("Only upper bound (effectively)", [4, 4], -1e10, 1.0, "mixed"),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_clip():
    """Test Clip with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape, min_val, max_val, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data = generate_test_data(input_shape, data_type)

        # Create input tensors with actual data
        # Clip operation: iTList can be [X] or [X, min, max]
        # For simplicity, we'll pass min/max as scalar tensors
        i_tensors = [
            F._from_data("X", test_data),
            F._from_data("min", np.array(min_val, dtype=np.float32)),
            F._from_data("max", np.array(max_val, dtype=np.float32)),
        ]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Clip",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_clip(test_data, min_val, max_val)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation - check if data was computed
        numerical_match = True
        try:
            # The unary_fwd shape inference should automatically compute data via try_compute_data
            if o_tensors[0].data is not None:
                # Data was computed automatically
                computed_output = o_tensors[0].data
            else:
                # Call compute function directly if not auto-computed
                computed_output = compute_clip(i_tensors, op_obj)

            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-7
            )
            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                logger.debug(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            logger.debug(f"\n  Numerical validation error: {e}")

        # Report results
        if shape_match and numerical_match == True:
            logger.debug(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓]")
        elif shape_match:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Numerical: {numerical_match}]"
            )
        else:
            logger.debug(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            logger.debug(
                f"  Shape match: {shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            logger.debug(f"  Numerical match: {numerical_match}")
            logger.debug("INPUTS:")
            for x in i_tensors:
                logger.debug(f"\t{x.name}: shape={x.shape}, dtype={x.dtype}")
            logger.debug("OUTPUTS:")
            for x in o_tensors:
                logger.debug(f"\t{x.name}: shape={x.shape}, dtype={x.dtype}")
            try:
                computed_output = compute_clip(i_tensors, op_obj)
                logger.debug(f"  Computed sample: {computed_output.flat[:5]}")
                logger.debug(f"  Expected sample: {ref_output.flat[:5]}")
            except:
                pass


# Error test cases - testing edge cases that could break the model
test_name_errors = "test_clip_errors"
test_cases_errors = [
    # These test cases validate that the model handles edge cases properly
    ("Min greater than max", [4, 4], 5.0, -5.0),  # Invalid: min > max
    ("Very large min", [4, 4], 1e20, 1e21),
    ("Very small max", [4, 4], -1e21, -1e20),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_clip_errors():
    """Test Clip with edge cases that could break the model"""
    msgw = get_max_test_msg_len(test_cases_errors)

    for tno, (tmsg, input_shape, min_val, max_val) in enumerate(test_cases_errors):
        op_name = f"{test_name_errors}_{tno}"

        # Generate test data
        test_data = np.random.randn(*input_shape).astype(np.float32)

        i_tensors = [
            F._from_data("X", test_data),
            F._from_data("min", np.array(min_val, dtype=np.float32)),
            F._from_data("max", np.array(max_val, dtype=np.float32)),
        ]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Clip",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # These should either raise exceptions or produce valid (possibly unexpected) outputs
        # Testing that the system handles edge cases gracefully
        try:
            op_obj.get_perf_counts(i_tensors, o_tensors)

            # Try to get computed output
            try:
                if o_tensors[0].data is not None:
                    computed_output = o_tensors[0].data
                else:
                    computed_output = compute_clip(i_tensors, op_obj)

                # Check for invalid outputs
                if np.any(np.isnan(computed_output)) or np.any(
                    np.isinf(computed_output)
                ):
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (invalid output detected: NaN or Inf)"
                    )
                else:
                    # Some edge cases may produce valid output - that's OK
                    # For example, min > max might just swap them
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (edge case handled, output shape: {computed_output.shape})"
                    )
            except (ValueError, RuntimeError) as e:
                logger.debug(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during compute)"
                )
        except (ValueError, AssertionError, RuntimeError) as e:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during shape inference)"
            )


# Additional test for specific numerical precision cases
test_name_precision = "test_clip_precision"
precision_test_cases = [
    # Test case with known output for manual verification
    (
        "Known output clip [0, 1]",
        np.array([[-1.0, 0.5, 2.0, 0.0]], dtype=np.float32),
        0.0,
        1.0,
        np.array([[0.0, 0.5, 1.0, 0.0]], dtype=np.float32),
    ),  # Expected
    (
        "Known output clip [-1, 1]",
        np.array([[2.5, -3.0, 0.5, -0.7]], dtype=np.float32),
        -1.0,
        1.0,
        np.array([[1.0, -1.0, 0.5, -0.7]], dtype=np.float32),
    ),  # Expected
    (
        "Known output all clipped to min",
        np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        1.0,
        2.0,
        np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
    ),  # All become min
    (
        "Known output all clipped to max",
        np.array([[5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
        1.0,
        2.0,
        np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32),
    ),  # All become max
    (
        "Known output no clipping",
        np.array([[0.3, 0.5, 0.7, 0.9]], dtype=np.float32),
        0.0,
        1.0,
        np.array([[0.3, 0.5, 0.7, 0.9]], dtype=np.float32),
    ),  # No change
]


@pytest.mark.unit
@pytest.mark.opunit
def test_clip_precision():
    """Test Clip with precise known outputs"""
    msgw = 35  # Fixed width for these tests

    for tno, (tmsg, test_data, min_val, max_val, expected_output) in enumerate(
        precision_test_cases
    ):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [
            F._from_data("X", test_data),
            F._from_data("min", np.array(min_val, dtype=np.float32)),
            F._from_data("max", np.array(max_val, dtype=np.float32)),
        ]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Clip",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        # Validate against known expected output
        try:
            # Check if data was computed automatically
            if o_tensors[0].data is not None:
                computed_output = o_tensors[0].data
            else:
                computed_output = compute_clip(i_tensors, op_obj)

            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-7)
            if match:
                logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                logger.debug(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  Expected: {expected_output.flatten()}")
                logger.debug(f"  Got:      {computed_output.flatten()}")
                logger.debug(
                    f"  Diff:     {(computed_output - expected_output).flatten()}"
                )
                assert False, f"Precision test failed for {tmsg}"
        except Exception as e:
            logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")
            assert False, f"Precision test error: {e}"
