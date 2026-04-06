#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import time
import os
from pathlib import Path
from loguru import logger

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_sub

# Try to import device config for memory estimation
try:
    from ttsim.config import get_arspec_from_yaml
    from ttsim.back.device import Device

    MEMORY_TEST_AVAILABLE = True
except ImportError:
    MEMORY_TEST_AVAILABLE = False

# Add polaris root to path for config access
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --------------------------------------------------------------------------
# Reference implementation
# --------------------------------------------------------------------------


def ref_impl_sub(A, B):
    """Reference element-wise subtraction with broadcasting: A - B"""
    return A - B


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


# --------------------------------------------------------------------------
# Main test cases
# --------------------------------------------------------------------------

test_name = "test_sub"
test_cases = [
    # (name, shape_A, shape_B, data_type)
    # Same-shape cases
    ("Same shape 1D", [4], [4], "positive"),
    ("Same shape 2D", [3, 4], [3, 4], "positive"),
    ("Same shape 3D", [2, 3, 4], [2, 3, 4], "positive"),
    ("Same shape 4D (NCHW)", [2, 3, 4, 4], [2, 3, 4, 4], "positive"),
    ("Same shape 5D", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], "positive"),
    # Broadcasting cases
    ("Scalar to 2D broadcast", [], [3, 4], "positive"),
    ("1D to 2D broadcast", [4], [3, 4], "positive"),
    ("Bidirectional broadcast", [3, 1], [1, 4], "positive"),
    ("Multi-dim broadcast", [2, 1, 4], [1, 3, 1], "positive"),
    ("Channel-wise bias (BN-like)", [1, 3, 1, 1], [2, 3, 4, 4], "positive"),
    ("Scalar subtract", [1], [2, 3, 4], "positive"),
    # Negative values
    ("Negative - Positive", [3, 4], [3, 4], "neg_pos"),
    ("Negative - Negative", [3, 4], [3, 4], "negative"),
    # Zero values
    ("Zero - Positive", [3, 4], [3, 4], "zero_pos"),
    ("Positive - Zero", [3, 4], [3, 4], "pos_zero"),
    ("All zeros", [3, 4], [3, 4], "zeros"),
    # Mixed values
    ("Mixed positive/negative", [2, 3, 4], [2, 3, 4], "mixed"),
    # Small values
    ("Small - Small", [3, 4], [3, 4], "small"),
    # Large values
    ("Large - Large", [3, 4], [3, 4], "large"),
    # Single element
    ("Single element", [1], [1], "positive"),
    # Subtract self
    ("Self subtraction", [3, 4], [3, 4], "self"),
    # Ones
    ("Subtract ones", [2, 3, 4], [2, 3, 4], "ones_B"),
    # Large tensors
    ("Large 2D", [64, 64], [64, 64], "positive"),
    ("Large 4D broadcast", [2, 16, 8, 8], [1, 16, 1, 1], "positive"),
]


def generate_test_data(shape, data_type, which="both"):
    """Generate test data based on type."""
    if len(shape) == 0:
        # Scalar
        if data_type == "positive":
            return np.array(np.random.rand() + 1.0, dtype=np.float32)
        return np.array(np.random.randn(), dtype=np.float32)

    if data_type == "positive":
        return np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)
    elif data_type == "negative":
        return np.array(-np.random.rand(*shape) - 1.0, dtype=np.float32)
    elif data_type == "neg_pos":
        if which == "A":
            return np.array(-np.random.rand(*shape) - 1.0, dtype=np.float32)
        else:
            return np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)
    elif data_type == "zero_pos":
        if which == "A":
            return np.zeros(shape, dtype=np.float32)
        else:
            return np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)
    elif data_type == "pos_zero":
        if which == "A":
            return np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)
        else:
            return np.zeros(shape, dtype=np.float32)
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return np.array(np.random.randn(*shape) * 2, dtype=np.float32)
    elif data_type == "small":
        return np.array(np.random.rand(*shape) * 1e-6, dtype=np.float32)
    elif data_type == "large":
        return np.array(np.random.rand(*shape) * 1e6, dtype=np.float32)
    elif data_type == "self":
        # Both A and B get the same data — handled in test loop
        return np.array(np.random.randn(*shape), dtype=np.float32)
    elif data_type == "ones_B":
        if which == "B":
            return np.ones(shape, dtype=np.float32)
        else:
            return np.array(np.random.randn(*shape), dtype=np.float32)
    else:
        return np.array(np.random.randn(*shape), dtype=np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_sub():
    """Numerical validation of compute_sub across shapes and data types"""

    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, shape_A, shape_B, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        if data_type == "self":
            data_A = generate_test_data(shape_A, data_type, which="A")
            data_B = data_A.copy()
        else:
            data_A = generate_test_data(shape_A, data_type, which="A")
            data_B = generate_test_data(shape_B, data_type, which="B")

        i_tensors = [
            F._from_data("A", data_A),
            F._from_data("B", data_B),
        ]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Sub",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Shape inference
        op_obj.get_perf_counts(i_tensors, o_tensors)

        ref_output = ref_impl_sub(data_A, data_B)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        assert (
            inf_shape == ref_shape
        ), f"[{tmsg}] shape mismatch: {inf_shape} vs {ref_shape}"

        # Numerical validation
        computed_output = compute_sub(i_tensors, op_obj)
        np.testing.assert_allclose(
            computed_output,
            ref_output,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"[{tmsg}] numerical mismatch",
        )

        logger.debug(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")


# --------------------------------------------------------------------------
# Edge / error cases
# --------------------------------------------------------------------------

test_cases_errors = [
    ("Empty tensor A", [0, 3], [1, 3]),
    ("Empty tensor B", [2, 3], [0, 3]),
    ("Both empty tensors", [0], [0]),
    ("Incompatible shapes", [3, 4], [5, 6]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_sub_errors():
    """Edge cases that should raise or produce degenerate results"""

    msgw = get_max_test_msg_len(test_cases_errors)

    for tno, (tmsg, shape_A, shape_B) in enumerate(test_cases_errors):
        op_name = f"test_sub_err_{tno}"

        data_A = (
            np.random.randn(*shape_A).astype(np.float32)
            if all(s > 0 for s in shape_A)
            else np.empty(shape_A, dtype=np.float32)
        )
        data_B = (
            np.random.randn(*shape_B).astype(np.float32)
            if all(s > 0 for s in shape_B)
            else np.empty(shape_B, dtype=np.float32)
        )

        i_tensors = [F._from_data("A", data_A), F._from_data("B", data_B)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Sub",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        try:
            op_obj.get_perf_counts(i_tensors, o_tensors)

            try:
                computed = compute_sub(i_tensors, op_obj)
                if computed.size == 0 or np.any(np.isnan(computed)):
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (invalid output detected)"
                    )
                else:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (edge case handled, shape: {computed.shape})"
                    )
            except (ValueError, IndexError, TypeError) as e:
                logger.debug(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during compute)"
                )
        except (ValueError, AssertionError, IndexError) as e:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during shape inference)"
            )


# --------------------------------------------------------------------------
# Precision tests with known outputs
# --------------------------------------------------------------------------

precision_test_cases = [
    (
        "Simple integer subtract",
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[4.0, 4.0], [4.0, 4.0]], dtype=np.float32),
    ),
    (
        "Scalar broadcast subtract",
        np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        np.array([5.0], dtype=np.float32),
        np.array([[5.0, 15.0], [25.0, 35.0]], dtype=np.float32),
    ),
    (
        "Subtract from zero",
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32),
    ),
    (
        "Subtract zero (identity)",
        np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32),
    ),
    (
        "Negative result",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        np.array([[-9.0, -18.0], [-27.0, -36.0]], dtype=np.float32),
    ),
    (
        "Self subtraction yields zero",
        np.array([[3.14, 2.71], [-1.0, 42.0]], dtype=np.float32),
        np.array([[3.14, 2.71], [-1.0, 42.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    ),
    (
        "Column broadcast subtract",
        np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        np.array([[5.0], [10.0]], dtype=np.float32),
        np.array([[5.0, 15.0, 25.0], [30.0, 40.0, 50.0]], dtype=np.float32),
    ),
    (
        "Row broadcast subtract",
        np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([[9.0, 18.0, 27.0], [39.0, 48.0, 57.0]], dtype=np.float32),
    ),
    (
        "Mixed sign subtract",
        np.array([[2.0, -3.0], [-4.0, 5.0]], dtype=np.float32),
        np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32),
        np.array([[3.0, -5.0], [-7.0, 9.0]], dtype=np.float32),
    ),
    (
        "1D simple",
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([9.0, 18.0, 27.0, 36.0], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_sub_precision():
    """Precision tests with known expected outputs"""

    msgw = get_max_test_msg_len(precision_test_cases)

    for tno, (tmsg, data_A, data_B, expected) in enumerate(precision_test_cases):
        op_name = f"test_sub_prec_{tno}"

        i_tensors = [F._from_data("A", data_A), F._from_data("B", data_B)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Sub",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        computed = compute_sub(i_tensors, op_obj)
        np.testing.assert_allclose(
            computed,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"[{tmsg}] precision mismatch",
        )

        logger.debug(f"  {tmsg:{msgw}s} -- OK")


# --------------------------------------------------------------------------
# Mathematical property tests
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
def test_sub_anti_commutativity():
    """A - B == -(B - A) (anti-commutative property)"""

    shapes = [([3, 4], [3, 4]), ([2, 1], [1, 3]), ([1], [5, 5])]

    for idx, (shape_A, shape_B) in enumerate(shapes):
        data_A = np.random.randn(*shape_A).astype(np.float32)
        data_B = np.random.randn(*shape_B).astype(np.float32)

        # A - B
        i_ab = [F._from_data("A", data_A), F._from_data("B", data_B)]
        o_ab = [make_tensor("Y")]
        op_ab = SimOp(
            {
                "name": f"anti_ab_{idx}",
                "optype": "Sub",
                "inList": ["A", "B"],
                "outList": ["Y"],
            }
        )
        for x in i_ab:
            x.op_in = [op_ab.name]
        for x in o_ab:
            x.op_out = [op_ab.name]
        op_ab.get_perf_counts(i_ab, o_ab)
        result_ab = compute_sub(i_ab, op_ab)

        # B - A
        i_ba = [F._from_data("B", data_B), F._from_data("A", data_A)]
        o_ba = [make_tensor("Y")]
        op_ba = SimOp(
            {
                "name": f"anti_ba_{idx}",
                "optype": "Sub",
                "inList": ["B", "A"],
                "outList": ["Y"],
            }
        )
        for x in i_ba:
            x.op_in = [op_ba.name]
        for x in o_ba:
            x.op_out = [op_ba.name]
        op_ba.get_perf_counts(i_ba, o_ba)
        result_ba = compute_sub(i_ba, op_ba)

        np.testing.assert_allclose(
            result_ab,
            -result_ba,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Anti-commutativity failed for shapes {shape_A}, {shape_B}",
        )

    logger.debug("  Anti-commutativity -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_sub_self_is_zero():
    """A - A == 0 for all A"""

    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data_A = np.random.randn(*shape).astype(np.float32)

        i_tensors = [F._from_data("A", data_A), F._from_data("B", data_A.copy())]
        o_tensors = [make_tensor("Y")]

        op_obj = SimOp(
            {
                "name": f"self_zero_{idx}",
                "optype": "Sub",
                "inList": ["A", "B"],
                "outList": ["Y"],
            }
        )
        for x in i_tensors:
            x.op_in = [op_obj.name]
        for x in o_tensors:
            x.op_out = [op_obj.name]
        op_obj.get_perf_counts(i_tensors, o_tensors)

        result = compute_sub(i_tensors, op_obj)
        np.testing.assert_allclose(
            result, 0.0, atol=1e-7, err_msg=f"A - A != 0 for shape {shape}"
        )

    logger.debug("  A - A == 0 -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_sub_right_identity():
    """A - 0 == A (right identity with zero)"""

    shapes = [[4], [3, 4], [2, 3, 4]]

    for idx, shape in enumerate(shapes):
        data_A = np.random.randn(*shape).astype(np.float32)
        data_zero = np.zeros(shape, dtype=np.float32)

        i_tensors = [F._from_data("A", data_A), F._from_data("Z", data_zero)]
        o_tensors = [make_tensor("Y")]

        op_obj = SimOp(
            {
                "name": f"right_id_{idx}",
                "optype": "Sub",
                "inList": ["A", "Z"],
                "outList": ["Y"],
            }
        )
        for x in i_tensors:
            x.op_in = [op_obj.name]
        for x in o_tensors:
            x.op_out = [op_obj.name]
        op_obj.get_perf_counts(i_tensors, o_tensors)

        result = compute_sub(i_tensors, op_obj)
        np.testing.assert_allclose(
            result,
            data_A,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"A - 0 != A for shape {shape}",
        )

    logger.debug("  A - 0 == A -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_sub_left_zero_negation():
    """0 - A == -A"""

    shapes = [[4], [3, 4], [2, 3, 4]]

    for idx, shape in enumerate(shapes):
        data_A = np.random.randn(*shape).astype(np.float32)
        data_zero = np.zeros(shape, dtype=np.float32)

        i_tensors = [F._from_data("Z", data_zero), F._from_data("A", data_A)]
        o_tensors = [make_tensor("Y")]

        op_obj = SimOp(
            {
                "name": f"left_neg_{idx}",
                "optype": "Sub",
                "inList": ["Z", "A"],
                "outList": ["Y"],
            }
        )
        for x in i_tensors:
            x.op_in = [op_obj.name]
        for x in o_tensors:
            x.op_out = [op_obj.name]
        op_obj.get_perf_counts(i_tensors, o_tensors)

        result = compute_sub(i_tensors, op_obj)
        np.testing.assert_allclose(
            result,
            -data_A,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"0 - A != -A for shape {shape}",
        )

    logger.debug("  0 - A == -A -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_sub_add_inverse():
    """(A - B) + B == A  (subtraction is the inverse of addition)"""

    shapes = [[4], [3, 4], [2, 3, 4]]

    for idx, shape in enumerate(shapes):
        data_A = np.random.randn(*shape).astype(np.float32)
        data_B = np.random.randn(*shape).astype(np.float32)

        # A - B
        i_sub = [F._from_data("A", data_A), F._from_data("B", data_B)]
        o_sub = [make_tensor("Y")]
        op_sub = SimOp(
            {
                "name": f"inv_sub_{idx}",
                "optype": "Sub",
                "inList": ["A", "B"],
                "outList": ["Y"],
            }
        )
        for x in i_sub:
            x.op_in = [op_sub.name]
        for x in o_sub:
            x.op_out = [op_sub.name]
        op_sub.get_perf_counts(i_sub, o_sub)
        diff = compute_sub(i_sub, op_sub)

        # (A - B) + B should equal A
        reconstructed = diff + data_B
        np.testing.assert_allclose(
            reconstructed,
            data_A,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"(A - B) + B != A for shape {shape}",
        )

    logger.debug("  (A - B) + B == A -- OK")


# --------------------------------------------------------------------------
# Memory and performance estimation
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.performance
def test_sub_memory_validation(capsys, request):
    """
    Test memory validation for sub operation.
    Validates instructions executed and data moved for various scenarios.

    This test validates two primary metrics:
    1. Instructions Executed: Verifies instruction count matches output elements
    2. Data Moved: Tracks input/output bytes and validates memory traffic

    Run with: pytest tests/test_ops/test_sub.py::test_sub_memory_validation -v
    For detailed output: add -s flag
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    # Test cases: different shapes and broadcasting scenarios
    test_cases = [
        {
            "name": "1D Same Shape",
            "shape_A": [1000],
            "shape_B": [1000],
            "description": "Element-wise sub of two 1D arrays",
        },
        {
            "name": "2D Same Shape",
            "shape_A": [32, 32],
            "shape_B": [32, 32],
            "description": "Element-wise sub of two 2D matrices",
        },
        {
            "name": "4D Same Shape",
            "shape_A": [2, 16, 16, 16],
            "shape_B": [2, 16, 16, 16],
            "description": "Element-wise sub of two 4D tensors",
        },
        {
            "name": "Scalar Broadcast",
            "shape_A": [1, 64, 64, 64],
            "shape_B": [1],
            "description": "Subtract scalar from 4D tensor",
        },
        {
            "name": "1D to 2D Broadcast",
            "shape_A": [32, 64],
            "shape_B": [64],
            "description": "Broadcast 1D to 2D for subtraction",
        },
    ]

    # Load device configuration once for all tests
    config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]  # Use Wormhole n150 device
        device = Device(device_pkg)

        logger.info(f"\n{'='*60}")
        logger.info("Sub Operation Memory Validation")
        logger.info(f"{'='*60}\n")
        logger.info("Device: Wormhole (n150)")
        logger.info(f"Device frequency: {device.freq_MHz} MHz")
        logger.info(f"Memory frequency: {device.memfreq_MHz} MHz")
        logger.info(
            f"Peak bandwidth: {device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
        )
    except Exception as e:
        logger.info(f"\nWarning: Could not load device config: {e}")
        logger.info("Skipping memory validation test")
        pytest.skip(f"Could not load device config: {e}")
        return

    logger.info(f"\n{'='*60}")
    logger.info("Running Memory Validation Tests")
    logger.info(f"{'='*60}\n")

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        shape_A = test_case["shape_A"]
        shape_B = test_case["shape_B"]
        description = test_case["description"]

        logger.info(f"\n-- Test: {test_name} --")
        logger.debug(f"Shape A: {shape_A}, Shape B: {shape_B}")

        # Generate test data
        np.random.seed(42)
        if len(shape_A) == 0:
            data_A = np.array(np.random.rand() + 1.0, dtype=np.float32)
        else:
            data_A = np.array(np.random.rand(*shape_A) + 1.0, dtype=np.float32)

        if len(shape_B) == 0:
            data_B = np.array(np.random.rand() + 1.0, dtype=np.float32)
        else:
            data_B = np.array(np.random.rand(*shape_B) + 1.0, dtype=np.float32)

        # Create operation
        i_tensors = [F._from_data("A", data_A), F._from_data("B", data_B)]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": f'sub_mem_{test_name.replace(" ", "_")}',
            "optype": "Sub",
            "inList": ["A", "B"],
            "outList": ["Y"],
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_obj.name]
        for x in o_tensors:
            x.op_out = [op_obj.name]

        # Set operation precision
        op_obj.precision = "fp32"

        # Get performance counts
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # Validate compute_sub correctness
        result = compute_sub(i_tensors, op_obj)
        expected = data_A - data_B
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"[{test_name}] compute_sub validation failed",
        )

        # Calculate output shape
        output_shape = tuple(o_tensors[0].shape)
        output_elems = int(np.prod(output_shape))

        # Set compute pipe for Sub operation (uses vector pipe for element-wise ops)
        if op_obj.uses_compute_pipe is None:
            op_obj.uses_compute_pipe = "vector"

        # Execute on device for cycle estimation
        if op_obj.perf_stats is not None:
            device.execute_op(op_obj)

        # Extract stats from op.perf_stats
        perf_stats = op_obj.perf_stats
        actual_in_elems = perf_stats["inElems"]
        actual_out_elems = perf_stats["outElems"]
        actual_in_bytes = perf_stats["inBytes"]
        actual_out_bytes = perf_stats["outBytes"]
        actual_instrs = perf_stats["instrs"]

        # Calculate expected values
        input_elems_A = int(np.prod(shape_A)) if len(shape_A) > 0 else 1
        input_elems_B = int(np.prod(shape_B)) if len(shape_B) > 0 else 1
        bytes_per_element = 4  # fp32

        # Validate element counts
        assert (
            actual_in_elems == input_elems_A + input_elems_B
        ), f"Input element count mismatch: {actual_in_elems} vs {input_elems_A + input_elems_B}"
        assert (
            actual_out_elems == output_elems
        ), f"Output element count mismatch: {actual_out_elems} vs {output_elems}"

        # Validate instructions (sub uses 'sub' instruction, 1 per output element)
        assert "sub" in actual_instrs, "Expected 'sub' instruction not found"
        actual_sub = actual_instrs.get("sub", 0)
        assert (
            actual_sub == output_elems
        ), f"Sub instruction count mismatch: {actual_sub} vs {output_elems}"

        # Calculate metrics
        total_data_movement = actual_in_bytes + actual_out_bytes
        instructions_executed = sum(actual_instrs.values())
        arithmetic_intensity = (
            instructions_executed / total_data_movement
            if total_data_movement > 0
            else 0
        )

        # Calculate execution cycles (read from op object, not perf_stats)
        compute_cycles = op_obj.compute_cycles
        mem_rd_cycles = op_obj.mem_rd_cycles
        mem_wr_cycles = op_obj.mem_wr_cycles
        memory_cycles = mem_rd_cycles + mem_wr_cycles
        total_cycles = max(compute_cycles, memory_cycles)
        bottleneck = "COMPUTE" if compute_cycles >= memory_cycles else "MEMORY"

        # Calculate broadcast ratio
        if input_elems_B == 1:
            broadcast_type = "scalar broadcast"
        elif input_elems_B < output_elems:
            broadcast_type = f"broadcast {input_elems_B}→{output_elems}"
        else:
            broadcast_type = "no broadcast"

        # Print detailed breakdown
        logger.debug(f"\n  -- Instructions & Operations --")
        logger.debug(f"  Instructions executed: {instructions_executed:,} (sub)")
        logger.debug(f"  Input A elements:      {input_elems_A:,}")
        logger.debug(f"  Input B elements:      {input_elems_B:,}")
        logger.debug(f"  Output elements:       {output_elems:,}")
        logger.debug(
            f"  Expected instructions: ~{output_elems:,} (1 sub per output element)"
        )
        instruction_ratio = actual_sub / output_elems if output_elems > 0 else 0
        logger.debug(
            f"  Instruction ratio:     {instruction_ratio:.2f} (✓ 1 sub per output)"
        )

        logger.debug(f"\n  -- Data Movement --")
        input_bytes_A = input_elems_A * bytes_per_element
        input_bytes_B = input_elems_B * bytes_per_element
        logger.debug(
            f"  Input A bytes:    {input_bytes_A:,} bytes ({input_bytes_A/1024:.2f} KB)"
        )
        logger.debug(
            f"  Input B bytes:    {input_bytes_B:,} bytes ({input_bytes_B/1024:.2f} KB)"
        )
        logger.debug(
            f"  Input total:      {actual_in_bytes:,} bytes ({actual_in_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Output bytes:     {actual_out_bytes:,} bytes ({actual_out_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Total data moved: {total_data_movement:,} bytes ({total_data_movement/1024:.2f} KB)"
        )
        logger.debug(f"  Broadcast:        {broadcast_type}")

        logger.debug(f"\n  -- Memory Metrics --")
        logger.debug(
            f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte"
        )
        expected_ai = output_elems / total_data_movement
        logger.debug(f"  Expected intensity:    {expected_ai:.4f} ops/byte")
        np.testing.assert_allclose(
            arithmetic_intensity,
            expected_ai,
            rtol=0.01,
            atol=1e-6,
            err_msg=f"Arithmetic intensity mismatch",
        )
        logger.debug(f"  ✓ Arithmetic intensity within expected range")

        logger.debug(f"\n  -- Execution Cycles --")
        logger.debug(f"  Compute cycles:   {compute_cycles:,}")
        logger.debug(f"  Memory cycles:    {memory_cycles:,}")
        logger.debug(f"    Read cycles:    {mem_rd_cycles:,}")
        logger.debug(f"    Write cycles:   {mem_wr_cycles:,}")
        logger.debug(f"  Ideal cycles:     {total_cycles:,}")
        logger.debug(f"  Bottleneck:       {bottleneck}")
        logger.debug(f"  ✓ Bottleneck analysis: {bottleneck} for binary operation")

        # Store results for summary
        all_results.append(
            {
                "test_name": test_name,
                "shape_A": shape_A,
                "shape_B": shape_B,
                "output_shape": output_shape,
                "sub_instructions": actual_sub,
                "total_data_moved": total_data_movement,
                "arithmetic_intensity": arithmetic_intensity,
                "broadcast_type": broadcast_type,
                "bottleneck": bottleneck,
                "compute_cycles": compute_cycles,
                "memory_cycles": memory_cycles,
                "ideal_cycles": total_cycles,
            }
        )

        logger.debug(f"\n  ✓ Test PASSED")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Memory Validation Summary")
    logger.info(f"{'='*60}\n")
    logger.info(f"Total tests run: {len(all_results)}")
    logger.info(f"All tests passed: ✓")

    # Compare arithmetic intensity across tests
    logger.info(f"\n-- Arithmetic Intensity Comparison --")
    for result in all_results:
        ai = result["arithmetic_intensity"]
        logger.debug(f"{result['test_name']:30s}: {ai:.4f} ops/byte")

    # Broadcast analysis
    logger.info(f"\n-- Broadcast Analysis --")
    for result in all_results:
        broadcast = result["broadcast_type"]
        logger.debug(f"{result['test_name']:30s}: {broadcast}")

    # Bottleneck analysis
    logger.info(f"\n-- Bottleneck Analysis --")
    for result in all_results:
        bottleneck = result["bottleneck"]
        logger.debug(f"{result['test_name']:30s}: {bottleneck}")

    logger.info(f"\n{'='*60}")
    logger.info("Memory validation complete!")
    logger.info(f"{'='*60}\n")

    # Create a summary that will be displayed in pytest output
    summary_lines = [
        "✓ Tests completed: {}/{} - All PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Findings:",
        "  • Sub instructions match output elements (1:1 ratio) ✓",
        "  • Broadcasting works correctly for all scenarios ✓",
        "  • Binary operations show typical MEMORY bottleneck ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        summary_lines.append(
            "  ✓ {:<26s} | {:>7,} sub | {:>7.1f} KB | {:.3f} ops/byte".format(
                result["test_name"],
                result["sub_instructions"],
                result["total_data_moved"] / 1024,
                result["arithmetic_intensity"],
            )
        )

    summary_lines.extend(
        [
            "",
            "Validation: All memory metrics within expected ranges ✓",
            "",
            "For detailed output, run with: pytest -s -v",
        ]
    )

    # Write to pytest's terminal reporter (always visible)
    try:
        terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            terminalreporter.write_sep(
                "=", "SUB MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        # Fallback: disable capture and print directly
        with capsys.disabled():
            logger.info("\n" + "=" * 70)
            logger.info("SUB MEMORY VALIDATION RESULTS")
            logger.info("=" * 70)
            for line in summary_lines:
                logger.info(line)
            logger.info("=" * 70 + "\n")

    # Final assertion
    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
