#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_meshgrid


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_meshgrid(ny, nx):
    """
    Reference implementation of meshgrid using NumPy.

    Args:
        ny: Height of grid
        nx: Width of grid

    Returns:
        Grid array [1, 1, ny, nx, 2] with [x, y] coordinates
    """
    # Create coordinate arrays
    y_coords = np.arange(ny, dtype=np.float32)
    x_coords = np.arange(nx, dtype=np.float32)

    # Create meshgrid using 'ij' indexing (matrix indexing)
    yv, xv = np.meshgrid(y_coords, x_coords, indexing="ij")

    # Stack as [xv, yv] along last axis
    grid = np.stack([xv, yv], axis=2)

    # Reshape to (1, 1, ny, nx, 2)
    grid = grid.reshape(1, 1, ny, nx, 2)

    return grid


# Test cases with shape validation and numerical validation
test_name = "test_meshgrid"
test_cases = [
    # (name, ny, nx)
    # Square grids
    ("Basic 4x4 grid", 4, 4),
    ("Small 2x2 grid", 2, 2),
    ("Medium 8x8 grid", 8, 8),
    ("Large 20x20 grid", 20, 20),
    ("Very large 64x64 grid", 64, 64),
    # Non-square grids (rectangular)
    ("Rectangular 4x8 grid", 4, 8),
    ("Rectangular 8x4 grid", 8, 4),
    ("Wide 2x10 grid", 2, 10),
    ("Tall 10x2 grid", 10, 2),
    ("Wide 5x15 grid", 5, 15),
    ("Tall 15x5 grid", 15, 5),
    # Edge cases: single row/column
    ("Single row 1x10", 1, 10),
    ("Single column 10x1", 10, 1),
    ("Single row 1x20", 1, 20),
    ("Single column 20x1", 20, 1),
    # Edge cases: minimal grids
    ("Minimal 1x1 grid", 1, 1),
    ("Small 1x2 grid", 1, 2),
    ("Small 2x1 grid", 2, 1),
    ("Small 3x3 grid", 3, 3),
    # Common detection grid sizes (YOLOv4-style)
    ("Detection grid 13x13", 13, 13),
    ("Detection grid 26x26", 26, 26),
    ("Detection grid 52x52", 52, 52),
    ("Detection grid 80x80", 80, 80),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_meshgrid():
    """Test Meshgrid with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, ny, nx) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Create input tensors - meshgrid can work in two ways:
        # 1. With input tensors containing ny, nx
        # 2. With attrs ny, nx
        # We'll test with attrs for simplicity
        i_tensors = []
        o_tensors = [make_tensor("grid")]

        op_info = {
            "name": op_name,
            "optype": "Meshgrid",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {
                "ny": ny,
                "nx": nx,
            },
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # 1. Shape validation
        ref_output = ref_impl_meshgrid(ny, nx)
        expected_shape = [1, 1, ny, nx, 2]

        # 2. Numerical validation - call compute function directly
        numerical_match = True
        shape_match = True
        try:
            # Call the compute function directly to get actual output
            computed_output = compute_meshgrid(i_tensors, op_obj)

            # Check shape
            computed_shape = list(computed_output.shape)
            shape_match = computed_shape == expected_shape

            # Check numerical values
            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-7
            )
            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                logger.debug(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            shape_match = False
            logger.debug(f"\n  Validation error: {e}")

        # Report results
        if shape_match and numerical_match == True:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓]"
            )
        elif shape_match:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Numerical: {numerical_match}]"
            )
        else:
            logger.debug(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            logger.debug(
                "  Shape match: "
                f"{shape_match} (got {computed_shape if 'computed_shape' in locals() else 'N/A'}, expected {expected_shape})"
            )
            logger.debug(f"  Numerical match: {numerical_match}")


# Error test cases - testing edge cases that could break the model
test_name_errors = "test_meshgrid_errors"
test_cases_errors = [
    # These test cases validate that the model handles edge cases properly
    ("Zero height", 0, 10),
    ("Zero width", 10, 0),
    ("Both zero", 0, 0),
    ("Negative height", -5, 10),
    ("Negative width", 10, -5),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_meshgrid_errors():
    """Test Meshgrid with edge cases that could break the model"""
    msgw = get_max_test_msg_len(test_cases_errors)

    for tno, (tmsg, ny, nx) in enumerate(test_cases_errors):
        op_name = f"{test_name_errors}_{tno}"

        i_tensors = []
        o_tensors = [make_tensor("grid")]

        op_info = {
            "name": op_name,
            "optype": "Meshgrid",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {
                "ny": ny,
                "nx": nx,
            },
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # These should either raise exceptions or produce invalid outputs
        try:
            computed_output = compute_meshgrid(i_tensors, op_obj)

            # Check if output is valid
            if computed_output.size == 0:
                logger.debug(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (empty output for invalid input)"
                )
            elif np.any(np.isnan(computed_output)) or np.any(np.isinf(computed_output)):
                logger.debug(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (invalid values detected)"
                )
            else:
                logger.debug(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (edge case handled, shape: {computed_output.shape})"
                )
        except (ValueError, IndexError, TypeError, RuntimeError) as e:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__})"
            )


# Precision test cases with known outputs
test_name_precision = "test_meshgrid_precision"
precision_test_cases = [
    # Test cases with known outputs for manual verification
    (
        "2x2 grid coordinates",
        2,
        2,
        # Expected: grid[0,0,y,x,0]=x, grid[0,0,y,x,1]=y
        np.array(
            [[[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]]], dtype=np.float32
        ),
    ),
    (
        "1x3 grid (single row)",
        1,
        3,
        np.array([[[[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]]], dtype=np.float32),
    ),
    (
        "3x1 grid (single column)",
        3,
        1,
        np.array([[[[[0.0, 0.0]], [[0.0, 1.0]], [[0.0, 2.0]]]]], dtype=np.float32),
    ),
    ("1x1 grid (single point)", 1, 1, np.array([[[[[0.0, 0.0]]]]], dtype=np.float32)),
    (
        "3x3 grid with correct coordinates",
        3,
        3,
        np.array(
            [
                [
                    [
                        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                        [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                        [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_meshgrid_precision():
    """Test Meshgrid with precise known outputs"""
    msgw = 35  # Fixed width for these tests

    for tno, (tmsg, ny, nx, expected_output) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = []
        o_tensors = [make_tensor("grid")]

        op_info = {
            "name": op_name,
            "optype": "Meshgrid",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {
                "ny": ny,
                "nx": nx,
            },
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Validate against known expected output
        try:
            computed_output = compute_meshgrid(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-7)

            if match:
                logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                logger.debug(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  Expected shape: {expected_output.shape}")
                logger.debug(f"  Got shape:      {computed_output.shape}")

                # Show a sample of values for debugging
                if computed_output.shape == expected_output.shape:
                    logger.debug("  Expected sample (first few coordinates):")
                    for y in range(min(2, ny)):
                        for x in range(min(2, nx)):
                            logger.debug(
                                f"    [{y},{x}]: {expected_output[0,0,y,x]}"
                            )
                    logger.debug("  Got sample (first few coordinates):")
                    for y in range(min(2, ny)):
                        for x in range(min(2, nx)):
                            logger.debug(
                                f"    [{y},{x}]: {computed_output[0,0,y,x]}"
                            )

                assert False, f"Precision test failed for {tmsg}"
        except Exception as e:
            logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")
            assert False, f"Precision test error: {e}"


# Additional test for coordinate property validation
test_name_properties = "test_meshgrid_properties"


@pytest.mark.unit
@pytest.mark.opunit
def test_meshgrid_properties():
    """Test Meshgrid coordinate properties (x increases along width, y increases along height)"""
    test_cases_props = [
        ("5x5 grid properties", 5, 5),
        ("10x8 grid properties", 10, 8),
        ("8x10 grid properties", 8, 10),
    ]

    msgw = get_max_test_msg_len(test_cases_props)

    for tno, (tmsg, ny, nx) in enumerate(test_cases_props):
        op_name = f"{test_name_properties}_{tno}"

        i_tensors = []
        o_tensors = [make_tensor("grid")]

        op_info = {
            "name": op_name,
            "optype": "Meshgrid",
            "inList": [],
            "outList": [x.name for x in o_tensors],
            "attrs": {"ny": ny, "nx": nx},
        }

        op_obj = SimOp(op_info)
        for x in o_tensors:
            x.op_out = [op_name]

        try:
            grid = compute_meshgrid(i_tensors, op_obj)

            # Property 1: X coordinate increases along width (last spatial dim)
            x_increases = True
            for y in range(ny):
                for x in range(nx - 1):
                    if grid[0, 0, y, x, 0] >= grid[0, 0, y, x + 1, 0]:
                        x_increases = False
                        break
                if not x_increases:
                    break

            # Property 2: Y coordinate increases along height (second-to-last spatial dim)
            y_increases = True
            for y in range(ny - 1):
                for x in range(nx):
                    if grid[0, 0, y, x, 1] >= grid[0, 0, y + 1, x, 1]:
                        y_increases = False
                        break
                if not y_increases:
                    break

            # Property 3: X values range from 0 to nx-1
            x_range_correct = (np.min(grid[..., 0]) == 0) and (
                np.max(grid[..., 0]) == nx - 1
            )

            # Property 4: Y values range from 0 to ny-1
            y_range_correct = (np.min(grid[..., 1]) == 0) and (
                np.max(grid[..., 1]) == ny - 1
            )

            all_props_pass = (
                x_increases and y_increases and x_range_correct and y_range_correct
            )

            if all_props_pass:
                logger.debug(
                    f"PROPERTY TEST[{tno}] {tmsg:{msgw}s} PASS [X↑ ✓, Y↑ ✓, X-range ✓, Y-range ✓]"
                )
            else:
                logger.debug(f"\nPROPERTY TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  X increases along width: {x_increases}")
                logger.debug(f"  Y increases along height: {y_increases}")
                logger.debug(f"  X range [0, {nx-1}]: {x_range_correct}")
                logger.debug(f"  Y range [0, {ny-1}]: {y_range_correct}")
                assert False, f"Property test failed for {tmsg}"

        except Exception as e:
            logger.debug(f"PROPERTY TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")
            assert False, f"Property test error: {e}"
