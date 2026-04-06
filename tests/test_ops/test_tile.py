#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_tile

# --------------------------------------------------------------------------
# Reference implementation
# --------------------------------------------------------------------------


def ref_impl_tile(data, repeats):
    """Reference tile: np.tile(data, repeats)"""
    return np.tile(data, repeats)


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def _run_tile(data, repeats, op_name):
    """Helper: build tensors + op, run shape inference + compute."""
    repeats_arr = np.array(repeats, dtype=np.int64)

    i_tensors = [F._from_data("X", data), F._from_data("R", repeats_arr)]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": op_name,
        "optype": "Tile",
        "inList": ["X", "R"],
        "outList": ["Y"],
    }
    op_obj = SimOp(op_info)
    for x in i_tensors:
        x.op_in = [op_name]
    for x in o_tensors:
        x.op_out = [op_name]

    op_obj.get_perf_counts(i_tensors, o_tensors)
    computed = compute_tile(i_tensors, op_obj)
    return computed, o_tensors


# --------------------------------------------------------------------------
# Main test cases  (name, input_shape, repeats)
# --------------------------------------------------------------------------

test_name = "test_tile"
test_cases = [
    # 1D
    ("1D repeat x1", [4], [1]),
    ("1D repeat x2", [4], [2]),
    ("1D repeat x3", [3], [3]),
    ("1D repeat x5", [6], [5]),
    # 2D
    ("2D repeat rows only", [2, 3], [2, 1]),
    ("2D repeat cols only", [2, 3], [1, 3]),
    ("2D repeat both", [2, 3], [3, 2]),
    ("2D no repeat", [4, 5], [1, 1]),
    # 3D
    ("3D repeat axis0", [2, 3, 4], [2, 1, 1]),
    ("3D repeat axis1", [2, 3, 4], [1, 3, 1]),
    ("3D repeat axis2", [2, 3, 4], [1, 1, 2]),
    ("3D repeat all", [2, 3, 4], [2, 2, 2]),
    # 4D (NCHW)
    ("4D repeat batch", [2, 3, 4, 4], [2, 1, 1, 1]),
    ("4D repeat spatial", [1, 3, 2, 2], [1, 1, 3, 3]),
    ("4D repeat all", [1, 2, 2, 2], [2, 2, 2, 2]),
    # Single element
    ("Single element x1", [1], [1]),
    ("Single element x10", [1], [10]),
    # Large
    ("Large 2D", [32, 16], [2, 2]),
    ("Large 1D", [128], [4]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_tile():
    """Numerical validation of compute_tile across shapes and repeat patterns"""

    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, shape, repeats) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        data = np.array(np.random.randn(*shape), dtype=np.float32)
        expected = ref_impl_tile(data, repeats)

        computed, o_tensors = _run_tile(data, repeats, op_name)

        # Shape check
        assert o_tensors[0].shape == list(
            expected.shape
        ), f"[{tmsg}] shape mismatch: {o_tensors[0].shape} vs {list(expected.shape)}"

        # Numerical check
        np.testing.assert_allclose(
            computed,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"[{tmsg}] numerical mismatch",
        )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

edge_cases = [
    ("All zeros tiled", [3, 4], [2, 2], "zeros"),
    ("All ones tiled", [2, 3], [3, 1], "ones"),
    ("Negative values", [3, 4], [2, 2], "negative"),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_tile_edge_cases():
    """Edge cases for Tile"""

    msgw = get_max_test_msg_len(edge_cases)

    for tno, (tmsg, shape, repeats, data_gen) in enumerate(edge_cases):
        op_name = f"test_tile_edge_{tno}"

        if data_gen == "zeros":
            data = np.zeros(shape, dtype=np.float32)
        elif data_gen == "ones":
            data = np.ones(shape, dtype=np.float32)
        elif data_gen == "negative":
            data = np.array(-np.random.rand(*shape) - 1, dtype=np.float32)
        else:
            data = np.array(np.random.randn(*shape), dtype=np.float32)

        expected = ref_impl_tile(data, repeats)
        computed, _ = _run_tile(data, repeats, op_name)

        np.testing.assert_allclose(
            computed, expected, rtol=1e-5, atol=1e-7, err_msg=f"[{tmsg}] mismatch"
        )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Precision tests with known outputs
# --------------------------------------------------------------------------

precision_test_cases = [
    (
        "1D tile x3",
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        [3],
        np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float32),
    ),
    (
        "2D tile rows",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        [2, 1],
        np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    ),
    (
        "2D tile cols",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        [1, 3],
        np.array(
            [[1.0, 2.0, 1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0, 3.0, 4.0]],
            dtype=np.float32,
        ),
    ),
    (
        "2D tile both",
        np.array([[1.0, 2.0]], dtype=np.float32),
        [3, 2],
        np.array(
            [[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0]],
            dtype=np.float32,
        ),
    ),
    (
        "Identity tile",
        np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32),
        [1, 1],
        np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32),
    ),
    (
        "Single element tile",
        np.array([42.0], dtype=np.float32),
        [5],
        np.array([42.0, 42.0, 42.0, 42.0, 42.0], dtype=np.float32),
    ),
    (
        "All zeros tile",
        np.array([[0.0, 0.0]], dtype=np.float32),
        [2, 3],
        np.zeros([2, 6], dtype=np.float32),
    ),
    (
        "Negative values tile",
        np.array([-1.0, -2.0], dtype=np.float32),
        [2],
        np.array([-1.0, -2.0, -1.0, -2.0], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_tile_precision():
    """Precision tests with known expected outputs"""

    msgw = get_max_test_msg_len(precision_test_cases)

    for tno, (tmsg, data, repeats, expected) in enumerate(precision_test_cases):
        op_name = f"test_tile_prec_{tno}"

        computed, o_tensors = _run_tile(data, repeats, op_name)

        # Shape check
        assert o_tensors[0].shape == list(
            expected.shape
        ), f"[{tmsg}] shape mismatch: {o_tensors[0].shape} vs {list(expected.shape)}"

        # Value check
        np.testing.assert_allclose(
            computed,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"[{tmsg}] precision mismatch",
        )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Mathematical property tests
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
def test_tile_output_shape():
    """Output shape = input_shape[i] * repeats[i] for each axis"""

    configs = [
        ([4], [3]),
        ([2, 3], [4, 2]),
        ([2, 3, 4], [1, 2, 3]),
        ([1, 2, 3, 4], [2, 3, 1, 2]),
    ]

    for shape, repeats in configs:
        data = np.array(np.random.randn(*shape), dtype=np.float32)
        _, o_tensors = _run_tile(data, repeats, "test_shape")

        expected_shape = [s * r for s, r in zip(shape, repeats)]
        assert (
            o_tensors[0].shape == expected_shape
        ), f"Shape {o_tensors[0].shape} != expected {expected_shape}"

    logger.debug("  Output shape = input * repeats -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_tile_identity_repeat():
    """Tiling with all 1s returns the original data"""

    for shape in [[4], [3, 4], [2, 3, 4], [1, 2, 3, 4]]:
        data = np.array(np.random.randn(*shape), dtype=np.float32)
        repeats = [1] * len(shape)

        computed, _ = _run_tile(data, repeats, "test_identity")

        np.testing.assert_array_equal(
            computed, data, err_msg=f"Identity tile failed for shape {shape}"
        )

    logger.debug("  Tile with all 1s = identity -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_tile_preserves_total_elements():
    """Total elements = input_elements * product(repeats)"""

    configs = [
        ([4], [3]),
        ([2, 3], [2, 2]),
        ([2, 3, 4], [1, 2, 3]),
    ]

    for shape, repeats in configs:
        data = np.array(np.random.randn(*shape), dtype=np.float32)
        computed, _ = _run_tile(data, repeats, "test_elems")

        expected_elems = data.size * int(np.prod(repeats))
        assert (
            computed.size == expected_elems
        ), f"Element count {computed.size} != {expected_elems}"

    logger.debug("  Total elements = input * prod(repeats) -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_tile_contains_original_data():
    """The tiled output contains the original data as a contiguous sub-block"""

    for shape in [[3], [2, 3], [2, 3, 4]]:
        data = np.array(np.random.randn(*shape), dtype=np.float32)
        repeats = [2] * len(shape)

        computed, _ = _run_tile(data, repeats, "test_contains")

        # Extract the first block (same shape as original)
        slices = tuple(slice(0, s) for s in shape)
        first_block = computed[slices]

        np.testing.assert_array_equal(
            first_block, data, err_msg=f"First tile block != original for shape {shape}"
        )

    logger.debug("  Output contains original as first block -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_tile_periodic_structure():
    """Tiling creates a periodic repetition -- data repeats along each axis"""

    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    repeats = [2, 3]

    computed, _ = _run_tile(data, repeats, "test_periodic")

    # Check that each tile block matches the original
    nr, nc = data.shape
    for ri in range(repeats[0]):
        for ci in range(repeats[1]):
            block = computed[ri * nr : (ri + 1) * nr, ci * nc : (ci + 1) * nc]
            np.testing.assert_array_equal(
                block, data, err_msg=f"Block ({ri},{ci}) doesn't match original"
            )

    logger.debug("  Periodic repetition verified -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_tile_constant_input():
    """Tiling a constant-valued tensor produces a constant-valued output"""

    val = 3.14
    data = np.full([2, 3], val, dtype=np.float32)
    repeats = [4, 5]

    computed, _ = _run_tile(data, repeats, "test_const")

    assert np.all(computed == val), "Constant input should produce constant output"

    logger.debug("  Constant input -> constant output -- OK")
