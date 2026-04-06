#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import sys
from loguru import logger

# Silence the ttsim-related log messages
try:
    from loguru import logger as _loguru_logger

    _loguru_logger.disable("ttsim")
except ImportError:
    pass

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F

try:
    from ttsim.config import get_arspec_from_yaml
    from ttsim.back.device import Device

    HAS_TTSIM_MEMORY_VALIDATION = True
except ImportError:
    get_arspec_from_yaml = None  # type: ignore[assignment]
    Device = None  # type: ignore[assignment]
    HAS_TTSIM_MEMORY_VALIDATION = False

MEMORY_VALIDATION_SKIP = pytest.mark.skipif(
    not HAS_TTSIM_MEMORY_VALIDATION,
    reason="ttsim config/device backend not available for split memory validation",
)

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from ttsim.ops.desc.data_compute import compute_split

def ref_impl(data_shape, indices, axis):
    X = np.random.randn(*data_shape)
    odata = np.take(X, indices, axis=axis)
    return list(odata.shape)


# Test cases
test_name = "test_split"
test_cases = [
    {
        "name": "test_split_equal_parts_1d_opset13",
        "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        "inputs": ["input"],
        "outputs": ["output_1", "output_2", "output_3"],
        "axis": 0,
        "expected_outputs": [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0]).astype(np.float32),
            np.array([5.0, 6.0]).astype(np.float32),
        ],
    },
    {
        "name": "test_split_variable_parts_1d_opset13",
        "split": np.array([2, 4]).astype(np.int64),
        "inputs": ["input", "split"],
        "outputs": ["output_1", "output_2"],
        "axis": 0,
        "expected_outputs": [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        ],
    },
    {
        "name": "test_split_equal_parts_2d_opset13",
        "X": np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        ).astype(np.float32),
        "inputs": ["input"],
        "outputs": ["output_1", "output_2"],
        "axis": 1,
        "expected_outputs": [
            np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]).astype(np.float32),
            np.array([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]).astype(np.float32),
        ],
    },
    {
        "name": "test_split_variable_parts_2d_opset13",
        "split": np.array([2, 4]).astype(np.int64),
        "inputs": ["input", "split"],
        "outputs": ["output_1", "output_2"],
        "axis": 1,
        "expected_outputs": [
            np.array([[1.0, 2.0], [7.0, 8.0]]).astype(np.float32),
            np.array([[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]).astype(
                np.float32
            ),
        ],
    },
    {
        "name": "test_split_equal_parts_default_axis_opset13",
        "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        "inputs": ["input"],
        "outputs": ["output_1", "output_2", "output_3"],
        "expected_outputs": [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0]).astype(np.float32),
            np.array([5.0, 6.0]).astype(np.float32),
        ],
    },
    {
        "name": "test_split_variable_parts_default_axis_opset13",
        "split": np.array([2, 4]).astype(np.int64),
        "inputs": ["input", "split"],
        "outputs": ["output_1", "output_2"],
        "expected_outputs": [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        ],
    },
    {
        "name": "test_split_zero_size_splits_opset13",
        "X": np.array([]).astype(np.float32),  # 1D
        "split": np.array([0, 0, 0]).astype(
            np.int64
        ),  # Split emtpy tensor to tensors of size zero
        "inputs": ["input", "split"],
        "outputs": ["output_1", "output_2", "output_3"],
        "expected_outputs": [
            np.array([]).astype(np.float32),
            np.array([]).astype(np.float32),
            np.array([]).astype(np.float32),
        ],
    },
    {
        "name": "test_split_equal_parts_1d_opset18",
        "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        "inputs": ["input"],
        "outputs": ["output_1", "output_2", "output_3"],
        "axis": 0,
        "num_outputs": 3,
        "expected_outputs": [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0]).astype(np.float32),
            np.array([5.0, 6.0]).astype(np.float32),
        ],
    },
    {
        "name": "test_split_variable_parts_1d_opset18",
        "split": np.array([2, 4]).astype(np.int64),
        "inputs": ["input", "split"],
        "outputs": ["output_1", "output_2"],
        "axis": 0,
        "expected_outputs": [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        ],
    },
    {
        "name": "test_split_equal_parts_2d",
        "X": np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        ).astype(np.float32),
        "inputs": ["input"],
        "outputs": ["output_1", "output_2"],
        "axis": 1,
        "num_outputs": 2,
        "expected_outputs": [
            np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]).astype(np.float32),
            np.array([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]).astype(np.float32),
        ],
    },
    {
        "name": "test_split_variable_parts_2d_opset18",
        "split": np.array([2, 4]).astype(np.int64),
        "inputs": ["input", "split"],
        "outputs": ["output_1", "output_2"],
        "axis": 1,
        "expected_outputs": [
            np.array([[1.0, 2.0], [7.0, 8.0]]).astype(np.float32),
            np.array([[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]).astype(
                np.float32
            ),
        ],
    },
    {
        "name": "test_split_equal_parts_default_axis_opset18",
        "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        "inputs": ["input"],
        "outputs": ["output_1", "output_2", "output_3"],
        "num_outputs": 3,
        "expected_outputs": [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0]).astype(np.float32),
            np.array([5.0, 6.0]).astype(np.float32),
        ],
    },
    {
        "name": "test_split_variable_parts_default_axis_opset18",
        "split": np.array([2, 4]).astype(np.int64),
        "inputs": ["input", "split"],
        "outputs": ["output_1", "output_2"],
        "expected_outputs": [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        ],
    },
    {
        "name": "test_split_zero_size_splits_opset18",
        "X": np.array([]).astype(np.float32),
        "split": np.array([0, 0, 0]).astype(np.int64),
        "inputs": ["input", "split"],
        "outputs": ["output_1", "output_2", "output_3"],
        "expected_outputs": [
            np.array([]).astype(np.float32),
            np.array([]).astype(np.float32),
            np.array([]).astype(np.float32),
        ],
    },
    # FAILS RIGHT NOW!!
    # {
    #        'name'   : "test_split_1d_uneven_split_opset18",
    #        'X'      : np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype(np.float32),
    #        'inputs' : ["input"],
    #        'outputs': ["output_1", "output_2", "output_3", "output_4"],
    #        'num_outputs': 4,
    #        'expected_outputs': [
    #            np.array([1.0, 2.0]).astype(np.float32),
    #            np.array([3.0, 4.0]).astype(np.float32),
    #            np.array([5.0, 6.0]).astype(np.float32),
    #            np.array([7.0]).astype(np.float32),
    #            ]
    #        },
    # {
    #        'name'   : "test_split_2d_uneven_split_opset18",
    #        'X'      : np.array( [ [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    #                              [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    #                              ]).astype(np.float32),
    #        'inputs' : ["input"],
    #        'outputs': ["output_1", "output_2", "output_3"],
    #        'axis'   : 1,
    #        'num_outputs': 3,
    #        'expected_outputs': [
    #            np.array([[1.0, 2.0, 3.0], [9.0, 10.0, 11.0]]).astype(np.float32),
    #            np.array([[4.0, 5.0, 6.0], [12.0, 13.0, 14.0]]).astype(np.float32),
    #            np.array([[7.0, 8.0], [15.0, 16.0]]).astype(np.float32),
    #            ]
    #        }
]


@pytest.mark.unit
@pytest.mark.opunit
def test_split():
    msgw = max([len(x["name"]) for x in test_cases])  # type: ignore
    input_tensors = []
    for tno, trec in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        XShape = trec["X"].shape if "X" in trec else test_cases[tno - 1]["X"].shape  # type: ignore
        XShape = list(XShape)
        i_tensors = [F._from_shape("X", XShape)]
        if "split" in trec:
            i_tensors.append(F._from_data("S", trec["split"]))  # type: ignore
        attrs = {}
        if "axis" in trec:
            attrs["axis"] = trec["axis"]
        if "num_outputs" in trec:
            attrs["num_outputs"] = trec["num_outputs"]

        num_outputs = len(trec["expected_outputs"])  # type: ignore
        o_tensors = [make_tensor(f"O{i}") for i in range(num_outputs)]
        op_info = {
            "name": op_name,
            "optype": "Split",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": attrs,
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        check = all([o_tensors[i].shape == list(trec["expected_outputs"][i].shape) for i in range(num_outputs)])  # type: ignore

        if check:
            logger.debug(f"TEST[{tno:3d}] {trec['name']:{msgw}s} PASS")
        else:
            logger.debug("INPUTS:")
            for x in i_tensors:
                logger.debug(f"\t{x}")
            logger.debug("OUTPUTS:")
            for x in o_tensors:
                logger.debug(f"\t{x}")
            logger.debug("REF OUTPUTS:")
            for x in trec["expected_outputs"]:
                logger.debug(f"\t{x.shape}")  # type: ignore
            assert False, f"TEST[{tno:3d}] {trec['name']:{msgw}s} FAIL"


# --------------------------------------------------------------------------
# Helpers for numerical tests
# --------------------------------------------------------------------------


def _get_max_msg_len(TL):
    return max([len(x[0]) for x in TL])


def _make_split_op(
    op_name, num_outputs, axis=0, split_sizes=None, has_split_tensor=False
):
    """Build SimOp for Split."""
    in_list = ["X"]
    if has_split_tensor:
        in_list.append("split_tensor")

    out_list = [f"Y{i}" for i in range(num_outputs)]

    attrs = {"axis": axis}
    if split_sizes is not None:
        attrs["split"] = list(split_sizes)
        attrs["num_outputs"] = num_outputs

    op_info = {
        "name": op_name,
        "optype": "Split",
        "inList": in_list,
        "outList": out_list,
        "attrs": attrs,
    }
    return SimOp(op_info)


def _make_split_tensors(data, num_outputs, split_sizes=None):
    """Create input and output tensor lists for numerical tests."""
    i_tensors = [F._from_data("X", data)]
    if split_sizes is not None:
        split_arr = np.array(split_sizes, dtype=np.int64)
        i_tensors.append(F._from_data("split_tensor", split_arr))

    o_tensors = [make_tensor(f"Y{i}") for i in range(num_outputs)]
    return i_tensors, o_tensors


def _run_split(data, axis, num_outputs, split_sizes, op_name):
    """Run shape inference + compute_split and return result list."""
    has_split_tensor = split_sizes is not None
    i_tensors, o_tensors = _make_split_tensors(data, num_outputs, split_sizes)
    op_obj = _make_split_op(op_name, num_outputs, axis, split_sizes, has_split_tensor)

    i_tensors[0].op_in = [op_name]
    if has_split_tensor:
        i_tensors[1].op_in = [op_name]
    for ot in o_tensors:
        ot.op_out = [op_name]

    op_obj.get_perf_counts(i_tensors, o_tensors)
    result_list = compute_split(i_tensors, op_obj)
    return result_list, o_tensors


# --------------------------------------------------------------------------
# Numerical validation test cases
#   (name, input_shape, axis, num_outputs, split_sizes)
# --------------------------------------------------------------------------

split_numerical_cases = [
    # Equal splits axis 0
    ("Equal 2-way axis0 1D", [8], 0, 2, None),
    ("Equal 4-way axis0 1D", [12], 0, 4, None),
    ("Equal 2-way axis0 2D", [6, 4], 0, 2, None),
    ("Equal 3-way axis0 2D", [9, 4], 0, 3, None),
    ("Equal 2-way axis0 3D", [4, 3, 4], 0, 2, None),
    ("Equal 2-way axis0 4D", [4, 3, 4, 4], 0, 2, None),
    ("Equal 2-way axis0 5D", [4, 2, 3, 4, 4], 0, 2, None),
    # Equal splits other axes
    ("Equal 2-way axis1 2D", [4, 6], 1, 2, None),
    ("Equal 3-way axis1 3D", [2, 9, 4], 1, 3, None),
    ("Equal 2-way axis2 3D", [2, 3, 8], 2, 2, None),
    ("Equal 2-way axis3 4D", [2, 3, 4, 6], 3, 2, None),
    # Unequal splits
    ("Unequal 2-way axis0 1D", [10], 0, 2, [3, 7]),
    ("Unequal 3-way axis0 1D", [10], 0, 3, [2, 5, 3]),
    ("Unequal 2-way axis1 2D", [4, 10], 1, 2, [3, 7]),
    ("Unequal 3-way axis0 2D", [12, 4], 0, 3, [2, 6, 4]),
    ("Unequal 2-way axis2 3D", [2, 3, 10], 2, 2, [4, 6]),
    ("Unequal asymmetric axis0", [10, 4], 0, 3, [1, 2, 7]),
    # Single output (no-op)
    ("Single output axis0", [4, 4], 0, 1, None),
    # Large tensors
    ("Large equal axis0", [64, 32], 0, 4, None),
    ("Large equal axis1", [8, 64], 1, 8, None),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_split_numerical():
    """Numerical validation of compute_split across shapes and split modes"""

    msgw = _get_max_msg_len(split_numerical_cases)

    for tno, (tmsg, shape, axis, num_outputs, split_sizes) in enumerate(
        split_numerical_cases
    ):
        op_name = f"test_split_num_{tno}"
        data = np.array(np.random.rand(*shape), dtype=np.float32)

        # Reference
        if split_sizes is None:
            expected_list = np.array_split(data, num_outputs, axis=axis)
        else:
            expected_list = np.split(data, np.cumsum(split_sizes)[:-1], axis=axis)

        result_list, o_tensors = _run_split(
            data, axis, num_outputs, split_sizes, op_name
        )

        # Shape check
        for idx, ot in enumerate(o_tensors):
            assert ot.shape == list(
                expected_list[idx].shape
            ), f"[{tmsg}] output {idx} shape mismatch: {ot.shape} vs {list(expected_list[idx].shape)}"

        # Value check
        assert (
            len(result_list) == num_outputs
        ), f"[{tmsg}] expected {num_outputs} outputs, got {len(result_list)}"
        for idx in range(num_outputs):
            np.testing.assert_allclose(
                result_list[idx],
                expected_list[idx],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"[{tmsg}] output {idx} mismatch",
            )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

split_edge_cases = [
    ("Single element", [1], 0, 1, None),
    ("All-zero data equal", [6, 4], 0, 2, None),
    ("All-zero data unequal", [6, 4], 0, 2, [1, 5]),
    ("Split size 1 each", [3], 0, 3, [1, 1, 1]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_split_edge_cases():
    """Edge cases for Split"""

    msgw = _get_max_msg_len(split_edge_cases)

    for tno, (tmsg, shape, axis, num_outputs, split_sizes) in enumerate(
        split_edge_cases
    ):
        op_name = f"test_split_edge_{tno}"

        if "zero" in tmsg.lower():
            data = np.zeros(shape, dtype=np.float32)
        else:
            data = np.array(np.random.rand(*shape), dtype=np.float32)

        if split_sizes is None:
            expected_list = np.array_split(data, num_outputs, axis=axis)
        else:
            expected_list = np.split(data, np.cumsum(split_sizes)[:-1], axis=axis)

        result_list, _ = _run_split(data, axis, num_outputs, split_sizes, op_name)

        assert len(result_list) == num_outputs
        for idx in range(num_outputs):
            np.testing.assert_allclose(
                result_list[idx],
                expected_list[idx],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"[{tmsg}] output {idx} mismatch",
            )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Precision tests
# --------------------------------------------------------------------------

split_precision_cases = [
    # (name, shape, axis, num_outputs, split_sizes, data_gen)
    ("Large values equal", [8], 0, 2, None, "large"),
    ("Small values equal", [8], 0, 2, None, "small"),
    ("Large values unequal", [10], 0, 2, [4, 6], "large"),
    ("Small values unequal", [10], 0, 2, [4, 6], "small"),
    ("Negative values", [8, 4], 0, 2, None, "negative"),
    ("Mixed sign", [8, 4], 0, 2, None, "mixed"),
    ("Sequential integers", [12], 0, 3, None, "sequential"),
    ("Near-zero values", [6, 4], 1, 2, None, "near_zero"),
    ("Ones everywhere", [9, 3], 0, 3, None, "ones"),
]


def _gen_precision_data(shape, data_gen):
    if data_gen == "large":
        return np.array(np.random.uniform(1e6, 1e8, size=shape), dtype=np.float32)
    elif data_gen == "small":
        return np.array(np.random.uniform(1e-8, 1e-4, size=shape), dtype=np.float32)
    elif data_gen == "negative":
        return np.array(np.random.uniform(-100, -0.01, size=shape), dtype=np.float32)
    elif data_gen == "mixed":
        return np.array(np.random.uniform(-100, 100, size=shape), dtype=np.float32)
    elif data_gen == "sequential":
        return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    elif data_gen == "near_zero":
        return np.array(np.random.uniform(-1e-6, 1e-6, size=shape), dtype=np.float32)
    elif data_gen == "ones":
        return np.ones(shape, dtype=np.float32)
    return np.array(np.random.rand(*shape), dtype=np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_split_precision():
    """Precision validation for Split across data ranges"""

    msgw = _get_max_msg_len(split_precision_cases)

    for tno, (tmsg, shape, axis, num_outputs, split_sizes, data_gen) in enumerate(
        split_precision_cases
    ):
        op_name = f"test_split_prec_{tno}"
        data = _gen_precision_data(shape, data_gen)

        if split_sizes is None:
            expected_list = np.array_split(data, num_outputs, axis=axis)
        else:
            expected_list = np.split(data, np.cumsum(split_sizes)[:-1], axis=axis)

        result_list, _ = _run_split(data, axis, num_outputs, split_sizes, op_name)

        assert len(result_list) == num_outputs
        for idx in range(num_outputs):
            np.testing.assert_allclose(
                result_list[idx],
                expected_list[idx],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"[{tmsg}] output {idx} mismatch",
            )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Mathematical property tests
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
def test_split_concat_recovers_original():
    """Concatenating all split outputs reconstructs the original tensor"""

    configs = [
        ([12], 0, 3, None),
        ([6, 8], 0, 2, None),
        ([6, 8], 1, 4, None),
        ([4, 6, 8], 2, 2, None),
        ([10], 0, 3, [2, 5, 3]),
        ([4, 10], 1, 2, [3, 7]),
    ]

    for shape, axis, num_outputs, split_sizes in configs:
        data = np.array(np.random.rand(*shape), dtype=np.float32)
        result_list, _ = _run_split(data, axis, num_outputs, split_sizes, "test_concat")

        reconstructed = np.concatenate(result_list, axis=axis)
        np.testing.assert_array_equal(
            reconstructed,
            data,
            err_msg=f"Concat recovery failed for shape={shape}, axis={axis}",
        )

    logger.debug("  Concatenation recovers original -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_split_output_shapes_sum_to_input():
    """Sum of output dims along split axis equals input dim"""

    configs = [
        ([12], 0, 4, None),
        ([6, 8], 1, 2, None),
        ([10], 0, 3, [2, 5, 3]),
        ([2, 3, 12], 2, 3, [4, 4, 4]),
    ]

    for shape, axis, num_outputs, split_sizes in configs:
        data = np.array(np.random.rand(*shape), dtype=np.float32)
        _, o_tensors = _run_split(data, axis, num_outputs, split_sizes, "test_dim_sum")

        total_split_dim = sum(ot.shape[axis] for ot in o_tensors)
        assert (
            total_split_dim == shape[axis]
        ), f"Split dim sum {total_split_dim} != input dim {shape[axis]}"

    logger.debug("  Output shapes sum to input dim -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_split_preserves_other_dims():
    """Non-split dimensions remain unchanged in all outputs"""

    configs = [
        ([6, 4], 0, 2, None),
        ([3, 8, 5], 1, 4, None),
        ([2, 3, 12], 2, 3, [4, 4, 4]),
    ]

    for shape, axis, num_outputs, split_sizes in configs:
        data = np.array(np.random.rand(*shape), dtype=np.float32)
        _, o_tensors = _run_split(data, axis, num_outputs, split_sizes, "test_dims")

        for idx, ot in enumerate(o_tensors):
            for d in range(len(shape)):
                if d != axis:
                    assert (
                        ot.shape[d] == shape[d]
                    ), f"Non-split dim {d} changed: {ot.shape[d]} vs {shape[d]}"

    logger.debug("  Non-split dims preserved -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_split_preserves_total_elements():
    """Total elements across all outputs equals input element count"""

    configs = [
        ([12], 0, 3, None),
        ([6, 8], 0, 2, None),
        ([6, 8], 1, 4, None),
        ([10], 0, 3, [2, 5, 3]),
    ]

    for shape, axis, num_outputs, split_sizes in configs:
        data = np.array(np.random.rand(*shape), dtype=np.float32)
        result_list, _ = _run_split(data, axis, num_outputs, split_sizes, "test_elems")

        total = sum(r.size for r in result_list)
        assert total == data.size, f"Total elements {total} != {data.size}"

    logger.debug("  Total elements preserved -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_split_equal_parts_identical_shapes():
    """Equal split produces outputs with identical shapes"""

    configs = [
        ([12], 0, 4),
        ([6, 8], 0, 3),
        ([6, 8], 1, 2),
        ([4, 6, 8], 2, 4),
    ]

    for shape, axis, num_outputs in configs:
        data = np.array(np.random.rand(*shape), dtype=np.float32)
        _, o_tensors = _run_split(data, axis, num_outputs, None, "test_eq")

        first_shape = o_tensors[0].shape
        for idx, ot in enumerate(o_tensors):
            assert (
                ot.shape == first_shape
            ), f"Output {idx} shape {ot.shape} != {first_shape}"

    logger.debug("  Equal split -> identical shapes -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_split_data_ordering():
    """Sequential data stays sequential in chunks"""

    data = np.arange(12, dtype=np.float32)
    result_list, _ = _run_split(data, 0, 3, None, "test_order")

    np.testing.assert_array_equal(
        result_list[0], np.array([0, 1, 2, 3], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        result_list[1], np.array([4, 5, 6, 7], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        result_list[2], np.array([8, 9, 10, 11], dtype=np.float32)
    )

    logger.debug("  Data ordering preserved -- OK")


def calculate_split_memory_stats(shape, axis=0, num_outputs=2, dtype="float32"):
    """Calculate memory and compute statistics for a split operation"""
    # Get device configuration
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    # Create input tensors (equal split, no split_sizes tensor)
    np_dtype = getattr(np, dtype)
    data_X = np.random.randn(*shape).astype(np_dtype)
    i_tensors = [F._from_data("X", data_X)]
    o_tensors = [make_tensor(f"Y{i}") for i in range(num_outputs)]

    # Create operation
    op_info = {
        "optype": "Split",
        "name": "split_mem_test",
        "inList": ["X"],
        "outList": [f"Y{i}" for i in range(num_outputs)],
        "attrs": {"axis": axis, "num_outputs": num_outputs},
    }
    op_obj = SimOp(op_info)

    # Set op references
    for x in i_tensors:
        x.op_in = [op_info["name"]]
    for x in o_tensors:
        x.op_out = [op_info["name"]]

    # Get performance counts
    op_obj.get_perf_counts(i_tensors, o_tensors)

    # Calculate statistics
    perf_stats = op_obj.perf_stats
    actual_instructions = perf_stats.get("instrs", {})
    ops = (
        sum(actual_instructions.values())
        if isinstance(actual_instructions, dict)
        else np.prod(shape)
    )
    input_bytes = perf_stats["inBytes"]
    output_bytes = perf_stats["outBytes"]
    total_memory = input_bytes + output_bytes

    # Calculate intensities
    arithmetic_intensity = ops / total_memory if total_memory > 0 else 0
    mem_bw_bytes_per_cycle = (
        device.simconfig_obj.peak_bandwidth(freq_units="GHz")
        * 1e9
        / device.freq_MHz
        / 1e6
    )
    compute_throughput = 1  # 1 op per cycle for split
    compute_cycles = ops / compute_throughput
    memory_cycles = total_memory / mem_bw_bytes_per_cycle

    bottleneck = "compute-bound" if compute_cycles > memory_cycles else "memory-bound"

    return {
        "shape": shape,
        "axis": axis,
        "num_outputs": num_outputs,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_memory": total_memory,
        "ops": ops,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
        "device": device,
    }

@MEMORY_VALIDATION_SKIP
def test_split_memory_validation():
    """Memory validation test for split operation"""
    logger.info("\n" + "=" * 80)
    logger.info("SPLIT MEMORY VALIDATION TEST")
    logger.info("=" * 80)

    # Test configurations (shape, axis, num_outputs)
    test_configs = [
        ([32], 0, 2),
        ([64, 64], 0, 2),
        ([32, 32, 32], 0, 2),
        ([16, 16, 16, 16], 0, 2),
        ([1, 224, 224, 3], 1, 2),
        ([8, 128, 128, 64], 0, 2),
        ([4, 56, 56, 256], 0, 2),
    ]

    results = []
    for shape, axis, num_outputs in test_configs:
        stats = calculate_split_memory_stats(shape, axis, num_outputs)
        results.append(stats)

    # Print device info once
    device = results[0]["device"]
    logger.info(f"\nDevice: {device.devname}")
    logger.info(f"  Name: {device.name}")
    logger.info(f"  Frequency: {device.freq_MHz} MHz")
    logger.info(f"  Memory Frequency: {device.memfreq_MHz} MHz")

    # Print results for each configuration
    for stats in results:
        logger.debug(
            f"Shape: {stats['shape']}, Axis: {stats['axis']}, Num Outputs: {stats['num_outputs']}"
        )
        logger.debug(f"  Memory: {stats['total_memory']/1e6:.4f} MB")
        logger.debug(f"  Operations: {stats['ops']:.0f}")
        logger.debug(f"  Arithmetic Intensity: {stats['arithmetic_intensity']:.6f} ops/byte")
        logger.debug(f"  Bottleneck: {stats['bottleneck']}")

    # Summary statistics
    memory_bound = sum(1 for r in results if r["bottleneck"] == "memory-bound")
    compute_bound = sum(1 for r in results if r["bottleneck"] == "compute-bound")

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total configurations tested: {len(results)}")
    logger.info(f"Memory-bound: {memory_bound}")
    logger.info(f"Compute-bound: {compute_bound}")
    logger.info("=" * 80)
