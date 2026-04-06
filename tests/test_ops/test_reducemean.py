#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import warnings
import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_reducemean

# --------------------------------------------------------------------------
# Reference implementation
# --------------------------------------------------------------------------


def ref_impl_reducemean(X, axes=None, keepdims=1, noop=0):
    """Reference ReduceMean"""
    if axes is None:
        if noop:
            return X.copy()
        else:
            return np.mean(X, axis=None, keepdims=bool(keepdims))
    else:
        axes_tuple = tuple(int(a) for a in axes)
        return np.mean(X, axis=axes_tuple, keepdims=bool(keepdims))


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def _make_reducemean_tensors(data, axes=None):
    """Build input tensor list: [X] or [X, axes_tensor]"""
    tensors = [F._from_data("X", data)]
    if axes is not None:
        tensors.append(F._from_data("axes", np.array(axes, dtype=np.int64)))
    return tensors


# --------------------------------------------------------------------------
# Main test cases
# --------------------------------------------------------------------------

reducemean_test_name = "test_reducemean"
reducemean_test_cases = [
    # (name, data_shape, axes, keepdims, noop_with_empty_axes)
    # Reduce all axes (axes=None, noop=0)
    ("1D reduce all", [8], None, 1, 0),
    ("2D reduce all", [3, 4], None, 1, 0),
    ("3D reduce all", [2, 3, 4], None, 1, 0),
    ("4D reduce all", [2, 3, 4, 4], None, 1, 0),
    # Reduce all without keepdims
    ("2D reduce all no keepdims", [3, 4], None, 0, 0),
    ("3D reduce all no keepdims", [2, 3, 4], None, 0, 0),
    # noop_with_empty_axes=1 (no reduction when axes=None)
    ("2D noop empty axes", [3, 4], None, 1, 1),
    ("3D noop empty axes", [2, 3, 4], None, 1, 1),
    # Reduce single axis with keepdims
    ("2D axis 0 keepdims", [3, 4], [0], 1, 0),
    ("2D axis 1 keepdims", [3, 4], [1], 1, 0),
    ("3D axis 0 keepdims", [2, 3, 4], [0], 1, 0),
    ("3D axis 1 keepdims", [2, 3, 4], [1], 1, 0),
    ("3D axis 2 keepdims", [2, 3, 4], [2], 1, 0),
    ("4D axis 1 (channel)", [2, 3, 4, 4], [1], 1, 0),
    # Reduce single axis without keepdims
    ("2D axis 0 no keepdims", [3, 4], [0], 0, 0),
    ("3D axis 2 no keepdims", [2, 3, 4], [2], 0, 0),
    ("4D axis 3 no keepdims", [2, 3, 4, 4], [3], 0, 0),
    # Reduce multiple axes
    ("3D axes [0,2] keepdims", [2, 3, 4], [0, 2], 1, 0),
    ("4D axes [2,3] keepdims", [2, 3, 4, 4], [2, 3], 1, 0),
    ("4D axes [0,1] keepdims", [2, 3, 4, 4], [0, 1], 1, 0),
    ("4D axes [2,3] no keepdims", [2, 3, 4, 4], [2, 3], 0, 0),
    # Negative axes
    ("3D axis -1 keepdims", [2, 3, 4], [-1], 1, 0),
    ("4D axis -2 keepdims", [2, 3, 4, 4], [-2], 1, 0),
    ("3D axes [-1,-2] keepdims", [2, 3, 4], [-1, -2], 1, 0),
    # Special sizes
    ("Single element", [1], [0], 1, 0),
    ("Ones shape", [1, 1, 1], [0, 1, 2], 1, 0),
    ("Large 4D spatial mean", [2, 16, 8, 8], [2, 3], 1, 0),
    ("5D input", [2, 2, 3, 4, 4], [3, 4], 1, 0),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_reducemean():
    """Numerical validation of compute_reducemean across shapes/axes/keepdims"""

    msgw = get_max_test_msg_len(reducemean_test_cases)

    for tno, (tmsg, shape, axes, keepdims, noop) in enumerate(reducemean_test_cases):
        op_name = f"{reducemean_test_name}_{tno}"

        data = np.array(np.random.randn(*shape), dtype=np.float32)
        expected = ref_impl_reducemean(data, axes, keepdims, noop)

        i_tensors = _make_reducemean_tensors(data, axes)
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "ReduceMean",
            "inList": [x.name for x in i_tensors],
            "outList": ["Y"],
            "attrs": {"keepdims": keepdims, "noop_with_empty_axes": noop},
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        # Shape check
        assert o_tensors[0].shape == list(
            expected.shape
        ), f"[{tno}] {tmsg}: shape mismatch {o_tensors[0].shape} != {list(expected.shape)}"

        # Numerical check
        computed = compute_reducemean(i_tensors, op_obj)
        assert np.allclose(
            computed, expected, rtol=1e-5, atol=1e-6
        ), f"[{tno}] {tmsg}: numerical mismatch"

        logger.debug(
            f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS  out_shape={list(expected.shape)}"
        )


# --------------------------------------------------------------------------
# Error/edge-case tests
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
def test_reducemean_errors():
    """Edge cases: empty tensor, single-element reduce"""

    edge_cases = [
        ("Empty tensor axis 0", [0, 4], [0], 1, 0),
        ("Reduce to scalar", [3, 4], None, 0, 0),
        ("Single-element tensor", [1], [0], 1, 0),
    ]

    msgw = get_max_test_msg_len(edge_cases)

    for tno, (tmsg, shape, axes, keepdims, noop) in enumerate(edge_cases):
        op_name = f"test_reducemean_edge_{tno}"

        data = np.array(
            (
                np.random.randn(*shape)
                if all(s > 0 for s in shape)
                else np.array([], dtype=np.float32).reshape(shape)
            ),
            dtype=np.float32,
        )

        i_tensors = _make_reducemean_tensors(data, axes)
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": op_name,
            "optype": "ReduceMean",
            "inList": [x.name for x in i_tensors],
            "outList": ["Y"],
            "attrs": {"keepdims": keepdims, "noop_with_empty_axes": noop},
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                op_obj.get_perf_counts(i_tensors, o_tensors)
                computed = compute_reducemean(i_tensors, op_obj)
            logger.debug(
                f"EDGE[{tno:2d}] {tmsg:{msgw}s} PASS (output shape: {computed.shape})"
            )
        except (ValueError, AssertionError, IndexError, RuntimeWarning) as e:
            logger.debug(
                f"EDGE[{tno:2d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__})"
            )


# --------------------------------------------------------------------------
# Precision test cases with known outputs
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
def test_reducemean_precision():
    """Test ReduceMean with precise known input/output pairs"""

    precision_cases = [
        (
            "1D mean",
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            None,
            1,
            0,
            np.array([2.5], dtype=np.float32),
        ),
        (
            "1D mean no keepdims",
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            None,
            0,
            0,
            np.float32(2.5),
        ),
        (
            "2D mean axis 0",
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            [0],
            1,
            0,
            np.array([[2.0, 3.0]], dtype=np.float32),
        ),
        (
            "2D mean axis 1",
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            [1],
            1,
            0,
            np.array([[1.5], [3.5]], dtype=np.float32),
        ),
        (
            "2D mean all axes",
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            None,
            1,
            0,
            np.array([[2.5]], dtype=np.float32),
        ),
        (
            "2D mean axis 0 no keepdims",
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            [0],
            0,
            0,
            np.array([2.0, 3.0], dtype=np.float32),
        ),
        (
            "3D mean axis 1",
            np.array(
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32
            ),
            [1],
            1,
            0,
            np.array([[[2.0, 3.0]], [[6.0, 7.0]]], dtype=np.float32),
        ),
        (
            "noop with empty axes",
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            None,
            1,
            1,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        ),
        (
            "All same values",
            np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32),
            None,
            1,
            0,
            np.array([5.0], dtype=np.float32),
        ),
        (
            "Negative values",
            np.array([-2.0, -4.0, -6.0, -8.0], dtype=np.float32),
            None,
            1,
            0,
            np.array([-5.0], dtype=np.float32),
        ),
    ]

    msgw = 30

    for tno, (tmsg, test_data, axes, keepdims, noop, expected) in enumerate(
        precision_cases
    ):
        op_name = f"test_reducemean_prec_{tno}"

        i_tensors = _make_reducemean_tensors(test_data, axes)
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": op_name,
            "optype": "ReduceMean",
            "inList": [x.name for x in i_tensors],
            "outList": ["Y"],
            "attrs": {"keepdims": keepdims, "noop_with_empty_axes": noop},
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]
        op_obj.get_perf_counts(i_tensors, o_tensors)

        computed = compute_reducemean(i_tensors, op_obj)
        assert np.allclose(
            computed, expected, rtol=1e-5, atol=1e-6
        ), f"Precision test '{tmsg}': expected {expected}, got {computed}"
        logger.debug(f"PRECISION[{tno:2d}] {tmsg:{msgw}s} PASS")


# --------------------------------------------------------------------------
# Mathematical property tests
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
def test_reducemean_bounds():
    """Mean is always between min and max of the input along reduced axes"""
    configs = [
        ([3, 4], [0], 1),
        ([3, 4], [1], 1),
        ([2, 3, 4], [2], 1),
        ([2, 3, 4, 4], [2, 3], 1),
    ]

    for idx, (shape, axes, keepdims) in enumerate(configs):
        data = np.array(np.random.randn(*shape), dtype=np.float32)

        i_tensors = _make_reducemean_tensors(data, axes)
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": f"bounds_{idx}",
            "optype": "ReduceMean",
            "inList": [x.name for x in i_tensors],
            "outList": ["Y"],
            "attrs": {"keepdims": keepdims},
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_info["name"]]
        for x in o_tensors:
            x.op_out = [op_info["name"]]
        op_obj.get_perf_counts(i_tensors, o_tensors)
        computed = compute_reducemean(i_tensors, op_obj)

        axes_tuple = tuple(axes)
        mins = np.min(data, axis=axes_tuple, keepdims=bool(keepdims))
        maxs = np.max(data, axis=axes_tuple, keepdims=bool(keepdims))

        assert np.all(computed >= mins - 1e-6) and np.all(
            computed <= maxs + 1e-6
        ), f"Mean out of [min, max] bounds for shape {shape}, axes {axes}"
        logger.debug(f"BOUNDS TEST[{idx}] shape={shape} axes={axes} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_reducemean_constant_input():
    """Mean of a constant tensor equals that constant"""
    shapes = [[5], [3, 4], [2, 3, 4]]

    for idx, shape in enumerate(shapes):
        val = float(np.random.uniform(-10, 10))
        data = np.full(shape, val, dtype=np.float32)

        i_tensors = _make_reducemean_tensors(data, None)
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": f"const_{idx}",
            "optype": "ReduceMean",
            "inList": [x.name for x in i_tensors],
            "outList": ["Y"],
            "attrs": {"keepdims": 1, "noop_with_empty_axes": 0},
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_info["name"]]
        for x in o_tensors:
            x.op_out = [op_info["name"]]
        op_obj.get_perf_counts(i_tensors, o_tensors)
        computed = compute_reducemean(i_tensors, op_obj)

        assert np.allclose(
            computed, val, rtol=1e-5
        ), f"Mean of constant {val} tensor should be {val}, got {computed}"
        logger.debug(f"CONSTANT INPUT[{idx}] val={val:.2f} shape={shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_reducemean_keepdims_shape():
    """Verify keepdims=1 preserves rank, keepdims=0 reduces rank"""
    shape = [2, 3, 4]
    data = np.array(np.random.randn(*shape), dtype=np.float32)
    axes = [1]

    # keepdims=1
    i1 = _make_reducemean_tensors(data, axes)
    o1 = [make_tensor("Y")]
    op1_info = {
        "name": "kd1",
        "optype": "ReduceMean",
        "inList": [x.name for x in i1],
        "outList": ["Y"],
        "attrs": {"keepdims": 1},
    }
    op1 = SimOp(op1_info)
    for x in i1:
        x.op_in = [op1.name]
    for x in o1:
        x.op_out = [op1.name]
    op1.get_perf_counts(i1, o1)
    c1 = compute_reducemean(i1, op1)
    assert len(c1.shape) == 3, f"keepdims=1 should preserve rank 3, got {len(c1.shape)}"
    assert c1.shape[1] == 1, f"Reduced dim should be 1, got {c1.shape[1]}"

    # keepdims=0
    i2 = _make_reducemean_tensors(data, axes)
    o2 = [make_tensor("Y")]
    op2_info = {
        "name": "kd0",
        "optype": "ReduceMean",
        "inList": [x.name for x in i2],
        "outList": ["Y"],
        "attrs": {"keepdims": 0},
    }
    op2 = SimOp(op2_info)
    for x in i2:
        x.op_in = [op2.name]
    for x in o2:
        x.op_out = [op2.name]
    op2.get_perf_counts(i2, o2)
    c2 = compute_reducemean(i2, op2)
    assert (
        len(c2.shape) == 2
    ), f"keepdims=0 should reduce rank to 2, got {len(c2.shape)}"

    # Both should have same values
    assert np.allclose(
        c1.squeeze(axis=1), c2, rtol=1e-5
    ), "keepdims=0 and keepdims=1 values should match after squeeze"
    logger.debug("KEEPDIMS SHAPE TEST PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_reducemean_noop():
    """noop_with_empty_axes=1 and axes=None should return input unchanged"""
    shape = [3, 4, 5]
    data = np.array(np.random.randn(*shape), dtype=np.float32)

    i_tensors = _make_reducemean_tensors(data, None)
    o_tensors = [make_tensor("Y")]
    op_info = {
        "name": "noop",
        "optype": "ReduceMean",
        "inList": [x.name for x in i_tensors],
        "outList": ["Y"],
        "attrs": {"keepdims": 1, "noop_with_empty_axes": 1},
    }
    op_obj = SimOp(op_info)
    for x in i_tensors:
        x.op_in = [op_info["name"]]
    for x in o_tensors:
        x.op_out = [op_info["name"]]
    op_obj.get_perf_counts(i_tensors, o_tensors)
    computed = compute_reducemean(i_tensors, op_obj)

    assert np.array_equal(
        computed, data
    ), "noop_with_empty_axes=1 should return input unchanged"
    logger.debug("NOOP TEST PASS")
