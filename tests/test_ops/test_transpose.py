#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_transpose


def ref_impl(shape0, perms0):
    _X0 = np.random.randn(*shape0)
    _Y = np.transpose(_X0, perms0)
    return list(_Y.shape)


# Test cases
test_name = "test_transpose"
test_cases = [
    ("2D Matrix Transpose", [3, 4], [1, 0]),
    ("1D Vector", [5], [0]),
    ("3D Tensor Transpose", [2, 3, 4], [1, 0, 2]),
    ("4D Tensor Transpose", [2, 3, 4, 5], [3, 2, 1, 0]),
    ("Empty Dimension Transpose", [3, 0, 2], [2, 1, 0]),
    ("Scalar Transpose", [], []),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_transpose():
    msgw = max([len(x[0]) for x in test_cases])
    for tno, (tmsg, shape, perms) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"
        i_tensors = [F._from_shape("X0", shape, np_dtype=np.float32)]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": op_name,
            "optype": "Transpose",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"perm": perms},
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(shape, perms)

        if inf_shape == ref_shape:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            print("INPUTS:")
            for x in i_tensors:
                print("\t", x)
            print("OUTPUTS:")
            for x in o_tensors:
                print("\t", x)
            assert (
                False
            ), f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"


# --------------------------------------------------------------------------
# Helpers for numerical tests
# --------------------------------------------------------------------------


def _get_max_msg_len(TL):
    return max([len(x[0]) for x in TL])


def _run_transpose(data, perm, op_name):
    """Build tensors + op, run shape inference + compute_transpose."""
    i_tensors = [F._from_data("X", data)]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": op_name,
        "optype": "Transpose",
        "inList": ["X"],
        "outList": ["Y"],
        "attrs": {"perm": list(perm)},
    }
    op_obj = SimOp(op_info)
    i_tensors[0].op_in = [op_name]
    o_tensors[0].op_out = [op_name]

    op_obj.get_perf_counts(i_tensors, o_tensors)
    computed = compute_transpose(i_tensors, op_obj)
    return computed, o_tensors


# --------------------------------------------------------------------------
# Numerical validation
# --------------------------------------------------------------------------

transpose_numerical_cases = [
    # 2D
    ("2D standard transpose", [3, 4], [1, 0]),
    ("2D square transpose", [4, 4], [1, 0]),
    ("2D identity perm", [3, 4], [0, 1]),
    # 3D
    ("3D swap last two", [2, 3, 4], [0, 2, 1]),
    ("3D swap first two", [2, 3, 4], [1, 0, 2]),
    ("3D full reverse", [2, 3, 4], [2, 1, 0]),
    ("3D identity perm", [2, 3, 4], [0, 1, 2]),
    ("3D cyclic left", [2, 3, 4], [1, 2, 0]),
    ("3D cyclic right", [2, 3, 4], [2, 0, 1]),
    # 4D (NCHW common permutations)
    ("4D NCHW->NHWC", [2, 3, 4, 5], [0, 2, 3, 1]),
    ("4D NHWC->NCHW", [2, 4, 5, 3], [0, 3, 1, 2]),
    ("4D identity perm", [2, 3, 4, 5], [0, 1, 2, 3]),
    ("4D full reverse", [2, 3, 4, 5], [3, 2, 1, 0]),
    ("4D swap spatial", [2, 3, 4, 5], [0, 1, 3, 2]),
    # 5D
    ("5D swap last two", [1, 2, 3, 4, 5], [0, 1, 2, 4, 3]),
    ("5D full reverse", [1, 2, 3, 4, 5], [4, 3, 2, 1, 0]),
    # 1D / single element
    ("1D trivial", [5], [0]),
    ("Single element", [1], [0]),
    # Large
    ("Large 2D", [64, 32], [1, 0]),
    ("Large 4D NCHW->NHWC", [2, 16, 8, 8], [0, 2, 3, 1]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_transpose_numerical():
    """Numerical validation of compute_transpose across shapes and perms"""

    msgw = _get_max_msg_len(transpose_numerical_cases)

    for tno, (tmsg, shape, perm) in enumerate(transpose_numerical_cases):
        op_name = f"test_transpose_num_{tno}"

        data = np.array(np.random.randn(*shape), dtype=np.float32)
        expected = np.transpose(data, perm)

        computed, o_tensors = _run_transpose(data, perm, op_name)

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

        print(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

transpose_edge_cases = [
    ("All zeros", [3, 4], [1, 0], "zeros"),
    ("All ones", [2, 3, 4], [2, 0, 1], "ones"),
    ("Negative values", [3, 4], [1, 0], "negative"),
    ("Ones in shape", [1, 1, 1, 1], [3, 2, 1, 0], "mixed"),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_transpose_edge_cases():
    """Edge cases for Transpose"""

    msgw = _get_max_msg_len(transpose_edge_cases)

    for tno, (tmsg, shape, perm, data_gen) in enumerate(transpose_edge_cases):
        op_name = f"test_transpose_edge_{tno}"

        if data_gen == "zeros":
            data = np.zeros(shape, dtype=np.float32)
        elif data_gen == "ones":
            data = np.ones(shape, dtype=np.float32)
        elif data_gen == "negative":
            data = np.array(-np.random.rand(*shape) - 1, dtype=np.float32)
        else:
            data = np.array(np.random.randn(*shape), dtype=np.float32)

        expected = np.transpose(data, perm)
        computed, _ = _run_transpose(data, perm, op_name)

        np.testing.assert_allclose(
            computed, expected, rtol=1e-5, atol=1e-7, err_msg=f"[{tmsg}] mismatch"
        )

        print(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Precision tests with known outputs
# --------------------------------------------------------------------------

transpose_precision_cases = [
    (
        "2D matrix transpose",
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        [1, 0],
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32),
    ),
    (
        "2D identity perm",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        [0, 1],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    ),
    (
        "3D perm [0,2,1]",
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        [0, 2, 1],
        np.transpose(np.arange(24, dtype=np.float32).reshape(2, 3, 4), [0, 2, 1]),
    ),
    (
        "3D perm [2,1,0]",
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        [2, 1, 0],
        np.transpose(np.arange(24, dtype=np.float32).reshape(2, 3, 4), [2, 1, 0]),
    ),
    (
        "1D trivial",
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
        [0],
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
    ),
    (
        "Square matrix transpose",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        [1, 0],
        np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
    ),
    (
        "4D NCHW->NHWC sequential",
        np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4),
        [0, 2, 3, 1],
        np.transpose(np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4), [0, 2, 3, 1]),
    ),
    (
        "Single element",
        np.array([42.0], dtype=np.float32),
        [0],
        np.array([42.0], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_transpose_precision():
    """Precision tests with known expected outputs"""

    msgw = _get_max_msg_len(transpose_precision_cases)

    for tno, (tmsg, data, perm, expected) in enumerate(transpose_precision_cases):
        op_name = f"test_transpose_prec_{tno}"

        computed, o_tensors = _run_transpose(data, perm, op_name)

        assert o_tensors[0].shape == list(
            expected.shape
        ), f"[{tmsg}] shape mismatch: {o_tensors[0].shape} vs {list(expected.shape)}"

        np.testing.assert_allclose(
            computed,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"[{tmsg}] precision mismatch",
        )

        print(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Mathematical property tests
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
def test_transpose_output_shape_rule():
    """output_shape[i] == input_shape[perm[i]]"""

    configs = [
        ([3, 4], [1, 0]),
        ([2, 3, 4], [2, 0, 1]),
        ([2, 3, 4, 5], [0, 2, 3, 1]),
    ]

    for shape, perm in configs:
        data = np.array(np.random.randn(*shape), dtype=np.float32)
        _, o_tensors = _run_transpose(data, perm, "test_shape_rule")

        expected_shape = [shape[p] for p in perm]
        assert (
            o_tensors[0].shape == expected_shape
        ), f"Shape {o_tensors[0].shape} != expected {expected_shape}"

    print("  output_shape[i] == input_shape[perm[i]] -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_transpose_identity_perm():
    """Identity permutation returns the original data"""

    for shape in [[4], [3, 4], [2, 3, 4], [1, 2, 3, 4]]:
        data = np.array(np.random.randn(*shape), dtype=np.float32)
        perm = list(range(len(shape)))
        computed, _ = _run_transpose(data, perm, "test_id")

        np.testing.assert_array_equal(
            computed, data, err_msg=f"Identity perm failed for shape {shape}"
        )

    print("  Identity permutation = no-op -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_transpose_inverse_is_identity():
    """Transpose followed by its inverse recovers original data"""

    configs = [
        ([3, 4], [1, 0]),
        ([2, 3, 4], [2, 0, 1]),
        ([2, 3, 4, 5], [0, 2, 3, 1]),
    ]

    for shape, perm in configs:
        data = np.array(np.random.randn(*shape), dtype=np.float32)

        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i

        first, _ = _run_transpose(data, perm, "test_inv_fwd")
        second, _ = _run_transpose(first, inv_perm, "test_inv_back")

        np.testing.assert_array_equal(
            second, data, err_msg=f"Transpose + inverse failed for perm {perm}"
        )

    print("  Transpose + inverse = identity -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_transpose_preserves_total_elements():
    """Element count is unchanged"""

    for shape, perm in [([3, 4], [1, 0]), ([2, 3, 4], [2, 0, 1])]:
        data = np.array(np.random.randn(*shape), dtype=np.float32)
        computed, _ = _run_transpose(data, perm, "test_elems")

        assert (
            computed.size == data.size
        ), f"Element count changed: {computed.size} vs {data.size}"

    print("  Total elements preserved -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_transpose_2d_is_matrix_T():
    """For 2D, perm=[1,0] is equivalent to .T"""

    for shape in [[3, 4], [5, 2], [1, 8]]:
        data = np.array(np.random.randn(*shape), dtype=np.float32)
        computed, _ = _run_transpose(data, [1, 0], "test_T")

        np.testing.assert_array_equal(
            computed, data.T, err_msg=f"2D transpose != .T for shape {shape}"
        )

    print("  2D perm=[1,0] == .T -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_transpose_preserves_values():
    """All values from input appear in output (just rearranged)"""

    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    computed, _ = _run_transpose(data, [2, 0, 1], "test_vals")

    assert sorted(data.flatten().tolist()) == sorted(
        computed.flatten().tolist()
    ), "Transpose should preserve all values"

    print("  All values preserved -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_transpose_constant_input():
    """Transposing a constant tensor gives a constant tensor with permuted shape"""

    val = 7.0
    data = np.full([2, 3, 4], val, dtype=np.float32)
    computed, o_tensors = _run_transpose(data, [2, 0, 1], "test_const")

    assert np.all(computed == val), "Constant input should stay constant"
    assert o_tensors[0].shape == [
        4,
        2,
        3,
    ], f"Shape should be [4,2,3], got {o_tensors[0].shape}"

    print("  Constant input -> constant output -- OK")
