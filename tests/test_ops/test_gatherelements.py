#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F


def _tensor_with_data(name, data):
    return SimTensor({"name": name, "shape": list(data.shape), "dtype": data.dtype, "data": data})


def _run_gatherelements(data_arr, indices_arr, axis):
    data_t = _tensor_with_data("data", data_arr)
    indices_t = _tensor_with_data("indices", indices_arr)
    o_tensors = [make_tensor("Y")]
    op_info = {
        "name": "test_gatherelements_op",
        "optype": "GatherElements",
        "inList": [data_t.name, indices_t.name],
        "outList": [x.name for x in o_tensors],
        "attrs": {"axis": axis},
    }
    op_obj = SimOp(op_info)
    for x in (data_t, indices_t):
        x.op_in = [op_info["name"]]
    for x in o_tensors:
        x.op_out = [op_info["name"]]
    op_obj.get_perf_counts([data_t, indices_t], o_tensors)
    return o_tensors[0]


@pytest.mark.unit
@pytest.mark.opunit
def test_gatherelements_onnx_spec_example():
    """The worked example from the ONNX GatherElements spec itself:
    data = [[1,2],[3,4]], indices = [[0,0],[1,0]], axis=1
    -> output = [[1,1],[4,3]]
    """
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    out = _run_gatherelements(data, indices, axis=1)

    expected = np.array([[1, 1], [4, 3]], dtype=np.float32)
    assert out.shape == [2, 2], f"expected [2,2], got {out.shape}"
    np.testing.assert_allclose(out.data, expected)
    logger.debug("GatherElements ONNX spec example PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_gatherelements_output_shape_equals_indices_shape():
    """Unlike Gather, GatherElements' output shape must exactly match the
    indices tensor's shape (same rank as data, but can differ in size along
    `axis`) -- this is the core distinction from Gather that caused the
    original SAM bug when compute_gather used take_along_axis by mistake."""
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = np.zeros((3, 2), dtype=np.int64)  # narrower than data along axis=1
    out = _run_gatherelements(data, indices, axis=1)
    assert out.shape == [3, 2], f"expected indices' shape [3,2], got {out.shape}"


@pytest.mark.unit
@pytest.mark.opunit
def test_gatherelements_negative_indices():
    """Negative indices count from the end of the axis, per ONNX spec."""
    data = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
    indices = np.array([[-1, 0], [-2, -1]], dtype=np.int64)
    out = _run_gatherelements(data, indices, axis=1)

    # row0: data[0][-1]=30, data[0][0]=10 -> [30, 10]
    # row1: data[1][-2]=50, data[1][-1]=60 -> [50, 60]
    expected = np.array([[30, 10], [50, 60]], dtype=np.float32)
    np.testing.assert_allclose(out.data, expected)


@pytest.mark.unit
@pytest.mark.opunit
def test_gatherelements_negative_axis():
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    out = _run_gatherelements(data, indices, axis=-1)  # same as axis=1 for rank-2
    expected = np.array([[1, 1], [4, 3]], dtype=np.float32)
    np.testing.assert_allclose(out.data, expected)


@pytest.mark.unit
@pytest.mark.opunit
def test_gatherelements_rank_mismatch_raises():
    """indices must have the same rank as data -- a mismatch should raise,
    not silently produce a wrong-shaped result."""
    data_t = F._from_shape("data", [3, 4], np_dtype=np.float32)
    indices_t = F._from_shape("indices", [3, 4, 1], np_dtype=np.int64)
    o_tensors = [make_tensor("Y")]
    op_info = {
        "name": "test_gatherelements_rank_mismatch",
        "optype": "GatherElements",
        "inList": [data_t.name, indices_t.name],
        "outList": [x.name for x in o_tensors],
        "attrs": {"axis": 1},
    }
    op_obj = SimOp(op_info)
    for x in (data_t, indices_t):
        x.op_in = [op_info["name"]]
    for x in o_tensors:
        x.op_out = [op_info["name"]]
    with pytest.raises((ValueError, AssertionError)):
        op_obj.get_perf_counts([data_t, indices_t], o_tensors)
    logger.debug("GatherElements rank-mismatch correctly raises PASS")
