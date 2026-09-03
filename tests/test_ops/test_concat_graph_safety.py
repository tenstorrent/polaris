#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Graph-safety contract for the rank-normalizing squeeze in concat_sinf / slice_sinf.

A SimTensor is a DAG edge that can fan out to several consumers, so a shape-inference
descriptor must not rewrite its inputs' .shape/.data, and must not invent .data that
the graph never produced. These tests pin that contract.
"""
import numpy as np
import pytest

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F


def _tensor(name, shape, data=None, dtype=np.float32):
    return SimTensor({"name": name, "shape": shape, "dtype": np.dtype(dtype), "data": data})


def _run(optype, i_tensors, attrs, op_name="gs_op"):
    o_tensors = [make_tensor("Y")]
    op_obj = SimOp({
        "name": op_name,
        "optype": optype,
        "inList": [x.name for x in i_tensors],
        "outList": [x.name for x in o_tensors],
        "attrs": attrs,
    })
    for x in i_tensors:
        x.op_in = [op_name]
    for x in o_tensors:
        x.op_out = [op_name]
    op_obj.get_perf_counts(i_tensors, o_tensors)
    return o_tensors[0]


@pytest.mark.unit
@pytest.mark.opunit
def test_concat_does_not_mutate_its_inputs():
    """concat_sinf's squeeze must run on copies, not on the shared graph tensors."""
    A = _tensor("A", [1, 64], np.arange(64, dtype=np.float32).reshape(1, 64))
    B = _tensor("B", [64], np.arange(64, 128, dtype=np.float32))

    out = _run("Concat", [A, B], {"axis": 0})
    assert list(out.shape) == [128]

    assert list(A.shape) == [1, 64], (
        f"Concat rewrote its input's shape in place: A.shape is now {list(A.shape)}, "
        "expected [1, 64]. A is a shared graph node -- other consumers now see the "
        "squeezed rank."
    )
    assert A.data.shape == (1, 64), (
        f"Concat rewrote its input's data in place: A.data.shape is now {A.data.shape}, "
        "expected (1, 64)."
    )


@pytest.mark.unit
@pytest.mark.opunit
def test_concat_propagates_hw_shape_on_channel_axis():
    """hw_shape must survive channel-concat -- it participates in LUT key construction."""
    P = F._from_shape("P", [1, 8, 4, 4], np_dtype=np.float32)
    Q = F._from_shape("Q", [1, 8, 4, 4], np_dtype=np.float32)
    P.hw_shape = [1, 1, 16, 8]
    Q.hw_shape = [1, 1, 16, 8]
    assert P.data is None and Q.data is None, "activations are expected to carry no data"

    out = _run("Concat", [P, Q], {"axis": 1}, op_name="gs_concat4d")

    assert list(out.shape) == [1, 16, 4, 4]
    assert out.hw_shape == [1, 1, 16, 16], (
        f"channel-concat dropped hw_shape (got {out.hw_shape}, expected [1, 1, 16, 16]). "
        "Downstream Halo/ITS/STI LUT keys fall back to raw NCHW and stop matching."
    )


@pytest.mark.unit
@pytest.mark.opunit
def test_concat_does_not_fabricate_data_for_dataless_inputs():
    """No input has data, so the output must have none -- not invented values."""
    R = F._from_shape("R", [4], np_dtype=np.float32)
    S = F._from_shape("S", [4], np_dtype=np.float32)
    assert R.data is None and S.data is None

    out = _run("Concat", [R, S], {"axis": 0}, op_name="gs_concat_nodata")

    assert out.data is None, (
        f"Concat fabricated output data {out.data} from inputs that carry none. "
        "Downstream ops that read .data as shape values would consume these."
    )


@pytest.mark.unit
@pytest.mark.opunit
def test_slice_does_not_mutate_its_index_inputs():
    """_squeeze_to_rank1 must not rewrite the shared starts/ends/axes tensors."""
    X = F._from_shape("X", [8, 16], np_dtype=np.float32)
    starts = _tensor("starts", [1, 1], np.array([[0]], dtype=np.int64), dtype=np.int64)
    ends = _tensor("ends", [1, 1], np.array([[4]], dtype=np.int64), dtype=np.int64)
    axes = _tensor("axes", [1, 1], np.array([[0]], dtype=np.int64), dtype=np.int64)

    out = _run("Slice", [X, starts, ends, axes], {}, op_name="gs_slice")
    assert list(out.shape) == [4, 16]

    for t, nm in ((starts, "starts"), (ends, "ends"), (axes, "axes")):
        assert list(t.shape) == [1, 1], (
            f"Slice rewrote its {nm} input's shape in place: now {list(t.shape)}, "
            "expected [1, 1]. These are shared graph nodes."
        )
        assert t.data.shape == (1, 1), (
            f"Slice rewrote its {nm} input's data in place: now {t.data.shape}, "
            "expected (1, 1)."
        )
