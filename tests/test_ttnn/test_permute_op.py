#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor
from ttsim.front.ttnn.ttnn_shim import permute_op


@pytest.mark.unit
def test_permute_op_creates_permute_simop_and_shape():
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0
    inp = Tensor(
        name="perm_in",
        shape=[2, 3, 224, 224],
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        device=device,
    )
    out = permute_op(inp, [0, 2, 3, 1])
    assert list(out.logical_shape()._shape) == [2, 224, 224, 3]
    assert out.layout == Layout.ROW_MAJOR_LAYOUT
    assert out.device == device

    permute_ops = [op for op in device.ops.values() if op.optype == "Permute"]
    assert len(permute_ops) == 1
    op = permute_ops[0]
    assert op.attrs["perm"] == [0, 2, 3, 1]
    assert op.perf_stats is not None
    assert "inElems" in op.perf_stats
    assert "outElems" in op.perf_stats
    assert inp.name in op.inList
    assert out.name in op.outList


@pytest.mark.unit
def test_permute_op_requires_device():
    class _NoDev:
        device = None

    with pytest.raises(AssertionError, match="permute_op requires input_tensor on device"):
        permute_op(_NoDev(), [0])  # type: ignore[arg-type]


@pytest.mark.unit
def test_permute_op_rejects_wrong_perm_length():
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0
    inp = Tensor(
        name="perm_in2",
        shape=[2, 3, 4],
        dtype=DataType.FLOAT32,
        device=device,
    )
    with pytest.raises(ValueError, match="must match input rank"):
        permute_op(inp, [0, 1])
