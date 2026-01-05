#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttsim.front.ttnn as ttnn
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
    assert out.logical_shape().as_list() == [2, 224, 224, 3]
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


@pytest.mark.unit
def test_ttnn_package_permute_records_permute_simop():
    """``import ttsim.front.ttnn as ttnn`` then ``ttnn.permute`` uses shim (Permute, not Transpose)."""
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0
    inp = Tensor(
        name="pkg_perm_in",
        shape=[2, 3, 4],
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        device=device,
    )
    out = ttnn.permute(inp, (0, 2, 1))
    assert out.logical_shape().as_list() == [2, 4, 3]
    ops = [op for op in device.ops.values() if op.optype == 'Permute']
    assert len(ops) == 1
    assert not any(op.optype == 'Transpose' for op in device.ops.values())


@pytest.mark.unit
def test_tensor_permute_variadic_and_sequence():
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0
    t4 = Tensor(
        name="t4",
        shape=[1, 2, 3, 4],
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        device=device,
    )
    out_v = t4.permute(0, 2, 1, 3)
    assert out_v.logical_shape().as_list() == [1, 3, 2, 4]

    device2 = Device(device_id=1)
    device2.architecture = ARCH.WORMHOLE_B0
    t4b = Tensor(
        name="t4b",
        shape=[1, 2, 3, 4],
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        device=device2,
    )
    out_s = t4b.permute([0, 2, 1, 3])
    assert out_s.logical_shape().as_list() == [1, 3, 2, 4]
    assert all(op.optype == 'Permute' for op in device2.ops.values())
