#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device TILE reshape records Untilize* + TilizeWithValPadding (not Reshape)."""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor
from ttsim.front.ttnn.ttnn_shim import ExecutionMode, set_execution_mode


@pytest.mark.unit
def test_ttnn_reshape_tile_device_track_is_untilize_then_tilize_with_val_padding():
    set_execution_mode(ExecutionMode.TRACK_ONLY)
    try:
        device = Device(device_id=0)
        device.architecture = ARCH.WORMHOLE_B0
        # 3*10*10 == 300 == 30*10
        inp = Tensor(
            name="tile_reshape_in",
            shape=[3, 10, 10],
            dtype=DataType.BFLOAT16,
            layout=Layout.TILE_LAYOUT,
            padded_shape=[3, 16, 16],
            device=device,
        )
        out = ttnn.reshape(inp, (30, 10))
        assert out.logical_shape().as_list() == [30, 10]
        assert out.layout == Layout.TILE_LAYOUT
        seq = [op.optype for op in device.ops.values()]
        assert seq == ["UntilizeWithUnpadding", "TilizeWithValPadding"]
        assert "Reshape" not in seq
        ops_list = list(device.ops.values())
        uwu = ops_list[0]
        assert uwu.optype == "UntilizeWithUnpadding"
        ps = uwu.perf_stats
        assert ps is not None
        assert ps["inElems"] == 3 * 16 * 16
        assert ps["outElems"] == 30 * 10
        assert ps["instrs"]["mov"] == 30 * 10
    finally:
        set_execution_mode(ExecutionMode.TRACK_ONLY)


@pytest.mark.unit
def test_ttnn_reshape_tile_aligned_uses_untilize_then_tilize():
    set_execution_mode(ExecutionMode.TRACK_ONLY)
    try:
        device = Device(device_id=0)
        device.architecture = ARCH.WORMHOLE_B0
        inp = Tensor(
            name="tile_aligned",
            shape=[1, 32, 32],
            dtype=DataType.BFLOAT16,
            layout=Layout.TILE_LAYOUT,
            padded_shape=[1, 32, 32],
            device=device,
        )
        out = ttnn.reshape(inp, (32, 32))
        assert out.logical_shape().as_list() == [32, 32]
        seq = [op.optype for op in device.ops.values()]
        assert seq == ["Untilize", "TilizeWithValPadding"]
        uop = list(device.ops.values())[0]
        assert uop.optype == "Untilize"
        ps = uop.perf_stats
        assert ps is not None
        assert ps["inElems"] == 1 * 32 * 32
        assert ps["outElems"] == 1 * 32 * 32
        assert ps["instrs"]["mov"] == 1 * 32 * 32
    finally:
        set_execution_mode(ExecutionMode.TRACK_ONLY)
