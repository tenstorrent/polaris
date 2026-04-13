#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device to_layout(TILE->ROW_MAJOR) with padding change: single UntilizeWithUnpadding SimOp."""

import pytest

from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor
from ttsim.front.ttnn.ttnn_shim import ExecutionMode, set_execution_mode, to_layout


@pytest.mark.unit
def test_to_layout_padding_change_emits_only_untilize_with_val_unpadding():
    set_execution_mode(ExecutionMode.TRACK_ONLY)
    try:
        device = Device(device_id=0)
        device.architecture = ARCH.WORMHOLE_B0
        inp = Tensor(
            name="tiled_in",
            shape=[1, 3, 30, 30],
            dtype=DataType.BFLOAT16,
            layout=Layout.TILE_LAYOUT,
            padded_shape=[1, 3, 32, 32],
            device=device,
        )
        out = to_layout(inp, layout=Layout.ROW_MAJOR_LAYOUT)
        assert out.logical_shape().as_list() == [1, 3, 30, 30]
        seq = [op.optype for op in device.ops.values()]
        assert seq[-1] == "UntilizeWithUnpadding"
        assert "Reshape" not in seq
    finally:
        set_execution_mode(ExecutionMode.TRACK_ONLY)


@pytest.mark.unit
def test_to_layout_execute_and_track_padding_change_emits_only_untilize_with_val_unpadding():
    set_execution_mode(ExecutionMode.EXECUTE_AND_TRACK)
    try:
        device = Device(device_id=0)
        device.architecture = ARCH.WORMHOLE_B0
        inp = Tensor(
            name="tiled_in_eat",
            shape=[1, 3, 30, 30],
            dtype=DataType.BFLOAT16,
            layout=Layout.TILE_LAYOUT,
            padded_shape=[1, 3, 32, 32],
            device=device,
        )
        out = to_layout(inp, layout=Layout.ROW_MAJOR_LAYOUT)
        assert out.logical_shape().as_list() == [1, 3, 30, 30]
        seq = [op.optype for op in device.ops.values()]
        assert seq[-1] == "UntilizeWithUnpadding"
        assert "Reshape" not in seq
    finally:
        set_execution_mode(ExecutionMode.TRACK_ONLY)
