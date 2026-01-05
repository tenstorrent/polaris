# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression: default ``ttnn.fold`` emits a ``Fold`` SimOp (ViT patch path)."""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor


@pytest.mark.unit
def test_fold_default_path_emits_fold_simop_vit_like_shapes() -> None:
    """Reshape + default ``fold`` (same pattern as ViT patch: prep reshape then ``Fold`` SimOp)."""
    device = Device(device_id=0)
    # Element count preserved: 2*32*8*16 == 2*32*4*32
    pixel_values = Tensor(
        shape=[2, 32, 8, 16],
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        device=device,
    )
    x = ttnn.reshape(pixel_values, (2, 32, 4, 32))
    y = ttnn.fold(x, stride_h=8, stride_w=2, use_transpose_as_fold=False)

    assert y.shape.as_list() == [2, 4, 2, 512]

    optypes = [op.optype for op in device.ops.values()]
    assert optypes[0] == 'Reshape'
    assert optypes[1] == 'Fold'
    assert 'Fold' in optypes
