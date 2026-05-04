# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fold SimOp shape-inference tests covering the flatten_nd design decision.

The three tests validate the three distinct code paths in fold_sinf:

1. L1-sharded / ROW_MAJOR (no explicit memory config) → flatten_nd=True
   → output [1, 1, N*Hs*Ws, Cs]  (HW prim::fold kernel behaviour)
2. DRAM-interleaved → flatten_nd=False
   → output [N, Hs, Ws, Cs]       (HW compute_output_specs override)
3. TILE_LAYOUT → flatten_nd=False
   → output [N, Hs, Ws, Cs]       (HW reshape-back-to-4D path)

See ttsim/front/ttnn/op.py (flatten_nd computation) and
ttsim/ops/desc/tensor.py (fold_sinf) for the design rationale.
"""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device
from ttsim.front.ttnn.memory import MemoryConfig
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor


@pytest.mark.unit
def test_fold_default_path_emits_fold_simop_vit_like_shapes() -> None:
    """L1/ROW_MAJOR fold flattens batch*spatial (ViT patch path)."""
    device = Device(device_id=0)
    pixel_values = Tensor(
        shape=[2, 32, 8, 16],
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        device=device,
    )
    x = ttnn.reshape(pixel_values, (2, 32, 4, 32))
    y = ttnn.fold(x, stride_h=8, stride_w=2, use_transpose_as_fold=False)

    # HW Fold kernel flattens batch*spatial: 2*4*2 = 16 rows, 512 channels
    assert y.shape.as_list() == [1, 1, 16, 512]

    optypes = [op.optype for op in device.ops.values()]
    assert optypes[0] == 'Reshape'
    assert optypes[1] == 'Fold'


@pytest.mark.unit
def test_fold_dram_preserves_4d_shape() -> None:
    """DRAM-interleaved fold preserves logical 4D shape [N, Hs, Ws, Cs]."""
    device = Device(device_id=0)
    pixel_values = Tensor(
        shape=[2, 32, 4, 32],
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        device=device,
    )
    # Tensor._memory_config defaults to None; set it directly here because
    # there is no public to_device/memory_config setter in the Polaris Tensor.
    pixel_values._memory_config = MemoryConfig.DRAM
    y = ttnn.fold(pixel_values, stride_h=8, stride_w=2, use_transpose_as_fold=False)

    assert y.shape.as_list() == [2, 4, 2, 512]

    fold_ops = [op for op in device.ops.values() if op.optype == 'Fold']
    assert len(fold_ops) == 1
    assert fold_ops[0].attrs['flatten_nd'] is False


@pytest.mark.unit
def test_fold_tiled_preserves_4d_shape() -> None:
    """Tiled-layout fold preserves logical 4D shape [N, Hs, Ws, Cs]."""
    device = Device(device_id=0)
    pixel_values = Tensor(
        shape=[2, 32, 4, 32],
        dtype=DataType.BFLOAT16,
        layout=Layout.TILE_LAYOUT,
        device=device,
    )
    y = ttnn.fold(pixel_values, stride_h=8, stride_w=2, use_transpose_as_fold=False)

    assert y.shape.as_list() == [2, 4, 2, 512]

    fold_ops = [op for op in device.ops.values() if op.optype == 'Fold']
    assert len(fold_ops) == 1
    assert fold_ops[0].attrs['flatten_nd'] is False
