# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the polaris-side mimic of tt-metal's conv2d parallel-config picker.

Ground-truth values come from cross-referencing the VGG UNet BH p100a refrun CSV
(``__refrun_cache/vgg_unet/bh/p100a/merged_ops_dualref_260519.csv``): each HW
``Conv2dDeviceOperation`` row reports ``CORE COUNT`` and ``INPUT_0_X_PAD[LOGICAL]``,
which together pin ``(num_cores_nhw * num_cores_c, padded_input_channels)``.
"""

from __future__ import annotations

import pytest

from tools.perf_lookup.conv_parallel_config import (
    determine_block_sharded_channel_padding,
    find_closest_largest_divisor_with_num_padding,
    find_closest_largest_divisor_with_num_padding_and_mult,
)


@pytest.mark.unit
def test_find_closest_largest_divisor_with_num_padding_no_padding_needed():
    # 32 / 8 = 4 with no remainder — divisor 8 returned immediately.
    assert find_closest_largest_divisor_with_num_padding(32, 8) == 8


@pytest.mark.unit
def test_find_closest_largest_divisor_with_num_padding_walks_down_to_11():
    # Start at 13: round_up(32, 13)=39, padding 7, shard 3 -> 7 >= 3 → step.
    # Try 12: round_up(32, 12)=36, padding 4, shard 3 -> step.
    # Try 11: round_up(32, 11)=33, padding 1, shard 3 -> stop, return 11.
    assert find_closest_largest_divisor_with_num_padding(32, 13) == 11


@pytest.mark.unit
def test_find_closest_largest_divisor_with_num_padding_and_mult_basic():
    # Same as the BH p100a num_cores_nhw case in d1.conv1: nhw_ntiles=32, grid_y=10, mult=1.
    # Result 8 (matches HW cores=88=8x11).
    assert find_closest_largest_divisor_with_num_padding_and_mult(32, 10, 1) == 8


@pytest.mark.unit
def test_determine_block_sharded_d1_conv1_bh_p100a():
    """BH p100a d1.conv1 — the case that drove this whole audit.

    HW refrun row: ``Conv2dDeviceOperation`` input ``y=1632, x=1056, cores=88,
    BLOCK_SHARDED``. The output spatial volume is 1024 (out_y), so
    ``out_nhw_ntiles = 1024 / 32 = 32``. Channels in = 1024 → 32 blocks of 32.
    Compute grid 12x10. Expected: (8, 11, 1056), total 88 cores.
    """
    result = determine_block_sharded_channel_padding(
        input_channels=1024,
        output_nhw=1024,
        compute_grid_size=(12, 10),
    )
    assert result == (8, 11, 1056)
    assert result[0] * result[1] == 88


@pytest.mark.unit
def test_determine_block_sharded_512ch_no_padding():
    """Most BH p100a convs with 512 channels land on a clean 8x8 = 64-core grid."""
    result = determine_block_sharded_channel_padding(
        input_channels=512,
        output_nhw=256,
        compute_grid_size=(12, 10),
    )
    assert result == (8, 8, 512)


@pytest.mark.unit
def test_determine_block_sharded_wh_n150_no_padding_at_1024ch():
    """WH n150's smaller grid (8x9) divides 32 blocks cleanly → no padding."""
    result = determine_block_sharded_channel_padding(
        input_channels=1024,
        output_nhw=1024,
        compute_grid_size=(8, 9),
    )
    assert result == (8, 8, 1024)


@pytest.mark.unit
def test_determine_block_sharded_rejects_zero_grid():
    with pytest.raises(ValueError, match="compute_grid_size"):
        determine_block_sharded_channel_padding(1024, 1024, (0, 10))


@pytest.mark.unit
def test_determine_block_sharded_rejects_zero_alignment():
    with pytest.raises(ValueError, match="input_channels_alignment"):
        determine_block_sharded_channel_padding(
            1024, 1024, (12, 10), input_channels_alignment=0
        )
