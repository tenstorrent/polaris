# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for ``ttsim/front/ttnn/conv_sharding.py``.

conv_sharding is 407 new lines and 12 functions ported from
``ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp``. Before this file only
``determine_parallel_config`` was referenced by any test, and only indirectly
through ``test_conv_input_reshard_gate``. That made it the largest untested
surface in the ResNet-50 port.

These functions do not change tensor shapes, which is what makes a regression
here quiet: a wrong answer changes whether ``needs_shard_or_reshard`` predicts a
reshard, which decides the conv's output sharding, which shifts the LUT key. The
failure mode is a silent drop in hit rate, not a crash.

Also covers ``op_canonical``'s averagepool rename, which had no test and is what
let ResNet-50's global average pool key as ``pool2d`` and match the capture.
"""

import pytest

from ttsim.front.ttnn.buffer import BufferType, ShardSpec, TensorMemoryLayout
from ttsim.front.ttnn.conv_sharding import (
    TILE_WIDTH,
    ParallelConfig,
    ShardOrientation,
    _grid_size,
    create_sharded_memory_config_from_parallel_config,
    determine_output_parallel_config,
    determine_parallel_config,
    find_closest_largest_divisor,
    get_input_channels_alignment,
    get_num_cores_channels,
    get_num_cores_nhw,
    needs_shard_or_reshard,
)
from ttsim.front.ttnn.core import CoreCoord, CoreRange, CoreRangeSet
from ttsim.front.ttnn.memory import MemoryConfig
from ttsim.front.ttnn.tensor import Layout

#: Real logical tensix grids, from ``config/tt_bh.yaml:95`` and
#: ``config/tt_wh.yaml:88``.  Used instead of round numbers so the divisor walks
#: below exercise the same values ResNet-50 hits on silicon.
BH_P100A_GRID = (12, 10)
WH_N150_GRID = (8, 9)


def _grid(x: int, y: int) -> CoreRangeSet:
    return CoreRangeSet([CoreRange(CoreCoord(0, 0), CoreCoord(x - 1, y - 1))])


def _sharded(layout, shard_shape, orientation=ShardOrientation.ROW_MAJOR, grid=None):
    """An L1 MemoryConfig carrying a real shard spec."""
    spec = ShardSpec(grid if grid is not None else _grid(8, 8), shard_shape, orientation)
    return MemoryConfig(layout, BufferType.L1, spec)


# ---------------------------------------------------------------------------
# find_closest_largest_divisor — tt::find_closest_largest_divisor
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    'num, start, expected',
    [
        (98, 98, 98),    # already a divisor
        (100, 98, 50),   # walks down to 50
        (98, 12, 7),     # 12..8 do not divide 98; 7 does
        (64, 8, 8),
        (97, 12, 1),     # prime: falls all the way to 1
    ],
)
def test_find_closest_largest_divisor(num, start, expected) -> None:
    """Largest d <= start dividing num. Never returns 0 — the loop stops at 1."""
    assert find_closest_largest_divisor(num, start) == expected


@pytest.mark.unit
def test_divisor_never_returns_zero() -> None:
    """A zero would be a division-by-zero further down determine_parallel_config."""
    for num in range(1, 200):
        assert find_closest_largest_divisor(num, 12) >= 1


# ---------------------------------------------------------------------------
# get_num_cores_nhw / get_num_cores_channels — cpp:302 and cpp:331
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_num_cores_nhw_by_layout() -> None:
    """HEIGHT spreads NHW over every core; WIDTH puts it on one; BLOCK on a row."""
    grid = _grid(8, 8)
    o = ShardOrientation.ROW_MAJOR
    assert get_num_cores_nhw(grid, TensorMemoryLayout.HEIGHT_SHARDED, o) == 64
    assert get_num_cores_nhw(grid, TensorMemoryLayout.WIDTH_SHARDED, o) == 1
    assert get_num_cores_nhw(grid, TensorMemoryLayout.BLOCK_SHARDED, o) == 8


@pytest.mark.unit
def test_num_cores_channels_by_layout() -> None:
    """The mirror image of the above — channels is the other axis."""
    grid = _grid(8, 8)
    o = ShardOrientation.ROW_MAJOR
    assert get_num_cores_channels(grid, TensorMemoryLayout.HEIGHT_SHARDED, o) == 1
    assert get_num_cores_channels(grid, TensorMemoryLayout.WIDTH_SHARDED, o) == 64
    assert get_num_cores_channels(grid, TensorMemoryLayout.BLOCK_SHARDED, o) == 8


@pytest.mark.unit
def test_nhw_times_channels_covers_block_grid() -> None:
    """For BLOCK the two factors must multiply back to the grid's core count."""
    for gx, gy in ((8, 8), (12, 10), (8, 9)):
        grid = _grid(gx, gy)
        o = ShardOrientation.ROW_MAJOR
        nhw = get_num_cores_nhw(grid, TensorMemoryLayout.BLOCK_SHARDED, o)
        ch = get_num_cores_channels(grid, TensorMemoryLayout.BLOCK_SHARDED, o)
        assert nhw * ch == gx * gy, (gx, gy, nhw, ch)


# ---------------------------------------------------------------------------
# get_input_channels_alignment — cpp:92
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_alignment_mm_conv_is_tile_width() -> None:
    """A matmul-lowered conv always aligns to TILE_WIDTH, whatever the layout."""
    assert get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.ROW_MAJOR_LAYOUT,
        False, True, None) == TILE_WIDTH


@pytest.mark.unit
def test_alignment_width_sharded_is_tile_width() -> None:
    assert get_input_channels_alignment(
        TensorMemoryLayout.WIDTH_SHARDED, Layout.ROW_MAJOR_LAYOUT,
        False, False, None) == TILE_WIDTH


@pytest.mark.unit
def test_alignment_tile_input_falls_through_to_tile_width() -> None:
    """The regression this port fixed: an earlier version returned a constant 16
    for the non-width, non-mm case, missing that the rule applies only to a
    ROW_MAJOR input. A TILE input must fall through to TILE_WIDTH."""
    assert get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.TILE_LAYOUT,
        False, False, None) == TILE_WIDTH


@pytest.mark.unit
def test_alignment_row_major_unsharded_is_not_tile_width() -> None:
    """A ROW_MAJOR interleaved input is the case the cpp rule actually targets."""
    got = get_input_channels_alignment(
        TensorMemoryLayout.INTERLEAVED, Layout.ROW_MAJOR_LAYOUT,
        False, False, None)
    assert got != TILE_WIDTH and got >= 1, got


@pytest.mark.unit
def test_alignment_accepts_none_layout() -> None:
    """Callers pass ``getattr(t, 'layout', None)``; None must not raise."""
    assert get_input_channels_alignment(None, None, False, False, None) >= 1


@pytest.mark.unit
@pytest.mark.parametrize(
    'shard_width, expected',
    [
        (64, TILE_WIDTH),   # 64 % 32 == 0
        (32, TILE_WIDTH),   # exactly one tile
        (48, 16),           # 48 % 32 == 16, but 48 % 16 == 0
        (24, 8),            # 24 % 16 == 8, but 24 % 8 == 0
        (20, TILE_WIDTH),   # divides by none of 32/16/8 -> the cpp fallthrough
    ],
)
def test_alignment_derived_from_shard_width(shard_width, expected) -> None:
    """cpp:92 — when the input is already sharded the alignment comes from the
    shard width itself (32 / 16 / 8), not from a constant.

    The 20 case is the one worth keeping: a width divisible by none of the three
    falls through to TILE_WIDTH rather than to the L1 element alignment, and an
    implementation that ended the chain with ``return 8`` would pass every other
    row here.
    """
    mc = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, shard_width))
    got = get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.ROW_MAJOR_LAYOUT, False, False, mc)
    assert got == expected, (shard_width, got, expected)


@pytest.mark.unit
def test_alignment_sharded_without_spec_uses_l1_element_alignment() -> None:
    """Sharded but spec-less: there is no shard width to read, so the rule falls
    back to the 8-element L1 stick.  Paired with the test above so that a change
    collapsing the two paths into one constant fails on one side or the other."""
    mc = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1, None)
    spec_less = get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.ROW_MAJOR_LAYOUT, False, False, mc)
    with_spec = get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.ROW_MAJOR_LAYOUT, False, False,
        _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 64)))
    assert spec_less == 8, spec_less
    assert with_spec == TILE_WIDTH, with_spec


@pytest.mark.unit
def test_alignment_sliced_op_enters_the_rule_without_row_major() -> None:
    """``is_row_major or sliced_op`` — a TILE input normally falls straight through
    to TILE_WIDTH (pinned above); ``sliced_op=True`` must let it in instead."""
    tile_plain = get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.TILE_LAYOUT, False, False, None)
    tile_sliced = get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.TILE_LAYOUT, True, False, None)
    assert tile_plain == TILE_WIDTH
    assert tile_sliced == 8, tile_sliced


# ---------------------------------------------------------------------------
# _grid_size — the empty-CoreRangeSet guard
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_grid_size_of_empty_corerangeset_is_zero() -> None:
    """``CoreRangeSet.bounding_box()`` returns None for an empty set (core.py:151).

    Without the guard this raises AttributeError on ``bb.end_coord`` deep inside
    get_num_cores_nhw, where the traceback says nothing useful.
    """
    assert _grid_size(CoreRangeSet([])) == (0, 0)
    assert _grid_size(_grid(12, 10)) == (12, 10)


# ---------------------------------------------------------------------------
# determine_parallel_config — the BLOCK and WIDTH branches (cpp:169)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_block_sharded_picks_resnet50_layer3_channel_cores() -> None:
    """The one exactly-known value in this file.

    ResNet-50 layer 3 on BH p100a: 1024 input channels, 32-byte alignment, a
    12x10 grid.  tt-metal's divisor walk lands on 11 channel cores and pads 32
    blocks to 33 -- 1056 channels.  1056 is the number the silicon capture
    records for those keys, which is why the padding pass exists at all
    (ttsim/back/device.py:621, and this module's own docstring).
    """
    pcfg = determine_parallel_config(
        TensorMemoryLayout.BLOCK_SHARDED,
        batch_size=16, input_channels=1024, output_height=14, output_width=14,
        output_channels=256, input_channels_alignment=TILE_WIDTH,
        compute_grid_size=BH_P100A_GRID)
    assert get_num_cores_channels(
        pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation) == 11
    assert get_num_cores_nhw(
        pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation) == 10
    assert pcfg.grid.num_cores() == 110


@pytest.mark.unit
@pytest.mark.parametrize(
    'grid_size, orientation, expected_grid',
    [
        # Traced by hand through cpp:188-204 with in_channels=1024 (=32 blocks at
        # 32-byte alignment) and out_nhw_ntiles = ceil(16*14*14 / 32) = 98.
        #   BH ROW_MAJOR: nhw walks from gy=10 -> 10 ; chan walks from gx=12 -> 11
        #   BH COL_MAJOR: nhw walks from gx=12 -> 11 ; chan walks from gy=10 ->  8
        #   WH ROW_MAJOR: nhw walks from gy= 9 ->  9 ; chan walks from gx= 8 ->  8
        #   WH COL_MAJOR: nhw walks from gx= 8 ->  8 ; chan walks from gy= 9 ->  8
        # and cpp:198-199 then puts nhw on x for COL_MAJOR, on y for ROW_MAJOR.
        (BH_P100A_GRID, ShardOrientation.ROW_MAJOR, (11, 10)),
        (BH_P100A_GRID, ShardOrientation.COL_MAJOR, (11, 8)),
        (WH_N150_GRID, ShardOrientation.ROW_MAJOR, (8, 9)),
        (WH_N150_GRID, ShardOrientation.COL_MAJOR, (8, 8)),
    ],
)
def test_block_sharded_places_cores_on_the_orientation_axis(
        grid_size, orientation, expected_grid) -> None:
    """cpp:196-200 assigns the nhw and channel counts to x or y according to the
    orientation.  The grids are pinned as (x, y) extents rather than as core
    totals because BH COL_MAJOR (11x8 = 88) and a transposed 8x11 would tie on
    any count-based assertion while meaning opposite things to the shard spec.

    The BH ROW_MAJOR row is also the layer-3 config the capture confirms.
    """
    pcfg = determine_parallel_config(
        TensorMemoryLayout.BLOCK_SHARDED,
        batch_size=16, input_channels=1024, output_height=14, output_width=14,
        output_channels=256, input_channels_alignment=TILE_WIDTH,
        compute_grid_size=grid_size, block_shard_orientation=orientation)
    assert pcfg.shard_scheme == TensorMemoryLayout.BLOCK_SHARDED
    assert pcfg.shard_orientation == orientation
    assert _grid_size(pcfg.grid) == expected_grid
    assert pcfg.grid.num_cores() <= grid_size[0] * grid_size[1]


@pytest.mark.unit
def test_block_sharded_padding_flag_switches_which_channel_count_is_read() -> None:
    """``enable_channels_padding`` is not a rounding toggle: the padded branch
    divides *input* channels into alignment blocks, the unpadded branch divides
    *output* channels into tiles (cpp:190-196).  op.py passes
    ``enable_channels_padding=not is_mm_conv``, so both are live.

    The channel counts are chosen so the two QUANTITIES disagree, not merely the
    two code paths.  1024 in-channels give 32 alignment blocks and 1536
    out-channels give 48 tiles; through the unpadded walk those are 8 and 12
    respectively.  An earlier draft of this test used 2048 out-channels, where
    both quantities happen to walk to 8 -- it passed just as happily against an
    implementation that read input channels in both branches.
    """
    common = dict(
        batch_size=16, input_channels=1024, output_height=14, output_width=14,
        output_channels=1536, input_channels_alignment=TILE_WIDTH,
        compute_grid_size=BH_P100A_GRID)
    padded = determine_parallel_config(
        TensorMemoryLayout.BLOCK_SHARDED, enable_channels_padding=True, **common)
    unpadded = determine_parallel_config(
        TensorMemoryLayout.BLOCK_SHARDED, enable_channels_padding=False, **common)
    ch_padded = get_num_cores_channels(
        padded.grid, padded.shard_scheme, padded.shard_orientation)
    ch_unpadded = get_num_cores_channels(
        unpadded.grid, unpadded.shard_scheme, unpadded.shard_orientation)
    assert (ch_padded, ch_unpadded) == (11, 12), (ch_padded, ch_unpadded)


@pytest.mark.unit
def test_width_sharded_spreads_channels_over_cores() -> None:
    """WIDTH puts every core on the channel axis, so num_cores_channels is the
    whole grid slice and num_cores_nhw is 1 (cpp:302/331)."""
    pcfg = determine_parallel_config(
        TensorMemoryLayout.WIDTH_SHARDED,
        batch_size=16, input_channels=1024, output_height=14, output_width=14,
        output_channels=1024, input_channels_alignment=TILE_WIDTH,
        compute_grid_size=BH_P100A_GRID)
    assert pcfg.shard_scheme == TensorMemoryLayout.WIDTH_SHARDED
    assert pcfg.shard_orientation == ShardOrientation.ROW_MAJOR
    assert get_num_cores_nhw(pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation) == 1
    assert get_num_cores_channels(
        pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation) == 32
    assert pcfg.grid.num_cores() == 32


@pytest.mark.unit
def test_non_conv_layout_raises_rather_than_guessing() -> None:
    """cpp supports Height, Block and Width only.  INTERLEAVED must raise, not
    fall through to an arbitrary grid -- a silent default here would produce a
    shard spec for an unsharded tensor and shift every downstream LUT key."""
    with pytest.raises(ValueError, match='Height, Block or Width'):
        determine_parallel_config(
            TensorMemoryLayout.INTERLEAVED,
            batch_size=16, input_channels=1024, output_height=14, output_width=14,
            output_channels=256, input_channels_alignment=TILE_WIDTH,
            compute_grid_size=BH_P100A_GRID)


# ---------------------------------------------------------------------------
# create_sharded_memory_config_from_parallel_config — cpp:357
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_channel_shard_pads_to_the_captured_1056() -> None:
    """The multi-channel-core branch, pinned against the capture.

    1024 channels / 16-byte alignment = 64 blocks; over 11 channel cores that is
    ceil(64/11) = 6 blocks each = 96 channels per core, so the tensor is padded
    to 11 * 96 = 1056.  Asserting 96 alone would not catch an off-by-one in the
    core count, so both are pinned.
    """
    pcfg = ParallelConfig(
        grid=_grid(11, 10),
        shard_scheme=TensorMemoryLayout.BLOCK_SHARDED,
        shard_orientation=ShardOrientation.ROW_MAJOR)
    mc = create_sharded_memory_config_from_parallel_config(
        nhw=16 * 14 * 14, channels=1024, pconfig=pcfg, input_channels_alignment=16)
    num_cores_channels = get_num_cores_channels(
        pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation)
    assert num_cores_channels == 11
    assert mc.shard_spec.shape[1] == 96, mc.shard_spec.shape
    assert num_cores_channels * mc.shard_spec.shape[1] == 1056


@pytest.mark.unit
def test_single_channel_core_takes_the_channels_verbatim() -> None:
    """The ``num_cores_channels <= 1`` branch must NOT pad.  Paired with the test
    above: an implementation that always padded would return 1024 -> 1024 here
    only by accident of 1024 already being aligned, so a non-aligned channel
    count is used instead."""
    pcfg = ParallelConfig(
        grid=_grid(8, 8),
        shard_scheme=TensorMemoryLayout.HEIGHT_SHARDED,
        shard_orientation=ShardOrientation.ROW_MAJOR)
    mc = create_sharded_memory_config_from_parallel_config(
        nhw=16 * 14 * 14, channels=1000, pconfig=pcfg, input_channels_alignment=16)
    assert get_num_cores_channels(
        pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation) == 1
    assert mc.shard_spec.shape[1] == 1000, mc.shard_spec.shape
    assert mc.memory_layout == TensorMemoryLayout.HEIGHT_SHARDED
    assert mc.buffer_type == BufferType.L1


@pytest.mark.unit
def test_width_sharded_does_not_pad_the_nhw_extent() -> None:
    """A WIDTH-sharded tensor keeps its whole NHW extent on one core, so there is
    no core count to round up against and cpp leaves the value alone.

    3000 is deliberately not a tile multiple: WIDTH must return it verbatim while
    HEIGHT, on the same grid, rounds 3000 up to 64 cores x 32 = 4096 and shards it
    to 64 per core.  Rounding WIDTH too would silently give 3008 -- a plausible
    number, a wrong shard spec, and a different LUT key.
    """
    def _shape(scheme):
        pcfg = ParallelConfig(
            grid=_grid(8, 8), shard_scheme=scheme,
            shard_orientation=ShardOrientation.ROW_MAJOR)
        return create_sharded_memory_config_from_parallel_config(
            nhw=3000, channels=1024, pconfig=pcfg,
            input_channels_alignment=16).shard_spec.shape

    assert _shape(TensorMemoryLayout.WIDTH_SHARDED) == (3000, 16)
    assert _shape(TensorMemoryLayout.HEIGHT_SHARDED) == (64, 1024)


# ---------------------------------------------------------------------------
# determine_output_parallel_config — cpp:239
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    'layout',
    [TensorMemoryLayout.HEIGHT_SHARDED, TensorMemoryLayout.BLOCK_SHARDED,
     TensorMemoryLayout.WIDTH_SHARDED],
)
@pytest.mark.parametrize(
    'orientation', [ShardOrientation.ROW_MAJOR, ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize('is_mm_conv', [False, True])
def test_output_config_inherits_scheme_and_orientation(
        layout, orientation, is_mm_conv) -> None:
    """The module docstring's central claim: only ``.grid`` is ever recomputed.

    If a future edit made the output adopt a requested layout instead of the
    input's, every conv output memory config would change and the LUT keys with
    them -- with no crash anywhere.  This is the assertion that would fail.
    """
    grid_before = _grid(11, 10)
    inp = ParallelConfig(
        grid=grid_before, shard_scheme=layout, shard_orientation=orientation)
    out = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation, is_mm_conv=is_mm_conv)
    assert out.shard_scheme == layout
    assert out.shard_orientation == orientation
    # cpp builds a fresh ParallelConfig; the caller's own config must survive the
    # call untouched, since op.py:2366 reuses ``pcfg`` after asking for ``ocfg``.
    assert inp.grid is grid_before
    assert inp.shard_scheme == layout


@pytest.mark.unit
def test_output_config_matmul_conv_keeps_the_input_grid() -> None:
    """cpp:243 returns early for a matmul-lowered conv -- the grid is inherited
    verbatim, not recomputed.  Paired with the WIDTH test below, which shows the
    same input DOES get a new grid when is_mm_conv is False; without that pair
    this would also pass against a function that never recomputed anything."""
    inp = ParallelConfig(
        grid=_grid(11, 10), shard_scheme=TensorMemoryLayout.WIDTH_SHARDED,
        shard_orientation=ShardOrientation.ROW_MAJOR)
    as_mm = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation, is_mm_conv=True)
    as_conv = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation, is_mm_conv=False)
    assert as_mm.grid is inp.grid
    assert as_conv.grid.num_cores() == 32
    assert as_conv.grid.num_cores() != inp.grid.num_cores()


@pytest.mark.unit
def test_output_config_block_sharded_rewrites_only_the_channel_axis() -> None:
    """BLOCK output: the channel cores come from the OUT channel count, the nhw
    cores are carried over from the input grid (cpp:248-254).  1024 out-channels
    on a 12-wide grid give 11 channel cores; the input's 10 nhw cores survive."""
    inp = ParallelConfig(
        grid=_grid(11, 10), shard_scheme=TensorMemoryLayout.BLOCK_SHARDED,
        shard_orientation=ShardOrientation.ROW_MAJOR)
    out = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation)
    assert get_num_cores_channels(
        out.grid, out.shard_scheme, out.shard_orientation) == 11
    assert get_num_cores_nhw(out.grid, out.shard_scheme, out.shard_orientation) == 10


@pytest.mark.unit
@pytest.mark.parametrize(
    'orientation, expected_grid, expected_nhw, expected_chan',
    [
        # Input grid 11x10 BLOCK, out_channels=1024 (32 tiles), compute grid 12x10.
        # ROW_MAJOR: channel divisor starts at gx=12 -> 11 cores; nhw read off the
        #            input grid's y = 10.  Laid out x=chan, y=nhw -> (11, 10).
        # COL_MAJOR: channel divisor starts at gy=10 -> 8 cores; nhw read off the
        #            input grid's x = 11.  Laid out x=nhw, y=chan -> (11, 8).
        (ShardOrientation.ROW_MAJOR, (11, 10), 10, 11),
        (ShardOrientation.COL_MAJOR, (11, 8), 11, 8),
    ],
)
def test_output_config_lays_the_grid_out_along_the_given_orientation(
        orientation, expected_grid, expected_nhw, expected_chan) -> None:
    """cpp:256-262 puts the channel and nhw counts on different axes per
    orientation, and picks a different starting divisor for the channel walk.

    THE REGRESSION.  ``block_shard_orientation`` used to default to ROW_MAJOR and
    both op.py call sites omitted it, so a COL_MAJOR input produced a config
    labelled COL_MAJOR whose grid had been laid out ROW_MAJOR -- (11, 11) with
    chan=11, where COL_MAJOR should give (11, 8) with chan=8.  Nothing crashed;
    get_num_cores_channels simply read the wrong extent and the shard width, and
    with it the LUT key, was wrong.  cpp:239-244 makes the argument mandatory,
    which is why the bug cannot be written there; the default is now gone here
    too, and cpp:833 / cpp:1077 show both callers passing the input's own
    orientation.

    Asserted as an (x, y) extent rather than a core count: (11, 8) and a
    transposed (8, 11) both total 88 and mean opposite things to a shard spec.
    """
    inp = ParallelConfig(
        grid=_grid(11, 10), shard_scheme=TensorMemoryLayout.BLOCK_SHARDED,
        shard_orientation=orientation)
    out = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation)
    assert _grid_size(out.grid) == expected_grid
    assert get_num_cores_nhw(
        out.grid, out.shard_scheme, out.shard_orientation) == expected_nhw
    assert get_num_cores_channels(
        out.grid, out.shard_scheme, out.shard_orientation) == expected_chan


@pytest.mark.unit
def test_output_config_requires_an_explicit_orientation() -> None:
    """The argument is mandatory, mirroring cpp:239-244 which gives it no default.

    That is the whole protection: with a default, a caller that forgets it gets a
    silently ROW_MAJOR grid on a COL_MAJOR config.  A TypeError is the correct
    outcome, and this pins it so nobody reinstates the convenience.
    """
    inp = ParallelConfig(
        grid=_grid(11, 10), shard_scheme=TensorMemoryLayout.BLOCK_SHARDED,
        shard_orientation=ShardOrientation.COL_MAJOR)
    with pytest.raises(TypeError, match='block_shard_orientation'):
        determine_output_parallel_config(inp, BH_P100A_GRID, 1024)  # type: ignore[call-arg]


@pytest.mark.unit
def test_output_config_height_sharded_grid_is_untouched() -> None:
    """HEIGHT matches neither elif, so the grid falls through unchanged.  Stated
    as an identity check rather than a core count, because a recomputed grid of
    the same size would satisfy the weaker assertion."""
    inp = ParallelConfig(
        grid=_grid(11, 10), shard_scheme=TensorMemoryLayout.HEIGHT_SHARDED,
        shard_orientation=ShardOrientation.ROW_MAJOR)
    out = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation)
    assert out.grid is inp.grid


# ---------------------------------------------------------------------------
# needs_shard_or_reshard — cpp:686-745
# ---------------------------------------------------------------------------
#
# Every clause below returns True.  A function body of ``return True`` would
# therefore satisfy all of them, which is why the first test here establishes the
# False case: it is the positive control for the whole section, and each clause
# test is a single-field mutation away from it.

_RESHARD_BASE = dict(in_channels=256, is_mm_conv=False,
                     input_on_device=True, input_tensor_layout=Layout.TILE_LAYOUT)


@pytest.mark.unit
def test_no_reshard_when_the_input_is_already_well_formed() -> None:
    """THE POSITIVE CONTROL.  A tile-aligned, ROW_MAJOR, height-sharded input on
    device needs nothing done to it.  If this ever starts returning True, every
    other test in this section has stopped proving anything."""
    mc = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 32))
    assert needs_shard_or_reshard(mc, **_RESHARD_BASE) is False


@pytest.mark.unit
def test_reshard_when_input_is_not_on_device() -> None:
    mc = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 32))
    assert needs_shard_or_reshard(
        mc, **{**_RESHARD_BASE, 'input_on_device': False}) is True


@pytest.mark.unit
def test_reshard_when_memory_config_is_absent() -> None:
    assert needs_shard_or_reshard(None, **_RESHARD_BASE) is True


@pytest.mark.unit
def test_reshard_when_input_is_interleaved() -> None:
    """An interleaved input is by definition not sharded, so the conv must shard it."""
    assert needs_shard_or_reshard(
        MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM),
        **_RESHARD_BASE) is True


@pytest.mark.unit
@pytest.mark.parametrize('spec', [None, ShardSpec(_grid(8, 8), None)])
def test_reshard_when_the_shard_shape_is_unknown(spec) -> None:
    """The regression this guard exists for: sharded-but-shapeless configs cannot
    be evaluated against cpp:686-745, and answering "no reshard" suppressed six
    of ResNet-50's nine Reshards.  Unknown must mean yes."""
    mc = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1, spec)
    assert needs_shard_or_reshard(mc, **_RESHARD_BASE) is True


@pytest.mark.unit
def test_reshard_when_shard_width_violates_the_alignment() -> None:
    """A TILE input aligns to 32, so a 24-wide shard must reshard.  The same
    24-wide shard with a ROW_MAJOR input derives an alignment of 8 from the width
    itself and does NOT -- which is the pairing that proves the clause is reading
    the alignment rather than hardcoding a tile."""
    mc = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 24))
    assert needs_shard_or_reshard(mc, **_RESHARD_BASE) is True
    assert needs_shard_or_reshard(
        mc, **{**_RESHARD_BASE, 'input_tensor_layout': Layout.ROW_MAJOR_LAYOUT}) is False


@pytest.mark.unit
def test_reshard_when_a_non_block_input_is_col_major() -> None:
    """Only BLOCK may be COL_MAJOR; anything else must be reshuffled to ROW_MAJOR."""
    col = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 32), ShardOrientation.COL_MAJOR)
    row = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 32), ShardOrientation.ROW_MAJOR)
    assert needs_shard_or_reshard(col, **_RESHARD_BASE) is True
    assert needs_shard_or_reshard(row, **_RESHARD_BASE) is False


@pytest.mark.unit
def test_block_sharded_matmul_checks_channels_against_the_core_split() -> None:
    """cpp:735 — for a matmul-lowered BLOCK conv the channels must divide exactly
    across the channel cores.  An 8x8 grid gives 8 channel cores at 32 wide, so
    256 fits and 512 does not."""
    mc = _sharded(TensorMemoryLayout.BLOCK_SHARDED, (32, 32))
    mm = {**_RESHARD_BASE, 'is_mm_conv': True}
    assert needs_shard_or_reshard(mc, **{**mm, 'in_channels': 256}) is False
    assert needs_shard_or_reshard(mc, **{**mm, 'in_channels': 512}) is True


@pytest.mark.unit
def test_matmul_conv_requires_a_tile_multiple_shard_height() -> None:
    """cpp:741 applies to matmul-lowered convs only, so the non-mm control must
    tolerate the same ragged height."""
    ragged = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (48, 32))
    assert needs_shard_or_reshard(ragged, **{**_RESHARD_BASE, 'is_mm_conv': True}) is True
    assert needs_shard_or_reshard(ragged, **_RESHARD_BASE) is False


# ---------------------------------------------------------------------------
# op_canonical: averagepool -> pool2d
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('spelling', ['averagepool', 'AveragePool', 'avgpool', 'maxpool', 'MaxPool'])
def test_pooling_canonicalises_to_pool2d(spelling) -> None:
    """tt-metal has ONE Pool2D device op, parameterised by pool type.

    Polaris models max- and average-pooling ONNX-style as two SimOps, so both
    must canonicalise to the hardware name or neither can match a profiler row or
    a LUT entry. ``maxpool`` was mapped from the start; ``averagepool`` was not,
    which is why ResNet-50's global average pool keyed as ('averagepool', ...)
    and missed, and why compare_layers showed pool2d 1 against hardware's 2.
    """
    from tools.profiling.op_canonical import normalize_polaris_optype
    assert normalize_polaris_optype(spelling) == 'pool2d'


@pytest.mark.unit
def test_averagepool_rename_can_only_turn_misses_into_hits() -> None:
    """No LUT in __ext/hlm-lut carries an 'averagepool' key, so nothing was
    relying on the old spelling. Asserted rather than assumed, because the
    opposite would mean this rename broke somebody else's lookups."""
    import glob

    import yaml

    luts = glob.glob('__ext/hlm-lut/*.yaml')
    if not luts:
        pytest.skip('no LUTs present (LFC cache)')
    for path in luts:
        with open(path) as f:
            doc = yaml.safe_load(f)
        codes = {e['key'].get('op_code') for e in (doc.get('entries') or [])}
        assert 'averagepool' not in codes, f'{path} has an averagepool key'
        assert 'avgpool' not in codes, f'{path} has an avgpool key'
