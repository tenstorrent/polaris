# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The two shard-spec constructors in ``ttsim/front/ttnn/memory.py``.

Both used to discard the shard spec -- ``create_sharded_memory_config`` returned
``shard_spec=None`` and ``create_sharded_memory_config_`` stored the placeholder
``(1, 1)``.  Three consumers read that spec: ``needs_shard_or_reshard`` (every
clause in ``conv2d_utils.cpp:686-745`` indexes ``input_shard_spec.shape``),
``get_input_channels_alignment``, and ``MemoryConfig.__eq__``.  With the stub,
``shard_w % alignment`` was ``1 % 32``, so the reshard predicate answered True for
every conv that reached it, and two configs that hardware treats as different
compared equal.

Neither constructor changes a tensor shape, so a regression here does not crash
anything -- it moves a LUT key.  That is what these tests are for.

The two are NOT interchangeable and the distinction is the easiest thing to get
wrong when editing either:

* ``create_sharded_memory_config``  takes the FULL tensor shape plus a
  ``ShardStrategy``, and DERIVES the per-core extent from the grid.  32 callers.
* ``create_sharded_memory_config_`` takes the per-core extent ALREADY COMPUTED by
  the caller plus a ``TensorMemoryLayout``, and stores it verbatim.  2 callers,
  both in ResNet-50.
"""

import pytest

from ttsim.front.ttnn.buffer import BufferType, ShardOrientation, TensorMemoryLayout
from ttsim.front.ttnn.core import CoreCoord, CoreRange, CoreRangeSet
from ttsim.front.ttnn.memory import (
    MemoryConfig,
    create_sharded_memory_config,
    create_sharded_memory_config_,
)
from ttsim.front.ttnn.tensor import ShardStrategy

ROW = ShardOrientation.ROW_MAJOR
COL = ShardOrientation.COL_MAJOR


def _core_grid(x: int, y: int):
    """``CoreGrid(x=, y=)`` -- the form that exposes ``.x`` / ``.y``."""
    from ttsim.front.ttnn.core import CoreGrid
    return CoreGrid(x=x, y=y)


def _bare_range_set(x: int, y: int) -> CoreRangeSet:
    """A plain CoreRangeSet: same cores, but no ``.x`` / ``.y`` attributes."""
    return CoreRangeSet([CoreRange(CoreCoord(0, 0), CoreCoord(x - 1, y - 1))])


# ===========================================================================
# create_sharded_memory_config -- derives the extent from the grid
# ===========================================================================

@pytest.mark.unit
@pytest.mark.parametrize(
    'strategy, expected_layout, expected_shard',
    [
        # grid 8x4 -> gx=8, gy=4, total=32; tensor (1, 1, 1024, 256) -> sh=1024, sw=256
        (ShardStrategy.BLOCK, TensorMemoryLayout.BLOCK_SHARDED, (256, 32)),
        (ShardStrategy.HEIGHT, TensorMemoryLayout.HEIGHT_SHARDED, (32, 256)),
        (ShardStrategy.WIDTH, TensorMemoryLayout.WIDTH_SHARDED, (1024, 8)),
    ],
)
def test_strategy_selects_both_the_layout_and_the_split_axis(
        strategy, expected_layout, expected_shard) -> None:
    """Ported from ``ttnn/ttnn/core.py::create_sharded_memory_config``::

        BLOCK  -> (sh / grid.y, sw / grid.x)
        HEIGHT -> (sh / (grid.x * grid.y), sw)
        WIDTH  -> (sh, sw / (grid.x * grid.y))

    The three shapes are mutually distinct on a non-square grid, which is why the
    grid here is 8x4 and not 8x8: on a square grid BLOCK and a transposed BLOCK
    produce the same pair and the test would not notice an axis swap.
    """
    mc = create_sharded_memory_config(
        (1, 1, 1024, 256), _core_grid(8, 4), strategy, ROW)
    assert mc.memory_layout == expected_layout
    assert mc.buffer_type == BufferType.L1
    assert mc.shard_spec is not None
    assert mc.shard_spec.shape == expected_shard


@pytest.mark.unit
def test_leading_dims_multiply_into_the_shard_height() -> None:
    """``shard_height = prod(batch dims) * height``.  A 2x3 batch makes the
    height six times larger; ignoring the leading dims would leave it unchanged,
    which the paired assertion below is what detects."""
    grid = _core_grid(8, 4)
    batched = create_sharded_memory_config(
        (2, 3, 1024, 256), grid, ShardStrategy.HEIGHT, ROW)
    plain = create_sharded_memory_config(
        (1, 1, 1024, 256), grid, ShardStrategy.HEIGHT, ROW)
    assert batched.shard_spec.shape == (192, 256)
    assert plain.shard_spec.shape == (32, 256)


@pytest.mark.unit
def test_indivisible_extents_round_up_instead_of_raising() -> None:
    """A deliberate divergence from real ttnn, which raises when a dimension does
    not divide the grid.  Polaris builds graphs for shapes silicon may never run,
    so a hard failure at graph-build time is worse than an approximate extent.

    1000 over 32 cores is 31.25 -> 32, and 100 over 8 is 12.5 -> 13.  Truncating
    division would give 31 and 12, so this also pins the direction of rounding.
    """
    grid = _core_grid(8, 4)
    height = create_sharded_memory_config(
        (1, 1, 1000, 100), grid, ShardStrategy.HEIGHT, ROW)
    block = create_sharded_memory_config(
        (1, 1, 1000, 100), grid, ShardStrategy.BLOCK, ROW)
    assert height.shard_spec.shape == (32, 100)
    assert block.shard_spec.shape == (250, 13)


@pytest.mark.unit
def test_use_height_and_width_flag_takes_the_last_two_dims_verbatim() -> None:
    """With the flag set, the caller has already computed the per-core extent, so
    the last two dims are stored as-is -- the batch dims AND the grid are both
    ignored.  ResNet-50 sets it at four sites (canonical C:1085, C:1421, C:1456,
    C:1510) and VGG UNet at four more.

    The control alongside it uses the same shape and grid with the flag off, so a
    change that made the flag a no-op would fail rather than coincide.
    """
    grid = _core_grid(8, 4)
    flagged = create_sharded_memory_config(
        (2, 3, 1024, 256), grid, ShardStrategy.BLOCK, ROW,
        use_height_and_width_as_shard_shape=True)
    derived = create_sharded_memory_config(
        (2, 3, 1024, 256), grid, ShardStrategy.BLOCK, ROW,
        use_height_and_width_as_shard_shape=False)
    assert flagged.shard_spec.shape == (1024, 256)
    assert derived.shard_spec.shape != (1024, 256)


@pytest.mark.unit
@pytest.mark.parametrize(
    'orientation, expected',
    [(ROW, ROW), (COL, COL), (None, ROW), ('not-an-orientation', ROW)],
)
def test_orientation_is_preserved_or_defaulted(orientation, expected) -> None:
    """``orientation`` defaults to None (many callers omit it) and real ttnn
    enums arrive as foreign objects, so anything that is not a shim
    ShardOrientation falls back to ROW_MAJOR.  COL_MAJOR must survive, or every
    block-sharded config silently becomes row-major."""
    mc = create_sharded_memory_config(
        (1, 1, 1024, 256), _core_grid(8, 4), ShardStrategy.BLOCK, orientation)
    assert mc.shard_spec.orientation == expected


@pytest.mark.unit
def test_a_grid_without_x_and_y_yields_no_shard_spec() -> None:
    """KNOWN LIMITATION, pinned so it is a decision rather than a surprise.

    The extent is derived from ``core_grid.x`` / ``core_grid.y``.  ``CoreGrid``
    always exposes both (core.py:198 and :204, even for the positional form), but
    a bare ``CoreRangeSet`` does not -- and three call sites in
    ``workloads/ttnn/tt_transformers_dualmode/attention.py`` (:268, :307, :330)
    pass exactly that.  For those the spec stays None and this constructor is
    inert, as it was before the port.

    Asserted against a CoreGrid over the identical cores, so the test fails if
    either half of that split changes.
    """
    same_cores = create_sharded_memory_config(
        (1, 1, 1024, 256), _core_grid(8, 4), ShardStrategy.BLOCK, ROW)
    bare = create_sharded_memory_config(
        (1, 1, 1024, 256), _bare_range_set(8, 4), ShardStrategy.BLOCK, ROW)
    assert same_cores.shard_spec is not None
    assert bare.shard_spec is None
    assert bare.memory_layout == TensorMemoryLayout.BLOCK_SHARDED


@pytest.mark.unit
@pytest.mark.parametrize(
    'shape, strategy, expected_layout',
    [
        ((1, 1, 1024, 256), 'nonsense', TensorMemoryLayout.INTERLEAVED),
        (None, ShardStrategy.BLOCK, TensorMemoryLayout.BLOCK_SHARDED),
        ((1024,), ShardStrategy.BLOCK, TensorMemoryLayout.BLOCK_SHARDED),
        (('a', 'b'), ShardStrategy.BLOCK, TensorMemoryLayout.BLOCK_SHARDED),
    ],
)
def test_unusable_input_degrades_quietly_rather_than_failing_the_build(
        shape, strategy, expected_layout) -> None:
    """An unmapped strategy, a missing shape, a 1-D shape and a non-numeric shape
    must all return the pre-port result -- the right layout with no spec -- rather
    than raise.  This runs during graph construction for every workload, so an
    exception here takes down the build of a model that would otherwise simulate.
    """
    mc = create_sharded_memory_config(shape, _core_grid(8, 4), strategy, ROW)
    assert mc.memory_layout == expected_layout
    assert mc.shard_spec is None
    assert mc.buffer_type == BufferType.L1


# ===========================================================================
# create_sharded_memory_config_ -- stores the caller's extent verbatim
# ===========================================================================

@pytest.mark.unit
def test_underscore_variant_stores_a_two_element_extent_verbatim() -> None:
    """ResNet-50's pre-layer4 block config passes
    ``[nearest_32(h // grid_y), w // grid_x]`` (canonical C:1415-1416) -- already
    per-core.  It must be stored, not recomputed: the grid here is 8x4, so a
    constructor that divided again would return something other than (320, 128).
    """
    mc = create_sharded_memory_config_(
        [320, 128], _core_grid(8, 4), TensorMemoryLayout.BLOCK_SHARDED, ROW, True)
    assert mc.shard_spec.shape == (320, 128)
    assert mc.memory_layout == TensorMemoryLayout.BLOCK_SHARDED
    assert mc.buffer_type == BufferType.L1


@pytest.mark.unit
@pytest.mark.parametrize(
    'shape, why',
    [
        ([1, 1, 320, 128], 'a full 4-D tensor shape, as C:1285 passes'),
        ([8, 4, 320, 128], 'a 4-D shape whose leading dims are NOT 1'),
        ([0, 128], 'a zero extent'),
        ([-4, 128], 'a negative extent'),
        (5, 'not a sequence at all'),
        (None, 'nothing'),
    ],
)
def test_underscore_variant_falls_back_to_the_placeholder(shape, why) -> None:
    """Anything that is not a usable 2-tuple of positive ints keeps the old
    ``(1, 1)`` placeholder rather than raising.

    The 4-D rows are not hypothetical: ``ttnn_functional_resnet50.py:1285`` passes
    ``x.shape`` to this constructor, so that call site still gets the placeholder
    and behaves exactly as it did before the port.

    Both 4-D rows are needed.  Relaxing the guard to ``len(shape) >= 2`` would
    read the first two entries instead of rejecting the shape -- and for
    ``[1, 1, 320, 128]`` those are (1, 1), the very value the fallback produces,
    so that row alone cannot see the difference.  ``[8, 4, 320, 128]`` can.
    Paired with the test above, which proves the stored-verbatim path is
    reachable: without it, a constructor that ALWAYS returned (1, 1) would
    satisfy every row here.
    """
    mc = create_sharded_memory_config_(
        shape, _core_grid(8, 4), TensorMemoryLayout.BLOCK_SHARDED, ROW, True)
    assert mc.shard_spec.shape == (1, 1), why


@pytest.mark.unit
def test_underscore_variant_normalises_a_list_grid_to_a_tuple() -> None:
    """A list grid is unhashable and compares unequal to the tuple form, which
    matters because ``ShardSpec.__eq__`` compares grids and ``MemoryConfig.__eq__``
    compares specs."""
    mc = create_sharded_memory_config_(
        [320, 128], [8, 4], TensorMemoryLayout.BLOCK_SHARDED, ROW, True)
    assert mc.shard_spec.grid == (8, 4)
    assert isinstance(mc.shard_spec.grid, tuple)


@pytest.mark.unit
def test_underscore_variant_maps_a_foreign_orientation_by_name() -> None:
    """On the hardware path the orientation is a real ttnn enum, not the shim's.
    It is converted by ``.name`` so COL_MAJOR survives the crossing; anything with
    no usable name defaults to ROW_MAJOR."""
    class _ForeignColMajor:
        name = 'COL_MAJOR'

    grid = _core_grid(8, 4)
    foreign = create_sharded_memory_config_(
        [320, 128], grid, TensorMemoryLayout.BLOCK_SHARDED, _ForeignColMajor(), True)
    native = create_sharded_memory_config_(
        [320, 128], grid, TensorMemoryLayout.BLOCK_SHARDED, COL, True)
    unusable = create_sharded_memory_config_(
        [320, 128], grid, TensorMemoryLayout.BLOCK_SHARDED, 'zzz', True)
    assert foreign.shard_spec.orientation == ShardOrientation.COL_MAJOR
    assert native.shard_spec.orientation == ShardOrientation.COL_MAJOR
    assert unusable.shard_spec.orientation == ShardOrientation.ROW_MAJOR


# ===========================================================================
# What the spec is actually FOR -- the consumers that read it back
# ===========================================================================

@pytest.mark.unit
def test_two_configs_differing_only_in_shard_shape_compare_unequal() -> None:
    """The reason the spec exists.  ResNet-50's bottleneck does::

        if ds_out.memory_config() != out.memory_config():   # canonical C:325-326
            ds_out = ttnn.to_memory_config(ds_out, out.memory_config())

    With ``shard_spec=None`` on both sides these compared equal and the Reshard
    was never emitted -- polaris showed 3 against the capture's 9.  Same layout,
    same buffer type, different per-core extent must be unequal.
    """
    grid = _core_grid(8, 4)
    a = create_sharded_memory_config((1, 1, 1024, 256), grid, ShardStrategy.BLOCK, ROW)
    b = create_sharded_memory_config((1, 1, 2048, 256), grid, ShardStrategy.BLOCK, ROW)
    same = create_sharded_memory_config((1, 1, 1024, 256), grid, ShardStrategy.BLOCK, ROW)
    assert a.memory_layout == b.memory_layout and a.buffer_type == b.buffer_type
    assert a != b
    assert a == same


@pytest.mark.unit
def test_the_shard_width_a_conv_would_read_is_tile_aligned_not_one() -> None:
    """``needs_shard_or_reshard`` computes ``shard_w % alignment``.  Under the old
    ``(1, 1)`` placeholder that was ``1 % 32`` -- non-zero for every conv, so the
    predicate returned True unconditionally and the conv's output sharding was
    decided by a constant.  Checked end-to-end through the real predicate rather
    than by asserting on the number, so the test tracks the consumer.
    """
    from ttsim.front.ttnn.conv_sharding import needs_shard_or_reshard
    from ttsim.front.ttnn.tensor import Layout

    good = create_sharded_memory_config_(
        [320, 128], _core_grid(8, 4), TensorMemoryLayout.BLOCK_SHARDED, ROW, True)
    placeholder = create_sharded_memory_config_(
        [1, 1, 320, 128], _core_grid(8, 4), TensorMemoryLayout.BLOCK_SHARDED, ROW, True)
    kw = dict(in_channels=1024, is_mm_conv=False, input_on_device=True,
              input_tensor_layout=Layout.TILE_LAYOUT)
    assert good.shard_spec.shape[1] % 32 == 0
    assert needs_shard_or_reshard(good, **kw) is False
    assert placeholder.shard_spec.shape[1] % 32 != 0
    assert needs_shard_or_reshard(placeholder, **kw) is True


@pytest.mark.unit
def test_an_unsharded_config_is_still_reported_as_unsharded() -> None:
    """``is_sharded()`` keys off the layout, not the spec, so the degraded paths
    above must not start claiming to be sharded just because a spec is absent --
    nor the reverse."""
    interleaved = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM)
    degraded = create_sharded_memory_config(
        None, _core_grid(8, 4), ShardStrategy.BLOCK, ROW)
    assert interleaved.is_sharded() is False
    assert degraded.is_sharded() is True
    assert degraded.shard_spec is None
