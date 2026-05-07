#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ttnn.concat wrapper (both calling conventions)."""

import pytest
import ttsim.front.ttnn as ttnn
import ttsim.front.ttnn.minitorch_shim as torch
from ttsim.front.ttnn.device import open_device, close_device, set_default_device
from ttsim.front.ttnn.tensor import ttnn_random


@pytest.fixture(scope='module')
def device():
    from ttsim.front.ttnn.device import get_default_device as _get
    try:
        prev = _get()
    except AssertionError:
        prev = None
    dev = open_device()
    set_default_device(dev)
    yield dev
    close_device(dev)
    set_default_device(prev)


def make_tensor(shape):
    raw = ttnn_random(shape, -1, 1, dtype=torch.bfloat16)
    return ttnn.from_torch(raw, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


# ---------------------------------------------------------------------------
# Calling-convention tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_concat_list_first_positional_dim(device):
    """ttnn.concat([t1, t2], dim) — canonical tt-metal list-first form."""
    t1 = make_tensor([1, 64, 32, 32])
    t2 = make_tensor([1, 64, 32, 32])
    out = ttnn.concat([t1, t2], 1)
    assert list(out.shape) == [1, 128, 32, 32]


@pytest.mark.unit
def test_concat_list_first_dim_kwarg(device):
    """ttnn.concat([t1, t2], dim=1) — list-first with dim= keyword."""
    t1 = make_tensor([1, 64, 32, 32])
    t2 = make_tensor([1, 64, 32, 32])
    out = ttnn.concat([t1, t2], dim=1)
    assert list(out.shape) == [1, 128, 32, 32]


@pytest.mark.unit
def test_concat_list_first_axis_kwarg(device):
    """ttnn.concat([t1, t2], axis=1) — list-first with axis= keyword."""
    t1 = make_tensor([1, 64, 32, 32])
    t2 = make_tensor([1, 64, 32, 32])
    out = ttnn.concat([t1, t2], axis=1)
    assert list(out.shape) == [1, 128, 32, 32]


@pytest.mark.unit
def test_concat_positional_form(device):
    """ttnn.concat(t1, t2, axis=1) — existing positional form unchanged."""
    t1 = make_tensor([1, 64, 32, 32])
    t2 = make_tensor([1, 64, 32, 32])
    out = ttnn.concat(t1, t2, axis=1)
    assert list(out.shape) == [1, 128, 32, 32]


@pytest.mark.unit
def test_concat_list_first_with_extra_kwargs(device):
    """ttnn.concat([t1, t2], dim, **kwargs) — extra kwargs are forwarded."""
    t1 = make_tensor([1, 512, 32, 32])
    t2 = make_tensor([1, 512, 32, 32])
    # memory_config is an extra kwarg; the wrapper must forward it intact.
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    mem_cfg = ttnn.create_sharded_memory_config(
        (16, 1024),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    out = ttnn.concat([t1, t2], 1, memory_config=mem_cfg)
    assert list(out.shape) == [1, 1024, 32, 32]


@pytest.mark.unit
def test_concat_three_tensors_list_first(device):
    """ttnn.concat([t1, t2, t3], dim) — more than two tensors."""
    t1 = make_tensor([1, 32, 8, 8])
    t2 = make_tensor([1, 32, 8, 8])
    t3 = make_tensor([1, 32, 8, 8])
    out = ttnn.concat([t1, t2, t3], 1)
    assert list(out.shape) == [1, 96, 8, 8]


@pytest.mark.unit
def test_concat_numpy_int_dim(device):
    """numpy.int64 positional dim must be accepted (not only plain int)."""
    import numpy as np
    t1 = make_tensor([1, 64, 8, 8])
    t2 = make_tensor([1, 64, 8, 8])
    out = ttnn.concat([t1, t2], np.int64(1))
    assert list(out.shape) == [1, 128, 8, 8]


@pytest.mark.unit
def test_concat_dim_axis_conflict_raises(device):
    """Passing both a positional dim and axis= kwarg must raise TypeError."""
    t1 = make_tensor([1, 64, 4, 4])
    t2 = make_tensor([1, 64, 4, 4])
    with pytest.raises(TypeError, match='conflicting values for axis'):
        ttnn.concat([t1, t2], 1, axis=2)
