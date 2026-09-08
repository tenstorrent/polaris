# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the arch-aware lm_head column-split transform (issue #477).

The front-end lm_head emits ONE canonical GEMM (dim x vocab); the backend
Device._split_column_ops expands it into the arch's L1-sized vocab column tiles + Concat,
mirroring tt-metal ModelArgs.get_lm_head_max_columns_per_device + LMHead dram_sharded split
(models/tt_transformers/tt/{model_config,lm_head}.py):
  - BH single-chip: vocab // 8      -> 8 x 16032 (vocab=128256)
  - WH:             668 * cores(dim) -> 3 x 42752 (dim=4096, 8x8 grid)
"""
import pytest

import ttsim.front.ttnn as ttnn
from ttsim.back.device import Device
from ttsim.config.wl2archmap import WL2ArchColumnSplit
from ttsim.front.ttnn.device import close_device, open_device, set_default_device

_VOCAB = 128256
_DIM = 4096


def _build_lm_head_graph():
    """A one-matmul graph mimicking the POLARIS lm_head: x[1,1,32,dim] @ W[dim,vocab] -> [1,1,32,vocab]."""
    dev = open_device()
    set_default_device(dev)
    x = ttnn.zeros([1, 1, 32, _DIM], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w = ttnn.zeros([_DIM, _VOCAB], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.linear(x, w, dtype=ttnn.bfloat8_b, compute_kernel_config=ttnn.MathFidelity.HiFi2)
    return dev, dev.get_graph()


class _FakeDev:
    """Borrows the real Device split methods without constructing a full backend Device."""

    def __init__(self, devname):
        self.devname = devname

    _is_blackhole = Device._is_blackhole
    _is_wormhole = Device._is_wormhole
    _lm_head_core_grid_num_cores = staticmethod(Device._lm_head_core_grid_num_cores)
    _lm_head_chunk_widths = Device._lm_head_chunk_widths
    _split_column_ops = Device._split_column_ops
    _split_one_matmul_columns = Device._split_one_matmul_columns


_SPEC = WL2ArchColumnSplit.from_list(
    [{'op_type': 'MatMul', 'match_output_x': _VOCAB, 'kind': 'lm_head_vocab'}]
)


def _matmuls(g):
    return [op for op in g._ops.values() if op.optype == 'MatMul']


def _concats(g):
    return [op for op in g._ops.values() if op.optype == 'Concat']


def _sts(g):
    return [op for op in g._ops.values() if op.optype == 'ShardedToInterleaved']


@pytest.mark.unit
@pytest.mark.parametrize('devname,n,width', [('Blackhole', 8, 16032), ('Wormhole', 3, 42752)])
def test_lm_head_column_split(devname, n, width):
    dev, g = _build_lm_head_graph()
    try:
        assert len(_matmuls(g)) == 1, 'precondition: exactly one lm_head matmul'
        _FakeDev(devname)._split_column_ops(g, _SPEC)

        mms = _matmuls(g)
        assert len(mms) == n, f'{devname}: expected {n} chunk matmuls, got {len(mms)}'
        # every chunk weight (input_1) carries the arch tile width, and preserves the
        # bf8 dtype + TILE layout of the original weight (clone fidelity).
        for op in mms:
            w_t = g._tensors[op.inList[1]]
            assert int(w_t.shape[-1]) == width, f'chunk weight width {w_t.shape[-1]} != {width}'
            assert w_t.get_layout() == ttnn.TILE_LAYOUT
            assert w_t._ttnn_dtype == ttnn.bfloat8_b
            assert op.attrs.get('math_fidelity') == 'HiFi2'  # attr carried to each chunk

        # full HW region: N chunk matmuls -> N ShardedToInterleaved -> Concat
        sts = _sts(g)
        assert len(sts) == n, f'expected {n} ShardedToInterleaved, got {len(sts)}'
        for op in sts:
            in_t = g._tensors[op.inList[0]]
            assert int(in_t.shape[-1]) == width, 'STS input is a chunk output'
            assert in_t._ttnn_dtype == ttnn.bfloat8_b, 'STS input keeps lm_head bf8 dtype'

        concats = _concats(g)
        assert len(concats) == 1, 'exactly one Concat reunifying the tiles'
        y_t = g._tensors[concats[0].outList[0]]
        assert int(y_t.shape[-1]) == _VOCAB, 'concat output restores full vocab'
        assert len(concats[0].inList) == n, 'concat consumes all N tiles (via STS)'
        # concat consumes the STS outputs, not the matmul outputs directly
        sts_out = {op.outList[0] for op in sts}
        assert set(concats[0].inList) == sts_out, 'concat inputs are the STS outputs'

        # idempotent: after split no op has output_x == vocab, so a re-run is a no-op.
        _FakeDev(devname)._split_column_ops(g, _SPEC)
        assert len(_matmuls(g)) == n
        assert len(_concats(g)) == 1
    finally:
        close_device(dev)


@pytest.mark.unit
def test_empty_split_spec_is_noop():
    dev, g = _build_lm_head_graph()
    try:
        _FakeDev('Blackhole')._split_column_ops(g, WL2ArchColumnSplit.from_list(None))
        assert len(_matmuls(g)) == 1
        assert len(_concats(g)) == 0
    finally:
        close_device(dev)


@pytest.mark.unit
def test_clone_preserves_layout_memory_dtype():
    """Regression for the shim Tensor.clone() fidelity fix (layout/memory/dtype preserved)."""
    dev = open_device()
    set_default_device(dev)
    try:
        w = ttnn.zeros([_DIM, _VOCAB], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        c = w.clone()
        assert c.get_layout() == w.get_layout() == ttnn.TILE_LAYOUT
        assert c._ttnn_dtype == w._ttnn_dtype == ttnn.bfloat8_b
        assert c._memory_config == w._memory_config
        assert list(c.shape) == list(w.shape)
    finally:
        close_device(dev)
