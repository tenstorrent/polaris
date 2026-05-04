#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Layout, shard, and transformer head op descriptors:
  Tilize, Untilize, TilizeWithValPadding, UntilizeWithUnpadding,
  InterleavedToSharded, ShardedToInterleaved, Reshard,
  ConcatHeads, CreateQKVHeads.
Used by the TTNN front-end tracking-only operator APIs (*_op helpers) in ttnn_shim.
"""

from ttsim.ops.desc.registry import register_ops
from ttsim.ops.tensor import _coerce_shape_to_list, require_shape_list

TILE_HEIGHT = 32
TILE_WIDTH = 32

# SimOp domain for layout ops (TTNN / device kernel names; not standard ONNX opset).
_TTNN_OP_DOMAIN = "com.tenstorrent.ttnn"


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _pad_to_tile_shape(shape):
    """Return list with last two dims rounded up to tile boundaries."""
    if len(shape) < 2:
        return list(shape)
    s = list(shape)
    s[-2] = _round_up(s[-2], TILE_HEIGHT)
    s[-1] = _round_up(s[-1], TILE_WIDTH)
    return s


def _nelems(shape):
    if not shape:
        return 1
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _input_storage_shape_for_perf(X, op, logical_shape_list):
    """Dims for input element/byte perf (padded TILE storage when known); output logical shape stays separate."""
    attr_ps = op.attrs.get('input_padded_shape')
    if attr_ps is not None:
        try:
            ps_list = _coerce_shape_to_list(attr_ps)
        except TypeError:
            ps_list = []
        if ps_list:
            return ps_list
    ps_fn = getattr(X, 'padded_shape', None)
    if callable(ps_fn):
        try:
            ps_obj = ps_fn()
        except Exception:
            ps_obj = None
        if ps_obj is not None:
            try:
                return _coerce_shape_to_list(ps_obj)
            except TypeError:
                pass
    return list(logical_shape_list)


def tilize_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Tilize: 1 input (ROW_MAJOR), output same logical shape, padded to tile."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "Tilize shape inference: input tensor shape must be known for element/byte accounting",
    )

    # Output logical shape is unchanged: tilize only reorders data within tile boundaries.
    out_logical = in_shape
    # Padded shape rounds last two dims up to TILE_HEIGHT/TILE_WIDTH so data fits in whole tiles.
    out_padded = _pad_to_tile_shape(in_shape) if len(in_shape) >= 2 else in_shape

    oTList[0].shape = out_logical
    oTList[0].dtype = X.dtype

    # perf_stats: element counts and bytes for input (row-major) and output (tiled); instrs model data movement.
    elem_size = op.attrs.get('element_size', 2)
    in_elems = _nelems(in_shape)
    out_elems = _nelems(out_padded)
    in_bytes = in_elems * elem_size
    out_bytes = out_elems * elem_size

    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_bytes,
        'outBytes': out_bytes,
        'instrs': {'mov': out_elems},
    }
    return


def untilize_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Untilize: 1 input (TILE), output same logical shape (ROW_MAJOR)."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "Untilize shape inference: input tensor shape must be known for element/byte accounting",
    )

    # Untilize preserves logical shape; output is row-major with same dimensions (no padding change in logical view).
    out_shape = in_shape

    oTList[0].shape = out_shape
    oTList[0].dtype = X.dtype

    # perf_stats: input from TILE storage (padded) when available; output logical row-major; mov = elements written out.
    elem_size = op.attrs.get('element_size', 2)
    in_storage = _input_storage_shape_for_perf(X, op, in_shape)
    in_elems = _nelems(in_storage)
    out_elems = _nelems(in_shape)

    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def tilize_with_val_padding_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for TilizeWithValPadding: 1 input, output padded from attrs."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "TilizeWithValPadding shape inference: input tensor shape must be known for element/byte accounting",
    )

    # Output padded shape comes from op attrs (caller may request larger tile-aligned shape); fallback = tile-round input.
    output_padded_shape = op.attrs.get('output_padded_shape', _pad_to_tile_shape(in_shape))
    out_padded = list(output_padded_shape)
    # Logical shape defaults to input; tile reshape may set output_logical_shape (row-major view before tilize).
    out_logical = list(op.attrs.get('output_logical_shape', in_shape))

    oTList[0].shape = out_logical
    oTList[0].dtype = X.dtype

    # perf_stats: in_elems from input shape, out_elems from padded shape (includes padding); mov = elements written out.
    elem_size = op.attrs.get('element_size', 2)
    in_elems = _nelems(in_shape)
    out_elems = _nelems(out_padded)
    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def untilize_with_unpadding_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for UntilizeWithUnpadding: 1 input, output shape from attrs."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "UntilizeWithUnpadding shape inference: input tensor shape must be known for element/byte accounting",
    )

    # Output logical shape is given in attrs (unpadded/cropped size); input may be tile-padded, output is smaller.
    output_shape = op.attrs.get('output_shape', in_shape)
    out_shape = list(output_shape)

    oTList[0].shape = out_shape
    oTList[0].dtype = X.dtype

    # perf_stats: input from TILE storage (padded) when available; out_elems from requested output shape.
    elem_size = op.attrs.get('element_size', 2)
    in_storage = _input_storage_shape_for_perf(X, op, in_shape)
    in_elems = _nelems(in_storage)
    out_elems = _nelems(out_shape)
    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def interleaved_to_sharded_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for InterleavedToSharded: 1->1, same logical shape (memory layout change only)."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "InterleavedToSharded shape inference: input tensor shape must be known",
    )
    oTList[0].shape = list(in_shape)
    oTList[0].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    elems = _nelems(in_shape)
    op.perf_stats = {
        'inElems': elems,
        'outElems': elems,
        'inBytes': elems * elem_size,
        'outBytes': elems * elem_size,
        'instrs': {'mov': elems},
    }
    return


def sharded_to_interleaved_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for ShardedToInterleaved: 1->1, same logical shape (memory layout change only)."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "ShardedToInterleaved shape inference: input tensor shape must be known",
    )
    oTList[0].shape = list(in_shape)
    oTList[0].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    elems = _nelems(in_shape)
    op.perf_stats = {
        'inElems': elems,
        'outElems': elems,
        'inBytes': elems * elem_size,
        'outBytes': elems * elem_size,
        'instrs': {'mov': elems},
    }
    return


def reshard_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Reshard: 1->1, same logical shape (shard-to-shard re-layout)."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "Reshard shape inference: input tensor shape must be known",
    )
    oTList[0].shape = list(in_shape)
    oTList[0].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    elems = _nelems(in_shape)
    op.perf_stats = {
        'inElems': elems,
        'outElems': elems,
        'inBytes': elems * elem_size,
        'outBytes': elems * elem_size,
        'instrs': {'mov': elems},
    }
    return


def nlp_concat_heads_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for ConcatHeads: [B, num_heads, S, head_dim] -> [B, S, num_heads*head_dim]."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "ConcatHeads shape inference: input tensor shape must be known",
    )
    assert len(in_shape) == 4, f"ConcatHeads expects rank-4 input, got rank {len(in_shape)}"
    B, num_heads, S, head_dim = in_shape
    out_shape = [B, S, num_heads * head_dim]

    oTList[0].shape = out_shape
    oTList[0].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    elems = _nelems(in_shape)
    op.perf_stats = {
        'inElems': elems,
        'outElems': elems,
        'inBytes': elems * elem_size,
        'outBytes': elems * elem_size,
        'instrs': {'mov': elems},
    }
    return


def nlp_create_qkv_heads_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for CreateQKVHeads: 1-2 inputs -> 3 outputs (Q, K, V).

    Input: [B, S, (num_heads + 2*num_kv_heads) * head_dim] (fused QKV)
    Optional second input: [B, S, 2*num_kv_heads * head_dim] (separate KV)
    Outputs: Q=[B, num_heads, S, head_dim],
             K=[B, num_kv_heads, head_dim, S] if transpose_k_heads else [B, num_kv_heads, S, head_dim],
             V=[B, num_kv_heads, S, head_dim]
    """
    assert 1 <= len(iTList) <= 2 and len(oTList) == 3
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "CreateQKVHeads shape inference: input tensor shape must be known",
    )

    num_heads = op.attrs.get('num_heads', 1)
    num_kv_heads = op.attrs.get('num_kv_heads', num_heads)
    head_dim = op.attrs.get('head_dim', None)
    # HW returns K pre-transposed when this attr is set; propagate to shape.
    transpose_k = op.attrs.get('transpose_k_heads', False)

    if len(iTList) == 2:
        B = in_shape[0] if len(in_shape) >= 3 else 1
        S = in_shape[-2] if len(in_shape) >= 2 else in_shape[0]
        if head_dim is None:
            head_dim = in_shape[-1] // num_heads
    else:
        B = in_shape[0] if len(in_shape) >= 3 else 1
        S = in_shape[-2] if len(in_shape) >= 2 else in_shape[0]
        if head_dim is None:
            head_dim = in_shape[-1] // (num_heads + 2 * num_kv_heads)

    q_shape = [B, num_heads, S, head_dim]
    k_shape = [B, num_kv_heads, head_dim, S] if transpose_k else [B, num_kv_heads, S, head_dim]
    v_shape = [B, num_kv_heads, S, head_dim]

    oTList[0].shape = q_shape
    oTList[0].dtype = X.dtype
    oTList[1].shape = k_shape
    oTList[1].dtype = X.dtype
    oTList[2].shape = v_shape
    oTList[2].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    in_elems = sum(_nelems(t.shape) for t in iTList if t.shape)
    out_elems = _nelems(q_shape) + _nelems(k_shape) + _nelems(v_shape)
    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def register_layout_ops():
    d = _TTNN_OP_DOMAIN
    _optbl = [
        ['Tilize', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, tilize_sinf, True, True, True, True, True],
        ['Untilize', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, untilize_sinf, True, True, True, True, True],
        ['TilizeWithValPadding', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, tilize_with_val_padding_sinf, True, True, True, True, True],
        ['UntilizeWithUnpadding', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, untilize_with_unpadding_sinf, True, True, True, True, True],
        ['InterleavedToSharded', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, interleaved_to_sharded_sinf, True, True, True, True, True],
        ['ShardedToInterleaved', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, sharded_to_interleaved_sinf, True, True, True, True, True],
        ['Reshard', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, reshard_sinf, True, True, True, True, True],
        ['ConcatHeads', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, nlp_concat_heads_sinf, True, True, True, True, True],
        ['CreateQKVHeads', 'ARITY_VARIADIC[1-2]->3', d, 'COMMON', 24, 21, 2, 1, 3, 3, nlp_create_qkv_heads_sinf, True, True, True, True, True],
    ]
    register_ops('layout', _optbl)
    return
