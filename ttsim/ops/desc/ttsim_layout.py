#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Layout op descriptors for Tilize, Untilize, TilizeWithValPadding, UntilizeWithUnpadding.
Used by the TTNN front-end tracking-only operator APIs (tilize_op, etc.) in ttnn_shim.
"""

from ttsim.ops.desc.registry import register_ops

TILE_HEIGHT = 32
TILE_WIDTH = 32


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


def tilize_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Tilize: 1 input (ROW_MAJOR), output same logical shape, padded to tile."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = list(X.shape) if X.shape is not None else []
    out_logical = in_shape
    out_padded = _pad_to_tile_shape(in_shape) if len(in_shape) >= 2 else in_shape

    oTList[0].shape = out_logical
    oTList[0].dtype = X.dtype

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
    in_shape = list(X.shape) if X.shape is not None else []
    out_shape = in_shape

    oTList[0].shape = out_shape
    oTList[0].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    n = _nelems(in_shape)
    nbytes = n * elem_size

    op.perf_stats = {
        'inElems': n,
        'outElems': n,
        'inBytes': nbytes,
        'outBytes': nbytes,
        'instrs': {'mov': n},
    }
    return


def tilize_with_val_padding_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for TilizeWithValPadding: 1 input, output padded from attrs."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = list(X.shape) if X.shape is not None else []
    output_padded_shape = op.attrs.get('output_padded_shape', _pad_to_tile_shape(in_shape))
    out_padded = list(output_padded_shape)
    out_logical = in_shape

    oTList[0].shape = out_logical
    oTList[0].dtype = X.dtype

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
    in_shape = list(X.shape) if X.shape is not None else []
    output_shape = op.attrs.get('output_shape', in_shape)
    out_shape = list(output_shape)

    oTList[0].shape = out_shape
    oTList[0].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    in_elems = _nelems(in_shape)
    out_elems = _nelems(out_shape)
    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def register_layout_ops():
    _optbl = [
        ['Tilize', 'ARITY_1->1', 'ttnn.layout', 'COMMON', 24, 21, 1, 1, 1, 1, tilize_sinf, True, True, True, True, True],
        ['Untilize', 'ARITY_1->1', 'ttnn.layout', 'COMMON', 24, 21, 1, 1, 1, 1, untilize_sinf, True, True, True, True, True],
        ['TilizeWithValPadding', 'ARITY_1->1', 'ttnn.layout', 'COMMON', 24, 21, 1, 1, 1, 1, tilize_with_val_padding_sinf, True, True, True, True, True],
        ['UntilizeWithUnpadding', 'ARITY_1->1', 'ttnn.layout', 'COMMON', 24, 21, 1, 1, 1, 1, untilize_with_unpadding_sinf, True, True, True, True, True],
    ]
    register_ops('layout', _optbl)
    return
