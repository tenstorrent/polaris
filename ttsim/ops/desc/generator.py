#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.desc.registry import register_ops
import numpy as np
from loguru import logger

def ConstantOpInference(inT, outT, op, **kwargs):
    # Constant op outputs a constant tensor defined via attribute 'value'
    value = op.attrs.get('value', None)
    if value is None:
        raise ValueError(f"ConstantOpInference: Constant op {op.name} missing 'value' attribute")
    outT[0].shape = list(value.shape)
    outT[0].dtype = value.dtype
    outT[0].data = value

    # For perf counts, we assume Constant op has zero compute and memory cycles
    op.perf_stats = {
            'inBytes' : 4,
            'outBytes': outT[0].nbytes(),
            'inElems' : 1,
            'outElems': outT[0].nelems(),
            'instrs'  : {'mov': outT[0].nelems()},
            }
    return

def ConstantOfShapeOpInference(inT, outT, op, **kwargs):
    if len(inT) < 1:
        raise ValueError(f"ConstantOfShapeOpInference: op {op.name} missing input shape tensor")

    if inT[0].data is None:
        # Fallback: some upstream shape tensors don't carry concrete data
        # (e.g. unresolved ONNX dims). inT[0] is itself a 1-D tensor whose
        # *values* are the target output shape, so the number of dims we
        # need to fabricate is inT[0]'s own declared shape (e.g.
        # inT[0].shape == [3] means ConstantOfShape's output should be
        # rank 3) -- default every dim to 1 rather than hard-failing or
        # assuming a fixed rank.
        # NOTE: keep this local -- inT[0] is a shared graph node, so writing
        # the fabricated values back onto it would leak them to its other
        # consumers.
        if inT[0].check_shape() and len(inT[0].shape) > 0:
            n_dims = int(inT[0].shape[0])
        else:
            n_dims = 1
        shape_data = np.ones(n_dims, dtype=np.int64)
        logger.warning(
            "ConstantOfShape input SimTensor({}) has no concrete data -- defaulting "
            "output to all-1s shape (rank={})", inT[0].name, n_dims,
        )
    else:
        shape_data = inT[0].data

    output_shape = [int(x) for x in shape_data]
    fill_value = op.attrs.get('value', np.array([0.0], dtype=np.float32))

    outT[0].shape = output_shape
    outT[0].dtype = np.asarray(fill_value).dtype
    # Also compute the actual fill data -- needed by downstream ops
    # (Concat/Cast/Pad etc.) that require concrete values, not just shape.
    outT[0].data = np.full(output_shape, np.asarray(fill_value).reshape(-1)[0], dtype=outT[0].dtype)

    op.perf_stats = {
        'inBytes': 4 * len(output_shape),
        'outBytes': outT[0].nbytes(),
        'inElems': len(output_shape),
        'outElems': outT[0].nelems(),
        'instrs': {'mov': outT[0].nelems()},
    }
    return

def RangeOpInference(inT, outT, op, **kwargs):
    start_t, limit_t, delta_t = inT[0], inT[1], inT[2]
    if start_t.data is not None and limit_t.data is not None and delta_t.data is not None:
        start = np.asarray(start_t.data).reshape(-1)[0]
        limit = np.asarray(limit_t.data).reshape(-1)[0]
        delta = np.asarray(delta_t.data).reshape(-1)[0]
        data = np.arange(start, limit, delta)
        if start_t.dtype is not None:
            data = data.astype(start_t.dtype)
        outT[0].shape = list(data.shape)
        outT[0].dtype = data.dtype
        outT[0].data = data
    else:
        logger.warning(
            "Range op {} has start/limit/delta without concrete data -- defaulting "
            "output to shape [1]", op.name,
        )
        outT[0].shape = [1]
        outT[0].dtype = start_t.dtype if getattr(start_t, 'dtype', None) is not None else np.int64
        outT[0].data = np.zeros([1], dtype=outT[0].dtype)
    op.perf_stats = {
        'inBytes': 4 * 3, 'outBytes': outT[0].nbytes(),
        'inElems': 3, 'outElems': outT[0].nelems(),
        'instrs': {'mov': outT[0].nelems()},
    }
    return

def register_generator_ops():
    _optbl = [
        ['RandomUniform',     'ARITY_0->1', 'ai.onnx', 'COMMON', 22, 22, 0, 0, 1, 1, 'g1_func', True, True, True, True, True],
        ['RandomNormal',      'ARITY_0->1', 'ai.onnx', 'COMMON', 22, 22, 0, 0, 1, 1, 'g1_func', True, True, True, True, True],
        ['EyeLike',           'ARITY_1->1', 'ai.onnx', 'COMMON', 22, 22, 1, 1, 1, 1, 'g1_func', True, True, True, True, True],
        ['RandomUniformLike', 'ARITY_1->1', 'ai.onnx', 'COMMON', 22, 22, 1, 1, 1, 1, 'g1_func', True, True, True, True, True],
        ['RandomNormalLike',  'ARITY_1->1', 'ai.onnx', 'COMMON', 22, 22, 1, 1, 1, 1, 'g1_func', True, True, True, True, True],
        ['Multinomial',       'ARITY_1->1', 'ai.onnx', 'COMMON', 22, 22, 1, 1, 1, 1, 'g1_func', True, True, True, True, True],
        ['Bernoulli',         'ARITY_1->1', 'ai.onnx', 'COMMON', 22, 22, 1, 1, 1, 1, 'g1_func', True, True, True, True, True],
        ['Range',             'ARITY_3->1', 'ai.onnx', 'COMMON', 11, 11, 3, 3, 1, 1, RangeOpInference, True, True, True, True, True],
        ['Constant',          'ARITY_0->1', 'ai.onnx', 'COMMON', 24, 21, 0, 0, 1, 1, ConstantOpInference, True, True, True, True, True],
        ['ConstantOfShape',   'ARITY_1->1', 'ai.onnx', 'COMMON', 24, 21, 1, 1, 1, 1, ConstantOfShapeOpInference, True, True, True, True, True],
    ]
    register_ops('generator', _optbl)
    return
