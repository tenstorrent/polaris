#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .registry import register_ops
import numpy as np
from loguru import logger

def if_sinf(iTList, oTList, op, **kwargs):
    if len(iTList) < 1:
        raise ValueError("If operator expects at least one input (the condition tensor)")
    cond_tensor = iTList[0]
    assert cond_tensor.check_shape(), f"Condition tensor shape not defined: {cond_tensor}"
    oTList[0].shape = [256]  # Placeholder shape; actual shape depends on the branches and is not known at this point
    oTList[0].dtype = np.dtype(np.float32)
    op.perf_stats = {
        'inElems': cond_tensor.nelems(),
        'outElems': oTList[0].nelems(),
        'inBytes': cond_tensor.nbytes(op.precision),
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': {'cmp': oTList[0].nelems()},
    }
    return

def register_controlflow_ops():
    _optbl = [
        ['If',   'ARITY_1->VARIADIC[1-*]',             'ai.onnx', 'COMMON',  24,  21,  1,           1,  2147483647,  1,  if_sinf,   True,  True,  True,  True,  True],
        ['Loop', 'ARITY_VARIADIC[2-*]->VARIADIC[1-*]', 'ai.onnx', 'COMMON',  24,  21,  2147483647,  2,  2147483647,  1,  'LoopInferenceFunction', True,  True,  True,  True,  True],
        ['Scan', 'ARITY_VARIADIC[1-*]->VARIADIC[1-*]', 'ai.onnx', 'COMMON',  24,  21,  2147483647,  1,  2147483647,  1,  'ScanInferenceFunction', True,  True,  True,  True,  True],
        ]

    register_ops('controlflow', _optbl)
    return

def register_image_ops():
    _optbl = [
        ['ImageDecoder',  'ARITY_1->1',  'ai.onnx',  'COMMON',  20,  20,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        ]
    register_ops('image', _optbl)
    return

def register_object_detection_ops():
    _optbl = [
        ['RoiAlign',          'ARITY_3->1',             'ai.onnx',  'COMMON',  22,  22,  3,  3,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        ['NonMaxSuppression', 'ARITY_VARIADIC[2-5]->1', 'ai.onnx',  'COMMON',  11,  11,  5,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        ]
    register_ops('object_detection', _optbl)
    return

def register_optional_ops():
    _optbl = [
        ['OptionalGetElement', 'ARITY_1->1',             'ai.onnx', 'COMMON',  18,  18,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        ['Optional',           'ARITY_VARIADIC[0-1]->1', 'ai.onnx', 'COMMON',  15,  15,  1,  0,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        ['OptionalHasElement', 'ARITY_VARIADIC[0-1]->1', 'ai.onnx', 'COMMON',  18,  18,  1,  0,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        ]
    register_ops('optional', _optbl)
    return

def dequantizelinear_sinf(iTList, oTList, op, **kwargs):
    assert len(iTList) >= 2, f"DequantizeLinear expects at least 2 inputs (x, scale), got {len(iTList)}"
    X = iTList[0]              
    scale = iTList[1]          
    Y = oTList[0]

    assert X.check_shape(), f"Input tensor shape not defined: {X}"
    assert scale.check_shape(), f"Scale tensor shape not defined: {scale}"
    assert scale.rank() <= X.rank(), \
        f"Scale rank ({scale.rank()}) must be <= input rank ({X.rank()}) for broadcasting"

    Y.shape = X.shape
    Y.dtype = np.dtype(np.float32)

    op.perf_stats = {
        'inElems': X.nelems(),
        'outElems': Y.nelems(),
        'inBytes': X.nbytes(op.precision),
        'outBytes': Y.nbytes(op.precision),
        'instrs': {'mov': Y.nelems()},
    }
    logger.warning(
        "DequantizeLinear perf_stats only account for the main data tensor X; "
        "scale/zero_point inputs are not included in inElems/inBytes accounting.",
        once=True,
    )
    return

def quantizelinear_sinf(iTList, oTList, op, **kwargs):
    assert len(iTList) >= 2, f"QuantizeLinear expects at least 2 inputs (x, scale), got {len(iTList)}"
    X = iTList[0]               
    scale = iTList[1]           
    zp = iTList[2] if len(iTList) >= 3 else None
    Y = oTList[0]

    assert X.check_shape(), f"Input tensor shape not defined: {X}"
    assert scale.check_shape(), f"Scale tensor shape not defined: {scale}"
    assert scale.rank() <= X.rank(), \
        f"Scale rank ({scale.rank()}) must be <= input rank ({X.rank()}) for broadcasting"
    if zp is not None:
        assert zp.check_shape(), f"ZeroPoint tensor shape not defined: {zp}"
        assert zp.rank() <= X.rank(), \
            f"ZeroPoint rank ({zp.rank()}) must be <= input rank ({X.rank()}) for broadcasting"

    Y.shape = X.shape

    if zp is not None and zp.dtype is not None:
        output_dtype = zp.dtype
    else:
        output_dtype = np.dtype(np.uint8)

    Y.dtype = np.dtype(output_dtype)

    op.perf_stats = {
        'inElems': X.nelems(),
        'outElems': Y.nelems(),
        'inBytes': X.nbytes(op.precision),
        'outBytes': Y.nbytes(op.precision),
        'instrs': {'mov': Y.nelems()},
    }
    logger.warning(
        "QuantizeLinear perf_stats only account for the main data tensor X; "
        "scale/zero_point inputs are not included in inElems/inBytes accounting.",
        once=True,
    )
    return

def register_quantization_ops():
    _optbl = [
        ['DynamicQuantizeLinear', 'ARITY_1->3',              'ai.onnx',  'COMMON',  11,  11,  1,  1,  3,  3,  'inline_lambda',  True,  True,  True,  True,  True],
        ['QuantizeLinear',        'ARITY_VARIADIC[2-3]->1',  'ai.onnx',  'COMMON',  24,  21,  3,  2,  1,  1,  quantizelinear_sinf,  True,  True,  True,  True,  True],
        ['DequantizeLinear',      'ARITY_VARIADIC[2-3]->1',  'ai.onnx',  'COMMON',  24,  21,  3,  2,  1,  1,  dequantizelinear_sinf,  True,  True,  True,  True,  True],
        ]
    register_ops('quantization', _optbl)
    return

def register_rnn_ops():
    _optbl = [
        ['RNN',  'ARITY_VARIADIC[3-6]->VARIADIC[0-2]',  'ai.onnx',  'COMMON',  22,  22,  6,  3,  2,  0,  'no_inference',  True,  True,  True,  True,  True],
        ['GRU',  'ARITY_VARIADIC[3-6]->VARIADIC[0-2]',  'ai.onnx',  'COMMON',  22,  22,  6,  3,  2,  0,  'no_inference',  True,  True,  True,  True,  True],
        ['LSTM', 'ARITY_VARIADIC[3-8]->VARIADIC[0-3]',  'ai.onnx',  'COMMON',  22,  22,  8,  3,  3,  0,  'no_inference',  True,  True,  True,  True,  True],
        ]
    register_ops('rnn', _optbl)
    return

def register_text_ops():
    _optbl = [
        ['RegexFullMatch',    'ARITY_1->1',  'ai.onnx',  'COMMON',  20,  20,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        ['StringNormalizer',  'ARITY_1->1',  'ai.onnx',  'COMMON',  10,  10,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        ['StringSplit',       'ARITY_1->2',  'ai.onnx',  'COMMON',  20,  20,  1,  1,  2,  2,  'inline_lambda',  True,  True,  True,  True,  True],
        ['StringConcat',      'ARITY_2->1',  'ai.onnx',  'COMMON',  20,  20,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        ]
    register_ops('text', _optbl)
    return

