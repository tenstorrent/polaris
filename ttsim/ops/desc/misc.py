#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .registry import register_ops

def register_controlflow_ops():
    _optbl = [
        ['If',   'ARITY_1->VARIADIC[1-*]',             'ai.onnx', 'COMMON',  24,  21,  1,           1,  2147483647,  1,  'IfInferenceFunction',   True,  True,  True,  True,  True],
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

import numpy as np

def quantize_linear_sinf(iTList, oTList, op, **kwargs):
    x = iTList[0]
    assert x.check_shape(), f"Illegal input shape: {x}"

    if not np.issubdtype(x.dtype, np.floating):
         raise AssertionError(f"Input must be float, got {x.dtype}")
    
    # Check axis
    axis = op.attrs.get('axis', 1)
    rank = x.rank()
    if axis < -rank or axis >= rank:
        raise ValueError(f"Invalid axis {axis} for rank {rank}")
    
    oTList[0].shape = x.shape
    # Default to uint8, or derived from zero_point if present
    oTList[0].dtype = np.dtype(np.uint8)
    if len(iTList) >= 3 and iTList[2].dtype is not None:
        oTList[0].dtype = iTList[2].dtype
        
    instr_count = {
        'div': x.nelems(),
        'round': x.nelems(),
        'add': x.nelems(),
        'cast': x.nelems(),
        'clip': 0
    }
    if op.attrs.get('saturate', 1):
        instr_count['clip'] = x.nelems()
    
    op.perf_stats = {
        'inElems': sum(t.nelems() for t in iTList),
        'outElems': oTList[0].nelems(),
        'inBytes': sum(t.nbytes(op.precision) for t in iTList),
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': instr_count
    }
    return

def dequantize_linear_sinf(iTList, oTList, op, **kwargs):
    x = iTList[0]
    assert x.check_shape()

    if not np.issubdtype(x.dtype, np.integer):
         raise AssertionError(f"Input must be integer, got {x.dtype}")

    # Check axis
    axis = op.attrs.get('axis', 1)
    rank = x.rank()
    if axis < -rank or axis >= rank:
        raise ValueError(f"Invalid axis {axis} for rank {rank}")

    oTList[0].shape = x.shape
    oTList[0].dtype = np.dtype(np.float32)
    
    instr_count = {
        'sub': x.nelems(),
        'mul': x.nelems(),
        'cast': x.nelems()
    }
    
    op.perf_stats = {
        'inElems': sum(t.nelems() for t in iTList),
        'outElems': oTList[0].nelems(),
        'inBytes': sum(t.nbytes(op.precision) for t in iTList),
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': instr_count
    }
    return

def dynamic_quantize_linear_sinf(iTList, oTList, op, **kwargs):
    x = iTList[0]
    assert x.check_shape()

    if not np.issubdtype(x.dtype, np.floating):
         raise AssertionError(f"Input must be float, got {x.dtype}")
    
    # Output 0: quantized data (uint8)
    oTList[0].shape = x.shape
    oTList[0].dtype = np.dtype(np.uint8)
    
    # Output 1: scale (scalar float32)
    oTList[1].shape = []
    oTList[1].dtype = np.dtype(np.float32)
    
    # Output 2: zero_point (scalar uint8)
    oTList[2].shape = []
    oTList[2].dtype = np.dtype(np.uint8)
    
    instr_count = {
        'cmp': x.nelems(),  # min/max
        'sub': x.nelems(),
        'div': x.nelems(),
        'round': x.nelems(),
        'add': x.nelems(),
        'clip': x.nelems(),
        'cast': x.nelems(),
        'mac': 0
    }
    
    op.perf_stats = {
        'inElems': sum(t.nelems() for t in iTList),
        'outElems': sum(t.nelems() for t in oTList),
        'inBytes': sum(t.nbytes(op.precision) for t in iTList),
        'outBytes': sum(t.nbytes(op.precision) for t in oTList),
        'instrs': instr_count
    }
    return

def register_quantization_ops():
    _optbl = [
        ['DynamicQuantizeLinear', 'ARITY_1->3',              'ai.onnx',  'COMMON',  11,  11,  1,  1,  3,  3,  dynamic_quantize_linear_sinf,  True,  True,  True,  True,  True],
        ['QuantizeLinear',        'ARITY_VARIADIC[2-3]->1',  'ai.onnx',  'COMMON',  24,  21,  3,  2,  1,  1,  quantize_linear_sinf,  True,  True,  True,  True,  True],
        ['DequantizeLinear',      'ARITY_VARIADIC[2-3]->1',  'ai.onnx',  'COMMON',  24,  21,  3,  2,  1,  1,  dequantize_linear_sinf,  True,  True,  True,  True,  True],
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

