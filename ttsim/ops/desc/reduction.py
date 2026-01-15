#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .registry import register_ops
import numpy as np

def r0_func(iTList, oTList, op, **kwargs):
    #select_last_index = op.attrs.get('select_last_index', 0) -- has no effect on outshape!!
    keepdims = op.attrs.get('keepdims', 1)
    axis     = op.attrs.get('axis',     0)
    dataT    = iTList[0].clone_by_shape()
    rank     = dataT.rank()
    if axis < 0: axis += rank
    assert (0 <= axis < rank), f"Arg axis {axis} out of range for rank {rank}"

    if keepdims:
        outShape = [i for i in dataT.shape]
        outShape[axis] = 1
    else:
        outShape = [d for i,d in enumerate(dataT.shape) if i != axis]

    oTList[0].shape = outShape
    #oTList[0].dtype = np.dtype(np.int64)
    #create dummy index data so that it can work downstream...
    oTList[0].data  = np.array([0 for i in range(oTList[0].nelems())], dtype=np.int64)
    oTList[0].dtype = oTList[0].data.dtype
    op.perf_stats = {
            'inElems' : dataT.nelems(),
            'inBytes' : dataT.nbytes(op.precision),
            'outElems': oTList[0].nelems(),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'cmp': dataT.nelems()}
            }
    return

def r1_func(iTList, oTList, op, **kwargs):
    keepdims = op.attrs.get('keepdims', 1)
    noop     = op.attrs.get('noop_with_empty_axes', 0)
    dataT    = iTList[0].clone_by_shape()
    axesT    = iTList[1].clone_by_shape(data_maybe_missing=False) if len(iTList) == 2 else None
    rank     = dataT.rank()
    if axesT is None:
        if noop:
            outShape = dataT.shape
            reduce_axes = None
        else:
            reduce_axes = [i for i in range(rank)]
    else:
        reduce_axes = [i for i in axesT.data]

    if reduce_axes:
        normalized_axes = []
        for a in reduce_axes:
            if a < 0: a += rank
            assert (0 <= a < rank), f"reduce axis {a} out of range for rank {rank}"
            normalized_axes.append(a)
        axes_set = sorted(set(normalized_axes))

        if keepdims:
            outShape = [1 if i in axes_set else d for i,d in enumerate(dataT.shape)]
        else:
            outShape = [d for i,d in enumerate(dataT.shape) if i not in axes_set]


    inelems = dataT.nelems()
    instr_profile = {
            'ReduceL1'        : {'abs': inelems, 'add': inelems},
            'ReduceL2'        : {'mul': inelems, 'add': inelems, 'sqrt': 1},
            'ReduceLogSum'    : {'log': inelems, 'add': inelems},
            'ReduceLogSumExp' : {'log': inelems, 'add': inelems, 'exp': 1},
            'ReduceMax'       : {'cmp': inelems},
            'ReduceMean'      : {'add': inelems, 'div': 1},
            'ReduceMin'       : {'cmp': inelems},
            'ReduceProd'      : {'mul': inelems},
            'ReduceSum'       : {'add': inelems},
            'ReduceSumSquare' : {'mul': inelems, 'add': inelems},
            }

    oTList[0].shape = outShape
    oTList[0].dtype = dataT.dtype
    op.perf_stats = {
            'inElems' : sum([x.nelems() for x in iTList]),
            'inBytes' : sum([x.nbytes(op.precision) for x in iTList]),
            'outElems': oTList[0].nelems(),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : instr_profile[op.optype]
            }
    return

def register_reduction_ops():
    _optbl = [
            ['ArgMax',          'ARITY_1->1',             'ai.onnx', 'COMMON', 13, 13, 1, 1, 1, 1, r0_func, True,  True,  True,  True, True],
            ['ArgMin',          'ARITY_1->1',             'ai.onnx', 'COMMON', 13, 13, 1, 1, 1, 1, r0_func, True,  True,  True,  True, True],

            ['ReduceL1',        'ARITY_VARIADIC[1-2]->1', 'ai.onnx', 'COMMON', 18, 18, 2, 1, 1, 1, r1_func, True,  True,  True,  True, True],
            ['ReduceL2',        'ARITY_VARIADIC[1-2]->1', 'ai.onnx', 'COMMON', 18, 18, 2, 1, 1, 1, r1_func, True,  True,  True,  True, True],
            ['ReduceLogSum',    'ARITY_VARIADIC[1-2]->1', 'ai.onnx', 'COMMON', 18, 18, 2, 1, 1, 1, r1_func, True,  True,  True,  True, True],
            ['ReduceLogSumExp', 'ARITY_VARIADIC[1-2]->1', 'ai.onnx', 'COMMON', 18, 18, 2, 1, 1, 1, r1_func, True,  True,  True,  True, True],
            ['ReduceMax',       'ARITY_VARIADIC[1-2]->1', 'ai.onnx', 'COMMON', 20, 20, 2, 1, 1, 1, r1_func, True,  True,  True,  True, True],
            ['ReduceMin',       'ARITY_VARIADIC[1-2]->1', 'ai.onnx', 'COMMON', 20, 20, 2, 1, 1, 1, r1_func, True,  True,  True,  True, True],
            ['ReduceMean',      'ARITY_VARIADIC[1-2]->1', 'ai.onnx', 'COMMON', 18, 18, 2, 1, 1, 1, r1_func, True,  True,  True,  True, True],
            ['ReduceProd',      'ARITY_VARIADIC[1-2]->1', 'ai.onnx', 'COMMON', 18, 18, 2, 1, 1, 1, r1_func, True,  True,  True,  True, True],
            ['ReduceSum',       'ARITY_VARIADIC[1-2]->1', 'ai.onnx', 'COMMON', 13, 13, 2, 1, 1, 1, r1_func, True,  True,  True,  True, True],
            ['ReduceSumSquare', 'ARITY_VARIADIC[1-2]->1', 'ai.onnx', 'COMMON', 18, 18, 2, 1, 1, 1, r1_func, True,  True,  True,  True, True],
            ]

    register_ops('reduction', _optbl)
    return
