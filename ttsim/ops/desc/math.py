#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.desc.registry import register_ops
from ttsim.ops.desc.helpers import unary_fwd, bidir_bcast, bidirectional_broadcast_shape_inference

import numpy as np

def variadic_sinf(iTList, oTList, op, **kwargs):
    dim = op.attrs.get('dim', None)
    input_tensor = iTList[0]
    assert input_tensor.check_shape(), f"Illegal Shape for {input_tensor}"

    rank = input_tensor.rank()
    if dim is not None and dim < 0:
        dim += rank

    if dim is None:
        # operate on all elements, output is scalar
        output_shape = []
    else:
        assert 0 <= dim < rank, f"dim {dim} out of bounds for rank {rank}"
        # Output shape is input shape with dim removed
        output_shape = [s for i, s in enumerate(input_tensor.shape) if i != dim]

    oTList[0].shape = output_shape
    oTList[0].dtype = input_tensor.dtype

    i_elems = input_tensor.nelems()
    o_elems = oTList[0].nelems()
    optype2instr = {
            'mean': {'add': i_elems, 'div': o_elems},
            'sum' : {'add': i_elems},
            'max' : {'cmp': i_elems - o_elems},
            'min' : {'cmp': i_elems - o_elems},
            }
    op.perf_stats = {
        'inElems': i_elems,
        'inBytes': input_tensor.nbytes(op.precision),
        'outElems': o_elems,
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': optype2instr[op.optype.lower()],
    }
    return

def matmul_shape_inf(iTList, oTList, op, **kwargs):
    A, B = iTList[0], iTList[1]
    assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
    assert B.check_shape(), f"Input tensor-B shape not defined: {B}"

    AShape, BShape, CShape = A.shape, B.shape, None
    if len(AShape) < 1 or len(BShape) < 1:
        raise ValueError("Shapes must have at least 1 dimension")

    # Handle 1D cases
    # Note: 1D cases must be handled separately and return early to avoid executing
    # the 2D+ code path, which would try to access mat2[-2] on 1-element lists,
    # causing IndexError. Additionally, we must set reduced_dim here since mat1/mat2
    # are not defined for 1D cases.
    if len(AShape) == 1 and len(BShape) == 1:
        # Vector-Vector: [n] × [n] -> [] (scalar result)
        if AShape[0] != BShape[0]:
            raise ValueError(f"Matmul incompatible: {AShape[0]} != {BShape[0]}")
        CShape = [] # Scalar result
        reduced_dim = AShape[0]  # The dimension being reduced
    elif len(AShape) == 1:
        # Vector-Matrix: [n] × [..., n, m] -> [..., m]
        if AShape[0] != BShape[-2]:
            raise ValueError(f"Matmul incompatible: {AShape[0]} != {BShape[-2]}")
        CShape = BShape[:-2] + [BShape[-1]]
        reduced_dim = AShape[0]  # The vector length (dimension being reduced)
    elif len(BShape) == 1:
        # Matrix-Vector: [..., m, n] × [n] -> [..., m]
        if AShape[-1] != BShape[0]:
            raise ValueError(f"Matmul incompatible: {AShape[-1]} != {BShape[0]}")
        CShape = AShape[:-1]
        reduced_dim = AShape[-1]  # The last dimension of the matrix (dimension being reduced)
    else:
        # Handle 2D+ cases
        # For 2D and higher-dimensional tensors, split into batch and matrix dimensions
        batch1, mat1 = AShape[:-2], AShape[-2:]
        batch2, mat2 = BShape[:-2], BShape[-2:]

        # Check matrix multiplication compatibility: inner dimensions must match
        if mat1[-1] != mat2[-2]:
            raise ValueError(f"Matmul incompatible: {mat1[-1]} != {mat2[-2]}")
        # Broadcast batch dimensions and compute output shape
        broadcast_batch = bidirectional_broadcast_shape_inference(batch1, batch2)
        CShape = broadcast_batch + [mat1[0], mat2[-1]]
        reduced_dim = mat1[-1]  # The inner dimension being reduced
    oTList[0].shape = CShape
    oTList[0].dtype = A.dtype
    op.perf_stats = {
            'inElems' : iTList[0].nelems() + iTList[1].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mac': oTList[0].nelems() * reduced_dim}
            }
    return

def grid_sample_inf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    B = iTList[1]
    assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
    assert B.check_shape(), f"Input tensor-B shape not defined: {B}"
    AShape = iTList[0].shape
    BShape = iTList[1].shape
    if len(AShape) != 4:
        raise ValueError(f"GridSample expects 4D input tensor-A shape; got {AShape}")
    if len(BShape) != 4:
        raise ValueError(f"GridSample expects 4D input tensor-B shape; got {BShape}")
    N, C, H, W = AShape
    N_grid, H_out, W_out, _ = BShape
    if N != N_grid:
        raise ValueError(f"GridSample batch-size mismatch: {N} != {N_grid}")
    C_out = C
    oTList[0].shape = [N, C_out, H_out, W_out]
    oTList[0].dtype = A.dtype

    mode = op.attrs.get('mode', 'bilinear')
    nElem_out = N * C_out * H_out * W_out
    instr_count = {}
    if mode == 'nearest':
        instr_count['mov'] = nElem_out
    elif mode == 'bilinear':
        instr_count['mul'] = nElem_out * 4
        instr_count['add'] = nElem_out * 3
    elif mode == 'bicubic':
        instr_count['mul'] = nElem_out * 16
        instr_count['add'] = nElem_out * 15
    else:
        raise ValueError(f"Unsupported GridSample mode: {mode}")

    op.perf_stats = {
            'inElems' : iTList[0].nelems() + iTList[1].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mov': oTList[0].nelems()} #TODO: refine this
            }
    return

def expand_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    AShape = iTList[0].shape

    if iTList[1].data is None:
        target_shape = AShape
    else:
        B = iTList[1].clone_by_shape(data_maybe_missing=False) #B.data should exist
        assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
        assert B.check_shape(), f"Input tensor-B shape not defined: {B}"
        assert B.dtype == np.int64, f"Input Data-Type should be np.int64 {B}"
        target_shape = [x.item() for x in B.data]

    CShape       = bidirectional_broadcast_shape_inference(AShape, target_shape)
    oTList[0].shape = CShape
    oTList[0].dtype = A.dtype
    op.perf_stats = {
            'inElems' : iTList[0].nelems() + iTList[1].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mov': oTList[0].nelems()}
            }
    return

def det_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
    AShape = iTList[0].shape
    ARank  = iTList[0].rank()
    if ARank < 2:
        raise ValueError("Det expects at least 2D input of shape [*, M, M]")

    if AShape[-2] != AShape[-1]:
        raise ValueError(f"Det expects [*,M,M] tensor shapes; got {AShape}")

    M = AShape[-1]
    oTList[0].shape = AShape[:-2]
    oTList[0].dtype = A.dtype
    op.perf_stats = {
            'inElems' : iTList[0].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mac': M**3/3, 'div': M**2/2}, #assume LU based algo for determinant
            }
    return

def data_window_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0].clone_by_shape(data_maybe_missing=False)
    if iTList[0].rank() != 0:
        raise ValueError("data windows expect a scalar as input: {A}")
    dtype = op.attrs['output_datatype']
    oTList[0].shape = [x.item() for x in A.data]
    oTList[0].dtype = op.precision
    op.perf_stats = {
            'inElems' : iTList[0].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mov': 0} #TODO: fix this for HammingWindow, HannWindow, BlackmanWindow costs
            }
    return

def topk_sinf(iTList, oTList, op, **kwargs):
    X = iTList[0]
    K = iTList[1].clone_by_shape(data_maybe_missing=False) #int64
    assert X.check_shape(), f"Input tensor-X shape not defined: {X}"
    assert K.dtype == np.int64, f"Input tensor-K Data-Type should be np.int64 {K}"
    XShape = X.shape
    XRank  = X.rank()
    _axis   : int = op.attrs.get('axis',   -1)
    _largest: int = op.attrs.get('largest', 1)
    _sorted : int = op.attrs.get('sorted',  1)

    if XRank < 1:
        raise ValueError("TopK expects input of rank >= 1: {X}")

    if _axis < 0: #type: ignore
        _axis = XRank + _axis #type: ignore

    if _axis < 0 or _axis >= XRank:
        raise ValueError(f"Axis {_axis} is out of bounds for X {X}")

    outshape = XShape
    d_axis   = XShape[_axis]
    k_value  = [x.item() for x in K.data]
    assert len(k_value) == 1, f"TopK requires K-Tensor should be 1D with a single scalar value"
    k_scalar_value = k_value[0]
    if k_scalar_value < 0:
        raise ValueError(f"TopK requires K value > 0: {k_scalar_value}")
    if k_scalar_value > d_axis:
        raise ValueError(f"TopK requires K value({k_scalar_value}) < dim(axis) = {d_axis}")
    outshape[_axis] = k_scalar_value

    oTList[0].shape = outshape
    oTList[1].shape = outshape
    oTList[0].dtype = X.dtype
    oTList[1].dtype = np.dtype(np.int64)
    op.perf_stats = {
            'inElems' : iTList[0].nelems() + iTList[1].nelems(),
            'outElems': oTList[0].nelems() + oTList[1].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision) + oTList[1].nbytes(op.precision),
            'instrs'  : {'mov': 0} #TODO: fix this for TopK
            }
    return

def nonzero_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
    AShape = A.shape
    AElems = A.nelems()
    nonzero_count = AElems // 2  # Assume half the elements are non-zero for perf estimation

    rank = A.rank()
    out_shape = AShape[:-1] + [nonzero_count] # Keep all dimensions except last one, append nonzero_count
    oTList[0].shape = out_shape
    oTList[0].dtype = np.dtype(np.int64)

    op.perf_stats = {
            'inElems' : iTList[0].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'cmp': AElems, 'mov': nonzero_count * rank}
            }
    return

def gemm_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    B = iTList[1]
    assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
    assert B.check_shape(), f"Input tensor-B shape not defined: {B}"
    transA = op.attrs.get('transA', 0)
    transB = op.attrs.get('transB', 0)
    AShape, BShape, CShape = A.shape, B.shape, None
    if transA:
        AShape = AShape[:-2] + [AShape[-1], AShape[-2]]
    if transB:
        BShape = BShape[:-2] + [BShape[-1], BShape[-2]]

    if len(AShape) < 2 or len(BShape) < 2:
        raise ValueError("Shapes must have at least 2 dimensions")

    batch1, mat1 = AShape[:-2], AShape[-2:]
    batch2, mat2 = BShape[:-2], BShape[-2:]

    # Check matrix multiplication compatibility
    if mat1[-1] != mat2[-2]:
        raise ValueError(f"Gemm incompatible: {mat1[-1]} != {mat2[-2]}")
    broadcast_batch = bidirectional_broadcast_shape_inference(batch1, batch2)
    CShape = broadcast_batch + [mat1[-2], mat2[-1]]

    reduced_dim     = mat1[-1]
    oTList[0].shape = CShape
    oTList[0].dtype = A.dtype
    op.perf_stats = {
            'inElems' : iTList[0].nelems() + iTList[1].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mac': oTList[0].nelems() * reduced_dim}
            }
    return

def register_math_ops():
    _xoptbl = [
            ['Clip', 'ARITY_VARIADIC[1-3]->1', 'ai.onnx', 'COMMON',  13,  13,  3,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['TopK',   'ARITY_2->2', 'ai.onnx', 'COMMON',  24,  11,  2,  2, 2,  2, topk_sinf, True,  True,  True,  True,  True],
            #complex shape inference
            #['NegativeLogLikelihoodLoss', 'ARITY_VARIADIC[2-3]->1',              'ai.onnx', 'COMMON',  22,  22,  3,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            #['SoftmaxCrossEntropyLoss',   'ARITY_VARIADIC[2-3]->VARIADIC[1-2]',  'ai.onnx', 'COMMON',  13,  13,  3,  2,  2,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            #['Einsum',           'ARITY_VARIADIC[1-*]->1', 'ai.onnx', 'COMMON',  12,  12,  2147483647,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            #['DFT',              'ARITY_VARIADIC[1-3]->1', 'ai.onnx', 'COMMON',  20,  20,  3,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ['Gemm',             'ARITY_VARIADIC[2-3]->1', 'ai.onnx', 'COMMON',  13,  13,  3,  2,  1,  1,  gemm_sinf,  True,  True,  True,  True,  True],
            #['MatMulInteger',    'ARITY_VARIADIC[2-4]->1', 'ai.onnx', 'COMMON',  10,  10,  4,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            #['STFT',             'ARITY_VARIADIC[2-4]->1', 'ai.onnx', 'COMMON',  17,  17,  4,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            #['QLinearMatMul',    'ARITY_8->1',             'ai.onnx', 'COMMON',  21,  21,  8,  8,  1,  1,  'QLinearMatMulShapeInference',  True,  True,  True,  True,  True],
            #['MelWeightMatrix',  'ARITY_5->1',             'ai.onnx', 'COMMON',  17,  17,  5,  5,  1,  1,  'inline_lambda', True,  True,  True,  True,  True],
            ]

    _binary_optbl = [
            ['Add',    'ARITY_2->1', 'ai.onnx', 'COMMON',  14,  14,  2,  2, 1,  1,  bidir_bcast,      True,  True,  True,  True,  True],
            ['Sub',    'ARITY_2->1', 'ai.onnx', 'COMMON',  14,  14,  2,  2, 1,  1,  bidir_bcast,      True,  True,  True,  True,  True],
            ['Mul',    'ARITY_2->1', 'ai.onnx', 'COMMON',  14,  14,  2,  2, 1,  1,  bidir_bcast,      True,  True,  True,  True,  True],
            ['Div',    'ARITY_2->1', 'ai.onnx', 'COMMON',  14,  14,  2,  2, 1,  1,  bidir_bcast,      True,  True,  True,  True,  True],
            ['Pow',    'ARITY_2->1', 'ai.onnx', 'COMMON',  15,  15,  2,  2, 1,  1,  bidir_bcast,      True,  True,  True,  True,  True],
            ['Mod',    'ARITY_2->1', 'ai.onnx', 'COMMON',  13,  13,  2,  2, 1,  1,  bidir_bcast ,     True,  True,  True,  True,  True],
            ['Expand', 'ARITY_2->1', 'ai.onnx', 'COMMON',  13,  13,  2,  2, 1,  1,  expand_sinf,      True,  True,  True,  True,  True],
            ['CumSum', 'ARITY_2->1', 'ai.onnx', 'COMMON',  14,  14,  2,  2, 1,  1,  unary_fwd,        True,  True,  True,  True,  True],
            ['PRelu',  'ARITY_2->1', 'ai.onnx', 'COMMON',  16,  16,  2,  2, 1,  1,  unary_fwd,        True,  True,  True,  True,  True],
            ['MatMul', 'ARITY_2->1', 'ai.onnx', 'COMMON',  13,  13,  2,  2, 1,  1,  matmul_shape_inf, True,  True,  True,  True,  True],
            ['GridSample', 'ARITY_2->1', 'ai.onnx', 'COMMON',  22,  22,  2,  2, 1,  1,  grid_sample_inf,        True,  True,  True,  True,  True],
            ]

    _variadic_input_unary_output_optbl = [
            ['Max',  'ARITY_VARIADIC[1-*]->1', 'ai.onnx', 'COMMON',  13,  13, 2147483647,  1,  1, 1,  variadic_sinf,  True,  True,  True,  True,  True],
            ['Min',  'ARITY_VARIADIC[1-*]->1', 'ai.onnx', 'COMMON',  13,  13, 2147483647,  1,  1, 1,  variadic_sinf,  True,  True,  True,  True,  True],
            ['Sum',  'ARITY_VARIADIC[1-*]->1', 'ai.onnx', 'COMMON',  13,  13, 2147483647,  1,  1, 1,  variadic_sinf,  True,  True,  True,  True,  True],
            ['Mean', 'ARITY_VARIADIC[1-*]->1', 'ai.onnx', 'COMMON',  13,  13, 2147483647,  1,  1, 1,  variadic_sinf,  True,  True,  True,  True,  True],
            ]

    _unary_optbl = [
            ['Softmax',          'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['LogSoftmax',       'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Hardmax',          'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Neg',              'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Abs',              'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Reciprocal',       'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Floor',            'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Ceil',             'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Sqrt',             'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Relu',             'ARITY_1->1', 'ai.onnx', 'COMMON',  14,  14,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['LeakyRelu',        'ARITY_1->1', 'ai.onnx', 'COMMON',  16,  16,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['ThresholdedRelu',  'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Selu',             'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Elu',              'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Mish',             'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Celu',             'ARITY_1->1', 'ai.onnx', 'COMMON',  12,  12,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Gelu',             'ARITY_1->1', 'ai.onnx', 'COMMON',  20,  20,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Exp',              'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Log',              'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Tanh',             'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Sigmoid',          'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['HardSigmoid',      'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['HardSwish',        'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Softsign',         'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Softplus',         'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Sin',              'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Cos',              'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Tan',              'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Asin',             'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Acos',             'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Atan',             'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Sinh',             'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Cosh',             'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Asinh',            'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Acosh',            'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Atanh',            'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Sign',             'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Erf',              'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Round',            'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],
            ['Det',              'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1,  det_sinf,   True,  True,  True,  True,  True],
            ['NonZero',          'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1,  nonzero_sinf,  True,  True,  True,  True,  True],

            ['HannWindow',       'ARITY_1->1', 'ai.onnx', 'COMMON',  17,  17,  1,  1,  1,  1, data_window_sinf,  True,  True,  True,  True,  True],
            ['HammingWindow',    'ARITY_1->1', 'ai.onnx', 'COMMON',  17,  17,  1,  1,  1,  1, data_window_sinf,  True,  True,  True,  True,  True],
            ['BlackmanWindow',   'ARITY_1->1', 'ai.onnx', 'COMMON',  17,  17,  1,  1,  1,  1, data_window_sinf,  True,  True,  True,  True,  True],
            ]

    _optbl = _xoptbl + _binary_optbl + _variadic_input_unary_output_optbl + _unary_optbl
    register_ops('math', _optbl)
    return
