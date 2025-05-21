#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import LayerNormalizationOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

# Layer normalization's reference implementation
def _layer_normalization(X, W, B, axis=-1, epsilon=1e-5):  # type: ignore
    X_shape = X.shape
    X_rank = len(X_shape)
    if axis < 0:
        # If axis = -1 and rank of X is 4,
        # the axis is changed to -1 + 4 = 3,
        # which means the last axis.
        axis = axis + X_rank
    unsqueezed_rank = X_rank - axis
    reduction_shape = X_shape[0:axis] + (1,) * unsqueezed_rank

    # Parameter used to convert N-D tensor layer
    # normalization to equivalent 2-D matirx operations.
    row_number = 1
    col_number = 1
    for i in range(X_rank):
        if i < axis:
            row_number *= X_shape[i]
        else:
            col_number *= X_shape[i]

    # After reshaping input tensor X into a matrix,
    # layer normalization is equivalent to conducting
    # standardization on each column vector (s.t. each
    # column has zero mean and unit variance).
    x_mat = np.reshape(X, (row_number, col_number))
    # This computes mean for every x_mat's column.
    x_mean = np.sum(x_mat, axis=1, keepdims=True) / col_number
    x_diff = x_mat - x_mean
    x_squared_diff = x_diff * x_diff
    # This computes variance for every x_mat's column.
    variance = np.sum(x_squared_diff, axis=1, keepdims=True) / col_number
    variance_eps = variance + epsilon
    std_dev = np.sqrt(variance_eps)
    inv_std_dev = np.reciprocal(std_dev)
    # Standardization step. y_mat is zero-mean and unit-variance.
    y_mat = x_diff * inv_std_dev
    # Apply affine transform on normalization outcome.
    # W is linear coefficient while B is bias.
    Y = np.reshape(y_mat, X_shape) * W + B
    # Matrix-level operations' outputs should be reshaped
    # to compensate the initial tensor-to-matrix reshape.
    X_mean = np.reshape(x_mean, reduction_shape)
    X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)

    return Y, X_mean, X_inv_std_dev

def calculate_normalized_shape(X_shape, axis):  # type: ignore
    X_rank = len(X_shape)
    if axis < 0:
        axis = axis + X_rank
    return X_shape[axis:]

# Test cases
test_name  = 'test_layernorm'
test_cases = [
        {
            'name': f"test_layer_normalization_4d",
            'x'   : [2, 3, 4, 5],
            'in'  : ["X", "W", "B"],
            'out' : ["Y", "Mean", "InvStdDev"],
            },
        {
            'name': "test_layer_normalization_default_axis",
            'x'   : [2, 3, 4, 5],
            'in'  : ["X", "W", "B"],
            'out' : ["Y", "Mean", "InvStdDev"],
            },
        {
            'name': "test_layer_normalization_2d",
            'x'   : [3, 4],
            'in'  : ["X", "W", "B"],
            'out' : ["Y", "Mean", "InvStdDev"],
            },
        {
            'name': f"test_layer_normalization_3d_epsilon",
            'x'   : [2, 3, 5],
            'in'  : ["X", "W", "B"],
            'out' : ["Y", "Mean", "InvStdDev"],
            'eps' : 1e-1,
            },
]

@pytest.mark.unit
@pytest.mark.opunit
def test_layernorm():
    for trec in test_cases:
        tname = trec['name'] #type: ignore
        if tname.endswith('default_axis'):
            axes = [-1]
            names = [tname]
        else:
            xrank = len(trec['x']) #type: ignore
            axes  = [i for i in range(xrank)]
            axes += [i - xrank for i in range(xrank)]
            names = [f'{tname}_neg_axis_{-a}' if a < 0 else f'{tname}_axis_{a}' for a in axes]
        trec['axes']  = axes #type: ignore
        trec['names'] = names #type: ignore
    msgw = max([len(y) for x in test_cases for y in x['names']]) #type: ignore
    for tno, trec in enumerate(test_cases):
        for cno, axis in enumerate(trec['axes']): #type: ignore
            test_name = trec['names'][cno] #type: ignore
            op_name = f'{test_name}_{tno}_{cno}'

            XShape = trec['x'] #type: ignore
            normalized_shape = calculate_normalized_shape(XShape, axis)
            X = np.random.randn(*XShape).astype(np.float32)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            B = np.random.randn(*normalized_shape).astype(np.float32)
            attrs = {'axis': axis}
            if 'eps' in trec: #type: ignore
                eps = trec['eps'] #type: ignore
                attrs['epsilon'] = eps
                Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis, eps)
            else:
                Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis)
            o0Shape = list(Y.shape)
            o1Shape = list(mean.shape)
            o2Shape = list(inv_std_dev.shape)

            i_tensors = [
                    F._from_shape('X', XShape,           np_dtype=np.float32), #data
                    F._from_shape('W', normalized_shape, np_dtype=np.float32), #scale
                    F._from_shape('B', normalized_shape, np_dtype=np.float32), #bias
                    ]
            o_tensors = [ make_tensor('Y'), make_tensor('mean'), make_tensor('inv_std_dev')]
            op_info = {
                    'name'   : op_name,
                    'optype' : 'LayerNorm',
                    'inList' : [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors],
                    'attrs'  : attrs,
                    }
            op_obj = LayerNormalizationOp(op_info)
            for x in i_tensors: x.op_in  = [op_name]
            for x in o_tensors: x.op_out = [op_name]
            op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

            assert o_tensors[0].shape == o0Shape, f"Y shape mismatch: {o_tensors[0].shape} != {o0Shape}"
            assert o_tensors[1].shape == o1Shape, f"mean shape mismatch: {o_tensors[1].shape} != {o1Shape}"
            assert o_tensors[2].shape == o2Shape, f"inv_std_dev shape mismatch: {o_tensors[2].shape} != {o2Shape}"

            eps_str = f"{eps:.2f}" if 'eps' in trec else '-' #type: ignore
            print(f"TEST[{tno:3d}] CASE[{cno:4d}] {test_name:{msgw}s} axis={axis:3d} eps={eps_str} PASS")
