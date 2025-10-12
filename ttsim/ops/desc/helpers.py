#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.tensor import SimTensor

import math
from loguru import logger
LOG   = logger
INFO  = LOG.info
DEBUG = LOG.debug

def update_output_tensor(op, in_tensor, out_tensor):
    assert in_tensor.check_shape(), f"ERROR: {op} Invalid Input SHAPE in {in_tensor}"
    if out_tensor.check_shape():
        DEBUG("Validated SimTensor({}) SHAPE: {}", out_tensor.name, out_tensor.shape)
        assert in_tensor.shape == out_tensor.shape, f"IO shape Mismatch {in_tensor.shape} != {out_tensor.shape} for {out_tensor.name}"
    else:
        DEBUG("Updating SimTensor({}) SHAPE: {} <- {}", out_tensor.name, out_tensor.shape, in_tensor.shape)
        out_tensor.shape = in_tensor.shape

    if in_tensor.data is not None:
        if out_tensor.data is None:
            out_tensor.data = in_tensor.data
            out_tensor.dtype = in_tensor.dtype
            DEBUG("Updating DATA SimTensor({})", out_tensor)

def build_tmp_data_tensor(data, name):
    return SimTensor({
        'name' : name,
        'shape': list(data.shape),
        'dtype': data.dtype,
        'data' : data,
        'resolve': '_',
        'op_in': [],
        'op_out': [],
        })

def bidirectional_broadcast_shape_inference(shape1, shape2):
    """
    more robust than multidirectional_broadcast_shape_inference
    can handle numpy like broadcasting with [1,0] and [1]
    """
    max_len = max(len(shape1), len(shape2))
    padded1 = list(shape1[::-1]) + [1] * (max_len - len(shape1))
    padded2 = list(shape2[::-1]) + [1] * (max_len - len(shape2))
    result  = []
    for d1, d2 in zip(padded1, padded2):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} not broadcast-compatible")
    return result[::-1]

def multidirectional_broadcast_shape_inference(shapes):
    """
    cannot handle numpy like broadcasting with [1,0] and [1]
    """
    max_len    = max(len(shape) for shape in shapes)
    num_shapes = len(shapes)

    padded_shapes = []
    for shape in shapes:
        shape_rev = list(shape[::-1])
        shape_rev.extend([1] * (max_len - len(shape)))
        padded_shapes.append(shape_rev)

    result = []
    for i in range(max_len):
        dim_list = [padded_shapes[j][i] for j in range(num_shapes)]
        max_dim  = max(dim_list)
        check = all(d == 1 or d == max_dim for d in dim_list)
        assert check, f"Incompatible shapes at dim: {max_len-1-i}: expect {max_dim} or 1, got {dim_list}"
        result.append(max_dim)

    return result[::-1]

def propagate_shape_and_type(inT, outT, i_idx, o_idx):
    assert len(inT) > i_idx
    assert len(outT) > o_idx
    assert inT[i_idx].check_shape(), f"Illegal Shape for input tensor @ index {i_idx}"
    outT[o_idx].shape = inT[i_idx].shape
    outT[o_idx].dtype = inT[i_idx].dtype
    return

def gelu_instr_counts(nElem):
    """
    ONNX opset-20 defines GELU w/ 2 variants controlled by the attribute 'approximate'
      if approximate = 'tanh', we use GELU (Gaussian Error Linear Unit) approximation:
         Y = 0.5 * X * (1 + tanh(math.sqrt(2 / math.pi) * (X + 0.044715 * pow(X, 3))))
      else (default)
         Y = 0.5 * X * (1 + erf(X/sqrt(2)))
    we assume approximate to be 'tanh' always for now....
    TODO: add default option as well...

    instr count calc.
     Y= <const> * X * ( <const> + tanh( <const> * ( X + <const> * X^3 ) ) )
    """
    mul_count, add_count, tanh_count = 0,0,0
    mul_count  += 2 * nElem # X^3
    mul_count  += nElem     # <const> * X^3
    add_count  += nElem     # X + <const> * X^3
    mul_count  += nElem     # <const> * ( X + ...)
    tanh_count += nElem     # tanh (...)
    add_count  += nElem     # <const + tanh(...)
    mul_count  += 2*nElem   # <const> * X * (...)
    return {'mul': mul_count, 'add': add_count, 'tanh': tanh_count}

def pooling_shape_inference(input_shape, kernel_shape, attrs):
    """Shape inference for pooling operators"""

    # Validate inputs
    if len(input_shape) < 2:
        raise ValueError(f"Expected at least 2D input tensor, got shape {input_shape}")

    num_spatial_dims = len(kernel_shape)
    if num_spatial_dims > len(input_shape) - 2:
        raise ValueError(f"Too many spatial dimensions ({num_spatial_dims}) for input shape {input_shape}")

    auto_pad      = attrs.get('auto_pad',      'NOTSET')
    ceil_mode     = attrs.get('ceil_mode',     0)
    dilations     = attrs.get('dilations',     [1] * num_spatial_dims)
    pads          = attrs.get('pads',          [0] * (2 * num_spatial_dims))
    storage_order = attrs.get('storage_order', 0)
    strides       = attrs.get('strides',       [1] * num_spatial_dims)

    # Extract spatial dimensions (assume last num_spatial_dims are spatial)
    non_spatial_dims = input_shape[:-num_spatial_dims]
    spatial_dims     = input_shape[-num_spatial_dims:]
    if len(spatial_dims) != num_spatial_dims:
        raise ValueError(f"Expected {num_spatial_dims} spatial dimensions, got {spatial_dims}")

    # Handle padding
    if pads is not None:
        if len(pads) != 2 * num_spatial_dims:
            raise ValueError(f"Expected pads length 2 * {num_spatial_dims}, got {pads}")
        pad_before = pads[:num_spatial_dims]
        pad_after = pads[num_spatial_dims:]
    else:
        if auto_pad == "VALID":
            pad_before = [0] * num_spatial_dims
            pad_after = [0] * num_spatial_dims
        elif auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            pad_before = []
            pad_after = []
            for i in range(num_spatial_dims):
                # Effective kernel size with dilation
                effective_kernel_size = (kernel_shape[i] - 1) * dilations[i] + 1
                # For SAME padding, output size is ceil(input_size / stride)
                out_size = math.ceil(spatial_dims[i] / strides[i])
                # Compute total padding needed
                pad_total = max((out_size - 1) * strides[i] + effective_kernel_size - spatial_dims[i], 0)
                # Distribute padding
                if auto_pad == "SAME_UPPER":
                    pad_b = pad_total // 2
                    pad_a = pad_total - pad_b
                else:  # SAME_LOWER
                    pad_a = pad_total // 2
                    pad_b = pad_total - pad_a
                pad_before.append(pad_b)
                pad_after.append(pad_a)
        else:  # NOTSET with pads=None
            pad_before = [0] * num_spatial_dims
            pad_after = [0] * num_spatial_dims

    # Compute output spatial dimensions
    output_spatial_dims = []
    for i in range(num_spatial_dims):
        # Compute effective kernel size with dilation
        effective_kernel_size = (kernel_shape[i] - 1) * dilations[i] + 1
        # Compute output size
        padded_size = spatial_dims[i] + pad_before[i] + pad_after[i]
        if ceil_mode == 0:
            out_size = math.floor((padded_size - effective_kernel_size) / strides[i]) + 1
        else:  # ceil_mode == 1
            out_size = math.ceil((padded_size - effective_kernel_size) / strides[i]) + 1
        if out_size <= 0:
            raise ValueError(
                f"Invalid output dimension {i}: size={out_size}. "
                f"Check input shape {input_shape}, kernel {kernel_shape}, "
                f"strides {strides}, pads {pads}, auto_pad {auto_pad}, "
                f"ceil_mode {ceil_mode}, dilations {dilations}."
            )
        output_spatial_dims.append(out_size)

    # Construct output shape
    output_shape = non_spatial_dims + output_spatial_dims
    return output_shape

#shape inference functions
def unary_fwd(iTList, oTList, op, **kwargs):
    X,Y = iTList[0], oTList[0]
    assert X.check_shape(), f"Input tensor shape not defined: {X}"
    Y.shape = X.shape
    Y.dtype = X.dtype
    optype2instr = {
            'Identity': {'mov': 0},
            'Clip'    : {'cmp': 2*Y.nelems(), 'mov': Y.nelems() },
            'CumSum'  : {'add': Y.nelems()}, #TODO: handle axis: we need to divide by X.shape[axis]
            'PRelu'   : {'cmp': Y.nelems(), 'mul': Y.nelems() }, #slope * x if x < 0 else = x

            'BitwiseNot': {'not': Y.nelems()},
            'Not'       : {'not': Y.nelems()},

            #everything below here is math unary operators
            'Softmax': {
                #TODO: take care of the axis
                'cmp': X.nelems(), # max_x = max(x)
                'sub': X.nelems(), # y = x - max_x
                'exp': X.nelems(), # exp(y)
                'add': X.nelems(), # z = sum(exp(y))
                'div': X.nelems(), # o = exp(y) / z
                },
            'LogSoftmax': {
                #TODO: take care of the axis
                #LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))
                #Softmax...
                'cmp': X.nelems(), # max_x = max(x)
                'sub': X.nelems(), # y = x - max_x
                'exp': X.nelems(), # exp(y)
                'add': X.nelems(), # z = sum(exp(y))
                'div': X.nelems(), # o = exp(y) / z
                'log': X.nelems(), # log(o)
                },
            'Gelu': gelu_instr_counts(X.nelems()),
            'Softplus': {
                # Softplus(x) = log(1 + exp(x))
                'exp': X.nelems(), # exp(x)
                'add': X.nelems(), # 1 + exp(x)
                'log': X.nelems(), # log(1 + exp(x))
                'mov': X.nelems(), # for output
                },
            'Sigmoid': {
                #1 / (1 + exp(-x))
                'exp': X.nelems(),
                'add': X.nelems(),
                'div': X.nelems(),
                },
            'Celu': {
                # max(0,x) + min(0,alpha*(exp(x/alpha)-1))
                #      y = max(0,x)
                #      z = x * 1/alpha
                #      z = exp(z)
                #      z = alpha * z
                #      z = z - 1
                #      z = min(0,z)
                #      w = y + z
                'cmp': 2*X.nelems(),
                'mul': 2*X.nelems(),
                'exp': X.nelems(),
                'sub': X.nelems(),
                'add': X.nelems(),
                },
            'Selu': {
                    #y = gamma * (alpha * e^x - alpha) if x <= 0 else gamma * x
                    #assume worst case: x <= 0
                    'exp': X.nelems(),
                    'mul': 2*X.nelems(),
                    'sub': X.nelems(),
                    },
            'Elu': {
                    #f(x) = alpha * (exp(x) - 1.) if x < 0 else  x
                    #we assume the worst case cost (i.e. x < 0)
                    'cmp': X.nelems(), # x < 0
                    'exp': X.nelems(), # y = exp(x)
                    'sub': X.nelems(), # y -= 1
                    'mul': X.nelems(), # y *= alpha
                    },
            'Mish': {
                    #x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
                    'exp' : X.nelems(), # y = exp(x)
                    'add' : X.nelems(), # y = y + 1
                    'log' : X.nelems(), # y = ln(y)
                    'tanh': X.nelems(), # y = tanh(y)
                    'mul' : X.nelems(), # y = x * y
                    },
            'HardSigmoid': {
                    #max(0, min(1, alpha * x + beta))
                    'mul': X.nelems(), # x = alpha * x
                    'add': X.nelems(), # x = x + beta
                    'cmp': 2*X.nelems(), # x = min(1, x); x = max(0, x)
                    },
            'HardSwish': {
                    #x * HardSigmoid<alpha, beta>(x)
                    'mul': 2*X.nelems(), # x = alpha * x; x * HardSigmoid
                    'add': X.nelems(), # x = x + beta
                    'cmp': 2*X.nelems(), # x = min(1, x); x = max(0, x)
                    },
            'Softsign': {
                    #x/(1+|x|) 
                    'abs': X.nelems(),
                    'add': X.nelems(),
                    'div': X.nelems()
                    },
            }

    for xopname in ['Sign', 'Relu', 'LeakyRelu', 'ThresholdedRelu', 'Hardmax']:
        #Sign            : x <=> 0
        #Relu            : max(0,x)
        #LeakyRelu       : alpha * x if x < 0 else x; additional 'add' accounted outside this loop
        #ThresholdedRelu : x for x > alpha else 0
        #Hardmax         : Hardmax(x, axis) = 1 if x == first_max_val_along_xis else 0; TODO: account for axis
        optype2instr[xopname] = {'cmp': X.nelems(), 'mov': X.nelems()}
    optype2instr['LeakyRelu']['add'] = X.nelems()

    for xopname in ['Reciprocal', 'Floor', 'Ceil', 'Sqrt', 'Exp', 'Log',
                    'Neg', 'Abs', 'Sin', 'Cos', 'Tan', 'Asin', 'Acos', 'Atan',
                    'Sinh', 'Cosh', 'Tanh', 'Asinh', 'Acosh', 'Atanh', 'Erf',
                    'Round', ]:
        optype2instr[xopname] = {xopname.lower(): X.nelems()}

    op.perf_stats =  {
            'inElems' : X.nelems(),
            'outElems': Y.nelems(),
            'inBytes' : X.nbytes(op.precision),
            'outBytes': Y.nbytes(op.precision),
            'instrs'  : optype2instr[op.optype],
            }
    return

def bidir_bcast(iTList, oTList, op, **kwargs):
    X0,X1, Y = iTList[0], iTList[1], oTList[0]
    assert X0.check_shape(), f"Input tensor-0 shape not defined: {X0}"
    assert X1.check_shape(), f"Input tensor-1 shape not defined: {X1}"
    Y.shape = bidirectional_broadcast_shape_inference(X0.shape, X1.shape)
    Y.dtype = X0.dtype
    op.perf_stats =  {
            'inElems' : X0.nelems() + X1.nelems(),
            'outElems': Y.nelems(),
            'inBytes' : X0.nbytes(op.precision) + X1.nbytes(op.precision),
            'outBytes': Y.nbytes(op.precision),
            'instrs'  : {op.optype.lower(): Y.nelems()}
            }
    return
