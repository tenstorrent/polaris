#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ttsim.ops.desc.helpers import build_tmp_data_tensor, unary_fwd, update_output_tensor
from ttsim.ops.desc.registry import register_ops
from ttsim.utils.common import prod_ints


def trilu_sinf(iTList, oTList, op, **kwargs):
    """
    TODO: this is very specific to DLRM usage right now
     need to generalize this as specified in ONNX opset!!
    """
    upper = op.attrs.get('upper', 1)
    assert len(iTList) == 1, f"More than 1 inputs not supported for Trilu for now!!"

    X = iTList[0].clone_by_shape()

    # Get the upper triangular indices manually (excluding diagonal)
    # This Code is DLRM specific
    row_indices, col_indices = [], []
    batch_size, num_features1, num_features2 = X.shape
    assert num_features1 == num_features2, f"Input should be an batch of square matrices: {X.shape}"
    num_features = num_features1
    for i in range(num_features):
        for j in range(i + 1, num_features):
            row_indices.append(i)
            col_indices.append(j)
    tmp_data = X.data[:, row_indices, col_indices]
    tmp_outT = build_tmp_data_tensor(tmp_data, op.name + '__tmp_out__')
    update_output_tensor(op, tmp_outT, oTList[0])

    op.perf_stats = {
            'inElems' : X.nelems(),
            'inBytes' : X.nbytes(op.precision),
            'outElems': oTList[0].nelems(),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mov': 100} #dummy - TODO get the real cost involved!!
            }
    return op.perf_stats

def where_sinf(iTList, oTList, op, **kwargs):
    '''
    ASSUME ONNX SHAPE INFERENCE
    condB = clone_tensor_by_shape(iTList[0], data_maybe_missing=False) #condB.data should exist
    X     = clone_tensor_by_shape(iTList[1])
    Y     = clone_tensor_by_shape(iTList[2])
    assert condB.dtype == np.bool_, f"Illegal Input Tensor Data Type: {condB}"
    np_out = np.where(condB.data, X.data, Y.data)
    tmp_outT = build_tmp_data_tensor(np_out, op.name + '__tmp_out__')
    update_output_tensor(op, tmp_outT, oTList[0])
    '''
    #assert oTList[0].check_shape(), f"SHAPE INFERENCE ERROR!!"
    oTList[0].shape = iTList[1].shape
    oTList[0].dtype = iTList[1].dtype
    op.perf_stats = {
            'inElems' : iTList[0].nelems() + iTList[1].nelems() + iTList[2].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision) + iTList[2].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mov': oTList[0].nelems(), 'cmp': oTList[0].nelems()}
            }
    return

def cast_sinf(iTList, oTList, op, **kwargs):
    '''
    ASSUME ONNX SHAPE INFERENCE
    saturate =  op.attrs.get('saturate', 1)
    to_type  =  op.attrs['to']
    A = clone_tensor_by_shape(iTList[0], data_maybe_missing=False) #A.data must be present
    tensor_type = TENSOR_TYPE_MAP[to_type]
    np_out = A.data.astype(tensor_type.np_dtype)
    tmp_outT = build_tmp_data_tensor(np_out, op.name + '__tmp_out__')
    update_output_tensor(op, tmp_outT, oTList[0])
    '''
    assert iTList[0].check_shape()
    assert oTList[0].check_shape()
    op.perf_stats = {
            'inBytes' : iTList[0].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'inElems' : iTList[0].nelems(),
            'outElems': oTList[0].nelems(),
            'instrs'  : {'mov': oTList[0].nelems()}
            }
    return

def pad_sinf(iTList, oTList, op, **kwargs):
    X = iTList[0]
    pad_tensor = iTList[1]
    mode = op.attrs.get('mode', 'constant')
    value = op.attrs.get('value', 0)

    assert pad_tensor.data is not None, "PadOp requires pad_tensor with data"
    pads = [int(x) for x in pad_tensor.data.tolist()]
    # Only pad the last 2 dimensions
    assert len(pads) == 4, f"pads length {len(pads)} != 4 (for last 2 dims only)"
    rank = X.rank()
    pad_before = [0] * (rank - 2) + pads[:2]
    pad_after = [0] * (rank - 2) + pads[2:]
    output_shape = [X.shape[i] + pad_before[i] + pad_after[i] for i in range(rank)]

    oTList[0].shape = output_shape
    oTList[0].dtype = X.dtype

    # Estimate instruction count: each output element is a copy or fill
    nElem_in = X.nelems()
    nElem_out = np.prod(output_shape)

    op.perf_stats = {
        'inElems': nElem_in + pad_tensor.nelems(),
        'inBytes': X.nbytes(op.precision) + pad_tensor.nbytes(op.precision),
        'outElems': nElem_out,
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': {'mov': nElem_out},
    }
    return

def tile_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    dataT    = iTList[0]
    repeatsT = iTList[1].clone_by_shape(data_maybe_missing=False)
    assert len(repeatsT.data) == dataT.rank(), \
            f"repeats={repeatsT.data} should have same length as input shape={dataT.shape}"

    checkshape = [d > 0 for d in repeatsT.data]
    assert all(checkshape), f"repeats={repeatsT.data} should be > 0"

    outshape   = [dim * repeatsT.data[i] for i,dim in enumerate(dataT.shape)]

    oTList[0].shape = outshape
    oTList[0].dtype = dataT.dtype

    op.perf_stats = {
            'inBytes' : int(iTList[0].nbytes(op.precision)) + int(iTList[1].nbytes(op.precision)),
            'inElems' : int(iTList[0].nelems()) + int(iTList[1].nelems()),
            'outBytes': int(oTList[0].nbytes(op.precision)),
            'outElems': int(oTList[0].nelems()),
            'instrs'  : {'mov': int(oTList[0].nelems())}
            }
    return

def squeeze_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    dataT = iTList[0]
    axesT = iTList[1].clone_by_shape(data_maybe_missing=False) #Y.data must be present

    data_rank  = dataT.rank()
    data_idx   = [ d + data_rank if d < 0 else d for d in axesT.data]
    checkshape = [d >= 0 and d < data_rank for d in data_idx]
    assert all(checkshape), f"axes={axesT.data} out of bounds: [-{data_rank}, {data_rank-1}]"
    outshape   = [dim for i,dim in enumerate(dataT.shape) if i not in data_idx]

    oTList[0].shape = outshape
    oTList[0].dtype = dataT.dtype

    op.perf_stats = {
            'inBytes' : int(iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision)),
            'outBytes': int(oTList[0].nbytes(op.precision)),
            'inElems' : int(iTList[0].nelems() + iTList[1].nelems()),
            'outElems': int(oTList[0].nelems()),
            'instrs'  : {'mov': int(oTList[0].nelems())}
            }
    return

def unsqueeze_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    Y = iTList[1].clone_by_shape(data_maybe_missing=False) #Y.data must be present
    newshape = list(iTList[0].shape)
    for d in Y.data:
        newrank = len(newshape)
        if d < 0: d = newrank + d + 1
        if d < 0 or d > newrank:
            raise ValueError(f"Axis {d} out of bounds: [-{newrank+1}, {newrank}]")
        newshape.insert(d, 1)

    #tmp_outT = build_tmp_data_tensor(np_out, op.name + '__tmp_out__')
    #update_output_tensor(op, tmp_outT, oTList[0])
    oTList[0].shape = [int(i) for i in newshape]
    oTList[0].dtype = iTList[0].dtype

    op.perf_stats = {
            'inBytes' : iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'inElems' : iTList[0].nelems() + iTList[1].nelems(),
            'outElems': oTList[0].nelems(),
            'instrs'  : {'mov': oTList[0].nelems()}
            }
    return

def slice_sinf(iTList, oTList, op, **kwargs):
    dataT   = iTList[0]
    startsT = iTList[1].clone_by_shape(data_maybe_missing=False) #startsT.data must be present
    endsT   = iTList[2].clone_by_shape(data_maybe_missing=False) #endsT.data must be present

    if len(iTList) >= 4:
        axesT = iTList[3].clone_by_shape(data_maybe_missing=False) #axesT.data must be present
    else:
        axesT = build_tmp_data_tensor(np.array([i for i in range(dataT.rank())]),
                                      op.name + '__tmp_axesT__')

    if len(iTList) == 5:
        stepsT = iTList[4].clone_by_shape(data_maybe_missing=False) #stepsT.data must be present
    else:
        stepsT = build_tmp_data_tensor(np.array([1 for _ in range(dataT.rank())]),
                                      op.name + '__tmp_stepsT')

    #print('Slice dataT=',   dataT)
    #print('Slice startsT=', startsT)
    #print('Slice endsT=',   endsT)
    #print('Slice axesT=',   axesT)
    #print('Slice stepsT=',  stepsT)

    #sanity checks...
    assert startsT.rank() == 1,           f"Slice Error 0, {startsT.shape}, rank != 1"
    assert startsT.shape == endsT.shape,  f"Slice Error 1, {startsT.shape} != {endsT.shape}"
    assert startsT.shape == axesT.shape,  f"Slice Error 2, {startsT.shape} != {axesT.shape}"
    assert startsT.shape == stepsT.shape, f"Slice Error 3, {startsT.shape} != {stepsT.shape}"

    #slices = [slice(None)] *  dataT.rank()
    #for s in range(startsT.rank()):
    #    s_axis  = axesT.data[s]
    #    s_start = startsT.data[s]
    #    s_end   = endsT.data[s]
    #    s_step  = stepsT.data[s]
    #    slices[s_axis] = slice(s_start, s_end, s_step)
    #np_out = dataT.data[tuple(slices)]
    #tmp_outT = build_tmp_data_tensor(np_out, op.name + '__tmp_out__')
    #update_output_tensor(op, tmp_outT, oTList[0])

    assert 'out_shape' in op.attrs, "No out_shape specified in Slice!! " + \
                         "Look at tensor_getitem implementation in ttsim/.../tensor_op.py"
    op.attrs['out_shape'] = [int(i) for i in op.attrs['out_shape']]
    oTList[0].shape = op.attrs['out_shape']
    oTList[0].dtype = dataT.dtype

    inBytes = dataT.nbytes(op.precision) + startsT.nbytes(op.precision) + endsT.nbytes(op.precision)
    inBytes += axesT.nbytes(op.precision)  if len(iTList) >= 4 else 0 #assume 4 bytes per axis spec
    inBytes += stepsT.nbytes(op.precision) if len(iTList) == 5 else 0 #assume 4 bytes per steps spec

    assert oTList[0].check_shape(), f"SHAPE INFERENCE ERROR!!"
    op.perf_stats = {
            'inBytes' : int(sum([x.nbytes(op.precision) for x in iTList])),
            'inElems' : int(sum([x.nelems() for x in iTList])),
            'outBytes': int(oTList[0].nbytes(op.precision)),
            'outElems': int(oTList[0].nelems()),
            'instrs'  : {'mov': int(oTList[0].nelems())}
            }
    return

def resize_sinf(iTList, oTList, op, **kwargs):
    if oTList[0].check_shape():
        pass
    else:
        assert len(iTList) >= 3, f"RESIZE #inputs ({len(iTList)}) should be >= 3"
        assert iTList[0].check_shape(), f"Illegal Resize Input  Tensor Shape"
        assert iTList[1].check_shape(), f"Illegal Resize ROI    Tensor Shape"
        assert iTList[2].check_shape(), f"Illegal Resize SCALES Tensor Shape"
        assert iTList[2].data is not None, f"SCALES data missing"
        XRank = iTList[0].rank()
        scales = [1.0] * XRank
        scales[-1] = iTList[2].data[-1]
        scales[-2] = iTList[2].data[-2]
        oTList[0].shape = [int(scales[i] * x) for i,x in enumerate(iTList[0].shape)]
        oTList[0].dtype = iTList[0].dtype

    nElem = iTList[0].nelems()
    op.perf_stats = {
            'inElems' : iTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision),
            'outElems': oTList[0].nelems(),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'cmp': nElem, 'mov': nElem}
            }
    return

def upsample_sinf(iTList, oTList, op, **kwargs):
    X = iTList[0]
    input_shape = X.shape
    mode = op.attrs.get('mode', 'nearest')
    #scales = op.attrs.get('scales', None)
    scale_factor = op.attrs.get('scale_factor', 1)
    size = op.attrs.get('size', None)
    align_corners = op.attrs.get('align_corners', False)

    # Determine output shape
    if size is not None:
        # size is a tuple/list of output spatial dims
        # torch.nn.Upsample expects size to specify the output spatial dims
        # For 2D: input (N, C, H, W), size = (out_H, out_W)
        output_shape = list(input_shape)
        spatial_dims = len(size)
        output_shape[-spatial_dims:] = list(size)
    # elif scales is not None:
    elif scale_factor is not None:
        # scales is a list/tuple of scaling factors for each dim
        output_shape = list(input_shape)
        if len(output_shape) >= 2:
            output_shape[-2] = int(round(output_shape[-2] * scale_factor))
            output_shape[-1] = int(round(output_shape[-1] * scale_factor))
    else:
        raise ValueError("UpsampleOp requires either 'size' or 'scale_factor' attribute")

    oTList[0].shape = output_shape
    oTList[0].dtype = X.dtype

    # Estimate instruction count: each output element is a copy or interpolation
    nElem_in = X.nelems()
    nElem_out = np.prod(output_shape)
    instr_count = {}
    if mode == 'nearest':
        instr_count['mov'] = nElem_out
    elif mode in ['linear', 'bilinear', 'trilinear', 'bicubic']:
        # Each output element: interpolation (mul/add per neighbor)
        # For bilinear: 4 neighbors, trilinear: 8, linear: 2, bicubic: 16
        neighbors = {'linear': 2, 'bilinear': 4, 'trilinear': 8, 'bicubic': 16}
        n = neighbors.get(mode, 4)
        instr_count['mul'] = nElem_out * n
        instr_count['add'] = nElem_out * (n - 1)
    else:
        raise ValueError(f"Unsupported Upsample mode: {mode}")

    op.perf_stats = {
        'inElems': nElem_in,
        'inBytes': X.nbytes(op.precision),
        'outElems': nElem_out,
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': instr_count
    }
    return

def concat_sinf(iTList, oTList, op, **kwargs):
    axis = op.attrs['axis']
    assert len(iTList) > 0, f"empty input list in Concat!!"
    base_rank = iTList[0].rank()
    assert all(x.rank() == base_rank for x in iTList), "input tensors rank mismatch"
    if axis < 0: axis = base_rank + axis
    if axis < 0 or axis >= base_rank:
        raise ValueError(f"Axis {axis} is out of bounds for tensors with rank {base_rank}. "
                    f"Valid range is [-{base_rank}, {base_rank-1}].")

    for i, x in enumerate(iTList[1:], 1):
        for dim in range(x.rank()):
            if dim != axis and x.shape[dim] != iTList[0].shape[dim]:
                    raise ValueError(f"Incompatible shapes at dim {i}: {x.shape} vs {iTList[0].shape}. "
                                     f"All dimensions except the concat axis ({axis}) must match.")

    oshape        = list(iTList[0].shape)
    oshape[axis]  = sum(x.shape[axis] for x in iTList)
    oTList[0].shape = oshape
    oTList[0].dtype = iTList[0].dtype

    # Placeholder: For Training, it may be required to output per-input tensor shape
    # Assumption: per-input tensor shape is a 1D-Tensor where each element
    # represents the length of the corresponding input along the axis
    #
    #out2_shape = [len(iTList)]
    #out2_data  = [x.shape for x in Xs]

    inBytes = sum((x.nbytes(op.precision) for x in iTList))
    inElems = sum((x.nelems() for x in iTList))
    op.perf_stats = {
            'inBytes' : inBytes,
            'inElems' : inElems,
            'outBytes': oTList[0].nbytes(op.precision),
            'outElems': oTList[0].nelems(),
            'instrs'  : {'mov': oTList[0].nelems()}
            }
    return

def reshape_sinf(iTList, oTList, op, **kwargs):
    allowzero = op.attrs.get('allowzero', 0)

    #A = iTList[0].clone_by_shape()
    B = iTList[1].clone_by_shape(data_maybe_missing=False) #B.data should exist
    assert iTList[0].check_shape(), f"Illegal Input Shape: {iTList[0].shape}"
    assert B.dtype == np.int64, f"Input data type should be np.int64, was {B.dtype} for reshape's shape tensor {B}"
    assert B.data is not None, f"Input Data should exist for reshape's shape tensor {B}"
    input_shape  = iTList[0].shape
    input_size   = iTList[0].nelems()
    target_shape = [x.item() for x in B.data]

    minus_one_count = 0
    minus_one_index = None
    zeros_count     = 0
    zeros_index     = []
    for i,x in enumerate(target_shape):
        if x == -1:
            minus_one_count += 1
            minus_one_index = i
        elif x == 0:
            zeros_count += 1
            zeros_index.append(i)
        else:
            pass
    assert minus_one_count <= 1, f"Only one -1 is allowed in target shape {target_shape}"

    if allowzero == 1 and minus_one_count == 1 and zeros_count > 0:
        assert False, f"Cannot have -1 and zeros simultaneously with allowzero in target_shape({target_shape})"

    #copy dims from input_shape, if required
    output_shape = [x for x in target_shape]
    if allowzero == 0:
        for idx in zeros_index:
            assert idx < len(input_shape), f"Illegal index({idx}) for input_shape({input_shape}) with allowzero=0"
            output_shape[idx] = input_shape[idx]

    # Handle -1 inference
    if minus_one_count == 1:
        output_size = prod_ints([x for x in output_shape if x != -1])
        assert input_size >= output_size and input_size % output_size == 0, \
                f"Cannot infer -1: input size {input_size}/{output_size}"
        inferred_dim = input_size // output_size
        output_shape[minus_one_index] = inferred_dim #type: ignore

    # Final validation
    final_output_size = prod_ints(output_shape)
    assert input_size  == final_output_size, \
            f"in({input_size}) & out({final_output_size}) sizes are not equal!!"

    #np_out = reshape_reference_implementation(A.data, B.data, allowzero=allowzero)
    #tmp_outT = build_tmp_data_tensor(np_out, op.name + '__tmp_C_out__')
    #update_output_tensor(op, tmp_outT, oTList[0])
    oTList[0].shape = output_shape
    oTList[0].dtype = iTList[0].dtype

    op.perf_stats = {
            'inElems' : int(iTList[0].nelems() + B.nelems()),
            'outElems': int(oTList[0].nelems()),
            'inBytes' : int(iTList[0].nbytes(op.precision) + B.nbytes(op.precision)),
            'outBytes': int(oTList[0].nbytes(op.precision)),
            'instrs'  : {'mov': int(oTList[0].nelems())}
            }
    return

def expand_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    shapeT = iTList[1].clone_by_shape(data_maybe_missing=True) #shapeT.data should exist
    target_shape = [x.item() for x in shapeT.data]
    input_shape  = A.shape

    # Align shapes by prepending 1s to input_shape if needed
    if len(target_shape) > len(input_shape):
        input_shape = [1] * (len(target_shape) - len(input_shape)) + input_shape

    assert len(target_shape) == len(input_shape), f"Input & Target shapes length mismatch: {input_shape} vs {target_shape}"
    for i, (in_dim, tgt_dim) in enumerate(zip(input_shape, target_shape)):
        if tgt_dim != in_dim and in_dim != 1:
            raise ValueError(f"Cannot expand dimension {i}: input dim {in_dim} to target dim {tgt_dim}")
    oTList[0].shape = target_shape
    oTList[0].dtype = A.dtype
    op.perf_stats = {
            'inElems' : A.nelems() + shapeT.nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : A.nbytes(op.precision) + shapeT.nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mov': oTList[0].nelems()}
            }
    return

def split_sinf(iTList, oTList, op, **kwargs):
    num_outputs = op.attrs.get('num_outputs', len(oTList))
    axis        = op.attrs.get('axis',0)

    A      = iTList[0]
    splitT = iTList[1] if len(iTList) == 2 else None
    assert A.check_shape(), "Illegal shape!!"
    if splitT is None or splitT.data is None:
        split_dim = A.shape[axis] // num_outputs
        split = [split_dim for i in range(num_outputs)]
    else:
        split = [x.item() for x in splitT.data]
    assert len(split) == num_outputs, f"split mismatch len( {split} ) != {num_outputs}"

    outShapes = []
    for tout_idx in range(num_outputs):
        tout_shape = A.shape.copy()
        tout_shape[axis] = split[tout_idx]
        outShapes.append(tout_shape)

    outBytes = 0
    outElems = 0
    for tidx, tout in enumerate(oTList):
        tshape0 = outShapes[tidx]
        tout.shape = tshape0
        tout.dtype = A.dtype
        outBytes += tout.nbytes(op.precision)
        outElems += tout.nelems()

    splitT_nelems = 0 if splitT is None else splitT.nelems()
    splitT_nbytes = 0 if splitT is None else splitT.nbytes(op.precision)
    op.perf_stats = {
            'inElems' : A.nelems() + splitT_nelems,
            'outElems': outElems,
            'inBytes' : A.nbytes(op.precision) + splitT_nbytes,
            'outBytes': outBytes,
            'instrs'  : {'mov': outElems}
            }
    return

def gather_sinf(iTList, oTList, op, **kwargs):
    axis = op.attrs.get('axis', 0)
    assert isinstance(axis, int), f"attribute axis ({axis}) is not an int!!"

    dataT    = iTList[0]
    indicesT = iTList[1]
    assert dataT.check_shape(), f"Illegal input dataT shape: {dataT}!!"
    assert indicesT.check_shape(), f"Illegal input indicesT shape: {indicesT}!!"

    data_rank  = dataT.rank()
    data_shape = dataT.shape
    # Normalize negative axis
    axis = axis if axis >= 0 else data_rank + axis
    assert axis >= 0 and axis < data_rank, f"Axis {axis} is out of bounds for dataT.shape {dataT.shape}"
    oTList[0].shape = data_shape[:axis] + indicesT.shape + data_shape[axis + 1:]
    oTList[0].dtype = dataT.dtype

    op.perf_stats = {
            'inElems' : int(oTList[0].nelems()), #read just what we need, not the whole embed. tbl
            'outElems': int(oTList[0].nelems()),
            'inBytes' : int(oTList[0].nbytes(op.precision)),
            'outBytes': int(oTList[0].nbytes(op.precision)),
            'instrs'  : {'mov': int(oTList[0].nelems())}
            }
    return

def shape_op_inf_func(iTList, oTList, op, **kwargs):
    A = iTList[0].clone_by_shape()

    start =  op.attrs.get('start', 0)
    end   =  op.attrs.get('end')
    start = 0 if start < 0 else start
    end   = A.rank() if end is None or end > A.rank() else end
    end   = A.rank() + end if end < 0 else end
    tdata = np.array(A.shape[start:end], dtype=np.int64)
    tmp_tensor = build_tmp_data_tensor(tdata, op.name + '_tmp_out_tensor_')
    update_output_tensor(op, tmp_tensor, oTList[0])
    op.perf_stats = {
            'inBytes' : A.rank() * 4,
            'outBytes': A.rank() * 4,
            'instrs'  : {'mov': A.rank()} # 4Bytes per Index
            }
    return

def transpose_op_inf_func(iTList, oTList, op, **kwargs):
    perms  = op.attrs['perm']
    assert len(perms) == iTList[0].rank(), f"perms({perms}) must be equal to input rank ({iTList[0].rank()})!!"
    oTList[0].shape = [iTList[0].shape[i] for i in perms]
    oTList[0].dtype = iTList[0].dtype
    op.perf_stats = {
            'inElems' : iTList[0].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mov': oTList[0].nelems()}
            }
    return

def register_tensor_ops():
    _optbl = [
            ##['Size',           'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['SpaceToDepth',   'ARITY_1->1', 'ai.onnx',  'COMMON',  13,  13,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['DepthToSpace',   'ARITY_1->1', 'ai.onnx',  'COMMON',  13,  13,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['IsNaN',          'ARITY_1->1', 'ai.onnx',  'COMMON',  20,  20,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['IsInf',          'ARITY_1->1', 'ai.onnx',  'COMMON',  20,  20,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['NonZero',        'ARITY_1->1', 'ai.onnx',  'COMMON',  13,  13,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['CastLike',       'ARITY_2->1', 'ai.onnx',  'COMMON',  24,  21,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['GatherElements', 'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['GridSample',     'ARITY_2->1', 'ai.onnx',  'COMMON',  22,  22,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['AffineGrid',     'ARITY_2->1', 'ai.onnx',  'COMMON',  20,  20,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['Compress',       'ARITY_2->1', 'ai.onnx',  'COMMON',  11,  11,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['ReverseSequence','ARITY_2->1', 'ai.onnx',  'COMMON',  10,  10,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['GatherND',       'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['CenterCropPad',  'ARITY_2->1', 'ai.onnx',  'COMMON',  18,  18,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['Scatter',        'ARITY_3->1', 'ai.onnx',  'COMMON',  11,  11,  3,  3,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['ScatterND',      'ARITY_3->1', 'ai.onnx',  'COMMON',  18,  18,  3,  3,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['ScatterElements','ARITY_3->1', 'ai.onnx',  'COMMON',  18,  18,  3,  3,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['OneHot',         'ARITY_3->1', 'ai.onnx',  'COMMON',  11,  11,  3,  3,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['Unique',  'ARITY_1->VARIADIC[1-4]', 'ai.onnx',  'COMMON',  11,  11,  1,  1,  4,  1,  'inline_lmbda', True, True, True, True, True],

            ['Shape',     'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1, shape_op_inf_func,  True,  True,  True,  True,  True],
            ['Transpose', 'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1, transpose_op_inf_func,  True,  True,  True,  True,  True],
            ['Identity',  'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1,  unary_fwd,  True,  True,  True,  True,  True],

            ['Gather',    'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1, gather_sinf,  True,  True,  True,  True,  True],
            ['Cast',      'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1, cast_sinf,  True,  True,  True,  True,  True],

            ['Reshape',   'ARITY_2->1', 'ai.onnx',  'COMMON',  24,  21,  2,  2,  1,  1, reshape_sinf,  True,  True,  True,  True,  True],
            ['Expand',    'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1, expand_sinf,  True,  True,  True,  True,  True],

            ['Unsqueeze', 'ARITY_2->1', 'ai.onnx',  'COMMON',  24,  21,  2,  2,  1,  1,
             unsqueeze_sinf,  True,  True,  True,  True,  True],
            ['Tile',      'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1, tile_sinf,  True,  True,  True,  True,  True],
            ['Upsample',  'ARITY_2->1', 'ai.onnx',  'COMMON',  10,  10,  2,  2,  1,  1,
             upsample_sinf,  True,  True,  True,  True,  True],
            ['Where',     'ARITY_3->1', 'ai.onnx',  'COMMON',  16,  16,  3,  3,  1,  1, where_sinf,  True,  True,  True,  True,  True],

            ['Squeeze',   'ARITY_VARIADIC[1-2]->1', 'ai.onnx',  'COMMON',  24,  21,  2,  1,  1,  1,
             squeeze_sinf, True, True, True, True, True],
            ['Trilu',     'ARITY_VARIADIC[1-2]->1', 'ai.onnx',  'COMMON',  14,  14,  2,  1,  1,  1,
             trilu_sinf, True, True, True, True, True],
            ['Resize',    'ARITY_VARIADIC[1-4]->1', 'ai.onnx',  'COMMON',  19,  19,  4,  1,  1,  1,
             resize_sinf, True, True, True, True, True],
            ['Slice',     'ARITY_VARIADIC[3-5]->1', 'ai.onnx',  'COMMON',  13,  13,  5,  3,  1,  1,
             slice_sinf, True, True, True, True, True],
            ['Pad',       'ARITY_VARIADIC[2-4]->1', 'ai.onnx',  'COMMON',  24,  21,  4,  2,  1,  1,
             pad_sinf, True, True, True, True, True],

            ['Concat', 'ARITY_VARIADIC[1-*]->1',             'ai.onnx', 'COMMON', 13, 13,
             2147483647, 1, 1, 1, concat_sinf, True, True, True, True, True],
            ['Split',  'ARITY_VARIADIC[1-2]->VARIADIC[1-*]', 'ai.onnx', 'COMMON', 18, 18, 2, 1,
             2147483647, 1, split_sinf, True, True, True, True, True],
            ]

    register_ops('tensor', _optbl)
    return
