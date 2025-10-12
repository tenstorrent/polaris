#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.desc.registry import register_ops
import numpy as np

def tgather_sinf(iTList, oTList, op, **kwargs):
    dataT    = iTList[0]
    indexT   = iTList[1]
    #axis     = op.attrs.get('axis', 0)

    assert dataT.check_shape(), f"Illegal input dataT shape: {dataT}!!"
    assert indexT.check_shape(), f"Illegal input indexT shape: {indexT}!!"

    # Output shape is same as index shape
    oTList[0].shape = indexT.shape
    oTList[0].dtype = dataT.dtype

    # Perf: each output element is a gather from input
    op.perf_stats = {
        'inElems' : dataT.nelems() + indexT.nelems(),
        'outElems': oTList[0].nelems(),
        'inBytes' : dataT.nbytes(op.precision) + indexT.nbytes(op.precision),
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs'  : {'mov': oTList[0].nelems()}
    }
    return

def as_sinf(iTList, oTList, op, **kwargs):
    # iTList[0]: output tensor to assign into
    # iTList[1]: input tensor to assign from
    # attrs: 'slice' specifies the slice indices for assignment

    output_tensor = iTList[0]
    input_tensor = iTList[1]
    slice_spec = op.attrs.get('slice', None)
    assert slice_spec is not None, "AssignOp requires 'slice' attribute specifying indices"

    # Validate slice shape matches input tensor shape
    # For example, to assign y[:, l, :, :] = temp:
    # - output_tensor: y
    # - input_tensor: temp
    # - slice_spec: (slice(None), l, slice(None), slice(None))
    # So, in AssignOp, set attrs['slice'] = (slice(None), l, slice(None), slice(None))
    # Then, output_tensor[slice_spec].shape == input_tensor.shape
    # This is a shape check only; actual assignment is not performed here

    # Compute the shape of the slice
    # Use numpy to infer the shape
    dummy = np.empty(output_tensor.shape)
    sliced = dummy[slice_spec]
    assert list(sliced.shape) == list(input_tensor.shape), \
        f"AssignOp: input tensor shape {input_tensor.shape} does not match slice shape {list(sliced.shape)}"

    oTList[0].shape = output_tensor.shape
    oTList[0].dtype = output_tensor.dtype

    # Count: 1 mov per element assigned
    instr_count = {'mov': input_tensor.nelems()}
    op.perf_stats = {
        'inElems': input_tensor.nelems(),
        'inBytes': input_tensor.nbytes(op.precision),
        'outElems': oTList[0].nelems(),
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': instr_count
    }
    return

def vp_sinf(iTList, oTList, op, **kwargs):
    for i in range(4):
        assert iTList[i].check_shape(), f"input[{i}] shape error: {iTList[i]}"
    assert iTList[3].data is not None, f"Missing voxel_num.data!!"

    geom_xyz         = iTList[0]
    depth_features   = iTList[1]
    context_features = iTList[2]
    voxel_num        = iTList[3]

    batch_size      = geom_xyz.shape[0]
    num_cams        = geom_xyz.shape[1]
    num_depth       = geom_xyz.shape[2]
    num_height      = geom_xyz.shape[3]
    num_width       = geom_xyz.shape[4]
    num_channels    = context_features.shape[1]
    #output_shape    = [batch_size, int(voxel_num.data[1]), int(voxel_num.data[0]), num_channels]
    #output = output.permute(0, 3, 1, 2)
    output_shape    = [batch_size, num_channels, int(voxel_num.data[1]), int(voxel_num.data[0])]

    oTList[0].shape = output_shape
    oTList[0].dtype = geom_xyz.dtype

    # Total number of samples ("points" to pool)
    total_samples = batch_size * num_cams * num_depth * num_height * num_width
    total_ops     = total_samples * num_channels
    instr_count = {
            'cmp': total_samples * 4, #4 bound checks per sample
            'mac': total_ops          #1 MAC per-sample, per-channel
            }
    loads = total_ops * 5 #5 loads per-sample, per-channel
    stores= total_ops     #1 store per-sample, per-channel

    bpe = 2 #assume fp16
    op.perf_stats = {
            'inElems' : loads,
            'inBytes' : loads * bpe,
            'outElems': stores,
            'outBytes': stores * bpe,
            'instrs'  : instr_count
            }

    return

def register_custom_ops():
    _optbl = [
            ['VoxelPooling', 'ARITY_4->1', 'custom', 'NA', -1, -1, 4, 4, 1, 1, vp_sinf,  False,  False, False,  False,  False],
            ['Assign', 'ARITY_2->1', 'custom', 'NA', -1, -1, 2, 2, 1, 1, as_sinf,  False,  False, False,  False,  False],

            #just to ungate checkin regression failures for now...
            ['TorchGather', 'ARITY_2->1', 'custom', 'NA', -1, -1, 2, 2, 1, 1, tgather_sinf,  False,  False, False,  False,  False],
            ]
    register_ops('custom', _optbl)
    return

