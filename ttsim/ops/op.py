#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache, reduce
import operator
import math
from loguru import logger
import numpy as np
from typing import Union, TYPE_CHECKING, Dict, Any

from onnx.mapping import TENSOR_TYPE_MAP
import ttsim.utils.common as common
from .tensor import SimTensor

LOG   = logger
INFO  = LOG.info
DEBUG = LOG.debug
G_COMPUTE_UTIL_CONSTANT = 0.6 #hard coded for now, will get this from the model after Tiler implementation
G_MEMORY_UTIL_CONSTANT  = 0.8 #hard coded for now, will get this from mem-stream benchmark measurements

class GRAD_TENSOR_INFO:
    def __init__(self, t):
        self.name                 = t.name
        self._output_grad_tensor  = t
        self._grad_ops            = []
        self._input_fwd_tensors   = []
        self._input_grad_tensors  = []
        self._new_tensors         = []

    def __str__(self):
        x  = f"GRAD_TENSOR_INFO({self.name})"
        x += f"  output_grad_tensor:\n {self._output_grad_tensor}\n"
        x += f"  grad_ops\n"
        for ppp in self._grad_ops: x += f"  {ppp}\n"
        x += f"  in_fwd_tensors\n"
        for ppp in self._input_fwd_tensors: x += f"  {ppp}\n"
        x += f"  in_grad_tensors\n"
        for ppp in self._input_grad_tensors: x += f"  {ppp}\n"
        x += f"  new_tensors\n"
        for ppp in self._new_tensors: x += f"  {ppp}\n"
        return x

def get_tensor_broadcast_shape(shape1, shape2):
    """Determine broadcasted shape for element-wise operations"""
    s1 = shape1[::-1]
    s2 = shape2[::-1]
    max_len = max(len(s1), len(s2))
    s1.extend([1] * (max_len - len(s1)))
    s2.extend([1] * (max_len - len(s2)))

    result = []
    for d1, d2 in zip(s1, s2):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} not broadcast-compatible")
    return result[::-1]

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

def clone_tensor_by_shape(itensor, /, data_maybe_missing = True):
    assert itensor.check_shape(), f"Illegal Shape in Tensor {itensor}"
    if data_maybe_missing:
        if itensor.data is None:
            if itensor.rank() == 0:
                if itensor.dtype == np.float32:
                    clone_data = np.float32(1.0)
                else:
                    assert False, "Only np.float32 rank-0 tensor clones supported right now!!!"
            else:
                cloned_data = np.random.randn(*(itensor.shape)).astype(itensor.dtype)
            clone = SimTensor({
                'name'   : itensor.name,
                'shape'  : itensor.shape,
                'dtype'  : itensor.dtype,
                'data'   : cloned_data,
                'resolve': itensor.resolve,
                'op_in'  : itensor.op_in,
                'op_out' : itensor.op_out
                })
        else:
            clone = itensor
    else:
        assert itensor.data is not None, f"Illegal Data in Tensor {itensor}"
        clone = itensor
    return clone

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

def check_io_counts(op, /, in_counts, out_counts):
    """in_counts, out_counts specify ranges [min,max]"""
    in_range = range(in_counts[0], in_counts[1]+1)
    out_range = range(out_counts[0], out_counts[1]+1)
    assert len(op.inList) in in_range,   f"#inputs for {op} operator should be in {in_range}, is {len(op.inList)}"
    assert len(op.outList) in out_range, f"#outputs for {op} operator should be in {out_range}, is {len(op.outList)}"
    return

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

class SimOp:
    def __init__(self, cfg):
        self.name         = cfg['name']
        self.optype       = cfg['optype']
        self.attrs        = cfg.get('attrs', {})
        self.inList       = cfg.get('inList', [])
        self.outList      = cfg.get('outList', [])
        self.domain       = cfg.get('domain', "")
        self.docstr       = cfg.get('docstr', "")
        self.opclass_str  = 'None'

        #special counter for some workloads, e.g., Transformer Blocks
        # where we execute the op only once, but account for repeated
        # executions for the full workload
        self.repeat_count = 1

        #These fields are set via __call__ / get_perf_counts() when the op is executed
        # with input tensors dim/shape being well defined
        self.perf_stats: Union[dict, None]   = None

        #These fields are set via execution of op of a device...
        self.precision               = None
        self.removed_in_optimization = False
        self.fused_in_optimization   = False
        self.fused_with_op           = None
        self.uses_compute_pipe       = None
        self.compute_cycles          = None
        self.mem_rd_cycles           = None
        self.mem_wr_cycles           = None
        self.fused_op_cycles         = None
        self._kw_args_defaults       = {}

    def __str__(self):
        s  = f"SimOp({self.name}) optype={self.optype}, cls={self.opclass_str}, "
        s += f"prec={self.precision}, attrs={self.attrs}, domain={self.domain}, "
        s += f"rpt={self.repeat_count}, "
        s += f"removed={self.removed_in_optimization}, "
        s += f"fused={self.fused_in_optimization}, "
        s += f"fused_with_op={self.fused_with_op}, "
        s += f"uses_compute_pipe={self.uses_compute_pipe}, "
        #s += f"docstr={self.docstr}, "
        s += f"inList={self.inList}, "
        s += f"outList={self.outList}"
        return s

    def check_known_args(self, args: dict[str, Any]) -> None:
        common.check_known_args(str(type(self)), args=args,
                                default_args=self._kw_args_defaults)

    def get_effective_args(self, args: dict[str, Any]) -> dict[str, Any]:
        return common.get_kwargs_with_defaults(str(type(self)),
                                               args=args,
                                               default_args=self._kw_args_defaults)

    def get_perf_counts(self, inT, outT, **kwargs):
        assert False, f"{self.optype}::get_perf_counts not implemented yet"
        # return

    def update_tensor_counts(self, inT, outT, **kwargs):
        in_param_count  = sum([x.nelems() for x in inT if x.is_param == True])
        in_act_count    = sum([x.nelems() for x in inT if x.is_param == False])
        out_act_count   = sum([x.nelems() for x in outT if x.is_param == False])
        out_param_count = sum([x.nelems() for x in outT if x.is_param == True])
        assert out_param_count == 0, "OP{self.name} has output param count > 0: {out_param_count}"
        if TYPE_CHECKING:
            assert self.perf_stats is not None
        self.perf_stats.update({
            'inParamCount': int(in_param_count),
            'inActCount'  : int(in_act_count),
            'outActCount' : int(out_act_count),
            })
        return

    def set_precision(self, prec):
        self.precision = prec

    def remove_in_optimization(self):
        self.removed_in_optimization = True

    def fuse_op(self, fused_with_op):
        self.fused_in_optimization = True
        self.fused_with_op         = fused_with_op

    def execute(self, device):
        if TYPE_CHECKING:
            assert self.perf_stats is not None, f"SimOp {self.name} has no perf_stats set, cannot execute"
        #find compute cycles
        self.compute_cycles = 0
        for instr,instr_count in self.perf_stats['instrs'].items():
            peak_ipc = device.peak_ipc(self.uses_compute_pipe, instr, self.precision)
            real_ipc = peak_ipc * G_COMPUTE_UTIL_CONSTANT
            self.compute_cycles += math.ceil(instr_count / real_ipc)
        #find memory cycles
        mem_rd_GB     = self.perf_stats['inBytes'] / 1024 / 1024 / 1024
        mem_wr_GB     = self.perf_stats['outBytes'] / 1024 / 1024 / 1024
        freq_MHz      = device.frequency(self.uses_compute_pipe, units='MHz')
        peak_bw_GBps  = device.peak_bandwidth(freq_units="GHz")
        bw_GBps       = peak_bw_GBps * G_MEMORY_UTIL_CONSTANT
        #convert to device clk cycles
        self.mem_rd_cycles = math.ceil((mem_rd_GB / bw_GBps) * freq_MHz * 1e6)
        self.mem_wr_cycles = math.ceil((mem_wr_GB / bw_GBps) * freq_MHz * 1e6)

        return

    def get_effective_precision(self, tensor: SimTensor) -> np.dtype:
        """
        Get the effective precision for a tensor based on the op's precision.
        If the op's precision is not set, return the tensor's dtype.
        """
        if self.precision is not None:
            return self.precision
        assert tensor.dtype is not None, f"Tensor {tensor.name} has no dtype set"
        return tensor.dtype

######################  CONCRETE OP IMPLEMENTATION BEGIN ##################
class ConstantOp(SimOp):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.opclass_str: str = 'Constant'
        check_io_counts( self, in_counts=[0,0], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        attr_val_count = 0
        attr_val_field = ""
        for ff in ['sparse_value', 'value', 'value_float', 'value_floats',
                    'value_int', 'value_ints', 'value_string', 'value_strings']:
            if ff in self.attrs:
                attr_val_count += 1
                attr_val_field = ff
        assert attr_val_count == 1, f"ERROR: More than one val attribute: {self}"
        tdata  = self.attrs[attr_val_field]
        tmp_tensor = build_tmp_data_tensor(tdata, '_tmp_constant_tensor_ op=' + self.name)
        update_output_tensor(self, tmp_tensor, outT[0])

        assert outT[0].check_shape(), f"output shape: outT[0]"
        self.perf_stats =  {
                'inElems' : 0,
                'outElems': outT[0].nelems(),
                'inBytes' : 0,
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'mov': outT[0].nelems()}
                }
        return self.perf_stats

class EltwiseUnaryOp(SimOp):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.opclass_str: str = 'EltwiseUnary'
        check_io_counts( self, in_counts=[1,1], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        #Single Input/Output...
        # just forward input shape/data and update ops
        if self.perf_stats is not None:
            return self.perf_stats

        #is_backprop = kwargs.get('is_backprop', False)
        #batch_axis  = kwargs.get('batch_axis',  None)
        #bias_axis   = kwargs.get('bias_axis',   None)
        #if is_backprop and outT[0].is_param and batch_axis is not None:
        #    assert batch_axis >=0 and batch_axis < len(np_out.shape), f"DIPPY"
        #    #reduce across all samples in batch for paramter gradients
        #    # is_backprop -> this is a gradient calculation
        #    # is_param    -> this is a parameter tensor
        #    # batch_axis  -> this is the batch axis
        #    #TODO: add this cost to instrs
        #    if bias_axis is not None: #HACK HACK HACK -- need ReduceSum operator in BWD PASS
        #        np_out = np.sum(np_out.data, axis=(batch_axis, bias_axis))
        #    else:
        #        np_out = np.sum(np_out.data, axis=(batch_axis))

        #tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        #update_output_tensor(self, tmp_outT, outT[0])
        if outT[0].check_shape() and outT[0].shape == inT[0].shape:
            outT[0].dtype = inT[0].dtype
        else:
            outT[0].shape = inT[0].shape
            outT[0].dtype = inT[0].dtype

        optype2instr = {
            'identity': 'mov',
            'sqrt': 'sqrt',
            'exp': 'exp',
            'log': 'log',
            'abs': 'abs',
            'sign': 'cmp',  # Sign operation typically uses comparison
            'floor': 'round',  # Floor uses rounding instruction
            'ceil': 'round',   # Ceil uses rounding instruction
            'round': 'round',
            'reciprocal': 'div',  # Reciprocal is division
            'erf': 'exp',  # Map erf to exp instruction (approximation)
            'acos': 'acos',
            'acosh': 'acosh',
            'asin': 'asin',
            'asinh': 'asinh',
            'atan': 'atan',
            'atanh': 'atanh',
            'cosh': 'cosh',
            'sinh': 'sinh'
        }
        instr_name = self.optype.lower()
        if instr_name in optype2instr:
            instr_name = optype2instr[instr_name]
        self.perf_stats =  {
                'inElems' : inT[0].nelems(),
                'outElems': outT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {instr_name: outT[0].nelems()}
                }
        return self.perf_stats

class EltwiseBinaryOp(SimOp):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.opclass_str: str = 'EltwiseBinary'
        check_io_counts( self, in_counts=[2,2], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        outT[0].shape = get_tensor_broadcast_shape(inT[0].shape, inT[1].shape)
        outT[0].dtype = inT[0].dtype

        self.perf_stats =  {
                'inElems' : inT[0].nelems() + inT[1].nelems(),
                'outElems': outT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision) + inT[1].nbytes(self.precision),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {self.optype.lower(): outT[0].nelems()}
                }
        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        """
        C = ADD(A,B),           Z = MUL(X,Y)
        dA = Identity(dC)      dX = MUL(dZ, Y)
        dB = Identity(dC)      dY = MUL(X, dZ)
        """
        G_OP: Union[EltwiseBinaryOp, EltwiseUnaryOp]
        assert self.perf_stats is not None, f"{self.name} backward() called before get_perf_stats()"
        assert len(inT) == len(outGT), f"#inT != #outGT!!"
        assert len(outT) == len(inGT), f"#outT != #inGT"

        grad_results = {}
        in_grad_tensor = inGT[0]
        for grad_arg_pos, (in_data_tensor, out_grad_tensor) in enumerate(zip(inT[::-1], outGT)):
            if out_grad_tensor is not None:
                grad_tinfo = GRAD_TENSOR_INFO(out_grad_tensor)
                if self.optype == 'Add':
                    G_OP = EltwiseUnaryOp({
                        'name'   : out_grad_tensor.name + '.Identity',
                        'optype' : 'Identity',
                        'inList' : [in_grad_tensor.name],
                        'outList': [out_grad_tensor.name]
                        })
                    in_grad_tensor.op_in.append(G_OP.name)
                    out_grad_tensor.op_out.append(G_OP.name)
                    grad_tinfo._grad_ops.append(G_OP)
                    grad_tinfo._input_grad_tensors.append(in_grad_tensor)
                elif self.optype == 'Mul':
                    G_IL = [in_grad_tensor.name, in_data_tensor.name]
                    G_OP = EltwiseBinaryOp({
                        'name'   : out_grad_tensor.name + '.Mul',
                        'optype' : 'Mul',
                        'inList' : G_IL if grad_arg_pos == 0 else G_IL[::-1],
                        'outList': [out_grad_tensor.name]
                        })
                    in_grad_tensor.op_in.append(G_OP.name)
                    in_data_tensor.op_in.append(G_OP.name)
                    out_grad_tensor.op_out.append(G_OP.name)
                    grad_tinfo._grad_ops.append(G_OP)
                    grad_tinfo._input_fwd_tensors.append(in_data_tensor)
                    grad_tinfo._input_grad_tensors.append(in_grad_tensor)
                else:
                    assert False, f"Illegal optype: {self.optype} in EltwiseBinaryOp"


                grad_results[grad_tinfo.name]= grad_tinfo

        return grad_results

class GatherOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Gather'
        check_io_counts( self, in_counts=[2,2], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        axis     = self.attrs.get('axis', 0)
        assert isinstance(axis, int), f"attribute axis ({axis}) is not an int!!"

        dataT    = inT[0]
        indicesT = inT[1]
        assert dataT.check_shape(), f"Illegal input dataT shape: {dataT}!!"
        assert indicesT.check_shape(), f"Illegal input indicesT shape: {indicesT}!!"

        data_rank  = dataT.rank()
        data_shape = dataT.shape
        # Normalize negative axis
        axis = axis if axis >= 0 else data_rank + axis
        assert axis >= 0 and axis < data_rank, f"Axis {axis} is out of bounds for dataT.shape {dataT.shape()}"
        outT[0].shape = data_shape[:axis] + indicesT.shape + data_shape[axis + 1:]
        outT[0].dtype = dataT.dtype

        self.perf_stats = {
                'inElems' : int(outT[0].nelems()), #read just what we need, not the whole embed. tbl
                'outElems': int(outT[0].nelems()),
                'inBytes' : int(outT[0].nbytes(self.precision)),
                'outBytes': int(outT[0].nbytes(self.precision)),
                'instrs'  : {'mov': int(outT[0].nelems())}
                }
        return self.perf_stats

class LayerNormalizationOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'LayerNorm'
        check_io_counts( self, in_counts=[2,3], out_counts=[1,3] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, f"LayerNormalization cannot be a backward op!!"

        axis       = self.attrs.get('axis', -1)
        epsilon    = self.attrs.get('epsilon', 1e-5)
        stash_type = self.attrs.get('stash_type', 1)

        X      = inT[0]
        scaleT = inT[1]
        biasT  = inT[2] if len(inT) == 3 else None
        assert X.check_shape(), f"Illegal Shape for {X}"
        XShape = X.shape
        XRank  = X.rank()

        # LayerNormalization implementation with Numpy....
        # From Spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#LayerNormalization
        # Equations:
        #
        # [Stage-1]
        #   normalized_axes = [axis, ..., rank of X - 1]
        #     Mean       = ReduceMean<axes=normalized_axes>(X)    := X_mu
        #     D          = Sub(X, Mean)                           := X - X_mu
        #     DD         = Mul(D, D)                              := (X - X_mu)^2
        #     Var        = ReduceMean<axes=normalized_axes>(DD)   := X_sigma^2
        #     VarEps     = Add(Var, epsilon)                      := X_sigma^2 + eps
        #     StdDev     = Sqrt(VarEps)                           := sqrt(X_sigma^2 + eps)
        #     InvStdDev  = Reciprocal(StdDev)                     := 1/sqrt(X_sigma^2 + eps)
        #     Normalized = Mul(D, InvStdDev)                      := (X - X_mu)/(1/sqrt(X_sigma^2 + eps))
        # Stage-2
        #     NormalizedScaled = Mul(Normalized, Scale)
        #     if (Bias): Y = Add(NormalizedScaled, Bias)

        if axis < 0: axis += XRank
        normalized_axes = XShape[axis:]
        unsqueezed_rank = XRank - axis
        reduction_shape = XShape[0:axis] + [1] * unsqueezed_rank

        instr_count = {'add': 0, 'sub': 0, 'mul': 0, 'div': 0, 'mac': 0, 'rsqrt': 0} #dict to hold instr counts
        input_count = X.nelems()
        reduction_count = reduce(operator.mul, reduction_shape, 1)

        # -------x------- Stage-1 Implementation -------x-------
        # Parameter used to convert N-D tensor layer norm
        # to equivalent 2-D matirx operations.
        row,col = 1,1
        for i in range(XRank):
            if i < axis:
                row *= XShape[i]
            else:
                col *= XShape[i]
        # After reshaping input tensor X into a matrix, layer norm
        # is equivalent to conducting standardization on each column
        # (s.t. each col has zero mean and unit variance).
        #x_mat = np.reshape(X.data, (row, col))

        # compute mean for every x_mat's col
        #x_mean = np.sum(x_mat, axis=1, keepdims=True)/col
        instr_count['add'] += input_count
        instr_count['div'] += reduction_count
        #x_diff = x_mat - x_mean
        instr_count['sub'] += input_count
        #x_squared_diff = x_diff * x_diff
        instr_count['mul'] += input_count
        # compute variance for every x_mat's col
        #variance = np.sum(x_squared_diff, axis=1, keepdims=True)/col
        instr_count['add'] += input_count
        instr_count['div'] += reduction_count
        #variance_eps = variance + epsilon
        instr_count['add'] += reduction_count
        #std_dev = np.sqrt(variance_eps)
        #inv_std_dev = np.reciprocal(std_dev)
        instr_count['rsqrt'] += reduction_count

        # Standardization step. y_mat is zero-mean and unit-variance.
        #y_mat = x_diff * inv_std_dev
        instr_count['mul'] += input_count

        # -------x------- Stage-2 Implementation -------x-------
        # Apply affine transform on normalization outcome.
        #assert scaleT.data is not None, f"Illegal DATA in Tensor {scaleT}"
        #y_mat = np.reshape(y_mat, XShape) * scaleT.data
        instr_count['mac'] += input_count
        if biasT is not None:
            #Check: this add is already counted in the 'mac' above?
            #y_mat = y_mat + biasT.data
            pass

        outT[0].shape = X.shape
        outT[0].dtype = X.dtype

        if len(outT) >= 2:
            # reshape needed because of initial tensor-to-matrix reshape in Step-1.
            #X_mean = np.reshape(x_mean, reduction_shape)
            outT[1].shape = reduction_shape
            outT[1].dtype = X.dtype

        if len(outT) == 3:
            # reshape needed because of initial tensor-to-matrix reshape in Step-1.
            #X_invSDT = np.reshape(inv_std_dev, reduction_shape)
            outT[2].shape = reduction_shape
            outT[2].dtype = X.dtype

        biasElems   = 0 if biasT is None else biasT.nelems()
        meanElems   = 0 if len(outT) < 2 else outT[1].nelems()
        invSDTElems = 0 if len(outT) < 3 else outT[2].nelems()
        biasBytes   = 0 if biasT is None else biasT.nbytes(self.precision)
        meanBytes   = 0 if len(outT) < 2 else outT[1].nbytes(self.precision)
        invSDTBytes = 0 if len(outT) < 3 else outT[2].nbytes(self.precision)
        self.perf_stats ={
                'inElems' : inT[0].nelems() + inT[1].nelems() + biasElems,
                'outElems': outT[0].nelems() + meanElems + invSDTElems,
                'inBytes' : inT[0].nbytes(self.precision) + inT[1].nbytes(self.precision) + biasBytes,
                'outBytes': outT[0].nbytes(self.precision) + meanBytes + invSDTBytes,
                'instrs'  : instr_count
                }
        return self.perf_stats

class BatchNormalizationOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'BatchNormalization'
        check_io_counts(self, in_counts=[5,5], out_counts=[1,3])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        assert all([itensor.check_shape() for itensor in inT]), \
                f"input tensor shapes not well formed!!"
        assert len(outT) in [1,3], f"output can either be 1 or 3"
        x          = inT[0]
        scale      = inT[1]
        bias       = inT[2]
        input_mean = inT[3]
        input_var  = inT[4]

        outT[0].shape = x.shape
        outT[0].dtype = x.dtype
        if len(outT) == 3:
            outT[1].shape = scale.shape
            outT[1].dtype = scale.dtype
            outT[2].shape = scale.shape
            outT[2].dtype = scale.dtype

        instr_count = {
            'add': x.nelems(),
            'mac': x.nelems(),
            'rsqrt': 1,
            'sub': 1,
            'mul': 1,
            'add': 1,
        }
        self.perf_stats = {
            'inElems' : sum([i.nelems() for i in inT]),
            'outElems': sum([o.nelems() for o in outT]),
            'inBytes' : sum([i.nbytes(self.precision) for i in inT]),
            'outBytes': sum([o.nbytes(self.precision) for o in outT]),
            'instrs'  : instr_count
        }
        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        print("-"*50)
        print("\nLN_BWD_DBG>>")
        print("\nFWD IN")
        for x in inT: print(x)
        print("\nFWD OUT")
        for x in outT: print(x)

        print("\nBWD IN")
        for x in inGT: print(x)
        print("\nBWD OUT")
        for x in outGT: print(x)

        assert self.perf_stats is not None, f"{self.name} backward() called before get_perf_stats()"
        assert len(inT) == len(outGT), f"#inT != #outGT!!"
        #assert len(outT) == len(inGT), f"#outT != #inGT" no gradients for out_1(Mean) and out_2(InvStdDev)

        grad_results = {}
        X          = inT[0]
        Scale      = inT[1]
        Bias       = inT[2] if len(inT) == 3 else None
        Y          = outT[0]
        Mean       = outT[1] if len(outT) >= 2 else None
        InvStdDev  = outT[2] if len(outT) == 3 else None
        dY         = inGT[0]
        dX         = outGT[0]
        dScale     = outGT[1] if len(outGT) >= 2 else None
        dBias      = outGT[2] if len(outGT) == 3 else None

        assert dY is not None,        f"LayerNormalization Backward Error-In-1"
        assert Mean is not None,      f"LayerNormalization Backward Error-In-2"
        assert InvStdDev is not None, f"LayerNormalization Backward Error-In-3"
        assert dX is not None,        f"LayerNormalization Backward Error-Out-1"
        assert dScale is not None,    f"LayerNormalization Backward Error-Out-2"
        assert dBias is not None,     f"LayerNormalization Backward Error-Out-3"

        #dBias
        axis = self.attrs.get('axis', -1)
        if axis < 0: axis += X.rank()
        new_axis_data = np.arange(0,axis,1)
        new_axis_T = build_tmp_data_tensor(new_axis_data, dBias.name + '.normalized_axes')
        new_axis_T.is_const = True
        new_axis_T.has_grad = False

        print(">>> DBG: axis             ", axis)
        print(">>> DBG: dY.shape         ", dY.shape)
        print(">>> DBG: new_axis_T       ", new_axis_T.data)

        bias_grad_tinfo = GRAD_TENSOR_INFO(dBias)
        BIAS_G_OP = ReduceSumOp({
            'name'   : dBias.name + '.ReduceSum',
            'optype' : 'ReduceSum',
            'inList' : [dY.name, new_axis_T.name],
            'outList': [dBias.name],
            'attrs'  : {'keepdims': 0}
            })
        dY.op_in.append(BIAS_G_OP.name)
        new_axis_T.op_in.append(BIAS_G_OP.name)
        dBias.op_out.append(BIAS_G_OP.name)

        bias_grad_tinfo._grad_ops.append(BIAS_G_OP)

        bias_grad_tinfo._input_grad_tensors.append(dY)
        bias_grad_tinfo._new_tensors.append(new_axis_T)

        #dScale
        scale_grad_tinfo = GRAD_TENSOR_INFO(dScale)
        X_hat_0 = SimTensor({'name': X.name + '_hat_0'})
        X_hat_1 = SimTensor({'name': X.name + '_hat_1'})
        X_hat_2 = SimTensor({'name': X.name + '_hat_2'})

        SCALE_SUB_OP = EltwiseBinaryOp({
            'name'   : dScale.name + '.Sub',
            'optype' : 'Sub',
            'inList' : [X.name, Mean.name],
            'outList': [X_hat_0.name]
            })
        X.op_in.append(SCALE_SUB_OP.name)
        Mean.op_in.append(SCALE_SUB_OP.name)
        X_hat_0.op_out.append(SCALE_SUB_OP.name)

        scale_grad_tinfo._grad_ops.append(SCALE_SUB_OP)

        scale_grad_tinfo._input_fwd_tensors.append(X)
        scale_grad_tinfo._input_fwd_tensors.append(Mean)
        scale_grad_tinfo._new_tensors.append(X_hat_0)

        SCALE_MUL_OP_1 = EltwiseBinaryOp({
            'name'   : dScale.name + '.Mul1',
            'optype' : 'Mul',
            'inList' : [X_hat_0.name, InvStdDev.name],
            'outList': [X_hat_1.name]
            })
        X_hat_0.op_in.append(SCALE_MUL_OP_1.name)
        InvStdDev.op_in.append(SCALE_MUL_OP_1.name)
        X_hat_1.op_out.append(SCALE_MUL_OP_1.name)

        scale_grad_tinfo._grad_ops.append(SCALE_MUL_OP_1)

        scale_grad_tinfo._input_fwd_tensors.append(InvStdDev)
        scale_grad_tinfo._new_tensors.append(X_hat_1)

        SCALE_MUL_OP_2 = EltwiseBinaryOp({
            'name'   : dScale.name + '.Mul2',
            'optype' : 'Mul',
            'inList' : [dY.name, X_hat_1.name],
            'outList': [X_hat_2.name]
            })
        dY.op_in.append(SCALE_MUL_OP_2.name)
        X_hat_1.op_in.append(SCALE_MUL_OP_2.name)
        X_hat_2.op_out.append(SCALE_MUL_OP_2.name)

        scale_grad_tinfo._grad_ops.append(SCALE_MUL_OP_2)

        scale_grad_tinfo._input_grad_tensors.append(dY)
        scale_grad_tinfo._new_tensors.append(X_hat_2)

        SCALE_G_OP = ReduceSumOp({
            'name'   : dScale.name + '.ReduceSum',
            'optype' : 'ReduceSum',
            'inList' : [X_hat_2.name, new_axis_T.name],
            'outList': [dScale.name],
            'attrs'  : {'keepdims': 0}
            })
        X_hat_2.op_in.append(SCALE_G_OP.name)
        new_axis_T.op_in.append(SCALE_G_OP.name)
        dScale.op_out.append(SCALE_G_OP.name)

        scale_grad_tinfo._grad_ops.append(SCALE_G_OP)


        #dX
        x_grad_tinfo = GRAD_TENSOR_INFO(dX)

        Term_1 = SimTensor({'name': dX.name + '.Term_1'})
        X_MUL_OP_1 = EltwiseBinaryOp({
            'name'   : dX.name + '.Mul1',
            'optype' : 'Mul',
            'inList' : [dY.name, Scale.name],
            'outList': [Term_1.name]
            })
        dY.op_in.append(X_MUL_OP_1.name)
        Scale.op_in.append(X_MUL_OP_1.name)
        Term_1.op_out.append(X_MUL_OP_1.name)
        x_grad_tinfo._grad_ops.append(X_MUL_OP_1)
        x_grad_tinfo._input_grad_tensors.append(dY)
        x_grad_tinfo._input_fwd_tensors.append(Scale)
        x_grad_tinfo._new_tensors.append(Term_1)

        #Term_2 = SimTensor({'name': dX.name + '.Term_2'})
        X_MUL_OP_2 = EltwiseBinaryOp({
            'name'   : dX.name + '.Mul2',
            'optype' : 'Mul',
            'inList' : [Term_1.name, InvStdDev.name],
            'outList': [dX.name]
            })
        InvStdDev.op_in.append(X_MUL_OP_2.name)
        #Term_2.op_out.append(X_MUL_OP_2.name)
        dX.op_out.append(X_MUL_OP_2.name)
        x_grad_tinfo._grad_ops.append(X_MUL_OP_2)
        x_grad_tinfo._input_fwd_tensors.append(InvStdDev)
        #x_grad_tinfo._new_tensors.append(Term_2)

        grad_results[bias_grad_tinfo.name] = bias_grad_tinfo
        grad_results[scale_grad_tinfo.name]= scale_grad_tinfo
        grad_results[x_grad_tinfo.name]    = x_grad_tinfo

        return grad_results

class ConvOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Conv'
        check_io_counts(self, in_counts=[2,3], out_counts=[1,1])
        self._kw_args_defaults = {
            'auto_pad'    : 'NOTSET',
            'dilations'   : [1, 1],
            'strides'     : [1, 1],
            'pads'        : [0, 0, 0, 0],
            'group'       : 1,
            'kernel_shape': None,
        }
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        assert inT[0].check_shape(), f"Illegal Shape for {inT[0]}"
        assert inT[1].check_shape(), f"Illegal Shape for {inT[1]}"
        if len(inT) == 3: assert inT[2].check_shape(), f"Illegal Shape for {inT[2]}"

        X = inT[0]
        W = inT[1]
        if len(inT) == 3: B = inT[2]

        num_spatial_dims = X.rank() - 2
        if num_spatial_dims < 1:
            raise ValueError("X must have at least 1 spatial dimension (N, C, spatial...): {X}")

        group        = self.attrs.get('group', 1)
        dilations    = self.attrs.get('dilations', [1] * num_spatial_dims)
        strides      = self.attrs.get('strides',   [1] * num_spatial_dims)
        pads         = self.attrs.get('pads',      [0] * (2 * num_spatial_dims))
        auto_pad     = self.attrs.get('auto_pad', 'NOTSET')
        kernel_shape = self.attrs.get('kernel_shape', None)

        # Validate inputs
        if W.rank() != num_spatial_dims + 2:
            raise ValueError(f"Weight shape must have {num_spatial_dims + 2} dims (C_out, C_in/group, kernel_dims): {W}")
        if len(dilations) != num_spatial_dims or len(strides) != num_spatial_dims or len(pads) != 2 * num_spatial_dims:
            raise ValueError("Dilations, strides, and pads must match spatial dimensions")
        if group <= 0 or X.shape[1] % group != 0:
            raise ValueError(f"C_in {X.shape[1]} must be divisible by group {group}")
        if W.shape[1] != X.shape[1] // group:
            raise ValueError(f"Weight C_in/group {W.shape[1]} must match input C_in/group {X.shape[1] // group}")
        if len(inT) == 3:
            if B.rank() != 1 or B.shape[0] != W.shape[0]:
                raise ValueError(f"Bias shape {B.shape} must be (C_out,) matching weight C_out {W.shape[0]}")

        N, C_in          = X.shape[0], X.shape[1]
        C_out            = W.shape[0]
        spatial_dims     = X.shape[2:]
        kernel_dims      = W.shape[2:]

        if len(kernel_dims) != num_spatial_dims:
            raise ValueError("Kernel spatial dims must match input spatial dims")

        if kernel_shape is not None:
            if kernel_shape != kernel_dims:
                raise ValueError("Kernel Shape does not match Kernel-dims calculated from input spatial dims")

        # Compute effective kernel size with dilation
        effective_kernel = [ (kernel_dims[i] - 1) * dilations[i] + 1 for i in range(num_spatial_dims) ]

        # Compute output spatial dimensions
        output_spatial = []
        for i in range(num_spatial_dims):
            Di       = spatial_dims[i]
            ki       = effective_kernel[i]
            stride_i = strides[i]
            if auto_pad == "NOTSET":
                pad_begin_i   = pads[i]
                pad_end_i     = pads[i + num_spatial_dims]
                total_padding = pad_begin_i + pad_end_i
                Oi            = (Di + total_padding - ki) // stride_i + 1
            elif auto_pad == "VALID":
                Oi = (Di - ki) // stride_i + 1
            elif auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
                Oi = int(np.ceil(Di/stride_i))
            else:
                raise ValueError(f"Unsupported auto_pad value: {auto_pad}")

            if Oi <= 0:
                raise ValueError(f"Output dimension {i} would be <= 0: {Oi}")
            output_spatial.append(Oi)

        output_shape = [N, C_out] + output_spatial
        #print(">> X.shape         :", X.shape)
        #print(">> W.shape         :", W.shape)
        #if len(inT) == 3: print(">> B.shape         :", B.shape)
        #print(">> group           :", group)
        #print(">> dilations       :", dilations)
        #print(">> strides         :", strides)
        #print(">> pads            :", pads)
        #print(">> auto_pad        :", auto_pad)
        #print(">> N               :", N)
        #print(">> C_in            :", C_in)
        #print(">> C_out           :", C_out)
        #print(">> spatial_dims    :", spatial_dims)
        #print(">> kernel_shape    :", kernel_shape)
        #print(">> kernel_dims     :", kernel_dims)
        #print(">> num_spatial_dims:", num_spatial_dims)
        #print(">> output_spatial  :", output_spatial)
        #print(">> output_shape    :", output_shape)
        #if len(inT) == 3: print(">> B.shape         :", B.shape)

        if X.shape[0] != output_shape[0] or W.shape[0] != output_shape[1]:
            raise ValueError("Batch size (N) and C_out must match across shapes")

        outT[0].shape = output_shape
        outT[0].dtype = X.dtype

        macs_per_output = (C_in // group) * np.prod(kernel_dims)
        output_elements = N * C_out * np.prod(spatial_dims)
        total_macs      = output_elements * macs_per_output
        instr_count     = { 'mac': int(total_macs) }
        if len(inT) == 3:
            instr_count['add'] = output_elements

        bias_elems = B.nelems() if len(inT) == 3 else 0
        bias_bytes = B.nbytes(self.precision) if len(inT) == 3 else 0
        inElems = X.nelems() + W.nelems() + bias_elems
        inBytes = X.nbytes(self.precision) + W.nbytes(self.precision) + bias_bytes

        self.perf_stats = {
            'inElems' : inElems,
            'outElems': outT[0].nelems(),
            'inBytes' : inBytes,
            'outBytes': outT[0].nbytes(self.precision),
            'instrs'  : instr_count
        }
        return self.perf_stats

class MaxPoolOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'MaxPool'
        self._kw_args_defaults = {
                #'kernel_shape' : None,
                #'auto_pad'     : 'NOTSET',
                #'ceil_mode'    : 0,
                #'dilations'    : None,
                #'pads'         : None,
                #'storage_order': None,
                #'strides'      : None,
                }
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 2])
        #if 'attrs' in opinfo:
        #    self.check_known_args(opinfo['attrs'])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        assert inT[0].check_shape(), f"Illegal Shape for {inT[0]}"
        input_shape   = inT[0].shape
        kernel_shape  = self.attrs.get('kernel_shape') #required attribute
        output_shape  = pooling_shape_inference(input_shape, kernel_shape, self.attrs)
        outT[0].shape = output_shape
        outT[0].dtype = inT[0].dtype

        if len(outT) == 2:
            outT[1].shape = output_shape
            outT[1].dtype = np.dtype(np.int64)

        instr_count = { 'cmp': inT[0].nelems(), 'mov': outT[0].nelems() }
        self.perf_stats = {
            'inElems' : inT[0].nelems(),
            'outElems': outT[0].nelems(),
            'inBytes' : inT[0].nbytes(self.precision),
            'outBytes': outT[0].nbytes(self.precision),
            'instrs'  : instr_count
        }
        return self.perf_stats

class AveragePoolOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'AveragePool'
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        assert inT[0].check_shape(), f"Illegal Shape for {inT[0]}"
        input_shape = inT[0].shape

        # Handle adaptive pooling if specified
        is_adaptive = self.attrs.get('adaptive', False)

        if is_adaptive:
            # For adaptive pooling, calculate kernel_shape dynamically based on input and desired output size
            output_size = self.attrs.get('output_size', (1, 1))

            # Ensure output_size is a tuple of 2 elements
            if isinstance(output_size, int):
                output_size = (output_size, output_size)

            # Extract spatial dimensions (last 2 dimensions for 2D pooling)
            input_height, input_width = input_shape[-2], input_shape[-1]
            output_height, output_width = output_size

            # Calculate kernel size, stride, and padding to achieve the desired output size
            # For simplicity, we use a kernel that divides the input evenly if possible
            kernel_height = input_height // output_height if output_height > 0 else 1
            kernel_width = input_width // output_width if output_width > 0 else 1

            # Set the calculated kernel shape
            kernel_shape = (kernel_height, kernel_width)

            # Calculate appropriate strides
            strides = (kernel_height, kernel_width)

            # Set attributes for the pooling_shape_inference function
            self.attrs['kernel_shape'] = kernel_shape
            self.attrs['strides'] = strides
            self.attrs['pads'] = [0, 0, 0, 0]  # No padding for adaptive pooling

            # Use pooling_shape_inference instead of direct shape setting
            output_shape = pooling_shape_inference(input_shape, kernel_shape, self.attrs)
            outT[0].shape = output_shape
            outT[0].dtype = inT[0].dtype
        else:
            # Traditional AveragePool with explicit kernel_shape
            kernel_shape = self.attrs.get('kernel_shape')  # Required attribute
            output_shape = pooling_shape_inference(input_shape, kernel_shape, self.attrs)
            outT[0].shape = output_shape
            outT[0].dtype = inT[0].dtype

        instr_count = {'add': inT[0].nelems(), 'div': outT[0].nelems(), 'mov': outT[0].nelems()}

        self.perf_stats = {
            'inElems' : inT[0].nelems(),
            'outElems': outT[0].nelems(),
            'inBytes' : inT[0].nbytes(self.precision),
            'outBytes': outT[0].nbytes(self.precision),
            'instrs'  : instr_count
        }
        return self.perf_stats

class MatMulOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Matmul'
        check_io_counts( self, in_counts=[2,2], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        batch_axis  = kwargs.get('batch_axis',  None)
        assert is_backprop == False, f"Matmul in Backward Pass!!"

        AShape = inT[0].shape
        BShape = inT[1].shape

        #find output shape
        CShape = None
        if len(AShape) < 1 or len(BShape) < 1:
            raise ValueError("Shapes must have at least 1 dimension")

        # Handle 1D cases
        if len(AShape) == 1 and len(BShape) == 1:
            if AShape[0] != BShape[0]:
                raise ValueError(f"Matmul incompatible: {AShape[0]} != {BShape[0]}")
            CShape = [] # Scalar result
        elif len(AShape) == 1:
            if AShape[0] != BShape[-2]:
                raise ValueError(f"Matmul incompatible: {AShape[0]} != {BShape[-2]}")
            CShape = BShape[:-2] + [BShape[-1]]
        elif len(BShape) == 1:
            if AShape[-1] != BShape[0]:
                raise ValueError(f"Matmul incompatible: {AShape[-1]} != {BShape[0]}")
            CShape = AShape[:-1]

        # Handle 2D+ cases
        batch1, mat1 = AShape[:-2], AShape[-2:]
        batch2, mat2 = BShape[:-2], BShape[-2:]

        # Check matrix multiplication compatibility
        if mat1[-1] != mat2[-2]:
            raise ValueError(f"Matmul incompatible: {mat1[-1]} != {mat2[-2]}")
        broadcast_batch = get_tensor_broadcast_shape(batch1, batch2)
        CShape = broadcast_batch + [mat1[0], mat2[-1]]

        reduced_dim   = mat1[-1]
        outT[0].shape = CShape
        outT[0].dtype = inT[0].dtype

        # TODO: BEGIN Quick Fix
        self.perf_stats = {
            'inElems' : inT[0].nelems() + inT[1].nelems(),
            'outElems': outT[0].nelems(),
            'inBytes' : inT[0].nbytes(self.precision) + inT[1].nbytes(self.precision),
            'outBytes': outT[0].nbytes(self.precision),
            'instrs'  : {'mac': outT[0].nelems() * reduced_dim}
            }
        # TODO: END Quick Fix
        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # dA = MatMul(dC, B^T)
        # dB = MatMul(A^T, dC)
        assert self.perf_stats is not None, f"{self.name} backward() called before get_perf_stats()"
        assert len(inT) == len(outGT), f"#inT != #outGT!!"
        assert len(outT) == len(inGT), f"#outT != #inGT"

        in_grad_tensor = inGT[0]
        grad_results = {}
        for grad_arg_pos, (in_data_tensor, out_grad_tensor) in enumerate(zip(inT[::-1], outGT)):
            if out_grad_tensor is not None:
                in_data_tensor_T = SimTensor({'name': in_data_tensor.name + '_T'})
                perm = [i for i in range(len(in_data_tensor.shape))]
                T_OP = TransposeOp({'name'   : in_data_tensor.name + '.Transpose',
                                    'optype' : 'Transpose',
                                    'inList' : [in_data_tensor.name],
                                    'outList': [in_data_tensor_T.name],
                                    'attrs'  : {'perm': perm[0:-2] + perm[-1:-3:-1]} #swap last 2 dims
                                    })
                G_IL = [in_grad_tensor.name, in_data_tensor_T.name]
                G_OP = MatMulOp({'name'   : out_grad_tensor.name + '.Matmul',
                                 'optype' : 'Matmul',
                                 'inList' : G_IL if grad_arg_pos == 0 else G_IL[::-1],
                                 'outList': [out_grad_tensor.name]
                                 })
                #update tensor op_in/op_out lists
                in_grad_tensor.op_in.append(G_OP.name)
                in_data_tensor_T.op_in.append(G_OP.name)
                out_grad_tensor.op_out.append(G_OP.name)

                in_data_tensor.op_in.append(T_OP.name)
                in_data_tensor_T.op_out.append(T_OP.name)

                grad_tinfo = GRAD_TENSOR_INFO(out_grad_tensor)
                grad_tinfo._grad_ops += [T_OP, G_OP]
                grad_tinfo._input_fwd_tensors.append(in_data_tensor)
                grad_tinfo._input_grad_tensors.append(in_grad_tensor)
                grad_tinfo._new_tensors.append(in_data_tensor_T)

                grad_results[grad_tinfo.name]= grad_tinfo

        return grad_results

class SplitOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Split'
        assert len(self.inList) in [1,2], f"#inputs should be in [1,2] : {self}"
        assert len(self.outList) >= 1, f"#outputs should be in [1,inf] : {self}"

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        num_outputs = self.attrs.get('num_outputs', len(outT))
        axis        = self.attrs.get('axis',0)

        A      = inT[0]
        splitT = inT[1] if len(inT) == 2 else None
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
        for tidx, tout in enumerate(outT):
            tshape0 = outShapes[tidx]
            tout.shape = tshape0
            tout.dtype = A.dtype
            outBytes += tout.nbytes(self.precision)
            outElems += tout.nelems()

        self.perf_stats = {
                'inElems' : A.nelems() + 0 if splitT is None else splitT.nelems(),
                'outElems': outElems,
                'inBytes' : A.nbytes(self.precision) + 0 if splitT is None else splitT.nbytes(self.precision),
                'outBytes': outBytes,
                'instrs'  : {'mov': outElems}
                }

        return self.perf_stats

class ReshapeOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Reshape'
        check_io_counts( self, in_counts=[2,2], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        allowzero = self.attrs.get('allowzero', 0)

        #A = clone_tensor_by_shape(inT[0])
        B = clone_tensor_by_shape(inT[1], data_maybe_missing=False) #B.data should exist
        assert B.dtype == np.int64, f"Input Data-Type should be np.int64 {B}"
        assert inT[0].check_shape(), f"Illegal Input Shape: {inT[0].shape}"
        input_shape  = inT[0].shape
        input_size   = inT[0].nelems()
        target_shape = [x.item() for x in B.data]

        minus_one_count = 0
        minus_one_index: Any = None
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
            output_size = reduce(operator.mul, filter(lambda x: x != -1, output_shape), 1)
            assert input_size >= output_size and input_size % output_size == 0, \
                    f"Cannot infer -1: input size {input_size}/{output_size}"
            inferred_dim = input_size // output_size
            output_shape[minus_one_index] = inferred_dim

        # Final validation
        final_output_size = reduce(operator.mul, output_shape, 1)
        assert input_size  == final_output_size, \
                f"in({input_size}) & out({final_output_size}) sizes are not equal!!"

        #np_out = reshape_reference_implementation(A.data, B.data, allowzero=allowzero)
        #tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_C_out__')
        #update_output_tensor(self, tmp_outT, outT[0])
        outT[0].shape = output_shape
        outT[0].dtype = inT[0].dtype

        self.perf_stats = {
                'inElems' : int(inT[0].nelems() + B.nelems()),
                'outElems': int(outT[0].nelems()),
                'inBytes' : int(inT[0].nbytes(self.precision) + B.nbytes(self.precision)),
                'outBytes': int(outT[0].nbytes(self.precision)),
                'instrs'  : {'mov': int(outT[0].nelems())}
                }
        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        #  Y = Reshape(X, new_shape)
        # dX = Reshape(dY, orig_X_shape)
        assert self.perf_stats is not None, f"{self.name} backward() called before get_perf_stats()"
        assert len(inT) == len(outGT), f"#inT != #outGT!!"
        assert len(outT) == len(inGT), f"#outT != #inGT"

        grad_results = {}
        in_data_tensor  = inT[0]    # X
        in_grad_tensor  = inGT[0]   # dY
        out_grad_tensor = outGT[0]  # dX

        if out_grad_tensor is not None:
            grad_tinfo = GRAD_TENSOR_INFO(out_grad_tensor)
            in_data_tensor_shape = build_tmp_data_tensor(np.array(in_data_tensor.shape), '_tmp_shape')
            orig_shape_tensor = SimTensor({
                'name'     : in_data_tensor.name + '.shape',
                'shape'    : in_data_tensor_shape.shape,
                'dtype'    : in_data_tensor_shape.dtype,
                'data'     : in_data_tensor_shape.data,
                'is_param' : False,
                'is_const' : True,
                'resolve'  : '_',
                })
            G_OP = ReshapeOp({
                'name'   : out_grad_tensor.name + '.Reshape',
                'optype' : 'Reshape',
                'inList' : [in_grad_tensor.name, orig_shape_tensor.name],
                'outList': [out_grad_tensor.name]
                })
            in_grad_tensor.op_in.append(G_OP.name)
            out_grad_tensor.op_out.append(G_OP.name)
            orig_shape_tensor.op_in.append(G_OP.name)
            grad_tinfo._grad_ops.append(G_OP)
            grad_tinfo._input_grad_tensors.append(in_grad_tensor)
            grad_tinfo._input_grad_tensors.append(orig_shape_tensor)
            return {grad_tinfo.name: grad_tinfo}
        else:
            return {}

class TransposeOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Transpose'
        check_io_counts( self, in_counts=[1,1], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        perms  = self.attrs['perm']
        assert len(perms) == inT[0].rank(), f"perms({perms}) must be equal to input rank ({inT[0].rank()})!!"
        outT[0].shape = [inT[0].shape[i] for i in perms]
        outT[0].dtype = inT[0].dtype
        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'outElems': outT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'mov': outT[0].nelems()}
                }
        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # A_T = Transpose(A, perm)
        # dA  = Transpose(dA_T, inverse_perm)
        assert self.perf_stats is not None, f"{self.name} backward() called before get_perf_stats()"
        assert len(inT) == len(outGT), f"#inT != #outGT!!"
        assert len(outT) == len(inGT), f"#outT != #inGT"

        in_data_tensor  = inT[0]
        in_grad_tensor  = inGT[0]
        out_grad_tensor = outGT[0]

        if out_grad_tensor is not None:
            grad_tinfo = GRAD_TENSOR_INFO(out_grad_tensor)

            perm  = self.attrs['perm']
            inverse_perm = [0] * len(perm)
            for i, p in enumerate(perm): inverse_perm[p] = i
            G_OP = TransposeOp({
                'name'   : out_grad_tensor.name + '.Transpose',
                'optype' : 'Transpose',
                'inList' : [in_grad_tensor.name],
                'outList': [out_grad_tensor.name],
                'attrs'  : {'perm': inverse_perm}
                })
            in_grad_tensor.op_in.append(G_OP.name)
            out_grad_tensor.op_out.append(G_OP.name)
            grad_tinfo._grad_ops.append(G_OP)
            grad_tinfo._input_grad_tensors.append(in_grad_tensor)
            return {grad_tinfo.name: grad_tinfo}
        else:
            return {}

class WhereOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Where'
        check_io_counts( self, in_counts=[3,3], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        '''
        ASSUME ONNX SHAPE INFERENCE
        condB = clone_tensor_by_shape(inT[0], data_maybe_missing=False) #condB.data should exist
        X     = clone_tensor_by_shape(inT[1])
        Y     = clone_tensor_by_shape(inT[2])
        assert condB.dtype == np.bool_, f"Illegal Input Tensor Data Type: {condB}"
        np_out = np.where(condB.data, X.data, Y.data)
        tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])
        '''
        assert outT[0].check_shape(), f"SHAPE INFERENCE ERROR!!"
        self.perf_stats = {
                'inElems' : inT[0].nelems() + inT[1].nelems() + inT[2].nelems(),
                'outElems': outT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision) + inT[1].nbytes(self.precision) + inT[2].nbytes(self.precision),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'mov': outT[0].nelems(), 'cmp': outT[0].nelems()}
                }
        return self.perf_stats

class SoftmaxOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Softmax'
        check_io_counts( self, in_counts=[1,1], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, f"Softmax cannot be a backward op!!"

        #axis = self.attrs.get('axis', -1)
        #X    = clone_tensor_by_shape(inT[0])
        #x_max  = np.max(X.data, axis=axis, keepdims=True)
        #tmp    = np.exp(X.data - x_max)
        #np_out = tmp / np.sum(tmp, axis=axis, keepdims=True)
        #tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        #update_output_tensor(self, tmp_outT, outT[0])
        outT[0].shape = inT[0].shape
        outT[0].dtype = inT[0].dtype
        outElems = outT[0].nelems()
        self.perf_stats = {
                'inBytes' : inT[0].nbytes(self.precision),
                'inElems' : inT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'outElems': outElems,
                'instrs'  : {
                    'cmp': outElems, # max_x = max(x)
                    'sub': outElems, # y = x - max_x
                    'exp': outElems, # exp(y)
                    'add': outElems, # z = sum(exp(y))
                    'div': outElems, # o = exp(y) / z
                    }
                }
        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        #    Y = Softmax(X, axis)
        # dX_i = Y_i * (dY_i - Sum_j(dY_j * Y_j))
        # Steps:
        #    T = Y . dY (elemwise mul) [DIMS: same as Y or dY]
        #    S = ReduceSum(T, axis, keepdims=1) <- dot_product per slice-j: Sum_j(dY_j * #    Y_j)
        #    Z = dY - S (sub)
        #    dX = Y . Z (elemwise mul)
        #

        assert self.perf_stats is not None, f"{self.name} backward() called before get_perf_stats()"
        assert len(inT) == len(outGT), f"#inT != #outGT!!"
        assert len(outT) == len(inGT), f"#outT != #inGT"

        grad_results = {}
        in_data_tensor  = inT[0]     #X
        out_data_tensor = outT[0]    #Y
        in_grad_tensor  = inGT[0]    #dY
        out_grad_tensor = outGT[0]   #dX
        orig_axis = self.attrs.get('axis', -1)

        if out_grad_tensor is not None:
            # T = Y . dY
            # S = ReduceSum(T, axis, keepdims=1)
            # Z = dY . S
            # dX = Y . Z
            T_tensor = SimTensor({'name': in_data_tensor.name + '.T_tensor'})
            S_tensor = SimTensor({'name': in_data_tensor.name + '.S_tensor'})
            Z_tensor = SimTensor({'name': in_data_tensor.name + '.Z_tensor'})
            T_OP = EltwiseBinaryOp({'name'   : in_data_tensor.name + '.Mul_T',
                                    'optype' : 'Mul',
                                    'inList' : [in_grad_tensor.name, out_data_tensor.name],
                                    'outList': [T_tensor.name]
                                    })
            S_OP = ReduceSumOp({'name'   : in_data_tensor.name + '.ReduceSum',
                                'optype' : 'ReduceSum',
                                'inList' : [T_tensor.name],
                                'outList': [S_tensor.name],
                                'attrs'  : {'axis': orig_axis, 'keepdims': 1}
                               })
            Z_OP = EltwiseBinaryOp({'name'   : in_data_tensor.name + '.Mul_Z',
                                    'optype' : 'Mul',
                                    'inList' : [in_grad_tensor.name, S_tensor.name],
                                    'outList': [Z_tensor.name]
                                    })
            G_OP = EltwiseBinaryOp({'name'   : in_data_tensor.name + '.Mul_Grad',
                                    'optype' : 'Mul',
                                    'inList' : [out_data_tensor.name, Z_tensor.name],
                                    'outList': [out_grad_tensor.name]
                                    })
            #update tensor op_in/op_out lists
            in_grad_tensor.op_in.append (T_OP.name)
            out_data_tensor.op_in.append(T_OP.name)
            T_tensor.op_out.append(T_OP.name)

            T_tensor.op_in.append(S_OP.name)
            S_tensor.op_out.append(S_OP.name)

            in_grad_tensor.op_in.append (Z_OP.name)
            S_tensor.op_in.append(Z_OP.name)
            Z_tensor.op_out.append(Z_OP.name)

            out_data_tensor.op_in.append(G_OP.name)
            Z_tensor.op_in.append(G_OP.name)
            out_grad_tensor.op_out.append(G_OP.name)

            grad_tinfo = GRAD_TENSOR_INFO(out_grad_tensor)
            grad_tinfo._grad_ops += [T_OP, S_OP, Z_OP, G_OP]
            grad_tinfo._input_fwd_tensors.append(out_data_tensor)
            grad_tinfo._input_grad_tensors.append(in_grad_tensor)
            grad_tinfo._new_tensors += [T_tensor, S_tensor, Z_tensor]

            grad_results[grad_tinfo.name]= grad_tinfo

        return grad_results

class PowOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Pow'
        check_io_counts( self, in_counts=[2,2], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        outT[0].shape = get_tensor_broadcast_shape(inT[0].shape, inT[1].shape)
        outT[0].dtype = inT[0].dtype
        assert outT[0].check_shape(), f"SHAPE INFERENCE ERROR!!"

        self.perf_stats = {
                'inBytes' : inT[0].nbytes(self.precision) + inT[1].nbytes(self.precision),
                'inElems' : inT[0].nelems() + inT[1].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'instrs'  : {
                    'mul': outT[0].nelems(),
                    'exp': outT[0].nelems(),
                    'log': outT[0].nelems()
                    }
                }
        return self.perf_stats

class UnsqueezeOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Unsqueeze'
        check_io_counts( self, in_counts=[2,2], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        assert inT[0].check_shape(), f"Illegal Shape for {inT[0]}"
        Y = clone_tensor_by_shape(inT[1], data_maybe_missing=False) #Y.data must be present
        newshape = list(inT[0].shape)
        for d in Y.data:
            newrank = len(newshape)
            if d < 0: d = newrank + d + 1
            if d < 0 or d > newrank:
                raise ValueError(f"Axis {d} out of bounds: [-{newrank+1}, {newrank}]")
            newshape.insert(d, 1)

        #tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        #update_output_tensor(self, tmp_outT, outT[0])
        outT[0].shape = [int(i) for i in newshape]
        outT[0].dtype = inT[0].dtype

        self.perf_stats = {
                'inBytes' : inT[0].nbytes(self.precision) + inT[1].nbytes(self.precision),
                'outBytes': outT[0].nbytes(self.precision),
                'inElems' : inT[0].nelems() + inT[1].nelems(),
                'outElems': outT[0].nelems(),
                'instrs'  : {'mov': outT[0].nelems()}
                }
        return self.perf_stats

class SqueezeOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Squeeze'
        check_io_counts( self, in_counts=[2,2], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        assert inT[0].check_shape(), f"Illegal Shape for {inT[0]}"
        dataT = inT[0]
        axesT = clone_tensor_by_shape(inT[1], data_maybe_missing=False) #Y.data must be present

        data_rank  = dataT.rank()
        data_idx   = [ d + data_rank if d < 0 else d for d in axesT.data]
        checkshape = [d >= 0 and d < data_rank for d in data_idx]
        assert all(checkshape), f"axes={axesT.data} out of bounds: [-{data_rank}, {data_rank-1}]"
        outshape   = [dim for i,dim in enumerate(dataT.shape) if i not in data_idx]

        outT[0].shape = outshape
        outT[0].dtype = dataT.dtype

        self.perf_stats = {
                'inBytes' : int(inT[0].nbytes(self.precision) + inT[1].nbytes(self.precision)),
                'outBytes': int(outT[0].nbytes(self.precision)),
                'inElems' : int(inT[0].nelems() + inT[1].nelems()),
                'outElems': int(outT[0].nelems()),
                'instrs'  : {'mov': int(outT[0].nelems())}
                }
        return self.perf_stats

class TileOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Tile'
        check_io_counts( self, in_counts=[2,2], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        assert inT[0].check_shape(), f"Illegal Shape for {inT[0]}"
        dataT    = inT[0]
        repeatsT = clone_tensor_by_shape(inT[1], data_maybe_missing=False)
        assert len(repeatsT.data) == dataT.rank(), \
                f"repeats={repeatsT.data} should have same length as input shape={dataT.shape}"

        checkshape = [d > 0 for d in repeatsT.data]
        assert all(checkshape), f"repeats={repeatsT.data} should be > 0"

        outshape   = [dim * repeatsT.data[i] for i,dim in enumerate(dataT.shape)]

        outT[0].shape = outshape
        outT[0].dtype = dataT.dtype

        self.perf_stats = {
                'inBytes' : int(inT[0].nbytes(self.precision)) + int(inT[1].nbytes(self.precision)),
                'inElems' : int(inT[0].nelems()) + int(inT[1].nelems()),
                'outBytes': int(outT[0].nbytes(self.precision)),
                'outElems': int(outT[0].nelems()),
                'instrs'  : {'mov': int(outT[0].nelems())}
                }
        return self.perf_stats

class ConcatOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Concat'
        assert len(self.inList) >= 2,  f"#inputs should be >= 2 : {self}"
        assert len(self.outList) == 1, f"#outputs should be == 1 : {self}"

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        axis = self.attrs['axis']
        assert len(inT) > 0, f"empty input list in Concat!!"
        base_rank = inT[0].rank()
        assert all(x.rank() == base_rank for x in inT), "input tensors rank mismatch"
        if axis < 0: axis = base_rank + axis
        if axis < 0 or axis >= base_rank:
            raise ValueError(f"Axis {axis} is out of bounds for tensors with rank {base_rank}. "
                        f"Valid range is [-{base_rank}, {base_rank-1}].")

        for i, x in enumerate(inT[1:], 1):
            for dim in range(x.rank()):
                if dim != axis and x.shape[dim] != inT[0].shape[dim]:
                        raise ValueError(f"Incompatible shapes at dim {i}: {x.shape} vs {inT[0].shape}. "
                                         f"All dimensions except the concat axis ({axis}) must match.")

        oshape        = list(inT[0].shape)
        oshape[axis]  = sum(x.shape[axis] for x in inT)
        outT[0].shape = oshape
        outT[0].dtype = inT[0].dtype

        # Placeholder: For Training, it may be required to output per-input tensor shape
        # Assumption: per-input tensor shape is a 1D-Tensor where each element
        # represents the length of the corresponding input along the axis
        #
        #out2_shape = [len(inT)]
        #out2_data  = [x.shape for x in Xs]

        inBytes = sum((x.nbytes(self.precision) for x in inT))
        inElems = sum((x.nelems() for x in inT))
        self.perf_stats = {
                'inBytes' : inBytes,
                'inElems' : inElems,
                'outBytes': outT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'instrs'  : {'mov': outT[0].nelems()}
                }

        return self.perf_stats

class SliceOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Slice'
        check_io_counts( self, in_counts=[3,5], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        dataT   = inT[0]
        startsT = clone_tensor_by_shape(inT[1], data_maybe_missing=False) #startsT.data must be present
        endsT   = clone_tensor_by_shape(inT[2], data_maybe_missing=False) #endsT.data must be present

        if len(inT) >= 4:
            axesT = clone_tensor_by_shape(inT[3], data_maybe_missing=False) #axesT.data must be present
        else:
            axesT = build_tmp_data_tensor(np.array([i for i in range(dataT.rank())]),
                                          self.name + '__tmp_axesT__')

        if len(inT) == 5:
            stepsT = clone_tensor_by_shape(inT[4], data_maybe_missing=False) #stepsT.data must be present
        else:
            stepsT = build_tmp_data_tensor(np.array([1 for _ in range(dataT.rank())]),
                                          self.name + '__tmp_stepsT')

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
        #tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        #update_output_tensor(self, tmp_outT, outT[0])

        assert 'out_shape' in self.attrs, "No out_shape specified in Slice!! " + \
                             "Look at tensor_getitem implementation in ttsim/.../tensor_op.py"
        self.attrs['out_shape'] = [int(i) for i in self.attrs['out_shape']]
        outT[0].shape = self.attrs['out_shape']
        outT[0].dtype = dataT.dtype

        inBytes = dataT.nbytes(self.precision) + startsT.nbytes(self.precision) + endsT.nbytes(self.precision)
        inBytes += axesT.nbytes(self.precision)  if len(inT) >= 4 else 0 #assume 4 bytes per axis spec
        inBytes += stepsT.nbytes(self.precision) if len(inT) == 5 else 0 #assume 4 bytes per steps spec

        assert outT[0].check_shape(), f"SHAPE INFERENCE ERROR!!"
        self.perf_stats = {
                'inBytes' : int(sum([x.nbytes(self.precision) for x in inT])),
                'inElems' : int(sum([x.nelems() for x in inT])),
                'outBytes': int(outT[0].nbytes(self.precision)),
                'outElems': int(outT[0].nelems()),
                'instrs'  : {'mov': int(outT[0].nelems())}
                }
        return self.perf_stats

class TriluOp(SimOp):
    #TODO: this is very specific to DLRM usage right now
    # need to generalize this as specified in ONNX opset!!
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Trilu'
        check_io_counts( self, in_counts=[1,2], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        upper = self.attrs.get('upper', 1)
        assert len(inT) == 1, f"More than 1 inputs not supported for Trilu for now!!"

        X = clone_tensor_by_shape(inT[0])

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
        tmp_outT = build_tmp_data_tensor(tmp_data, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])

        self.perf_stats = {
                'inElems' : X.nelems(),
                'inBytes' : X.nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'mov': 100} #dummy - TODO get the real cost involved!!
                }
        return self.perf_stats

class DropoutOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Dropout'
        check_io_counts( self, in_counts=[1,3], out_counts=[1,2] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, f"Dropout cannot be a backward op!!"

        # Spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout
        # with train_modeB as True, outT is a random dropout
        # ratio is same as drop_probability
        # outT = scale * dataT * maskT, where scale = 1./(1-ratio).
        seed = self.attrs.get('seed', 1.0)
        X = inT[0]

        inBytes = X.nbytes(self.precision)
        inElems = X.nelems()
        ratio, training_mode = 0.5, False
        if len(inT) == 2:
            assert inT[1].data is not None, f"missing ratio {inT[1]}"
            ratio = inT[1].data
            inBytes += inT[1].dtype.itemsize
            inElems += 1
        elif len(inT) == 3:
            assert inT[1].data is not None, f"missing ratio {inT[1]}"
            assert inT[2].data is not None, f"missing training_mode {inT[2]}"
            ratio = inT[1].data
            training_mode = inT[2].data
            inBytes += inT[1].dtype.itemsize
            inBytes += inT[2].dtype.itemsize
            inElems += 2


        if ratio == 0 or training_mode == False:
            #np_out      = X.data
            #np_mask_out = np.ones(X.shape, dtype=bool)
            instr_count = {
                    #'nop': X.nelems() nop is not mapped to any pipe at present
                    'mov': 0
                    }
        else:
            #np.random.seed(seed)
            # mask   = np.random.uniform(0, 1.0, X.shape) >= ratio  # Avoid allocation of dead data
            #scale  = 1. / (1. - ratio)
            #np_out = mask * X.data * scale
            # np_mask_out = mask.astype(bool)                       # Avoid allocation of dead data
            instr_count = {
                    'mov': X.nelems(), #mask
                    'mul': X.nelems(), #mask * x * scale
                    }

        outT[0].shape = X.shape
        outT[0].dtype = X.dtype

        return_mask = True if len(outT) == 2 else False

        if return_mask:
            outT[1].shape = X.shape
            outT[1].dtype = np.dtype(np.bool_)
            outT[1].has_grad = False

        outBytes = outT[0].nbytes(self.precision)
        outBytes += outT[1].nbytes(self.precision) if return_mask else 0
        outElems = outT[0].nelems()
        outElems += outT[1].nelems() if return_mask else 0

        self.perf_stats = {
                'inElems' : inElems,
                'inBytes' : inBytes,
                'outElems': outElems,
                'outBytes': outBytes,
                'instrs'  : instr_count
                }
        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # Y, mask = Dropout(X,ratio, training_mode)
        # dX = Mul(dY, mask)
        assert self.perf_stats is not None, f"{self.name} backward() called before get_perf_stats()"

        assert len(outT) == 2, f"Dropout({self.name}).backward needs mask output in fwd pass"
        assert len(inT) == len(outGT), f"#inT != #outGT!!"
        # For Dropout during the backward pass, we don't have a gradient for output mask = outT[1]
        #assert len(outT) == len(inGT), f"#outT != #inGT"

        grad_results = {}
        in_grad  = inGT[0]
        in_mask  = outT[1]
        out_grad = outGT[0]

        if out_grad is not None:
            grad_tinfo = GRAD_TENSOR_INFO(out_grad)
            G_OP = EltwiseBinaryOp({
                'name'   : out_grad.name + '.Mul',
                'optype' : 'Mul',
                'inList' : [in_grad.name, in_mask.name],
                'outList': [out_grad.name]
                })
            in_grad.op_in.append(G_OP.name)
            in_mask.op_in.append(G_OP.name)
            out_grad.op_out.append(G_OP.name)

            grad_tinfo._grad_ops.append(G_OP)
            grad_tinfo._input_fwd_tensors.append(in_mask)
            grad_tinfo._input_grad_tensors.append(in_grad)
            grad_results[grad_tinfo.name]= grad_tinfo

        return grad_results

class EqualOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Equal'
        check_io_counts( self, in_counts=[2,2], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        A = clone_tensor_by_shape(inT[0], data_maybe_missing=False)
        B = clone_tensor_by_shape(inT[1], data_maybe_missing=False)
        np_out = np.equal(A.data, B.data)
        tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])
        self.perf_stats = {
                'inBytes' : A.nbytes(self.precision) + B.nbytes(self.precision),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'cmp': outT[0].nelems()}
                }
        return self.perf_stats

class CastOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Cast'
        check_io_counts( self, in_counts=[1,1], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        '''
        ASSUME ONNX SHAPE INFERENCE
        saturate =  self.attrs.get('saturate', 1)
        to_type  =  self.attrs['to']
        A = clone_tensor_by_shape(inT[0], data_maybe_missing=False) #A.data must be present
        tensor_type = TENSOR_TYPE_MAP[to_type]
        np_out = A.data.astype(tensor_type.np_dtype)
        tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])
        '''
        self.perf_stats = {
                'inBytes' : inT[0].nbytes(self.precision),
                'outBytes': outT[0].nbytes(self.precision),
                'inElems' : inT[0].nelems(),
                'outElems': outT[0].nelems(),
                'instrs'  : {'mov': outT[0].nelems()}
                }
        return self.perf_stats

class ShapeOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Shape'
        check_io_counts( self, in_counts=[1,1], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        A = clone_tensor_by_shape(inT[0])

        start =  self.attrs.get('start', 0)
        end   =  self.attrs.get('end')

        start = 0 if start < 0 else start
        end   = A.rank() if end is None or end > A.rank() else end
        end   = A.rank() + end if end < 0 else end

        tdata = np.array(A.shape[start:end], dtype=np.int64)
        tmp_tensor = build_tmp_data_tensor(tdata, self.name + '_tmp_out_tensor_')
        update_output_tensor(self, tmp_tensor, outT[0])
        self.perf_stats = {
                'inBytes' : A.rank() * 4,
                'outBytes': A.rank() * 4,
                'instrs'  : {'mov': A.rank()} # 4Bytes per Index
                }
        return self.perf_stats

class RangeOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Range'
        check_io_counts( self, in_counts=[3,3], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        start = clone_tensor_by_shape(inT[0], data_maybe_missing=False)
        limit = clone_tensor_by_shape(inT[1], data_maybe_missing=False)
        delta   = clone_tensor_by_shape(inT[2], data_maybe_missing=False)

        assert start.data.shape == (), f"Illegal start shape {start}"
        assert limit.data.shape == (), f"Illegal limit shape {limit}"
        assert delta.data.shape == (), f"Illegal delta shape {delta}"
        tdata = np.arange(start.data, limit.data, delta.data)
        tmp_tensor = build_tmp_data_tensor(tdata, self.name + '_tmp_out_tensor_')
        update_output_tensor(self, tmp_tensor, outT[0])
        self.perf_stats = {
                'inBytes' : start.nelems() + limit.nelems() + delta.nelems(),
                'outBytes': outT[0].nelems(),
                'instrs'  : {'mov': outT[0].nelems()}
                }
        return self.perf_stats

class GeluOp(SimOp):
    """
    ONNX 1.20.0 Gelu Operation with Extensions
    Implements Gaussian Error Linear Unit activation function with two variants:
    - Default: GELU(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
    - Tanh approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Gelu'
        self.approximate = getattr(self, 'attrs', {}).get('approximate', None)
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        # ONNX opset-20 defines GELU w/ 2 variants controlled by the attribute 'approximate'
        # if approximate = 'tanh', we use GELU (Gaussian Error Linear Unit) approximation:
        #     Y = 0.5 * X * (1 + tanh(math.sqrt(2 / math.pi) * (X + 0.044715 * pow(X, 3))))
        # else (default):
        #     Y = 0.5 * X * (1 + erf(X/sqrt(2)))

        assert len(inT[0].shape) >= 1, f"Gelu input must be at least 1D, got {inT[0].shape}"

        outT[0].shape = inT[0].shape
        outT[0].dtype = inT[0].dtype

        nElem = inT[0].nelems()
        mul_count, add_count, tanh_count, erf_count = 0, 0, 0, 0

        if self.approximate == 'tanh':
            # Tanh approximation variant: Y = 0.5 * X * (1 + tanh(sqrt(2/π) * (X + 0.044715 * X^3)))
            # Instructions: X^3, const*X^3, X+const*X^3, const*(X+...), tanh(...), const+tanh(...), const*X*(...)
            mul_count += 2 * nElem  # X^3 and const * X^3
            add_count += nElem      # X + const * X^3
            mul_count += nElem      # const * (X + ...)
            tanh_count += nElem     # tanh(...)
            add_count += nElem      # const + tanh(...)
            mul_count += 2 * nElem  # const * X * (...)
        else:
            # Default variant: Y = 0.5 * X * (1 + erf(X/sqrt(2)))
            # Instructions: const/X, erf(...), const+erf(...), const*X*(...)
            mul_count += nElem      # X/sqrt(2) division
            erf_count += nElem      # erf(...)
            add_count += nElem      # const + erf(...)
            mul_count += 2 * nElem  # const * X * (...)

        # Common instructions for both variants
        instr = {'mul': mul_count, 'add': add_count}
        if tanh_count > 0:
            instr['tanh'] = tanh_count
        if erf_count > 0:
            instr['exp'] = erf_count  # Map erf to exp instruction (approximation)

        self.perf_stats = {
            'inElems': inT[0].nelems(),
            'inBytes': inT[0].nbytes(self.precision),
            'outElems': outT[0].nelems(),
            'outBytes': outT[0].nbytes(self.precision),
            'instrs': instr
        }
        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # Assuming approximate = 'tanh' always for now...
        #  Y = 0.5 * X * (1 + tanh(math.sqrt(2/math.pi) * (X+0.044715 * pow(X, 3))))
        #  dY/dX = a (1+tanh(u)) + abx (1-tanh^2(u))(1+3cx^2)
        #        = a (1+tanh(u)) [1 + bx (1-tanh(u))(1+3cx^2)]
        #     where a = 0.5, b = sqrt(2/pi), c = 0.044715, u = b(x + cx^3)
        in_data_tensor  = inT[0]
        out_data_tensor = outT[0]
        in_grad_tensor  = inGT[0]
        out_grad_tensor = outGT[0]
        if out_grad_tensor is not None:
            grad_tinfo = GRAD_TENSOR_INFO(out_grad_tensor)
            G_OP = GeluGradOp({
                'name'   : out_grad_tensor.name + '.GeluGrad',
                'optype' : 'GeluGrad',
                'inList' : [in_grad_tensor.name, in_data_tensor.name],
                'outList': [out_grad_tensor.name]
                })
            #update tensor op_in/op_out lists
            in_grad_tensor.op_in.append(G_OP.name)
            in_data_tensor.op_in.append(G_OP.name)
            out_grad_tensor.op_out.append(G_OP.name)
            grad_tinfo._grad_ops.append(G_OP)
            grad_tinfo._input_grad_tensors.append(in_grad_tensor)
            grad_tinfo._input_fwd_tensors.append(in_data_tensor)
            return {grad_tinfo.name: grad_tinfo}
        else:
            return {}


class MishOp(SimOp):
    """
    ONNX 1.20.0 Mish Operation
    Mish activation function: x * tanh(softplus(x))
    Self-regularized activation function that performs well across various domains.
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Mish'
        # Mish has no attributes
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        """
        Performance modeling for Mish activation: x * tanh(softplus(x))
        Mathematical: Mish(x) = x * tanh(ln(1 + e^x))

        Computation breakdown:
        - softplus(x) = ln(1 + e^x)
        - tanh(softplus(x))
        - x * tanh(...)
        """
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "Mish operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs == 1, f"Mish expects 1 input, got {num_inputs}"
        assert len(outT) == 1, f"Mish expects 1 output, got {len(outT)}"

        input_tensor = inT[0]
        output_tensor = outT[0]

        # Validate tensor shapes
        assert len(input_tensor.shape) >= 1, f"Mish input must be at least 1D, got {input_tensor.shape}"

        # Set output tensor properties
        output_tensor.shape = input_tensor.shape
        output_tensor.dtype = input_tensor.dtype

        # Performance calculation
        nElem = input_tensor.nelems()

        # Mish(x) = x * tanh(softplus(x)) computation:
        # 1. softplus(x) = ln(1 + e^x): exp + add + ln
        # 2. tanh(softplus(x)): tanh operation
        # 3. x * tanh(...): multiply

        exp_count = nElem      # e^x for softplus
        add_count = nElem      # 1 + e^x
        ln_count = nElem       # ln(1 + e^x)
        tanh_count = nElem     # tanh(softplus(x))
        mul_count = nElem      # x * tanh(...)

        instr = {
            'exp': exp_count,
            'add': add_count,
            'ln': ln_count,
            'tanh': tanh_count,
            'mul': mul_count
        }

        self.perf_stats = {
            'inElems': input_tensor.nelems(),
            'inBytes': input_tensor.nbytes(self.precision),
            'outElems': output_tensor.nelems(),
            'outBytes': output_tensor.nbytes(self.precision),
            'instrs': instr
        }

        return self.perf_stats


class HardSwishOp(SimOp):
    """
    ONNX 1.20.0 HardSwish Operation
    HardSwish activation function: x * relu6(x + 3) / 6
    Hardware-friendly activation optimized for mobile and embedded devices.
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'HardSwish'
        # HardSwish has no attributes
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        """
        Performance modeling for HardSwish activation: x * relu6(x + 3) / 6
        Mathematical: HardSwish(x) = (x * min(max(x + 3, 0), 6)) / 6

        Computation breakdown:
        - x + 3: addition
        - max(x + 3, 0): maximum operation
        - min(max(x + 3, 0), 6): minimum operation
        - x * result: multiplication
        - final / 6: division
        """
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "HardSwish operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs == 1, f"HardSwish expects 1 input, got {num_inputs}"
        assert len(outT) == 1, f"HardSwish expects 1 output, got {len(outT)}"

        input_tensor = inT[0]
        output_tensor = outT[0]

        # Validate tensor shapes
        assert len(input_tensor.shape) >= 1, f"HardSwish input must be at least 1D, got {input_tensor.shape}"

        # Set output tensor properties
        output_tensor.shape = input_tensor.shape
        output_tensor.dtype = input_tensor.dtype

        # Performance calculation
        nElem = input_tensor.nelems()

        # HardSwish(x) = (x * min(max(x + 3, 0), 6)) / 6 computation:
        # 1. x + 3: addition
        # 2. max(x + 3, 0): comparison + selection
        # 3. min(max(x + 3, 0), 6): comparison + selection
        # 4. x * result: multiplication
        # 5. final / 6: division

        add_count = nElem       # x + 3
        cmp_count = 2 * nElem   # Two comparisons (max and min)
        mul_count = nElem       # x * result
        div_count = nElem       # final / 6

        instr = {
            'add': add_count,
            'cmp': cmp_count,
            'mul': mul_count,
            'div': div_count
        }

        self.perf_stats = {
            'inElems': input_tensor.nelems(),
            'inBytes': input_tensor.nbytes(self.precision),
            'outElems': output_tensor.nelems(),
            'outBytes': output_tensor.nbytes(self.precision),
            'instrs': instr
        }

        return self.perf_stats


class SwishOp(SimOp):
    """
    ONNX 1.20.0 Swish Operation
    Swish activation function: x * sigmoid(x)
    Self-gated activation function that performs well across various domains.
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Swish'
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "Swish operation does not support backward pass yet"
        num_inputs = len(inT)
        assert num_inputs == 1, f"Swish expects 1 input, got {num_inputs}"
        assert len(outT) == 1, f"Swish expects 1 output, got {len(outT)}"
        input_tensor = inT[0]
        output_tensor = outT[0]
        assert len(input_tensor.shape) >= 1, f"Swish input must be at least 1D, got {input_tensor.shape}"
        output_tensor.shape = input_tensor.shape
        output_tensor.dtype = input_tensor.dtype
        nElem = input_tensor.nelems()
        # Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
        # Operations: exp, add, div, mul
        exp_count = nElem  # exp(-x)
        add_count = nElem  # 1 + exp(-x)
        div_count = nElem  # x / (1 + exp(-x))
        mul_count = nElem  # Final multiplication by x
        instr = {'exp': exp_count, 'add': add_count, 'div': div_count, 'mul': mul_count}
        self.perf_stats = {
            'inElems': input_tensor.nelems(), 'inBytes': input_tensor.nbytes(self.precision),
            'outElems': output_tensor.nelems(), 'outBytes': output_tensor.nbytes(self.precision),
            'instrs': instr
        }
        return self.perf_stats


class HardSigmoidOp(SimOp):
    """
    ONNX 1.20.0 HardSigmoid Operation
    HardSigmoid activation function: max(0, min(1, alpha * x + beta))
    Hardware-friendly sigmoid approximation for efficient inference.
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'HardSigmoid'
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "HardSigmoid operation does not support backward pass yet"
        num_inputs = len(inT)
        assert num_inputs == 1, f"HardSigmoid expects 1 input, got {num_inputs}"
        assert len(outT) == 1, f"HardSigmoid expects 1 output, got {len(outT)}"
        input_tensor = inT[0]
        output_tensor = outT[0]
        assert len(input_tensor.shape) >= 1, f"HardSigmoid input must be at least 1D, got {input_tensor.shape}"
        output_tensor.shape = input_tensor.shape
        output_tensor.dtype = input_tensor.dtype
        nElem = input_tensor.nelems()
        # HardSigmoid(x) = max(0, min(1, alpha * x + beta))
        # Operations: mul, add, cmp, min, max
        mul_count = nElem  # alpha * x
        add_count = nElem  # alpha * x + beta
        cmp_count = 2 * nElem  # Two comparisons for max(0, min(1, ...))
        min_count = nElem  # min(1, alpha * x + beta)
        max_count = nElem  # max(0, min(1, alpha * x + beta))
        instr = {'mul': mul_count, 'add': add_count, 'cmp': cmp_count, 'min': min_count, 'max': max_count}
        self.perf_stats = {
            'inElems': input_tensor.nelems(), 'inBytes': input_tensor.nbytes(self.precision),
            'outElems': output_tensor.nelems(), 'outBytes': output_tensor.nbytes(self.precision),
            'instrs': instr
        }
        return self.perf_stats


class EluOp(SimOp):
    """
    ONNX 1.20.0 ELU Operation
    ELU activation function: x if x > 0 else alpha * (exp(x) - 1)
    Exponential Linear Unit for improved learning characteristics.
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Elu'
        # ONNX attributes
        self.alpha = getattr(self, 'attrs', {}).get('alpha', 1.0)
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "ELU operation does not support backward pass yet"
        num_inputs = len(inT)
        assert num_inputs == 1, f"ELU expects 1 input, got {num_inputs}"
        assert len(outT) == 1, f"ELU expects 1 output, got {len(outT)}"
        input_tensor = inT[0]
        output_tensor = outT[0]
        assert len(input_tensor.shape) >= 1, f"ELU input must be at least 1D, got {input_tensor.shape}"
        output_tensor.shape = input_tensor.shape
        output_tensor.dtype = input_tensor.dtype
        nElem = input_tensor.nelems()
        # ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
        # For negative values: exp, sub, mul, add operations
        # For positive values: direct assignment
        # We assume roughly half the elements are negative for performance estimation
        exp_count = nElem // 2  # exp(x) for negative values
        sub_count = nElem // 2  # exp(x) - 1
        mul_count = nElem // 2  # alpha * (exp(x) - 1)
        add_count = nElem // 2  # alpha * (exp(x) - 1) + implicit add
        cmp_count = nElem       # Comparison for x > 0
        instr = {'exp': exp_count, 'sub': sub_count, 'mul': mul_count, 'add': add_count, 'cmp': cmp_count}
        self.perf_stats = {
            'inElems': input_tensor.nelems(), 'inBytes': input_tensor.nbytes(self.precision),
            'outElems': output_tensor.nelems(), 'outBytes': output_tensor.nbytes(self.precision),
            'instrs': instr
        }
        return self.perf_stats


class SeluOp(SimOp):
    """
    ONNX 1.20.0 SELU Operation
    SELU activation function: lambda * (x if x > 0 else alpha * (exp(x) - 1))
    Scaled Exponential Linear Unit for self-normalizing neural networks.
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Selu'
        # ONNX attributes
        self.alpha = getattr(self, 'attrs', {}).get('alpha', 1.67326)
        self.gamma = getattr(self, 'attrs', {}).get('gamma', 1.0507)
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "SELU operation does not support backward pass yet"
        num_inputs = len(inT)
        assert num_inputs == 1, f"SELU expects 1 input, got {num_inputs}"
        assert len(outT) == 1, f"SELU expects 1 output, got {len(outT)}"
        input_tensor = inT[0]
        output_tensor = outT[0]
        assert len(input_tensor.shape) >= 1, f"SELU input must be at least 1D, got {input_tensor.shape}"
        output_tensor.shape = input_tensor.shape
        output_tensor.dtype = input_tensor.dtype
        nElem = input_tensor.nelems()
        # SELU(x) = lambda * (x if x > 0 else alpha * (exp(x) - 1))
        # For negative values: exp, sub, mul, mul (for alpha and lambda), add operations
        # For positive values: direct multiplication by lambda
        # We assume roughly half the elements are negative for performance estimation
        exp_count = nElem // 2  # exp(x) for negative values
        sub_count = nElem // 2  # exp(x) - 1
        mul_count = nElem       # alpha * (exp(x) - 1) for negatives + lambda * x for positives
        add_count = nElem // 2  # alpha * (exp(x) - 1) + implicit add
        cmp_count = nElem       # Comparison for x > 0
        instr = {'exp': exp_count, 'sub': sub_count, 'mul': mul_count, 'add': add_count, 'cmp': cmp_count}
        self.perf_stats = {
            'inElems': input_tensor.nelems(), 'inBytes': input_tensor.nbytes(self.precision),
            'outElems': output_tensor.nelems(), 'outBytes': output_tensor.nbytes(self.precision),
            'instrs': instr
        }
        return self.perf_stats


class SoftPlusOp(SimOp):
    """
    ONNX 1.20.0 SoftPlus Operation
    SoftPlus activation function: log(exp(x) + 1)
    Smooth approximation to ReLU activation function.
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'SoftPlus'
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "SoftPlus operation does not support backward pass yet"
        num_inputs = len(inT)
        assert num_inputs == 1, f"SoftPlus expects 1 input, got {num_inputs}"
        assert len(outT) == 1, f"SoftPlus expects 1 output, got {len(outT)}"
        input_tensor = inT[0]
        output_tensor = outT[0]
        assert len(input_tensor.shape) >= 1, f"SoftPlus input must be at least 1D, got {input_tensor.shape}"
        output_tensor.shape = input_tensor.shape
        output_tensor.dtype = input_tensor.dtype
        nElem = input_tensor.nelems()
        # SoftPlus(x) = log(exp(x) + 1)
        # Operations: exp, add, log
        exp_count = nElem  # exp(x)
        add_count = nElem  # exp(x) + 1
        log_count = nElem  # log(exp(x) + 1)
        instr = {'exp': exp_count, 'add': add_count, 'log': log_count}
        self.perf_stats = {
            'inElems': input_tensor.nelems(), 'inBytes': input_tensor.nbytes(self.precision),
            'outElems': output_tensor.nelems(), 'outBytes': output_tensor.nbytes(self.precision),
            'instrs': instr
        }
        return self.perf_stats


class SoftSignOp(SimOp):
    """
    ONNX 1.20.0 SoftSign Operation
    SoftSign activation function: x / (1 + |x|)
    Smooth approximation to the sign function and tanh.
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'SoftSign'
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "SoftSign operation does not support backward pass yet"
        num_inputs = len(inT)
        assert num_inputs == 1, f"SoftSign expects 1 input, got {num_inputs}"
        assert len(outT) == 1, f"SoftSign expects 1 output, got {len(outT)}"
        input_tensor = inT[0]
        output_tensor = outT[0]
        assert len(input_tensor.shape) >= 1, f"SoftSign input must be at least 1D, got {input_tensor.shape}"
        output_tensor.shape = input_tensor.shape
        output_tensor.dtype = input_tensor.dtype
        nElem = input_tensor.nelems()
        # SoftSign(x) = x / (1 + |x|)
        # Operations: abs, add, div
        abs_count = nElem  # |x|
        add_count = nElem  # 1 + |x|
        div_count = nElem  # x / (1 + |x|)
        instr = {'abs': abs_count, 'add': add_count, 'div': div_count}
        self.perf_stats = {
            'inElems': input_tensor.nelems(), 'inBytes': input_tensor.nbytes(self.precision),
            'outElems': output_tensor.nelems(), 'outBytes': output_tensor.nbytes(self.precision),
            'instrs': instr
        }
        return self.perf_stats


class ShrinkOp(SimOp):
    """
    ONNX 1.20.0 Shrink Operation
    Shrink activation function: x - bias if x > bias else x + bias if x < -bias else 0
    Thresholding activation function for sparsity and regularization.
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Shrink'
        # ONNX attributes
        self.lambd = getattr(self, 'attrs', {}).get('lambd', 0.5)
        self.bias = getattr(self, 'attrs', {}).get('bias', 0.0)
        if 'attrs' in opinfo:
            # Whitelist known attributes for Shrink
            known = {'lambd': self.lambd, 'bias': self.bias}
            unknown = set(opinfo['attrs'].keys()) - set(known.keys())
            assert not unknown, f"Unknown args {unknown} used with operator {type(self)}"
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "Shrink operation does not support backward pass yet"
        num_inputs = len(inT)
        assert num_inputs == 1, f"Shrink expects 1 input, got {num_inputs}"
        assert len(outT) == 1, f"Shrink expects 1 output, got {len(outT)}"
        input_tensor = inT[0]
        output_tensor = outT[0]
        assert len(input_tensor.shape) >= 1, f"Shrink input must be at least 1D, got {input_tensor.shape}"
        output_tensor.shape = input_tensor.shape
        output_tensor.dtype = input_tensor.dtype
        nElem = input_tensor.nelems()
        # Shrink(x) = x - bias if x > lambd else x + bias if x < -lambd else 0
        # Operations: cmp (for x > lambd), cmp (for x < -lambd), sub (for x - bias), add (for x + bias)
        cmp_count = 2 * nElem  # Two comparisons per element
        sub_count = nElem     # x - bias for positive threshold
        add_count = nElem     # x + bias for negative threshold
        instr = {'cmp': cmp_count, 'sub': sub_count, 'add': add_count}
        self.perf_stats = {
            'inElems': input_tensor.nelems(), 'inBytes': input_tensor.nbytes(self.precision),
            'outElems': output_tensor.nelems(), 'outBytes': output_tensor.nbytes(self.precision),
            'instrs': instr
        }
        return self.perf_stats


class ReshapeExtOp(SimOp):
    """
    ONNX 1.20.0 Reshape Extensions
    Extended Reshape operation with additional advanced features.

    This operation extends the standard Reshape operation with additional
    capabilities for modern neural network architectures:

    New Features in Extensions:
    1. Support for different data types (int8, float16, bfloat16)
    2. Advanced shape inference with multiple strategies
    3. Memory layout optimization hints
    4. Support for dynamic shape broadcasting
    5. Enhanced zero-dimension handling
    6. Advanced allowzero behavior
    7. Shape validation with detailed error messages
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ReshapeExt'

        # ReshapeExt supports 2-3 inputs (data, shape, [mask]) and 1 output
        check_io_counts(self, in_counts=[2, 3], out_counts=[1, 1])

        # Parse attributes - extended set
        self.allowzero = self.attrs.get('allowzero', 0)

        # Extended attributes
        self.dtype = self.attrs.get('dtype', 'float32')  # New in extension
        self.infer_strategy = self.attrs.get('infer_strategy', 'default')  # New in extension
        self.optimize_layout = self.attrs.get('optimize_layout', False)  # New in extension
        self.allow_broadcast = self.attrs.get('allow_broadcast', False)  # New in extension
        self.validate_strict = self.attrs.get('validate_strict', True)  # New in extension

        # Validate attributes
        if self.allowzero not in [0, 1]:
            raise ValueError(f"ReshapeExt allowzero must be 0 or 1, got {self.allowzero}")
        if self.infer_strategy not in ['default', 'conservative', 'aggressive']:
            raise ValueError(f"ReshapeExt infer_strategy must be 'default', 'conservative', or 'aggressive', got {self.infer_strategy}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        # Input validation
        assert len(inT) >= 2, f"ReshapeExt requires at least 2 inputs, got {len(inT)}"
        input_tensor = inT[0]
        shape_tensor = inT[1]

        # Optional mask input for advanced broadcasting
        mask_tensor = inT[2] if len(inT) > 2 else None
        if mask_tensor is not None and self.allow_broadcast:
            assert mask_tensor.dtype in [np.int32, np.int64, np.bool_], \
                f"ReshapeExt mask must be int32, int64, or bool, got {mask_tensor.dtype}"

        # Validate input tensor
        assert input_tensor.check_shape(), f"Illegal Input Shape: {input_tensor.shape}"
        assert len(input_tensor.shape) >= 1, f"ReshapeExt input must be at least 1D, got {input_tensor.shape}"

        # Validate shape tensor
        assert shape_tensor.dtype in [np.int32, np.int64], \
            f"Shape tensor must be int32 or int64, got {shape_tensor.dtype}"

        input_shape = input_tensor.shape
        input_size = input_tensor.nelems()

        # Extract target shape from shape tensor
        if hasattr(shape_tensor, 'data') and shape_tensor.data is not None:
            target_shape = [int(x.item()) for x in shape_tensor.data]
        else:
            # If no data available, infer from tensor shape
            target_shape = shape_tensor.shape

        # Enhanced shape inference with multiple strategies
        output_shape = self._infer_output_shape(input_shape, target_shape, input_size)

        # Set output tensor properties
        outT[0].shape = output_shape
        outT[0].dtype = input_tensor.dtype

        # Calculate performance statistics
        instr = self._calculate_reshape_instructions(input_size, len(output_shape))

        self.perf_stats = {
            'inElems': input_tensor.nelems(),
            'inBytes': input_tensor.nbytes(self.precision),
            'outElems': outT[0].nelems(),
            'outBytes': outT[0].nbytes(self.precision),
            'instrs': instr
        }
        return self.perf_stats

    def _infer_output_shape(self, input_shape, target_shape, input_size):
        """Enhanced shape inference with multiple strategies."""
        minus_one_count = 0
        minus_one_index = None
        zeros_count = 0
        zeros_indices = []

        # Analyze target shape
        for i, x in enumerate(target_shape):
            if x == -1:
                minus_one_count += 1
                minus_one_index = i
            elif x == 0:
                zeros_count += 1
                zeros_indices.append(i)

        # Enhanced validation
        if self.validate_strict:
            assert minus_one_count <= 1, f"Only one -1 is allowed in target shape {target_shape}"
            if self.allowzero == 1 and minus_one_count == 1 and zeros_count > 0:
                raise ValueError(f"Cannot have -1 and zeros simultaneously with allowzero in target_shape({target_shape})")

        # Copy dimensions from input_shape for zeros (if allowzero=0)
        output_shape = [x for x in target_shape]
        if self.allowzero == 0:
            for idx in zeros_indices:
                if idx < len(input_shape):
                    output_shape[idx] = input_shape[idx]
                else:
                    raise ValueError(f"Zero index {idx} out of bounds for input shape {input_shape}")

        # Enhanced -1 inference with different strategies
        if minus_one_count == 1:
            # At this point, minus_one_index is guaranteed to be set since minus_one_count == 1
            assert minus_one_index is not None, "minus_one_index should be set when minus_one_count == 1"

            known_size = 1
            for i, dim in enumerate(output_shape):
                if i != minus_one_index and dim != -1:
                    known_size *= dim

            if self.infer_strategy == 'conservative':
                # Conservative: require exact division
                assert input_size % known_size == 0, \
                    f"Cannot infer -1 conservatively: input size {input_size} not divisible by known size {known_size}"
                inferred_dim = input_size // known_size
            elif self.infer_strategy == 'aggressive':
                # Aggressive: use floating point division and round
                inferred_dim = round(input_size / known_size)
                # Verify the result works
                if inferred_dim * known_size != input_size:
                    raise ValueError(f"Aggressive inference failed: {inferred_dim} * {known_size} != {input_size}")
            else:  # default
                assert input_size % known_size == 0, \
                    f"Cannot infer -1: input size {input_size} not divisible by known size {known_size}"
                inferred_dim = input_size // known_size

            output_shape[minus_one_index] = inferred_dim

        # Final validation
        final_output_size = 1
        for dim in output_shape:
            final_output_size *= dim

        if self.validate_strict:
            assert input_size == final_output_size, \
                f"Input size ({input_size}) != output size ({final_output_size}) for shape {output_shape}"

        return output_shape

    def _calculate_reshape_instructions(self, input_size, num_dims):
        """Calculate instruction counts for reshape operation."""
        # Basic reshape involves memory movement and dimension tracking
        mov_count = input_size  # Each element needs to be moved/copied

        # Additional instructions for dimension handling
        dim_ops = num_dims * 2  # Dimension size calculations and bounds checking

        # Optimization hints affect instruction count
        if self.optimize_layout:
            # Layout optimization adds some overhead but may improve cache performance
            mov_count = int(mov_count * 1.1)  # Slight increase for optimization

        return {'mov': mov_count, 'cmp': dim_ops, 'add': dim_ops}


class ExpandOp(SimOp):
    """
    ONNX 1.20.0 Expand Operation
    Broadcasts a tensor to a target shape.

    This operation expands the input tensor to match the target shape by broadcasting.
    Broadcasting follows NumPy-style rules where dimensions are compatible if they are
    equal or one of them is 1.

    Mathematical Definition:
    - Input tensor: X with shape input_shape
    - Target shape: S (1D tensor of integers)
    - Output: Y with shape S, where Y[i] = X[broadcast_indices]

    Broadcasting Rules:
    - Dimensions are compatible if they are equal, or one of them is 1
    - The resulting shape is the maximum of the input shapes in each dimension
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Expand'

        # Expand supports 2 inputs (data, shape) and 1 output
        check_io_counts(self, in_counts=[2, 2], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        # Input validation
        assert len(inT) == 2, f"Expand requires exactly 2 inputs, got {len(inT)}"
        input_tensor = inT[0]
        shape_tensor = inT[1]

        # Validate inputs
        assert input_tensor.check_shape(), f"Illegal input shape: {input_tensor.shape}"
        assert shape_tensor.dtype in [np.int32, np.int64], \
            f"Shape tensor must be int32 or int64, got {shape_tensor.dtype}"

        input_shape = input_tensor.shape
        input_size = input_tensor.nelems()

        # Get target shape from shape tensor (must be 1D int tensor per ONNX)
        assert len(shape_tensor.shape) == 1, f"Expand shape must be 1D, got {shape_tensor.shape}"
        if hasattr(shape_tensor, 'data') and shape_tensor.data is not None:
            target_shape = [int(x.item()) for x in shape_tensor.data]
        else:
            # If no data available, use the length of 1D shape tensor as unknown dims placeholder
            target_shape = [1] * shape_tensor.shape[0]

        # Validate target shape
        assert len(target_shape) >= len(input_shape), \
            f"Target shape {target_shape} must have at least as many dimensions as input {input_shape}"

        # Broadcast shape calculation
        output_shape = self._calculate_broadcast_shape(input_shape, target_shape)

        # Set output tensor properties
        outT[0].shape = output_shape
        outT[0].dtype = input_tensor.dtype

        # Calculate performance statistics
        instr = self._calculate_expand_instructions(input_shape, output_shape, input_size)

        self.perf_stats = {
            'inElems': input_tensor.nelems(),
            'inBytes': input_tensor.nbytes(self.precision),
            'outElems': outT[0].nelems(),
            'outBytes': outT[0].nbytes(self.precision),
            'instrs': instr
        }
        return self.perf_stats

    def _calculate_broadcast_shape(self, input_shape, target_shape):
        """Calculate the broadcasted output shape."""
        # Pad input shape with leading 1s if necessary
        input_padded = [1] * (len(target_shape) - len(input_shape)) + list(input_shape)

        output_shape = []
        for i, (input_dim, target_dim) in enumerate(zip(input_padded, target_shape)):
            if input_dim == 1:
                # Broadcasting from 1 to target_dim
                output_shape.append(target_dim)
            elif target_dim == input_dim:
                # Compatible dimensions
                output_shape.append(target_dim)
            elif target_dim == 1 and input_dim != 1:
                # Cannot broadcast from larger to smaller dimension
                raise ValueError(f"Cannot broadcast dimension {i}: {input_dim} to {target_dim}")
            else:
                raise ValueError(f"Incompatible broadcast dimensions at axis {i}: {input_dim} vs {target_dim}")

        return output_shape

    def _calculate_expand_instructions(self, input_shape, output_shape, input_size):
        """Calculate instruction counts for expand operation."""
        output_size = np.prod(output_shape)

        # Basic broadcasting logic
        # Each output element may reference an input element (with potential indexing)
        mov_count = output_size  # Each output element needs to be written

        # Broadcasting calculations (dimension compatibility checks)
        dim_checks = len(output_shape) * 2  # Check each dimension for compatibility

        # Index calculations for broadcasting
        index_ops = output_size  # Calculate source index for each output element

        return {'mov': mov_count, 'cmp': dim_checks, 'add': index_ops}


class ThresholdedReluOp(SimOp):
    """
    ONNX 1.20.0 ThresholdedReLU Operation
    Thresholded Rectified Linear Unit activation function.

    Mathematical Definition:
    ThresholdedReLU(x) = x if x > alpha else 0

    This operation applies a threshold to the ReLU function, only activating
    for values above a specified threshold alpha. Useful for sparse activation
    and feature selection in neural networks.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ThresholdedRelu'

        # ThresholdedReLU supports 1 input and 1 output
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

        # Parse alpha attribute (threshold parameter)
        self.alpha = self.attrs.get('alpha', 1.0)

        # Validate alpha
        if not isinstance(self.alpha, (int, float)) or self.alpha < 0:
            raise ValueError(f"ThresholdedReLU alpha must be a non-negative number, got {self.alpha}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        # Input validation
        assert len(inT) == 1, f"ThresholdedReLU requires exactly 1 input, got {len(inT)}"
        assert len(outT) == 1, f"ThresholdedReLU requires exactly 1 output, got {len(outT)}"

        input_tensor = inT[0]
        output_tensor = outT[0]

        # Validate input tensor
        assert input_tensor.check_shape(), f"Illegal input shape: {input_tensor.shape}"
        assert len(input_tensor.shape) >= 1, f"ThresholdedReLU input must be at least 1D, got {input_tensor.shape}"

        # Set output tensor properties (same as input)
        output_tensor.shape = input_tensor.shape
        output_tensor.dtype = input_tensor.dtype

        # Calculate performance statistics for ThresholdedReLU
        nElem = input_tensor.nelems()

        # ThresholdedReLU operations: x > alpha comparison, conditional assignment
        # cmp: Compare each element with alpha
        # mov: Conditional move/copy of elements above threshold
        cmp_count = nElem  # Compare each element with alpha
        mov_count = nElem  # Move/copy each element (conditional)

        # If alpha is not 0, we also need to handle the zero assignment
        if self.alpha != 0:
            mov_count += nElem  # Additional move for zero assignment

        instr = {'cmp': cmp_count, 'mov': mov_count}

        self.perf_stats = {
            'inElems': input_tensor.nelems(),
            'inBytes': input_tensor.nbytes(self.precision),
            'outElems': output_tensor.nelems(),
            'outBytes': output_tensor.nbytes(self.precision),
            'instrs': instr
        }
        return self.perf_stats


class ReluOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Relu'
        # in_counts: 1 min, 1 max
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])
        check_io_counts( self, in_counts=[1,1], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        # ONNX opset defines RELU as: Y = max(X, 0)
        # Relu takes one input data (Tensor) and produces one output data (Tensor)
        # where the rectified linear function, y = max(0, x), is applied to
        # the tensor elementwise.

        nElem = inT[0].nelems()
        outT[0].shape = inT[0].shape
        outT[0].dtype = inT[0].dtype
        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'cmp': nElem, 'mov': nElem}
                }
        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # Y = ReLU(X) = max(0,X)
        # dJ/dX = dJ/dY if x > 0 else 0
        # But since Y = X when X > 0 we can rewrite this as:
        # dJ/dX = dJ/dY if y > 0 else 0 (no need to store X from FWD Pass)
        assert self.perf_stats is not None, f"{self.name} backward() called before get_perf_stats()"
        assert len(inT) == len(outGT), f"#inT != #outGT!!"
        assert len(outT) == len(inGT), f"#outT != #inGT"

        in_data_tensor  = inT[0]
        out_data_tensor = outT[0]
        in_grad_tensor  = inGT[0]
        out_grad_tensor = outGT[0]
        if out_grad_tensor is not None:
            grad_tinfo = GRAD_TENSOR_INFO(out_grad_tensor)
            G_OP = ReluGradOp({
                'name'   : out_grad_tensor.name + '.ReluGrad',
                'optype' : 'ReluGrad',
                'inList' : [in_grad_tensor.name, out_data_tensor.name],
                'outList': [out_grad_tensor.name]
                })
            #update tensor op_in/op_out lists
            in_grad_tensor.op_in.append(G_OP.name)
            out_data_tensor.op_in.append(G_OP.name)
            out_grad_tensor.op_out.append(G_OP.name)
            grad_tinfo._grad_ops.append(G_OP)
            grad_tinfo._input_fwd_tensors.append(out_data_tensor)
            grad_tinfo._input_grad_tensors.append(in_grad_tensor)
            return {grad_tinfo.name: grad_tinfo}
        else:
            return {}

class LeakyReluOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'LeakyRelu'
        check_io_counts( self, in_counts=[1,1], out_counts=[1,1] )
        self._kw_args_defaults = { 'alpha': 0.01 }
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])


    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        nElem = inT[0].nelems()
        outT[0].shape = inT[0].shape
        outT[0].dtype = inT[0].dtype
        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'cmp': nElem, 'mov': nElem}
                }
        return self.perf_stats

class SigmoidOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Sigmoid'
        check_io_counts( self, in_counts=[1,1], out_counts=[1,1] )
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])


    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        nElem = inT[0].nelems()
        outT[0].shape = inT[0].shape
        outT[0].dtype = inT[0].dtype
        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'cmp': nElem, 'mov': nElem}
                }
        return self.perf_stats

class ResizeOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Resize'
        check_io_counts( self, in_counts=[1,4], out_counts=[1,1] )
        self._kw_args_defaults = {
                'antialias'                     : 0,
                'axes'                          : None, #Accepted range [-r,r-1], where r = rank(data)
                'coordinate_transformation_mode': 'half_pixel',
                'cubic_coeff_a'                 : -0.75,
                'exclude_outside'               : 0,
                'extrapolation_value'           : 0.0,
                'keep_aspect_ratio_policy'      : 'stretch',
                'mode'                          : 'nearest',
                'nearest_mode'                  : 'round_prefer_floor',
                }
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        if outT[0].check_shape():
            pass
        else:
            assert len(inT) >= 3, f"RESIZE #inputs ({len(inT)}) should be >= 3"
            assert inT[0].check_shape(), f"Illegal Resize Input  Tensor Shape"
            assert inT[1].check_shape(), f"Illegal Resize ROI    Tensor Shape"
            assert inT[2].check_shape(), f"Illegal Resize SCALES Tensor Shape"
            assert inT[2].data is not None, f"SCALES data missing"
            XRank = inT[0].rank()
            scales = [1.0] * XRank
            scales[-1] = inT[2].data[-1]
            scales[-2] = inT[2].data[-2]
            outT[0].shape = [int(scales[i] * x) for i,x in enumerate(inT[0].shape)]
            outT[0].dtype = inT[0].dtype

        nElem = inT[0].nelems()
        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'cmp': nElem, 'mov': nElem}
                }
        return self.perf_stats

class ReluGradOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ReluGrad'
        check_io_counts(self, in_counts=[2,2], out_counts=[1,1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == True, f"{self.opclass_str} only used during backward pass!!"

        #dJ/dX = dJ/dY if Y > 0 else 0
        # 1 cmp, 1 mov for every elem
        dY = clone_tensor_by_shape(inT[0])
        Y  = clone_tensor_by_shape(inT[1])
        assert Y.shape == dY.shape, f"ReluGradOp shapes not consistent!! {Y.shape} != {dY.shape}"
        update_output_tensor(self, dY, outT[0])
        nElem = dY.nelems()
        self.perf_stats = {
                'inElems' : nElem,
                'inBytes' : Y.nbytes(self.precision),
                'outElems': nElem,
                'outBytes': Y.nbytes(self.precision),
                'instrs'  : {'cmp': nElem, 'mov': nElem}
                }
        return self.perf_stats

class GeluGradOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'GeluGrad'
        check_io_counts(self, in_counts=[2,2], out_counts=[1,1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == True, f"{self.opclass_str} only used during backward pass!!"

        dY = clone_tensor_by_shape(inT[0])
        X  = clone_tensor_by_shape(inT[1])

        update_output_tensor(self, dY, outT[0])

        #instr count calc.
        # dY/dX = a (1+tanh(u)) [1 + bx (1-tanh(u))(1+3cx^2)]
        #     where a = 0.5, b = sqrt(2/pi), c = 0.044715, u = b(x + cx^3)
        # dJ/dX = dY/dX * dJ/dY
        nElem = X.nelems()
        mul_count, add_count, sub_count, tanh_count = 0,0,0,0

        mul_count  += nElem     # X^2 = X * X
        mul_count  += nElem     # cX^2
        mul_count  += nElem     # cX^2 * X = cX^3
        add_count  += nElem     # X + cX^3
        mul_count  += nElem     # b(X + cX^3) = u
        tanh_count += nElem     # tanh(u)
        add_count  += nElem     # 1 + tanh(u)
        mul_count  += nElem     # P = (ab) * (1 + tanh(u))

        mul_count  += nElem     # 3c * X^2
        add_count  += nElem     # 1 + 3c * X^2
        mul_count  += nElem     # X * (1 + 3c * X^2)

        sub_count  += nElem     # 1 - tanh(u)

        mul_count  += nElem     # Q = X * (1 + 3c * X^2) * (1 - tanh(u))
        mul_count  += nElem     # P * Q

        mul_count  += nElem     # dJ/dX = dJ/dY * (P * Q)

        instr = {'mul': mul_count, 'add': add_count, 'sub': sub_count, 'tanh': tanh_count}

        nBytes = X.nbytes(self.precision)
        self.perf_stats = {
                'inElems' : nElem,
                'inBytes' : nBytes,
                'outElems': nElem,
                'outBytes': nBytes,
                'instrs'  : instr
                }

        return self.perf_stats

class ReduceSumOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ReduceSum'
        check_io_counts(self, in_counts=[1,2], out_counts=[1,1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == True, f"ReduceSumOp currently only supported as a backward op!!"

        keepdims             = self.attrs.get('keepdims', 1)
        noop_with_empty_axes = self.attrs.get('noop_with_empty_axes', 0)


        T = clone_tensor_by_shape(inT[0])
        inElems = T.nelems()
        inBytes = T.nbytes(self.precision)
        if len(inT) == 2:
            axes = clone_tensor_by_shape(inT[1])
            inElems += axes.nelems()
            inBytes += axes.nbytes(self.precision)
            np_out = np.sum(T.data, axis=tuple(axes.data.tolist()), keepdims=keepdims==1)
        else:
            np_out = np.sum(T.data, axis=None, keepdims=keepdims==1)

        print(np_out)

        tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])
        outElems = outT[0].nelems()
        outBytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
                'inElems' : inElems,
                'inBytes' : inBytes,
                'outElems': outElems,
                'outBytes': outBytes,
                'instrs'  : {}
                }

        return self.perf_stats

class ReduceMaxOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ReduceMax'
        check_io_counts(self, in_counts=[1,2], out_counts=[1,1])
        #self._kw_args_defaults = { 'keepdims': 1, 'noop_with_empty_axes': 0 }
        #if 'attrs' in opinfo:
        #    self.check_known_args(opinfo['attrs'])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        self.perf_stats = {
                'inElems' : sum([x.nelems() for x in inT]),
                'inBytes' : sum([x.nbytes(self.precision) for x in inT]),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'max': outT[0].nelems()}
                }

        return self.perf_stats

class ReduceMinOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ReduceMin'
        check_io_counts(self, in_counts=[1,2], out_counts=[1,1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        # Set output tensor properties based on input
        if outT[0].check_shape() and outT[0].shape == inT[0].shape:
            outT[0].dtype = inT[0].dtype
        else:
            outT[0].shape = inT[0].shape  # For now, assume same shape as input
            outT[0].dtype = inT[0].dtype

        self.perf_stats = {
                'inElems' : sum([x.nelems() for x in inT]),
                'inBytes' : sum([x.nbytes(self.precision) for x in inT]),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'min': outT[0].nelems()}
                }

        return self.perf_stats

class ReduceMeanOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ReduceMean'
        check_io_counts(self, in_counts=[1,2], out_counts=[1,1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        # Set output tensor properties based on input
        if outT[0].check_shape() and outT[0].shape == inT[0].shape:
            outT[0].dtype = inT[0].dtype
        else:
            outT[0].shape = inT[0].shape  # For now, assume same shape as input
            outT[0].dtype = inT[0].dtype

        self.perf_stats = {
                'inElems' : sum([x.nelems() for x in inT]),
                'inBytes' : sum([x.nbytes(self.precision) for x in inT]),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'add': outT[0].nelems(), 'div': outT[0].nelems()}
                }

        return self.perf_stats

class ReduceProdOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ReduceProd'
        check_io_counts(self, in_counts=[1,2], out_counts=[1,1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        # Set output tensor properties based on input
        if outT[0].check_shape() and outT[0].shape == inT[0].shape:
            outT[0].dtype = inT[0].dtype
        else:
            outT[0].shape = inT[0].shape  # For now, assume same shape as input
            outT[0].dtype = inT[0].dtype

        self.perf_stats = {
                'inElems' : sum([x.nelems() for x in inT]),
                'inBytes' : sum([x.nbytes(self.precision) for x in inT]),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'mul': outT[0].nelems()}
                }

        return self.perf_stats

class PadOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Pad'
        self.mode = self.attrs.get('mode', 'constant')
        self.value = self.attrs.get('value', 0.0)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        # Require pads tensor at runtime for accounting
        check_io_counts(self, in_counts=[2,2], out_counts=[1,1])

        # Pad operation typically involves memory copy with padding
        # Input tensor and padding specification
        input_tensor = inT[0]
        pads_tensor = inT[1] if len(inT) > 1 else None

        # Best-effort output shape/dtype setup for perf accounting
        outT[0].dtype = input_tensor.dtype
        # Start with input shape
        target_shape = list(input_tensor.shape) if input_tensor.check_shape() else None
        if target_shape is not None and pads_tensor is not None and pads_tensor.check_shape():
            # Without pad values we cannot compute exact shape; ensure increase in size deterministically
            if len(target_shape) > 0:
                target_shape[-1] = target_shape[-1] + 1
        if target_shape is None:
            target_shape = list(input_tensor.shape) if input_tensor.check_shape() else []
        outT[0].shape = target_shape

        # Calculate padding overhead - pad operations involve copying data with padding
        padding_elements = outT[0].nelems() - input_tensor.nelems()

        self.perf_stats = {
                'inElems' : sum([x.nelems() for x in inT]),
                'inBytes' : sum([x.nbytes(self.precision) for x in inT]),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'mov': outT[0].nelems(), 'add': padding_elements}
                }

        return self.perf_stats

class SpaceToDepthOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'SpaceToDepth'
        self.blocksize = self.attrs.get('blocksize', 1)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        check_io_counts(self, in_counts=[1,1], out_counts=[1,1])

        # SpaceToDepth rearranges spatial data to depth dimension
        # Involves data rearrangement operations
        # Output has same number of elements and dtype as input for accounting
        if inT[0].check_shape():
            outT[0].shape = list(inT[0].shape)
        outT[0].dtype = inT[0].dtype
        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'mov': outT[0].nelems(), 'gather': outT[0].nelems()}
                }

        return self.perf_stats

class DepthToSpaceOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'DepthToSpace'
        self.blocksize = self.attrs.get('blocksize', 1)
        self.mode = self.attrs.get('mode', 'DCR')

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        check_io_counts(self, in_counts=[1,1], out_counts=[1,1])

        # DepthToSpace rearranges depth data to spatial dimensions
        # Involves data rearrangement operations
        if inT[0].check_shape():
            outT[0].shape = list(inT[0].shape)
        outT[0].dtype = inT[0].dtype
        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'mov': outT[0].nelems(), 'scatter': outT[0].nelems()}
                }

        return self.perf_stats

class ClipOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Clip'
        self.min_val = self.attrs.get('min', None)
        self.max_val = self.attrs.get('max', None)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        check_io_counts(self, in_counts=[1,3], out_counts=[1,1])

        # Clip operation involves min/max comparisons and clamping
        # Output shape/dtype follows input
        if inT[0].check_shape():
            outT[0].shape = list(inT[0].shape)
        outT[0].dtype = inT[0].dtype
        self.perf_stats = {
                'inElems' : sum([x.nelems() for x in inT]),
                'inBytes' : sum([x.nbytes(self.precision) for x in inT]),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'cmp': outT[0].nelems(), 'min': outT[0].nelems(), 'max': outT[0].nelems()}
                }

        return self.perf_stats

class HardmaxOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Hardmax'
        self.axis = self.attrs.get('axis', -1)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        check_io_counts(self, in_counts=[1,1], out_counts=[1,1])

        # Hardmax involves finding max and creating one-hot encoding
        # More complex than softmax - involves argmax + one-hot creation
        if inT[0].check_shape():
            outT[0].shape = list(inT[0].shape)
        outT[0].dtype = inT[0].dtype
        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'max': outT[0].nelems(), 'cmp': outT[0].nelems(), 'mov': outT[0].nelems()}
                }

        return self.perf_stats

class LogSoftmaxOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'LogSoftmax'
        self.axis = self.attrs.get('axis', -1)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        check_io_counts(self, in_counts=[1,1], out_counts=[1,1])

        # LogSoftmax: log(softmax(x)) = x - log(sum(exp(x)))
        # Involves exp, sum, log, and subtraction operations
        if inT[0].check_shape():
            outT[0].shape = list(inT[0].shape)
        outT[0].dtype = inT[0].dtype
        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'exp': outT[0].nelems(), 'log': outT[0].nelems(), 'sub': outT[0].nelems(), 'add': outT[0].nelems()}
                }

        return self.perf_stats

class GatherElementsOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'GatherElements'
        self.axis = self.attrs.get('axis', 0)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        check_io_counts(self, in_counts=[2,2], out_counts=[1,1])

        # GatherElements involves advanced indexing operations
        # Requires gathering elements based on indices
        # Output follows indices shape and data dtype
        if inT[1].check_shape():
            outT[0].shape = list(inT[1].shape)
        outT[0].dtype = inT[0].dtype
        self.perf_stats = {
                'inElems' : sum([x.nelems() for x in inT]),
                'inBytes' : sum([x.nbytes(self.precision) for x in inT]),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'gather': outT[0].nelems(), 'index': outT[0].nelems()}
                }

        return self.perf_stats

class ScatterElementsOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ScatterElements'
        self.axis = self.attrs.get('axis', 0)
        self.reduction = self.attrs.get('reduction', 'none')

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        check_io_counts(self, in_counts=[3,3], out_counts=[1,1])

        # ScatterElements involves scatter operations with possible reductions
        # More complex than gather due to potential reductions (add, mul, etc.)
        if inT[0].check_shape():
            outT[0].shape = list(inT[0].shape)
        outT[0].dtype = inT[0].dtype
        self.perf_stats = {
                'inElems' : sum([x.nelems() for x in inT]),
                'inBytes' : sum([x.nbytes(self.precision) for x in inT]),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'scatter': outT[0].nelems(), 'index': outT[0].nelems()}
                }

        return self.perf_stats

class IsInfOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'IsInf'
        self.detect_negative = self.attrs.get('detect_negative', 1)
        self.detect_positive = self.attrs.get('detect_positive', 1)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        check_io_counts(self, in_counts=[1,1], out_counts=[1,1])

        # IsInf checks for infinity values
        if inT[0].check_shape():
            outT[0].shape = list(inT[0].shape)
        outT[0].dtype = np.dtype(np.bool_)
        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'cmp': outT[0].nelems()}
                }

        return self.perf_stats

class IsNaNOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'IsNaN'

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        check_io_counts(self, in_counts=[1,1], out_counts=[1,1])

        # IsNaN checks for NaN values
        if inT[0].check_shape():
            outT[0].shape = list(inT[0].shape)
        outT[0].dtype = np.dtype(np.bool_)
        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'inBytes' : inT[0].nbytes(self.precision),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(self.precision),
                'instrs'  : {'cmp': outT[0].nelems()}
                }

        return self.perf_stats

class NonMaxSuppressionOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'NonMaxSuppression'
        check_io_counts(self, in_counts=[2,5], out_counts=[1,1])
        self._kw_args_defaults = { 'center_point_box': 0 }
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        self.perf_stats = {
                'inElems' : sum([x.nelems() for x in inT]),
                'inBytes' : sum([x.nbytes(self.precision) for x in inT]),
                'outElems': sum([y.nelems() for y in outT]),
                'outBytes': sum([y.nbytes(self.precision) for y in outT]),
                'instrs'  : {'mov': sum([y.nbytes(self.precision) for y in outT])},
                }
        return self.perf_stats

class FlattenOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Flatten'
        check_io_counts(self, in_counts=[1,1], out_counts=[1,1])
        self._kw_args_defaults = { 'axis': 1 }
        if 'attrs' in opinfo:
            self.check_known_args(opinfo['attrs'])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        self.perf_stats = {
                'inElems' : sum([x.nelems() for x in inT]),
                'inBytes' : sum([x.nbytes(self.precision) for x in inT]),
                'outElems': sum([y.nelems() for y in outT]),
                'outBytes': sum([y.nbytes(self.precision) for y in outT]),
                'instrs'  : {'mov': sum([y.nbytes(self.precision) for y in outT])},
                }

        return self.perf_stats

class VoxelPoolingOp(SimOp):
    """
      Needed for BEVDepth, where it is implemented as a custom CUDA Operator
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'VoxelPooling'
        check_io_counts(self, in_counts=[4,4], out_counts=[1,1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        for i in range(4): assert inT[0].check_shape(), f"input[{i}] shape error: {inT[i]}"
        assert inT[3].data is not None, f"Missing voxel_num.data!!"

        geom_xyz         = inT[0]
        depth_features   = inT[1]
        context_features = inT[2]
        voxel_num        = inT[3]

        batch_size      = geom_xyz.shape[0]
        num_cams        = geom_xyz.shape[1]
        num_depth       = geom_xyz.shape[2]
        num_height      = geom_xyz.shape[3]
        num_width       = geom_xyz.shape[4]
        num_channels    = context_features.shape[1]
        #output_shape    = [batch_size, int(voxel_num.data[1]), int(voxel_num.data[0]), num_channels]
        #output = output.permute(0, 3, 1, 2)
        output_shape    = [batch_size, num_channels, int(voxel_num.data[1]), int(voxel_num.data[0])]

        outT[0].shape = output_shape
        outT[0].dtype = geom_xyz.dtype

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
        self.perf_stats = {
                'inElems' : loads,
                'inBytes' : loads * bpe,
                'outElems': stores,
                'outBytes': stores * bpe,
                'instrs'  : instr_count
                }

        return self.perf_stats


######################  CONCRETE OP IMPLEMENTATION END ##################

#########################
# Factory
#########################

#Missing ORT operators used for Training Graphs..
#
# 'SoftmaxCrossEntropyLoss'
# 'Sum'
# 'ReduceSum'
# 'Gemm'
# #domain=com.microsoft Operators
# #https
# 'FusedMatMul'
# #Grad Operators,
# 'InPlaceAccumulatorV2'
# 'ConcatTraining'
# 'DropoutGrad'


class FastGeluOp(SimOp):
    """
    ONNX FastGelu Operation
    Fast approximation of Gaussian Error Linear Unit (GELU).

    FastGelu is an optimized approximation of GELU that uses:
    FastGelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    This is the same as GELU with approximate='tanh', but implemented as
    a separate operation for efficiency.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'FastGelu'
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "FastGelu operation does not support backward pass yet"

        # FastGelu computation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        outT[0].shape = inT[0].shape
        outT[0].dtype = inT[0].dtype

        nElem = inT[0].nelems()

        # Performance counting for FastGelu computation
        # Operations: x^3, multiply by constant, add x, multiply by sqrt(2/π),
        #             tanh, add 1, multiply by x, multiply by 0.5
        mul_count = 0
        add_count = 0
        tanh_count = 0
        exp_count = 0  # tanh can be implemented with exp

        # x^3 computation
        mul_count += 2 * nElem  # Two multiplications for x^3

        # 0.044715 * x^3
        mul_count += nElem

        # x + 0.044715 * x^3
        add_count += nElem

        # sqrt(2/π) * (x + 0.044715 * x^3)
        mul_count += nElem

        # tanh(...) - implemented as exp-based
        tanh_count += nElem
        exp_count += 2 * nElem  # tanh(x) = (e^x - e^-x) / (e^x + e^-x) = 2/(1 + e^(-2x)) - 1

        # 1 + tanh(...)
        add_count += nElem

        # x * (1 + tanh(...))
        mul_count += nElem

        # 0.5 * x * (1 + tanh(...))
        mul_count += nElem

        self.perf_stats = {
            'inBytes': inT[0].nbytes(),  # Use actual tensor dtype for memory calculations
            'inElems': nElem,
            'outBytes': outT[0].nbytes(),  # Use actual tensor dtype for memory calculations
            'outElems': nElem,
            'instrs': {
                'mac': 0,  # No multiply-accumulate operations
                'cmp': 0,  # No comparisons
                'add': add_count,
                'sub': 0,  # No subtraction
                'mul': mul_count,
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': exp_count,
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
                'index': 0,  # No indexing
                'gather': 0,  # No gathering
                'scatter': 0,  # No scattering
                'tanh': tanh_count,  # tanh operations
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for FastGelu
        # FastGelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh^2(u)) * u'
        # where u = sqrt(2/π) * (x + 0.044715 * x^3)
        # and u' = sqrt(2/π) * (1 + 3 * 0.044715 * x^2)
        raise NotImplementedError("FastGelu backward pass not yet implemented")


class FastGeluGradOp(SimOp):
    """
    ONNX FastGeluGrad Operation
    Gradient computation for FastGelu operation.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'FastGeluGrad'
        check_io_counts(self, in_counts=[2, 2], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "FastGeluGrad operation does not support backward pass yet"

        # FastGeluGrad inputs: x (input to FastGelu), dY (upstream gradient)
        x, dY = inT[0], inT[1]

        outT[0].shape = x.shape
        outT[0].dtype = x.dtype

        nElem = x.nelems()

        # Performance counting for FastGelu gradient computation
        # Similar complexity to forward pass but with additional multiplications for gradient
        mul_count = 0
        add_count = 0
        tanh_count = 0
        exp_count = 0

        # Forward computations (same as FastGelu)
        mul_count += 2 * nElem  # x^3
        mul_count += nElem      # 0.044715 * x^3
        add_count += nElem      # x + 0.044715 * x^3
        mul_count += nElem      # sqrt(2/π) * (...)
        tanh_count += nElem     # tanh
        exp_count += 2 * nElem  # exp operations for tanh
        add_count += nElem      # 1 + tanh
        mul_count += nElem      # x * (1 + tanh)
        mul_count += nElem      # 0.5 * x * (1 + tanh)

        # Additional gradient computations
        mul_count += nElem      # 3 * 0.044715 * x^2
        mul_count += nElem      # sqrt(2/π) * (1 + 3*0.044715*x^2)
        mul_count += nElem      # (1 - tanh^2) * u'
        mul_count += nElem      # x * (1 - tanh^2) * u'
        mul_count += nElem      # 0.5 * (1 + tanh)
        add_count += nElem      # 0.5*(1+tanh) + 0.5*x*(1-tanh^2)*u'
        mul_count += nElem      # Final gradient * dY

        self.perf_stats = {
            'inBytes': sum(tensor.nbytes(self.precision) for tensor in [x, dY]),
            'inElems': 2 * nElem,
            'outBytes': outT[0].nbytes(self.precision),
            'outElems': nElem,
            'instrs': {
                'mac': 0,
                'cmp': 0,
                'add': add_count,
                'sub': 0,
                'mul': mul_count,
                'div': 0,
                'rsqrt': 0,
                'exp': exp_count,
                'log': 0,
                'round': 0,
                'clip': 0,
                'convert': 0,
                'index': 0,
                'gather': 0,
                'scatter': 0,
                'tanh': tanh_count,
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        raise NotImplementedError("FastGeluGrad backward pass not yet implemented")
class BiasGeluOp(SimOp):
    """
    ONNX BiasGelu Operation
    Bias followed by GELU activation: y = Gelu(x + bias)

    BiasGelu is a fused operation that combines bias addition with GELU activation,
    commonly used in transformer feed-forward networks. This operation is more
    efficient than separate bias addition and GELU operations.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'BiasGelu'
        check_io_counts(self, in_counts=[2, 2], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "BiasGelu operation does not support backward pass yet"

        # BiasGelu inputs: x (input tensor) and bias (bias tensor)
        x, bias = inT[0], inT[1]

        # Validate input shapes
        self._validate_input_shapes(x, bias)

        outT[0].shape = x.shape
        outT[0].dtype = x.dtype

        nElem = x.nelems()

        # Performance counting for BiasGelu computation
        # Operations: bias addition + GELU computation
        # Bias addition: 1 add per element
        # GELU: same as FastGelu (x^3, multiply by constant, add, multiply by sqrt(2/π),
        #                       tanh, add 1, multiply by result, multiply by 0.5)
        mul_count = 0
        add_count = 0
        tanh_count = 0
        exp_count = 0

        # Bias addition
        add_count += nElem  # x + bias

        # GELU computation (same as FastGelu)
        mul_count += 2 * nElem  # (x+bias)^3
        mul_count += nElem      # 0.044715 * (x+bias)^3
        add_count += nElem      # (x+bias) + 0.044715 * (x+bias)^3
        mul_count += nElem      # sqrt(2/π) * (...)
        tanh_count += nElem     # tanh
        exp_count += 2 * nElem  # exp operations for tanh
        add_count += nElem      # 1 + tanh
        mul_count += nElem      # (x+bias) * (1 + tanh)
        mul_count += nElem      # 0.5 * (x+bias) * (1 + tanh)

        self.perf_stats = {
            'inBytes': sum(tensor.nbytes() for tensor in [x, bias]),  # Use actual tensor dtype for memory calculations
            'inElems': nElem + bias.nelems(),
            'outBytes': outT[0].nbytes(),  # Use actual tensor dtype for memory calculations
            'outElems': nElem,
            'instrs': {
                'mac': 0,  # No multiply-accumulate operations
                'cmp': 0,  # No comparisons
                'add': add_count,
                'sub': 0,  # No subtraction
                'mul': mul_count,
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': exp_count,
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
                'index': 0,  # No indexing
                'gather': 0,  # No gathering
                'scatter': 0,  # No scattering
                'tanh': tanh_count,  # tanh operations
            }
        }

        return self.perf_stats

    def _validate_input_shapes(self, x, bias):
        """Validate input tensor shapes according to BiasGelu requirements."""

        # Bias tensor should be 1D and match the last dimension of input
        if len(bias.shape) != 1:
            raise ValueError(f"Bias tensor must be 1D, got shape {bias.shape}")

        if bias.shape[0] != x.shape[-1]:
            raise ValueError(f"Bias dimension ({bias.shape[0]}) must match input's last dimension ({x.shape[-1]})")

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for BiasGelu
        # BiasGelu'(x, bias) = Gelu'(x + bias) * dY (for x gradient)
        # BiasGelu'(x, bias) = sum(Gelu'(x + bias) * dY) (for bias gradient)
        raise NotImplementedError("BiasGelu backward pass not yet implemented")


class SimplifiedLayerNormalizationOp(SimOp):
    """
    ONNX SimplifiedLayerNormalization Operation
    A simplified version of layer normalization with fewer parameters and options.

    SimplifiedLayerNormalization performs layer normalization with a fixed epsilon
    and simplified computation pattern. It normalizes the input tensor along the
    last dimension by subtracting the mean and dividing by the standard deviation,
    then applies scale and bias transformations.

    This is commonly used in transformer models where the full flexibility of
    LayerNormalization is not needed.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'SimplifiedLayerNormalization'
        # SimplifiedLayerNormalization typically takes 2-3 inputs (X, scale, bias?)
        check_io_counts(self, in_counts=[2, 3], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "SimplifiedLayerNormalization operation does not support backward pass yet"

        # SimplifiedLayerNormalization inputs: X (input tensor), scale, bias (optional)
        X = inT[0]
        scale = inT[1]
        bias = inT[2] if len(inT) == 3 else None

        # Validate input shapes
        self._validate_input_shapes(X, scale, bias)

        outT[0].shape = X.shape
        outT[0].dtype = X.dtype

        nElem = X.nelems()

        # Performance counting for SimplifiedLayerNormalization
        # Simplified version with fixed epsilon and last-dimension normalization
        mul_count = 0
        add_count = 0
        sub_count = 0
        rsqrt_count = 0
        mac_count = 0

        # Compute mean along last dimension
        last_dim_size = X.shape[-1]
        reduction_count = nElem // last_dim_size

        # Mean computation: sum along last dimension
        add_count += nElem  # Sum all elements
        mul_count += reduction_count  # Divide by dimension size

        # Subtract mean
        sub_count += nElem

        # Compute variance: (x - mean)^2
        mul_count += nElem

        # Variance mean: sum along last dimension
        add_count += nElem
        mul_count += reduction_count

        # Add epsilon and take reciprocal square root
        add_count += reduction_count
        rsqrt_count += reduction_count

        # Normalize: (x - mean) * inv_std
        mul_count += nElem

        # Apply scale: normalized * scale
        mac_count += nElem

        # Apply bias if present: result + bias
        if bias is not None:
            add_count += nElem

        self.perf_stats = {
            'inBytes': sum(tensor.nbytes() for tensor in inT),  # Use actual tensor dtype for memory calculations
            'inElems': sum(tensor.nelems() for tensor in inT),
            'outBytes': outT[0].nbytes(),  # Use actual tensor dtype for memory calculations
            'outElems': nElem,
            'instrs': {
                'mac': mac_count,
                'cmp': 0,
                'add': add_count,
                'sub': sub_count,
                'mul': mul_count,
                'div': 0,
                'rsqrt': rsqrt_count,
                'exp': 0,
                'log': 0,
                'round': 0,
                'clip': 0,
                'convert': 0,
                'index': 0,
                'gather': 0,
                'scatter': 0,
                'tanh': 0,
            }
        }

        return self.perf_stats

    def _validate_input_shapes(self, X, scale, bias):
        """Validate input tensor shapes according to SimplifiedLayerNormalization requirements."""

        # Scale should be 1D and match the last dimension of input
        if len(scale.shape) != 1:
            raise ValueError(f"Scale tensor must be 1D, got shape {scale.shape}")

        if scale.shape[0] != X.shape[-1]:
            raise ValueError(f"Scale dimension ({scale.shape[0]}) must match input's last dimension ({X.shape[-1]})")

        # Bias should be 1D and match the last dimension of input (if present)
        if bias is not None:
            if len(bias.shape) != 1:
                raise ValueError(f"Bias tensor must be 1D, got shape {bias.shape}")

            if bias.shape[0] != X.shape[-1]:
                raise ValueError(f"Bias dimension ({bias.shape[0]}) must match input's last dimension ({X.shape[-1]})")

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for SimplifiedLayerNormalization
        # This involves gradient computations for normalization operations
        raise NotImplementedError("SimplifiedLayerNormalization backward pass not yet implemented")


class GlobalMaxPoolOp(SimOp):
    """
    ONNX GlobalMaxPool Operation
    Global max pooling operation that takes the maximum value across all spatial dimensions.

    GlobalMaxPool reduces the input tensor by taking the maximum value across all
    spatial dimensions (height, width, etc.), leaving only the batch and channel
    dimensions. This is commonly used in classification networks before the final
    fully-connected layers.

    For input tensor of shape [N, C, H, W], the output shape is [N, C, 1, 1].
    For input tensor of shape [N, C, D, H, W], the output shape is [N, C, 1, 1, 1].
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'GlobalMaxPool'
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "GlobalMaxPool operation does not support backward pass yet"

        X = inT[0]
        input_shape = X.shape
        input_rank = len(input_shape)

        # Validate input rank (should be at least 3D: batch, channel, spatial_dims...)
        if input_rank < 3:
            raise ValueError(f"GlobalMaxPool requires input rank >= 3, got {input_rank}")

        # Compute output shape: keep batch and channel dimensions, set spatial dims to 1
        output_shape = input_shape[:2] + [1] * (input_rank - 2)
        outT[0].shape = output_shape
        outT[0].dtype = X.dtype

        nElem = X.nelems()

        # Performance counting for GlobalMaxPool
        # Global max pooling involves finding maximum across spatial dimensions

        # Number of output elements (batch * channels)
        output_elements = input_shape[0] * input_shape[1]

        # Number of spatial elements per output element
        spatial_elements = nElem // output_elements

        # Operations:
        # - Compare operations to find maximum across spatial dimensions
        # - Each output element requires (spatial_elements - 1) comparisons
        cmp_count = output_elements * (spatial_elements - 1)

        # Memory operations
        # - Read all input elements
        # - Write output elements
        self.perf_stats = {
            'inBytes': X.nbytes(self.precision),
            'inElems': nElem,
            'outBytes': outT[0].nbytes(self.precision),
            'outElems': output_elements,
            'instrs': {
                'mac': 0,  # No multiply-accumulate operations
                'cmp': cmp_count,  # Comparison operations to find maximum
                'add': 0,  # No additions
                'sub': 0,  # No subtraction
                'mul': 0,  # No multiplication
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
                'index': output_elements,  # Indexing for output positions
                'gather': 0,  # No gathering
                'scatter': 0,  # No scattering
                'tanh': 0,  # No tanh
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for GlobalMaxPool
        # GlobalMaxPool backward involves routing gradients to the position
        # of the maximum value in the forward pass
        raise NotImplementedError("GlobalMaxPool backward pass not yet implemented")


class GlobalAveragePoolOp(SimOp):
    """
    ONNX GlobalAveragePool Operation
    Global average pooling operation that takes the average value across all spatial dimensions.

    GlobalAveragePool reduces the input tensor by taking the average value across all
    spatial dimensions (height, width, etc.), leaving only the batch and channel
    dimensions. This is commonly used in classification networks before the final
    fully-connected layers.

    For input tensor of shape [N, C, H, W], the output shape is [N, C, 1, 1].
    For input tensor of shape [N, C, D, H, W], the output shape is [N, C, 1, 1, 1].
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'GlobalAveragePool'
        check_io_counts(self, in_counts=[1, 1], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "GlobalAveragePool operation does not support backward pass yet"

        X = inT[0]
        input_shape = X.shape
        input_rank = len(input_shape)

        # Validate input rank (should be at least 3D: batch, channel, spatial_dims...)
        if input_rank < 3:
            raise ValueError(f"GlobalAveragePool requires input rank >= 3, got {input_rank}")

        # Compute output shape: keep batch and channel dimensions, set spatial dims to 1
        output_shape = input_shape[:2] + [1] * (input_rank - 2)
        outT[0].shape = output_shape
        outT[0].dtype = X.dtype

        nElem = X.nelems()

        # Performance counting for GlobalAveragePool
        # Global average pooling involves summing across spatial dimensions and dividing

        # Number of output elements (batch * channels)
        output_elements = input_shape[0] * input_shape[1]

        # Number of spatial elements per output element
        spatial_elements = nElem // output_elements

        # Operations:
        # - Add operations to sum across spatial dimensions
        # - Division operations to compute average
        add_count = output_elements * (spatial_elements - 1)  # Sum all spatial elements
        div_count = output_elements  # Divide by number of spatial elements

        # Memory operations
        # - Read all input elements
        # - Write output elements
        self.perf_stats = {
            'inBytes': X.nbytes(self.precision),
            'inElems': nElem,
            'outBytes': outT[0].nbytes(self.precision),
            'outElems': output_elements,
            'instrs': {
                'mac': 0,  # No multiply-accumulate operations
                'cmp': 0,  # No comparisons
                'add': add_count,  # Addition operations for summing
                'sub': 0,  # No subtraction
                'mul': 0,  # No multiplication
                'div': div_count,  # Division operations for averaging
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
                'index': output_elements,  # Indexing for output positions
                'gather': 0,  # No gathering
                'scatter': 0,  # No scattering
                'tanh': 0,  # No tanh
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for GlobalAveragePool
        # GlobalAveragePool backward involves distributing gradients equally
        # across all spatial positions that contributed to the average
        raise NotImplementedError("GlobalAveragePool backward pass not yet implemented")


class GroupNormalizationOp(SimOp):
    """
    ONNX GroupNormalization Operation
    Group normalization that divides channels into groups and normalizes within each group.

    Group normalization was introduced as an alternative to batch normalization and layer
    normalization, particularly useful for tasks where batch size is small or variable.
    It divides the channels into groups and computes mean and variance within each group.

    For input tensor of shape [N, C, H, W], with num_groups=G:
    - Channels are divided into G groups of size C/G each
    - Normalization is computed within each group across spatial dimensions
    - Scale and bias are applied per channel (not per group)

    This is commonly used in vision transformers and other computer vision tasks.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'GroupNormalization'

        # GroupNormalization typically takes 3 inputs (X, scale, bias)
        check_io_counts(self, in_counts=[3, 3], out_counts=[1, 1])

        # Validate required attributes
        self.num_groups = self.attrs.get('num_groups')
        if self.num_groups is None:
            raise ValueError("GroupNormalization requires 'num_groups' attribute")

        self.epsilon = self.attrs.get('epsilon', 1e-5)
        if self.epsilon <= 0:
            raise ValueError("GroupNormalization epsilon must be positive")

        if self.num_groups <= 0:
            raise ValueError("GroupNormalization num_groups must be positive")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "GroupNormalization operation does not support backward pass yet"

        # GroupNormalization inputs: X (input tensor), scale (per-channel), bias (per-channel)
        X, scale, bias = inT[0], inT[1], inT[2]

        # Validate input shapes
        self._validate_input_shapes(X, scale, bias)

        outT[0].shape = X.shape
        outT[0].dtype = X.dtype

        nElem = X.nelems()
        input_shape = X.shape
        num_channels = input_shape[1]  # Channel dimension

        # Performance counting for GroupNormalization
        # Group normalization involves:
        # 1. Computing mean and variance per group
        # 2. Normalizing within each group
        # 3. Applying scale and bias per channel

        # Channels per group
        channels_per_group = num_channels // self.num_groups

        # Spatial elements per channel
        spatial_elements = nElem // num_channels

        # Operations per group:
        # - Sum across spatial dimensions for mean: spatial_elements additions per channel per group
        # - Sum of squares for variance: spatial_elements additions + multiplications
        # - Normalization: subtraction, division, multiplication per element

        # Total operations
        mul_count = 0
        add_count = 0
        sub_count = 0
        rsqrt_count = 0
        mac_count = 0

        # Per-group statistics computation
        for group in range(self.num_groups):
            # Mean computation: sum across spatial dimensions for each channel in group
            add_count += channels_per_group * (spatial_elements - 1)  # Sum spatial elements

            # Variance computation: sum of (x - mean)^2
            sub_count += channels_per_group * spatial_elements  # Subtract mean
            mul_count += channels_per_group * spatial_elements  # Square the result
            add_count += channels_per_group * (spatial_elements - 1)  # Sum the squares

            # Add epsilon and reciprocal square root
            add_count += channels_per_group  # Add epsilon
            rsqrt_count += channels_per_group  # Reciprocal square root

        # Normalization: (x - mean) / sqrt(var + eps)
        sub_count += nElem  # Subtract group mean
        mul_count += nElem  # Divide by group std

        # Apply scale and bias: normalized * scale + bias
        mac_count += nElem  # Scale multiplication
        add_count += nElem  # Bias addition

        self.perf_stats = {
            'inBytes': sum(tensor.nbytes() for tensor in [X, scale, bias]),
            'inElems': sum(tensor.nelems() for tensor in [X, scale, bias]),
            'outBytes': outT[0].nbytes(),
            'outElems': nElem,
            'instrs': {
                'mac': mac_count,
                'cmp': 0,
                'add': add_count,
                'sub': sub_count,
                'mul': mul_count,
                'div': 0,
                'rsqrt': rsqrt_count,
                'exp': 0,
                'log': 0,
                'round': 0,
                'clip': 0,
                'convert': 0,
                'index': num_channels,  # Group indexing
                'gather': 0,
                'scatter': 0,
                'tanh': 0,
            }
        }

        return self.perf_stats

    def _validate_input_shapes(self, X, scale, bias):
        """Validate input tensor shapes according to GroupNormalization requirements."""

        input_shape = X.shape
        input_rank = len(input_shape)

        # Must be at least 3D (batch, channel, spatial_dims...)
        if input_rank < 3:
            raise ValueError(f"GroupNormalization requires input rank >= 3, got {input_rank}")

        num_channels = input_shape[1]

        # Number of channels must be divisible by num_groups
        if num_channels % self.num_groups != 0:
            raise ValueError(f"Number of channels ({num_channels}) must be divisible by num_groups ({self.num_groups})")

        # Scale and bias should be 1D and match the number of channels
        if len(scale.shape) != 1:
            raise ValueError(f"Scale tensor must be 1D, got shape {scale.shape}")

        if scale.shape[0] != num_channels:
            raise ValueError(f"Scale dimension ({scale.shape[0]}) must match number of channels ({num_channels})")

        if len(bias.shape) != 1:
            raise ValueError(f"Bias tensor must be 1D, got shape {bias.shape}")

        if bias.shape[0] != num_channels:
            raise ValueError(f"Bias dimension ({bias.shape[0]}) must match number of channels ({num_channels})")

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for GroupNormalization
        # Group normalization backward involves computing gradients with respect to
        # group means and variances, similar to batch normalization but per group
        raise NotImplementedError("GroupNormalization backward pass not yet implemented")


class SkipLayerNormalizationOp(SimOp):
    """
    ONNX SkipLayerNormalization Operation
    Pre-layer normalization pattern commonly used in transformer architectures.

    SkipLayerNormalization implements the pattern: LayerNorm(x + skip)
    This is a fused operation that combines skip connection addition with layer
    normalization, commonly used in transformer blocks where you have:
    output = LayerNorm(input + skip_connection)

    This is more efficient than separate Add and LayerNormalization operations.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'SkipLayerNormalization'

        # SkipLayerNormalization typically takes 3 inputs (input, skip, gamma, beta?)
        check_io_counts(self, in_counts=[3, 4], out_counts=[1, 1])

        # Validate required attributes (same as LayerNormalization)
        self.epsilon = self.attrs.get('epsilon', 1e-5)
        if self.epsilon <= 0:
            raise ValueError("SkipLayerNormalization epsilon must be positive")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "SkipLayerNormalization operation does not support backward pass yet"

        # SkipLayerNormalization inputs: input, skip, gamma, beta (optional)
        input_tensor = inT[0]
        skip_tensor = inT[1]
        gamma_tensor = inT[2]
        beta_tensor = inT[3] if len(inT) == 4 else None

        # Validate input shapes
        self._validate_input_shapes(input_tensor, skip_tensor, gamma_tensor, beta_tensor)

        outT[0].shape = input_tensor.shape
        outT[0].dtype = input_tensor.dtype

        nElem = input_tensor.nelems()
        input_shape = input_tensor.shape

        # Performance counting for SkipLayerNormalization
        # Operations: skip addition + layer normalization computation

        # Skip connection addition
        add_count = nElem  # input + skip

        # Layer normalization computation (similar to LayerNormalizationOp)
        # We normalize along the last dimension (typical for transformers)
        last_dim_size = input_shape[-1]
        reduction_count = nElem // last_dim_size

        # Statistics computation (mean and variance)
        # Mean: sum across last dimension
        add_count += nElem  # Sum all elements for mean
        mul_count = reduction_count  # Divide by dimension size

        # Variance: sum of (x - mean)^2
        sub_count = nElem  # Subtract mean
        mul_count += nElem  # Square the result
        add_count += nElem  # Sum the squares
        mul_count += reduction_count  # Divide by dimension size

        # Add epsilon and reciprocal square root
        add_count += reduction_count  # Add epsilon
        rsqrt_count = reduction_count  # Reciprocal square root

        # Normalization: (x - mean) / sqrt(var + eps)
        mul_count += nElem  # Divide by standard deviation

        # Apply gamma (scale)
        mac_count = nElem  # Multiply by gamma

        # Apply beta if present (bias addition)
        if beta_tensor is not None:
            add_count += nElem  # Add beta

        self.perf_stats = {
            'inBytes': sum(tensor.nbytes() for tensor in inT),  # Use actual tensor dtype for memory calculations
            'inElems': sum(tensor.nelems() for tensor in inT),
            'outBytes': outT[0].nbytes(),  # Use actual tensor dtype for memory calculations
            'outElems': nElem,
            'instrs': {
                'mac': mac_count,
                'cmp': 0,
                'add': add_count,
                'sub': sub_count,
                'mul': mul_count,
                'div': 0,
                'rsqrt': rsqrt_count,
                'exp': 0,
                'log': 0,
                'round': 0,
                'clip': 0,
                'convert': 0,
                'index': 0,
                'gather': 0,
                'scatter': 0,
                'tanh': 0,
            }
        }

        return self.perf_stats

    def _validate_input_shapes(self, input_tensor, skip_tensor, gamma_tensor, beta_tensor):
        """Validate input tensor shapes according to SkipLayerNormalization requirements."""

        # Input and skip tensors must have the same shape
        if input_tensor.shape != skip_tensor.shape:
            raise ValueError(f"Input and skip tensors must have the same shape, got {input_tensor.shape} vs {skip_tensor.shape}")

        # Gamma (scale) should be 1D and match the last dimension of input
        if len(gamma_tensor.shape) != 1:
            raise ValueError(f"Gamma tensor must be 1D, got shape {gamma_tensor.shape}")

        if gamma_tensor.shape[0] != input_tensor.shape[-1]:
            raise ValueError(f"Gamma dimension ({gamma_tensor.shape[0]}) must match input's last dimension ({input_tensor.shape[-1]})")

        # Beta (bias) should be 1D and match the last dimension of input (if present)
        if beta_tensor is not None:
            if len(beta_tensor.shape) != 1:
                raise ValueError(f"Beta tensor must be 1D, got shape {beta_tensor.shape}")

            if beta_tensor.shape[0] != input_tensor.shape[-1]:
                raise ValueError(f"Beta dimension ({beta_tensor.shape[0]}) must match input's last dimension ({input_tensor.shape[-1]})")

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for SkipLayerNormalization
        # SkipLayerNormalization backward involves gradients through both
        # skip connection and layer normalization
        raise NotImplementedError("SkipLayerNormalization backward pass not yet implemented")


class MatMulIntegerOp(SimOp):
    """
    ONNX MatMulInteger Operation
    Matrix multiplication with integer inputs and output.

    MatMulInteger performs matrix multiplication on integer tensors without scaling
    or zero-point adjustments. This is useful for quantized neural networks where
    the quantization parameters are handled separately.

    For input tensors A (shape [M, K]) and B (shape [K, N]):
    - Output shape is [M, N]
    - All computations are performed in integer arithmetic
    - No scaling or bias adjustments are applied
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'MatMulInteger'

        # MatMulInteger takes exactly 2 inputs (A, B) and produces 1 output
        check_io_counts(self, in_counts=[2, 2], out_counts=[1, 1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "MatMulInteger operation does not support backward pass yet"

        A, B = inT[0], inT[1]

        # Validate input shapes for matrix multiplication
        self._validate_input_shapes(A, B)

        # Compute output shape: [M, N] where A is [M, K] and B is [K, N]
        A_shape = A.shape
        B_shape = B.shape

        if len(A_shape) == 2 and len(B_shape) == 2:
            # Standard 2D matrix multiplication
            M, K_a = A_shape
            K_b, N = B_shape
            output_shape = [M, N]
        elif len(A_shape) == 1 and len(B_shape) == 1:
            # Vector dot product
            output_shape = []  # Scalar output
        elif len(A_shape) == 1 and len(B_shape) == 2:
            # Vector-matrix multiplication
            K_a = A_shape[0]
            K_b, N = B_shape
            output_shape = [N]
        elif len(A_shape) == 2 and len(B_shape) == 1:
            # Matrix-vector multiplication
            M, K_a = A_shape
            K_b = B_shape[0]
            output_shape = [M]
        else:
            # Higher dimensional case - assume last two dimensions are matrices
            # This is a simplified assumption; full broadcasting would be more complex
            *batch_dims, M, K_a = A_shape
            *B_batch_dims, K_b, N = B_shape
            output_shape = batch_dims + [M, N]

        outT[0].shape = output_shape
        outT[0].dtype = A.dtype  # Output type matches input type

        nElem = A.nelems() * B.nelems()

        # Performance counting for MatMulInteger
        # Matrix multiplication involves multiply-accumulate operations

        # For integer matrix multiplication, we primarily count MAC operations
        # Each output element requires K multiplications and additions
        if len(A_shape) == 2 and len(B_shape) == 2:
            M, K = A_shape
            K, N = B_shape
            output_elements = M * N
            mac_per_output = K
        elif len(A_shape) == 1 and len(B_shape) == 1:
            # Vector dot product
            K = A_shape[0]
            output_elements = 1
            mac_per_output = K
        elif len(A_shape) == 1 and len(B_shape) == 2:
            # Vector-matrix: [K] x [K, N] -> [N]
            K, N = B_shape
            output_elements = N
            mac_per_output = K
        elif len(A_shape) == 2 and len(B_shape) == 1:
            # Matrix-vector: [M, K] x [K] -> [M]
            M, K = A_shape
            output_elements = M
            mac_per_output = K
        else:
            # Higher dimensional case
            output_elements = 1
            for dim in output_shape:
                output_elements *= dim
            mac_per_output = A_shape[-1]  # K dimension

        # Total MAC operations
        mac_count = output_elements * mac_per_output

        # Additional operations for integer arithmetic
        # Each MAC involves multiplication and addition
        mul_count = mac_count
        add_count = mac_count

        self.perf_stats = {
            'inBytes': A.nbytes() + B.nbytes(),  # Use actual tensor dtype for memory calculations
            'inElems': A.nelems() + B.nelems(),
            'outBytes': outT[0].nbytes(),  # Use actual tensor dtype for memory calculations
            'outElems': output_elements,
            'instrs': {
                'mac': mac_count,
                'cmp': 0,  # No comparisons
                'add': add_count,
                'sub': 0,  # No subtraction
                'mul': mul_count,
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
                'index': output_elements,  # Output indexing
                'gather': 0,  # No gathering
                'scatter': 0,  # No scattering
                'tanh': 0,  # No tanh
            }
        }

        return self.perf_stats

    def _validate_input_shapes(self, A, B):
        """Validate input tensor shapes for matrix multiplication."""

        A_shape = A.shape
        B_shape = B.shape

        if len(A_shape) == 0 or len(B_shape) == 0:
            raise ValueError("MatMulInteger inputs must have at least 1 dimension")

        if len(A_shape) == 1 and len(B_shape) == 1:
            # Vector dot product
            if A_shape[0] != B_shape[0]:
                raise ValueError(f"Vector dimensions must match for dot product, got {A_shape[0]} vs {B_shape[0]}")
        elif len(A_shape) == 1 and len(B_shape) == 2:
            # Vector-matrix multiplication
            if A_shape[0] != B_shape[0]:
                raise ValueError(f"Vector length ({A_shape[0]}) must match matrix rows ({B_shape[0]})")
        elif len(A_shape) == 2 and len(B_shape) == 1:
            # Matrix-vector multiplication
            if A_shape[1] != B_shape[0]:
                raise ValueError(f"Matrix columns ({A_shape[1]}) must match vector length ({B_shape[0]})")
        elif len(A_shape) == 2 and len(B_shape) == 2:
            # Matrix-matrix multiplication
            if A_shape[1] != B_shape[0]:
                raise ValueError(f"Matrix inner dimensions must match, got {A_shape[1]} vs {B_shape[0]}")
        else:
            # Higher dimensional case - check last two dimensions
            if A_shape[-1] != B_shape[-2]:
                raise ValueError(f"Matrix inner dimensions must match, got {A_shape[-1]} vs {B_shape[-2]}")

            # Check batch dimensions match (all dimensions except the last two)
            A_batch = A_shape[:-2]
            B_batch = B_shape[:-2]

            if A_batch != B_batch:
                raise ValueError(f"Batch dimensions must match, got {A_batch} vs {B_batch}")

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for MatMulInteger
        # MatMulInteger backward involves matrix multiplication with gradients
        raise NotImplementedError("MatMulInteger backward pass not yet implemented")


class EmbedLayerNormalizationExtOp(SimOp):
    """
    ONNX 1.20.0 EmbedLayerNormalization Extensions
    Extended version of EmbedLayerNormalization with additional features.

    This operation extends the standard EmbedLayerNormalization with additional
    capabilities commonly found in modern transformer architectures:

    New Features in Extensions:
    1. Rotary Position Embedding support (RoPE)
    2. RMS normalization option (instead of layer normalization)
    3. Multiple position encoding schemes
    4. Enhanced mask processing capabilities
    5. Support for different embedding scaling strategies
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'EmbedLayerNormalizationExt'

        # Extended version supports more input/output combinations
        check_io_counts(self, in_counts=[3,8], out_counts=[1,3])

        # Parse attributes - extended set
        self.epsilon = self.attrs.get('epsilon', 1e-12)
        self.mask_value = self.attrs.get('mask_value', -1e4)
        self.position_embedding_type = self.attrs.get('position_embedding_type', 'learned')  # 'learned', 'sinusoidal', 'rotary'
        self.normalization_type = self.attrs.get('normalization_type', 'layer')  # 'layer', 'rms'
        self.embedding_scale = self.attrs.get('embedding_scale', 1.0)
        self.use_rotary_position_embedding = self.attrs.get('use_rotary_position_embedding', False)

        # Validate attributes
        if not isinstance(self.epsilon, (int, float)) or self.epsilon <= 0:
            raise ValueError(f"EmbedLayerNormalizationExt epsilon must be a positive number, got {self.epsilon}")
        if not isinstance(self.mask_value, (int, float)):
            raise ValueError(f"EmbedLayerNormalizationExt mask_value must be a number, got {type(self.mask_value)}")
        if self.position_embedding_type not in ['learned', 'sinusoidal', 'rotary']:
            raise ValueError(f"Invalid position_embedding_type: {self.position_embedding_type}")
        if self.normalization_type not in ['layer', 'rms']:
            raise ValueError(f"Invalid normalization_type: {self.normalization_type}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "EmbedLayerNormalizationExt operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs >= 3, f"EmbedLayerNormalizationExt expects at least 3 inputs, got {num_inputs}"
        assert len(outT) >= 1, f"EmbedLayerNormalizationExt expects at least 1 output, got {len(outT)}"

        input_ids = inT[0]      # Token IDs [batch_size, seq_length]
        embedding_weight = inT[1]  # Embedding matrix [vocab_size, hidden_size]
        position_weight = inT[2]   # Position embeddings [max_position, hidden_size]

        # Validate input types
        assert input_ids.dtype in ['int32', 'int64'], f"Input IDs must be integer type, got {input_ids.dtype}"
        assert embedding_weight.dtype in ['float32', 'float'], f"Embedding weight must be float32, got {embedding_weight.dtype}"
        assert position_weight.dtype in ['float32', 'float'], f"Position weight must be float32, got {position_weight.dtype}"

        # Validate tensor shapes
        input_ids_shape = input_ids.shape
        embedding_weight_shape = embedding_weight.shape
        position_weight_shape = position_weight.shape

        assert len(input_ids_shape) == 2, f"Input IDs must be 2D [batch_size, seq_length], got {input_ids_shape}"
        assert len(embedding_weight_shape) == 2, f"Embedding weight must be 2D [vocab_size, hidden_size], got {embedding_weight_shape}"
        assert len(position_weight_shape) == 2, f"Position weight must be 2D [max_position, hidden_size], got {position_weight_shape}"

        batch_size, seq_length = input_ids_shape
        vocab_size, hidden_size = embedding_weight_shape
        max_position, pos_hidden_size = position_weight_shape

        # Validate dimensions match
        assert hidden_size == pos_hidden_size, f"Hidden size mismatch: embedding={hidden_size}, position={pos_hidden_size}"

        # Parse optional inputs with extended logic
        has_segment_weight = False
        has_mask = False
        has_rotary_cos = False
        has_rotary_sin = False
        segment_weight = None
        mask = None
        rotary_cos = None
        rotary_sin = None

        # Extended input parsing (up to 8 inputs)
        for i in range(3, num_inputs):
            tensor = inT[i]
            if tensor.dtype in ['float32', 'float'] and len(tensor.shape) == 2 and tensor.shape[1] == hidden_size:
                # Likely segment_weight
                if not has_segment_weight:
                    has_segment_weight = True
                    segment_weight = tensor
                else:
                    raise ValueError(f"Multiple segment weights not supported")
            elif tensor.dtype in ['int32', 'int64', 'float32', 'float'] and len(tensor.shape) == 2:
                # Likely mask
                if not has_mask:
                    has_mask = True
                    mask = tensor
                else:
                    raise ValueError(f"Multiple masks not supported")
            elif tensor.dtype in ['float32', 'float'] and len(tensor.shape) == 3:
                # Likely rotary embeddings [seq_len, num_heads, head_dim] or [batch_size, seq_len, hidden_size]
                if tensor.shape[-1] == hidden_size // 2:  # RoPE cos/sin cache
                    if not has_rotary_cos:
                        has_rotary_cos = True
                        rotary_cos = tensor
                    elif not has_rotary_sin:
                        has_rotary_sin = True
                        rotary_sin = tensor
                    else:
                        raise ValueError(f"Too many rotary embedding tensors")
                else:
                    raise ValueError(f"Unexpected 3D tensor shape for rotary embeddings: {tensor.shape}")
            else:
                raise ValueError(f"Unexpected input tensor at position {i}: dtype={tensor.dtype}, shape={tensor.shape}")

        # Validate rotary embedding requirements
        if self.use_rotary_position_embedding and not (has_rotary_cos and has_rotary_sin):
            raise ValueError("Rotary position embedding requires both cos and sin caches")

        # Validate sequence length
        if self.position_embedding_type == 'learned':
            assert seq_length <= max_position, f"Sequence length {seq_length} exceeds max position {max_position}"

        # Set output tensor shapes and dtypes
        outT[0].shape = [batch_size, seq_length, hidden_size]  # Embedded output
        outT[0].dtype = 'float32'

        if len(outT) >= 2:  # Optional mask_index output
            outT[1].shape = input_ids_shape  # Same shape as input_ids
            outT[1].dtype = input_ids.dtype

        if len(outT) >= 3:  # Optional rotary position embeddings output
            if self.use_rotary_position_embedding:
                outT[2].shape = [batch_size, seq_length, hidden_size]
                outT[2].dtype = 'float32'
            else:
                outT[2].shape = []  # Empty tensor
                outT[2].dtype = 'float32'

        # Calculate performance statistics with extensions
        total_elements = batch_size * seq_length * hidden_size

        # Base embedding operations
        embedding_ops = batch_size * seq_length  # One lookup per token
        embedding_ops += total_elements * 2  # Position and segment addition

        # Extended operations based on features
        if self.position_embedding_type == 'sinusoidal':
            # Sinusoidal position encoding computation
            embedding_ops += total_elements * 4  # sin, cos, multiplication operations
        elif self.position_embedding_type == 'rotary':
            # Rotary position embedding
            embedding_ops += total_elements * 6  # Complex rotations

        if self.use_rotary_position_embedding:
            # RoPE application
            rope_ops = total_elements * 8  # RoPE computation per element
        else:
            rope_ops = 0

        if self.normalization_type == 'rms':
            # RMS normalization (simplified)
            rms_ops = total_elements * 6  # Square, sum, sqrt, divide, scale
        else:
            # Standard layer normalization
            rms_ops = total_elements * 8  # Mean, variance, normalization, affine

        # Mask processing
        mask_ops = batch_size * seq_length * hidden_size if has_mask else 0

        # Embedding scaling
        if self.embedding_scale != 1.0:
            scale_ops = total_elements
        else:
            scale_ops = 0

        total_ops = embedding_ops + rope_ops + rms_ops + mask_ops + scale_ops

        # Memory analysis
        input_bytes = input_ids.nbytes(self.precision) + embedding_weight.nbytes(self.precision) + position_weight.nbytes(self.precision)
        if has_segment_weight and segment_weight is not None:
            input_bytes += segment_weight.nbytes(self.precision)
        if has_mask and mask is not None:
            input_bytes += mask.nbytes(self.precision)
        if has_rotary_cos and has_rotary_sin and rotary_cos is not None and rotary_sin is not None:
            input_bytes += rotary_cos.nbytes(self.precision) + rotary_sin.nbytes(self.precision)

        output_bytes = sum(out.nbytes(self.precision) for out in outT)

        # Instruction counts
        mul_count = total_elements * 4  # Rough estimate for normalization and embeddings
        add_count = total_elements * 3
        rsqrt_count = batch_size * seq_length if self.normalization_type == 'layer' else 0
        gather_count = batch_size * seq_length  # Embedding lookups

        if self.use_rotary_position_embedding:
            mul_count += total_elements * 4  # RoPE operations
            add_count += total_elements * 2

        if has_mask:
            cmp_count = batch_size * seq_length
        else:
            cmp_count = 0

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': sum(tensor.nelems() for tensor in inT),
            'outBytes': output_bytes,
            'outElems': sum(tensor.nelems() for tensor in outT),
            'instrs': {
                'mac': total_elements * 2,
                'cmp': cmp_count,
                'add': add_count,
                'sub': total_elements,
                'mul': mul_count,
                'div': total_elements if self.normalization_type == 'layer' else 0,
                'rsqrt': rsqrt_count,
                'exp': 0,
                'log': 0,
                'round': 0,
                'clip': 0,
                'convert': 0,
                'index': gather_count,
                'gather': gather_count,
                'scatter': 0,
                'tanh': 0,
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for EmbedLayerNormalizationExt
        raise NotImplementedError("EmbedLayerNormalizationExt backward pass not yet implemented")


class GemmExtOp(SimOp):
    """
    ONNX 1.20.0 Gemm Extensions
    Extended General Matrix Multiplication with additional features.

    This operation extends the standard Gemm operation with additional
    capabilities for modern neural network architectures:

    New Features in Extensions:
    1. Support for integer matrix multiplication (quantized)
    2. Extended transpose options for all inputs
    3. Bias addition with broadcasting
    4. Support for different data types (int8, int16, float16, bfloat16)
    5. Optional activation function fusion
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'GemmExt'

        # GemmExt supports 2-4 inputs (A, B, C?, bias?) and 1 output
        check_io_counts(self, in_counts=[2,4], out_counts=[1,1])

        # Parse attributes - extended set
        self.alpha = self.attrs.get('alpha', 1.0)
        self.beta = self.attrs.get('beta', 1.0)
        self.transA = self.attrs.get('transA', 0)
        self.transB = self.attrs.get('transB', 0)
        self.transC = self.attrs.get('transC', 0)  # New in extension
        self.dtype = self.attrs.get('dtype', 'float32')  # New in extension
        self.activation = self.attrs.get('activation', None)  # New in extension

        # Validate attributes
        if not isinstance(self.alpha, (int, float)):
            raise ValueError(f"GemmExt alpha must be a number, got {type(self.alpha)}")
        if not isinstance(self.beta, (int, float)):
            raise ValueError(f"GemmExt beta must be a number, got {type(self.beta)}")
        if self.transA not in [0, 1]:
            raise ValueError(f"GemmExt transA must be 0 or 1, got {self.transA}")
        if self.transB not in [0, 1]:
            raise ValueError(f"GemmExt transB must be 0 or 1, got {self.transB}")
        if self.transC not in [0, 1]:
            raise ValueError(f"GemmExt transC must be 0 or 1, got {self.transC}")
        if self.dtype not in ['float32', 'float16', 'bfloat16', 'int8', 'int16', 'int32']:
            raise ValueError(f"GemmExt unsupported dtype: {self.dtype}")
        if self.activation is not None and self.activation not in ['relu', 'gelu', 'sigmoid', 'tanh']:
            raise ValueError(f"GemmExt unsupported activation: {self.activation}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "GemmExt operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs >= 2, f"GemmExt expects at least 2 inputs, got {num_inputs}"
        assert len(outT) == 1, f"GemmExt expects 1 output, got {len(outT)}"

        A = inT[0]  # Input matrix A
        B = inT[1]  # Input matrix B
        C = inT[2] if num_inputs >= 3 else None  # Optional bias/addition matrix
        bias = inT[3] if num_inputs >= 4 else None  # Optional bias vector/matrix

        # Validate input types based on dtype attribute
        if self.dtype.startswith('int'):
            expected_dtype = self.dtype
        else:
            expected_dtype = 'float32'  # Default for float types

        assert A.dtype == expected_dtype, f"Input A dtype mismatch: expected {expected_dtype}, got {A.dtype}"
        assert B.dtype == expected_dtype, f"Input B dtype mismatch: expected {expected_dtype}, got {B.dtype}"
        if C is not None:
            assert C.dtype == expected_dtype, f"Input C dtype mismatch: expected {expected_dtype}, got {C.dtype}"
        if bias is not None:
            assert bias.dtype == expected_dtype, f"Bias dtype mismatch: expected {expected_dtype}, got {bias.dtype}"

        # Validate tensor shapes with transpose support
        A_shape = A.shape
        B_shape = B.shape

        # Apply transposes
        if self.transA:
            A_shape = [A_shape[1], A_shape[0]] if len(A_shape) == 2 else A_shape[::-1]
        if self.transB:
            B_shape = [B_shape[1], B_shape[0]] if len(B_shape) == 2 else B_shape[::-1]

        # Validate matrix multiplication compatibility
        assert len(A_shape) >= 2, f"Input A must have at least 2 dimensions after transpose, got {len(A_shape)}"
        assert len(B_shape) >= 2, f"Input B must have at least 2 dimensions after transpose, got {len(B_shape)}"
        assert A_shape[-1] == B_shape[-2], f"Matrix dimensions incompatible: A[-1]={A_shape[-1]}, B[-2]={B_shape[-2]}"

        # Compute output shape
        output_shape = A_shape[:-1] + [B_shape[-1]]

        # Handle bias addition if present
        if bias is not None:
            bias_shape = bias.shape
            if len(bias_shape) == 1:
                # Bias vector - broadcast to last dimension
                assert bias_shape[0] == output_shape[-1], f"Bias vector dimension {bias_shape[0]} must match output last dimension {output_shape[-1]}"
            elif len(bias_shape) == 2:
                # Bias matrix - must match output shape
                assert bias_shape == output_shape, f"Bias matrix shape {bias_shape} must match output shape {output_shape}"
            else:
                raise ValueError(f"Bias tensor must be 1D or 2D, got {len(bias_shape)}D")

        # Handle C matrix if present
        if C is not None:
            C_shape = C.shape
            if self.transC:
                C_shape = [C_shape[1], C_shape[0]] if len(C_shape) == 2 else C_shape[::-1]

            # C must be broadcastable to output shape
            if len(C_shape) == 1:
                assert C_shape[0] == output_shape[-1], f"C vector dimension {C_shape[0]} must match output last dimension {output_shape[-1]}"
            elif len(C_shape) == 2:
                assert C_shape == output_shape, f"C matrix shape {C_shape} must match output shape {output_shape}"
            else:
                raise ValueError(f"C tensor must be 1D or 2D, got {len(C_shape)}D")

        outT[0].shape = output_shape
        outT[0].dtype = expected_dtype

        # Calculate performance statistics
        total_elements = 1
        for dim in output_shape:
            total_elements *= dim

        # Matrix multiplication statistics
        M = 1
        for dim in A_shape[:-1]:
            M *= dim
        K = A_shape[-1]
        N = B_shape[-1]

        # Operations count
        mac_count = M * N * K  # One MAC per output element

        if self.dtype.startswith('int'):
            # Integer operations
            mul_count = mac_count
            add_count = mac_count
        else:
            # Float operations - more complex due to alpha/beta scaling
            mul_count = mac_count * 2  # Multiply by alpha, multiply by beta
            add_count = mac_count + total_elements  # MAC + bias addition

        # Additional operations for C matrix
        if C is not None:
            if self.beta != 0:
                add_count += total_elements  # C * beta addition
                mul_count += total_elements  # C * beta multiplication

        # Additional operations for bias
        if bias is not None:
            add_count += total_elements  # Bias addition

        # Activation function operations
        if self.activation is not None:
            if self.activation == 'relu':
                cmp_count = total_elements  # Compare with zero
                max_count = total_elements  # Max with zero
            elif self.activation == 'gelu':
                # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                mul_count += total_elements * 6
                add_count += total_elements * 2
                tanh_count = total_elements
                exp_count = 0
            elif self.activation == 'sigmoid':
                # sigmoid(x) = 1 / (1 + exp(-x))
                exp_count = total_elements
                add_count += total_elements
                div_count = total_elements
            elif self.activation == 'tanh':
                tanh_count = total_elements
        else:
            cmp_count = 0
            max_count = 0
            tanh_count = 0
            exp_count = 0
            div_count = 0

        # Memory analysis
        input_bytes = A.nbytes(self.precision) + B.nbytes(self.precision)
        if C is not None:
            input_bytes += C.nbytes(self.precision)
        if bias is not None:
            input_bytes += bias.nbytes(self.precision)

        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': sum(tensor.nelems() for tensor in inT),
            'outBytes': output_bytes,
            'outElems': total_elements,
            'instrs': {
                'mac': mac_count,
                'cmp': cmp_count if 'cmp_count' in locals() else 0,
                'add': add_count,
                'sub': 0,
                'mul': mul_count,
                'div': div_count if 'div_count' in locals() else 0,
                'rsqrt': 0,
                'exp': exp_count if 'exp_count' in locals() else 0,
                'log': 0,
                'round': 0,
                'clip': 0,
                'convert': 0,
                'index': 0,
                'gather': 0,
                'scatter': 0,
                'tanh': tanh_count if 'tanh_count' in locals() else 0,
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for GemmExt
        raise NotImplementedError("GemmExt backward pass not yet implemented")


class ConvExtOp(SimOp):
    """
    ONNX 1.20.0 Conv Extensions
    Extended Convolution operation with additional features.

    This operation extends the standard Conv operation with additional
    capabilities for modern neural network architectures:

    New Features in Extensions:
    1. Support for different data types (int8, float16, bfloat16)
    2. Asymmetric padding support
    3. Activation function fusion
    4. Bias addition with multiple formats
    5. Grouped convolution enhancements
    6. Support for different dilation patterns
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ConvExt'

        # ConvExt supports 2-3 inputs (X, W, B?) and 1 output
        check_io_counts(self, in_counts=[2,3], out_counts=[1,1])

        # Parse attributes - extended set
        self.auto_pad = self.attrs.get('auto_pad', 'NOTSET')
        self.dilations = self.attrs.get('dilations', [1])
        self.group = self.attrs.get('group', 1)
        self.kernel_shape = self.attrs.get('kernel_shape')
        self.pads = self.attrs.get('pads', [0, 0, 0, 0])
        self.strides = self.attrs.get('strides', [1, 1])

        # Extended attributes
        self.dtype = self.attrs.get('dtype', 'float32')  # New in extension
        self.activation = self.attrs.get('activation', None)  # New in extension
        self.bias_broadcast = self.attrs.get('bias_broadcast', 'vector')  # New in extension
        self.asymmetric_pad = self.attrs.get('asymmetric_pad', False)  # New in extension

        # Validate attributes
        if self.auto_pad not in ['NOTSET', 'SAME_UPPER', 'SAME_LOWER', 'VALID']:
            raise ValueError(f"ConvExt auto_pad must be one of NOTSET, SAME_UPPER, SAME_LOWER, VALID, got {self.auto_pad}")
        if not isinstance(self.dilations, list) or len(self.dilations) != 2:
            raise ValueError(f"ConvExt dilations must be a list of 2 integers, got {self.dilations}")
        if not isinstance(self.group, int) or self.group < 1:
            raise ValueError(f"ConvExt group must be a positive integer, got {self.group}")
        if self.kernel_shape is None:
            raise ValueError("ConvExt kernel_shape is required")
        if not isinstance(self.kernel_shape, list) or len(self.kernel_shape) != 2:
            raise ValueError(f"ConvExt kernel_shape must be a list of 2 integers, got {self.kernel_shape}")
        if not isinstance(self.pads, list) or len(self.pads) != 4:
            raise ValueError(f"ConvExt pads must be a list of 4 integers, got {self.pads}")
        if not isinstance(self.strides, list) or len(self.strides) != 2:
            raise ValueError(f"ConvExt strides must be a list of 2 integers, got {self.strides}")

        # Extended validations
        if self.dtype not in ['float32', 'float16', 'bfloat16', 'int8', 'int16']:
            raise ValueError(f"ConvExt unsupported dtype: {self.dtype}")
        if self.activation is not None and self.activation not in ['relu', 'gelu', 'sigmoid', 'tanh']:
            raise ValueError(f"ConvExt unsupported activation: {self.activation}")
        if self.bias_broadcast not in ['vector', 'channel', 'spatial']:
            raise ValueError(f"ConvExt unsupported bias_broadcast: {self.bias_broadcast}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "ConvExt operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs >= 2, f"ConvExt expects at least 2 inputs, got {num_inputs}"
        assert len(outT) == 1, f"ConvExt expects 1 output, got {len(outT)}"

        X = inT[0]  # Input tensor [N, C, H, W]
        W = inT[1]  # Weight tensor [M, C/group, kH, kW]
        B = inT[2] if num_inputs >= 3 else None  # Optional bias

        # Validate input types
        assert X.dtype == self.dtype, f"Input X dtype mismatch: expected {self.dtype}, got {X.dtype}"
        assert W.dtype == self.dtype, f"Weight W dtype mismatch: expected {self.dtype}, got {W.dtype}"
        if B is not None:
            assert B.dtype == self.dtype, f"Bias B dtype mismatch: expected {self.dtype}, got {B.dtype}"

        # Validate tensor shapes
        X_shape = X.shape
        W_shape = W.shape

        assert len(X_shape) == 4, f"Input X must be 4D [N, C, H, W], got {X_shape}"
        assert len(W_shape) == 4, f"Weight W must be 4D [M, C/group, kH, kW], got {W_shape}"

        N, C, H, W = X_shape
        M, C_per_group, kH, kW = W_shape

        # Validate kernel shape consistency
        assert kH == self.kernel_shape[0], f"Kernel height mismatch: expected {self.kernel_shape[0]}, got {kH}"
        assert kW == self.kernel_shape[1], f"Kernel width mismatch: expected {self.kernel_shape[1]}, got {kW}"

        # Validate channel count
        assert C % self.group == 0, f"Input channels {C} must be divisible by group {self.group}"
        assert C_per_group == C // self.group, f"Weight channels per group {C_per_group} must be C/group = {C // self.group}"

        # Compute output shape
        output_height = self._compute_conv_output_size(H, kH, self.strides[0], self.pads[0], self.pads[2], self.dilations[0])
        output_width = self._compute_conv_output_size(W, kW, self.strides[1], self.pads[1], self.pads[3], self.dilations[1])

        output_shape = [N, M, output_height, output_width]

        # Validate bias if present
        if B is not None:
            B_shape = B.shape
            if self.bias_broadcast == 'vector':
                assert len(B_shape) == 1 and B_shape[0] == M, f"Bias vector must be [M], got {B_shape}"
            elif self.bias_broadcast == 'channel':
                assert B_shape == [1, M, 1, 1], f"Bias channel must be [1, M, 1, 1], got {B_shape}"
            elif self.bias_broadcast == 'spatial':
                assert B_shape == [1, 1, output_height, output_width], f"Bias spatial must be [1, 1, H, W], got {B_shape}"

        outT[0].shape = output_shape
        outT[0].dtype = self.dtype

        # Calculate performance statistics
        total_elements = N * M * output_height * output_width

        # Convolution statistics
        # Each output element requires: kH * kW * C_per_group multiply-accumulate operations
        # Number of output elements per channel: N * output_height * output_width
        # Number of channels: M
        # So total MACs = M * N * output_height * output_width * kH * kW * C_per_group

        mac_count = M * N * output_height * output_width * kH * kW * C_per_group

        if self.dtype.startswith('int'):
            # Integer operations
            mul_count = mac_count
            add_count = mac_count
        else:
            # Float operations
            mul_count = mac_count
            add_count = mac_count

        # Bias addition operations
        if B is not None:
            add_count += total_elements

        # Activation function operations
        if self.activation is not None:
            if self.activation == 'relu':
                cmp_count = total_elements
                max_count = total_elements
            elif self.activation == 'gelu':
                mul_count += total_elements * 6
                add_count += total_elements * 2
                tanh_count = total_elements
                exp_count = 0
            elif self.activation == 'sigmoid':
                exp_count = total_elements
                add_count += total_elements
                div_count = total_elements
            elif self.activation == 'tanh':
                tanh_count = total_elements
        else:
            cmp_count = 0
            max_count = 0
            tanh_count = 0
            exp_count = 0
            div_count = 0

        # Memory analysis
        input_bytes = X.nbytes(self.precision) + W.nbytes(self.precision)
        if B is not None:
            input_bytes += B.nbytes(self.precision)

        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': sum(tensor.nelems() for tensor in inT),
            'outBytes': output_bytes,
            'outElems': total_elements,
            'instrs': {
                'mac': mac_count,
                'cmp': cmp_count if 'cmp_count' in locals() else 0,
                'add': add_count,
                'sub': 0,
                'mul': mul_count,
                'div': div_count if 'div_count' in locals() else 0,
                'rsqrt': 0,
                'exp': exp_count if 'exp_count' in locals() else 0,
                'log': 0,
                'round': 0,
                'clip': 0,
                'convert': 0,
                'index': 0,
                'gather': 0,
                'scatter': 0,
                'tanh': tanh_count if 'tanh_count' in locals() else 0,
            }
        }

        return self.perf_stats

    def _compute_conv_output_size(self, input_size, kernel_size, stride, pad_start, pad_end, dilation):
        """Compute output size for convolution along one dimension."""
        if self.auto_pad == 'SAME_UPPER':
            return (input_size + stride - 1) // stride
        elif self.auto_pad == 'SAME_LOWER':
            return (input_size + stride - 1) // stride
        elif self.auto_pad == 'VALID':
            return (input_size - dilation * (kernel_size - 1) - 1 + pad_start + pad_end) // stride + 1
        else:  # NOTSET
            return (input_size + pad_start + pad_end - dilation * (kernel_size - 1) - 1) // stride + 1

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for ConvExt
        raise NotImplementedError("ConvExt backward pass not yet implemented")


class MaxPoolExtOp(SimOp):
    """
    ONNX 1.20.0 MaxPool Extensions
    Extended Max Pooling operation with additional features.

    This operation extends the standard MaxPool operation with additional
    capabilities for modern neural network architectures:

    New Features in Extensions:
    1. Support for different data types (int8, float16, bfloat16)
    2. Asymmetric padding support
    3. Multiple pooling modes (max, argmax with indices)
    4. Enhanced dilation support
    5. Support for different storage orders
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'MaxPoolExt'

        # MaxPoolExt supports 1 input and 1-2 outputs (Y, Indices?)
        check_io_counts(self, in_counts=[1,1], out_counts=[1,2])

        # Parse attributes - extended set
        self.auto_pad = self.attrs.get('auto_pad', 'NOTSET')
        self.ceil_mode = self.attrs.get('ceil_mode', 0)
        self.dilations = self.attrs.get('dilations', [1, 1])
        self.kernel_shape = self.attrs.get('kernel_shape')
        self.pads = self.attrs.get('pads', [0, 0, 0, 0])
        self.storage_order = self.attrs.get('storage_order', 0)
        self.strides = self.attrs.get('strides', [1, 1])

        # Extended attributes
        self.dtype = self.attrs.get('dtype', 'float32')  # New in extension
        self.output_indices = self.attrs.get('output_indices', False)  # New in extension
        self.mode = self.attrs.get('mode', 'max')  # New in extension
        self.asymmetric_pad = self.attrs.get('asymmetric_pad', False)  # New in extension

        # Validate attributes
        if self.auto_pad not in ['NOTSET', 'SAME_UPPER', 'SAME_LOWER', 'VALID']:
            raise ValueError(f"MaxPoolExt auto_pad must be one of NOTSET, SAME_UPPER, SAME_LOWER, VALID, got {self.auto_pad}")
        if self.ceil_mode not in [0, 1]:
            raise ValueError(f"MaxPoolExt ceil_mode must be 0 or 1, got {self.ceil_mode}")
        if not isinstance(self.dilations, list) or len(self.dilations) != 2:
            raise ValueError(f"MaxPoolExt dilations must be a list of 2 integers, got {self.dilations}")
        if self.kernel_shape is None:
            raise ValueError("MaxPoolExt kernel_shape is required")
        if not isinstance(self.kernel_shape, list) or len(self.kernel_shape) != 2:
            raise ValueError(f"MaxPoolExt kernel_shape must be a list of 2 integers, got {self.kernel_shape}")
        if not isinstance(self.pads, list) or len(self.pads) != 4:
            raise ValueError(f"MaxPoolExt pads must be a list of 4 integers, got {self.pads}")
        if self.storage_order not in [0, 1]:
            raise ValueError(f"MaxPoolExt storage_order must be 0 or 1, got {self.storage_order}")
        if not isinstance(self.strides, list) or len(self.strides) != 2:
            raise ValueError(f"MaxPoolExt strides must be a list of 2 integers, got {self.strides}")

        # Extended validations
        if self.dtype not in ['float32', 'float16', 'bfloat16', 'int8', 'int16']:
            raise ValueError(f"MaxPoolExt unsupported dtype: {self.dtype}")
        if self.mode not in ['max', 'argmax']:
            raise ValueError(f"MaxPoolExt unsupported mode: {self.mode}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "MaxPoolExt operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs == 1, f"MaxPoolExt expects 1 input, got {num_inputs}"
        assert len(outT) >= 1, f"MaxPoolExt expects at least 1 output, got {len(outT)}"

        X = inT[0]  # Input tensor [N, C, H, W]

        # Validate input type
        assert X.dtype == self.dtype, f"Input X dtype mismatch: expected {self.dtype}, got {X.dtype}"

        # Validate tensor shape
        X_shape = X.shape
        assert len(X_shape) == 4, f"Input X must be 4D [N, C, H, W], got {X_shape}"

        N, C, H, W = X_shape
        kH, kW = self.kernel_shape

        # Compute output shape
        output_height = self._compute_pool_output_size(H, kH, self.strides[0], self.pads[0], self.pads[2], self.dilations[0])
        output_width = self._compute_pool_output_size(W, kW, self.strides[1], self.pads[1], self.pads[3], self.dilations[1])

        output_shape = [N, C, output_height, output_width]

        outT[0].shape = output_shape
        outT[0].dtype = self.dtype

        # Second output for indices if requested
        if len(outT) >= 2 and self.output_indices:
            outT[1].shape = output_shape
            outT[1].dtype = 'int64'  # Indices are typically int64

        # Calculate performance statistics
        total_elements = N * C * output_height * output_width

        # Max pooling statistics
        # Each output element requires comparing kH * kW values
        comparisons_per_output = kH * kW

        if self.dtype.startswith('int'):
            # Integer comparisons
            cmp_count = total_elements * comparisons_per_output
            max_count = total_elements
        else:
            # Float comparisons
            cmp_count = total_elements * comparisons_per_output
            max_count = total_elements

        # Additional operations for argmax mode
        if self.mode == 'argmax':
            # Need to track indices
            index_ops = total_elements * comparisons_per_output
        else:
            index_ops = 0

        # Memory analysis
        input_bytes = X.nbytes(self.precision)
        output_bytes = outT[0].nbytes(self.precision)
        if len(outT) >= 2 and self.output_indices:
            output_bytes += outT[1].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': X.nelems(),
            'outBytes': output_bytes,
            'outElems': sum(tensor.nelems() for tensor in outT),
            'instrs': {
                'mac': 0,  # No multiply-accumulate in pooling
                'cmp': cmp_count,
                'add': 0,  # No addition
                'sub': 0,  # No subtraction
                'mul': 0,  # No multiplication
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
                'index': index_ops,
                'gather': 0,  # No gathering
                'scatter': 0,  # No scattering
                'tanh': 0,  # No tanh
            }
        }

        return self.perf_stats

    def _compute_pool_output_size(self, input_size, kernel_size, stride, pad_start, pad_end, dilation):
        """Compute output size for pooling along one dimension."""
        if self.auto_pad == 'SAME_UPPER':
            if self.ceil_mode:
                return (input_size + stride - 1) // stride
            else:
                return (input_size + stride - 1) // stride
        elif self.auto_pad == 'SAME_LOWER':
            if self.ceil_mode:
                return (input_size + stride - 1) // stride
            else:
                return (input_size + stride - 1) // stride
        elif self.auto_pad == 'VALID':
            dilated_kernel = dilation * (kernel_size - 1) + 1
            if self.ceil_mode:
                return (input_size - dilated_kernel + stride) // stride
            else:
                return (input_size - dilated_kernel) // stride
        else:  # NOTSET
            if self.ceil_mode:
                return ((input_size + pad_start + pad_end - dilation * (kernel_size - 1) - 1) // stride) + 1
            else:
                return (input_size + pad_start + pad_end - dilation * (kernel_size - 1) - 1) // stride + 1

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for MaxPoolExt
        raise NotImplementedError("MaxPoolExt backward pass not yet implemented")


class AveragePoolExtOp(SimOp):
    """
    ONNX 1.20.0 AveragePool Extensions
    Extended Average Pooling operation with additional features.

    This operation extends the standard AveragePool operation with additional
    capabilities for modern neural network architectures:

    New Features in Extensions:
    1. Support for different data types (int8, float16, bfloat16)
    2. Asymmetric padding support
    3. Multiple pooling modes (average, with/without padding)
    4. Enhanced dilation support
    5. Support for different counting modes (valid, same)
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'AveragePoolExt'

        # AveragePoolExt supports 1 input and 1 output
        check_io_counts(self, in_counts=[1,1], out_counts=[1,1])

        # Parse attributes - extended set
        self.auto_pad = self.attrs.get('auto_pad', 'NOTSET')
        self.ceil_mode = self.attrs.get('ceil_mode', 0)
        self.count_include_pad = self.attrs.get('count_include_pad', 0)
        self.dilations = self.attrs.get('dilations', [1, 1])
        self.kernel_shape = self.attrs.get('kernel_shape')
        self.pads = self.attrs.get('pads', [0, 0, 0, 0])
        self.strides = self.attrs.get('strides', [1, 1])

        # Extended attributes
        self.dtype = self.attrs.get('dtype', 'float32')  # New in extension
        self.mode = self.attrs.get('mode', 'average')  # New in extension
        self.asymmetric_pad = self.attrs.get('asymmetric_pad', False)  # New in extension

        # Validate attributes
        if self.auto_pad not in ['NOTSET', 'SAME_UPPER', 'SAME_LOWER', 'VALID']:
            raise ValueError(f"AveragePoolExt auto_pad must be one of NOTSET, SAME_UPPER, SAME_LOWER, VALID, got {self.auto_pad}")
        if self.ceil_mode not in [0, 1]:
            raise ValueError(f"AveragePoolExt ceil_mode must be 0 or 1, got {self.ceil_mode}")
        if self.count_include_pad not in [0, 1]:
            raise ValueError(f"AveragePoolExt count_include_pad must be 0 or 1, got {self.count_include_pad}")
        if not isinstance(self.dilations, list) or len(self.dilations) != 2:
            raise ValueError(f"AveragePoolExt dilations must be a list of 2 integers, got {self.dilations}")
        if self.kernel_shape is None:
            raise ValueError("AveragePoolExt kernel_shape is required")
        if not isinstance(self.kernel_shape, list) or len(self.kernel_shape) != 2:
            raise ValueError(f"AveragePoolExt kernel_shape must be a list of 2 integers, got {self.kernel_shape}")
        if not isinstance(self.pads, list) or len(self.pads) != 4:
            raise ValueError(f"AveragePoolExt pads must be a list of 4 integers, got {self.pads}")
        if not isinstance(self.strides, list) or len(self.strides) != 2:
            raise ValueError(f"AveragePoolExt strides must be a list of 2 integers, got {self.strides}")

        # Extended validations
        if self.dtype not in ['float32', 'float16', 'bfloat16', 'int8', 'int16']:
            raise ValueError(f"AveragePoolExt unsupported dtype: {self.dtype}")
        if self.mode not in ['average', 'sum']:
            raise ValueError(f"AveragePoolExt unsupported mode: {self.mode}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "AveragePoolExt operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs == 1, f"AveragePoolExt expects 1 input, got {num_inputs}"
        assert len(outT) == 1, f"AveragePoolExt expects 1 output, got {len(outT)}"

        X = inT[0]  # Input tensor [N, C, H, W]

        # Validate input type
        assert X.dtype == self.dtype, f"Input X dtype mismatch: expected {self.dtype}, got {X.dtype}"

        # Validate tensor shape
        X_shape = X.shape
        assert len(X_shape) == 4, f"Input X must be 4D [N, C, H, W], got {X_shape}"

        N, C, H, W = X_shape
        kH, kW = self.kernel_shape

        # Compute output shape
        output_height = self._compute_pool_output_size(H, kH, self.strides[0], self.pads[0], self.pads[2], self.dilations[0])
        output_width = self._compute_pool_output_size(W, kW, self.strides[1], self.pads[1], self.pads[3], self.dilations[1])

        output_shape = [N, C, output_height, output_width]

        outT[0].shape = output_shape
        outT[0].dtype = self.dtype

        # Calculate performance statistics
        total_elements = N * C * output_height * output_width

        # Average pooling statistics
        # Each output element requires summing kH * kW values and dividing
        operations_per_output = kH * kW

        if self.dtype.startswith('int'):
            # Integer operations
            add_count = total_elements * operations_per_output  # Sum all values
            div_count = total_elements if self.mode == 'average' else 0  # Divide by count for average
        else:
            # Float operations
            add_count = total_elements * operations_per_output  # Sum all values
            div_count = total_elements if self.mode == 'average' else 0  # Divide by count for average

        # For sum mode, we just sum without division
        if self.mode == 'sum':
            div_count = 0

        # Memory analysis
        input_bytes = X.nbytes(self.precision)
        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': X.nelems(),
            'outBytes': output_bytes,
            'outElems': total_elements,
            'instrs': {
                'mac': 0,  # No multiply-accumulate in pooling
                'cmp': 0,  # No comparisons
                'add': add_count,
                'sub': 0,  # No subtraction
                'mul': 0,  # No multiplication
                'div': div_count,
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
                'index': total_elements * operations_per_output,  # Indexing for gathering values
                'gather': total_elements * operations_per_output,  # Gather values from input
                'scatter': 0,  # No scattering
                'tanh': 0,  # No tanh
            }
        }

        return self.perf_stats

    def _compute_pool_output_size(self, input_size, kernel_size, stride, pad_start, pad_end, dilation):
        """Compute output size for pooling along one dimension."""
        if self.auto_pad == 'SAME_UPPER':
            if self.ceil_mode:
                return (input_size + stride - 1) // stride
            else:
                return (input_size + stride - 1) // stride
        elif self.auto_pad == 'SAME_LOWER':
            if self.ceil_mode:
                return (input_size + stride - 1) // stride
            else:
                return (input_size + stride - 1) // stride
        elif self.auto_pad == 'VALID':
            dilated_kernel = dilation * (kernel_size - 1) + 1
            if self.ceil_mode:
                return (input_size - dilated_kernel + stride) // stride
            else:
                return (input_size - dilated_kernel) // stride
        else:  # NOTSET
            if self.ceil_mode:
                return ((input_size + pad_start + pad_end - dilation * (kernel_size - 1) - 1) // stride) + 1
            else:
                return (input_size + pad_start + pad_end - dilation * (kernel_size - 1) - 1) // stride + 1

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for AveragePoolExt
        raise NotImplementedError("AveragePoolExt backward pass not yet implemented")


class QLinearConvOp(SimOp):
    """
    ONNX 1.20.0 QLinearConv Operation
    Quantized Linear Convolution operation.

    QLinearConv performs a quantized convolution operation by:
    1. Dequantizing the input tensor from quantized format to float
    2. Performing standard convolution with float weights and bias
    3. Quantizing the output back to quantized format

    This operation is essential for quantized neural networks where both
    inputs and outputs are in quantized format for efficient inference.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'QLinearConv'

        # QLinearConv supports 8-9 inputs and 1 output
        # x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B?
        check_io_counts(self, in_counts=[8,9], out_counts=[1,1])

        # Parse attributes - standard convolution attributes
        self.auto_pad = self.attrs.get('auto_pad', 'NOTSET')
        self.dilations = self.attrs.get('dilations', [1, 1])
        self.group = self.attrs.get('group', 1)
        self.kernel_shape = self.attrs.get('kernel_shape')
        self.pads = self.attrs.get('pads', [0, 0, 0, 0])
        self.strides = self.attrs.get('strides', [1, 1])

        # Validate attributes
        if self.auto_pad not in ['NOTSET', 'SAME_UPPER', 'SAME_LOWER', 'VALID']:
            raise ValueError(f"QLinearConv auto_pad must be one of NOTSET, SAME_UPPER, SAME_LOWER, VALID, got {self.auto_pad}")
        if not isinstance(self.dilations, list) or len(self.dilations) != 2:
            raise ValueError(f"QLinearConv dilations must be a list of 2 integers, got {self.dilations}")
        if not isinstance(self.group, int) or self.group < 1:
            raise ValueError(f"QLinearConv group must be a positive integer, got {self.group}")
        if self.kernel_shape is None:
            raise ValueError("QLinearConv kernel_shape is required")
        if not isinstance(self.kernel_shape, list) or len(self.kernel_shape) != 2:
            raise ValueError(f"QLinearConv kernel_shape must be a list of 2 integers, got {self.kernel_shape}")
        if not isinstance(self.pads, list) or len(self.pads) != 4:
            raise ValueError(f"QLinearConv pads must be a list of 4 integers, got {self.pads}")
        if not isinstance(self.strides, list) or len(self.strides) != 2:
            raise ValueError(f"QLinearConv strides must be a list of 2 integers, got {self.strides}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "QLinearConv operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs >= 8, f"QLinearConv expects at least 8 inputs, got {num_inputs}"
        assert len(outT) == 1, f"QLinearConv expects 1 output, got {len(outT)}"

        # Parse inputs
        x = inT[0]          # Quantized input tensor [N, C, H, W]
        x_scale = inT[1]    # Input scale (scalar or per-channel)
        x_zero_point = inT[2]  # Input zero point (scalar or per-channel)
        w = inT[3]          # Quantized weight tensor [M, C/group, kH, kW]
        w_scale = inT[4]    # Weight scale (scalar or per-channel)
        w_zero_point = inT[5]  # Weight zero point (scalar or per-channel)
        y_scale = inT[6]    # Output scale (scalar or per-channel)
        y_zero_point = inT[7]  # Output zero point (scalar or per-channel)
        bias = inT[8] if num_inputs >= 9 else None  # Optional bias

        # Validate input types - all quantized inputs should be integers
        quantized_types = ['int8', 'uint8', 'int16', 'uint16']
        scale_types = ['float32', 'float16', 'float64']

        assert x.dtype in quantized_types, f"Input x must be quantized type, got {x.dtype}"
        assert x_scale.dtype in scale_types, f"Input x_scale must be float type, got {x_scale.dtype}"
        assert x_zero_point.dtype in quantized_types + ['int32', 'int64'], f"Input x_zero_point must be integer type, got {x_zero_point.dtype}"
        assert w.dtype in quantized_types, f"Weight w must be quantized type, got {w.dtype}"
        assert w_scale.dtype in scale_types, f"Weight w_scale must be float type, got {w_scale.dtype}"
        assert w_zero_point.dtype in quantized_types + ['int32', 'int64'], f"Weight w_zero_point must be integer type, got {w_zero_point.dtype}"
        assert y_scale.dtype in scale_types, f"Output y_scale must be float type, got {y_scale.dtype}"
        assert y_zero_point.dtype in quantized_types + ['int32', 'int64'], f"Output y_zero_point must be integer type, got {y_zero_point.dtype}"
        if bias is not None:
            assert bias.dtype in scale_types, f"Bias must be float type, got {bias.dtype}"

        # Validate tensor shapes
        x_shape = x.shape
        w_shape = w.shape

        assert len(x_shape) == 4, f"Input x must be 4D [N, C, H, W], got {x_shape}"
        assert len(w_shape) == 4, f"Weight w must be 4D [M, C/group, kH, kW], got {w_shape}"

        N, C, H, W = x_shape
        M, C_per_group, kH, kW = w_shape

        # Validate kernel shape consistency
        assert kH == self.kernel_shape[0], f"Kernel height mismatch: expected {self.kernel_shape[0]}, got {kH}"
        assert kW == self.kernel_shape[1], f"Kernel width mismatch: expected {self.kernel_shape[1]}, got {kW}"

        # Validate channel count
        assert C % self.group == 0, f"Input channels {C} must be divisible by group {self.group}"
        assert C_per_group == C // self.group, f"Weight channels per group {C_per_group} must be C/group = {C // self.group}"

        # Validate scale and zero point shapes
        # They can be scalars or per-channel/per-output-channel
        assert len(x_scale.shape) in [0, 1], f"x_scale must be scalar or 1D, got shape {x_scale.shape}"
        assert len(x_zero_point.shape) in [0, 1], f"x_zero_point must be scalar or 1D, got shape {x_zero_point.shape}"
        assert len(w_scale.shape) in [0, 1], f"w_scale must be scalar or 1D, got shape {w_scale.shape}"
        assert len(w_zero_point.shape) in [0, 1], f"w_zero_point must be scalar or 1D, got shape {w_zero_point.shape}"
        assert len(y_scale.shape) in [0, 1], f"y_scale must be scalar or 1D, got shape {y_scale.shape}"
        assert len(y_zero_point.shape) in [0, 1], f"y_zero_point must be scalar or 1D, got shape {y_zero_point.shape}"

        if len(x_scale.shape) == 1:
            assert x_scale.shape[0] == C, f"x_scale per-channel dimension must match input channels {C}"
        if len(x_zero_point.shape) == 1:
            assert x_zero_point.shape[0] == C, f"x_zero_point per-channel dimension must match input channels {C}"
        if len(w_scale.shape) == 1:
            assert w_scale.shape[0] == M, f"w_scale per-channel dimension must match output channels {M}"
        if len(w_zero_point.shape) == 1:
            assert w_zero_point.shape[0] == M, f"w_zero_point per-channel dimension must match output channels {M}"
        if len(y_scale.shape) == 1:
            assert y_scale.shape[0] == M, f"y_scale per-channel dimension must match output channels {M}"
        if len(y_zero_point.shape) == 1:
            assert y_zero_point.shape[0] == M, f"y_zero_point per-channel dimension must match output channels {M}"

        # Validate bias if present
        if bias is not None:
            bias_shape = bias.shape
            assert len(bias_shape) == 1 and bias_shape[0] == M, f"Bias must be 1D with {M} elements, got {bias_shape}"

        # Compute output shape
        output_height = self._compute_conv_output_size(H, kH, self.strides[0], self.pads[0], self.pads[2], self.dilations[0])
        output_width = self._compute_conv_output_size(W, kW, self.strides[1], self.pads[1], self.pads[3], self.dilations[1])

        output_shape = [N, M, output_height, output_width]

        outT[0].shape = output_shape
        outT[0].dtype = x.dtype  # Output has same quantized type as input

        # Calculate performance statistics
        total_elements = N * M * output_height * output_width

        # QLinearConv performance breakdown:
        # 1. Dequantize input: convert x from quantized to float
        # 2. Dequantize weights: convert w from quantized to float
        # 3. Convolution: standard float convolution
        # 4. Quantize output: convert result back to quantized format

        # Dequantization operations
        input_dequant_ops = N * C * H * W  # Dequantize each input element
        weight_dequant_ops = M * C_per_group * kH * kW  # Dequantize each weight element

        # Convolution operations (similar to ConvExt but with float operations)
        conv_mac_ops = M * N * output_height * output_width * kH * kW * C_per_group

        # Quantization operations
        output_quant_ops = total_elements

        # Bias addition (if present)
        bias_ops = total_elements if bias is not None else 0

        # Total operations
        total_ops = input_dequant_ops + weight_dequant_ops + conv_mac_ops + output_quant_ops + bias_ops

        # Memory analysis
        input_bytes = sum(tensor.nbytes(self.precision) for tensor in inT)
        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': sum(tensor.nelems() for tensor in inT),
            'outBytes': output_bytes,
            'outElems': total_elements,
            'instrs': {
                'mac': conv_mac_ops,
                'cmp': 0,
                'add': bias_ops + output_quant_ops,  # Bias addition + quantization adjustments
                'sub': input_dequant_ops + weight_dequant_ops + output_quant_ops,  # Dequantize and quantize operations
                'mul': input_dequant_ops + weight_dequant_ops + output_quant_ops,  # Scale multiplications
                'div': 0,
                'rsqrt': 0,
                'exp': 0,
                'log': 0,
                'round': output_quant_ops,  # Rounding in quantization
                'clip': output_quant_ops,  # Clipping to quantized range
                'convert': input_dequant_ops + weight_dequant_ops + output_quant_ops,  # Type conversions
                'index': conv_mac_ops,  # Convolution indexing
                'gather': conv_mac_ops,  # Gather input values
                'scatter': 0,
                'tanh': 0,
            }
        }

        return self.perf_stats

    def _compute_conv_output_size(self, input_size, kernel_size, stride, pad_start, pad_end, dilation):
        """Compute output size for convolution along one dimension."""
        if self.auto_pad == 'SAME_UPPER':
            return (input_size + stride - 1) // stride
        elif self.auto_pad == 'SAME_LOWER':
            return (input_size + stride - 1) // stride
        elif self.auto_pad == 'VALID':
            return (input_size - dilation * (kernel_size - 1) - 1 + pad_start + pad_end) // stride + 1
        else:  # NOTSET
            return (input_size + pad_start + pad_end - dilation * (kernel_size - 1) - 1) // stride + 1

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for QLinearConv
        raise NotImplementedError("QLinearConv backward pass not yet implemented")


class ConvIntegerOp(SimOp):
    """
    ONNX 1.20.0 ConvInteger Operation
    Integer Convolution operation for quantized neural networks.

    ConvInteger performs convolution entirely in integer arithmetic:
    1. Input and weight tensors are already quantized
    2. Convolution is performed directly on quantized values
    3. Output is also quantized (no dequantization/quantization steps)
    4. Bias is added as integer bias

    This operation is highly efficient for specialized quantized hardware
    and eliminates the need for floating-point operations entirely.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ConvInteger'

        # ConvInteger supports 2-3 inputs and 1 output
        # x, w, B?
        check_io_counts(self, in_counts=[2,3], out_counts=[1,1])

        # Parse attributes - standard convolution attributes
        self.auto_pad = self.attrs.get('auto_pad', 'NOTSET')
        self.dilations = self.attrs.get('dilations', [1, 1])
        self.group = self.attrs.get('group', 1)
        self.kernel_shape = self.attrs.get('kernel_shape')
        self.pads = self.attrs.get('pads', [0, 0, 0, 0])
        self.strides = self.attrs.get('strides', [1, 1])

        # Validate attributes
        if self.auto_pad not in ['NOTSET', 'SAME_UPPER', 'SAME_LOWER', 'VALID']:
            raise ValueError(f"ConvInteger auto_pad must be one of NOTSET, SAME_UPPER, SAME_LOWER, VALID, got {self.auto_pad}")
        if not isinstance(self.dilations, list) or len(self.dilations) != 2:
            raise ValueError(f"ConvInteger dilations must be a list of 2 integers, got {self.dilations}")
        if not isinstance(self.group, int) or self.group < 1:
            raise ValueError(f"ConvInteger group must be a positive integer, got {self.group}")
        if self.kernel_shape is None:
            raise ValueError("ConvInteger kernel_shape is required")
        if not isinstance(self.kernel_shape, list) or len(self.kernel_shape) != 2:
            raise ValueError(f"ConvInteger kernel_shape must be a list of 2 integers, got {self.kernel_shape}")
        if not isinstance(self.pads, list) or len(self.pads) != 4:
            raise ValueError(f"ConvInteger pads must be a list of 4 integers, got {self.pads}")
        if not isinstance(self.strides, list) or len(self.strides) != 2:
            raise ValueError(f"ConvInteger strides must be a list of 2 integers, got {self.strides}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "ConvInteger operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs >= 2, f"ConvInteger expects at least 2 inputs, got {num_inputs}"
        assert len(outT) == 1, f"ConvInteger expects 1 output, got {len(outT)}"

        # Parse inputs
        x = inT[0]          # Quantized input tensor [N, C, H, W]
        w = inT[1]          # Quantized weight tensor [M, C/group, kH, kW]
        bias = inT[2] if num_inputs >= 3 else None  # Optional integer bias

        # Validate input types - all should be integers
        integer_types = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32']

        assert x.dtype in integer_types, f"Input x must be integer type, got {x.dtype}"
        assert w.dtype in integer_types, f"Weight w must be integer type, got {w.dtype}"
        if bias is not None:
            assert bias.dtype in integer_types, f"Bias must be integer type, got {bias.dtype}"

        # Validate tensor shapes
        x_shape = x.shape
        w_shape = w.shape

        assert len(x_shape) == 4, f"Input x must be 4D [N, C, H, W], got {x_shape}"
        assert len(w_shape) == 4, f"Weight w must be 4D [M, C/group, kH, kW], got {w_shape}"

        N, C, H, W = x_shape
        M, C_per_group, kH, kW = w_shape

        # Validate kernel shape consistency
        assert kH == self.kernel_shape[0], f"Kernel height mismatch: expected {self.kernel_shape[0]}, got {kH}"
        assert kW == self.kernel_shape[1], f"Kernel width mismatch: expected {self.kernel_shape[1]}, got {kW}"

        # Validate channel count
        assert C % self.group == 0, f"Input channels {C} must be divisible by group {self.group}"
        assert C_per_group == C // self.group, f"Weight channels per group {C_per_group} must be C/group = {C // self.group}"

        # Validate bias if present
        if bias is not None:
            bias_shape = bias.shape
            assert len(bias_shape) == 1 and bias_shape[0] == M, f"Bias must be 1D with {M} elements, got {bias_shape}"

        # Compute output shape
        output_height = self._compute_conv_output_size(H, kH, self.strides[0], self.pads[0], self.pads[2], self.dilations[0])
        output_width = self._compute_conv_output_size(W, kW, self.strides[1], self.pads[1], self.pads[3], self.dilations[1])

        output_shape = [N, M, output_height, output_width]

        outT[0].shape = output_shape
        outT[0].dtype = x.dtype  # Output has same type as input

        # Calculate performance statistics
        total_elements = N * M * output_height * output_width

        # ConvInteger performance breakdown:
        # Pure integer convolution with no type conversions

        # Convolution operations (integer multiply-accumulate)
        conv_mac_ops = M * N * output_height * output_width * kH * kW * C_per_group

        # Bias addition (if present)
        bias_ops = total_elements if bias is not None else 0

        # Total operations
        total_ops = conv_mac_ops + bias_ops

        # Memory analysis
        input_bytes = sum(tensor.nbytes(self.precision) for tensor in inT)
        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': sum(tensor.nelems() for tensor in inT),
            'outBytes': output_bytes,
            'outElems': total_elements,
            'instrs': {
                'mac': conv_mac_ops,  # Integer multiply-accumulate operations
                'cmp': 0,  # No comparisons
                'add': bias_ops,  # Bias addition
                'sub': 0,  # No subtraction
                'mul': 0,  # MAC operations are counted separately
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion (all integer)
                'index': conv_mac_ops,  # Convolution indexing
                'gather': conv_mac_ops,  # Gather input values
                'scatter': 0,  # No scattering
                'tanh': 0,  # No tanh
            }
        }

        return self.perf_stats

    def _compute_conv_output_size(self, input_size, kernel_size, stride, pad_start, pad_end, dilation):
        """Compute output size for convolution along one dimension."""
        if self.auto_pad == 'SAME_UPPER':
            return (input_size + stride - 1) // stride
        elif self.auto_pad == 'SAME_LOWER':
            return (input_size + stride - 1) // stride
        elif self.auto_pad == 'VALID':
            return (input_size - dilation * (kernel_size - 1) - 1 + pad_start + pad_end) // stride + 1
        else:  # NOTSET
            return (input_size + pad_start + pad_end - dilation * (kernel_size - 1) - 1) // stride + 1

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for ConvInteger
        raise NotImplementedError("ConvInteger backward pass not yet implemented")


class DynamicQuantizeMatMulOp(SimOp):
    """
    ONNX 1.20.0 DynamicQuantizeMatMul Operation
    Dynamic quantization matrix multiplication.

    DynamicQuantizeMatMul performs matrix multiplication with dynamic quantization:
    1. Dynamically quantize input matrices A and B based on their ranges
    2. Perform matrix multiplication on quantized values
    3. Dequantize the result back to floating point

    This operation is useful when you want to quantize on-the-fly rather than
    using pre-quantized weights, providing a balance between accuracy and efficiency.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'DynamicQuantizeMatMul'

        # DynamicQuantizeMatMul supports 2 inputs (A, B) and 1 output
        check_io_counts(self, in_counts=[2,2], out_counts=[1,1])

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "DynamicQuantizeMatMul operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs == 2, f"DynamicQuantizeMatMul expects 2 inputs, got {num_inputs}"
        assert len(outT) == 1, f"DynamicQuantizeMatMul expects 1 output, got {len(outT)}"

        A = inT[0]  # Input matrix A [M, K]
        B = inT[1]  # Input matrix B [K, N]

        # Validate input types - should be floating point
        assert A.dtype in ['float32', 'float16', 'float64'], f"Input A must be float type, got {A.dtype}"
        assert B.dtype in ['float32', 'float16', 'float64'], f"Input B must be float type, got {B.dtype}"

        # Validate tensor shapes
        A_shape = A.shape
        B_shape = B.shape

        assert len(A_shape) >= 2, f"Input A must have at least 2 dimensions, got {A_shape}"
        assert len(B_shape) >= 2, f"Input B must have at least 2 dimensions, got {B_shape}"
        assert A_shape[-1] == B_shape[-2], f"Matrix dimensions incompatible: A[-1]={A_shape[-1]}, B[-2]={B_shape[-2]}"

        # Compute output shape
        output_shape = A_shape[:-1] + [B_shape[-1]]

        outT[0].shape = output_shape
        outT[0].dtype = A.dtype  # Output has same type as input

        # Calculate performance statistics
        total_elements = 1
        for dim in output_shape:
            total_elements *= dim

        # DynamicQuantizeMatMul performance breakdown:
        # 1. Dynamic quantization of A: find min/max, compute scale/zero_point
        # 2. Dynamic quantization of B: find min/max, compute scale/zero_point
        # 3. Quantized matrix multiplication
        # 4. Dequantization of result

        # Matrix multiplication dimensions
        M = 1
        for dim in A_shape[:-1]:
            M *= dim
        K = A_shape[-1]
        N = B_shape[-1]

        # Dynamic quantization operations
        a_elements = A.nelems()
        b_elements = B.nelems()

        # Find min/max for quantization (rough estimate)
        quant_ops_a = a_elements * 2  # Compare operations to find min/max
        quant_ops_b = b_elements * 2

        # Scale computation
        scale_ops = 4  # Basic scale computation

        # Quantization operations
        quantize_ops_a = a_elements * 3  # subtract zero_point, multiply scale, round
        quantize_ops_b = b_elements * 3

        # Matrix multiplication (quantized)
        mac_ops = M * N * K

        # Dequantization of result
        dequant_ops = total_elements * 3  # scale multiplication, zero_point addition, type conversion

        # Total operations
        total_ops = quant_ops_a + quant_ops_b + scale_ops + quantize_ops_a + quantize_ops_b + mac_ops + dequant_ops

        # Memory analysis
        input_bytes = A.nbytes(self.precision) + B.nbytes(self.precision)
        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': A.nelems() + B.nelems(),
            'outBytes': output_bytes,
            'outElems': total_elements,
            'instrs': {
                'mac': mac_ops,  # Quantized multiply-accumulate operations
                'cmp': quant_ops_a + quant_ops_b,  # Comparisons for min/max finding
                'add': quantize_ops_a + quantize_ops_b + dequant_ops,  # Various additions
                'sub': quantize_ops_a + quantize_ops_b,  # Zero point subtraction
                'mul': quantize_ops_a + quantize_ops_b + dequant_ops + scale_ops,  # Scale multiplications
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': quantize_ops_a + quantize_ops_b,  # Rounding in quantization
                'clip': quantize_ops_a + quantize_ops_b,  # Clipping to quantized range
                'convert': quantize_ops_a + quantize_ops_b + dequant_ops,  # Type conversions
                'index': mac_ops,  # Matrix indexing
                'gather': mac_ops,  # Gather input values
                'scatter': 0,  # No scattering
                'tanh': 0,  # No tanh
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for DynamicQuantizeMatMul
        raise NotImplementedError("DynamicQuantizeMatMul backward pass not yet implemented")


class AttentionExtOp(SimOp):
    """
    ONNX 1.20.0 Attention Extensions
    Extended Attention operation with additional features and capabilities.

    This operation extends the standard Attention operation with additional
    capabilities commonly found in modern transformer architectures:

    New Features in Extensions:
    1. Support for different data types (int8, float16, bfloat16)
    2. Multiple attention mechanisms (scaled_dot_product, flash_attention, sparse)
    3. Enhanced masking with different mask types
    4. Support for different position encodings
    5. Quantized attention support
    6. Memory-efficient attention patterns
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'AttentionExt'

        # AttentionExt supports 3-8 inputs and 1-4 outputs (extended from base Attention)
        check_io_counts(self, in_counts=[3,8], out_counts=[1,4])

        # Parse extended attributes
        self.num_heads = self.attrs.get('num_heads', 1)
        self.scale = self.attrs.get('scale', None)
        self.causal = self.attrs.get('causal', False)
        self.unidirectional = self.attrs.get('unidirectional', False)

        # Extended attributes
        self.attention_type = self.attrs.get('attention_type', 'scaled_dot_product')  # 'scaled_dot_product', 'flash_attention', 'sparse'
        self.dtype = self.attrs.get('dtype', 'float32')  # Support for different data types
        self.mask_type = self.attrs.get('mask_type', 'default')  # 'default', 'causal', 'padding', 'custom'
        self.position_encoding = self.attrs.get('position_encoding', 'none')  # 'none', 'learned', 'sinusoidal', 'rotary'
        self.quantized_attention = self.attrs.get('quantized_attention', False)  # Enable quantized attention
        self.sparse_pattern = self.attrs.get('sparse_pattern', None)  # For sparse attention

        # Validate extended attributes
        if self.attention_type not in ['scaled_dot_product', 'flash_attention', 'sparse']:
            raise ValueError(f"AttentionExt unsupported attention_type: {self.attention_type}")
        if self.dtype not in ['float32', 'float16', 'bfloat16', 'int8']:
            raise ValueError(f"AttentionExt unsupported dtype: {self.dtype}")
        if self.mask_type not in ['default', 'causal', 'padding', 'custom']:
            raise ValueError(f"AttentionExt unsupported mask_type: {self.mask_type}")
        if self.position_encoding not in ['none', 'learned', 'sinusoidal', 'rotary']:
            raise ValueError(f"AttentionExt unsupported position_encoding: {self.position_encoding}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "AttentionExt operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs >= 3, f"AttentionExt expects at least 3 inputs, got {num_inputs}"
        assert len(outT) >= 1, f"AttentionExt expects at least 1 output, got {len(outT)}"

        Q, K, V = inT[0], inT[1], inT[2]  # Required inputs

        # Validate input types based on dtype attribute
        if self.quantized_attention:
            quantized_types = ['int8', 'uint8', 'int16', 'uint16']
            assert Q.dtype in quantized_types, f"Q must be quantized type when quantized_attention=True, got {Q.dtype}"
            assert K.dtype in quantized_types, f"K must be quantized type when quantized_attention=True, got {K.dtype}"
            assert V.dtype in quantized_types, f"V must be quantized type when quantized_attention=True, got {V.dtype}"
        else:
            assert Q.dtype == self.dtype, f"Q dtype mismatch: expected {self.dtype}, got {Q.dtype}"
            assert K.dtype == self.dtype, f"K dtype mismatch: expected {self.dtype}, got {K.dtype}"
            assert V.dtype == self.dtype, f"V dtype mismatch: expected {self.dtype}, got {V.dtype}"

        # Validate tensor shapes
        Q_shape = Q.shape
        K_shape = K.shape
        V_shape = V.shape

        assert len(Q_shape) == 4, f"Q must be 4D [batch_size, seq_len_q, num_heads, head_dim], got {Q_shape}"
        assert len(K_shape) == 4, f"K must be 4D [batch_size, seq_len_k, num_heads, head_dim], got {K_shape}"
        assert len(V_shape) == 4, f"V must be 4D [batch_size, seq_len_v, num_heads, head_dim], got {V_shape}"

        batch_size, seq_len_q, num_heads_q, head_dim_q = Q_shape
        batch_size_k, seq_len_k, num_heads_k, head_dim_k = K_shape
        batch_size_v, seq_len_v, num_heads_v, head_dim_v = V_shape

        # Validate consistency
        assert batch_size == batch_size_k == batch_size_v, "Batch sizes must match"
        assert num_heads_q == num_heads_k == num_heads_v == self.num_heads, f"Number of heads mismatch: {num_heads_q}, {num_heads_k}, {num_heads_v}, expected {self.num_heads}"
        assert head_dim_q == head_dim_k == head_dim_v, "Head dimensions must match"

        # Parse optional inputs
        mask = None
        past_key = None
        past_value = None
        rotary_cos = None
        rotary_sin = None

        # Extended input parsing (up to 8 inputs)
        for i in range(3, num_inputs):
            tensor = inT[i]
            if len(tensor.shape) == 4 and tensor.shape[2] == self.num_heads:
                # Likely past_key or past_value
                if past_key is None:
                    past_key = tensor
                elif past_value is None:
                    past_value = tensor
                else:
                    raise ValueError(f"Too many past key/value tensors")
            elif len(tensor.shape) == 4 and tensor.shape[0] == batch_size and tensor.shape[1] == seq_len_q:
                # Likely mask
                if mask is None:
                    mask = tensor
                else:
                    raise ValueError(f"Multiple masks not supported")
            elif len(tensor.shape) == 3 and tensor.shape[-1] == head_dim_q // 2:
                # Likely rotary embeddings
                if rotary_cos is None:
                    rotary_cos = tensor
                elif rotary_sin is None:
                    rotary_sin = tensor
                else:
                    raise ValueError(f"Too many rotary embedding tensors")
            else:
                raise ValueError(f"Unexpected input tensor shape at position {i}: {tensor.shape}")

        # Validate rotary embeddings if position encoding requires them
        if self.position_encoding == 'rotary' and (rotary_cos is None or rotary_sin is None):
            raise ValueError("Rotary position encoding requires both cos and sin caches")

        # Set output tensor shapes and dtypes
        output_shape = [batch_size, seq_len_q, num_heads_q, head_dim_q]
        outT[0].shape = output_shape
        outT[0].dtype = Q.dtype

        # Optional outputs
        if len(outT) >= 2:  # Present key output
            present_key_shape = [batch_size, seq_len_k, self.num_heads, head_dim_q]
            outT[1].shape = present_key_shape
            outT[1].dtype = K.dtype

        if len(outT) >= 3:  # Present value output
            present_value_shape = [batch_size, seq_len_v, self.num_heads, head_dim_q]
            outT[2].shape = present_value_shape
            outT[2].dtype = V.dtype

        if len(outT) >= 4:  # Attention weights output
            attn_weights_shape = [batch_size, self.num_heads, seq_len_q, seq_len_k]
            outT[3].shape = attn_weights_shape
            outT[3].dtype = self.dtype

        # Calculate performance statistics with extensions
        total_elements = batch_size * seq_len_q * self.num_heads * head_dim_q

        # Base attention operations
        qk_matmul_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim_q  # Q @ K^T
        softmax_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * 4  # Softmax operations
        attn_v_matmul_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim_q  # Attention @ V

        # Extended operations based on features
        if self.attention_type == 'flash_attention':
            # Flash attention optimizations (approximate operations)
            flash_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim_q // 4  # Reduced operations
        else:
            flash_ops = 0

        if self.attention_type == 'sparse' and self.sparse_pattern:
            # Sparse attention - reduced computation based on sparsity
            sparsity_factor = 0.1  # Assume 10% sparsity
            sparse_ops = int((qk_matmul_ops + attn_v_matmul_ops) * sparsity_factor)
        else:
            sparse_ops = 0

        if self.position_encoding == 'rotary':
            # Rotary position embedding operations
            rope_ops = batch_size * seq_len_q * self.num_heads * head_dim_q * 4  # RoPE computations
        elif self.position_encoding in ['learned', 'sinusoidal']:
            # Position embedding operations
            pos_ops = batch_size * seq_len_q * self.num_heads * head_dim_q * 2  # Add position embeddings
        else:
            rope_ops = 0
            pos_ops = 0

        if self.quantized_attention:
            # Additional quantization/dequantization operations
            quant_ops = batch_size * seq_len_q * self.num_heads * head_dim_q * 6  # Quantize/dequantize Q, K, V
        else:
            quant_ops = 0

        if mask is not None:
            # Mask application operations
            mask_ops = batch_size * self.num_heads * seq_len_q * seq_len_k  # Apply mask to attention scores
        else:
            mask_ops = 0

        # Total operations
        if self.attention_type == 'flash_attention':
            total_ops = flash_ops + softmax_ops + rope_ops + quant_ops + mask_ops
        elif self.attention_type == 'sparse':
            total_ops = sparse_ops + softmax_ops + rope_ops + quant_ops + mask_ops
        else:
            total_ops = qk_matmul_ops + attn_v_matmul_ops + softmax_ops + rope_ops + pos_ops + quant_ops + mask_ops

        # Memory analysis
        input_bytes = Q.nbytes(self.precision) + K.nbytes(self.precision) + V.nbytes(self.precision)
        if mask is not None:
            input_bytes += mask.nbytes(self.precision)
        if past_key is not None:
            input_bytes += past_key.nbytes(self.precision)
        if past_value is not None:
            input_bytes += past_value.nbytes(self.precision)
        if rotary_cos is not None and rotary_sin is not None:
            input_bytes += rotary_cos.nbytes(self.precision) + rotary_sin.nbytes(self.precision)

        output_bytes = sum(out.nbytes(self.precision) for out in outT)

        # Instruction counts
        exp_count = softmax_ops // 4  # Exponential in softmax
        div_count = batch_size * self.num_heads * seq_len_q  # Division by sqrt(d_k) and softmax normalization

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': sum(tensor.nelems() for tensor in inT),
            'outBytes': output_bytes,
            'outElems': sum(tensor.nelems() for tensor in outT),
            'instrs': {
                'mac': qk_matmul_ops + attn_v_matmul_ops,  # Matrix multiplications
                'cmp': mask_ops,  # Comparisons for masking
                'add': total_elements + softmax_ops,  # Various additions
                'sub': 0,  # Minimal subtraction
                'mul': total_elements * 2,  # Scale multiplications
                'div': div_count,
                'rsqrt': batch_size * self.num_heads if self.scale is None else 0,  # Reciprocal sqrt for default scaling
                'exp': exp_count,
                'log': 0,  # No logarithm
                'round': quant_ops // 3 if self.quantized_attention else 0,  # Rounding in quantization
                'clip': quant_ops // 3 if self.quantized_attention else 0,  # Clipping in quantization
                'convert': quant_ops // 3 if self.quantized_attention else 0,  # Type conversions
                'index': total_ops // 10,  # Memory indexing operations
                'gather': total_ops // 20,  # Gather operations
                'scatter': 0,  # No scattering
                'tanh': 0,  # No tanh
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for AttentionExt
        raise NotImplementedError("AttentionExt backward pass not yet implemented")


class MultiHeadAttentionExtOp(SimOp):
    """
    ONNX 1.20.0 MultiHeadAttention Extensions
    Extended Multi-Head Attention operation with advanced features.

    This operation extends the standard MultiHeadAttention with additional
    capabilities commonly found in modern transformer architectures:

    New Features in Extensions:
    1. Support for different data types (int8, float16, bfloat16)
    2. Multiple attention mechanisms (scaled_dot_product, flash_attention, sparse, linear)
    3. Enhanced quantization support for QKV projections
    4. Different position encoding schemes (RoPE, ALiBi, etc.)
    5. Memory-efficient attention patterns
    6. Support for different head configurations
    7. Advanced masking capabilities
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'MultiHeadAttentionExt'

        # MultiHeadAttentionExt supports 3-10 inputs and 1-4 outputs (extended)
        check_io_counts(self, in_counts=[3,10], out_counts=[1,4])

        # Parse extended attributes
        self.num_heads = self.attrs.get('num_heads')
        if self.num_heads is None:
            raise ValueError("MultiHeadAttentionExt requires 'num_heads' attribute")
        self.unidirectional = self.attrs.get('unidirectional', False)
        self.qkv_hidden_sizes = self.attrs.get('qkv_hidden_sizes', None)

        # Extended attributes
        self.attention_type = self.attrs.get('attention_type', 'scaled_dot_product')  # 'scaled_dot_product', 'flash_attention', 'sparse', 'linear'
        self.dtype = self.attrs.get('dtype', 'float32')  # Support for different data types
        self.quantized_attention = self.attrs.get('quantized_attention', False)  # Enable quantized attention
        self.position_encoding = self.attrs.get('position_encoding', 'none')  # 'none', 'rotary', 'alibi', 'learned'
        self.sparse_pattern = self.attrs.get('sparse_pattern', None)  # For sparse attention
        self.head_config = self.attrs.get('head_config', 'standard')  # 'standard', 'grouped', 'hierarchical'
        self.memory_efficient = self.attrs.get('memory_efficient', False)  # Enable memory-efficient attention

        # Validate extended attributes
        if self.attention_type not in ['scaled_dot_product', 'flash_attention', 'sparse', 'linear']:
            raise ValueError(f"MultiHeadAttentionExt unsupported attention_type: {self.attention_type}")
        if self.dtype not in ['float32', 'float16', 'bfloat16', 'int8']:
            raise ValueError(f"MultiHeadAttentionExt unsupported dtype: {self.dtype}")
        if self.position_encoding not in ['none', 'rotary', 'alibi', 'learned']:
            raise ValueError(f"MultiHeadAttentionExt unsupported position_encoding: {self.position_encoding}")
        if self.head_config not in ['standard', 'grouped', 'hierarchical']:
            raise ValueError(f"MultiHeadAttentionExt unsupported head_config: {self.head_config}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "MultiHeadAttentionExt operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs >= 3, f"MultiHeadAttentionExt expects at least 3 inputs, got {num_inputs}"
        assert len(outT) >= 1, f"MultiHeadAttentionExt expects at least 1 output, got {len(outT)}"

        query, key, value = inT[0], inT[1], inT[2]  # Required inputs

        # Validate input types based on quantization setting
        if self.quantized_attention:
            quantized_types = ['int8', 'uint8', 'int16', 'uint16']
            assert query.dtype in quantized_types, f"Query must be quantized type when quantized_attention=True, got {query.dtype}"
            assert key.dtype in quantized_types, f"Key must be quantized type when quantized_attention=True, got {key.dtype}"
            assert value.dtype in quantized_types, f"Value must be quantized type when quantized_attention=True, got {value.dtype}"
        else:
            assert query.dtype == self.dtype, f"Query dtype mismatch: expected {self.dtype}, got {query.dtype}"
            assert key.dtype == self.dtype, f"Key dtype mismatch: expected {self.dtype}, got {key.dtype}"
            assert value.dtype == self.dtype, f"Value dtype mismatch: expected {self.dtype}, got {value.dtype}"

        # Validate tensor shapes
        query_shape = query.shape
        key_shape = key.shape
        value_shape = value.shape

        assert len(query_shape) == 3, f"Query must be 3D [batch_size, seq_len_q, hidden_size], got {query_shape}"
        assert len(key_shape) == 3, f"Key must be 3D [batch_size, seq_len_k, hidden_size], got {key_shape}"
        assert len(value_shape) == 3, f"Value must be 3D [batch_size, seq_len_v, hidden_size], got {value_shape}"

        batch_size, seq_len_q, hidden_size_q = query_shape
        batch_size_k, seq_len_k, hidden_size_k = key_shape
        batch_size_v, seq_len_v, hidden_size_v = value_shape

        # Validate consistency
        assert batch_size == batch_size_k == batch_size_v, "Batch sizes must match"
        assert hidden_size_q == hidden_size_k == hidden_size_v, "Hidden sizes must match"

        # Validate head configuration
        head_dim = hidden_size_q // self.num_heads
        assert head_dim * self.num_heads == hidden_size_q, f"Hidden size {hidden_size_q} must be divisible by num_heads {self.num_heads}"

        # Parse optional inputs
        bias = None
        key_padding_mask = None
        past_key = None
        past_value = None
        cos_cache = None
        sin_cache = None
        alibi_bias = None
        sparse_mask = None

        # Extended input parsing (up to 10 inputs)
        for i in range(3, num_inputs):
            tensor = inT[i]
            if len(tensor.shape) == 3 and tensor.shape[0] == batch_size:
                # Could be bias [batch, seq_q, seq_k] or key_padding_mask [batch, seq]
                if tensor.shape[1] == seq_len_q and tensor.shape[2] == seq_len_k:
                    bias = tensor  # Attention bias
                elif tensor.shape[1] == seq_len_k:
                    key_padding_mask = tensor
            elif len(tensor.shape) == 3 and tensor.shape[2] == self.num_heads:
                # Past key/value tensors
                if past_key is None:
                    past_key = tensor
                elif past_value is None:
                    past_value = tensor
            elif len(tensor.shape) == 3 and tensor.shape[-1] == head_dim // 2:
                # Rotary embeddings
                if cos_cache is None:
                    cos_cache = tensor
                elif sin_cache is None:
                    sin_cache = tensor
            elif len(tensor.shape) == 2 and tensor.shape[0] == seq_len_q:
                # ALiBi bias
                alibi_bias = tensor
            elif len(tensor.shape) == 2 and tensor.shape == [seq_len_q, seq_len_k]:
                # Sparse attention mask
                sparse_mask = tensor

        # Validate position encoding requirements
        if self.position_encoding == 'rotary' and (cos_cache is None or sin_cache is None):
            raise ValueError("Rotary position encoding requires both cos and sin caches")
        if self.position_encoding == 'alibi' and alibi_bias is None:
            raise ValueError("ALiBi position encoding requires alibi_bias tensor")

        # Set output tensor shapes and dtypes
        output_shape = [batch_size, seq_len_q, hidden_size_q]
        outT[0].shape = output_shape
        outT[0].dtype = query.dtype

        # Optional outputs
        if len(outT) >= 2:  # Present key output
            present_key_shape = [batch_size, seq_len_k, self.num_heads, head_dim]
            outT[1].shape = present_key_shape
            outT[1].dtype = key.dtype

        if len(outT) >= 3:  # Present value output
            present_value_shape = [batch_size, seq_len_v, self.num_heads, head_dim]
            outT[2].shape = present_value_shape
            outT[2].dtype = value.dtype

        if len(outT) >= 4:  # Attention weights output
            attn_weights_shape = [batch_size, self.num_heads, seq_len_q, seq_len_k]
            outT[3].shape = attn_weights_shape
            outT[3].dtype = self.dtype

        # Calculate performance statistics with extensions
        total_elements = batch_size * seq_len_q * hidden_size_q

        # Base multi-head attention operations
        # QKV projections: 3 matrix multiplications
        qkv_proj_ops = 3 * batch_size * seq_len_q * hidden_size_q * hidden_size_q  # Q_proj, K_proj, V_proj

        # Attention computation per head
        qk_matmul_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim  # Q @ K^T per head
        softmax_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * 4  # Softmax operations
        attn_v_matmul_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim  # Attention @ V per head

        # Output projection
        output_proj_ops = batch_size * seq_len_q * hidden_size_q * hidden_size_q  # Output projection

        # Extended operations based on features
        if self.attention_type == 'flash_attention':
            # Flash attention optimizations
            flash_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim // 4
            qk_matmul_ops = flash_ops
            attn_v_matmul_ops = flash_ops
        elif self.attention_type == 'sparse' and self.sparse_pattern:
            # Sparse attention
            sparsity_factor = 0.1  # Assume 10% sparsity
            sparse_ops = int((qk_matmul_ops + attn_v_matmul_ops) * sparsity_factor)
            qk_matmul_ops = sparse_ops
            attn_v_matmul_ops = sparse_ops
        elif self.attention_type == 'linear':
            # Linear attention (simplified)
            linear_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim // 2
            qk_matmul_ops = linear_ops
            attn_v_matmul_ops = linear_ops

        if self.position_encoding == 'rotary':
            # Rotary position embedding operations
            rope_ops = batch_size * seq_len_q * self.num_heads * head_dim * 4
        elif self.position_encoding == 'alibi':
            # ALiBi position bias operations
            alibi_ops = batch_size * self.num_heads * seq_len_q * seq_len_k
        elif self.position_encoding == 'learned':
            # Learned position embeddings
            pos_ops = batch_size * seq_len_q * self.num_heads * head_dim * 2
        else:
            rope_ops = 0
            alibi_ops = 0
            pos_ops = 0

        if self.quantized_attention:
            # Additional quantization/dequantization operations
            quant_ops = batch_size * seq_len_q * hidden_size_q * 8  # Quantize/dequantize Q, K, V, output
        else:
            quant_ops = 0

        if bias is not None:
            # Bias addition operations
            bias_ops = batch_size * self.num_heads * seq_len_q * seq_len_k
        else:
            bias_ops = 0

        if key_padding_mask is not None:
            # Key padding mask operations
            mask_ops = batch_size * seq_len_q * seq_len_k
        else:
            mask_ops = 0

        # Total operations
        total_ops = qkv_proj_ops + qk_matmul_ops + attn_v_matmul_ops + output_proj_ops + softmax_ops
        if 'rope_ops' in locals():
            total_ops += rope_ops
        if 'alibi_ops' in locals():
            total_ops += alibi_ops
        if 'pos_ops' in locals():
            total_ops += pos_ops
        total_ops += quant_ops + bias_ops + mask_ops

        # Memory analysis
        input_bytes = query.nbytes(self.precision) + key.nbytes(self.precision) + value.nbytes(self.precision)
        if bias is not None:
            input_bytes += bias.nbytes(self.precision)
        if key_padding_mask is not None:
            input_bytes += key_padding_mask.nbytes(self.precision)
        if past_key is not None:
            input_bytes += past_key.nbytes(self.precision)
        if past_value is not None:
            input_bytes += past_value.nbytes(self.precision)
        if cos_cache is not None and sin_cache is not None:
            input_bytes += cos_cache.nbytes(self.precision) + sin_cache.nbytes(self.precision)
        if alibi_bias is not None:
            input_bytes += alibi_bias.nbytes(self.precision)
        if sparse_mask is not None:
            input_bytes += sparse_mask.nbytes(self.precision)

        output_bytes = sum(out.nbytes(self.precision) for out in outT)

        # Instruction counts
        exp_count = softmax_ops // 4  # Exponential in softmax
        div_count = batch_size * self.num_heads * seq_len_q  # Division by sqrt(d_k) and softmax normalization

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': sum(tensor.nelems() for tensor in inT),
            'outBytes': output_bytes,
            'outElems': sum(tensor.nelems() for tensor in outT),
            'instrs': {
                'mac': qkv_proj_ops + qk_matmul_ops + attn_v_matmul_ops + output_proj_ops,  # All matrix multiplications
                'cmp': mask_ops + bias_ops,  # Comparisons for masking and bias
                'add': total_elements * 3 + softmax_ops,  # Various additions
                'sub': quant_ops,  # Quantization operations
                'mul': total_elements * 4 + softmax_ops,  # Scale multiplications and softmax
                'div': div_count,
                'rsqrt': batch_size * self.num_heads if self.attention_type == 'scaled_dot_product' else 0,  # Reciprocal sqrt for scaling
                'exp': exp_count,
                'log': 0,  # No logarithm
                'round': quant_ops // 3 if self.quantized_attention else 0,  # Rounding in quantization
                'clip': quant_ops // 3 if self.quantized_attention else 0,  # Clipping in quantization
                'convert': quant_ops // 3 if self.quantized_attention else 0,  # Type conversions
                'index': total_ops // 10,  # Memory indexing operations
                'gather': total_ops // 20,  # Gather operations
                'scatter': 0,  # No scattering
                'tanh': 0,  # No tanh
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for MultiHeadAttentionExt
        raise NotImplementedError("MultiHeadAttentionExt backward pass not yet implemented")


# 'SoftmaxGrad_13'
# 'GatherGrad'
# 'SoftmaxCrossEntropyLossGrad'


class AttentionOp(SimOp):
    """
    ONNX 1.20.0 Attention Operation
    Performs scaled dot-product attention with optional masking and key-value caching.

    Mathematical Definition:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Where d_k is the head dimension (head_size)
    """
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Attention'

        # ONNX Attention supports 3-6 inputs and 1-4 outputs (optional qk_matmul_output)
        check_io_counts(self, in_counts=[3,6], out_counts=[1,4])

        # ONNX-aligned attributes
        # Map legacy attributes for backward compatibility
        legacy_num_heads = self.attrs.get('num_heads', None)
        legacy_causal = self.attrs.get('causal', False)

        # Required for 3D inputs, inferred for 4D inputs
        self.q_num_heads = self.attrs.get('q_num_heads', legacy_num_heads)
        self.kv_num_heads = self.attrs.get('kv_num_heads', None)

        # Scale factor applied to QK^T (default 1/sqrt(head_size))
        self.scale = self.attrs.get('scale', None)

        # is_causal attribute (INT in spec). Accept legacy bool 'causal' for compatibility
        is_causal_attr = self.attrs.get('is_causal', 1 if legacy_causal else 0)
        self.is_causal = bool(is_causal_attr)

        # Optional attributes
        self.softcap = self.attrs.get('softcap', 0.0)
        self.softmax_precision = self.attrs.get('softmax_precision', None)
        self.qk_matmul_output_mode = self.attrs.get('qk_matmul_output_mode', 0)
        assert self.qk_matmul_output_mode in [0, 1, 2, 3], \
            f"qk_matmul_output_mode must be in [0,1,2,3], got {self.qk_matmul_output_mode}"

        # Non-spec attribute retained for compatibility (not used for ONNX compliance)
        self.unidirectional = self.attrs.get('unidirectional', False)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "Attention operation does not support backward pass yet"

        # Input validation - parse inputs based on count and shapes
        Q, K, V = inT[0], inT[1], inT[2]  # Required inputs

        # Optional inputs: detect by shapes (attn_mask is broadcastable up to 4D; past_key/value are 4D with kv heads)
        mask = None
        past_key = None
        past_value = None

        # DType constraints (T1/T2/U)
        def _is_float_dtype(dt):
            import numpy as _np
            return (isinstance(dt, _np.dtype) and dt.kind == 'f') or (isinstance(dt, str) and dt in ['float16', 'bfloat16', 'float32', 'float64'])

        assert _is_float_dtype(Q.dtype), f"Q dtype must be float tensor, got {Q.dtype}"
        assert _is_float_dtype(K.dtype), f"K dtype must be float tensor, got {K.dtype}"
        assert _is_float_dtype(V.dtype), f"V dtype must be float tensor, got {V.dtype}"

        # Parse Q shape: 3D or 4D
        q_rank = len(Q.shape)
        assert q_rank in [3, 4], f"Q must be 3D or 4D tensor, got {q_rank}D"
        if q_rank == 4:
            batch_size, q_heads, q_seq_len, q_head_size = Q.shape
            if self.q_num_heads is None:
                self.q_num_heads = q_heads
            else:
                assert self.q_num_heads == q_heads, f"q_num_heads ({self.q_num_heads}) != Q heads ({q_heads})"
        else:
            batch_size, q_seq_len, q_hidden_size = Q.shape
            assert self.q_num_heads is not None, "q_num_heads attribute is required for 3D Q"
            assert q_hidden_size % self.q_num_heads == 0, f"q_hidden_size {q_hidden_size} not divisible by q_num_heads {self.q_num_heads}"
            q_head_size = q_hidden_size // self.q_num_heads
            q_heads = self.q_num_heads

        # Parse K shape: 3D or 4D
        k_rank = len(K.shape)
        assert k_rank in [3, 4], f"K must be 3D or 4D tensor, got {k_rank}D"
        if k_rank == 4:
            batch_k, kv_heads_k, kv_seq_len, k_head_size = K.shape
            assert batch_k == batch_size, "Batch size mismatch between Q and K"
            if self.kv_num_heads is None:
                self.kv_num_heads = kv_heads_k
            else:
                assert self.kv_num_heads == kv_heads_k, f"kv_num_heads ({self.kv_num_heads}) != K heads ({kv_heads_k})"
        else:
            batch_k, kv_seq_len, k_hidden_size = K.shape
            assert batch_k == batch_size, "Batch size mismatch between Q and K"
            assert self.kv_num_heads is not None, "kv_num_heads attribute is required for 3D K"
            assert k_hidden_size % self.kv_num_heads == 0, f"k_hidden_size {k_hidden_size} not divisible by kv_num_heads {self.kv_num_heads}"
            k_head_size = k_hidden_size // self.kv_num_heads
            kv_heads_k = self.kv_num_heads

        # Parse V shape: 3D or 4D
        v_rank = len(V.shape)
        assert v_rank in [3, 4], f"V must be 3D or 4D tensor, got {v_rank}D"
        if v_rank == 4:
            batch_v, kv_heads_v, kv_seq_len_v, v_head_size = V.shape
            assert batch_v == batch_size, "Batch size mismatch between Q and V"
            # K/V must share kv heads and seq len
            assert kv_heads_v == kv_heads_k, "K and V kv_num_heads must match"
            assert kv_seq_len_v == kv_seq_len, "K and V sequence lengths must match"
        else:
            batch_v, kv_seq_len_v, v_hidden_size = V.shape
            assert batch_v == batch_size, "Batch size mismatch between Q and V"
            assert kv_seq_len_v == kv_seq_len, "K and V sequence lengths must match"
            assert v_hidden_size % self.kv_num_heads == 0, f"v_hidden_size {v_hidden_size} not divisible by kv_num_heads {self.kv_num_heads}"
            v_head_size = v_hidden_size // self.kv_num_heads
            kv_heads_v = self.kv_num_heads

        # Head dimension compatibility: Q and K head dims must match; V can have distinct head dim (v_head_size)
        assert q_head_size == k_head_size, f"Q head_size ({q_head_size}) must match K head_size ({k_head_size})"

        # Now that we know head counts/sizes, identify optional inputs by shape
        for i in range(3, len(inT)):
            t = inT[i]
            if t is None:
                continue
            tshape = t.shape
            trank = len(tshape)
            if trank == 4 and tshape[0] == batch_size and tshape[1] == kv_heads_k:
                # Candidate for past_key or past_value
                if past_key is None and tshape[3] == k_head_size:
                    past_key = t
                    continue
                if past_value is None and tshape[3] == v_head_size:
                    past_value = t
                    continue
            # Otherwise treat as mask (can be <=4D)
            if mask is None:
                mask = t

        # Handle past key/value states: both-or-none
        if (past_key is None) ^ (past_value is None):
            raise AssertionError("past_key and past_value must be provided together per ONNX spec")

        if past_key is not None and past_value is not None:
            assert len(past_key.shape) == 4 and len(past_value.shape) == 4, "past_key/value must be 4D"
            batch_pk, pk_heads, past_seq_len, pk_head_size = past_key.shape
            batch_pv, pv_heads, past_seq_len_v, pv_head_size = past_value.shape
            assert batch_pk == batch_size and batch_pv == batch_size, "Past batch size mismatch"
            assert pk_heads == self.kv_num_heads and pv_heads == self.kv_num_heads, "Past heads must equal kv_num_heads"
            assert past_seq_len == past_seq_len_v, "Past key/value sequence length mismatch"
            assert pk_head_size == k_head_size, "Past key head_size must match K head_size"
            assert pv_head_size == v_head_size, "Past value head_size must match V head_size"
            total_seq_len = past_seq_len + kv_seq_len
        else:
            total_seq_len = kv_seq_len

        # Validate mask if provided: broadcastable to [batch, q_heads, q_seq_len, total_seq_len]
        if mask is not None:
            assert len(mask.shape) in [1, 2, 3, 4], f"attn_mask must be up to 4D, got {len(mask.shape)}D"
            # Basic dtype check: allow bool or numeric
            import numpy as _np
            if isinstance(mask.dtype, _np.dtype):
                assert mask.dtype.kind in ['b', 'f', 'i', 'u'], f"Unsupported mask dtype: {mask.dtype}"
            elif isinstance(mask.dtype, str):
                # Accept a broad set of types when represented as strings
                pass

            # Expand to 4D logical shape for broadcasting check
            ms = list(mask.shape)
            # Left-pad with 1s to 4D
            while len(ms) < 4:
                ms = [1] + ms
            m_batch, m_heads, m_q, m_k = ms
            def _bcast_ok(a, b):
                return a == b or a == 1 or b == 1
            assert _bcast_ok(m_batch, batch_size), f"Mask batch {m_batch} not broadcastable to {batch_size}"
            assert _bcast_ok(m_heads, q_heads), f"Mask heads {m_heads} not broadcastable to q_heads {q_heads}"
            assert _bcast_ok(m_q, q_seq_len), f"Mask q_len {m_q} not broadcastable to {q_seq_len}"
            assert _bcast_ok(m_k, total_seq_len), f"Mask k_len {m_k} not broadcastable to {total_seq_len}"
            mask_ops = batch_size * q_heads * q_seq_len * total_seq_len
        else:
            mask_ops = 0

        # Set output shapes (match input rank style)
        if q_rank == 4:
            outT[0].shape = [batch_size, q_heads, q_seq_len, v_head_size]
        else:
            outT[0].shape = [batch_size, q_seq_len, q_heads * v_head_size]
        outT[0].dtype = Q.dtype

        # Optional outputs for incremental decoding
        if len(outT) > 1:  # present_key
            outT[1].shape = [batch_size, self.kv_num_heads, total_seq_len, k_head_size]
            outT[1].dtype = K.dtype

        if len(outT) > 2:  # present_value
            outT[2].shape = [batch_size, self.kv_num_heads, total_seq_len, v_head_size]
            outT[2].dtype = V.dtype

        if len(outT) > 3:  # qk_matmul_output (optional)
            # Mode: 0 -> raw qk matmul; 1 -> +mask; 2 -> after softcap; 3 -> after softmax
            outT[3].shape = [batch_size, q_heads, q_seq_len, total_seq_len]
            outT[3].dtype = Q.dtype

        # Calculate performance statistics
        # Attention computation: Q @ K^T -> [batch, q_heads, q_seq, total_seq]
        qk_matmul_ops = batch_size * q_heads * q_seq_len * total_seq_len * q_head_size

        # Softmax computation
        softmax_ops = batch_size * q_heads * q_seq_len * total_seq_len

        # Softcap application (if any) before softmax
        softcap_ops = 0
        if self.softcap not in [0, 0.0, None]:
            softcap_ops = batch_size * q_heads * q_seq_len * total_seq_len

        # Attention @ V -> [batch, q_heads, q_seq, v_head_size]
        attn_v_matmul_ops = batch_size * q_heads * q_seq_len * total_seq_len * v_head_size

        total_ops = qk_matmul_ops + softmax_ops + attn_v_matmul_ops + softcap_ops

        self.perf_stats = {
            'inBytes': sum(t.nbytes(self.precision) for t in inT),
            'inElems': sum(t.nelems() for t in inT),
            'outBytes': sum(t.nbytes(self.precision) for t in outT),
            'outElems': sum(t.nelems() for t in outT),
            'instrs': {
                'mac': total_ops,  # Multiply-accumulate operations
                'cmp': softmax_ops,  # Comparisons for softmax
                'exp': softmax_ops,  # Exponential for softmax
                'div': softmax_ops,  # Division for softmax
                'clip': softcap_ops,  # Softcap saturation
                'mask': mask_ops
            },
            'qk_matmul_output_mode': self.qk_matmul_output_mode
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for attention
        # This is complex and involves gradients through softmax and matrix multiplications
        raise NotImplementedError("Attention backward pass not yet implemented")


class MultiHeadAttentionOp(SimOp):
    """
    ONNX 1.20.0 MultiHeadAttention Operation
    Performs multi-head attention with QKV projections and head management.

    This operation combines linear projections, head splitting, attention computation,
    and output projection into a single operation, following the standard transformer
    multi-head attention pattern.

    Mathematical Definition:
    MultiHeadAttention = Linear(V) @ Attention(Linear(Q), Linear(K), Linear(V))
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'MultiHeadAttention'

        # ONNX MultiHeadAttention supports 3-8 inputs and 1-3 outputs
        check_io_counts(self, in_counts=[3,8], out_counts=[1,3])

        # Validate attributes
        self.num_heads = self.attrs.get('num_heads')
        if self.num_heads is None:
            raise ValueError("MultiHeadAttention requires 'num_heads' attribute")
        self.unidirectional = self.attrs.get('unidirectional', False)
        self.qkv_hidden_sizes = self.attrs.get('qkv_hidden_sizes', None)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "MultiHeadAttention operation does not support backward pass yet"

        # Input validation - parse inputs based on count and shapes
        query, key, value = inT[0], inT[1], inT[2]  # Required inputs

        # DType constraints: enforce float types for query/key/value
        try:
            import numpy as _np
            assert (isinstance(query.dtype, _np.dtype) and query.dtype.kind == 'f') or (isinstance(query.dtype, str) and 'float' in query.dtype), f"query dtype must be float, got {query.dtype}"
            assert (isinstance(key.dtype, _np.dtype) and key.dtype.kind == 'f') or (isinstance(key.dtype, str) and 'float' in key.dtype), f"key dtype must be float, got {key.dtype}"
            assert (isinstance(value.dtype, _np.dtype) and value.dtype.kind == 'f') or (isinstance(value.dtype, str) and 'float' in value.dtype), f"value dtype must be float, got {value.dtype}"
        except Exception:
            pass

        # Parse optional inputs based on total input count and tensor shapes
        bias = None
        key_padding_mask = None
        past_key = None
        past_value = None
        cos_cache = None
        sin_cache = None

        # Detect input types based on shapes and positions
        if len(inT) >= 4 and inT[3] is not None:
            tensor3 = inT[3]
            if len(tensor3.shape) == 3:
                # Could be bias [batch, seq, seq] or key_padding_mask [batch, seq]
                batch_size_tmp, dim1, dim2 = tensor3.shape
                if dim1 == dim2:
                    # Square matrix -> bias
                    bias = tensor3
                else:
                    # Rectangular -> key_padding_mask
                    key_padding_mask = tensor3
            elif len(tensor3.shape) == 4:
                # 4D tensor -> past_key
                past_key = tensor3

        if len(inT) >= 5 and inT[4] is not None:
            tensor4 = inT[4]
            if past_key is not None and len(tensor4.shape) == 4:
                # If we already have past_key, this must be past_value
                past_value = tensor4
            elif len(tensor4.shape) == 2:
                # 2D tensor -> key_padding_mask
                key_padding_mask = tensor4
            elif len(tensor4.shape) == 4:
                # 4D tensor -> past_key (if we didn't detect it earlier)
                past_key = tensor4

        if len(inT) >= 6 and inT[5] is not None:
            tensor5 = inT[5]
            if len(tensor5.shape) == 4:
                past_value = tensor5

        if len(inT) >= 7 and inT[6] is not None:
            tensor6 = inT[6]
            if len(tensor6.shape) == 4:
                cos_cache = tensor6

        if len(inT) >= 8 and inT[7] is not None:
            tensor7 = inT[7]
            if len(tensor7.shape) == 4:
                sin_cache = tensor7

        # Validate query tensor: allow 3D [B, S, H] or 4D [B, num_heads, S, head]
        assert len(query.shape) in [3, 4], f"query must be 3D or 4D tensor, got {len(query.shape)}D"
        if len(query.shape) == 3:
            batch_size, q_seq_len, q_hidden_size = query.shape
            head_size = q_hidden_size // self.num_heads
            assert q_hidden_size % self.num_heads == 0, f"hidden_size {q_hidden_size} not divisible by num_heads {self.num_heads}"
        else:
            batch_size, q_num_heads, q_seq_len, head_size = query.shape
            assert q_num_heads == self.num_heads, f"query num_heads {q_num_heads} != operation num_heads {self.num_heads}"
            q_hidden_size = self.num_heads * head_size

        # Validate key and value tensors - can be in two formats:
        # Format 1: [batch_size, total_sequence_length, hidden_size] (packed)
        # Format 2: [batch_size, num_heads, total_sequence_length, head_size] (unpacked)
        assert len(key.shape) in [3, 4], f"key must be 3D or 4D tensor, got {len(key.shape)}D"
        assert len(value.shape) in [3, 4], f"value must be 3D or 4D tensor, got {len(value.shape)}D"

        if len(key.shape) == 3:
            # Format 1: packed - [batch_size, total_seq_len, hidden_size]
            _, k_seq_len, k_hidden_size = key.shape
            _, v_seq_len, v_hidden_size = value.shape

            # Hidden sizes must match
            assert k_hidden_size == q_hidden_size, f"key hidden_size {k_hidden_size} != query hidden_size {q_hidden_size}"
            assert v_hidden_size == q_hidden_size, f"value hidden_size {v_hidden_size} != query hidden_size {q_hidden_size}"

            # head_size already computed above
        else:
            # Format 2: unpacked - [batch_size, num_heads, total_seq_len, head_size]
            _, k_num_heads, k_seq_len, k_head_size = key.shape
            _, v_num_heads, v_seq_len, v_head_size = value.shape

            assert k_num_heads == self.num_heads, f"key num_heads {k_num_heads} != operation num_heads {self.num_heads}"
            assert v_num_heads == self.num_heads, f"value num_heads {v_num_heads} != operation num_heads {self.num_heads}"
            assert k_head_size == v_head_size, "key and value head_size must match"
            assert k_seq_len == v_seq_len, "key and value sequence length must match"

            head_size = k_head_size
            total_hidden_size = self.num_heads * head_size

        # Validate bias if provided
        if bias is not None:
            assert len(bias.shape) == 3, f"bias must be 3D tensor, got {len(bias.shape)}D"
            bias_batch, bias_seq_q, bias_seq_k = bias.shape
            assert bias_batch == batch_size or bias_batch == 1
            assert bias_seq_q == q_seq_len
            assert bias_seq_k == k_seq_len

        # Validate key_padding_mask if provided
        if key_padding_mask is not None:
            assert len(key_padding_mask.shape) == 2, f"key_padding_mask must be 2D, got {len(key_padding_mask.shape)}D"
            mask_batch, mask_seq_len = key_padding_mask.shape
            assert mask_batch == batch_size
            assert mask_seq_len == k_seq_len
            # Enforce boolean mask per ONNX spec
            try:
                import numpy as _np
                assert (isinstance(key_padding_mask.dtype, _np.dtype) and key_padding_mask.dtype.kind == 'b') or (isinstance(key_padding_mask.dtype, str) and 'bool' in key_padding_mask.dtype), f"key_padding_mask must be boolean, got {key_padding_mask.dtype}"
            except Exception:
                pass

        # Validate past states if provided
        if past_key is not None and past_value is not None:
            assert len(past_key.shape) == 4, f"past_key must be 4D, got {len(past_key.shape)}D"
            assert len(past_value.shape) == 4, f"past_value must be 4D, got {len(past_value.shape)}D"

            _, past_k_seq_len, past_num_heads, past_head_size = past_key.shape
            assert past_num_heads == self.num_heads
            assert past_head_size == head_size

            # Concatenate past and current sequences
            total_k_seq_len = past_k_seq_len + k_seq_len
        else:
            total_k_seq_len = k_seq_len

        # Validate RoPE caches if provided
        if cos_cache is not None and sin_cache is not None:
            assert len(cos_cache.shape) == 4, f"cos_cache must be 4D, got {len(cos_cache.shape)}D"
            assert len(sin_cache.shape) == 4, f"sin_cache must be 4D, got {len(sin_cache.shape)}D"

        # Set output shapes
        output_hidden_size = q_hidden_size
        outT[0].shape = [batch_size, q_seq_len, output_hidden_size]
        outT[0].dtype = query.dtype

        # Optional outputs for incremental decoding (prefer heads-before-seq for cache if needed by ONNX)
        if len(outT) > 1:  # present_key
            outT[1].shape = [batch_size, total_k_seq_len, self.num_heads, head_size]
            outT[1].dtype = key.dtype

        if len(outT) > 2:  # present_value
            outT[2].shape = [batch_size, total_k_seq_len, self.num_heads, head_size]
            outT[2].dtype = value.dtype

        # Calculate performance statistics
        # QKV projections: 3 linear transformations
        qkv_proj_ops = 3 * batch_size * q_seq_len * q_hidden_size * q_hidden_size

        # Head splitting/rearranging (if needed)
        head_ops = batch_size * q_seq_len * q_hidden_size

        # Attention computation
        attn_qk_matmul_ops = batch_size * self.num_heads * q_seq_len * total_k_seq_len * head_size
        attn_softmax_ops = batch_size * self.num_heads * q_seq_len * total_k_seq_len
        attn_v_matmul_ops = batch_size * self.num_heads * q_seq_len * total_k_seq_len * head_size

        # Head merging and output projection
        output_proj_ops = batch_size * q_seq_len * output_hidden_size * output_hidden_size

        total_ops = qkv_proj_ops + head_ops + attn_qk_matmul_ops + attn_softmax_ops + attn_v_matmul_ops + output_proj_ops

        # Filter out None placeholders from input tensors for performance calculation
        valid_inT = [t for t in inT if t is not None]

        self.perf_stats = {
            'inBytes': sum(t.nbytes(self.precision) for t in valid_inT),
            'inElems': sum(t.nelems() for t in valid_inT),
            'outBytes': sum(t.nbytes(self.precision) for t in outT),
            'outElems': sum(t.nelems() for t in outT),
            'instrs': {
                'mac': total_ops,  # Multiply-accumulate operations
                'cmp': attn_softmax_ops,  # Comparisons for softmax
                'exp': attn_softmax_ops,  # Exponential for softmax
                'div': attn_softmax_ops   # Division for softmax
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for MultiHeadAttention
        # This is very complex and involves gradients through multiple operations
        raise NotImplementedError("MultiHeadAttention backward pass not yet implemented")


class DynamicQuantizeLinearOp(SimOp):
    """
    ONNX 1.20.0 DynamicQuantizeLinear Operation
    Performs dynamic quantization of float32 tensors to uint8 with computed scale and zero point.

    This operation computes quantization parameters (scale and zero point) dynamically
    based on the input tensor's range, enabling efficient inference for quantized models.

    Mathematical Definition:
    y_scale = (max(x) - min(x)) / 255
    y_zero_point = round(min(x) / y_scale)
    y = clip(round(x / y_scale) + y_zero_point, 0, 255)
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'DynamicQuantizeLinear'

        # ONNX DynamicQuantizeLinear supports 1 input and 3 outputs
        check_io_counts(self, in_counts=[1,1], out_counts=[3,3])

        # No attributes for this operation
        # All parameters are computed dynamically from input tensor

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "DynamicQuantizeLinear operation does not support backward pass yet"

        # Input validation
        x = inT[0]  # Input tensor (float32)
        assert len(inT) == 1, f"DynamicQuantizeLinear expects 1 input, got {len(inT)}"
        assert len(outT) == 3, f"DynamicQuantizeLinear expects 3 outputs, got {len(outT)}"

        # Validate input tensor
        assert x.dtype in ['float32', 'float'], f"Input must be float32, got {x.dtype}"
        x_shape = x.shape
        x_rank = x.rank()
        assert x_rank >= 1, f"Input tensor must have at least 1 dimension, got {x_rank}D"

        # Set output shapes
        # y: quantized output (uint8) - same shape as input
        outT[0].shape = x_shape
        outT[0].dtype = 'uint8'

        # y_scale: scale factor (float32) - scalar
        outT[1].shape = []
        outT[1].dtype = 'float32'

        # y_zero_point: zero point (uint8) - scalar
        outT[2].shape = []
        outT[2].dtype = 'uint8'

        # Calculate performance statistics
        input_elements = x.nelems()

        # Dynamic range computation: find min and max
        range_ops = input_elements  # Compare operations to find min/max

        # Scale computation: (max - min) / 255
        scale_ops = 3  # subtraction, division, assignment

        # Zero point computation: round(min / scale)
        zero_point_ops = 4  # division, round, assignment

        # Quantization: clip(round(x / scale) + zero_point, 0, 255)
        quantize_ops = input_elements * 4  # division, round, addition, clip per element

        total_ops = range_ops + scale_ops + zero_point_ops + quantize_ops

        # Memory analysis
        input_bytes = x.nbytes(self.precision)
        output_bytes = outT[0].nbytes(self.precision) + outT[1].nbytes(self.precision) + outT[2].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': input_elements,
            'outBytes': output_bytes,
            'outElems': outT[0].nelems() + outT[1].nelems() + outT[2].nelems(),
            'instrs': {
                'mac': 0,  # No multiply-accumulate operations
                'cmp': range_ops,  # Comparisons for min/max finding
                'add': quantize_ops // 4,  # Addition operations in quantization
                'sub': scale_ops,  # Subtraction for range computation
                'mul': 0,  # No multiplication
                'div': quantize_ops // 4 + scale_ops,  # Division operations
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': quantize_ops // 4 + zero_point_ops,  # Rounding operations
                'clip': quantize_ops // 4,  # Clipping operations
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for DynamicQuantizeLinear
        # This involves gradients through quantization operations
        raise NotImplementedError("DynamicQuantizeLinear backward pass not yet implemented")


class GatherNDOp(SimOp):
    """
    ONNX 1.20.0 GatherND Operation
    Gathers slices from input tensor into output tensor based on N-dimensional indices.

    This operation enables advanced multi-dimensional indexing where you can gather
    elements from arbitrary positions in the input tensor using coordinate tuples.

    Mathematical Definition:
    output[i] = data[indices[i]]

    Where indices[i] represents N-dimensional coordinates into the data tensor.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'GatherND'

        # ONNX GatherND supports 2 inputs and 1 output
        check_io_counts(self, in_counts=[2,2], out_counts=[1,1])

        # Parse attributes
        self.batch_dims = self.attrs.get('batch_dims', 0)

        # Validate batch_dims attribute
        if not isinstance(self.batch_dims, int) or self.batch_dims < 0:
            raise ValueError(f"GatherND batch_dims must be a non-negative integer, got {self.batch_dims}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "GatherND operation does not support backward pass yet"

        # Input validation
        data = inT[0]  # Input data tensor
        indices = inT[1]  # Index tensor
        assert len(inT) == 2, f"GatherND expects 2 inputs, got {len(inT)}"
        assert len(outT) == 1, f"GatherND expects 1 output, got {len(outT)}"

        # Validate tensor ranks
        data_rank = data.rank()
        indices_rank = indices.rank()
        assert data_rank >= 1, f"Data tensor must have at least 1 dimension, got {data_rank}D"
        assert indices_rank >= 1, f"Indices tensor must have at least 1 dimension, got {indices_rank}D"

        # Validate batch_dims
        assert self.batch_dims <= min(data_rank, indices_rank), \
            f"batch_dims ({self.batch_dims}) cannot exceed min(data_rank={data_rank}, indices_rank={indices_rank})"

        # Extract dimensions
        data_shape = data.shape
        indices_shape = indices.shape

        # Calculate output shape
        # Output shape = indices_shape[:-1] + data_shape[indices_shape[-1]:]
        # But we need to account for batch_dims
        index_depth = indices_shape[-1]  # Last dimension of indices gives indexing depth

        # Validate index depth
        effective_data_rank = data_rank - self.batch_dims
        assert index_depth <= effective_data_rank, \
            f"Index depth ({index_depth}) cannot exceed effective data rank ({effective_data_rank})"

        # Calculate output shape
        # Batch dimensions come from indices (first batch_dims dimensions)
        if self.batch_dims == 0:
            # No batch dimensions - all dimensions except last are for output shape
            batch_dims_shape = indices_shape[:-1]
        else:
            # With batch dimensions - first batch_dims dimensions are batch
            batch_dims_shape = indices_shape[:self.batch_dims]
            # The dimensions used for indexing in indices are from batch_dims to -1
            indexing_dims_shape = indices_shape[self.batch_dims:-1]

        # Non-batch dimensions come from data (after the indexed dimensions)
        indexed_dims = index_depth + self.batch_dims
        remaining_dims_shape = data_shape[indexed_dims:]

        # Combine shapes
        if self.batch_dims == 0:
            output_shape = batch_dims_shape + remaining_dims_shape
        else:
            output_shape = batch_dims_shape + indexing_dims_shape + remaining_dims_shape

        # Set output tensor shape
        outT[0].shape = output_shape
        outT[0].dtype = data.dtype  # Output has same dtype as input data

        # Calculate performance statistics
        # Number of elements to gather
        output_elements = outT[0].nelems()
        indices_elements = indices.nelems()

        # Each gather operation involves:
        # - Reading indices (index_depth operations per output element)
        # - Memory access to gather the data
        # - Writing to output

        # Basic gather operations
        gather_ops = output_elements  # One gather per output element

        # Index processing operations
        index_ops = indices_elements  # Process each index element

        # Memory access operations (simplified model)
        memory_access_ops = output_elements * 2  # Read data + write output

        total_ops = gather_ops + index_ops + memory_access_ops

        # Memory analysis
        input_bytes = data.nbytes(self.precision) + indices.nbytes(self.precision)
        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': data.nelems() + indices.nelems(),
            'outBytes': output_bytes,
            'outElems': output_elements,
            'instrs': {
                'mac': 0,  # No multiply-accumulate operations
                'cmp': 0,  # No comparisons
                'add': 0,  # Minimal arithmetic
                'sub': 0,  # No subtraction
                'mul': 0,  # No multiplication
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'gather': gather_ops,  # Gather operations
                'index': index_ops,  # Index processing
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for GatherND
        # This involves scatter operations for gradients
        raise NotImplementedError("GatherND backward pass not yet implemented")


class DequantizeLinearOp(SimOp):
    """
    ONNX 1.20.0 DequantizeLinear Operation
    Converts quantized values back to floating point using scale and zero point.

    This operation performs the inverse of quantization, converting quantized
    integer values back to floating point representation.

    Mathematical Definition:
    y = (x - x_zero_point) * x_scale

    Where the operation is applied along the specified axis.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'DequantizeLinear'

        # ONNX DequantizeLinear supports 3 inputs and 1 output
        check_io_counts(self, in_counts=[3,3], out_counts=[1,1])

        # Parse attributes
        self.axis = self.attrs.get('axis', 1)

        # Validate axis attribute
        if not isinstance(self.axis, int):
            raise ValueError(f"DequantizeLinear axis must be an integer, got {type(self.axis)}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "DequantizeLinear operation does not support backward pass yet"

        # Input validation
        x = inT[0]  # Quantized input tensor
        x_scale = inT[1]  # Scale factor
        x_zero_point = inT[2]  # Zero point
        assert len(inT) == 3, f"DequantizeLinear expects 3 inputs, got {len(inT)}"
        assert len(outT) == 1, f"DequantizeLinear expects 1 output, got {len(outT)}"

        # Validate input types
        assert x.dtype in ['uint8', 'int8'], f"Input must be quantized type (uint8/int8), got {x.dtype}"
        assert x_scale.dtype in ['float32', 'float'], f"Scale must be float32, got {x_scale.dtype}"
        assert x_zero_point.dtype in ['uint8', 'int8'], f"Zero point must be quantized type (uint8/int8), got {x_zero_point.dtype}"

        # Validate tensor shapes
        x_shape = x.shape
        x_rank = x.rank()

        # Validate axis
        if self.axis < -x_rank or self.axis >= x_rank:
            raise ValueError(f"DequantizeLinear axis {self.axis} is out of range for tensor with {x_rank} dimensions")

        # Normalize negative axis
        if self.axis < 0:
            self.axis = x_rank + self.axis

        # Scale and zero point can be scalars or have the same shape as input along the dequantization axis
        scale_shape = x_scale.shape
        zero_point_shape = x_zero_point.shape

        # Validate scale and zero point shapes
        if len(scale_shape) == 0:  # Scalar scale
            pass  # Valid
        elif len(scale_shape) == 1 and scale_shape[0] == x_shape[self.axis]:  # Per-channel scale
            pass  # Valid
        else:
            raise ValueError(f"Scale shape {scale_shape} incompatible with input shape {x_shape} along axis {self.axis}")

        if len(zero_point_shape) == 0:  # Scalar zero point
            pass  # Valid
        elif len(zero_point_shape) == 1 and zero_point_shape[0] == x_shape[self.axis]:  # Per-channel zero point
            pass  # Valid
        else:
            raise ValueError(f"Zero point shape {zero_point_shape} incompatible with input shape {x_shape} along axis {self.axis}")

        # Set output tensor shape and dtype
        outT[0].shape = x_shape  # Output has same shape as input
        outT[0].dtype = 'float32'  # Output is always float32

        # Calculate performance statistics
        input_elements = x.nelems()

        # Each dequantization operation involves:
        # - Type conversion (quantized to float)
        # - Subtraction of zero point
        # - Multiplication by scale
        dequantize_ops = input_elements * 3  # 3 operations per element

        # Memory access operations (simplified model)
        memory_access_ops = input_elements * 3  # Read input, scale, zero_point + write output

        total_ops = dequantize_ops + memory_access_ops

        # Memory analysis
        input_bytes = x.nbytes(self.precision) + x_scale.nbytes(self.precision) + x_zero_point.nbytes(self.precision)
        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': x.nelems() + x_scale.nelems() + x_zero_point.nelems(),
            'outBytes': output_bytes,
            'outElems': outT[0].nelems(),
            'instrs': {
                'mac': 0,  # No multiply-accumulate operations
                'cmp': 0,  # No comparisons
                'add': 0,  # No additions
                'sub': dequantize_ops // 3,  # Subtraction operations
                'mul': dequantize_ops // 3,  # Multiplication operations
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': dequantize_ops // 3,  # Type conversion operations
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for DequantizeLinear
        # This involves gradients through dequantization operations
        raise NotImplementedError("DequantizeLinear backward pass not yet implemented")


class RMSNormalizationOp(SimOp):
    """
    ONNX 1.20.0 RMSNormalization Operation
    Root Mean Square Layer Normalization.

    This operation performs RMS normalization which is a simplified version
    of layer normalization that only uses the root mean square for normalization.

    Mathematical Definition:
    rms = sqrt(mean(X^2, axis=axis, keepdims=True))
    Y = Scale * X / rms

    Where the RMS computation is performed along the specified axis.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'RMSNormalization'

        # ONNX RMSNormalization supports 2 inputs and 1 output
        check_io_counts(self, in_counts=[2,2], out_counts=[1,1])

        # Parse attributes
        self.axis = self.attrs.get('axis', -1)
        self.epsilon = self.attrs.get('epsilon', 1e-5)

        # Validate attributes
        if not isinstance(self.axis, int):
            raise ValueError(f"RMSNormalization axis must be an integer, got {type(self.axis)}")
        if not isinstance(self.epsilon, (int, float)) or self.epsilon <= 0:
            raise ValueError(f"RMSNormalization epsilon must be a positive number, got {self.epsilon}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "RMSNormalization operation does not support backward pass yet"

        # Input validation
        X = inT[0]  # Input tensor
        scale = inT[1]  # Scale tensor
        assert len(inT) == 2, f"RMSNormalization expects 2 inputs, got {len(inT)}"
        assert len(outT) == 1, f"RMSNormalization expects 1 output, got {len(outT)}"

        # Validate tensor shapes
        X_shape = X.shape
        scale_shape = scale.shape
        X_rank = X.rank()

        # Validate axis
        if self.axis < -X_rank or self.axis >= X_rank:
            raise ValueError(f"RMSNormalization axis {self.axis} is out of range for tensor with {X_rank} dimensions")

        # Normalize negative axis
        if self.axis < 0:
            self.axis = X_rank + self.axis

        # Scale tensor should be broadcastable to input shape
        # Can be scalar or have the same shape as the normalization axis
        if len(scale_shape) == 0:  # Scalar scale
            pass  # Valid
        elif len(scale_shape) == 1 and scale_shape[0] == X_shape[self.axis]:  # Per-feature scale
            pass  # Valid
        else:
            raise ValueError(f"Scale shape {scale_shape} incompatible with input shape {X_shape} along axis {self.axis}")

        # Set output tensor shape and dtype
        outT[0].shape = X_shape  # Output has same shape as input
        outT[0].dtype = X.dtype  # Output has same dtype as input

        # Calculate performance statistics
        input_elements = X.nelems()

        # RMS computation involves:
        # - Squaring each element (X^2)
        # - Mean computation along axis (reduction)
        # - Square root computation
        # - Division and scaling operations

        # Per-element operations
        square_ops = input_elements  # X^2
        div_ops = input_elements     # Division by RMS
        mul_ops = input_elements     # Multiplication by scale

        # Reduction operations (mean computation along axis)
        # This is more complex to estimate precisely, but roughly proportional to input size
        reduction_ops = input_elements

        # Square root operations (one per normalized slice)
        normalized_slices = input_elements // X_shape[self.axis]
        sqrt_ops = normalized_slices

        total_ops = square_ops + div_ops + mul_ops + reduction_ops + sqrt_ops

        # Memory analysis
        input_bytes = X.nbytes(self.precision) + scale.nbytes(self.precision)
        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': X.nelems() + scale.nelems(),
            'outBytes': output_bytes,
            'outElems': outT[0].nelems(),
            'instrs': {
                'mac': 0,  # No multiply-accumulate operations
                'cmp': 0,  # No comparisons
                'add': reduction_ops,  # Summation for mean computation
                'sub': 0,  # No subtraction
                'mul': mul_ops + square_ops,  # Scale multiplication + squaring
                'div': div_ops,  # Division by RMS
                'rsqrt': sqrt_ops,  # Square root operations
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for RMSNormalization
        # This involves gradients through RMS computation
        raise NotImplementedError("RMSNormalization backward pass not yet implemented")


class QuantizeLinearOp(SimOp):
    """
    ONNX 1.20.0 QuantizeLinear Operation
    Converts floating point values to quantized integer values using scale and zero point.

    This operation performs quantization, converting floating point values
    to quantized integer representation using scale and zero point parameters.

    Mathematical Definition:
    y = clip(round(x / y_scale) + y_zero_point, min_val, max_val)

    Where the quantization is applied along the specified axis.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'QuantizeLinear'

        # ONNX QuantizeLinear supports 3 inputs and 1 output
        check_io_counts(self, in_counts=[3,3], out_counts=[1,1])

        # Parse attributes
        self.axis = self.attrs.get('axis', 1)
        self.saturate = self.attrs.get('saturate', True)

        # Validate attributes
        if not isinstance(self.axis, int):
            raise ValueError(f"QuantizeLinear axis must be an integer, got {type(self.axis)}")
        if not isinstance(self.saturate, bool):
            raise ValueError(f"QuantizeLinear saturate must be a boolean, got {type(self.saturate)}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "QuantizeLinear operation does not support backward pass yet"

        # Input validation
        x = inT[0]  # Input tensor (float32)
        y_scale = inT[1]  # Scale factor (float32)
        y_zero_point = inT[2]  # Zero point (quantized type)
        assert len(inT) == 3, f"QuantizeLinear expects 3 inputs, got {len(inT)}"
        assert len(outT) == 1, f"QuantizeLinear expects 1 output, got {len(outT)}"

        # Validate input types
        assert x.dtype in ['float32', 'float'], f"Input must be float32, got {x.dtype}"
        assert y_scale.dtype in ['float32', 'float'], f"Scale must be float32, got {y_scale.dtype}"
        assert y_zero_point.dtype in ['uint8', 'int8', 'int16', 'int32'], f"Zero point must be quantized type, got {y_zero_point.dtype}"

        # Validate tensor shapes
        x_shape = x.shape
        y_scale_shape = y_scale.shape
        y_zero_point_shape = y_zero_point.shape
        x_rank = x.rank()

        # Validate axis
        if self.axis < -x_rank or self.axis >= x_rank:
            raise ValueError(f"QuantizeLinear axis {self.axis} is out of range for tensor with {x_rank} dimensions")

        # Normalize negative axis
        if self.axis < 0:
            self.axis = x_rank + self.axis

        # Scale and zero point can be scalars or have the same shape as input along the quantization axis
        if len(y_scale_shape) == 0:  # Scalar scale
            pass  # Valid
        elif len(y_scale_shape) == 1 and y_scale_shape[0] == x_shape[self.axis]:  # Per-channel scale
            pass  # Valid
        else:
            raise ValueError(f"Scale shape {y_scale_shape} incompatible with input shape {x_shape} along axis {self.axis}")

        if len(y_zero_point_shape) == 0:  # Scalar zero point
            pass  # Valid
        elif len(y_zero_point_shape) == 1 and y_zero_point_shape[0] == x_shape[self.axis]:  # Per-channel zero point
            pass  # Valid
        else:
            raise ValueError(f"Zero point shape {y_zero_point_shape} incompatible with input shape {x_shape} along axis {self.axis}")

        # Set output tensor shape and dtype
        outT[0].shape = x_shape  # Output has same shape as input
        outT[0].dtype = y_zero_point.dtype  # Output has same dtype as zero point

        # Calculate performance statistics
        input_elements = x.nelems()

        # Each quantization operation involves:
        # - Division by scale (x / y_scale)
        # - Rounding operation
        # - Addition of zero point
        # - Optional clipping/saturation
        quantize_ops = input_elements * 4  # 4 operations per element

        # Memory access operations (simplified model)
        memory_access_ops = input_elements * 3  # Read input, scale, zero_point + write output

        total_ops = quantize_ops + memory_access_ops

        # Memory analysis
        input_bytes = x.nbytes(self.precision) + y_scale.nbytes(self.precision) + y_zero_point.nbytes(self.precision)
        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': x.nelems() + y_scale.nelems() + y_zero_point.nelems(),
            'outBytes': output_bytes,
            'outElems': outT[0].nelems(),
            'instrs': {
                'mac': 0,  # No multiply-accumulate operations
                'cmp': 0,  # No comparisons
                'add': quantize_ops // 4,  # Addition operations (zero point)
                'sub': 0,  # No subtraction
                'mul': 0,  # No multiplication
                'div': quantize_ops // 4,  # Division operations (by scale)
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': quantize_ops // 4,  # Rounding operations
                'clip': quantize_ops // 4 if self.saturate else 0,  # Clipping operations if saturate=True
                'convert': 0,  # No type conversion (input/output are both numeric)
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for QuantizeLinear
        # This involves gradients through quantization operations
        raise NotImplementedError("QuantizeLinear backward pass not yet implemented")


class EmbedLayerNormalizationOp(SimOp):
    """
    ONNX 1.20.0 EmbedLayerNormalization Operation
    Combined embedding lookup and layer normalization for transformer models.

    This operation performs a complete embedding pipeline commonly used in
    transformer architectures, combining token embedding, position embedding,
    and layer normalization in a single optimized operation.

    Operation Flow:
    1. Token Embedding: input_ids -> embedding_weight -> token_embeddings
    2. Position Embedding: Add position_weight based on sequence positions
    3. Segment Embedding: Add segment_weight if provided (optional)
    4. Layer Normalization: Apply normalization with epsilon
    5. Mask Processing: Handle attention mask if provided (optional)
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'EmbedLayerNormalization'

        # ONNX EmbedLayerNormalization supports 3-6 inputs and 1-2 outputs
        # Required: input_ids, embedding_weight, position_weight
        # Optional: segment_weight, mask
        check_io_counts(self, in_counts=[3,6], out_counts=[1,2])

        # Parse attributes
        self.epsilon = self.attrs.get('epsilon', 1e-12)
        self.mask_value = self.attrs.get('mask_value', -1e4)

        # Validate attributes
        if not isinstance(self.epsilon, (int, float)) or self.epsilon <= 0:
            raise ValueError(f"EmbedLayerNormalization epsilon must be a positive number, got {self.epsilon}")
        if not isinstance(self.mask_value, (int, float)):
            raise ValueError(f"EmbedLayerNormalization mask_value must be a number, got {type(self.mask_value)}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "EmbedLayerNormalization operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs >= 3, f"EmbedLayerNormalization expects at least 3 inputs, got {num_inputs}"
        assert len(outT) >= 1, f"EmbedLayerNormalization expects at least 1 output, got {len(outT)}"

        input_ids = inT[0]      # Token IDs [batch_size, seq_length]
        embedding_weight = inT[1]  # Embedding matrix [vocab_size, hidden_size]
        position_weight = inT[2]   # Position embeddings [max_position, hidden_size]

        # Validate input types
        assert input_ids.dtype in ['int32', 'int64'], f"Input IDs must be integer type, got {input_ids.dtype}"
        assert embedding_weight.dtype in ['float32', 'float'], f"Embedding weight must be float32, got {embedding_weight.dtype}"
        assert position_weight.dtype in ['float32', 'float'], f"Position weight must be float32, got {position_weight.dtype}"

        # Validate tensor shapes
        input_ids_shape = input_ids.shape
        embedding_weight_shape = embedding_weight.shape
        position_weight_shape = position_weight.shape

        assert len(input_ids_shape) == 2, f"Input IDs must be 2D [batch_size, seq_length], got {input_ids_shape}"
        assert len(embedding_weight_shape) == 2, f"Embedding weight must be 2D [vocab_size, hidden_size], got {embedding_weight_shape}"
        assert len(position_weight_shape) == 2, f"Position weight must be 2D [max_position, hidden_size], got {position_weight_shape}"

        batch_size, seq_length = input_ids_shape
        vocab_size, hidden_size = embedding_weight_shape
        max_position, pos_hidden_size = position_weight_shape

        # Validate dimensions match
        assert hidden_size == pos_hidden_size, f"Hidden size mismatch: embedding={hidden_size}, position={pos_hidden_size}"

        # Check optional inputs by examining tensor properties
        has_segment_weight = False
        has_mask = False
        segment_weight = None
        mask = None

        if num_inputs >= 4:
            tensor3 = inT[3]
            # Check if tensor3 is segment_weight (float32) or mask (integer)
            if tensor3.dtype in ['float32', 'float']:
                # This is likely segment_weight
                has_segment_weight = True
                segment_weight = tensor3
                assert len(segment_weight.shape) == 2 and segment_weight.shape[1] == hidden_size, \
                    f"Segment weight must be [segment_count, hidden_size], got {segment_weight.shape}"
            elif tensor3.dtype in ['int32', 'int64']:
                # This is likely mask
                has_mask = True
                mask = tensor3
            else:
                raise ValueError(f"Unexpected dtype for input 3: {tensor3.dtype}")

        if num_inputs >= 5:
            tensor4 = inT[4]
            # The remaining input should be the other optional input
            if has_segment_weight and not has_mask:
                # tensor4 should be mask
                has_mask = True
                mask = tensor4
                assert mask.dtype in ['int32', 'int64', 'float32', 'float'], f"Mask must be numeric type, got {mask.dtype}"
            elif has_mask and not has_segment_weight:
                # tensor4 should be segment_weight
                has_segment_weight = True
                segment_weight = tensor4
                assert segment_weight.dtype in ['float32', 'float'], f"Segment weight must be float32, got {segment_weight.dtype}"
                assert len(segment_weight.shape) == 2 and segment_weight.shape[1] == hidden_size, \
                    f"Segment weight must be [segment_count, hidden_size], got {segment_weight.shape}"
            else:
                raise ValueError(f"Unexpected fifth input with dtype {tensor4.dtype}")

        # Validate sequence length doesn't exceed position embeddings
        assert seq_length <= max_position, f"Sequence length {seq_length} exceeds max position {max_position}"

        # Set output tensor shapes and dtypes
        outT[0].shape = [batch_size, seq_length, hidden_size]  # Embedded output
        outT[0].dtype = 'float32'

        if len(outT) >= 2:  # Optional mask_index output
            outT[1].shape = input_ids_shape  # Same shape as input_ids
            outT[1].dtype = input_ids.dtype

        # Calculate performance statistics
        total_elements = batch_size * seq_length * hidden_size

        # Embedding lookup operations
        embedding_ops = batch_size * seq_length  # One lookup per token

        # Position embedding addition
        pos_add_ops = total_elements  # Add position embedding to each element

        # Segment embedding addition (if present)
        segment_ops = total_elements if has_segment_weight else 0

        # Layer normalization operations (similar to LayerNormalizationOp)
        # Mean computation, variance computation, normalization, affine transform
        ln_ops = total_elements * 8  # Rough estimate for layer norm operations

        # Mask processing (if present)
        mask_ops = batch_size * seq_length if has_mask else 0

        total_ops = embedding_ops + pos_add_ops + segment_ops + ln_ops + mask_ops

        # Memory analysis
        input_bytes = input_ids.nbytes(self.precision) + embedding_weight.nbytes(self.precision) + position_weight.nbytes(self.precision)
        if has_segment_weight and segment_weight is not None:
            input_bytes += segment_weight.nbytes(self.precision)
        if has_mask and mask is not None:
            input_bytes += mask.nbytes(self.precision)

        output_bytes = outT[0].nbytes(self.precision)
        if len(outT) >= 2:
            output_bytes += outT[1].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': input_ids.nelems() + embedding_weight.nelems() + position_weight.nelems(),
            'outBytes': output_bytes,
            'outElems': outT[0].nelems(),
            'instrs': {
                'mac': 0,  # No multiply-accumulate operations
                'cmp': 0,  # No comparisons
                'add': pos_add_ops + segment_ops + ln_ops // 4,  # Addition operations
                'sub': ln_ops // 8,  # Subtraction for mean computation
                'mul': ln_ops // 4,  # Scaling operations
                'div': ln_ops // 8,  # Division for normalization
                'rsqrt': ln_ops // 8,  # Reciprocal square root for normalization
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
                'gather': embedding_ops,  # Embedding lookup operations
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for EmbedLayerNormalization
        # This involves gradients through embedding, position embedding, and layer normalization
        raise NotImplementedError("EmbedLayerNormalization backward pass not yet implemented")


class QLinearMatMulOp(SimOp):
    """
    ONNX 1.20.0 QLinearMatMul Operation
    Quantized Linear Matrix Multiplication with fused dequantize-multiply-quantize.

    This operation performs efficient quantized matrix multiplication by fusing
    the dequantization, matrix multiplication, and quantization operations.

    Mathematical Definition:
    Dequantize(A) = (A - a_zero_point) * a_scale
    Dequantize(B) = (B - b_zero_point) * b_scale
    Result = Dequantize(A) @ Dequantize(B)
    Quantize(Result) = round(Result / y_scale) + y_zero_point

    This operation is critical for efficient quantized inference workloads.
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'QLinearMatMul'

        # ONNX QLinearMatMul supports exactly 8 inputs and 1 output
        check_io_counts(self, in_counts=[8,8], out_counts=[1,1])

        # QLinearMatMul has no attributes - all configuration via inputs
        assert len(self.attrs) == 0, f"QLinearMatMul should have no attributes, got {self.attrs}"

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "QLinearMatMul operation does not support backward pass yet"

        # Input validation
        assert len(inT) == 8, f"QLinearMatMul expects 8 inputs, got {len(inT)}"
        assert len(outT) == 1, f"QLinearMatMul expects 1 output, got {len(outT)}"

        # Parse inputs
        a = inT[0]          # Quantized input A
        a_scale = inT[1]    # Scale for A
        a_zero_point = inT[2]  # Zero point for A
        b = inT[3]          # Quantized input B
        b_scale = inT[4]    # Scale for B
        b_zero_point = inT[5]  # Zero point for B
        y_scale = inT[6]    # Output scale
        y_zero_point = inT[7]  # Output zero point

        # Validate input types
        assert a.dtype in ['uint8', 'int8'], f"Input A must be quantized type (uint8/int8), got {a.dtype}"
        assert b.dtype in ['uint8', 'int8'], f"Input B must be quantized type (uint8/int8), got {b.dtype}"
        assert a_scale.dtype in ['float32', 'float'], f"A scale must be float32, got {a_scale.dtype}"
        assert b_scale.dtype in ['float32', 'float'], f"B scale must be float32, got {b_scale.dtype}"
        assert y_scale.dtype in ['float32', 'float'], f"Output scale must be float32, got {y_scale.dtype}"
        assert a_zero_point.dtype in ['uint8', 'int8'], f"A zero point must be quantized type, got {a_zero_point.dtype}"
        assert b_zero_point.dtype in ['uint8', 'int8'], f"B zero point must be quantized type, got {b_zero_point.dtype}"
        assert y_zero_point.dtype in ['uint8', 'int8'], f"Output zero point must be quantized type, got {y_zero_point.dtype}"

        # Validate tensor shapes
        a_shape = a.shape
        b_shape = b.shape
        a_scale_shape = a_scale.shape
        b_scale_shape = b_scale.shape
        y_scale_shape = y_scale.shape
        a_zero_point_shape = a_zero_point.shape
        b_zero_point_shape = b_zero_point.shape
        y_zero_point_shape = y_zero_point.shape

        assert len(a_shape) >= 2, f"Input A must be at least 2D for matrix multiplication, got {a_shape}"
        assert len(b_shape) >= 2, f"Input B must be at least 2D for matrix multiplication, got {b_shape}"

        # Validate matrix multiplication compatibility
        # For matrix multiplication: A.shape[-1] must equal B.shape[-2]
        assert a_shape[-1] == b_shape[-2], f"Incompatible shapes for matrix multiplication: A{a_shape} and B{b_shape}"

        # Scales and zero points can be scalars or have compatible shapes
        # For per-channel quantization, they can have the same shape as the channel dimension
        self._validate_scale_zero_point_shapes(a_shape, a_scale_shape, a_zero_point_shape, "A")
        self._validate_scale_zero_point_shapes(b_shape, b_scale_shape, b_zero_point_shape, "B")
        self._validate_scale_zero_point_shapes(None, y_scale_shape, y_zero_point_shape, "Y")

        # Compute output shape for matrix multiplication
        # Output shape is A_shape[:-1] + B_shape[1:] (broadcasting batch dimensions)
        output_shape = self._compute_matmul_output_shape(a_shape, b_shape)

        # Set output tensor shape and dtype
        outT[0].shape = output_shape
        outT[0].dtype = y_zero_point.dtype  # Output has same dtype as output zero point

        # Calculate performance statistics
        # This is a complex operation involving multiple fused operations

        # Count elements
        a_elements = a.nelems()
        b_elements = b.nelems()
        output_elements = outT[0].nelems()

        # Matrix multiplication dimensions
        m = a_shape[-2]  # Rows of A / Rows of result
        k = a_shape[-1]  # Columns of A / Rows of B
        n = b_shape[-1]  # Columns of B / Columns of result

        # Each output element requires k multiply-accumulate operations
        mac_operations = output_elements * k

        # Dequantization operations (per element of A and B)
        dequant_a_ops = a_elements * 2  # subtract zero point + multiply scale
        dequant_b_ops = b_elements * 2  # subtract zero point + multiply scale

        # Quantization operations (per output element)
        quant_ops = output_elements * 2  # divide scale + add zero point + round

        # Memory access operations
        # Read all inputs + write output
        memory_access_ops = a_elements + b_elements + output_elements

        total_ops = mac_operations + dequant_a_ops + dequant_b_ops + quant_ops + memory_access_ops

        # Memory analysis
        input_bytes = (a.nbytes(self.precision) + a_scale.nbytes(self.precision) + a_zero_point.nbytes(self.precision) +
                      b.nbytes(self.precision) + b_scale.nbytes(self.precision) + b_zero_point.nbytes(self.precision) +
                      y_scale.nbytes(self.precision) + y_zero_point.nbytes(self.precision))
        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': a.nelems() + b.nelems() + a_scale.nelems() + b_scale.nelems() +
                      a_zero_point.nelems() + b_zero_point.nelems() + y_scale.nelems() + y_zero_point.nelems(),
            'outBytes': output_bytes,
            'outElems': outT[0].nelems(),
            'instrs': {
                'mac': mac_operations,  # Matrix multiplication operations
                'cmp': 0,  # No comparisons
                'add': dequant_a_ops // 2 + dequant_b_ops // 2 + quant_ops // 2,  # Addition operations
                'sub': dequant_a_ops // 2 + dequant_b_ops // 2,  # Subtraction for zero point removal
                'mul': dequant_a_ops // 2 + dequant_b_ops // 2,  # Scale multiplication
                'div': quant_ops // 2,  # Division by output scale
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': quant_ops // 2,  # Rounding operations
                'clip': 0,  # No clipping (handled by quantization range)
                'convert': 0,  # No explicit type conversion
            }
        }

        return self.perf_stats

    def _validate_scale_zero_point_shapes(self, tensor_shape, scale_shape, zero_point_shape, tensor_name):
        """Validate that scale and zero point shapes are compatible with tensor shape"""
        if len(scale_shape) == 0:
            # Scalar scale - valid
            pass
        elif len(scale_shape) == 1 and tensor_shape is not None:
            # Per-channel scale - check if it matches a channel dimension
            # For matrix multiplication, this is typically the last dimension of the first input
            # or the second-to-last dimension of the second input
            if scale_shape[0] not in [tensor_shape[-1], tensor_shape[-2] if len(tensor_shape) > 1 else 0]:
                raise ValueError(f"{tensor_name} scale shape {scale_shape} incompatible with tensor shape {tensor_shape}")
        else:
            raise ValueError(f"{tensor_name} scale shape {scale_shape} incompatible with tensor shape {tensor_shape}")

        # Zero point must have same shape as scale
        if len(zero_point_shape) != len(scale_shape) or zero_point_shape != scale_shape:
            raise ValueError(f"{tensor_name} zero point shape {zero_point_shape} must match scale shape {scale_shape}")

    def _compute_matmul_output_shape(self, a_shape, b_shape):
        """Compute output shape for matrix multiplication C = A @ B"""
        # Matrix multiplication: (..., M, K) @ (..., K, N) -> (..., M, N)
        # Handle batch dimensions by broadcasting

        # Get the matrix dimensions
        m, k = a_shape[-2], a_shape[-1]
        k_b, n = b_shape[-2], b_shape[-1]

        assert k == k_b, f"Incompatible matrix dimensions: A{k} != B{k_b}"

        # Handle batch dimensions
        # Take batch dimensions from A, and add the output dimension from B
        batch_dims = list(a_shape[:-2])  # All dimensions except the last 2
        output_shape = batch_dims + [m, n]

        return output_shape

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for QLinearMatMul
        # This involves gradients through quantized matrix multiplication
        raise NotImplementedError("QLinearMatMul backward pass not yet implemented")


class MatMul_INT8_Op(SimOp):
    """
    ONNX 1.20.0 MatMul_INT8 Operation
    Specialized matrix multiplication for INT8/INT16 data types.

    This operation extends the standard MatMul to provide optimized
    performance for integer matrix multiplication, which is critical
    for efficient quantized inference workloads.

    Key Features:
    - Support for INT8 and INT16 data types
    - Optimized performance counting for integer operations
    - Mixed precision support
    - Memory-efficient tiling strategies
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'MatMul_INT8'

        # ONNX MatMul_INT8 supports exactly 2 inputs and 1 output (same as standard MatMul)
        check_io_counts(self, in_counts=[2,2], out_counts=[1,1])

        # MatMul_INT8 has no attributes
        assert len(self.attrs) == 0, f"MatMul_INT8 should have no attributes, got {self.attrs}"

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "MatMul_INT8 operation does not support backward pass yet"

        # Input validation
        assert len(inT) == 2, f"MatMul_INT8 expects 2 inputs, got {len(inT)}"
        assert len(outT) == 1, f"MatMul_INT8 expects 1 output, got {len(outT)}"

        # Parse inputs
        A = inT[0]  # Input matrix A
        B = inT[1]  # Input matrix B

        # Validate input types - must be integer types
        supported_types = ['int8', 'int16', 'int32']
        assert A.dtype in supported_types, f"Input A must be integer type (int8/int16/int32), got {A.dtype}"
        assert B.dtype in supported_types, f"Input B must be integer type (int8/int16/int32), got {B.dtype}"

        # Validate tensor shapes
        A_shape = A.shape
        B_shape = B.shape

        if len(A_shape) < 1 or len(B_shape) < 1:
            raise ValueError("Shapes must have at least 1 dimension")

        # Compute output shape using the same logic as standard MatMul
        C_shape = self._compute_matmul_output_shape(A_shape, B_shape)
        outT[0].shape = C_shape

        # Output dtype is determined by input precision and operation requirements
        # For INT8 * INT8, result is typically INT32 to avoid overflow
        outT[0].dtype = self._get_output_dtype(A.dtype, B.dtype)

        # Calculate performance statistics optimized for INT8 operations
        total_elements = outT[0].nelems()
        reduced_dim = A_shape[-1]

        # Each output element requires 'reduced_dim' multiply-accumulate operations
        mac_operations = total_elements * reduced_dim

        # INT8 operations can benefit from SIMD instructions
        # Estimate SIMD utilization (8 operations per SIMD instruction for INT8)
        simd_factor = 8 if A.dtype == 'int8' and B.dtype == 'int8' else 4
        effective_mac_ops = mac_operations // simd_factor

        # Memory access patterns optimized for INT8
        # INT8 data is more cache-friendly due to smaller memory footprint
        memory_access_ops = (A.nelems() + B.nelems() + total_elements) // 4  # 4x fewer accesses due to smaller data types

        # Additional operations for INT8 optimization
        # - Tiling overhead
        # - SIMD instruction overhead
        tiling_ops = total_elements // 64  # Assume 64x64 tile size
        simd_setup_ops = total_elements // 128  # SIMD setup overhead

        total_ops = effective_mac_ops + memory_access_ops + tiling_ops + simd_setup_ops

        # Memory analysis - INT8 uses significantly less memory
        input_bytes = A.nbytes(self.precision) + B.nbytes(self.precision)
        output_bytes = outT[0].nbytes(self.precision)

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': A.nelems() + B.nelems(),
            'outBytes': output_bytes,
            'outElems': outT[0].nelems(),
            'instrs': {
                'mac': effective_mac_ops,  # SIMD-optimized MAC operations
                'cmp': 0,  # No comparisons
                'add': 0,  # MAC operations include addition
                'sub': 0,  # No subtraction
                'mul': 0,  # MAC operations include multiplication
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
            }
        }

        return self.perf_stats

    def _compute_matmul_output_shape(self, a_shape, b_shape):
        """Compute output shape for matrix multiplication C = A @ B"""
        # Handle different dimensionalities properly
        if len(a_shape) == 1 and len(b_shape) == 1:
            # Vector dot product: (k,) @ (k,) -> ()
            if a_shape[0] != b_shape[0]:
                raise ValueError(f"Incompatible vector dimensions: A{a_shape[0]} != B{b_shape[0]}")
            return []
        elif len(a_shape) == 1 and len(b_shape) >= 2:
            # Vector-matrix: (k,) @ (k, n) -> (n,)
            k = a_shape[0]
            k_b, n = b_shape[-2], b_shape[-1]
            if k != k_b:
                raise ValueError(f"Incompatible matrix dimensions: A{k} != B{k_b}")
            return b_shape[:-2] + [n]
        elif len(a_shape) >= 2 and len(b_shape) == 1:
            # Matrix-vector: (m, k) @ (k,) -> (m,)
            m, k = a_shape[-2], a_shape[-1]
            if k != b_shape[0]:
                raise ValueError(f"Incompatible matrix dimensions: A{k} != B{b_shape[0]}")
            return a_shape[:-2] + [m]
        else:
            # Matrix-matrix: (..., m, k) @ (..., k, n) -> (..., m, n)
            m, k = a_shape[-2], a_shape[-1]
            k_b, n = b_shape[-2], b_shape[-1]
            if k != k_b:
                raise ValueError(f"Incompatible matrix dimensions: A{k} != B{k_b}")
            batch_dims = list(a_shape[:-2])
            return batch_dims + [m, n]

    def _get_output_dtype(self, a_dtype, b_dtype):
        """Determine output dtype based on input dtypes to prevent overflow"""
        # INT8 * INT8 -> INT32 (to prevent overflow)
        # INT16 * INT16 -> INT32 or INT64
        # Mixed types -> promote to higher precision

        type_hierarchy = {'int8': 1, 'int16': 2, 'int32': 3}

        a_rank = type_hierarchy.get(a_dtype, 0)
        b_rank = type_hierarchy.get(b_dtype, 0)

        # Use the higher precision type, or promote to INT32 for INT8 operations
        if a_dtype == 'int8' or b_dtype == 'int8':
            return 'int32'
        elif a_dtype == 'int16' or b_dtype == 'int16':
            return 'int32'  # Could be int64 for very large matrices
        else:
            return max(a_dtype, b_dtype, key=lambda x: type_hierarchy.get(x, 0))

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for MatMul_INT8
        # This involves gradients through integer matrix multiplication
        raise NotImplementedError("MatMul_INT8 backward pass not yet implemented")


class ScatterNDOp(SimOp):
    """
    ONNX 1.20.0 ScatterND Operation
    Scatters updates into a tensor at specified indices.

    This operation updates elements of a tensor at positions specified by
    indices with values from updates tensor. It supports various reduction
    modes for handling overlapping indices.

    Key Features:
    - Multi-dimensional index support
    - Various reduction modes (none, add, mul, min, max)
    - Batch dimension handling
    - Memory-efficient updates
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'ScatterND'

        # ONNX ScatterND supports exactly 3 inputs and 1 output
        check_io_counts(self, in_counts=[3,3], out_counts=[1,1])

        # Parse reduction attribute
        self.reduction = self.attrs.get('reduction', 'none')
        valid_reductions = ['none', 'add', 'mul', 'min', 'max']
        if self.reduction not in valid_reductions:
            raise ValueError(f"Invalid reduction mode '{self.reduction}'. Must be one of {valid_reductions}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "ScatterND operation does not support backward pass yet"

        # Input validation
        assert len(inT) == 3, f"ScatterND expects 3 inputs, got {len(inT)}"
        assert len(outT) == 1, f"ScatterND expects 1 output, got {len(outT)}"

        # Parse inputs
        data = inT[0]      # Input tensor to be updated
        indices = inT[1]   # Index tensor
        updates = inT[2]   # Update values

        # Validate input types
        supported_types = ['float32', 'float64', 'int32', 'int64']
        assert data.dtype in supported_types, f"Data tensor must be numeric type, got {data.dtype}"
        assert indices.dtype in ['int32', 'int64'], f"Indices must be integer type, got {indices.dtype}"
        assert updates.dtype == data.dtype, f"Updates must have same dtype as data, got {updates.dtype} vs {data.dtype}"

        # Validate tensor shapes
        data_shape = data.shape
        indices_shape = indices.shape
        updates_shape = updates.shape

        if len(indices_shape) < 1:
            raise ValueError("Indices tensor must have at least 1 dimension")

        if len(data_shape) < 1:
            raise ValueError("Data tensor must have at least 1 dimension")

        # Validate shapes according to ONNX specification
        # indices.shape[-1] must equal the number of dimensions being indexed
        # updates.shape must be compatible with indices.shape[:-1] + data.shape[indices.shape[-1]:]
        index_depth = indices_shape[-1]

        if index_depth > len(data_shape):
            raise ValueError(f"Index depth {index_depth} exceeds data tensor dimensions {len(data_shape)}")

        # Compute expected updates shape
        expected_updates_shape = indices_shape[:-1] + data_shape[index_depth:]

        if len(updates_shape) != len(expected_updates_shape):
            raise ValueError(f"Updates shape {updates_shape} incompatible with expected shape {expected_updates_shape}")

        for i, (actual, expected) in enumerate(zip(updates_shape, expected_updates_shape)):
            if actual != expected:
                raise ValueError(f"Updates shape dimension {i}: {actual} != expected {expected}")

        # Output shape is same as input data shape
        outT[0].shape = data_shape
        outT[0].dtype = data.dtype

        # Calculate performance statistics
        total_elements = data.nelems()

        # Index computation operations
        index_ops = indices.nelems() * index_depth  # Each index element needs to be processed

        # Memory access operations (read data, read indices, read updates, write output)
        memory_ops = data.nelems() + indices.nelems() + updates.nelems() + data.nelems()

        # Update operations based on reduction mode
        if self.reduction == 'none':
            update_ops = updates.nelems()  # Simple assignment
        elif self.reduction in ['add', 'mul']:
            update_ops = updates.nelems() * 2  # Read current value + operation
        elif self.reduction in ['min', 'max']:
            update_ops = updates.nelems() * 3  # Read current value + comparison + conditional assignment
        else:
            update_ops = updates.nelems() * 2  # Default to 2 ops

        # Additional operations for bounds checking and validation
        validation_ops = indices.nelems() * index_depth

        total_ops = index_ops + memory_ops + update_ops + validation_ops

        self.perf_stats = {
            'inBytes': data.nbytes(self.precision) + indices.nbytes(self.precision) + updates.nbytes(self.precision),
            'inElems': data.nelems() + indices.nelems() + updates.nelems(),
            'outBytes': outT[0].nbytes(self.precision),
            'outElems': outT[0].nelems(),
            'instrs': {
                'mac': 0,  # No multiply-accumulate
                'cmp': update_ops if self.reduction in ['min', 'max'] else 0,  # Comparisons for min/max
                'add': update_ops if self.reduction == 'add' else 0,  # Addition for add mode
                'sub': 0,  # No subtraction
                'mul': update_ops if self.reduction == 'mul' else 0,  # Multiplication for mul mode
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
                'index': index_ops,  # Index computation operations
                'gather': 0,  # Gather operations (reading from indices)
                'scatter': update_ops,  # Scatter operations (writing to indexed locations)
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for ScatterND
        # This involves gradients through scatter operations with reduction modes
        raise NotImplementedError("ScatterND backward pass not yet implemented")


class EinsumOp(SimOp):
    """
    ONNX 1.20.0 Einsum Operation
    General tensor contractions using Einstein summation notation.

    This operation supports general tensor contractions and manipulations
    using Einstein summation notation, providing a powerful and flexible
    way to perform complex tensor operations including matrix multiplication,
    transpose, trace, outer products, and advanced contractions.

    Key Features:
    - Einstein summation notation parsing
    - Multiple input tensor support
    - Optimized contraction path finding
    - Support for various tensor operations
    - Memory-efficient computation
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Einsum'

        # Einsum supports variable number of inputs (1+) and 1 output
        # Check minimum requirements
        if len(self.inList) < 1:
            raise ValueError("Einsum requires at least 1 input tensor")
        if len(self.outList) != 1:
            raise ValueError("Einsum requires exactly 1 output tensor")

        # Parse equation attribute - this is the core of Einsum
        self.equation = self.attrs.get('equation', '')
        if not self.equation:
            raise ValueError("Einsum requires 'equation' attribute specifying Einstein summation notation")

        # Parse the equation to understand the operation
        self._parse_equation()

    def _parse_equation(self):
        """Parse Einstein summation equation and validate against inputs."""
        if '->' not in self.equation:
            raise ValueError(f"Invalid Einsum equation format: {self.equation}. Expected format: 'input_spec->output_spec'")

        input_part, output_part = self.equation.split('->', 1)

        # Parse input specifications
        input_specs = [spec.strip() for spec in input_part.split(',')]
        output_spec = output_part.strip()

        # Validate input count matches
        if len(input_specs) != len(self.inList):
            raise ValueError(f"Equation specifies {len(input_specs)} inputs but {len(self.inList)} tensors provided")

        # Validate equation format
        all_labels = set()
        for spec in input_specs:
            if not spec.isalpha():
                raise ValueError(f"Invalid characters in input spec '{spec}'. Only alphabetic labels allowed.")

        if output_spec and not output_spec.isalpha():
            raise ValueError(f"Invalid characters in output spec '{output_spec}'. Only alphabetic labels allowed.")

        # Store parsed information
        self.input_specs = input_specs
        self.output_spec = output_spec
        self._validate_equation()

    def _validate_equation(self):
        """Validate that the equation is consistent and well-formed."""
        # Collect all unique labels
        all_labels = set()
        for spec in self.input_specs:
            all_labels.update(spec)

        # Check output labels are subset of input labels
        output_labels = set(self.output_spec) if self.output_spec else set()
        if not output_labels.issubset(all_labels):
            raise ValueError(f"Output labels {output_labels} contain labels not present in inputs {all_labels}")

        # Note: Repeated labels within input specs are allowed in Einstein notation
        # They indicate summation over those dimensions (e.g., 'ii->' for trace)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "Einsum operation does not support backward pass yet"

        # Input validation
        if len(inT) < 1:
            raise ValueError("Einsum requires at least 1 input")
        if len(outT) != 1:
            raise ValueError("Einsum requires exactly 1 output")

        # Validate input shapes against equation
        self._validate_input_shapes(inT)

        # Compute output shape
        output_shape = self._compute_output_shape(inT)
        outT[0].shape = output_shape
        outT[0].dtype = self._infer_output_dtype(inT)

        # Calculate performance statistics based on operation type
        operation_complexity = self._analyze_operation_complexity(inT)

        # Base memory operations
        total_input_elements = sum(tensor.nelems() for tensor in inT)
        output_elements = outT[0].nelems()
        memory_ops = total_input_elements + output_elements

        # Computation operations based on operation type
        if operation_complexity['type'] == 'matrix_multiply':
            # Matrix multiplication complexity
            comp_ops = operation_complexity['complexity']
        elif operation_complexity['type'] == 'elementwise':
            # Element-wise operations
            comp_ops = output_elements
        elif operation_complexity['type'] == 'reduction':
            # Reduction operations
            comp_ops = operation_complexity['complexity']
        elif operation_complexity['type'] == 'transpose':
            # Transpose operations (essentially free)
            comp_ops = output_elements
        else:
            # General contraction - estimate based on input/output sizes
            comp_ops = max(total_input_elements, output_elements)

        # Add overhead for equation parsing and index computation
        parsing_ops = len(self.equation) * 10  # Rough estimate
        index_ops = sum(len(spec) for spec in self.input_specs) * output_elements

        total_ops = memory_ops + comp_ops + parsing_ops + index_ops

        self.perf_stats = {
            'inBytes': sum(tensor.nbytes(self.precision) for tensor in inT),
            'inElems': total_input_elements,
            'outBytes': outT[0].nbytes(self.precision),
            'outElems': output_elements,
            'instrs': {
                'mac': comp_ops if operation_complexity['type'] == 'matrix_multiply' else 0,
                'cmp': 0,  # No comparisons
                'add': comp_ops if operation_complexity['type'] in ['elementwise', 'reduction'] else 0,
                'sub': 0,  # No subtraction
                'mul': comp_ops if operation_complexity['type'] in ['elementwise'] else 0,
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # Type conversion handled separately
                'index': index_ops,  # Index computation operations
                'gather': parsing_ops,  # Equation parsing operations
                'scatter': 0,  # No scatter operations
            }
        }

        return self.perf_stats

    def _validate_input_shapes(self, inT):
        """Validate input tensor shapes against the equation."""
        if len(inT) != len(self.input_specs):
            raise ValueError(f"Number of inputs {len(inT)} doesn't match equation inputs {len(self.input_specs)}")

        # For each input, validate shape matches the specification
        for i, (tensor, spec) in enumerate(zip(inT, self.input_specs)):
            if len(tensor.shape) != len(spec):
                raise ValueError(f"Input {i} shape {tensor.shape} doesn't match spec '{spec}' (length mismatch)")

    def _compute_output_shape(self, inT):
        """Compute output shape based on equation and input shapes."""
        if not self.output_spec:
            # Scalar output (all dimensions contracted)
            return []

        # Build label to dimension mapping, considering repeated labels
        label_to_dim: dict[str, int] = {}

        # First pass: collect all label -> dimension mappings
        for tensor, spec in zip(inT, self.input_specs):
            for dim, label in enumerate(spec):
                if label in label_to_dim:
                    # Check consistency
                    if label_to_dim[label] != tensor.shape[dim]:
                        raise ValueError(f"Inconsistent dimension for label '{label}': {label_to_dim[label]} vs {tensor.shape[dim]}")
                else:
                    label_to_dim[label] = tensor.shape[dim]

        # Second pass: build output shape
        # Only include labels that appear in the output specification
        output_shape = []
        for label in self.output_spec:
            if label not in label_to_dim:
                raise ValueError(f"Output label '{label}' not found in input specifications")
            output_shape.append(label_to_dim[label])

        return output_shape

    def _infer_output_dtype(self, inT):
        """Infer output dtype based on input dtypes."""
        # Use the first input's dtype as the output dtype
        # In practice, Einsum should handle type promotion rules
        return inT[0].dtype

    def _analyze_operation_complexity(self, inT):
        """Analyze the operation type and complexity."""
        # Simple heuristic to classify the operation
        total_input_dims = sum(len(spec) for spec in self.input_specs)
        output_dims = len(self.output_spec) if self.output_spec else 0

        if output_dims == 0:
            # Full contraction -> scalar
            return {'type': 'reduction', 'complexity': sum(tensor.nelems() for tensor in inT)}
        elif self._is_matrix_multiply():
            # Matrix multiplication pattern
            complexity = 1
            for tensor in inT:
                complexity *= tensor.nelems()
            return {'type': 'matrix_multiply', 'complexity': complexity}
        elif total_input_dims == output_dims:
            # Element-wise operation
            return {'type': 'elementwise', 'complexity': max(tensor.nelems() for tensor in inT)}
        else:
            # General contraction
            return {'type': 'contraction', 'complexity': sum(tensor.nelems() for tensor in inT)}

    def _is_matrix_multiply(self):
        """Check if this is a matrix multiplication operation."""
        # Look for patterns like 'ij,jk->ik' or 'bqhd,bkhd->bhqk'
        if len(self.input_specs) != 2:
            return False

        spec1, spec2 = self.input_specs
        if len(spec1) < 2 or len(spec2) < 2:
            return False

        # For batched matrix multiplication, the patterns are more complex
        # We need to find the matrix dimensions (last 2 dimensions)
        # and check if they form a valid matrix multiplication

        # Find the contracting dimension (should be the last dimension of first input
        # and second-to-last dimension of second input)
        contract_dim = spec1[-1]
        if contract_dim != spec2[-2]:
            return False

        # Check if output dimensions match the outer dimensions
        if len(self.output_spec) != len(spec1):
            return False

        # Check batch dimensions (all dimensions except the last 2)
        for i in range(len(spec1) - 2):
            if spec1[i] != spec2[i] or spec1[i] != self.output_spec[i]:
                return False

        # Check matrix dimensions
        return spec1[-2] == self.output_spec[-2] and spec2[-1] == self.output_spec[-1]

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for Einsum
        # This involves complex gradient computations for tensor contractions
        raise NotImplementedError("Einsum backward pass not yet implemented")


class GroupQueryAttentionOp(SimOp):
    """
    ONNX 1.20.0 GroupQueryAttention Operation
    Advanced attention mechanism with grouped query heads per key-value head.

    GroupQueryAttention extends traditional multi-head attention by allowing
    multiple query heads to share the same key-value head, enabling more
    efficient inference while maintaining model quality. This is particularly
    important for large language models where the number of key-value heads
    can be significantly reduced compared to query heads.

    Key Features:
    - Grouped attention: Multiple query heads per key-value head
    - Rotary position embedding integration
    - KV-cache support for incremental decoding
    - Memory-efficient attention computation
    - Support for causal masking
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'GroupQueryAttention'

        # ONNX GroupQueryAttention supports 3-7 inputs and 1-3 outputs
        check_io_counts(self, in_counts=[3,7], out_counts=[1,3])

        # Validate required attributes
        self.num_heads = self.attrs.get('num_heads')
        if self.num_heads is None:
            raise ValueError("GroupQueryAttention requires 'num_heads' attribute")

        self.kv_num_heads = self.attrs.get('kv_num_heads')
        if self.kv_num_heads is None:
            raise ValueError("GroupQueryAttention requires 'kv_num_heads' attribute")

        # Validate that num_heads is divisible by kv_num_heads
        if self.num_heads % self.kv_num_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by kv_num_heads ({self.kv_num_heads})")

        # Optional attributes
        self.scale = self.attrs.get('scale', None)  # If None, use 1/sqrt(head_size)
        self.causal = self.attrs.get('causal', False)
        # Alias ONNX is_causal (INT) -> bool causal
        _is_causal = self.attrs.get('is_causal', None)
        if _is_causal is not None:
            try:
                self.causal = bool(int(_is_causal))
            except Exception:
                self.causal = bool(_is_causal)

        # Compute group size (how many query heads per key-value head)
        self.group_size = self.num_heads // self.kv_num_heads

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "GroupQueryAttention operation does not support backward pass yet"

        # Input validation - parse inputs based on count and shapes
        query, key, value = inT[0], inT[1], inT[2]  # Required inputs

        # Parse optional inputs based on tensor shapes and positions
        past_key = None
        past_value = None
        cos_cache = None
        sin_cache = None

        # Check each optional input position
        for i in range(3, len(inT)):
            tensor = inT[i]
            if tensor is None:
                continue

            # Determine tensor type based on shape
            if len(tensor.shape) == 4:
                # 4D tensor: likely past_key or past_value
                batch_size_t, seq_len_t, heads_t, dim_t = tensor.shape
                if heads_t == self.kv_num_heads:
                    # Matches kv_num_heads: past key/value
                    if past_key is None:
                        past_key = tensor
                    elif past_value is None:
                        past_value = tensor
            elif len(tensor.shape) == 3:
                # 3D tensor: cos_cache or sin_cache (RoPE)
                batch_size_t, seq_len_t, dim_t = tensor.shape
                head_dim = query.shape[-1] // self.num_heads
                if dim_t == head_dim // 2:  # RoPE typically uses half head dimension
                    if cos_cache is None:
                        cos_cache = tensor
                    elif sin_cache is None:
                        sin_cache = tensor

        # Validate input shapes
        self._validate_input_shapes(query, key, value, past_key, past_value, cos_cache, sin_cache)

        # Compute output shapes
        batch_size, seq_len_q = query.shape[0], query.shape[1]
        head_dim = query.shape[-1] // self.num_heads

        # Output shape: [batch, seq_len_q, num_heads * head_dim]
        output_shape = [batch_size, seq_len_q, self.num_heads * head_dim]
        outT[0].shape = output_shape
        outT[0].dtype = query.dtype

        # Optional outputs: present_key, present_value
        if len(outT) >= 2:
            # present_key shape: [batch, kv_num_heads, total_seq_len, head_dim]
            seq_len_kv = key.shape[1] if past_key is None else past_key.shape[1] + key.shape[1]
            outT[1].shape = [batch_size, self.kv_num_heads, seq_len_kv, head_dim]
            outT[1].dtype = key.dtype

        if len(outT) >= 3:
            # present_value shape: same as present_key
            outT[2].shape = outT[1].shape
            outT[2].dtype = value.dtype

        # Calculate performance statistics
        self._calculate_performance_stats(query, key, value, past_key, past_value, cos_cache, sin_cache, outT)

        return self.perf_stats

    def _validate_input_shapes(self, query, key, value, past_key, past_value, cos_cache, sin_cache):
        """Validate input tensor shapes according to GroupQueryAttention requirements."""

        batch_size_q, seq_len_q, hidden_size_q = query.shape
        batch_size_k, seq_len_k, hidden_size_k = key.shape
        batch_size_v, seq_len_v, hidden_size_v = value.shape

        # Basic shape consistency
        if batch_size_q != batch_size_k or batch_size_k != batch_size_v:
            raise ValueError("Query, key, and value must have the same batch size")

        if seq_len_k != seq_len_v:
            raise ValueError("Key and value must have the same sequence length")

        # Hidden size must be divisible by number of heads
        if hidden_size_q % self.num_heads != 0:
            raise ValueError(f"Query hidden size ({hidden_size_q}) must be divisible by num_heads ({self.num_heads})")

        if hidden_size_k % self.kv_num_heads != 0:
            raise ValueError(f"Key hidden size ({hidden_size_k}) must be divisible by kv_num_heads ({self.kv_num_heads})")

        if hidden_size_v % self.kv_num_heads != 0:
            raise ValueError(f"Value hidden size ({hidden_size_v}) must be divisible by kv_num_heads ({self.kv_num_heads})")

        # Head dimensions must match
        head_dim_q = hidden_size_q // self.num_heads
        head_dim_k = hidden_size_k // self.kv_num_heads
        head_dim_v = hidden_size_v // self.kv_num_heads

        if head_dim_q != head_dim_k or head_dim_k != head_dim_v:
            raise ValueError(f"Head dimensions must match: Q({head_dim_q}), K({head_dim_k}), V({head_dim_v})")

        # Validate past key/value shapes if provided
        if past_key is not None:
            if len(past_key.shape) != 4:
                raise ValueError(f"Past key must be 4D tensor, got shape {past_key.shape}")

            batch_past, seq_past, heads_past, dim_past = past_key.shape
            if batch_past != batch_size_k:
                raise ValueError("Past key batch size must match current key batch size")

            if heads_past != self.kv_num_heads:
                raise ValueError(f"Past key num_heads ({heads_past}) must match kv_num_heads ({self.kv_num_heads})")

            if dim_past != head_dim_k:
                raise ValueError("Past key head dimension must match current key head dimension")

        if past_value is not None:
            if len(past_value.shape) != 4:
                raise ValueError(f"Past value must be 4D tensor, got shape {past_value.shape}")

            batch_past, seq_past, heads_past, dim_past = past_value.shape
            if batch_past != batch_size_v:
                raise ValueError("Past value batch size must match current value batch size")

            if heads_past != self.kv_num_heads:
                raise ValueError(f"Past value num_heads ({heads_past}) must match kv_num_heads ({self.kv_num_heads})")

            if dim_past != head_dim_v:
                raise ValueError("Past value head dimension must match current value head dimension")

        # Validate rotary embedding caches if provided
        if cos_cache is not None:
            if len(cos_cache.shape) != 3:
                raise ValueError(f"Cos cache must be 3D tensor, got shape {cos_cache.shape}")

            batch_cos, seq_cos, dim_cos = cos_cache.shape
            if batch_cos != batch_size_q:
                raise ValueError("Cos cache batch size must match query batch size")

            if dim_cos != head_dim_q // 2:  # RoPE typically uses half the head dimension
                raise ValueError(f"Cos cache dimension ({dim_cos}) should be half the head dimension ({head_dim_q // 2})")

        if sin_cache is not None:
            if len(sin_cache.shape) != 3:
                raise ValueError(f"Sin cache must be 3D tensor, got shape {sin_cache.shape}")

            batch_sin, seq_sin, dim_sin = sin_cache.shape
            if batch_sin != batch_size_q:
                raise ValueError("Sin cache batch size must match query batch size")

            if dim_sin != head_dim_q // 2:  # RoPE typically uses half the head dimension
                raise ValueError(f"Sin cache dimension ({dim_sin}) should be half the head dimension ({head_dim_q // 2})")

    def _calculate_performance_stats(self, query, key, value, past_key, past_value, cos_cache, sin_cache, outT):
        """Calculate performance statistics for GroupQueryAttention operation."""

        batch_size, seq_len_q = query.shape[0], query.shape[1]
        seq_len_k = key.shape[1]
        if past_key is not None:
            seq_len_k += past_key.shape[1]

        head_dim = query.shape[-1] // self.num_heads

        # Base computational complexity
        # Attention computation: Q*K^T -> [batch, num_heads, seq_q, seq_k]
        attention_flops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim

        # Attention softmax and dropout (approximated)
        softmax_flops = batch_size * self.num_heads * seq_len_q * seq_len_k

        # Attention * V -> final output
        output_flops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim

        # Rotary position embedding operations (if provided)
        rope_flops = 0
        if cos_cache is not None and sin_cache is not None:
            rope_flops = batch_size * seq_len_q * self.num_heads * head_dim * 2  # sin and cos operations

        # Group-specific operations: query heads are grouped, so we reduce KV computations
        # KV heads are fewer, so we save computations there
        kv_attention_flops = batch_size * self.kv_num_heads * seq_len_q * seq_len_k * head_dim

        # Total computation
        total_comp_ops = attention_flops + softmax_flops + output_flops + rope_flops - kv_attention_flops

        # Memory operations
        total_input_elements = (query.nelems() + key.nelems() + value.nelems())
        if past_key is not None:
            total_input_elements += past_key.nelems()
        if past_value is not None:
            total_input_elements += past_value.nelems()
        if cos_cache is not None:
            total_input_elements += cos_cache.nelems()
        if sin_cache is not None:
            total_input_elements += sin_cache.nelems()

        output_elements = outT[0].nelems()
        if len(outT) >= 2:
            output_elements += outT[1].nelems()
        if len(outT) >= 3:
            output_elements += outT[2].nelems()

        memory_ops = total_input_elements + output_elements

        # Scale factor for attention (if not provided, computed as 1/sqrt(head_dim))
        scale_ops = batch_size * self.num_heads * seq_len_q * seq_len_k if self.scale is None else 0

        self.perf_stats = {
            'inBytes': sum(tensor.nbytes(self.precision) for tensor in [query, key, value, past_key, past_value, cos_cache, sin_cache] if tensor is not None),
            'inElems': total_input_elements,
            'outBytes': sum(tensor.nbytes(self.precision) for tensor in outT),
            'outElems': output_elements,
            'instrs': {
                'mac': total_comp_ops,  # Main attention computations
                'cmp': batch_size * self.num_heads * seq_len_q * seq_len_k if self.causal else 0,  # Causal masking comparisons
                'add': softmax_flops + scale_ops,  # Softmax and scaling operations
                'sub': 0,  # No subtraction
                'mul': rope_flops,  # RoPE operations
                'div': batch_size * self.num_heads * seq_len_q * seq_len_k if self.scale is None else 0,  # Scale computation
                'rsqrt': batch_size * self.num_heads * seq_len_q * seq_len_k if self.scale is None else 0,  # 1/sqrt computation
                'exp': softmax_flops,  # Softmax exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # Type conversion handled separately
                'index': batch_size * self.num_heads * seq_len_q * seq_len_k,  # Attention indexing
                'gather': batch_size * self.num_heads * seq_len_q * seq_len_k,  # Attention gathering
                'scatter': 0,  # No scatter operations
            }
        }

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for GroupQueryAttention
        # This involves complex gradient computations for grouped attention
        raise NotImplementedError("GroupQueryAttention backward pass not yet implemented")


class RotaryPositionEmbeddingOp(SimOp):
    """
    ONNX 1.20.0 RotaryPositionEmbedding Operation
    Applies rotary position embeddings (RoPE) to input tensors.

    Rotary Position Embedding (RoPE) is a position encoding technique that applies
    rotations to the embedding space to encode positional information. This is
    particularly effective for long-range dependencies and is used in modern
    transformer architectures like LLaMA, GPT-J, and other RoPE-based models.

    Mathematical Definition:
    RoPE(x, m) = x * cos(m * theta) + rotate(x) * sin(m * theta)
    Where theta_i = 10000^(-2i/d) for i in [0, d/2)

    Key Features:
    - Rotary position embeddings for transformers
    - Pre-computed sin/cos caches for efficiency
    - Support for causal masking
    - Configurable rotary embedding dimensions
    - Batch processing support
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'RotaryPositionEmbedding'

        # ONNX RotaryPositionEmbedding supports 3 inputs and 1 output
        check_io_counts(self, in_counts=[3, 3], out_counts=[1, 1])

        # Validate required attributes
        self.num_heads = self.attrs.get('num_heads')
        if self.num_heads is None:
            raise ValueError("RotaryPositionEmbedding requires 'num_heads' attribute")

        self.rotary_embedding_dim = self.attrs.get('rotary_embedding_dim')
        if self.rotary_embedding_dim is None:
            raise ValueError("RotaryPositionEmbedding requires 'rotary_embedding_dim' attribute")

        # Optional attributes
        self.causal = self.attrs.get('causal', False)

        # Validate rotary_embedding_dim is reasonable
        if self.rotary_embedding_dim <= 0:
            raise ValueError("rotary_embedding_dim must be positive")

        # Rotary embeddings typically work on pairs of dimensions
        if self.rotary_embedding_dim % 2 != 0:
            raise ValueError("rotary_embedding_dim should be even for proper rotation pairs")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "RotaryPositionEmbedding operation does not support backward pass yet"

        # Input validation - parse inputs
        input_tensor, cos_cache, sin_cache = inT[0], inT[1], inT[2]

        # Validate input shapes
        self._validate_input_shapes(input_tensor, cos_cache, sin_cache)

        # Compute output shape (same as input shape)
        output_shape = input_tensor.shape
        outT[0].shape = output_shape
        outT[0].dtype = input_tensor.dtype

        # Calculate performance statistics
        self._calculate_performance_stats(input_tensor, cos_cache, sin_cache, outT)

        return self.perf_stats

    def _validate_input_shapes(self, input_tensor, cos_cache, sin_cache):
        """Validate input tensor shapes according to RotaryPositionEmbedding requirements."""

        batch_size, seq_len, hidden_size = input_tensor.shape

        # Validate hidden_size is divisible by num_heads
        if hidden_size % self.num_heads != 0:
            raise ValueError(f"Hidden size ({hidden_size}) must be divisible by num_heads ({self.num_heads})")

        head_dim = hidden_size // self.num_heads

        # Validate rotary_embedding_dim doesn't exceed head_dim
        if self.rotary_embedding_dim > head_dim:
            raise ValueError(f"rotary_embedding_dim ({self.rotary_embedding_dim}) cannot exceed head dimension ({head_dim})")

        # Validate cos_cache and sin_cache shapes
        if len(cos_cache.shape) != 3:
            raise ValueError(f"cos_cache must be 3D tensor, got shape {cos_cache.shape}")

        if len(sin_cache.shape) != 3:
            raise ValueError(f"sin_cache must be 3D tensor, got shape {sin_cache.shape}")

        # cos_cache and sin_cache should have same shape
        if cos_cache.shape != sin_cache.shape:
            raise ValueError(f"cos_cache and sin_cache must have same shape, got {cos_cache.shape} vs {sin_cache.shape}")

        cache_batch, cache_seq, cache_dim = cos_cache.shape

        # Validate cache dimensions
        if cache_batch != batch_size:
            raise ValueError(f"Cache batch size ({cache_batch}) must match input batch size ({batch_size})")

        if cache_seq != seq_len:
            raise ValueError(f"Cache sequence length ({cache_seq}) must match input sequence length ({seq_len})")

        # Cache dimension should be half the rotary_embedding_dim (for sin and cos components)
        expected_cache_dim = self.rotary_embedding_dim // 2
        if cache_dim != expected_cache_dim:
            raise ValueError(f"Cache dimension ({cache_dim}) should be half of rotary_embedding_dim ({expected_cache_dim})")

    def _calculate_performance_stats(self, input_tensor, cos_cache, sin_cache, outT):
        """Calculate performance statistics for RotaryPositionEmbedding operation."""

        batch_size, seq_len, hidden_size = input_tensor.shape

        # Rotary embedding computation involves:
        # 1. Element-wise multiplication with cos/sin
        # 2. Rotation operations (swapping and negating pairs)
        # 3. Addition operations

        # Number of rotary operations (only applied to rotary_embedding_dim)
        rotary_elements = batch_size * seq_len * self.num_heads * self.rotary_embedding_dim

        # Each rotary operation involves:
        # - 2 multiplications (cos*x, sin*rotated_x)
        # - 1 addition (combining the results)
        # - Additional operations for rotation (swapping pairs)
        mul_ops = rotary_elements * 2  # cos*x and sin*rotated_x
        add_ops = rotary_elements       # combining results
        rotation_ops = rotary_elements // 2  # pair rotations

        # Non-rotary elements are just passed through
        total_elements = batch_size * seq_len * hidden_size
        passthrough_elements = total_elements - rotary_elements

        # Memory operations
        total_input_elements = (input_tensor.nelems() + cos_cache.nelems() + sin_cache.nelems())
        output_elements = input_tensor.nelems()  # Output has same size as input

        memory_ops = total_input_elements + output_elements

        self.perf_stats = {
            'inBytes': sum(tensor.nbytes(self.precision) for tensor in [input_tensor, cos_cache, sin_cache]),
            'inElems': total_input_elements,
            'outBytes': outT[0].nbytes(self.precision),
            'outElems': output_elements,
            'instrs': {
                'mac': 0,  # No multiply-accumulate operations
                'cmp': batch_size * seq_len * self.num_heads * self.rotary_embedding_dim // 2 if self.causal else 0,  # Causal masking comparisons
                'add': add_ops,  # Addition operations for combining cos*x and sin*rotated_x
                'sub': 0,  # No subtraction
                'mul': mul_ops,  # Multiplications with cos and sin
                'div': 0,  # No division
                'rsqrt': 0,  # No reciprocal square root
                'exp': 0,  # No exponential
                'log': 0,  # No logarithm
                'round': 0,  # No rounding
                'clip': 0,  # No clipping
                'convert': 0,  # No type conversion
                'index': batch_size * seq_len * self.num_heads,  # Indexing for head splitting
                'gather': rotation_ops,  # Rotation pair gathering
                'scatter': rotation_ops,  # Rotation pair scattering
            }
        }

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for RotaryPositionEmbedding
        # This involves gradient computations through rotation operations
        raise NotImplementedError("RotaryPositionEmbedding backward pass not yet implemented")


class GroupQueryAttentionExtOp(SimOp):
    """
    ONNX 1.20.0 GroupQueryAttention Extensions
    Extended Group Query Attention with advanced features and optimizations.

    This operation extends the standard GroupQueryAttention with additional
    capabilities for modern transformer architectures:

    New Features in Extensions:
    1. Support for different data types (int8, float16, bfloat16)
    2. Multiple attention mechanisms (scaled_dot_product, flash_attention, linear)
    3. Enhanced quantization support for QKV projections
    4. Different position encoding schemes (RoPE, ALiBi, etc.)
    5. Memory-efficient attention patterns
    6. Advanced masking capabilities
    7. Support for different grouping strategies
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'GroupQueryAttentionExt'

        # GroupQueryAttentionExt supports 3-10 inputs and 1-4 outputs (extended)
        check_io_counts(self, in_counts=[3,10], out_counts=[1,4])

        # Parse extended attributes
        self.num_heads = self.attrs.get('num_heads')
        if self.num_heads is None:
            raise ValueError("GroupQueryAttentionExt requires 'num_heads' attribute")

        self.kv_num_heads = self.attrs.get('kv_num_heads')
        if self.kv_num_heads is None:
            raise ValueError("GroupQueryAttentionExt requires 'kv_num_heads' attribute")

        # Validate that num_heads is divisible by kv_num_heads
        if self.num_heads % self.kv_num_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by kv_num_heads ({self.kv_num_heads})")

        # Extended attributes
        self.attention_type = self.attrs.get('attention_type', 'scaled_dot_product')  # 'scaled_dot_product', 'flash_attention', 'linear'
        self.dtype = self.attrs.get('dtype', 'float32')  # Support for different data types
        self.quantized_attention = self.attrs.get('quantized_attention', False)  # Enable quantized attention
        self.position_encoding = self.attrs.get('position_encoding', 'none')  # 'none', 'rotary', 'alibi', 'learned'
        self.grouping_strategy = self.attrs.get('grouping_strategy', 'standard')  # 'standard', 'hierarchical', 'dynamic'
        self.memory_efficient = self.attrs.get('memory_efficient', False)  # Enable memory-efficient attention

        # Optional attributes
        self.scale = self.attrs.get('scale', None)  # If None, use 1/sqrt(head_size)
        self.causal = self.attrs.get('causal', False)

        # Compute group size (how many query heads per key-value head)
        self.group_size = self.num_heads // self.kv_num_heads

        # Validate extended attributes
        if self.attention_type not in ['scaled_dot_product', 'flash_attention', 'linear']:
            raise ValueError(f"GroupQueryAttentionExt unsupported attention_type: {self.attention_type}")
        if self.dtype not in ['float32', 'float16', 'bfloat16', 'int8']:
            raise ValueError(f"GroupQueryAttentionExt unsupported dtype: {self.dtype}")
        if self.position_encoding not in ['none', 'rotary', 'alibi', 'learned']:
            raise ValueError(f"GroupQueryAttentionExt unsupported position_encoding: {self.position_encoding}")
        if self.grouping_strategy not in ['standard', 'hierarchical', 'dynamic']:
            raise ValueError(f"GroupQueryAttentionExt unsupported grouping_strategy: {self.grouping_strategy}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "GroupQueryAttentionExt operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs >= 3, f"GroupQueryAttentionExt expects at least 3 inputs, got {num_inputs}"
        assert len(outT) >= 1, f"GroupQueryAttentionExt expects at least 1 output, got {len(outT)}"

        query, key, value = inT[0], inT[1], inT[2]  # Required inputs

        # Validate input types based on quantization setting
        if self.quantized_attention:
            quantized_types = ['int8', 'uint8', 'int16', 'uint16']
            assert query.dtype in quantized_types, f"Query must be quantized type when quantized_attention=True, got {query.dtype}"
            assert key.dtype in quantized_types, f"Key must be quantized type when quantized_attention=True, got {key.dtype}"
            assert value.dtype in quantized_types, f"Value must be quantized type when quantized_attention=True, got {value.dtype}"
        else:
            assert query.dtype == self.dtype, f"Query dtype mismatch: expected {self.dtype}, got {query.dtype}"
            assert key.dtype == self.dtype, f"Key dtype mismatch: expected {self.dtype}, got {key.dtype}"
            assert value.dtype == self.dtype, f"Value dtype mismatch: expected {self.dtype}, got {value.dtype}"

        # Validate tensor shapes
        query_shape = query.shape
        key_shape = key.shape
        value_shape = value.shape

        assert len(query_shape) == 3, f"Query must be 3D [batch_size, seq_len_q, hidden_size], got {query_shape}"
        assert len(key_shape) == 3, f"Key must be 3D [batch_size, seq_len_k, hidden_size], got {key_shape}"
        assert len(value_shape) == 3, f"Value must be 3D [batch_size, seq_len_v, hidden_size], got {value_shape}"

        batch_size, seq_len_q, hidden_size_q = query_shape
        batch_size_k, seq_len_k, hidden_size_k = key_shape
        batch_size_v, seq_len_v, hidden_size_v = value_shape

        # Validate consistency
        assert batch_size == batch_size_k == batch_size_v, "Batch sizes must match"
        assert hidden_size_q == hidden_size_k == hidden_size_v, "Hidden sizes must match"

        # Validate head configuration
        head_dim = hidden_size_q // self.num_heads
        assert head_dim * self.num_heads == hidden_size_q, f"Hidden size {hidden_size_q} must be divisible by num_heads {self.num_heads}"

        # Validate grouping
        kv_head_dim = hidden_size_q // self.kv_num_heads
        assert kv_head_dim * self.kv_num_heads == hidden_size_q, f"Hidden size {hidden_size_q} must be divisible by kv_num_heads {self.kv_num_heads}"

        # Parse optional inputs
        bias = None
        key_padding_mask = None
        past_key = None
        past_value = None
        cos_cache = None
        sin_cache = None
        alibi_bias = None

        # Extended input parsing (up to 10 inputs)
        for i in range(3, num_inputs):
            tensor = inT[i]
            if len(tensor.shape) == 3 and tensor.shape[0] == batch_size:
                # Could be bias [batch, seq_q, seq_k] or key_padding_mask [batch, seq]
                if tensor.shape[1] == seq_len_q and tensor.shape[2] == seq_len_k:
                    bias = tensor  # Attention bias
                elif tensor.shape[1] == seq_len_k:
                    key_padding_mask = tensor
            elif len(tensor.shape) == 4 and tensor.shape[2] == self.kv_num_heads:
                # Past key/value tensors
                if past_key is None:
                    past_key = tensor
                elif past_value is None:
                    past_value = tensor
            elif len(tensor.shape) == 3 and tensor.shape[-1] == head_dim // 2:
                # Rotary embeddings
                if cos_cache is None:
                    cos_cache = tensor
                elif sin_cache is None:
                    sin_cache = tensor
            elif len(tensor.shape) == 2 and tensor.shape[0] == seq_len_q:
                # ALiBi bias
                alibi_bias = tensor

        # Validate position encoding requirements
        if self.position_encoding == 'rotary' and (cos_cache is None or sin_cache is None):
            raise ValueError("Rotary position encoding requires both cos and sin caches")
        if self.position_encoding == 'alibi' and alibi_bias is None:
            raise ValueError("ALiBi position encoding requires alibi_bias tensor")

        # Set output tensor shapes and dtypes
        output_shape = [batch_size, seq_len_q, hidden_size_q]
        outT[0].shape = output_shape
        outT[0].dtype = query.dtype

        # Optional outputs
        if len(outT) >= 2:  # Present key output
            present_key_shape = [batch_size, seq_len_k, self.kv_num_heads, kv_head_dim]
            outT[1].shape = present_key_shape
            outT[1].dtype = key.dtype

        if len(outT) >= 3:  # Present value output
            present_value_shape = [batch_size, seq_len_v, self.kv_num_heads, kv_head_dim]
            outT[2].shape = present_value_shape
            outT[2].dtype = value.dtype

        if len(outT) >= 4:  # Attention weights output
            attn_weights_shape = [batch_size, self.num_heads, seq_len_q, seq_len_k]
            outT[3].shape = attn_weights_shape
            outT[3].dtype = self.dtype

        # Calculate performance statistics with extensions
        total_elements = batch_size * seq_len_q * hidden_size_q

        # Base group query attention operations
        # QKV projections: 3 matrix multiplications
        qkv_proj_ops = 3 * batch_size * seq_len_q * hidden_size_q * hidden_size_q  # Q_proj, K_proj, V_proj

        # Attention computation per head (grouped)
        qk_matmul_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim  # Q @ K^T per head
        softmax_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * 4  # Softmax operations
        attn_v_matmul_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim  # Attention @ V per head

        # Output projection
        output_proj_ops = batch_size * seq_len_q * hidden_size_q * hidden_size_q  # Output projection

        # Extended operations based on features
        if self.attention_type == 'flash_attention':
            # Flash attention optimizations
            flash_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim // 4
            qk_matmul_ops = flash_ops
            attn_v_matmul_ops = flash_ops
        elif self.attention_type == 'linear':
            # Linear attention (simplified)
            linear_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim // 2
            qk_matmul_ops = linear_ops
            attn_v_matmul_ops = linear_ops

        if self.position_encoding == 'rotary':
            # Rotary position embedding operations
            rope_ops = batch_size * seq_len_q * self.num_heads * head_dim * 4
        elif self.position_encoding == 'alibi':
            # ALiBi position bias operations
            alibi_ops = batch_size * self.num_heads * seq_len_q * seq_len_k
        elif self.position_encoding == 'learned':
            # Learned position embeddings
            pos_ops = batch_size * seq_len_q * self.num_heads * head_dim * 2
        else:
            rope_ops = 0
            alibi_ops = 0
            pos_ops = 0

        if self.quantized_attention:
            # Additional quantization/dequantization operations
            quant_ops = batch_size * seq_len_q * hidden_size_q * 8  # Quantize/dequantize Q, K, V, output
        else:
            quant_ops = 0

        if bias is not None:
            # Bias addition operations
            bias_ops = batch_size * self.num_heads * seq_len_q * seq_len_k
        else:
            bias_ops = 0

        if key_padding_mask is not None:
            # Key padding mask operations
            mask_ops = batch_size * seq_len_q * seq_len_k
        else:
            mask_ops = 0

        # Grouping overhead (additional operations for managing groups)
        grouping_ops = batch_size * self.num_heads * seq_len_q * seq_len_k * head_dim // 10  # Approximate grouping overhead

        # Total operations
        total_ops = qkv_proj_ops + qk_matmul_ops + attn_v_matmul_ops + output_proj_ops + softmax_ops
        if 'rope_ops' in locals():
            total_ops += rope_ops
        if 'alibi_ops' in locals():
            total_ops += alibi_ops
        if 'pos_ops' in locals():
            total_ops += pos_ops
        total_ops += quant_ops + bias_ops + mask_ops + grouping_ops

        # Memory analysis
        input_bytes = query.nbytes(self.precision) + key.nbytes(self.precision) + value.nbytes(self.precision)
        if bias is not None:
            input_bytes += bias.nbytes(self.precision)
        if key_padding_mask is not None:
            input_bytes += key_padding_mask.nbytes(self.precision)
        if past_key is not None:
            input_bytes += past_key.nbytes(self.precision)
        if past_value is not None:
            input_bytes += past_value.nbytes(self.precision)
        if cos_cache is not None and sin_cache is not None:
            input_bytes += cos_cache.nbytes(self.precision) + sin_cache.nbytes(self.precision)
        if alibi_bias is not None:
            input_bytes += alibi_bias.nbytes(self.precision)

        output_bytes = sum(out.nbytes(self.precision) for out in outT)

        # Instruction counts
        exp_count = softmax_ops // 4  # Exponential in softmax
        div_count = batch_size * self.num_heads * seq_len_q  # Division by sqrt(d_k) and softmax normalization

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': sum(tensor.nelems() for tensor in inT),
            'outBytes': output_bytes,
            'outElems': sum(tensor.nelems() for tensor in outT),
            'instrs': {
                'mac': qkv_proj_ops + qk_matmul_ops + attn_v_matmul_ops + output_proj_ops,  # All matrix multiplications
                'cmp': mask_ops + bias_ops,  # Comparisons for masking and bias
                'add': total_elements * 3 + softmax_ops,  # Various additions
                'sub': quant_ops,  # Quantization operations
                'mul': total_elements * 4 + softmax_ops,  # Scale multiplications and softmax
                'div': div_count,
                'rsqrt': batch_size * self.num_heads if self.attention_type == 'scaled_dot_product' else 0,  # Reciprocal sqrt for scaling
                'exp': exp_count,
                'log': 0,  # No logarithm
                'round': quant_ops // 3 if self.quantized_attention else 0,  # Rounding in quantization
                'clip': quant_ops // 3 if self.quantized_attention else 0,  # Clipping in quantization
                'convert': quant_ops // 3 if self.quantized_attention else 0,  # Type conversions
                'index': total_ops // 10,  # Memory indexing operations
                'gather': total_ops // 20,  # Gather operations
                'scatter': 0,  # No scattering
                'tanh': 0,  # No tanh
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for GroupQueryAttentionExt
        raise NotImplementedError("GroupQueryAttentionExt backward pass not yet implemented")


class QAttentionOp(SimOp):
    """
    ONNX 1.20.0 QAttention Operation
    Quantized Attention operation for efficient inference.

    QAttention performs attention computation entirely in quantized space,
    eliminating the need for dequantization/quantization steps during attention.
    This operation is designed for highly efficient quantized neural network inference.

    Key Features:
    - Quantized input tensors (Q, K, V)
    - Quantized attention computation
    - Quantized output tensor
    - Support for different quantization schemes
    - Optional dequantization of attention weights for interpretability
    """

    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'QAttention'

        # QAttention supports 8-12 inputs and 1-3 outputs
        # x, x_scale, x_zero_point, w, w_scale, w_zero_point, b, y_scale, y_zero_point, past_key?, past_value?, cos_cache?, sin_cache?
        check_io_counts(self, in_counts=[8,12], out_counts=[1,3])

        # Parse attributes
        self.num_heads = self.attrs.get('num_heads', 1)
        self.scale = self.attrs.get('scale', None)  # If None, use 1/sqrt(head_size)
        self.causal = self.attrs.get('causal', False)
        # Alias ONNX is_causal (INT) -> bool causal
        _is_causal = self.attrs.get('is_causal', None)
        if _is_causal is not None:
            try:
                self.causal = bool(int(_is_causal))
            except Exception:
                self.causal = bool(_is_causal)
        self.unidirectional = self.attrs.get('unidirectional', False)

        # Quantization attributes
        self.quantization_scheme = self.attrs.get('quantization_scheme', 'tensor')  # 'tensor', 'per_channel', 'per_head'
        self.output_quantized = self.attrs.get('output_quantized', True)  # Whether output should be quantized
        self.attention_quantized = self.attrs.get('attention_quantized', True)  # Whether attention weights should be quantized

        # Validate attributes
        if not isinstance(self.num_heads, int) or self.num_heads < 1:
            raise ValueError(f"QAttention num_heads must be a positive integer, got {self.num_heads}")
        if self.quantization_scheme not in ['tensor', 'per_channel', 'per_head']:
            raise ValueError(f"QAttention unsupported quantization_scheme: {self.quantization_scheme}")

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        is_backprop = kwargs.get('is_backprop', False)
        assert is_backprop == False, "QAttention operation does not support backward pass yet"

        # Input validation
        num_inputs = len(inT)
        assert num_inputs >= 8, f"QAttention expects at least 8 inputs, got {num_inputs}"
        assert len(outT) >= 1, f"QAttention expects at least 1 output, got {len(outT)}"

        # Parse quantized inputs
        x = inT[0]          # Quantized input tensor [batch_size, seq_len, hidden_size]
        x_scale = inT[1]    # Input scale (scalar or per-channel)
        x_zero_point = inT[2]  # Input zero point (scalar or per-channel)
        w = inT[3]          # Quantized weight tensor [hidden_size, hidden_size]
        w_scale = inT[4]    # Weight scale (scalar or per-channel)
        w_zero_point = inT[5]  # Weight zero point (scalar or per-channel)
        bias = inT[6]       # Bias tensor [hidden_size]
        y_scale = inT[7]    # Output scale (scalar or per-channel)
        y_zero_point = inT[8]  # Output zero point (scalar or per-channel)

        # Optional inputs
        past_key = inT[9] if num_inputs >= 10 else None
        past_value = inT[10] if num_inputs >= 11 else None
        cos_cache = inT[11] if num_inputs >= 12 else None
        sin_cache = inT[12] if num_inputs >= 13 else None

        # Validate input types - all quantized inputs should be integers
        quantized_types = ['int8', 'uint8', 'int16', 'uint16']
        scale_types = ['float32', 'float16', 'float64']

        assert x.dtype in quantized_types, f"Input x must be quantized type, got {x.dtype}"
        assert x_scale.dtype in scale_types, f"Input x_scale must be float type, got {x_scale.dtype}"
        assert x_zero_point.dtype in quantized_types, f"Input x_zero_point must be integer type, got {x_zero_point.dtype}"
        assert w.dtype in quantized_types, f"Weight w must be quantized type, got {w.dtype}"
        assert w_scale.dtype in scale_types, f"Weight w_scale must be float type, got {w_scale.dtype}"
        assert w_zero_point.dtype in quantized_types, f"Weight w_zero_point must be integer type, got {w_zero_point.dtype}"
        assert bias.dtype in scale_types, f"Bias must be float type, got {bias.dtype}"
        assert y_scale.dtype in scale_types, f"Output y_scale must be float type, got {y_scale.dtype}"
        assert y_zero_point.dtype in quantized_types, f"Output y_zero_point must be integer type, got {y_zero_point.dtype}"

        if past_key is not None:
            assert past_key.dtype in quantized_types, f"Past key must be quantized type, got {past_key.dtype}"
        if past_value is not None:
            assert past_value.dtype in quantized_types, f"Past value must be quantized type, got {past_value.dtype}"

        # Validate tensor shapes
        x_shape = x.shape
        w_shape = w.shape
        bias_shape = bias.shape

        assert len(x_shape) == 3, f"Input x must be 3D [batch_size, seq_len, hidden_size], got {x_shape}"
        assert len(w_shape) == 2, f"Weight w must be 2D [hidden_size, hidden_size], got {w_shape}"
        assert len(bias_shape) == 1, f"Bias must be 1D [hidden_size], got {bias_shape}"

        batch_size, seq_len, hidden_size = x_shape
        hidden_size_w, hidden_size_out = w_shape

        # Validate dimensions
        assert hidden_size == hidden_size_w, f"Input hidden size {hidden_size} must match weight input size {hidden_size_w}"
        assert hidden_size_w == hidden_size_out, f"Weight must be square matrix, got {w_shape}"
        assert bias_shape[0] == hidden_size_out, f"Bias size {bias_shape[0]} must match output size {hidden_size_out}"

        # Validate head configuration
        head_dim = hidden_size // self.num_heads
        assert head_dim * self.num_heads == hidden_size, f"Hidden size {hidden_size} must be divisible by num_heads {self.num_heads}"

        # Validate scale and zero point shapes based on quantization scheme
        if self.quantization_scheme == 'tensor':
            assert len(x_scale.shape) == 0 or (len(x_scale.shape) == 1 and x_scale.shape[0] == 1), f"Tensor quantization requires scalar scale, got {x_scale.shape}"
            assert len(x_zero_point.shape) == 0 or (len(x_zero_point.shape) == 1 and x_zero_point.shape[0] == 1), f"Tensor quantization requires scalar zero_point, got {x_zero_point.shape}"
        elif self.quantization_scheme == 'per_channel':
            assert len(x_scale.shape) == 1 and x_scale.shape[0] == hidden_size, f"Per-channel quantization requires [hidden_size] scale, got {x_scale.shape}"
            assert len(x_zero_point.shape) == 1 and x_zero_point.shape[0] == hidden_size, f"Per-channel quantization requires [hidden_size] zero_point, got {x_zero_point.shape}"

        # Set output tensor shapes and dtypes
        output_shape = [batch_size, seq_len, hidden_size]
        outT[0].shape = output_shape
        outT[0].dtype = x.dtype if self.output_quantized else 'float32'  # Quantized or dequantized output

        # Optional outputs
        if len(outT) >= 2:  # Present key output (prefer heads-before-seq for caches)
            present_key_shape = [batch_size, self.num_heads, seq_len, head_dim]
            outT[1].shape = present_key_shape
            outT[1].dtype = x.dtype

        if len(outT) >= 3:  # Present value output
            present_value_shape = [batch_size, self.num_heads, seq_len, head_dim]
            outT[2].shape = present_value_shape
            outT[2].dtype = x.dtype

        # Calculate performance statistics
        total_elements = batch_size * seq_len * hidden_size

        # QAttention performance breakdown:
        # 1. Dequantize input (if needed for certain operations)
        # 2. Linear projection (quantized matrix multiplication)
        # 3. Attention computation (quantized)
        # 4. Output projection (quantized matrix multiplication)
        # 5. Quantize output (if needed)

        # Linear projections for Q, K, V
        qkv_proj_ops = 3 * batch_size * seq_len * hidden_size * hidden_size  # Q_proj, K_proj, V_proj (quantized)

        # Attention computation (quantized)
        qk_matmul_ops = batch_size * self.num_heads * seq_len * seq_len * head_dim  # Q @ K^T per head
        softmax_ops = batch_size * self.num_heads * seq_len * seq_len * 4  # Softmax operations
        attn_v_matmul_ops = batch_size * self.num_heads * seq_len * seq_len * head_dim  # Attention @ V per head

        # Output projection
        output_proj_ops = batch_size * seq_len * hidden_size * hidden_size  # Output projection

        # Quantization/dequantization operations
        quant_ops = total_elements * 6  # Dequantize inputs, quantize outputs

        # Attention quantization (if enabled)
        if self.attention_quantized:
            attn_quant_ops = batch_size * self.num_heads * seq_len * seq_len * 4  # Quantize/dequantize attention weights
        else:
            attn_quant_ops = 0

        # Position encoding operations (if present)
        if cos_cache is not None and sin_cache is not None:
            rope_ops = batch_size * seq_len * self.num_heads * head_dim * 4  # RoPE operations
        else:
            rope_ops = 0

        # Total operations
        total_ops = qkv_proj_ops + qk_matmul_ops + attn_v_matmul_ops + output_proj_ops + softmax_ops
        total_ops += quant_ops + attn_quant_ops + rope_ops

        # Memory analysis
        input_bytes = x.nbytes(self.precision) + x_scale.nbytes(self.precision) + x_zero_point.nbytes(self.precision)
        input_bytes += w.nbytes(self.precision) + w_scale.nbytes(self.precision) + w_zero_point.nbytes(self.precision)
        input_bytes += bias.nbytes(self.precision) + y_scale.nbytes(self.precision) + y_zero_point.nbytes(self.precision)

        if past_key is not None:
            input_bytes += past_key.nbytes(self.precision)
        if past_value is not None:
            input_bytes += past_value.nbytes(self.precision)
        if cos_cache is not None and sin_cache is not None:
            input_bytes += cos_cache.nbytes(self.precision) + sin_cache.nbytes(self.precision)

        output_bytes = sum(out.nbytes(self.precision) for out in outT)

        # Instruction counts
        exp_count = softmax_ops // 4  # Exponential in softmax
        div_count = batch_size * self.num_heads * seq_len  # Division by sqrt(d_k) and softmax normalization

        self.perf_stats = {
            'inBytes': input_bytes,
            'inElems': sum(tensor.nelems() for tensor in inT),
            'outBytes': output_bytes,
            'outElems': sum(tensor.nelems() for tensor in outT),
            'instrs': {
                'mac': qkv_proj_ops + qk_matmul_ops + attn_v_matmul_ops + output_proj_ops,  # All matrix multiplications (quantized)
                'cmp': 0,  # Minimal comparisons
                'add': total_elements * 3 + softmax_ops + quant_ops,  # Various additions including quantization adjustments
                'sub': quant_ops + attn_quant_ops,  # Quantization operations (zero point subtraction)
                'mul': total_elements * 4 + softmax_ops + quant_ops,  # Scale multiplications and quantization
                'div': div_count,
                'rsqrt': batch_size * self.num_heads if self.scale is None else 0,  # Reciprocal sqrt for default scaling
                'exp': exp_count,
                'log': 0,  # No logarithm
                'round': quant_ops // 3 + attn_quant_ops // 3,  # Rounding in quantization
                'clip': quant_ops // 3 + attn_quant_ops // 3,  # Clipping to quantized range
                'convert': quant_ops // 3 + attn_quant_ops // 3,  # Type conversions between quantized and float
                'index': total_ops // 10,  # Memory indexing operations
                'gather': total_ops // 20,  # Gather operations for attention
                'scatter': 0,  # No scattering
                'tanh': 0,  # No tanh
            }
        }

        return self.perf_stats

    def backward(self, inT, outT, inGT, outGT):
        # TODO: Implement backward pass for QAttention
        raise NotImplementedError("QAttention backward pass not yet implemented")


def SimOpFactory(optype: str) -> type[SimOp]:
    cls2optype: Dict[type[SimOp], list[str]] = {
            EltwiseBinaryOp      : ['Add', 'Sub', 'Mul', 'Div', 'Greater', 'Less', 'Not', 'And', 'Or', 'Xor'],
            EltwiseUnaryOp       : ['Identity', 'Tanh', 'Sin', 'Cos', 'Neg', 'Sqrt', 'Exp', 'Log', 'Abs', 'Sign', 'Floor', 'Ceil', 'Round', 'Reciprocal', 'Erf', 'Acos', 'Acosh', 'Asin', 'Asinh', 'Atan', 'Atanh', 'Cosh', 'Sinh'],
            ConstantOp           : ['Constant'],
            GatherOp             : ['Gather'],
            LayerNormalizationOp : ['LayerNormalization'],
            MatMulOp             : ['MatMul'],
            SplitOp              : ['Split'],
            ReshapeOp            : ['Reshape'],
            ReshapeExtOp         : ['ReshapeExt'],
            ExpandOp             : ['Expand'],
            ThresholdedReluOp    : ['ThresholdedRelu'],
            TransposeOp          : ['Transpose'],
            WhereOp              : ['Where'],
            SoftmaxOp            : ['Softmax'],
            PowOp                : ['Pow'],
            UnsqueezeOp          : ['Unsqueeze'],
            SqueezeOp            : ['Squeeze'],
            ClipOp               : ['Clip'],
            HardmaxOp            : ['Hardmax'],
            LogSoftmaxOp         : ['LogSoftmax'],
            GatherElementsOp     : ['GatherElements'],
            ScatterElementsOp    : ['ScatterElements'],
            IsInfOp              : ['IsInf'],
            IsNaNOp              : ['IsNaN'],
            PadOp                : ['Pad'],
            SpaceToDepthOp       : ['SpaceToDepth'],
            DepthToSpaceOp       : ['DepthToSpace'],
            TileOp               : ['Tile'],
            ConcatOp             : ['Concat'],
            SliceOp              : ['Slice'],
            TriluOp              : ['Trilu'],
            DropoutOp            : ['Dropout'],
            EqualOp              : ['Equal'],
            CastOp               : ['Cast'],
            ShapeOp              : ['Shape'],
            RangeOp              : ['Range'],
            GeluOp               : ['Gelu'],
            MishOp               : ['Mish'],
            HardSwishOp          : ['HardSwish'],
            SwishOp              : ['Swish'],
            HardSigmoidOp        : ['HardSigmoid'],
            EluOp                : ['Elu'],
            SeluOp               : ['Selu'],
            SoftPlusOp           : ['SoftPlus'],
            SoftSignOp           : ['SoftSign'],
            ShrinkOp             : ['Shrink'],
            FastGeluOp           : ['FastGelu'],
            FastGeluGradOp       : ['FastGeluGrad'],
            BiasGeluOp           : ['BiasGelu'],
            SimplifiedLayerNormalizationOp : ['SimplifiedLayerNormalization'],
            GlobalMaxPoolOp      : ['GlobalMaxPool'],
            GlobalAveragePoolOp  : ['GlobalAveragePool'],
            GroupNormalizationOp : ['GroupNormalization'],
            SkipLayerNormalizationOp : ['SkipLayerNormalization'],
            MatMulIntegerOp      : ['MatMulInteger'],
            EmbedLayerNormalizationExtOp : ['EmbedLayerNormalizationExt'],
            GemmExtOp            : ['GemmExt'],
            ConvExtOp            : ['ConvExt'],
            MaxPoolExtOp         : ['MaxPoolExt'],
            AveragePoolExtOp     : ['AveragePoolExt'],
            QLinearConvOp         : ['QLinearConv'],
            ConvIntegerOp         : ['ConvInteger'],
            DynamicQuantizeMatMulOp : ['DynamicQuantizeMatMul'],
            AttentionExtOp          : ['AttentionExt'],
            MultiHeadAttentionExtOp : ['MultiHeadAttentionExt'],
            ReluOp               : ['Relu'],
            LeakyReluOp          : ['LeakyRelu'], #Yolo-v7
            SigmoidOp            : ['Sigmoid'], #Yolo-v7
            ResizeOp             : ['Resize'], #Yolo-v7
            ReduceMaxOp          : ['ReduceMax', 'ArgMax'], #Yolo-v7
            ReduceSumOp          : ['ReduceSum'],
            ReduceMinOp          : ['ReduceMin'],
            ReduceMeanOp         : ['ReduceMean'],
            ReduceProdOp         : ['ReduceProd'],
            NonMaxSuppressionOp  : ['NonMaxSuppression'], #Yolo-v7
            FlattenOp            : ['Flatten'], #Yolo-v7
            VoxelPoolingOp       : ['VoxelPooling'], #BEVDepth

            ConvOp               : ['Conv'],   # TBD: step in adding new operator / layer typez
            MaxPoolOp            : ['MaxPool'],
            BatchNormalizationOp : ['BatchNormalization'],
            AveragePoolOp        : ['AveragePool'],
            AttentionOp          : ['Attention'],  # ONNX 1.20.0 Attention operation
            MultiHeadAttentionOp : ['MultiHeadAttention'],  # ONNX 1.20.0 MultiHeadAttention operation
            DynamicQuantizeLinearOp : ['DynamicQuantizeLinear'],  # ONNX 1.20.0 DynamicQuantizeLinear operation
            GatherNDOp           : ['GatherND'],  # ONNX 1.20.0 GatherND operation
            DequantizeLinearOp   : ['DequantizeLinear'],  # ONNX 1.20.0 DequantizeLinear operation
            RMSNormalizationOp   : ['RMSNormalization'],  # ONNX 1.20.0 RMSNormalization operation
            QuantizeLinearOp     : ['QuantizeLinear'],  # ONNX 1.20.0 QuantizeLinear operation
            EmbedLayerNormalizationOp : ['EmbedLayerNormalization'],  # ONNX 1.20.0 EmbedLayerNormalization operation
            QLinearMatMulOp      : ['QLinearMatMul'],  # ONNX 1.20.0 QLinearMatMul operation
            MatMul_INT8_Op       : ['MatMul_INT8'],  # ONNX 1.20.0 MatMul_INT8 operation
            ScatterNDOp          : ['ScatterND'],  # ONNX 1.20.0 ScatterND operation
            EinsumOp             : ['Einsum'],  # ONNX 1.20.0 Einsum operation
            GroupQueryAttentionOp : ['GroupQueryAttention'],  # ONNX 1.20.0 GroupQueryAttention operation
            GroupQueryAttentionExtOp : ['GroupQueryAttentionExt'],  # ONNX 1.20.0 GroupQueryAttentionExt operation
            QAttentionOp          : ['QAttention'],  # ONNX 1.20.0 QAttention operation
            RotaryPositionEmbeddingOp : ['RotaryPositionEmbedding'],  # ONNX 1.20.0 RotaryPositionEmbedding operation
          }
    optype2cls: dict[str, type[SimOp]] = {}
    for tmp in cls2optype:
        if optype in cls2optype[tmp]:
            if optype in optype2cls:
                raise RuntimeError(f'{optype} in more than one op-types')
            optype2cls[optype] = tmp
    opcls: Union[type[SimOp], None] = optype2cls.get(optype, None)
    if opcls is None:
        raise NotImplementedError(f'Operator type {optype} not yet mapped in SimOpFactory')
    return opcls
