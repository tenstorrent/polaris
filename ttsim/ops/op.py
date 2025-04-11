from functools import lru_cache, reduce
import operator
import math
import logging
import numpy as np
from typing import Union, TYPE_CHECKING, Dict, Any

from onnx.mapping import TENSOR_TYPE_MAP
import ttsim.utils.common as common
from .tensor import SimTensor

LOG   = logging.getLogger(__name__)
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
        DEBUG(f"Validated SimTensor({out_tensor.name}) SHAPE: {out_tensor.shape}")
        assert in_tensor.shape == out_tensor.shape, f"IO shape Mismatch {in_tensor.shape} != {out_tensor.shape} for {out_tensor.name}"
    else:
        DEBUG(f"Updating SimTensor({out_tensor.name}) SHAPE: {out_tensor.shape} <- {in_tensor.shape}")
        out_tensor.shape = in_tensor.shape

    if in_tensor.data is not None:
        if out_tensor.data is None:
            out_tensor.data = in_tensor.data
            out_tensor.dtype = in_tensor.dtype
            DEBUG(f"Updating DATA SimTensor({out_tensor})")

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
        return

    def update_tensor_counts(self, inT, outT, **kwargs):
        in_param_count  = sum([x.nelems() for x in inT if x.is_param == True])
        in_act_count    = sum([x.nelems() for x in inT if x.is_param == False])
        out_act_count   = sum([x.nelems() for x in outT if x.is_param == False])
        out_param_count = sum([x.nelems() for x in outT if x.is_param == True])
        assert out_param_count == 0, "OP{self.name} has output param count > 0: {out_param_count}"
        if TYPE_CHECKING:
            assert self.perf_stats is not None
        self.perf_stats.update({
            'inParamCount': in_param_count,
            'inActCount'  : in_act_count,
            'outActCount' : out_act_count,
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
        #find compute cycles
        self.compute_cycles = 0
        if TYPE_CHECKING:
            assert self.perf_stats is not None
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

######################  CONCRETE OP IMPLEMENTATION BEGIN ##################
class ConstantOp(SimOp):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.opclass_str: str = 'Constant'
        check_io_counts( self, in_counts=[0,0], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        assert False, f"{self.opclass_str} get_perf_stats not supported at present!!"

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
        self.perf_stats =  {
                'inElems' : 0,
                'outElems': outT[0].nelems(),
                'inBytes' : 0,
                'outBytes': outT[0].nbytes(),
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
        outT[0].shape = inT[0].shape
        outT[0].dtype = inT[0].dtype

        optype2instr = {'identity': 'mov'}
        instr_name = self.optype.lower()
        if instr_name in optype2instr:
            instr_name = optype2instr[instr_name]
        self.perf_stats =  {
                'inElems' : inT[0].nelems(),
                'outElems': outT[0].nelems(),
                'inBytes' : inT[0].nbytes(),
                'outBytes': outT[0].nbytes(),
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
                'inBytes' : inT[0].nbytes() + inT[1].nbytes(),
                'outBytes': outT[0].nbytes(),
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
                'inElems' : dataT.nelems(),
                'outElems': outT[0].nelems(),
                'inBytes' : dataT.nbytes(),
                'outBytes': outT[0].nbytes(),
                'instrs'  : {'mov': outT[0].nelems()}
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
        biasBytes   = 0 if biasT is None else biasT.nbytes()
        meanBytes   = 0 if len(outT) < 2 else outT[1].nbytes()
        invSDTBytes = 0 if len(outT) < 3 else outT[2].nbytes()
        self.perf_stats ={
                'inElems' : inT[0].nelems() + inT[1].nelems() + biasElems,
                'outElems': outT[0].nelems() + meanElems + invSDTElems,
                'inBytes' : inT[0].nbytes() + inT[1].nbytes() + biasBytes,
                'outBytes': outT[0].nbytes() + meanBytes + invSDTBytes,
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
            'inBytes' : sum([i.nbytes() for i in inT]),
            'outBytes': sum([o.nbytes() for o in outT]),
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
        instr_count     = { 'mac': total_macs }
        if len(inT) == 3:
            instr_count['add'] = output_elements

        inElems = X.nelems() + W.nelems() + B.nelems() if len(inT) == 3 else 0
        inBytes = X.nbytes() + W.nbytes() + B.nbytes() if len(inT) == 3 else 0

        self.perf_stats = {
            'inElems' : inElems,
            'outElems': outT[0].nelems(),
            'inBytes' : inBytes,
            'outBytes': outT[0].nbytes(),
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
            'inBytes' : inT[0].nbytes(),
            'outBytes': outT[0].nbytes(),
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
        input_shape   = inT[0].shape
        kernel_shape  = self.attrs.get('kernel_shape') #required attribute
        output_shape  = pooling_shape_inference(input_shape, kernel_shape, self.attrs)
        outT[0].shape = output_shape
        outT[0].dtype = inT[0].dtype

        instr_count = {'add': inT[0].nelems(), 'div': outT[0].nelems(), 'mov': outT[0].nelems()}

        self.perf_stats = {
            'inElems' : inT[0].nelems(),
            'outElems': outT[0].nelems(),
            'inBytes' : inT[0].nbytes(),
            'outBytes': outT[0].nbytes(),
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

        ######## New Implementation Begin #########
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
        ######## New Implementation Begin #########

        self.perf_stats = {
            'inElems' : inT[0].nelems() + inT[1].nelems(),
            'outElems': outT[0].nelems(),
            'inBytes' : inT[0].nelems() * inT[0].dtype.itemsize + inT[1].nelems() * inT[1].dtype.itemsize,
            'outBytes': outT[0].nelems() * outT[0].dtype.itemsize,
            'instrs'  : {'mac': outT[0].nelems() * reduced_dim}
            }
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
            outBytes += tout.nbytes()
            outElems += tout.nelems()

        self.perf_stats = {
                'inElems' : A.nelems() + 0 if splitT is None else splitT.nelems(),
                'outElems': outElems,
                'inBytes' : A.nbytes() + 0 if splitT is None else splitT.nbytes(),
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
                'inElems' : inT[0].nelems() + B.nelems(),
                'outElems': outT[0].nelems(),
                'inBytes' : inT[0].nbytes() + B.nbytes(),
                'outBytes': outT[0].nbytes(),
                'instrs'  : {'mov': outT[0].nelems()}
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
                'inBytes' : inT[0].nbytes(),
                'outBytes': outT[0].nbytes(),
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
        condB = clone_tensor_by_shape(inT[0], data_maybe_missing=False) #condB.data should exist
        X     = clone_tensor_by_shape(inT[1])
        Y     = clone_tensor_by_shape(inT[2])
        assert condB.dtype == np.bool_, f"Illegal Input Tensor Data Type: {condB}"
        np_out = np.where(condB.data, X.data, Y.data)
        tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])
        self.perf_stats = {
                'inElems' : condB.nelems() + X.nelems() + Y.nelems(),
                'outElems': outT[0].nelems(),
                'inBytes' : condB.nbytes() + X.nbytes() + Y.nbytes(),
                'outBytes': outT[0].nbytes(),
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
                'inBytes' : inT[0].nbytes(),
                'inElems' : inT[0].nelems(),
                'outBytes': outT[0].nbytes(),
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
        X = clone_tensor_by_shape(inT[0])
        Y = clone_tensor_by_shape(inT[1])
        np_out = pow(X.data, Y.data)
        tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])
        outElems = outT[0].nelems()
        self.perf_stats = {
                'inBytes' : X.nbytes() + Y.nbytes(),
                'outBytes': outT[0].nbytes(),
                'instrs'  : {
                    'mul': outElems,
                    'exp': outElems,
                    'log': outElems
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
        X = clone_tensor_by_shape(inT[0], data_maybe_missing=False) #X.data must be present
        Y = clone_tensor_by_shape(inT[1], data_maybe_missing=False) #Y.data must be present

        np_out = X.data
        for d in Y.data:
            np_out = np.expand_dims(np_out, axis=d)
        tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])
        self.perf_stats = {
                'inBytes' : X.nbytes() + Y.nbytes(),
                'outBytes': outT[0].nbytes(),
                'instrs'  : {'mov': outT[0].nelems()}
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
        Xs   = [clone_tensor_by_shape(x) for x in inT]
        #some sanity checks...
        chk_in_ranks = all([x.rank() == Xs[0].rank() for x in Xs])
        assert chk_in_ranks, f"Input Rank Mismatch: {[x.shape for x in Xs]}"

        np_x = [x.data for x in Xs]
        np_out = np.concatenate(np_x, axis)
        tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])
        inBytes = sum((x.nbytes() for x in Xs))

        # Placeholder: For Training, it may be required to output per-input tensor shape
        # Assumption: per-input tensor shape is a 1D-Tensor where each element
        # represents the length of the corresponding input along the axis
        #
        #out2_shape = [len(inT)]
        #out2_data  = [x.shape for x in Xs]
        self.perf_stats = {
                'inBytes' : inBytes,
                'outBytes': outT[0].nbytes(),
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
        dataT   = clone_tensor_by_shape(inT[0], data_maybe_missing=False) #dataT.data must be present
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

        slices = [slice(None)] *  dataT.rank()
        for s in range(startsT.rank()):
            s_axis  = axesT.data[s]
            s_start = startsT.data[s]
            s_end   = endsT.data[s]
            s_step  = stepsT.data[s]
            slices[s_axis] = slice(s_start, s_end, s_step)
        np_out = dataT.data[tuple(slices)]
        tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])

        inBytes = dataT.nbytes() + startsT.nbytes() + endsT.nbytes()
        inBytes += axesT.nbytes()  if len(inT) >= 4 else 0 #assume 4 bytes per axis spec
        inBytes += stepsT.nbytes() if len(inT) == 5 else 0 #assume 4 bytes per steps spec

        self.perf_stats = {
                'inBytes' : inBytes,
                'outBytes': outT[0].nbytes(),
                'instrs'  : {'mov': outT[0].nelems()}
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
                'inBytes' : X.nbytes(),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(),
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

        inBytes = X.nbytes()
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
            instr_count = {'nop': X.nelems()}
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

        outBytes = outT[0].nbytes()
        outBytes += outT[1].nbytes() if return_mask else 0
        outElems = outT[0].nbytes()
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
                'inBytes' : A.nbytes() + B.nbytes(),
                'outBytes': outT[0].nbytes(),
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
        saturate =  self.attrs.get('saturate', 1)
        to_type  =  self.attrs['to']
        A = clone_tensor_by_shape(inT[0], data_maybe_missing=False) #A.data must be present
        tensor_type = TENSOR_TYPE_MAP[to_type]
        np_out = A.data.astype(tensor_type.np_dtype)
        tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])
        self.perf_stats = {
                'inBytes' : A.nbytes(),
                'outBytes': outT[0].nbytes(),
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
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Gelu'
        check_io_counts( self, in_counts=[1,1], out_counts=[1,1] )

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats
        # ONNX opset-20 defines GELU w/ 2 variants controlled by the attribute 'approximate'
        #  if approximate = 'tanh', we use GELU (Gaussian Error Linear Unit) approximation:
        #     Y = 0.5 * X * (1 + tanh(math.sqrt(2 / math.pi) * (X + 0.044715 * pow(X, 3))))
        #  else (default)
        #     Y = 0.5 * X * (1 + erf(X/sqrt(2)))
        # we assume approximate to be 'tanh' always for now....
        # TODO: add default option as well...

        outT[0].shape = inT[0].shape
        outT[0].dtype = inT[0].dtype
        #instr count calc.
        # Y= <const> * X * ( <const> + tanh( <const> * ( X + <const> * X^3 ) ) )
        nElem = inT[0].nelems()
        mul_count, add_count, tanh_count = 0,0,0
        mul_count  += 2 * nElem # X^3
        mul_count  += nElem     # <const> * X^3
        add_count  += nElem     # X + <const> * X^3
        mul_count  += nElem     # <const> * ( X + ...)
        tanh_count += nElem     # tanh (...)
        add_count  += nElem     # <const + tanh(...)
        mul_count  += 2*nElem   # <const> * X * (...)
        instr = {'mul': mul_count, 'add': add_count, 'tanh': tanh_count}

        self.perf_stats = {
                'inElems' : inT[0].nelems(),
                'inBytes' : inT[0].nbytes(),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(),
                'instrs'  : instr
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
                'inBytes' : inT[0].nbytes(),
                'outElems': outT[0].nelems(),
                'outBytes': outT[0].nbytes(),
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
                'inBytes' : Y.nbytes(),
                'outElems': nElem,
                'outBytes': Y.nbytes(),
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

        nBytes = X.nbytes()
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
        inBytes = T.nbytes()
        if len(inT) == 2:
            axes = clone_tensor_by_shape(inT[1])
            inElems += axes.nelems()
            inBytes += axes.nbytes()
            np_out = np.sum(T.data, axis=tuple(axes.data.tolist()), keepdims=keepdims==1)
        else:
            np_out = np.sum(T.data, axis=None, keepdims=keepdims==1)

        print(np_out)

        tmp_outT = build_tmp_data_tensor(np_out, self.name + '__tmp_out__')
        update_output_tensor(self, tmp_outT, outT[0])
        outElems = outT[0].nelems()
        outBytes = outT[0].nbytes()

        self.perf_stats = {
                'inElems' : inElems,
                'inBytes' : inBytes,
                'outElems': outElems,
                'outBytes': outBytes,
                'instrs'  : {}
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
# 'FastGelu'                    
# 'FusedMatMul'                 
# #Grad Operators,
# 'InPlaceAccumulatorV2'        
# 'ConcatTraining'              
# 'DropoutGrad'                 
# 'FastGeluGrad'                
# 'SoftmaxGrad_13'              
# 'GatherGrad'                  
# 'SoftmaxCrossEntropyLossGrad'


@lru_cache(maxsize=128)
def SimOpFactory(optype: str) -> type[SimOp]:
    cls2optype: Dict[type[SimOp], list[str]] = {
            EltwiseBinaryOp      : ['Add', 'Mul'],
            EltwiseUnaryOp       : ['Identity', 'Tanh'],
            ConstantOp           : ['Constant'],
            GatherOp             : ['Gather'],
            LayerNormalizationOp : ['LayerNormalization'],
            MatMulOp             : ['MatMul'],
            SplitOp              : ['Split'],
            ReshapeOp            : ['Reshape'],
            TransposeOp          : ['Transpose'],
            WhereOp              : ['Where'],
            SoftmaxOp            : ['Softmax'],
            PowOp                : ['Pow'],
            UnsqueezeOp          : ['Unsqueeze'],
            ConcatOp             : ['Concat'],
            SliceOp              : ['Slice'],
            TriluOp              : ['Trilu'],
            DropoutOp            : ['Dropout'],
            EqualOp              : ['Equal'],
            CastOp               : ['Cast'],
            ShapeOp              : ['Shape'],
            RangeOp              : ['Range'],
            GeluOp               : ['Gelu'],
            ReluOp               : ['Relu'],
            ConvOp               : ['Conv'],   # TBD: step in adding new operator / layer typez
            MaxPoolOp            : ['MaxPool'],
            BatchNormalizationOp : ['BatchNormalization'],
            AveragePoolOp        : ['AveragePool', 'GlobalAveragePool'],
          }
    optype2cls: dict[str, type[SimOp]] = {}
    for tmp in cls2optype:
        if optype in cls2optype[tmp]:
            if optype in optype2cls:
                raise RuntimeError(f'{optype} in more than one op-types')
            optype2cls[optype] = tmp
    opcls: Union[type[SimOp], None] = optype2cls.get(optype, None)
    if opcls is None:
        raise RuntimeError(f'Operator type {optype} not yet mapped in SimOpFactory')
    return opcls
