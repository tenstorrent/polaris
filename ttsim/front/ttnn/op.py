#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.op import SimOp
from .tensor import Tensor, DataType
from .memory import MemoryConfig

from enum import Enum, auto
from itertools import count
import numpy as np

class MathFidelity(Enum):
    LoFi  = auto()
    HiFi2 = auto()
    HiFi3 = auto()
    HiFi4 = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return MathFidelity[s]

    @property
    def cname(self)->str:
        return self.name.lower()

op_counter = count(start=1, step=1)
def generate_new_op_name(): return f"ttsim.ttnn.Op_{next(op_counter)}"

def single_output_immediate_op(optype, /, preprocess=None):

    def _impl(*args, **kwargs):

        if preprocess:
            args, kwargs = preprocess(args, kwargs)

        tensor_args = [x for x in args if isinstance(x, Tensor)]
        devchk_list = [x.device for x in tensor_args]
        device      = devchk_list[0]
        assert device and all(x == device for x in devchk_list), f"device check: {devchk_list}"

        op_name = generate_new_op_name()
        opinfo  = {'name': op_name, 'optype': optype, 'inList': [], 'attrs': kwargs}
        C       = Tensor(name=op_name + '.out',  op_out= [op_name], device=device)

        new_args = []
        for i,x in enumerate(args):
            if isinstance(x, Tensor):
                x.op_in.append(op_name)
                opinfo['inList'].append(x.name)
                new_args.append(x)
            elif isinstance(x, (int,float)):
                #print(f"FOUND not Tensor input in ttnn.op({optype}) : {type(x)}")
                if optype in ['Add', 'Sub', 'Mul']:
                    tmp = Tensor(name=f"{op_name}.in.{i}", shape=[], dtype=DataType.FLOAT32, device=device)
                    tmp.op_in.append(op_name)
                    opinfo['inList'].append(tmp.name)
                    new_args.append(tmp)
                else:
                    exit(0)
            else:
                assert False, f"Unknown input type in ttnn.op({optype}) : {type(x)}"
        opinfo['outList'] = [C.name]

        opobj      = SimOp(opinfo)
        perf_stats = opobj.get_perf_counts(new_args, [C])
        # print(f"{optype}:: {perf_stats}")
        opobj.update_tensor_counts(new_args, [C])

        device.add_op(opobj)

        return C

    return _impl

def argmax_pp(args_list, kwargs_dict):
    #translate attribs
    kwargs_dict['axis'] = kwargs_dict.get('dim', 0)
    if 'keepdim' in kwargs_dict:
        kwargs_dict['keepdims'] = 1 if kwargs_dict['keepdim'] else 0
    else:
        kwargs_dict['keepdims'] = 0
    return args_list, kwargs_dict

def reshape_pp(args_list, kwargs_dict):
    assert len(args_list) <= 3, f"ttnn.reshape has 3 inputs (special case for TT h/w)"
    inT      = args_list[0]
    outShape = args_list[1]
    assert isinstance(inT, Tensor), f"ttnn.reshape 1st input should be a ttnn.Tensor"
    assert isinstance(outShape, (list, tuple)), f"ttnn.reshape 2nd input should be a list|tuple of ints"
    assert all(isinstance(x, int) for x in outShape), f"ttnn.reshape 2nd input should be a list|tuple of ints"

    if (len(args_list) == 3):
        # write code to get slice (batch 0) of the input tensor and return it
        inT = Tensor(shape=outShape, device=inT.device, dtype=DataType.from_numpy(inT.dtype))

    outData = np.array(outShape, dtype=np.int64)
    outT = Tensor(shape=outData.shape, dtype=DataType.INT64, device=inT.device, data=outData)
    return (inT, outT), kwargs_dict

def permute_pp(args_list, kwargs_dict):
    inT = args_list[0]
    assert isinstance(inT, Tensor), f"ttnn.permute 1st input should be a ttnn.Tensor"
    assert isinstance(args_list[1], (list, tuple)), f"ttnn.permute 2nd input should be a list|tuple of ints"
    kwargs_dict['perm'] = list(args_list[1])
    return (inT, ), kwargs_dict

def embedding_pp(args_list, kwargs_dict):
    # TTNN passes in the order indices, weights while Polaris takes weights, indices
    assert len(args_list) == 2, f"ttnn.embedding has 2 inputs"
    input_tensor  = args_list[0]
    weight_tensor = args_list[1]
    assert isinstance(input_tensor, Tensor),  f"ttnn.embedding 1st input should be a ttnn.Tensor: {input_tensor}"
    assert isinstance(weight_tensor, Tensor), f"ttnn.embedding 2nd input should be a ttnn.Tensor: {weight_tensor}"
    return (weight_tensor, input_tensor), kwargs_dict

def layer_norm_pp(args_list, kwargs_dict):
    input_tensor          = args_list[0]
    weight_tensor         = kwargs_dict['weight']                if 'weight'                in kwargs_dict else None
    bias_tensor           = kwargs_dict['bias']                  if 'bias'                  in kwargs_dict else None
    epsilon               = kwargs_dict['epsilon']               if 'epsilon'               in kwargs_dict else None
    memory_config         = kwargs_dict['memory_config']         if 'memory_config'         in kwargs_dict else None
    compute_kernel_config = kwargs_dict['compute_kernel_config'] if 'compute_kernel_config' in kwargs_dict else None

    assert isinstance(input_tensor, Tensor), f"ttnn.layer_norm 1st input should be a ttnn.Tensor"
    assert isinstance(weight_tensor, Tensor), f"ttnn.layer_norm 2nd input should be a ttnn.Tensor"
    if bias_tensor is not None:
        assert isinstance(bias_tensor, Tensor), f"ttnn.layer_norm 3rd input should be a ttnn.Tensor"

    kwargs_dict = {}
    if bias_tensor is not None:
        return (input_tensor, weight_tensor, bias_tensor), kwargs_dict
    else:
        return (input_tensor, weight_tensor), kwargs_dict

def conv2d_pp(args_list, kwargs_dict):
    input_tensor  = kwargs_dict['input_tensor']
    weight_tensor = kwargs_dict['weight_tensor']
    bias_tensor   = kwargs_dict['bias_tensor']
    padding_size  = kwargs_dict['padding'][0]
    pads = [padding_size for i in range(4)]
    kwargs_dict = {'pads': pads, 'kernel_shape': list(kwargs_dict['kernel_size'])}
    return (input_tensor, weight_tensor, bias_tensor), kwargs_dict

def outer_pp(args_list, kwargs_dict):
    """Preprocessor for outer product operation."""
    assert len(args_list) == 2, f"ttnn.outer has 2 inputs"
    tensor_a = args_list[0]
    tensor_b = args_list[1]
    assert isinstance(tensor_a, Tensor), f"ttnn.outer 1st input should be a ttnn.Tensor"
    assert isinstance(tensor_b, Tensor), f"ttnn.outer 2nd input should be a ttnn.Tensor"

    # Validate that inputs are 1D tensors
    if len(tensor_a.shape) != 1:
        raise ValueError(f"ttnn.outer expects 1D tensors, got tensor_a with shape {tensor_a.shape}")
    if len(tensor_b.shape) != 1:
        raise ValueError(f"ttnn.outer expects 1D tensors, got tensor_b with shape {tensor_b.shape}")

    # Output shape will be (len(tensor_a), len(tensor_b))
    output_shape = [tensor_a.shape[0], tensor_b.shape[0]]
    kwargs_dict['output_shape'] = output_shape

    return (tensor_a.unsqueeze(1), tensor_b.unsqueeze(0)), kwargs_dict

def torchgather_pp(args_list, kwargs_dict):
    """Preprocessor for torch gather operation. Torch Gather differs vs. ONNX Gather."""
    assert len(args_list) == 3, f"ttnn.gather has 3 inputs"
    input_tensor = args_list[0]
    dim = args_list[1]
    index_tensor = args_list[2]

    # Ensure the index tensor is of integer type
    if index_tensor.dtype != np.int64:
        raise ValueError(f"ttnn.gather expects index tensor to be of type int64, got {index_tensor.dtype}")

    kwargs_dict['axis'] = dim
    return (input_tensor, index_tensor), kwargs_dict

def transpose_pp(args_list, kwargs_dict):
    inT = args_list[0]
    dim1 = args_list[1]
    dim2 = args_list[2]
    out_dims = [i for i in range(len(inT.shape))]
    out_dims[dim2] = dim1
    out_dims[dim1] = dim2
    kwargs_dict = {'perm': out_dims}
    return ([inT], kwargs_dict)

def cat(tensors, dim=0):
    """Concatenate a list of tensors along a specified dimension."""
    if not tensors:
        raise ValueError("Input list of tensors is empty")

    first_tensor = tensors[0]
    # Handle negative dimension
    original_rank = len(first_tensor.shape)
    if dim < 0:
        dim += original_rank

    # Validate dimension
    if dim < 0 or dim >= original_rank:
        raise ValueError(f"Dimension {dim} is out of range for tensors of rank {original_rank}. "
                        f"Valid range is [-{original_rank}, {original_rank - 1}]")

    # Calculate new shape
    new_shape = list(first_tensor.shape)
    new_shape[dim] = sum(tensor.shape[dim] for tensor in tensors)

    # Create the concatenated tensor
    return Tensor(
        shape=new_shape,
        dtype=DataType.from_numpy(first_tensor.dtype.name),
        device=first_tensor.device
    )

def where_pp(args, kwargs_dict):
    mask_tensor = args[0]
    input_tensor = args[1]
    value_tensor = args[2]
    return (mask_tensor, input_tensor, value_tensor), kwargs_dict

def sharded_to_interleaved(input_tensor, memory_config=None):
    return input_tensor  # No actual conversion, just returning the input tensor

def rms_norm(input_tensor, weight_tensor=None, bias_tensor=None, epsilon=1e-6, memory_config=None, compute_kernel_config=None, dim=3072):
    # Compute RMS (using layernorm for now)
    rms = layer_norm(input_tensor, weight=weight_tensor, epsilon=epsilon, axis=-1)
    normalized = div(input_tensor, rms)
    if weight_tensor is not None:
        weight_tensor = reshape(weight_tensor, (1, 1, 1, dim))
        if (normalized.shape[-1] != weight_tensor.shape[-1]):
            weight_tensor = weight_tensor.repeat((1, 1, 1, normalized.shape[-1]//dim)) ## only repeat if needed
        normalized = multiply(normalized, weight_tensor)

    if bias_tensor is not None:
        normalized = add(normalized, bias_tensor)
    return normalized

def max_pool2d_pp(args_list, kwargs_dict):
    input_tensor = kwargs_dict['input_tensor']
    kernel_size  = kwargs_dict['kernel_size']
    stride       = kwargs_dict.get('stride', kernel_size)
    padding      = kwargs_dict.get('padding', 0)
    dilation     = kwargs_dict.get('dilation', 1)
    ceil_mode    = kwargs_dict.get('ceil_mode', False)

    kwargs_dict = {
        'kernel_shape': list(kernel_size),
        'strides': list(stride),
        'pads': [padding[0], padding[0], padding[1], padding[1]], # [pad_top, pad_left, pad_bottom, pad_right]
        'dilations': list(dilation),
        'ceil_mode': ceil_mode
    }
    return (input_tensor,), kwargs_dict

def conv_transpose2d_pp(args_list, kwargs_dict):
    input_tensor  = kwargs_dict['input_tensor']
    weight_tensor = kwargs_dict['weight_tensor']
    bias_tensor   = kwargs_dict['bias_tensor']
    padding_size  = kwargs_dict['padding'][0]
    pads = [padding_size for i in range(4)]
    output_padding = kwargs_dict.get('output_padding', (0,0))
    strides = kwargs_dict.get('stride', (1,1))
    kwargs_dict = {
        'padding': pads,
        'kernel_size': list(kwargs_dict['kernel_size']),
        'output_padding': list(output_padding),
        'strides': list(strides)
    }
    return (input_tensor, weight_tensor, bias_tensor), kwargs_dict

class transformer:
    def __init__(self, config):
        pass

    def paged_scaled_dot_product_attention_decode(self, *args, **kwargs):
        pass

class experimental:
    def __init__(self):
        pass

    def all_gather_matmul(self, *args, **kwargs):
        pass

def all_gather(*args, **kwargs):
    assert "Not implemented yet!!"

#Pointwise Unary
cos         = single_output_immediate_op('Cos')
gelu        = single_output_immediate_op('Gelu')
identity    = single_output_immediate_op('Identity')
leaky_relu  = single_output_immediate_op('LeakyRelu')
neg         = single_output_immediate_op('Neg')
relu        = single_output_immediate_op('Relu')
sigmoid     = single_output_immediate_op('Sigmoid')
sin         = single_output_immediate_op('Sin')
softmax     = single_output_immediate_op('Softmax')
tanh        = single_output_immediate_op('Tanh')

#Pointwise Binary
add         = single_output_immediate_op('Add')
multiply    = single_output_immediate_op('Mul')
subtract    = single_output_immediate_op('Sub')
div         = single_output_immediate_op('Div')
pow         = single_output_immediate_op('Pow')

#Pointwise Ternary
where       = single_output_immediate_op('Where', preprocess=where_pp)

#Reduction
argmax      = single_output_immediate_op('ArgMax', preprocess=argmax_pp)

#Data Movement
concat      = single_output_immediate_op('Concat')
reshape     = single_output_immediate_op('Reshape',   preprocess=reshape_pp)
embedding   = single_output_immediate_op('Gather',    preprocess=embedding_pp)
permute     = single_output_immediate_op('Transpose', preprocess=permute_pp)
gather      = single_output_immediate_op('TorchGather', preprocess=torchgather_pp)
transpose   = single_output_immediate_op('Transpose', preprocess=transpose_pp)

#Normalization
layer_norm  = single_output_immediate_op('LayerNormalization', preprocess=layer_norm_pp)
batch_norm  = single_output_immediate_op('BatchNormalization')

#Convolution
conv2d      = single_output_immediate_op('Conv', preprocess=conv2d_pp)
conv_transpose2d = single_output_immediate_op('ConvTranspose', preprocess=conv_transpose2d_pp)

#Pooling
global_avg_pool2d = single_output_immediate_op('GlobalAveragePool')
max_pool2d        = single_output_immediate_op('MaxPool', preprocess=max_pool2d_pp)

#Matrix Multiplication
matmul      = single_output_immediate_op('MatMul')
outer       = single_output_immediate_op('MatMul', preprocess=outer_pp)


Tensor.__add__    = add       #type: ignore
Tensor.__sub__    = subtract  #type: ignore
Tensor.__mul__    = multiply  #type: ignore
Tensor.__div__    = div       #type: ignore
Tensor.__pow__    = pow       #type: ignore
Tensor.__matmul__ = matmul    #type: ignore
Tensor.reshape    = reshape   #type: ignore

#Mutli-operator functions
def linear(*args, **kwargs):
    assert len(args) == 2, f"linear args #-inputs({len(args)}) != 2"
    A, B        = args[0], args[1]
    bias        = kwargs.get('bias',                   None)
    act         = kwargs.get('activation',             None)
    #t_A         = kwargs.get('transpose_a',            False)
    #t_B         = kwargs.get('transpose_b',            False)
    dtype       = kwargs.get('dtype',                  None)
    #otile       = kwargs.get('output_tile',            None)
    #opt_otensor = kwargs.get('optional_output_tensor', None)
    core_grid   = kwargs.get('core_grid',              None)
    #mem_cfg     = kwargs.get('memory_config',          MemoryConfig.DRAM)
    #pgm_cfg     = kwargs.get('program_config',         None)
    ckrnl_cfg   = kwargs.get('compute_kernel_config',  None)

    not_impl_attrs = {
            'transpose_a'           : False,
            'transpose_b'           : False,
            #'dtype'                 : None,
            'output_tile'           : None,
            'optional_output_tensor': None,
            #'core_grid'             : None,
            #'memory_config'         : MemoryConfig.DRAM,
            'program_config'        : None,
            # 'compute_kernel_config' : None,
            }

    for aname,adefval in not_impl_attrs.items():
        if aname in kwargs:
            assert kwargs[aname] == adefval, f"linear.attrib: {aname} = {kwargs[aname]} not implemented yet!!"

    C = matmul(A, B)
    if bias is not None:
        C = add(C, bias)
    if act is not None:
        act_op = { 'relu': relu, 'gelu': gelu }[act]
        C = act_op(C)
    return C

# fold:
# takes an input tensor with shape (N, H, W, C) and transforms it to shape
# (N, H//stride_h, W//stride_w, C*stride_h*stride_w) by reshaping and permuting
# the spatial dimensions. This operation is commonly used as a preprocessing step
# for convolution operations, similar to the im2col operation in other deep learning
# frameworks, to reorganize input data in a format suitable for efficient matrix
# multiplication on Tenstorrent hardware.
def fold(ttnn_tensor_like,
         stride_h : int,
         stride_w : int,
         *,
         use_transpose_as_fold = False,
         output_shape = None, #ttnn.Shape
         pad_c : int = 0,
         pad_h : int = 0,
         pad_w : int = 0,
         grid_size = None, #ttnn.CoreRangeSet
         override_memory_config: MemoryConfig = None, #type: ignore
         ):

    assert ttnn_tensor_like.rank() == 4, f"fold input should be a rank-4 [N, H, W, C] tensor!!\n{ttnn_tensor_like}"
    N, H, W, C = ttnn_tensor_like.shape

    assert isinstance(stride_h, int) and stride_h > 0 and stride_h <= H, f"stride_h({stride_h}) should be in [0, {H}]"
    assert isinstance(stride_w, int) and stride_w > 0 and stride_w <= H, f"stride_w({stride_w}) should be in [0, {W}]"

    if pad_h > 0: H += pad_h
    if pad_w > 0: W += pad_w
    if pad_c > 0: C += pad_c

    Hs = H // stride_h
    Ws = W // stride_w

    if use_transpose_as_fold:
        #fold implemented as a series of reshape/transpose
        reshaped1  = ttnn_tensor_like.reshape(N, Hs, stride_h, Ws, stride_w, C)
        transposed = reshaped1.permute(0, 1, 3, 2, 4, 5)
        reshaped2  = transposed.reshape(N, Hs, Ws, C * stride_h * stride_w)
    else:
        #fold implemented as device specific efficient kernel: for DRAM/L1 memory configs
        #sharded tensor support: special handling for height sharded tensors
        reshaped2  = ttnn_tensor_like.reshape(N, Hs, Ws, C * stride_h * stride_w)

    return reshaped2
