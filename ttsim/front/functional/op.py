#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from functools import lru_cache, partial
from typing import Union, Iterator

import logging
from ttsim.graph import WorkloadGraph
from ttsim.ops import SimOpFactory, SimTensor
import ttsim.utils.common as common

#creates a tensor from shape/dtype specification
def _from_shape(name: str, shape: list[int], is_param: bool = False, is_const=False, np_dtype=np.float32) -> SimTensor:
    return SimTensor({
        'name' : name,
        'shape': shape,
        'dtype': np.dtype(np_dtype),
        'is_param': is_param,
        'is_const': is_const,
        })

#creates a tensor from data specification
def _from_data(name: str, data: Union[np.ndarray, np.floating, np.bool_], is_param: bool = False, is_const=False):
    return SimTensor({
        'name' : name,
        'shape': list(data.shape),
        'dtype': data.dtype,
        'data' : data,
        'resolve': '_',
        'is_param': is_param,
        'is_const': is_const,
        'op_in': [],
        'op_out': [],
        })

@lru_cache(maxsize=128)
def required_attrs(optype: str) -> list[str]:
    _tbl = {
        'Transpose': ['perm'],
        'Cast'     : ['to'],
        }
    return _tbl[optype] if optype in _tbl else []

def check_required_attrs(name, optype, attr_names, **kwargs):
    chktbl = {aname: aname in kwargs for aname in attr_names}
    chk = all([v for a,v in chktbl.items()])
    assert chk, f"{optype} op {name} requires attributes {attr_names}.\nattrs_present: {chktbl}"
    return

def get_output(name):
    return SimTensor({'name': name + '.out', 'op_out': [name]})

def get_opinfo(name, optype, **kwargs):
    return {'name': name, 'optype': optype, 'attrs': kwargs, 'domain': 'ttsim.common', 'inList': []}

def get_sim_op(opinfo):
    opcls = SimOpFactory(opinfo['optype'])
    opobj = opcls(opinfo)
    return opobj

#####################################################################################################
# SimOpHandle is a simple class to collect required information for a SimOp which includes input/output
# SimTensors as well as parameter SimTensors. Makes the functional interface implementation easy
# and proivdes an interface which mimics PyTorch Operators (poor-man's) for easy
# experimentation/tinkering
#####################################################################################################
class SimOpHandle:
    #for each parameter, we store the position in the input tensor list with the tensor
    # via (pos, tensor)
    # e.g. params = [(0,param1_tensor0), (3, param_tensor1), (6, param_tensor_2)]
    # then when we get the inputs in the __call__, we can create the extended
    # input list with params at the correct positions
    def __init__(self, name, optype, /, params, ipos, **kwargs):
        self.name       = name
        self.optype     = optype
        self.opinfo     = get_opinfo(name, optype, **kwargs)
        self.params     = params
        self.ipos       = ipos
        self.sim_op     = None
        self.otensor    = None
        self.perf_stats = None
        self.implicit_inputs = [] #needed for graph2onnx dump
        self.link_module = None
        check_required_attrs(name, optype, required_attrs(optype), **kwargs)

    def set_module(self, m):
        if self.link_module == None:
            self.link_module = m

    def __call__(self, *xargs):
        assert len(xargs) == len(self.ipos), \
                f"Length for inputs {len(xargs)} & ipos {len(self.ipos)} don't match"

        all_itensors = self.params + list(zip(self.ipos, xargs))
        sorted_all_itensors = sorted(all_itensors, key=lambda v: v[0])
        xinput = [x for _,x in sorted_all_itensors]

        #input tensor setup
        for x in xinput:
            x.op_in.append(self.name)
            self.opinfo['inList'].append(x.name)

        #output tensor setup
        self.otensor           = get_output(self.name)
        self.opinfo['outList'] = [self.otensor.name]

        #create relevant SimOp
        self.sim_op = get_sim_op(self.opinfo)

        #get perf stats for the SimOp -- this also ensures that the output tensor shape/data
        #is well formed
        self.perf_stats = self.sim_op.get_perf_counts(xinput,[self.otensor])
        self.sim_op.update_tensor_counts(xinput,[self.otensor])

        #return result
        if self.link_module is not None:
            if self.otensor not in self.link_module._tensors:
                self.link_module._tensors[self.otensor.name] = self.otensor
        return self.otensor

# SimOpHandle assumes only N inputs/params & 1 output
# Split has variadic outputs, need special handling
class SplitOpHandle:
    def __init__(self, name, /, count, **kwargs):
        self.name       = name
        self.optype     = 'Split'
        self.opinfo     = get_opinfo(name, 'Split', **kwargs)
        self.count      = count
        self.axis       = kwargs.get('axis', 0)
        self.params     = []
        self.implicit_inputs     = [] #needed for graph2onnx dump
        self.sim_op     = None
        self.otensors   = []
        self.perf_stats = None
        self.link_module = None
        check_required_attrs(name, 'Split', required_attrs('Split'), **kwargs)

    def set_module(self, m):
        self.link_module = m

    def __str__(self):
        s  = "SplitOpHandle:\n"
        s += f"    name       : {self.name      }\n"
        s += f"    optype     : {self.optype    }\n"
        s += f"    opinfo     :\n"
        for k,v in self.opinfo.items(): s += f"        {k:7s}: {v}\n"
        s += f"    count      : {self.count     }\n"
        s += f"    axis       : {self.axis      }\n"
        s += f"    sim_op     : {self.sim_op    }\n"
        s += f"    otensors   :\n"
        for ox in self.otensors: s += f"        {ox}\n"
        s += f"    perf_stats : {self.perf_stats}\n"
        return s

    def __call__(self, x):
        #ensure axis is within x.rank bounds
        if self.axis < 0:
            axis = x.rank() + self.axis
        elif self.axis >= x.rank():
            axis = x.rank() - 1
        else:
            axis = self.axis
        assert self.axis >=0 and self.axis < x.rank(), f"SplitOpHandle: axis={axis} should be in [0,{x.rank()})"

        out_dim = x.shape[axis] // self.count
        assert out_dim >= 1, f"SplitOpHandle: out_dim={out_dim} should be >=1"

        y = _from_data(self.name + '.in2', np.array([out_dim for _ in range(self.count)],
                                                    dtype=np.int64), is_param=False, is_const=True)
        self.implicit_inputs.append(y)

        #input tensor setup
        x.op_in.append(self.name)
        self.opinfo['inList'].append(x.name)
        y.op_in.append(self.name)
        self.opinfo['inList'].append(y.name)

        #output tensor setup
        self.otensors = [SimTensor({'name': self.name + "_" + str(i),
                                    'op_out': [self.name]}) for i in range(self.count)]
        self.opinfo['outList'] = [ot.name for ot in self.otensors]

        #create relevant SimOp
        self.sim_op = get_sim_op(self.opinfo)

        #get perf stats for the SimOp -- this also ensures that the output tensor shape/data
        #is well formed
        self.perf_stats = self.sim_op.get_perf_counts([x, y], self.otensors)
        self.sim_op.update_tensor_counts([x,y],self.otensors)

        #return result
        if self.link_module is not None:
            for x in self.otensors:
                if x not in self.link_module._tensors:
                    self.link_module._tensors[x.name] = x
        return tuple(self.otensors)

# VariadicInputOpHandle assumes only any number of inputs
# & 1 output; Also, there may be some constraints on the
# number of inputs, e.g., should be within a range
# Importantly, we don't allow any params to be specified!!
class VariadicInputOpHandle:
    def __init__(self, name, /, optype, input_range, **kwargs):
        assert len(input_range) == 2, f"input_range({input_range}) specification should be (min, max+1)!!"
        self.name        = name
        self.optype      = optype
        self.opinfo      = get_opinfo(name, optype, **kwargs)
        self.input_range = input_range
        self.sim_op      = None
        self.otensor     = None
        self.perf_stats  = None
        self.implicit_inputs = [] #needed for graph2onnx dump
        self.link_module = None
        check_required_attrs(name, optype, required_attrs(optype), **kwargs)

    def set_module(self, m):
        self.link_module = m

    def __call__(self, *xargs):
        min_in_val, max_in_val = self.input_range

        assert len(xargs) >= min_in_val and len(xargs) < max_in_val, \
                f"Length for inputs {len(xargs)} should be in range: [{min_in_val}, {max_in_val})"

        xinput = [x for x in xargs]
        #input tensor setup
        for x in xinput:
            x.op_in.append(self.name)
            self.opinfo['inList'].append(x.name)

        #output tensor setup
        self.otensor           = get_output(self.name)
        self.opinfo['outList'] = [self.otensor.name]

        #create relevant SimOp
        self.sim_op = get_sim_op(self.opinfo)

        #get perf stats for the SimOp -- this also ensures that the output tensor shape/data
        #is well formed
        self.perf_stats = self.sim_op.get_perf_counts(xinput,[self.otensor])
        self.sim_op.update_tensor_counts(xinput,[self.otensor])

        #return result
        if self.link_module is not None:
            if self.otensor not in self.link_module._tensors:
                self.link_module._tensors[self.otensor.name] = self.otensor
        return self.otensor

#sequence a list of layers...
# run as a pipeline:
# inputs -> lyr0 -> lyr1 -> .... -> lyrN-1 -> output
# implicitly name the intermediate tensors...
#
# Basic Implementation is :
#     lambda inList : functools.reduce(lambda x, f: f(x), lyr_list, inList)
# Additional Book-keeping required for tracking intermediate tensors etc.
#def SEQ(lyr_list):
#    def _run(inList):
#        val = inList
#        for lyr_num, lyr in enumerate(lyr_list):
#            print("sequence:", lyr_num)
#            print_layer(lyr)
#            print("layer input=", val)
#            val = lyr(val)
#            print("layer output=", val)
#        return val
#    return _run

class SimOpHandleList:
    # We don't allow SplitOpHandle or VariadicInputOpHandle in SimOpHandleList for now...
    # because how do we chain multiple inputs or outputs into the chain??
    def __init__(self, _ops):
        self._ops_in_list = {}
        assert len(_ops) > 0, f"Empty OpList at construction!!"

        for i, _op in enumerate(_ops):
            assert _op is not None, f"'None' _op passed to OpList"
            #assert isinstance(_op, (SimOpHandle, SplitOpHandle, VariadicInputOpHandle)), f"{_op} is not a SimOpHandle subclass"
            assert isinstance(_op, SimOpHandle), f"{_op} is not a SimOpHandle subclass"
            self._ops_in_list[str(i)] = _op

        #check all _op names in the list are unique...
        assert len(self) == len(set(self._ops_in_list)), \
                f"Op Names in OpList are not unique : {[o.name for o in self._ops_in_list.values()]}!!"

    def __len__(self):
        return len(self._ops_in_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._ops_in_list[str(i)] for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, int):
            idx = idx + len(self) if idx < 0 else idx
            if idx < 0 or idx >= len(self):
                raise IndexError(f'out-of-bound-index: {idx}')
            return self._ops_in_list[str(idx)]
        else:
            raise TypeError(f'Invalid index Type: {type(idx)}')

    def __iter__(self) -> Iterator[SimOpHandle]:
        for i in range(len(self)):
            yield self[i]

    #we want to make OpList Callable...
    def __call__(self, x):
        val = x
        for op_name, op_obj in self._ops_in_list.items():
            val = op_obj(val)
        return val

    #we want to make this immutable after construction...
    # so restricting setitem / delitem / append / insert / extend
    def __setitem__(self, idx, module):
        raise RuntimeError("OpList is immutable after construction")

    def __delitem__(self, idx):
        raise RuntimeError("OpList is immutable after construction")

    def append(self, module):
        raise RuntimeError("OpList is immutable after construction")

    def extend(self, modules):
        raise RuntimeError("OpList is immutable after construction")

    def insert(self, index, module):
        raise RuntimeError("OpList is immutable after construction")

######################################################################################################
# Operators With Implicit Parameters/Inputs
######################################################################################################
def Embedding(name, tbl_size, emb_dim, **kwargs):
    emb_wt = _from_shape(name + '.param', [tbl_size, emb_dim], is_param=True)
    emb_wt.op_in.append(name)
    op_hndl =  SimOpHandle(name, 'Gather', params=[(0,emb_wt)], ipos=[1], **kwargs)
    return op_hndl

def Bias(name, shape0, **kwargs):
    bias_term = _from_shape(name + '.bias', shape0, is_param=True)
    bias_term.op_in.append(name)
    op_hndl = SimOpHandle(name, 'Add', params=[(0,bias_term)], ipos=[1], **kwargs)
    return op_hndl

def MulFixed(name, dname, data0, **kwargs):
    mul_term = _from_data(name + '.' + dname, is_const=True, data=data0)
    mul_term.op_in.append(name)
    op_hndl = SimOpHandle(name, 'Mul', params=[(0,mul_term)], ipos=[1], **kwargs)
    return op_hndl

def ReshapeFixed(name, shape1, **kwargs):
    shape_term = _from_data(name + '.fixshape', is_const=True, data=np.array(shape1, dtype=np.int64))
    shape_term.op_in.append(name)
    op_hndl = SimOpHandle(name, 'Reshape', params=[(1,shape_term)], ipos=[0], **kwargs)
    return op_hndl

def Linear(name, nrow, ncol, **kwargs):
    mm_param = _from_shape(name + '.param', [nrow, ncol], is_param=True)
    mm_param.op_in.append(name)
    op_hndl =  SimOpHandle(name, 'MatMul', params=[(1,mm_param)], ipos=[0], **kwargs)
    return op_hndl

def Conv2d(name, in_channels, out_channels, kernel_size, **kwargs):
    kernel_dims = (kernel_size, kernel_size)
    arg_defaults = {
            'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1,
            'bias': True, 'padding_mode': 'zeros', 'device': None, 'dtype': None
            }
    eff_args   = common.get_kwargs_with_defaults('Conv', args=kwargs, default_args=arg_defaults)
    stride     = common.make_tuple(eff_args['stride'], 2)
    padding    = common.make_tuple(eff_args['padding'], 2*2)
    dilation   = common.make_tuple(eff_args['dilation'], 2)
    param_dims = [out_channels, in_channels // eff_args['groups'], *kernel_dims]
    conv_param = _from_shape(name+'.param', param_dims, is_param=True)
    # NOTE: 'bias' is a fixed argument, not kwarg for ONNX
    op_hndl = SimOpHandle(name, 'Conv', params=[(1, conv_param)], ipos=[0],
                          group=eff_args['groups'],  # Torch names this attr 'groups', ONNX names it 'group'
                          strides=stride,      # Torch / ONNX names differ
                          pads=padding,        # Torch / ONNX names differ
                          dilations=dilation,  # Torch / ONNX names differ
                          )
    return op_hndl

def MaxPool2d(name, kernel_size, **kwargs):
    arg_defaults = {
        'stride': None,
        'padding': 0,
        'dilation': 1,
        'return_indices': False,
        'ceil_mode': False,
    }
    eff_args   = common.get_kwargs_with_defaults('Maxpool', args=kwargs, default_args=arg_defaults)
    kernel_shape = common.make_tuple(kernel_size, 2)
    stride = eff_args['stride']
    if stride is None:
        # stride defaults to kernel_size and not 1!!
        stride = common.make_tuple(kernel_size, 2)
    else:
        stride = common.make_tuple(stride, 2)
    padding = common.make_tuple(eff_args['padding'], 2*2)
    dilation = common.make_tuple(eff_args['dilation'], 2)
    ceil_mode = eff_args['ceil_mode']
    op_hndl = SimOpHandle(name, 'MaxPool',
                          params=[], ipos=[0],
                          kernel_shape=kernel_shape,
                          pads=padding,                       # Torch/ONNX names differ
                          ceil_mode=1 if ceil_mode else 0,    # Torch/ONNX types differ
                          dilations=dilation,                 # Torch/ONNX names differ
                          strides=stride,                     # Torch/ONNX names differ
                          )
    return op_hndl


def Dropout(name, prob=0.5, train_mode=True, /, **kwargs):
    #SimTensor(/drop/Dropout_output_1) shape=[1, 7, 48], dtype=bool, op_in=[], op_out=['/drop/Dropout'], data=None
    # There are no trainable parameters for Dropout, 'prob' fixes the 'ratio' input1,
    # 'train_mode' fixes the 'training_mode' input2; So we fix in1, and in2 here...
    # We are abusing the name 'params' in the SimOpHandle constructor call below, eventually I will
    # rename it to something like params_or_inputs, but living with it for now.
    #Note:
    # Even if Dropout can potentially generate 2 outputs, the 2nd o/p being the mask,
    # I don't see the mask being used by any other operators downstream in real workloads,
    # so neglecting that for now
    ratio = _from_data(name + '.ratio', np.float32(prob), is_param=False, is_const=True)
    ratio.op_in.append(name)
    training_mode = _from_data(name + '.training_mode', np.bool_(train_mode), is_param=False, is_const=True)
    training_mode.op_in.append(name)
    op_hndl =  SimOpHandle(name, 'Dropout', params=[(1,ratio), (2,training_mode)], ipos=[0], **kwargs)
    return op_hndl


def LayerNorm(name, count, /, **kwargs):
    #Note:
    # ONNX LayerNorm can generate upto 3 outputs, but we are only generating 1
    # Ok for now, because simple LLMs behave the same way...
    scale = _from_shape(name + '.scale', [count], is_param=True)
    scale.op_in.append(name)
    bias = _from_shape(name + '.bias', [count], is_param=True)
    bias.op_in.append(name)
    op_hndl =  SimOpHandle(name, 'LayerNormalization', params=[(1,scale), (2,bias)], ipos=[0], **kwargs)
    return op_hndl

def BatchNorm2d(name, channels, /, **kwargs):
    # TODO:
    # ONNX BatchNorm can generate upto 3 outputs, but we are currently only generating 1
    # This implementation might need to be revised if 3-output batch norm is used in some network
    scale = _from_shape(name + '.scale', [channels], is_param=True)
    scale.op_in.append(name)
    bias = _from_shape(name + '.bias', [channels], is_param=True)
    bias.op_in.append(name)
    input_mean = _from_shape(name + '.input_mean', [channels], is_param=True)
    input_mean.op_in.append(name)
    input_var = _from_shape(name + '.input_var', [channels], is_param=True)
    input_var.op_in.append(name)
    op_hndl = SimOpHandle(name, 'BatchNormalization', params=[(1,scale), (2,bias),(3,input_mean), (4,input_var)], ipos=[0], **kwargs)
    return op_hndl

def AveragePool2d(name: str, kernel_shape: tuple[int, int], /, **kwargs):
    op_hndl = SimOpHandle(name, 'AveragePool', params=[], ipos=[0], **kwargs)
    return op_hndl

def Resize(name: str, /, scale_factor, **kwargs):
    roi     = _from_data(name + '.roi',    np.array([], dtype=np.float32), is_param=False, is_const=True)
    scales  = _from_data(name + '.scales', np.array([scale_factor, scale_factor], dtype=np.float32), is_param=False, is_const=True)
    op_hndl = SimOpHandle(name, 'Resize', params=[(1, roi), (2, scales)], ipos=[0], **kwargs)
    return op_hndl

######################################################################################################
# Simple Operator Mapping
######################################################################################################
def UniversalOperator(name, /, optype, params, ipos, **kwargs):
    return SimOpHandle(name, optype, params=params, ipos=ipos, **kwargs)

#Unary Operators
UnaryOperator = partial(UniversalOperator, params=[], ipos=[0])
Identity      = partial(UnaryOperator, optype='Identity')
Tanh          = partial(UnaryOperator, optype='Tanh')
Softmax       = partial(UnaryOperator, optype='Softmax')
Cast          = partial(UnaryOperator, optype='Cast')
Shape         = partial(UnaryOperator, optype='Shape')
Transpose     = partial(UnaryOperator, optype='Transpose')
Gelu          = partial(UnaryOperator, optype='Gelu')
Relu          = partial(UnaryOperator, optype='Relu')
LeakyReLU     = partial(UnaryOperator, optype='LeakyRelu')
Sigmoid       = partial(UnaryOperator, optype='Sigmoid')

#Binary Operators
BinaryOperator = partial(UniversalOperator, params=[], ipos=[0,1])
Add            = partial(BinaryOperator, optype='Add')
Mul            = partial(BinaryOperator, optype='Mul')
Gather         = partial(BinaryOperator, optype='Gather')
MatMul         = partial(BinaryOperator, optype='MatMul')
Reshape        = partial(BinaryOperator, optype='Reshape')
Pow            = partial(BinaryOperator, optype='Pow')
Unsqueeze      = partial(BinaryOperator, optype='Unsqueeze')
Equal          = partial(BinaryOperator, optype='Equal')

#Ternary Operators
TernaryOperator = partial(UniversalOperator, params=[], ipos=[0,1,2])
Where   = partial(TernaryOperator, optype='Where')
Range   = partial(TernaryOperator, optype='Range')

#Variadic Input Operator
#class VariadicInputOpHandle:
#    def __init__(self, name, optype, input_range, /, **kwargs):
ConcatX = partial(VariadicInputOpHandle, optype='Concat', input_range=(2,float('inf')))
TriluX  = partial(VariadicInputOpHandle, optype='Trilu',  input_range=(1,2))
SliceF  = partial(VariadicInputOpHandle, optype='Slice',  input_range=(3,6))
