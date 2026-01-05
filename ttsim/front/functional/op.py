#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from functools import lru_cache, partial
from typing import Union, Iterator

from loguru import logger
from ttsim.graph import WorkloadGraph
from ttsim.ops import SimOp, SimTensor
from ttsim.ops.tensor import require_shape_list
import ttsim.utils.common as common
from ttsim.config.wl2archmap import WL2ArchTypeSpec


# creates a tensor from shape/dtype specification
def _from_shape(
    name: str,
    shape: list[int],
    is_param: bool = False,
    is_const=False,
    np_dtype=np.float32,
) -> SimTensor:
    return SimTensor(
        {
            "name": name,
            "shape": shape,
            "dtype": np.dtype(np_dtype),
            "is_param": is_param,
            "is_const": is_const,
        }
    )


# creates a tensor from data specification
def _from_data(
    name: str,
    data: Union[np.ndarray, np.floating, np.bool_],
    is_param: bool = False,
    is_const=False,
):
    return SimTensor(
        {
            "name": name,
            "shape": list(data.shape),
            "dtype": data.dtype,
            "data": data,
            "resolve": "_",
            "is_param": is_param,
            "is_const": is_const,
            "op_in": [],
            "op_out": [],
        }
    )


# Helper functions for creating tensors (similar to PyTorch)
def ones(name: str, shape: list[int], dtype=np.float32) -> SimTensor:
    """Create a tensor filled with ones"""
    data = np.ones(shape, dtype=dtype)
    return _from_data(name, data, is_const=True)


def zeros(name: str, shape: list[int], dtype=np.float32) -> SimTensor:
    """Create a tensor filled with zeros"""
    data = np.zeros(shape, dtype=dtype)
    return _from_data(name, data, is_const=True)


def full(name: str, shape: list[int], fill_value: float, dtype=np.float32) -> SimTensor:
    """Create a tensor filled with a specific value"""
    data = np.full(shape, fill_value, dtype=dtype)
    return _from_data(name, data, is_const=True)


def full_like(name: str, tensor: SimTensor, fill_value: float) -> SimTensor:
    """Create a tensor filled with a specific value, with shape and dtype matching another tensor"""
    data = np.full(
        require_shape_list(tensor.shape, "full_like requires tensor.shape to be set"),
        fill_value,
        dtype=tensor.dtype.type,
    )
    return _from_data(name, data, is_const=True)


def as_tensor(
    name: str, data: Union[np.ndarray, list, tuple, float, int], dtype=None
) -> SimTensor:
    """Convert data to a SimTensor"""
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=dtype if dtype is not None else np.float32)
    elif dtype is not None:
        data = data.astype(dtype)
    return _from_data(name, data, is_const=True)


@lru_cache(maxsize=128)
def required_attrs(optype: str) -> list[str]:
    _tbl = {
        "Transpose": ["perm"],
        "Cast": ["to"],
    }
    return _tbl[optype] if optype in _tbl else []


def check_required_attrs(name, optype, attr_names, **kwargs):
    chktbl = {aname: aname in kwargs for aname in attr_names}
    chk = all([v for a, v in chktbl.items()])
    assert (
        chk
    ), f"{optype} op {name} requires attributes {attr_names}.\nattrs_present: {chktbl}"
    return


def get_output(name):
    return SimTensor({"name": name + ".out", "op_out": [name]})


def get_opinfo(name, optype, **kwargs):
    return {
        "name": name,
        "optype": optype,
        "attrs": kwargs,
        "domain": "ttsim.common",
        "inList": [],
    }


def get_sim_op(opinfo, default_dtype=None):
    optype: str = opinfo["optype"]
    opobj = SimOp(opinfo)
    if WL2ArchTypeSpec.has_instance():
        opobj.set_precision(WL2ArchTypeSpec.layer_2_datatype(optype.upper()))
    else:
        if default_dtype is None:
            raise AssertionError(
                f"Cannot determine data precision for {optype} as neither workload-arch map nor default precision is set. "
                "Consider setting a workload-arch map (WL2ArchTypeSpec) or providing a default precision parameter to resolve this issue."
            )
        opobj.set_precision(default_dtype)
    return opobj


#####################################################################################################
# SimOpHandle is a simple class to collect required information for a SimOp which includes input/output
# SimTensors as well as parameter SimTensors. Makes the functional interface implementation easy
# and proivdes an interface which mimics PyTorch Operators (poor-man's) for easy
# experimentation/tinkering
#####################################################################################################
class SimOpHandle:
    # for each parameter, we store the position in the input tensor list with the tensor
    # via (pos, tensor)
    # e.g. params = [(0,param1_tensor0), (3, param_tensor1), (6, param_tensor_2)]
    # then when we get the inputs in the __call__, we can create the extended
    # input list with params at the correct positions
    def __init__(self, name, optype, /, params, ipos, **kwargs):
        self.name = name
        self.optype = optype
        self.opinfo = get_opinfo(name, optype, **kwargs)
        self.params = params
        self.ipos = ipos
        self.sim_op = None
        self.otensor = None
        self.perf_stats = None
        self.implicit_inputs = []  # needed for graph2onnx dump
        self.link_module = None
        check_required_attrs(name, optype, required_attrs(optype), **kwargs)

    def set_module(self, m):
        if self.link_module == None:
            self.link_module = m

    def __call__(self, *xargs):
        assert len(xargs) == len(
            self.ipos
        ), f"Length for inputs {len(xargs)} & ipos {len(self.ipos)} don't match"

        all_itensors = self.params + list(zip(self.ipos, xargs))
        sorted_all_itensors = sorted(all_itensors, key=lambda v: v[0])
        xinput = [x for _, x in sorted_all_itensors]

        # input tensor setup
        # TODO: some ops are shared, i.e. they can be used multiple times for different
        # input tensors, and as such, their op_in gets appended several times...
        # how do we deal with this? Maybe in shared cases, op_in/op_out could be a list of
        # lists, with the last entry being the latest use???
        # At present, we need to create ops that CANNOT BE SHARED, because the assert on
        #  number or inputs/outputs being in range inside SimOp is set and the program crashes!!!
        for x in xinput:
            x.op_in.append(self.name)
            self.opinfo["inList"].append(x.name)

        # output tensor setup
        self.otensor = get_output(self.name)
        self.opinfo["outList"] = [self.otensor.name]

        # create relevant SimOp
        self.sim_op = get_sim_op(self.opinfo, default_dtype=x.dtype)

        # get perf stats for the SimOp -- this also ensures that the output tensor shape/data
        # is well formed
        self.perf_stats = self.sim_op.get_perf_counts(xinput, [self.otensor])
        self.sim_op.update_tensor_counts(xinput, [self.otensor])

        # return result
        if self.link_module is not None:
            self.otensor.link_module = self.link_module
            if self.otensor.name not in self.link_module._tensors:
                self.link_module._tensors[self.otensor.name] = self.otensor

        return self.otensor


# MultiOutputSimOpHandle mirrors SimOpHandle but materializes a list of
# output SimTensors, enabling ops like TopK that produce multiple values.
class MultiOutputSimOpHandle:
    def __init__(self, name, optype, /, params, ipos, num_outputs, **kwargs):
        self.name = name
        self.optype = optype
        self.opinfo = get_opinfo(name, optype, **kwargs)
        self.params = params
        self.ipos = ipos
        self.num_outputs = num_outputs
        self.sim_op = None
        self.otensors = []
        self.perf_stats = None
        self.implicit_inputs = []
        self.link_module = None
        check_required_attrs(name, optype, required_attrs(optype), **kwargs)

    def set_module(self, m):
        if self.link_module is None:
            self.link_module = m

    def __call__(self, *xargs):
        assert len(xargs) == len(
            self.ipos
        ), f"Length for inputs {len(xargs)} & ipos {len(self.ipos)} don't match"

        all_itensors = self.params + list(zip(self.ipos, xargs))
        sorted_all_itensors = sorted(all_itensors, key=lambda v: v[0])
        xinput = [x for _, x in sorted_all_itensors]

        for tensor in xinput:
            tensor.op_in.append(self.name)
            self.opinfo["inList"].append(tensor.name)

        self.otensors = [
            SimTensor({"name": f"{self.name}.out.{idx}", "op_out": [self.name]})
            for idx in range(self.num_outputs)
        ]
        self.opinfo["outList"] = [tensor.name for tensor in self.otensors]

        default_dtype = xinput[0].dtype if xinput else np.float32
        self.sim_op = get_sim_op(self.opinfo, default_dtype=default_dtype)

        self.perf_stats = self.sim_op.get_perf_counts(xinput, self.otensors)
        self.sim_op.update_tensor_counts(xinput, self.otensors)

        if self.link_module is not None:
            for tensor in self.otensors:
                tensor.link_module = self.link_module
                if tensor.name not in self.link_module._tensors:
                    self.link_module._tensors[tensor.name] = tensor
            for _, param_tensor in self.params:
                if param_tensor.name not in self.link_module._tensors:
                    self.link_module._tensors[param_tensor.name] = param_tensor

        return self.otensors


# SimOpHandle assumes only N inputs/params & 1 output
# Split has variadic outputs, need special handling
class SplitOpHandle:
    def __init__(self, name, /, count, **kwargs):
        self.name = name
        self.optype = "Split"
        self.opinfo = get_opinfo(name, "Split", **kwargs)
        self.count = count
        self.axis = kwargs.get("axis", 0)
        self.params = []
        self.implicit_inputs = []  # needed for graph2onnx dump
        self.sim_op = None
        self.otensors = []
        self.perf_stats = None
        self.link_module = None
        check_required_attrs(name, "Split", required_attrs("Split"), **kwargs)

    def set_module(self, m):
        self.link_module = m

    def __str__(self):
        s = "SplitOpHandle:\n"
        s += f"    name       : {self.name      }\n"
        s += f"    optype     : {self.optype    }\n"
        s += f"    opinfo     :\n"
        for k, v in self.opinfo.items():
            s += f"        {k:7s}: {v}\n"
        s += f"    count      : {self.count     }\n"
        s += f"    axis       : {self.axis      }\n"
        s += f"    sim_op     : {self.sim_op    }\n"
        s += f"    otensors   :\n"
        for ox in self.otensors:
            s += f"        {ox}\n"
        s += f"    perf_stats : {self.perf_stats}\n"
        return s

    def __call__(self, x, y=None):
        # ensure axis is within x.rank bounds
        if self.axis < 0:
            axis = x.rank() + self.axis
        elif self.axis >= x.rank():
            axis = x.rank() - 1
        else:
            axis = self.axis
        assert (
            self.axis >= 0 and self.axis < x.rank()
        ), f"SplitOpHandle: axis={axis} should be in [0,{x.rank()})"

        out_dim = x.shape[axis] // self.count
        assert out_dim >= 1, f"SplitOpHandle: out_dim={out_dim} should be >=1"

        if y is None:
            y = _from_data(
                self.name + ".in2",
                np.array([out_dim for _ in range(self.count)], dtype=np.int64),
                is_param=False,
                is_const=True,
            )
        self.implicit_inputs.append(y)

        # input tensor setup
        x.op_in.append(self.name)
        self.opinfo["inList"].append(x.name)
        y.op_in.append(self.name)
        self.opinfo["inList"].append(y.name)

        # output tensor setup
        self.otensors = [
            SimTensor({"name": self.name + "_" + str(i), "op_out": [self.name]})
            for i in range(self.count)
        ]
        self.opinfo["outList"] = [ot.name for ot in self.otensors]

        # create relevant SimOp
        self.sim_op = get_sim_op(self.opinfo, default_dtype=x.dtype)

        # get perf stats for the SimOp -- this also ensures that the output tensor shape/data
        # is well formed
        self.perf_stats = self.sim_op.get_perf_counts([x, y], self.otensors)
        self.sim_op.update_tensor_counts([x, y], self.otensors)

        if self.link_module is not None:
            for x in self.otensors:
                x.link_module = self.link_module
                if x not in self.link_module._tensors:
                    self.link_module._tensors[x.name] = x
        return tuple(self.otensors)


# VariadicInputOpHandle assumes only any number of inputs
# & 1 output; Also, there may be some constraints on the
# number of inputs, e.g., should be within a range
# Importantly, we don't allow any params to be specified!!
class VariadicInputOpHandle:
    def __init__(self, name, /, optype, input_range, **kwargs):
        assert (
            len(input_range) == 2
        ), f"input_range({input_range}) specification should be (min, max+1)!!"
        self.name = name
        self.optype = optype
        self.opinfo = get_opinfo(name, optype, **kwargs)
        self.input_range = input_range
        self.sim_op = None
        self.otensor = None
        self.perf_stats = None
        self.implicit_inputs = []  # needed for graph2onnx dump
        self.link_module = None
        check_required_attrs(name, optype, required_attrs(optype), **kwargs)

    def set_module(self, m):
        self.link_module = m

    def __call__(self, *xargs):
        min_in_val, max_in_val = self.input_range

        assert (
            len(xargs) >= min_in_val and len(xargs) < max_in_val
        ), f"Length for inputs {len(xargs)} should be in range: [{min_in_val}, {max_in_val})"

        xinput = [x for x in xargs]
        # input tensor setup
        for x in xinput:
            x.op_in.append(self.name)
            self.opinfo["inList"].append(x.name)

        # output tensor setup
        self.otensor = get_output(self.name)
        self.opinfo["outList"] = [self.otensor.name]

        # create relevant SimOp
        if xinput:
            default_dtype = xinput[0].dtype
        else:
            default_dtype = np.float32
        self.sim_op = get_sim_op(self.opinfo, default_dtype=default_dtype)

        # get perf stats for the SimOp -- this also ensures that the output tensor shape/data
        # is well formed
        self.perf_stats = self.sim_op.get_perf_counts(xinput, [self.otensor])
        self.sim_op.update_tensor_counts(xinput, [self.otensor])

        if self.link_module is not None:
            self.otensor.link_module = self.link_module
            if self.otensor not in self.link_module._tensors:
                self.link_module._tensors[self.otensor.name] = self.otensor
        return self.otensor


# sequence a list of layers...
# run as a pipeline:
# inputs -> lyr0 -> lyr1 -> .... -> lyrN-1 -> output
# implicitly name the intermediate tensors...
#
# Basic Implementation is :
#     lambda inList : functools.reduce(lambda x, f: f(x), lyr_list, inList)
# Additional Book-keeping required for tracking intermediate tensors etc.
# def SEQ(lyr_list):
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
            # assert isinstance(_op, (SimOpHandle, SplitOpHandle, VariadicInputOpHandle)), f"{_op} is not a SimOpHandle subclass"
            assert isinstance(_op, SimOpHandle), f"{_op} is not a SimOpHandle subclass"
            self._ops_in_list[str(i)] = _op

        # check all _op names in the list are unique...
        assert len(self) == len(
            set(self._ops_in_list)
        ), f"Op Names in OpList are not unique : {[o.name for o in self._ops_in_list.values()]}!!"

    def __len__(self):
        return len(self._ops_in_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._ops_in_list[str(i)] for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, int):
            idx = idx + len(self) if idx < 0 else idx
            if idx < 0 or idx >= len(self):
                raise IndexError(f"out-of-bound-index: {idx}")
            return self._ops_in_list[str(idx)]
        else:
            raise TypeError(f"Invalid index Type: {type(idx)}")

    def __iter__(self) -> Iterator[SimOpHandle]:
        for i in range(len(self)):
            yield self[i]

    # we want to make OpList Callable...
    def __call__(self, x):
        val = x
        for op_name, op_obj in self._ops_in_list.items():
            val = op_obj(val)
        return val

    # we want to make this immutable after construction...
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
    emb_wt = _from_shape(name + ".param", [tbl_size, emb_dim], is_param=True)
    emb_wt.op_in.append(name)
    op_hndl = SimOpHandle(name, "Gather", params=[(0, emb_wt)], ipos=[1], **kwargs)
    return op_hndl


def Bias(name, shape0, **kwargs):
    bias_term = _from_shape(name + ".bias", shape0, is_param=True)
    bias_term.op_in.append(name)
    op_hndl = SimOpHandle(name, "Add", params=[(0, bias_term)], ipos=[1], **kwargs)
    return op_hndl


def MulFixed(name, dname, data0, **kwargs):
    mul_term = _from_data(name + "." + dname, is_const=True, data=data0)
    mul_term.op_in.append(name)
    op_hndl = SimOpHandle(name, "Mul", params=[(0, mul_term)], ipos=[1], **kwargs)
    return op_hndl


def ReshapeFixed(name, shape1, **kwargs):
    shape_term = _from_data(
        name + ".fixshape", is_const=True, data=np.array(shape1, dtype=np.int64)
    )
    shape_term.op_in.append(name)
    op_hndl = SimOpHandle(name, "Reshape", params=[(1, shape_term)], ipos=[0], **kwargs)
    return op_hndl


def Linear(name, nrow, ncol, module=None, **kwargs):
    mm_param = _from_shape(name + ".param", [nrow, ncol], is_param=True)
    mm_param.op_in.append(name)
    op_hndl = SimOpHandle(name, "MatMul", params=[(1, mm_param)], ipos=[0], **kwargs)
    if module is not None:
        module._tensors[mm_param.name] = mm_param
    return op_hndl


def Conv2d(name, in_channels, out_channels, kernel_size, **kwargs):
    kernel_dims = (kernel_size, kernel_size)
    arg_defaults = {
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": "zeros",
        "device": None,
        "dtype": None,
    }
    eff_args = common.get_kwargs_with_defaults(
        "Conv", args=kwargs, default_args=arg_defaults
    )
    stride = common.make_tuple(eff_args["stride"], 2)
    padding = common.make_tuple(eff_args["padding"], 2 * 2)
    dilation = common.make_tuple(eff_args["dilation"], 2)
    param_dims = [out_channels, in_channels // eff_args["groups"], *kernel_dims]
    conv_param = _from_shape(name + ".param", param_dims, is_param=True)

    # Create bias tensor if enabled
    params_list = [(1, conv_param)]
    if eff_args["bias"]:
        bias_param = _from_shape(name + ".bias", [out_channels], is_param=True)
        params_list.append((2, bias_param))

    # NOTE: 'bias' is a fixed argument, not kwarg for ONNX
    op_hndl = SimOpHandle(
        name,
        "Conv",
        params=params_list,
        ipos=[0],
        group=eff_args[
            "groups"
        ],  # Torch names this attr 'groups', ONNX names it 'group'
        strides=stride,  # Torch / ONNX names differ
        pads=padding,  # Torch / ONNX names differ
        dilations=dilation,  # Torch / ONNX names differ
    )
    return op_hndl


def ConvTranspose2d(name, in_channels, out_channels, kernel_size=2, stride=2):
    kernel_dims = (kernel_size, kernel_size)
    stride = common.make_tuple(stride, 2)
    param_dims = [in_channels, out_channels, *kernel_dims]
    convt_param = _from_shape(name + ".param", param_dims, is_param=True)
    op_hndl = SimOpHandle(
        name,
        "ConvTranspose",
        params=[(1, convt_param)],
        ipos=[0],
        strides=stride,
    )
    return op_hndl


def Upsample(name, scale_factor, mode="nearest", align_corners=True):
    op_hndl = SimOpHandle(
        name,
        "Upsample",
        ipos=[0],
        params=[],
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )
    return op_hndl


def MaxPool2d(name, kernel_size, **kwargs):
    arg_defaults = {
        "stride": None,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
    }
    eff_args = common.get_kwargs_with_defaults(
        "Maxpool", args=kwargs, default_args=arg_defaults
    )
    kernel_shape = common.make_tuple(kernel_size, 2)
    stride = eff_args["stride"]
    if stride is None:
        # stride defaults to kernel_size and not 1!!
        stride = common.make_tuple(kernel_size, 2)
    else:
        stride = common.make_tuple(stride, 2)
    padding = common.make_tuple(eff_args["padding"], 2 * 2)
    dilation = common.make_tuple(eff_args["dilation"], 2)
    ceil_mode = eff_args["ceil_mode"]
    op_hndl = SimOpHandle(
        name,
        "MaxPool",
        params=[],
        ipos=[0],
        kernel_shape=kernel_shape,
        pads=padding,  # Torch/ONNX names differ
        ceil_mode=1 if ceil_mode else 0,  # Torch/ONNX types differ
        dilations=dilation,  # Torch/ONNX names differ
        strides=stride,  # Torch/ONNX names differ
    )
    return op_hndl


def Dropout(name, /, prob=0.5, train_mode=True, *, module=None, **kwargs):
    # SimTensor(/drop/Dropout_output_1) shape=[1, 7, 48], dtype=bool, op_in=[], op_out=['/drop/Dropout'], data=None
    # There are no trainable parameters for Dropout, 'prob' fixes the 'ratio' input1,
    # 'train_mode' fixes the 'training_mode' input2; So we fix in1, and in2 here...
    # We are abusing the name 'params' in the SimOpHandle constructor call below, eventually I will
    # rename it to something like params_or_inputs, but living with it for now.
    # Note:
    # Even if Dropout can potentially generate 2 outputs, the 2nd o/p being the mask,
    # I don't see the mask being used by any other operators downstream in real workloads,
    # so neglecting that for now
    ratio = _from_data(name + ".ratio", np.float32(prob), is_param=False, is_const=True)
    ratio.op_in.append(name)
    training_mode = _from_data(
        name + ".training_mode", np.bool_(train_mode), is_param=False, is_const=True
    )
    training_mode.op_in.append(name)
    if module is not None:
        module._tensors[ratio.name] = ratio
        module._tensors[training_mode.name] = training_mode
    # Store prob and train_mode in attrs so dropout_sinf can access them directly
    # Remove any existing prob/train_mode from kwargs to avoid conflicts
    kwargs.pop("prob", None)
    kwargs.pop("train_mode", None)
    op_hndl = SimOpHandle(
        name,
        "Dropout",
        params=[(1, ratio), (2, training_mode)],
        ipos=[0],
        prob=prob,
        train_mode=train_mode,
        **kwargs,
    )
    return op_hndl


def LayerNorm(name, count, /, **kwargs):
    # Note:
    # ONNX LayerNorm can generate upto 3 outputs, but we are only generating 1
    # Ok for now, because simple LLMs behave the same way...
    scale = _from_shape(name + ".scale", [count], is_param=True)
    scale.op_in.append(name)
    bias = _from_shape(name + ".bias", [count], is_param=True)
    bias.op_in.append(name)
    op_hndl = SimOpHandle(
        name, "LayerNormalization", params=[(1, scale), (2, bias)], ipos=[0], **kwargs
    )
    return op_hndl


def BatchNorm2d(name, channels, /, **kwargs):
    # TODO:
    # ONNX BatchNorm can generate upto 3 outputs, but we are currently only generating 1
    # This implementation might need to be revised if 3-output batch norm is used in some network
    scale = _from_shape(name + ".scale", [channels], is_param=True)
    scale.op_in.append(name)
    bias = _from_shape(name + ".bias", [channels], is_param=True)
    bias.op_in.append(name)
    input_mean = _from_shape(name + ".input_mean", [channels], is_param=True)
    input_mean.op_in.append(name)
    input_var = _from_shape(name + ".input_var", [channels], is_param=True)
    input_var.op_in.append(name)
    op_hndl = SimOpHandle(
        name,
        "BatchNormalization",
        params=[(1, scale), (2, bias), (3, input_mean), (4, input_var)],
        ipos=[0],
        **kwargs,
    )
    return op_hndl

def GroupNorm(name, num_groups, num_channels, /, **kwargs):
    weight = _from_shape(name + '.weight', [num_channels], is_param=True)
    weight.op_in.append(name)
    bias = _from_shape(name + '.bias', [num_channels], is_param=True)
    bias.op_in.append(name)
    op_hndl = SimOpHandle(name, 'GroupNormalization', params=[(1,weight), (2,bias)], ipos=[0], num_groups=num_groups, num_channels=num_channels, **kwargs)
    return op_hndl

def Resize(name: str, /, scale_factor, **kwargs):
    roi = _from_data(
        name + ".roi", np.array([], dtype=np.float32), is_param=False, is_const=True
    )
    if isinstance(scale_factor, (float, int)):
        scales = _from_data(
            name + ".scales",
            np.array([scale_factor, scale_factor], dtype=np.float32),
            is_param=False,
            is_const=True,
        )
    elif isinstance(scale_factor, (list, tuple)):
        assert (
            len(scale_factor) == 2
        ), f"Need to pass scale_factor list with 2 elems: {scale_factor}"
        assert isinstance(scale_factor[0], (float, int)) and isinstance(
            scale_factor[1], (float, int)
        ), f"scale_factor list should be of type: int/float"
        scales = _from_data(
            name + ".scales",
            np.array(scale_factor, dtype=np.float32),
            is_param=False,
            is_const=True,
        )
    else:
        assert False, f"Illegal scale_factor={scale_factor} input into F.Resize"

    # Translate PyTorch-style kwargs to ONNX-compatible Resize attributes
    if 'align_corners' in kwargs:
        ac = kwargs.pop('align_corners')
        kwargs['coordinate_transformation_mode'] = 'align_corners' if ac else 'half_pixel'
    if 'mode' in kwargs:
        m = kwargs['mode']
        if m == 'bilinear':
            kwargs['mode'] = 'linear'
        # 'nearest' and 'cubic' are already ONNX-compatible
    op_hndl = SimOpHandle(
        name, "Resize", params=[(1, roi), (2, scales)], ipos=[0], **kwargs
    )
    return op_hndl


def Split(name, **kwargs):
    return SplitOpHandle(name, ipos=[0, 1], **kwargs)


def AdaptiveAvgPool1d(name, adaptive=True, output_size=1, **kwargs):
    op_hndl = SimOpHandle(
        name,
        "AveragePool",
        params=[],
        ipos=[0],
        adaptive=adaptive,
        output_size=output_size,
        **kwargs,
    )
    return op_hndl


def AdaptiveAvgPool2d(name, adaptive=True, output_size=1, **kwargs):
    op_hndl = SimOpHandle(
        name,
        "AveragePool",
        params=[],
        ipos=[0],
        adaptive=adaptive,
        output_size=output_size,
        **kwargs,
    )
    return op_hndl


def conv1d(name, **kwargs):
    op_hndl = SimOpHandle(name, "Conv", params=[], ipos=[0, 1, 2], **kwargs)
    return op_hndl


def ReduceSum(name: str, axis=None, axes=None, **kwargs):
    """
    ReduceSum operator
    Args:
        name: Operation name
        axis: Single axis to reduce (int) - deprecated, use axes instead
        axes: Axis or axes to reduce (int, list of ints, or None for all axes)
        **kwargs: Additional kwargs (e.g., keepdims)
    """
    if axes is None and axis is not None:
        axes = axis

    # Convert to list if single int
    if axes is not None:
        axes_list = [axes] if isinstance(axes, int) else axes
        axesT = _from_data(name + '.axes', np.array(axes_list, dtype=np.int64), is_param=False, is_const=True)
        op_hndl = SimOpHandle(name, 'ReduceSum', params=[(1, axesT)], ipos=[0], **kwargs)
        op_hndl.implicit_inputs.append(axesT)
    else:
        # For axes=None, don't pass axes tensor - reduce over all dimensions
        # Set noop_with_empty_axes=0 to ensure reduction happens
        if 'noop_with_empty_axes' not in kwargs:
            kwargs['noop_with_empty_axes'] = 0
        op_hndl = SimOpHandle(name, 'ReduceSum', params=[], ipos=[0], **kwargs)
    return op_hndl

def ReduceMean(name: str, axes=None, **kwargs):
    """
    ReduceMean operator
    Args:
        name: Operation name
        axes: Axis or axes to reduce (int, list of ints, or None for all axes)
        **kwargs: Additional kwargs (e.g., keepdims)
    """
    # Convert to list if single int
    if axes is not None:
        axes_list = [axes] if isinstance(axes, int) else axes
        axesT = _from_data(name + '.axes', np.array(axes_list, dtype=np.int64), is_param=False, is_const=True)
        op_hndl = SimOpHandle(name, 'ReduceMean', params=[(1, axesT)], ipos=[0], **kwargs)
        op_hndl.implicit_inputs.append(axesT)
    else:
        # For axes=None, don't pass axes tensor - reduce over all dimensions
        # Set noop_with_empty_axes=0 to ensure reduction happens
        if 'noop_with_empty_axes' not in kwargs:
            kwargs['noop_with_empty_axes'] = 0
        op_hndl = SimOpHandle(name, 'ReduceMean', params=[], ipos=[0], **kwargs)
    return op_hndl

def ReduceMax(name: str, axes=None, **kwargs):
    """
    ReduceMax operator
    Args:
        name: Operation name
        axes: Axis or axes to reduce (int, list of ints, or None for all axes)
        **kwargs: Additional kwargs (e.g., keepdims)
    """
    # Convert to list if single int
    if axes is not None:
        axes_list = [axes] if isinstance(axes, int) else axes
        axesT = _from_data(name + '.axes', np.array(axes_list, dtype=np.int64), is_param=False, is_const=True)
        op_hndl = SimOpHandle(name, 'ReduceMax', params=[(1, axesT)], ipos=[0], **kwargs)
        op_hndl.implicit_inputs.append(axesT)
    else:
        # For axes=None, don't pass axes tensor - reduce over all dimensions
        # Set noop_with_empty_axes=0 to ensure reduction happens
        if 'noop_with_empty_axes' not in kwargs:
            kwargs['noop_with_empty_axes'] = 0
        op_hndl = SimOpHandle(name, 'ReduceMax', params=[], ipos=[0], **kwargs)
    return op_hndl


def ArgMax(name: str, axis=-1, **kwargs):
    """
    ArgMax operator - returns indices of maximum values along an axis
    Args:
        name: Operation name
        axis: Axis along which to find argmax (default: -1)
        **kwargs: Additional kwargs (e.g., keepdims, select_last_index)
    """
    kwargs['axis'] = axis
    op_hndl = SimOpHandle(name, 'ArgMax', params=[], ipos=[0], **kwargs)
    return op_hndl


def permute(name, dims, **kwargs):
    kwargs["perm"] = dims
    op_hndl = SimOpHandle(name, "Transpose", params=[], ipos=[0], **kwargs)
    return op_hndl


def topk(name, *, k, **kwargs):
    k_tensor = _from_data(
        name + ".k", data=np.array([k], dtype=np.int64), is_param=False, is_const=True
    )
    op_hndl = MultiOutputSimOpHandle(
        name,
        "TopK",
        params=[(1, k_tensor)],
        ipos=[0],
        num_outputs=2,
        **kwargs,
    )
    return op_hndl


######################################################################################################
# Simple Operator Mapping
######################################################################################################
def UniversalOperator(name, /, optype, params, ipos, **kwargs):
    return SimOpHandle(name, optype, params=params, ipos=ipos, **kwargs)


# Unary Operators
UnaryOperator = partial(UniversalOperator, params=[], ipos=[0])
Identity = partial(UnaryOperator, optype="Identity")
Tanh = partial(UnaryOperator, optype="Tanh")
Neg = partial(UnaryOperator, optype="Neg")
Abs = partial(UnaryOperator, optype="Abs")
exp = partial(UnaryOperator, optype="Exp")
Cos = partial(UnaryOperator, optype="Cos")
Sin = partial(UnaryOperator, optype="Sin")
Log = partial(UnaryOperator, optype="Log")
Sqrt = partial(UnaryOperator, optype="Sqrt")
Exp = partial(UnaryOperator, optype="Exp")
Softmax = partial(UnaryOperator, optype="Softmax")
softplus = partial(UnaryOperator, optype="Softplus")
Clip = partial(UnaryOperator, optype="Clip")
Cast = partial(UnaryOperator, optype="Cast")
Shape = partial(UnaryOperator, optype="Shape")
Transpose = partial(UnaryOperator, optype="Transpose")
Gelu = partial(UnaryOperator, optype="Gelu")
Relu = partial(UnaryOperator, optype="Relu")
Relu6 = partial(UnaryOperator, optype="Relu6")
Mish = partial(UnaryOperator, optype="Mish")
LeakyReLU = partial(UnaryOperator, optype="LeakyRelu")
Sigmoid = partial(UnaryOperator, optype="Sigmoid")
InverseSigmoid = partial(UnaryOperator, optype="InverseSigmoid")
Glu = partial(UnaryOperator, optype="Glu")
Diag = partial(UnaryOperator, optype="Diag")
AveragePool2d = partial(UnaryOperator, optype="AveragePool")
Sum = partial(UnaryOperator, optype="Sum")
Mean = partial(UnaryOperator, optype="Mean")
Reciprocal = partial(UnaryOperator, optype="Reciprocal")
Hardswish = partial(UnaryOperator, optype="HardSwish")
Atan = partial(UnaryOperator, optype='Atan')

# Binary Operators
BinaryOperator = partial(UniversalOperator, params=[], ipos=[0, 1])
Add = partial(BinaryOperator, optype="Add")
Sub = partial(BinaryOperator, optype="Sub")
Mul = partial(BinaryOperator, optype="Mul")
Div = partial(BinaryOperator, optype="Div")
Gather = partial(BinaryOperator, optype="Gather")
MatMul = partial(BinaryOperator, optype="MatMul")
Reshape = partial(BinaryOperator, optype="Reshape")
Pow = partial(BinaryOperator, optype="Pow")
Unsqueeze = partial(BinaryOperator, optype="Unsqueeze")
Squeeze = partial(BinaryOperator, optype="Squeeze")
Tile = partial(BinaryOperator, optype="Tile")
Equal = partial(BinaryOperator, optype="Equal")
Assign = partial(BinaryOperator, optype="Assign")
Pad = partial(BinaryOperator, optype="Pad")
L1Loss = partial(BinaryOperator, optype="L1Loss")  # legacy; prefer SimNN.L1Loss
BinaryCrossEntropyWithLogits = partial(
    BinaryOperator, optype="BinaryCrossEntropyWithLogits"
)  # added
Greater = partial(BinaryOperator, optype="Greater")  # added
GridSample = partial(BinaryOperator, optype='GridSample')
Less = partial(BinaryOperator, optype='Less')
Maximum = partial(BinaryOperator, optype='Max')  # Element-wise maximum of two tensors
Minimum = partial(BinaryOperator, optype='Min')  # Element-wise minimum of two tensors


def Cdist(name, x1, x2, p=2.0):
    """
    Pairwise distance computation between two collections of row vectors.

    Args:
        name: Operation name
        x1: SimTensor [..., P, M] - First collection of row vectors
        x2: SimTensor [..., R, M] - Second collection of row vectors
        p: Norm order (default 2.0)
            p=1: Manhattan distance (L1)
            p=2: Euclidean distance (L2)

    Returns:
        SimTensor [..., P, R] - Pairwise distances
        output[..., i, j] = ||x1[..., i, :] - x2[..., j, :]||_p

    Example:
        # Compute L1 distance between predictions and targets
        distances = F.Cdist('bbox_dist', pred_boxes, target_boxes, p=1.0)
    """
    # Create opinfo with p as attribute
    opinfo = get_opinfo(name, "Cdist", p=p)

    # Setup input tensors
    x1.op_in.append(name)
    opinfo["inList"].append(x1.name)
    x2.op_in.append(name)
    opinfo["inList"].append(x2.name)

    # Create output tensor
    otensor = get_output(name)
    opinfo["outList"] = [otensor.name]

    # Create SimOp
    sim_op = get_sim_op(opinfo, default_dtype=x1.dtype)

    # Get perf stats
    sim_op.get_perf_counts([x1, x2], [otensor])
    sim_op.update_tensor_counts([x1, x2], [otensor])

    return otensor


# Variadic Operators
def Einsum(name, subscripts, *operands):
    """
    Einstein summation operation.

    Args:
        name: Operation name
        subscripts: Einsum notation string (e.g., "bqnc,bnchw->bqnhw")
        *operands: Variable number of input tensors

    Returns:
        Output tensor from einsum operation

    Example:
        # Batched matrix multiplication with specific dimensions
        result = F.Einsum('attn_weights', 'bqnc,bnchw->bqnhw', queries, keys)
    """
    # Create opinfo with subscripts as attribute
    opinfo = get_opinfo(name, "Einsum", subscripts=subscripts)

    # Setup input tensors
    for x in operands:
        x.op_in.append(name)
        opinfo["inList"].append(x.name)

    # Create output tensor
    otensor = get_output(name)
    opinfo["outList"] = [otensor.name]

    # Create SimOp
    sim_op = get_sim_op(
        opinfo, default_dtype=operands[0].dtype if operands else np.float32
    )

    # Get perf stats
    sim_op.get_perf_counts(list(operands), [otensor])
    sim_op.update_tensor_counts(list(operands), [otensor])

    return otensor


# Ternary Operators
TernaryOperator = partial(UniversalOperator, params=[], ipos=[0, 1, 2])
Where = partial(TernaryOperator, optype="Where")
Range = partial(TernaryOperator, optype="Range")
ScatterND = partial(TernaryOperator, optype='ScatterND')
GroupNormalization = partial(TernaryOperator, optype="GroupNormalization")

# 4-ary Operators
FourAryOperator = partial(UniversalOperator, params=[], ipos=[0, 1, 2, 3])
VoxelPooling = partial(FourAryOperator, optype="VoxelPooling")

# Variadic Input Operator
# class VariadicInputOpHandle:
#    def __init__(self, name, optype, input_range, /, **kwargs):
ConcatX = partial(VariadicInputOpHandle, optype="Concat", input_range=(2, float("inf")))
TriluX = partial(VariadicInputOpHandle, optype="Trilu", input_range=(1, 2))
SliceF = partial(VariadicInputOpHandle, optype="Slice", input_range=(3, 6))

#Generator Operators (zero inputs)
def Constant(name_or_value, value=None, shape=None, dtype=None, **kwargs):
    """Create a constant tensor with the given value(s)

    Can be called as:
        Constant(name, value) - standard form
        Constant(value, shape=..., dtype=...) - compatibility form
    """
    # Determine if first arg is name or value
    if value is None:
        # Called as Constant(value, shape=..., dtype=...)
        value = name_or_value
        name = 'const_' + str(id(value))
    else:
        # Called as Constant(name, value)
        name = name_or_value

    # Handle shape parameter if provided
    if shape is not None:
        if isinstance(value, (int, float)):
            final_dtype = dtype if dtype is not None else np.float32
            data = np.full(shape, value, dtype=final_dtype)
        else:
            data = np.array(value, dtype=dtype if dtype is not None else np.float32)
            if list(data.shape) != list(shape):
                data = np.broadcast_to(data, shape)
    else:
        if isinstance(value, (int, float)):
            final_dtype = dtype if dtype is not None else np.float32
            data = np.array([value], dtype=final_dtype)
        elif isinstance(value, np.ndarray):
            data = value if dtype is None else value.astype(dtype)
        else:
            final_dtype = dtype if dtype is not None else np.float32
            data = np.array(value, dtype=final_dtype)

    # For constants, we can just return a SimTensor with the data
    # Since Constant is a generator op with no inputs
    const_tensor = _from_data(name + '.const', data, is_const=True)
    return const_tensor
