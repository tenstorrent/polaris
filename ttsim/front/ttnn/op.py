#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto
from itertools import count

import numpy as np
from loguru import logger

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import Shape, require_shape_list

from .buffer import BufferType, TensorMemoryLayout
from .memory import MemoryConfig
from .tensor import DataType, Layout, Tensor, require_ttnn_tensor, zeros


class MathFidelity(Enum):
    LoFi = auto()
    HiFi2 = auto()
    HiFi3 = auto()
    HiFi4 = auto()

    @classmethod
    def enumvalue(cls, s: str):
        return MathFidelity[s]

    @property
    def cname(self) -> str:
        return self.name.lower()


op_counter = count(start=1, step=1)


def generate_new_op_name():
    return f"ttsim.ttnn.Op_{next(op_counter)}"


_COMPACT_DTYPES: frozenset[DataType] = frozenset()  # populated after DataType import below
# Precedence order for compact dtypes (most compact first)
_COMPACT_DTYPE_PRECEDENCE: list[DataType] = []  # populated after DataType import


def _propagate_ttnn_dtype(inputs: list[Tensor], outputs: list[Tensor]) -> None:
    """Propagate _ttnn_dtype from inputs to unannotated outputs.

    When inputs carry different DataTypes (e.g. BFLOAT16 + BFLOAT8_B), the more
    compact type wins.  This mirrors HW behaviour where ``activations_dtype``
    (typically BFLOAT8_B) dominates after the first conversion point.

    Compact dtypes are prioritized by explicit precedence: BFLOAT4_B (most compact)
    is preferred over BFLOAT8_B, ensuring deterministic propagation regardless of
    input order.
    """
    candidates = [
        getattr(t, "_ttnn_dtype", None)
        for t in inputs
        if getattr(t, "_ttnn_dtype", None) is not None
    ]
    if not candidates:
        return

    # Find the most compact dtype according to precedence order
    compact = [d for d in candidates if d in _COMPACT_DTYPES]
    if compact:
        # Pick the first dtype in precedence order that appears in candidates
        src = next((d for d in _COMPACT_DTYPE_PRECEDENCE if d in compact), compact[0])
    else:
        src = candidates[0]

    for o in outputs:
        if getattr(o, "_ttnn_dtype", None) is None:
            o._ttnn_dtype = src


_COMPACT_DTYPES = frozenset({DataType.BFLOAT8_B, DataType.BFLOAT4_B})
# BFLOAT4_B (4 bits) is more compact than BFLOAT8_B (8 bits)
_COMPACT_DTYPE_PRECEDENCE = [DataType.BFLOAT4_B, DataType.BFLOAT8_B]


def _propagate_memory_config(inputs: list[Tensor], outputs: list[Tensor]) -> None:
    """Propagate _memory_config from the first input that has one set.

    On real tt-metal, op outputs inherit the memory placement of their
    inputs unless the op explicitly changes it (e.g. to_memory_config).
    """
    src = None
    for t in inputs:
        mc = getattr(t, "_memory_config", None)
        if mc is not None:
            src = mc
            break
    if src is None:
        return
    for o in outputs:
        if getattr(o, "_memory_config", None) is None:
            o._memory_config = src


def single_output_immediate_op(optype, /, preprocess=None):

    def _impl(*args, **kwargs):

        if preprocess:
            args, kwargs = preprocess(args, kwargs)

        tensor_args: list[Tensor] = []
        for i, x in enumerate(args):
            if isinstance(x, (int, float)):
                continue
            tensor_args.append(
                require_ttnn_tensor(x, f"ttnn.op({optype}) argument {i}")
            )
        if not tensor_args:
            raise TypeError(
                f"ttnn.op({optype}) requires at least one ttsim.front.ttnn.tensor.Tensor argument"
            )
        devchk_list = [x.device for x in tensor_args]
        device = devchk_list[0]
        # assert device and all(x == device for x in devchk_list), f"device check: {devchk_list}"

        op_name = generate_new_op_name()
        opinfo  = {'name': op_name, 'optype': optype, 'inList': [], 'attrs': kwargs}
        C = Tensor(name=op_name + ".out", op_out=[op_name], device=device)

        new_args = []
        for i, x in enumerate(args):
            if isinstance(x, Tensor):
                x.op_in.append(op_name)
                opinfo["inList"].append(x.name)
                new_args.append(x)
            elif isinstance(x, (int, float)):
                if optype in ["Add", "Sub", "Mul"]:
                    assert (
                        len(tensor_args) == 1
                    ), f"Only one tensor input supported for {optype} with scalar input"
                    opinfo["attrs"]["scalar"] = x
                    # On tt-metal, the scalar is broadcast-filled into a
                    # full-shape tensor buffer.  The profiler records both
                    # inputs, so include this broadcast tensor in inList to
                    # produce an arity-2 LUT key matching the profiler.
                    tensor_shape = list(tensor_args[0].shape) if tensor_args[0].shape is not None else [1]
                    tensor_dtype = DataType.from_numpy(tensor_args[0].dtype)

                    # Populate data for EXECUTE/EXECUTE_AND_TRACK modes so that
                    # compute helpers (compute_add/sub/mul) can access iTList[1].data
                    scalar_data = None
                    if hasattr(tensor_args[0], 'has_data') and tensor_args[0].has_data():
                        # Create broadcast-filled array matching tensor shape
                        np_dtype = tensor_args[0].dtype
                        scalar_data = np.full(tensor_shape, x, dtype=np_dtype)

                    tmp = Tensor(
                        name=f"{op_name}.scalar",
                        shape=tensor_shape,
                        dtype=tensor_dtype,
                        layout=tensor_args[0].get_layout(),
                        device=device,
                        data=scalar_data,
                    )
                    src_mc = getattr(tensor_args[0], "_memory_config", None)
                    if src_mc is not None:
                        tmp._memory_config = src_mc
                    src_dt = getattr(tensor_args[0], "_ttnn_dtype", None)
                    if src_dt is not None:
                        tmp._ttnn_dtype = src_dt
                    tmp.op_in.append(op_name)
                    opinfo["inList"].append(tmp.name)
                    new_args.append(tmp)
                else:
                    logger.warning(
                        "Scalar operand not supported for {} (only Add, Sub, Mul accept scalars); "
                        "got type {}",
                        optype,
                        type(x).__name__,
                    )
                    raise TypeError(
                        f"Scalar input is not supported for ttnn.op({optype}); "
                        f"only Add, Sub, and Mul accept scalar operands"
                    )
            else:
                assert False, f"Unknown input type in ttnn.op({optype}) : {type(x)}"
        opinfo["outList"] = [C.name]

        opobj = SimOp(opinfo)
        perf_stats = opobj.get_perf_counts(new_args, [C])
        opobj.update_tensor_counts(new_args, [C])

        _propagate_ttnn_dtype(tensor_args, [C])
        _propagate_memory_config(tensor_args, [C])

        mem_cfg = kwargs.get("memory_config", None)
        if mem_cfg is not None:
            C._memory_config = mem_cfg

        device.add_op(opobj)  # type: ignore[union-attr]

        return C

    return _impl


def multiple_output_immediate_op(optype, /, preprocess=None):
    def _impl(*args, **kwargs):

        if preprocess:
            args, kwargs = preprocess(args, kwargs)

        for i, x in enumerate(args):
            require_ttnn_tensor(x, f"ttnn.op({optype}) argument {i}")
        tensor_args: list[Tensor] = list(args)
        devchk_list = [x.device for x in tensor_args]
        device = devchk_list[0]
        # assert device and all(x == device for x in devchk_list), f"device check: {devchk_list}"

        op_name = generate_new_op_name()
        opinfo = {"name": op_name, "optype": optype, "inList": [], "attrs": kwargs}
        out_tensors = []
        num_outputs = kwargs.get("num_outputs", 2)
        for out_idx in range(num_outputs):
            C = Tensor(name=f"{op_name}.out.{out_idx}", op_out=[op_name], device=device)
            out_tensors.append(C)

        new_args = []
        for i, x in enumerate(args):
            if isinstance(x, Tensor):
                x.op_in.append(op_name)
                opinfo["inList"].append(x.name)
                new_args.append(x)
            else:
                assert False, f"Unknown input type in ttnn.op({optype}) : {type(x)}"
        opinfo["outList"] = [C.name for C in out_tensors]

        opobj = SimOp(opinfo)
        perf_stats = opobj.get_perf_counts(new_args, out_tensors)
        # print(f"{optype}:: {perf_stats}")
        opobj.update_tensor_counts(new_args, out_tensors)

        _propagate_ttnn_dtype(tensor_args, out_tensors)
        _propagate_memory_config(tensor_args, out_tensors)
        device.add_op(opobj)  # type: ignore[union-attr]

        return tuple(out_tensors)

    return _impl


def argmax_pp(args_list, kwargs_dict):
    # translate attribs
    kwargs_dict["axis"] = kwargs_dict.get("dim", 0)
    if "keepdim" in kwargs_dict:
        kwargs_dict["keepdims"] = 1 if kwargs_dict["keepdim"] else 0
    else:
        kwargs_dict["keepdims"] = 0
    return args_list, kwargs_dict


def reshape_pp(args_list, kwargs_dict):
    assert len(args_list) <= 3, f"ttnn.reshape has 3 inputs (special case for TT h/w)"
    inT = require_ttnn_tensor(args_list[0], "ttnn.reshape input")
    outShape = args_list[1]
    if isinstance(outShape, Shape):
        outShape = outShape.as_list()
    elif isinstance(outShape, (list, tuple)):
        outShape = list(outShape)
    else:
        assert False, (
            "ttnn.reshape 2nd input should be a list, tuple, or ttsim.ops.tensor.Shape"
        )
    assert all(
        isinstance(x, (int, np.integer)) for x in outShape
    ), f"ttnn.reshape 2nd input should be a sequence of integer sizes"

    in_dtype = DataType.from_numpy(inT.dtype)
    if len(args_list) == 3:
        # write code to get slice (batch 0) of the input tensor and return it
        inT = Tensor(shape=outShape, device=inT.device, dtype=in_dtype)

    # NOTE: dimensions in the shape should be integer type
    shapeData = np.array(outShape, dtype=np.int64)
    # NOTE: shapeData is not the reshape output, but it holds the shape tensor data,
    # so it should be of type INT64
    shapeT = Tensor(
        shape=shapeData.shape, dtype=DataType.INT64, device=inT.device, data=shapeData
    )
    return (inT, shapeT), kwargs_dict


def expand_pp(args_list, kwargs_dict):
    inT = require_ttnn_tensor(args_list[0], "ttnn.expand input")
    outShape = args_list[1]
    if isinstance(outShape, Shape):
        outShape = outShape.as_list()
    elif isinstance(outShape, (list, tuple)):
        outShape = list(outShape)
    else:
        assert False, (
            "ttnn.expand 2nd input should be a list, tuple, or ttsim.ops.tensor.Shape"
        )
    outData = np.array(outShape, dtype=np.int64)
    outT = Tensor(
        shape=outData.shape, dtype=DataType.INT64, device=inT.device, data=outData
    )
    return (inT, outT), kwargs_dict


def split_pp(args_list, kwargs_dict):
    inT = require_ttnn_tensor(args_list[0], "ttnn.split input")
    outT = require_ttnn_tensor(args_list[1], "ttnn.split output template")
    split_sizes = kwargs_dict.get("split_sizes", None)
    num_splits = kwargs_dict.get("num_splits", None)
    axis = kwargs_dict.get("dim", 0)
    kwargs_dict["split_sizes"] = split_sizes
    kwargs_dict["num_splits"] = num_splits
    kwargs_dict["axis"] = axis
    return (inT, outT), kwargs_dict


def permute_pp(args_list, kwargs_dict):
    inT = require_ttnn_tensor(args_list[0], "ttnn.permute input")
    assert isinstance(
        args_list[1], (list, tuple)
    ), f"ttnn.permute 2nd input should be a list|tuple of ints"
    kwargs_dict["perm"] = list(args_list[1])
    return (inT,), kwargs_dict


def embedding_pp(args_list, kwargs_dict):
    # TTNN passes in the order indices, weights while Polaris takes weights, indices
    assert len(args_list) == 2, f"ttnn.embedding has 2 inputs"
    input_tensor = require_ttnn_tensor(args_list[0], "ttnn.embedding indices")
    weight_tensor = require_ttnn_tensor(args_list[1], "ttnn.embedding weight")
    return (weight_tensor, input_tensor), kwargs_dict


def layer_norm_pp(args_list, kwargs_dict):
    input_tensor = args_list[0]
    weight_tensor = kwargs_dict["weight"] if "weight" in kwargs_dict else None
    bias_tensor = kwargs_dict["bias"] if "bias" in kwargs_dict else None
    epsilon = kwargs_dict["epsilon"] if "epsilon" in kwargs_dict else None
    memory_config = (
        kwargs_dict["memory_config"] if "memory_config" in kwargs_dict else None
    )
    compute_kernel_config = (
        kwargs_dict["compute_kernel_config"]
        if "compute_kernel_config" in kwargs_dict
        else None
    )

    input_tensor = require_ttnn_tensor(input_tensor, "ttnn.layer_norm input")
    if weight_tensor is not None:
        weight_tensor = require_ttnn_tensor(weight_tensor, "ttnn.layer_norm weight")
    if bias_tensor is not None:
        bias_tensor = require_ttnn_tensor(bias_tensor, "ttnn.layer_norm bias")

    kwargs_dict = {}
    if bias_tensor is not None:
        return (input_tensor, weight_tensor, bias_tensor), kwargs_dict
    else:
        return (input_tensor, weight_tensor), kwargs_dict


def conv2d_pp(args_list, kwargs_dict):
    input_tensor = require_ttnn_tensor(
        kwargs_dict["input_tensor"], "ttnn.conv2d input_tensor"
    )
    weight_tensor = require_ttnn_tensor(
        kwargs_dict["weight_tensor"], "ttnn.conv2d weight_tensor"
    )
    bias_tensor = require_ttnn_tensor(
        kwargs_dict["bias_tensor"], "ttnn.conv2d bias_tensor"
    )
    strides = kwargs_dict.get("stride", (1, 1))
    padding_size = kwargs_dict["padding"][0]
    pads = [padding_size for i in range(4)]
    kwargs_dict = {
        "pads": pads,
        "kernel_shape": list(kwargs_dict["kernel_size"]),
        "strides": list(strides),
    }
    return (input_tensor, weight_tensor, bias_tensor), kwargs_dict


def outer_pp(args_list, kwargs_dict):
    """Preprocessor for outer product operation."""
    assert len(args_list) == 2, f"ttnn.outer has 2 inputs"
    tensor_a = require_ttnn_tensor(args_list[0], "ttnn.outer input a")
    tensor_b = require_ttnn_tensor(args_list[1], "ttnn.outer input b")
    assert tensor_a.shape is not None and tensor_b.shape is not None

    # Validate that inputs are 1D tensors
    if len(tensor_a.shape) != 1:
        raise ValueError(
            f"ttnn.outer expects 1D tensors, got tensor_a with shape {tensor_a.shape}"
        )
    if len(tensor_b.shape) != 1:
        raise ValueError(
            f"ttnn.outer expects 1D tensors, got tensor_b with shape {tensor_b.shape}"
        )

    # Output shape will be (len(tensor_a), len(tensor_b))
    output_shape = [tensor_a.shape[0], tensor_b.shape[0]]
    kwargs_dict["output_shape"] = output_shape

    return (tensor_a.unsqueeze(1), tensor_b.unsqueeze(0)), kwargs_dict


def torchgather_pp(args_list, kwargs_dict):
    """Preprocessor for torch gather operation. Torch Gather differs vs. ONNX Gather."""
    assert len(args_list) == 3, f"ttnn.gather has 3 inputs"
    input_tensor = require_ttnn_tensor(args_list[0], "ttnn.gather input")
    dim = args_list[1]
    index_tensor = require_ttnn_tensor(args_list[2], "ttnn.gather index")

    # Ensure the index tensor is of integer type
    if index_tensor.dtype != np.int64:
        raise ValueError(
            f"ttnn.gather expects index tensor to be of type int64, got {index_tensor.dtype}"
        )

    kwargs_dict["axis"] = dim
    return (input_tensor, index_tensor), kwargs_dict


def transpose_pp(args_list, kwargs_dict):
    inT = require_ttnn_tensor(args_list[0], "ttnn.transpose input")
    dim1 = args_list[1]
    dim2 = args_list[2]
    in_rank = len(
        require_shape_list(inT.shape, "ttnn.transpose input shape must be set")
    )
    out_dims = [i for i in range(in_rank)]
    out_dims[dim2] = dim1
    out_dims[dim1] = dim2
    kwargs_dict = {"perm": out_dims}
    return ([inT], kwargs_dict)


def cat(tensors, dim=0):
    """Concatenate a list of tensors along a specified dimension.

    Delegates to ``concat`` (which emits a ``Concat`` SimOp) so that the
    operation is visible in the device graph regardless of whether the
    caller spells it ``ttnn.cat`` or ``ttnn.concat``.
    """
    return concat(*tensors, axis=dim)


def where_pp(args, kwargs_dict):
    mask_tensor = require_ttnn_tensor(args[0], "ttnn.where condition")
    input_tensor = require_ttnn_tensor(args[1], "ttnn.where x")
    value_tensor = require_ttnn_tensor(args[2], "ttnn.where y")
    return (mask_tensor, input_tensor, value_tensor), kwargs_dict


def rms_norm(
    input_tensor,
    weight_tensor=None,
    bias_tensor=None,
    epsilon=1e-6,
    memory_config=None,
    compute_kernel_config=None,
    dim=3072,
):
    input_tensor = require_ttnn_tensor(input_tensor, "ttnn.rms_norm input")
    if weight_tensor is not None:
        weight_tensor = require_ttnn_tensor(weight_tensor, "ttnn.rms_norm weight")
    if bias_tensor is not None:
        bias_tensor = require_ttnn_tensor(bias_tensor, "ttnn.rms_norm bias")
    # Compute RMS (using layernorm for now)
    rms = layer_norm(input_tensor, weight=weight_tensor, epsilon=epsilon, axis=-1)
    normalized = div(input_tensor, rms)
    if weight_tensor is not None:
        weight_tensor = reshape(weight_tensor, (1, 1, 1, dim))
        if normalized.shape[-1] != weight_tensor.shape[-1]:
            weight_tensor = weight_tensor.repeat(
                (1, 1, 1, normalized.shape[-1] // dim)
            )  ## only repeat if needed
        normalized = multiply(normalized, weight_tensor)

    if bias_tensor is not None:
        normalized = add(normalized, bias_tensor)
    return normalized


def max_pool2d_pp(args_list, kwargs_dict):
    input_tensor = require_ttnn_tensor(
        kwargs_dict["input_tensor"], "ttnn.max_pool2d input_tensor"
    )
    kernel_size = kwargs_dict["kernel_size"]
    stride = kwargs_dict.get("stride", kernel_size)
    padding = kwargs_dict.get("padding", 0)
    dilation = kwargs_dict.get("dilation", 1)
    ceil_mode = kwargs_dict.get("ceil_mode", False)

    kwargs_dict = {
        "kernel_shape": list(kernel_size),
        "strides": list(stride),
        "pads": [
            padding[0],
            padding[0],
            padding[1],
            padding[1],
        ],  # [pad_top, pad_left, pad_bottom, pad_right]
        "dilations": list(dilation),
        "ceil_mode": ceil_mode,
    }
    return (input_tensor,), kwargs_dict


def conv_transpose2d_pp(args_list, kwargs_dict):
    input_tensor = require_ttnn_tensor(
        kwargs_dict["input_tensor"], "ttnn.conv_transpose2d input_tensor"
    )
    weight_tensor = require_ttnn_tensor(
        kwargs_dict["weight_tensor"], "ttnn.conv_transpose2d weight_tensor"
    )
    bias_tensor = require_ttnn_tensor(
        kwargs_dict["bias_tensor"], "ttnn.conv_transpose2d bias_tensor"
    )
    padding_size = kwargs_dict["padding"][0]
    pads = [padding_size for i in range(4)]
    output_padding = kwargs_dict.get("output_padding", (0, 0))
    strides = kwargs_dict.get("stride", (1, 1))
    kwargs_dict = {
        "padding": pads,
        "kernel_size": list(kwargs_dict["kernel_size"]),
        "output_padding": list(output_padding),
        "strides": list(strides),
    }
    return (input_tensor, weight_tensor, bias_tensor), kwargs_dict


def as_pp(args_list, kwargs_dict):
    input_tensor = require_ttnn_tensor(args_list[0], "ttnn.slice input")
    slice_spec = kwargs_dict.get("slice", None)
    assert (
        slice_spec is not None
    ), "ttnn.slice requires 'slice' attribute specifying indices"

    # Compute the shape of the slice
    # Use numpy to infer the shape
    in_shape = require_shape_list(
        input_tensor.shape, "ttnn.slice input shape must be set"
    )
    dummy = np.empty(in_shape)
    sliced = dummy[slice_spec]
    out_shape = list(sliced.shape)

    kwargs_dict["output_shape"] = out_shape
    return (input_tensor,), kwargs_dict


def topk_pp(args_list, kwargs_dict):
    input_tensor = require_ttnn_tensor(args_list[0], "ttnn.topk input")
    k_tensor = require_ttnn_tensor(args_list[1], "ttnn.topk k")
    axis = kwargs_dict.get("dim", -1)
    largest = kwargs_dict.get("largest", True)
    sorted = kwargs_dict.get("sorted", True)

    k_shape = require_shape_list(k_tensor.shape, "ttnn.topk k shape must be set")
    kwargs_dict = {
        "k": k_shape[0],
        "axis": axis,
        "largest": 1 if largest else 0,
        "sorted": 1 if sorted else 0,
    }
    return (input_tensor, k_tensor), kwargs_dict


def zeros_like(input_tensor, memory_config=None):
    require_ttnn_tensor(input_tensor, "ttnn.zeros_like input")
    return zeros(
        shape=input_tensor.shape,
        dtype=DataType.from_numpy(input_tensor.dtype.name),
        layout=(
            Layout.from_numpy(input_tensor.layout.name)
            if input_tensor.layout
            else Layout.TILE_LAYOUT
        ),
        device=input_tensor.device,
    )


def compare(input_tensor_a, input_tensor_b, op_type):
    require_ttnn_tensor(input_tensor_a, "ttnn.compare input a")
    require_ttnn_tensor(input_tensor_b, "ttnn.compare input b")
    assert op_type in [
        "equal",
        "not_equal",
        "greater",
        "less",
        "greater_equal",
        "less_equal",
    ], f"Unsupported compare op_type: {op_type}"
    # For simplicity, we return a tensor of the same shape with boolean dtype
    return Tensor(
        shape=input_tensor_a.shape, dtype=DataType.BOOL, device=input_tensor_a.device
    )


def maximum(input_tensor_a, input_tensor_b):
    require_ttnn_tensor(input_tensor_a, "ttnn.maximum input a")
    require_ttnn_tensor(input_tensor_b, "ttnn.maximum input b")
    return input_tensor_a  # Placeholder implementation


def unsqueeze(input_tensor, dim):
    require_ttnn_tensor(input_tensor, "ttnn.unsqueeze input")
    return input_tensor.unsqueeze(dim)


def divide(input_tensor, divisor, use_legacy=False):
    require_ttnn_tensor(input_tensor, "ttnn.divide input")
    if isinstance(divisor, (int, float)):
        divisor = Tensor(
            shape=input_tensor.shape,
            dtype=DataType.from_numpy(input_tensor.dtype.name),
            device=input_tensor.device,
            data=np.full(input_tensor.shape, divisor, dtype=input_tensor.dtype),
        )
    require_ttnn_tensor(divisor, "ttnn.divide divisor")
    return div(input_tensor, divisor)


def clone(input_tensor, memory_config=None):
    require_ttnn_tensor(input_tensor, "ttnn.clone input")
    return input_tensor


def squeeze(input_tensor, dim):
    require_ttnn_tensor(input_tensor, "ttnn.squeeze input")
    return input_tensor.squeeze(dim)


def repeat(input_tensor, repeats):
    require_ttnn_tensor(input_tensor, "ttnn.repeat input")
    output_shape = [i * j for i, j in zip(list(input_tensor.shape), repeats)]
    return Tensor(
        shape=output_shape,
        dtype=DataType.from_numpy(input_tensor.dtype.name),
        device=input_tensor.device,
    )


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

    @staticmethod
    def nlp_create_qkv_heads(input_tensor, kv_input_tensor=None, *,
                              num_heads, num_kv_heads=None,
                              transpose_k_heads=False, memory_config=None):
        """Delegate to ttnn_shim; mirrors HW's single-op QKV head split."""
        from .ttnn_shim import nlp_create_qkv_heads as _nlp_create_qkv_heads
        return _nlp_create_qkv_heads(
            input_tensor, kv_input_tensor,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            transpose_k_heads=transpose_k_heads, memory_config=memory_config,
        )

    @staticmethod
    def nlp_concat_heads(input_tensor, memory_config=None):
        """Delegate to ttnn_shim; mirrors HW's single-op head concatenation."""
        from .ttnn_shim import nlp_concat_heads as _nlp_concat_heads
        return _nlp_concat_heads(input_tensor, memory_config=memory_config)


def all_gather(*args, **kwargs):
    raise NotImplementedError("all_gather is not implemented yet!!")


def eqz(input_tensor):
    return compare(input_tensor, zeros_like(input_tensor), op_type='equal')


def moe(gate_logits, expert_mask, topE_mask, k, k_tensor):
    gate_logits = require_ttnn_tensor(gate_logits, "ttnn.moe gate_logits")
    expert_mask = require_ttnn_tensor(expert_mask, "ttnn.moe expert_mask")
    topE_mask = require_ttnn_tensor(topE_mask, "ttnn.moe topE_mask")
    k_tensor = require_ttnn_tensor(k_tensor, "ttnn.moe k_tensor")
    N, C, H, W = gate_logits.shape
    assert expert_mask.shape == [N, C, 1, W], "expert_mask must be [N, C, 1, W]"
    assert topE_mask.shape[-1] == k, f"topE_mask last dim must be k"

    # 1) Apply expert_mask to zero out padded experts (set to -inf)
    #    Broadcast over H: [N,C,H,W] + [N,C,1,W] -> [N,C,H,W]
    masked_logits = sum(gate_logits, expert_mask)

    # 2) Top-k over experts (last dimension)
    topk_values, topk_indices = topk(masked_logits, k_tensor, dim=-1)  # [N,C,H,k]

    # 3) Apply topE_mask (implements "e" effective experts within top-k)
    #    topE_mask: [N,C,1,k] -> broadcast to [N,C,H,k]
    topk_scores = softmax(topk_values + topE_mask, dim=-1)      # [N,C,H,k]

    # 4) Select only entries that correspond to `target_expert`
    expert_selector = eqz(topk_indices)       # [N,C,H,k]

    # 5) Aggregate weight for that expert over its (up to e) positions in top-k
    #    Result is shape [N,C,H,1]
    weights = sum(topk_scores * expert_selector, dim=-1)
    weights = unsqueeze(weights, -1)

    return weights


# Pointwise Unary
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
clamp       = single_output_immediate_op('Clip')
log         = single_output_immediate_op('Log')
min         = single_output_immediate_op('Min')
max         = single_output_immediate_op('Max')
sqrt        = single_output_immediate_op('Sqrt')

# Pointwise Binary
add = single_output_immediate_op("Add")
sub = single_output_immediate_op("Sub")
multiply = single_output_immediate_op("Mul")
subtract = single_output_immediate_op("Sub")
div = single_output_immediate_op("Div")
pow = single_output_immediate_op("Pow")
mean = single_output_immediate_op("Mean")
sum = single_output_immediate_op("Sum")
atan = single_output_immediate_op("Atan")
exp = single_output_immediate_op("Exp")

# Pointwise Ternary
where = single_output_immediate_op("Where", preprocess=where_pp)
nonzero = single_output_immediate_op("NonZero")

# Reduction
argmax = single_output_immediate_op("ArgMax", preprocess=argmax_pp)

# Data Movement
concat = single_output_immediate_op("Concat")
reshape = single_output_immediate_op("Reshape", preprocess=reshape_pp)
expand = single_output_immediate_op("Expand", preprocess=expand_pp)
embedding = single_output_immediate_op("Gather", preprocess=embedding_pp)
permute = single_output_immediate_op("Transpose", preprocess=permute_pp)
gather = single_output_immediate_op("TorchGather", preprocess=torchgather_pp)
transpose = single_output_immediate_op("Transpose", preprocess=transpose_pp)
split = multiple_output_immediate_op("Split", preprocess=split_pp)

# Normalization
layer_norm = single_output_immediate_op("LayerNormalization", preprocess=layer_norm_pp)
batch_norm = single_output_immediate_op("BatchNormalization")

# Convolution
conv2d = single_output_immediate_op("Conv", preprocess=conv2d_pp)
conv_transpose2d = single_output_immediate_op(
    "ConvTranspose", preprocess=conv_transpose2d_pp
)

# Pooling
global_avg_pool2d = single_output_immediate_op("GlobalAveragePool")
max_pool2d = single_output_immediate_op("MaxPool", preprocess=max_pool2d_pp)

# Matrix Multiplication
matmul = single_output_immediate_op("MatMul")
outer = single_output_immediate_op("MatMul", preprocess=outer_pp)

# Funky Ops
grid_sample = single_output_immediate_op("GridSample")
assign = single_output_immediate_op("Assign", preprocess=as_pp)
topk = multiple_output_immediate_op("TopK", preprocess=topk_pp)


Tensor.__add__ = add  # type: ignore
Tensor.__sub__ = subtract  # type: ignore
Tensor.__mul__ = multiply  # type: ignore
Tensor.__div__ = div  # type: ignore
Tensor.__pow__ = pow  # type: ignore
Tensor.__matmul__ = matmul  # type: ignore
Tensor.reshape = reshape  # type: ignore


def silu(x):
    return x * sigmoid(x)


# Multi-operator functions
def linear(*args, **kwargs):
    """Fused linear: emits a single MatMul SimOp with optional bias (3rd input)
    and optional fused activation, matching HW's MatmulDeviceOperation.

    Previous implementation decomposed linear into separate matmul → add → activation
    SimOps.  HW's MatmulDeviceOperation fuses all three into one kernel, so the
    decomposed graph produced extra ops that did not appear in profiler traces.
    Emitting a single MatMul SimOp (with bias as an optional 3rd input and activation
    as an attribute) keeps the POLARIS op graph 1-to-1 with HW profiler output.
    """
    assert len(args) == 2, f"linear args #-inputs({len(args)}) != 2"
    A = require_ttnn_tensor(args[0], "ttnn.linear input")
    B = require_ttnn_tensor(args[1], "ttnn.linear weight")
    bias_tensor = kwargs.get("bias", None)
    act = kwargs.get("activation", None)

    device = A.device if hasattr(A, 'device') and A.device else (
        B.device if hasattr(B, 'device') else None)

    op_name = generate_new_op_name()
    attrs = {}
    if act is not None:
        attrs["fused_activation"] = act
    opinfo = {'name': op_name, 'optype': 'MatMul', 'inList': [], 'attrs': attrs}
    C = Tensor(name=op_name + ".out", op_out=[op_name], device=device)

    input_tensors = []
    for x in [A, B]:
        x.op_in.append(op_name)
        opinfo["inList"].append(x.name)
        input_tensors.append(x)

    if bias_tensor is not None:
        bias_tensor = require_ttnn_tensor(bias_tensor, "ttnn.linear bias")
        bias_tensor.op_in.append(op_name)
        opinfo["inList"].append(bias_tensor.name)
        input_tensors.append(bias_tensor)

    opinfo["outList"] = [C.name]
    opobj = SimOp(opinfo)
    opobj.get_perf_counts(input_tensors, [C])
    opobj.update_tensor_counts(input_tensors, [C])

    # Handle both dtype and output_dtype parameters (dtype is an alias for output_dtype)
    output_dtype = kwargs.get("output_dtype", None)
    if output_dtype is None:
        output_dtype = kwargs.get("dtype", None)
    if output_dtype is not None and isinstance(output_dtype, DataType):
        C._ttnn_dtype = output_dtype
    else:
        _propagate_ttnn_dtype(input_tensors, [C])

    _propagate_memory_config(input_tensors, [C])

    mem_cfg = kwargs.get("memory_config", None)
    if mem_cfg is not None:
        C._memory_config = mem_cfg

    if device is not None:
        device.add_op(opobj)
    return C


# fold:
# takes an input tensor with shape (N, H, W, C) and transforms it to shape
# (N, H//stride_h, W//stride_w, C*stride_h*stride_w) by reshaping and permuting
# the spatial dimensions. This operation is commonly used as a preprocessing step
# for convolution operations, similar to the im2col operation in other deep learning
# frameworks, to reorganize input data in a format suitable for efficient matrix
# multiplication on Tenstorrent hardware.
def fold(
    ttnn_tensor_like,
    stride_h: int,
    stride_w: int,
    *,
    use_transpose_as_fold=False,
    output_shape=None,  # ttnn.Shape
    pad_c: int = 0,
    pad_h: int = 0,
    pad_w: int = 0,
    grid_size=None,  # ttnn.CoreRangeSet
    override_memory_config: MemoryConfig = None,  # type: ignore
):

    ttnn_tensor_like = require_ttnn_tensor(ttnn_tensor_like, "ttnn.fold input")
    assert (
        ttnn_tensor_like.rank() == 4
    ), f"fold input should be a rank-4 [N, H, W, C] tensor!!\n{ttnn_tensor_like}"
    N, H, W, C = ttnn_tensor_like.shape

    assert (
        isinstance(stride_h, int) and stride_h > 0 and stride_h <= H
    ), f"stride_h({stride_h}) should be in [0, {H}]"
    assert (
        isinstance(stride_w, int) and stride_w > 0 and stride_w <= H
    ), f"stride_w({stride_w}) should be in [0, {W}]"

    if pad_h > 0:
        H += pad_h
    if pad_w > 0:
        W += pad_w
    if pad_c > 0:
        C += pad_c

    Hs = H // stride_h
    Ws = W // stride_w

    if use_transpose_as_fold:
        # fold implemented as a series of reshape/transpose
        reshaped1 = ttnn_tensor_like.reshape(N, Hs, stride_h, Ws, stride_w, C)
        transposed = reshaped1.permute(0, 1, 3, 2, 4, 5)
        reshaped2 = transposed.reshape(N, Hs, Ws, C * stride_h * stride_w)
    else:
        # Fold as first-class SimOp (matches hardware Fold kernel naming vs reshape shortcut).
        assert ttnn_tensor_like.device is not None, "fold requires input tensor on device"
        op_name = generate_new_op_name()
        # Design decision: flatten_nd is *computed* from the input tensor's
        # layout and memory configuration rather than being a caller-supplied
        # parameter.  This mirrors the tt-metal C++ implementation where
        # prim::fold always produces [1,1,N*Hs*Ws,Cs] but the higher-level
        # ttnn::fold conditionally reshapes back to [N,Hs,Ws,Cs] for tiled
        # or DRAM-interleaved inputs (see fold.cpp / fold_device_op.cpp).
        #
        # The choice of `not (is_tiled or is_dram)` means:
        #   - L1-sharded ROW_MAJOR → flatten_nd=True  (ViT, typical models)
        #   - DRAM-interleaved     → flatten_nd=False  (preserve 4D)
        #   - TILE_LAYOUT          → flatten_nd=False  (preserve 4D)
        #   - Unknown/no memcfg    → flatten_nd=True   (safe HW default)
        #
        # This attr is forwarded to fold_sinf in tensor.py, which is
        # frontend-agnostic; see the comment there for the default rationale.
        is_tiled = getattr(ttnn_tensor_like, 'layout', None) == Layout.TILE_LAYOUT
        # Tensor.memory_config() returns None when no config has been set
        # (the common L1-sharded path); hasattr guards against non-Tensor inputs.
        mc = ttnn_tensor_like.memory_config() if hasattr(ttnn_tensor_like, 'memory_config') else None
        is_dram = (getattr(mc, 'buffer_type', None) is BufferType.DRAM) if mc is not None else False
        fold_attrs = {
            'stride_h': stride_h,
            'stride_w': stride_w,
            'pad_h': pad_h,
            'pad_w': pad_w,
            'pad_c': pad_c,
        }
        out_tensor = Tensor(
            name=op_name + '.out',
            op_out=[op_name],
            device=ttnn_tensor_like.device,
            dtype=ttnn_tensor_like.dtype,
            layout=ttnn_tensor_like.layout,
        )
        ttnn_tensor_like.op_in.append(op_name)
        opinfo = {
            'name': op_name,
            'optype': 'Fold',
            'inList': [ttnn_tensor_like.name],
            'outList': [out_tensor.name],
            'attrs': fold_attrs,
        }
        opobj = SimOp(opinfo)
        opobj.get_perf_counts([ttnn_tensor_like], [out_tensor])
        opobj.update_tensor_counts([ttnn_tensor_like], [out_tensor])
        _propagate_ttnn_dtype([ttnn_tensor_like], [out_tensor])
        out_tensor._memory_config = MemoryConfig(
            TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1,
        )
        ttnn_tensor_like.device.add_op(opobj)
        reshaped2 = out_tensor

    return reshaped2
