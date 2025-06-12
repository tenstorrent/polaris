#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Construct Backward Graph From a Forward Graph
import ttsim.front.functional.op as F
from ttsim.ops.tensor import SimTensor

from itertools import count
import numpy as np

counter = count(start=1, step=1)

######## helper functions ##################
def torch2onnx_slice_plan(input_shape, slice_spec):
    """
    Analyze PyTorch/Numpy-style slice_spec for a tensor of shape input_shape,
    and return an ONNX execution plan as a dict:
        {
            'unsqueezes'  : [axis, ...],  # axes at which to insert new dims (None in slice_spec)
            'gathers'     : [(axis, index), ...],
            'slice'       : {'axes': [...], 'starts': [...], 'ends': [...], 'steps': [...]},
            'squeeze_axes': [...],    # always [0, 1, ...] after gathers, or None
            'output_shape': list,     # shape after all slicing and unsqueezing
        }
    Raises AssertionError or NotImplementedError for invalid or unsupported cases.
    """
    # Ensure list for easy manipulation
    slice_spec = list(slice_spec)
    ndim       = len(input_shape)

    # Check for multiple ellipsis
    if slice_spec.count(Ellipsis) > 1:
        raise AssertionError("Multiple ellipsis not allowed in slice_spec")

    # Expand ellipsis
    if Ellipsis in slice_spec:
        idx = slice_spec.index(Ellipsis)
        n_specified = len([s for s in slice_spec if s is not Ellipsis and s is not None])
        num_missing = ndim - n_specified
        expanded_spec = (slice_spec[:idx] + [slice(None)] * num_missing + slice_spec[idx+1:])
    else:
        expanded_spec = slice_spec[:]

    # After ellipsis expansion, insert None axes remain untouched
    # Now, pad with slice(None) if needed (excluding None axes)
    n_not_none = len([s for s in expanded_spec if s is not None])
    if n_not_none < ndim:
        expanded_spec += [slice(None)] * (ndim - n_not_none)

    # --- Check for too many non-None indices ---
    n_non_none = len([s for s in expanded_spec if s is not None])
    if n_non_none > ndim:
        raise AssertionError(
            f"Too many non-None indices for tensor: expected <= {ndim}, got {n_non_none}"
        )

    # Identify where None occurs (axes for Unsqueeze)
    unsqueeze_axes = [i for i, s in enumerate(expanded_spec) if s is None]

    # Remove None entries to get the actual slice/gather spec for data axes
    data_spec = [s for s in expanded_spec if s is not None]
    assert len(data_spec) == ndim, (
        f"Internal error: after removing Nones, number of axes is {len(data_spec)}, expected {ndim}"
    )

    # Build gathers
    gathers = []
    working_shape = list(input_shape)
    axes_removed = 0
    for i, spec in enumerate(data_spec):
        if isinstance(spec, int):
            axis = i - axes_removed
            idx = spec if spec >= 0 else working_shape[axis] + spec
            if not (0 <= idx < working_shape[axis]):
                raise AssertionError(
                    f"Integer index {idx} out of bounds for axis {axis} with dim {working_shape[axis]}"
                )
            gathers.append((axis, idx))
            working_shape.pop(axis)
            axes_removed += 1

    # Remaining axes for slice
    post_gather_spec = [spec for spec in data_spec if not isinstance(spec, int)]
    post_gather_shape = working_shape
    slice_axes = []
    starts = []
    ends = []
    steps = []
    out_shape = []
    for i, (dim, spec) in enumerate(zip(post_gather_shape, post_gather_spec)):
        if not isinstance(spec, slice):
            raise AssertionError(f"Non-slice object found where slice expected: {spec}")
        s = 0 if spec.start is None else (spec.start if spec.start >= 0 else dim + spec.start)
        e = dim if spec.stop is None else (spec.stop if spec.stop >= 0 else dim + spec.stop)
        step = 1 if spec.step is None else spec.step
        if step < 0:
            raise NotImplementedError("Negative steps are not supported by ONNX Slice (opset 13).")
        if step > 0:
            e = min(e, dim)
        else:
            e = max(e, -1)
        slice_axes.append(i)
        starts.append(s)
        ends.append(e)
        steps.append(step)
        # Compute output shape for this axis
        length = (e - s + (step - 1)) // step if step > 0 else 0
        out_shape.append(max(0, length))

    squeeze_axes = list(range(len(gathers))) if gathers else None

    # Compute final output shape, including unsqueezes
    final_shape = out_shape
    for axis in unsqueeze_axes:
        # Each unsqueeze is on the original index before any insertions,
        # so as we insert, subsequent indices shift.
        if axis < 0 or axis > len(final_shape):
            raise AssertionError(f"Invalid None position at {axis}")
        final_shape.insert(axis, 1)

    plan = {
        'unsqueezes': unsqueeze_axes if unsqueeze_axes else None,
        'gathers': gathers if gathers else None,
        'slice': {
            'axes': slice_axes,
            'starts': starts,
            'ends': ends,
            'steps': steps,
        } if slice_axes else None,
        'squeeze_axes': squeeze_axes,
        'output_shape': final_shape,
    }
    return plan

######## specific tensor operations ##################
def tensor_contiguous(self):
    return self #identity function for now

def tensor_view(self, *shape):
    assert isinstance(self, SimTensor), f"tensor_view self = {self} not a SimTensor!!"
    shape      = list(shape) #type: ignore
    orig_numel = self.nelems()

    # Handle a single shape passed as a tuple/list
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = list(shape[0]) #type: ignore

    # Handle -1 for one of the dimensions (PyTorch-style inference)
    infer_idx = None
    known = 1
    for i, d in enumerate(shape):
        if d == -1:
            if infer_idx is not None:
                raise ValueError("Only one dimension can be inferred (-1)")
            infer_idx = i
        else:
            known *= d

    if infer_idx is not None:
        if known == 0:
            raise ValueError("Known product of shape dimensions is zero.")
        if orig_numel % known != 0:
            raise ValueError("Shape is not compatible for view (cannot infer dimension)")
        shape[infer_idx] = orig_numel // known #type: ignore

    # Check total elements match
    new_numel = 1
    for d in shape: new_numel *= d
    if new_numel != orig_numel:
        raise ValueError("Shape is not compatible for view (element count mismatch)")

    assert self.link_module is not None, f"link_module for {self.name} not specified!!"
    op_name = f"{self.link_module.name}.view.impl_{next(counter)}"
    op = F.Reshape(op_name)
    op.set_module(self.link_module)
    self.link_module._op_hndls[op.name] = op
    shapeTensor = F._from_data(op_name + '.fixshape', is_const=True, data=np.array(shape, dtype=np.int64))
    for x in [self, shapeTensor]:
        if x.name not in self.link_module._tensors:
            self.link_module._tensors[x.name] = x
    return op(self, shapeTensor)

def tensor_transpose(self, dim0, dim1):
    assert isinstance(self, SimTensor), f"transpose self = {self} not a SimTensor!!"
    if self.rank() < 1: raise ValueError("Tensor rank must be at least 1.")

    # Handle negative indices (convert to positive)
    if dim0 < 0: dim0 = self.rank() + dim0
    if dim1 < 0: dim1 = self.rank() + dim1

    # Validate dimension indices
    if dim0 < 0 or dim0 >= self.rank():
        raise ValueError(f"dim0 ({dim0}) is out of bounds for tensor rank {self.rank()}.")
    if dim1 < 0 or dim1 >= self.rank():
        raise ValueError(f"dim1 ({dim1}) is out of bounds for tensor rank {self.rank()}.")

    # Create default permutation [0, 1, 2, ..., tensor_rank-1]
    perm = list(range(self.rank()))

    # Swap dim0 and dim1 in the permutation
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]

    assert self.link_module is not None, f"link_module for {self.name} not specified!!"
    op_name = f"{self.link_module.name}.transpose.impl_{next(counter)}"
    op = F.Transpose(op_name, perm=perm)
    op.set_module(self.link_module)
    self.link_module._op_hndls[op.name] = op
    if self.name not in self.link_module._tensors:
        self.link_module._tensors[self.name] = self
    return op(self)

def tensor_unsqueeze(self, dim):
    assert isinstance(self, SimTensor), f"unsqueeze self = {self} not a SimTensor!!"
    if self.rank() < 0: raise ValueError("Tensor rank must be at least 0.")

    # Handle negative indices (convert to positive)
    if dim < 0: dim = self.rank() + dim + 1

    # Validate dimension indices
    if dim < 0 or dim > self.rank():
        raise ValueError(f"dim ({dim}) is out of bounds for tensor rank {self.rank()}. "
                         f"Valid range is [-{self.rank()+1}, {self.rank()}]"
                         )
    assert self.link_module is not None, f"link_module for {self.name} not specified!!"
    op_name = f"{self.link_module.name}.unsqueeze.impl_{next(counter)}"
    axesTensor = F._from_data(op_name + '.axes', is_const=True, data=np.array([dim]))
    op = F.Unsqueeze(op_name)
    op.set_module(self.link_module)
    self.link_module._op_hndls[op.name] = op
    for x in [self, axesTensor]:
        if x.name not in self.link_module._tensors:
            self.link_module._tensors[x.name] = x
    return op(self, axesTensor)

def tensor_squeeze(self, dim):
    assert isinstance(self, SimTensor), f"unsqueeze self = {self} not a SimTensor!!"
    if self.rank() < 0: raise ValueError("Tensor rank must be at least 0.")

    if dim < 0: dim = self.rank() + dim

    # Validate dimension indices
    if dim < 0 or dim >= self.rank():
        raise ValueError(f"dim ({dim}) is out of bounds for tensor rank {self.rank()}. "
                         f"Valid range is [-{self.rank()}, {self.rank()-1}]"
                         )

    assert self.link_module is not None, f"link_module for {self.name} not specified!!"
    op_name = f"{self.link_module.name}.unsqueeze.impl_{next(counter)}"
    axesTensor = F._from_data(op_name + '.axes', is_const=True, data=np.array([dim]))
    op = F.Unsqueeze(op_name)
    op.set_module(self.link_module)
    self.link_module._op_hndls[op.name] = op
    for x in [self, axesTensor]:
        if x.name not in self.link_module._tensors:
            self.link_module._tensors[x.name] = x
    return op(self, axesTensor)

def tensor_getitem(self, idx):
    assert self.link_module is not None, f"link_module for {self.name} not specified!!"
    plan = torch2onnx_slice_plan(self.shape, idx)

    X = self
    op_num = next(counter)
    op_sub_num = 0

    # 1. Unsqueeze for new axes (from None in slice_spec)
    if plan['unsqueezes']:
        op_name   = f"{self.link_module.name}.slice.impl_{op_num}.{op_sub_num}"
        op        = F.Unsqueeze(op_name)
        op.set_module(self.link_module)
        self.link_module._op_hndls[op.name] = op
        op_sub_num += 1

        unsq_axes = F._from_data(op_name + '.axes', data=np.array(plan['unsqueezes'], dtype=np.int64), is_const=True)
        if unsq_axes not in self.link_module._tensors:
            self.link_module._tensors[unsq_axes.name] = unsq_axes

        X = op(X, unsq_axes)
        #print("HAPPY>>> 1", X)

    # 2. Gather+Squeeze for integer indices
    if plan['gathers']:
        for i, (axis, idx) in enumerate(plan['gathers']):
            assert 0 <= axis < X.rank(), f"Gather axis {axis} out of bounds for {X.shape}"
            assert 0 <= idx < X.shape[axis], f"Gather idx {idx} out of bounds for axis {axis} (shape={X.shape})"
            op_name   = f"{self.link_module.name}.slice.impl_{op_num}.{op_sub_num}"
            op        = F.Gather(op_name)
            op.set_module(self.link_module)
            self.link_module._op_hndls[op.name] = op
            op_sub_num += 1
            idx_tensor = F._from_data(op.name + '.idx', np.array([idx], dtype=np.int64), is_const=True)
            self.link_module._tensors[idx_tensor.name] = idx_tensor
            idx_tensor.set_module(self.link_module)
            X = op(X, idx_tensor)
            #print("HAPPY>>> 2", X)

            # Squeeze axis immediately after gather (axis always 0 after gather)
            #axes_tensor = oh.make_tensor(f'squeeze_axes_{i}', TensorProto.INT64, [1], [0])
            op_name   = f"{self.link_module.name}.slice.impl_{op_num}.{op_sub_num}"
            op        = F.Squeeze(op_name)
            op.set_module(self.link_module)
            self.link_module._op_hndls[op.name] = op
            op_sub_num += 1
            axes_tensor = F._from_data(op.name + '.axes', np.array([0], dtype=np.int64), is_const=True)
            self.link_module._tensors[axes_tensor.name] = axes_tensor
            axes_tensor.set_module(self.link_module)
            X = op(X, axes_tensor)
            #print("HAPPY>>> 3", X)

    # 3. Slice (if any)
    if plan['slice']:
        op_name   = f"{self.link_module.name}.slice.impl_{op_num}.{op_sub_num}"
        op        = F.SliceF(op_name, out_shape=plan['output_shape'])
        op.set_module(self.link_module)
        self.link_module._op_hndls[op.name] = op
        op_sub_num += 1

        s = plan['slice']
        starts_init = F._from_data('starts', data=np.array(s['starts'], dtype=np.int64), is_const=True)
        ends_init   = F._from_data('ends',   data=np.array(s['ends'],   dtype=np.int64), is_const=True)
        axes_init   = F._from_data('axes',   data=np.array(s['axes'],   dtype=np.int64), is_const=True)
        steps_init  = F._from_data('steps',  data=np.array(s['steps'],  dtype=np.int64), is_const=True)
        for t in [starts_init, ends_init, axes_init, steps_init]:
            if t not in self.link_module._tensors:
                self.link_module._tensors[t.name] = t

        X = op(X, starts_init, ends_init, axes_init, steps_init)
        #print("HAPPY>>> 4", X)

    return X

def tensor_repeat(self, *sizes):
    if len(sizes) != len(self.shape):
        raise ValueError(f"repeat expects {len(self.shape)} arguments, got {len(sizes)}!!")

    assert self.link_module is not None, f"link_module for {self.name} not specified!!"
    op_name = f"{self.link_module.name}.repeat.impl_{next(counter)}"
    op = F.Tile(op_name)
    op.set_module(self.link_module)
    self.link_module._op_hndls[op.name] = op
    repeatsT = F._from_data(op.name + '.repeats', data=np.array(sizes, dtype=np.int64), is_const=True)
    for x in [self, repeatsT]:
        if x.name not in self.link_module._tensors:
            self.link_module._tensors[x.name] = x
    return op(self, repeatsT)

def tensor_flatten(self, start_dim=0, end_dim=-1):
    shape = self.shape
    ndim  = self.rank()
    # Handle negative indices
    start_dim = start_dim if start_dim >= 0 else ndim + start_dim
    end_dim = end_dim if end_dim >= 0 else ndim + end_dim
    if not (0 <= start_dim <= end_dim < ndim):
        raise ValueError("Invalid start_dim or end_dim for flatten.")

    # Compute product of dimensions to flatten
    flat_size = 1
    for d in shape[start_dim:end_dim+1]:
        flat_size *= d

    # New shape: axes before start_dim, flat_size, axes after end_dim
    new_shape = shape[:start_dim] + [flat_size] + shape[end_dim+1:]

    assert self.link_module is not None, f"link_module for {self.name} not specified!!"
    op_name = f"{self.link_module.name}.flatten.impl_{next(counter)}"
    shapeTensor = F._from_data(op_name + '.shape', is_const=True, data=np.array(new_shape, dtype=np.int64))
    op = F.Reshape(op_name)
    op.set_module(self.link_module)
    self.link_module._op_hndls[op.name] = op
    for x in [self, shapeTensor]:
        if x.name not in self.link_module._tensors:
            self.link_module._tensors[x.name] = x
    return op(self, shapeTensor)

######## unary-tensor operations ##################
def unary_op(optype, ophndl_cls, self):
    assert isinstance(self, SimTensor), f"{optype} self = {self} not a SimTensor!!"
    assert self.link_module is not None, f"link_module for {self.name} not specified!!"
    op_name = f"{self.link_module.name}.{optype}.impl_{next(counter)}"
    assert op_name not in self.link_module._op_hndls, \
            f"Implicit op_name created via SimTensor.func_op not unique!! {op_name}"
    op = ophndl_cls(op_name)
    op.set_module(self.link_module)
    self.link_module._op_hndls[op.name] = op
    if self.name not in self.link_module._tensors:
            self.link_module._tensors[self.name] = self
    return op

def tensor_neg(self):
    o = unary_op('neg', F.Neg, self)
    return o(self)

def tensor_cos(self):
    o = unary_op('cos', F.Cos, self)
    return o(self)

def tensor_sin(self):
    o = unary_op('sin', F.Sin, self)
    return o(self)

def tensor_softmax(self, **kwargs):
    o = unary_op('softmax', F.Softmax, self)
    return o(self)

######## binary-tensor operations ##################
def binary_op(optype, ophndl_cls, self, arg):
    assert isinstance(self, SimTensor), f"{optype} self = {self} not a SimTensor!!"
    assert isinstance(arg, SimTensor), f"{optype} arg = {arg} not a SimTensor!!"
    use_link_module = self.link_module if self.link_module is not None else arg.link_module
    assert use_link_module is not None, f"link_module for {self.name} or {arg.name} not specified!!"
    op_name = f"{use_link_module.name}.{optype}.impl_{next(counter)}"
    assert op_name not in use_link_module._op_hndls, \
            f"Implicit op_name created via SimTensor.func_op not unique!! {op_name}"
    op = ophndl_cls(op_name)
    op.set_module(use_link_module)
    use_link_module._op_hndls[op.name] = op
    for x in (self, arg):
        if x.name not in use_link_module._tensors:
            use_link_module._tensors[x.name] = x
    return op

def tensor_add(self, x):
    o = binary_op('add', F.Add, self, x)
    return o(self, x)

def tensor_sub(self, x):
    o = binary_op('sub', F.Sub, self, x)
    return o(self, x)

def tensor_mul(self, x):
    o = binary_op('mul', F.Mul, self, x)
    return o(self, x)

def tensor_div(self, x):
    o = binary_op('div', F.Div, self, x)
    return o(self, x)

def tensor_pow(self, x):
    o = binary_op('pow', F.Pow, self, x)
    return o(self, x)

def matmul(A, B):
    o = binary_op('matmul', F.MatMul, A, B)
    C = o(A, B)
    return C

######## multi-tensor operations ##################
def cat(simtensor_list, dim=0):
    link_module = None
    for i,x in  enumerate(simtensor_list):
        assert isinstance(x, SimTensor), f"cat: input[{i}] = {x} not a SimTensor!!"
        if link_module is None and x.link_module is not None:
            link_module = x.link_module
    assert link_module is not None, f"cat: none of the input tensors link_module is specified!!"
    op_name = f"{link_module.name}.cat.impl_{next(counter)}"
    op = F.ConcatX(op_name, axis=dim)
    op.set_module(link_module)
    link_module._op_hndls[op.name] = op
    for x in simtensor_list:
        if x.name not in link_module._tensors:
            link_module._tensors[x.name] = x
    result = op(*simtensor_list)
    result.set_module(link_module)
    return result

def stack(simtensor_list, dim=0):
    if not simtensor_list:
        raise ValueError("simtensor_list must not be empty")
    base_shape = simtensor_list[0].shape
    base_dtype = simtensor_list[0].dtype
    for t in simtensor_list:
        if t.shape != base_shape:
            raise ValueError("All tensors must have the same shape to stack")
        if t.dtype != base_dtype:
            raise ValueError("All tensors must have the same dtype to stack")

    base_rank = simtensor_list[0].rank()
    if dim < 0:
        dim += base_rank + 1 #because a new axis will be added

    if dim < 0 or dim > base_rank:
        raise ValueError(f"dim {dim} is out of range for tensors of rank {base_rank}!!")

    link_module = None
    for tt in simtensor_list:
        if tt.link_module is not None:
            link_module = tt.link_module
            break
    assert link_module is not None, f"link_module not specified for any input tensors!!"

    op_num     = next(counter)
    op_sub_num = 0
    t_name = f"{link_module.name}.stack.unsqueeze.impl_{op_num}.{op_sub_num}.axesTensor"
    op_sub_num += 1
    axesTensor = F._from_data(t_name, is_const=True, data=np.array([dim]))
    link_module._tensors[axesTensor.name] = axesTensor
    axesTensor.set_module(link_module)

    outTensors = []
    for tt in simtensor_list:
        op_name = f"{link_module.name}.stack.unsqueeze.impl_{op_num}.{op_sub_num}"
        op_sub_num += 1
        op = F.Unsqueeze(op_name)
        op.set_module(link_module)
        link_module._op_hndls[op.name] = op
        if tt.name not in link_module._tensors:
            link_module._tensors[tt.name] = tt
        ott = op(tt, axesTensor)
        outTensors.append(ott)

    op_name = f"{link_module.name}.stack.concat.impl_{op_num}.{op_sub_num}"
    op = F.ConcatX(op_name, axis=dim)
    op.set_module(link_module)
    final_res = op(*outTensors)
    return final_res

def interpolate(self, **kwargs):
    # TODO:
    # minimal interpolate implementation as used in BEVDepth
    # enables ResizeOp in the backend via scale_factor
    # need to generalize this to enable torch.nn.functional.interpolate
    # which further requires the complete implementation of ResizeOp according
    # to ONNX Spec:https://onnx.ai/onnx/operators/onnx__Resize.html 

    assert self.link_module is not None, f"link_module not specified for any input tensor!!"

    scale_factor = kwargs.get('scale_factor', None)
    size         = kwargs.get('size', None)
    if scale_factor is not None:
        resize_scale_factor = scale_factor
    elif size is not None:
        resize_scale_factor = size
    else:
        raise ValueError(f"Need to specify either scale_factor={scale_factor} or size={size}")

    op_name = f"{self.link_module.name}.interpolate.impl_{next(counter)}"
    op = F.Resize(op_name, scale_factor=resize_scale_factor)
    op.set_module(self.link_module)
    return op(self)

# torch.matmul, triu, full, masked_fill_, zeros
SimTensor.__add__     = tensor_add         #type: ignore
SimTensor.__sub__     = tensor_sub         #type: ignore
SimTensor.__mul__     = tensor_mul         #type: ignore
SimTensor.__truediv__ = tensor_div         #type: ignore
SimTensor.__pow__     = tensor_pow         #type: ignore
SimTensor.__neg__     = tensor_neg         #type: ignore
SimTensor.__getitem__ = tensor_getitem     #type: ignore

SimTensor.cos         = tensor_cos         #type: ignore
SimTensor.sin         = tensor_sin         #type: ignore

SimTensor.view        = tensor_view        #type: ignore
SimTensor.reshape     = tensor_view        #type: ignore
SimTensor.transpose   = tensor_transpose   #type: ignore
SimTensor.unsqueeze   = tensor_unsqueeze   #type: ignore
SimTensor.squeeze     = tensor_squeeze     #type: ignore
SimTensor.contiguous  = tensor_contiguous  #type: ignore
SimTensor.flatten     = tensor_flatten     #type: ignore
SimTensor.repeat      = tensor_repeat      #type: ignore
SimTensor.softmax     = tensor_softmax     #type: ignore

#TODO:
# 0. Cleanup op-naming/link-module/tensor etc... (refactor)
# 2. Fix outshape in Slice (Onnx-like) implementation in ttsim/ops/op.py
# 3. Add tensor op tests (pytest)
