#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

###############################################################
# Poor Man's Module/ModuleList inspired by PyTorch Signature
###############################################################
from typing import Iterator
import ttsim.front.functional.op as F
import ttsim.ops.op as Ops
from ttsim.ops import SimTensor
from ttsim.graph import WorkloadGraph


class Module:
    # Type declarations for INSTANCE attributes
    name: str

    def __init__(self):
        self._tensors = {}
        self._op_hndls = {}
        self._submodules = {}

    def __setattr__(self, name, value):
        if isinstance(value, SimTensor):
            self._tensors[name] = value
        elif isinstance(
            value,
            (
                F.SimOpHandle,
                F.SplitOpHandle,
                F.VariadicInputOpHandle,
                F.MultiOutputSimOpHandle,
            ),
        ):
            # IMPLICIT_INPUTS are not constructed till SimOpHandle::__call__
            # is executed; so, we need to do this after the __call__ is done :-(
            # For an example of this: check SplitOpHandle::__call__
            self._op_hndls[name] = value
            if hasattr(value, "params") and len(value.params) > 0:
                for _, ptensor in value.params:
                    self._tensors[ptensor.name] = ptensor
        elif isinstance(value, F.SimOpHandleList):
            for o in value:
                self._op_hndls[o.name] = o
                if len(o.params) > 0:
                    for _, ptensor in o.params:
                        self._tensors[ptensor.name] = ptensor
        elif isinstance(value, Module):
            self._submodules[name] = value
        elif isinstance(value, ModuleList):
            for m in value:
                self._submodules[name + "." + m.name] = m
        else:
            pass
        super().__setattr__(name, value)

    def link_op2module(self):
        for _op_name, _op in self._op_hndls.items():
            _op.set_module(self)
        for k, v in self._submodules.items():
            v.link_op2module()
        return

    def create_intermediate_tensor(self, tname):
        return SimTensor({"name": self.name + "." + tname})

    def create_data_tensor(self, tname, /, data, is_param=False, is_const=False):
        return F._from_data(
            self.name + "." + tname, data, is_param=is_param, is_const=is_const
        )

    def create_shape_tensor(self, tname, /, shape, is_param=False, is_const=False):
        return F._from_shape(
            self.name + "." + tname, shape, is_param, is_const=is_const
        )

    def get_tensors(self, tbl):
        # v.imp note: attributes across instances have same names, so we should not
        # use the attr-name to accumulate ALL tensors across submodules...
        for k, v in self._tensors.items():
            tbl[v.name] = v
        for k, v in self._op_hndls.items():
            if len(v.implicit_inputs) > 0:
                itensor = v.implicit_inputs[0]
                tbl[itensor.name] = itensor
        for k, v in self._submodules.items():
            v.get_tensors(tbl)
        return tbl

    def get_ops(self, tbl: dict):
        # v.imp note: attributes across instances have same names, so we should not
        # use the attr-name to accumulate ALL ops across submodules...
        for k, v in self._op_hndls.items():
            if v.sim_op is not None:
                # sometimes we use ops conditionally in workloads, when it is not known at
                # construction time whether a SimOpHandle.__call__() will be invoked
                # in those cases, sim_op is None
                tbl[v.sim_op.name] = v.sim_op
        for k, v in self._submodules.items():
            v.get_ops(tbl)
        return tbl

    def _get_forward_graph(self, input_tensors):
        # Intended to be called only from subclasses
        # Get Tensors...
        ttbl = {}
        for tname, t in input_tensors.items():
            if isinstance(t, list):
                for ii, tt in enumerate(t):
                    assert isinstance(
                        tt, SimTensor
                    ), f"{tname}[{ii}] not a SimTensor!!\n{tt}"
                    ttbl[tt.name] = tt
            elif isinstance(t, SimTensor):
                ttbl[t.name] = t
            else:
                assert (
                    False
                ), f"input_tensor {tname} should be an instance of (SimTensor|List[SimTensor])!!\n{t}"

        self.get_tensors(ttbl)

        # Get Ops...
        otbl: dict = {}
        self.get_ops(otbl)

        # Graph Construction...
        gg = WorkloadGraph(self.name)

        # Add Tensors to Graph...
        for _, tensor in ttbl.items():
            gg.add_tensor(tensor)

        # Add Ops to Graph...
        for _, op in otbl.items():
            gg.add_op(op)

        # Construct Graph
        gg.construct_graph()

        return gg

    def __str__(self, indent_width=0):
        indent0 = " " * indent_width * 4
        indent1 = " " * (indent_width + 1) * 4
        indent2 = " " * (indent_width + 2) * 4
        s = f"{indent0}MODULE: {self.name}\n"
        s += f"{indent0}TENSORS:\n"
        for k, v in self._tensors.items():
            s += f"{indent1}{k}:{v}\n"

        s += f"{indent0}OPS:\n"
        for k, v in self._op_hndls.items():
            s += f"{indent1}{k}:{v.sim_op}\n"
            if hasattr(v, "params") and len(v.params) > 0:
                s += f"{indent2}PARAMS:\n"
                for _, ptensor in v.params:
                    s += f"{indent2}{ptensor}\n"
            if len(v.implicit_inputs) > 0:
                s += f"{indent2}IMPLICIT_INPUTS:\n"
                for itensor in v.implicit_inputs:
                    s += f"{indent2}{itensor}\n"

        s += f"{indent0}SUBMODULES:\n"
        for k, v in self._submodules.items():
            s += f"{indent1}{k}:\n"
            s += v.__str__(indent_width + 1)
        return s

    def __call__(self, *args, **kwargs):
        # "Pure Virtual" function, should never get called.
        # Defined to ensure Module class is "callable", and ensure no static type check fails (mypy)
        print(f"__call__() not implemented for {self.__class__.__name__}::{self.name}")
        raise AssertionError


class ModuleList:
    def __init__(self, modules):
        self._modules_in_list = {}

        assert len(modules) > 0, f"Empty ModuleList at construction!!"

        for i, module in enumerate(modules):
            assert module is not None, f"'None' module passed to ModuleList"
            assert isinstance(module, Module), f"{module} is not a Module subclass"
            self._modules_in_list[str(i)] = module

        # check all module names in the list are unique...
        assert len(self) == len(
            set(self._modules_in_list)
        ), f"Module Names in ModuleList are not unique : {[m.name for m in self._modules_in_list.values()]}!!"

    def __len__(self):
        return len(self._modules_in_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [
                self._modules_in_list[str(i)] for i in range(*idx.indices(len(self)))
            ]
        elif isinstance(idx, int):
            idx = idx + len(self) if idx < 0 else idx
            if idx < 0 or idx >= len(self):
                raise IndexError(f"out-of-bound-index: {idx}")
            return self._modules_in_list[str(idx)]
        else:
            raise TypeError(f"Invalid index Type: {type(idx)}")

    def __iter__(self) -> Iterator[Module]:
        for i in range(len(self)):
            yield self[i]

    # we want to make this immutable after construction...
    # so restricting setitem / delitem / append / insert / extend
    def __setitem__(self, idx, module):
        raise RuntimeError("ModuleList is immutable after construction")

    def __delitem__(self, idx):
        raise RuntimeError("ModuleList is immutable after construction")

    def append(self, module):
        raise RuntimeError("ModuleList is immutable after construction")

    def extend(self, modules):
        raise RuntimeError("ModuleList is immutable after construction")

    def insert(self, index, module):
        raise RuntimeError("ModuleList is immutable after construction")

    def __call__(self, *x):  # type: ignore[override]
        raise RuntimeError("ModuleList is not Callable")


class Sequential(Module):
    """Sequential container — calls submodules in order (mirrors nn.Sequential).

    The name is derived from the first module's parent namespace: e.g. if the
    first module is named ``"backbone.layer0.conv"``, the Sequential gets the
    name ``"backbone.layer0"``.
    """

    def __init__(self, modules: list):
        super().__init__()
        assert len(modules) > 0, "Empty Sequential at construction"

        # Derive a stable, human-readable name from the first module
        first_name: str = modules[0].name if hasattr(modules[0], "name") else "seq"
        self.name = (
            first_name.rsplit(".", 1)[0] if "." in first_name else first_name + "_seq"
        )

        # Keep ordered list for __call__; register each child correctly so
        # get_ops / get_tensors / link_op2module traverse them.
        self._sequential: list = modules
        for m in modules:
            if not hasattr(m, "name"):
                continue
            if isinstance(m, Module):
                self._submodules[m.name] = m
            elif isinstance(
                m,
                (
                    F.SimOpHandle,
                    F.SplitOpHandle,
                    F.VariadicInputOpHandle,
                    F.MultiOutputSimOpHandle,
                ),
            ):
                self._op_hndls[m.name] = m
                if hasattr(m, "params"):
                    for _, ptensor in m.params:
                        self._tensors[ptensor.name] = ptensor
        super().link_op2module()

    def __call__(self, x):  # type: ignore[override]
        for m in self._sequential:
            x = m(x)
        return x


############## Specific Modules ################
class Linear(Module):
    def __init__(self, name, in_features, out_features, bias=True):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.matmul = F.MatMul(name + ".matmul")
        self.transpose = F.Transpose(name + ".transpose", perm=[1, 0])
        self.param = F._from_shape(
            name + ".param", [out_features, in_features], is_param=True
        )
        self.bias = (
            F._from_shape(name + ".bias", [out_features], is_param=True)
            if bias
            else None
        )
        self.param.set_module(self)
        if bias:
            self.bias.set_module(self)  # type: ignore
        super().link_op2module()

    def __call__(self, x):
        param_t = self.transpose(self.param)
        Y = self.matmul(x, param_t)
        if self.bias is not None:
            Y += self.bias
        return Y

    def analytical_param_count(self, lvl):
        param_count = self.in_features * self.out_features
        if self.bias:
            param_count += self.out_features
        return param_count


class Dropout(Module):
    """
    Dropout module that wraps F.Dropout SimOpHandle.

    Acts as identity when prob=0 or train_mode=False.
    Uses compute_dropout from data_compute for numerical computation.

    Args:
        name: Module name for tracking
        prob: Dropout probability (default: 0.0)
        train_mode: Whether in training mode (default: False)

    Example:
        >>> dropout = Dropout('my_dropout', prob=0.1, train_mode=False)
        >>> output = dropout(x)
    """

    def __init__(self, name, prob=0.0, train_mode=False):
        super().__init__()
        self.name = name
        self.prob = prob
        self.train_mode = train_mode
        self.dropout_op = F.Dropout(name + ".dropout", prob=prob, train_mode=train_mode)
        super().link_op2module()

    def __call__(self, x):
        return self.dropout_op(x)

    def analytical_param_count(self, lvl):
        return 0


class Silu(Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.sigmoidop = F.Sigmoid(name + ".sigmoid")
        super().link_op2module()

    def __call__(self, x):
        return x * self.sigmoidop(x)

    def analytical_param_count(self, lvl):
        return 0


class bmm(Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        super().link_op2module()

    def __call__(self, a, b):
        batch_shape = Ops.get_tensor_broadcast_shape(a.shape[:-2], b.shape[:-2])
        m, k1 = a.shape[-2], a.shape[-1]
        k2, n = b.shape[-2], b.shape[-1]
        out_shape = batch_shape + [m, n]

        for n in range(a.shape[0]):
            matmulop = F.MatMul(self.name + f".matmul_{n}")
            matmulop.set_module(self)
            self._op_hndls[matmulop.name] = matmulop
            input_1 = F._from_shape(
                self.name + f".input1_{n}", [a.shape[-2], a.shape[-1]], is_const=True
            )
            input_2 = F._from_shape(
                self.name + f".input2_{n}", [b.shape[-2], b.shape[-1]], is_const=True
            )
            self._tensors[input_1.name] = input_1
            self._tensors[input_2.name] = input_2
            matmulop(input_1, input_2)
        out = F._from_shape(self.name + ".out", out_shape)
        out.set_module(self)
        self._tensors[out.name] = out
        return out

    def analytical_param_count(self, lvl):
        return 0


class GroupNorm(Module):
    def __init__(self, name, num_groups=1, num_channels=1, eps=1e-5, affine=True):
        super().__init__()
        self.name = name
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = F._from_shape(name + ".weight", [num_channels], is_param=True)
        self.bias = F._from_shape(name + ".bias", [num_channels], is_param=True)
        self.gn_op = F.GroupNormalization(
            self.name + ".gn",
            num_groups=self.num_groups,
            num_channels=self.num_channels,
            eps=self.eps,
        )
        super().link_op2module()

    def __call__(self, x, latent_embeds=None):
        return self.gn_op(x, self.weight, self.bias)

    def analytical_param_count(self, lvl):
        return 2 * self.num_channels


class MultiheadAttention(Module):
    """
    Multi-head attention mechanism.

    Implements scaled dot-product attention with multiple attention heads.
    Compatible with PyTorch's nn.MultiheadAttention interface.

    Args:
        name: Module name for tracking
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability (default: 0.0)
        bias: If True, add bias to input/output projection layers (default: True)
        add_bias_kv: If True, add bias to key and value sequences (default: False)
        add_zero_attn: If True, add a new batch of zeros to key and value (default: False)
        kdim: Total number of features in key (default: None, uses embed_dim)
        vdim: Total number of features in value (default: None, uses embed_dim)
        batch_first: If True, input/output shape is (batch, seq, feature) (default: False)
    """

    def __init__(self, name, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.name = name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout
        self.use_bias = bias

        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        self.head_dim = embed_dim // num_heads

        # Q, K, V projection weights (combined for efficiency)
        self.in_proj_weight = F._from_shape(
            name + ".in_proj_weight", [3 * embed_dim, embed_dim], is_param=True
        )
        self.in_proj_weight.set_module(self)

        self.in_proj_bias: SimTensor | None = None
        if bias:
            self.in_proj_bias = F._from_shape(
                name + ".in_proj_bias", [3 * embed_dim], is_param=True
            )
            self.in_proj_bias.set_module(self)

        # Output projection
        self.out_proj = Linear(name + ".out_proj", embed_dim, embed_dim, bias=bias)
        self._submodules[self.out_proj.name] = self.out_proj

        super().link_op2module()

    def __call__(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=True,
    ):
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor [*, tgt_len, embed_dim]
            key: Key tensor [*, src_len, embed_dim]
            value: Value tensor [*, src_len, embed_dim]
            key_padding_mask: Mask for padded keys [batch, src_len] (optional)
            attn_mask: Attention mask [tgt_len, src_len] (optional)
            need_weights: Whether to return attention weights (default: True)

        Returns:
            output: [*, tgt_len, embed_dim]
            or
            (output, attn_weights): if need_weights=True
                attn_weights: [*, num_heads, tgt_len, src_len]
        """
        # Build input list for the operation
        input_list = [query, key, value]
        if key_padding_mask is not None:
            input_list.append(key_padding_mask)
        if attn_mask is not None:
            input_list.append(attn_mask)

        # Create ipos based on number of inputs (all are tensors)
        ipos = list(range(len(input_list)))

        # Create the multi-head attention operation
        mha_op_name = self.name + ".mha_op"

        # Pass projection weight data so compute_multihead_attention can apply them
        extra_attrs = {}
        if self.in_proj_weight is not None and self.in_proj_weight.data is not None:
            extra_attrs["in_proj_weight_data"] = self.in_proj_weight.data
        if self.in_proj_bias is not None and self.in_proj_bias.data is not None:
            extra_attrs["in_proj_bias_data"] = self.in_proj_bias.data
        if hasattr(self, "out_proj") and self.out_proj is not None:
            if (
                hasattr(self.out_proj, "param")
                and self.out_proj.param is not None
                and self.out_proj.param.data is not None
            ):
                extra_attrs["out_proj_weight_data"] = self.out_proj.param.data
            if (
                hasattr(self.out_proj, "bias")
                and self.out_proj.bias is not None
                and self.out_proj.bias.data is not None
            ):
                extra_attrs["out_proj_bias_data"] = self.out_proj.bias.data

        mha_op = F.SimOpHandle(
            mha_op_name,
            "MultiheadAttention",
            params=[],
            ipos=ipos,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_prob,
            **extra_attrs,
        )
        mha_op.set_module(self)
        self._op_hndls[mha_op_name] = mha_op

        # Call the operation
        # Note: The operation may return a tuple (output, attn_weights) or just output
        result = mha_op(*input_list)

        # If need_weights is False but the op still returns a tuple, extract first element
        if need_weights:
            if isinstance(result, tuple):
                return result  # (output, attn_weights)
            else:
                # If only output is returned, create a placeholder for attn_weights
                return result, None
        else:
            if isinstance(result, tuple):
                return result[0]  # Extract just the output
            else:
                return result

    def analytical_param_count(self, lvl):
        """Calculate number of parameters in this module."""
        # in_proj: embed_dim * 3 * embed_dim
        # out_proj: embed_dim * embed_dim
        param_count = self.embed_dim * 3 * self.embed_dim  # Q, K, V projections
        param_count += self.embed_dim * self.embed_dim  # Output projection

        if self.use_bias:
            param_count += 3 * self.embed_dim  # Q, K, V biases
            param_count += self.embed_dim  # Output bias

        return param_count


class L1Loss(Module):
    """L1Loss (Mean Absolute Error) — not an ONNX op, defined as a SimNN module.

    Computes: loss = mean(|prediction - target|)  (when reduction='mean')

    Args:
        name: Module name for tracking
        reduction: 'none' | 'mean' | 'sum'  (default: 'mean')
    """

    def __init__(self, name, reduction="mean"):
        super().__init__()
        self.name = name
        self.reduction = reduction
        self.sub_op = F.Sub(name + ".sub")
        self.abs_op = F.Abs(name + ".abs")
        if reduction == "mean":
            self.reduce_op = F.ReduceMean(name + ".reduce_mean")
        elif reduction == "sum":
            self.reduce_op = F.ReduceSum(name + ".reduce_sum")
        else:
            self.reduce_op = None
        super().link_op2module()

    def __call__(self, prediction, target):
        diff = self.sub_op(prediction, target)
        abs_diff = self.abs_op(diff)
        if self.reduce_op is not None:
            return self.reduce_op(abs_diff)
        return abs_diff

    def analytical_param_count(self, lvl):
        return 0

