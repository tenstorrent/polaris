#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .tensor import SimTensor
from .desc.registry import get_opdesc_registry
import ttsim.utils.common as common

from typing import TYPE_CHECKING, Any, Union
import numpy as np

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
        self.exec_stats              = None
        self._kw_args_defaults       = {}

    def __str__(self):
        s  = f"SimOp({self.name}) optype={self.optype}, cls={self.opclass_str}, "
        s += f"prec={self.precision}, attrs={self.attrs}, domain={self.domain}, "
        s += f"rpt={self.repeat_count}, "
        s += f"removed={self.removed_in_optimization}, "
        s += f"fused={self.fused_in_optimization}, "
        s += f"fused_with_op={self.fused_with_op}, "
        s += f"uses_compute_pipe={self.uses_compute_pipe}, "
        s += f"inList={self.inList}, "
        s += f"outList={self.outList}"
        return s

    def check_known_args(self, args) -> None:
        common.check_known_args(str(type(self)), args=args,
                                default_args=self._kw_args_defaults)

    def get_effective_args(self, args: dict[str, Any]) -> dict[str, Any]:
        return common.get_kwargs_with_defaults(str(type(self)),
                                               args=args,
                                               default_args=self._kw_args_defaults)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        opdesc = get_opdesc_registry().get_opdesc(self.optype)

        #check in/out counts
        in_range  = range(opdesc['min_input'], opdesc['max_input']+1)
        out_range = range(opdesc['min_output'], opdesc['max_output']+1)
        assert len(inT) in in_range,   f"#inputs for {self} operator should be in {in_range}, is {len(inT)}"
        assert len(outT) in out_range, f"#outputs for {self} operator should be in {out_range}, is {len(outT)}"

        #Do Shape Inference, Update perf_stats
        shape_inf_func = opdesc['shape_inf_func']
        shape_inf_func(inT, outT, self, **kwargs)

        return self.perf_stats

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

    def get_effective_precision(self, tensor: SimTensor) -> np.dtype:
        """
        Get the effective precision for a tensor based on the op's precision.
        If the op's precision is not set, return the tensor's dtype.
        """
        if self.precision is not None:
            return self.precision
        assert tensor.dtype is not None, f"Tensor {tensor.name} has no dtype set"
        return tensor.dtype
