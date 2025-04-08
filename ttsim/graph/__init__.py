#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Construct Backward Graph From a Forward Graph
import numpy as np
import logging

from .wl_graph import WorkloadGraph
from ttsim.ops.tensor import SimTensor

#TODO: Move to SimTensor as SimTensor.Grad()....
def CREATE_GRAD_TENSOR(t, set_data=False):
    if t.is_const == True:
        # need this because it is backward computation is easy
        # when we zip input and gradient tensors
        return None
    else:
        #gradient tensors should have same shape as orig
        gt = SimTensor({
            'name'    : t.name + '_grad',
            'shape'   : t.shape,
            'dtype'   : t.dtype,
            'is_param': t.is_param,
            'is_grad' : True
            })

        if set_data:
            if gt.rank() == 0:
                if gt.dtype == np.float32:
                    gt.data = np.float32(1.0)
                else:
                    assert False, "Only np.float32 rank-0 grad tensor supported right now!!!"
            else:
                gt.data = np.random.randn(*(gt.shape)).astype(gt.dtype)
        return gt

def COPY_FWD_TENSOR(t):
    #op_in/op_out = [] for easy visualization
    return SimTensor({
        'name'    : t.name,
        'shape'   : t.shape,
        'dtype'   : t.dtype,
        'data'    : t.data,
        'is_param': t.is_param,
        'is_const': t.is_const
        })

class BackwardWorkloadGraph:
    def __init__(self, fwd_graph):
        self._fwd_graph = fwd_graph
        self._bwd_graph = self.reverse()

    def reverse(self):
        #create input gradient tensors
        bwd_tensors  = {}
        for otname in self._fwd_graph.get_output_tensors():
            ot          = self._fwd_graph._tensors[otname]
            assert len(ot.op_in) == 0, f"output tensor {ot} has a source!!"
            if ot.has_grad == True:
                g_ot        = CREATE_GRAD_TENSOR(ot, set_data=True)
                bwd_tensors[g_ot.name] = g_ot

        logging.info("BWD INIT GRADS")
        for _,x in bwd_tensors.items(): logging.info('\t%s', x)

        ordered_nodes = self._fwd_graph.get_ordered_nodes()
        sorted_nodes  = reversed(ordered_nodes)
        bwd_ops       = {}
        for opnum, opname in enumerate(sorted_nodes):
            fwd_op        = self._fwd_graph._ops[opname]
            itensors      = [COPY_FWD_TENSOR(self._fwd_graph._tensors[x]) for x in fwd_op.inList]
            otensors      = [self._fwd_graph._tensors[x] for x in fwd_op.outList]
            grad_itensors = [bwd_tensors[x.name + '_grad'] for x in otensors if x.has_grad == True]
            grad_otensors = [CREATE_GRAD_TENSOR(x) for x in itensors]

            x_grad_results = fwd_op.backward(itensors, otensors, grad_itensors, grad_otensors)

            for x_i, (x_name, x_obj) in enumerate(x_grad_results.items()):
                assert x_name not in bwd_tensors, f"out grad tensor: {x_name} already in bwd_tensors"
                bwd_tensors[x_name] = x_obj._output_grad_tensor

                for x_tensor in x_obj._input_grad_tensors:
                    #special case handling for Reshape Orig-X-Shape put in as a new constant tensor
                    if fwd_op.optype == 'Reshape' and x_tensor.is_const == True and x_tensor.dtype == np.int64:
                        assert x_tensor.name not in bwd_tensors, \
                            f"in_grad_tensor: {x_tensor.name} for Reshape already in bwd_tensors!!"
                        bwd_tensors[x_tensor.name] = x_tensor
                    assert x_tensor.name in bwd_tensors, \
                            f"in_grad_tensor: {x_tensor.name} not in bwd_tensors!!\n{x_tensor}"

                for x_op in x_obj._grad_ops:
                    bwd_ops.update({x_op.name: x_op})
                for x_tensor in x_obj._input_fwd_tensors:
                    bwd_tensors.update({x_tensor.name: x_tensor})
                for x_tensor in x_obj._new_tensors:
                    bwd_tensors.update({x_tensor.name: x_tensor})


        bgg = WorkloadGraph(self._fwd_graph._name + '_bwd')

        for k,v in bwd_tensors.items():
            bgg.add_tensor(v)
        for k,v in bwd_ops.items():
            bgg.add_op(v)

        bgg.construct_graph()
        return bgg
