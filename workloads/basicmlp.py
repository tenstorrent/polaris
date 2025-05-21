#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
import numpy as np

class BasicMLP(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name           = name
        self.mm_dims        = cfg['mm_dims']
        self.bs             = cfg['bs']
        self.with_transpose = cfg.get('with_transpose', False)
        self.with_relu      = cfg.get('with_relu',      False)
        self.with_gelu      = cfg.get('with_gelu',      False)
        self.with_bias      = cfg.get('with_bias',      False)
        self.with_mul       = cfg.get('with_mul',       False)
        self.with_softmax   = cfg.get('with_softmax',   False)
        self.with_residual  = cfg.get('with_residual',  False)

        #ops
        dim_pairs = zip(self.mm_dims[0:-1], self.mm_dims[1:])
        myops = []
        for i, (M,N) in enumerate(dim_pairs):
            myops.append(F.Linear(self.name + f'.Linear{i}', M, N))
            if self.with_bias:
                myops.append(F.Bias(self.name + f'.Bias{i}', [N]))
            if self.with_transpose:
                myops.append(F.Transpose(self.name + f'.Transpose_0{i}', perm=[0,2,1]))
                myops.append(F.Transpose(self.name + f'.Transpose_1{i}', perm=[0,2,1]))
            if self.with_relu:
                myops.append(F.Relu(self.name + f'.Relu{i}'))
            if self.with_gelu:
                myops.append(F.Gelu(self.name + f'.Gelu{i}'))
            if self.with_mul:
                myops.append(F.MulFixed(self.name + f'.Mul{i}', 'PI', np.float32((i+1)*3.141592)))
            if self.with_softmax:
                myops.append(F.Softmax(self.name +'.Softmax{i}'))
        self.olist = F.SimOpHandleList(myops)
        if self.with_residual:
            self.residual = F.Add(self.name + f'.Residual')

        super().link_op2module()

    def set_batch_size(self, b):
        self.bs = b

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def create_input_tensors(self):
        self.input_tensors = {
                'x' : F._from_shape('x', [self.bs, 5, self.mm_dims[0]], is_param=False, np_dtype=np.float32),
                }
        return

    def __call__(self):
        X = self.input_tensors['x']
        if self.with_residual:
            return self.residual(self.olist(X), X)
        else:
            return self.olist(X)

def run_standalone(outdir: str='.') -> None:
    XXX = BasicMLP('BasicMLP', {'mm_dims'  : [32, 128, 256, 64, 10], 'bs': 1})
    XXX.set_batch_size(4)
    XXX.create_input_tensors()
    y = XXX()
    gg = XXX.get_forward_graph()
    onnx_ofilename = f'{outdir}/basic_mlp.onnx'
    gg.graph2onnx(onnx_ofilename)


if __name__ == '__main__':
    run_standalone()
