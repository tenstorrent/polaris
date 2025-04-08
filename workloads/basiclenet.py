#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import numpy as np
from ttsim.ops import SimTensor
import logging

class LeNet(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name       = name
        self.in_channels = cfg['in_channels']
        self.out_channels = cfg['out_channels']
        self.kernel_size = cfg['kernel_size']
        self.pooldim = cfg['pooldim']
        self.fcdim1 = cfg['fcdim1']
        self.fcdim2 = cfg['fcdim2']
        self.bs = cfg['bs']
        self.conv2d1 = F.Conv2d(self.name+'.conv', self.in_channels, self.out_channels, self.kernel_size)
        self.relu1 = F.Relu(self.name+'.relu')
        self.pool1 = F.MaxPool2d(self.name+'.pool', self.pooldim)
        self.reshape = F.ReshapeFixed(self.name+'.reshape', [self.bs, -1])
        self.linear = F.Linear(self.name+'.linear', self.fcdim1, self.fcdim2)
        super().link_op2module()


    def set_batch_size(self, batch):
        self.bs = batch

    def create_input_tensors(self):
        self.input_tensors = {
            'x' : F._from_shape('x',  [self.bs, self.in_channels, 28, 28], is_param=False, np_dtype=np.float32),
        }
        logging.debug('input tensors: %s', self.input_tensors['x'])
        return

    def get_forward_graph(self):
        #need call forwarding because the base class expects
        # to get the input tensors as input
        GG = super()._get_forward_graph(self.input_tensors)
        return GG


    def __call__(self):  # Take care of all required arguments through attributes in the same object
        t1 = self.conv2d1(self.input_tensors['x'])
        t2 = self.relu1(t1)
        t3 = self.pool1(t2)
        t4 = self.reshape(t3)
        t5 = self.linear(t4)
        return t5

    def analytical_param_count(self):
        return 10

def run_standalone(outdir: str ='.')->None:
    configs = {
        '1654': {'in_channels': 1, 'out_channels': 6, 'kernel_size': 5, 'bs': 4, 'pooldim': 2, 'fcdim1': 864, 'fcdim2': 120},
    }
    for cfgname, config in configs.items():
        XXX = LeNet('LeNet', config)
        # XXX.set_batch_size(4)
        XXX.create_input_tensors()
        XXX()
        gr = XXX.get_forward_graph()
        gr.graph2onnx(f'{outdir}/testlenet_{cfgname}.onnx', do_model_check=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s")
    run_standalone(outdir='.')
