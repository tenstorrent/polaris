#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import sys, os
sys.path.append(os.getcwd())
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import numpy as np
import tests.common
from pathlib import Path
import pytest
from ttsim.ops import SimTensor
import logging

class ConvAndBatchnormTester(tests.common.SimOpTester):
    def __init__(self, name: str, cfgentry: dict):
        super().__init__(name, cfgentry)


    def __setup__(self, cfgentry: dict)->None:
        cfg: dict = cfgentry['cfg']
        self.in_channels: int = cfg['in_channels']
        self.out_channels: int = cfg['out_channels']
        self.kernel_size: int = cfg['kernel_size']
        self.h: int = cfg['h']
        self.w: int = cfg['w']
        self.bs: int = cfg['bs']
        unsupported_attributes = {k for k in cfg} - {'in_channels', 'out_channels', 'kernel_size', 'bs', 'h', 'w'}
        assert not unsupported_attributes, f'unsupported attributes {unsupported_attributes} for testbatchnorm'
        self.conv2d1 = F.Conv2d(self.name+'.conv', self.in_channels, self.out_channels, self.kernel_size)
        self.batchnorm1 = F.BatchNorm2d(self.name+'.batchnorm', self.out_channels)
        super().link_op2module()

    def set_batch_size(self, batch: int)->None:
        self.bs = batch

    def create_input_tensors(self)->None:
        self.input_tensors: dict[str, SimTensor] = {
            'x' : F._from_shape('x',
                                [self.bs, self.in_channels, self.h, self.w],
                                is_param=False, np_dtype=np.float32),
        }

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def __call__(self):  # Take care of all required arguments through attributes in the same object
        t1 = self.conv2d1(self.input_tensors['x'])
        t2 = self.batchnorm1(t1)
        return t2

    def analytical_param_count(self):
        return 10

@pytest.mark.unit
@pytest.mark.opunit
def test_conv_and_batchnorm(tmp_path_factory):
    testname: str = 'conv_and_batchnorm'

    configs: dict[str, dict[str, dict]] = {
        f'{testname}01': {
            'cfg': {'in_channels': 1, 'out_channels': 6, 'kernel_size': 5, 'bs': 4, 'h': 28, 'w': 28},
            'expected': {'shape': [4, 6, 24, 24]}
        }
    }
    outdir: Path = tmp_path_factory.mktemp('onnx')
    for cfgname, config in configs.items():
        cbtest: ConvAndBatchnormTester = ConvAndBatchnormTester('conv_batchnorm', config)
        cbtest.create_input_tensors()
        res = cbtest()
        assert res.shape == config['expected']['shape']
        gr = cbtest.get_forward_graph()
        fname = outdir / f'{testname}_{cfgname}.onnx'
        gr.graph2onnx(fname, do_model_check=True)

