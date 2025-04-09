#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import sys, os
sys.path.append(os.getcwd())
import ttsim.front.functional.op as funcop
from ttsim.ops.tensor import SimTensor
import numpy as np
import tests.common
from pathlib import Path
import pytest

class Batchnorm2dTester(tests.common.SimOpTester):
    def __init__(self, name: str, cfgentry: dict):
        super().__init__(name, cfgentry)

    def __setup__(self, cfgentry: dict):
        cfg: dict = cfgentry['cfg']
        self.channels: int = cfg['channels']
        self.h: int = cfg['h']
        self.w: int = cfg['w']
        self.bs: int = cfg['bs']
        unsupported_attributes: set = {k for k in cfg} - {'channels', 'bs', 'h', 'w'}
        assert not unsupported_attributes, f'unsupported attributes {unsupported_attributes} for testbatchnorm'
        self.batchnorm1 = funcop.BatchNorm2d(self.name+'.batchnorm', self.channels)

    def create_input_tensors(self):
        self.input_tensors: dict[str, SimTensor] = {
            'x' : funcop._from_shape('x',
                                [self.bs, self.channels, self.h, self.w],
                                is_param=False, np_dtype=np.float32),
        }

    def __call__(self):  # Take care of all required arguments through attributes in the same object
        t1 = self.batchnorm1(self.input_tensors['x'])
        return t1


@pytest.mark.unit
@pytest.mark.opunit
def test_batchnorm(tmp_path_factory):
    testname: str = 'batchnormtest'
    configs: dict[str, dict[str, dict]] = {
        f'{testname}01': {
            'cfg': {'channels': 6, 'bs': 4, 'h': 24, 'w': 24},
            'expected': {'shape': [4, 6, 24, 24]}
        },
    }
    outdir: Path = tmp_path_factory.mktemp('onnx')
    for cfgname, config in configs.items():
        btest: Batchnorm2dTester = Batchnorm2dTester('batchnorm', config)
        btest.create_input_tensors()
        res = btest()
        assert res.shape == config['expected']['shape']
        gr = btest.forward_graph()
        fname = outdir / f'{testname}_{cfgname}.onnx'
        gr.graph2onnx(fname, do_model_check=True)
        # TODO: Run this generated onnx through polaris

