#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys, os
sys.path.append(os.getcwd())
import ttsim.front.functional.op as funcop
from ttsim.ops.tensor import SimTensor
import numpy as np
import tests.common
from pathlib import Path
import pytest

class Maxpool2dTester(tests.common.SimOpTester):
    def __init__(self, name: str, cfgentry: dict):
        super().__init__(name, cfgentry)

    def __setup__(self, cfgentry: dict):
        cfg: dict = cfgentry['cfg']
        self.kernel_size: int = cfg['kernel_size']
        self.channels: int = cfg['channels']
        self.h: int = cfg['h']
        self.w: int = cfg['w']
        self.bs: int = cfg['bs']
        unsupported_attributes: set = {k for k in cfg} - {'kernel_size', 'channels', 'bs', 'h', 'w'}
        assert not unsupported_attributes, f'unsupported attributes {unsupported_attributes} for testMaxpool'
        self.Maxpool1 = funcop.MaxPool2d(self.name+'.Maxpool', self.kernel_size)

    def create_input_tensors(self):
        self.input_tensors: dict[str, SimTensor] = {
            'x' : funcop._from_shape('x',
                                [self.bs, self.channels, self.h, self.w],
                                is_param=False, np_dtype=np.float32),
        }

    def __call__(self):  # Take care of all required arguments through attributes in the same object
        t1 = self.Maxpool1(self.input_tensors['x'])
        return t1


@pytest.mark.unit
@pytest.mark.opunit
def test_Maxpool(tmp_path_factory):
    testname: str = 'Maxpooltest'
    configs: dict[str, dict[str, dict]] = {
        f'{testname}01': {
            'cfg': {'kernel_size': 3, 'channels': 6, 'bs': 4, 'h': 24, 'w': 24},
            'expected': {'shape': [4, 6, 8, 8]}
        },
    }
    outdir: Path = tmp_path_factory.mktemp('onnx')
    for cfgname, config in configs.items():
        btest: Maxpool2dTester = Maxpool2dTester('Maxpool', config)
        btest.create_input_tensors()
        res = btest()
        assert res.shape == config['expected']['shape']
        gr = btest.forward_graph()
        fname = outdir / f'{testname}_{cfgname}.onnx'
        gr.graph2onnx(fname, do_model_check=True)
        # TODO: Run this generated onnx through polaris

