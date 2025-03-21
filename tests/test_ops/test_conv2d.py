import sys, os
sys.path.append(os.getcwd())
import ttsim.front.functional.op as funcop
import numpy as np
import tests.common
from pathlib import Path
import pytest
from typing import TYPE_CHECKING

class Conv2dTester(tests.common.SimOpTester):
    def __init__(self, name: str, cfgentry: dict):
        super().__init__(name, cfgentry)

    def __setup__(self, cfgentry: dict):
        cfg: dict = cfgentry['cfg']
        self.in_channels: int = cfg['in_channels']
        self.out_channels: int = cfg['out_channels']
        self.kernel_size: int = cfg['kernel_size']
        self.bs: int = cfg['bs']
        unsupported_attributes: set = {k for k in cfg} - {'in_channels', 'out_channels', 'kernel_size', 'bs'}
        assert not unsupported_attributes, f'unsupported attributes {unsupported_attributes} for testbatchnorm'
        self.conv2d1 = funcop.Conv2d(self.name+'.conv', self.in_channels, self.out_channels, self.kernel_size)

    def create_input_tensors(self):
        self.input_tensors = {
            'x' : funcop._from_shape('x',
                                [self.bs, self.in_channels, 28, 28],
                                is_param=False, np_dtype=np.float32),
        }

    def __call__(self):  # Take care of all required arguments through attributes in the same object
        if TYPE_CHECKING:
            assert self.input_tensors is not None
        t1 = self.conv2d1(self.input_tensors['x'])
        # t2 = self.batchnorm1(t1)
        return t1


@pytest.mark.unit
@pytest.mark.opunit
def test_conv(tmp_path_factory):
    testname: str = 'convtest'
    configs: dict[str, dict[str, dict]] = {
        f'{testname}01': {
            'cfg': {'in_channels': 1, 'out_channels': 6, 'kernel_size': 5, 'bs': 4},
            'expected': {'shape': [4, 6, 24, 24]}
        },
    }
    outdir: Path = tmp_path_factory.mktemp('onnx')
    for cfgname, config in configs.items():
        btest: Conv2dTester = Conv2dTester('conv', config)
        btest.create_input_tensors()
        res = btest()
        assert res.shape == config['expected']['shape']
        gr = btest.forward_graph()
        fname = outdir / f'{testname}_{cfgname}.onnx'
        gr.graph2onnx(fname, do_model_check=True)
        # TODO: Run this generated onnx through polaris

