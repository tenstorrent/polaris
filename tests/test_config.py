#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
import pytest
from ttsim.utils.common import parse_yaml, parse_csv
from ttsim.config import create_ipblock, create_package, parse_xlsx_config, \
    get_arspec_from_yaml, get_wlspec_from_yaml, get_wlmapspec_from_yaml
from ttsim.front.onnx.onnx2nx import onnx2graph

def test_xlsx_config():
    result = parse_xlsx_config('GPU@config/Nvidia.xlsx')
    a100 = result['A100']
    a100.normalize_param('sample-param')
    a100.value('target-power-w')
    with pytest.raises(AttributeError, match='undefined value'):
        a100.value('undef-attribute-1')
    a100.value('undef-attribute-2', defvalue=100)
    a100.set_value('dummy', 100)
    with pytest.raises(RuntimeError, match='already defined'):
        a100.set_value('dummy', 200)

def test_arspec_config():
    result = get_arspec_from_yaml('config/all_archs.yaml')
    nv = result[0]['nvidia_sm']
    assert nv.kind() == 'ComputeIP'
    ignore = str(nv)
    vec = nv.pipes['vector']
    mat = nv.pipes['matrix']
    vec.set_frequency(vec.frequency()+100)
    vec.peak_ipc('mac', 'fp16')
    vec.peak_flops('mac', 'fp16')
    nv.set_frequency(1000)

    gddr6 = result[0]['gddr6']
    gddr6.size()
    gddr6.frequency()
    gddr6.peak_bandwidth()
    a100 = result[1]['A100']
    a100.set_frequency(1000)
    a100.peak_ipc('matrix', 'mac', 'fp16')
    a100.peak_flops('matrix', 'mac', 'fp16')
    a100.ramp_penalty()
    a100.frequency('matrix')
    a100.mem_size()
    a100.peak_bandwidth()
    a100.mem_frequency()

def test_wlspec_config():
    result = get_wlspec_from_yaml('config/all_workloads.yaml')
    ttsim = next((wl for wlname, wl in result.items() if wl.api == 'TTSIM'))
    instances = ttsim.get_instances()


def test_wlmapspec_config():
    result = get_wlmapspec_from_yaml('config/wl2archmapping.yaml')
