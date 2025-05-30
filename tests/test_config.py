#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
from ttsim.utils.common import parse_yaml, parse_csv
from ttsim.config import parse_xlsx_config, \
    get_arspec_from_yaml, get_wlspec_from_yaml, get_wlmapspec_from_yaml
from ttsim.front.onnx.onnx2nx import onnx2graph
from math import exp

@pytest.mark.unit
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

@pytest.mark.unit
def test_arspec_config():
    result = get_arspec_from_yaml('config/all_archs.yaml')
    ipgroups, packages = result
    nv = ipgroups.get_ipblock('nvidia_sm')
    assert nv.iptype == 'compute'
    vec = nv.get_pipe('vector')
    mat = nv.get_pipe('matrix')
    old_freq = vec.freq_MHz
    vec.set_frequency(vec.frequency()+100)
    assert vec.freq_MHz == old_freq + 100
    assert vec.peak_ipc('mac', 'fp16') == 128
    assert vec.peak_flops('mac', 'fp16') == 193280 * (1000 ** -2)
    nv.set_frequency(1000)

    gddr6 = ipgroups.get_ipblock('gddr6')
    assert gddr6.size() == 4
    assert gddr6.frequency() == 1250
    assert gddr6.peak_bandwidth() == 80
    a100 = packages['A100']
    a100.set_frequency(1000)
    assert a100.peak_ipc('matrix', 'mac', 'fp16') == 110592
    assert a100.peak_flops('matrix', 'mac', 'fp16') == 110.592
    assert a100.ramp_penalty() == 100
    assert a100.frequency('matrix') == 1000
    assert a100.mem_size() == 40
    assert a100.peak_bandwidth() == 1555.2
    assert a100.mem_frequency() == 1215

@pytest.mark.unit
def test_wlspec_config():
    result = get_wlspec_from_yaml('config/all_workloads.yaml')
    ttsim = next((wl for wlname, wl in result.items() if wl.api == 'TTSIM'))
    instances = ttsim.get_instances()

@pytest.mark.unit
def test_wlmapspec_config():
    result = get_wlmapspec_from_yaml('config/wl2archmapping.yaml')
