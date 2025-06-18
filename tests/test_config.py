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
def test_arspec_config():
    result = get_arspec_from_yaml('config/all_archs.yaml')
    ipgroups, packages = result
    tensix = ipgroups.get_ipblock('tensix')
    assert tensix.iptype == 'compute'
    vec = tensix.get_pipe('vector')
    mat = tensix.get_pipe('matrix')
    old_freq = vec.freq_MHz
    vec.set_frequency(vec.frequency()+100)
    assert vec.freq_MHz == old_freq + 100
    assert vec.peak_ipc('mac', 'fp16') == 128
    assert vec.peak_flops('mac', 'fp16') == 140800 * (1000 ** -2)
    tensix.set_frequency(1000)

    gddr6 = ipgroups.get_ipblock('gddr6')
    assert gddr6.size() == 4
    assert gddr6.frequency() == 1250
    assert gddr6.peak_bandwidth() == 80
    q1a1 = packages['Q1_A1']
    q1a1.set_frequency(1000)
    assert q1a1.peak_ipc('matrix', 'mac', 'fp16') == 65536
    assert q1a1.peak_flops('matrix', 'mac', 'fp16') == 65.536
    assert q1a1.ramp_penalty() == 50
    assert q1a1.frequency('matrix') == 1000
    assert q1a1.mem_size() == 32
    assert q1a1.peak_bandwidth() == 2560
    assert q1a1.mem_frequency() == 1250

@pytest.mark.unit
def test_wlspec_config():
    result = get_wlspec_from_yaml('config/all_workloads.yaml')
    ttsim = next((wl[0] for wlname, wl in result.items() if wl[0].api == 'TTSIM'))
    instances = ttsim.get_instances()

@pytest.mark.unit
def test_wlmapspec_config():
    result = get_wlmapspec_from_yaml('config/wl2archmapping.yaml')
