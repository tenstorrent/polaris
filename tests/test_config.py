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
    assert q1a1.peak_ipc('matrix', 'mac', 'fp16') == 262144
    assert q1a1.peak_flops('matrix', 'mac', 'fp16') == 262.144
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

@pytest.mark.unit
def test_archname_population():
    """Test that architecture package name (archname) is correctly populated from YAML to Device."""
    from ttsim.back.device import Device

    # Load architecture configuration
    ipgroups, packages = get_arspec_from_yaml('config/all_archs.yaml')

    # Test with Q1_A1 instance from Grendel package
    q1a1 = packages['Q1_A1']
    assert q1a1.devname == 'Grendel', f"Expected package name 'Grendel', got '{q1a1.devname}'"
    assert q1a1.name == 'Q1_A1', f"Expected instance name 'Q1_A1', got '{q1a1.name}'"

    # Create Device from package instance
    device = Device(q1a1)
    assert device.devname == 'Grendel', f"Device.devname should be 'Grendel', got '{device.devname}'"
    assert device.name == 'Q1_A1', f"Device.name should be 'Q1_A1', got '{device.name}'"

    # Test with n150 instance from Wormhole package (if available in config)
    if 'n150' in packages:
        n150 = packages['n150']
        assert n150.devname == 'Wormhole', f"Expected package name 'Wormhole', got '{n150.devname}'"
        assert n150.name == 'n150', f"Expected instance name 'n150', got '{n150.name}'"

        device_wh = Device(n150)
        assert device_wh.devname == 'Wormhole', f"Device.devname should be 'Wormhole', got '{device_wh.devname}'"
        assert device_wh.name == 'n150', f"Device.name should be 'n150', got '{device_wh.name}'"

@pytest.mark.unit
def test_archname_in_output():
    """Test that HLMStats correctly maps Device fields to output archname and devname."""
    from ttsim.back.device import Device
    from ttsim.stats.hlmstats import HLMStats
    from ttsim.front.onnx.onnx2nx import onnx2graph
    from ttsim.graph import WorkloadGraph

    # Load architecture configuration
    ipgroups, packages = get_arspec_from_yaml('config/all_archs.yaml')

    # Create a Device from Q1_A1 (Grendel package)
    q1a1 = packages['Q1_A1']
    device = Device(q1a1)

    # Create a minimal workload graph for testing
    graph = WorkloadGraph('TestGraph')

    # Create minimal workload info and stats info
    wlinfo = {
        'wlg': 'TTSIM',
        'wln': 'test_workload',
        'wli': 'test_instance',
        'wlb': 1
    }

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stats_info = {
            'flag_dump_stats_csv': False,
            'outputfmt': None,
            'stat_dir': tmpdir_path,
            'config_dir': tmpdir_path,
            'odir': tmpdir_path,
            'saved_devices': set()
        }

        # Create HLMStats instance
        hlm_stats = HLMStats(device, graph, wlinfo, stats_info)

        # Verify that HLMStats correctly maps the fields
        assert hlm_stats.archname == 'Grendel', f"HLMStats.archname should be 'Grendel', got '{hlm_stats.archname}'"
        assert hlm_stats.devname == 'Q1_A1', f"HLMStats.devname should be 'Q1_A1', got '{hlm_stats.devname}'"


@pytest.mark.unit
def test_memory_transfer_rate_validation():
    """Test that MemoryBlockModel validates transfer_rate_GTs matches calculated Tps."""
    from ttsim.config.simconfig import MemoryBlockModel
    from pydantic import ValidationError

    # Test case 1: Valid GDDR6 configuration
    gddr6_config = {
        'name': 'gddr6',
        'iptype': 'memory',
        'technology': 'GDDR',
        'data_bits': 32,
        'data_rate': 8,
        'freq_MHz': 1250,
        'size_GB': 4,
        'transfer_rate_GTs': 20.0  # 2 * (1250/1000) * 1 * 8 = 20.0
    }
    memory = MemoryBlockModel(**gddr6_config)
    assert memory.transfer_rate_GTs == 20.0

    # Test case 2: Valid HBM2 configuration
    hbm2_config = {
        'name': 'hbm2',
        'iptype': 'memory',
        'technology': 'HBM',
        'freq_MHz': 1200,
        'stacks': 8,
        'data_bits': 128,
        'size_GB': 8,
        'transfer_rate_GTs': 19.2  # 2 * (1200/1000) * 8 * 1 = 19.2
    }
    memory_hbm2 = MemoryBlockModel(**hbm2_config)
    assert memory_hbm2.transfer_rate_GTs == 19.2

    # Test case 3: Invalid configuration - mismatch in transfer_rate_GTs
    invalid_config = {
        'name': 'invalid_mem',
        'iptype': 'memory',
        'technology': 'GDDR',
        'data_bits': 32,
        'data_rate': 8,
        'freq_MHz': 1250,
        'size_GB': 4,
        'transfer_rate_GTs': 15.0  # Wrong! Should be 20.0
    }
    with pytest.raises(ValidationError) as exc_info:
        MemoryBlockModel(**invalid_config)
    assert 'transfer_rate_GTs' in str(exc_info.value)

    # Test case 4: Valid LPDDR5x configuration
    lpddr5x_config = {
        'name': 'lpddr5x',
        'iptype': 'memory',
        'technology': 'LPDDR',
        'data_bits': 16,
        'data_rate': 4,
        'freq_MHz': 1200,
        'size_GB': 4,
        'transfer_rate_GTs': 9.6  # 2 * (1200/1000) * 1 * 4 = 9.6
    }
    memory_lpddr = MemoryBlockModel(**lpddr5x_config)
    assert memory_lpddr.transfer_rate_GTs == 9.6

    # Test case 5: Default values for stacks and data_rate
    default_config = {
        'name': 'simple_mem',
        'iptype': 'memory',
        'technology': 'HBM',
        'freq_MHz': 1000,
        'data_bits': 128,
        'size_GB': 4,
        'transfer_rate_GTs': 2.0  # 2 * (1000/1000) * 1 * 1 = 2.0
    }
    memory_default = MemoryBlockModel(**default_config)
    assert memory_default.transfer_rate_GTs == 2.0
    assert memory_default.stacks == 1
    assert memory_default.data_rate == 1
