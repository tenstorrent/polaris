#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

import pytest

from ttsim.back.device import Device
from ttsim.config.simconfig import (ComputeBlockModel, ComputeInsnModel, ComputePipeModel, IPGroupComputeModel,
                                    IPGroupMemoryModel, MemoryBlockModel, PackageInstanceModel)
from ttsim.graph.wl_graph import WorkloadGraph
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import SimTensor


@pytest.mark.unit
def test_peak_bandwidth_per_cycle():
    """Test peak_bandwidth_per_cycle() with various memory configurations."""
    # Test case 1: Single stack, SDR (data_rate=1), 32-bit data
    mem1 = MemoryBlockModel(
        name='test_mem1',
        iptype='memory',
        technology='GDDR',
        data_bits=32,
        freq_MHz=1000,
        size_GB=2,
        stacks=1,
        data_rate=1
    )
    # Expected: 2 * 1 * 1 * 32 / 8 = 8 bytes per cycle
    assert mem1.peak_bandwidth_per_cycle() == 8.0

    # Test case 2: Single stack, DDR (data_rate=2), 32-bit data
    mem2 = MemoryBlockModel(
        name='test_mem2',
        iptype='memory',
        technology='GDDR',
        data_bits=32,
        freq_MHz=1000,
        size_GB=2,
        stacks=1,
        data_rate=2
    )
    # Expected: 2 * 1 * 2 * 32 / 8 = 16 bytes per cycle
    assert mem2.peak_bandwidth_per_cycle() == 16.0

    # Test case 3: Multiple stacks (4), DDR (data_rate=2), 32-bit data
    mem3 = MemoryBlockModel(
        name='test_mem3',
        iptype='memory',
        technology='GDDR',
        data_bits=32,
        freq_MHz=1000,
        size_GB=2,
        stacks=4,
        data_rate=2
    )
    # Expected: 2 * 4 * 2 * 32 / 8 = 64 bytes per cycle
    assert mem3.peak_bandwidth_per_cycle() == 64.0

    # Test case 4: Single stack, DDR, 64-bit data
    mem4 = MemoryBlockModel(
        name='test_mem4',
        iptype='memory',
        technology='GDDR',
        data_bits=64,
        freq_MHz=1000,
        size_GB=2,
        stacks=1,
        data_rate=2
    )
    # Expected: 2 * 1 * 2 * 64 / 8 = 32 bytes per cycle
    assert mem4.peak_bandwidth_per_cycle() == 32.0


@pytest.mark.unit
def test_peak_bandwidth_per_cycle_package():
    """Test peak_bandwidth_per_cycle() at package level with multiple memory units."""
    mem_block = MemoryBlockModel(
        name='test_mem',
        iptype='memory',
        technology='GDDR',
        data_bits=32,
        freq_MHz=1000,
        size_GB=2,
        stacks=1,
        data_rate=2
    )

    # Create IP group with 4 memory units
    ipgroup_mem = IPGroupMemoryModel(
        ipname='test_mem_group',
        iptype='memory',
        num_units=4,
        ipobj=mem_block
    )

    # Create minimal compute group for package
    compute_pipe = ComputePipeModel(
        name='matrix',
        num_units=1,
        freq_MHz=1000,
        instructions=[ComputeInsnModel(name='mac', tpt={'fp16': 512})]
    )
    compute_block = ComputeBlockModel(
        name='tensix',
        iptype='compute',
        pipes=[compute_pipe]
    )
    ipgroup_comp = IPGroupComputeModel(
        ipname='tensix',
        iptype='compute',
        num_units=1,
        ipobj=compute_block
    )

    package = PackageInstanceModel(
        devname='TestArch',
        name='test_device',
        ipgroups=[ipgroup_comp, ipgroup_mem]
    )

    # Expected: 16 bytes/cycle per unit * 4 units = 64 bytes/cycle
    assert package.peak_bandwidth_per_cycle() == 64.0


@pytest.mark.unit
def test_memory_bandwidth_validation():
    """Test validation logic detects inconsistencies in memory traffic accounting."""
    from ttsim.config import get_arspec_from_yaml

    ipgroups, packages = get_arspec_from_yaml('config/all_archs.yaml')
    device_pkg = packages['Q1_A1']
    device = Device(device_pkg)

    graph = WorkloadGraph('test_graph')

    # Create tensors first (required by WorkloadGraph.add_op)
    input_tensor = SimTensor({'name': 'input1', 'shape': [10, 10], 'dtype': 'float32'})
    output_tensor = SimTensor({'name': 'output1', 'shape': [10, 10], 'dtype': 'float32'})
    input_tensor.op_in = ['test_op']
    output_tensor.op_out = ['test_op']
    graph.add_tensor(input_tensor)
    graph.add_tensor(output_tensor)

    # Create an op with known memory traffic
    op = SimOp({
        'name': 'test_op',
        'optype': 'Add',
        'inList': ['input1'],
        'outList': ['output1']
    })
    op.uses_compute_pipe = 'matrix'
    op.precision = 'fp16'
    op.repeat_count = 1

    # Calculate correct byte counts for a given number of cycles
    memfreq_MHz = device.simconfig_obj.mem_frequency(units='MHz')
    devfreq_MHz = device.simconfig_obj.frequency('matrix', units='MHz')
    mem_to_dev_ratio = devfreq_MHz / memfreq_MHz
    peak_bw = device.simconfig_obj.peak_bandwidth_per_cycle()
    bw_cycle = peak_bw * device.DG_MEMORY_UTIL_CONSTANT

    test_cycles = 100
    correct_bytes = int(bw_cycle * test_cycles / mem_to_dev_ratio)

    op.perf_stats = {
        'inBytes': correct_bytes,
        'outBytes': correct_bytes,
        'instrs': {'add': 100},
        'inParamCount': 0,
        'inActCount': 100,
        'outActCount': 100
    }

    graph.add_op(op)
    graph.construct_graph()

    # Execute op to calculate cycles
    device.execute_op(op)

    # Verify cycles match expected (within rounding tolerance)
    if TYPE_CHECKING:
        assert op.mem_rd_cycles is not None
        assert op.mem_wr_cycles is not None
    assert abs(op.mem_rd_cycles - test_cycles) <= 1
    assert abs(op.mem_wr_cycles - test_cycles) <= 1

    # Test validation - should pass with correct values
    summary = device.get_exec_stats(graph, bs=1)
    assert 'inBytes' in summary
    assert 'outBytes' in summary

    # Test with incorrect values - should raise ValueError
    # Double the bytes - this should significantly exceed the +1 cycle tolerance
    op.perf_stats['inBytes'] = int(correct_bytes * 2)  # Double the bytes
    # NOTE: Do NOT call device.execute_op(op) again, as it would recalculate cycles correctly
    # However, we need to manually set the fractional cycles to be inconsistent as well
    # since they are calculated in execute_op
    # The validation compares aggregate cycles against expected based on total bytes
    # With doubled bytes but same cycles, we expect: 
    #   expected_cycles = (doubled_bytes / bw) * ratio ≈ 2 * test_cycles
    #   actual_cycles = test_cycles  
    #   This is within the +1 tolerance, so we need more deviation
    
    # To ensure validation fails, we need a bigger mismatch
    # Let's use 10x the bytes which will definitely fail validation
    op.perf_stats['inBytes'] = int(correct_bytes * 10)  # 10x the bytes
    
    # Validation should catch the inconsistency
    with pytest.raises(ValueError, match="Memory bandwidth validation failed"):
        device.get_exec_stats(graph, bs=1)


@pytest.mark.unit
def test_zero_memory_cycles():
    """Test edge cases where tot_mem_rd_cycles or tot_mem_wr_cycles are zero."""
    from ttsim.config import get_arspec_from_yaml

    ipgroups, packages = get_arspec_from_yaml('config/all_archs.yaml')
    device_pkg = packages['Q1_A1']
    device = Device(device_pkg)

    graph = WorkloadGraph('test_graph')

    # Create tensors
    input_tensor = SimTensor({'name': 'input1', 'shape': [10, 10], 'dtype': 'float32'})
    output_tensor = SimTensor({'name': 'output1', 'shape': [10, 10], 'dtype': 'float32'})
    input_tensor.op_in = ['compute_only_op']
    output_tensor.op_out = ['compute_only_op']
    graph.add_tensor(input_tensor)
    graph.add_tensor(output_tensor)

    # Create an op with zero memory traffic (compute-only)
    op = SimOp({
        'name': 'compute_only_op',
        'optype': 'Add',
        'inList': ['input1'],
        'outList': ['output1']
    })
    op.uses_compute_pipe = 'matrix'
    op.precision = 'fp16'
    op.repeat_count = 1
    op.perf_stats = {
        'inBytes': 0,
        'outBytes': 0,
        'instrs': {'add': 100},
        'inParamCount': 0,
        'inActCount': 0,
        'outActCount': 0
    }

    graph.add_op(op)
    graph.construct_graph()

    # Execute op
    device.execute_op(op)

    # Verify cycles are zero
    assert op.mem_rd_cycles == 0
    assert op.mem_wr_cycles == 0

    # Should not raise error - validation skips when cycles are zero
    summary = device.get_exec_stats(graph, bs=1)
    assert summary['tot_mem_rd_cycles'] == 0
    assert summary['tot_mem_wr_cycles'] == 0

    # Test with only read cycles (zero write cycles)
    input_tensor2 = SimTensor({'name': 'input2', 'shape': [10, 10], 'dtype': 'float32'})
    output_tensor2 = SimTensor({'name': 'output2', 'shape': [10, 10], 'dtype': 'float32'})
    input_tensor2.op_in = ['read_only_op']
    output_tensor2.op_out = ['read_only_op']
    graph.add_tensor(input_tensor2)
    graph.add_tensor(output_tensor2)

    op2 = SimOp({
        'name': 'read_only_op',
        'optype': 'Read',
        'inList': ['input2'],
        'outList': ['output2']
    })
    op2.uses_compute_pipe = 'vector'
    op2.precision = 'fp32'
    op2.repeat_count = 1

    memfreq_MHz = device.simconfig_obj.mem_frequency(units='MHz')
    devfreq_MHz = device.simconfig_obj.frequency('matrix', units='MHz')
    mem_to_dev_ratio = devfreq_MHz / memfreq_MHz
    peak_bw = device.simconfig_obj.peak_bandwidth_per_cycle()
    bw_cycle = peak_bw * device.DG_MEMORY_UTIL_CONSTANT
    test_cycles = 50
    correct_bytes = int(bw_cycle * test_cycles / mem_to_dev_ratio)

    op2.perf_stats = {
        'inBytes': correct_bytes,
        'outBytes': 0,  # No writes
        'instrs': {'mov': 50},
        'inParamCount': 0,
        'inActCount': 100,
        'outActCount': 100
    }

    graph.add_op(op2)
    graph.construct_graph()
    device.execute_op(op2)

    if TYPE_CHECKING:
        assert op2.mem_rd_cycles is not None
        assert op2.mem_wr_cycles is not None
    assert op2.mem_rd_cycles > 0
    assert op2.mem_wr_cycles == 0

    # Should not raise error - validation handles zero write cycles
    summary2 = device.get_exec_stats(graph, bs=1)
    assert summary2['tot_mem_wr_cycles'] == 0
    assert summary2['tot_mem_rd_cycles'] > 0


@pytest.mark.unit
def test_different_device_memory_frequencies():
    """Test correct bandwidth and time accounting when device frequency != memory frequency."""
    from ttsim.config import get_arspec_from_yaml

    ipgroups, packages = get_arspec_from_yaml('config/all_archs.yaml')
    device_pkg = packages['Q1_A1']  # Has device freq 1200 MHz, memory freq 1250 MHz

    # Verify frequencies are different
    memfreq_MHz = device_pkg.mem_frequency(units='MHz')
    devfreq_MHz = device_pkg.frequency('matrix', units='MHz')
    assert memfreq_MHz != devfreq_MHz, "Test requires different frequencies"

    device = Device(device_pkg)
    mem_to_dev_ratio = devfreq_MHz / memfreq_MHz

    graph = WorkloadGraph('test_graph')

    # Create tensors
    input_tensor = SimTensor({'name': 'input1', 'shape': [100, 100], 'dtype': 'float32'})
    output_tensor = SimTensor({'name': 'output1', 'shape': [100, 100], 'dtype': 'float32'})
    input_tensor.op_in = ['freq_test_op']
    output_tensor.op_out = ['freq_test_op']
    graph.add_tensor(input_tensor)
    graph.add_tensor(output_tensor)

    # Create an op with known memory traffic
    op = SimOp({
        'name': 'freq_test_op',
        'optype': 'Add',
        'inList': ['input1'],
        'outList': ['output1']
    })
    op.uses_compute_pipe = 'matrix'
    op.precision = 'fp16'
    op.repeat_count = 1

    # Calculate expected values
    peak_bw = device.simconfig_obj.peak_bandwidth_per_cycle()
    bw_cycle = peak_bw * device.DG_MEMORY_UTIL_CONSTANT

    # Choose a target number of device clock cycles
    target_device_cycles = 200
    # Calculate how many memory clock cycles that corresponds to
    target_mem_cycles = target_device_cycles / mem_to_dev_ratio
    # Calculate bytes needed for that many memory cycles
    required_bytes = int(bw_cycle * target_mem_cycles)

    op.perf_stats = {
        'inBytes': required_bytes,
        'outBytes': required_bytes,
        'instrs': {'add': 100},
        'inParamCount': 0,
        'inActCount': 10000,
        'outActCount': 10000
    }

    graph.add_op(op)
    graph.construct_graph()

    # Execute op to calculate cycles
    device.execute_op(op)

    # Verify cycles are in device clock domain and match expected (within rounding tolerance)
    # The cycles should be approximately target_device_cycles
    if TYPE_CHECKING:
        assert op.mem_rd_cycles is not None
        assert op.mem_wr_cycles is not None
    assert abs(op.mem_rd_cycles - target_device_cycles) <= 2, \
        f"Expected ~{target_device_cycles} device cycles, got {op.mem_rd_cycles}"
    assert abs(op.mem_wr_cycles - target_device_cycles) <= 2

    # Verify the conversion is correct: mem_cycles_memclk * ratio = mem_cycles_devclk
    # We can verify by checking: bytes / bw_cycle should give mem cycles, then * ratio = dev cycles
    calculated_mem_cycles = required_bytes / bw_cycle
    calculated_dev_cycles = calculated_mem_cycles * mem_to_dev_ratio
    assert abs(op.mem_rd_cycles - calculated_dev_cycles) <= 2, \
        f"Cycle conversion incorrect: {op.mem_rd_cycles} != {calculated_dev_cycles}"

    # Test validation - should pass with correct values
    summary = device.get_exec_stats(graph, bs=1)
    assert 'inBytes' in summary
    assert 'outBytes' in summary

    # Verify bandwidth accounting is correct
    # tot_mem_rd_cycles is in device clock domain
    # bytes_per_device_clock = bytes_per_memory_clock / mem_to_dev_ratio
    expected_bytes_per_device_clock = bw_cycle / mem_to_dev_ratio
    actual_bytes_per_device_clock = summary['inBytes'] / summary['tot_mem_rd_cycles']
    # Allow 10% tolerance for rounding differences
    assert abs(actual_bytes_per_device_clock - expected_bytes_per_device_clock) / expected_bytes_per_device_clock < 0.1, \
        f"Bandwidth accounting incorrect: {actual_bytes_per_device_clock} != {expected_bytes_per_device_clock}"


@pytest.mark.unit
def test_memory_bandwidth_with_repeat_count():
    """Test memory traffic and bandwidth calculations with ops that have repeat_count > 1."""
    from ttsim.config import get_arspec_from_yaml

    ipgroups, packages = get_arspec_from_yaml('config/all_archs.yaml')
    device_pkg = packages['Q1_A1']
    device = Device(device_pkg)

    graph = WorkloadGraph('test_graph')

    # Create tensors
    input_tensor = SimTensor({'name': 'input1', 'shape': [100, 100], 'dtype': 'float32'})
    output_tensor = SimTensor({'name': 'output1', 'shape': [100, 100], 'dtype': 'float32'})
    input_tensor.op_in = ['repeat_op']
    output_tensor.op_out = ['repeat_op']
    graph.add_tensor(input_tensor)
    graph.add_tensor(output_tensor)

    # Create an op with repeat_count = 5
    op = SimOp({
        'name': 'repeat_op',
        'optype': 'Add',
        'inList': ['input1'],
        'outList': ['output1']
    })
    op.uses_compute_pipe = 'matrix'
    op.precision = 'fp16'
    repeat_count = 5
    op.repeat_count = repeat_count

    # Calculate expected values for a single execution
    memfreq_MHz = device.simconfig_obj.mem_frequency(units='MHz')
    devfreq_MHz = device.simconfig_obj.frequency('matrix', units='MHz')
    mem_to_dev_ratio = devfreq_MHz / memfreq_MHz
    peak_bw = device.simconfig_obj.peak_bandwidth_per_cycle()
    bw_cycle = peak_bw * device.DG_MEMORY_UTIL_CONSTANT

    # Set memory traffic for a single execution
    single_execution_cycles = 100
    single_mem_cycles = single_execution_cycles / mem_to_dev_ratio
    single_execution_bytes = int(bw_cycle * single_mem_cycles)

    op.perf_stats = {
        'inBytes': single_execution_bytes,
        'outBytes': single_execution_bytes,
        'instrs': {'add': 50},
        'inParamCount': 0,
        'inActCount': 10000,
        'outActCount': 10000
    }

    graph.add_op(op)
    graph.construct_graph()

    # Execute op
    device.execute_op(op)

    # Verify cycles for single execution
    if TYPE_CHECKING:
        assert op.mem_rd_cycles is not None
        assert op.mem_wr_cycles is not None
    assert abs(op.mem_rd_cycles - single_execution_cycles) <= 2, \
        f"Single execution cycles incorrect: {op.mem_rd_cycles} vs {single_execution_cycles}"

    # Get execution stats - should account for repeat_count
    summary = device.get_exec_stats(graph, bs=1)

    # Verify that total bytes accounts for repeat_count
    expected_total_inBytes = single_execution_bytes * repeat_count
    expected_total_outBytes = single_execution_bytes * repeat_count
    assert summary['inBytes'] == expected_total_inBytes, \
        f"Total inBytes incorrect: {summary['inBytes']} vs {expected_total_inBytes}"
    assert summary['outBytes'] == expected_total_outBytes, \
        f"Total outBytes incorrect: {summary['outBytes']} vs {expected_total_outBytes}"

    # Verify that total cycles accounts for repeat_count
    expected_total_rd_cycles = op.mem_rd_cycles * repeat_count
    expected_total_wr_cycles = op.mem_wr_cycles * repeat_count
    assert abs(summary['tot_mem_rd_cycles'] - expected_total_rd_cycles) <= repeat_count, \
        f"Total read cycles incorrect: {summary['tot_mem_rd_cycles']} vs {expected_total_rd_cycles}"
    assert abs(summary['tot_mem_wr_cycles'] - expected_total_wr_cycles) <= repeat_count, \
        f"Total write cycles incorrect: {summary['tot_mem_wr_cycles']} vs {expected_total_wr_cycles}"

    # Verify bandwidth validation still passes
    expected_bytes_per_device_clock = bw_cycle / mem_to_dev_ratio
    actual_bytes_per_device_clock = summary['inBytes'] / summary['tot_mem_rd_cycles']
    assert abs(actual_bytes_per_device_clock - expected_bytes_per_device_clock) / expected_bytes_per_device_clock < 0.1, \
        f"Bandwidth accounting incorrect with repeat_count: {actual_bytes_per_device_clock} != {expected_bytes_per_device_clock}"


@pytest.mark.unit
def test_multiple_ops_with_different_repeat_counts():
    """Test memory traffic calculations with multiple ops having different repeat counts."""
    from ttsim.config import get_arspec_from_yaml

    ipgroups, packages = get_arspec_from_yaml('config/all_archs.yaml')
    device_pkg = packages['Q1_A1']
    device = Device(device_pkg)

    graph = WorkloadGraph('test_graph')

    # Create tensors for op1
    input1_tensor = SimTensor({'name': 'input1', 'shape': [100, 100], 'dtype': 'float32'})
    intermediate_tensor = SimTensor({'name': 'intermediate', 'shape': [100, 100], 'dtype': 'float32'})
    output_tensor = SimTensor({'name': 'output1', 'shape': [100, 100], 'dtype': 'float32'})

    input1_tensor.op_in = ['op1']
    intermediate_tensor.op_out = ['op1']
    intermediate_tensor.op_in = ['op2']
    output_tensor.op_out = ['op2']

    graph.add_tensor(input1_tensor)
    graph.add_tensor(intermediate_tensor)
    graph.add_tensor(output_tensor)

    # Setup bandwidth parameters
    memfreq_MHz = device.simconfig_obj.mem_frequency(units='MHz')
    devfreq_MHz = device.simconfig_obj.frequency('matrix', units='MHz')
    mem_to_dev_ratio = devfreq_MHz / memfreq_MHz
    peak_bw = device.simconfig_obj.peak_bandwidth_per_cycle()
    bw_cycle = peak_bw * device.DG_MEMORY_UTIL_CONSTANT

    # Create op1 with repeat_count = 3
    op1 = SimOp({
        'name': 'op1',
        'optype': 'Add',
        'inList': ['input1'],
        'outList': ['intermediate']
    })
    op1.uses_compute_pipe = 'matrix'
    op1.precision = 'fp16'
    op1_repeat_count = 3
    op1.repeat_count = op1_repeat_count

    op1_single_cycles = 50
    op1_single_mem_cycles = op1_single_cycles / mem_to_dev_ratio
    op1_single_bytes = int(bw_cycle * op1_single_mem_cycles)

    op1.perf_stats = {
        'inBytes': op1_single_bytes,
        'outBytes': op1_single_bytes,
        'instrs': {'add': 30},
        'inParamCount': 0,
        'inActCount': 10000,
        'outActCount': 10000
    }

    # Create op2 with repeat_count = 7
    op2 = SimOp({
        'name': 'op2',
        'optype': 'Mul',
        'inList': ['intermediate'],
        'outList': ['output1']
    })
    op2.uses_compute_pipe = 'vector'
    op2.precision = 'fp32'
    op2_repeat_count = 7
    op2.repeat_count = op2_repeat_count

    op2_single_cycles = 80
    op2_single_mem_cycles = op2_single_cycles / mem_to_dev_ratio
    op2_single_bytes = int(bw_cycle * op2_single_mem_cycles)

    op2.perf_stats = {
        'inBytes': op2_single_bytes,
        'outBytes': op2_single_bytes,
        'instrs': {'mul': 40},
        'inParamCount': 0,
        'inActCount': 10000,
        'outActCount': 10000
    }

    graph.add_op(op1)
    graph.add_op(op2)
    graph.construct_graph()

    # Execute ops
    device.execute_op(op1)
    device.execute_op(op2)

    # Get execution stats
    summary = device.get_exec_stats(graph, bs=1)

    # Verify total bytes accounts for both ops with their respective repeat counts
    expected_total_inBytes = (op1_single_bytes * op1_repeat_count) + (op2_single_bytes * op2_repeat_count)
    expected_total_outBytes = (op1_single_bytes * op1_repeat_count) + (op2_single_bytes * op2_repeat_count)

    assert summary['inBytes'] == expected_total_inBytes, \
        f"Total inBytes incorrect: {summary['inBytes']} vs {expected_total_inBytes}"
    assert summary['outBytes'] == expected_total_outBytes, \
        f"Total outBytes incorrect: {summary['outBytes']} vs {expected_total_outBytes}"

    # Verify total cycles accounts for both ops with their respective repeat counts
    if TYPE_CHECKING:
        assert op1.mem_rd_cycles is not None
        assert op1.mem_wr_cycles is not None
        assert op2.mem_rd_cycles is not None
        assert op2.mem_wr_cycles is not None

    expected_total_rd_cycles = (op1.mem_rd_cycles * op1_repeat_count) + (op2.mem_rd_cycles * op2_repeat_count)
    expected_total_wr_cycles = (op1.mem_wr_cycles * op1_repeat_count) + (op2.mem_wr_cycles * op2_repeat_count)

    # Allow some tolerance for rounding
    tolerance = op1_repeat_count + op2_repeat_count + 2
    assert abs(summary['tot_mem_rd_cycles'] - expected_total_rd_cycles) <= tolerance, \
        f"Total read cycles incorrect: {summary['tot_mem_rd_cycles']} vs {expected_total_rd_cycles}"
    assert abs(summary['tot_mem_wr_cycles'] - expected_total_wr_cycles) <= tolerance, \
        f"Total write cycles incorrect: {summary['tot_mem_wr_cycles']} vs {expected_total_wr_cycles}"

    # Verify bandwidth validation still passes for aggregate
    expected_bytes_per_device_clock = bw_cycle / mem_to_dev_ratio
    actual_bytes_per_device_clock = summary['inBytes'] / summary['tot_mem_rd_cycles']
    assert abs(actual_bytes_per_device_clock - expected_bytes_per_device_clock) / expected_bytes_per_device_clock < 0.1, \
        f"Bandwidth accounting incorrect with multiple repeat counts: {actual_bytes_per_device_clock} != {expected_bytes_per_device_clock}"
