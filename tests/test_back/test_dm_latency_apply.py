#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Option C (--dm_latency_mode apply): the exposed read latency is added to
memory-read cycles, while 'report' mode (default) leaves timing untouched.

Uses the real Blackhole p100a package (config/tt_bh.yaml), which carries the
calibrated ``read_latency`` block, and drives a single op through
``Device.execute_op`` on the legacy (inBytes) descriptor path.
"""

import math

import pytest

from ttsim.back.device import Device
from ttsim.config import get_arspec_from_yaml
from ttsim.graph.wl_graph import WorkloadGraph
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import SimTensor

ARCH_YAML = 'config/tt_bh.yaml'
PACKAGE = 'p100a'


def _make_device(**kwargs) -> Device:
    _, packages = get_arspec_from_yaml(ARCH_YAML)
    return Device(packages[PACKAGE], **kwargs)


def _run_single_op(device: Device) -> SimOp:
    graph = WorkloadGraph('g')
    in_t = SimTensor({'name': 'in1', 'shape': [512, 512], 'dtype': 'float32'})
    out_t = SimTensor({'name': 'out1', 'shape': [512, 512], 'dtype': 'float32'})
    in_t.op_in = ['op']
    out_t.op_out = ['op']
    graph.add_tensor(in_t)
    graph.add_tensor(out_t)
    op = SimOp({'name': 'op', 'optype': 'Add', 'inList': ['in1'], 'outList': ['out1']})
    op.uses_compute_pipe = 'matrix'
    op.precision = 'fp16'
    op.repeat_count = 1
    op.perf_stats = {
        'inBytes': 512 * 512 * 2, 'outBytes': 512 * 512 * 2, 'instrs': {'add': 1000},
        'inParamCount': 0, 'inActCount': 100, 'outActCount': 100,
    }
    graph.add_op(op)
    graph.construct_graph()
    device.execute_op(op)
    return op


@pytest.mark.unit
def test_bad_mode_raises():
    with pytest.raises(ValueError, match="dm_latency_mode"):
        _make_device(dm_latency_mode='bogus')


@pytest.mark.unit
def test_apply_mode_implies_enabled_and_report_default():
    report = _make_device(dm_latency_mode='report', enable_dm_latency=True)
    assert report.enable_dm_latency and not report.dm_apply_timing
    apply = _make_device(dm_latency_mode='apply')  # no explicit enable
    assert apply.enable_dm_latency and apply.dm_apply_timing
    # fclk defaults to the matrix compute clock when the YAML omits fclk_mhz.
    assert apply.dm_fclk_MHz == apply.freq_MHz


@pytest.mark.unit
def test_report_mode_does_not_change_read_cycles():
    baseline = _run_single_op(_make_device())  # model off entirely
    report = _run_single_op(_make_device(dm_latency_mode='report', enable_dm_latency=True))
    # Report mode computes the prediction but must not perturb timing.
    assert report.mem_rd_cycles == baseline.mem_rd_cycles
    assert report.mem_rd_cycles_fractional == pytest.approx(baseline.mem_rd_cycles_fractional)
    assert report.mem_rd_cycles_fractional == pytest.approx(report.mem_rd_cycles_bw_fractional)
    assert report.dm_read_exposed_devclk_cycles == 0.0
    # The prediction itself is still produced.
    assert report.dm_read_exposed_cycles > 0.0


@pytest.mark.unit
def test_apply_mode_adds_exposed_to_read_cycles():
    apply = _run_single_op(_make_device(dm_latency_mode='apply'))
    # Effective read cycles = bandwidth-only + exposed (converted to devclk).
    assert apply.dm_read_exposed_devclk_cycles > 0.0
    assert apply.mem_rd_cycles_fractional == pytest.approx(
        apply.mem_rd_cycles_bw_fractional + apply.dm_read_exposed_devclk_cycles
    )
    # p100a matrix op: fclk == matrix devclk, so the conversion ratio is 1.
    assert apply.dm_read_exposed_devclk_cycles == pytest.approx(apply.dm_read_exposed_cycles)
    # And effective read cycles strictly exceed the bandwidth-only baseline.
    assert apply.mem_rd_cycles_fractional > apply.mem_rd_cycles_bw_fractional


@pytest.mark.unit
def test_apply_mode_aggregate_bandwidth_check_passes():
    # get_exec_stats runs the memory-bandwidth self-check; in apply mode it must use
    # the BW-only cycles and therefore not raise despite inflated read cycles.
    device = _make_device(dm_latency_mode='apply')
    graph = WorkloadGraph('g')
    in_t = SimTensor({'name': 'in1', 'shape': [512, 512], 'dtype': 'float32'})
    out_t = SimTensor({'name': 'out1', 'shape': [512, 512], 'dtype': 'float32'})
    in_t.op_in = ['op']
    out_t.op_out = ['op']
    graph.add_tensor(in_t)
    graph.add_tensor(out_t)
    op = SimOp({'name': 'op', 'optype': 'Add', 'inList': ['in1'], 'outList': ['out1']})
    op.uses_compute_pipe = 'matrix'
    op.precision = 'fp16'
    op.repeat_count = 1
    op.perf_stats = {
        'inBytes': 512 * 512 * 2, 'outBytes': 512 * 512 * 2, 'instrs': {'add': 1000},
        'inParamCount': 0, 'inActCount': 100, 'outActCount': 100,
    }
    graph.add_op(op)
    graph.construct_graph()
    device.execute_op(op)
    summary = device.get_exec_stats(graph, bs=1)
    # No exception raised; reported read cycles include the exposed latency, so they
    # exceed the bandwidth-only cycles the self-check validated against.
    assert summary['tot_mem_rd_cycles'] >= math.ceil(op.mem_rd_cycles_bw_fractional)
    assert op.mem_rd_cycles_fractional > op.mem_rd_cycles_bw_fractional
