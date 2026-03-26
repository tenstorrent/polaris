# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for Device.execute_graph disable_fusion knob."""

from unittest.mock import MagicMock, Mock

import pytest

from ttsim.back.device import Device


class MockIPGroup:
    def __init__(self, iptype):
        self.iptype = iptype


class MockSimConfigWithPeakIpc:
    """Minimal sim config for Device.execute_op (used by execute_graph)."""

    def __init__(self, freq_mhz=1000, ramp_penalty=10):
        self._freq_mhz = freq_mhz
        self._ramp_penalty = ramp_penalty
        self._mem_size_gb = 32.0
        self.devname = 'TestArch'
        self.name = 'test_device'
        self.ipgroups = [MockIPGroup('compute'), MockIPGroup('memory')]

    def frequency(self, pipe, units='MHz'):
        return self._freq_mhz

    def mem_frequency(self, units='MHz'):
        return self._freq_mhz

    def ramp_penalty(self):
        return self._ramp_penalty

    def mem_size(self, units='GB'):
        return self._mem_size_gb

    def peak_bandwidth(self, freq_units="GHz"):
        return 1000.0

    def peak_bandwidth_per_cycle(self):
        return 10.0

    def peak_flops(self, pipe, instr, precision, mul_factor=1):
        return 100.0

    def peak_ipc(self, pipe, instr, precision):
        return 128.0


def _create_mock_op(name, optype, compute_pipe='matrix'):
    op = Mock()
    op.name = name
    op.optype = optype
    op.removed_in_optimization = False
    op.fused_in_optimization = False
    op.fused_with_op = None
    op.uses_compute_pipe = compute_pipe
    op.compute_cycles = 1000
    op.mem_rd_cycles = 500
    op.mem_wr_cycles = 500
    op.mem_rd_cycles_fractional = 500.0
    op.mem_wr_cycles_fractional = 500.0
    op.repeat_count = 1
    op.precision = 'fp32'
    op.fused_op_cycles = None
    op.exec_stats = {}
    op.perf_stats = {
        'inElems': 1000,
        'outElems': 1000,
        'inBytes': 4000,
        'outBytes': 4000,
        'instrs': {'mac': 1000},
        'inParamCount': 500,
        'inActCount': 500,
        'outActCount': 1000,
    }
    return op


@pytest.mark.unit
def test_execute_graph_disable_fusion_skips_fuse_nodes():
    """When disable_fusion=True, wlgraph.fuse_nodes must not be called."""
    device = Device(MockSimConfigWithPeakIpc())
    op = _create_mock_op('op1', 'MatMul', compute_pipe='matrix')
    graph = MagicMock()
    graph.get_ordered_nodes.return_value = ['op1']
    graph.get_op.return_value = op
    graph.get_optype = MagicMock(return_value='MatMul')

    wlmapspec = MagicMock()
    wlmapspec.removal_spec = MagicMock()
    wlmapspec.removal_spec.is_included.return_value = False
    wlmapspec.fusion_spec = MagicMock()

    fuse_nodes = MagicMock(return_value=[])
    graph.fuse_nodes = fuse_nodes

    device.execute_graph(graph, wlmapspec, disable_fusion=True)

    fuse_nodes.assert_not_called()
    graph.set_precision.assert_called_once_with(wlmapspec.data_type_spec)
    graph.set_resources.assert_called_once_with(wlmapspec.rsrc_spec)
    graph.remove_nodes.assert_called_once_with(wlmapspec.removal_spec)


@pytest.mark.unit
def test_execute_graph_fusion_calls_fuse_nodes_by_default():
    """When disable_fusion=False (default), fuse_nodes is invoked."""
    device = Device(MockSimConfigWithPeakIpc())
    op = _create_mock_op('op1', 'MatMul', compute_pipe='matrix')
    graph = MagicMock()
    graph.get_ordered_nodes.return_value = ['op1']
    graph.get_op.return_value = op
    graph.get_optype = MagicMock(return_value='MatMul')

    wlmapspec = MagicMock()
    wlmapspec.removal_spec = MagicMock()
    wlmapspec.removal_spec.is_included.return_value = False
    wlmapspec.fusion_spec = MagicMock()

    fuse_nodes = MagicMock(return_value=[])
    graph.fuse_nodes = fuse_nodes

    device.execute_graph(graph, wlmapspec, disable_fusion=False)

    fuse_nodes.assert_called_once_with(wlmapspec.fusion_spec)
