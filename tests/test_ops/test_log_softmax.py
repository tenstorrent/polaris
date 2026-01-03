#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from typing import Any
from ttsim.ops.op import SimOp
from ttsim.ops.desc.registry import get_opdesc_registry
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


class TestLogSoftmax:
    """Test cases for LogSoftmax operation"""

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_factory_integration(self):
        """Test that LogSoftmaxOp is registered in the op mapping"""
        assert get_opdesc_registry().has_shape_inference_function('LogSoftmax')

    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("shape", [
        [4],           # 1D
        [3, 4],        # 2D
        [2, 3, 4],     # 3D
        [2, 3, 4, 5]   # 4D
    ])
    def test_log_softmax_basic_functionality(self, shape):
        """Test basic LogSoftmax operation with different shapes"""
        op_name = f'test_log_softmax_{len(shape)}d'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'LogSoftmax',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Verify output shape is valid
        assert o_tensors[0].shape is not None
        assert o_tensors[0].dtype == np.float32

        # Verify performance stats include exp, log, sub, and add instructions
        assert 'inElems' in op_perf
        assert 'outElems' in op_perf
        assert 'inBytes' in op_perf
        assert 'outBytes' in op_perf
        assert 'instrs' in op_perf
        assert 'exp' in op_perf['instrs']
        assert 'log' in op_perf['instrs']
        assert 'sub' in op_perf['instrs']
        assert 'add' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("axis", [-1, 0, 1, 2])
    def test_log_softmax_with_axis_attribute(self, axis):
        """Test LogSoftmax with different axis values"""
        shape = [3, 4, 5]
        op_name = f'test_log_softmax_axis_{axis}'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'LogSoftmax',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'axis': axis}
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Verify axis is set correctly
        assert op_obj.attrs['axis'] == axis

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_log_softmax_default_axis(self):
        """Test LogSoftmax with default axis value"""
        op_name = 'test_log_softmax_default_axis'
        i_tensors = [F._from_shape('X', [3, 4], np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'LogSoftmax',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'axis': -1}  # Default axis
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Verify default axis is -1
        assert op_obj.attrs['axis'] == -1

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_log_softmax_performance_consistency(self):
        """Test that LogSoftmax performance is consistent across runs"""
        shape = [3, 4, 5]
        op_name = 'test_log_softmax_perf_consistency'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'LogSoftmax',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        # Run multiple times to check consistency
        perf_results = []
        for _ in range(3):
            perf = op_obj.get_perf_counts(i_tensors, o_tensors)
            perf_results.append(perf)

        # All results should be identical
        for key in ['inElems', 'outElems', 'inBytes', 'outBytes']:
            values = [p[key] for p in perf_results]
            assert all(v == values[0] for v in values), f"Inconsistent {key}: {values}"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_log_softmax_memory_usage(self):
        """Test LogSoftmax memory usage patterns"""
        shape = [10, 20]
        op_name = 'test_log_softmax_memory'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'LogSoftmax',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Memory usage should be reasonable
        input_elements = i_tensors[0].nelems()
        assert op_perf['inElems'] >= input_elements
        assert op_perf['inBytes'] >= input_elements * 4  # float32 = 4 bytes

        # Output should have same memory usage as input
        assert op_perf['outElems'] == input_elements

        # Should have exp, log, sub, and add instructions
        assert op_perf['instrs']['exp'] > 0
        assert op_perf['instrs']['log'] > 0
        assert op_perf['instrs']['sub'] > 0
        assert op_perf['instrs']['add'] > 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_log_softmax_invalid_inputs(self):
        """Test LogSoftmax with invalid inputs"""
        op_name = 'test_log_softmax_invalid'
        i_tensors: list[Any] = []  # Empty input list
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'LogSoftmax',
            'inList': [],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        # This should raise an error due to invalid input count
        with pytest.raises(Exception, match="should be in range"):
            op_obj.get_perf_counts(i_tensors, o_tensors)

