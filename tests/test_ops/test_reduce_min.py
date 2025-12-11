#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


class TestReduceMin:
    """Test cases for ReduceMin operation"""

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_factory_integration(self):
        """Basic sanity test for ReduceMin creation via SimOp."""
        op_info = {
            'name': 'test_reduce_min_factory',
            'optype': 'ReduceMin',
            'inList': ['X'],
            'outList': ['Y'],
            'attrs': {},
        }
        op = SimOp(op_info)
        assert op.optype == 'ReduceMin'

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_reduce_min_1d(self):
        """Test ReduceMin operation with 1D tensor"""
        op_name = 'test_reduce_min_1d'
        i_tensors = [F._from_shape('X', [4], np_dtype=np.float32)]
        o_tensors = [F._from_shape('Y', [4], np_dtype=np.float32)]

        op_info = {
            'name': op_name,
            'optype': 'ReduceMin',
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

        # Verify performance stats
        assert 'inElems' in op_perf
        assert 'outElems' in op_perf
        assert 'inBytes' in op_perf
        assert 'outBytes' in op_perf
        assert 'instrs' in op_perf
        # ReduceMin is modeled using min operations
        assert 'min' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_reduce_min_2d(self):
        """Test ReduceMin operation with 2D tensor"""
        op_name = 'test_reduce_min_2d'
        i_tensors = [F._from_shape('X', [3, 4], np_dtype=np.float32)]
        o_tensors = [F._from_shape('Y', [3, 4], np_dtype=np.float32)]

        op_info = {
            'name': op_name,
            'optype': 'ReduceMin',
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

        # Verify performance stats
        assert 'inElems' in op_perf
        assert 'outElems' in op_perf
        assert 'inBytes' in op_perf
        assert 'outBytes' in op_perf
        assert 'instrs' in op_perf
        assert 'min' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_reduce_min_with_axes(self):
        """Test ReduceMin with axes specification"""
        op_name = 'test_reduce_min_with_axes'
        X = F._from_shape('X', [2, 3, 4], np_dtype=np.float32)
        axes = F._from_shape('axes', [2], np_dtype=np.int64)
        axes.data = np.array([1, 2], dtype=np.int64)
        i_tensors = [X, axes]
        o_tensors = [F._from_shape('Y', [2, 3, 4], np_dtype=np.float32)]

        op_info = {
            'name': op_name,
            'optype': 'ReduceMin',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Verify performance stats include axes tensor
        assert op_perf['inElems'] > i_tensors[0].nelems()  # Should include axes elements

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_reduce_min_performance_consistency(self):
        """Test that ReduceMin performance is consistent across runs"""
        shape = [3, 4, 5]
        op_name = 'test_reduce_min_perf_consistency'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [F._from_shape('Y', shape, np_dtype=np.float32)]

        op_info = {
            'name': op_name,
            'optype': 'ReduceMin',
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
    def test_reduce_min_memory_usage(self):
        """Test ReduceMin memory usage patterns"""
        shape = [10, 20]
        op_name = 'test_reduce_min_memory'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [F._from_shape('Y', shape, np_dtype=np.float32)]

        op_info = {
            'name': op_name,
            'optype': 'ReduceMin',
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

        # Output should have reasonable memory usage
        assert op_perf['outBytes'] > 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_reduce_min_invalid_inputs(self):
        """Test ReduceMin with invalid inputs"""
        op_name = 'test_reduce_min_invalid'
        i_tensors = []  # Empty input list
        o_tensors = [F._from_shape('Y', [1], np_dtype=np.float32)]

        op_info = {
            'name': op_name,
            'optype': 'ReduceMin',
            'inList': [],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        # This should raise an error due to invalid input count
        with pytest.raises(AssertionError, match="should be in"):
            op_obj.get_perf_counts(i_tensors, o_tensors)
