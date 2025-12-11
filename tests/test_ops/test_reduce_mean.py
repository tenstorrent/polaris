#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


class TestReduceMean:
    """Test cases for ReduceMean operation"""

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_factory_integration(self):
        """Basic sanity test for ReduceMean creation via SimOp."""
        op_info = {
            'name': 'test_reduce_mean_factory',
            'optype': 'ReduceMean',
            'inList': ['X'],
            'outList': ['Y'],
            'attrs': {},
        }
        op = SimOp(op_info)
        assert op.optype == 'ReduceMean'

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_reduce_mean_basic_functionality(self):
        """Test basic ReduceMean operation with different shapes"""
        test_shapes = [
            [4],           # 1D
            [3, 4],        # 2D
            [2, 3, 4],     # 3D
            [2, 3, 4, 5]   # 4D
        ]

        for shape in test_shapes:
            op_name = f'test_reduce_mean_{len(shape)}d'
            i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
            o_tensors = [F._from_shape('Y', [2, 3, 4], np_dtype=np.float32)]

            op_info = {
                'name': op_name,
                'optype': 'ReduceMean',
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

            # Verify performance stats include add and div instructions
            assert 'inElems' in op_perf
            assert 'outElems' in op_perf
            assert 'inBytes' in op_perf
            assert 'outBytes' in op_perf
            assert 'instrs' in op_perf
            assert 'add' in op_perf['instrs']
            assert 'div' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_reduce_mean_with_axes(self):
        """Test ReduceMean with axes specification"""
        op_name = 'test_reduce_mean_with_axes'
        X = F._from_shape('X', [2, 3, 4], np_dtype=np.float32)
        axes = F._from_shape('axes', [1], np_dtype=np.int64)
        axes.data = np.array([1], dtype=np.int64)
        i_tensors = [X, axes]
        o_tensors = [F._from_shape('Y', [2, 3, 4], np_dtype=np.float32)]

        op_info = {
            'name': op_name,
            'optype': 'ReduceMean',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Verify performance stats include axes tensor contribution
        assert op_perf['inElems'] >= sum(t.nelems() for t in i_tensors)
        assert 'add' in op_perf['instrs']
        assert 'div' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_reduce_mean_performance_consistency(self):
        """Test that ReduceMean performance is consistent across runs"""
        shape = [3, 4, 5]
        op_name = 'test_reduce_mean_perf_consistency'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [F._from_shape('Y', [2, 3, 4], np_dtype=np.float32)]

        op_info = {
            'name': op_name,
            'optype': 'ReduceMean',
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
    def test_reduce_mean_memory_usage(self):
        """Test ReduceMean memory usage patterns"""
        shape = [10, 20]
        op_name = 'test_reduce_mean_memory'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [F._from_shape('Y', [2, 3, 4], np_dtype=np.float32)]

        op_info = {
            'name': op_name,
            'optype': 'ReduceMean',
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

        # Should have add and div instructions for averaging
        assert op_perf['instrs']['add'] > 0
        assert op_perf['instrs']['div'] > 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_reduce_mean_invalid_inputs(self):
        """Test ReduceMean with invalid inputs"""
        op_name = 'test_reduce_mean_invalid'
        i_tensors = []  # Empty input list
        o_tensors = [F._from_shape('Y', [2, 3, 4], np_dtype=np.float32)]

        op_info = {
            'name': op_name,
            'optype': 'ReduceMean',
            'inList': [],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        # This should raise an error due to invalid input count
        with pytest.raises(AssertionError, match="should be in"):
            op_obj.get_perf_counts(i_tensors, o_tensors)
