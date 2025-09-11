#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import unittest
import numpy as np
from typing import Any
from ttsim.ops.op import SimOpFactory
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


class TestIsNaN(unittest.TestCase):
    """Test cases for IsNaN operation"""

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_factory_integration(self):
        """Test that IsNaNOp can be created through SimOpFactory"""
        op_cls = SimOpFactory('IsNaN')
        assert op_cls.__name__ == 'IsNaNOp'

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_isnan_basic_functionality(self):
        """Test basic IsNaN operation with different shapes"""
        test_shapes = [
            [4],           # 1D
            [3, 4],        # 2D
            [2, 3, 4],     # 3D
            [2, 3, 4, 5]   # 4D
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                op_name = f'test_isnan_{len(shape)}d'
                i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'IsNaN',
                    'inList': [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors]
                }

                op_cls = SimOpFactory('IsNaN')
                op_obj = op_cls(op_info)

                for x in i_tensors: x.op_in = [op_name]
                for x in o_tensors: x.op_out = [op_name]

                op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

                # Verify output shape is valid
                assert o_tensors[0].shape is not None
                assert o_tensors[0].dtype == np.bool_  # IsNaN outputs boolean

                # Verify performance stats include comparison instruction
                assert 'inElems' in op_perf
                assert 'outElems' in op_perf
                assert 'inBytes' in op_perf
                assert 'outBytes' in op_perf
                assert 'instrs' in op_perf
                assert 'cmp' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_isnan_performance_consistency(self):
        """Test that IsNaN performance is consistent across runs"""
        shape = [3, 4, 5]
        op_name = 'test_isnan_perf_consistency'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'IsNaN',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_cls = SimOpFactory('IsNaN')
        op_obj = op_cls(op_info)

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
    def test_isnan_memory_usage(self):
        """Test IsNaN memory usage patterns"""
        shape = [10, 20]
        op_name = 'test_isnan_memory'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'IsNaN',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_cls = SimOpFactory('IsNaN')
        op_obj = op_cls(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Memory usage should be reasonable
        input_elements = i_tensors[0].nelems()
        assert op_perf['inElems'] >= input_elements
        assert op_perf['inBytes'] >= input_elements * 4  # float32 = 4 bytes

        # Output should have same number of elements as input
        assert op_perf['outElems'] == input_elements

        # Should have comparison instructions
        assert op_perf['instrs']['cmp'] > 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_isnan_different_dtypes(self):
        """Test IsNaN with different data types"""
        dtypes = [np.float32, np.float64]
        shape = [3, 4]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                op_name = f'test_isnan_{dtype.__name__}'
                i_tensors = [F._from_shape('X', shape, np_dtype=dtype)]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'IsNaN',
                    'inList': [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors]
                }

                op_cls = SimOpFactory('IsNaN')
                op_obj = op_cls(op_info)

                for x in i_tensors: x.op_in = [op_name]
                for x in o_tensors: x.op_out = [op_name]

                op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

                # Verify output is boolean regardless of input type
                assert o_tensors[0].dtype == np.bool_

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_isnan_invalid_inputs(self):
        """Test IsNaN with invalid inputs"""
        op_name = 'test_isnan_invalid'
        i_tensors: list[Any] = []  # Empty input list
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'IsNaN',
            'inList': [],
            'outList': [x.name for x in o_tensors]
        }

        op_cls = SimOpFactory('IsNaN')
        op_obj = op_cls(op_info)

        # This should raise an error due to invalid input count
        with pytest.raises(Exception, match="should be in range"):
            op_obj.get_perf_counts(i_tensors, o_tensors)

