#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import unittest
import numpy as np
from typing import Any
from ttsim.ops.op import SimOp
from ttsim.ops.desc.registry import get_opdesc_registry
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


class TestClip(unittest.TestCase):
    """Test cases for Clip operation"""

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_factory_integration(self):
        """Test that ClipOp is registered in the op mapping"""
        assert get_opdesc_registry().has_shape_inference_function('Clip')

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_clip_basic_functionality(self):
        """Test basic Clip operation with different shapes"""
        test_shapes = [
            [4],           # 1D
            [3, 4],        # 2D
            [2, 3, 4],     # 3D
            [2, 3, 4, 5]   # 4D
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                op_name = f'test_clip_{len(shape)}d'
                i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'Clip',
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

                # Verify performance stats include comparison and min/max instructions
                assert 'inElems' in op_perf
                assert 'outElems' in op_perf
                assert 'inBytes' in op_perf
                assert 'outBytes' in op_perf
                assert 'instrs' in op_perf
                assert 'cmp' in op_perf['instrs']
                assert 'min' in op_perf['instrs']
                assert 'max' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_clip_with_min_max_tensors(self):
        """Test Clip with min and max tensors"""
        op_name = 'test_clip_with_min_max'
        i_tensors = [
            F._from_shape('X', [3, 4], np_dtype=np.float32),
            F._from_shape('min', [1], np_dtype=np.float32),  # min tensor
            F._from_shape('max', [1], np_dtype=np.float32)   # max tensor
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'Clip',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Verify performance stats include min/max tensors
        assert op_perf['inElems'] > i_tensors[0].nelems()  # Should include min/max elements

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_clip_with_attributes(self):
        """Test Clip with min and max attributes"""
        op_name = 'test_clip_with_attributes'
        i_tensors = [F._from_shape('X', [3, 4], np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'Clip',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'min': 0.0, 'max': 1.0}
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Verify attributes are set correctly
        assert op_obj.attrs['min'] == 0.0
        assert op_obj.attrs['max'] == 1.0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_clip_performance_consistency(self):
        """Test that Clip performance is consistent across runs"""
        shape = [3, 4, 5]
        op_name = 'test_clip_perf_consistency'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'Clip',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'min': -1.0, 'max': 1.0}
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
    def test_clip_memory_usage(self):
        """Test Clip memory usage patterns"""
        shape = [10, 20]
        op_name = 'test_clip_memory'
        i_tensors = [F._from_shape('X', shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'Clip',
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

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_clip_invalid_inputs(self):
        """Test Clip with invalid inputs"""
        op_name = 'test_clip_invalid'
        i_tensors: list[Any] = []  # Empty input list
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'Clip',
            'inList': [],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        # This should raise an error due to invalid input count
        with pytest.raises(Exception, match="should be in range"):
            op_obj.get_perf_counts(i_tensors, o_tensors)

