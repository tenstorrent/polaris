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


class TestSpaceToDepth(unittest.TestCase):
    """Test cases for SpaceToDepth operation"""

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_factory_integration(self):
        """Test that SpaceToDepthOp can be created through SimOpFactory"""
        op_cls = SimOpFactory('SpaceToDepth')
        assert op_cls.__name__ == 'SpaceToDepthOp'

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_space_to_depth_basic_functionality(self):
        """Test basic SpaceToDepth operation with different configurations"""
        test_configs = [
            ([1, 8, 8, 3], 2),    # blocksize=2
            ([1, 12, 12, 3], 3),  # blocksize=3
            ([2, 16, 16, 4], 4)   # blocksize=4
        ]

        for input_shape, blocksize in test_configs:
            with self.subTest(input_shape=input_shape, blocksize=blocksize):
                op_name = f'test_space_to_depth_{blocksize}x'
                i_tensors = [F._from_shape('X', input_shape, np_dtype=np.float32)]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'SpaceToDepth',
                    'inList': [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors],
                    'attrs': {'blocksize': blocksize}
                }

                op_cls = SimOpFactory('SpaceToDepth')
                op_obj = op_cls(op_info)

                for x in i_tensors: x.op_in = [op_name]
                for x in o_tensors: x.op_out = [op_name]

                op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

                # Verify attributes are set correctly
                assert op_obj.attrs['blocksize'] == blocksize

                # Verify output shape is valid
                assert o_tensors[0].shape is not None
                assert o_tensors[0].dtype == np.float32

                # Verify performance stats
                assert 'inElems' in op_perf
                assert 'outElems' in op_perf
                assert 'inBytes' in op_perf
                assert 'outBytes' in op_perf
                assert 'instrs' in op_perf
                assert 'mov' in op_perf['instrs']
                assert 'gather' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_space_to_depth_different_blocksizes(self):
        """Test SpaceToDepth with different blocksize values"""
        input_shape = [1, 8, 8, 3]
        blocksizes = [2, 4]

        for blocksize in blocksizes:
            with self.subTest(blocksize=blocksize):
                op_name = f'test_space_to_depth_bs_{blocksize}'
                i_tensors = [F._from_shape('X', input_shape, np_dtype=np.float32)]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'SpaceToDepth',
                    'inList': [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors],
                    'attrs': {'blocksize': blocksize}
                }

                op_cls = SimOpFactory('SpaceToDepth')
                op_obj = op_cls(op_info)

                for x in i_tensors: x.op_in = [op_name]
                for x in o_tensors: x.op_out = [op_name]

                op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

                # Verify blocksize is set correctly
                assert op_obj.attrs['blocksize'] == blocksize

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_space_to_depth_performance_consistency(self):
        """Test that SpaceToDepth performance is consistent across runs"""
        input_shape = [1, 8, 8, 3]
        blocksize = 2
        op_name = 'test_space_to_depth_perf_consistency'
        i_tensors = [F._from_shape('X', input_shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'SpaceToDepth',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'blocksize': blocksize}
        }

        op_cls = SimOpFactory('SpaceToDepth')
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
    def test_space_to_depth_memory_usage(self):
        """Test SpaceToDepth memory usage patterns"""
        input_shape = [2, 12, 12, 4]
        blocksize = 3
        op_name = 'test_space_to_depth_memory'
        i_tensors = [F._from_shape('X', input_shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'SpaceToDepth',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'blocksize': blocksize}
        }

        op_cls = SimOpFactory('SpaceToDepth')
        op_obj = op_cls(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Memory usage should be reasonable
        input_elements = i_tensors[0].nelems()
        assert op_perf['inElems'] == input_elements
        assert op_perf['inBytes'] == input_elements * 4  # float32 = 4 bytes

        # Output should have same number of elements (rearranged)
        assert op_perf['outElems'] == input_elements

        # Should have gather instructions for data rearrangement
        assert op_perf['instrs']['gather'] > 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_space_to_depth_invalid_inputs(self):
        """Test SpaceToDepth with invalid inputs"""
        op_name = 'test_space_to_depth_invalid'
        i_tensors: list[Any] = []  # Empty input list
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'SpaceToDepth',
            'inList': [],
            'outList': [x.name for x in o_tensors]
        }

        op_cls = SimOpFactory('SpaceToDepth')
        op_obj = op_cls(op_info)

        # This should raise an error due to invalid input count
        with pytest.raises(Exception, match="should be in range"):
            op_obj.get_perf_counts(i_tensors, o_tensors)

