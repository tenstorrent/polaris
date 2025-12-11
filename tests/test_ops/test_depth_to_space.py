#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import unittest
import numpy as np
from typing import Any
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


class TestDepthToSpace(unittest.TestCase):
    """Test cases for DepthToSpace operation"""

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_factory_integration(self):
        """Basic sanity test for DepthToSpace creation via SimOp."""
        op_info = {
            'name': 'test_depth_to_space_factory',
            'optype': 'DepthToSpace',
            'inList': ['X'],
            'outList': ['Y'],
            'attrs': {'blocksize': 2, 'mode': 'DCR'},
        }
        op_obj = SimOp(op_info)
        assert op_obj.optype == 'DepthToSpace'

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_depth_to_space_basic_functionality(self):
        """Test basic DepthToSpace operation with different configurations"""
        test_configs = [
            ([1, 2, 2, 12], 2, 'DCR'),    # blocksize=2, DCR mode
            ([1, 3, 3, 27], 3, 'CRD'),    # blocksize=3, CRD mode
            ([2, 4, 4, 64], 4, 'DCR')     # blocksize=4, DCR mode
        ]

        for input_shape, blocksize, mode in test_configs:
            with self.subTest(input_shape=input_shape, blocksize=blocksize, mode=mode):
                op_name = f'test_depth_to_space_{blocksize}x_{mode}'
                i_tensors = [F._from_shape('X', input_shape, np_dtype=np.float32)]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'DepthToSpace',
                    'inList': [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors],
                    'attrs': {'blocksize': blocksize, 'mode': mode}
                }

                op_obj = SimOp(op_info)

                for x in i_tensors: x.op_in = [op_name]
                for x in o_tensors: x.op_out = [op_name]

                op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

                # Verify attributes are set correctly
                assert op_obj.attrs['blocksize'] == blocksize
                assert op_obj.attrs['mode'] == mode

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
                assert 'scatter' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_depth_to_space_different_modes(self):
        """Test DepthToSpace with different mode values"""
        input_shape = [1, 2, 2, 12]
        blocksize = 2
        modes = ['DCR', 'CRD']

        for mode in modes:
            with self.subTest(mode=mode):
                op_name = f'test_depth_to_space_mode_{mode}'
                i_tensors = [F._from_shape('X', input_shape, np_dtype=np.float32)]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'DepthToSpace',
                    'inList': [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors],
                    'attrs': {'blocksize': blocksize, 'mode': mode}
                }

                op_obj = SimOp(op_info)

                for x in i_tensors: x.op_in = [op_name]
                for x in o_tensors: x.op_out = [op_name]

                op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

                # Verify mode is set correctly
                assert op_obj.attrs['mode'] == mode

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_depth_to_space_performance_consistency(self):
        """Test that DepthToSpace performance is consistent across runs"""
        input_shape = [1, 2, 2, 12]
        blocksize = 2
        mode = 'DCR'
        op_name = 'test_depth_to_space_perf_consistency'
        i_tensors = [F._from_shape('X', input_shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'DepthToSpace',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'blocksize': blocksize, 'mode': mode}
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
    def test_depth_to_space_memory_usage(self):
        """Test DepthToSpace memory usage patterns"""
        input_shape = [2, 3, 3, 27]
        blocksize = 3
        mode = 'DCR'
        op_name = 'test_depth_to_space_memory'
        i_tensors = [F._from_shape('X', input_shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'DepthToSpace',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'blocksize': blocksize, 'mode': mode}
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Memory usage should be reasonable
        input_elements = i_tensors[0].nelems()
        assert op_perf['inElems'] == input_elements
        assert op_perf['inBytes'] == input_elements * 4  # float32 = 4 bytes

        # Output should have same number of elements (rearranged)
        assert op_perf['outElems'] == input_elements

        # Should have scatter instructions for spatial redistribution
        assert op_perf['instrs']['scatter'] > 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_depth_to_space_invalid_inputs(self):
        """Test DepthToSpace with invalid inputs"""
        op_name = 'test_depth_to_space_invalid'
        i_tensors: list[Any] = []  # Empty input list
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'DepthToSpace',
            'inList': [],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        # This should raise an error due to invalid input count
        with pytest.raises(AssertionError, match="should be in"):
            op_obj.get_perf_counts(i_tensors, o_tensors)

