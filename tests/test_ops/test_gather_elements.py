#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import unittest
import numpy as np
from ttsim.ops.op import SimOpFactory
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


class TestGatherElements(unittest.TestCase):
    """Test cases for GatherElements operation"""

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_factory_integration(self):
        """Test that GatherElementsOp can be created through SimOpFactory"""
        op_cls = SimOpFactory('GatherElements')
        assert op_cls.__name__ == 'GatherElementsOp'

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_gather_elements_basic_functionality(self):
        """Test basic GatherElements operation with different configurations"""
        test_configs = [
            ([3, 4], [3, 4], 0),    # Same shape, axis=0
            ([3, 4], [3, 4], 1),    # Same shape, axis=1
            ([2, 3, 4], [2, 3, 4], -1),  # 3D, axis=-1
        ]

        for data_shape, indices_shape, axis in test_configs:
            with self.subTest(data_shape=data_shape, indices_shape=indices_shape, axis=axis):
                op_name = f'test_gather_elements_axis_{axis}'
                i_tensors = [
                    F._from_shape('data', data_shape, np_dtype=np.float32),
                    F._from_shape('indices', indices_shape, np_dtype=np.int64)
                ]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'GatherElements',
                    'inList': [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors],
                    'attrs': {'axis': axis}
                }

                op_cls = SimOpFactory('GatherElements')
                op_obj = op_cls(op_info)

                for x in i_tensors: x.op_in = [op_name]
                for x in o_tensors: x.op_out = [op_name]

                op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

                # Verify attributes are set correctly
                assert op_obj.axis == axis

                # Verify output shape is valid
                assert o_tensors[0].shape is not None
                assert o_tensors[0].dtype == np.float32

                # Verify performance stats
                assert 'inElems' in op_perf
                assert 'outElems' in op_perf
                assert 'inBytes' in op_perf
                assert 'outBytes' in op_perf
                assert 'instrs' in op_perf
                assert 'gather' in op_perf['instrs']
                assert 'index' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_gather_elements_different_axes(self):
        """Test GatherElements with different axis values"""
        data_shape = [2, 3, 4]
        indices_shape = [2, 3, 4]
        axes = [0, 1, 2, -1, -2, -3]

        for axis in axes:
            with self.subTest(axis=axis):
                op_name = f'test_gather_elements_axis_{axis}'
                i_tensors = [
                    F._from_shape('data', data_shape, np_dtype=np.float32),
                    F._from_shape('indices', indices_shape, np_dtype=np.int64)
                ]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'GatherElements',
                    'inList': [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors],
                    'attrs': {'axis': axis}
                }

                op_cls = SimOpFactory('GatherElements')
                op_obj = op_cls(op_info)

                for x in i_tensors: x.op_in = [op_name]
                for x in o_tensors: x.op_out = [op_name]

                op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

                # Verify axis is set correctly
                assert op_obj.axis == axis

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_gather_elements_default_axis(self):
        """Test GatherElements with default axis value"""
        op_name = 'test_gather_elements_default_axis'
        i_tensors = [
            F._from_shape('data', [3, 4], np_dtype=np.float32),
            F._from_shape('indices', [3, 4], np_dtype=np.int64)
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'GatherElements',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_cls = SimOpFactory('GatherElements')
        op_obj = op_cls(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Verify default axis is 0
        assert op_obj.axis == 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_gather_elements_performance_consistency(self):
        """Test that GatherElements performance is consistent across runs"""
        data_shape = [2, 3, 4]
        indices_shape = [2, 3, 4]
        op_name = 'test_gather_elements_perf_consistency'
        i_tensors = [
            F._from_shape('data', data_shape, np_dtype=np.float32),
            F._from_shape('indices', indices_shape, np_dtype=np.int64)
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'GatherElements',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_cls = SimOpFactory('GatherElements')
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
    def test_gather_elements_memory_usage(self):
        """Test GatherElements memory usage patterns"""
        data_shape = [10, 20]
        indices_shape = [10, 20]
        op_name = 'test_gather_elements_memory'
        i_tensors = [
            F._from_shape('data', data_shape, np_dtype=np.float32),
            F._from_shape('indices', indices_shape, np_dtype=np.int64)
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'GatherElements',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_cls = SimOpFactory('GatherElements')
        op_obj = op_cls(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Memory usage should be reasonable
        data_elements = i_tensors[0].nelems()
        indices_elements = i_tensors[1].nelems()
        assert op_perf['inElems'] >= data_elements + indices_elements

        # Output should have same number of elements as indices
        assert op_perf['outElems'] == indices_elements

        # Should have gather and index instructions
        assert op_perf['instrs']['gather'] > 0
        assert op_perf['instrs']['index'] > 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_gather_elements_invalid_inputs(self):
        """Test GatherElements with invalid inputs"""
        op_name = 'test_gather_elements_invalid'
        i_tensors = [F._from_shape('data', [3, 4], np_dtype=np.float32)]  # Missing indices tensor
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'GatherElements',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_cls = SimOpFactory('GatherElements')
        op_obj = op_cls(op_info)

        # This should raise an error due to missing indices tensor
        with pytest.raises(Exception, match="should be in range"):
            op_obj.get_perf_counts(i_tensors, o_tensors)

