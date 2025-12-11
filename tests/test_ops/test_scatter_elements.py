#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import unittest
import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


class TestScatterElements(unittest.TestCase):
    """Test cases for ScatterElements operation"""

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_factory_integration(self):
        """Test that ScatterElements is registered in the op mapping"""
        from ttsim.ops.desc.registry import get_opdesc_registry
        assert get_opdesc_registry().has_shape_inference_function('ScatterElements')

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_scatter_elements_basic_functionality(self):
        """Test basic ScatterElements operation with different configurations"""
        test_configs = [
            ([3, 4], [3, 4], [3, 4], 0, 'none'),    # Same shapes, axis=0, reduction=none
            ([3, 4], [3, 4], [3, 4], 1, 'add'),     # Same shapes, axis=1, reduction=add
            ([2, 3, 4], [2, 3, 4], [2, 3, 4], -1, 'mul'),  # 3D, axis=-1, reduction=mul
        ]

        for data_shape, indices_shape, updates_shape, axis, reduction in test_configs:
            with self.subTest(data_shape=data_shape, indices_shape=indices_shape, axis=axis, reduction=reduction):
                op_name = f'test_scatter_elements_axis_{axis}_{reduction}'
                i_tensors = [
                    F._from_shape('data', data_shape, np_dtype=np.float32),
                    F._from_shape('indices', indices_shape, np_dtype=np.int64),
                    F._from_shape('updates', updates_shape, np_dtype=np.float32)
                ]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'ScatterElements',
                    'inList': [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors],
                    'attrs': {'axis': axis, 'reduction': reduction}
                }

                op_obj = SimOp(op_info)

                for x in i_tensors: x.op_in = [op_name]
                for x in o_tensors: x.op_out = [op_name]

                op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

                # Verify attributes are set correctly
                assert op_obj.attrs['axis'] == axis
                assert op_obj.attrs['reduction'] == reduction

                # Verify output shape is valid
                assert o_tensors[0].shape is not None
                assert o_tensors[0].dtype == np.float32

                # Verify performance stats
                assert 'inElems' in op_perf
                assert 'outElems' in op_perf
                assert 'inBytes' in op_perf
                assert 'outBytes' in op_perf
                assert 'instrs' in op_perf
                assert 'scatter' in op_perf['instrs']
                assert 'index' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_scatter_elements_different_reductions(self):
        """Test ScatterElements with different reduction modes"""
        data_shape = [3, 4]
        indices_shape = [3, 4]
        updates_shape = [3, 4]
        axis = 0
        reductions = ['none', 'add', 'mul']

        for reduction in reductions:
            with self.subTest(reduction=reduction):
                op_name = f'test_scatter_elements_reduction_{reduction}'
                i_tensors = [
                    F._from_shape('data', data_shape, np_dtype=np.float32),
                    F._from_shape('indices', indices_shape, np_dtype=np.int64),
                    F._from_shape('updates', updates_shape, np_dtype=np.float32)
                ]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'ScatterElements',
                    'inList': [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors],
                    'attrs': {'axis': axis, 'reduction': reduction}
                }

                op_obj = SimOp(op_info)

                for x in i_tensors: x.op_in = [op_name]
                for x in o_tensors: x.op_out = [op_name]

                op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

                # Verify reduction is set correctly
                assert op_obj.attrs['reduction'] == reduction

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_scatter_elements_default_attributes(self):
        """Test ScatterElements with default attributes"""
        op_name = 'test_scatter_elements_default_attrs'
        i_tensors = [
            F._from_shape('data', [3, 4], np_dtype=np.float32),
            F._from_shape('indices', [3, 4], np_dtype=np.int64),
            F._from_shape('updates', [3, 4], np_dtype=np.float32)
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'ScatterElements',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'axis': 0, 'reduction': 'none'}  # Default attributes
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Verify default attributes
        assert op_obj.attrs['axis'] == 0
        assert op_obj.attrs['reduction'] == 'none'

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_scatter_elements_performance_consistency(self):
        """Test that ScatterElements performance is consistent across runs"""
        data_shape = [2, 3, 4]
        indices_shape = [2, 3, 4]
        updates_shape = [2, 3, 4]
        op_name = 'test_scatter_elements_perf_consistency'
        i_tensors = [
            F._from_shape('data', data_shape, np_dtype=np.float32),
            F._from_shape('indices', indices_shape, np_dtype=np.int64),
            F._from_shape('updates', updates_shape, np_dtype=np.float32)
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'ScatterElements',
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
    def test_scatter_elements_memory_usage(self):
        """Test ScatterElements memory usage patterns"""
        data_shape = [10, 20]
        indices_shape = [10, 20]
        updates_shape = [10, 20]
        op_name = 'test_scatter_elements_memory'
        i_tensors = [
            F._from_shape('data', data_shape, np_dtype=np.float32),
            F._from_shape('indices', indices_shape, np_dtype=np.int64),
            F._from_shape('updates', updates_shape, np_dtype=np.float32)
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'ScatterElements',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Memory usage should be reasonable
        data_elements = i_tensors[0].nelems()
        indices_elements = i_tensors[1].nelems()
        updates_elements = i_tensors[2].nelems()
        assert op_perf['inElems'] >= data_elements + indices_elements + updates_elements

        # Output should have same memory usage as input data
        assert op_perf['outElems'] == data_elements

        # Should have scatter and index instructions
        assert op_perf['instrs']['scatter'] > 0
        assert op_perf['instrs']['index'] > 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_scatter_elements_invalid_inputs(self):
        """Test ScatterElements with invalid inputs"""
        op_name = 'test_scatter_elements_invalid'
        i_tensors = [
            F._from_shape('data', [3, 4], np_dtype=np.float32),
            F._from_shape('indices', [3, 4], np_dtype=np.int64)
            # Missing updates tensor
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'ScatterElements',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_obj = SimOp(op_info)

        # This should raise an error due to missing updates tensor
        with pytest.raises(Exception, match="should be in range"):
            op_obj.get_perf_counts(i_tensors, o_tensors)

