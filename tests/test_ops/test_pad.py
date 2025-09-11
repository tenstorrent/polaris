#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import unittest
import numpy as np
from ttsim.ops.op import SimOpFactory
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


class TestPad(unittest.TestCase):
    """Test cases for Pad operation"""

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_factory_integration(self):
        """Test that PadOp can be created through SimOpFactory"""
        op_cls = SimOpFactory('Pad')
        assert op_cls.__name__ == 'PadOp'

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_pad_basic_functionality(self):
        """Test basic Pad operation with different configurations"""
        test_configs = [
            ([4], [2, 2]),              # 1D padding
            ([3, 4], [1, 2, 1, 2]),     # 2D padding
            ([2, 3, 4], [1, 1, 2, 2, 1, 1])  # 3D padding
        ]

        for input_shape, pads_shape in test_configs:
            with self.subTest(input_shape=input_shape, pads_shape=pads_shape):
                op_name = f'test_pad_{len(input_shape)}d'
                i_tensors = [
                    F._from_shape('X', input_shape, np_dtype=np.float32),
                    F._from_shape('pads', pads_shape, np_dtype=np.int64)  # pads tensor
                ]
                o_tensors = [make_tensor('Y')]

                op_info = {
                    'name': op_name,
                    'optype': 'Pad',
                    'inList': [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors]
                }

                op_cls = SimOpFactory('Pad')
                op_obj = op_cls(op_info)

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
                assert 'mov' in op_perf['instrs']

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_pad_with_attributes(self):
        """Test Pad with mode and value attributes"""
        op_name = 'test_pad_with_attributes'
        i_tensors = [
            F._from_shape('X', [3, 4], np_dtype=np.float32),
            F._from_shape('pads', [4], np_dtype=np.int64)  # pads tensor
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'Pad',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'mode': 'constant', 'value': 5.0}
        }

        op_cls = SimOpFactory('Pad')
        op_obj = op_cls(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Verify attributes are set correctly
        assert op_obj.attrs['mode'] == 'constant'
        assert op_obj.attrs['value'] == 5.0

        # Verify performance stats include padding overhead
        assert 'add' in op_perf['instrs']  # For padding elements

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_pad_performance_consistency(self):
        """Test that Pad performance is consistent across runs"""
        input_shape = [3, 4]
        pads_shape = [4]
        op_name = 'test_pad_perf_consistency'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_shape('pads', pads_shape, np_dtype=np.int64)
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'Pad',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_cls = SimOpFactory('Pad')
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
    def test_pad_memory_usage(self):
        """Test Pad memory usage patterns"""
        input_shape = [10, 20]
        pads_shape = [4]
        op_name = 'test_pad_memory'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_shape('pads', pads_shape, np_dtype=np.int64)
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'Pad',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_cls = SimOpFactory('Pad')
        op_obj = op_cls(op_info)

        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        # Memory usage should be reasonable
        input_elements = i_tensors[0].nelems()
        pads_elements = i_tensors[1].nelems()
        assert op_perf['inElems'] >= input_elements + pads_elements

        # Output should be larger than input due to padding
        assert op_perf['outElems'] > input_elements

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_pad_invalid_inputs(self):
        """Test Pad with invalid inputs"""
        op_name = 'test_pad_invalid'
        i_tensors = [F._from_shape('X', [3, 4], np_dtype=np.float32)]  # Missing pads tensor
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name': op_name,
            'optype': 'Pad',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }

        op_cls = SimOpFactory('Pad')
        op_obj = op_cls(op_info)

        # This should raise an error due to missing pads tensor
        with pytest.raises(Exception, match="should be in range"):
            op_obj.get_perf_counts(i_tensors, o_tensors)

