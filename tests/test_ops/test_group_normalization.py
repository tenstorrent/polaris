#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for GroupNormalization operation implementation.

This module tests the ONNX GroupNormalization operation, which divides channels
into groups and normalizes within each group. This is commonly used in vision
transformers and other computer vision tasks where batch normalization may not
be suitable due to small or variable batch sizes.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, GroupNormalizationOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_group_normalization_test_tensors(batch_size=2, num_channels=32, height=14, width=14, num_groups=8):
    """
    Helper function to create test tensors for GroupNormalization operation.

    Args:
        batch_size: Batch size for input tensor
        num_channels: Number of channels (must be divisible by num_groups)
        height: Spatial height dimension
        width: Spatial width dimension
        num_groups: Number of groups to divide channels into

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    # Create 4D input tensor [batch_size, num_channels, height, width]
    input_tensor = F._from_shape('input', [batch_size, num_channels, height, width], np_dtype=np.dtype('float32'))

    # Create scale tensor with 1D shape [num_channels]
    scale_tensor = F._from_shape('scale', [num_channels], np_dtype=np.dtype('float32'))

    # Create bias tensor with 1D shape [num_channels]
    bias_tensor = F._from_shape('bias', [num_channels], np_dtype=np.dtype('float32'))

    input_tensors = [input_tensor, scale_tensor, bias_tensor]

    # Create output tensor (same shape as input)
    output_tensor = F._from_shape('output', [batch_size, num_channels, height, width], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


def create_group_normalization_5d_test_tensors(batch_size=2, num_channels=32, depth=8, height=7, width=7, num_groups=8):
    """
    Helper function to create 5D test tensors for GroupNormalization operation.

    Args:
        batch_size: Batch size for input tensor
        num_channels: Number of channels (must be divisible by num_groups)
        depth: Spatial depth dimension
        height: Spatial height dimension
        width: Spatial width dimension
        num_groups: Number of groups to divide channels into

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    # Create 5D input tensor [batch_size, num_channels, depth, height, width]
    input_tensor = F._from_shape('input', [batch_size, num_channels, depth, height, width], np_dtype=np.dtype('float32'))

    # Create scale tensor with 1D shape [num_channels]
    scale_tensor = F._from_shape('scale', [num_channels], np_dtype=np.dtype('float32'))

    # Create bias tensor with 1D shape [num_channels]
    bias_tensor = F._from_shape('bias', [num_channels], np_dtype=np.dtype('float32'))

    input_tensors = [input_tensor, scale_tensor, bias_tensor]

    # Create output tensor (same shape as input)
    output_tensor = F._from_shape('output', [batch_size, num_channels, depth, height, width], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


class TestGroupNormalization:
    """Test class for GroupNormalization operation"""

    def test_factory_integration(self):
        """Test that GroupNormalization is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('GroupNormalization')
        assert opcls == GroupNormalizationOp

    def test_basic_group_normalization(self):
        """Test basic GroupNormalization with standard 4D configuration"""
        batch_size, num_channels, height, width, num_groups = 2, 32, 14, 14, 8

        inT, outT = create_group_normalization_test_tensors(
            batch_size=batch_size, num_channels=num_channels,
            height=height, width=width, num_groups=num_groups
        )

        op_info = {
            'name': 'test_group_norm_basic',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': num_groups,
                'epsilon': 1e-5,
            },
        }
        op = GroupNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape (same as input)
        expected_shape = [batch_size, num_channels, height, width]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mac'] > 0  # Should have multiply-accumulate operations
        assert perf_stats['instrs']['add'] > 0  # Should have additions
        assert perf_stats['instrs']['sub'] > 0  # Should have subtractions
        assert perf_stats['instrs']['rsqrt'] > 0  # Should have reciprocal square root

    def test_group_normalization_5d_tensor(self):
        """Test GroupNormalization with 5D tensor (3D spatial dimensions)"""
        batch_size, num_channels, depth, height, width, num_groups = 2, 32, 8, 7, 7, 8

        inT, outT = create_group_normalization_5d_test_tensors(
            batch_size=batch_size, num_channels=num_channels, depth=depth,
            height=height, width=width, num_groups=num_groups
        )

        op_info = {
            'name': 'test_group_norm_5d',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': num_groups,
                'epsilon': 1e-5,
            },
        }
        op = GroupNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        expected_shape = [batch_size, num_channels, depth, height, width]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        total_elements = batch_size * num_channels * depth * height * width
        channels_per_group = num_channels // num_groups
        spatial_elements = total_elements // num_channels

        assert perf_stats['inElems'] == total_elements + num_channels + num_channels  # input + scale + bias
        assert perf_stats['outElems'] == total_elements

        # Verify group-based operations
        assert perf_stats['instrs']['rsqrt'] == num_groups * channels_per_group  # One per channel per group

    def test_group_normalization_different_groups(self):
        """Test GroupNormalization with different group configurations"""
        test_configs = [
            (32, 8, 1),    # 32 channels, 8 groups (4 channels per group)
            (32, 4, 2),    # 32 channels, 4 groups (8 channels per group)
            (32, 16, 3),   # 32 channels, 16 groups (2 channels per group)
            (64, 8, 4),    # 64 channels, 8 groups (8 channels per group)
        ]

        batch_size, height, width = 2, 14, 14

        for num_channels, num_groups, config_id in test_configs:
            inT, outT = create_group_normalization_test_tensors(
                batch_size=batch_size, num_channels=num_channels,
                height=height, width=width, num_groups=num_groups
            )

            op_info = {
                'name': f'test_group_norm_groups_{config_id}',
                'optype': 'GroupNormalization',
                'inList': ['input', 'scale', 'bias'],
                'outList': ['output'],
                'attrs': {
                    'num_groups': num_groups,
                    'epsilon': 1e-5,
                },
            }
            op = GroupNormalizationOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, num_channels, height, width]
            assert outT[0].shape == expected_shape

            # Verify channels per group calculation
            channels_per_group = num_channels // num_groups
            assert channels_per_group > 0

            # Verify performance scales correctly with group configuration
            total_elements = batch_size * num_channels * height * width
            spatial_elements = total_elements // num_channels

            expected_rsqrt = num_groups * channels_per_group
            assert perf_stats['instrs']['rsqrt'] == expected_rsqrt

    def test_group_normalization_different_shapes(self):
        """Test GroupNormalization with different tensor shapes"""
        test_configs = [
            (1, 32, 28, 28, 4),    # Small input
            (4, 64, 14, 14, 8),    # Standard size
            (2, 128, 7, 7, 16),    # Large channels
            (8, 256, 6, 6, 32),    # Very large
        ]

        for batch_size, num_channels, height, width, num_groups in test_configs:
            inT, outT = create_group_normalization_test_tensors(
                batch_size=batch_size, num_channels=num_channels,
                height=height, width=width, num_groups=num_groups
            )

            op_info = {
                'name': f'test_group_norm_shape_{batch_size}_{num_channels}',
                'optype': 'GroupNormalization',
                'inList': ['input', 'scale', 'bias'],
                'outList': ['output'],
                'attrs': {
                    'num_groups': num_groups,
                    'epsilon': 1e-5,
                },
            }
            op = GroupNormalizationOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, num_channels, height, width]
            assert outT[0].shape == expected_shape

            # Verify element counts
            total_elements = batch_size * num_channels * height * width
            assert perf_stats['inElems'] == total_elements + num_channels + num_channels
            assert perf_stats['outElems'] == total_elements

    def test_group_normalization_large_tensor(self):
        """Test GroupNormalization with large tensor to verify performance scaling"""
        batch_size, num_channels, height, width, num_groups = 4, 256, 14, 14, 32

        inT, outT = create_group_normalization_test_tensors(
            batch_size=batch_size, num_channels=num_channels,
            height=height, width=width, num_groups=num_groups
        )

        op_info = {
            'name': 'test_group_norm_large',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': num_groups,
                'epsilon': 1e-5,
            },
        }
        op = GroupNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        total_elements = batch_size * num_channels * height * width
        expected_elements = 4 * 256 * 14 * 14

        assert perf_stats['inElems'] == expected_elements + num_channels + num_channels
        assert perf_stats['outElems'] == expected_elements

        # Verify performance scaling - should have substantial operations for large tensors
        channels_per_group = num_channels // num_groups
        spatial_elements = total_elements // num_channels

        expected_rsqrt = num_groups * channels_per_group
        assert perf_stats['instrs']['rsqrt'] == expected_rsqrt
        assert perf_stats['instrs']['rsqrt'] == 256  # 32 groups * 8 channels per group

        # Should have substantial operations for large tensors
        assert perf_stats['instrs']['mac'] >= 200000
        assert perf_stats['instrs']['add'] >= 200000
        assert perf_stats['instrs']['sub'] >= 200000

    def test_group_normalization_memory_efficiency(self):
        """Test that GroupNormalization has reasonable memory requirements"""
        batch_size, num_channels, height, width, num_groups = 4, 128, 14, 14, 16

        inT, outT = create_group_normalization_test_tensors(
            batch_size=batch_size, num_channels=num_channels,
            height=height, width=width, num_groups=num_groups
        )

        op_info = {
            'name': 'test_group_norm_memory',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': num_groups,
                'epsilon': 1e-5,
            },
        }
        op = GroupNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Memory requirements should be reasonable
        # Input + scale + bias should be larger than output
        total_elements = batch_size * num_channels * height * width
        assert perf_stats['inBytes'] > perf_stats['outBytes']
        assert perf_stats['inElems'] == total_elements + num_channels + num_channels
        assert perf_stats['outElems'] == total_elements

    def test_invalid_num_groups_none(self):
        """Test error handling for missing num_groups attribute"""
        op_info = {
            'name': 'test_invalid_num_groups',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }

        with pytest.raises(ValueError, match="GroupNormalization requires 'num_groups' attribute"):
            GroupNormalizationOp(op_info)

    def test_invalid_num_groups_zero(self):
        """Test error handling for zero num_groups"""
        op_info = {
            'name': 'test_invalid_num_groups_zero',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': 0,
                'epsilon': 1e-5,
            },
        }

        with pytest.raises(ValueError, match="GroupNormalization num_groups must be positive"):
            GroupNormalizationOp(op_info)

    def test_invalid_epsilon_zero(self):
        """Test error handling for zero epsilon"""
        op_info = {
            'name': 'test_invalid_epsilon',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': 8,
                'epsilon': 0,
            },
        }

        with pytest.raises(ValueError, match="GroupNormalization epsilon must be positive"):
            GroupNormalizationOp(op_info)

    def test_invalid_input_rank_2d(self):
        """Test error handling for 2D input tensor"""
        input_tensor = F._from_shape('input', [4, 64], np_dtype=np.dtype('float32'))
        scale_tensor = F._from_shape('scale', [64], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [64], np_dtype=np.dtype('float32'))

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [4, 64, 1, 1], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_invalid_rank_2d',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': 8,
                'epsilon': 1e-5,
            },
        }
        op = GroupNormalizationOp(op_info)

        with pytest.raises(ValueError, match="GroupNormalization requires input rank >= 3"):
            op.get_perf_counts(inT, outT)

    def test_channels_not_divisible_by_groups(self):
        """Test error handling for channels not divisible by num_groups"""
        input_tensor = F._from_shape('input', [2, 30, 14, 14], np_dtype=np.dtype('float32'))  # 30 channels
        scale_tensor = F._from_shape('scale', [30], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [30], np_dtype=np.dtype('float32'))

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [2, 30, 14, 14], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_channels_not_divisible',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': 8,  # 30 % 8 != 0
                'epsilon': 1e-5,
            },
        }
        op = GroupNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Number of channels .* must be divisible by num_groups"):
            op.get_perf_counts(inT, outT)

    def test_scale_dimension_mismatch(self):
        """Test error handling for scale dimension mismatch"""
        input_tensor = F._from_shape('input', [2, 32, 14, 14], np_dtype=np.dtype('float32'))
        scale_tensor = F._from_shape('scale', [16], np_dtype=np.dtype('float32'))  # Wrong dimension
        bias_tensor = F._from_shape('bias', [32], np_dtype=np.dtype('float32'))

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [2, 32, 14, 14], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_scale_mismatch',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': 8,
                'epsilon': 1e-5,
            },
        }
        op = GroupNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Scale dimension .* must match number of channels"):
            op.get_perf_counts(inT, outT)

    def test_bias_dimension_mismatch(self):
        """Test error handling for bias dimension mismatch"""
        input_tensor = F._from_shape('input', [2, 32, 14, 14], np_dtype=np.dtype('float32'))
        scale_tensor = F._from_shape('scale', [32], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [16], np_dtype=np.dtype('float32'))  # Wrong dimension

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [2, 32, 14, 14], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_bias_mismatch',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': 8,
                'epsilon': 1e-5,
            },
        }
        op = GroupNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Bias dimension .* must match number of channels"):
            op.get_perf_counts(inT, outT)

    def test_scale_not_1d(self):
        """Test error handling for non-1D scale tensor"""
        input_tensor = F._from_shape('input', [2, 32, 14, 14], np_dtype=np.dtype('float32'))
        scale_tensor = F._from_shape('scale', [4, 8], np_dtype=np.dtype('float32'))  # 2D scale
        bias_tensor = F._from_shape('bias', [32], np_dtype=np.dtype('float32'))

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [2, 32, 14, 14], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_scale_not_1d',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': 8,
                'epsilon': 1e-5,
            },
        }
        op = GroupNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Scale tensor must be 1D"):
            op.get_perf_counts(inT, outT)

    def test_bias_not_1d(self):
        """Test error handling for non-1D bias tensor"""
        input_tensor = F._from_shape('input', [2, 32, 14, 14], np_dtype=np.dtype('float32'))
        scale_tensor = F._from_shape('scale', [32], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [4, 8], np_dtype=np.dtype('float32'))  # 2D bias

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [2, 32, 14, 14], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_bias_not_1d',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': 8,
                'epsilon': 1e-5,
            },
        }
        op = GroupNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Bias tensor must be 1D"):
            op.get_perf_counts(inT, outT)

    def test_group_normalization_backward_not_implemented(self):
        """Test that backward pass raises NotImplementedError"""
        batch_size, num_channels, height, width, num_groups = 2, 32, 7, 7, 8

        inT, outT = create_group_normalization_test_tensors(
            batch_size=batch_size, num_channels=num_channels,
            height=height, width=width, num_groups=num_groups
        )

        op_info = {
            'name': 'test_group_norm_backward',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': num_groups,
                'epsilon': 1e-5,
            },
        }
        op = GroupNormalizationOp(op_info)

        # Create dummy gradient tensors
        inGT = [F._from_shape('input_grad', [batch_size, num_channels, height, width], np_dtype=np.dtype('float32')),
                F._from_shape('scale_grad', [num_channels], np_dtype=np.dtype('float32')),
                F._from_shape('bias_grad', [num_channels], np_dtype=np.dtype('float32'))]
        outGT = [F._from_shape('output_grad', [batch_size, num_channels, height, width], np_dtype=np.dtype('float32'))]

        with pytest.raises(NotImplementedError, match="GroupNormalization backward pass not yet implemented"):
            op.backward(inT, outT, inGT, outGT)

    def test_group_normalization_different_dtypes(self):
        """Test GroupNormalization with different data types"""
        dtypes = [np.dtype('float32'), np.dtype('float64')]

        for dtype in dtypes:
            input_tensor = F._from_shape('input', [2, 32, 7, 7], np_dtype=dtype)
            scale_tensor = F._from_shape('scale', [32], np_dtype=dtype)
            bias_tensor = F._from_shape('bias', [32], np_dtype=dtype)

            inT = [input_tensor, scale_tensor, bias_tensor]
            outT = [F._from_shape('output', [2, 32, 7, 7], np_dtype=dtype)]

            op_info = {
                'name': f'test_group_norm_{dtype}',
                'optype': 'GroupNormalization',
                'inList': ['input', 'scale', 'bias'],
                'outList': ['output'],
                'attrs': {
                    'num_groups': 8,
                    'epsilon': 1e-5,
                },
            }
            op = GroupNormalizationOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shapes and element counts
            assert outT[0].shape == [2, 32, 7, 7]
            assert perf_stats['inElems'] == 2 * 32 * 7 * 7 + 32 + 32  # input + scale + bias
            assert perf_stats['outElems'] == 2 * 32 * 7 * 7

            # Memory usage should scale with data type size
            if dtype == np.dtype('float64'):
                # input + scale + bias vs output (same shape as input)
                expected_in_bytes = (2 * 32 * 7 * 7 + 32 + 32) * 8
                expected_out_bytes = (2 * 32 * 7 * 7) * 8
                assert perf_stats['inBytes'] == expected_in_bytes
                assert perf_stats['outBytes'] == expected_out_bytes

    def test_group_normalization_vision_transformer_config(self):
        """Test GroupNormalization with vision transformer configuration"""
        # Vision Transformer configuration: typically uses group norm with small groups
        batch_size, num_channels, height, width, num_groups = 8, 384, 14, 14, 12

        inT, outT = create_group_normalization_test_tensors(
            batch_size=batch_size, num_channels=num_channels,
            height=height, width=width, num_groups=num_groups
        )

        op_info = {
            'name': 'test_vit_group_norm',
            'optype': 'GroupNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {
                'num_groups': num_groups,
                'epsilon': 1e-6,  # ViT often uses smaller epsilon
            },
        }
        op = GroupNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify this represents a realistic ViT configuration
        total_elements = batch_size * num_channels * height * width
        channels_per_group = num_channels // num_groups  # 384 / 12 = 32

        assert perf_stats['inElems'] == total_elements + num_channels + num_channels
        assert perf_stats['outElems'] == total_elements
        assert perf_stats['instrs']['rsqrt'] == num_groups * channels_per_group  # 12 * 32 = 384

        # Should have significant computational requirements for ViT scale
        assert perf_stats['instrs']['mac'] >= 500000
        assert perf_stats['instrs']['add'] >= 500000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
