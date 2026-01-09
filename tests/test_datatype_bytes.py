#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests to ensure inBytes and outBytes are never negative when using datatype parameter.
This test specifically addresses the bug where --datatype bf16 caused negative byte values.
"""

import pytest

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import SimTensor
from ttsim.utils.types import SimDataType, get_sim_dtype


@pytest.mark.unit
def test_nbytes_with_bf16_precision():
    """Test that nbytes calculation with bf16 precision produces non-negative values."""
    # Create a tensor with some elements
    tensor = SimTensor({
        'name': 'test_tensor',
        'shape': [10, 20, 30],  # 6000 elements
        'dtype': 'float32'
    })

    # Test with bf16 precision (the problematic case)
    nbytes_bf16 = tensor.nbytes('bf16')
    assert nbytes_bf16 > 0, f"nbytes with bf16 precision returned {nbytes_bf16}, expected positive value"
    assert nbytes_bf16 == 6000 * 2, f"Expected 6000 * 2 = 12000 bytes for bf16, got {nbytes_bf16}"

    # Test with BF16 (uppercase)
    nbytes_BF16 = tensor.nbytes('BF16')
    assert nbytes_BF16 > 0, f"nbytes with BF16 precision returned {nbytes_BF16}, expected positive value"
    assert nbytes_BF16 == nbytes_bf16, "BF16 and bf16 should produce same result"

    # Test with BFLOAT16 (full name)
    nbytes_bfloat16 = tensor.nbytes('bfloat16')
    assert nbytes_bfloat16 > 0, f"nbytes with bfloat16 precision returned {nbytes_bfloat16}, expected positive value"
    assert nbytes_bfloat16 == nbytes_bf16, "bfloat16 and bf16 should produce same result"


@pytest.mark.unit
def test_nbytes_with_all_datatype_abbreviations():
    """Test that nbytes calculation works correctly with all datatype abbreviations."""
    tensor = SimTensor({
        'name': 'test_tensor',
        'shape': [100],  # 100 elements
        'dtype': 'float32'
    })

    # Test all abbreviations that should work
    test_cases = [
        ('bf16', 2),      # bfloat16 = 2 bytes
        ('fp16', 2),      # float16 = 2 bytes
        ('fp32', 4),      # float32 = 4 bytes
        ('fp64', 8),      # float64 = 8 bytes
        ('int8', 1),      # int8 = 1 byte
        ('int32', 4),     # int32 = 4 bytes
        ('bfloat16', 2),  # full name
        ('float32', 4),   # full name
    ]

    for dtype_str, expected_bytes_per_elem in test_cases:
        nbytes = tensor.nbytes(dtype_str)
        expected = 100 * expected_bytes_per_elem
        assert nbytes > 0, f"nbytes with {dtype_str} returned {nbytes}, expected positive value"
        assert nbytes == expected, \
            f"nbytes with {dtype_str}: expected {expected}, got {nbytes}"


@pytest.mark.unit
def test_nbytes_never_negative_with_valid_precisions():
    """Test that nbytes never returns negative values for any valid precision."""
    tensor = SimTensor({
        'name': 'test_tensor',
        'shape': [5, 10, 15],  # 750 elements
        'dtype': 'float32'
    })

    # Test all valid datatype strings from the command line
    valid_datatypes = [
        'bf16', 'fp16', 'fp32', 'fp64', 'int8',
        'bfloat16', 'float16', 'float32', 'float64',
        'int16', 'int32', 'int64',
        'uint8', 'uint16', 'uint32', 'uint64',
    ]

    for dtype in valid_datatypes:
        sim_dtype = get_sim_dtype(dtype)
        if sim_dtype != SimDataType.UNKNOWN:
            nbytes = tensor.nbytes(dtype)
            assert nbytes >= 0, \
                f"nbytes with {dtype} returned {nbytes}, expected non-negative value"
            # For valid types, should be positive (tensor has elements)
            if tensor.nelems() > 0:
                assert nbytes > 0, \
                    f"nbytes with {dtype} returned {nbytes}, expected positive value for non-empty tensor"


@pytest.mark.unit
def test_perf_stats_inbytes_outbytes_non_negative():
    """Test that perf_stats inBytes and outBytes are non-negative for operations with different precisions."""
    from ttsim.ops.desc.tensor import cast_sinf

    # Create input and output tensors
    input_tensor = SimTensor({
        'name': 'input',
        'shape': [10, 20],  # 200 elements
        'dtype': 'float32'
    })

    output_tensor = SimTensor({
        'name': 'output',
        'shape': [10, 20],  # 200 elements
        'dtype': 'float32'
    })

    # Test with bf16 precision (the problematic case)
    op_bf16 = SimOp({
        'name': 'cast_op_bf16',
        'optype': 'Cast',
        'inList': ['input'],
        'outList': ['output']
    })
    op_bf16.precision = 'bf16'

    # Call shape inference which sets perf_stats
    cast_sinf([input_tensor], [output_tensor], op_bf16)

    assert op_bf16.perf_stats is not None, "perf_stats should be set"
    assert 'inBytes' in op_bf16.perf_stats
    assert 'outBytes' in op_bf16.perf_stats

    inBytes = op_bf16.perf_stats['inBytes']
    outBytes = op_bf16.perf_stats['outBytes']

    assert inBytes > 0, f"inBytes with bf16 precision returned {inBytes}, expected positive value"
    assert outBytes > 0, f"outBytes with bf16 precision returned {outBytes}, expected positive value"
    assert inBytes == 200 * 2, f"Expected 200 * 2 = 400 bytes for bf16 input, got {inBytes}"
    assert outBytes == 200 * 2, f"Expected 200 * 2 = 400 bytes for bf16 output, got {outBytes}"

    # Test with other precisions
    for precision in ['fp16', 'fp32', 'int8', 'int32']:
        op = SimOp({
            'name': f'cast_op_{precision}',
            'optype': 'Cast',
            'inList': ['input'],
            'outList': ['output']
        })
        op.precision = precision

        cast_sinf([input_tensor], [output_tensor], op)

        assert op.perf_stats is not None, "perf_stats should be set"
        assert op.perf_stats['inBytes'] > 0, \
            f"inBytes with {precision} precision returned {op.perf_stats['inBytes']}, expected positive value"
        assert op.perf_stats['outBytes'] > 0, \
            f"outBytes with {precision} precision returned {op.perf_stats['outBytes']}, expected positive value"


@pytest.mark.unit
def test_invalid_datatype_causes_error():
    """Test that using an invalid datatype causes appropriate error handling."""
    tensor = SimTensor({
        'name': 'test_tensor',
        'shape': [10, 10],  # 100 elements
        'dtype': 'float32'
    })

    # Test that invalid datatype results in UNKNOWN -> -1 bytes per element
    invalid_dtype = 'invalid_type_xyz'
    sim_dtype = get_sim_dtype(invalid_dtype)
    assert sim_dtype == SimDataType.UNKNOWN, "Invalid dtype should map to UNKNOWN"

    # nbytes with invalid precision should return negative value (this is the bug scenario)
    nbytes = tensor.nbytes(invalid_dtype)
    assert nbytes < 0, f"nbytes with invalid dtype should return negative value, got {nbytes}"
    assert nbytes == -100, f"Expected -100 (100 elements * -1 bytes), got {nbytes}"

    # This demonstrates why validate_datatype should be used to catch this early
    from ttsim.utils.types import validate_datatype
    with pytest.raises(ValueError, match="Invalid datatype"):
        validate_datatype(invalid_dtype)


@pytest.mark.unit
def test_validate_datatype_terminates_on_invalid():
    """Test that validate_datatype raises ValueError for invalid datatypes, causing termination."""
    from ttsim.utils.types import validate_datatype

    # Test various invalid datatype strings
    invalid_datatypes = [
        'invalid_type',
        'xyz123',
        'unknown',
        'bad_datatype',
        'not_a_type',
        '',  # Empty string
    ]

    for invalid_dtype in invalid_datatypes:
        with pytest.raises(ValueError) as exc_info:
            validate_datatype(invalid_dtype)

        # Verify the error message is informative
        error_msg = str(exc_info.value)
        assert 'Invalid datatype' in error_msg or 'Invalid' in error_msg
        assert invalid_dtype in error_msg or 'Valid options' in error_msg


@pytest.mark.unit
def test_invalid_datatype_in_operation_causes_negative_bytes():
    """Test that operations with invalid datatype precision produce negative inBytes/outBytes."""
    from ttsim.ops.desc.tensor import cast_sinf

    input_tensor = SimTensor({
        'name': 'input',
        'shape': [10, 10],  # 100 elements
        'dtype': 'float32'
    })

    output_tensor = SimTensor({
        'name': 'output',
        'shape': [10, 10],  # 100 elements
        'dtype': 'float32'
    })

    # Create operation with invalid precision
    op_invalid = SimOp({
        'name': 'cast_op_invalid',
        'optype': 'Cast',
        'inList': ['input'],
        'outList': ['output']
    })
    op_invalid.precision = 'invalid_type_xyz'  # Invalid datatype

    # Call shape inference which sets perf_stats
    cast_sinf([input_tensor], [output_tensor], op_invalid)

    # This demonstrates the bug: invalid datatype leads to negative bytes
    assert op_invalid.perf_stats is not None, "perf_stats should be set"
    assert op_invalid.perf_stats['inBytes'] < 0, \
        "Invalid datatype should produce negative inBytes (demonstrating the bug)"
    assert op_invalid.perf_stats['outBytes'] < 0, \
        "Invalid datatype should produce negative outBytes (demonstrating the bug)"

    # This is why validate_datatype should be called before setting precision
    from ttsim.utils.types import validate_datatype
    with pytest.raises(ValueError):
        validate_datatype('invalid_type_xyz')


@pytest.mark.unit
def test_override_datatypes_are_validated():
    """Test that datatypes in op_data_type_spec.override are validated when loading config."""
    from ttsim.config.wl2archmap import WL2ArchDatatypes

    # Test valid override datatypes
    valid_spec = {
        'global_type': 'int8',
        'override': {
            'dropout': 'int32',
            'conv': 'fp16'
        }
    }
    # Should not raise an error
    data_type_spec = WL2ArchDatatypes.from_dict(valid_spec)
    assert data_type_spec.global_type == 'int8'
    assert data_type_spec.override['DROPOUT'] == 'int32'
    assert data_type_spec.override['CONV'] == 'fp16'

    # Test invalid override datatype
    invalid_spec = {
        'global_type': 'int8',
        'override': {
            'dropout': 'invalid_type_xyz'
        }
    }
    # Should raise ValueError
    with pytest.raises(ValueError, match="Invalid datatype"):
        WL2ArchDatatypes.from_dict(invalid_spec)

    # Test invalid global_type
    invalid_global_spec = {
        'global_type': 'invalid_type_xyz',
        'override': {}
    }
    # Should raise ValueError
    with pytest.raises(ValueError, match="Invalid datatype"):
        WL2ArchDatatypes.from_dict(invalid_global_spec)

    # Test that update_global_type also validates
    valid_spec_obj = WL2ArchDatatypes.from_dict(valid_spec)
    # Valid update should work
    valid_spec_obj.update_global_type('bf16')
    assert valid_spec_obj.global_type == 'bf16'

    # Invalid update should raise ValueError
    with pytest.raises(ValueError, match="Invalid datatype"):
        valid_spec_obj.update_global_type('invalid_type_xyz')
