#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import ttsim.utils.types as types


@pytest.mark.unit
def test_types():
    assert types.socnodetype2str(types.SOCNodeType.CORE) == 'C'
    assert types.str2df('float16') == types.DataFormat.FP16A
    assert types.str2df(['float16', 'float16_b']) == [types.DataFormat.FP16A, types.DataFormat.FP16B]
    assert types.str2mf('lofi') == types.MathFidelity.LoFi
    assert types.get_sim_dtype('BOOL') == types.SimDataType.BOOL
    assert types.get_sim_dtype('wrongtype') == types.SimDataType.UNKNOWN
    assert types.get_bpe(types.SimDataType.INT64) == 8


@pytest.mark.unit
def test_get_sim_dtype_abbreviations():
    """Test that common datatype abbreviations are correctly mapped."""
    # Test bf16 abbreviation (the main bug fix)
    assert types.get_sim_dtype('bf16') == types.SimDataType.BFLOAT16
    assert types.get_sim_dtype('BF16') == types.SimDataType.BFLOAT16
    assert types.get_sim_dtype('Bf16') == types.SimDataType.BFLOAT16

    # Test other abbreviations
    assert types.get_sim_dtype('fp16') == types.SimDataType.FLOAT16
    assert types.get_sim_dtype('fp32') == types.SimDataType.FLOAT32
    assert types.get_sim_dtype('fp64') == types.SimDataType.FLOAT64

    # Test full names still work
    assert types.get_sim_dtype('bfloat16') == types.SimDataType.BFLOAT16
    assert types.get_sim_dtype('float16') == types.SimDataType.FLOAT16
    assert types.get_sim_dtype('float32') == types.SimDataType.FLOAT32
    assert types.get_sim_dtype('float64') == types.SimDataType.FLOAT64


@pytest.mark.unit
def test_get_sim_dtype_invalid():
    """Test that invalid datatypes return UNKNOWN."""
    assert types.get_sim_dtype('invalid_type') == types.SimDataType.UNKNOWN
    assert types.get_sim_dtype('xyz123') == types.SimDataType.UNKNOWN
    assert types.get_sim_dtype('') == types.SimDataType.UNKNOWN


@pytest.mark.unit
def test_validate_datatype_valid():
    """Test that validate_datatype accepts valid datatypes."""
    # Test abbreviations
    assert types.validate_datatype('bf16') == 'bf16'
    assert types.validate_datatype('fp16') == 'fp16'
    assert types.validate_datatype('fp32') == 'fp32'
    assert types.validate_datatype('fp64') == 'fp64'

    # Test full names
    assert types.validate_datatype('bfloat16') == 'bfloat16'
    assert types.validate_datatype('float32') == 'float32'
    assert types.validate_datatype('int8') == 'int8'

    # Test case insensitivity
    assert types.validate_datatype('BF16') == 'BF16'
    assert types.validate_datatype('Bf16') == 'Bf16'


@pytest.mark.unit
def test_validate_datatype_invalid():
    """Test that validate_datatype raises ValueError for invalid datatypes."""
    with pytest.raises(ValueError, match="Invalid datatype"):
        types.validate_datatype('invalid_type')

    with pytest.raises(ValueError, match="Invalid datatype"):
        types.validate_datatype('xyz123')

    with pytest.raises(ValueError, match="Invalid datatype"):
        types.validate_datatype('unknown')

    # Verify the error message includes valid options
    try:
        types.validate_datatype('bad_type')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        assert 'bad_type' in error_msg
        assert 'Valid options' in error_msg


@pytest.mark.unit
def test_get_bpe_never_negative():
    """Test that get_bpe never returns negative values for valid datatypes."""
    # Test all valid datatypes
    valid_dtypes = [
        types.SimDataType.BOOL,
        types.SimDataType.INT4,
        types.SimDataType.INT8,
        types.SimDataType.INT16,
        types.SimDataType.INT32,
        types.SimDataType.INT64,
        types.SimDataType.UINT4,
        types.SimDataType.UINT8,
        types.SimDataType.UINT16,
        types.SimDataType.UINT32,
        types.SimDataType.UINT64,
        types.SimDataType.BFLOAT8,
        types.SimDataType.BFLOAT16,
        types.SimDataType.FLOAT16,
        types.SimDataType.FLOAT32,
        types.SimDataType.FLOAT64,
    ]

    for dtype in valid_dtypes:
        bpe = types.get_bpe(dtype)
        assert bpe > 0, f"get_bpe({dtype}) returned {bpe}, expected positive value"

    # UNKNOWN should return -1 (this is expected behavior)
    assert types.get_bpe(types.SimDataType.UNKNOWN) == -1
