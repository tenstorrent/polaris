# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`tools.profiling.profiler_polaris_opname_mapping`."""

from __future__ import annotations

import pytest

from tools.profiling.profiler_polaris_opname_mapping import (
    _coerce_output_padded_shape,
    _map_profiler_opcode_to_polaris_optype,
    _untilize_unpadding_logical_output_from_attrs,
    polaris_op_signature,
    profiler_op_signature,
)


@pytest.mark.unit
def test_coerce_output_padded_shape_from_list() -> None:
    assert _coerce_output_padded_shape([8, 14, 32, 1024]) == (8, 14, 32, 1024)
    assert _coerce_output_padded_shape((1, 1, 32, 64)) == (32, 64)


@pytest.mark.unit
def test_coerce_output_padded_shape_from_shape_string() -> None:
    assert _coerce_output_padded_shape('Shape([8, 224, 32, 64])') == (8, 224, 32, 64)
    assert _coerce_output_padded_shape('[8x14x32x1024]') == (8, 14, 32, 1024)


@pytest.mark.unit
def test_untilize_unpadding_logical_output_from_attrs() -> None:
    assert _untilize_unpadding_logical_output_from_attrs(
        {'output_shape': [8, 224, 12, 64]}
    ) == (8, 224, 12, 64)
    assert _untilize_unpadding_logical_output_from_attrs(
        {'output_tensor_end': 'Shape([7, 223, 11, 63])'}
    ) == (8, 224, 12, 64)
    # output_shape takes precedence over output_tensor_end (avoid leading 1 dims for normalize)
    assert _untilize_unpadding_logical_output_from_attrs(
        {
            'output_shape': [8, 16, 32, 64],
            'output_tensor_end': [0, 0, 0, 0],
        }
    ) == (8, 16, 32, 64)


@pytest.mark.unit
def test_polaris_op_signature_tilize_padded_vs_logical() -> None:
    row = {
        'optype': 'TilizeWithValPadding',
        'input_tensors': 'input_0[8x224x12x64]:BFLOAT16',
        'output_tensors': 'output_0[8x224x12x64]:BFLOAT16',
        'attrs': (
            "{'output_padded_shape': 'Shape([8, 224, 32, 64])', 'pad_value': '0'}"
        ),
    }
    sig_default = polaris_op_signature(row)
    sig_tensor_only = polaris_op_signature(row, use_layout_attr_shapes=False)
    assert sig_default[2] == ((8, 224, 32, 64),)
    assert sig_tensor_only[2] == ((8, 224, 12, 64),)
    assert sig_default[0] == sig_tensor_only[0] == 'tilizewithvalpadding'
    assert sig_default[1] == sig_tensor_only[1]


@pytest.mark.unit
def test_polaris_op_signature_untilize_unpadding_uses_attrs() -> None:
    """When output_tensors wrongly show tile-padded size, attrs carry true logical output."""
    row = {
        'optype': 'UntilizeWithUnpadding',
        'input_tensors': 'input_0[8x224x32x64]:BFLOAT16',
        'output_tensors': 'output_0[8x224x32x64]:BFLOAT16',
        'attrs': "{'output_tensor_end': 'Shape([7, 223, 11, 63])'}",
    }
    sig_tensor_only = polaris_op_signature(row, use_layout_attr_shapes=False)
    sig_default = polaris_op_signature(row)
    assert sig_tensor_only[2] == ((8, 224, 32, 64),)
    assert sig_default[2] == ((8, 224, 12, 64),)


def _profiler_tensor_cols(prefix: str, idx: int, w: int, z: int, y: int, x: int, *, tag: str) -> dict[str, str]:
    """Build TTNN-style ``*_W_PAD[tag]`` columns; bracket value is the parsed dimension."""
    p = f'{prefix}_{idx}'
    return {
        f'{p}_W_PAD[{tag}]': f'0[{w}]',
        f'{p}_Z_PAD[{tag}]': f'0[{z}]',
        f'{p}_Y_PAD[{tag}]': f'0[{y}]',
        f'{p}_X_PAD[{tag}]': f'0[{x}]',
        f'{p}_DATATYPE': 'BFLOAT16',
    }


@pytest.mark.unit
def test_profiler_op_signature_tilize_prefers_output_padded_columns() -> None:
    """When *_PAD[PADDED] exists on output, use it for TilizeWithValPadding (knob on)."""
    row = {
        'OP CODE': 'TilizeWithValPadding',
        'ATTRIBUTES': '',
        **_profiler_tensor_cols('INPUT', 0, 8, 224, 12, 64, tag='LOGICAL'),
        **_profiler_tensor_cols('OUTPUT', 0, 8, 224, 12, 64, tag='LOGICAL'),
        **_profiler_tensor_cols('OUTPUT', 0, 8, 224, 32, 64, tag='PADDED'),
    }
    sig_off = profiler_op_signature(row, use_layout_attr_shapes=False)
    sig_on = profiler_op_signature(row, use_layout_attr_shapes=True)
    assert sig_off[2] == ((8, 224, 12, 64),)
    assert sig_on[2] == ((8, 224, 32, 64),)


@pytest.mark.unit
def test_profiler_untilize_unpadding_opcode_identity_mapping() -> None:
    """``UntilizeWithUnpadding`` maps to itself (names now aligned between profiler and Polaris)."""
    from tools.profiling.profiler_polaris_opname_mapping import _map_profiler_opcode_to_polaris_optype

    assert _map_profiler_opcode_to_polaris_optype('UntilizeWithUnpadding', {}) == 'UntilizeWithUnpadding'
    assert (
        _map_profiler_opcode_to_polaris_optype('UntilizeWithUnpaddingDeviceOperation', {})
        == 'UntilizeWithUnpadding'
    )


@pytest.mark.unit
def test_profiler_op_signature_untilize_prefers_input_padded_columns() -> None:
    """UntilizeWithUnpadding: padded tile extent on input_0 when PADDED columns exist."""
    row = {
        'OP CODE': 'UntilizeWithUnpadding',
        'ATTRIBUTES': '',
        **_profiler_tensor_cols('INPUT', 0, 8, 224, 12, 64, tag='LOGICAL'),
        **_profiler_tensor_cols('INPUT', 0, 8, 224, 32, 64, tag='PADDED'),
        **_profiler_tensor_cols('OUTPUT', 0, 8, 224, 12, 64, tag='LOGICAL'),
    }
    sig_off = profiler_op_signature(row, use_layout_attr_shapes=False)
    sig_on = profiler_op_signature(row, use_layout_attr_shapes=True)
    assert sig_off[1] == ((8, 224, 12, 64),)
    assert sig_on[1] == ((8, 224, 32, 64),)
    assert sig_off[2] == sig_on[2] == ((8, 224, 12, 64),)

    legacy = {**row, 'OP CODE': 'UntilizeWithUnpadding'}
    assert profiler_op_signature(legacy, use_layout_attr_shapes=True) == sig_on


@pytest.mark.unit
def test_binary_ng_device_operation_maps_to_add_mul() -> None:
    """New convention: BinaryNgDeviceOperation with binary_op_type in attrs."""
    assert (
        _map_profiler_opcode_to_polaris_optype(
            'BinaryNgDeviceOperation', {'binary_op_type': 'BinaryOpType::ADD'}
        )
        == 'Add'
    )
    assert (
        _map_profiler_opcode_to_polaris_optype(
            'BinaryNgDeviceOperation', {'binary_op_type': 'BinaryOpType::MUL'}
        )
        == 'Mul'
    )


@pytest.mark.unit
def test_legacy_bare_add_mul_still_works() -> None:
    """Old convention: bare ADD/MUL as OP CODE, no binary_op_type in attrs."""
    assert _map_profiler_opcode_to_polaris_optype('ADD', {}) == 'Add'
    assert _map_profiler_opcode_to_polaris_optype('MUL', {}) == 'Mul'


@pytest.mark.unit
def test_create_qkv_heads_maps_to_nlp_create_qkv_heads() -> None:
    """CreateQKVHeadsDeviceOperation maps to NLPCreateQKVHeads via stem_to_lookup."""
    assert (
        _map_profiler_opcode_to_polaris_optype('CreateQKVHeadsDeviceOperation', {})
        == 'NLPCreateQKVHeads'
    )
