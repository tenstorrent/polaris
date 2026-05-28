#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools.profiling.show_layers_polaris."""

import pytest

from tools.profiling.show_layers_polaris import _parse_lut_key_cell


@pytest.mark.unit
def test_parse_lut_key_cell_none_and_empty_return_none():
    assert _parse_lut_key_cell(None) is None
    assert _parse_lut_key_cell('') is None
    assert _parse_lut_key_cell('   ') is None


@pytest.mark.unit
def test_parse_lut_key_cell_string_none_variants_return_none():
    """Cells that hold the literal string ``None`` / ``NA`` / ``null`` (CSV null sentinels)
    must parse back to None, matching how the polaris CSV writer renders missing values.
    """
    assert _parse_lut_key_cell('None') is None
    assert _parse_lut_key_cell('none') is None
    assert _parse_lut_key_cell('NA') is None
    assert _parse_lut_key_cell('null') is None


@pytest.mark.unit
def test_parse_lut_key_cell_arity1_tuple_roundtrip():
    """str(tuple) → tuple round-trip for a 9-tuple arity-1 LUT key."""
    key = ('halo', 1, 1, 65536, 16, 'ROW_MAJOR', 'BFLOAT16', 'DEV_1_L1_HEIGHT_SHARDED', 'N/A')
    parsed = _parse_lut_key_cell(repr(key))
    assert parsed == key


@pytest.mark.unit
def test_parse_lut_key_cell_arity2_tuple_roundtrip():
    """str(tuple) → tuple round-trip for a 16-tuple arity-2 LUT key (Move dup form)."""
    key = (
        'move', 1, 1, 99072, 16, 'ROW_MAJOR', 'BFLOAT16', 'DEV_1_L1_HEIGHT_SHARDED', 'N/A',
        1, 1, 99072, 16, 'ROW_MAJOR', 'BFLOAT16', 'DEV_1_L1_HEIGHT_SHARDED',
    )
    parsed = _parse_lut_key_cell(repr(key))
    assert parsed == key


@pytest.mark.unit
def test_parse_lut_key_cell_invalid_returns_none():
    """Malformed cells degrade gracefully — round-trip is best-effort, not a hard contract."""
    assert _parse_lut_key_cell('not a tuple') is None
    assert _parse_lut_key_cell('(unterminated') is None
    # Lists are not tuples — they're a valid Python literal but the wrong shape.
    assert _parse_lut_key_cell('[1, 2, 3]') is None
    # Plain int / string don't roundtrip to a tuple either.
    assert _parse_lut_key_cell('42') is None
