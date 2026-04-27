#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools.profiling.compare_layers."""

import pytest
from tools.profiling.compare_layers import (
    compare_layers,
    find_next_match,
    normalize_optype,
)


# ── find_next_match tests ──────────────────────────────────────────────


@pytest.mark.unit
def test_find_next_match_basic():
    """Test basic matching of operation types."""
    layers = [
        {'optype': 'MatMul'},
        {'optype': 'Add'},
        {'optype': 'Softmax'},
        {'optype': 'MatMul'},
    ]

    # Find first MatMul
    result = find_next_match(layers, 0, normalize_optype('MatMul'))
    assert result == 0

    # Find second MatMul
    result = find_next_match(layers, 1, normalize_optype('MatMul'))
    assert result == 3

    # Find Add
    result = find_next_match(layers, 0, normalize_optype('Add'))
    assert result == 1


@pytest.mark.unit
def test_find_next_match_not_found():
    """Test when operation type is not found."""
    layers = [
        {'optype': 'MatMul'},
        {'optype': 'Add'},
    ]

    result = find_next_match(layers, 0, normalize_optype('Softmax'))
    assert result is None


@pytest.mark.unit
def test_find_next_match_with_max_distance():
    """Test max_distance parameter limits search range."""
    layers = [
        {'optype': 'MatMul'},
        {'optype': 'Add'},
        {'optype': 'Softmax'},
        {'optype': 'MatMul'},
    ]

    # Should find it within distance
    result = find_next_match(layers, 0, normalize_optype('Softmax'), max_distance=5)
    assert result == 2

    # Should not find it (too far)
    result = find_next_match(layers, 0, normalize_optype('Softmax'), max_distance=2)
    assert result is None


@pytest.mark.unit
def test_find_next_match_normalization():
    """Test that operation type normalization works in matching."""
    layers = [
        {'optype': 'MatMul'},
        {'optype': 'Add'},
    ]

    # 'MatMul' normalizes to 'matmul'
    result = find_next_match(layers, 0, normalize_optype('MatMul'))
    assert result == 0

    # Test with different case
    result = find_next_match(layers, 0, normalize_optype('matmul'))
    assert result == 0


# ── compare_layers tests ──────────────────────────────────────────────


@pytest.mark.unit
def test_compare_layers_exact_match():
    """Test comparison with exact matching layers."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        }
    ]

    stats = compare_layers(polaris_layers, profiler_layers)
    assert stats.total_matches == 1
    assert stats.unmatched_polaris == 0
    assert stats.unmatched_profiler == 0


@pytest.mark.unit
def test_compare_layers_type_mismatch():
    """Test comparison with mismatched operation types."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'Add',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x128'],
        }
    ]

    stats = compare_layers(polaris_layers, profiler_layers)
    assert stats.total_matches == 0
    # When types don't match and polaris op not found ahead, marked as name_mismatch
    assert stats.name_mismatches == 1
    # Remaining profiler entry becomes unmatched
    assert stats.unmatched_profiler == 1


@pytest.mark.unit
def test_compare_layers_out_of_order():
    """Test comparison with out-of-order operations - forward search only."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        },
        {
            'seqno': 1,
            'optype': 'Add',
            'input_tensors': ['1x64x256'],
            'output_tensors': ['1x64x256'],
        },
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'Add',
            'input_tensors': ['1x64x256'],
            'output_tensors': ['1x64x256'],
        },
        {
            'seqno': 1,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        },
    ]

    # Algorithm searches forward in profiler for polaris ops
    # Polaris MatMul finds profiler MatMul at index 1, skipping profiler Add
    # Polaris Add is then not found in remaining profiler entries (exhausted)
    stats = compare_layers(polaris_layers, profiler_layers)
    assert stats.total_matches == 1  # Only MatMul matches
    assert stats.unmatched_polaris == 1  # Polaris Add not matched
    assert stats.unmatched_profiler == 1  # Profiler Add skipped


@pytest.mark.unit
def test_compare_layers_unmatched_polaris():
    """Test when Polaris has extra operations."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        },
        {
            'seqno': 1,
            'optype': 'Softmax',
            'input_tensors': ['1x64x256'],
            'output_tensors': ['1x64x256'],
        },
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        }
    ]

    stats = compare_layers(polaris_layers, profiler_layers)
    assert stats.total_matches == 1
    assert stats.unmatched_polaris == 1
    assert stats.unmatched_profiler == 0


@pytest.mark.unit
def test_compare_layers_unmatched_profiler():
    """Test when profiler has extra operations."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        },
        {
            'seqno': 1,
            'optype': 'Add',
            'input_tensors': ['1x64x256'],
            'output_tensors': ['1x64x256'],
        },
    ]

    stats = compare_layers(polaris_layers, profiler_layers)
    assert stats.total_matches == 1
    assert stats.unmatched_polaris == 0
    assert stats.unmatched_profiler == 1


@pytest.mark.unit
def test_compare_layers_binary_special_case():
    """Test binary operation special handling (scalar operands)."""
    # Binary ops may have scalar operands that don't appear in profiler
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'Add',
            'input_tensors': ['1x64x256', '1'],
            'output_tensors': ['1x64x256'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'Add',
            'input_tensors': ['1x64x256'],  # Scalar operand not tracked
            'output_tensors': ['1x64x256'],
        }
    ]

    stats = compare_layers(polaris_layers, profiler_layers)
    # Should still match due to binary op special handling
    assert stats.total_matches == 1


@pytest.mark.unit
def test_compare_layers_reshape_special_case():
    """Test reshape operation with strip_leading_ones."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'Reshape',
            'input_tensors': ['1x64x256'],
            'output_tensors': ['16384'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'Reshape',
            'input_tensors': ['64x256'],
            'output_tensors': ['16384'],
        }
    ]

    # With strip_leading_ones, the input shapes become equivalent
    stats = compare_layers(polaris_layers, profiler_layers, strip_leading_ones=True)
    assert stats.total_matches == 1


@pytest.mark.unit
def test_compare_layers_strip_leading_ones():
    """Test strip_leading_ones option."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x1x64x128'],
            'output_tensors': ['1x1x64x256'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['64x128'],
            'output_tensors': ['64x256'],
        }
    ]

    # Without strip_leading_ones, shapes don't match
    stats = compare_layers(polaris_layers, profiler_layers, strip_leading_ones=False)
    assert stats.total_matches == 0

    # With strip_leading_ones, shapes should match
    stats = compare_layers(polaris_layers, profiler_layers, strip_leading_ones=True)
    assert stats.total_matches == 1


@pytest.mark.unit
def test_compare_layers_empty_lists():
    """Test comparison with empty lists."""
    stats = compare_layers([], [])
    assert stats.total_matches == 0
    assert stats.unmatched_polaris == 0
    assert stats.unmatched_profiler == 0
