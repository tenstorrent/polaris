#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools.profiling.shape_canonical — the single source of truth
for tensor shape parsing, normalization, and attribute normalization."""

import pytest
from tools.profiling.shape_canonical import (
    DTYPE_NORMALIZATION,
    LAYOUT_NORMALIZATION,
    compare_tensor_attributes,
    compare_tensor_shapes,
    coerce_shape_to_list,
    normalize_attr,
    normalize_dtype,
    normalize_layout,
    normalize_memory_tag,
    normalize_shape,
    parse_shape_string,
    promote_to_rank4,
    reshape_input0_wzyx,
    validate_binary_compatibility,
    validate_reshape_compatibility,
)


# ── parse_shape_string ───────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("1x224x224", [1, 224, 224]),
        ("8x197x768", [8, 197, 768]),
        ("1x1x1024x768", [1, 1, 1024, 768]),
        ("768", [768]),
        ("", []),
        ("  ", []),
    ],
)
def test_parse_shape_string(input_str, expected):
    assert parse_shape_string(input_str) == expected


@pytest.mark.unit
def test_parse_shape_string_invalid():
    assert parse_shape_string("axb") == []


# ── normalize_shape ──────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "shape,strip_leading,strip_singleton,expected",
    [
        ([1, 1, 1, 224, 224], False, False, [1, 224, 224]),
        ([1, 1, 1024, 768], True, False, [1024, 768]),
        ([8, 1, 197, 768], False, True, [8, 197, 768]),
        ([1, 8, 197, 768], False, True, [8, 197, 768]),
        ([1, 1, 1], False, False, [1]),
        ([1, 1, 1], False, True, [1]),
        ([], False, False, []),
        ([768], False, False, [768]),
        ([1, 768], False, False, [1, 768]),
    ],
)
def test_normalize_shape(shape, strip_leading, strip_singleton, expected):
    assert normalize_shape(shape, strip_leading, strip_singleton) == expected


# ── promote_to_rank4 ────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "dims,expected",
    [
        ([768], (1, 1, 1, 768)),
        ([197, 768], (1, 1, 197, 768)),
        ([8, 197, 768], (1, 8, 197, 768)),
        ([1, 8, 197, 768], (1, 8, 197, 768)),
    ],
)
def test_promote_to_rank4(dims, expected):
    assert promote_to_rank4(dims) == expected


# ── reshape_input0_wzyx ─────────────────────────────────────────────────


@pytest.mark.unit
def test_reshape_input0_wzyx():
    assert reshape_input0_wzyx(1, 1, 197, 768) == (1, 1, 197, 768)
    assert reshape_input0_wzyx(1, 8, 197, 768) == (1, 1, 1576, 768)


# ── coerce_shape_to_list ────────────────────────────────────────────────


@pytest.mark.unit
def test_coerce_shape_to_list_from_tuple():
    assert coerce_shape_to_list((1, 2, 3)) == [1, 2, 3]


@pytest.mark.unit
def test_coerce_shape_to_list_none():
    assert coerce_shape_to_list(None) == []


# ── normalize_layout / normalize_dtype ───────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("tile_layout", "TILE"),
        ("TILE", "TILE"),
        ("row_major_layout", "ROW_MAJOR"),
        ("row_major", "ROW_MAJOR"),
        (None, None),
    ],
)
def test_normalize_layout(raw, expected):
    assert normalize_layout(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("float16", "BFLOAT16"),
        ("bfloat16", "BFLOAT16"),
        ("float32", "FLOAT32"),
        ("bfloat8_b", "BFLOAT8_B"),
        (None, None),
    ],
)
def test_normalize_dtype(raw, expected):
    assert normalize_dtype(raw) == expected


# ── compare_tensor_shapes ───────────────────────────────────────────────


@pytest.mark.unit
def test_compare_tensor_shapes_match():
    match, details = compare_tensor_shapes(
        ["1x224x224"], ["1x224x224"],
    )
    assert match is True
    assert details == ""


@pytest.mark.unit
def test_compare_tensor_shapes_mismatch():
    match, details = compare_tensor_shapes(
        ["1x224x224"], ["1x112x112"],
    )
    assert match is False
    assert "tensor 0" in details


@pytest.mark.unit
def test_compare_tensor_shapes_leading_ones():
    match, _ = compare_tensor_shapes(
        ["1x1x224x224"], ["1x224x224"],
    )
    assert match is True


@pytest.mark.unit
def test_compare_tensor_shapes_count_mismatch():
    match, details = compare_tensor_shapes(
        ["1x224x224", "1x224x224"], ["1x224x224"],
    )
    assert match is False
    assert "count mismatch" in details


# ── validate_binary_compatibility ────────────────────────────────────────


@pytest.mark.unit
def test_validate_binary_overlap_prefix():
    match, details = validate_binary_compatibility(
        ["1x224x224", "1x224x224"],
        ["1x224x224"],
    )
    assert match is True
    assert "binary compatible" in details


@pytest.mark.unit
def test_validate_binary_all_scalar():
    match, _ = validate_binary_compatibility([], ["1x224x224"])
    assert match is True


# ── validate_reshape_compatibility ───────────────────────────────────────


@pytest.mark.unit
def test_validate_reshape_not_2_inputs():
    match, details = validate_reshape_compatibility(
        ["1x224x224"],
        ["1x224x224"],
        ["1x224x224"],
    )
    assert match is False
    assert "polaris doesn't have 2 inputs" in details


# ── normalize_memory_tag ────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "polaris_repr,profiler_short",
    [
        (
            "MemoryConfig(memory_layout=<TensorMemoryLayout.BLOCK_SHARDED: 4>, "
            "buffer_type=<BufferType.L1: 2>)",
            "L1_BLOCK_SHARDED",
        ),
        (
            "MemoryConfig(memory_layout=<TensorMemoryLayout.INTERLEAVED: 1>, "
            "buffer_type=<BufferType.DRAM: 1>)",
            "DRAM_INTERLEAVED",
        ),
        ("DEV_1_L1_BLOCK_SHARDED", "L1_BLOCK_SHARDED"),
        ("l1_block_sharded", "L1_BLOCK_SHARDED"),
    ],
)
def test_normalize_memory_tag_polaris_repr_matches_profiler(polaris_repr, profiler_short):
    assert normalize_memory_tag(polaris_repr) == normalize_memory_tag(profiler_short)


@pytest.mark.unit
def test_normalize_memory_tag_semantic_mismatch_still_differs():
    """P3-style: different buffer/layout must not normalize to equality."""
    dram = normalize_memory_tag(
        "MemoryConfig(memory_layout=<TensorMemoryLayout.INTERLEAVED: 1>, "
        "buffer_type=<BufferType.DRAM: 1>)"
    )
    l1_block = normalize_memory_tag(
        "MemoryConfig(memory_layout=<TensorMemoryLayout.BLOCK_SHARDED: 4>, "
        "buffer_type=<BufferType.L1: 2>)"
    )
    assert dram != l1_block


@pytest.mark.unit
def test_memory_config_str_matches_canonical_tag():
    """New CSV exports use ``str(mc)`` → short profiler-style tag."""
    from ttsim.front.ttnn.buffer import BufferType, TensorMemoryLayout
    from ttsim.front.ttnn.memory import MemoryConfig

    mc = MemoryConfig(TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1)
    assert str(mc) == "L1_BLOCK_SHARDED"
    assert normalize_memory_tag(str(mc)) == normalize_memory_tag(repr(mc))


# ── compare_tensor_attributes ───────────────────────────────────────────


@pytest.mark.unit
def test_compare_tensor_attributes_match():
    p = {"input_dtypes": ["bfloat16"], "input_layouts": ["tile"], "input_memories": ["DRAM"]}
    f = {"input_dtypes": ["BFLOAT16"], "input_layouts": ["TILE"], "input_memories": ["DRAM"]}
    match, details = compare_tensor_attributes(p, f, "input")
    assert match is True


@pytest.mark.unit
def test_compare_tensor_attributes_memory_repr_vs_profiler_short():
    p = {
        "input_dtypes": ["bfloat16"],
        "input_layouts": ["tile"],
        "input_memories": [
            "MemoryConfig(memory_layout=<TensorMemoryLayout.BLOCK_SHARDED: 4>, "
            "buffer_type=<BufferType.L1: 2>)"
        ],
    }
    f = {
        "input_dtypes": ["BFLOAT16"],
        "input_layouts": ["TILE"],
        "input_memories": ["L1_BLOCK_SHARDED"],
    }
    match, details = compare_tensor_attributes(p, f, "input")
    assert match is True
    assert details == ""


@pytest.mark.unit
def test_compare_tensor_attributes_dtype_mismatch():
    p = {"input_dtypes": ["float32"], "input_layouts": [], "input_memories": []}
    f = {"input_dtypes": ["bfloat16"], "input_layouts": [], "input_memories": []}
    match, details = compare_tensor_attributes(p, f, "input")
    assert match is False
    assert "dtype" in details


@pytest.mark.unit
def test_compare_tensor_attributes_none_skipped():
    p = {"input_dtypes": [None], "input_layouts": [], "input_memories": []}
    f = {"input_dtypes": ["bfloat16"], "input_layouts": [], "input_memories": []}
    match, _ = compare_tensor_attributes(p, f, "input")
    assert match is True
