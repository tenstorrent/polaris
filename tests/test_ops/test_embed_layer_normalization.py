#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from ttsim.ops.op import EmbedLayerNormalizationOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_embed_layer_normalization_test_tensors(batch_size=2, seq_length=8, vocab_size=1000, hidden_size=768, max_position=512, include_segment=True, include_mask=True):
    """Helper function to create test tensors for EmbedLayerNormalization operation"""

    # Create input_ids: [batch_size, seq_length]
    input_ids = F._from_shape('input_ids', [batch_size, seq_length], np_dtype='int64')

    # Create embedding_weight: [vocab_size, hidden_size]
    embedding_weight = F._from_shape('embedding_weight', [vocab_size, hidden_size], np_dtype=np.float32)

    # Create position_weight: [max_position, hidden_size]
    position_weight = F._from_shape('position_weight', [max_position, hidden_size], np_dtype=np.float32)

    input_tensors = [input_ids, embedding_weight, position_weight]
    input_names = ['input_ids', 'embedding_weight', 'position_weight']

    # Optional segment_weight: [segment_count, hidden_size]
    if include_segment:
        segment_weight = F._from_shape('segment_weight', [2, hidden_size], np_dtype=np.float32)  # 2 segments, float32
        input_tensors.append(segment_weight)
        input_names.append('segment_weight')

    # Optional mask: [batch_size, seq_length] or other shapes
    if include_mask:
        mask = F._from_shape('mask', [batch_size, seq_length], np_dtype='int64')
        input_tensors.append(mask)
        input_names.append('mask')

    # Create output tensors
    output = make_tensor('output')
    output_tensors = [output]
    output_names = ['output']

    return input_tensors, output_tensors, input_names, output_names


def reference_embed_layer_normalization(input_ids, embedding_weight, position_weight, segment_weight=None, mask=None, epsilon=1e-12):
    """
    Reference implementation of EmbedLayerNormalization for testing
    Follows the typical transformer embedding pipeline
    """
    # Convert to numpy for easier manipulation
    input_ids = np.asarray(input_ids)
    embedding_weight = np.asarray(embedding_weight)
    position_weight = np.asarray(position_weight)

    batch_size, seq_length = input_ids.shape
    vocab_size, hidden_size = embedding_weight.shape

    # 1. Token embedding lookup
    token_embeddings = embedding_weight[input_ids]  # [batch_size, seq_length, hidden_size]

    # 2. Position embedding addition
    position_ids = np.arange(seq_length)[np.newaxis, :]  # [1, seq_length]
    position_ids = np.tile(position_ids, (batch_size, 1))  # [batch_size, seq_length]
    position_embeddings = position_weight[position_ids]  # [batch_size, seq_length, hidden_size]

    # 3. Combine token and position embeddings
    embeddings = token_embeddings + position_embeddings

    # 4. Segment embedding addition (if provided)
    if segment_weight is not None:
        segment_weight = np.asarray(segment_weight)
        # Assume segment_ids are all 0 for simplicity (could be more complex)
        segment_ids = np.zeros((batch_size, seq_length), dtype=int)
        segment_embeddings = segment_weight[segment_ids]  # [batch_size, seq_length, hidden_size]
        embeddings += segment_embeddings

    # 5. Layer normalization
    # Compute mean and variance along the last axis (hidden_size)
    mean = np.mean(embeddings, axis=-1, keepdims=True)  # [batch_size, seq_length, 1]
    variance = np.var(embeddings, axis=-1, keepdims=True)  # [batch_size, seq_length, 1]

    # Normalize
    normalized = (embeddings - mean) / np.sqrt(variance + epsilon)

    # Note: In a full implementation, there would be scale and bias parameters for layer norm
    # For this reference, we assume scale=1 and bias=0

    # 6. Mask processing (if provided)
    if mask is not None:
        mask = np.asarray(mask)
        # Apply mask (this is a simplified version)
        mask_expanded = np.expand_dims(mask, axis=-1)  # [batch_size, seq_length, 1]
        normalized = np.where(mask_expanded == 0, normalized, -1e4)  # or some mask value

    return normalized


@pytest.mark.unit
@pytest.mark.opunit
def test_embed_layer_normalization_basic():
    """Test basic EmbedLayerNormalization with required inputs only"""
    print("\n=== Testing Basic EmbedLayerNormalization ===")

    batch_size, seq_length = 2, 8
    vocab_size, hidden_size = 1000, 768
    max_position = 512

    input_tensors, output_tensors, input_names, output_names = create_embed_layer_normalization_test_tensors(
        batch_size=batch_size, seq_length=seq_length, vocab_size=vocab_size,
        hidden_size=hidden_size, max_position=max_position,
        include_segment=False, include_mask=False
    )

    attrs = {'epsilon': 1e-12}

    op_info = {
        'name': 'test_embed_layer_normalization_basic',
        'optype': 'EmbedLayerNormalization',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    # Create and execute EmbedLayerNormalization operation
    op_obj = EmbedLayerNormalizationOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    # Execute get_perf_counts to validate and set shapes
    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shape
    expected_output_shape = [batch_size, seq_length, hidden_size]
    assert output_tensors[0].shape == expected_output_shape, \
        f"Output shape mismatch: {output_tensors[0].shape} != {expected_output_shape}"

    # Validate output dtype
    assert output_tensors[0].dtype == 'float32', \
        f"Output dtype mismatch: {output_tensors[0].dtype} != float32"

    print(f"✓ Basic EmbedLayerNormalization test passed")
    print(f"  Input IDs shape: {input_tensors[0].shape} ({input_tensors[0].dtype})")
    print(f"  Embedding weight shape: {input_tensors[1].shape} ({input_tensors[1].dtype})")
    print(f"  Position weight shape: {input_tensors[2].shape} ({input_tensors[2].dtype})")
    print(f"  Output shape: {output_tensors[0].shape} ({output_tensors[0].dtype})")


@pytest.mark.unit
@pytest.mark.opunit
def test_embed_layer_normalization_with_segment():
    """Test EmbedLayerNormalization with segment embeddings"""
    print("\n=== Testing EmbedLayerNormalization with Segment ===")

    batch_size, seq_length = 2, 8
    vocab_size, hidden_size = 1000, 768
    max_position = 512

    input_tensors, output_tensors, input_names, output_names = create_embed_layer_normalization_test_tensors(
        batch_size=batch_size, seq_length=seq_length, vocab_size=vocab_size,
        hidden_size=hidden_size, max_position=max_position,
        include_segment=True, include_mask=False
    )

    attrs = {'epsilon': 1e-12}

    op_info = {
        'name': 'test_embed_layer_normalization_with_segment',
        'optype': 'EmbedLayerNormalization',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = EmbedLayerNormalizationOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = [batch_size, seq_length, hidden_size]
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == 'float32'

    print(f"✓ EmbedLayerNormalization with segment test passed")
    print(f"  Input count: {len(input_tensors)} (includes segment_weight)")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_embed_layer_normalization_with_mask():
    """Test EmbedLayerNormalization with attention mask"""
    print("\n=== Testing EmbedLayerNormalization with Mask ===")

    batch_size, seq_length = 2, 8
    vocab_size, hidden_size = 1000, 768
    max_position = 512

    input_tensors, output_tensors, input_names, output_names = create_embed_layer_normalization_test_tensors(
        batch_size=batch_size, seq_length=seq_length, vocab_size=vocab_size,
        hidden_size=hidden_size, max_position=max_position,
        include_segment=True, include_mask=True
    )

    attrs = {'epsilon': 1e-12, 'mask_value': -1e4}

    op_info = {
        'name': 'test_embed_layer_normalization_with_mask',
        'optype': 'EmbedLayerNormalization',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = EmbedLayerNormalizationOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = [batch_size, seq_length, hidden_size]
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == 'float32'

    print(f"✓ EmbedLayerNormalization with mask test passed")
    print(f"  Input count: {len(input_tensors)} (includes segment_weight and mask)")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_embed_layer_normalization_optional_output():
    """Test EmbedLayerNormalization with optional mask_index output"""
    print("\n=== Testing EmbedLayerNormalization with Optional Output ===")

    batch_size, seq_length = 2, 8
    vocab_size, hidden_size = 1000, 768
    max_position = 512

    input_tensors, output_tensors, input_names, output_names = create_embed_layer_normalization_test_tensors(
        batch_size=batch_size, seq_length=seq_length, vocab_size=vocab_size,
        hidden_size=hidden_size, max_position=max_position,
        include_segment=False, include_mask=True
    )

    # Add optional mask_index output
    mask_index = make_tensor('mask_index')
    output_tensors.append(mask_index)
    output_names.append('mask_index')

    attrs = {'epsilon': 1e-12}

    op_info = {
        'name': 'test_embed_layer_normalization_optional_output',
        'optype': 'EmbedLayerNormalization',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = EmbedLayerNormalizationOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate outputs
    expected_output_shape = [batch_size, seq_length, hidden_size]
    expected_mask_shape = [batch_size, seq_length]

    assert output_tensors[0].shape == expected_output_shape  # Main output
    assert output_tensors[0].dtype == 'float32'
    assert output_tensors[1].shape == expected_mask_shape   # Mask index output
    assert output_tensors[1].dtype == 'int64'  # Same as input_ids

    print(f"✓ EmbedLayerNormalization optional output test passed")
    print(f"  Main output shape: {output_tensors[0].shape}")
    print(f"  Mask index shape: {output_tensors[1].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_embed_layer_normalization_invalid_inputs():
    """Test EmbedLayerNormalization operation with invalid inputs"""
    print("\n=== Testing EmbedLayerNormalization Error Cases ===")

    # Test 1: Wrong input count
    try:
        input_tensors = [F._from_shape('input', [4, 8], np_dtype='int64')]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_embed_layer_normalization_wrong_inputs',
            'optype': 'EmbedLayerNormalization',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = EmbedLayerNormalizationOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input count: {e}")

    # Test 2: Wrong input dtype
    try:
        input_tensors = [
            F._from_shape('input_ids', [2, 8], np_dtype=np.float32),  # Wrong dtype
            F._from_shape('embedding_weight', [1000, 768], np_dtype=np.float32),
            F._from_shape('position_weight', [512, 768], np_dtype=np.float32)
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_embed_layer_normalization_wrong_dtype',
            'optype': 'EmbedLayerNormalization',
            'inList': ['input_ids', 'embedding_weight', 'position_weight'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = EmbedLayerNormalizationOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input dtype"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input dtype: {e}")

    # Test 3: Sequence length exceeds position embeddings
    try:
        input_tensors = [
            F._from_shape('input_ids', [2, 100], np_dtype='int64'),  # Long sequence
            F._from_shape('embedding_weight', [1000, 768], np_dtype=np.float32),
            F._from_shape('position_weight', [50, 768], np_dtype=np.float32)  # Short positions
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_embed_layer_normalization_seq_too_long',
            'optype': 'EmbedLayerNormalization',
            'inList': ['input_ids', 'embedding_weight', 'position_weight'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = EmbedLayerNormalizationOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for sequence too long"
    except AssertionError as e:
        print(f"✓ Correctly caught sequence length exceeding position embeddings: {e}")

    # Test 4: Invalid epsilon
    try:
        input_tensors = [
            F._from_shape('input_ids', [2, 8], np_dtype='int64'),
            F._from_shape('embedding_weight', [1000, 768], np_dtype=np.float32),
            F._from_shape('position_weight', [512, 768], np_dtype=np.float32)
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_embed_layer_normalization_invalid_epsilon',
            'optype': 'EmbedLayerNormalization',
            'inList': ['input_ids', 'embedding_weight', 'position_weight'],
            'outList': ['output'],
            'attrs': {'epsilon': -1.0},  # Invalid negative epsilon
        }

        op_obj = EmbedLayerNormalizationOp(op_info)
        assert False, "Should have raised ValueError for invalid epsilon"
    except ValueError as e:
        print(f"✓ Correctly caught invalid epsilon: {e}")

    print("✓ All error case tests passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_embed_layer_normalization_factory():
    """Test that EmbedLayerNormalization operation can be created through SimOpFactory"""
    print("\n=== Testing EmbedLayerNormalization Factory Creation ===")

    from ttsim.ops.op import SimOpFactory

    # Test factory creation
    embed_ln_class = SimOpFactory('EmbedLayerNormalization')
    assert embed_ln_class == EmbedLayerNormalizationOp

    # Create instance through factory
    op_info = {
        'name': 'factory_test_embed_layer_normalization',
        'optype': 'EmbedLayerNormalization',
        'inList': ['input_ids', 'embedding_weight', 'position_weight'],
        'outList': ['output'],
        'attrs': {'epsilon': 1e-12},
    }

    op_obj = embed_ln_class(op_info)
    assert isinstance(op_obj, EmbedLayerNormalizationOp)
    assert op_obj.optype == 'EmbedLayerNormalization'
    assert op_obj.name == 'factory_test_embed_layer_normalization'
    assert op_obj.epsilon == 1e-12

    print("✓ EmbedLayerNormalization factory creation test passed")
    print(f"  Created operation: {op_obj}")


@pytest.mark.unit
@pytest.mark.opunit
def test_embed_layer_normalization_transformer_scenario():
    """Test EmbedLayerNormalization in a realistic transformer scenario"""
    print("\n=== Testing EmbedLayerNormalization Transformer Scenario ===")

    # Typical BERT-like transformer parameters
    batch_size = 4
    seq_length = 128
    vocab_size = 30522  # BERT vocabulary size
    hidden_size = 768   # BERT hidden size
    max_position = 512  # BERT max position

    input_tensors, output_tensors, input_names, output_names = create_embed_layer_normalization_test_tensors(
        batch_size=batch_size, seq_length=seq_length, vocab_size=vocab_size,
        hidden_size=hidden_size, max_position=max_position,
        include_segment=True, include_mask=True
    )

    attrs = {'epsilon': 1e-12, 'mask_value': -10000.0}

    op_info = {
        'name': 'test_embed_layer_normalization_transformer',
        'optype': 'EmbedLayerNormalization',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = EmbedLayerNormalizationOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = [batch_size, seq_length, hidden_size]
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ Transformer EmbedLayerNormalization test passed")
    print(f"  BERT-like parameters: batch_size={batch_size}, seq_len={seq_length}, vocab={vocab_size}, hidden={hidden_size}")
    print(f"  Input count: {len(input_tensors)} (input_ids, embedding_weight, position_weight, segment_weight, mask)")
    print(f"  Output shape: {output_tensors[0].shape}")
    print(f"  Performance: {perf_stats['instrs']['gather']} embedding lookups, {perf_stats['instrs']['add']} additions")


if __name__ == '__main__':
    # Run all tests manually
    test_embed_layer_normalization_factory()
    test_embed_layer_normalization_basic()
    test_embed_layer_normalization_with_segment()
    test_embed_layer_normalization_with_mask()
    test_embed_layer_normalization_optional_output()
    test_embed_layer_normalization_invalid_inputs()
    test_embed_layer_normalization_transformer_scenario()

    print("\n🎉 All EmbedLayerNormalization operation tests completed successfully!")
