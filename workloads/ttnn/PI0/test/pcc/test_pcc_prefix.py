# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Shape Correctness Test: Prefix Embedding - TTSim
Tests that the ttsim PrefixEmbeddingTTNN module produces outputs with correct shapes.
NOTE: ttsim is a shape-tracking performance simulator — it does NOT
compute real numerical values. Tests assert output shapes only.
Usage:
    pytest test_ttsim_prefix.py -v
    python test_ttsim_prefix.py
"""
import sys
from pathlib import Path
from typing import List
import pytest
import ttsim.front.ttnn as ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from workloads.ttnn.PI0.tt.ttnn_prefix import PrefixEmbeddingTTNN
from workloads.ttnn.PI0.common.configs import PrefixConfig

SEQ_LEN = 32


def safe_shape_list(shape) -> List[int]:
    """Convert shape to list, raising if None."""
    if shape is None:
        raise ValueError("Shape cannot be None")
    return list(shape)


def create_prefix_config() -> PrefixConfig:
    """Create PrefixConfig — matches typical PI0 dimensions."""
    return PrefixConfig(
        vlm_hidden_size=2048,
        num_image_tokens=256,
    )


def make_input(shape: list, device) -> ttnn.Tensor:
    """Create a ttsim input tensor of given shape."""
    return ttnn.zeros(
        shape,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def create_mock_embed_language_fn(config: PrefixConfig, device):
    """
    Create a mock language embedding function.
    Returns tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def embed_language(tokens: ttnn.Tensor) -> ttnn.Tensor:
        shape = tokens.shape
        if shape is None:
            raise ValueError("tokens must have a valid shape")
        batch_size = shape[0]
        seq_len = shape[1]
        return ttnn.zeros(
            (batch_size, seq_len, config.vlm_hidden_size),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    return embed_language


def assert_shape(tensor: ttnn.Tensor, expected: list, label: str):
    """Assert ttsim tensor has the expected shape."""
    actual = safe_shape_list(tensor.shape)
    assert actual == expected, f"[{label}] Shape mismatch: got {actual}, expected {expected}"
    print(f"\n✅ [{label}] Shape OK: {actual}")


@pytest.fixture
def device():
    """Open a ttsim device and close it after each test."""
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_pcc_prefix_language_embedding(device):
    """Test prefix language embedding shape: (batch, seq_len) -> (batch, seq_len, hidden_dim)"""
    config = create_prefix_config()
    embed_lang_fn = create_mock_embed_language_fn(config, device)
    model = PrefixEmbeddingTTNN(config, device, embed_language_fn=embed_lang_fn)
    
    lang_tokens = make_input([1, SEQ_LEN], device)
    lang_masks = make_input([1, SEQ_LEN], device)
    out = model.embed_language(lang_tokens, lang_masks)
    
    assert_shape(out, [1, SEQ_LEN, config.vlm_hidden_size], "prefix_language_embedding")


def main():
    """Standalone shape smoke test without pytest."""
    print("=" * 70)
    print("  Prefix Embedding Shape Test (TTSim)")
    print("=" * 70)
    
    config = create_prefix_config()
    device = ttnn.open_device(device_id=0)
    
    try:
        print("\n1. Setting up mock embedding function...")
        embed_lang_fn = create_mock_embed_language_fn(config, device)
        model = PrefixEmbeddingTTNN(config, device, embed_language_fn=embed_lang_fn)
        
        # Test: embed_language
        print("2. Testing embed_language...")
        lang_tokens = make_input([1, SEQ_LEN], device)
        lang_masks = make_input([1, SEQ_LEN], device)
        out = model.embed_language(lang_tokens, lang_masks)
        
        expected = [1, SEQ_LEN, config.vlm_hidden_size]
        actual = safe_shape_list(out.shape)
        passed = actual == expected
        
        print("\n" + "=" * 70)
        print("  RESULTS")
        print("=" * 70)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  prefix_language_embedding  got={actual}  expected={expected}")
        print("=" * 70)
        return 0 if passed else 1
        
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())