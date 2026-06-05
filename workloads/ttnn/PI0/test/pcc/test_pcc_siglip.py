# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Shape Correctness Test: SigLIP Vision Tower - TTSim
Tests that the ttsim SigLIP vision tower produces outputs with correct shapes.
NOTE: ttsim is a shape-tracking performance simulator — it does NOT
compute real numerical values. Tests assert output shapes only.
Usage:
    pytest test_ttsim_siglip.py -v
    python test_ttsim_siglip.py
"""
import sys
from pathlib import Path
from typing import List
import pytest
import ttsim.front.ttnn as ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from workloads.ttnn.PI0.tt.ttnn_siglip import SigLIPVisionTowerTTNN
from workloads.ttnn.PI0.common.configs import SigLIPConfig


def safe_shape_list(shape) -> List[int]:
    """Convert shape to list, raising if None."""
    if shape is None:
        raise ValueError("Shape cannot be None")
    return list(shape)


def create_siglip_config() -> SigLIPConfig:
    """Create SigLIP config matching checkpoint."""
    return SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )


def create_small_siglip_config() -> SigLIPConfig:
    """Create smaller SigLIP config for fast testing."""
    return SigLIPConfig(
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=4,
        num_attention_heads=6,
        image_size=224,
        patch_size=14,
    )


def make_input(shape: list, device) -> ttnn.Tensor:
    """Create a ttsim input tensor of given shape."""
    return ttnn.zeros(
        shape,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def create_vision_tower_weights(config: SigLIPConfig, device) -> dict:
    """Create full vision tower weight tensors."""
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    num_patches = (config.image_size // config.patch_size) ** 2

    weights = {
        "patch_embedding.weight": make_input([hidden, 3, config.patch_size, config.patch_size], device),
        "patch_embedding.bias": make_input([1, hidden], device),
        "position_embedding.weight": make_input([1, num_patches, hidden], device),
        "post_layernorm.weight": make_input([1, hidden], device),
        "post_layernorm.bias": make_input([1, hidden], device),
    }

    for i in range(config.num_hidden_layers):
        prefix = f"encoder.layers.{i}."
        weights[f"{prefix}layer_norm1.weight"] = make_input([1, hidden], device)
        weights[f"{prefix}layer_norm1.bias"] = make_input([1, hidden], device)
        weights[f"{prefix}layer_norm2.weight"] = make_input([1, hidden], device)
        weights[f"{prefix}layer_norm2.bias"] = make_input([1, hidden], device)
        weights[f"{prefix}self_attn.q_proj.weight"] = make_input([hidden, hidden], device)
        weights[f"{prefix}self_attn.q_proj.bias"] = make_input([1, hidden], device)
        weights[f"{prefix}self_attn.k_proj.weight"] = make_input([hidden, hidden], device)
        weights[f"{prefix}self_attn.k_proj.bias"] = make_input([1, hidden], device)
        weights[f"{prefix}self_attn.v_proj.weight"] = make_input([hidden, hidden], device)
        weights[f"{prefix}self_attn.v_proj.bias"] = make_input([1, hidden], device)
        weights[f"{prefix}self_attn.out_proj.weight"] = make_input([hidden, hidden], device)
        weights[f"{prefix}self_attn.out_proj.bias"] = make_input([1, hidden], device)
        weights[f"{prefix}mlp.fc1.weight"] = make_input([hidden, intermediate], device)
        weights[f"{prefix}mlp.fc1.bias"] = make_input([1, intermediate], device)
        weights[f"{prefix}mlp.fc2.weight"] = make_input([intermediate, hidden], device)
        weights[f"{prefix}mlp.fc2.bias"] = make_input([1, hidden], device)

    return weights


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


def test_pcc_siglip_vision_tower(device):
    """Test SigLIP Vision Tower shape: (batch, 3, 224, 224) -> (batch, num_patches, hidden)"""
    config = create_small_siglip_config()
    weights = create_vision_tower_weights(config, device)
    model = SigLIPVisionTowerTTNN(config, weights, device)

    pixel_values = make_input([1, 3, config.image_size, config.image_size], device)
    out = model.forward(pixel_values)

    num_patches = (config.image_size // config.patch_size) ** 2
    assert_shape(out, [1, num_patches, config.hidden_size], "siglip_vision_tower")


def main():
    """Standalone shape smoke test without pytest."""
    print("=" * 70)
    print("  SigLIP Vision Tower Shape Test (TTSim)")
    print("=" * 70)

    config = create_small_siglip_config()
    device = ttnn.open_device(device_id=0)

    try:
        num_patches = (config.image_size // config.patch_size) ** 2

        print("\n1. Creating weights and model...")
        weights = create_vision_tower_weights(config, device)
        model = SigLIPVisionTowerTTNN(config, weights, device)

        print("2. Testing vision tower forward...")
        pixel_values = make_input([1, 3, config.image_size, config.image_size], device)
        out = model.forward(pixel_values)

        expected = [1, num_patches, config.hidden_size]
        actual = safe_shape_list(out.shape)
        passed = actual == expected

        print("\n" + "=" * 70)
        print("  RESULTS")
        print("=" * 70)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  siglip_vision_tower  got={actual}  expected={expected}")
        print("=" * 70)
        return 0 if passed else 1

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())