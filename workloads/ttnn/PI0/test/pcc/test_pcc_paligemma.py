# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Shape Correctness Test: PaliGemma Backbone - TTSim
Tests that the ttsim PaliGemma backbone produces outputs with correct shapes.
NOTE: ttsim is a shape-tracking performance simulator — it does NOT
compute real numerical values. Tests assert output shapes only.
Usage:
    pytest test_ttsim_paligemma.py -v
    python test_ttsim_paligemma.py
"""
import sys
from pathlib import Path
from typing import List
import pytest
import ttsim.front.ttnn as ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from workloads.ttnn.PI0.tt.ttnn_paligemma import PaliGemmaBackboneTTNN
from workloads.ttnn.PI0.common.configs import SigLIPConfig, GemmaConfig, PaliGemmaConfig

SEQ_LEN = 64


def safe_shape_list(shape) -> List[int]:
    """Convert shape to list, raising if None."""
    if shape is None:
        raise ValueError("Shape cannot be None")
    return list(shape)


def create_small_config() -> PaliGemmaConfig:
    """Create small PaliGemma config for testing."""
    siglip = SigLIPConfig(
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=2,
        num_attention_heads=6,
        image_size=224,
        patch_size=14,
    )
    vlm = GemmaConfig(
        width=512,
        depth=2,
        mlp_dim=2048,
        num_heads=8,
        num_kv_heads=1,
        head_dim=64,
    )
    expert = GemmaConfig(
        width=256,
        depth=2,
        mlp_dim=1024,
        num_heads=4,
        num_kv_heads=1,
        head_dim=64,
    )
    return PaliGemmaConfig(
        siglip_config=siglip,
        vlm_config=vlm,
        expert_config=expert,
        max_seq_len=256,
    )


def make_input(shape: list, device) -> ttnn.Tensor:
    """Create a ttsim input tensor of given shape."""
    return ttnn.zeros(
        shape,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def create_siglip_weights(config: SigLIPConfig, device) -> dict:
    """Create SigLIP weight tensors."""
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


def create_gemma_weights(config: GemmaConfig, device) -> dict:
    """Create Gemma weight tensors."""
    width = config.width
    mlp_dim = config.mlp_dim
    q_dim = config.num_heads * config.head_dim
    kv_dim = config.num_kv_heads * config.head_dim
    
    weights = {
        "model.embed_tokens.weight": make_input([10000, width], device),
        "model.norm.weight": make_input([1, width], device),
    }
    
    for i in range(config.depth):
        prefix = f"model.layers.{i}."
        weights[f"{prefix}input_layernorm.weight"] = make_input([1, width], device)
        weights[f"{prefix}post_attention_layernorm.weight"] = make_input([1, width], device)
        weights[f"{prefix}self_attn.q_proj.weight"] = make_input([width, q_dim], device)
        weights[f"{prefix}self_attn.k_proj.weight"] = make_input([width, kv_dim], device)
        weights[f"{prefix}self_attn.v_proj.weight"] = make_input([width, kv_dim], device)
        weights[f"{prefix}self_attn.o_proj.weight"] = make_input([q_dim, width], device)
        weights[f"{prefix}mlp.gate_proj.weight"] = make_input([width, mlp_dim], device)
        weights[f"{prefix}mlp.up_proj.weight"] = make_input([width, mlp_dim], device)
        weights[f"{prefix}mlp.down_proj.weight"] = make_input([mlp_dim, width], device)
    
    return weights


def create_projector_weights(in_size: int, out_size: int, device) -> dict:
    """Create multi-modal projector weight tensors."""
    return {
        "linear.weight": make_input([out_size, in_size], device),
        "linear.bias": make_input([1, out_size], device),
    }


def create_paligemma_weights(config: PaliGemmaConfig, device) -> dict:
    """Create all PaliGemma weight tensors."""
    return {
        "vlm_vision": create_siglip_weights(config.siglip_config, device),
        "vlm_language": create_gemma_weights(config.vlm_config, device),
        "vlm_projector": create_projector_weights(
            config.siglip_config.hidden_size,
            config.vlm_config.width,
            device,
        ),
        "action_expert": create_gemma_weights(config.expert_config, device),
    }


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


def test_paligemma_embed_image_shape(device):
    """Test embed_image: (batch, 3, 224, 224) -> (batch, num_patches, vlm_width)"""
    config = create_small_config()
    weights = create_paligemma_weights(config, device)
    model = PaliGemmaBackboneTTNN(config, weights, device)
    
    pixel_values = make_input(
        [1, 3, config.siglip_config.image_size, config.siglip_config.image_size],
        device,
    )
    out = model.embed_image(pixel_values)
    
    num_patches = (config.siglip_config.image_size // config.siglip_config.patch_size) ** 2
    assert_shape(out, [1, num_patches, config.vlm_config.width], "embed_image")


def test_paligemma_forward_vlm_shape(device):
    """Test forward_vlm: (batch, seq_len, vlm_width) -> (batch, seq_len, vlm_width)"""
    config = create_small_config()
    weights = create_paligemma_weights(config, device)
    model = PaliGemmaBackboneTTNN(config, weights, device)
    
    hidden_states = make_input([1, SEQ_LEN, config.vlm_config.width], device)
    out, _ = model.forward_vlm(hidden_states)
    
    assert_shape(out, [1, SEQ_LEN, config.vlm_config.width], "forward_vlm")


def main():
    """Standalone shape smoke test without pytest."""
    print("=" * 70)
    print("  PaliGemma Backbone Shape Test (TTSim)")
    print("=" * 70)
    
    config = create_small_config()
    device = ttnn.open_device(device_id=0)
    
    try:
        results = []
        num_patches = (config.siglip_config.image_size // config.siglip_config.patch_size) ** 2
        
        print("\n1. Creating weights and model...")
        weights = create_paligemma_weights(config, device)
        model = PaliGemmaBackboneTTNN(config, weights, device)
        
        # Test 1: embed_image
        print("2. Testing embed_image...")
        pixel_values = make_input(
            [1, 3, config.siglip_config.image_size, config.siglip_config.image_size],
            device,
        )
        out = model.embed_image(pixel_values)
        expected = [1, num_patches, config.vlm_config.width]
        actual = safe_shape_list(out.shape)
        results.append(("embed_image", actual, expected, actual == expected))
        
        # Test 2: forward_vlm
        print("3. Testing forward_vlm...")
        hidden = make_input([1, SEQ_LEN, config.vlm_config.width], device)
        out, _ = model.forward_vlm(hidden)
        expected = [1, SEQ_LEN, config.vlm_config.width]
        actual = safe_shape_list(out.shape)
        results.append(("forward_vlm", actual, expected, actual == expected))
        
        # Results
        print("\n" + "=" * 70)
        print("  RESULTS")
        print("=" * 70)
        all_passed = all(ok for _, _, _, ok in results)
        for name, actual, expected, ok in results:
            status = "✅ PASS" if ok else "❌ FAIL"
            print(f"  {status}  {name:<15} got={actual}  expected={expected}")
        print("=" * 70)
        print(f"  Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        print("=" * 70)
        return 0 if all_passed else 1
        
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())