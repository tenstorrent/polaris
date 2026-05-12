#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Run all shape tests for PI0 model components using TTSim.
This is a TTSim port - shape tracking only, no numerical computation.
Usage:
    python run_all_pcc_tests.py
"""
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
import ttsim.front.ttnn as ttnn


def get_tensor_shape(tensor: Any) -> Tuple[int, ...]:
    """Extract shape from tensor."""
    if tensor is None:
        return ()
    if hasattr(tensor, 'shape'):
        shape = tensor.shape
        if shape is None:
            return ()
        if isinstance(shape, (list, tuple)):
            return tuple(shape)
        try:
            return tuple(shape)
        except Exception:
            return ()
    return ()


def make_input(
    shape: List[int], device: Any, dtype: Any = ttnn.bfloat16, layout: Any = ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    """Create a ttsim input tensor of given shape."""
    return ttnn.zeros(shape, device=device, dtype=dtype, layout=layout)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================
def test_suffix(device: Any) -> Tuple[bool, str]:
    """Test Suffix Embedding shapes."""
    from workloads.ttnn.PI0.tt.ttnn_suffix import SuffixEmbeddingTTNN
    from workloads.ttnn.PI0.common.configs import SuffixConfig

    config = SuffixConfig(
        action_dim=32,
        action_horizon=50,
        expert_width=256,
        state_dim=32,
        time_emb_dim=256,
        pi05=False,
    )

    weights: Dict[str, ttnn.Tensor] = {
        "action_in_proj.weight": make_input([config.action_dim, config.expert_width], device),
        "action_in_proj.bias": make_input([1, config.expert_width], device),
        "action_out_proj.weight": make_input([config.expert_width, config.action_dim], device),
        "action_out_proj.bias": make_input([1, config.action_dim], device),
        "state_proj.weight": make_input([config.state_dim, config.expert_width], device),
        "state_proj.bias": make_input([1, config.expert_width], device),
        "action_time_mlp_in.weight": make_input([config.expert_width * 2, config.time_emb_dim], device),
        "action_time_mlp_in.bias": make_input([1, config.time_emb_dim], device),
        "action_time_mlp_out.weight": make_input([config.time_emb_dim, config.expert_width], device),
        "action_time_mlp_out.bias": make_input([1, config.expert_width], device),
    }

    model = SuffixEmbeddingTTNN(config, weights, device)

    state = make_input([1, config.state_dim], device)
    noisy_actions = make_input([1, config.action_horizon, config.action_dim], device)
    timestep = make_input([1], device, layout=ttnn.ROW_MAJOR_LAYOUT)

    suffix_embs, _, _, _ = model.embed_suffix(state, noisy_actions, timestep)

    expected = (1, 1 + config.action_horizon, config.expert_width)
    actual = get_tensor_shape(suffix_embs)
    passed = actual == expected
    return passed, f"{'✅ PASS' if passed else '❌ FAIL'} - got {list(actual)}, expected {list(expected)}"


def test_prefix(device: Any) -> Tuple[bool, str]:
    """Test Prefix Language Embedding shapes."""
    from workloads.ttnn.PI0.tt.ttnn_prefix import PrefixEmbeddingTTNN
    from workloads.ttnn.PI0.common.configs import PrefixConfig, SigLIPConfig

    siglip_config = SigLIPConfig(image_size=224, patch_size=14)
    config = PrefixConfig(
        vlm_hidden_size=512,
        num_image_tokens=siglip_config.num_patches,
    )
    seq_len = 32
    num_patches = siglip_config.num_patches

    def embed_image_fn(img: ttnn.Tensor) -> ttnn.Tensor:
        shape = img.shape
        batch_size = shape[0] if shape else 1
        return make_input([batch_size, num_patches, config.vlm_hidden_size], device)

    def embed_language_fn(tokens: ttnn.Tensor) -> ttnn.Tensor:
        shape = tokens.shape
        batch_size = shape[0] if shape else 1
        seq = shape[1] if shape and len(shape) > 1 else seq_len
        return make_input([batch_size, seq, config.vlm_hidden_size], device)

    model = PrefixEmbeddingTTNN(
        config, device,
        embed_image_fn=embed_image_fn,
        embed_language_fn=embed_language_fn,
    )

    images = [make_input([1, 3, 224, 224], device)]
    img_masks = [make_input([1], device, layout=ttnn.ROW_MAJOR_LAYOUT)]
    lang_tokens = make_input([1, seq_len], device)
    lang_masks = make_input([1, seq_len], device)

    prefix_embs, _, _ = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    expected = (1, num_patches + seq_len, config.vlm_hidden_size)
    actual = get_tensor_shape(prefix_embs)
    passed = actual == expected
    return passed, f"{'✅ PASS' if passed else '❌ FAIL'} - got {list(actual)}, expected {list(expected)}"


def test_gemma_mlp(device: Any) -> Tuple[bool, str]:
    """Test Gemma MLP shapes."""
    from workloads.ttnn.PI0.tt.ttnn_gemma import GemmaMLPTTNN
    from workloads.ttnn.PI0.common.configs import GemmaConfig

    config = GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=128)
    seq_len = 32

    weights: Dict[str, ttnn.Tensor] = {
        "mlp.gate_proj.weight": make_input([config.width, config.mlp_dim], device),
        "mlp.up_proj.weight": make_input([config.width, config.mlp_dim], device),
        "mlp.down_proj.weight": make_input([config.mlp_dim, config.width], device),
    }

    model = GemmaMLPTTNN(config, weights, device)
    x = make_input([1, seq_len, config.width], device)
    out = model.forward(x)

    expected = (1, seq_len, config.width)
    actual = get_tensor_shape(out)
    passed = actual == expected
    return passed, f"{'✅ PASS' if passed else '❌ FAIL'} - got {list(actual)}, expected {list(expected)}"


def test_siglip(device: Any) -> Tuple[bool, str]:
    """Test SigLIP Vision Tower shapes."""
    from workloads.ttnn.PI0.tt.ttnn_siglip import SigLIPVisionTowerTTNN
    from workloads.ttnn.PI0.common.configs import SigLIPConfig

    config = SigLIPConfig(
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=4,
        num_attention_heads=6,
        image_size=224,
        patch_size=14,
    )
    num_patches = (config.image_size // config.patch_size) ** 2

    weights: Dict[str, ttnn.Tensor] = {
        "patch_embedding.weight": make_input([config.hidden_size, 3, config.patch_size, config.patch_size], device),
        "patch_embedding.bias": make_input([1, config.hidden_size], device),
        "position_embedding.weight": make_input([1, num_patches, config.hidden_size], device),
        "post_layernorm.weight": make_input([1, config.hidden_size], device),
        "post_layernorm.bias": make_input([1, config.hidden_size], device),
    }

    for i in range(config.num_hidden_layers):
        prefix = f"encoder.layers.{i}."
        weights[f"{prefix}layer_norm1.weight"] = make_input([1, config.hidden_size], device)
        weights[f"{prefix}layer_norm1.bias"] = make_input([1, config.hidden_size], device)
        weights[f"{prefix}layer_norm2.weight"] = make_input([1, config.hidden_size], device)
        weights[f"{prefix}layer_norm2.bias"] = make_input([1, config.hidden_size], device)
        weights[f"{prefix}self_attn.q_proj.weight"] = make_input([config.hidden_size, config.hidden_size], device)
        weights[f"{prefix}self_attn.q_proj.bias"] = make_input([1, config.hidden_size], device)
        weights[f"{prefix}self_attn.k_proj.weight"] = make_input([config.hidden_size, config.hidden_size], device)
        weights[f"{prefix}self_attn.k_proj.bias"] = make_input([1, config.hidden_size], device)
        weights[f"{prefix}self_attn.v_proj.weight"] = make_input([config.hidden_size, config.hidden_size], device)
        weights[f"{prefix}self_attn.v_proj.bias"] = make_input([1, config.hidden_size], device)
        weights[f"{prefix}self_attn.out_proj.weight"] = make_input([config.hidden_size, config.hidden_size], device)
        weights[f"{prefix}self_attn.out_proj.bias"] = make_input([1, config.hidden_size], device)
        weights[f"{prefix}mlp.fc1.weight"] = make_input([config.hidden_size, config.intermediate_size], device)
        weights[f"{prefix}mlp.fc1.bias"] = make_input([1, config.intermediate_size], device)
        weights[f"{prefix}mlp.fc2.weight"] = make_input([config.intermediate_size, config.hidden_size], device)
        weights[f"{prefix}mlp.fc2.bias"] = make_input([1, config.hidden_size], device)

    model = SigLIPVisionTowerTTNN(config, weights, device)
    x = make_input([1, 3, config.image_size, config.image_size], device)
    out = model.forward(x)

    expected = (1, num_patches, config.hidden_size)
    actual = get_tensor_shape(out)
    passed = actual == expected
    return passed, f"{'✅ PASS' if passed else '❌ FAIL'} - got {list(actual)}, expected {list(expected)}"


def test_paligemma(device: Any) -> Tuple[bool, str]:
    """Test PaliGemma Backbone shapes (image embedding)."""
    from workloads.ttnn.PI0.tt.ttnn_paligemma import PaliGemmaBackboneTTNN
    from workloads.ttnn.PI0.common.configs import PaliGemmaConfig, SigLIPConfig, GemmaConfig

    siglip = SigLIPConfig(
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=2,
        num_attention_heads=6,
        image_size=224,
        patch_size=14,
    )
    vlm = GemmaConfig(width=512, depth=2, mlp_dim=2048, num_heads=8, num_kv_heads=1, head_dim=64)
    expert = GemmaConfig(width=256, depth=2, mlp_dim=1024, num_heads=4, num_kv_heads=1, head_dim=64)
    config = PaliGemmaConfig(siglip_config=siglip, vlm_config=vlm, expert_config=expert, max_seq_len=256)

    num_patches = (siglip.image_size // siglip.patch_size) ** 2

    weights: Dict[str, Dict[str, ttnn.Tensor]] = {
        "vlm_vision": {},
        "vlm_language": {},
        "vlm_projector": {},
        "action_expert": {},
    }

    # SigLIP weights
    weights["vlm_vision"]["patch_embedding.weight"] = make_input(
        [siglip.hidden_size, 3, siglip.patch_size, siglip.patch_size], device
    )
    weights["vlm_vision"]["patch_embedding.bias"] = make_input([1, siglip.hidden_size], device)
    weights["vlm_vision"]["position_embedding.weight"] = make_input([1, num_patches, siglip.hidden_size], device)
    weights["vlm_vision"]["post_layernorm.weight"] = make_input([1, siglip.hidden_size], device)
    weights["vlm_vision"]["post_layernorm.bias"] = make_input([1, siglip.hidden_size], device)

    for i in range(siglip.num_hidden_layers):
        prefix = f"encoder.layers.{i}."
        weights["vlm_vision"][f"{prefix}layer_norm1.weight"] = make_input([1, siglip.hidden_size], device)
        weights["vlm_vision"][f"{prefix}layer_norm1.bias"] = make_input([1, siglip.hidden_size], device)
        weights["vlm_vision"][f"{prefix}layer_norm2.weight"] = make_input([1, siglip.hidden_size], device)
        weights["vlm_vision"][f"{prefix}layer_norm2.bias"] = make_input([1, siglip.hidden_size], device)
        weights["vlm_vision"][f"{prefix}self_attn.q_proj.weight"] = make_input(
            [siglip.hidden_size, siglip.hidden_size], device
        )
        weights["vlm_vision"][f"{prefix}self_attn.q_proj.bias"] = make_input([1, siglip.hidden_size], device)
        weights["vlm_vision"][f"{prefix}self_attn.k_proj.weight"] = make_input(
            [siglip.hidden_size, siglip.hidden_size], device
        )
        weights["vlm_vision"][f"{prefix}self_attn.k_proj.bias"] = make_input([1, siglip.hidden_size], device)
        weights["vlm_vision"][f"{prefix}self_attn.v_proj.weight"] = make_input(
            [siglip.hidden_size, siglip.hidden_size], device
        )
        weights["vlm_vision"][f"{prefix}self_attn.v_proj.bias"] = make_input([1, siglip.hidden_size], device)
        weights["vlm_vision"][f"{prefix}self_attn.out_proj.weight"] = make_input(
            [siglip.hidden_size, siglip.hidden_size], device
        )
        weights["vlm_vision"][f"{prefix}self_attn.out_proj.bias"] = make_input([1, siglip.hidden_size], device)
        weights["vlm_vision"][f"{prefix}mlp.fc1.weight"] = make_input(
            [siglip.hidden_size, siglip.intermediate_size], device
        )
        weights["vlm_vision"][f"{prefix}mlp.fc1.bias"] = make_input([1, siglip.intermediate_size], device)
        weights["vlm_vision"][f"{prefix}mlp.fc2.weight"] = make_input(
            [siglip.intermediate_size, siglip.hidden_size], device
        )
        weights["vlm_vision"][f"{prefix}mlp.fc2.bias"] = make_input([1, siglip.hidden_size], device)

    # Projector weights
    weights["vlm_projector"]["linear.weight"] = make_input([vlm.width, siglip.hidden_size], device)
    weights["vlm_projector"]["linear.bias"] = make_input([1, vlm.width], device)

    # VLM language weights (Gemma)
    weights["vlm_language"]["model.embed_tokens.weight"] = make_input([10000, vlm.width], device)
    weights["vlm_language"]["model.norm.weight"] = make_input([1, vlm.width], device)

    vlm_q_dim = vlm.num_heads * vlm.head_dim
    vlm_kv_dim = vlm.num_kv_heads * vlm.head_dim
    for i in range(vlm.depth):
        prefix = f"model.layers.{i}."
        weights["vlm_language"][f"{prefix}input_layernorm.weight"] = make_input([1, vlm.width], device)
        weights["vlm_language"][f"{prefix}post_attention_layernorm.weight"] = make_input([1, vlm.width], device)
        weights["vlm_language"][f"{prefix}self_attn.q_proj.weight"] = make_input([vlm.width, vlm_q_dim], device)
        weights["vlm_language"][f"{prefix}self_attn.k_proj.weight"] = make_input([vlm.width, vlm_kv_dim], device)
        weights["vlm_language"][f"{prefix}self_attn.v_proj.weight"] = make_input([vlm.width, vlm_kv_dim], device)
        weights["vlm_language"][f"{prefix}self_attn.o_proj.weight"] = make_input([vlm_q_dim, vlm.width], device)
        weights["vlm_language"][f"{prefix}mlp.gate_proj.weight"] = make_input([vlm.width, vlm.mlp_dim], device)
        weights["vlm_language"][f"{prefix}mlp.up_proj.weight"] = make_input([vlm.width, vlm.mlp_dim], device)
        weights["vlm_language"][f"{prefix}mlp.down_proj.weight"] = make_input([vlm.mlp_dim, vlm.width], device)

    # Action expert weights (Gemma)
    weights["action_expert"]["model.embed_tokens.weight"] = make_input([10000, expert.width], device)
    weights["action_expert"]["model.norm.weight"] = make_input([1, expert.width], device)

    expert_q_dim = expert.num_heads * expert.head_dim
    expert_kv_dim = expert.num_kv_heads * expert.head_dim
    for i in range(expert.depth):
        prefix = f"model.layers.{i}."
        weights["action_expert"][f"{prefix}input_layernorm.weight"] = make_input([1, expert.width], device)
        weights["action_expert"][f"{prefix}post_attention_layernorm.weight"] = make_input([1, expert.width], device)
        weights["action_expert"][f"{prefix}self_attn.q_proj.weight"] = make_input([expert.width, expert_q_dim], device)
        weights["action_expert"][f"{prefix}self_attn.k_proj.weight"] = make_input([expert.width, expert_kv_dim], device)
        weights["action_expert"][f"{prefix}self_attn.v_proj.weight"] = make_input([expert.width, expert_kv_dim], device)
        weights["action_expert"][f"{prefix}self_attn.o_proj.weight"] = make_input([expert_q_dim, expert.width], device)
        weights["action_expert"][f"{prefix}mlp.gate_proj.weight"] = make_input([expert.width, expert.mlp_dim], device)
        weights["action_expert"][f"{prefix}mlp.up_proj.weight"] = make_input([expert.width, expert.mlp_dim], device)
        weights["action_expert"][f"{prefix}mlp.down_proj.weight"] = make_input([expert.mlp_dim, expert.width], device)

    model = PaliGemmaBackboneTTNN(config, weights, device)
    x = make_input([1, 3, siglip.image_size, siglip.image_size], device)
    out = model.embed_image(x)

    expected = (1, num_patches, vlm.width)
    actual = get_tensor_shape(out)
    passed = actual == expected
    return passed, f"{'✅ PASS' if passed else '❌ FAIL'} - got {list(actual)}, expected {list(expected)}"


def test_pi0_model(device: Any) -> Tuple[bool, str]:
    """Test full PI0 Model shapes."""
    from workloads.ttnn.PI0.tt.ttnn_pi0_model import PI0ModelTTNN

    model = PI0ModelTTNN(
        device=device,
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        num_denoising_steps=10,
        max_seq_len=2048,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
        bs=1,
        image_size=224,
        patch_size=14,
        num_images=1,
        lang_seq_len=256,
    )

    actions = model.sample_actions()

    expected = (1, 50, 32)
    actual = get_tensor_shape(actions)
    passed = actual == expected
    return passed, f"{'✅ PASS' if passed else '❌ FAIL'} - got {list(actual)}, expected {list(expected)}"


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    print("=" * 80)
    print("  PI0 Shape Tests - All Components (TTSim)")
    print("=" * 80)

    device = ttnn.open_device(device_id=0)

    results: List[Tuple[str, bool, str, float]] = []

    tests: List[Tuple[str, Callable[[Any], Tuple[bool, str]]]] = [
        ("Suffix", test_suffix),
        ("Prefix", test_prefix),
        ("Gemma MLP", test_gemma_mlp),
        ("SigLIP", test_siglip),
        ("PaliGemma", test_paligemma),
        ("PI0 Model", test_pi0_model),
    ]

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"  {test_name}")
        print("=" * 60)

        try:
            start = time.time()
            passed, message = test_func(device)
            elapsed = time.time() - start
            print(f"  {message} ({elapsed:.3f}s)")
            results.append((test_name, passed, message, elapsed))
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, f"ERROR: {e}", 0.0))

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"\n{'Component':<15} {'Status':<10} {'Time':<10}")
    print("-" * 40)

    for test_name, passed, _, elapsed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<15} {status:<10} {elapsed:.3f}s")

    passed_count = sum(1 for _, passed, _, _ in results if passed)
    total_count = len(results)

    print("\n" + "=" * 80)
    print(f"  Results: {passed_count}/{total_count} tests passed")
    print("=" * 80)

    ttnn.close_device(device)

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())