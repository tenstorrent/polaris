# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Shape Correctness Test: Suffix Embedding - TTSim
Tests that the ttsim SuffixEmbeddingTTNN module produces outputs with correct shapes.
NOTE: ttsim is a shape-tracking performance simulator — it does NOT
compute real numerical values. Tests assert output shapes only.
Usage:
    pytest test_ttsim_suffix.py -v
    python test_ttsim_suffix.py
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
import ttsim.front.ttnn as ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from workloads.ttnn.PI0.tt.ttnn_suffix import SuffixEmbeddingTTNN
from workloads.ttnn.PI0.common.configs import SuffixConfig


def safe_shape_list(shape: Any) -> List[int]:
    """Convert shape to list, raising if None."""
    if shape is None:
        raise ValueError("Shape cannot be None")
    return list(shape)


def create_suffix_config() -> SuffixConfig:
    """Create SuffixConfig matching checkpoint."""
    return SuffixConfig(
        action_dim=32,
        action_horizon=50,
        expert_width=1024,
        state_dim=32,
        time_emb_dim=1024,
        pi05=False,
    )


def make_input(
    shape: List[int], device: Any, dtype: Any = ttnn.bfloat16, layout: Any = ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    """Create a ttsim input tensor of given shape."""
    return ttnn.zeros(
        shape,
        device=device,
        dtype=dtype,
        layout=layout,
    )


def create_suffix_weights(config: SuffixConfig, device: Any) -> Dict[str, ttnn.Tensor]:
    """Create suffix weight tensors for testing."""
    return {
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


def assert_shape(tensor: ttnn.Tensor, expected: List[int], label: str) -> None:
    """Assert ttsim tensor has the expected shape."""
    actual = safe_shape_list(tensor.shape)
    assert actual == expected, f"[{label}] Shape mismatch: got {actual}, expected {expected}"
    print(f"\n✅ [{label}] Shape OK: {actual}")


@pytest.fixture
def device() -> Any:
    """Open a ttsim device and close it after each test."""
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_pcc_suffix_embed_actions(device: Any) -> None:
    """Test suffix action embedding shape: (batch, horizon, action_dim) -> (batch, horizon, expert_width)"""
    config = create_suffix_config()
    weights = create_suffix_weights(config, device)
    model = SuffixEmbeddingTTNN(config, weights, device)

    noisy_actions = make_input([1, config.action_horizon, config.action_dim], device)
    out = model.embed_actions(noisy_actions)

    assert_shape(out, [1, config.action_horizon, config.expert_width], "embed_actions")


def test_pcc_suffix_embed_state(device: Any) -> None:
    """Test suffix state embedding shape: (batch, state_dim) -> (batch, 1, expert_width)"""
    config = create_suffix_config()
    weights = create_suffix_weights(config, device)
    model = SuffixEmbeddingTTNN(config, weights, device)

    state = make_input([1, config.state_dim], device)
    out = model.embed_state(state)

    assert out is not None, "embed_state returned None"
    assert_shape(out, [1, 1, config.expert_width], "embed_state")


def test_pcc_suffix_project_output(device: Any) -> None:
    """Test suffix output projection shape: (batch, horizon, expert_width) -> (batch, horizon, action_dim)"""
    config = create_suffix_config()
    weights = create_suffix_weights(config, device)
    model = SuffixEmbeddingTTNN(config, weights, device)

    expert_output = make_input([1, config.action_horizon, config.expert_width], device)
    out = model.project_output(expert_output)

    assert_shape(out, [1, config.action_horizon, config.action_dim], "project_output")


def test_pcc_suffix_full_embed(device: Any) -> None:
    """Test full suffix embedding shape: returns (batch, 1+horizon, expert_width)"""
    config = create_suffix_config()
    weights = create_suffix_weights(config, device)
    model = SuffixEmbeddingTTNN(config, weights, device)

    state = make_input([1, config.state_dim], device)
    noisy_actions = make_input([1, config.action_horizon, config.action_dim], device)
    timestep = make_input([1], device, layout=ttnn.ROW_MAJOR_LAYOUT)

    suffix_embs, _, _, _ = model.embed_suffix(state, noisy_actions, timestep)

    expected_len = 1 + config.action_horizon
    assert_shape(suffix_embs, [1, expected_len, config.expert_width], "embed_suffix")


def main() -> int:
    """Standalone shape smoke test without pytest."""
    print("=" * 70)
    print("  Suffix Embedding Shape Test (TTSim)")
    print("=" * 70)

    config = create_suffix_config()
    device = ttnn.open_device(device_id=0)

    try:
        results: List[tuple[str, Optional[List[int]], List[int], bool]] = []

        print("\n1. Creating weights and model...")
        weights = create_suffix_weights(config, device)
        model = SuffixEmbeddingTTNN(config, weights, device)

        # Test 1: embed_actions
        print("2. Testing embed_actions...")
        noisy_actions = make_input([1, config.action_horizon, config.action_dim], device)
        out = model.embed_actions(noisy_actions)
        expected: List[int] = [1, config.action_horizon, config.expert_width]
        actual: List[int] = safe_shape_list(out.shape)
        results.append(("embed_actions", actual, expected, actual == expected))

        # Test 2: embed_state
        print("3. Testing embed_state...")
        state = make_input([1, config.state_dim], device)
        out_state = model.embed_state(state)
        expected = [1, 1, config.expert_width]
        actual_state: Optional[List[int]] = safe_shape_list(out_state.shape) if out_state is not None else None
        results.append(("embed_state", actual_state, expected, actual_state == expected))

        # Test 3: project_output
        print("4. Testing project_output...")
        expert_output = make_input([1, config.action_horizon, config.expert_width], device)
        out_proj = model.project_output(expert_output)
        expected = [1, config.action_horizon, config.action_dim]
        actual = safe_shape_list(out_proj.shape)
        results.append(("project_output", actual, expected, actual == expected))

        # Test 4: embed_suffix
        print("5. Testing embed_suffix...")
        state = make_input([1, config.state_dim], device)
        noisy_actions = make_input([1, config.action_horizon, config.action_dim], device)
        timestep = make_input([1], device, layout=ttnn.ROW_MAJOR_LAYOUT)
        suffix_embs, _, _, _ = model.embed_suffix(state, noisy_actions, timestep)
        expected_len = 1 + config.action_horizon
        expected = [1, expected_len, config.expert_width]
        actual = safe_shape_list(suffix_embs.shape)
        results.append(("embed_suffix", actual, expected, actual == expected))

        # Results
        print("\n" + "=" * 70)
        print("  RESULTS")
        print("=" * 70)
        all_passed = all(ok for _, _, _, ok in results)
        for name, actual_res, expected_res, ok in results:
            status = "✅ PASS" if ok else "❌ FAIL"
            print(f"  {status}  {name:<20} got={actual_res}  expected={expected_res}")
        print("=" * 70)
        print(f"  Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        print("=" * 70)
        return 0 if all_passed else 1

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())