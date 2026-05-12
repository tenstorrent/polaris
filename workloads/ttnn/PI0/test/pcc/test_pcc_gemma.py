# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Shape Correctness Test: Gemma MLP - TTSim
Tests that the ttsim Gemma MLP modules produce outputs with correct shapes.
NOTE: ttsim is a shape-tracking performance simulator — it does NOT
compute real numerical values. Tests assert output shapes only.
Usage:
    pytest test_ttsim_gemma.py -v
    python test_ttsim_gemma.py
"""
import sys
from pathlib import Path
from typing import List
import pytest
import ttsim.front.ttnn as ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from workloads.ttnn.PI0.tt.ttnn_gemma import GemmaMLPTTNN
from workloads.ttnn.PI0.common.configs import GemmaConfig

SEQ_LEN = 64


def safe_shape_list(shape) -> List[int]:
    """Convert shape to list, raising if None."""
    if shape is None:
        raise ValueError("Shape cannot be None")
    return list(shape)


def create_vlm_config() -> GemmaConfig:
    """Create VLM Gemma config (2B)."""
    return GemmaConfig(
        width=2048,
        depth=18,
        mlp_dim=16384,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    )


def create_expert_config() -> GemmaConfig:
    """Create Expert Gemma config (300M)."""
    return GemmaConfig(
        width=1024,
        depth=18,
        mlp_dim=4096,
        num_heads=8,
        num_kv_heads=1,
        head_dim=128,
    )


def make_input(shape: list, device) -> ttnn.Tensor:
    """Create a ttsim input tensor of given shape."""
    return ttnn.zeros(
        shape,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def create_mlp_weights(config: GemmaConfig, device) -> dict:
    """Create MLP weight tensors for testing."""
    return {
        "mlp.gate_proj.weight": make_input([config.width, config.mlp_dim], device),
        "mlp.up_proj.weight": make_input([config.width, config.mlp_dim], device),
        "mlp.down_proj.weight": make_input([config.mlp_dim, config.width], device),
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


def test_gemma_vlm_mlp_shape(device):
    """Test VLM MLP shape: (batch, seq, 2048) -> (batch, seq, 2048)"""
    config = create_vlm_config()
    weights = create_mlp_weights(config, device)
    mlp = GemmaMLPTTNN(config, weights, device)
    
    x = make_input([1, SEQ_LEN, config.width], device)
    out = mlp.forward(x)
    
    assert_shape(out, [1, SEQ_LEN, config.width], "vlm_mlp")


def test_gemma_expert_mlp_shape(device):
    """Test Expert MLP shape: (batch, seq, 1024) -> (batch, seq, 1024)"""
    config = create_expert_config()
    weights = create_mlp_weights(config, device)
    mlp = GemmaMLPTTNN(config, weights, device)
    
    x = make_input([1, SEQ_LEN, config.width], device)
    out = mlp.forward(x)
    
    assert_shape(out, [1, SEQ_LEN, config.width], "expert_mlp")


def main():
    """Standalone shape smoke test without pytest."""
    print("=" * 70)
    print("  Gemma MLP Shape Test (TTSim)")
    print("=" * 70)
    
    device = ttnn.open_device(device_id=0)
    try:
        results = []
        
        # Test 1: VLM MLP
        print("\n1. Testing VLM MLP (2B)...")
        config = create_vlm_config()
        weights = create_mlp_weights(config, device)
        mlp = GemmaMLPTTNN(config, weights, device)
        x = make_input([1, SEQ_LEN, config.width], device)
        out = mlp.forward(x)
        expected = [1, SEQ_LEN, config.width]
        actual = safe_shape_list(out.shape)
        results.append(("vlm_mlp", actual, expected, actual == expected))
        
        # Test 2: Expert MLP
        print("2. Testing Expert MLP (300M)...")
        config = create_expert_config()
        weights = create_mlp_weights(config, device)
        mlp = GemmaMLPTTNN(config, weights, device)
        x = make_input([1, SEQ_LEN, config.width], device)
        out = mlp.forward(x)
        expected = [1, SEQ_LEN, config.width]
        actual = safe_shape_list(out.shape)
        results.append(("expert_mlp", actual, expected, actual == expected))
        
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