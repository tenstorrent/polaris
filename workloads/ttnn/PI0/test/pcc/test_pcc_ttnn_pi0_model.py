# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Shape Correctness Test: PI0 Model - TTSim
Tests that the ttsim PI0 model produces outputs with correct shapes.
NOTE: ttsim is a shape-tracking performance simulator — it does NOT
compute real numerical values. Tests assert output shapes only.
Usage:
    pytest test_ttsim_pi0_model.py -v
    python test_ttsim_pi0_model.py
"""
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import pytest
import ttsim.front.ttnn as ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from workloads.ttnn.PI0.tt.ttnn_pi0_model import PI0ModelTTNN

BATCH_SIZE = 1
EXPECTED_SHAPE = [BATCH_SIZE, 50, 32]

PI0_CONFIG: Dict[str, Any] = {
    'action_dim': 32,
    'action_horizon': 50,
    'state_dim': 32,
    'num_denoising_steps': 10,
    'max_seq_len': 2048,
    'paligemma_variant': 'gemma_2b',
    'action_expert_variant': 'gemma_300m',
    'pi05': False,
    'bs': BATCH_SIZE,
    'image_size': 224,
    'patch_size': 14,
    'num_images': 1,
    'lang_seq_len': 256,
}


def safe_shape_list(shape: Any) -> List[int]:
    """Convert shape to list, raising if None."""
    if shape is None:
        raise ValueError("Shape cannot be None")
    return list(shape)


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


def test_pcc_pi0_ttnn(device: Any) -> None:
    """Test PI0 model sample_actions shape."""
    model = PI0ModelTTNN(device=device, **PI0_CONFIG)
    ttnn_actions = model.sample_actions()
    assert_shape(ttnn_actions, EXPECTED_SHAPE, "PI0ModelTTNN.sample_actions")


def main() -> int:
    """Standalone shape smoke test without pytest."""
    print("=" * 80)
    print("  PI0 TTNN SHAPE TEST")
    print("  TTSim (shape-tracking only, no numerical values)")
    print("=" * 80)

    device = ttnn.open_device(device_id=0)

    try:
        # Initialize model
        print("\n1. Initializing PI0ModelTTNN...")
        start = time.time()
        model = PI0ModelTTNN(device=device, **PI0_CONFIG)
        print(f"   Model initialized in {(time.time() - start) * 1000:.2f}ms")

        # Test sample_actions
        print("\n2. Testing sample_actions()...")
        start = time.time()
        ttnn_actions = model.sample_actions()
        actual_shape = safe_shape_list(ttnn_actions.shape)
        passed = actual_shape == EXPECTED_SHAPE
        print(f"   Shape: {actual_shape}, Time: {(time.time() - start) * 1000:.2f}ms")

        # Results
        print("\n" + "=" * 80)
        print("  RESULTS")
        print("=" * 80)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}  sample_actions  got={actual_shape}  expected={EXPECTED_SHAPE}")
        print("=" * 80)
        return 0 if passed else 1

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())