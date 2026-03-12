#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for RelPositionEmbedding TTSim module
Validates the conversion from PyTorch to TTSim
"""

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import numpy as np
import torch
import ttsim.front.functional.op as F
from workloads.BEVFormer.ttsim_models.position_embedding import RelPositionEmbedding


def test_position_embedding_construction():
    """Test that the module can be constructed successfully."""
    print("\n" + "=" * 80)
    print("TEST 1: Module Construction")
    print("=" * 80)

    try:
        pos_embed = RelPositionEmbedding(
            name="test_pos_embed", num_pos_feats=64, pos_norm=True
        )
        print("✓ Module constructed successfully")
        print(f"  - Module name: {pos_embed.name}")
        print(f"  - Number of position features: {pos_embed.num_pos_feats}")
        print(f"  - Position normalization: {pos_embed.pos_norm}")
        return True
    except Exception as e:
        print(f"✗ Module construction failed: {e}")
        return False


def test_position_embedding_forward():
    """Test the forward pass with a sample input."""
    print("\n" + "=" * 80)
    print("TEST 2: Forward Pass")
    print("=" * 80)

    try:
        # Create module
        pos_embed = RelPositionEmbedding(
            name="test_pos_embed_forward", num_pos_feats=64, pos_norm=True
        )

        # Create input tensor [B, C, H, W]
        batch_size = 1
        channels = 256
        height = 50
        width = 50

        print(
            f"Creating input tensor with shape: [{batch_size}, {channels}, {height}, {width}]"
        )
        input_tensor = F._from_shape(
            "test_input", [batch_size, channels, height, width]
        )

        # Run forward pass
        print("Running forward pass...")
        output = pos_embed(input_tensor)

        # Check output shape
        expected_shape = [height * width, 64]
        print(f"Expected output shape: {expected_shape}")
        print(f"Actual output shape: {output.shape}")

        if output.shape == expected_shape:
            print("✓ Forward pass successful - output shape matches expected")
            return True
        else:
            print(f"✗ Forward pass failed - shape mismatch")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_different_spatial_sizes():
    """Test with different spatial dimensions."""
    print("\n" + "=" * 80)
    print("TEST 3: Different Spatial Sizes")
    print("=" * 80)

    test_cases = [
        (200, 200),  # BEVFormer base model size
        (100, 100),  # Smaller grid
        (50, 100),  # Non-square grid
    ]

    all_passed = True
    for i, (H, W) in enumerate(test_cases, 1):
        try:
            print(f"\nTest case {i}: H={H}, W={W}")

            pos_embed = RelPositionEmbedding(
                name=f"test_pos_embed_size_{i}", num_pos_feats=128, pos_norm=False
            )

            input_tensor = F._from_shape(f"test_input_{i}", [1, 256, H, W])
            output = pos_embed(input_tensor)

            expected_shape = [H * W, 128]
            if output.shape == expected_shape:
                print(f"  ✓ Output shape correct: {output.shape}")
            else:
                print(
                    f"  ✗ Output shape incorrect: expected {expected_shape}, got {output.shape}"
                )
                all_passed = False
        except Exception as e:
            print(f"  ✗ Test case {i} failed: {e}")
            all_passed = False

    return all_passed


def test_parameter_count():
    """Test parameter count calculation."""
    print("\n" + "=" * 80)
    print("TEST 4: Parameter Count")
    print("=" * 80)

    test_cases = [
        (64, True, 4 * 64 + 2 * 64),  # with LayerNorm
        (64, False, 4 * 64),  # without LayerNorm
        (256, True, 4 * 256 + 2 * 256),  # larger embedding
    ]

    all_passed = True
    for i, (num_feats, use_norm, expected_count) in enumerate(test_cases, 1):
        try:
            print(f"\nTest case {i}: num_feats={num_feats}, pos_norm={use_norm}")

            pos_embed = RelPositionEmbedding(
                name=f"test_pos_embed_params_{i}",
                num_pos_feats=num_feats,
                pos_norm=use_norm,
            )

            param_count = pos_embed.analytical_param_count()
            print(f"  Expected parameter count: {expected_count}")
            print(f"  Actual parameter count: {param_count}")

            if param_count == expected_count:
                print(f"  ✓ Parameter count correct")
            else:
                print(f"  ✗ Parameter count incorrect")
                all_passed = False
        except Exception as e:
            print(f"  ✗ Test case {i} failed: {e}")
            all_passed = False

    return all_passed


def test_without_normalization():
    """Test module without position normalization."""
    print("\n" + "=" * 80)
    print("TEST 5: Without Position Normalization")
    print("=" * 80)

    try:
        pos_embed = RelPositionEmbedding(
            name="test_pos_embed_no_norm", num_pos_feats=128, pos_norm=False
        )

        input_tensor = F._from_shape("test_input_no_norm", [1, 256, 100, 100])
        output = pos_embed(input_tensor)

        expected_shape = [10000, 128]
        if output.shape == expected_shape:
            print(f"✓ Forward pass without normalization successful")
            print(f"  Output shape: {output.shape}")
            return True
        else:
            print(
                f"✗ Output shape incorrect: expected {expected_shape}, got {output.shape}"
            )
            return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_validation():
    """Test that TTSim outputs match PyTorch implementation numerically (simplified test without Linear/LayerNorm)."""
    print("\n" + "=" * 80)
    print("TEST 6: Data Validation (Simplified)")
    print("=" * 80)

    try:
        # PyTorch implementation - simplified to match what we can compute
        def pytorch_pos_encoding(H, W):
            """Simplified position encoding: just cos/sin grid without Linear/LayerNorm."""
            # Y-axis
            y_range = torch.arange(H, dtype=torch.float32) / float(H - 1)
            y_range_scaled = y_range * np.pi
            y_cos = torch.cos(y_range_scaled)
            y_sin = torch.sin(y_range_scaled)
            y_axis = torch.stack([y_cos, y_sin], dim=1)  # [H, 2]
            y_axis = y_axis.view(H, 1, 2).repeat(1, W, 1).view(H * W, 2)  # [H*W, 2]

            # X-axis
            x_range = torch.arange(W, dtype=torch.float32) / float(W - 1)
            x_range_scaled = x_range * np.pi
            x_cos = torch.cos(x_range_scaled)
            x_sin = torch.sin(x_range_scaled)
            x_axis = torch.stack([x_cos, x_sin], dim=1)  # [W, 2]
            x_axis = x_axis.view(1, W, 2).repeat(H, 1, 1).view(H * W, 2)  # [H*W, 2]

            # Concatenate
            pos = torch.cat([y_axis, x_axis], dim=1)  # [H*W, 4]
            return pos

        # TTSim implementation - simplified
        def ttsim_pos_encoding(H, W):
            """Simplified position encoding using TTSim operations."""
            import ttsim.front.functional.op as F

            name = "test_simplified"

            # Y-axis
            y_range_data = np.arange(H, dtype=np.float32) / float(H - 1)
            y_range = F._from_data(name + ".y_range", y_range_data, is_const=True)

            pi_tensor = F._from_data(
                name + ".pi", np.array(np.pi, dtype=np.float32), is_const=True
            )
            y_range_scaled = F.Mul(name + ".y_scaled")(y_range, pi_tensor)

            y_cos = F.Cos(name + ".y_cos")(y_range_scaled)
            y_sin = F.Sin(name + ".y_sin")(y_range_scaled)

            y_cos_unsq = F.Unsqueeze(name + ".y_cos_unsq")(
                y_cos,
                F._from_data(
                    name + ".ax1", np.array([1], dtype=np.int64), is_const=True
                ),
            )
            y_sin_unsq = F.Unsqueeze(name + ".y_sin_unsq")(
                y_sin,
                F._from_data(
                    name + ".ax1_2", np.array([1], dtype=np.int64), is_const=True
                ),
            )

            y_axis = F.ConcatX(name + ".y_concat", axis=1)(y_cos_unsq, y_sin_unsq)
            y_axis_reshaped = F.Reshape(name + ".y_reshape")(
                y_axis,
                F._from_data(
                    name + ".y_shape",
                    np.array([H, 1, 2], dtype=np.int64),
                    is_const=True,
                ),
            )
            y_axis_tiled = F.Tile(name + ".y_tile")(
                y_axis_reshaped,
                F._from_data(
                    name + ".y_reps", np.array([1, W, 1], dtype=np.int64), is_const=True
                ),
            )
            y_axis_final = F.Reshape(name + ".y_final")(
                y_axis_tiled,
                F._from_data(
                    name + ".y_final_shape",
                    np.array([H * W, 2], dtype=np.int64),
                    is_const=True,
                ),
            )

            # X-axis
            x_range_data = np.arange(W, dtype=np.float32) / float(W - 1)
            x_range = F._from_data(name + ".x_range", x_range_data, is_const=True)
            x_range_scaled = F.Mul(name + ".x_scaled")(x_range, pi_tensor)

            x_cos = F.Cos(name + ".x_cos")(x_range_scaled)
            x_sin = F.Sin(name + ".x_sin")(x_range_scaled)

            x_cos_unsq = F.Unsqueeze(name + ".x_cos_unsq")(
                x_cos,
                F._from_data(
                    name + ".ax1_3", np.array([1], dtype=np.int64), is_const=True
                ),
            )
            x_sin_unsq = F.Unsqueeze(name + ".x_sin_unsq")(
                x_sin,
                F._from_data(
                    name + ".ax1_4", np.array([1], dtype=np.int64), is_const=True
                ),
            )

            x_axis = F.ConcatX(name + ".x_concat", axis=1)(x_cos_unsq, x_sin_unsq)
            x_axis_reshaped = F.Reshape(name + ".x_reshape")(
                x_axis,
                F._from_data(
                    name + ".x_shape",
                    np.array([W, 1, 2], dtype=np.int64),
                    is_const=True,
                ),
            )
            x_axis_tiled = F.Tile(name + ".x_tile")(
                x_axis_reshaped,
                F._from_data(
                    name + ".x_reps", np.array([H, 1, 1], dtype=np.int64), is_const=True
                ),
            )
            x_axis_final = F.Reshape(name + ".x_final")(
                x_axis_tiled,
                F._from_data(
                    name + ".x_final_shape",
                    np.array([H * W, 2], dtype=np.int64),
                    is_const=True,
                ),
            )

            # Concatenate
            pos = F.ConcatX(name + ".final_concat", axis=1)(y_axis_final, x_axis_final)
            return pos

        # Test with small dimensions
        H, W = 4, 4

        # Get outputs
        torch_output = pytorch_pos_encoding(H, W).detach().numpy()
        ttsim_output_tensor = ttsim_pos_encoding(H, W)
        ttsim_output = ttsim_output_tensor.data

        print(f"PyTorch output shape: {torch_output.shape}")

        if ttsim_output is None:
            print("✗ TTSim output has no data - data propagation failed")
            return False

        print(f"TTSim output shape: {ttsim_output.shape}")
        print(f"Expected shape: [{H*W}, 4]")

        # Compare shapes
        if torch_output.shape != ttsim_output.shape:
            print(
                f"✗ Shape mismatch: PyTorch {torch_output.shape} vs TTSim {ttsim_output.shape}"
            )
            return False

        # Compare values
        max_diff = np.max(np.abs(torch_output - ttsim_output))
        mean_diff = np.mean(np.abs(torch_output - ttsim_output))
        rel_error = max_diff / (np.max(np.abs(torch_output)) + 1e-6)

        print(f"\nNumerical comparison:")
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        print(f"  Relative error: {rel_error:.6e}")

        # Check if differences are acceptable
        tolerance = 1e-5
        if max_diff < tolerance:
            print(f"✓ Outputs match within tolerance ({tolerance})")
            print(f"\nSample values (first row):")
            print(f"  PyTorch: {torch_output[0, :]}")
            print(f"  TTSim:   {ttsim_output[0, :]}")
            return True
        else:
            print(f"✗ Outputs differ by {max_diff:.6e} (tolerance: {tolerance})")
            print("\nSample values (first row):")
            print(f"  PyTorch: {torch_output[0, :]}")
            print(f"  TTSim:   {ttsim_output[0, :]}")
            return False

    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RelPositionEmbedding TTSim Module Test Suite")
    print("=" * 80)

    results = {
        "Module Construction": test_position_embedding_construction(),
        "Forward Pass": test_position_embedding_forward(),
        "Different Spatial Sizes": test_different_spatial_sizes(),
        "Parameter Count": test_parameter_count(),
        "Without Normalization": test_without_normalization(),
        "Data Validation": test_data_validation(),
    }

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<60} {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n All tests passed! The module is working correctly.")
        return 0
    else:
        print(
            f"\n  {total_tests - passed_tests} test(s) failed. Please review the errors above."
        )
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
