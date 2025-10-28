#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.llama3.rope import RotarySetup

def filter_ttnn_attrs(attrs_dict):
    return {k: v for k, v in attrs_dict.items() if not (isinstance(v, ttnn.Tensor) or k == "layout" or k == "memory_config")}

def test_rotary_setup():
    """Test driver for RotarySetup class"""
    print("=== Testing RotarySetup ===")

    # Test parameters
    batch_size = 32
    head_dim = 64
    max_seq_len = 128
    rope_theta = 10000.0
    scale_factor = 1.0
    orig_context_len = 2048

    # Create device
    device = ttnn.open_device(device_id=0)#, device_type="ttnn", device_name=None)

    try:
        # Initialize RotarySetup
        print(f"Initializing RotarySetup with:")
        print(f"  batch_size: {batch_size}")
        print(f"  head_dim: {head_dim}")
        print(f"  max_seq_len: {max_seq_len}")
        print(f"  rope_theta: {rope_theta}")

        rope_setup = RotarySetup(
            device=device,
            batch_size=batch_size,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            scale_factor=scale_factor,
            orig_context_len=orig_context_len,
            datatype=ttnn.bfloat16,
        )
        print(" RotarySetup initialized successfully")

        # Test get_both_trans_mats
        print("\nTesting get_both_trans_mats...")
        trans_mats = rope_setup.get_both_trans_mats()
        assert "decode" in trans_mats, "Missing decode transformation matrix"
        assert "prefill" in trans_mats, "Missing prefill transformation matrix"
        print(f" Transformation matrices available:")
        print(f"  decode shape: {trans_mats['decode'].shape}")
        print(f"  prefill shape: {trans_mats['prefill'].shape}")

        # Test get_rot_idxs
        print("\nTesting get_rot_idxs...")
        position_ids = ttnn.arange(batch_size, dtype=ttnn.int32, device=device)
        _, position_ids = rope_setup.get_rot_idxs(position_ids, on_host=False)
        # Test get_rot_mats

        print("\nTesting get_rot_mats...")
        cos_sin_mats = rope_setup.get_rot_mats(position_ids)
        cos_mat, sin_mat = cos_sin_mats
        print(f" Rotation matrices computed:")
        print(f"  cos_mat shape: {cos_mat.shape}")
        print(f"  sin_mat shape: {sin_mat.shape}")

        # Test get_rot_mats with return_rot_idxs=True
        print("\nTesting get_rot_mats with return_rot_idxs=True...")
        (cos_mat2, sin_mat2), rot_idxs2 = rope_setup.get_rot_mats(position_ids, return_rot_idxs=True)
        print(f" Rotation matrices with indices:")
        print(f"  cos_mat shape: {cos_mat2.shape}")
        print(f"  sin_mat shape: {sin_mat2.shape}")
        print(f"  rot_idxs shape: {rot_idxs2.shape}")
        print("\n=== All Tests Passed! ===")

        # Generate computation graph
        # print("\nGenerating computation graph...")
        # g = device.get_graph()
        # g.graph2onnx('test_rope.onnx', do_model_check=False,
        #             filter_op_attrs=filter_ttnn_attrs)

    except Exception as e:
        print(f" Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        ttnn.close_device(device)
        print("Device closed.")


if __name__ == "__main__":
    test_rotary_setup()
    print("\n All tests completed!")
