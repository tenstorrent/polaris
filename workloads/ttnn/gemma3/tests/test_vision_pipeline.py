# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 Vision Model Tests
Shape-only validation (ttsim simulation mode does not compute values)
"""
import os
import sys
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
import numpy as np
from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device
from workloads.ttnn.gemma3.tt.model_config import ModelArgs
from workloads.ttnn.gemma3.tt.gemma_vision_block import TtSiglipGemmaVisionModel


# =============================================================================
# Helper Functions
# =============================================================================
def safe_deallocate(tensor):
    if hasattr(ttnn, "deallocate"):
        try:
            ttnn.deallocate(tensor)
        except Exception:
            pass
    elif hasattr(tensor, "deallocate"):
        try:
            tensor.deallocate()
        except Exception:
            pass


def get_shape_tuple(tensor):
    """Get shape as tuple from various tensor types."""
    if hasattr(tensor, 'shape'):
        shape = tensor.shape
        if hasattr(shape, '__iter__'):
            return tuple(shape)
        return shape
    if hasattr(tensor, 'get_shape'):
        return tuple(tensor.get_shape())
    return ()


def check_shape(actual_tensor, expected_shape, test_name):
    """
    Shape-only validation for ttsim simulation mode.
    Compares core dimensions, ignoring extra leading dimensions.
    """
    actual = get_shape_tuple(actual_tensor)
    expected = tuple(expected_shape)

    # Get the last N dimensions where N = len(expected)
    if len(actual) >= len(expected):
        actual_core = actual[-len(expected):]
    else:
        actual_core = actual

    # Check if core dimensions match
    match = actual_core == expected

    print(f"\n=== {test_name} SHAPE CHECK ===")
    print(f" Expected: {expected}")
    print(f" Actual (full): {actual}")
    print(f" Actual (core): {actual_core}")
    print(f" Match: {'✅' if match else '❌'}")

    if not match:
        logger.warning(f"{test_name}: shape mismatch! expected {expected}, got {actual}")
    return match


def numpy_to_ttnn_tensor(np_array, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
    """Convert a numpy array to a ttnn.Tensor."""
    return ttnn.Tensor(
        np_array,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def get_device():
    return Device(name="test_device")


# =============================================================================
# Test 1: Full Vision Model
# =============================================================================
def test_gemma_vision(wln, device, cfg, **kwargs):
    """
    Test full Gemma Vision Model.
    Validates output shapes only (ttsim simulation mode).
    """
    bsz = int(cfg.get("bs", 1))
    model_name = cfg.get("model_name", "gemma3-vision")

    logger.info("=" * 60)
    logger.info(f"Running: {model_name} (wln={wln})")
    logger.info(f" bs: {bsz}")
    logger.info("=" * 60)

    all_tests_pass = True
    results = {}

    try:
        model_args = ModelArgs(device, max_batch_size=bsz, max_seq_len=256)
        state_dict = model_args.load_state_dict()
        first_layer_prefix = "model.vision_tower.vision_model."

        image_size = model_args.vision_chunk_size
        in_channels = model_args.vision_in_channels
        patch_size = model_args.vision_patch_size
        vision_dim = model_args.vision_dim
        num_patches = (image_size // patch_size) ** 2

        logger.info(
            f"Model config: image_size={image_size}, patch_size={patch_size}, "
            f"vision_dim={vision_dim}, num_patches={num_patches}"
        )

        # Create input tensor
        input_tensor_np = np.random.rand(bsz, in_channels, image_size, image_size).astype(np.float32)

        print("\n=== POLARIS VISION FULL INPUT ===")
        print(f" image numpy shape: {input_tensor_np.shape}")

        # Convert to ttnn tensor
        input_tensor_tt = numpy_to_ttnn_tensor(
            input_tensor_np,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        tt_model = TtSiglipGemmaVisionModel(
            device,
            state_dict=state_dict,
            state_dict_prefix=first_layer_prefix,
            dtype=ttnn.bfloat16,
            configuration=model_args,
        )

        test_output = tt_model(input_tensor_tt)

        output_shape = get_shape_tuple(test_output)
        print("\n=== POLARIS VISION FULL OUTPUT ===")
        print(f" output shape: {output_shape}")

        expected_shape = (bsz, num_patches, vision_dim)
        passed = check_shape(test_output, expected_shape, "full_vision_model")

        results["vision_model"] = "PASSED (shape)" if passed else "FAILED (shape mismatch)"
        if not passed:
            all_tests_pass = False

        # Cleanup
        safe_deallocate(input_tensor_tt)
        safe_deallocate(test_output)

    except Exception as e:
        logger.error(f"Error in test_gemma_vision: {e}")
        traceback.print_exc()
        results["vision_model"] = f"ERROR: {e}"
        all_tests_pass = False

    logger.info("\n" + "=" * 60)
    logger.info("FULL VISION TEST SUMMARY")
    logger.info("=" * 60)
    for k, v in results.items():
        icon = "✅" if "PASSED" in v else "❌"
        logger.info(f" {icon} {k}: {v}")

    return {
        "model_name": model_name,
        "batch_size": bsz,
        "results": results,
        "all_passed": all_tests_pass,
    }


# =============================================================================
# Test 2: Piecewise Vision Model
# =============================================================================
def test_gemma_vision_piecewise(wln, device, cfg, **kwargs):
    """
    Test Gemma Vision Model component by component.
    Validates shapes of: embeddings → encoder → ln_post → full model.
    """
    from workloads.ttnn.gemma3.tt.gemma_image_transformer import TtGemmaImageTransformer
    from workloads.ttnn.gemma3.tt.siglip_vision_embedding import TtSiglipVisionEmbeddings
    from workloads.ttnn.tt_transformers.llama_layernorm import TtLayerNorm

    bsz = int(cfg.get("bs", 1))
    model_name = cfg.get("model_name", "gemma3-vision-piecewise")

    logger.info("=" * 60)
    logger.info(f"Running: {model_name} (wln={wln})")
    logger.info(f" bs: {bsz}")
    logger.info("=" * 60)

    all_tests_pass = True
    results = {}

    try:
        model_args = ModelArgs(device, max_batch_size=bsz, max_seq_len=256)
        state_dict = model_args.load_state_dict()
        first_layer_prefix = "model.vision_tower.vision_model."

        image_size = model_args.vision_chunk_size
        in_channels = model_args.vision_in_channels
        patch_size = model_args.vision_patch_size
        vision_dim = model_args.vision_dim
        num_patches = (image_size // patch_size) ** 2

        logger.info(
            f"Model config: image_size={image_size}, patch_size={patch_size}, "
            f"vision_dim={vision_dim}, num_patches={num_patches}"
        )

        # Create input tensor
        input_tensor_np = np.random.rand(bsz, in_channels, image_size, image_size).astype(np.float32)

        print("\n=== POLARIS VISION PIECEWISE INPUT ===")
        print(f" image numpy shape: {input_tensor_np.shape}")

        # Expected shapes
        expected_embed_shape = (bsz, num_patches, vision_dim)
        expected_encoder_shape = (bsz, num_patches, vision_dim)
        expected_lnpost_shape = (bsz, num_patches, vision_dim)
        expected_full_shape = (bsz, num_patches, vision_dim)

        # ── Full model run first ──────────────────────────────────────────────
        input_tensor_tt_full = numpy_to_ttnn_tensor(
            input_tensor_np,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        tt_full_model = TtSiglipGemmaVisionModel(
            device,
            state_dict=state_dict,
            state_dict_prefix=first_layer_prefix,
            dtype=ttnn.bfloat16,
            configuration=model_args,
        )

        test_output_full = tt_full_model(input_tensor_tt_full)

        output_shape = get_shape_tuple(test_output_full)
        print("\n=== POLARIS VISION FULL OUTPUT ===")
        print(f" output shape: {output_shape}")

        passed_full = check_shape(test_output_full, expected_full_shape, "full_vision_model")
        results["gemma_vision"] = "PASSED (shape)" if passed_full else "FAILED (shape mismatch)"
        if not passed_full:
            all_tests_pass = False

        safe_deallocate(test_output_full)
        safe_deallocate(input_tensor_tt_full)
        del tt_full_model, test_output_full

        # ── Embeddings ────────────────────────────────────────────────────────
        input_tensor_tt_embed = numpy_to_ttnn_tensor(
            input_tensor_np,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        test_embeddings = TtSiglipVisionEmbeddings(
            mesh_device=device,
            state_dict=state_dict,
            state_dict_prefix=f"{first_layer_prefix}embeddings.",
            dtype=ttnn.bfloat16,
            image_size=model_args.vision_chunk_size,
            patch_size=model_args.vision_patch_size,
            num_channels=model_args.vision_in_channels,
            hidden_dim=model_args.vision_dim,
            bias=True,
        )

        embed_output = test_embeddings(input_tensor_tt_embed)

        embed_shape = get_shape_tuple(embed_output)
        print("\n=== POLARIS VISION EMBEDDINGS OUTPUT ===")
        print(f" output shape: {embed_shape}")

        passed_embed = check_shape(embed_output, expected_embed_shape, "embeddings")
        results["embeddings"] = "PASSED (shape)" if passed_embed else "FAILED (shape mismatch)"
        if not passed_embed:
            all_tests_pass = False

        safe_deallocate(input_tensor_tt_embed)

        # ── Encoder ───────────────────────────────────────────────────────────
        test_encoder = TtGemmaImageTransformer(
            mesh_device=device,
            state_dict=state_dict,
            state_dict_prefix=f"{first_layer_prefix}encoder.",
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat16),
            dtype=ttnn.bfloat16,
            configuration=model_args,
            layers=model_args.vision_n_layers,
            block_key="layers",
        )

        # Get sequence length from embed_output shape
        if len(embed_shape) >= 2:
            seq_len = embed_shape[-2]  # num_patches dimension
        else:
            seq_len = num_patches

        attention_mask_np = np.zeros((bsz, 1, seq_len, seq_len), dtype=np.float32)
        tt_mask = ttnn.Tensor(
            attention_mask_np,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        encoder_output = test_encoder(embed_output, mask=tt_mask)

        encoder_shape = get_shape_tuple(encoder_output)
        print("\n=== POLARIS VISION ENCODER OUTPUT ===")
        print(f" output shape: {encoder_shape}")

        passed_encoder = check_shape(encoder_output, expected_encoder_shape, "encoder")
        results["encoder"] = "PASSED (shape)" if passed_encoder else "FAILED (shape mismatch)"
        if not passed_encoder:
            all_tests_pass = False

        safe_deallocate(tt_mask)

        # ── LN Post ───────────────────────────────────────────────────────────
        test_ln_post = TtLayerNorm(
            device=device,
            dim=model_args.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{first_layer_prefix}post_layernorm.",
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat16),
            weight_dtype=ttnn.bfloat16,
            eps=model_args.norm_eps,
        )

        ln_post_output = test_ln_post(encoder_output)

        ln_shape = get_shape_tuple(ln_post_output)
        print("\n=== POLARIS VISION LN_POST OUTPUT ===")
        print(f" output shape: {ln_shape}")

        passed_lnpost = check_shape(ln_post_output, expected_lnpost_shape, "ln_post")
        results["ln_post"] = "PASSED (shape)" if passed_lnpost else "FAILED (shape mismatch)"
        if not passed_lnpost:
            all_tests_pass = False

        # Cleanup
        safe_deallocate(embed_output)
        safe_deallocate(encoder_output)
        safe_deallocate(ln_post_output)

    except Exception as e:
        logger.error(f"Error in test_gemma_vision_piecewise: {e}")
        traceback.print_exc()
        results["error"] = f"ERROR: {e}"
        all_tests_pass = False

    logger.info("\n" + "=" * 60)
    logger.info("PIECEWISE VISION TEST SUMMARY")
    logger.info("=" * 60)
    for k, v in results.items():
        icon = "✅" if "PASSED" in v else "❌"
        logger.info(f" {icon} {k}: {v}")

    if all_tests_pass:
        logger.info("\n✅ All piecewise vision tests PASSED!")
    else:
        logger.warning("\n❌ Some piecewise vision tests FAILED!")

    return {
        "model_name": model_name,
        "batch_size": bsz,
        "results": results,
        "all_passed": all_tests_pass,
    }


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("GEMMA3 VISION MODEL TESTS")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Created device: {device}")

    test_configs = [
        {"bs": 1, "model_name": "gemma3-vision-bs1"},
    ]

    all_passed = True
    results = {}

    # Test 1: Full Vision Model
    for config in test_configs:
        test_name = f"test_gemma_vision_bs{config['bs']}"
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# {test_name}")
        logger.info(f"{'#' * 60}")

        try:
            result = test_gemma_vision(
                wln="gemma3_vision_test",
                device=device,
                cfg=config,
            )
            if result["all_passed"]:
                results[test_name] = "PASSED"
            else:
                results[test_name] = "FAILED"
                all_passed = False
        except Exception as e:
            logger.error(f"Test failed: {e}")
            traceback.print_exc()
            results[test_name] = f"ERROR: {e}"
            all_passed = False

    # Test 2: Piecewise Vision Model
    for config in test_configs:
        test_name = f"test_gemma_vision_piecewise_bs{config['bs']}"
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# {test_name}")
        logger.info(f"{'#' * 60}")

        try:
            result = test_gemma_vision_piecewise(
                wln="gemma3_vision_piecewise_test",
                device=device,
                cfg=config,
            )
            if result["all_passed"]:
                results[test_name] = "PASSED"
            else:
                results[test_name] = "FAILED"
                all_passed = False
        except Exception as e:
            logger.error(f"Test failed: {e}")
            traceback.print_exc()
            results[test_name] = f"ERROR: {e}"
            all_passed = False

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 60)
    for test_name, result in results.items():
        icon = "✅" if result == "PASSED" else "❌"
        logger.info(f" {icon} {test_name}: {result}")
    logger.info("=" * 60)

    if all_passed:
        logger.info("✅ ALL TESTS PASSED!")
    else:
        logger.error("❌ SOME TESTS FAILED - SEE ABOVE FOR DETAILS")
        sys.exit(1)

    logger.info("=" * 60)