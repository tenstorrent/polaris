# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 Conv2D Patch Embedding test.
Mirrors tt-metal's models/demos/multimodal/gemma3/tests/test_patch_embedding.py
(run_conv2d_inference), adapted to Polaris's shape-only simulation mode.
"""
import os
import sys
import traceback

import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.tt.gemma_conv2d_patch import TtGemmaConv2dPatch
from workloads.ttnn.gemma3.tt.model_config import ModelArgs

sys.path.append(os.path.dirname(__file__))
from _test_utils import get_device, check_shape, safe_deallocate, numpy_to_ttnn_tensor, print_summary  # type: ignore[import-not-found]


def run_conv2d_inference(wln, device, cfg, **kwargs):
    """Reference: run_conv2d_inference. B, NCH, H, W input -> (B, num_patches, vision_dim)."""
    bsz = int(cfg.get("bs", 1))
    dtype = ttnn.bfloat16

    logger.info("=" * 60)
    logger.info(f"Running: {cfg.get('model_name', 'gemma3-patch-embedding')} (wln={wln})")
    logger.info("=" * 60)

    results = {}
    try:
        model_args = ModelArgs(device, max_batch_size=bsz, max_seq_len=256)
        state_dict = model_args.load_state_dict()

        tt_layer_prefix = "model.vision_tower.vision_model.embeddings.patch_embedding."

        in_channels = model_args.vision_in_channels
        out_channels = model_args.vision_dim
        kernel_size = model_args.vision_patch_size
        stride = model_args.vision_patch_size
        H = W = model_args.vision_chunk_size
        num_patches = (H // kernel_size) ** 2

        assert H % kernel_size == 0, "Height should be divisible by kernel_size."
        assert W % kernel_size == 0, "Width should be divisible by kernel_size."

        input_tensor_np = np.random.rand(bsz, in_channels, H, W).astype(np.float32)
        input_tensor_tt = numpy_to_ttnn_tensor(input_tensor_np, device=device, dtype=dtype)

        tt_model = TtGemmaConv2dPatch(
            device,
            state_dict,
            tt_layer_prefix,
            dtype,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            True,
            image_size=H,
        )
        tt_output = tt_model(input_tensor_tt)

        expected_shape = (bsz, num_patches, out_channels)
        passed = check_shape(tt_output, expected_shape, "conv2d_patch")
        results["conv2d_patch"] = "PASSED (shape)" if passed else "FAILED (shape mismatch)"

        safe_deallocate(input_tensor_tt)
        safe_deallocate(tt_output)
    except Exception as e:
        logger.error(f"Error in run_conv2d_inference: {e}")
        traceback.print_exc()
        results["conv2d_patch"] = f"ERROR: {e}"

    all_passed = print_summary(results, "PATCH EMBEDDING TEST SUMMARY")
    return {"results": results, "all_passed": all_passed}


if __name__ == "__main__":
    device = get_device()
    result = run_conv2d_inference(wln="gemma3_patch_embedding_test", device=device, cfg={"bs": 1})
    sys.exit(0 if result["all_passed"] else 1)
