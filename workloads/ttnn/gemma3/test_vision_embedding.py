# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 Siglip Vision Embedding test.
Mirrors tt-metal's test_vision_embedding.py (run_vision_embedding_integration),
adapted to Polaris's shape-only simulation mode.
"""
import os
import sys
import traceback

import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.tt.model_config import ModelArgs
from workloads.ttnn.gemma3.tt.siglip_vision_embedding import TtSiglipVisionEmbeddings

sys.path.append(os.path.dirname(__file__))
from _test_utils import get_device, check_shape, safe_deallocate, numpy_to_ttnn_tensor, print_summary  # type: ignore[import-not-found]


def run_vision_embedding_integration(wln, device, cfg, **kwargs):
    bsz = int(cfg.get("bs", 1))
    dtype = ttnn.bfloat16

    logger.info("=" * 60)
    logger.info(f"Running: {cfg.get('model_name', 'gemma3-vision-embedding')} (wln={wln})")
    logger.info("=" * 60)

    results = {}
    try:
        model_args = ModelArgs(device, max_batch_size=bsz, max_seq_len=256)
        state_dict = model_args.load_state_dict()

        first_layer_prefix = "model.vision_tower.vision_model.embeddings."

        image_size = model_args.vision_chunk_size
        patch_size = model_args.vision_patch_size
        hidden_dim = model_args.vision_dim
        in_channels = model_args.vision_in_channels
        num_patches = (image_size // patch_size) ** 2

        input_tensor_np = np.random.rand(bsz, in_channels, image_size, image_size).astype(np.float32)
        input_tensor_tt = numpy_to_ttnn_tensor(input_tensor_np, device=device, dtype=dtype)

        vision_embed = TtSiglipVisionEmbeddings(
            mesh_device=device,
            state_dict=state_dict,
            state_dict_prefix=first_layer_prefix,
            dtype=dtype,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=in_channels,
            hidden_dim=hidden_dim,
            bias=True,
        )

        embeddings = vision_embed(input_tensor_tt)

        expected_shape = (bsz, num_patches, hidden_dim)
        passed = check_shape(embeddings, expected_shape, "vision_embedding")
        results["vision_embedding"] = "PASSED (shape)" if passed else "FAILED (shape mismatch)"

        safe_deallocate(input_tensor_tt)
        safe_deallocate(embeddings)
    except Exception as e:
        logger.error(f"Error in run_vision_embedding_integration: {e}")
        traceback.print_exc()
        results["vision_embedding"] = f"ERROR: {e}"

    all_passed = print_summary(results, "VISION EMBEDDING TEST SUMMARY")
    return {"results": results, "all_passed": all_passed}


if __name__ == "__main__":
    device = get_device()
    result = run_vision_embedding_integration(wln="gemma3_vision_embedding_test", device=device, cfg={"bs": 1})
    sys.exit(0 if result["all_passed"] else 1)
