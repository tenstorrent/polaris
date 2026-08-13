# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 vision (SigLIP) feed-forward/MLP test.
Mirrors tt-metal's test_vision_mlp.py (run_mlp_inference),
adapted to Polaris's shape-only simulation mode.
"""
import os
import sys
import traceback

import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.tt.gemma_image_mlp import TtGemmaImageFeedForward
from workloads.ttnn.gemma3.tt.model_config import ModelArgs

sys.path.append(os.path.dirname(__file__))
from _test_utils import get_device, check_shape, safe_deallocate, numpy_to_ttnn_tensor, print_summary  # type: ignore[import-not-found]


def run_mlp_inference(wln, device, cfg, **kwargs):
    batch = int(cfg.get("batch", 1))

    logger.info("=" * 60)
    logger.info(f"Running: {cfg.get('model_name', 'gemma3-vision-mlp')} (wln={wln})")
    logger.info("=" * 60)

    results = {}
    try:
        dtype = ttnn.bfloat16
        model_args = ModelArgs(device, max_batch_size=batch, max_seq_len=256)
        state_dict = model_args.load_state_dict()

        first_layer_prefix = "model.vision_tower.vision_model.encoder.layers.0.mlp."

        dim = model_args.vision_dim
        seq_len = (model_args.vision_chunk_size // model_args.vision_patch_size) ** 2  # num_patches

        tt_model = TtGemmaImageFeedForward(
            mesh_device=device,
            args=model_args,
            state_dict=state_dict,
            state_dict_prefix=first_layer_prefix,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
        )

        torch_input = np.random.rand(1, batch, seq_len, dim).astype(np.float32)
        tt_input = numpy_to_ttnn_tensor(torch_input, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        tt_output = tt_model(tt_input)

        expected_shape = (batch, 1, seq_len, dim)
        passed = check_shape(tt_output, expected_shape, "vision_mlp")
        results["vision_mlp"] = "PASSED (shape)" if passed else "FAILED (shape mismatch)"

        safe_deallocate(tt_input)
        safe_deallocate(tt_output)
    except Exception as e:
        logger.error(f"Error in run_mlp_inference: {e}")
        traceback.print_exc()
        results["vision_mlp"] = f"ERROR: {e}"

    all_passed = print_summary(results, "VISION MLP TEST SUMMARY")
    return {"results": results, "all_passed": all_passed}


if __name__ == "__main__":
    device = get_device()
    result = run_mlp_inference(wln="gemma3_vision_mlp_test", device=device, cfg={"batch": 1})
    sys.exit(0 if result["all_passed"] else 1)
