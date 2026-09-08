# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 vision transformer block test (attention + MLP + residuals, one layer).
Mirrors tt-metal's test_vision_transformer_block.py (run_block_inference),
adapted to Polaris's shape-only simulation mode.
"""
import os
import sys
import traceback

import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.tt.gemma_image_block import TtGemmaImageTransformerBlock
from workloads.ttnn.gemma3.tt.model_config import ModelArgs

sys.path.append(os.path.dirname(__file__))
from _test_utils import get_device, check_shape, safe_deallocate, numpy_to_ttnn_tensor, print_summary  # type: ignore[import-not-found]


def run_block_inference(wln, device, cfg, **kwargs):
    batch = int(cfg.get("batch", 1))

    logger.info("=" * 60)
    logger.info(f"Running: {cfg.get('model_name', 'gemma3-vision-transformer-block')} (wln={wln})")
    logger.info("=" * 60)

    results = {}
    try:
        dtype = ttnn.bfloat16
        model_args = ModelArgs(device, max_batch_size=batch, max_seq_len=256)
        state_dict = model_args.load_state_dict()

        first_layer_prefix = "model.vision_tower.vision_model.encoder.layers.0."

        dim = model_args.vision_dim
        seq_len = (model_args.vision_chunk_size // model_args.vision_patch_size) ** 2  # num_patches

        tt_model = TtGemmaImageTransformerBlock(
            device,
            state_dict=state_dict,
            state_dict_prefix=first_layer_prefix,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
            configuration=model_args,
        )

        pt_attention_input = np.random.rand(1, batch, seq_len, dim).astype(np.float32)
        attention_input = numpy_to_ttnn_tensor(pt_attention_input, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        mask_np = np.zeros((batch, 1, seq_len, seq_len), dtype=np.float32)
        tt_mask = numpy_to_ttnn_tensor(mask_np, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        tt_out = tt_model(attention_input, mask=tt_mask)

        expected_shape = (batch, seq_len, dim)
        passed = check_shape(tt_out, expected_shape, "vision_transformer_block")
        results["vision_transformer_block"] = "PASSED (shape)" if passed else "FAILED (shape mismatch)"

        safe_deallocate(attention_input)
        safe_deallocate(tt_mask)
        safe_deallocate(tt_out)
    except Exception as e:
        logger.error(f"Error in run_block_inference: {e}")
        traceback.print_exc()
        results["vision_transformer_block"] = f"ERROR: {e}"

    all_passed = print_summary(results, "VISION TRANSFORMER BLOCK TEST SUMMARY")
    return {"results": results, "all_passed": all_passed}


if __name__ == "__main__":
    device = get_device()
    result = run_block_inference(wln="gemma3_vision_transformer_block_test", device=device, cfg={"batch": 1})
    sys.exit(0 if result["all_passed"] else 1)
