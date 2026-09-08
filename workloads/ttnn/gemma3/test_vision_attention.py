# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 vision (SigLIP) attention test.
Mirrors tt-metal's test_vision_attention.py (run_attention_inference),
adapted to Polaris's shape-only simulation mode.
"""
import os
import sys
import traceback

import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.tt.gemma_image_attention import TtGemmaImageAttention
from workloads.ttnn.gemma3.tt.model_config import ModelArgs

sys.path.append(os.path.dirname(__file__))
from _test_utils import get_device, check_shape, safe_deallocate, numpy_to_ttnn_tensor, print_summary  # type: ignore[import-not-found]


def run_attention_inference(wln, device, cfg, **kwargs):
    batch = int(cfg.get("batch", 1))

    logger.info("=" * 60)
    logger.info(f"Running: {cfg.get('model_name', 'gemma3-vision-attention')} (wln={wln})")
    logger.info("=" * 60)

    results = {}
    try:
        dtype = ttnn.bfloat16
        model_args = ModelArgs(device, max_batch_size=batch, max_seq_len=256)
        state_dict = model_args.load_state_dict()

        # Same key naming _gemma_dummy_hf_model populates: "...encoder.layers.0.attn."
        first_layer_prefix = "model.vision_tower.vision_model.encoder.layers.0.attn."

        dim = model_args.vision_dim
        seq_len = (model_args.vision_chunk_size // model_args.vision_patch_size) ** 2  # num_patches

        tt_model = TtGemmaImageAttention(
            mesh_device=device,
            state_dict=state_dict,
            state_dict_prefix=first_layer_prefix,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
            configuration=model_args,
        )

        pt_attention_input = np.random.rand(1, batch, seq_len, dim).astype(np.float32)
        attention_input = numpy_to_ttnn_tensor(pt_attention_input, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        tt_out = tt_model(attention_input)

        expected_shape = (batch, seq_len, dim)
        passed = check_shape(tt_out, expected_shape, "vision_attention")
        results["vision_attention"] = "PASSED (shape)" if passed else "FAILED (shape mismatch)"

        safe_deallocate(attention_input)
        safe_deallocate(tt_out)
    except Exception as e:
        logger.error(f"Error in run_attention_inference: {e}")
        traceback.print_exc()
        results["vision_attention"] = f"ERROR: {e}"

    all_passed = print_summary(results, "VISION ATTENTION TEST SUMMARY")
    return {"results": results, "all_passed": all_passed}


if __name__ == "__main__":
    device = get_device()
    result = run_attention_inference(wln="gemma3_vision_attention_test", device=device, cfg={"batch": 1})
    sys.exit(0 if result["all_passed"] else 1)
