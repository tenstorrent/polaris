# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 vision norm tests: the Gemma-specific RMSNorm (mm_soft_emb_norm) and the shared
TtLayerNorm used for ln_post. Mirrors tt-metal's test_vision_rmsnorm.py
(run_rmsnorm_inference + run_llama_rms_norm), adapted to Polaris's shape-only mode.
"""
import os
import sys
import traceback

import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.tt.gemma_vision_rmsnorm import RMSNorm
from workloads.ttnn.gemma3.tt.model_config import ModelArgs
from workloads.ttnn.tt_transformers.llama_layernorm import TtLayerNorm

sys.path.append(os.path.dirname(__file__))
from _test_utils import get_device, check_shape, safe_deallocate, numpy_to_ttnn_tensor, print_summary  # type: ignore[import-not-found]


def run_rmsnorm_inference(wln, device, cfg, **kwargs):
    """mm_soft_emb_norm: Gemma3-specific RMSNorm with unit-offset weight handling."""
    seq_len = int(cfg.get("seq_len", 128))
    batch_size = int(cfg.get("batch_size", 1))
    dim = 1152  # matches multi_modal_projector.py's hardcoded RMSNorm(dim=1152, ...)
    mode = "decode" if seq_len <= 32 else "prefill"

    logger.info("=" * 60)
    logger.info(f"Running: {cfg.get('model_name', 'gemma3-vision-rmsnorm')} (wln={wln})")
    logger.info("=" * 60)

    results = {}
    try:
        dtype = ttnn.bfloat16
        model_args = ModelArgs(device, max_batch_size=batch_size, max_seq_len=128)
        model_args.n_layers = 1
        state_dict = model_args.load_state_dict()

        tt_model = RMSNorm(
            device=device,
            dim=dim,
            state_dict=state_dict,
            state_dict_prefix="",
            weight_key="model.multi_modal_projector.mm_soft_emb_norm",
            weight_dtype=dtype,
            is_distributed=False,
            sharded_program_config=None,
            sharded_output_config=None,
        )

        input_np = np.random.rand(1, 1, dim).astype(np.float32)
        tt_input = numpy_to_ttnn_tensor(
            input_np, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        tt_output = tt_model(tt_input, mode=mode)

        expected_shape = (1, 1, dim)
        passed = check_shape(tt_output, expected_shape, "mm_soft_emb_norm")
        results["mm_soft_emb_norm"] = "PASSED (shape)" if passed else "FAILED (shape mismatch)"

        safe_deallocate(tt_input)
        safe_deallocate(tt_output)
    except Exception as e:
        logger.error(f"Error in run_rmsnorm_inference: {e}")
        traceback.print_exc()
        results["mm_soft_emb_norm"] = f"ERROR: {e}"

    all_passed = print_summary(results, "VISION RMSNORM (mm_soft_emb_norm) TEST SUMMARY")
    return {"results": results, "all_passed": all_passed}


def run_llama_rms_norm(wln, device, cfg, **kwargs):
    """ln_post: the shared TtLayerNorm, sized for the vision tower."""
    logger.info("=" * 60)
    logger.info(f"Running: {cfg.get('model_name', 'gemma3-vision-ln-post')} (wln={wln})")
    logger.info("=" * 60)

    results = {}
    try:
        dtype = ttnn.bfloat16
        model_args = ModelArgs(device)
        state_dict = model_args.load_state_dict()

        batch_size = 1
        seq_len = 4096
        model_dim = model_args.vision_dim

        x_np = np.random.rand(batch_size, seq_len, model_dim).astype(np.float32)
        tt_input = numpy_to_ttnn_tensor(x_np, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        ln_post = TtLayerNorm(
            device=device,
            dim=model_args.vision_dim,
            state_dict=state_dict,
            state_dict_prefix="model.vision_tower.vision_model.ln_post.",
            weight_cache_path=None,
            weight_dtype=dtype,
            eps=model_args.norm_eps,
        )

        test_output = ln_post(tt_input)

        expected_shape = (batch_size, seq_len, model_dim)
        passed = check_shape(test_output, expected_shape, "ln_post")
        results["ln_post"] = "PASSED (shape)" if passed else "FAILED (shape mismatch)"

        safe_deallocate(tt_input)
        safe_deallocate(test_output)
    except Exception as e:
        logger.error(f"Error in run_llama_rms_norm: {e}")
        traceback.print_exc()
        results["ln_post"] = f"ERROR: {e}"

    all_passed = print_summary(results, "VISION LAYERNORM (ln_post) TEST SUMMARY")
    return {"results": results, "all_passed": all_passed}


if __name__ == "__main__":
    device = get_device()
    r1 = run_rmsnorm_inference(wln="gemma3_vision_rmsnorm_test", device=device, cfg={"seq_len": 128, "batch_size": 1})
    r2 = run_llama_rms_norm(wln="gemma3_vision_ln_post_test", device=device, cfg={})
    sys.exit(0 if (r1["all_passed"] and r2["all_passed"]) else 1)
