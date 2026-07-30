# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode sampling head — minimal port of the tt_transformers on-device sampler.

Ports the decode happy-path of ``models/common/sampling/tt_sampling.py``
(``TTSampling.sample``) so the Polaris graph spans the full HW capture
(logits -> sampled token), instead of stopping at lm_head.

Uses the multi_step_reduction gather: split the vocab into ``num_splits`` chunks,
top-k each chunk, then concat — matching the capture's ``TopK x2 / Concat x2``
(chunk width = vocab/num_splits = 64128, matching the capture's TopK input).

Known residuals vs the capture (layout-level, deferred):
  - the shim emits one ``Split`` op where HW emits ``Slice`` kernels (the indices
    split + slice granularity; sim Split vs HW Slice x4);
  - the pre-sampling ``Untilize`` is omitted — shim ``topk`` emits ROW_MAJOR (HW
    emits TILE), so untilize's TILE-only guard would reject the tensor; restore
    once topk models the TILE output.

Host/multichip steps (deallocate / from_torch / all_gather / to_memory_config)
are omitted — they do not belong in the single-chip sim graph. Param tensors
(seeds / user_ids / k / p / temp / device offsets) are config-only dummies:
their values are irrelevant to op shapes and device time (accuracy not validated).
"""
import os

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]


def _dummy(shape, device, dtype):
    """Config-only dummy tensor (fabricated on-device; values irrelevant)."""
    return ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def sample_decode(logits, device, max_top_k=32, num_splits=2):
    """logits ``[1, 1, batch, vocab]`` -> sampled token id ``[1, 1, batch, 1]``.

    Sequence: typecast -> topk -> typecast -> add(device offsets) -> untilize
    -> manual_seed -> sampling.
    """
    batch = list(logits.shape)[-2]
    vocab = list(logits.shape)[-1]

    x_bf16 = ttnn.typecast(logits, dtype=ttnn.bfloat16)
    # multi_step_reduction: split the vocab across sub-core grids, top-k each chunk,
    # then gather (matches the capture's TopK x2 + Concat x2; chunk width = vocab/2).
    x_chunks = ttnn.split(x_bf16, vocab // num_splits, dim=3)
    tv_list, ti_list = [], []
    for chunk in x_chunks:
        tv, ti = ttnn.topk(chunk, k=max_top_k, dim=-1)
        tv_list.append(tv)
        ti_list.append(ti)
    topk_values = ttnn.concat(tv_list, dim=3)
    topk_indices = ttnn.concat(ti_list, dim=3)

    topk_indices_i32 = ttnn.typecast(topk_indices, dtype=ttnn.int32)
    offsets = _dummy(list(topk_indices_i32.shape), device, ttnn.int32)
    global_indices = ttnn.add(offsets, topk_indices_i32, dtype=ttnn.uint32)
    # NOTE (first-cut gap): the capture has an Untilize here. Shim topk emits
    # ROW_MAJOR (HW emits TILE), so untilize is redundant and its TILE-only guard
    # would reject this tensor. Restore once topk models the TILE output.

    seeds = _dummy([1, batch], device, ttnn.uint32)
    user_ids = _dummy([1, batch], device, ttnn.uint32)
    ttnn.manual_seed(seeds=seeds, user_ids=user_ids)

    k_t = _dummy([1, batch], device, ttnn.uint32)
    p_t = _dummy([1, batch], device, ttnn.float32)
    temp_t = _dummy([1, batch], device, ttnn.float32)
    return ttnn.sampling(topk_values, global_indices, k=k_t, p=p_t, temp=temp_t)
