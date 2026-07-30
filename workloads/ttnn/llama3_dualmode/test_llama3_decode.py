#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode llama3 DECODE runner.

Companion to test_llama3_prefill.py, using the same workloads.ttnn.tt_transformers_dualmode
components in decode mode (seqlen=1 per user, batch users, paged KV cache). The decode path
in attention.forward_decode emits NLPCreateQKVHeadsDecode -> RotaryEmbeddingLlamaFusedQK ->
PagedFusedUpdateCache -> SdpaDecode -> NLPConcatHeadsDecode, and rope.get_rot_mats supplies the
decode cos/sin via embedding + transpose + interleaved_to_sharded — matching the BH decode
capture (project_llama3_prefill_op_sequence). Builds the decode graph on-device; Polaris
extracts the WorkloadGraph after this returns.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from loguru import logger

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]

from workloads.ttnn.tt_transformers_dualmode.model import Transformer
from workloads.ttnn.tt_transformers_dualmode.model_config import ModelArgs

_NUM_LAYERS = {'llama3-8B': 32, 'llama3-3B': 28, 'llama3-1B': 16, 'llama3-70B': 80}


class PagedAttentionConfig:
    def __init__(self, block_size=32, max_num_blocks=1024):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks


def run_llama3_dualmode_decode(wlname: str, ttnn_device, cfg: dict):
    assert isinstance(cfg, dict), 'cfg must be a dictionary'
    assert isinstance(wlname, str), 'wlname must be a string'

    model_name = cfg.get('model_name', 'llama3-8B')
    batch = cfg.get('bs', 32)            # decode generates one token per user across `batch` users
    num_layers = cfg.get('n_layers', _NUM_LAYERS.get(model_name, 32))
    dtype = ttnn.bfloat8_b
    seqlen = 1

    paged_attention_config = PagedAttentionConfig(
        block_size=cfg.get('page_block_size', 32),
        max_num_blocks=cfg.get('page_max_num_blocks', 1024),
    )

    logger.info(f'Loading dual-mode TT model {model_name} ({num_layers} layers, decode batch={batch})...')
    args = ModelArgs(
        ttnn_device, model_name=model_name, instruct=False,
        max_batch_size=batch, max_seq_len=cfg.get('max_seq_len', 4096), dummy_weights=True,
    )
    args.n_layers = num_layers

    state_dict = None
    model = Transformer(
        args=args, mesh_device=ttnn_device, dtype=dtype, state_dict=state_dict,
        weight_cache_path=None, paged_attention_config=paged_attention_config,
        use_paged_kv_cache=False,
    )

    # One token per user -> embedding -> [1,1,batch,dim] decode hidden state
    if IS_POLARIS:
        tokens = ttnn._rand(shape=[1, batch], device=ttnn_device, dtype=ttnn.int32)
    else:
        import torch  # type: ignore[import-not-found]
        tokens = ttnn.from_torch(
            torch.randint(0, args.vocab_size, (1, batch)), dtype=ttnn.uint32, device=ttnn_device,
        )
    x = model.embd(tokens)
    x = ttnn.unsqueeze_to_4D(x)

    # Per-user current positions; rot_idxs is [1, batch] for rope.get_rot_mats (decode)
    if IS_POLARIS:
        current_pos = ttnn._rand(shape=[1, batch], device=ttnn_device, dtype=ttnn.int32)
    else:
        import torch  # type: ignore[import-not-found]
        current_pos = ttnn.from_torch(torch.zeros(1, batch, dtype=torch.int32), dtype=ttnn.int32, device=ttnn_device)

    # Separate rope-index counter — real llama3 decode keeps rot_mat_idxs distinct from the
    # KV-cache current_pos and advances both per step (HW emits PlusOne x2).
    if IS_POLARIS:
        rot_idxs = ttnn._rand(shape=[1, batch], device=ttnn_device, dtype=ttnn.int32)
    else:
        import torch  # type: ignore[import-not-found]
        rot_idxs = ttnn.from_torch(torch.zeros(1, batch, dtype=torch.int32), dtype=ttnn.int32, device=ttnn_device)
    rot_mats = model.rope_setup.get_rot_mats(rot_idxs)

    tt_out = model(x, current_pos, rot_mats=rot_mats, mode='decode', page_table=None)
    logger.info(f'Decode output shape {list(tt_out.shape)}')

    # Advance position counters for the next decode step (matches HW PlusOne x2).
    ttnn.plus_one(current_pos, skip_negative_entries=True)
    ttnn.plus_one(rot_idxs)

    # Sampling head: extend the graph through the decode sampling tail so it spans
    # the full HW capture (logits -> sampled token), not just lm_head.
    from workloads.ttnn.tt_transformers_dualmode.sampling import sample_decode
    tt_tok = sample_decode(tt_out, ttnn_device)
    logger.info(f'Decode sampled token shape {list(tt_tok.shape)}')
    logger.info(f'Finished dual-mode decode for {model_name}.')


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'llama3-8B'
    dev = ttnn.open_device(device_id=0)
    run_llama3_dualmode_decode(wlname='llama3_dualmode', ttnn_device=dev, cfg={'model_name': name, 'n_layers': 1})
