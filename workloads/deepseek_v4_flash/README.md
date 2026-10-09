# DeepSeek-V4-Flash — TTSIM workload

TTSIM functional perf-simulation workload for
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
(`model_type: deepseek_v4`, `DeepseekV4ForCausalLM`): **284B total / ~13B active**,
1M context, fp8 + fp4 mixed checkpoint.

Entry class: `Dsv4FlashMoe@dsv4f_model.py` (registered as workload `deepseek_v4_flash`
in `config/all_workloads.yaml`).

## Architecture (pinned from `config.json`)

| Component | Detail |
|---|---|
| Embedding / LM head | vocab 129280, hidden 4096, untied |
| Layers | 43, all sparse-MoE (no dense FFN layers) |
| Attention | shared-KV MQA: 64 Q-heads x 512, **one** shared 512-dim KV vector per token (k == v), q_lora 1024, grouped low-rank o_proj (o_groups 8, o_lora 1024), partial RoPE 64/512 (theta 1e4), sliding_window 128, per-head attention sink |
| `compress_ratios` | `[0, 0, (4, 128) x 20, 0]` -> 3 sliding-window-only layers, 20 **CSA** layers (4x compressed KV, Lightning Indexer top-512 blocks), 20 **HCA** layers (128x compressed KV, all blocks) |
| Compressor | learned gated pooling: `wkv`/`wgate` projections + APE + softmax over the pool window; overlapping windows at ratio 4 |
| Lightning Indexer | CSA layers only: 64 heads x 128, scores compressed blocks, selects top-512; own 128-dim compressor |
| MoE | 256 routed + 1 shared, **top-6**, intermediate 2048, **sqrtsoftplus** scoring (`sqrt(softplus(x))`), `noaux_tc` bias (routed layers), **hash routing** (`tid2eid` lookup) on first 3 layers, `norm_topk_prob`, routed scale 1.5, `swiglu_limit` 10, fp4 experts |
| mHC | manifold-constrained hyper-connections: 4 residual streams, per-sublayer `hc_fn` mixes (sinkhorn, 20 iters, eps 1e-6) |
| MTP | 1 draft layer — excluded (same policy as the other LLM workloads) |

## Effective attention width per layer type

For a static window of `L` tokens (prefill) the graph attends `S` KV positions:

| Layer type | S |
|---|---|
| sliding-window-only (ratio 0) | `min(L, 128)` |
| CSA (ratio 4) | `min(L, 128) + min(L/4, 512)` |
| HCA (ratio 128) | `min(L, 128) + L/128` |

E.g. at prefill L=2048: 128 / 640 / 144 — the hybrid compression keeps every
layer's attention window tiny, which is the core of the Flash efficiency.
Decode (L=1) runs the graph over the current step only (per-token compressor /
indexer projections still execute; see abstractions below).

## Validation

- **Parameters**: 284,315,752,941 (~284.3B) vs published **284B** (0.1% off;
  excludes the 1 MTP layer, includes the 3 hash tables).
- **Active params/token** (reads per decode step, incl. embedding + LM head):
  ~13.4B — matches the advertised **~13B activated**.

## Performance-simulation abstractions

1. **Layer multiplicity**: one proxy block per unique `(gate_kind, compress_ratio)`
   layer type — `(hash,0)`x2, `(hash,4)`x1, `(routed,0)`x1, `(routed,4)`x19,
   `(routed,128)`x20 — executed once, op `repeat_count` scales (BasicLLM pattern).
2. **MoE active compute**: top-6 expert slots + shared expert applied to routed
   activations; dispatch/combine data movement not modeled.
3. **Hash routing** (first 3 layers): the real gate still computes
   sqrtsoftplus router scores (modeled), but expert *selection* is a `tid2eid`
   table lookup; the gathered routing weights are approximated by a length-6
   slice of the scores (cost-equivalent; the hash tables count toward params).
4. **Sparse windows**: dynamic KV gathers are approximated by static width-`S`
   slices (the real `sparse_attn` also gathers; matmul cost is identical).
   Decode KV-cache reads are not modeled — attention runs over the current
   window only (same limitation as the other LLM workloads).
5. **mHC sinkhorn**: 20 sinkhorn iterations over tiny mixing tensors are
   approximated by softmax normalization of the `hc_fn` mixes; the 4-stream
   `hc_pre`/`hc_post` residual mixing is modeled op-for-op. The final
   `hc_head` stream reduction is approximated by a plain sum (params counted).
6. **Compressor overlap pooling**: the joint 2*4-token overlap window pool is
   modeled as two independent 4-token pools summed (same softmax/MAC count).
7. **swiglu_limit** clamps (2 eltwise ops/expert) and the indexer Hadamard
   rotation / fp4-fp8 quantization ops are omitted (format transforms, not
   perf-relevant matmul/eltwise work).
8. **RoPE** is approximated by elementwise cos/sin mixing over the full head_dim
   (the true model rotates only 64/512 dims); cost is negligible.
9. **MTP** draft layer is not part of the main inference loop and is not simulated.

## Files

- `dsv4f_model.py` — the workload (`RMSNorm`/`MoEExpert` reused from
  `workloads/qwen3_5/qwen35_model.py`, the established cross-workload pattern)
- `test_model.py` — standalone smoke tests (tiny, decode, prefill 128/2048/4096)
- `tests/test_workloads/test_dsv4flash.py` — pytest suite (`@pytest.mark.unit`)

## Instances

| Instance | seq_len | Notes |
|---|---|---|
| `dsv4f_decode` | 1 | decode step |
| `dsv4f_prefill_2k` | 2048 | CSA S=640, HCA S=144, window layers S=128 |
| `dsv4f_tiny` | 8 | 6-layer scaled-down model for fast CI |

Run:

```sh
python polaris.py --archspec config/all_archs.yaml --wlspec config/all_workloads.yaml \
    --wlmapspec config/wl2archmapping.yaml --filterwlg TTSIM --filterwl deepseek_v4_flash \
    --filterarch Q1_A1 --study dsv4f --odir __dsv4f
```