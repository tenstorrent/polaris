# GLM-5.3 — Polaris TTSIM Workload

Perf simulation workload for **zai-org/GLM-5.3** (753B total / ~40B active,
DeepSeek-style MLA + DSA sparse attention, sparse MoE), built on the TTSIM
frontend. Shares `RMSNorm` / `MoEExpert` with the Qwen3.5 workload
(`workloads/qwen3_5/qwen35_model.py`).

## Architecture (pinned from HF `config.json`, `model_type: glm_moe_dsa`)

| Component | Value |
|---|---|
| Layers | 78: first 3 dense FFN (intermediate 12288), remaining 75 sparse MoE |
| Hidden / vocab | 6144 / 154,880 (untied embedding / LM head) |
| Attention | MLA: 64 Q-heads, q_lora_rank 2048, kv_lora_rank 512, qk_nope 192 + qk_rope 64, v_head_dim 256, rope_theta 8e6 |
| DSA indexer | 32 index-heads × 128 dims, top-2048 tokens/query; `full` at layers 0-3 + every 4th (22 layers), `shared` elsewhere (56) |
| MoE | 256 routed + 1 shared expert, top-8, expert width 2048, sigmoid scoring + `noaux_tc` bias, norm_topk_prob, routed_scaling 2.5 |
| MTP | 1 nextn layer (speculative decoding) — **not simulated** |
| Quant | checkpoint ships FP8 (e4m3, block 128×128); sim dtypes come from the wl2arch mapping |

## Files

- `glm53_model.py` — TTSIM model (`Glm53Moe` entry class + `MlaDsaAttention`,
  `GlmSparseMoE`, `DenseFFN`, `Glm53Layer`; `RMSNorm`/`MoEExpert` imported from
  the Qwen3.5 workload)
- `test_model.py` — standalone smoke test (incl. the DSA sparse-window path)
- `tests/test_workloads/test_glm53.py` (repo root) — pytest unit tests

## Usage

```bash
python polaris.py \
  --archspec config/all_archs.yaml \
  --wlspec  config/all_workloads.yaml \
  --wlmapspec config/wl2archmapping.yaml \
  --filterwlg TTSIM --filterwl glm5_3 \
  --filterarch Q1_A1 \
  --study glm53 --odir __glm53
```

Registered instances: `glm53_decode` (bs=1, L=1), `glm53_prefill_2k` (bs=1,
L=2048), `glm53_tiny` (reduced CI config).

## Op decomposition

- **MLA**: q path `x → q_a(2048) → RMSNorm → q_b(64×256)`; KV path
  `x → kv_a(512+64) → RMSNorm(latent) → kv_b(64×(192+256))`; scores split into
  nope (192-dim) and rope (64-dim, shared across heads) matmuls; context over
  v_head_dim 256; `o_proj` 16384→6144.
- **DSA indexer** (full layers only): `wq_idx`/`wk_idx` projections (32×128),
  head-averaged score matmul, `topk` selection. Attention matmuls run against
  `min(L, index_topk)` KV positions.
- **MoE**: sigmoid router + `noaux_tc` correction bias → top-8 → normalize →
  ×2.5 scaling; 8 active expert FFNs + shared expert.
- **Proxy blocks** (×repeat_count): dense+full-indexer ×3, MoE+full-indexer ×19,
  MoE+shared-indexer ×56.

## Modeling abstractions

1. **Active-expert MoE**: top-8 + shared experts applied to routed activations;
   dispatch/combine data movement not modeled.
2. **DSA top-k gather** is data-dependent; approximated by a static slice to the
   first `min(L, index_topk)` KV positions when L > index_topk — cost-equivalent
   for the attention matmuls.
3. **Shared-indexer layers** perform no indexer compute (indices reused);
   indexer weights still count toward parameters.
4. **RoPE** approximated by elementwise cos/sin mixing (negligible cost).
5. **MTP layer** excluded from the main inference loop (hence 747B modeled vs
   753.3B checkpoint total, which includes the MTP layer).

## Validation

- Analytical parameter count: **747,106,078,464** (~0.8% under the published
  753.3B; delta = excluded MTP layer).
- Graph builds for decode, prefill 512 (dense window), prefill 4096 with
  topk 2048 (sparse window), and a tiny CI config.
- `polaris.py` end-to-end projection runs clean.