# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for llama3 transformer device ops in the TTNN shim (issue #468).

Shapes mirror the llama3-8B prefill/decode HW profiler traces
(blackhole p100a, 2026-06-18).
"""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor
from ttsim.front.ttnn.ttnn_shim import (
    nlp_concat_heads_decode_op,
    nlp_create_qkv_heads_decode_op,
    paged_fill_cache_op,
    paged_fused_update_cache_op,
    rotary_embedding_llama_fused_qk_op,
    rotary_embedding_llama_op,
    scaled_dot_product_attention_op,
)


def _make_device():
    device = Device(device_id=0)
    device.architecture = ARCH.BLACKHOLE
    return device


def _t(device, shape, name, dtype=DataType.BFLOAT16, layout=Layout.TILE_LAYOUT):
    return Tensor(name=name, shape=shape, dtype=dtype, layout=layout, device=device)


# =========================================================================
# rotary_embedding_llama (prefill) — output = x shape
# =========================================================================

@pytest.mark.unit
def test_rotary_embedding_llama_op_shape_and_perf():
    device = _make_device()
    x = _t(device, [1, 32, 128, 128], "rope_x")
    cos = _t(device, [1, 1, 1024, 128], "rope_cos")
    sin = _t(device, [1, 1, 1024, 128], "rope_sin")
    trans = _t(device, [1, 1, 32, 32], "rope_trans")
    out = rotary_embedding_llama_op(x, cos, sin, trans)
    assert out.logical_shape().as_list() == [1, 32, 128, 128]
    ops = [op for op in device.ops.values() if op.optype == "RotaryEmbeddingLlama"]
    assert len(ops) == 1
    assert x.name in ops[0].inList and cos.name in ops[0].inList
    assert out.name in ops[0].outList
    assert ops[0].perf_stats is not None


@pytest.mark.unit
def test_rotary_embedding_llama_via_experimental():
    device = _make_device()
    x = _t(device, [1, 32, 128, 128], "rope2_x")
    cos = _t(device, [1, 1, 1024, 128], "rope2_cos")
    sin = _t(device, [1, 1, 1024, 128], "rope2_sin")
    trans = _t(device, [1, 1, 32, 32], "rope2_trans")
    out = ttnn.experimental.rotary_embedding_llama(x, cos, sin, trans, is_decode_mode=False)
    assert out.logical_shape().as_list() == [1, 32, 128, 128]
    assert any(op.optype == "RotaryEmbeddingLlama" for op in device.ops.values())


# =========================================================================
# rotary_embedding_llama_fused_qk (decode) — multi-output (q, k)
# =========================================================================

@pytest.mark.unit
def test_rotary_embedding_llama_fused_qk_op_shapes():
    device = _make_device()
    q = _t(device, [1, 1, 32, 128], "fqk_q")
    k = _t(device, [1, 1, 32, 128], "fqk_k")
    cos = _t(device, [1, 2, 32, 128], "fqk_cos")
    sin = _t(device, [1, 2, 32, 128], "fqk_sin")
    trans = _t(device, [1, 1, 64, 32], "fqk_trans")
    q_out, k_out = rotary_embedding_llama_fused_qk_op(q, k, cos, sin, trans)
    assert q_out.logical_shape().as_list() == [1, 1, 32, 128]
    assert k_out.logical_shape().as_list() == [1, 1, 32, 128]
    ops = [op for op in device.ops.values() if op.optype == "RotaryEmbeddingLlamaFusedQK"]
    assert len(ops) == 1
    assert len(ops[0].outList) == 2


@pytest.mark.unit
def test_rotary_embedding_llama_fused_qk_via_experimental():
    device = _make_device()
    q = _t(device, [1, 1, 32, 128], "fqk2_q")
    k = _t(device, [1, 1, 32, 128], "fqk2_k")
    cos = _t(device, [1, 2, 32, 128], "fqk2_cos")
    sin = _t(device, [1, 2, 32, 128], "fqk2_sin")
    trans = _t(device, [1, 1, 64, 32], "fqk2_trans")
    q_out, k_out = ttnn.experimental.rotary_embedding_llama_fused_qk(q, k, cos, sin, trans)
    assert q_out.logical_shape().as_list() == [1, 1, 32, 128]
    assert k_out.logical_shape().as_list() == [1, 1, 32, 128]


# =========================================================================
# paged_fill_cache (prefill) — output = cache shape (passthrough)
# =========================================================================

@pytest.mark.unit
def test_paged_fill_cache_op_shape():
    device = _make_device()
    cache = _t(device, [1024, 8, 32, 128], "pfc_cache")
    inp = _t(device, [1, 8, 128, 128], "pfc_in")
    page_table = _t(device, [1, 1, 1, 4], "pfc_pt", dtype=DataType.INT32,
                    layout=Layout.ROW_MAJOR_LAYOUT)
    out = paged_fill_cache_op(cache, inp, page_table=page_table, batch_idx=0)
    assert out.logical_shape().as_list() == [1024, 8, 32, 128]
    ops = [op for op in device.ops.values() if op.optype == "PagedFillCache"]
    assert len(ops) == 1
    assert ops[0].attrs["batch_idx"] == 0
    assert cache.name in ops[0].inList and inp.name in ops[0].inList


@pytest.mark.unit
def test_paged_fill_cache_via_experimental():
    device = _make_device()
    cache = _t(device, [1024, 8, 32, 128], "pfc2_cache")
    inp = _t(device, [1, 8, 128, 128], "pfc2_in")
    pt = _t(device, [1, 1, 1, 4], "pfc2_pt", dtype=DataType.INT32,
            layout=Layout.ROW_MAJOR_LAYOUT)
    out = ttnn.experimental.paged_fill_cache(cache, inp, pt, batch_idx=3)
    assert out.logical_shape().as_list() == [1024, 8, 32, 128]


# =========================================================================
# paged_fused_update_cache (decode) — multi-output (k_cache, v_cache)
# =========================================================================

@pytest.mark.unit
def test_paged_fused_update_cache_op_shapes():
    device = _make_device()
    k_cache = _t(device, [1024, 8, 32, 128], "pfu_kc")
    k = _t(device, [1, 1, 32, 128], "pfu_k")
    v_cache = _t(device, [1024, 8, 32, 128], "pfu_vc")
    v = _t(device, [1, 1, 32, 128], "pfu_v")
    upd = _t(device, [1, 1, 1, 1], "pfu_upd", dtype=DataType.INT32,
             layout=Layout.ROW_MAJOR_LAYOUT)
    pt = _t(device, [1, 1, 1, 1024], "pfu_pt", dtype=DataType.INT32,
            layout=Layout.ROW_MAJOR_LAYOUT)
    k_out, v_out = paged_fused_update_cache_op(
        k_cache, k, v_cache, v, update_idxs_tensor=upd, page_table=pt,
    )
    assert k_out.logical_shape().as_list() == [1024, 8, 32, 128]
    assert v_out.logical_shape().as_list() == [1024, 8, 32, 128]
    ops = [op for op in device.ops.values() if op.optype == "PagedFusedUpdateCache"]
    assert len(ops) == 1
    assert len(ops[0].outList) == 2


@pytest.mark.unit
def test_paged_fused_update_cache_via_experimental():
    device = _make_device()
    k_cache = _t(device, [1024, 8, 32, 128], "pfu2_kc")
    k = _t(device, [1, 1, 32, 128], "pfu2_k")
    v_cache = _t(device, [1024, 8, 32, 128], "pfu2_vc")
    v = _t(device, [1, 1, 32, 128], "pfu2_v")
    upd = _t(device, [1, 1, 1, 1], "pfu2_upd", dtype=DataType.INT32,
             layout=Layout.ROW_MAJOR_LAYOUT)
    pt = _t(device, [1, 1, 1, 1024], "pfu2_pt", dtype=DataType.INT32,
            layout=Layout.ROW_MAJOR_LAYOUT)
    k_out, v_out = ttnn.experimental.paged_fused_update_cache(
        k_cache, k, v_cache, v, update_idxs_tensor=upd, page_table=pt,
    )
    assert k_out.logical_shape().as_list() == [1024, 8, 32, 128]
    assert v_out.logical_shape().as_list() == [1024, 8, 32, 128]


# =========================================================================
# nlp_create_qkv_heads_decode (decode) — multi-output (q, k, v)
# =========================================================================

@pytest.mark.unit
def test_nlp_create_qkv_heads_decode_op_shapes():
    device = _make_device()
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    hidden = (num_heads + 2 * num_kv_heads) * head_dim  # 6144
    x = _t(device, [1, 1, 32, hidden], "ncqd_x")
    q, k, v = nlp_create_qkv_heads_decode_op(
        x, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    assert q.logical_shape().as_list() == [1, 1, num_heads, head_dim]
    assert k.logical_shape().as_list() == [1, 1, num_kv_heads, head_dim]
    assert v.logical_shape().as_list() == [1, 1, num_kv_heads, head_dim]
    ops = [op for op in device.ops.values() if op.optype == "NLPCreateQKVHeadsDecode"]
    assert len(ops) == 1
    assert len(ops[0].outList) == 3


@pytest.mark.unit
def test_nlp_create_qkv_heads_decode_via_experimental():
    device = _make_device()
    hidden = (32 + 2 * 8) * 128
    x = _t(device, [1, 1, 32, hidden], "ncqd2_x")
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        x, num_heads=32, num_kv_heads=8,
    )
    assert q.logical_shape().as_list() == [1, 1, 32, 128]
    assert k.logical_shape().as_list() == [1, 1, 8, 128]
    assert v.logical_shape().as_list() == [1, 1, 8, 128]


# =========================================================================
# nlp_concat_heads_decode (decode) — single output, head_dim folded into X
# =========================================================================

@pytest.mark.unit
def test_nlp_concat_heads_decode_op_shape():
    device = _make_device()
    x = _t(device, [1, 1, 32, 128], "nchd_x")
    out = nlp_concat_heads_decode_op(x, num_heads=32)
    assert out.logical_shape().as_list() == [1, 1, 32, 32 * 128]  # X = 4096
    ops = [op for op in device.ops.values() if op.optype == "NLPConcatHeadsDecode"]
    assert len(ops) == 1
    assert ops[0].perf_stats is not None


@pytest.mark.unit
def test_nlp_concat_heads_decode_via_experimental():
    device = _make_device()
    x = _t(device, [1, 1, 32, 128], "nchd2_x")
    out = ttnn.experimental.nlp_concat_heads_decode(x, num_heads=32)
    assert out.logical_shape().as_list() == [1, 1, 32, 4096]


# =========================================================================
# scaled_dot_product_attention (prefill + decode) — output = q shape
# =========================================================================

@pytest.mark.unit
def test_sdpa_prefill_op_shape():
    device = _make_device()
    q = _t(device, [1, 32, 128, 128], "sdpa_q")
    k = _t(device, [1, 8, 128, 128], "sdpa_k")
    v = _t(device, [1, 8, 128, 128], "sdpa_v")
    out = scaled_dot_product_attention_op(q, k, v, is_causal=True, scale=0.088)
    assert out.logical_shape().as_list() == [1, 32, 128, 128]
    ops = [op for op in device.ops.values() if op.optype == "ScaledDotProductAttention"]
    assert len(ops) == 1
    assert ops[0].attrs["is_causal"] is True


@pytest.mark.unit
def test_sdpa_prefill_via_transformer():
    device = _make_device()
    q = _t(device, [1, 32, 128, 128], "sdpa2_q")
    k = _t(device, [1, 8, 128, 128], "sdpa2_k")
    v = _t(device, [1, 8, 128, 128], "sdpa2_v")
    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.088)
    assert out.logical_shape().as_list() == [1, 32, 128, 128]


@pytest.mark.unit
def test_paged_sdpa_decode_via_transformer():
    device = _make_device()
    q = _t(device, [1, 1, 32, 128], "psd_q")
    k_cache = _t(device, [1024, 8, 32, 128], "psd_kc")
    v_cache = _t(device, [1024, 8, 32, 128], "psd_vc")
    cur_pos = _t(device, [1, 1, 1, 1], "psd_cp", dtype=DataType.INT32,
                 layout=Layout.ROW_MAJOR_LAYOUT)
    pt = _t(device, [1, 1, 1, 1024], "psd_pt", dtype=DataType.INT32,
            layout=Layout.ROW_MAJOR_LAYOUT)
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q, k_cache, v_cache, page_table_tensor=pt, cur_pos_tensor=cur_pos, scale=0.088,
    )
    assert out.logical_shape().as_list() == [1, 1, 32, 128]
    ops = [op for op in device.ops.values() if op.optype == "ScaledDotProductAttention"]
    assert len(ops) == 1


# =========================================================================
# UnaryOpType — fused activation on a binary op (MLP silu fused into the Mul,
# matching the HW BinaryNg single-op; no separate silu op emitted)
# =========================================================================

@pytest.mark.unit
def test_binary_fused_activation_silu():
    assert ttnn.UnaryOpType.SILU == "SILU"  # resolves on the shim path (mirrors ttnn.UnaryOpType)
    device = _make_device()
    a = _t(device, [1, 1, 32, 14336], "mlp_w1_out")
    b = _t(device, [1, 1, 32, 14336], "mlp_w3_out")
    out = ttnn.multiply(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    assert out.logical_shape().as_list() == [1, 1, 32, 14336]
    # ONE Mul op carrying the fused activation as an attr — no separate silu op
    muls = [op for op in device.ops.values() if op.optype == "Mul"]
    assert len(muls) == 1
    assert muls[0].attrs.get("input_tensor_a_activations") == [ttnn.UnaryOpType.SILU]
