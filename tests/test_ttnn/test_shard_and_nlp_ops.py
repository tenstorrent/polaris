#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for shard layout ops and transformer head ops (issue #394)."""

import numpy as np
import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor
from ttsim.front.ttnn.ttnn_shim import (ExecutionMode, interleaved_to_sharded, interleaved_to_sharded_op,
                                        nlp_concat_heads, nlp_concat_heads_op, nlp_create_qkv_heads,
                                        nlp_create_qkv_heads_op, reshard, reshard_op, set_execution_mode,
                                        sharded_to_interleaved, sharded_to_interleaved_op)


def _make_device():
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0
    return device


# =========================================================================
# InterleavedToSharded
# =========================================================================

@pytest.mark.unit
def test_interleaved_to_sharded_op_shape_and_perf():
    device = _make_device()
    inp = Tensor(
        name="i2s_in", shape=[1, 32, 64], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = interleaved_to_sharded_op(inp)
    assert out.logical_shape().as_list() == [1, 32, 64]
    ops = [op for op in device.ops.values() if op.optype == "InterleavedToSharded"]
    assert len(ops) == 1
    ps = ops[0].perf_stats
    assert ps is not None
    assert ps["inElems"] == 1 * 32 * 64
    assert ps["outElems"] == 1 * 32 * 64
    assert "mov" in ps["instrs"]
    assert inp.name in ops[0].inList
    assert out.name in ops[0].outList


@pytest.mark.unit
def test_interleaved_to_sharded_op_requires_device():
    class _NoDev:
        device = None
    with pytest.raises(AssertionError, match="interleaved_to_sharded_op requires"):
        interleaved_to_sharded_op(_NoDev())


@pytest.mark.unit
def test_interleaved_to_sharded_shim_delegates_to_op():
    """High-level shim delegates to _op when device is present, producing a SimOp."""
    device = _make_device()
    inp = Tensor(
        name="i2s_track", shape=[2, 16, 32], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = interleaved_to_sharded(inp)
    assert out.logical_shape().as_list() == [2, 16, 32]
    ops = [op for op in device.ops.values() if op.optype == "InterleavedToSharded"]
    assert len(ops) == 1
    assert inp.name in ops[0].inList


@pytest.mark.unit
def test_ttnn_interleaved_to_sharded_package_level():
    """ttnn.interleaved_to_sharded resolves to the shim, not the old stub."""
    device = _make_device()
    inp = Tensor(
        name="i2s_pkg", shape=[4, 64], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = ttnn.interleaved_to_sharded(inp)
    assert out.logical_shape().as_list() == [4, 64]


# =========================================================================
# ShardedToInterleaved
# =========================================================================

@pytest.mark.unit
def test_sharded_to_interleaved_op_shape_and_perf():
    device = _make_device()
    inp = Tensor(
        name="s2i_in", shape=[1, 32, 64], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = sharded_to_interleaved_op(inp)
    assert out.logical_shape().as_list() == [1, 32, 64]
    ops = [op for op in device.ops.values() if op.optype == "ShardedToInterleaved"]
    assert len(ops) == 1
    ps = ops[0].perf_stats
    assert ps is not None
    assert ps["inElems"] == 1 * 32 * 64
    assert "mov" in ps["instrs"]


@pytest.mark.unit
def test_sharded_to_interleaved_shim_delegates_to_op():
    """High-level shim delegates to _op when device is present, producing a SimOp."""
    device = _make_device()
    inp = Tensor(
        name="s2i_track", shape=[2, 16, 32], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = sharded_to_interleaved(inp)
    assert out.logical_shape().as_list() == [2, 16, 32]
    ops = [op for op in device.ops.values() if op.optype == "ShardedToInterleaved"]
    assert len(ops) == 1
    assert inp.name in ops[0].inList


@pytest.mark.unit
def test_ttnn_sharded_to_interleaved_package_level():
    device = _make_device()
    inp = Tensor(
        name="s2i_pkg", shape=[4, 64], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = ttnn.sharded_to_interleaved(inp)
    assert out.logical_shape().as_list() == [4, 64]


# =========================================================================
# Reshard
# =========================================================================

@pytest.mark.unit
def test_reshard_op_shape_and_perf():
    device = _make_device()
    inp = Tensor(
        name="rs_in", shape=[1, 32, 64], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = reshard_op(inp)
    assert out.logical_shape().as_list() == [1, 32, 64]
    ops = [op for op in device.ops.values() if op.optype == "Reshard"]
    assert len(ops) == 1
    ps = ops[0].perf_stats
    assert ps is not None
    assert ps["inElems"] == 1 * 32 * 64
    assert "mov" in ps["instrs"]


@pytest.mark.unit
def test_reshard_shim_delegates_to_op():
    """High-level shim delegates to _op when device is present, producing a SimOp."""
    device = _make_device()
    inp = Tensor(
        name="rs_track", shape=[2, 16, 32], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = reshard(inp, memory_config=None)
    assert out.logical_shape().as_list() == [2, 16, 32]
    ops = [op for op in device.ops.values() if op.optype == "Reshard"]
    assert len(ops) == 1
    assert inp.name in ops[0].inList


@pytest.mark.unit
def test_ttnn_reshard_package_level():
    device = _make_device()
    inp = Tensor(
        name="rs_pkg", shape=[4, 64], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = ttnn.reshard(inp, memory_config=None)
    assert out.logical_shape().as_list() == [4, 64]


# =========================================================================
# ConcatHeads
# =========================================================================

@pytest.mark.unit
def test_nlp_concat_heads_op_shape_and_perf():
    device = _make_device()
    inp = Tensor(
        name="nch_in", shape=[1, 8, 128, 64], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = nlp_concat_heads_op(inp)
    assert out.logical_shape().as_list() == [1, 128, 8 * 64]
    ops = [op for op in device.ops.values() if op.optype == "ConcatHeads"]
    assert len(ops) == 1
    ps = ops[0].perf_stats
    assert ps is not None
    assert ps["inElems"] == 1 * 8 * 128 * 64
    assert ps["outElems"] == 1 * 8 * 128 * 64
    assert "mov" in ps["instrs"]


@pytest.mark.unit
def test_nlp_concat_heads_shim_delegates_to_op():
    """High-level shim delegates to _op when device is present, producing a SimOp."""
    device = _make_device()
    inp = Tensor(
        name="nch_track", shape=[1, 8, 128, 64], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = nlp_concat_heads(inp)
    assert out.logical_shape().as_list() == [1, 128, 512]
    ops = [op for op in device.ops.values() if op.optype == "ConcatHeads"]
    assert len(ops) == 1
    assert inp.name in ops[0].inList


@pytest.mark.unit
def test_nlp_concat_heads_execute_roundtrip():
    """Data roundtrip without a device so the execute path computes actual values."""
    set_execution_mode(ExecutionMode.EXECUTE)
    try:
        B, H, S, D = 1, 2, 3, 4
        arr = np.arange(B * H * S * D, dtype=np.float32).reshape(B, H, S, D)
        inp = Tensor(
            name="nch_exec", shape=[B, H, S, D], dtype=DataType.FLOAT32,
            layout=Layout.ROW_MAJOR_LAYOUT, device=None, data=arr.flatten(),
        )
        out = nlp_concat_heads(inp)
        assert out.logical_shape().as_list() == [B, S, H * D]
        expected = arr.transpose(0, 2, 1, 3).reshape(B, S, H * D)
        result = np.array(out.get_data()).reshape(B, S, H * D)
        np.testing.assert_allclose(result, expected)
    finally:
        set_execution_mode(ExecutionMode.TRACK_ONLY)


@pytest.mark.unit
def test_ttnn_experimental_nlp_concat_heads():
    device = _make_device()
    inp = Tensor(
        name="exp_nch", shape=[1, 4, 32, 64], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = ttnn.experimental.nlp_concat_heads(inp)
    assert out.logical_shape().as_list() == [1, 32, 256]


# =========================================================================
# CreateQKVHeads
# =========================================================================

@pytest.mark.unit
def test_nlp_create_qkv_heads_op_shape_and_perf():
    device = _make_device()
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    fused_dim = (num_heads + 2 * num_kv_heads) * head_dim
    inp = Tensor(
        name="qkv_in", shape=[1, 128, fused_dim], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    q, k, v = nlp_create_qkv_heads_op(
        inp, num_heads=num_heads, num_kv_heads=num_kv_heads,
    )
    assert q.logical_shape().as_list() == [1, num_heads, 128, head_dim]
    assert k.logical_shape().as_list() == [1, num_kv_heads, 128, head_dim]
    assert v.logical_shape().as_list() == [1, num_kv_heads, 128, head_dim]

    ops = [op for op in device.ops.values() if op.optype == "CreateQKVHeads"]
    assert len(ops) == 1
    op = ops[0]
    assert len(op.outList) == 3
    ps = op.perf_stats
    assert ps is not None
    assert ps["inElems"] == 1 * 128 * fused_dim
    expected_out = (num_heads + 2 * num_kv_heads) * 128 * head_dim
    assert ps["outElems"] == expected_out
    assert "mov" in ps["instrs"]


@pytest.mark.unit
def test_nlp_create_qkv_heads_shim_delegates_to_op():
    """High-level shim delegates to _op when device is present, producing a SimOp."""
    device = _make_device()
    num_heads = 4
    num_kv_heads = 2
    head_dim = 32
    fused_dim = (num_heads + 2 * num_kv_heads) * head_dim
    inp = Tensor(
        name="qkv_track", shape=[1, 16, fused_dim], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    q, k, v = nlp_create_qkv_heads(
        inp, num_heads=num_heads, num_kv_heads=num_kv_heads,
    )
    assert q.logical_shape().as_list() == [1, num_heads, 16, head_dim]
    assert k.logical_shape().as_list() == [1, num_kv_heads, 16, head_dim]
    assert v.logical_shape().as_list() == [1, num_kv_heads, 16, head_dim]
    ops = [op for op in device.ops.values() if op.optype == "CreateQKVHeads"]
    assert len(ops) == 1
    assert inp.name in ops[0].inList


@pytest.mark.unit
def test_nlp_create_qkv_heads_execute_roundtrip():
    """Data roundtrip without a device so the execute path computes actual values."""
    set_execution_mode(ExecutionMode.EXECUTE)
    try:
        B, S, num_heads, num_kv_heads, head_dim = 1, 4, 2, 1, 3
        fused_dim = (num_heads + 2 * num_kv_heads) * head_dim
        arr = np.arange(B * S * fused_dim, dtype=np.float32).reshape(B, S, fused_dim)
        inp = Tensor(
            name="qkv_exec", shape=[B, S, fused_dim], dtype=DataType.FLOAT32,
            layout=Layout.ROW_MAJOR_LAYOUT, device=None, data=arr.flatten(),
        )
        q, k, v = nlp_create_qkv_heads(
            inp, num_heads=num_heads, num_kv_heads=num_kv_heads,
        )

        q_end = num_heads * head_dim
        k_end = q_end + num_kv_heads * head_dim
        expected_q = arr[:, :, :q_end].reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
        expected_k = arr[:, :, q_end:k_end].reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        expected_v = arr[:, :, k_end:].reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        np.testing.assert_allclose(
            np.array(q.get_data()).reshape(B, num_heads, S, head_dim), expected_q)
        np.testing.assert_allclose(
            np.array(k.get_data()).reshape(B, num_kv_heads, S, head_dim), expected_k)
        np.testing.assert_allclose(
            np.array(v.get_data()).reshape(B, num_kv_heads, S, head_dim), expected_v)
    finally:
        set_execution_mode(ExecutionMode.TRACK_ONLY)


@pytest.mark.unit
def test_ttnn_experimental_nlp_create_qkv_heads():
    device = _make_device()
    fused_dim = (8 + 2 * 2) * 64
    inp = Tensor(
        name="exp_qkv", shape=[1, 128, fused_dim], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        inp, num_heads=8, num_kv_heads=2,
    )
    assert q.logical_shape().as_list() == [1, 8, 128, 64]
    assert k.logical_shape().as_list() == [1, 2, 128, 64]
    assert v.logical_shape().as_list() == [1, 2, 128, 64]


# =========================================================================
# Device ops sequence tests
# =========================================================================

@pytest.mark.unit
def test_shard_ops_device_sequence():
    device = _make_device()
    inp = Tensor(
        name="seq_in", shape=[1, 32, 64], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    t1 = interleaved_to_sharded_op(inp)
    t2 = reshard_op(t1)
    _t3 = sharded_to_interleaved_op(t2)
    seq = [op.optype for op in device.ops.values()]
    assert seq == ["InterleavedToSharded", "Reshard", "ShardedToInterleaved"]


@pytest.mark.unit
def test_nlp_head_ops_device_sequence():
    device = _make_device()
    fused_dim = (4 + 2 * 2) * 32
    inp = Tensor(
        name="head_seq_in", shape=[1, 16, fused_dim], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    q, k, v = nlp_create_qkv_heads_op(inp, num_heads=4, num_kv_heads=2)
    _out = nlp_concat_heads_op(q)
    seq = [op.optype for op in device.ops.values()]
    assert seq == ["CreateQKVHeads", "ConcatHeads"]


# =========================================================================
# use_fused_qkv_op toggle (fused= parameter on utils wrappers)
# =========================================================================

@pytest.mark.unit
def test_utils_nlp_create_qkv_heads_fused_true():
    """fused=True dispatches to the single CreateQKVHeads shim op."""
    import workloads.ttnn.tt_transformers.utils as utils
    device = _make_device()
    num_heads, num_kv_heads, head_dim = 4, 2, 32
    fused_dim = (num_heads + 2 * num_kv_heads) * head_dim
    inp = Tensor(
        name="fused_t_in", shape=[1, 1, 16, fused_dim], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    q, k, v = utils.nlp_create_qkv_heads(
        inp, num_heads=num_heads, num_kv_heads=num_kv_heads, fused=True,
    )
    assert q.logical_shape().as_list() == [1, num_heads, 16, head_dim]
    assert k.logical_shape().as_list() == [1, num_kv_heads, 16, head_dim]
    assert v.logical_shape().as_list() == [1, num_kv_heads, 16, head_dim]


@pytest.mark.unit
def test_utils_nlp_create_qkv_heads_fused_false():
    """fused=False uses the decomposed reshape+permute path."""
    import workloads.ttnn.tt_transformers.utils as utils
    device = _make_device()
    num_heads, num_kv_heads, head_dim = 4, 2, 32
    fused_dim = (num_heads + 2 * num_kv_heads) * head_dim
    inp = Tensor(
        name="decomp_t_in", shape=[1, 1, 16, fused_dim], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    q, k, v = utils.nlp_create_qkv_heads(
        inp, num_heads=num_heads, num_kv_heads=num_kv_heads, fused=False,
    )
    assert q.logical_shape().as_list() == [1, num_heads, 16, head_dim]
    assert k.logical_shape().as_list() == [1, num_kv_heads, 16, head_dim]
    assert v.logical_shape().as_list() == [1, num_kv_heads, 16, head_dim]
    permute_ops = [op for op in device.ops.values() if op.optype == "Permute"]
    assert len(permute_ops) >= 3, "Decomposed path should produce Permute ops"


@pytest.mark.unit
def test_utils_nlp_concat_heads_fused_true():
    import workloads.ttnn.tt_transformers.utils as utils
    device = _make_device()
    inp = Tensor(
        name="cat_fused_in", shape=[1, 4, 16, 32], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = utils.nlp_concat_heads(inp, fused=True)
    assert out.logical_shape().as_list() == [1, 16, 128]


@pytest.mark.unit
def test_utils_nlp_concat_heads_fused_false():
    import workloads.ttnn.tt_transformers.utils as utils
    device = _make_device()
    inp = Tensor(
        name="cat_decomp_in", shape=[1, 4, 16, 32], dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT, device=device,
    )
    out = utils.nlp_concat_heads(inp, fused=False)
    assert out.logical_shape().as_list() == [1, 16, 128]
    permute_ops = [op for op in device.ops.values() if op.optype == "Permute"]
    assert len(permute_ops) >= 1, "Decomposed path should produce Permute ops"
