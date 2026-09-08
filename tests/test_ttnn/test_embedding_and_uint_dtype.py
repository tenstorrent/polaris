# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ttnn.embedding op-code/operand-order + uint dtype LUT-key reader (issue #478).

- ttnn.embedding emits optype 'Embedding' (canonicalizes to 'embedding', matching the HW
  EmbeddingsDeviceOperation) with tokens-first operand order (input_0=tokens, input_1=weight),
  NOT a generic ONNX Gather (data-first). See ttsim/ops/desc/tensor.py:embedding_sinf.
- tensor_datatype must report UINT32/UINT16 (not INT32) — "INT32" is a substring of "UINT32",
  which previously mis-keyed every uint index tensor (embedding tokens, PlusOne, ManualSeed).
"""
import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import close_device, open_device, set_default_device
from tools.profiling.shape_canonical import tensor_datatype


@pytest.fixture()
def device():
    dev = open_device()
    set_default_device(dev)
    yield dev
    close_device(dev)


@pytest.mark.unit
def test_embedding_optype_and_operand_order(device):
    tokens = ttnn.zeros([1, 32], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    weight = ttnn.zeros([128256, 4096], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.embedding(tokens, weight)
    g = device.get_graph()
    emb = [op for op in g._ops.values() if op.optype == 'Embedding']
    assert len(emb) == 1, 'ttnn.embedding must emit exactly one Embedding op'
    op = emb[0]
    # tokens-first operand order (matches HW EmbeddingsDeviceOperation)
    assert g._tensors[op.inList[0]].name == tokens.name, 'input_0 must be the tokens (indices)'
    assert g._tensors[op.inList[1]].name == weight.name, 'input_1 must be the weight'
    # output = tokens.shape + [embed_dim]
    assert list(out.shape)[-1] == 4096
    assert not any(o.optype == 'Gather' for o in g._ops.values()), 'embedding must not emit a Gather'


@pytest.mark.unit
@pytest.mark.parametrize('dt,expected', [
    (ttnn.uint32, 'UINT32'),
    (ttnn.int32, 'INT32'),
    (ttnn.bfloat16, 'BFLOAT16'),
])
def test_tensor_datatype_uint_not_misread_as_int(device, dt, expected):
    t = ttnn.zeros([1, 32], dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    assert tensor_datatype(t, None) == expected
