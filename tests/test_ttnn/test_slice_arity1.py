# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ttnn.slice arity-1 shape/arity contract (issue #478).

ttnn.slice is an arity-1 op: the bounds travel as an 'output_shape' attr rather than as
starts/ends tensor inputs (the ONNX form), mirroring the hardware SliceDeviceOperation. The
decode sampling path relies on this for op-alignment, and the LUT key arity is derived from the
operand count — so an extra operand or a wrong inferred shape would silently re-key the op.
Covers both the shim emission (ttsim/front/ttnn/op.py:as_pp) and the descriptor's arity-1
branch (ttsim/ops/desc/tensor.py:slice_sinf).

`slice_spec_out_shape` replaced a dummy `np.empty(in_shape)` + `dummy[spec].shape`; the
equivalence test below pins it to numpy basic-indexing semantics so a future refactor cannot
drift.
"""
import numpy as np
import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import close_device, open_device, set_default_device
from ttsim.front.ttnn.op import slice_spec_out_shape


@pytest.fixture()
def device():
    dev = open_device()
    set_default_device(dev)
    yield dev
    close_device(dev)


# (in_shape, slice_spec) — representative specs incl. the decode sampling chunk slice,
# integer indexing, Ellipsis, partial specs and negative bounds/steps.
SLICE_SPECS = [
    ([1, 1, 32, 128256], (slice(None), slice(None), slice(None), slice(0, 16032))),
    ([1, 1, 32, 128256], (slice(None), slice(None), slice(None), slice(112224, 128256))),
    ([1, 1, 32, 64128], (slice(None), slice(None), slice(None), slice(8016, 16032))),
    ([4, 6], (slice(1, 3), slice(2, 5))),
    ([4, 6], (2, slice(None))),                       # integer index drops the dim
    ([3, 5, 7, 9], (Ellipsis, slice(0, 4))),          # Ellipsis expansion
    ([3, 5, 7, 9], (slice(1, 3),)),                   # partial spec: trailing dims kept whole
    ([10], (slice(-4, -1),)),                         # negative bounds
    ([10], (slice(None, None, 2),)),                  # step
    ([10], (slice(None, None, -1),)),                 # negative step
    ([2, 4, 8], (slice(0, 2), 1, slice(None, None, 2))),
]


@pytest.mark.unit
@pytest.mark.parametrize("in_shape,spec", SLICE_SPECS)
def test_slice_spec_out_shape_matches_numpy(in_shape, spec):
    """Analytical shape inference is exactly numpy basic indexing, with no allocation."""
    expected = list(np.empty(in_shape, dtype=np.int8)[spec].shape)
    assert slice_spec_out_shape(in_shape, spec) == expected


@pytest.mark.unit
@pytest.mark.parametrize("spec", [Ellipsis, slice(2, 8), 3])
def test_slice_spec_out_shape_accepts_bare_spec(spec):
    """A non-tuple spec is treated as a 1-tuple, as numpy does."""
    in_shape = [10]
    expected = list(np.empty(in_shape, dtype=np.int8)[spec].shape)
    assert slice_spec_out_shape(in_shape, spec) == expected


@pytest.mark.unit
def test_slice_spec_out_shape_rejects_overindexing():
    with pytest.raises(AssertionError, match="over-indexes"):
        slice_spec_out_shape([4, 6], (slice(None), slice(None), slice(None)))
    with pytest.raises(AssertionError, match="at most one Ellipsis"):
        slice_spec_out_shape([4, 6], (Ellipsis, Ellipsis))


@pytest.mark.unit
@pytest.mark.parametrize("in_shape,spec", SLICE_SPECS)
def test_shim_emits_single_arity1_slice(device, in_shape, spec):
    """The shim emits exactly one Slice op, with one operand and the inferred output shape."""
    x = ttnn.zeros(in_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.slice(x, slice=spec)

    g = device.get_graph()
    slices = [op for op in g._ops.values() if op.optype == 'Slice']
    assert len(slices) == 1, 'ttnn.slice must emit exactly one Slice op'
    op = slices[0]
    # arity-1: the LUT key arity is derived from the operand count, so an extra
    # starts/ends operand here would re-key the op away from the HW SliceDeviceOperation.
    assert len(op.inList) == 1, f'Slice must be arity-1, got {len(op.inList)} operands'
    assert g._tensors[op.inList[0]].name == x.name
    assert op.attrs['output_shape'] == list(np.empty(in_shape, dtype=np.int8)[spec].shape)
    assert list(out.shape) == op.attrs['output_shape']
    assert not any(o.optype == 'Split' for o in g._ops.values()), 'slice must not emit a Split'


@pytest.mark.unit
def test_shim_slice_chunks_are_disjoint_and_cover(device):
    """The decode sampling chunk pattern: n slices tiling the vocab dim, one Slice op each."""
    vocab, n = 128256, 8
    step = vocab // n
    x = ttnn.zeros([1, 1, 32, vocab], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    chunks = [
        ttnn.slice(x, slice=(slice(None), slice(None), slice(None), slice(i * step, (i + 1) * step)))
        for i in range(n)
    ]
    g = device.get_graph()
    slices = [op for op in g._ops.values() if op.optype == 'Slice']
    assert len(slices) == n
    assert all(len(op.inList) == 1 for op in slices)
    assert all(list(c.shape) == [1, 1, 32, step] for c in chunks)
