#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for TTSim SoftmaxCrossEntropyLoss shape/perf inference.
"""

import numpy as np
import pytest

from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.math import softmax_cross_entropy_loss_sinf

SEED = 42

def _seed():
    np.random.seed(SEED)

def make_sim_tensor(data, name="t"):
    return SimTensor(
        {
            "name": name,
            "shape": list(data.shape),
            "data": data.copy(),
            "dtype": np.dtype(np.float32),
        }
    )

class _FakeOp:
    def __init__(self, **attrs):
        self.attrs = attrs
        self.precision = np.dtype(np.float32)
        self.perf_stats = None

@pytest.mark.unit
@pytest.mark.opunit
def test_softmax_ce_reduction_none_shapes_and_perf():
    _seed()
    B, C = 4, 7
    scores_data = np.random.randn(B, C).astype(np.float32)
    # Use probabilities / soft labels; shape must match scores
    labels_data = np.random.rand(B, C).astype(np.float32)
    labels_data /= labels_data.sum(axis=-1, keepdims=True)

    scores = make_sim_tensor(scores_data, "sce_scores")
    labels = make_sim_tensor(labels_data, "sce_labels")
    loss_out = make_sim_tensor(np.zeros_like(labels_data), "sce_loss")

    op = _FakeOp(reduction="none")

    softmax_cross_entropy_loss_sinf([scores, labels], [loss_out], op)

    assert list(loss_out.shape) == [B, C]
    assert loss_out.dtype == scores.dtype

    assert op.perf_stats is not None
    for key in ("inElems", "outElems", "inBytes", "outBytes"):
        assert key in op.perf_stats
        assert op.perf_stats[key] >= 0

@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_softmax_ce_scalar_reduction(reduction):
    _seed()
    B, C = 2, 5
    scores_data = np.random.randn(B, C).astype(np.float32)
    labels_data = np.random.rand(B, C).astype(np.float32)
    labels_data /= labels_data.sum(axis=-1, keepdims=True)

    scores = make_sim_tensor(scores_data, f"sce_scores_{reduction}")
    labels = make_sim_tensor(labels_data, f"sce_labels_{reduction}")
    loss_out = make_sim_tensor(np.array(0.0, dtype=np.float32), f"sce_loss_{reduction}")

    op = _FakeOp(reduction=reduction)

    softmax_cross_entropy_loss_sinf([scores, labels], [loss_out], op)

    assert loss_out.shape == []
    assert loss_out.dtype == scores.dtype

@pytest.mark.unit
@pytest.mark.opunit
def test_softmax_ce_optional_log_prob_output_shape():
    _seed()
    B, C = 3, 6
    scores_data = np.random.randn(B, C).astype(np.float32)
    labels_data = np.random.rand(B, C).astype(np.float32)
    labels_data /= labels_data.sum(axis=-1, keepdims=True)

    scores = make_sim_tensor(scores_data, "sce_scores_logp")
    labels = make_sim_tensor(labels_data, "sce_labels_logp")
    loss_out = make_sim_tensor(np.array(0.0, dtype=np.float32), "sce_loss_logp")
    log_prob_out = make_sim_tensor(np.zeros_like(scores_data), "sce_log_prob")

    op = _FakeOp(reduction="mean")

    softmax_cross_entropy_loss_sinf(
        [scores, labels],
        [loss_out, log_prob_out],
        op,
    )

    assert list(log_prob_out.shape) == [B, C]
    assert log_prob_out.dtype == scores.dtype