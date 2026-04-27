#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for UniAD BEVFormer track detection head."""

import pytest
import numpy as np
import ttsim.front.functional.op as F
from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.bevformer_head import (
    BEVFormerTrackHead,
)

_SMALL_CFG = dict(
    embed_dims=32,
    num_query=10,
    num_classes=4,
    code_size=10,
    num_dec_layers=2,
    num_heads=4,
    ffn_dim=64,
    num_cls_fcs=2,
    num_reg_fcs=2,
    bs=1,
    bev_h=4,
    bev_w=4,
)


@pytest.mark.unit
def test_track_head_construction():
    head = BEVFormerTrackHead("track", _SMALL_CFG)
    assert head is not None


@pytest.mark.unit
def test_track_head_output_keys():
    """Track head should return a dict with expected keys."""
    cfg = _SMALL_CFG
    head = BEVFormerTrackHead("track", cfg)
    E = cfg["embed_dims"]
    bev = F._from_shape("bev", [cfg["bs"], 16, E])
    out = head(bev)
    assert "query_feats" in out, "missing key: query_feats"
    assert "cls_scores" in out, "missing key: cls_scores"
    assert "bbox_preds" in out, "missing key: bbox_preds"


@pytest.mark.unit
def test_track_head_output_shapes():
    """Track head output shapes should be correct."""
    cfg = _SMALL_CFG
    head = BEVFormerTrackHead("track2", cfg)
    E = cfg["embed_dims"]
    bs = cfg["bs"]
    nq = cfg["num_query"]
    nc = cfg["num_classes"]
    cs = cfg["code_size"]

    bev = F._from_shape("bev2", [bs, 16, E])
    out = head(bev)

    assert out["query_feats"].shape == [bs, nq, E]
    assert out["cls_scores"].shape == [bs, nq, nc]
    assert out["bbox_preds"].shape == [bs, nq, cs]
