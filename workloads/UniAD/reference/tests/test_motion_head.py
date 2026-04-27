#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for UniAD MotionHead."""

import pytest
import numpy as np
import ttsim.front.functional.op as F
from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.motion_head import (
    MotionHead,
)

_SMALL_CFG = dict(
    embed_dims=32,
    num_query=10,
    predict_steps=4,
    num_anchor=2,
    num_dec_layers=2,
    num_heads=4,
    ffn_dim=64,
    bs=1,
    bev_h=4,
    bev_w=4,
)


def _make_track_out(cfg):
    E = cfg["embed_dims"]
    bs = cfg["bs"]
    nq = cfg["num_query"]
    qf = F._from_shape("query_feats", [bs, nq, E])
    return {"query_feats": qf}


@pytest.mark.unit
def test_motion_head_construction():
    head = MotionHead("motion", _SMALL_CFG)
    assert head is not None


@pytest.mark.unit
def test_motion_head_output_keys():
    cfg = _SMALL_CFG
    head = MotionHead("motion", cfg)
    E = cfg["embed_dims"]
    bev = F._from_shape("bev", [cfg["bs"], 16, E])
    track_out = _make_track_out(cfg)
    out = head(bev, track_out)
    assert "traj_preds" in out
    assert "motion_feats" in out


@pytest.mark.unit
def test_motion_head_traj_shape():
    cfg = _SMALL_CFG
    head = MotionHead("motion2", cfg)
    E = cfg["embed_dims"]
    bs = cfg["bs"]
    nq = cfg["num_query"]
    K = cfg["num_anchor"]
    T = cfg["predict_steps"]
    bev = F._from_shape("bev2", [bs, 16, E])
    track_out = _make_track_out(cfg)
    out = head(bev, track_out)
    assert out["traj_preds"].shape == [
        bs,
        nq,
        K,
        T,
        2,
    ], f"unexpected traj_preds shape: {out['traj_preds'].shape}"
