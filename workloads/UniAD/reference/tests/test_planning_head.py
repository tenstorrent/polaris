#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for UniAD PlanningHead."""

import pytest
import numpy as np
import ttsim.front.functional.op as F
from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.planning_head import (
    PlanningHead,
)

_SMALL_CFG = dict(
    embed_dims=32,
    planning_steps=4,
    num_dec_layers=2,
    num_heads=4,
    ffn_dim=64,
    bs=1,
    bev_h=4,
    bev_w=4,
)


def _make_motion_out(cfg):
    E = cfg["embed_dims"]
    bs = cfg["bs"]
    mf = F._from_shape("motion_feats", [bs, 8, E])
    return {"motion_feats": mf}


@pytest.mark.unit
def test_planning_head_construction():
    head = PlanningHead("plan", _SMALL_CFG)
    assert head is not None


@pytest.mark.unit
def test_planning_head_output_key():
    cfg = _SMALL_CFG
    head = PlanningHead("plan", cfg)
    E = cfg["embed_dims"]
    bs = cfg["bs"]
    bev = F._from_shape("bev", [bs, 16, E])
    motion_out = _make_motion_out(cfg)
    out = head(bev, motion_out)
    assert "plan_traj" in out, "missing key: plan_traj"


@pytest.mark.unit
def test_planning_head_traj_shape():
    """plan_traj should be [bs, planning_steps, 2]."""
    cfg = _SMALL_CFG
    head = PlanningHead("plan2", cfg)
    E = cfg["embed_dims"]
    bs = cfg["bs"]
    T = cfg["planning_steps"]
    bev = F._from_shape("bev2", [bs, 16, E])
    motion_out = _make_motion_out(cfg)
    out = head(bev, motion_out)
    assert out["plan_traj"].shape == [
        bs,
        T,
        2,
    ], f"unexpected plan_traj shape: {out['plan_traj'].shape}"


@pytest.mark.unit
def test_planning_head_with_occ():
    """Planning head should accept occ_out without error."""
    cfg = _SMALL_CFG
    head = PlanningHead("plan3", cfg)
    E = cfg["embed_dims"]
    bs = cfg["bs"]
    bev = F._from_shape("bev3", [bs, 16, E])
    motion_out = _make_motion_out(cfg)
    occ_out = {"occ_pred": F._from_shape("occ", [bs, 2, 8, 8])}
    out = head(bev, motion_out, occ_out)
    assert "plan_traj" in out
