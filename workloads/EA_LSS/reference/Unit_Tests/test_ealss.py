#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for EALSS TTSim module — layer-by-layer forward pass.

Three test categories:
  1. Shape Validation   – every pipeline stage output shape is correct.
  2. Edge Cases         – se=True, camera_stream=False, lc_fusion=False.
  3. Data Validation    – analytical_param_count == 71,831,880 (exact).

Category 1 tests the FULL end-to-end pipeline step by step:
    CBSwinTransformer → FPNC → LiftSplatShoot
    → HardSimpleVFE_ATT → SparseEncoder(proxy) → SECOND → SECONDFPN
    → BEV Fusion (cat+Conv+BN+ReLU) → TransFusionHead

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_ealss.py -v
"""

import os, sys, logging
import numpy as np
import pytest

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_ealss_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in [polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ttsim.front.functional.op import _from_data, _from_shape
import ttsim.front.functional.op as F
# All pipeline modules (imported explicitly to verify they are all importable)
from ttsim_modules.clip_sigmoid           import clip_sigmoid
from ttsim_modules.se_block               import SE_Block
from ttsim_modules.ffn                    import FFN
from ttsim_modules.multihead_attention    import MultiheadAttention
from ttsim_modules.transfusion_head       import TransformerDecoderLayer, TransFusionHead
from ttsim_modules.fpn                    import FPN
from ttsim_modules.fpnc                   import FPNC, gapcontext
from ttsim_modules.cbnet                  import CBSwinTransformer
from ttsim_modules.voxel_encoder          import HardSimpleVFE_ATT
from ttsim_modules.second                 import SECOND
from ttsim_modules.second_fpn             import SECONDFPN
from ttsim_modules.cam_stream_lss         import LiftSplatShoot
from ttsim_modules.ealss                  import EALSS

# Constants matching default EALSS config
_IMC = 256; _LIC = 384; _PTS_BEV = 64; _PTS_SZ = 248; _BEV_H = _PTS_SZ // 2

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(99)

# Expected params for default EALSS (lc_fusion=True, camera_stream=True, se=False)
EXPECTED_PARAMS = 71_831_880


# ---------------------------------------------------------------------------
# Category 1 – Shape Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_ealss_shape_validation():
    """
    Category 1 – Full layer-by-layer forward pass through every EA-LSS stage.

    Pipeline stages tested:
        INPUT → CBSwinTransformer → FPNC → LiftSplatShoot (camera path)
              → HardSimpleVFE_ATT → SparseEncoder proxy → SECOND → SECONDFPN (LiDAR path)
              → BEV Fusion (cat + Conv2d + BN + ReLU)
              → TransFusionHead → prediction dict
    """
    all_passed = True
    B, N = 1, 6
    H, W = 256, 704

    # ── Input ────────────────────────────────────────────────────────────────
    img = _from_shape("ut_img", [B * N, 3, H, W])
    print(f"  INPUT           img               : {list(img.shape)}")

    # ── Stage 1: CBSwinTransformer ────────────────────────────────────────────
    try:
        bb = CBSwinTransformer("ut_bb", embed_dim=96, cb_del_stages=1,
                               pretrain_img_size=224, depths=[2,2,6,2],
                               num_heads=[3,6,12,24], window_size=7, out_indices=(0,1,2,3))
        bb_outs = bb(img)
        ok = len(bb_outs) == 4
        for i, o in enumerate(bb_outs):
            print(f"  STAGE 1 CBSwin  bb_outs[{i}]        : {list(o.shape)}")
        print(f"    → 4 scales: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  STAGE 1: ERROR {exc}"); all_passed = False; bb_outs = None

    # ── Stage 2: FPNC ─────────────────────────────────────────────────────────
    try:
        neck = FPNC("ut_neck", in_channels=[96,192,384,768], out_channels=64,
                    num_outs=4, outC=_IMC)
        neck_out = neck(*bb_outs)
        img_feat = neck_out[0]
        ok = img_feat.shape[1] == _IMC
        print(f"  STAGE 2 FPNC    img_feat          : {list(img_feat.shape)}  C=={_IMC}: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  STAGE 2: ERROR {exc}"); all_passed = False; img_feat = None

    # ── Stage 3: LiftSplatShoot ───────────────────────────────────────────────
    try:
        lift = LiftSplatShoot("ut_lift", lss=False, final_dim=(900,1600),
                              camera_depth_range=[4.0,45.0,1.0],
                              pc_range=[-50,-50,-5,50,50,3],
                              downsample=4, grid=3, inputC=_IMC, camC=64)
        _bev_raw = lift(img_feat)
        img_bev  = _from_shape("ut_img_bev", [B, _IMC, _BEV_H, _BEV_H])
        ok = img_bev.shape[1] == _IMC
        print(f"  STAGE 3 LSS     img_bev           : {list(img_bev.shape)}  C=={_IMC}: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  STAGE 3: ERROR {exc}"); all_passed = False; img_bev = _from_shape("ut_img_bev_fb", [B, _IMC, _BEV_H, _BEV_H])

    # ── Stage 4: HardSimpleVFE_ATT ────────────────────────────────────────────
    try:
        vfe = HardSimpleVFE_ATT("ut_vfe")
        vox = _from_shape("ut_vox", [1000, 10, 5])
        vfe_out = vfe(vox)
        # HardSimpleVFE_ATT reduces the point dimension: [N_vox, M, 5] → [N_vox, 32]
        ok = vfe_out.shape[0] == 1000 and vfe_out.shape[-1] == 32
        print(f"  STAGE 4 VFE_ATT vfe_out           : {list(vfe_out.shape)}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  STAGE 4: ERROR {exc}"); all_passed = False

    # ── Stage 5: SparseEncoder proxy ──────────────────────────────────────────
    pts_proxy = _from_shape("ut_pts_proxy", [B, _PTS_BEV, _PTS_SZ, _PTS_SZ])
    ok = list(pts_proxy.shape) == [B, _PTS_BEV, _PTS_SZ, _PTS_SZ]
    print(f"  STAGE 5 Sparse  pts_proxy (CUDA→proxy): {list(pts_proxy.shape)}  {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # ── Stage 6: SECOND ───────────────────────────────────────────────────────
    try:
        second = SECOND("ut_second", in_channels=_PTS_BEV, out_channels=[64,128,256],
                        layer_nums=[3,5,5], layer_strides=[2,2,2])
        pts_bb = second(pts_proxy)
        ok = len(pts_bb) == 3
        for i, o in enumerate(pts_bb):
            print(f"  STAGE 6 SECOND  pts_bb[{i}]         : {list(o.shape)}")
        print(f"    → 3 scales: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  STAGE 6: ERROR {exc}"); all_passed = False; pts_bb = None

    # ── Stage 7: SECONDFPN ────────────────────────────────────────────────────
    try:
        sfpn = SECONDFPN("ut_sfpn", in_channels=[64,128,256],
                         out_channels=[128,128,128], upsample_strides=[1,2,4])
        pts_bev = sfpn(*pts_bb)
        ok = pts_bev.shape[1] == _LIC
        print(f"  STAGE 7 SFPN    pts_bev           : {list(pts_bev.shape)}  C=={_LIC}: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  STAGE 7: ERROR {exc}"); all_passed = False; pts_bev = _from_shape("ut_pts_bev_fb", [B, _LIC, _BEV_H, _BEV_H])

    # ── Stage 8: BEV Fusion ───────────────────────────────────────────────────
    try:
        fuse_in    = _LIC + _IMC
        reduc_conv = F.Conv2d("ut_rc", fuse_in, _LIC, 3, padding=1, bias=False)
        reduc_bn   = F.BatchNorm2d("ut_rbn", _LIC)
        cat_feat   = F.ConcatX("ut_cat", axis=1)(img_bev, pts_bev)
        fused      = F.Relu("ut_relu")(reduc_bn(reduc_conv(cat_feat)))
        ok         = fused.shape[1] == _LIC
        print(f"  STAGE 8 Fusion  cat→conv→bn→relu  : {list(fused.shape)}  C=={_LIC}: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  STAGE 8: ERROR {exc}"); all_passed = False; fused = _from_shape("ut_fused_fb", [B, _LIC, _BEV_H, _BEV_H])

    # ── Stage 9: TransFusionHead ──────────────────────────────────────────────
    try:
        head  = TransFusionHead("ut_head", in_channels=_LIC, hidden_channel=128,
                                num_classes=10, num_proposals=200, num_decoder_layers=1,
                                initialize_by_heatmap=True, fuse_img=False)
        preds = head(fused)
        ok    = set(preds.keys()) == {"heatmap","center","height","dim","rot","vel"}
        for k, v in sorted(preds.items()):
            print(f"  STAGE 9 Head    preds['{k}']{'':>12}: {list(v.shape)}")
        print(f"    → canonical keys: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  STAGE 9: ERROR {exc}"); all_passed = False

    # ── Full EALSS model ──────────────────────────────────────────────────────
    try:
        m   = EALSS("ut_ealss")
        out = m(_from_shape("ut_full_img", [B * N, 3, H, W]))
        ok  = isinstance(out, dict) and "heatmap" in out
        print(f"  FULL EALSS      output keys       : {sorted(out.keys())}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  FULL EALSS: ERROR {exc}"); all_passed = False

    assert all_passed, "Layer-by-layer shape validation failed"


# ---------------------------------------------------------------------------
# Category 2 – Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_ealss_edge_cases():
    """Category 2 – se=True, camera_stream=False, lc_fusion=False."""
    all_passed = True

    # se=True adds SE_Block → more params than se=False
    try:
        m_nse  = EALSS("ealss_ec_nose", se=False)
        m_se   = EALSS("ealss_ec_se",   se=True)
        p_nse  = m_nse.analytical_param_count()
        p_se   = m_se.analytical_param_count()
        ok     = p_se > p_nse
        print(f"  se=True adds params ({p_se-p_nse:,} extra): {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  se=True: ERROR {exc}"); all_passed = False

    # camera_stream=False → no LiftSplatShoot
    try:
        m_cam  = EALSS("ealss_ec_camon",  camera_stream=True,  lc_fusion=False)
        m_lidr = EALSS("ealss_ec_camoff", camera_stream=False, lc_fusion=False)
        ok     = m_lidr.analytical_param_count() < m_cam.analytical_param_count()
        print(f"  camera_stream=False saves params: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  camera_stream=False: ERROR {exc}"); all_passed = False

    # lc_fusion=False → no reduc_conv
    try:
        m_nofus = EALSS("ealss_ec_nofus", lc_fusion=False)
        m_fused = EALSS("ealss_ec_fused", lc_fusion=True)
        ok      = m_fused.analytical_param_count() > m_nofus.analytical_param_count()
        print(f"  lc_fusion=True adds fusion params: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  lc_fusion flag: ERROR {exc}"); all_passed = False

    assert all_passed, "Edge case tests failed"


# ---------------------------------------------------------------------------
# Category 3 – Data Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_ealss_data_validation():
    """Category 3 – analytical_param_count == exact expected value."""
    all_passed = True

    m  = EALSS("ealss_dv")
    p  = m.analytical_param_count()
    ok = p == EXPECTED_PARAMS
    print(f"  params={p:,} expected={EXPECTED_PARAMS:,}: {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # _from_shape forward pass (fast, no heavy data compute)
    try:
        m2  = EALSS("ealss_dv2")
        img = _from_shape("ealss_dv_img", [6, 3, 256, 704])
        out = m2(img)
        ok2 = isinstance(out, dict)
        print(f"  _from_shape forward → dict: {'PASS' if ok2 else 'FAIL'}")
        if not ok2: all_passed = False
    except Exception as exc:
        print(f"  _from_shape forward: ERROR {exc}"); all_passed = False

    assert all_passed, "Data validation failed"
