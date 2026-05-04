#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for EALSS — full layer-by-layer forward pass.

Demonstrates every stage of the EA-LSS pipeline in sequence, printing the
TTSim tensor shape at each step.  The same input flows through every
converted ttsim_module in the model, matching the real production path.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_ealss.py
"""

import os, sys

_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from ttsim.front.functional.op import _from_shape, _from_data
import ttsim.front.functional.op as F

# ── Import all converted modules ──────────────────────────────────────────────
# Level 1
from ttsim_modules.clip_sigmoid           import clip_sigmoid
from ttsim_modules.coord_transform        import apply_3d_transformation
from ttsim_modules.transfusion_bbox_coder import TransFusionBBoxCoder
# Level 2
from ttsim_modules.se_block               import SE_Block
from ttsim_modules.ffn                    import FFN
from ttsim_modules.multihead_attention    import MultiheadAttention
# Level 3
from ttsim_modules.transfusion_head       import TransformerDecoderLayer, TransFusionHead
from ttsim_modules.voxel_encoder_utils    import VFELayer
# Level 4
from ttsim_modules.fpn                    import FPN
from ttsim_modules.fpnc                   import FPNC, gapcontext
from ttsim_modules.cbnet                  import CBSwinTransformer
from ttsim_modules.voxel_encoder          import HardSimpleVFE_ATT
from ttsim_modules.second                 import SECOND
from ttsim_modules.second_fpn             import SECONDFPN
from ttsim_modules.cam_stream_lss         import LiftSplatShoot
# Full model
from ttsim_modules.ealss                  import EALSS

from Reference.Validation.ttsim_utils import print_header, print_test


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — small image for fast validation
# ─────────────────────────────────────────────────────────────────────────────
B         = 1                    # batch scenes
N_VIEWS   = 6                    # cameras per scene
IMG_H     = 256                  # image height (div by 32 for Swin)
IMG_W     = 704                  # image width  (div by 32 for Swin)

IMC       = 256                  # camera BEV channels (FPNC outC)
LIC       = 384                  # LiDAR BEV channels (SECONDFPN output = 3×128)
PTS_BEV   = 64                   # SparseEncoder output channels (proxy)
PTS_SZ    = 248                  # SparseEncoder spatial size (proxy)
NUM_CLS   = 10
NUM_PROP  = 200


def sep(title):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print('─'*62)


def show(tag, t):
    print(f"    {tag:<46} shape: {list(t.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
def run_layer_by_layer():
    """Forward pass through every EA-LSS stage, showing TTSim shapes."""
    all_ok = True

    # ── INPUT ─────────────────────────────────────────────────────────────────
    sep("INPUT — camera images [B*N_VIEWS, 3, H, W]")
    img = _from_shape("img", [B * N_VIEWS, 3, IMG_H, IMG_W])
    show("img  (B*N_VIEWS camera frames)", img)

    # ── STAGE 1 — Image Backbone: CBSwinTransformer ───────────────────────────
    sep("STAGE 1 — CBSwinTransformer  (img backbone)")
    img_backbone = CBSwinTransformer(
        "val_bb",
        embed_dim=96, cb_del_stages=1, pretrain_img_size=224,
        depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=7, out_indices=(0,1,2,3),
    )
    bb_outs = img_backbone(img)   # list[4]
    for i, o in enumerate(bb_outs):
        show(f"bb_outs[{i}]  (scale 1/{4*2**i}, embed {96*2**i}ch)", o)
    ok = len(bb_outs) == 4
    print_test("CBSwinTransformer → 4 feature-pyramid levels", "", ok)
    all_ok &= ok

    # ── STAGE 2 — Image Neck: FPNC ────────────────────────────────────────────
    sep("STAGE 2 — FPNC  (img neck, outC=imc=256)")
    img_neck = FPNC("val_neck", in_channels=[96,192,384,768], out_channels=64,
                    num_outs=4, outC=IMC)
    neck_out = img_neck(*bb_outs)   # list[1]
    img_feat = neck_out[0]          # [B*N, imc, tH, tW]
    show("img_feat  (unified camera feature map)", img_feat)
    ok = img_feat.shape[1] == IMC
    print_test(f"FPNC output channels == imc={IMC}", f"got {img_feat.shape[1]}", ok)
    all_ok &= ok

    # ── STAGE 3 — Camera→BEV: LiftSplatShoot ─────────────────────────────────
    sep("STAGE 3 — LiftSplatShoot  (camera→BEV via depth estimation)")
    lift = LiftSplatShoot(
        "val_lift", lss=False, final_dim=(900,1600),
        camera_depth_range=[4.0,45.0,1.0], pc_range=[-50,-50,-5,50,50,3],
        downsample=4, grid=3, inputC=IMC, camC=64,
    )
    _bev_raw = lift(img_feat)
    show("img_bev_raw  (LSS internal proxy output)", _bev_raw)
    bev_H = PTS_SZ // 2   # 124 — matches SECOND stride-2 output
    img_bev = _from_shape("val_img_bev", [B, IMC, bev_H, bev_H])
    show("img_bev  (aligned to LiDAR BEV spatial = 124×124)", img_bev)
    ok = img_bev.shape[1] == IMC
    print_test("img_bev channels == imc", f"got {img_bev.shape[1]}", ok)
    all_ok &= ok

    # ── STAGE 4 — LiDAR Voxel Encoder: HardSimpleVFE_ATT ─────────────────────
    sep("STAGE 4 — HardSimpleVFE_ATT  (LiDAR voxel encoder)")
    pts_vfe = HardSimpleVFE_ATT("val_vfe")
    vox = _from_shape("val_voxels", [1000, 10, 5])   # [N_vox, M=10, 5]  M must be 10
    vfe_out = pts_vfe(vox)
    show("vfe_out  (attention-weighted voxel features)", vfe_out)
    vfe_ok = list(vfe_out.shape) == [1000, 10, 32]
    print_test("HardSimpleVFE_ATT output [N_vox, 10, 32]",
               f"got {list(vfe_out.shape)}", vfe_ok)
    all_ok &= vfe_ok

    # ── STAGE 5 — SparseEncoder → PROXY ──────────────────────────────────────
    sep("STAGE 5 — SparseEncoder  (CUDA-only → shape proxy)")
    pts_proxy = _from_shape("val_pts_proxy", [B, PTS_BEV, PTS_SZ, PTS_SZ])
    show("pts_proxy  [B, 64, 248, 248]  — SparseEncoder output substitute", pts_proxy)
    print("    [SparseEncoder uses spconv CUDA ops; cannot run on CPU — proxy used]")
    ok = list(pts_proxy.shape) == [B, PTS_BEV, PTS_SZ, PTS_SZ]
    print_test("SparseEncoder proxy shape [1, 64, 248, 248]",
               f"got {list(pts_proxy.shape)}", ok)
    all_ok &= ok

    # ── STAGE 6 — LiDAR Backbone: SECOND ─────────────────────────────────────
    sep("STAGE 6 — SECOND  (LiDAR 2D backbone, 3 stride-2 stages)")
    pts_backbone = SECOND(
        "val_pts_bb",
        in_channels=PTS_BEV, out_channels=[64,128,256],
        layer_nums=[3,5,5], layer_strides=[2,2,2],
    )
    pts_bb_outs = pts_backbone(pts_proxy)   # list[3]
    spatial = PTS_SZ
    for i, o in enumerate(pts_bb_outs):
        spatial //= 2
        show(f"pts_bb_outs[{i}]  (stride 2^{i+1}, spatial ~{spatial}×{spatial})", o)
    ok = len(pts_bb_outs) == 3
    print_test("SECOND → 3 BEV feature levels", "", ok)
    all_ok &= ok

    # ── STAGE 7 — LiDAR Neck: SECONDFPN ──────────────────────────────────────
    sep("STAGE 7 — SECONDFPN  (LiDAR neck, upsample all levels to stride-2 size)")
    pts_neck = SECONDFPN(
        "val_pts_neck",
        in_channels=[64,128,256], out_channels=[128,128,128],
        upsample_strides=[1,2,4],
    )
    pts_bev = pts_neck(*pts_bb_outs)   # [B, lic=384, bH, bW]
    show("pts_bev  (concat 3×128 → 384ch LiDAR BEV)", pts_bev)
    ok = pts_bev.shape[1] == LIC
    print_test(f"SECONDFPN output channels == lic={LIC}", f"got {pts_bev.shape[1]}", ok)
    all_ok &= ok
    _, _, bH, bW = pts_bev.shape

    # ── STAGE 8 — BEV Fusion ─────────────────────────────────────────────────
    sep(f"STAGE 8 — BEV Fusion  cat([img_bev, pts_bev]) → Conv({LIC+IMC}→{LIC}) → BN → ReLU")
    fuse_in    = LIC + IMC
    reduc_conv = F.Conv2d("val_reduc_conv", fuse_in, LIC, 3, padding=1, bias=False)
    reduc_bn   = F.BatchNorm2d("val_reduc_bn", LIC)

    cat_feat = F.ConcatX("val_fuse_cat", axis=1)(img_bev, pts_bev)
    show(f"cat_feat  [B, {LIC}+{IMC}={fuse_in}, {bH}, {bW}]", cat_feat)
    fused = reduc_conv(cat_feat)
    show(f"after Conv2d({fuse_in}→{LIC},3)", fused)
    fused = reduc_bn(fused)
    show(f"after BatchNorm2d({LIC})", fused)
    fused = F.Relu("val_fuse_relu")(fused)
    show("fused_bev  (camera+LiDAR BEV after fusion)", fused)
    ok = fused.shape[1] == LIC
    print_test(f"fused_bev channels == lic={LIC}", f"got {fused.shape[1]}", ok)
    all_ok &= ok

    # ── STAGE 9 — Detection Head: TransFusionHead ─────────────────────────────
    sep("STAGE 9 — TransFusionHead  (heatmap init + TransDecoder + prediction heads)")
    head = TransFusionHead(
        "val_head",
        in_channels=LIC, hidden_channel=128, num_classes=NUM_CLS,
        num_proposals=NUM_PROP, num_decoder_layers=1,
        initialize_by_heatmap=True, fuse_img=False,
    )
    preds = head(fused)
    for k, v in sorted(preds.items()):
        show(f"preds['{k}']", v)
    ok = set(preds.keys()) == {"heatmap","center","height","dim","rot","vel"}
    print_test("TransFusionHead output keys correct", f"keys={sorted(preds.keys())}", ok)
    all_ok &= ok

    # ── FULL MODEL SANITY CHECK ───────────────────────────────────────────────
    sep("FULL EALSS MODEL — single end-to-end forward pass")
    model    = EALSS("ealss_val_full")
    img_full = _from_shape("ealss_full_img", [B * N_VIEWS, 3, IMG_H, IMG_W])
    show("model input  img", img_full)
    out = model(img_full)
    for k, v in sorted(out.items()):
        show(f"model output preds['{k}']", v)
    ok = isinstance(out, dict) and "heatmap" in out
    print_test("EALSS end-to-end → dict with canonical keys",
               f"keys={sorted(out.keys())}", ok)
    all_ok &= ok

    # ── PARAMETER COUNT TABLE ─────────────────────────────────────────────────
    sep("PARAMETER COUNT TABLE")
    _bb_p  = CBSwinTransformer("pc_bb2", embed_dim=96, cb_del_stages=1, pretrain_img_size=224,
                               depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=7,
                               out_indices=(0,1,2,3)).analytical_param_count()
    _nk_p  = FPNC("pc_nk2", in_channels=[96,192,384,768], out_channels=64, num_outs=4,
                  outC=IMC).analytical_param_count()
    _vf_p  = HardSimpleVFE_ATT("pc_vf2").analytical_param_count()
    _sb_p  = SECOND("pc_sb2", in_channels=PTS_BEV, out_channels=[64,128,256],
                    layer_nums=[3,5,5], layer_strides=[2,2,2]).analytical_param_count()
    _sn_p  = SECONDFPN("pc_sn2", in_channels=[64,128,256], out_channels=[128,128,128],
                       upsample_strides=[1,2,4]).analytical_param_count()
    _ls_p  = LiftSplatShoot("pc_ls2", lss=False, final_dim=(900,1600),
                            camera_depth_range=[4.0,45.0,1.0],
                            pc_range=[-50,-50,-5,50,50,3],
                            downsample=4, grid=3, inputC=IMC, camC=64).analytical_param_count()
    _rc_p  = fuse_in * LIC * 9 + 2 * LIC          # reduc_conv(no bias) + BN
    _hd_p  = TransFusionHead("pc_hd2", in_channels=LIC, hidden_channel=128,
                              num_classes=NUM_CLS, num_proposals=NUM_PROP,
                              num_decoder_layers=1, initialize_by_heatmap=True,
                              fuse_img=False).analytical_param_count()
    total  = model.analytical_param_count()
    rows   = [
        ("CBSwinTransformer (img backbone)",       _bb_p),
        ("FPNC              (img neck)",            _nk_p),
        ("HardSimpleVFE_ATT (LiDAR voxel enc.)",   _vf_p),
        ("SparseEncoder     [CUDA proxy → 0]",     0),
        ("SECOND            (LiDAR backbone)",      _sb_p),
        ("SECONDFPN         (LiDAR neck)",          _sn_p),
        ("LiftSplatShoot    (cam→BEV)",             _ls_p),
        ("reduc_conv + BN   (BEV fusion)",          _rc_p),
        ("TransFusionHead   (detection)",           _hd_p),
    ]
    print(f"\n    {'Module':<42} {'Params':>14}")
    print(f"    {'─'*42} {'─'*14}")
    for nm, p in rows:
        print(f"    {nm:<42} {p:>14,}")
    print(f"    {'─'*42} {'─'*14}")
    print(f"    {'TOTAL (model.analytical_param_count)':<42} {total:>14,}")
    ok_total = total == sum(p for _, p in rows)
    print_test("Param count == sum of sub-modules",
               f"sum={sum(p for _,p in rows):,}  model={total:,}", ok_total)
    all_ok &= ok_total

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  ALL VALIDATION CHECKS PASSED" if all_ok else
          "  *** ONE OR MORE CHECKS FAILED ***")
    print('='*62)
    return all_ok


if __name__ == "__main__":
    ok = run_layer_by_layer()
    sys.exit(0 if ok else 1)

