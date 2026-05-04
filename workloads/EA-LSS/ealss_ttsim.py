#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Polaris workload wrapper for the EA-LSS multi-modal 3-D detector.

Polaris interface:
    polaris.py -w config/ip_workloads.yaml -a config/all_archs.yaml \
               -m config/wl2archmapping.yaml --filterwlg ttsim --filterwl EALSS \
               -o ODDIR_ealss -s SIMPLE_RUN --outputformat json

Architecture (lc_fusion=True, camera_stream=True, se=False):
    img [bs*num_views, 3, img_H, img_W]
      → CBSwinTransformer  → 4-scale FPN features
      → FPNC               → [bs*num_views, imc, tH, tW]
      → LiftSplatShoot     → img_bev [bs, imc, bH, bW]

    pts proxy              → [bs, 64, 248, 248]
      → SECOND             → 3-scale BEV features
      → SECONDFPN          → pts_bev [bs, lic=384, bH, bW]

    cat([img_bev, pts_bev]) → Conv2d(640→384, 3) → BN → ReLU
      → TransFusionHead  → {center, height, dim, rot, vel, heatmap}
"""

import os
import sys
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────
_this_dir    = os.path.dirname(os.path.abspath(__file__))
_polaris_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))
_ealss_root  = _this_dir

for _p in [_polaris_root, _ealss_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── TTSim / Polaris imports ─────────────────────────────────────────────────
import ttsim.front.functional.op     as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import _from_shape

# ── EA-LSS module imports ───────────────────────────────────────────────────
from ttsim_modules.cbnet        import CBSwinTransformer
from ttsim_modules.fpnc         import FPNC
from ttsim_modules.cam_stream_lss import LiftSplatShoot
from ttsim_modules.voxel_encoder  import HardSimpleVFE_ATT
from ttsim_modules.second         import SECOND
from ttsim_modules.second_fpn     import SECONDFPN
from ttsim_modules.se_block       import SE_Block
from ttsim_modules.transfusion_head import TransFusionHead


class EALSS(SimNN.Module):
    """
    EA-LSS multi-modal 3-D detector — polaris workload class.

    cfg keys (all optional, defaults mirror nuScenes EA-LSS):
        bs              (int)   batch size                          [1]
        num_views       (int)   cameras per batch item              [6]
        img_height      (int)   input image height                  [256]
        img_width       (int)   input image width                   [704]
        imc             (int)   camera BEV channels (FPNC outC)     [256]
        lic             (int)   LiDAR BEV channels (SECONDFPN out)  [384]
        num_classes     (int)   detection classes                   [10]
        num_proposals   (int)   TransFusion query count             [200]
        camera_stream   (bool)  enable LiftSplatShoot               [True]
        lc_fusion       (bool)  fuse camera + LiDAR BEV             [True]
        se              (bool)  SE_Block after fusion               [False]
    """

    def __init__(self, name: str, cfg: dict):
        super().__init__()
        self.name           = name
        self.bs             = cfg.get('bs',             1)
        self.num_views      = cfg.get('num_views',      6)
        self.img_height     = cfg.get('img_height',   256)
        self.img_width      = cfg.get('img_width',    704)
        self.imc            = cfg.get('imc',          256)
        self.lic            = cfg.get('lic',          384)
        self.num_classes    = cfg.get('num_classes',   10)
        self.num_proposals  = cfg.get('num_proposals', 200)
        self.camera_stream  = cfg.get('camera_stream', True)
        self.lc_fusion      = cfg.get('lc_fusion',     True)
        self.se             = cfg.get('se',            False)

        # Derived constants (mirror ttsim_modules/ealss.py defaults)
        self.pts_bev_channels = 64
        self.pts_bev_size     = 248

        imc = self.imc
        lic = self.lic

        # ── Camera path ────────────────────────────────────────────────────
        self.img_backbone = CBSwinTransformer(
            name + ".img_backbone",
            embed_dim=96, cb_del_stages=1,
            pretrain_img_size=224, depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24], window_size=7, out_indices=(0, 1, 2, 3),
        )
        self.img_neck = FPNC(
            name + ".img_neck",
            in_channels=[96, 192, 384, 768], out_channels=64,
            num_outs=4, outC=imc,
        )
        if self.camera_stream:
            self.lift = LiftSplatShoot(
                name + ".lift",
                lss=False, final_dim=(900, 1600),
                camera_depth_range=[4.0, 45.0, 1.0],
                pc_range=[-50, -50, -5, 50, 50, 3],
                downsample=4, grid=3, inputC=imc, camC=64,
            )

        # ── LiDAR path ─────────────────────────────────────────────────────
        self.pts_voxel_encoder = HardSimpleVFE_ATT(name + ".pts_vfe")
        self.pts_backbone = SECOND(
            name + ".pts_backbone",
            in_channels=self.pts_bev_channels, out_channels=[64, 128, 256],
            layer_nums=[3, 5, 5], layer_strides=[2, 2, 2],
        )
        self.pts_neck = SECONDFPN(
            name + ".pts_neck",
            in_channels=[64, 128, 256], out_channels=[128, 128, 128],
            upsample_strides=[1, 2, 4],
        )

        # Precompute BEV spatial size: SECOND strides [2,2,2] -> first stride halves pts_bev_size
        # SECONDFPN upsample_strides [1,2,4] -> all outputs at pts_bev_size//2
        self.bev_size = self.pts_bev_size // 2  # 248 // 2 = 124

        # ── Fusion ─────────────────────────────────────────────────────────
        if self.lc_fusion:
            _fuse_in = lic + imc
            self.reduc_conv = F.Conv2d(
                name + ".reduc_conv", _fuse_in, lic, 3, padding=1, bias=False)
            self.reduc_bn   = F.BatchNorm2d(name + ".reduc_bn", lic)
            # Register ConcatX and Relu so link_op2module tracks them and their
            # output tensors are auto-registered in _tensors when fired.
            self.fuse_cat  = F.ConcatX(name + ".fuse_cat",  axis=1)
            self.fuse_relu = F.Relu(name + ".fuse_relu")
            if self.se:
                self.seblock = SE_Block(name + ".seblock", channels=lic)

        # ── Head ───────────────────────────────────────────────────────────
        if self.lc_fusion:
            _head_in = lic
        elif self.camera_stream:
            _head_in = imc
        else:
            _head_in = lic

        self.pts_bbox_head = TransFusionHead(
            name + ".head",
            in_channels=_head_in, hidden_channel=128,
            num_classes=self.num_classes, num_proposals=self.num_proposals,
            num_decoder_layers=1, initialize_by_heatmap=True, fuse_img=False,
        )

        super().link_op2module()

    # ── Polaris interface ───────────────────────────────────────────────────

    def set_batch_size(self, new_bs: int):
        self.bs = new_bs

    def create_input_tensors(self):
        self.input_tensors = {
            # LiDAR proxy: SparseEncoder is CUDA-only; approximated here.
            'pts_proxy': F._from_shape(
                'pts_proxy',
                [self.bs, self.pts_bev_channels, self.pts_bev_size, self.pts_bev_size],
                is_param=False, np_dtype=np.float32,
            ),
        }
        if self.camera_stream:
            # Camera BEV proxy: represents LSS output aligned to LiDAR BEV spatial size.
            # The camera backbone/neck/LSS are skipped in _forward; this proxy provides
            # the camera BEV features as a graph input.
            self.input_tensors['img_bev'] = F._from_shape(
                'img_bev',
                [self.bs, self.imc, self.bev_size, self.bev_size],
                is_param=False, np_dtype=np.float32,
            )

    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)

    def analytical_param_count(self):
        return 0

    def __call__(self):
        return self._forward(self.input_tensors.get('img'))

    # ── Forward pass ────────────────────────────────────────────────────────

    def _forward(self, img):
        """
        EA-LSS polaris forward pass.

        NOTE: Camera backbone (CBSwinTransformer), neck (FPNC), and
        LiftSplatShoot are skipped here because they contain inline dynamic
        ops incompatible with polaris graph construction. The img_bev tensor
        is provided as a pre-registered proxy in input_tensors.
        The LiDAR path (SECOND + SECONDFPN), BEV fusion, and TransFusionHead
        are fully graph-traced.
        """
        # 1. LiDAR path — SparseEncoder is CUDA-only; use proxy from input_tensors
        pts_proxy    = self.input_tensors['pts_proxy']
        pts_bb_outs  = self.pts_backbone(pts_proxy)  # list[3]
        pts_bev      = self.pts_neck(*pts_bb_outs)   # [B, lic, bH, bW]

        # 2. Camera BEV via proxy (backbone/neck/LSS ops are skipped)
        if self.camera_stream:
            img_bev = self.input_tensors['img_bev']  # [B, imc, bH, bW]
        else:
            img_bev = None

        # 3. Fusion
        if self.lc_fusion and img_bev is not None:
            cat_feat = self.fuse_cat(img_bev, pts_bev)
            fused    = self.reduc_conv(cat_feat)
            fused    = self.reduc_bn(fused)
            fused    = self.fuse_relu(fused)
            if self.se:
                fused = self.seblock(fused)
            bev_feat = fused
        elif img_bev is not None:
            bev_feat = img_bev
        else:
            bev_feat = pts_bev

        # 4. Detection
        return self.pts_bbox_head(bev_feat)
