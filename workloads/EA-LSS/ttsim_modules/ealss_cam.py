#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of EALSS_CAM — camera-only variant of the EA-LSS detector.

Original file: mmdet3d/models/detectors/ealss_cam.py

Differences from EALSS:
  - imc=512  (default)                — larger camera BEV feature channels
  - lc_fusion=False (default)         — camera-only, no LiDAR fusion
  - No SparseEncoder / SECOND / SECONDFPN when lc_fusion=False
  - TransFusionHead gets imc channels when lc_fusion=False, lic when True
  - When lc_fusion=True: reduc_conv(lic+imc=896, lic=384, 3, bias=False)+BN

Architecture (default lc_fusion=False):
    img [B*N, 3, fH, fW]
      → CBSwinTransformer
      → FPNC                  → [B*N, imc=512, tH, tW]
      → LiftSplatShoot        → img_bev [B, imc, bH, bW]
      → TransFusionHead       → prediction dict

Architecture (lc_fusion=True variant):
    ... same as EALSS but with imc=512 giving reduc_conv(896, 384)

No torch / mmcv imports.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import SimOpHandle, _from_shape, _from_data

_ealss_root = os.path.abspath(os.path.join(current_dir, ".."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim_modules.cbnet            import CBSwinTransformer
from ttsim_modules.fpnc             import FPNC
from ttsim_modules.voxel_encoder    import HardSimpleVFE_ATT
from ttsim_modules.second           import SECOND
from ttsim_modules.second_fpn       import SECONDFPN
from ttsim_modules.cam_stream_lss   import LiftSplatShoot
from ttsim_modules.se_block         import SE_Block
from ttsim_modules.transfusion_head import TransFusionHead


class EALSS_CAM(SimNN.Module):
    """
    EA-LSS Camera-only (or optionally LiDAR-fused) 3D detector.

    Like EALSS but defaults to camera-only operation (lc_fusion=False, imc=512).

    Args:
        name (str): Module prefix.
        camera_stream (bool): Enable camera-to-BEV path. Default: True.
        lc_fusion (bool): Fuse camera BEV with LiDAR BEV. Default: False.
        se (bool): Use SE_Block after fusion. Default: False.
        imc (int): Camera BEV feature channels. Default: 512.
        lic (int): LiDAR BEV channels (only used when lc_fusion=True). Default: 384.
        num_views (int): Number of cameras per batch item. Default: 6.
        num_classes (int): Detection classes. Default: 10.
        num_proposals (int): Query proposals for TransFusion. Default: 200.
        num_decoder_layers (int): Transformer decoder depth. Default: 1.
        camera_depth_range (list): [d_min, d_max, d_step]. Default: [4,45,1].
        pc_range (list): Point-cloud range. Default: [-50,-50,-5,50,50,3].
        final_dim (tuple): Full image resolution. Default: (900, 1600).
        downsample (int): Image spatial downsample factor. Default: 4.
        grid (int): Z-axis voxel grid step. Default: 3.
        lss (bool): Use full ResNet encoder in LiftSplatShoot. Default: False.
        pts_bev_channels (int): SparseEncoder proxy channel dim. Default: 64.
        pts_bev_size (int): SparseEncoder proxy spatial dim. Default: 248.
    """

    def __init__(
        self,
        name: str,
        camera_stream: bool = True,
        lc_fusion: bool = False,
        se: bool = False,
        imc: int = 512,
        lic: int = 384,
        num_views: int = 6,
        num_classes: int = 10,
        num_proposals: int = 200,
        num_decoder_layers: int = 1,
        camera_depth_range: list = None,
        pc_range: list = None,
        final_dim: tuple = (900, 1600),
        downsample: int = 4,
        grid: int = 3,
        lss: bool = False,
        pts_bev_channels: int = 64,
        pts_bev_size: int = 248,
    ):
        super().__init__()
        self.name             = name
        self.camera_stream    = camera_stream
        self.lc_fusion        = lc_fusion
        self.se               = se
        self.imc              = imc
        self.lic              = lic
        self.num_views        = num_views
        self.pts_bev_channels = pts_bev_channels
        self.pts_bev_size     = pts_bev_size

        if camera_depth_range is None:
            camera_depth_range = [4.0, 45.0, 1.0]
        if pc_range is None:
            pc_range = [-50, -50, -5, 50, 50, 3]

        # ------------------------------------------------------------------
        # Camera path: CBSwinTransformer → FPNC → LiftSplatShoot
        # ------------------------------------------------------------------
        _swin_kwargs = dict(
            pretrain_img_size=224,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            out_indices=(0, 1, 2, 3),
        )
        self.img_backbone = CBSwinTransformer(
            name + ".img_backbone",
            embed_dim=96,
            cb_del_stages=1,
            **_swin_kwargs,
        )
        self.img_neck = FPNC(
            name + ".img_neck",
            in_channels=[96, 192, 384, 768],
            out_channels=64,
            num_outs=4,
            outC=imc,
        )

        if camera_stream:
            self.lift = LiftSplatShoot(
                name + ".lift",
                lss=lss,
                final_dim=final_dim,
                camera_depth_range=camera_depth_range,
                pc_range=pc_range,
                downsample=downsample,
                grid=grid,
                inputC=imc,
                camC=64,
            )

        # ------------------------------------------------------------------
        # LiDAR path (only instantiated when lc_fusion=True)
        # ------------------------------------------------------------------
        if lc_fusion:
            self.pts_voxel_encoder = HardSimpleVFE_ATT(name + ".pts_vfe")
            self.pts_backbone = SECOND(
                name + ".pts_backbone",
                in_channels=pts_bev_channels,
                out_channels=[64, 128, 256],
                layer_nums=[3, 5, 5],
                layer_strides=[2, 2, 2],
            )
            self.pts_neck = SECONDFPN(
                name + ".pts_neck",
                in_channels=[64, 128, 256],
                out_channels=[128, 128, 128],
                upsample_strides=[1, 2, 4],
            )

            # Fusion: cat → Conv2d(lic+imc, lic) + BN + ReLU
            _fuse_in = lic + imc
            self.reduc_conv = F.Conv2d(
                name + ".reduc_conv",
                _fuse_in, lic, 3,
                padding=1, bias=False,
            )
            self.reduc_bn = F.BatchNorm2d(name + ".reduc_bn", lic)
            self._reduc_conv_params = _fuse_in * lic * 9
            self._reduc_bn_params   = 2 * lic
            if se:
                self.seblock = SE_Block(name + ".seblock", channels=lic)

        # ------------------------------------------------------------------
        # Detection head
        # ------------------------------------------------------------------
        if lc_fusion:
            _head_in = lic
        elif camera_stream:
            _head_in = imc
        else:
            _head_in = lic                  # fallback (rarely used)

        self.pts_bbox_head = TransFusionHead(
            name + ".head",
            in_channels=_head_in,
            hidden_channel=128,
            num_classes=num_classes,
            num_proposals=num_proposals,
            num_decoder_layers=num_decoder_layers,
            initialize_by_heatmap=True,
            fuse_img=False,
        )

        super().link_op2module()

    # ------------------------------------------------------------------
    def __call__(self, img):
        """
        Full forward pass.

        Args:
            img (SimTensor): [B*N, 3, fH, fW] camera images.

        Returns:
            dict: Prediction dict from TransFusionHead.
        """
        B_N, _, fH, fW = img.shape
        B = max(B_N // self.num_views, 1)

        # 1. Image feature extraction
        img_bb_outs  = self.img_backbone(img)
        img_neck_out = self.img_neck(*img_bb_outs)
        img_feat     = img_neck_out[0]              # [B*N, imc, tH, tW]

        # 2. Camera BEV
        if self.camera_stream:
            _img_bev_raw = self.lift(img_feat)      # shape passthrough proxy
            # spatial dims matched to LiDAR BEV (248/2 = 124 with stride=2 SECOND)
            _bH = self.pts_bev_size // 2
            img_bev = _from_shape(
                self.name + ".img_bev",
                [B, self.imc, _bH, _bH],
            )
        else:
            img_bev = None

        # 3. LiDAR path + fusion (only when lc_fusion=True)
        if self.lc_fusion:
            pts_proxy   = _from_shape(
                self.name + ".pts_proxy",
                [B, self.pts_bev_channels, self.pts_bev_size, self.pts_bev_size],
            )
            pts_bb_outs = self.pts_backbone(pts_proxy)
            pts_bev     = self.pts_neck(*pts_bb_outs)   # [B, lic, bH, bW]
            if img_bev is not None:
                cat_feat = F.ConcatX(self.name + ".fuse_cat", axis=1)(img_bev, pts_bev)
            else:
                cat_feat = pts_bev
            fused = self.reduc_conv(cat_feat)
            fused = self.reduc_bn(fused)
            fused = F.Relu(self.name + ".fuse_relu")(fused)
            if self.se:
                fused = self.seblock(fused)
            bev_feat = fused
        elif img_bev is not None:
            bev_feat = img_bev                          # [B, imc, bH, bW]
        else:
            bev_feat = _from_shape(
                self.name + ".bev_fallback",
                [B, self.lic, self.pts_bev_size // 2, self.pts_bev_size // 2],
            )

        return self.pts_bbox_head(bev_feat)

    def analytical_param_count(self, lvl: int = 0) -> int:
        p  = self.img_backbone.analytical_param_count(lvl + 1)
        p += self.img_neck.analytical_param_count(lvl + 1)
        if self.camera_stream:
            p += self.lift.analytical_param_count(lvl + 1)
        if self.lc_fusion:
            p += self.pts_voxel_encoder.analytical_param_count(lvl + 1)
            # SparseEncoder (pts_middle_encoder) is CUDA-only → 0
            p += self.pts_backbone.analytical_param_count(lvl + 1)
            p += self.pts_neck.analytical_param_count(lvl + 1)
            p += self._reduc_conv_params
            p += self._reduc_bn_params
            if self.se:
                p += self.seblock.analytical_param_count(lvl + 1)
        p += self.pts_bbox_head.analytical_param_count(lvl + 1)
        return p
