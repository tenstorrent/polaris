#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of EALSS — the primary EA-LSS multi-modal 3D detector.

Original file: mmdet3d/models/detectors/ealss.py

Architecture:
    Camera stream (image path):
        img [B*N, 3, fH, fW]
          → CBSwinTransformer       → 4-scale features
          → FPNC                    → [B*N, imc, tH, tW]
          → LiftSplatShoot          → img_bev [B, imc, bH, bW]

    LiDAR stream (point cloud path):
        pts [N_pts, 5]
          → HardSimpleVFE_ATT       → [N_vox, 32]
          → SparseEncoder (CUDA)    → proxy [B, 64, bH, bW]  (not modelled)
          → SECOND                  → list of BEV feature maps
          → SECONDFPN               → pts_bev [B, lic, bH, bW]

    Fusion (when lc_fusion=True):
        cat([img_bev, pts_bev], dim=1)   → [B, lic+imc, bH, bW]
          → Conv2d(lic+imc, lic, 3)      → BN → ReLU
          → SE_Block (if se=True)
          → fused_bev [B, lic, bH, bW]

    Detection:
        fused_bev → TransFusionHead → prediction dict

Default EA-LSS config (nuScenes, lc_fusion=True, camera_stream=True):
    imc=256, lic=384, num_views=6
    img_backbone: CBSwinTransformer(embed_dim=96, depths=[2,2,6,2])
    img_neck    : FPNC(in=[96,192,384,768], out=64, num_outs=4, outC=imc)
    pts_voxel_encoder: HardSimpleVFE_ATT()
    pts_backbone: SECOND(in=64, out=[64,128,256], layers=[3,5,5])
    pts_neck    : SECONDFPN(in=[64,128,256], out=[128,128,128], strides=[1,2,4])
    pts_bbox_head: TransFusionHead(in_channels=lic=384)

SparseEncoder (pts_middle_encoder) is CUDA-only and is replaced by a
proxy _from_shape call inside __call__.

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

# ── All converted EA-LSS TTSim modules ──────────────────────────────────────
# Level 1 – Atomic ops / utilities
from ttsim_modules.clip_sigmoid             import clip_sigmoid          # Sigmoid clamped to (0,1) used by heatmap head
from ttsim_modules.gaussian                 import gaussian_radius, draw_heatmap_gaussian    # Gaussian heatmap target generation
from ttsim_modules.coord_transform          import apply_3d_transformation                # Camera↔LiDAR coordinate transforms
from ttsim_modules.box3d_nms                import box3d_multiclass_nms                   # 3-D NMS for detection output
from ttsim_modules.transfusion_bbox_coder   import TransFusionBBoxCoder                   # Box encode/decode for TransFusion

# Level 2 – Sub-network building blocks
from ttsim_modules.norm                     import NaiveSyncBatchNorm1d, NaiveSyncBatchNorm2d, NaiveSyncBatchNorm3d  # Sync BatchNorm wrappers
from ttsim_modules.mlp                      import MLP as _MLP          # Generic multi-layer perceptron
from ttsim_modules.ffn                      import FFN                   # Feed-Forward Network (Transformer FFN)
from ttsim_modules.multihead_attention      import MultiheadAttention as _MHA  # Multi-head self/cross attention
from ttsim_modules.position_embedding_learned import PositionEmbeddingLearned  # Learned 2-D positional embedding
from ttsim_modules.se_block                 import SE_Block              # Squeeze-and-Excitation channel attention

# Level 3 – Composite sub-modules
from ttsim_modules.swin_transformer         import SwinTransformer       # Standard Swin Transformer backbone
from ttsim_modules.voxel_encoder_utils      import VFELayer, get_paddings_indicator  # VFE building blocks
from ttsim_modules.transfusion_head         import TransformerDecoderLayer, TransFusionHead  # Detection head

# Level 4 – Full functional modules
from ttsim_modules.fpn                      import FPN                   # Feature Pyramid Network (used by FPNC)
from ttsim_modules.fpnc                     import FPNC, gapcontext      # Extended FPN for camera stream (gap context + multi-scale concat)
from ttsim_modules.cbnet                    import CBSwinTransformer     # Composite Backbone: dual-CBSwin w/ cross-backbone fusion
from ttsim_modules.voxel_encoder            import HardSimpleVFE, HardSimpleVFE_ATT   # LiDAR voxel feature encoders
from ttsim_modules.second                   import SECOND                # Stacked Encoder Conv blocks for sparse BEV
from ttsim_modules.second_fpn               import SECONDFPN             # FPN neck on top of SECOND, merges 3 strides
from ttsim_modules.cam_stream_lss           import LiftSplatShoot        # Camera→BEV via lift-splat-shoot
from ttsim_modules.cam_stream_lss_quickcumsum import QuickCumsum         # Fast cumulative sum kernel (used by LSS)

# Level 5–6 – Base detector classes (kept for inheritance context)
from ttsim_modules.base_3d_detector         import Base3DDetector        # Abstract 3-D detector shell
from ttsim_modules.mvx_two_stage            import MVXTwoStageDetector   # Generic multi-modal orchestrator
from ttsim_modules.mvx_faster_rcnn          import MVXFasterRCNN, DynamicMVXFasterRCNN  # FasterRCNN-style detector wrappers
from ttsim_modules.transfusion_detector     import TransFusionDetector   # TransFusion top-level (alternative to EALSS)


class EALSS(SimNN.Module):
    """
    Full EA-LSS multi-modal 3D detector.

    Composes camera and LiDAR feature extraction paths, BEV fusion, and
    TransFusion detection head.

    Args:
        name (str): Module prefix.
        camera_stream (bool): Enable LiftSplatShoot camera-to-BEV. Default: True.
        lc_fusion (bool): Fuse camera BEV with LiDAR BEV. Default: True.
        se (bool): Use SE_Block after fusion. Default: False.
        imc (int): Camera BEV feature channels (FPNC outC). Default: 256.
        lic (int): LiDAR BEV feature channels (SECONDFPN output). Default: 384.
        num_views (int): Number of cameras (views) per batch item. Default: 6.
        num_classes (int): Detection classes. Default: 10.
        num_proposals (int): Query proposals for TransFusion. Default: 200.
        num_decoder_layers (int): Transformer decoder depth. Default: 1.
        camera_depth_range (list): [d_min, d_max, d_step]. Default: [4,45,1].
        pc_range (list): Point-cloud range. Default: [-50,-50,-5,50,50,3].
        final_dim (tuple): Full image resolution. Default: (900, 1600).
        downsample (int): Image spatial downsample factor. Default: 4.
        grid (int): Z-axis voxel grid step for LiftSplatShoot. Default: 3.
        lss (bool): Use full ResNet BEV encoder in LiftSplatShoot. Default: False.
        pts_bev_channels (int): Channel dim of SparseEncoder proxy. Default: 64.
        pts_bev_size (int): Spatial dim of SparseEncoder proxy. Default: 248.
    """

    def __init__(
        self,
        name: str,
        camera_stream: bool = True,
        lc_fusion: bool = True,
        se: bool = False,
        imc: int = 256,
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
        # LiDAR path: HardSimpleVFE_ATT → (SparseEncoder proxy) → SECOND → SECONDFPN
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Fusion: cat → Conv2d(lic+imc, lic, 3) + BN + ReLU [+ SE_Block]
        # ------------------------------------------------------------------
        if lc_fusion:
            _fuse_in = lic + imc
            self.reduc_conv = F.Conv2d(
                name + ".reduc_conv",
                _fuse_in, lic, 3,
                padding=1, bias=False,
            )
            self.reduc_bn = F.BatchNorm2d(name + ".reduc_bn", lic)
            # Param counts for the fusion ops (computed analytically)
            self._reduc_conv_params = _fuse_in * lic * 9          # no bias
            self._reduc_bn_params   = 2 * lic                      # BN scale+bias
            if se:
                self.seblock = SE_Block(name + ".seblock", channels=lic)

        # ------------------------------------------------------------------
        # Detection head: TransFusionHead
        # When lc_fusion=True  → head receives lic channels
        # When camera_stream only → head receives imc channels
        # ------------------------------------------------------------------
        if lc_fusion:
            _head_in = lic
        elif camera_stream:
            _head_in = imc
        else:
            _head_in = lic

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
        img_bb_outs  = self.img_backbone(img)           # list[4] SimTensors
        img_neck_out = self.img_neck(*img_bb_outs)      # list[1]
        img_feat     = img_neck_out[0]                  # [B*N, imc, tH, tW]

        # 2. LiDAR feature extraction
        #    SparseEncoder (CUDA) is approximated by a proxy _from_shape.
        pts_proxy    = _from_shape(
            self.name + ".pts_proxy",
            [B, self.pts_bev_channels, self.pts_bev_size, self.pts_bev_size],
        )
        pts_bb_outs  = self.pts_backbone(pts_proxy)     # list[3]
        pts_bev      = self.pts_neck(*pts_bb_outs)      # [B, lic, bH, bW]
        _, _, bH, bW = pts_bev.shape

        # 3. Camera BEV via LiftSplatShoot
        if self.camera_stream:
            _img_bev_raw = self.lift(img_feat)          # shape passthrough proxy
            img_bev      = _from_shape(
                self.name + ".img_bev",
                [B, self.imc, bH, bW],
            )
        else:
            img_bev = None

        # 4. Fusion
        if self.lc_fusion and img_bev is not None:
            cat_feat = F.ConcatX(self.name + ".fuse_cat", axis=1)(img_bev, pts_bev)
            fused    = self.reduc_conv(cat_feat)
            fused    = self.reduc_bn(fused)
            fused    = F.Relu(self.name + ".fuse_relu")(fused)
            if self.se:
                fused = self.seblock(fused)
            bev_feat = fused                            # [B, lic, bH, bW]
        elif img_bev is not None:
            bev_feat = img_bev                          # [B, imc, bH, bW]
        else:
            bev_feat = pts_bev                          # [B, lic, bH, bW]

        # 5. Detection
        return self.pts_bbox_head(bev_feat)

    def analytical_param_count(self, lvl: int = 0) -> int:
        p  = self.img_backbone.analytical_param_count(lvl + 1)
        p += self.img_neck.analytical_param_count(lvl + 1)
        p += self.pts_voxel_encoder.analytical_param_count(lvl + 1)
        # SparseEncoder (pts_middle_encoder) is CUDA-only → 0
        p += self.pts_backbone.analytical_param_count(lvl + 1)
        p += self.pts_neck.analytical_param_count(lvl + 1)
        if self.camera_stream:
            p += self.lift.analytical_param_count(lvl + 1)
        if self.lc_fusion:
            p += self._reduc_conv_params
            p += self._reduc_bn_params
            if self.se:
                p += self.seblock.analytical_param_count(lvl + 1)
        p += self.pts_bbox_head.analytical_param_count(lvl + 1)
        return p
