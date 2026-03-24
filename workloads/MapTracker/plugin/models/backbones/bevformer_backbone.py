#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
BEVFormerBackbone and UpsampleBlock
"""

# -------------------------------PyTorch--------------------------------

# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from mmdet.models import BACKBONES
# from mmcv.runner import force_fp32, auto_fp16
# from mmdet.models.utils import build_transformer
# from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
# from .bevformer.grid_mask import GridMask
# from mmdet3d.models import builder
# from contextlib import nullcontext
#
#
# class UpsampleBlock(nn.Module):
#     def __init__(self, ins, outs):
#         super(UpsampleBlock, self).__init__()
#         self.gn = nn.GroupNorm(32, outs)
#         self.conv = nn.Conv2d(ins, outs, kernel_size=3,
#                               stride=1, padding=1)  # same
#         self.relu = nn.ReLU(inplace=True)
#
#     def init_weights(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward(self, x):
#
#         x = self.conv(x)
#         x = self.relu(self.gn(x))
#         x = self.upsample2x(x)
#
#         return x
#
#     def upsample2x(self, x):
#         _, _, h, w = x.shape
#         x = F.interpolate(x, size=(h*2, w*2),
#                           mode='bilinear', align_corners=True)
#         return x
#
# @BACKBONES.register_module()
# class BEVFormerBackbone(nn.Module):
#     """Head of Detr3D.
#     Args:
#         with_box_refine (bool): Whether to refine the reference points
#             in the decoder. Defaults to False.
#         as_two_stage (bool) : Whether to generate the proposal from
#             the outputs of encoder.
#         transformer (obj:`ConfigDict`): ConfigDict is used for building
#             the Encoder and Decoder.
#         bev_h, bev_w (int): spatial shape of BEV queries.
#     """
#
#     def __init__(self,
#                  roi_size,
#                  bev_h,
#                  bev_w,
#                  img_backbone=None,
#                  img_neck=None,
#                  transformer=None,
#                  positional_encoding=None,
#                  use_grid_mask=True,
#                  upsample=False,
#                  up_outdim=128,
#                  history_steps=None,
#                  **kwargs):
#         super(BEVFormerBackbone, self).__init__()
#
#         # image feature
#         self.default_ratio = 0.5
#         self.default_prob = 0.7
#         self.grid_mask = GridMask(
#             True, True, rotate=1, offset=False, ratio=self.default_ratio, mode=1,
#                 prob=self.default_prob)
#         self.use_grid_mask = use_grid_mask
#
#         if img_backbone:
#             self.img_backbone = builder.build_backbone(img_backbone)
#         if img_neck is not None:
#             self.img_neck = builder.build_neck(img_neck)
#             self.with_img_neck = True
#         else:
#             self.with_img_neck = False
#
#         self.bev_h = bev_h
#         self.bev_w = bev_w
#
#         self.real_w = roi_size[0]
#         self.real_h = roi_size[1]
#
#         self.positional_encoding = build_positional_encoding(
#             positional_encoding)
#         self.transformer = build_transformer(transformer)
#         self.embed_dims = self.transformer.embed_dims
#
#         self.upsample = upsample
#         if self.upsample:
#             self.up = UpsampleBlock(self.transformer.embed_dims, up_outdim)
#
#         self.history_steps = history_steps
#
#         self._init_layers()
#         self.init_weights()
#
#
#     def _init_layers(self):
#         """Initialize classification branch and regression branch of head."""
#         self.bev_embedding = nn.Embedding(
#             self.bev_h * self.bev_w, self.embed_dims)
#
#
#     def init_weights(self):
#         """Initialize weights of the DeformDETR head."""
#         self.transformer.init_weights()
#         self.img_backbone.init_weights()
#         self.img_neck.init_weights()
#
#         if self.upsample:
#             self.up.init_weights()
#
#     # @auto_fp16(apply_to=('img'))
#     def extract_img_feat(self, img, img_metas, len_queue=None):
#         """Extract features of images."""
#         B = img.size(0)
#         if img is not None:
#
#             # input_shape = img.shape[-2:]
#             # # update real input shape of each single img
#             # for img_meta in img_metas:
#             #     img_meta.update(input_shape=input_shape)
#
#             if img.dim() == 5 and img.size(0) == 1:
#                 img = img.squeeze(0)
#             elif img.dim() == 5 and img.size(0) > 1:
#                 B, N, C, H, W = img.size()
#                 img = img.reshape(B * N, C, H, W)
#             if self.use_grid_mask:
#                 img = self.grid_mask(img)
#
#             img_feats = self.img_backbone(img)
#             if isinstance(img_feats, dict):
#                 img_feats = list(img_feats.values())
#         else:
#             return None
#         if self.with_img_neck:
#             img_feats = self.img_neck(img_feats)
#
#         img_feats_reshaped = []
#         for img_feat in img_feats:
#             BN, C, H, W = img_feat.size()
#             if len_queue is not None:
#                 img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
#             else:
#                 img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
#
#         return img_feats_reshaped
#
#     def forward(self, img, img_metas, timestep, history_bev_feats, history_img_metas, all_history_coord, *args, prev_bev=None,
#                 img_backbone_gradient=True, **kwargs):
#         """Forward function.
#         Args:
#             mlvl_feats (tuple[Tensor]): Features from the upstream
#                 network, each is a 5D-tensor with shape
#                 (B, N, C, H, W).
#             prev_bev: previous bev featues
#         Returns:
#             all_cls_scores (Tensor): Outputs from the classification head, \
#                 shape [nb_dec, bs, num_query, cls_out_channels]. Note \
#                 cls_out_channels should includes background.
#             all_bbox_preds (Tensor): Sigmoid outputs from the regression \
#                 head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
#                 Shape [nb_dec, bs, num_query, 9].
#         """
#         # Optionally turn off the gradient backprop for the 2D image backbones
#         # but always keep the gradients on for the BEV transformer part
#         backprop_context = torch.no_grad if img_backbone_gradient is False else nullcontext
#         with backprop_context():
#             mlvl_feats = self.extract_img_feat(img=img, img_metas=img_metas)
#
#         bs, num_cam, _, _, _ = mlvl_feats[0].shape
#         dtype = mlvl_feats[0].dtype
#         bev_queries = self.bev_embedding.weight.to(dtype)
#
#         # Prepare the transformed history bev features, add the bev prop fusion here
#         if len(history_bev_feats) > 0:
#             all_warped_history_feat = []
#             for b_i in range(bs):
#                 history_coord = all_history_coord[b_i]
#                 history_bev_feats_i = torch.stack([feats[b_i] for feats in history_bev_feats], 0)
#                 warped_history_feat_i = F.grid_sample(history_bev_feats_i,
#                             history_coord, padding_mode='zeros', align_corners=False)
#                 all_warped_history_feat.append(warped_history_feat_i)
#             all_warped_history_feat = torch.stack(all_warped_history_feat, dim=0) # BTCHW
#             prop_bev_feat = all_warped_history_feat[:, -1]
#         else:
#             all_warped_history_feat = None
#             prop_bev_feat = None
#
#         # pad the bev history buffer to fixed length
#         if len(history_bev_feats) < self.history_steps:
#             num_repeat = self.history_steps - len(history_bev_feats)
#             zero_bev_feats = torch.zeros([bs, bev_queries.shape[1], self.bev_h, self.bev_w]).to(bev_queries.device)
#             padding_history_bev_feats = torch.stack([zero_bev_feats,] * num_repeat, dim=1)
#             if all_warped_history_feat is not None:
#                 all_warped_history_feat = torch.cat([padding_history_bev_feats, all_warped_history_feat], dim=1)
#             else:
#                 all_warped_history_feat = padding_history_bev_feats
#
#         bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
#         bev_pos = self.positional_encoding(bev_mask).to(dtype)
#
#         outs =  self.transformer.get_bev_features(
#                 mlvl_feats,
#                 bev_queries,
#                 self.bev_h,
#                 self.bev_w,
#                 grid_length=(self.real_h / self.bev_h,
#                             self.real_w / self.bev_w),
#                 bev_pos=bev_pos,
#                 prop_bev=prop_bev_feat,
#                 img_metas=img_metas,
#                 prev_bev=prev_bev,
#                 warped_history_bev=all_warped_history_feat,
#             )
#
#         outs = outs.unflatten(1,(self.bev_h,self.bev_w)).permute(0,3,1,2).contiguous()
#
#         if self.upsample:
#             outs = self.up(outs)
#
#         return outs, mlvl_feats

# -------------------------------TTSIM-----------------------------------

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import numpy as np

# ============================================================================
# Image Backbone (ResNet-50) and FPN Neck
# ============================================================================


class Bottleneck(SimNN.Module):
    """ResNet bottleneck (1x1->3x3->1x1 + residual)."""

    expansion = 4

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg["out_channels"]
        self.stride = cfg.get("stride", 1)
        self.downsample = cfg.get("downsample", None)

        conv_dims = [
            (self.in_channels, self.out_channels, 1, 0, 1),
            (self.out_channels, self.out_channels, 3, 1, self.stride),
            (self.out_channels, self.out_channels * Bottleneck.expansion, 1, 0, 1),
        ]
        oplist = []
        for i, (ic, oc, k, p, s) in enumerate(conv_dims):
            conv = F.Conv2d(
                f"{name}.conv{i}", ic, oc, kernel_size=k, padding=p, stride=s
            )
            bn = F.BatchNorm2d(f"{name}.bn{i}", oc)
            oplist += [conv, bn]

        self.op_blk = F.SimOpHandleList(oplist)
        self.relu = F.Relu(f"{name}.relu")
        self.add = F.Add(f"{name}.add")

        if self.downsample is not None:
            xi = self.downsample["in_channels"]
            xo = self.downsample["out_channels"]
            xs = self.downsample["stride"]
            self.conv_ds = F.Conv2d(
                f"{name}.conv_ds", xi, xo, kernel_size=1, padding=0, stride=xs
            )
            self.bn_ds = F.BatchNorm2d(f"{name}.bn_ds", xo)

        super().link_op2module()

    def __call__(self, x):
        y = self.op_blk(x)
        if self.downsample is None:
            z = self.add(y, x)
        else:
            x = self.conv_ds(x)
            x = self.bn_ds(x)
            z = self.add(y, x)
        return self.relu(z)


class ResNet50Backbone(SimNN.Module):
    """ResNet backbone producing features at selected stages.

    Default config matches MapTracker: ResNet-50 with out_indices=(1,2,3)
    returning [C3, C4, C5] at channels [512, 1024, 2048].
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.in_channels = 64
        layers = cfg.get("layers", [3, 4, 6, 3])
        img_channels = cfg.get("img_channels", 3)
        self.out_indices = cfg.get("out_indices", (1, 2, 3))

        # Stem: Conv7x7/2 + BN + ReLU + MaxPool/2
        self.conv1 = F.Conv2d(
            f"{name}.conv1", img_channels, 64, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = F.BatchNorm2d(f"{name}.bn1", 64)
        self.relu = F.Relu(f"{name}.relu")
        self.maxpool = F.MaxPool2d(
            f"{name}.maxpool", kernel_size=3, stride=2, padding=1
        )

        self.stage1 = SimNN.ModuleList(
            self._make_stage(f"{name}.layer1", layers[0], 64, stride=1)
        )
        self.stage2 = SimNN.ModuleList(
            self._make_stage(f"{name}.layer2", layers[1], 128, stride=2)
        )
        self.stage3 = SimNN.ModuleList(
            self._make_stage(f"{name}.layer3", layers[2], 256, stride=2)
        )
        self.stage4 = SimNN.ModuleList(
            self._make_stage(f"{name}.layer4", layers[3], 512, stride=2)
        )

        super().link_op2module()

    def _make_stage(self, name, num_blocks, planes, stride):
        blocks = []
        exp = Bottleneck.expansion
        downsample_cfg = None
        if stride != 1 or self.in_channels != planes * exp:
            downsample_cfg = {
                "in_channels": self.in_channels,
                "out_channels": planes * exp,
                "stride": stride,
            }
        blocks.append(
            Bottleneck(
                f"{name}.0",
                {
                    "in_channels": self.in_channels,
                    "out_channels": planes,
                    "stride": stride,
                    "downsample": downsample_cfg,
                },
            )
        )
        self.in_channels = planes * exp

        for i in range(1, num_blocks):
            blocks.append(
                Bottleneck(
                    f"{name}.{i}",
                    {
                        "in_channels": self.in_channels,
                        "out_channels": planes,
                    },
                )
            )
        return blocks

    def __call__(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        stages = [self.stage1, self.stage2, self.stage3, self.stage4]
        outs = []
        for idx, stage in enumerate(stages):
            for blk in stage:
                x = blk(x)
            if idx in self.out_indices:
                outs.append(x)
        return outs


class FPN(SimNN.Module):
    """Feature Pyramid Network: uniform-channel multi-scale features."""

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        in_channels = cfg["in_channels"]
        out_channels = cfg["out_channels"]
        num_outs = cfg.get("num_outs", 3)
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.lateral_convs = []
        for i, ic in enumerate(in_channels):
            c = F.Conv2d(f"{name}.lat{i}", ic, out_channels, kernel_size=1)
            self.lateral_convs.append(c)
            setattr(self, f"lat{i}", c)

        self.fpn_convs = []
        self.fpn_adds = []
        self.fpn_upsamples = []
        for i in range(self.num_ins):
            c = F.Conv2d(
                f"{name}.fpn{i}", out_channels, out_channels, kernel_size=3, padding=1
            )
            self.fpn_convs.append(c)
            setattr(self, f"fpn{i}", c)
            a = F.Add(f"{name}.fpn_add{i}")
            self.fpn_adds.append(a)
            setattr(self, f"fpn_add{i}", a)
            if i < self.num_ins - 1:
                r = F.Resize(f"{name}.fpn_up{i}", scale_factor=2, mode="nearest")
                self.fpn_upsamples.append(r)
                setattr(self, f"fpn_up{i}", r)

        self.extra_convs = []
        if num_outs > self.num_ins:
            for i in range(num_outs - self.num_ins):
                ic = in_channels[-1] if i == 0 else out_channels
                c = F.Conv2d(
                    f"{name}.extra{i}",
                    ic,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                self.extra_convs.append(c)
                setattr(self, f"extra{i}", c)

        super().link_op2module()

    def __call__(self, features):
        assert len(features) == self.num_ins
        laterals = [self.lateral_convs[i](features[i]) for i in range(self.num_ins)]

        for i in range(self.num_ins - 2, -1, -1):
            up = self.fpn_upsamples[i](laterals[i + 1])
            laterals[i] = self.fpn_adds[i](laterals[i], up)

        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        if self.num_outs > self.num_ins:
            x_extra = features[-1]
            for j, conv in enumerate(self.extra_convs):
                x_extra = conv(x_extra if j == 0 else outs[-1])
                outs.append(x_extra)

        return outs


# ============================================================================
# BEV Backbone Components
# ============================================================================


class UpsampleBlock(SimNN.Module):
    """
    Upsample block: Conv2d -> GroupNorm -> ReLU -> Upsample2x

    Args:
        name: Module name
        ins: Input channels
        outs: Output channels

    Architecture:
        1. Conv2d(ins, outs, kernel_size=3, stride=1, padding=1)
        2. GroupNorm(32 groups, outs channels)
        3. ReLU
        4. Bilinear interpolation upsample 2x
    """

    def __init__(self, name, ins, outs):
        super().__init__()
        self.name = name
        self.ins = ins
        self.outs = outs

        # Pre-create all operators
        # Conv2d: weight only, bias is separate
        self.conv_weight = F.Conv2d(
            name + ".conv",
            in_channels=ins,
            out_channels=outs,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Conv2d bias (reshaped for broadcasting: [outs] -> [1, outs, 1, 1])
        self.conv_bias_param = F._from_shape(
            name + ".conv.bias_param", [outs], is_param=True
        )
        # Reshape bias for broadcasting with Conv output [batch, outs, h, w]
        self.conv_bias_shape = F._from_data(
            name + ".conv.bias_shape",
            np.array([1, outs, 1, 1], dtype=np.int64),
            is_const=True,
        )
        self.conv_bias_reshape = F.Reshape(name + ".conv.bias_reshape")
        self.conv_add = F.Add(name + ".conv.add")

        self.gn = F.GroupNorm(name + ".gn", 32, outs)
        self.relu = F.Relu(name + ".relu")

        # Upsample 2x using bilinear interpolation
        self.upsample = F.Resize(
            name + ".upsample",
            scale_factor=2.0,
            mode="linear",
            coordinate_transformation_mode="align_corners",
        )

        super().link_op2module()

    def __call__(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch, ins, h, w]

        Returns:
            out: Output tensor [batch, outs, h*2, w*2]
        """
        # 1. Conv2d (weight only)
        x = self.conv_weight(x)
        setattr(self, x.name, x)

        # 1b. Reshape bias and add
        bias_reshaped = self.conv_bias_reshape(
            self.conv_bias_param, self.conv_bias_shape
        )
        setattr(self, bias_reshaped.name, bias_reshaped)

        x = self.conv_add(x, bias_reshaped)
        setattr(self, x.name, x)

        # 2. GroupNorm
        x = self.gn(x)
        setattr(self, x.name, x)

        # 3. ReLU
        x = self.relu(x)
        setattr(self, x.name, x)

        # 4. Upsample 2x
        x = self.upsample(x)
        setattr(self, x.name, x)

        return x

    def analytical_param_count(self, lvl=0):
        """
        Calculate total number of trainable parameters

        Args:
            lvl: Verbosity level (0=silent, 1=summary, 2=detailed)

        Returns:
            int: Total parameter count
        """
        # Conv2d: weight + bias
        # Shape: [outs, ins, kernel, kernel]
        conv_weight = self.outs * self.ins * 3 * 3
        conv_bias = self.outs

        # GroupNorm: weight + bias = 2 * outs
        gn_params = 2 * self.outs

        total = conv_weight + conv_bias + gn_params

        if lvl >= 1:
            print(f"UpsampleBlock ({self.name}):")
            print(f"  Conv2d weight: {conv_weight:,}")
            print(f"  Conv2d bias: {conv_bias:,}")
            print(f"  GroupNorm: {gn_params:,}")
            print(f"  Total: {total:,}")

        return total


class BEVFormerBackbone(SimNN.Module):
    """
    BEVFormer backbone for generating BEV features from multi-view features.

    This module orchestrates:
    1. ResNet-50 image backbone + FPN neck (extract_img_feat)
    2. BEV query embedding initialization
    3. History BEV feature warping (temporal fusion)
    4. BEV transformer encoding (spatial cross-attention)
    5. Optional upsampling

    Args:
        name: Module name
        bev_h: BEV grid height
        bev_w: BEV grid width
        embed_dims: BEV feature dimension
        real_h: Real-world height in meters
        real_w: Real-world width in meters
        transformer: PerceptionTransformer instance (already converted to TTSim)
        max_batch_size: Maximum batch size (for pre-allocating operators)
        upsample: Whether to apply upsampling
        up_outdim: Output dimension after upsampling
        history_steps: Number of history frames to buffer
        num_cams: Number of cameras (default 7 for MapTracker/AV2)
        img_channels: Input image channels (default 3)
        backbone_layers: ResNet stage block counts (default [3,4,6,3] = ResNet-50)
        out_indices: Which ResNet stages to output (default (1,2,3) = C3,C4,C5)
        fpn_in_channels: FPN input channels per level (default [512,1024,2048])
        fpn_num_outs: Number of FPN output levels (default 3)

    TTSim Implementation Notes:
    - Full pipeline from raw images: ResNet-50 -> FPN -> BEV transformer
    - Forward accepts raw images [B*num_cams, C, H, W] as input
    - BEV embedding uses Embedding operator
    - History warping uses GridSample operator (batched processing)
    - All operators pre-created in __init__ (no dynamic creation)
    """

    def __init__(
        self,
        name,
        bev_h,
        bev_w,
        embed_dims,
        real_h,
        real_w,
        transformer,
        max_batch_size=8,
        upsample=False,
        up_outdim=128,
        history_steps=3,
        num_cams=7,
        img_channels=3,
        backbone_layers=None,
        out_indices=(1, 2, 3),
        fpn_in_channels=None,
        fpn_num_outs=3,
    ):
        super().__init__()
        self.name = name
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.real_h = real_h
        self.real_w = real_w
        self.transformer = transformer
        self.max_batch_size = max_batch_size
        self.upsample = upsample
        self.history_steps = history_steps
        self.num_cams = num_cams

        # Image backbone (ResNet-50) and neck (FPN)
        if backbone_layers is None:
            backbone_layers = [3, 4, 6, 3]  # ResNet-50
        if fpn_in_channels is None:
            fpn_in_channels = [512, 1024, 2048]  # C3, C4, C5 for ResNet-50

        self.img_backbone = ResNet50Backbone(
            f"{name}.img_backbone",
            {
                "layers": backbone_layers,
                "img_channels": img_channels,
                "out_indices": out_indices,
            },
        )
        self.img_neck = FPN(
            f"{name}.img_neck",
            {
                "in_channels": fpn_in_channels,
                "out_channels": embed_dims,
                "num_outs": fpn_num_outs,
            },
        )

        # BEV query embedding: [bev_h * bev_w, embed_dims]
        self.bev_embedding = F.Embedding(
            name + ".bev_embedding", tbl_size=bev_h * bev_w, emb_dim=embed_dims
        )

        # Pre-create operators for history warping
        # Stack implementation: Unsqueeze each tensor at dim=0, then concat
        _history_unsqueezes = []
        for i in range(history_steps):
            unsqueeze = F.Unsqueeze(f"{name}.history_unsqueeze_{i}")
            _history_unsqueezes.append(unsqueeze)
        self.history_unsqueezes = F.SimOpHandleList(_history_unsqueezes)

        self.history_concat = F.ConcatX(name + ".history_concat", axis=0)
        self.history_transpose = F.Transpose(
            name + ".history_transpose", perm=[1, 0, 2, 3, 4]
        )  # [T, bs, C, H, W] -> [bs, T, C, H, W]

        # Reshape for batched GridSample: [bs, T, C, H, W] -> [bs*T, C, H, W]
        self.history_reshape_in = F.Reshape(name + ".history_reshape_in")
        self.coord_reshape_in = F.Reshape(name + ".coord_reshape_in")

        # GridSample for history warping (batched)
        self.grid_sample = F.GridSample(
            name + ".grid_sample", padding_mode="zeros", align_corners=False
        )

        # Reshape back: [bs*T, C, H, W] -> [bs, T, C, H, W]
        self.history_reshape_out = F.Reshape(name + ".history_reshape_out")

        # Note: last_slice will be created dynamically in forward() with actual batch size

        # Reshape to squeeze time dimension: [bs, 1, C, H, W] -> [bs, C, H, W]
        self.prop_reshape = F.Reshape(name + ".prop_reshape")

        # Concatenation for history padding
        self.concat_history = F.ConcatX(name + ".concat_history", axis=1)

        # Optional upsampling
        if self.upsample:
            self.up = UpsampleBlock(name + ".up", embed_dims, up_outdim)

        # Reshape FPN features: [B*nc, d, h, w] -> [B, nc, d, h, w]
        self.feat_reshapes = []
        for i in range(fpn_num_outs):
            r = F.Reshape(f"{name}.feat_reshape_{i}")
            self.feat_reshapes.append(r)
            setattr(self, f"feat_reshape_{i}", r)

        # Reshape for unflatten: [bs, bev_h*bev_w, C] -> [bs, bev_h, bev_w, C]
        self.unflatten_reshape = F.Reshape(name + ".unflatten_reshape")

        # Permute: [bs, bev_h, bev_w, C] -> [bs, C, bev_h, bev_w]
        self.permute = F.Transpose(name + ".permute", perm=[0, 3, 1, 2])

        super().link_op2module()

        # Call counter for unique tensor names across multiple invocations
        self._call_count = 0

    def analytical_param_count(self, lvl=0):
        """
        Calculate total number of trainable parameters

        Args:
            lvl: Verbosity level (0=silent, 1=summary, 2=detailed)

        Returns:
            int: Total parameter count
        """
        # BEV embedding: [bev_h * bev_w, embed_dims]
        bev_embedding_params = self.bev_h * self.bev_w * self.embed_dims

        total = bev_embedding_params

        # Add upsampling block params if enabled
        if self.upsample:
            upsample_params = self.up.analytical_param_count(lvl=max(0, lvl - 1))
            total += upsample_params

        if lvl >= 1:
            print(f"\nBEVFormerBackbone ({self.name}):")
            print(
                f"  BEV Embedding: {bev_embedding_params:,} ({self.bev_h}x{self.bev_w}x{self.embed_dims})"
            )
            if self.upsample:
                print(f"  UpsampleBlock: {upsample_params:,}")
            print(f"  Total: {total:,}")

        return total

    def extract_img_feat(self, img):
        """
        Extract multi-level features from raw images using ResNet-50 + FPN.

        Args:
            img: [B*num_cams, C, H, W] raw camera images
        Returns:
            mlvl_feats: list of [B, num_cams, embed_dims, H_l, W_l]
        """
        nc = self.num_cams
        bnc = img.shape[0]
        B = bnc // nc

        # ResNet-50 backbone: [B*nc, C, H, W] -> list of [B*nc, ch, h, w]
        backbone_feats = self.img_backbone(img)

        # FPN neck: list of [B*nc, ch, h, w] -> list of [B*nc, embed_dims, h, w]
        fpn_feats = self.img_neck(backbone_feats)

        # Reshape: [B*nc, d, h, w] -> [B, nc, d, h, w]
        mlvl_feats = []
        for idx, feat in enumerate(fpn_feats):
            _, d, fh, fw = feat.shape
            shape_tensor = F._from_data(
                f"{self.name}.feat_shape_{idx}_c{self._call_count}",
                np.array([B, nc, d, fh, fw], dtype=np.int64),
                is_const=True,
            )
            setattr(self, shape_tensor.name, shape_tensor)
            feat_reshaped = self.feat_reshapes[idx](feat, shape_tensor)
            mlvl_feats.append(feat_reshaped)

        return mlvl_feats

    def __call__(
        self,
        img,
        bev_pos,
        history_bev_feats=None,
        all_history_coord=None,
        prev_bev=None,
        img_metas=None,
    ):
        """
        Forward pass

        Args:
            img: [B*num_cams, C, H, W] raw camera images
            bev_pos: BEV positional encoding [bs, C, bev_h, bev_w]
            history_bev_feats: List of history BEV features (length <= history_steps)
                              Each element: [bs, C, bev_h, bev_w]
            all_history_coord: Warp coordinates [bs, T, H, W, 2] (NOT a list, single tensor)
            prev_bev: Previous BEV features (optional)
            img_metas: Image metadata (for transformer)

        Returns:
            outs: BEV features [bs, C_out, bev_h, bev_w]
            mlvl_feats: Multi-level image features
        """
        # Extract image features through ResNet-50 + FPN
        mlvl_feats = self.extract_img_feat(img)

        bs = mlvl_feats[0].shape[0]

        # Unique prefix per call to avoid tensor name collisions
        _cc = self._call_count
        self._call_count += 1
        pfx = f"{self.name}_c{_cc}"

        # 1. Get BEV query embeddings [bev_h*bev_w, embed_dims]
        # Create indices tensor: [0, 1, 2, ..., bev_h*bev_w-1]
        bev_indices = F._from_data(
            f"{pfx}.bev_indices",
            np.arange(self.bev_h * self.bev_w, dtype=np.int64),
            is_const=True,
        )
        setattr(self, bev_indices.name, bev_indices)
        bev_queries = self.bev_embedding(bev_indices)
        setattr(self, bev_queries.name, bev_queries)

        # 2. Warp history BEV features if provided
        if history_bev_feats is not None and len(history_bev_feats) > 0:
            T = len(history_bev_feats)

            # Stack history features: list of [bs, C, H, W] -> [T, bs, C, H, W]
            # Using Unsqueeze + ConcatX pattern (no F.Stack operator in TTSim)
            # Create axes tensor for unsqueeze at dim=0
            axes_tensor = F._from_data(
                f"{pfx}.unsqueeze_axes", np.array([0], dtype=np.int64), is_const=True
            )
            setattr(self, axes_tensor.name, axes_tensor)

            # Unsqueeze each history tensor: [bs, C, H, W] -> [1, bs, C, H, W]
            unsqueezed_history = []
            for i, hist_feat in enumerate(history_bev_feats):
                unsqueezed = self.history_unsqueezes[i](hist_feat, axes_tensor)
                setattr(self, unsqueezed.name, unsqueezed)
                unsqueezed_history.append(unsqueezed)

            # Concat along axis=0: [1, bs, C, H, W] x T -> [T, bs, C, H, W]
            history_stacked = self.history_concat(*unsqueezed_history)
            setattr(self, history_stacked.name, history_stacked)

            # Transpose: [T, bs, C, H, W] -> [bs, T, C, H, W]
            history_transposed = self.history_transpose(history_stacked)
            setattr(self, history_transposed.name, history_transposed)

            # Reshape for batched GridSample: [bs, T, C, H, W] -> [bs*T, C, H, W]
            history_flat_shape = F._from_data(
                f"{pfx}.history_flat_shape",
                np.array(
                    [bs * T, self.embed_dims, self.bev_h, self.bev_w], dtype=np.int64
                ),
                is_const=True,
            )
            setattr(self, history_flat_shape.name, history_flat_shape)
            history_flat = self.history_reshape_in(
                history_transposed, history_flat_shape
            )
            setattr(self, history_flat.name, history_flat)

            # Reshape coords: [bs, T, H, W, 2] -> [bs*T, H, W, 2]
            coord_flat_shape = F._from_data(
                f"{pfx}.coord_flat_shape",
                np.array([bs * T, self.bev_h, self.bev_w, 2], dtype=np.int64),
                is_const=True,
            )
            setattr(self, coord_flat_shape.name, coord_flat_shape)
            coord_flat = self.coord_reshape_in(all_history_coord, coord_flat_shape)
            setattr(self, coord_flat.name, coord_flat)

            # GridSample: [bs*T, C, H, W] with [bs*T, H, W, 2] -> [bs*T, C, H, W]
            warped_flat = self.grid_sample(history_flat, coord_flat)
            setattr(self, warped_flat.name, warped_flat)

            # Reshape back: [bs*T, C, H, W] -> [bs, T, C, H, W]
            warped_shape = F._from_data(
                f"{pfx}.warped_shape",
                np.array(
                    [bs, T, self.embed_dims, self.bev_h, self.bev_w], dtype=np.int64
                ),
                is_const=True,
            )
            setattr(self, warped_shape.name, warped_shape)
            all_warped_history_feat = self.history_reshape_out(
                warped_flat, warped_shape
            )
            setattr(self, all_warped_history_feat.name, all_warped_history_feat)

            # Extract last timestep: [bs, T, C, H, W] -> [bs, 1, C, H, W]
            # SliceF along axis=1, start=T-1, end=T (gets last timestep)
            # Create SliceF dynamically with actual batch size
            last_slice_shape = [bs, 1, self.embed_dims, self.bev_h, self.bev_w]
            starts = F._from_data(
                f"{pfx}.last_starts",
                np.array([0, T - 1, 0, 0, 0], dtype=np.int64),
                is_const=True,
            )
            setattr(self, starts.name, starts)
            ends = F._from_data(
                f"{pfx}.last_ends",
                np.array(
                    [bs, T, self.embed_dims, self.bev_h, self.bev_w], dtype=np.int64
                ),
                is_const=True,
            )
            setattr(self, ends.name, ends)
            axes = F._from_data(
                f"{pfx}.last_axes",
                np.array([0, 1, 2, 3, 4], dtype=np.int64),
                is_const=True,
            )
            setattr(self, axes.name, axes)
            steps = F._from_data(
                f"{pfx}.last_steps",
                np.array([1, 1, 1, 1, 1], dtype=np.int64),
                is_const=True,
            )
            setattr(self, steps.name, steps)

            last_slice = F.SliceF(f"{pfx}.last_slice", out_shape=last_slice_shape)
            prop_bev_feat_5d = last_slice(
                all_warped_history_feat, starts, ends, axes, steps
            )
            setattr(self, prop_bev_feat_5d.name, prop_bev_feat_5d)

            # Squeeze time dimension: [bs, 1, C, H, W] -> [bs, C, H, W]
            prop_squeeze_shape = F._from_data(
                f"{pfx}.prop_squeeze_shape",
                np.array([bs, self.embed_dims, self.bev_h, self.bev_w], dtype=np.int64),
                is_const=True,
            )
            setattr(self, prop_squeeze_shape.name, prop_squeeze_shape)
            prop_bev_feat = self.prop_reshape(prop_bev_feat_5d, prop_squeeze_shape)
            setattr(self, prop_bev_feat.name, prop_bev_feat)

            # Pad history buffer to fixed length if needed
            if T < self.history_steps:
                num_repeat = self.history_steps - T
                # Create zero padding: [bs, num_repeat, C, H, W]
                zero_padding_shape = [
                    bs,
                    num_repeat,
                    self.embed_dims,
                    self.bev_h,
                    self.bev_w,
                ]
                zero_bev_feats = F._from_data(
                    f"{pfx}.zero_padding",
                    np.zeros(zero_padding_shape, dtype=np.float32),
                    is_const=True,
                )
                setattr(self, zero_bev_feats.name, zero_bev_feats)

                # Concat padding with warped history
                all_warped_history_feat = self.concat_history(
                    zero_bev_feats, all_warped_history_feat
                )
                setattr(self, all_warped_history_feat.name, all_warped_history_feat)
        else:
            all_warped_history_feat = None
            prop_bev_feat = None

        # 3. Call transformer to get BEV features
        # Output: [bs, bev_h*bev_w, C]
        outs = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            prop_bev=prop_bev_feat,
            img_metas=img_metas,
            prev_bev=prev_bev,
            warped_history_bev=all_warped_history_feat,
        )
        setattr(self, outs.name, outs)

        # 4. Unflatten: [bs, bev_h*bev_w, C] -> [bs, bev_h, bev_w, C]
        unflatten_shape = F._from_data(
            f"{pfx}.unflatten_shape",
            np.array([bs, self.bev_h, self.bev_w, self.embed_dims], dtype=np.int64),
            is_const=True,
        )
        setattr(self, unflatten_shape.name, unflatten_shape)
        outs = self.unflatten_reshape(outs, unflatten_shape)
        setattr(self, outs.name, outs)

        # 5. Permute: [bs, bev_h, bev_w, C] -> [bs, C, bev_h, bev_w]
        outs = self.permute(outs)
        setattr(self, outs.name, outs)

        # 6. Optional upsampling
        if self.upsample:
            outs = self.up(outs)
            setattr(self, outs.name, outs)

        return outs, mlvl_feats
