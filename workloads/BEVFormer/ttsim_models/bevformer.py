#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BEVFormer TTSim Model – Self-Contained Polaris Workload

Architecture:
  ResNet backbone → FPN neck → BEVFormer encoder (MHA self + cross attn)
  → Detection decoder (MHA self + cross attn on BEV) → cls head

All building blocks are defined inline following the basicresnet.py / BasicLLM.py
pattern.  No external submodule imports; no numpy/SimTensor mixing in the forward pass.

Polaris interface:
  __init__(name, cfg), set_batch_size(), create_input_tensors(),
  __call__(), get_forward_graph(), analytical_param_count()
"""

import os, sys
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import math
import numpy as np

import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

# ============================================================================
# Backbone
# ============================================================================


class Bottleneck(SimNN.Module):
    """ResNet bottleneck (1x1→3x3→1x1 + residual).  cfg-dict style like basicresnet."""

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
            z = y + x
        else:
            x = self.conv_ds(x)
            x = self.bn_ds(x)
            z = y + x
        return self.relu(z)


class ResNetBackbone(SimNN.Module):
    """4-stage ResNet producing [C2, C3, C4, C5] at strides {4, 8, 16, 32}."""

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.in_channels = 64
        layers = cfg.get("layers", [3, 4, 6, 3])
        img_channels = cfg.get("img_channels", 3)

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
        c2 = x
        for blk in self.stage1:
            c2 = blk(c2)
        c3 = c2
        for blk in self.stage2:
            c3 = blk(c3)
        c4 = c3
        for blk in self.stage3:
            c4 = blk(c4)
        c5 = c4
        for blk in self.stage4:
            c5 = blk(c5)
        return [c2, c3, c4, c5]


# ============================================================================
# FPN Neck
# ============================================================================


class FPN(SimNN.Module):
    """Feature Pyramid Network: uniform-channel multi-scale features."""

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        in_channels = cfg["in_channels"]
        out_channels = cfg["out_channels"]
        num_outs = cfg.get("num_outs", 4)
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.lateral_convs = []
        for i, ic in enumerate(in_channels):
            c = F.Conv2d(f"{name}.lat{i}", ic, out_channels, kernel_size=1)
            self.lateral_convs.append(c)
            setattr(self, f"lat{i}", c)

        self.fpn_convs = []
        for i in range(self.num_ins):
            c = F.Conv2d(
                f"{name}.fpn{i}", out_channels, out_channels, kernel_size=3, padding=1
            )
            self.fpn_convs.append(c)
            setattr(self, f"fpn{i}", c)

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
            target_h = laterals[i].shape[-2]
            target_w = laterals[i].shape[-1]
            up = laterals[i + 1].interpolate(size=(target_h, target_w))
            laterals[i] = laterals[i] + up

        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        if self.num_outs > self.num_ins:
            x_extra = features[-1]
            for j, conv in enumerate(self.extra_convs):
                x_extra = conv(x_extra if j == 0 else outs[-1])
                outs.append(x_extra)

        return outs


# ============================================================================
# Transformer building blocks (self-contained, SimTensor-only)
# ============================================================================


class FFN(SimNN.Module):
    """Two-layer feed-forward network with ReLU."""

    def __init__(self, name, embed_dims, feedforward_channels):
        super().__init__()
        self.name = name
        self.fc1 = SimNN.Linear(f"{name}.fc1", embed_dims, feedforward_channels)
        self.relu = F.Relu(f"{name}.relu")
        self.fc2 = SimNN.Linear(f"{name}.fc2", feedforward_channels, embed_dims)
        super().link_op2module()

    def __call__(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MultiHeadAttn(SimNN.Module):
    """
    Multi-head attention – handles both self-attention and cross-attention.

    For self-attention pass the same tensor for query, key, value.
    For cross-attention pass different tensors; Q and KV may differ in
    sequence length (Nq vs Nk).
    """

    def __init__(self, name, embed_dims, num_heads):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        assert (
            embed_dims % num_heads == 0
        ), f"embed_dims {embed_dims} must be divisible by num_heads {num_heads}"
        self.dh = embed_dims // num_heads

        # Stored as a const SimTensor (mirrors BasicLLM.py pattern)
        self.scale = F._from_data(f"{name}.scale", np.float32(1.0 / math.sqrt(self.dh)))

        self.w_q = SimNN.Linear(f"{name}.w_q", embed_dims, embed_dims)
        self.w_k = SimNN.Linear(f"{name}.w_k", embed_dims, embed_dims)
        self.w_v = SimNN.Linear(f"{name}.w_v", embed_dims, embed_dims)
        self.w_o = SimNN.Linear(f"{name}.w_o", embed_dims, embed_dims)
        self.softmax = F.Softmax(f"{name}.softmax", axis=-1)

        super().link_op2module()

    def __call__(self, query, key, value):
        """
        Args:
            query: [B, Nq, d]
            key:   [B, Nk, d]
            value: [B, Nk, d]
        Returns:
            output: [B, Nq, d]
        """
        B, Nq, _ = query.shape
        _, Nk, _ = key.shape
        nH = self.num_heads
        dh = self.dh

        Q = self.w_q(query)  # [B, Nq, d]
        K = self.w_k(key)  # [B, Nk, d]
        V = self.w_v(value)  # [B, Nk, d]

        # Split into heads
        # Q/V: [B, N, d] → [B, N, nH, dh] → transpose(1,2) → [B, nH, N, dh]
        # K:   same, then transpose(2,3) → [B, nH, dh, Nk]  (ready for matmul with Q)
        Q = Q.reshape(B, Nq, nH, dh).transpose(1, 2)  # [B, nH, Nq, dh]
        K = K.reshape(B, Nk, nH, dh).transpose(1, 2).transpose(2, 3)  # [B, nH, dh, Nk]
        V = V.reshape(B, Nk, nH, dh).transpose(1, 2)  # [B, nH, Nk, dh]

        # Scaled dot-product attention
        attn = T.matmul(Q, K) * self.scale  # [B, nH, Nq, Nk]
        attn = self.softmax(attn)

        # Aggregate values and merge heads
        out = T.matmul(attn, V)  # [B, nH, Nq, dh]
        out = out.transpose(1, 2).reshape(B, Nq, self.embed_dims)  # [B, Nq, d]
        out = self.w_o(out)
        return out


class BEVEncoderLayer(SimNN.Module):
    """
    Single BEVFormer encoder layer.

    Operation order (matching the original paper):
      BEV self-attn  (temporal self-attention)  → LayerNorm
      Cross-attn (BEV queries ← camera features) → LayerNorm
      FFN                                         → LayerNorm
    """

    def __init__(self, name, embed_dims, num_heads, feedforward_channels):
        super().__init__()
        self.name = name

        self.self_attn = MultiHeadAttn(f"{name}.self_attn", embed_dims, num_heads)
        self.cross_attn = MultiHeadAttn(f"{name}.cross_attn", embed_dims, num_heads)
        self.ffn = FFN(f"{name}.ffn", embed_dims, feedforward_channels)
        self.norm1 = F.LayerNorm(f"{name}.norm1", embed_dims)
        self.norm2 = F.LayerNorm(f"{name}.norm2", embed_dims)
        self.norm3 = F.LayerNorm(f"{name}.norm3", embed_dims)

        super().link_op2module()

    def __call__(self, bev_query, cam_feats):
        """
        Args:
            bev_query: [B, bev_h*bev_w, d]
            cam_feats: [B, Nk, d]  (flattened multi-cam multi-scale features)
        Returns:
            Updated BEV: [B, bev_h*bev_w, d]
        """
        bev = bev_query + self.self_attn(bev_query, bev_query, bev_query)
        bev = self.norm1(bev)
        bev = bev + self.cross_attn(bev, cam_feats, cam_feats)
        bev = self.norm2(bev)
        bev = bev + self.ffn(bev)
        bev = self.norm3(bev)
        return bev


class BEVDecoderLayer(SimNN.Module):
    """
    Single BEVFormer decoder layer.

    Operation order:
      Object self-attn                          → LayerNorm
      Cross-attn (object queries ← BEV embed)  → LayerNorm
      FFN                                       → LayerNorm
    """

    def __init__(self, name, embed_dims, num_heads, feedforward_channels):
        super().__init__()
        self.name = name

        self.self_attn = MultiHeadAttn(f"{name}.self_attn", embed_dims, num_heads)
        self.cross_attn = MultiHeadAttn(f"{name}.cross_attn", embed_dims, num_heads)
        self.ffn = FFN(f"{name}.ffn", embed_dims, feedforward_channels)
        self.norm1 = F.LayerNorm(f"{name}.norm1", embed_dims)
        self.norm2 = F.LayerNorm(f"{name}.norm2", embed_dims)
        self.norm3 = F.LayerNorm(f"{name}.norm3", embed_dims)

        super().link_op2module()

    def __call__(self, query, bev_embed):
        """
        Args:
            query:     [B, num_query, d]
            bev_embed: [B, bev_h*bev_w, d]
        Returns:
            Updated object features [B, num_query, d]
        """
        q = query + self.self_attn(query, query, query)
        q = self.norm1(q)
        q = q + self.cross_attn(q, bev_embed, bev_embed)
        q = self.norm2(q)
        q = q + self.ffn(q)
        q = self.norm3(q)
        return q


# ============================================================================
# Main workload class
# ============================================================================


class BEVFormer(SimNN.Module):
    """
    BEVFormer detector – Polaris workload interface.

    Architecture pipeline:
      1. ResNet backbone  → [C2, C3, C4, C5]
      2. FPN neck         → [P0, P1, P2, P3]  (uniform embed_dims channels)
      3. Flatten/concat   → cam_feats  [B, nc*Σ(Hl·Wl), d]
      4. BEV encoder layers (self-attn + cam cross-attn + FFN) × N_enc
                          → bev_embed  [B, bev_h*bev_w, d]
      5. Detection decoder layers (self-attn + BEV cross-attn + FFN) × N_dec
                          → object features [B, num_query, d]
      6. Classification head → [B, num_query, num_classes]
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name

        # ---- hyperparameters ------------------------------------------
        self.num_cams = cfg.get("num_cams", 6)
        self.img_height = cfg.get("img_height", 256)
        self.img_width = cfg.get("img_width", 256)
        self.img_channels = cfg.get("img_channels", 3)
        self.bs = cfg.get("bs", 1)
        self.embed_dims = cfg.get("embed_dims", 256)
        self.num_classes = cfg.get("num_classes", 10)
        self.bev_h = cfg.get("bev_h", 50)
        self.bev_w = cfg.get("bev_w", 50)
        self.num_query = cfg.get("num_query", 900)
        self.num_encoder_layers = cfg.get("num_encoder_layers", 6)
        self.num_decoder_layers = cfg.get("num_decoder_layers", 6)
        self.num_heads = cfg.get("num_heads", 8)
        self.ffn_dim = cfg.get("ffn_dim", self.embed_dims * 2)
        self.backbone_layers = cfg.get("backbone_layers", [3, 4, 6, 3])
        self.backbone_channels = cfg.get("backbone_channels", [256, 512, 1024, 2048])
        self.num_feature_levels = cfg.get("num_feature_levels", 4)

        d = self.embed_dims
        nH = self.num_heads
        ffn = self.ffn_dim

        # ---- backbone -------------------------------------------------
        self.backbone = ResNetBackbone(
            f"{name}.backbone",
            {
                "img_channels": self.img_channels,
                "layers": self.backbone_layers,
            },
        )

        # ---- FPN neck -------------------------------------------------
        self.neck = FPN(
            f"{name}.neck",
            {
                "in_channels": self.backbone_channels,
                "out_channels": d,
                "num_outs": self.num_feature_levels,
            },
        )

        # ---- BEV encoder layers ---------------------------------------
        self.enc_layers = SimNN.ModuleList(
            [
                BEVEncoderLayer(f"{name}.enc{i}", d, nH, ffn)
                for i in range(self.num_encoder_layers)
            ]
        )

        # ---- Detection decoder layers ---------------------------------
        self.dec_layers = SimNN.ModuleList(
            [
                BEVDecoderLayer(f"{name}.dec{i}", d, nH, ffn)
                for i in range(self.num_decoder_layers)
            ]
        )

        # ---- Classification head  (Linear d → num_classes) -----------
        self.cls_head = SimNN.Linear(f"{name}.cls_head", d, self.num_classes)

        super().link_op2module()

    # ==================================================================
    # Polaris interface
    # ==================================================================

    def set_batch_size(self, new_bs):
        self.bs = new_bs

    def create_input_tensors(self):
        """
        Stacked multi-camera images: [B*num_cams, C, H, W]
        """
        self.input_tensors = {
            "img": F._from_shape(
                "img",
                [
                    self.bs * self.num_cams,
                    self.img_channels,
                    self.img_height,
                    self.img_width,
                ],
                is_param=False,
                np_dtype=np.float32,
            ),
        }

    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)

    def analytical_param_count(self):
        """Approximate parameter count for the full model."""
        d = self.embed_dims
        ffn = self.ffn_dim

        # --- Backbone ---
        total = self.img_channels * 64 * 7 * 7 + 64 * 4  # stem conv + BN
        in_ch = 64
        for nb, pl in zip(self.backbone_layers, [64, 128, 256, 512]):
            exp = Bottleneck.expansion
            out_c = pl * exp
            for bi in range(nb):
                ic = in_ch if bi == 0 else out_c
                mid = pl
                total += ic * mid + mid * 4  # 1×1 conv + BN
                total += mid * mid * 9 + mid * 4  # 3×3 conv + BN
                total += mid * out_c + out_c * 4  # 1×1 conv + BN
                if bi == 0 and (ic != out_c):
                    total += ic * out_c + out_c * 4  # downsample
            in_ch = out_c

        # --- FPN ---
        for ic in self.backbone_channels:
            total += ic * d + d  # lateral 1×1 conv + BN
            total += d * d * 9 + d  # fpn 3×3 conv + BN

        # --- Per transformer layer: 2×MHA (Q,K,V,O) + FFN (2-layer) + 3×LN ---
        per_attn = 4 * (d * d + d)  # w_q/k/v/o each [d,d] + bias
        per_ffn = d * ffn + ffn + ffn * d + d
        per_ln = 2 * d  # weight + bias
        per_layer = 2 * per_attn + per_ffn + 3 * per_ln

        total += self.num_encoder_layers * per_layer
        total += self.num_decoder_layers * per_layer

        # --- BEV queries + object queries + cls head ---
        total += self.bev_h * self.bev_w * d  # BEV query embedding
        total += self.num_query * d  # object query embedding
        total += d * self.num_classes + self.num_classes  # cls head weights + bias

        return total

    # ==================================================================
    # Forward pass
    # ==================================================================

    def __call__(self):
        """
        img [B*nc, C, H, W]
        → backbone → FPN → flatten cam feats [B, nc*Σ(Hl·Wl), d]
        → encoder layers  → bev_embed [B, HW_bev, d]
        → decoder layers  → object feats [B, nq, d]
        → cls head        → [B, nq, num_classes]
        """
        img = self.input_tensors["img"]  # [B*nc, C, H, W]
        batch = self.bs
        nc = self.num_cams
        d = self.embed_dims

        # ----------------------------------------------------------
        # 1. Backbone: [B*nc, C, H, W] → list of [B*nc, Ch, Hh, Wh]
        # ----------------------------------------------------------
        feats = self.backbone(img)  # [c2, c3, c4, c5]

        # ----------------------------------------------------------
        # 2. FPN: → list of [B*nc, d, Hl, Wl]
        # ----------------------------------------------------------
        fpn_feats = self.neck(feats)

        # ----------------------------------------------------------
        # 3. Flatten camera features to [B, nc*Σ(Hl·Wl), d]
        # ----------------------------------------------------------
        cam_feats_list = []
        for feat in fpn_feats:
            bnc, fd, fh, fw = feat.shape  # bnc == batch * nc
            # [B*nc, d, H, W] → [B*nc, H*W, d]
            feat_flat = feat.reshape(bnc, fd, fh * fw).transpose(1, 2)
            # [B*nc, H*W, d] → [B, nc*H*W, d]
            feat_flat = feat_flat.reshape(batch, nc * fh * fw, fd)
            cam_feats_list.append(feat_flat)

        cam_feats = (
            T.cat(cam_feats_list, dim=1)
            if len(cam_feats_list) > 1
            else cam_feats_list[0]
        )
        # cam_feats: [B, nc*Σ(Hl·Wl), d]

        # ----------------------------------------------------------
        # 4. BEV queries (learnable): [B, bev_h*bev_w, d]
        # ----------------------------------------------------------
        bev_queries = F._from_shape(
            f"{self.name}.bev_queries",
            [batch, self.bev_h * self.bev_w, d],
            is_param=True,
            np_dtype=np.float32,
        )

        # ----------------------------------------------------------
        # 5. BEVFormer encoder
        #    Each layer: BEV self-attn → cam cross-attn → FFN
        # ----------------------------------------------------------
        bev_embed = bev_queries
        for layer in self.enc_layers:
            bev_embed = layer(bev_embed, cam_feats)
        # bev_embed: [B, bev_h*bev_w, d]

        # ----------------------------------------------------------
        # 6. Object queries (learnable): [B, num_query, d]
        # ----------------------------------------------------------
        object_queries = F._from_shape(
            f"{self.name}.object_queries",
            [batch, self.num_query, d],
            is_param=True,
            np_dtype=np.float32,
        )

        # ----------------------------------------------------------
        # 7. Detection decoder
        #    Each layer: object self-attn → BEV cross-attn → FFN
        # ----------------------------------------------------------
        q = object_queries
        for layer in self.dec_layers:
            q = layer(q, bev_embed)
        # q: [B, num_query, d]

        # ----------------------------------------------------------
        # 8. Classification head → [B, num_query, num_classes]
        # ----------------------------------------------------------
        cls_logits = self.cls_head(q)
        return cls_logits


# ============================================================================
# Standalone driver (mirrors basicresnet.py pattern)
# ============================================================================


def run_standalone(outdir: str = ".") -> None:
    bevformer_cfgs = {
        "bevformer_tiny": {
            "num_cams": 6,
            "img_height": 256,
            "img_width": 256,
            "img_channels": 3,
            "bs": 1,
            "embed_dims": 128,
            "num_classes": 10,
            "bev_h": 25,
            "bev_w": 25,
            "num_query": 300,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "num_heads": 8,
            "ffn_dim": 256,
            "backbone_layers": [2, 2, 2, 2],
            "backbone_channels": [256, 512, 1024, 2048],
            "num_feature_levels": 4,
        },
        "bevformer_small": {
            "num_cams": 6,
            "img_height": 512,
            "img_width": 512,
            "img_channels": 3,
            "bs": 1,
            "embed_dims": 256,
            "num_classes": 10,
            "bev_h": 50,
            "bev_w": 50,
            "num_query": 900,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "num_heads": 8,
            "ffn_dim": 512,
            "backbone_layers": [3, 4, 6, 3],
            "backbone_channels": [256, 512, 1024, 2048],
            "num_feature_levels": 4,
        },
    }

    for k, v in bevformer_cfgs.items():
        logger.info(f"Creating BEVFormer({k})...")
        model = BEVFormer(k, v)
        model.create_input_tensors()
        y = model()
        logger.debug("Input:  ", model.input_tensors["img"].shape)
        logger.debug("Output: ", y.shape)
        gg = model.get_forward_graph()
        logger.info("Dumping ONNX...")
        gg.graph2onnx(f"{outdir}/{k}.onnx", do_model_check=True)
        logger.info("-" * 40, "\n")


if __name__ == "__main__":
    run_standalone()
