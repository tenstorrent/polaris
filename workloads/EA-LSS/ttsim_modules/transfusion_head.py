#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of TransformerDecoderLayer and TransFusionHead.

Original file: mmdet3d/models/dense_heads/transfusion_head.py
  - TransformerDecoderLayer  (class at ~line 45)
  - TransFusionHead          (class at ~line 500)

Both classes are combined in this single file because they appear in the same
source file. Existing sub-modules (MultiheadAttention, PositionEmbeddingLearned,
FFN) are imported from their respective ttsim_modules.

-------------------------------------------------------------------------------
TransformerDecoderLayer
    Transformer decoder block with optional self-attention.

    cross_only=False:
        self_attn        MHA(d, nhead)
        multihead_attn   MHA(d, nhead)
        linear1,2        MLP: Linear(d→ffn) + act + Linear(ffn→d)
        norm1,2,3        LayerNorm(d) × 3
        self_posembed    PositionEmbeddingLearned(2, d)
        cross_posembed   PositionEmbeddingLearned(2, d)

    cross_only=True:
        omit self_attn and norm1

    Default params (d=128, nhead=8, ffn=256, cross_only=False): 233,088

TransFusionHead
    Full detection head combining BEV feature, transformer decoder,
    and per-attribute prediction heads.

    Default EA-LSS config (fuse_img=False, initialize_by_heatmap=True):
        in_channels=1024, hidden=128, num_classes=10,
        num_decoder_layers=1, num_proposals=200

    Key sub-modules:
        shared_conv        Conv2d(in_channels, hidden, 3)
        heatmap_head       ConvModule2d(hidden,hidden,3) + Conv2d(hidden,classes,3)
        class_encoding     Conv1d(classes, hidden, 1)   [if initialize_by_heatmap]
        decoder            ModuleList of TransformerDecoderLayer
        prediction_heads   ModuleList of FFN

No torch / mmcv imports.
"""

import os
import sys
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import SimOpHandle, _from_shape, _from_data

_ealss_root = os.path.abspath(os.path.join(current_dir, ".."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim_modules.multihead_attention import MultiheadAttention
from ttsim_modules.position_embedding_learned import PositionEmbeddingLearned
from ttsim_modules.ffn import FFN
from ttsim_modules.swin_transformer import _LinearModule


# ---------------------------------------------------------------------------
# TransformerDecoderLayer
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(SimNN.Module):
    """
    One decoder layer as used by TransFusionHead.

    Call signature:
        out = layer(query, key, query_pos, key_pos)
        query      (SimTensor): [B, C, Pq]
        key        (SimTensor): [B, C, Pk]
        query_pos  (SimTensor): [B, Pq, 2]
        key_pos    (SimTensor): [B, Pk, 2]
        out        (SimTensor): [B, C, Pq]

    Args:
        name (str): Module name.
        d_model (int): Embedding dimension. Default: 128.
        nhead (int): Number of attention heads. Default: 8.
        dim_feedforward (int): FFN hidden dim. Default: 256.
        dropout (float): Dropout (ignored in TTSim). Default: 0.1.
        activation (str): Activation function name. Default: 'relu'.
        cross_only (bool): If True, omit self-attention and norm1. Default: False.
    """

    def __init__(
        self,
        name: str,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        cross_only: bool = False,
    ):
        super().__init__()
        self.name    = name
        self.d_model = d_model
        self.nhead   = nhead
        self.ffn_dim = dim_feedforward
        self.cross_only = cross_only

        if not cross_only:
            self.self_attn = MultiheadAttention(name + ".self_attn", d_model, nhead)

        self.multihead_attn = MultiheadAttention(name + ".mha", d_model, nhead)

        # Linear FFN
        self.linear1 = _LinearModule(name + ".lin1", d_model, dim_feedforward, bias=True)
        self.linear2 = _LinearModule(name + ".lin2", dim_feedforward, d_model, bias=True)

        # LayerNorms
        if not cross_only:
            self.norm1 = F.LayerNorm(name + ".norm1", d_model)
        self.norm2 = F.LayerNorm(name + ".norm2", d_model)
        self.norm3 = F.LayerNorm(name + ".norm3", d_model)

        # Position embeddings
        self.self_posembed  = PositionEmbeddingLearned(name + ".spel", input_channel=2, num_pos_feats=d_model)
        self.cross_posembed = PositionEmbeddingLearned(name + ".cpel", input_channel=2, num_pos_feats=d_model)

        # ── Register all dynamic ops so link_op2module tracks them and their
        # output tensors are auto-stored in _tensors when fired (polaris graph). ──
        self.op_tr_q      = SimOpHandle(name + ".tr_q",      "Transpose", params=[], ipos=[0], perm=[2,0,1])
        self.op_tr_k      = SimOpHandle(name + ".tr_k",      "Transpose", params=[], ipos=[0], perm=[2,0,1])
        self.op_tr_qp     = SimOpHandle(name + ".tr_qp",     "Transpose", params=[], ipos=[0], perm=[2,0,1])
        self.op_tr_kp     = SimOpHandle(name + ".tr_kp",     "Transpose", params=[], ipos=[0], perm=[2,0,1])
        self.op_q_add_pos = SimOpHandle(name + ".q_add_pos", "Add",       params=[], ipos=[0,1])
        self.op_k_add_pos = SimOpHandle(name + ".k_add_pos", "Add",       params=[], ipos=[0,1])
        if not cross_only:
            self.op_sa_res     = SimOpHandle(name + ".sa_res",     "Add", params=[], ipos=[0,1])
            self.op_q_add_pos2 = SimOpHandle(name + ".q_add_pos2", "Add", params=[], ipos=[0,1])
        self.op_ca_res    = SimOpHandle(name + ".ca_res",  "Add",       params=[], ipos=[0,1])
        self.op_ffn_act   = F.Relu(name + ".ffn_act")
        self.op_ffn_res   = SimOpHandle(name + ".ffn_res", "Add",       params=[], ipos=[0,1])
        self.op_tr_out    = SimOpHandle(name + ".tr_out",  "Transpose", params=[], ipos=[0], perm=[1,2,0])

        super().link_op2module()

    def __call__(self, query, key, query_pos, key_pos):
        """
        Args:
            query     [B, C, Pq]
            key       [B, C, Pk]
            query_pos [B, Pq, 2]
            key_pos   [B, Pk, 2]
        Returns:
            SimTensor [B, C, Pq]
        """
        B, C, Pq = query.shape
        _, _, Pk = key.shape

        # PositionEmbeddingLearned: [B, P, 2] -> [B, C, P]
        q_pos_embed = self.self_posembed(query_pos)   # [B, C, Pq]
        k_pos_embed = self.cross_posembed(key_pos)    # [B, C, Pk]

        # Permute query/key to [P, B, C] for MHA
        q_pbc  = self.op_tr_q(query)                  # [Pq, B, C]
        k_pbc  = self.op_tr_k(key)                    # [Pk, B, C]
        qp_pbc = self.op_tr_qp(q_pos_embed)           # [Pq, B, C]
        kp_pbc = self.op_tr_kp(k_pos_embed)           # [Pk, B, C]

        # Add positional embeddings
        q_with_pos = self.op_q_add_pos(q_pbc, qp_pbc)
        k_with_pos = self.op_k_add_pos(k_pbc, kp_pbc)

        if not self.cross_only:
            # Self-attention on query
            sa_out, _ = self.self_attn(q_with_pos, q_with_pos, q_with_pos)
            q_pbc = self.op_sa_res(q_pbc, sa_out)
            q_pbc = self.norm1(q_pbc)
            q_with_pos = self.op_q_add_pos2(q_pbc, qp_pbc)

        # Cross-attention
        ca_out, _ = self.multihead_attn(q_with_pos, k_with_pos, k_with_pos)
        q_pbc = self.op_ca_res(q_pbc, ca_out)
        q_pbc = self.norm2(q_pbc)

        # FFN
        ffn_out = self.linear1(q_pbc)
        ffn_out = self.op_ffn_act(ffn_out)
        ffn_out = self.linear2(ffn_out)
        q_pbc = self.op_ffn_res(q_pbc, ffn_out)
        q_pbc = self.norm3(q_pbc)

        # Permute back [P, B, C] -> [B, C, P]
        out = self.op_tr_out(q_pbc)                   # [B, C, Pq]
        return out

    def analytical_param_count(self, lvl: int = 0) -> int:
        p = 0
        if not self.cross_only:
            p += self.self_attn.analytical_param_count()
        p += self.multihead_attn.analytical_param_count()
        p += self.linear1.analytical_param_count()
        p += self.linear2.analytical_param_count()
        # LayerNorms: 2 * d_model params each
        if not self.cross_only:
            p += 2 * self.d_model   # norm1
        p += 2 * self.d_model       # norm2
        p += 2 * self.d_model       # norm3
        p += self.self_posembed.analytical_param_count()
        p += self.cross_posembed.analytical_param_count()
        return p


# ---------------------------------------------------------------------------
# TransFusionHead
# ---------------------------------------------------------------------------

class TransFusionHead(SimNN.Module):
    """
    Full TransFusion detection head (fuse_img=False case).

    Args:
        name (str): Module name.
        in_channels (int): Input BEV feature channels. Default: 1024.
        hidden_channel (int): Hidden dim for queries. Default: 128.
        num_classes (int): Number of foreground classes. Default: 10.
        num_proposals (int): Number of object proposals. Default: 200.
        num_decoder_layers (int): Decoder depth. Default: 1.
        num_heads (int): MHA heads. Default: 8.
        ffn_channel (int): Decoder FFN dim. Default: 256.
        common_heads (dict): Per-attribute prediction heads.
        num_heatmap_convs (int): Conv layers in heatmap head. Default: 2.
        initialize_by_heatmap (bool): Init queries via heatmap. Default: True.
        fuse_img (bool): Image fusion (not modelled here). Default: False.
    """

    def __init__(
        self,
        name: str,
        in_channels: int = 1024,
        hidden_channel: int = 128,
        num_classes: int = 10,
        num_proposals: int = 200,
        num_decoder_layers: int = 1,
        num_heads: int = 8,
        ffn_channel: int = 256,
        common_heads: dict = None,
        num_heatmap_convs: int = 2,
        initialize_by_heatmap: bool = True,
        fuse_img: bool = False,
        dropout: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.in_channels     = in_channels
        self.hidden_channel  = hidden_channel
        self.num_classes     = num_classes
        self.num_proposals   = num_proposals
        self.num_decoder_layers = num_decoder_layers
        self.num_heads       = num_heads
        self.ffn_channel     = ffn_channel
        self.initialize_by_heatmap = initialize_by_heatmap
        self.fuse_img        = fuse_img

        if common_heads is None:
            common_heads = dict(center=(2, 2), height=(1, 2),
                                dim=(3, 2), rot=(2, 2), vel=(2, 2))
        self.common_heads = common_heads

        # --- shared_conv: Conv2d(in_channels, hidden, 3, pad=1, bias=True) ---
        # bias='auto' in source → True when no norm follows
        self.shared_conv = F.Conv2d(
            name + ".shared_conv",
            in_channels, hidden_channel,
            kernel_size=3, padding=1, bias=True,
        )

        # --- heatmap_head and class_encoding ---
        if initialize_by_heatmap:
            # heatmap_head = ConvModule(hid,hid,3,BN) + Conv2d(hid,classes,3,bias)
            self.hm_conv0 = F.Conv2d(
                name + ".hm_conv0",
                hidden_channel, hidden_channel,
                kernel_size=3, padding=1, bias=False,    # BN follows → no bias
            )
            self.hm_bn0 = F.BatchNorm2d(name + ".hm_bn0", hidden_channel)
            # Register hm_relu so its output tensor is tracked for polaris graph.
            self.hm_relu = F.Relu(name + ".hm_relu")
            self.hm_conv1 = F.Conv2d(
                name + ".hm_conv1",
                hidden_channel, num_classes,
                kernel_size=3, padding=1, bias=True,
            )
            # class_encoding: Conv1d(num_classes, hidden, 1, bias=True)
            from ttsim_modules.mlp import ConvModule1d
            self.class_enc = ConvModule1d(
                name + ".class_enc",
                in_channels=num_classes,
                out_channels=hidden_channel,
                kernel_size=1,
                with_bn=False, with_relu=False, bias=True,
            )
        else:
            # Learnable query features: [1, hidden, num_proposals]
            # and query positions: [1, num_proposals, 2]
            self.query_feat = _from_shape(name + ".query_feat",
                                          [1, hidden_channel, num_proposals], is_param=True)
            self.query_pos  = _from_shape(name + ".query_pos",
                                          [1, num_proposals, 2], is_param=True)

        # --- decoder ---
        self.num_decoder_total = num_decoder_layers
        for i in range(num_decoder_layers):
            setattr(self, f"decoder_{i}",
                    TransformerDecoderLayer(
                        name + f".dec{i}",
                        d_model=hidden_channel,
                        nhead=num_heads,
                        dim_feedforward=ffn_channel,
                        dropout=dropout,
                        activation=activation,
                        cross_only=False,
                    ))

        # --- prediction_heads ---
        for i in range(num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads["heatmap"] = (num_classes, num_heatmap_convs)
            setattr(self, f"pred_head_{i}",
                    FFN(name + f".ph{i}",
                        in_channels=hidden_channel,
                        heads=heads,
                        head_conv=hidden_channel,
                        final_kernel=1))

        super().link_op2module()

    def __call__(self, x):
        """
        Args:
            x (SimTensor): [B, in_channels, H, W] BEV feature map.
        Returns:
            dict: prediction dict from the last decoder layer's FFN,
                  e.g. {'center': [B,2,P], 'heatmap': [B,10,P], ...}
        """
        B, C, H, W = x.shape
        P = self.num_proposals

        # Shared conv
        x = self.shared_conv(x)                           # [B, hidden, H, W]

        if self.initialize_by_heatmap:
            # Heatmap branch
            hm = self.hm_conv0(x)
            hm = self.hm_bn0(hm)
            hm = self.hm_relu(hm)                         # registered op
            hm = self.hm_conv1(hm)                        # [B, num_classes, H, W]

            # Top-K selection: approximate as identity (shape only) → [B, num_classes, P]
            top_hm = _from_shape(self.name + ".top_hm", [B, self.num_classes, P])
            self._tensors[top_hm.name] = top_hm           # make visible to graph

            # Class encoding: [B, num_classes, P] -> [B, hidden, P]
            query = self.class_enc(top_hm)                # [B, hidden, P]
        else:
            query = _from_shape(self.name + ".q0", [B, self.hidden_channel, P])
            self._tensors[query.name] = query             # make visible to graph

        # BEV spatial features flattened: [B, hidden, H*W]
        # Approximate: just use a proxy [B, hidden, H*W]
        bev_feat = _from_shape(self.name + ".bev_feat", [B, self.hidden_channel, H * W])
        self._tensors[bev_feat.name] = bev_feat           # make visible to graph

        # Query and key positional info
        query_pos = _from_shape(self.name + ".qpos", [B, P, 2])
        self._tensors[query_pos.name] = query_pos         # make visible to graph
        key_pos   = _from_shape(self.name + ".kpos", [B, H * W, 2])
        self._tensors[key_pos.name]   = key_pos           # make visible to graph

        # Decoder loops
        last_preds = None
        for i in range(self.num_decoder_total):
            dec = getattr(self, f"decoder_{i}")
            query = dec(query, bev_feat, query_pos, key_pos)  # [B, hidden, P]

            ph = getattr(self, f"pred_head_{i}")
            last_preds = ph(query)                             # dict

        return last_preds

    def analytical_param_count(self, lvl: int = 0) -> int:
        # shared_conv: Conv2d(in_ch, hid, 3) with bias
        p = self.in_channels * self.hidden_channel * 9 + self.hidden_channel

        if self.initialize_by_heatmap:
            # hm_conv0 (no bias) + bn0 (scale+bias) + hm_conv1 (bias)
            p += self.hidden_channel * self.hidden_channel * 9          # hm_conv0
            p += 2 * self.hidden_channel                                 # hm_bn0
            p += self.hidden_channel * self.num_classes * 9 + self.num_classes  # hm_conv1
            # class_enc: Conv1d(classes, hidden, 1) + bias
            p += self.num_classes * self.hidden_channel + self.hidden_channel
        else:
            # query_feat + query_pos
            p += self.hidden_channel * self.num_proposals                # query_feat
            p += self.num_proposals * 2                                  # query_pos

        for i in range(self.num_decoder_total):
            dec = getattr(self, f"decoder_{i}")
            p += dec.analytical_param_count(lvl + 1)

        for i in range(self.num_decoder_total):
            ph = getattr(self, f"pred_head_{i}")
            p += ph.analytical_param_count(lvl + 1)

        return p
