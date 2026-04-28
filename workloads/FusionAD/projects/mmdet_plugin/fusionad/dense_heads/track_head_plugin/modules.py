
# =============================================================================
# ORIGINAL TORCH CODE (from FusionAD)
# Source: FusionAD/projects/mmdet3d_plugin/fusionad/dense_heads/track_head_plugin/modules.py
# =============================================================================
# import torch
# import torch.nn.functional as F
# from torch import nn
# from .track_instance import Instances
#
# # MemoryBank
# class MemoryBank(nn.Module):
#
#     def __init__(self,
#                  args,
#                  dim_in, hidden_dim, dim_out,
#                  ):
#         super().__init__()
#         self._build_layers(args, dim_in, hidden_dim, dim_out)
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def _build_layers(self, args, dim_in, hidden_dim, dim_out):
#         self.save_thresh = args['memory_bank_score_thresh']
#         self.save_period = 3
#         self.max_his_length = args['memory_bank_len']
#
#         self.save_proj = nn.Linear(dim_in, dim_in)
#
#         self.temporal_attn = nn.MultiheadAttention(dim_in, 8, dropout=0)
#         self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)
#         self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)
#         self.temporal_norm1 = nn.LayerNorm(dim_in)
#         self.temporal_norm2 = nn.LayerNorm(dim_in)
#
#     def update(self, track_instances):
#         embed = track_instances.output_embedding[:, None]  #( N, 1, 256)
#         scores = track_instances.scores
#         mem_padding_mask = track_instances.mem_padding_mask
#         device = embed.device
#
#         save_period = track_instances.save_period
#         if self.training:
#             saved_idxes = scores > 0
#         else:
#             saved_idxes = (save_period == 0) & (scores > self.save_thresh)
#             # saved_idxes = (save_period == 0)
#             save_period[save_period > 0] -= 1
#             save_period[saved_idxes] = self.save_period
#
#         saved_embed = embed[saved_idxes]
#         if len(saved_embed) > 0:
#             prev_embed = track_instances.mem_bank[saved_idxes]
#             save_embed = self.save_proj(saved_embed)
#             mem_padding_mask[saved_idxes] = torch.cat([mem_padding_mask[saved_idxes, 1:], torch.zeros((len(saved_embed), 1), dtype=torch.bool, device=device)], dim=1)
#             track_instances.mem_bank = track_instances.mem_bank.clone()
#             track_instances.mem_bank[saved_idxes] = torch.cat([prev_embed[:, 1:], save_embed], dim=1)
#
#     def _forward_temporal_attn(self, track_instances):
#         if len(track_instances) == 0:
#             return track_instances
#
#         key_padding_mask = track_instances.mem_padding_mask
#
#         valid_idxes = key_padding_mask[:, -1] == 0
#         embed = track_instances.output_embedding[valid_idxes]  # (n, 256)
#
#         if len(embed) > 0:
#             prev_embed = track_instances.mem_bank[valid_idxes]
#             key_padding_mask = key_padding_mask[valid_idxes]
#             embed2 = self.temporal_attn(
#                 embed[None],                  # (num_track, dim) to (1, num_track, dim)
#                 prev_embed.transpose(0, 1),   # (num_track, mem_len, dim) to (mem_len, num_track, dim)
#                 prev_embed.transpose(0, 1),
#                 key_padding_mask=key_padding_mask,
#             )[0][0]
#
#             embed = self.temporal_norm1(embed + embed2)
#             embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))
#             embed = self.temporal_norm2(embed + embed2)
#             track_instances.output_embedding = track_instances.output_embedding.clone()
#             track_instances.output_embedding[valid_idxes] = embed
#
#         return track_instances
#
#     def forward_temporal_attn(self, track_instances):
#         return self._forward_temporal_attn(track_instances)
#
#     def forward(self, track_instances: Instances, update_bank=True) -> Instances:
#         track_instances = self._forward_temporal_attn(track_instances)
#         if update_bank:
#             self.update(track_instances)
#         return track_instances
#
#
# # QIM
# class QueryInteractionBase(nn.Module):
#
#     def __init__(self, args, dim_in, hidden_dim, dim_out):
#         super().__init__()
#         self.args = args
#         self._build_layers(args, dim_in, hidden_dim, dim_out)
#         self._reset_parameters()
#
#     def _build_layers(self, args, dim_in, hidden_dim, dim_out):
#         raise NotImplementedError()
#
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def _select_active_tracks(self, data: dict) -> Instances:
#         raise NotImplementedError()
#
#     def _update_track_embedding(self, track_instances):
#         raise NotImplementedError()
#
# class QueryInteractionModule(QueryInteractionBase):
#
#     def __init__(self, args, dim_in, hidden_dim, dim_out):
#         super().__init__(args, dim_in, hidden_dim, dim_out)
#         self.random_drop = args["random_drop"]
#         self.fp_ratio = args["fp_ratio"]
#         self.update_query_pos = args["update_query_pos"]
#
#     def _build_layers(self, args, dim_in, hidden_dim, dim_out):
#         dropout = args["merger_dropout"]
#
#         self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
#         self.linear1 = nn.Linear(dim_in, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(hidden_dim, dim_in)
#
#         if args["update_query_pos"]:
#             self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
#             self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
#             self.dropout_pos1 = nn.Dropout(dropout)
#             self.dropout_pos2 = nn.Dropout(dropout)
#             self.norm_pos = nn.LayerNorm(dim_in)
#
#         self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
#         self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
#         self.dropout_feat1 = nn.Dropout(dropout)
#         self.dropout_feat2 = nn.Dropout(dropout)
#         self.norm_feat = nn.LayerNorm(dim_in)
#
#         self.norm1 = nn.LayerNorm(dim_in)
#         self.norm2 = nn.LayerNorm(dim_in)
#
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = F.relu
#
#     def _update_track_embedding(self, track_instances: Instances) -> Instances:
#         if len(track_instances) == 0:
#             return track_instances
#         dim = track_instances.query.shape[1]
#         out_embed = track_instances.output_embedding
#         query_pos = track_instances.query[:, :dim // 2]
#         query_feat = track_instances.query[:, dim // 2:]
#         q = k = query_pos + out_embed
#
#         # attention
#         tgt = out_embed
#         tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:,
#                                                                              0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)
#
#         # ffn
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)
#
#         if self.update_query_pos:
#             # ffn: linear_pos2
#             query_pos2 = self.linear_pos2(
#                 self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
#             query_pos = query_pos + self.dropout_pos2(query_pos2)
#             query_pos = self.norm_pos(query_pos)
#             track_instances.query[:, :dim // 2] = query_pos
#
#         query_feat2 = self.linear_feat2(
#             self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
#         query_feat = query_feat + self.dropout_feat2(query_feat2)
#         query_feat = self.norm_feat(query_feat)
#         track_instances.query[:, dim // 2:] = query_feat
#         # track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
#         # update ref_pts using track_instances.pred_boxes
#         return track_instances
#
#     def _random_drop_tracks(self, track_instances: Instances) -> Instances:
#         drop_probability = self.random_drop
#         if drop_probability > 0 and len(track_instances) > 0:
#             keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
#             track_instances = track_instances[keep_idxes]
#         return track_instances
#
#     def _add_fp_tracks(self, track_instances: Instances,
#                        active_track_instances: Instances) -> Instances:
#         """
#         self.fp_ratio is used to control num(add_fp) / num(active)
#         """
#         inactive_instances = track_instances[track_instances.obj_idxes < 0]
#
#         # add fp for each active track in a specific probability.
#         fp_prob = torch.ones_like(
#             active_track_instances.scores) * self.fp_ratio
#         selected_active_track_instances = active_track_instances[
#             torch.bernoulli(fp_prob).bool()]
#         num_fp = len(selected_active_track_instances)
#
#         if len(inactive_instances) > 0 and num_fp > 0:
#             if num_fp >= len(inactive_instances):
#                 fp_track_instances = inactive_instances
#             else:
#                 # randomly select num_fp from inactive_instances
#                 # fp_indexes = np.random.permutation(len(inactive_instances))
#                 # fp_indexes = fp_indexes[:num_fp]
#                 # fp_track_instances = inactive_instances[fp_indexes]
#
#                 # v2: select the fps with top scores rather than random selection
#                 fp_indexes = torch.argsort(inactive_instances.scores)[-num_fp:]
#                 fp_track_instances = inactive_instances[fp_indexes]
#
#             merged_track_instances = Instances.cat(
#                 [active_track_instances, fp_track_instances])
#             return merged_track_instances
#
#         return active_track_instances
#
#     def _select_active_tracks(self, data: dict) -> Instances:
#         track_instances: Instances = data["track_instances"]
#         if self.training:
#             active_idxes = (track_instances.obj_idxes >=
#                             0) & (track_instances.iou > 0.5)
#             active_track_instances = track_instances[active_idxes]
#             # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
#             active_track_instances = self._random_drop_tracks(
#                 active_track_instances)
#             if self.fp_ratio > 0:
#                 active_track_instances = self._add_fp_tracks(
#                     track_instances, active_track_instances)
#         else:
#             active_track_instances = track_instances[
#                 track_instances.obj_idxes >= 0]
#
#         return active_track_instances
#
#     def forward(self, data) -> Instances:
#         active_track_instances = self._select_active_tracks(data)
#         active_track_instances = self._update_track_embedding(
#             active_track_instances)
#         init_track_instances: Instances = data["init_track_instances"]
#         merged_track_instances = Instances.cat(
#             [init_track_instances, active_track_instances])
#         return merged_track_instances
# =============================================================================
# END OF ORIGINAL TORCH CODE
# =============================================================================


#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of MemoryBank and QueryInteractionModule for FusionAD tracking.

Inference-only conversion.  Training-specific methods and runtime tracking
orchestration (update, _select_active_tracks, _random_drop_tracks,
_add_fp_tracks) are omitted — they operate outside the ONNX graph.

Original: projects/mmdet3d_plugin/fusionad/dense_heads/track_head_plugin/modules.py

Classes:
  - MemoryBank              : Temporal memory bank with multi-head attention + FFN.
  - QueryInteractionModule  : Self-attention + FFN for track query refinement.
"""

import sys
import os
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))

# Add fusionad directory so "from modules.xxx" imports resolve
fusionad_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if fusionad_dir not in sys.path:
    sys.path.insert(0, fusionad_dir)

# Add polaris root for ttsim
polaris_root = os.path.abspath(
    os.path.join(current_dir, '..', '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.multihead_attention import MultiheadAttention
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.builder_utils import LayerNorm


# ======================================================================
# MemoryBank
# ======================================================================

class MemoryBank(SimNN.Module):
    """
    TTSim implementation of MemoryBank temporal attention.

    The forward path applies multi-head attention between current track
    embeddings (query) and the memory bank (key/value), followed by a
    two-layer FFN with residual connections and layer normalization.

    The ``update()`` method (between-frame bookkeeping for the memory bank)
    is runtime-only and not part of the ONNX graph.

    Original architecture (per-track):
        1. MHA(embed, prev_embed, prev_embed, key_padding_mask)
        2. Residual + LayerNorm
        3. FC1 -> ReLU -> FC2
        4. Residual + LayerNorm

    Args:
        name (str): Module name for TTSim graph.
        dim_in (int): Input / embedding dimension.
        hidden_dim (int): Hidden dimension for the FFN.
        dim_out (int): Output dimension (unused, kept for API compat).
        num_heads (int): Number of attention heads. Default: 8.
        save_thresh (float): Score threshold for memory update. Default: 0.0.
        save_period (int): Frames between saves. Default: 3.
        max_his_length (int): Maximum memory bank length. Default: 4.
    """

    def __init__(self, name, dim_in, hidden_dim, dim_out,
                 num_heads=8,
                 save_thresh=0.0, save_period=3, max_his_length=4):
        super().__init__()
        self.name = name
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.save_thresh = save_thresh
        self.save_period = save_period
        self.max_his_length = max_his_length

        # -- Linear projection for save (runtime update, not in graph) --
        self.save_proj = SimNN.Linear(
            f'{name}.save_proj', in_features=dim_in, out_features=dim_in)

        # -- Temporal multi-head attention --
        self.temporal_attn = MultiheadAttention(
            f'{name}.temporal_attn',
            embed_dims=dim_in, num_heads=num_heads,
            attn_drop=0.0, proj_drop=0.0, batch_first=False)

        # -- FFN --
        self.temporal_fc1 = SimNN.Linear(
            f'{name}.temporal_fc1', in_features=dim_in, out_features=hidden_dim)
        self.temporal_fc2 = SimNN.Linear(
            f'{name}.temporal_fc2', in_features=hidden_dim, out_features=dim_in)

        # -- Layer norms --
        self.temporal_norm1 = LayerNorm(f'{name}.temporal_norm1', dim_in)
        self.temporal_norm2 = LayerNorm(f'{name}.temporal_norm2', dim_in)

        # -- Pre-create ops --
        self.unsqueeze_q = F.Unsqueeze(f'{name}.unsqueeze_q')
        self.unsq_dim = F._from_data(
            f'{name}.unsq_dim', np.array([0]), is_const=True)

        self.transpose_kv = F.Transpose(
            f'{name}.transpose_kv', perm=[1, 0, 2])

        self.reshape_attn = F.Reshape(f'{name}.reshape_attn')

        self.add_res1 = F.Add(f'{name}.add_res1')
        self.relu = F.Relu(f'{name}.relu')
        self.add_res2 = F.Add(f'{name}.add_res2')

        super().link_op2module()

    def __call__(self, embed, prev_embed, key_padding_mask=None):
        """
        Forward temporal attention.

        Args:
            embed: (N, dim) current track embeddings.
            prev_embed: (N, mem_len, dim) memory bank embeddings.
            key_padding_mask: (N, mem_len) boolean padding mask (optional).

        Returns:
            out: (N, dim) updated embeddings.
        """
        N = embed.shape[0]
        dim = embed.shape[1]

        # query: (N, dim) -> unsqueeze axis=0 -> (1, N, dim) for MHA seq-first
        q = self.unsqueeze_q(embed, self.unsq_dim)
        setattr(self, q.name, q)

        # key/value: (N, mem_len, dim) -> transpose -> (mem_len, N, dim)
        kv = self.transpose_kv(prev_embed)
        setattr(self, kv.name, kv)

        # Temporal multi-head attention (batch_first=False)
        attn_out = self.temporal_attn(
            query=q, key=kv, value=kv,
            key_padding_mask=key_padding_mask)
        setattr(self, attn_out.name, attn_out)

        # Reshape: (1, N, dim) -> (N, dim)
        self.attn_out_shape = F._from_data(
            f'{self.name}.attn_out_shape',
            np.array([N, dim], dtype=np.int64), is_const=True)
        attn_out = self.reshape_attn(attn_out, self.attn_out_shape)
        setattr(self, attn_out.name, attn_out)

        # Residual + LayerNorm 1
        out = self.add_res1(embed, attn_out)
        setattr(self, out.name, out)
        out = self.temporal_norm1(out)
        setattr(self, out.name, out)

        # FFN: fc1 -> relu -> fc2
        ffn = self.temporal_fc1(out)
        setattr(self, ffn.name, ffn)
        ffn = self.relu(ffn)
        setattr(self, ffn.name, ffn)
        ffn = self.temporal_fc2(ffn)
        setattr(self, ffn.name, ffn)

        # Residual + LayerNorm 2
        out = self.add_res2(out, ffn)
        setattr(self, out.name, out)
        out = self.temporal_norm2(out)
        setattr(self, out.name, out)

        return out

    def analytical_param_count(self, lvl=0):
        """Total learnable parameters in MemoryBank."""
        indent = "  " * lvl
        total = 0

        # save_proj: dim_in * dim_in + dim_in
        sp = self.dim_in * self.dim_in + self.dim_in
        total += sp
        if lvl >= 2:
            logger.debug(f"{indent}  save_proj: {sp:,}")

        # temporal_attn: 4 projections (q,k,v,out) each dim*dim+dim
        mha = self.temporal_attn.analytical_param_count(lvl=max(0, lvl - 1))
        total += mha
        if lvl >= 2:
            logger.debug(f"{indent}  temporal_attn: {mha:,}")

        # temporal_fc1: dim_in * hidden_dim + hidden_dim
        fc1 = self.dim_in * self.hidden_dim + self.hidden_dim
        total += fc1
        if lvl >= 2:
            logger.debug(f"{indent}  temporal_fc1: {fc1:,}")

        # temporal_fc2: hidden_dim * dim_in + dim_in
        fc2 = self.hidden_dim * self.dim_in + self.dim_in
        total += fc2
        if lvl >= 2:
            logger.debug(f"{indent}  temporal_fc2: {fc2:,}")

        # LayerNorms: each has weight + bias = 2 * dim_in
        ln = 2 * self.dim_in * 2  # two layer norms
        total += ln
        if lvl >= 2:
            logger.debug(f"{indent}  layer_norms (x2): {ln:,}")

        if lvl >= 1:
            logger.debug(f"{indent}Total MemoryBank params: {total:,}")

        return total


# ======================================================================
# QueryInteractionModule
# ======================================================================

class QueryInteractionModule(SimNN.Module):
    """
    TTSim implementation of QueryInteractionModule (inference path).

    Refines track query embeddings through self-attention and feed-forward
    networks.  Runtime track selection/filtering is handled outside the
    ONNX graph by the orchestrator.

    Forward path:
        1. Split query -> query_pos, query_feat
        2. q = k = query_pos + output_embedding
        3. Self-attention(q, k, v=output_embedding)
        4. Residual + LayerNorm
        5. FFN + Residual + LayerNorm
        6. [Optional] Pos FFN + Residual + LayerNorm
        7. Feat FFN + Residual + LayerNorm
        8. Concat(query_pos, query_feat) -> updated query

    Args:
        name (str): Module name for TTSim graph.
        dim_in (int): Input / embedding dimension.
        hidden_dim (int): Hidden dimension for FFN layers.
        dim_out (int): Output dimension (unused, kept for API compat).
        num_heads (int): Number of attention heads. Default: 8.
        update_query_pos (bool): Whether to update query positional part.
            Default: False.
    """

    def __init__(self, name, dim_in, hidden_dim, dim_out,
                 num_heads=8, update_query_pos=False):
        super().__init__()
        self.name = name
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.update_query_pos = update_query_pos

        # -- Self-attention (inference: dropout=0) --
        self.self_attn = MultiheadAttention(
            f'{name}.self_attn',
            embed_dims=dim_in, num_heads=num_heads,
            attn_drop=0.0, proj_drop=0.0, batch_first=False)

        # -- Main FFN --
        self.linear1 = SimNN.Linear(
            f'{name}.linear1', in_features=dim_in, out_features=hidden_dim)
        self.linear2 = SimNN.Linear(
            f'{name}.linear2', in_features=hidden_dim, out_features=dim_in)
        self.norm1 = LayerNorm(f'{name}.norm1', dim_in)
        self.norm2 = LayerNorm(f'{name}.norm2', dim_in)

        # -- Optional pos FFN --
        if update_query_pos:
            self.linear_pos1 = SimNN.Linear(
                f'{name}.linear_pos1', in_features=dim_in, out_features=hidden_dim)
            self.linear_pos2 = SimNN.Linear(
                f'{name}.linear_pos2', in_features=hidden_dim, out_features=dim_in)
            self.norm_pos = LayerNorm(f'{name}.norm_pos', dim_in)

        # -- Feat FFN --
        self.linear_feat1 = SimNN.Linear(
            f'{name}.linear_feat1', in_features=dim_in, out_features=hidden_dim)
        self.linear_feat2 = SimNN.Linear(
            f'{name}.linear_feat2', in_features=hidden_dim, out_features=dim_in)
        self.norm_feat = LayerNorm(f'{name}.norm_feat', dim_in)

        # -- Pre-create ops --
        # Unsqueeze for adding seq dim (axis=1): (N, dim) -> (N, 1, dim)
        self.unsqueeze_batch = F.Unsqueeze(f'{name}.unsqueeze_batch')
        self.unsq_dim = F._from_data(
            f'{name}.unsq_dim', np.array([1]), is_const=True)

        self.unsqueeze_v = F.Unsqueeze(f'{name}.unsqueeze_v')

        self.reshape_attn = F.Reshape(f'{name}.reshape_attn')

        self.add_qk = F.Add(f'{name}.add_qk')  # q = k = query_pos + out_embed
        self.add_attn_res = F.Add(f'{name}.add_attn_res')  # residual after attn
        self.relu_main = F.Relu(f'{name}.relu_main')
        self.add_ffn_res = F.Add(f'{name}.add_ffn_res')  # residual after main FFN

        if update_query_pos:
            self.relu_pos = F.Relu(f'{name}.relu_pos')
            self.add_pos_res = F.Add(f'{name}.add_pos_res')

        self.relu_feat = F.Relu(f'{name}.relu_feat')
        self.add_feat_res = F.Add(f'{name}.add_feat_res')

        self.concat_query = F.ConcatX(f'{name}.concat_query', axis=-1)

        super().link_op2module()

    def _slice(self, src, field_name, start, end):
        """Slice the last dimension: src[..., start:end]."""
        ndim = len(src.shape)
        starts = [0] * (ndim - 1) + [start]
        ends = [int(s) for s in src.shape[:-1]] + [end]
        axes = list(range(ndim))
        out_shape = list(src.shape[:-1]) + [end - start]

        st = F._from_data(f'{self.name}.{field_name}_st',
                          np.array(starts, dtype=np.int64), is_const=True)
        setattr(self, st.name, st)
        en = F._from_data(f'{self.name}.{field_name}_en',
                          np.array(ends, dtype=np.int64), is_const=True)
        setattr(self, en.name, en)
        ax = F._from_data(f'{self.name}.{field_name}_ax',
                          np.array(axes, dtype=np.int64), is_const=True)
        setattr(self, ax.name, ax)
        sl = F.SliceF(f'{self.name}.{field_name}_sl', out_shape=out_shape)
        setattr(self, sl.name, sl)
        result = sl(src, st, en, ax)
        setattr(self, result.name, result)
        return result

    def __call__(self, query, output_embedding):
        """
        Forward pass: refine track query embeddings.

        Args:
            query: (N, dim*2) combined [query_pos | query_feat].
            output_embedding: (N, dim) detection output embedding.

        Returns:
            updated_query: (N, dim*2) refined [query_pos | query_feat].
        """
        N = query.shape[0]
        full_dim = query.shape[1]
        dim = full_dim // 2

        # Split query -> pos, feat
        query_pos = self._slice(query, 'query_pos', 0, dim)
        query_feat = self._slice(query, 'query_feat', dim, full_dim)

        # q = k = query_pos + output_embedding
        qk = self.add_qk(query_pos, output_embedding)
        setattr(self, qk.name, qk)

        # Unsqueeze for MHA: (N, dim) -> (N, 1, dim) = (seq=N, batch=1, dim)
        q_unsq = self.unsqueeze_batch(qk, self.unsq_dim)
        setattr(self, q_unsq.name, q_unsq)

        # Also unsqueeze value (output_embedding)
        v_unsq = self.unsqueeze_v(output_embedding, self.unsq_dim)
        setattr(self, v_unsq.name, v_unsq)

        # Self-attention: query=key=qk_unsq, value=out_embed_unsq
        attn_out = self.self_attn(query=q_unsq, key=q_unsq, value=v_unsq)
        setattr(self, attn_out.name, attn_out)

        # Reshape: (N, 1, dim) -> (N, dim)
        self.attn_out_shape = F._from_data(
            f'{self.name}.attn_out_shape',
            np.array([N, dim], dtype=np.int64), is_const=True)
        attn_out = self.reshape_attn(attn_out, self.attn_out_shape)
        setattr(self, attn_out.name, attn_out)

        # Residual + LayerNorm 1  (dropout skipped at inference)
        tgt = self.add_attn_res(output_embedding, attn_out)
        setattr(self, tgt.name, tgt)
        tgt = self.norm1(tgt)
        setattr(self, tgt.name, tgt)

        # Main FFN: linear1 -> relu -> linear2
        ffn = self.linear1(tgt)
        setattr(self, ffn.name, ffn)
        ffn = self.relu_main(ffn)
        setattr(self, ffn.name, ffn)
        ffn = self.linear2(ffn)
        setattr(self, ffn.name, ffn)

        # Residual + LayerNorm 2
        tgt = self.add_ffn_res(tgt, ffn)
        setattr(self, tgt.name, tgt)
        tgt = self.norm2(tgt)
        setattr(self, tgt.name, tgt)

        # -- Optional: update query_pos --
        if self.update_query_pos:
            pos_ffn = self.linear_pos1(tgt)
            setattr(self, pos_ffn.name, pos_ffn)
            pos_ffn = self.relu_pos(pos_ffn)
            setattr(self, pos_ffn.name, pos_ffn)
            pos_ffn = self.linear_pos2(pos_ffn)
            setattr(self, pos_ffn.name, pos_ffn)
            query_pos = self.add_pos_res(query_pos, pos_ffn)
            setattr(self, query_pos.name, query_pos)
            query_pos = self.norm_pos(query_pos)
            setattr(self, query_pos.name, query_pos)

        # -- Update query_feat --
        feat_ffn = self.linear_feat1(tgt)
        setattr(self, feat_ffn.name, feat_ffn)
        feat_ffn = self.relu_feat(feat_ffn)
        setattr(self, feat_ffn.name, feat_ffn)
        feat_ffn = self.linear_feat2(feat_ffn)
        setattr(self, feat_ffn.name, feat_ffn)
        query_feat = self.add_feat_res(query_feat, feat_ffn)
        setattr(self, query_feat.name, query_feat)
        query_feat = self.norm_feat(query_feat)
        setattr(self, query_feat.name, query_feat)

        # Concat: [query_pos, query_feat] -> (N, dim*2)
        updated_query = self.concat_query(query_pos, query_feat)
        setattr(self, updated_query.name, updated_query)

        return updated_query

    def analytical_param_count(self, lvl=0):
        """Total learnable parameters in QueryInteractionModule."""
        indent = "  " * lvl
        total = 0

        # self_attn: MHA
        mha = self.self_attn.analytical_param_count(lvl=max(0, lvl - 1))
        total += mha
        if lvl >= 2:
            logger.debug(f"{indent}  self_attn: {mha:,}")

        # Main FFN: linear1 + linear2
        l1 = self.dim_in * self.hidden_dim + self.hidden_dim
        l2 = self.hidden_dim * self.dim_in + self.dim_in
        total += l1 + l2
        if lvl >= 2:
            logger.debug(f"{indent}  linear1: {l1:,}")
            logger.debug(f"{indent}  linear2: {l2:,}")

        # norm1 + norm2: each weight + bias = 2 * dim_in
        ln_main = 2 * self.dim_in * 2
        total += ln_main
        if lvl >= 2:
            logger.debug(f"{indent}  norm1+norm2: {ln_main:,}")

        # Optional pos FFN
        if self.update_query_pos:
            lp1 = self.dim_in * self.hidden_dim + self.hidden_dim
            lp2 = self.hidden_dim * self.dim_in + self.dim_in
            ln_pos = 2 * self.dim_in
            pos_total = lp1 + lp2 + ln_pos
            total += pos_total
            if lvl >= 2:
                logger.debug(f"{indent}  pos FFN (linear_pos1/2 + norm_pos): {pos_total:,}")

        # Feat FFN: linear_feat1 + linear_feat2 + norm_feat
        lf1 = self.dim_in * self.hidden_dim + self.hidden_dim
        lf2 = self.hidden_dim * self.dim_in + self.dim_in
        ln_feat = 2 * self.dim_in
        feat_total = lf1 + lf2 + ln_feat
        total += feat_total
        if lvl >= 2:
            logger.debug(f"{indent}  feat FFN (linear_feat1/2 + norm_feat): {feat_total:,}")

        if lvl >= 1:
            logger.debug(f"{indent}Total QueryInteractionModule params: {total:,}")

        return total


# ======================================================================
# Self-test
# ======================================================================

if __name__ == '__main__':
    logger.info("track_head_plugin/modules.py — TTSim Module Tests")
    logger.info("=" * 70)

    num_tracks = 10
    dim_in = 256
    hidden_dim = 256
    mem_len = 4
    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # Test 1: MemoryBank construction
    # ------------------------------------------------------------------
    try:
        mb = MemoryBank('test_mb', dim_in=dim_in, hidden_dim=hidden_dim,
                        dim_out=dim_in, num_heads=8, max_his_length=mem_len)
        logger.debug("  [Test 1] MemoryBank construction: PASS")
        passed += 1
    except Exception as e:
        logger.debug(f"  [Test 1] MemoryBank construction: FAIL — {e}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 2: MemoryBank forward
    # ------------------------------------------------------------------
    try:
        embed = F._from_data('mb_embed',
                             np.random.randn(num_tracks, dim_in).astype(np.float32))
        prev_embed = F._from_data('mb_prev_embed',
                                  np.random.randn(num_tracks, mem_len, dim_in).astype(np.float32))
        out = mb(embed, prev_embed)
        assert list(out.shape) == [num_tracks, dim_in], \
            f"Expected ({num_tracks}, {dim_in}), got {out.shape}"
        logger.debug(f"  [Test 2] MemoryBank forward shape: PASS — {out.shape}")
        passed += 1
    except Exception as e:
        logger.debug(f"  [Test 2] MemoryBank forward shape: FAIL — {e}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 3: MemoryBank forward with key_padding_mask
    # ------------------------------------------------------------------
    try:
        mb2 = MemoryBank('test_mb2', dim_in=dim_in, hidden_dim=hidden_dim,
                         dim_out=dim_in, num_heads=8, max_his_length=mem_len)
        embed2 = F._from_data('mb2_embed',
                              np.random.randn(num_tracks, dim_in).astype(np.float32))
        prev2 = F._from_data('mb2_prev',
                             np.random.randn(num_tracks, mem_len, dim_in).astype(np.float32))
        kpm = F._from_data('mb2_kpm',
                           np.zeros((num_tracks, mem_len), dtype=np.float32))
        out2 = mb2(embed2, prev2, key_padding_mask=kpm)
        assert list(out2.shape) == [num_tracks, dim_in], \
            f"Expected ({num_tracks}, {dim_in}), got {out2.shape}"
        logger.debug(f"  [Test 3] MemoryBank with mask: PASS — {out2.shape}")
        passed += 1
    except Exception as e:
        logger.debug(f"  [Test 3] MemoryBank with mask: FAIL — {e}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 4: QueryInteractionModule construction
    # ------------------------------------------------------------------
    try:
        qim = QueryInteractionModule(
            'test_qim', dim_in=dim_in, hidden_dim=hidden_dim,
            dim_out=dim_in, num_heads=8, update_query_pos=False)
        logger.debug("  [Test 4] QIM construction (no pos update): PASS")
        passed += 1
    except Exception as e:
        logger.debug(f"  [Test 4] QIM construction: FAIL — {e}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 5: QIM forward
    # ------------------------------------------------------------------
    try:
        query = F._from_data('qim_query',
                             np.random.randn(num_tracks, dim_in * 2).astype(np.float32))
        out_emb = F._from_data('qim_out_emb',
                               np.random.randn(num_tracks, dim_in).astype(np.float32))
        updated = qim(query, out_emb)
        assert list(updated.shape) == [num_tracks, dim_in * 2], \
            f"Expected ({num_tracks}, {dim_in * 2}), got {updated.shape}"
        logger.debug(f"  [Test 5] QIM forward shape: PASS — {updated.shape}")
        passed += 1
    except Exception as e:
        logger.debug(f"  [Test 5] QIM forward shape: FAIL — {e}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 6: QIM with update_query_pos=True
    # ------------------------------------------------------------------
    try:
        qim_pos = QueryInteractionModule(
            'test_qim_pos', dim_in=dim_in, hidden_dim=hidden_dim,
            dim_out=dim_in, num_heads=8, update_query_pos=True)
        query2 = F._from_data('qim_pos_query',
                              np.random.randn(num_tracks, dim_in * 2).astype(np.float32))
        out_emb2 = F._from_data('qim_pos_out_emb',
                                np.random.randn(num_tracks, dim_in).astype(np.float32))
        updated2 = qim_pos(query2, out_emb2)
        assert list(updated2.shape) == [num_tracks, dim_in * 2], \
            f"Expected ({num_tracks}, {dim_in * 2}), got {updated2.shape}"
        logger.debug(f"  [Test 6] QIM with pos update: PASS — {updated2.shape}")
        passed += 1
    except Exception as e:
        logger.debug(f"  [Test 6] QIM with pos update: FAIL — {e}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 7: Graph connectivity (MemoryBank)
    # ------------------------------------------------------------------
    try:
        mb3 = MemoryBank('test_mb3', dim_in=dim_in, hidden_dim=hidden_dim,
                         dim_out=dim_in, num_heads=8)
        emb = F._from_data('mb3_e',
                           np.random.randn(num_tracks, dim_in).astype(np.float32))
        p = F._from_data('mb3_p',
                         np.random.randn(num_tracks, mem_len, dim_in).astype(np.float32))
        r = mb3(emb, p)
        # Verify output is a SimTensor with graph connectivity
        assert hasattr(r, 'op_in'), "Output is not a SimTensor"
        logger.debug("  [Test 7] MemoryBank graph connectivity: PASS")
        passed += 1
    except Exception as e:
        logger.debug(f"  [Test 7] MemoryBank graph connectivity: FAIL — {e}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 8: Graph connectivity (QIM)
    # ------------------------------------------------------------------
    try:
        qim3 = QueryInteractionModule(
            'test_qim3', dim_in=dim_in, hidden_dim=hidden_dim,
            dim_out=dim_in, num_heads=8, update_query_pos=False)
        q3 = F._from_data('qim3_q',
                          np.random.randn(num_tracks, dim_in * 2).astype(np.float32))
        oe3 = F._from_data('qim3_oe',
                           np.random.randn(num_tracks, dim_in).astype(np.float32))
        r3 = qim3(q3, oe3)
        assert hasattr(r3, 'op_in'), "Output is not a SimTensor"
        logger.debug("  [Test 8] QIM graph connectivity: PASS")
        passed += 1
    except Exception as e:
        logger.debug(f"  [Test 8] QIM graph connectivity: FAIL — {e}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 9: MemoryBank analytical_param_count
    # ------------------------------------------------------------------
    try:
        mb4 = MemoryBank('test_mb4', dim_in=dim_in, hidden_dim=hidden_dim,
                         dim_out=dim_in, num_heads=8)
        count = mb4.analytical_param_count(lvl=1)
        assert count > 0, f"Expected positive param count, got {count}"
        logger.debug(f"  [Test 9] MemoryBank param count: PASS — {count:,}")
        passed += 1
    except Exception as e:
        logger.debug(f"  [Test 9] MemoryBank param count: FAIL — {e}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 10: QIM analytical_param_count (no pos)
    # ------------------------------------------------------------------
    try:
        qim4 = QueryInteractionModule(
            'test_qim4', dim_in=dim_in, hidden_dim=hidden_dim,
            dim_out=dim_in, num_heads=8, update_query_pos=False)
        count_no_pos = qim4.analytical_param_count(lvl=1)
        assert count_no_pos > 0
        logger.debug(f"  [Test 10] QIM param count (no pos): PASS — {count_no_pos:,}")
        passed += 1
    except Exception as e:
        logger.debug(f"  [Test 10] QIM param count (no pos): FAIL — {e}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 11: QIM analytical_param_count (with pos)
    # ------------------------------------------------------------------
    try:
        qim5 = QueryInteractionModule(
            'test_qim5', dim_in=dim_in, hidden_dim=hidden_dim,
            dim_out=dim_in, num_heads=8, update_query_pos=True)
        count_pos = qim5.analytical_param_count(lvl=1)
        assert count_pos > count_no_pos, \
            f"With pos ({count_pos}) should > without ({count_no_pos})"
        logger.debug(f"  [Test 11] QIM param count (with pos): PASS — {count_pos:,}")
        passed += 1
    except Exception as e:
        logger.debug(f"  [Test 11] QIM param count (with pos): FAIL — {e}")
        failed += 1

    logger.info("=" * 70)
    logger.info(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed > 0:
        logger.info("[FAIL] Some tests failed!")
    else:
        logger.info("[OK] All tests passed.")
