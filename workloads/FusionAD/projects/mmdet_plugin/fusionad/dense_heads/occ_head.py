

#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of OccHead for FusionAD.

Inference-only conversion.  Training losses, forward_train, forward_test
and metric wrappers are omitted.

Classes:
  - OccHead : Occupancy prediction head with transformer-based future
              state forecasting and instance segmentation logits.

Dependencies (TTSim, already converted):
  - occ_head_plugin.modules : BevFeatureSlicer, MLP, SimpleConv2d,
                              CVT_Decoder, Bottleneck, UpsamplingAdd
  - modules.custom_base_transformer_layer : MyCustomBaseTransformerLayer
  - modules.builder_utils : LayerNorm

Weight mapping notes (PyTorch → TTSim):
  - bev_sampler             → bev_sampler
  - bev_light_proj          → bev_light_proj   (SimpleConv2d)
  - base_downscale.0        → base_ds_0        (Bottleneck)
  - base_downscale.1        → base_ds_1        (Bottleneck)
  - downscale_convs[i]      → ds_conv_{i}      (Bottleneck)
  - temporal_mlps[i]        → temporal_mlp_{i}  (MLP)
  - temporal_mlp_for_mask   → mask_mlp_{0..4}   (same weights to all 5)
  - transformer_decoder.layers[k] → trans_layer_{k}
  - upsample_adds[i]        → upsample_add_{i}
  - dense_decoder           → dense_decoder     (CVT_Decoder)
  - query_to_occ_feat       → query_to_occ_feat (MLP)
  - mode_fuser.0 (Linear)   → mf_linear
  - mode_fuser.1 (LN)       → mf_ln
  - mode_fuser.2 (ReLU)     → mf_relu
  - multi_query_fuser.0     → mqf_linear1
  - multi_query_fuser.1     → mqf_ln1
  - multi_query_fuser.2     → mqf_relu1
  - multi_query_fuser.3     → mqf_linear2
"""

# =============================================================================
# ORIGINAL TORCH CODE
# =============================================================================

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmdet.models.builder import HEADS, build_loss
# from mmcv.runner import BaseModule
# from einops import rearrange
# from mmdet.core import reduce_mean
# from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
# import copy
# from .occ_head_plugin import MLP, BevFeatureSlicer, SimpleConv2d, CVT_Decoder, Bottleneck, UpsamplingAdd, \
#                              predict_instance_segmentation_and_trajectories
#
# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
#
# @HEADS.register_module()
# class OccHead(BaseModule):
#     def __init__(self,
#                  # General
#                  receptive_field=3,
#                  n_future=4,
#                  spatial_extent=(50, 50),
#                  ignore_index=255,
#
#                  # BEV
#                  grid_conf = None,
#
#                  bev_size=(200, 200),
#                  bev_emb_dim=256,
#                  bev_proj_dim=64,
#                  bev_proj_nlayers=1,
#
#                  # Query
#                  query_dim=256,
#                  query_mlp_layers=3,
#                  detach_query_pos=True,
#                  temporal_mlp_layer=2,
#
#                  # Transformer
#                  transformer_decoder=None,
#
#                  attn_mask_thresh=0.5,
#                  # Loss
#                  sample_ignore_mode='all_valid',
#                  aux_loss_weight=1.,
#
#                  loss_mask=None,
#                  loss_dice=None,
#
#                  # Cfgs
#                  init_cfg=None,
#
#                  # Eval
#                  pan_eval=False,
#                  test_seg_thresh:float=0.5,
#                  test_with_track_score=False,
#                  ):
#         assert init_cfg is None, 'To prevent abnormal initialization ' \
#             'behavior, init_cfg is not allowed to be set'
#         super().__init__(init_cfg)
#         self.receptive_field = receptive_field  # NOTE: Used by prepare_future_labels in E2EPredTransformer
#         self.n_future = n_future
#         self.spatial_extent = spatial_extent
#         self.ignore_index  = ignore_index
#
#         bevformer_bev_conf = {
#             'xbound': [-51.2, 51.2, 0.512],
#             'ybound': [-51.2, 51.2, 0.512],
#             'zbound': [-10.0, 10.0, 20.0],
#         }
#         self.bev_sampler =  BevFeatureSlicer(bevformer_bev_conf, grid_conf)
#
#         self.bev_size = bev_size
#         self.bev_proj_dim = bev_proj_dim
#
#         if bev_proj_nlayers == 0:
#             self.bev_light_proj = nn.Sequential()
#         else:
#             self.bev_light_proj = SimpleConv2d(
#                 in_channels=bev_emb_dim,
#                 conv_channels=bev_emb_dim,
#                 out_channels=bev_proj_dim,
#                 num_conv=bev_proj_nlayers,
#             )
#
#         # Downscale bev_feat -> /4
#         self.base_downscale = nn.Sequential(
#             Bottleneck(in_channels=bev_proj_dim, downsample=True),
#             Bottleneck(in_channels=bev_proj_dim, downsample=True)
#         )
#
#         # Future blocks with transformer
#         self.n_future_blocks = self.n_future + 1
#
#         # - transformer
#         self.attn_mask_thresh = attn_mask_thresh
#
#         self.num_trans_layers = transformer_decoder.num_layers
#         assert self.num_trans_layers % self.n_future_blocks == 0
#
#         self.num_heads = transformer_decoder.transformerlayers.\
#             attn_cfgs.num_heads
#         self.transformer_decoder = build_transformer_layer_sequence(
#             transformer_decoder)
#
#         # - temporal-mlps
#         # query_out_dim = bev_proj_dim
#
#         temporal_mlp = MLP(query_dim, query_dim, bev_proj_dim, num_layers=temporal_mlp_layer)
#         self.temporal_mlps = _get_clones(temporal_mlp, self.n_future_blocks)
#
#         # - downscale-convs
#         downscale_conv = Bottleneck(in_channels=bev_proj_dim, downsample=True)
#         self.downscale_convs = _get_clones(downscale_conv, self.n_future_blocks)
#
#         # - upsampleAdds
#         upsample_add = UpsamplingAdd(in_channels=bev_proj_dim, out_channels=bev_proj_dim)
#         self.upsample_adds = _get_clones(upsample_add, self.n_future_blocks)
#
#         # Decoder
#         self.dense_decoder = CVT_Decoder(
#             dim=bev_proj_dim,
#             blocks=[bev_proj_dim, bev_proj_dim],
#         )
#
#         # Query
#         self.mode_fuser = nn.Sequential(
#                 nn.Linear(query_dim, bev_proj_dim),
#                 nn.LayerNorm(bev_proj_dim),
#                 nn.ReLU(inplace=True)
#             )
#         self.multi_query_fuser =  nn.Sequential(
#                 nn.Linear(query_dim * 3, query_dim * 2),
#                 nn.LayerNorm(query_dim * 2),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(query_dim * 2, bev_proj_dim),
#             )
#
#         self.detach_query_pos = detach_query_pos
#
#         self.query_to_occ_feat = MLP(
#             query_dim, query_dim, bev_proj_dim, num_layers=query_mlp_layers
#         )
#         self.temporal_mlp_for_mask = copy.deepcopy(self.query_to_occ_feat)
#
#         # Loss
#         # For matching
#         self.sample_ignore_mode = sample_ignore_mode
#         assert self.sample_ignore_mode in ['all_valid', 'past_valid', 'none']
#
#         self.aux_loss_weight = aux_loss_weight
#
#         self.loss_dice = build_loss(loss_dice)
#         self.loss_mask = build_loss(loss_mask)
#
#         self.pan_eval = pan_eval
#         self.test_seg_thresh  = test_seg_thresh
#
#         self.test_with_track_score = test_with_track_score
#         self.init_weights()
#
#     def init_weights(self):
#         for p in self.transformer_decoder.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_normal_(p)
#
#     def get_attn_mask(self, state, ins_query):
#         # state: b, c, h, w
#         # ins_query: b, q, c
#         ins_embed = self.temporal_mlp_for_mask(
#             ins_query
#         )
#         mask_pred = torch.einsum("bqc,bchw->bqhw", ins_embed, state)
#         attn_mask = mask_pred.sigmoid() < self.attn_mask_thresh
#         attn_mask = rearrange(attn_mask, 'b q h w -> b (h w) q').unsqueeze(1).repeat(
#             1, self.num_heads, 1, 1).flatten(0, 1)
#         attn_mask = attn_mask.detach()
#
#         # if a mask is all True(all background), then set it all False.
#         attn_mask[torch.where(
#             attn_mask.sum(-1) == attn_mask.shape[-1])] = False
#
#         upsampled_mask_pred = F.interpolate(
#             mask_pred,
#             self.bev_size,
#             mode='bilinear',
#             align_corners=False
#         )  # Supervised by gt
#
#         return attn_mask, upsampled_mask_pred, ins_embed
#
#     def forward(self, x, ins_query):
#         base_state = rearrange(x, '(h w) b d -> b d h w', h=self.bev_size[0])
#
#         base_state = self.bev_sampler(base_state)
#         base_state = self.bev_light_proj(base_state)
#         base_state = self.base_downscale(base_state)
#         base_ins_query = ins_query
#
#         last_state = base_state
#         last_ins_query = base_ins_query
#         future_states = []
#         mask_preds = []
#         temporal_query = []
#         temporal_embed_for_mask_attn = []
#         n_trans_layer_each_block = self.num_trans_layers // self.n_future_blocks
#         assert n_trans_layer_each_block >= 1
#
#         for i in range(self.n_future_blocks):
#             # Downscale
#             cur_state = self.downscale_convs[i](last_state)  # /4 -> /8
#
#             # Attention
#             # temporal_aware ins_query
#             cur_ins_query = self.temporal_mlps[i](last_ins_query)  # [b, q, d]
#             temporal_query.append(cur_ins_query)
#
#             # Generate attn mask
#             attn_mask, mask_pred, cur_ins_emb_for_mask_attn = self.get_attn_mask(cur_state, cur_ins_query)
#             attn_masks = [None, attn_mask]
#
#             mask_preds.append(mask_pred)  # /1
#             temporal_embed_for_mask_attn.append(cur_ins_emb_for_mask_attn)
#
#             cur_state = rearrange(cur_state, 'b c h w -> (h w) b c')
#             cur_ins_query = rearrange(cur_ins_query, 'b q c -> q b c')
#
#             for j in range(n_trans_layer_each_block):
#                 trans_layer_ind = i * n_trans_layer_each_block + j
#                 trans_layer = self.transformer_decoder.layers[trans_layer_ind]
#                 cur_state = trans_layer(
#                     query=cur_state,  # [h'*w', b, c]
#                     key=cur_ins_query,  # [nq, b, c]
#                     value=cur_ins_query,  # [nq, b, c]
#                     query_pos=None,
#                     key_pos=None,
#                     attn_masks=attn_masks,
#                     query_key_padding_mask=None,
#                     key_padding_mask=None
#                 )  # out size: [h'*w', b, c]
#
#             cur_state = rearrange(cur_state, '(h w) b c -> b c h w', h=self.bev_size[0]//8)
#
#             # Upscale to /4
#             cur_state = self.upsample_adds[i](cur_state, last_state)
#
#             # Out
#             future_states.append(cur_state)  # [b, d, h/4, w/4]
#             last_state = cur_state
#
#         future_states = torch.stack(future_states, dim=1)  # [b, t, d, h/4, w/4]
#         temporal_query = torch.stack(temporal_query, dim=1)  # [b, t, q, d]
#         mask_preds = torch.stack(mask_preds, dim=2)  # [b, q, t, h, w]
#         ins_query = torch.stack(temporal_embed_for_mask_attn, dim=1)  # [b, t, q, d]
#
#         # Decode future states to larger resolution
#         future_states = self.dense_decoder(future_states)
#         ins_occ_query = self.query_to_occ_feat(ins_query)    # [b, t, q, query_out_dim]
#
#         # Generate final outputs
#         ins_occ_logits = torch.einsum("btqc,btchw->bqthw", ins_occ_query, future_states)
#
#         return mask_preds, ins_occ_logits
#
#     def merge_queries(self, outs_dict, detach_query_pos=True):
#         ins_query = outs_dict.get('traj_query', None)       # [n_dec, b, nq, n_modes, dim]
#         track_query = outs_dict['track_query']              # [b, nq, d]
#         track_query_pos = outs_dict['track_query_pos']      # [b, nq, d]
#
#         if detach_query_pos:
#             track_query_pos = track_query_pos.detach()
#
#         ins_query = ins_query[-1]
#         ins_query = self.mode_fuser(ins_query).max(2)[0]
#         ins_query = self.multi_query_fuser(torch.cat([ins_query, track_query, track_query_pos], dim=-1))
#
#         return ins_query
#
#     # With matched queries [a small part of all queries] and matched_gt results
#     def forward_train(
#                     self,
#                     bev_feat,
#                     outs_dict,
#                     gt_inds_list=None,
#                     gt_segmentation=None,
#                     gt_instance=None,
#                     gt_img_is_valid=None,
#                 ):
#         # Generate warpped gt and related inputs
#         gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(gt_segmentation, gt_instance, gt_img_is_valid)
#
#         all_matched_gt_ids = outs_dict['all_matched_idxes']  # list of tensor, length bs
#
#         ins_query = self.merge_queries(outs_dict, self.detach_query_pos)
#
#         # Forward the occ-flow model
#         mask_preds_batch, ins_seg_preds_batch = self(bev_feat, ins_query=ins_query)
#
#         # Get pred and gt
#         ins_seg_targets_batch  = gt_instance # [1, 5, 200, 200] [b, t, h, w] # ins targets of a batch
#
#         # img_valid flag, for filtering out invalid samples in sequence when calculating loss
#         img_is_valid = gt_img_is_valid  # [1, 7]
#         assert img_is_valid.size(1) == self.receptive_field + self.n_future,  \
#                 f"Img_is_valid can only be 7 as for loss calculation and evaluation!!! Don't change it"
#         frame_valid_mask = img_is_valid.bool()
#         past_valid_mask  = frame_valid_mask[:, :self.receptive_field]
#         future_frame_mask = frame_valid_mask[:, (self.receptive_field-1):]  # [1, 5]  including current frame
#
#         # only supervise when all 3 past frames are valid
#         past_valid = past_valid_mask.all(dim=1)
#         future_frame_mask[~past_valid] = False
#
#         # Calculate loss in the batch
#         loss_dict = dict()
#         loss_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
#         loss_mask = ins_seg_preds_batch.new_zeros(1)[0].float()
#         loss_aux_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
#         loss_aux_mask = ins_seg_preds_batch.new_zeros(1)[0].float()
#
#         bs = ins_query.size(0)
#         assert bs == 1
#         for ind in range(bs):
#             # Each gt_bboxes contains 3 frames, we only use the last one
#             cur_gt_inds   = gt_inds_list[ind][-1]
#
#             cur_matched_gt = all_matched_gt_ids[ind]  # [n_gt]
#
#             # Re-order gt according to matched_gt_inds
#             cur_gt_inds   = cur_gt_inds[cur_matched_gt]
#
#             # Deal matched_gt: -1, its actually background(unmatched)
#             cur_gt_inds[cur_matched_gt == -1] = -1  # Bugfixed
#             cur_gt_inds[cur_matched_gt == -2] = -2
#
#             frame_mask = future_frame_mask[ind]  # [t]
#
#             # Prediction
#             ins_seg_preds = ins_seg_preds_batch[ind]   # [q(n_gt for matched), t, h, w]
#             ins_seg_targets = ins_seg_targets_batch[ind]  # [t, h, w]
#             mask_preds = mask_preds_batch[ind]
#
#             # Assigned-gt
#             ins_seg_targets_ordered = []
#             for ins_id in cur_gt_inds:
#                 # -1 for unmatched query
#                 # If ins_seg_targets is all 255, ignore (directly append occ-and-flow gt to list)
#                 # 255 for special object --> change to -20 (same as in occ_label.py)
#                 # -2 for no_query situation
#                 if (ins_seg_targets == self.ignore_index).all().item() is True:
#                     ins_tgt = ins_seg_targets.long()
#                 elif ins_id.item() in [-1, -2] :  # false positive query (unmatched)
#                     ins_tgt = torch.ones_like(ins_seg_targets).long() * self.ignore_index
#                 else:
#                     SPECIAL_INDEX = -20
#                     if ins_id.item() == self.ignore_index:
#                         ins_id = torch.ones_like(ins_id) * SPECIAL_INDEX
#                     ins_tgt = (ins_seg_targets == ins_id).long()  # [t, h, w], 0 or 1
#
#                 ins_seg_targets_ordered.append(ins_tgt)
#
#             ins_seg_targets_ordered = torch.stack(ins_seg_targets_ordered, dim=0)  # [n_gt, t, h, w]
#
#             # Sanity check
#             t, h, w = ins_seg_preds.shape[-3:]
#             assert t == 1+self.n_future, f"{ins_seg_preds.size()}"
#             assert ins_seg_preds.size() == ins_seg_targets_ordered.size(),   \
#                             f"{ins_seg_preds.size()}, {ins_seg_targets_ordered.size()}"
#
#             num_total_pos = ins_seg_preds.size(0)  # Check this line
#
#             # loss for a sample in batch
#             num_total_pos = ins_seg_preds.new_tensor([num_total_pos])
#             num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
#
#             cur_dice_loss = self.loss_dice(
#                 ins_seg_preds, ins_seg_targets_ordered, avg_factor=num_total_pos, frame_mask=frame_mask)
#
#             cur_mask_loss = self.loss_mask(
#                 ins_seg_preds, ins_seg_targets_ordered, frame_mask=frame_mask
#             )
#
#             cur_aux_dice_loss = self.loss_dice(
#                 mask_preds, ins_seg_targets_ordered, avg_factor=num_total_pos, frame_mask=frame_mask
#             )
#             cur_aux_mask_loss = self.loss_mask(
#                 mask_preds, ins_seg_targets_ordered, frame_mask=frame_mask
#             )
#
#             loss_dice += cur_dice_loss
#             loss_mask += cur_mask_loss
#             loss_aux_dice += cur_aux_dice_loss * self.aux_loss_weight
#             loss_aux_mask += cur_aux_mask_loss * self.aux_loss_weight
#
#         loss_dict['loss_dice'] = loss_dice / bs
#         loss_dict['loss_mask'] = loss_mask / bs
#         loss_dict['loss_aux_dice'] = loss_aux_dice / bs
#         loss_dict['loss_aux_mask'] = loss_aux_mask / bs
#
#         return loss_dict
#
#     def forward_test(
#                     self,
#                     bev_feat,
#                     outs_dict,
#                     no_query=False,
#                     gt_segmentation=None,
#                     gt_instance=None,
#                     gt_img_is_valid=None,
#                 ):
#         gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(gt_segmentation, gt_instance, gt_img_is_valid)
#
#         out_dict = dict()
#         out_dict['seg_gt']  = gt_segmentation[:, :1+self.n_future]  # [1, 5, 1, 200, 200]
#         out_dict['ins_seg_gt'] = self.get_ins_seg_gt(gt_instance[:, :1+self.n_future])  # [1, 5, 200, 200]
#         if no_query:
#             # output all zero results
#             out_dict['seg_out'] = torch.zeros_like(out_dict['seg_gt']).long()  # [1, 5, 1, 200, 200]
#             out_dict['ins_seg_out'] = torch.zeros_like(out_dict['ins_seg_gt']).long()  # [1, 5, 200, 200]
#             return out_dict
#
#         ins_query = self.merge_queries(outs_dict, self.detach_query_pos)
#
#         _, pred_ins_logits = self(bev_feat, ins_query=ins_query)
#
#         out_dict['pred_ins_logits'] = pred_ins_logits
#
#         pred_ins_logits = pred_ins_logits[:,:,:1+self.n_future]  # [b, q, t, h, w]
#         pred_ins_sigmoid = pred_ins_logits.sigmoid()  # [b, q, t, h, w]
#
#         if self.test_with_track_score:
#             track_scores = outs_dict['track_scores'].to(pred_ins_sigmoid)  # [b, q]
#             track_scores = track_scores[:, :, None, None, None]
#             pred_ins_sigmoid = pred_ins_sigmoid * track_scores  # [b, q, t, h, w]
#
#         out_dict['pred_ins_sigmoid'] = pred_ins_sigmoid
#         pred_seg_scores = pred_ins_sigmoid.max(1)[0]
#         seg_out = (pred_seg_scores > self.test_seg_thresh).long().unsqueeze(2)  # [b, t, 1, h, w]
#         out_dict['seg_out'] = seg_out
#         if self.pan_eval:
#             # ins_pred
#             pred_consistent_instance_seg =  \
#                 predict_instance_segmentation_and_trajectories(seg_out, pred_ins_sigmoid)  # bg is 0, fg starts with 1, consecutive
#
#             out_dict['ins_seg_out'] = pred_consistent_instance_seg  # [1, 5, 200, 200]
#
#         return out_dict
#
#     def get_ins_seg_gt(self, gt_instance):
#         ins_gt_old = gt_instance  # Not consecutive, 0 for bg, otherwise ins_ind(start from 1)
#         ins_gt_new = torch.zeros_like(ins_gt_old).to(ins_gt_old)  # Make it consecutive
#         ins_inds_unique = torch.unique(ins_gt_old)
#         new_id = 1
#         for uni_id in ins_inds_unique:
#             if uni_id.item() in [0, self.ignore_index]:  # ignore background_id
#                 continue
#             ins_gt_new[ins_gt_old == uni_id] = new_id
#             new_id += 1
#         return ins_gt_new  # Consecutive
#
#     def get_occ_labels(self, gt_segmentation, gt_instance, gt_img_is_valid):
#         if not self.training:
#             gt_segmentation = gt_segmentation[0]
#             gt_instance = gt_instance[0]
#             gt_img_is_valid = gt_img_is_valid[0]
#
#         gt_segmentation = gt_segmentation[:, :self.n_future+1].long().unsqueeze(2)
#         gt_instance = gt_instance[:, :self.n_future+1].long()
#         gt_img_is_valid = gt_img_is_valid[:, :self.receptive_field + self.n_future]
#         return gt_segmentation, gt_instance, gt_img_is_valid


# =============================================================================
# TTsim CODE
# =============================================================================


import sys
import os
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))

# Add dense_heads directory (so occ_head_plugin is importable as a package)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# fusionad directory (for modules.*)
fusionad_dir = os.path.abspath(os.path.join(current_dir, '..'))
if fusionad_dir not in sys.path:
    sys.path.insert(0, fusionad_dir)

# polaris root
polaris_root = os.path.abspath(
    os.path.join(current_dir, '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from .occ_head_plugin.modules import (
    BevFeatureSlicer, MLP, SimpleConv2d, CVT_Decoder,
    Bottleneck, UpsamplingAdd,
)
from ..modules.custom_base_transformer_layer import MyCustomBaseTransformerLayer
from ..modules.builder_utils import LayerNorm


class OccHead(SimNN.Module):
    """
    Occupancy prediction head for FusionAD.

    Produces instance occupancy logits from BEV features and instance
    queries.  Uses a series of future-prediction blocks, each with:
      - Bottleneck downscale  (/4 → /8)
      - Temporal MLP for query projection
      - Attention-mask generation (einsum decomposition)
      - Transformer decoder layer (self-attn + cross-attn + FFN)
      - UpsamplingAdd for feature fusion (/8 → /4)

    Followed by:
      - CVT_Decoder for final up-sampling
      - query_to_occ_feat MLP
      - Final einsum (decomposed) for logit generation

    Args:
        name (str): Module name.
        n_future (int): Future timesteps. Default: 4.
        grid_conf (dict): BEV grid config for sub-region slicing.
        bev_size (tuple): BEV spatial size (H, W). Default: (200, 200).
        bev_emb_dim (int): Input BEV embedding dim. Default: 256.
        bev_proj_dim (int): Projected BEV dim. Default: 256.
        bev_proj_nlayers (int): BEV projection conv layers. Default: 4.
        query_dim (int): Instance query dim. Default: 256.
        query_mlp_layers (int): Layers in query-to-occ MLP. Default: 3.
        temporal_mlp_layer (int): Layers in temporal MLP. Default: 2.
        num_trans_layers (int): Total transformer layers. Default: 5.
        num_heads (int): Attention heads. Default: 8.
        attn_mask_thresh (float): Sigmoid threshold for mask. Default: 0.3.
        embed_dims (int): Transformer embedding dim. Default: 256.
        feedforward_channels (int): FFN hidden dim. Default: 2048.
    """

    def __init__(self, name,
                 n_future=4,
                 grid_conf=None,
                 bev_size=(200, 200),
                 bev_emb_dim=256,
                 bev_proj_dim=256,
                 bev_proj_nlayers=4,
                 query_dim=256,
                 query_mlp_layers=3,
                 temporal_mlp_layer=2,
                 num_trans_layers=5,
                 num_heads=8,
                 attn_mask_thresh=0.3,
                 embed_dims=256,
                 feedforward_channels=2048):
        super().__init__()
        self.name = name
        self.n_future = n_future
        self.bev_size = bev_size
        self.bev_proj_dim = bev_proj_dim
        self.attn_mask_thresh = attn_mask_thresh
        self.num_heads = num_heads
        self.n_future_blocks = n_future + 1
        self.num_trans_layers = num_trans_layers
        assert num_trans_layers % self.n_future_blocks == 0
        self.n_trans_layer_each_block = num_trans_layers // self.n_future_blocks

        # ── BEV Sampler ──────────────────────────────────────────────
        bevformer_bev_conf = {
            'xbound': [-51.2, 51.2, 0.512],
            'ybound': [-51.2, 51.2, 0.512],
            'zbound': [-10.0, 10.0, 20.0],
        }
        self.bev_sampler = BevFeatureSlicer(
            f'{name}.bev_sampler', bevformer_bev_conf, grid_conf)

        # ── BEV Light Projection ─────────────────────────────────────
        self.has_bev_proj = bev_proj_nlayers > 0
        if self.has_bev_proj:
            self.bev_light_proj = SimpleConv2d(
                f'{name}.bev_light_proj',
                in_channels=bev_emb_dim,
                conv_channels=bev_emb_dim,
                out_channels=bev_proj_dim,
                num_conv=bev_proj_nlayers)

        # ── Base Downscale (2× Bottleneck) ───────────────────────────
        self.base_ds_0 = Bottleneck(
            f'{name}.base_ds_0', in_channels=bev_proj_dim, downsample=True)
        self.base_ds_1 = Bottleneck(
            f'{name}.base_ds_1', in_channels=bev_proj_dim, downsample=True)

        # ── Input reshape: (H*W, B, D) → (B, D, H, W) ──────────────
        self.input_transpose = F.Transpose(
            f'{name}.input_transpose', perm=[1, 2, 0])
        self.input_reshape = F.Reshape(f'{name}.input_reshape')

        # ── Per-block modules ────────────────────────────────────────
        for i in range(self.n_future_blocks):
            pfx = f'{name}.blk_{i}'

            # Downscale conv /4 → /8
            setattr(self, f'ds_conv_{i}', Bottleneck(
                f'{pfx}.ds_conv', in_channels=bev_proj_dim, downsample=True))

            # Temporal MLP
            setattr(self, f'temporal_mlp_{i}', MLP(
                f'{pfx}.temporal_mlp', query_dim, query_dim, bev_proj_dim,
                num_layers=temporal_mlp_layer))

            # Mask MLP (5 copies of temporal_mlp_for_mask — one per block,
            # all receive the same PyTorch weights at test time)
            setattr(self, f'mask_mlp_{i}', MLP(
                f'{pfx}.mask_mlp', query_dim, query_dim, bev_proj_dim,
                num_layers=query_mlp_layers))

            # ── get_attn_mask ops ──
            # einsum "bqc,bchw->bqhw" decomposition
            setattr(self, f'mask_state_reshape_{i}',
                    F.Reshape(f'{pfx}.mask_state_reshape'))
            setattr(self, f'mask_matmul_{i}',
                    F.MatMul(f'{pfx}.mask_matmul'))
            setattr(self, f'mask_pred_reshape_{i}',
                    F.Reshape(f'{pfx}.mask_pred_reshape'))

            # sigmoid + less → boolean mask
            setattr(self, f'mask_sigmoid_{i}',
                    F.Sigmoid(f'{pfx}.mask_sigmoid'))
            setattr(self, f'mask_less_{i}',
                    F.Less(f'{pfx}.mask_less'))
            setattr(self, f'mask_thresh_{i}',
                    F._from_data(
                        f'{pfx}.mask_thresh',
                        np.array(attn_mask_thresh, dtype=np.float32).reshape(1),
                        is_const=True))

            # reshape + transpose: [B,Q,H',W'] → [B,H'*W',Q]
            setattr(self, f'mask_flat_{i}',
                    F.Reshape(f'{pfx}.mask_flat'))
            setattr(self, f'mask_transpose_{i}',
                    F.Transpose(f'{pfx}.mask_transpose', perm=[0, 2, 1]))

            # unsqueeze(1) → [B,1,H'*W',Q]
            setattr(self, f'mask_unsq_{i}',
                    F.Unsqueeze(f'{pfx}.mask_unsq'))
            setattr(self, f'mask_unsq_ax_{i}',
                    F._from_data(
                        f'{pfx}.mask_unsq_ax',
                        np.array([1], dtype=np.int64), is_const=True))

            # tile [1,nhead,1,1] → [B,nhead,H'*W',Q]
            setattr(self, f'mask_tile_{i}',
                    F.Tile(f'{pfx}.mask_tile'))
            setattr(self, f'mask_tile_reps_{i}',
                    F._from_data(
                        f'{pfx}.mask_tile_reps',
                        np.array([1, num_heads, 1, 1], dtype=np.int64),
                        is_const=True))

            # bool → float mask via Where(bool, -1e9, 0)
            setattr(self, f'mask_where_{i}',
                    F.Where(f'{pfx}.mask_where'))

            # ── Transformer state reshaping ──
            # state [B,C,H',W'] → [B,C,H'*W'] → perm → [H'*W',B,C]
            setattr(self, f'state_flat_{i}',
                    F.Reshape(f'{pfx}.state_flat'))
            setattr(self, f'state_to_seq_{i}',
                    F.Transpose(f'{pfx}.state_to_seq', perm=[2, 0, 1]))

            # query [B,Q,C] → [Q,B,C]
            setattr(self, f'query_to_seq_{i}',
                    F.Transpose(f'{pfx}.query_to_seq', perm=[1, 0, 2]))

            # state back [H'*W',B,C] → [B,C,H'*W'] → [B,C,H',W']
            setattr(self, f'seq_to_state_perm_{i}',
                    F.Transpose(f'{pfx}.seq_to_state_perm', perm=[1, 2, 0]))
            setattr(self, f'seq_to_state_reshape_{i}',
                    F.Reshape(f'{pfx}.seq_to_state_reshape'))

            # ── Transformer layers ──
            for j in range(self.n_trans_layer_each_block):
                lid = i * self.n_trans_layer_each_block + j
                setattr(self, f'trans_layer_{lid}',
                        MyCustomBaseTransformerLayer(
                            f'{name}.trans_layer_{lid}',
                            attn_cfgs=dict(
                                type='MultiheadAttention',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                batch_first=False),
                            ffn_cfgs=dict(
                                embed_dims=embed_dims,
                                feedforward_channels=feedforward_channels,
                                num_fcs=2,
                                act_cfg=dict(type='ReLU', inplace=True),
                                ffn_drop=0.0,
                                add_identity=True),
                            operation_order=(
                                'self_attn', 'norm', 'cross_attn',
                                'norm', 'ffn', 'norm'),
                            batch_first=False))

            # ── Upsample + Add ──
            setattr(self, f'upsample_add_{i}', UpsamplingAdd(
                f'{pfx}.upsample_add',
                in_channels=bev_proj_dim, out_channels=bev_proj_dim))

            # ── Unsqueeze for stacking ──
            setattr(self, f'state_unsq_{i}',
                    F.Unsqueeze(f'{pfx}.state_unsq'))
            setattr(self, f'embed_unsq_{i}',
                    F.Unsqueeze(f'{pfx}.embed_unsq'))

        # ── Stack axis constant (dim=1) ──────────────────────────────
        self.stack_ax1 = F._from_data(
            f'{name}.stack_ax1',
            np.array([1], dtype=np.int64), is_const=True)

        # ── ConcatX for stacking ─────────────────────────────────────
        self.concat_states = F.ConcatX(
            f'{name}.concat_states', axis=1)
        self.concat_embeds = F.ConcatX(
            f'{name}.concat_embeds', axis=1)

        # ── Dense Decoder ────────────────────────────────────────────
        self.dense_decoder = CVT_Decoder(
            f'{name}.dense_decoder',
            dim=bev_proj_dim, blocks=[bev_proj_dim, bev_proj_dim])

        # ── query_to_occ_feat MLP ────────────────────────────────────
        self.query_to_occ_feat = MLP(
            f'{name}.query_to_occ_feat',
            query_dim, query_dim, bev_proj_dim,
            num_layers=query_mlp_layers)

        # ── Final einsum "btqc,btchw->bqthw" decomposition ──────────
        self.final_q_reshape = F.Reshape(f'{name}.final_q_reshape')
        self.final_s_reshape = F.Reshape(f'{name}.final_s_reshape')
        self.final_matmul = F.MatMul(f'{name}.final_matmul')
        self.final_reshape_out = F.Reshape(f'{name}.final_reshape_out')
        self.final_transpose = F.Transpose(
            f'{name}.final_transpose', perm=[0, 2, 1, 3, 4])

        # ── merge_queries ops ────────────────────────────────────────
        # mode_fuser: Linear + LN + ReLU
        self.mf_linear = SimNN.Linear(
            f'{name}.mf_linear',
            in_features=query_dim, out_features=bev_proj_dim)
        self.mf_ln = LayerNorm(f'{name}.mf_ln', bev_proj_dim)
        self.mf_relu = F.Relu(f'{name}.mf_relu')

        # ReduceMax over modes (axis=2), drop that dim
        self.mf_reduce_max = F.ReduceMax(
            f'{name}.mf_reduce_max', axes=[2], keepdims=0)

        # ConcatX for merging queries
        self.mq_concat = F.ConcatX(f'{name}.mq_concat', axis=-1)

        # multi_query_fuser: Linear + LN + ReLU + Linear
        self.mqf_linear1 = SimNN.Linear(
            f'{name}.mqf_linear1',
            in_features=query_dim * 3, out_features=query_dim * 2)
        self.mqf_ln1 = LayerNorm(f'{name}.mqf_ln1', query_dim * 2)
        self.mqf_relu1 = F.Relu(f'{name}.mqf_relu1')
        self.mqf_linear2 = SimNN.Linear(
            f'{name}.mqf_linear2',
            in_features=query_dim * 2, out_features=bev_proj_dim)

        super().link_op2module()

    def _reg(self, tensor):
        """Register a dynamic SimTensor into self._tensors for graph visibility."""
        self._tensors[tensor.name] = tensor
        return tensor

    # ------------------------------------------------------------------
    # Attention mask helper
    # ------------------------------------------------------------------
    def _get_attn_mask(self, i, state, ins_query, B):
        """
        Compute float attention mask for block *i*.

        Decomposes  einsum("bqc,bchw->bqhw")  into reshape + matmul,
        then converts sigmoid < threshold into an additive float mask
        (-1e9 for masked positions, 0 otherwise).

        Args:
            i (int):       Block index (0 … n_future_blocks-1).
            state:         (B, C, H', W') spatial-state features.
            ins_query:     (B, Q, C) temporal instance query.
            B (int):       Batch size.

        Returns:
            ins_embed:  (B, Q, C)  — mask embedding (reused later).
            attn_mask:  (B, nhead, H'*W', Q)  — additive float mask.
        """
        # Mask embedding
        ins_embed = getattr(self, f'mask_mlp_{i}')(ins_query)  # [B,Q,C]

        B_s, C_s, H_s, W_s = state.shape
        Q = ins_query.shape[1]
        hw = H_s * W_s

        # einsum "bqc,bchw->bqhw"  →  matmul([B,Q,C],[B,C,H'W'])
        state_flat = getattr(self, f'mask_state_reshape_{i}')(
            state,
            self._reg(F._from_data(f'{self.name}._msrs_{i}',
                         np.array([B_s, C_s, hw], dtype=np.int64),
                         is_const=True)))

        mask_pred = getattr(self, f'mask_matmul_{i}')(
            ins_embed, state_flat)  # [B,Q,H'W']

        mask_pred = getattr(self, f'mask_pred_reshape_{i}')(
            mask_pred,
            self._reg(F._from_data(f'{self.name}._mprs_{i}',
                         np.array([B_s, Q, H_s, W_s], dtype=np.int64),
                         is_const=True)))  # [B,Q,H',W']

        # sigmoid < threshold → boolean
        sig = getattr(self, f'mask_sigmoid_{i}')(mask_pred)
        bool_mask = getattr(self, f'mask_less_{i}')(
            sig, getattr(self, f'mask_thresh_{i}'))  # [B,Q,H',W']

        # reshape [B,Q,H',W'] → [B,Q,H'W'] → transpose → [B,H'W',Q]
        bool_mask = getattr(self, f'mask_flat_{i}')(
            bool_mask,
            self._reg(F._from_data(f'{self.name}._mfrs_{i}',
                         np.array([B_s, Q, hw], dtype=np.int64),
                         is_const=True)))
        bool_mask = getattr(self, f'mask_transpose_{i}')(bool_mask)

        # unsqueeze(1) → [B,1,H'W',Q]
        bool_mask = getattr(self, f'mask_unsq_{i}')(
            bool_mask, getattr(self, f'mask_unsq_ax_{i}'))

        # tile [1,nhead,1,1] → [B,nhead,H'W',Q]
        bool_mask = getattr(self, f'mask_tile_{i}')(
            bool_mask, getattr(self, f'mask_tile_reps_{i}'))

        # bool → float additive mask:  True → -1e9,  False → 0
        # TTSim Where requires all inputs to have the same shape (no broadcast).
        neg_inf_full = self._reg(F._from_data(
            f'{self.name}._neg_inf_{i}',
            np.full(bool_mask.shape, -1e9, dtype=np.float32),
            is_const=True))
        zero_full = self._reg(F._from_data(
            f'{self.name}._zero_{i}',
            np.zeros(bool_mask.shape, dtype=np.float32),
            is_const=True))
        attn_mask = getattr(self, f'mask_where_{i}')(
            bool_mask, neg_inf_full, zero_full)

        return ins_embed, attn_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(self, x, ins_query):
        """
        Forward pass (inference only).

        Args:
            x:         (H*W, B, D)  flattened BEV features.
            ins_query: (B, Q, D)    merged instance queries.

        Returns:
            ins_occ_logits: (B, Q, T, H_out, W_out)
        """
        H, W = self.bev_size
        B = x.shape[1]
        D = x.shape[2]

        # (H*W, B, D) → (B, D, H*W) → (B, D, H, W)
        base_state = self.input_transpose(x)
        base_state = self.input_reshape(
            base_state,
            self._reg(F._from_data(f'{self.name}._inp_shape',
                         np.array([B, D, H, W], dtype=np.int64),
                         is_const=True)))

        # BEV sampling and projection
        base_state = self.bev_sampler(base_state)
        if self.has_bev_proj:
            base_state = self.bev_light_proj(base_state)

        # Base downscale /1 → /4
        base_state = self.base_ds_0(base_state)
        base_state = self.base_ds_1(base_state)

        last_state = base_state
        last_ins_query = ins_query

        state_unsq_list = []
        embed_unsq_list = []

        for i in range(self.n_future_blocks):
            # ── Downscale /4 → /8 ──
            cur_state = getattr(self, f'ds_conv_{i}')(last_state)

            # ── Temporal MLP ──
            cur_ins_query = getattr(self, f'temporal_mlp_{i}')(
                last_ins_query)  # [B,Q,D]

            # ── Attention mask ──
            ins_embed, attn_mask = self._get_attn_mask(
                i, cur_state, cur_ins_query, B)
            attn_masks = [None, attn_mask]  # None for self-attn

            # ── Reshape state for transformer ──
            B_s, C_s, H_s, W_s = cur_state.shape
            hw_s = H_s * W_s

            cur_state_seq = getattr(self, f'state_flat_{i}')(
                cur_state,
                self._reg(F._from_data(f'{self.name}._sfrs_{i}',
                             np.array([B_s, C_s, hw_s], dtype=np.int64),
                             is_const=True)))
            cur_state_seq = getattr(self, f'state_to_seq_{i}')(
                cur_state_seq)  # [H'W', B, C]

            # Reshape query [B,Q,C] → [Q,B,C]
            cur_ins_q_seq = getattr(self, f'query_to_seq_{i}')(
                cur_ins_query)

            # ── Transformer decoder layer(s) ──
            for j in range(self.n_trans_layer_each_block):
                lid = i * self.n_trans_layer_each_block + j
                cur_state_seq = getattr(self, f'trans_layer_{lid}')(
                    query=cur_state_seq,
                    key=cur_ins_q_seq,
                    value=cur_ins_q_seq,
                    query_pos=None,
                    key_pos=None,
                    attn_masks=attn_masks,
                    query_key_padding_mask=None,
                    key_padding_mask=None)

            # ── Reshape state back ──
            cur_state = getattr(self, f'seq_to_state_perm_{i}')(
                cur_state_seq)  # [B, C, H'W']
            cur_state = getattr(self, f'seq_to_state_reshape_{i}')(
                cur_state,
                self._reg(F._from_data(f'{self.name}._sbrs_{i}',
                             np.array([B_s, C_s, H_s, W_s], dtype=np.int64),
                             is_const=True)))  # [B, C, H', W']

            # ── Upsample + skip-add: /8 → /4 ──
            cur_state = getattr(self, f'upsample_add_{i}')(
                cur_state, last_state)

            # ── Unsqueeze for stacking ──
            state_unsq_list.append(
                getattr(self, f'state_unsq_{i}')(
                    cur_state, self.stack_ax1))   # [B,1,D,H/4,W/4]
            embed_unsq_list.append(
                getattr(self, f'embed_unsq_{i}')(
                    ins_embed, self.stack_ax1))   # [B,1,Q,D]

            last_state = cur_state

        # ── Stack results ────────────────────────────────────────────
        future_states = self.concat_states(
            *state_unsq_list)   # [B, T, D, H/4, W/4]
        ins_embeds = self.concat_embeds(
            *embed_unsq_list)   # [B, T, Q, D]

        # ── Dense decoder ────────────────────────────────────────────
        future_states = self.dense_decoder(future_states)

        # ── query_to_occ_feat ────────────────────────────────────────
        ins_occ_query = self.query_to_occ_feat(ins_embeds)  # [B,T,Q,D']

        # ── Final einsum "btqc,btchw->bqthw" ────────────────────────
        B_f, T_f, Q_f, C_f = ins_occ_query.shape
        _, _, C_s2, H_f, W_f = future_states.shape

        # reshape query → [B*T, Q, C]
        q_flat = self.final_q_reshape(
            ins_occ_query,
            self._reg(F._from_data(f'{self.name}._fqrs',
                         np.array([B_f * T_f, Q_f, C_f], dtype=np.int64),
                         is_const=True)))

        # reshape states → [B*T, C, H*W]
        s_flat = self.final_s_reshape(
            future_states,
            self._reg(F._from_data(f'{self.name}._fsrs',
                         np.array([B_f * T_f, C_s2, H_f * W_f],
                                  dtype=np.int64),
                         is_const=True)))

        # matmul → [B*T, Q, H*W]
        logits = self.final_matmul(q_flat, s_flat)

        # reshape → [B, T, Q, H, W]
        logits = self.final_reshape_out(
            logits,
            self._reg(F._from_data(f'{self.name}._fors',
                         np.array([B_f, T_f, Q_f, H_f, W_f],
                                  dtype=np.int64),
                         is_const=True)))

        # transpose → [B, Q, T, H, W]
        ins_occ_logits = self.final_transpose(logits)

        return ins_occ_logits

    # ------------------------------------------------------------------
    # merge_queries  (called externally before __call__)
    # ------------------------------------------------------------------
    def merge_queries(self, traj_query_last, track_query, track_query_pos):
        """
        Merge tracking outputs into instance queries for OccHead.

        In the original PyTorch code this is called before ``forward()``
        from ``forward_train`` / ``forward_test``.

        Args:
            traj_query_last: (B, Q, n_modes, D) — last decoder traj query.
            track_query:     (B, Q, D)          — track feature query.
            track_query_pos: (B, Q, D)          — track positional query.

        Returns:
            ins_query: (B, Q, bev_proj_dim)
        """
        # mode_fuser: Linear + LN + ReLU
        x = self.mf_linear(traj_query_last)  # [B,Q,modes,proj_dim]
        x = self.mf_ln(x)
        x = self.mf_relu(x)

        # Max over modes → [B,Q,proj_dim]
        x = self.mf_reduce_max(x)

        # Concat with track queries
        x = self.mq_concat(x, track_query, track_query_pos)  # [B,Q,3D]

        # multi_query_fuser
        x = self.mqf_linear1(x)
        x = self.mqf_ln1(x)
        x = self.mqf_relu1(x)
        x = self.mqf_linear2(x)  # [B,Q,proj_dim]

        return x


# ──────────────────────────────────────────────────────────────────────
# Self-tests
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)

    logger.info('=' * 60)
    logger.info('OccHead self-tests')
    logger.info('=' * 60)

    occflow_grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
    }

    # ── Test 1: creation ──
    logger.info('\nTEST 1  —  Module creation ... ')
    head = OccHead(
        'occ_head',
        n_future=4,
        grid_conf=occflow_grid_conf,
        bev_size=(200, 200),
        bev_emb_dim=256,
        bev_proj_dim=256,
        bev_proj_nlayers=4,
        query_dim=256,
        query_mlp_layers=3,
        temporal_mlp_layer=2,
        num_trans_layers=5,
        num_heads=8,
        attn_mask_thresh=0.3,
        embed_dims=256,
        feedforward_channels=2048,
    )
    logger.info('[OK]')

    # ── Test 2: forward shape ──
    logger.info('TEST 2  —  Forward pass shape ... ')
    B, Q, D = 1, 10, 256
    H, W = 200, 200
    x = F._from_data('t2_x',
                      np.random.randn(H * W, B, D).astype(np.float32))
    ins_q = F._from_data('t2_q',
                          np.random.randn(B, Q, D).astype(np.float32))
    out = head(x, ins_q)
    T = head.n_future_blocks
    expected = [B, Q, T, H, W]
    assert list(out.shape) == expected, \
        f'shape {list(out.shape)} != {expected}'
    logger.info('[OK]')

    # ── Test 3: merge_queries shape ──
    logger.info('TEST 3  —  merge_queries shape ... ')
    n_modes = 6
    traj_q = F._from_data('t3_traj',
                            np.random.randn(B, Q, n_modes, D).astype(np.float32))
    trk_q = F._from_data('t3_trk',
                           np.random.randn(B, Q, D).astype(np.float32))
    trk_pos = F._from_data('t3_pos',
                             np.random.randn(B, Q, D).astype(np.float32))
    mq = head.merge_queries(traj_q, trk_q, trk_pos)
    assert list(mq.shape) == [B, Q, head.bev_proj_dim], \
        f'shape {list(mq.shape)} != {[B, Q, head.bev_proj_dim]}'
    logger.info('[OK]')

    logger.info('\nAll OccHead self-tests passed.')
