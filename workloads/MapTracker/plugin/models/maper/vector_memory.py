#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
ttsim implementation of VectorInstanceMemory from MapTracker
Converts PyTorch memory bank and positional encoding to TensTorrent simulation framework
"""

# -------------------------------PyTorch--------------------------------

# import torch
# from torch import nn
#
# from einops import repeat, rearrange
# from scipy.spatial.transform import Rotation as R
# import numpy as np
#
#
# def get_emb(sin_inp):
#     """
#     Gets a base embedding for one dimension with sin and cos intertwined
#     """
#     emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
#     return torch.flatten(emb, -2, -1)
#
#
# class PositionalEncoding1D(nn.Module):
#     def __init__(self, channels):
#         """
#         :param channels: The last dimension of the tensor you want to apply pos emb to.
#         """
#         super(PositionalEncoding1D, self).__init__()
#         self.org_channels = channels
#         channels = int(np.ceil(channels / 2) * 2)
#         self.channels = channels
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
#         self.register_buffer("inv_freq", inv_freq)
#         self.register_buffer("cached_penc", None)
#
#     def forward(self, tensor):
#         """
#         :param tensor: A 3d tensor of size (batch_size, x, ch)
#         :return: Positional Encoding Matrix of size (batch_size, x, ch)
#         """
#         if len(tensor.shape) != 3:
#             raise RuntimeError("The input tensor has to be 3d!")
#
#         if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
#             return self.cached_penc
#
#         self.cached_penc = None
#         batch_size, x, orig_ch = tensor.shape
#         pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
#         sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
#         emb_x = get_emb(sin_inp_x)
#         emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
#         emb[:, : self.channels] = emb_x
#
#         self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
#         return self.cached_penc
#
#
# class VectorInstanceMemory(nn.Module):
#
#     def __init__(self,
#                  dim_in, number_ins, bank_size, mem_len, mem_select_dist_ranges
#                  ):
#         super().__init__()
#         self.max_number_ins = 3 * number_ins # make sure this is not exceeded at initial training when results could be quite random
#         self.bank_size = bank_size
#         self.mem_len = mem_len
#         self.dim_in = dim_in
#         self.mem_select_dist_ranges = mem_select_dist_ranges
#
#         p_enc_1d = PositionalEncoding1D(dim_in)
#         fake_tensor = torch.zeros((1, 1000, dim_in)) # suppose all sequences are shorter than 1000
#         self.cached_pe = p_enc_1d(fake_tensor)[0]
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def set_bank_size(self, bank_size):
#         self.bank_size = bank_size
#
#     def init_memory(self, bs):
#         self.mem_bank = torch.zeros((self.bank_size, bs, self.max_number_ins, self.dim_in), dtype=torch.float32).cuda()
#         self.mem_bank_seq_id = torch.zeros((self.bank_size, bs, self.max_number_ins), dtype=torch.long).cuda()
#         self.mem_bank_trans = torch.zeros((self.bank_size, bs,  3),dtype=torch.float32).cuda()
#         self.mem_bank_rot = torch.zeros((self.bank_size, bs, 3, 3),dtype=torch.float32).cuda()
#         self.batch_mem_embeds_dict = {}
#         self.batch_mem_relative_pe_dict = {}
#         self.batch_key_padding_dict = {}
#         self.curr_rot = torch.zeros((bs,3,3),dtype=torch.float32).cuda()
#         self.curr_trans = torch.zeros((bs,3),dtype=torch.float32).cuda()
#         self.gt_lines_info = {}
#
#         # memory recording information
#         self.instance2mem = [{} for _ in range(bs)]
#         self.num_ins = [0 for _ in range(bs)]
#         self.active_mem_ids = [None for _ in range(bs)]
#         self.valid_track_idx = [None for _ in range(bs)]
#         self.random_bev_masks = [None for _ in range(bs)]
#         init_entry_length = torch.tensor([0]*self.max_number_ins).long()
#         self.mem_entry_lengths = [init_entry_length.clone() for _ in range(bs)]
#
#     def update_memory(self, batch_i, is_first_frame, propagated_ids, prev_out, num_tracks,
#                       seq_idx, timestep):
#         if is_first_frame:
#             mem_instance_ids = torch.arange(propagated_ids.shape[0])
#             track2mem_info = {i: i for i in range(len(propagated_ids))}
#             num_instances = len(propagated_ids)
#         else:
#             track2mem_info_prev = self.instance2mem[batch_i]
#             track2mem_info = {}
#             num_instances = self.num_ins[batch_i]
#             for pred_i, propagated_id in enumerate(propagated_ids):
#                 if propagated_id < num_tracks: # existing tracks
#                     track2mem_info[pred_i] = track2mem_info_prev[propagated_id.item()]
#                 else: # newborn instances
#                     track2mem_info[pred_i] = num_instances
#                     num_instances += 1
#             mem_instance_ids = torch.tensor([track2mem_info[item] for item in range(len(propagated_ids))]).long()
#
#         assert num_instances < self.max_number_ins, 'Number of instances larger than mem size!'
#
#         #NOTE: put information into the memory, need to detach the scores to block gradient backprop
#         # from future time steps
#         prev_embeddings = prev_out['hs_embeds'][batch_i]
#         prev_scores = prev_out['scores'][batch_i]
#         prev_scores, prev_labels = prev_scores.max(-1)
#         prev_scores = prev_scores.sigmoid().detach()
#
#         mem_lens_per_ins = self.mem_entry_lengths[batch_i][mem_instance_ids]
#
#         # insert information into mem bank
#         for ins_idx, mem_id in enumerate(mem_instance_ids):
#             if mem_lens_per_ins[ins_idx] < self.bank_size:
#                 self.mem_bank[mem_lens_per_ins[ins_idx], batch_i, mem_id] = prev_embeddings[propagated_ids[ins_idx]]
#                 self.mem_bank_seq_id[mem_lens_per_ins[ins_idx], batch_i, mem_id] = seq_idx
#             else:
#                 self.mem_bank[:self.bank_size-1, batch_i, mem_id] = self.mem_bank[1:self.bank_size, batch_i, mem_id]
#                 self.mem_bank[-1, batch_i, mem_id] = prev_embeddings[propagated_ids[ins_idx]]
#                 self.mem_bank_seq_id[:self.bank_size-1, batch_i, mem_id] = self.mem_bank_seq_id[1:self.bank_size, batch_i, mem_id]
#                 self.mem_bank_seq_id[-1, batch_i, mem_id] = seq_idx
#
#         if self.curr_t < self.bank_size:
#             self.mem_bank_rot[self.curr_t, batch_i] = self.curr_rot[batch_i]
#             self.mem_bank_trans[self.curr_t, batch_i] = self.curr_trans[batch_i]
#         else:
#             self.mem_bank_rot[:self.bank_size-1, batch_i] = self.mem_bank_rot[1:, batch_i].clone()
#             self.mem_bank_rot[-1, batch_i] = self.curr_rot[batch_i]
#             self.mem_bank_trans[:self.bank_size-1, batch_i] = self.mem_bank_trans[1:, batch_i].clone()
#             self.mem_bank_trans[-1, batch_i] = self.curr_trans[batch_i]
#
#         # Update the mem recording information
#         self.instance2mem[batch_i] = track2mem_info
#         self.num_ins[batch_i] = num_instances
#         self.mem_entry_lengths[batch_i][mem_instance_ids] += 1
#         self.active_mem_ids[batch_i] = mem_instance_ids.long().to(propagated_ids.device)
#         active_mem_entry_lens = self.mem_entry_lengths[batch_i][self.active_mem_ids[batch_i]]
#         self.valid_track_idx[batch_i] = torch.where(active_mem_entry_lens >= 1)[0]
#
#         #print('Active memory ids:', self.active_mem_ids[batch_i])
#         #print('Memory entry lens:', active_mem_entry_lens)
#         #print('Valid track idx:', self.valid_track_idx[batch_i])
#
#     def prepare_transformation_batch(self,history_e2g_trans,history_e2g_rot,curr_e2g_trans,curr_e2g_rot):
#         history_g2e_matrix = torch.stack([torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device),]*len(history_e2g_trans), dim=0)
#         history_g2e_matrix[:, :3, :3] = torch.transpose(history_e2g_rot, 1, 2)
#         history_g2e_matrix[:, :3, 3] = -torch.bmm(torch.transpose(history_e2g_rot, 1, 2), history_e2g_trans[..., None]).squeeze(-1)
#
#         curr_g2e_matrix = torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device)
#         curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
#         curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)
#
#         curr_e2g_matrix = torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device)
#         curr_e2g_matrix[:3, :3] = curr_e2g_rot
#         curr_e2g_matrix[:3, 3] = curr_e2g_trans
#
#         history_e2g_matrix = torch.stack([torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device),]*len(history_e2g_trans), dim=0)
#         history_e2g_matrix[:, :3, :3] = history_e2g_rot
#         history_e2g_matrix[:, :3, 3] = history_e2g_trans
#
#         history_curr2prev_matrix = torch.bmm(history_g2e_matrix, repeat(curr_e2g_matrix,'n1 n2 -> r n1 n2', r=len(history_g2e_matrix)))
#         history_prev2curr_matrix = torch.bmm(repeat(curr_g2e_matrix, 'n1 n2 -> r n1 n2', r=len(history_e2g_matrix)), history_e2g_matrix)
#
#         return history_curr2prev_matrix, history_prev2curr_matrix
#
#     def clear_dict(self,):
#         self.batch_mem_embeds_dict = {}
#         self.batch_mem_relative_pe_dict = {}
#         self.batch_key_padding_dict = {}
#
#     def trans_memory_bank(self, query_prop, b_i, metas):
#         seq_id = metas['local_idx']
#
#         active_mem_ids = self.active_mem_ids[b_i]
#         mem_entry_lens = self.mem_entry_lengths[b_i][active_mem_ids]
#         num_track_ins = len(active_mem_ids)
#         valid_mem_len = min(self.curr_t, self.mem_len)
#         valid_bank_size = min(self.curr_t, self.bank_size)
#         mem_trans = self.mem_bank_trans[:, b_i]
#         mem_rots = self.mem_bank_rot[:, b_i]
#
#         if self.training:
#             # Note: at training time, bank_size must be the same as mem_len, no selection needed
#             assert self.mem_len == self.bank_size, 'at training time, bank_size must be the same as mem_len'
#             mem_embeds = self.mem_bank[:, b_i, active_mem_ids]
#             mem_seq_ids = self.mem_bank_seq_id[:, b_i, active_mem_ids]
#         else:
#             # at test time, the bank size can be much longer, and we need the selection strategy
#             mem_embeds = torch.zeros_like(self.mem_bank[:self.mem_len, b_i, active_mem_ids])
#             mem_seq_ids = torch.zeros_like(self.mem_bank_seq_id[:self.mem_len, b_i, active_mem_ids])
#
#         # Put information into mem embeddings and pos_ids, prepare for attention-fusion
#         # Also prepare the pose information for the query propagation
#         all_pose_select_indices = []
#         all_select_indices = []
#         for idx, active_idx in enumerate(active_mem_ids):
#             effective_len = mem_entry_lens[idx]
#             valid_mem_trans = mem_trans[:valid_bank_size]
#             trunc_eff_len = min(effective_len, self.bank_size)
#             valid_pose_ids = torch.arange(valid_bank_size-trunc_eff_len, valid_bank_size)
#             #print('ins {}, valid pose ids {}'.format(idx, valid_pose_ids))
#             if effective_len <= self.mem_len:
#                 select_indices = torch.arange(effective_len)
#             else:
#                 select_indices = self.select_memory_entries(valid_mem_trans[-trunc_eff_len:], metas)
#             pose_select_indices = valid_pose_ids[select_indices]
#             mem_embeds[:len(select_indices), idx] = self.mem_bank[select_indices, b_i, active_idx]
#             mem_seq_ids[:len(select_indices), idx] = self.mem_bank_seq_id[select_indices, b_i, active_idx]
#             all_pose_select_indices.append(pose_select_indices)
#             all_select_indices.append(select_indices)
#
#         # prepare mem padding mask
#         key_padding_mask = torch.ones((self.mem_len, num_track_ins)).bool().cuda()
#         padding_trunc_loc = torch.clip(mem_entry_lens, max=self.mem_len)
#         for ins_i in range(num_track_ins):
#             key_padding_mask[:padding_trunc_loc[ins_i], ins_i] = False
#         key_padding_mask = key_padding_mask.T
#
#         # prepare relative seq idx gap
#         relative_seq_idx = torch.zeros_like(mem_embeds[:,:,0]).long()
#         relative_seq_idx[:valid_mem_len] = seq_id - mem_seq_ids[:valid_mem_len]
#         relative_seq_pe = self.cached_pe[relative_seq_idx].to(mem_embeds.device)
#
#         # prepare relative pose information for each active instance
#         curr2prev_matrix, prev2curr_matrix = self.prepare_transformation_batch(mem_trans[:valid_bank_size],
#             mem_rots[:valid_bank_size], self.curr_trans[b_i], self.curr_rot[b_i])
#         pose_matrix = prev2curr_matrix.float()[:,:3]
#         rot_mat = pose_matrix[..., :3].cpu().numpy()
#         rot = R.from_matrix(rot_mat)
#         translation = pose_matrix[..., 3]
#
#         if self.training:
#             rot, translation = self.add_noise_to_pose(rot, translation)
#
#         rot_quat = torch.tensor(rot.as_quat()).float().to(pose_matrix.device)
#         pose_info = torch.cat([rot_quat, translation], dim=1)
#         pose_info_per_ins = torch.zeros((valid_mem_len, num_track_ins, pose_info.shape[1])).to(pose_info.device)
#
#         for ins_idx in range(num_track_ins):
#             pose_select_indices = all_pose_select_indices[ins_idx]
#             pose_info_per_ins[:len(pose_select_indices), ins_idx] = pose_info[pose_select_indices]
#
#         mem_embeds_new = mem_embeds.clone()
#         mem_embeds_valid = rearrange(mem_embeds[:valid_mem_len], 't n c -> (t n) c')
#         pose_info_per_ins = rearrange(pose_info_per_ins, 't n c -> (t n) c')
#         mem_embeds_prop = query_prop(
#             mem_embeds_valid,
#             pose_info_per_ins
#         )
#         mem_embeds_new[:valid_mem_len] = rearrange(mem_embeds_prop, '(t n) c -> t n c', t=valid_mem_len)
#
#         self.batch_mem_embeds_dict[b_i] = mem_embeds_new.clone().detach()
#         self.batch_mem_relative_pe_dict[b_i] = relative_seq_pe
#         self.batch_key_padding_dict[b_i] = key_padding_mask
#
#     def add_noise_to_pose(self, rot, trans):
#         rot_euler = rot.as_euler('zxy')
#         # 0.08 mean is around 5-degree, 3-sigma is 15-degree
#         noise_euler = np.random.randn(*list(rot_euler.shape)) * 0.08
#         rot_euler += noise_euler
#         noisy_rot = R.from_euler('zxy', rot_euler)
#
#         # error within 0.25 meter
#         noise_trans = torch.randn_like(trans) * 0.25
#         noise_trans[:, 2] = 0
#         noisy_trans = trans + noise_trans
#
#         return noisy_rot, noisy_trans
#
#     def select_memory_entries(self, mem_trans, curr_meta):
#         history_e2g_trans = mem_trans[:, :2].cpu().numpy()
#         curr_e2g_trans = np.array(curr_meta['ego2global_translation'][:2])
#         dists = np.linalg.norm(history_e2g_trans - curr_e2g_trans[None, :], axis=1)
#
#         sorted_indices = np.argsort(dists)
#         sorted_dists = dists[sorted_indices]
#         covered = np.zeros_like(sorted_indices).astype(np.bool)
#         selected_ids = []
#         for dist_range in self.mem_select_dist_ranges[::-1]:
#             outter_valid_flags = (sorted_dists >= dist_range) & ~covered
#             if outter_valid_flags.any():
#                 pick_id = np.where(outter_valid_flags)[0][0]
#                 covered[pick_id:] = True
#             else:
#                 inner_valid_flags = (sorted_dists < dist_range) & ~covered
#                 if inner_valid_flags.any():
#                     pick_id = np.where(inner_valid_flags)[0][-1]
#                     covered[pick_id] = True
#                 else:
#                     # return the mem_len closest one, but in the order of far -> close
#                     return np.array(sorted_indices[:4][::-1])
#             selected_ids.append(pick_id)
#
#         selected_mem_ids = sorted_indices[np.array(selected_ids)]
#         return selected_mem_ids
#
#
#

# -------------------------------TTSIM-----------------------------------


import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore[import-untyped]
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class PositionalEncoding1D(SimNN.Module):
    """1D Positional Encoding using sine and cosine functions"""

    def __init__(self, name, channels):
        super().__init__()
        self.name = name
        self.org_channels = channels
        # Round up to nearest even number
        self.channels = int(np.ceil(channels / 2) * 2)

        # Compute inverse frequencies: 1.0 / (10000 ^ (k / channels))
        k_values = np.arange(0, self.channels, 2).astype(np.float32)
        inv_freq = 1.0 / (10000 ** (k_values / self.channels))

        # Position indices [0, 1, 2, ..., max_len-1]
        self.max_cached_len = 1000
        pos_x = np.arange(self.max_cached_len, dtype=np.float32)

        # Create constant tensors
        pos_tensor = F._from_data(
            f"{self.name}.pos_x",
            data=pos_x.reshape(-1, 1),  # [max_len, 1]
            is_const=True,
        )
        inv_freq_tensor = F._from_data(
            f"{self.name}.inv_freq",
            data=inv_freq.reshape(1, -1),  # [1, channels//2]
            is_const=True,
        )
        setattr(self, pos_tensor.name, pos_tensor)
        setattr(self, inv_freq_tensor.name, inv_freq_tensor)

        # Compute sin_inp = outer product using F.Mul (broadcasts automatically)
        self.mul_op = F.Mul(f"{self.name}.mul")
        sin_inp = self.mul_op(pos_tensor, inv_freq_tensor)  # [max_len, channels//2]
        setattr(self, sin_inp.name, sin_inp)

        # Create Sin and Cos operators
        self.sin_op = F.Sin(f"{self.name}.sin")
        self.cos_op = F.Cos(f"{self.name}.cos")

        # Apply sin and cos operations
        sin_vals = self.sin_op(sin_inp)  # [max_len, channels//2]
        cos_vals = self.cos_op(sin_inp)  # [max_len, channels//2]
        setattr(self, sin_vals.name, sin_vals)
        setattr(self, cos_vals.name, cos_vals)

        # Interleave sin and cos using TTSim ops (keeps graph connected)
        # Unsqueeze both to [max_len, channels//2, 1]
        unsq_axis = F._from_data(
            f"{self.name}.unsq_axis", np.array([-1], dtype=np.int64), is_const=True
        )
        setattr(self, unsq_axis.name, unsq_axis)

        self.sin_unsq_op = F.Unsqueeze(f"{self.name}.sin_unsq")
        self.cos_unsq_op = F.Unsqueeze(f"{self.name}.cos_unsq")
        sin_unsq = self.sin_unsq_op(sin_vals, unsq_axis)  # [max_len, channels//2, 1]
        cos_unsq = self.cos_unsq_op(cos_vals, unsq_axis)  # [max_len, channels//2, 1]
        setattr(self, sin_unsq.name, sin_unsq)
        setattr(self, cos_unsq.name, cos_unsq)

        # Concat on last axis to interleave: [max_len, channels//2, 2]
        self.interleave_concat = F.ConcatX(f"{self.name}.interleave_concat", axis=-1)
        interleaved = self.interleave_concat(
            sin_unsq, cos_unsq
        )  # [max_len, channels//2, 2]
        setattr(self, interleaved.name, interleaved)

        # Reshape to flatten last two dims: [max_len, channels]
        self.interleave_reshape = F.Reshape(f"{self.name}.interleave_reshape")
        reshape_shape = F._from_data(
            f"{self.name}.interleave_shape",
            np.array([self.max_cached_len, self.channels], dtype=np.int64),
            is_const=True,
        )
        setattr(self, reshape_shape.name, reshape_shape)
        pe_table = self.interleave_reshape(
            interleaved, reshape_shape
        )  # [max_len, channels]
        setattr(self, pe_table.name, pe_table)
        self.pe_table = pe_table

        # Extract numpy cache from the graph-connected SimTensor
        if pe_table.data is not None:
            emb_full = pe_table.data
        else:
            # Fallback: compute manually
            sin_d = sin_vals.data if sin_vals.data is not None else np.sin(sin_inp.data)
            cos_d = cos_vals.data if cos_vals.data is not None else np.cos(sin_inp.data)
            emb_full = np.stack([sin_d, cos_d], axis=-1).reshape(
                self.max_cached_len, -1
            )

        # Slice to original channels and cache
        self.emb_cache = emb_full[:, : self.org_channels]

        super().link_op2module()

    def __call__(self, tensor):
        batch_size, seq_len, orig_ch = tensor.shape

        # Slice cached encoding to sequence length
        emb = self.emb_cache[:seq_len, :]  # [seq_len, orig_ch]

        # Add batch dimension and tile: [1, seq_len, orig_ch] -> [batch_size, seq_len, orig_ch]
        emb_batched = np.tile(emb[np.newaxis, :, :], (batch_size, 1, 1))

        # Convert to SimTensor
        pe_tensor = F._from_data(
            f"{self.name}.pe_output_{batch_size}_{seq_len}",
            data=emb_batched,
            is_const=True,
        )
        setattr(self, pe_tensor.name, pe_tensor)

        return pe_tensor

    def analytical_param_count(self, lvl):
        """
        Calculate total number of parameters (PositionalEncoding1D has no trainable params)

        Returns:
            int: 0 (no trainable parameters, only cached constants)
        """
        return 0


class VectorInstanceMemory(SimNN.Module):
    """
    Memory bank for tracking instances across frames.
    Stores embeddings, sequence IDs, and pose information.
    """

    def __init__(
        self,
        name,
        dim_in,
        number_ins,
        bank_size,
        mem_len,
        mem_select_dist_ranges=(20, 40, 60, 80),
    ):
        super().__init__()
        self.name = name

        self.dim_in = dim_in
        self.number_ins = number_ins
        self.max_number_ins = 3 * number_ins  # Match original MapTracker source
        self.bank_size = bank_size
        self.mem_len = mem_len
        self.mem_select_dist_ranges = mem_select_dist_ranges

        # Positional encoding for temporal information
        self.pos_encoder = PositionalEncoding1D(f"{name}.pos_encoder", dim_in)
        self.cached_pe = self.pos_encoder.emb_cache

        # Tracking state (not tensors, just python dicts/lists)
        self.instance2mem = {}
        self.num_ins = {}
        self.mem_entry_lengths = {}
        self.active_mem_ids = {}
        self.valid_track_idx = {}
        self.curr_t = 0
        self.curr_rot = {}
        self.curr_trans = {}

        # Memory banks (initialized per batch)
        self.mem_bank = None
        self.mem_bank_seq_id = None
        self.mem_bank_trans = None
        self.mem_bank_rot = None

        # Dictionary for batch memory
        self.batch_mem_embeds_dict = {}
        self.batch_mem_relative_pe_dict = {}
        self.batch_key_padding_dict = {}

        super().link_op2module()

    def set_bank_size(self, bank_size):
        """Set the memory bank size (used for test-time configuration)."""
        self.bank_size = bank_size

    def init_memory(self, bs):
        """Initialize memory banks for batch size bs"""
        # Memory bank for embeddings (bank_size, bs, max_number_ins, dim_in)
        self.mem_bank = np.zeros(
            (self.bank_size, bs, self.max_number_ins, self.dim_in), dtype=np.float32
        )

        # Memory bank for sequence IDs (bank_size, bs, max_number_ins)
        self.mem_bank_seq_id = np.zeros(
            (self.bank_size, bs, self.max_number_ins), dtype=np.int64
        )

        # Memory bank for transformations (bank_size, bs, 3)
        self.mem_bank_trans = np.zeros((self.bank_size, bs, 3), dtype=np.float32)

        # Memory bank for rotations (bank_size, bs, 3, 3)
        self.mem_bank_rot = np.zeros((self.bank_size, bs, 3, 3), dtype=np.float32)

        # Initialize tracking state for each batch
        for b_i in range(bs):
            self.instance2mem[b_i] = {}
            self.num_ins[b_i] = 0
            self.mem_entry_lengths[b_i] = np.zeros(self.max_number_ins, dtype=np.int64)
            self.active_mem_ids[b_i] = np.array([], dtype=np.int64)
            self.valid_track_idx[b_i] = np.array([], dtype=np.int64)
            self.curr_rot[b_i] = np.eye(3, dtype=np.float32)
            self.curr_trans[b_i] = np.zeros(3, dtype=np.float32)

    def update_memory(
        self, batch_i, is_first_frame, propagated_ids, prev_out, seq_idx, num_tracks
    ):
        """
        Update memory bank with new observations

        Args:
            batch_i: Batch index
            is_first_frame: Whether this is the first frame
            propagated_ids: Indices of propagated instances
            prev_out: Previous output dict with 'hs_embeds' and 'scores'
            seq_idx: Current sequence index
            num_tracks: Number of existing tracks
        """
        mem_instance_ids: np.ndarray
        if is_first_frame:
            mem_instance_ids = np.arange(len(propagated_ids), dtype=np.int64)
            track2mem_info = {i: i for i in range(len(propagated_ids))}
            num_instances = len(propagated_ids)
        else:
            track2mem_info_prev = self.instance2mem[batch_i]
            track2mem_info = {}
            num_instances = self.num_ins[batch_i]

            mem_instance_ids_list: list[int] = []
            for pred_i, propagated_id in enumerate(propagated_ids):
                if propagated_id < num_tracks:  # existing tracks
                    track2mem_info[pred_i] = track2mem_info_prev[int(propagated_id)]
                    mem_instance_ids_list.append(track2mem_info[pred_i])
                else:  # newborn instances
                    track2mem_info[pred_i] = num_instances
                    mem_instance_ids_list.append(num_instances)
                    num_instances += 1
            mem_instance_ids = np.array(mem_instance_ids_list, dtype=np.int64)

        assert (
            num_instances < self.max_number_ins
        ), "Number of instances larger than mem size!"

        # Extract previous embeddings and scores
        # prev_out may contain SimTensors or numpy arrays
        prev_embeddings = prev_out["hs_embeds"]
        if hasattr(prev_embeddings, "data"):
            # SimTensor path
            prev_embeddings = prev_embeddings[batch_i]
            if hasattr(prev_embeddings, "data"):
                prev_embeddings = (
                    np.asarray(prev_embeddings.data)
                    if prev_embeddings.data is not None
                    else np.zeros((1,), dtype=np.float32)
                )
        elif isinstance(prev_embeddings, np.ndarray):
            if prev_embeddings.ndim == 3:
                prev_embeddings = prev_embeddings[batch_i]
        else:
            prev_embeddings = np.asarray(prev_embeddings)
            if prev_embeddings.ndim == 3:
                prev_embeddings = prev_embeddings[batch_i]

        prev_scores_raw = prev_out["scores"]
        if hasattr(prev_scores_raw, "data"):
            # SimTensor path
            prev_scores_raw = prev_scores_raw[batch_i]
            if hasattr(prev_scores_raw, "data"):
                prev_scores_raw = (
                    np.asarray(prev_scores_raw.data)
                    if prev_scores_raw.data is not None
                    else np.zeros((1,), dtype=np.float32)
                )
        elif isinstance(prev_scores_raw, np.ndarray):
            if prev_scores_raw.ndim == 3:
                prev_scores_raw = prev_scores_raw[batch_i]
        else:
            prev_scores_raw = np.asarray(prev_scores_raw)
            if prev_scores_raw.ndim == 3:
                prev_scores_raw = prev_scores_raw[batch_i]

        # Ensure we have numpy for wrapping
        if not isinstance(prev_scores_raw, np.ndarray):
            prev_scores_raw = np.asarray(prev_scores_raw).astype(np.float32)

        # Use TTSim ops for score processing (leverages try_compute_data)
        prev_scores_tensor = F._from_data(
            f"{self.name}.prev_scores_{batch_i}_{seq_idx}",
            prev_scores_raw.astype(np.float32),
            is_const=False,
        )

        # Get max scores along last dimension using ttsim
        argmax_op = F.ArgMax(f"{self.name}.argmax_{batch_i}_{seq_idx}", axis=-1)
        prev_labels_tensor = argmax_op(prev_scores_tensor)
        if prev_labels_tensor.data is not None:
            prev_labels = np.asarray(prev_labels_tensor.data)
        else:
            prev_labels = np.argmax(prev_scores_raw, axis=-1)

        reducemax_op = F.ReduceMax(
            f"{self.name}.reducemax_{batch_i}_{seq_idx}", axes=[-1]
        )
        prev_scores_max = reducemax_op(prev_scores_tensor)

        # Sigmoid using ttsim Sigmoid operator
        sigmoid_op = F.Sigmoid(f"{self.name}.sigmoid_{batch_i}_{seq_idx}")
        prev_scores_sig = sigmoid_op(prev_scores_max)

        # Get data (detached)
        if prev_scores_sig.data is not None:
            prev_scores = np.asarray(prev_scores_sig.data)
        else:
            prev_scores_max_np = np.max(prev_scores_raw, axis=-1)
            prev_scores = 1.0 / (1.0 + np.exp(-prev_scores_max_np))

        mem_lens_per_ins = self.mem_entry_lengths[batch_i][mem_instance_ids]

        # Insert information into mem bank
        assert self.mem_bank is not None
        assert self.mem_bank_seq_id is not None
        for ins_idx, mem_id in enumerate(mem_instance_ids):
            prop_id = int(propagated_ids[ins_idx])
            if mem_lens_per_ins[ins_idx] < self.bank_size:
                self.mem_bank[mem_lens_per_ins[ins_idx], batch_i, mem_id] = (
                    prev_embeddings[prop_id]
                )
                self.mem_bank_seq_id[mem_lens_per_ins[ins_idx], batch_i, mem_id] = (
                    seq_idx
                )
            else:
                # Shift memory bank (FIFO)
                self.mem_bank[: self.bank_size - 1, batch_i, mem_id] = self.mem_bank[
                    1 : self.bank_size, batch_i, mem_id
                ].copy()
                self.mem_bank[-1, batch_i, mem_id] = prev_embeddings[prop_id]

                self.mem_bank_seq_id[: self.bank_size - 1, batch_i, mem_id] = (
                    self.mem_bank_seq_id[1 : self.bank_size, batch_i, mem_id].copy()
                )
                self.mem_bank_seq_id[-1, batch_i, mem_id] = seq_idx

        # Update transformation memory
        assert self.mem_bank_rot is not None
        assert self.mem_bank_trans is not None
        if self.curr_t < self.bank_size:
            self.mem_bank_rot[self.curr_t, batch_i] = self.curr_rot[batch_i]
            self.mem_bank_trans[self.curr_t, batch_i] = self.curr_trans[batch_i]
        else:
            self.mem_bank_rot[: self.bank_size - 1, batch_i] = self.mem_bank_rot[
                1:, batch_i
            ].copy()
            self.mem_bank_rot[-1, batch_i] = self.curr_rot[batch_i]

            self.mem_bank_trans[: self.bank_size - 1, batch_i] = self.mem_bank_trans[
                1:, batch_i
            ].copy()
            self.mem_bank_trans[-1, batch_i] = self.curr_trans[batch_i]

        # Update tracking information
        self.instance2mem[batch_i] = track2mem_info
        self.num_ins[batch_i] = num_instances
        self.mem_entry_lengths[batch_i][mem_instance_ids] += 1
        self.active_mem_ids[batch_i] = mem_instance_ids

        active_mem_entry_lens = self.mem_entry_lengths[batch_i][
            self.active_mem_ids[batch_i]
        ]
        self.valid_track_idx[batch_i] = np.where(active_mem_entry_lens >= 1)[0]

    def prepare_transformation_batch(
        self, history_e2g_trans, history_e2g_rot, curr_e2g_trans, curr_e2g_rot
    ):
        """
        Prepare transformation matrices between current and historical frames

        Args:
            history_e2g_trans: Historical ego-to-global translations (N, 3)
            history_e2g_rot: Historical ego-to-global rotations (N, 3, 3)
            curr_e2g_trans: Current ego-to-global translation (3,)
            curr_e2g_rot: Current ego-to-global rotation (3, 3)

        Returns:
            history_curr2prev_matrix: Transformation from current to historical (N, 4, 4)
            history_prev2curr_matrix: Transformation from historical to current (N, 4, 4)
        """
        N = len(history_e2g_trans)

        # Convert to numpy if SimTensor
        if hasattr(history_e2g_trans, "data"):
            history_e2g_trans = history_e2g_trans.data
        if hasattr(history_e2g_rot, "data"):
            history_e2g_rot = history_e2g_rot.data
        if hasattr(curr_e2g_trans, "data"):
            curr_e2g_trans = curr_e2g_trans.data
        if hasattr(curr_e2g_rot, "data"):
            curr_e2g_rot = curr_e2g_rot.data

        # Ensure inputs are numpy arrays (handle memoryview from scipy)
        history_e2g_trans = np.asarray(history_e2g_trans, dtype=np.float32)
        history_e2g_rot = np.asarray(history_e2g_rot, dtype=np.float32)
        curr_e2g_trans = np.asarray(curr_e2g_trans, dtype=np.float32)
        curr_e2g_rot = np.asarray(curr_e2g_rot, dtype=np.float32)

        # Create historical global-to-ego matrices
        history_g2e_matrix = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))

        # Transpose rotation matrices using ttsim
        history_rot_tensor = F._from_data(f"{self.name}.history_rot", history_e2g_rot)
        transpose_op = F.Transpose(f"{self.name}.transpose_rot", perm=[0, 2, 1])
        history_rot_T = transpose_op(history_rot_tensor)
        history_rot_T_data = (
            history_rot_T.data
            if history_rot_T.data is not None
            else np.transpose(history_e2g_rot, (0, 2, 1))
        )

        history_g2e_matrix[:, :3, :3] = history_rot_T_data

        # MatMul for rotation @ translation using ttsim
        trans_tensor = F._from_data(
            f"{self.name}.history_trans", np.expand_dims(history_e2g_trans, -1)
        )
        matmul_op1 = F.MatMul(f"{self.name}.matmul_g2e")
        neg_trans = matmul_op1(history_rot_T, trans_tensor)
        neg_trans_data = (
            neg_trans.data
            if neg_trans.data is not None
            else np.matmul(history_rot_T_data, np.expand_dims(history_e2g_trans, -1))
        )

        neg_op = F.Neg(f"{self.name}.neg_trans")
        neg_trans_result = neg_op(neg_trans)
        history_g2e_matrix[:, :3, 3] = (
            neg_trans_result.data
            if neg_trans_result.data is not None
            else -neg_trans_data
        ).squeeze(-1)

        # Create current global-to-ego matrix (small matrices, numpy is fine)
        curr_g2e_matrix = np.eye(4, dtype=np.float64)
        curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
        curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

        # Create current ego-to-global matrix
        curr_e2g_matrix = np.eye(4, dtype=np.float64)
        curr_e2g_matrix[:3, :3] = curr_e2g_rot
        curr_e2g_matrix[:3, 3] = curr_e2g_trans

        # Create historical ego-to-global matrices
        history_e2g_matrix = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
        history_e2g_matrix[:, :3, :3] = history_e2g_rot
        history_e2g_matrix[:, :3, 3] = history_e2g_trans

        # Compute transformations using ttsim MatMul
        curr_e2g_repeated = np.tile(np.expand_dims(curr_e2g_matrix, 0), (N, 1, 1))
        curr_g2e_repeated = np.tile(np.expand_dims(curr_g2e_matrix, 0), (N, 1, 1))

        # MatMul for transformation matrices
        history_g2e_tensor = F._from_data(
            f"{self.name}.history_g2e", history_g2e_matrix.astype(np.float32)
        )
        curr_e2g_repeated_tensor = F._from_data(
            f"{self.name}.curr_e2g_rep", curr_e2g_repeated.astype(np.float32)
        )

        matmul_op2 = F.MatMul(f"{self.name}.matmul_curr2prev")
        curr2prev_tensor = matmul_op2(history_g2e_tensor, curr_e2g_repeated_tensor)
        history_curr2prev_matrix = (
            curr2prev_tensor.data
            if curr2prev_tensor.data is not None
            else np.matmul(history_g2e_matrix, curr_e2g_repeated)
        )

        curr_g2e_repeated_tensor = F._from_data(
            f"{self.name}.curr_g2e_rep", curr_g2e_repeated.astype(np.float32)
        )
        history_e2g_tensor = F._from_data(
            f"{self.name}.history_e2g", history_e2g_matrix.astype(np.float32)
        )

        matmul_op3 = F.MatMul(f"{self.name}.matmul_prev2curr")
        prev2curr_tensor = matmul_op3(curr_g2e_repeated_tensor, history_e2g_tensor)
        history_prev2curr_matrix = (
            prev2curr_tensor.data
            if prev2curr_tensor.data is not None
            else np.matmul(curr_g2e_repeated, history_e2g_matrix)
        )

        return history_curr2prev_matrix, history_prev2curr_matrix

    def clear_dict(self):
        """Clear batch memory dictionaries"""
        self.batch_mem_embeds_dict = {}
        self.batch_mem_relative_pe_dict = {}
        self.batch_key_padding_dict = {}

    def trans_memory_bank(self, query_prop, b_i, metas, training=False):
        """
        Transform memory bank for current query

        Args:
            query_prop: Query propagation module/function
            b_i: Batch index
            metas: Metadata dict containing 'local_idx', 'ego2global_translation', etc.
            training: Whether in training mode
        """
        seq_id = metas["local_idx"]

        active_mem_ids = self.active_mem_ids[b_i]
        mem_entry_lens = self.mem_entry_lengths[b_i][active_mem_ids]
        num_track_ins = len(active_mem_ids)
        valid_mem_len = min(self.curr_t, self.mem_len)
        valid_bank_size = min(self.curr_t, self.bank_size)

        assert self.mem_bank is not None
        assert self.mem_bank_seq_id is not None
        assert self.mem_bank_trans is not None
        assert self.mem_bank_rot is not None
        mem_trans = self.mem_bank_trans[:, b_i]
        mem_rots = self.mem_bank_rot[:, b_i]

        if training:
            # At training time, bank_size must be the same as mem_len
            assert (
                self.mem_len == self.bank_size
            ), "at training time, bank_size must be the same as mem_len"
            mem_embeds = self.mem_bank[:, b_i, active_mem_ids].copy()
            mem_seq_ids = self.mem_bank_seq_id[:, b_i, active_mem_ids].copy()
        else:
            # At test time, need selection strategy
            mem_embeds = np.zeros(
                (self.mem_len, num_track_ins, self.dim_in), dtype=np.float32
            )
            mem_seq_ids = np.zeros((self.mem_len, num_track_ins), dtype=np.int64)

        # Put information into mem embeddings and pos_ids
        all_pose_select_indices = []
        all_select_indices = []

        for idx, active_idx in enumerate(active_mem_ids):
            effective_len = mem_entry_lens[idx]
            valid_mem_trans = mem_trans[:valid_bank_size]
            trunc_eff_len = min(effective_len, self.bank_size)
            valid_pose_ids = np.arange(valid_bank_size - trunc_eff_len, valid_bank_size)

            if effective_len <= self.mem_len:
                select_indices = np.arange(effective_len)
            else:
                select_indices = self.select_memory_entries(
                    valid_mem_trans[-trunc_eff_len:], metas
                )

            pose_select_indices = valid_pose_ids[select_indices]
            mem_embeds[: len(select_indices), idx] = self.mem_bank[
                select_indices, b_i, active_idx
            ]
            mem_seq_ids[: len(select_indices), idx] = self.mem_bank_seq_id[
                select_indices, b_i, active_idx
            ]
            all_pose_select_indices.append(pose_select_indices)
            all_select_indices.append(select_indices)

        # Prepare memory padding mask
        key_padding_mask = np.ones((self.mem_len, num_track_ins), dtype=bool)
        padding_trunc_loc = np.clip(mem_entry_lens, 0, self.mem_len)
        for ins_i in range(num_track_ins):
            key_padding_mask[: padding_trunc_loc[ins_i], ins_i] = False
        key_padding_mask = key_padding_mask.T

        # Prepare relative sequence index gap
        relative_seq_idx = np.zeros((self.mem_len, num_track_ins), dtype=np.int64)
        relative_seq_idx[:valid_mem_len] = seq_id - mem_seq_ids[:valid_mem_len]

        # Get positional encoding from cached embeddings
        relative_seq_pe = np.zeros(
            (self.mem_len, num_track_ins, self.dim_in), dtype=np.float32
        )
        for i in range(self.mem_len):
            for j in range(num_track_ins):
                idx = relative_seq_idx[i, j]
                if idx >= 0 and idx < len(self.cached_pe):
                    relative_seq_pe[i, j] = self.cached_pe[idx]

        # Prepare relative pose information
        curr2prev_matrix, prev2curr_matrix = self.prepare_transformation_batch(
            mem_trans[:valid_bank_size],
            mem_rots[:valid_bank_size],
            self.curr_trans[b_i],
            self.curr_rot[b_i],
        )

        pose_matrix = prev2curr_matrix.astype(np.float32)[:, :3]
        rot_mat = pose_matrix[..., :3]
        rot = R.from_matrix(rot_mat)
        translation = pose_matrix[..., 3]

        if training:
            rot, translation = self.add_noise_to_pose(rot, translation)

        rot_quat = rot.as_quat().astype(np.float32)

        # Concatenate rotation quaternion and translation using ttsim
        rot_quat_tensor = F._from_data(f"{self.name}.rot_quat_{b_i}", rot_quat)
        translation_tensor = F._from_data(f"{self.name}.translation_{b_i}", translation)

        concat_op = F.ConcatX(f"{self.name}.concat_pose_{b_i}", axis=1)
        pose_info_tensor = concat_op(rot_quat_tensor, translation_tensor)
        pose_info = (
            pose_info_tensor.data
            if pose_info_tensor.data is not None
            else np.concatenate([rot_quat, translation], axis=1)
        )

        pose_info_per_ins = np.zeros(
            (valid_mem_len, num_track_ins, pose_info.shape[1]), dtype=np.float32
        )

        for ins_idx in range(num_track_ins):
            pose_select_indices = all_pose_select_indices[ins_idx]
            pose_info_per_ins[: len(pose_select_indices), ins_idx] = pose_info[
                pose_select_indices
            ]

        # Propagate memory embeddings using ttsim Reshape
        mem_embeds_new = mem_embeds.copy()

        # Guard: skip reshape/propagation if no valid memory entries
        if valid_mem_len == 0 or num_track_ins == 0:
            # Nothing to propagate — store as-is
            self.batch_mem_embeds_dict[b_i] = mem_embeds_new.copy()
            self.batch_mem_relative_pe_dict[b_i] = relative_seq_pe
            self.batch_key_padding_dict[b_i] = key_padding_mask
            return

        # Reshape: 't n c -> (t n) c'
        mem_embeds_valid_data = mem_embeds[:valid_mem_len]
        mem_embeds_tensor = F._from_data(
            f"{self.name}.mem_embeds_{b_i}", mem_embeds_valid_data
        )
        reshape_op1 = F.Reshape(f"{self.name}.reshape_embeds_{b_i}")
        shape1 = F._from_data(
            f"{self.name}.reshape_embeds_shape_{b_i}",
            np.array([-1, self.dim_in], dtype=np.int64),
            is_const=True,
        )
        mem_embeds_flat = reshape_op1(mem_embeds_tensor, shape1)
        mem_embeds_valid = (
            mem_embeds_flat.data
            if mem_embeds_flat.data is not None
            else mem_embeds_valid_data.reshape(-1, self.dim_in)
        )

        # Reshape pose info
        pose_info_tensor2 = F._from_data(
            f"{self.name}.pose_info_per_ins_{b_i}", pose_info_per_ins
        )
        reshape_op2 = F.Reshape(f"{self.name}.reshape_pose_{b_i}")
        shape2 = F._from_data(
            f"{self.name}.reshape_pose_shape_{b_i}",
            np.array([-1, pose_info.shape[1]], dtype=np.int64),
            is_const=True,
        )
        pose_info_flat_tensor = reshape_op2(pose_info_tensor2, shape2)
        pose_info_flat = (
            pose_info_flat_tensor.data
            if pose_info_flat_tensor.data is not None
            else pose_info_per_ins.reshape(-1, pose_info.shape[1])
        )

        # Call query propagation (expecting SimTensor or callable)
        if callable(query_prop):
            mem_embeds_valid_np = (
                np.asarray(mem_embeds_valid).copy()
                if not isinstance(mem_embeds_valid, np.ndarray)
                else mem_embeds_valid.copy()
            )
            mem_embeds_prop_tensor = query_prop(
                F._from_data(
                    f"{self.name}.mem_embeds_valid_{b_i}", mem_embeds_valid_np
                ),
                F._from_data(f"{self.name}.pose_info_flat_{b_i}", pose_info_flat),
            )
            if (
                hasattr(mem_embeds_prop_tensor, "data")
                and mem_embeds_prop_tensor.data is not None
            ):
                mem_embeds_prop_data = np.asarray(mem_embeds_prop_tensor.data).copy()
            else:
                # Fallback: use the original valid embeddings (no propagation applied)
                mem_embeds_prop_data = mem_embeds_valid_np
            mem_embeds_prop_tensor = F._from_data(
                f"{self.name}.mem_embeds_prop_{b_i}", mem_embeds_prop_data
            )
        else:
            mem_embeds_prop_data = (
                mem_embeds_valid
                if isinstance(mem_embeds_valid, np.ndarray)
                else np.asarray(mem_embeds_valid).copy()
            )
            mem_embeds_prop_tensor = F._from_data(
                f"{self.name}.mem_embeds_prop_{b_i}", mem_embeds_prop_data
            )

        # Reshape back using ttsim: '(t n) c -> t n c'
        reshape_op3 = F.Reshape(f"{self.name}.reshape_back_{b_i}")
        shape3 = F._from_data(
            f"{self.name}.reshape_back_shape_{b_i}",
            np.array([valid_mem_len, num_track_ins, -1], dtype=np.int64),
            is_const=True,
        )
        mem_embeds_reshaped = reshape_op3(mem_embeds_prop_tensor, shape3)
        if mem_embeds_reshaped.data is not None:
            mem_embeds_new_data = np.asarray(mem_embeds_reshaped.data).reshape(
                valid_mem_len, num_track_ins, -1
            )
        elif isinstance(mem_embeds_prop_data, np.ndarray):
            mem_embeds_new_data = mem_embeds_prop_data.reshape(
                valid_mem_len, num_track_ins, -1
            )
        else:  # fallback: convert to ndarray
            mem_embeds_new_data = np.asarray(mem_embeds_prop_data).reshape(valid_mem_len, num_track_ins, -1)  # type: ignore[unreachable]

        mem_embeds_new[:valid_mem_len] = mem_embeds_new_data

        # Store in dictionaries
        self.batch_mem_embeds_dict[b_i] = mem_embeds_new.copy()
        self.batch_mem_relative_pe_dict[b_i] = relative_seq_pe
        self.batch_key_padding_dict[b_i] = key_padding_mask

    def add_noise_to_pose(self, rot, trans):
        """
        Add noise to pose for data augmentation during training

        Args:
            rot: Rotation object from scipy
            trans: Translation array (N, 3)

        Returns:
            noisy_rot: Rotation with added noise
            noisy_trans: Translation with added noise
        """
        rot_euler = rot.as_euler("zxy")
        # 0.08 std means ~5 degree, 3-sigma is ~15 degree
        noise_euler = np.random.randn(*rot_euler.shape) * 0.08
        rot_euler += noise_euler
        noisy_rot = R.from_euler("zxy", rot_euler)

        # Error within 0.25 meter
        noise_trans = np.random.randn(*trans.shape).astype(np.float32) * 0.25
        noise_trans[:, 2] = 0  # No noise in z-direction
        noisy_trans = trans + noise_trans

        return noisy_rot, noisy_trans

    def select_memory_entries(self, mem_trans, curr_meta):
        """
        Select memory entries based on distance ranges

        Args:
            mem_trans: Memory translations (N, 3)
            curr_meta: Current metadata with 'ego2global_translation'

        Returns:
            selected_mem_ids: Indices of selected memory entries
        """
        history_e2g_trans = mem_trans[:, :2]
        curr_e2g_trans = np.array(curr_meta["ego2global_translation"][:2])
        dists = np.linalg.norm(history_e2g_trans - curr_e2g_trans[None, :], axis=1)

        sorted_indices = np.argsort(dists)
        sorted_dists = dists[sorted_indices]
        covered = np.zeros(len(sorted_indices), dtype=bool)
        selected_ids = []

        for dist_range in self.mem_select_dist_ranges[::-1]:
            outer_valid_flags = (sorted_dists >= dist_range) & ~covered
            if outer_valid_flags.any():
                pick_id = np.where(outer_valid_flags)[0][0]
                covered[pick_id:] = True
            else:
                inner_valid_flags = (sorted_dists < dist_range) & ~covered
                if inner_valid_flags.any():
                    pick_id = np.where(inner_valid_flags)[0][-1]
                    covered[pick_id] = True
                else:
                    # Return the mem_len closest ones, but in order far -> close
                    return sorted_indices[:4][::-1]
            selected_ids.append(pick_id)

        selected_mem_ids = sorted_indices[np.array(selected_ids)]
        return selected_mem_ids

    def __call__(self, *args, **kwargs):
        """Forward pass - placeholder for compatibility"""
        # VectorInstanceMemory doesn't have a typical forward pass
        # It's used via update_memory and trans_memory_bank methods
        pass

    def analytical_param_count(self, lvl):
        """
        Calculate total number of parameters in VectorInstanceMemory

        Returns:
            int: Total parameter count (only positional encoder parameters)
        """
        # VectorInstanceMemory has no trainable parameters itself
        # Memory banks (mem_bank, mem_bank_seq_id, etc.) are state, not parameters
        # Only the positional encoder has cached constants (but no trainable params)
        pe_params = self.pos_encoder.analytical_param_count(lvl)

        # Note: If query propagation modules are added, their params should be counted here
        return pe_params
