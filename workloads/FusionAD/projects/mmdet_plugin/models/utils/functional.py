

"""
FusionAD functional utilities - TTSim Implementation

Shared math helpers: positional embeddings, Gaussian activation,
coordinate normalization, and trajectory transforms.
"""
# =============================================================================
# TORCH CODE
# =============================================================================
# import math
# import torch
# from einops import rearrange, repeat
#
# def bivariate_gaussian_activation(ip):
#     """
#     Activation function to output parameters of bivariate Gaussian distribution.
#
#     Args:
#         ip (torch.Tensor): Input tensor.
#
#     Returns:
#         torch.Tensor: Output tensor containing the parameters of the bivariate Gaussian distribution.
#     """
#     mu_x = ip[..., 0:1]
#     mu_y = ip[..., 1:2]
#     sig_x = ip[..., 2:3]
#     sig_y = ip[..., 3:4]
#     rho = ip[..., 4:5]
#     sig_x = torch.exp(sig_x)
#     sig_y = torch.exp(sig_y)
#     rho = torch.tanh(rho)
#     out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
#     return out
#
# def norm_points(pos, pc_range):
#     """
#     Normalize the end points of a given position tensor.
#
#     Args:
#         pos (torch.Tensor): Input position tensor.
#         pc_range (List[float]): Point cloud range.
#
#     Returns:
#         torch.Tensor: Normalized end points tensor.
#     """
#     x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
#     y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
#     return torch.stack([x_norm, y_norm], dim=-1)
#
# def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
#     """
#     Convert 2D position into positional embeddings.
#
#     Args:
#         pos (torch.Tensor): Input 2D position tensor.
#         num_pos_feats (int, optional): Number of positional features. Default is 128.
#         temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.
#
#     Returns:
#         torch.Tensor: Positional embeddings tensor.
#     """
#     scale = 2 * math.pi
#     pos = pos * scale
#     dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
#     dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
#     pos_x = pos[..., 0, None] / dim_t
#     pos_y = pos[..., 1, None] / dim_t
#     pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
#     pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
#     posemb = torch.cat((pos_y, pos_x), dim=-1)
#     return posemb
#
# def rot_2d(yaw):
#     """
#     Compute 2D rotation matrix for a given yaw angle tensor.
#
#     Args:
#         yaw (torch.Tensor): Input yaw angle tensor.
#
#     Returns:
#         torch.Tensor: 2D rotation matrix tensor.
#     """
#     sy, cy = torch.sin(yaw), torch.cos(yaw)
#     out = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).permute([2,0,1])
#     return out
#
# def anchor_coordinate_transform(anchors, bbox_results, with_translation_transform=True, with_rotation_transform=True):
#     """
#     Transform anchor coordinates with respect to detected bounding boxes in the batch.
#
#     Args:
#         anchors (torch.Tensor): A tensor containing the k-means anchor values.
#         bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
#         with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
#         with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.
#
#     Returns:
#         torch.Tensor: A tensor containing the transformed anchor coordinates.
#     """
#     batch_size = len(bbox_results)
#     batched_anchors = []
#     transformed_anchors = anchors[None, ...] # expand num agents: num_groups, num_modes, 12, 2 -> 1, ...
#     for i in range(batch_size):
#         bboxes, scores, labels, bbox_index, mask = bbox_results[i]
#         yaw = bboxes.yaw.to(transformed_anchors.device)
#         bbox_centers = bboxes.gravity_center.to(transformed_anchors.device)
#         if with_rotation_transform:
#             angle = yaw - 3.1415953 # num_agents, 1
#             rot_yaw = rot_2d(angle) # num_agents, 2, 2
#             rot_yaw = rot_yaw[:, None, None,:, :] # num_agents, 1, 1, 2, 2
#             transformed_anchors = rearrange(transformed_anchors, 'b g m t c -> b g m c t')  # 1, num_groups, num_modes, 12, 2 -> 1, num_groups, num_modes, 2, 12
#             transformed_anchors = torch.matmul(rot_yaw, transformed_anchors)# -> num_agents, num_groups, num_modes, 12, 2
#             transformed_anchors = rearrange(transformed_anchors, 'b g m c t -> b g m t c')
#         if with_translation_transform:
#             transformed_anchors = bbox_centers[:, None, None, None, :2] + transformed_anchors
#         batched_anchors.append(transformed_anchors)
#     return torch.stack(batched_anchors)
#
#
# def trajectory_coordinate_transform(trajectory, bbox_results, with_translation_transform=True, with_rotation_transform=True):
#     """
#     Transform trajectory coordinates with respect to detected bounding boxes in the batch.
#     Args:
#         trajectory (torch.Tensor): predicted trajectory.
#         bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
#         with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
#         with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.
#
#     Returns:
#         torch.Tensor: A tensor containing the transformed trajectory coordinates.
#     """
#     batch_size = len(bbox_results)
#     batched_trajectories = []
#     for i in range(batch_size):
#         bboxes, scores, labels, bbox_index, mask = bbox_results[i]
#         yaw = bboxes.yaw.to(trajectory.device)
#         bbox_centers = bboxes.gravity_center.to(trajectory.device)
#         transformed_trajectory = trajectory[i,...]
#         if with_rotation_transform:
#             # we take negtive here, to reverse the trajectory back to ego centric coordinate
#             angle = -(yaw - 3.1415953)
#             rot_yaw = rot_2d(angle)
#             rot_yaw = rot_yaw[:,None, None,:, :] # A, 1, 1, 2, 2
#             transformed_trajectory = rearrange(transformed_trajectory, 'a g p t c -> a g p c t') # A, G, P, 12 ,2 -> # A, G, P, 2, 12
#             transformed_trajectory = torch.matmul(rot_yaw, transformed_trajectory)# -> A, G, P, 12, 2
#             transformed_trajectory = rearrange(transformed_trajectory, 'a g p c t -> a g p t c')
#         if with_translation_transform:
#             transformed_trajectory = bbox_centers[:, None, None, None, :2] + transformed_trajectory
#         batched_trajectories.append(transformed_trajectory)
#     return torch.stack(batched_trajectories)
# =============================================================================
# END OF ORIGINAL TORCH CODE
# =============================================================================


#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


# =============================================================================
# bivariate_gaussian_activation
# =============================================================================
#
# ---- PyTorch ----
# def bivariate_gaussian_activation(ip):
#     mu_x  = ip[..., 0:1]
#     mu_y  = ip[..., 1:2]
#     sig_x = ip[..., 2:3]
#     sig_y = ip[..., 3:4]
#     rho   = ip[..., 4:5]
#     sig_x = torch.exp(sig_x)
#     sig_y = torch.exp(sig_y)
#     rho   = torch.tanh(rho)
#     out   = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
#     return out

# =============================================================================
# norm_points
# =============================================================================
#
# ---- PyTorch ----
# def norm_points(pos, pc_range):
#     x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
#     y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
#     return torch.stack([x_norm, y_norm], dim=-1)

# =============================================================================
# pos2posemb2d
# =============================================================================
#
# ---- PyTorch ----
# def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
#     scale = 2 * math.pi
#     pos = pos * scale
#     dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
#     dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
#     pos_x = pos[..., 0, None] / dim_t
#     pos_y = pos[..., 1, None] / dim_t
#     pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
#     pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
#     posemb = torch.cat((pos_y, pos_x), dim=-1)
#     return posemb

# =============================================================================
# rot_2d
# =============================================================================
#
# ---- PyTorch ----
# def rot_2d(yaw):
#     sy, cy = torch.sin(yaw), torch.cos(yaw)
#     out = torch.stack([torch.stack([cy, -sy]),
#                        torch.stack([sy,  cy])]).permute([2,0,1])
#     return out

# =============================================================================
# anchor_coordinate_transform
# =============================================================================
#
# ---- PyTorch ----
# def anchor_coordinate_transform(anchors, bbox_results, ...):
#     for each batch:
#         yaw = bboxes.yaw; centers = bboxes.gravity_center
#         angle = yaw - pi; rot_mat = rot_2d(angle)
#         anchors = rot_mat @ anchors + centers
#     return stacked

# =============================================================================
# trajectory_coordinate_transform
# =============================================================================
#
# ---- PyTorch ----
# def trajectory_coordinate_transform(trajectory, bbox_results, ...):
#     for each batch:
#         angle = -(yaw - pi); rot_mat = rot_2d(angle)
#         traj = rot_mat @ traj + centers
#     return stacked
# ----------------------------- TTSIM ------------------------------------

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — dynamic full-rank SliceF creation
# ═══════════════════════════════════════════════════════════════════════════════

def _slice_last_dim(module, tensor, start, end, op_name):
    """Dynamically create a full-rank SliceF for tensor[..., start:end]."""
    shape = list(tensor.shape)
    ndim = len(shape)
    out_shape = shape[:-1] + [end - start]

    starts_arr = np.zeros(ndim, dtype=np.int64);  starts_arr[-1] = start
    ends_arr   = np.array(shape, dtype=np.int64);  ends_arr[-1]   = end
    axes_arr   = np.arange(ndim, dtype=np.int64)
    steps_arr  = np.ones(ndim, dtype=np.int64)

    starts = F._from_data(f'{op_name}.starts', starts_arr, is_const=True)
    ends   = F._from_data(f'{op_name}.ends',   ends_arr,   is_const=True)
    axes   = F._from_data(f'{op_name}.axes',   axes_arr,   is_const=True)
    steps  = F._from_data(f'{op_name}.steps',  steps_arr,  is_const=True)
    for t in (starts, ends, axes, steps):
        setattr(module, t.name, t)

    slice_op = F.SliceF(op_name, out_shape=out_shape)
    setattr(module, slice_op.name, slice_op)
    slice_op.set_module(module)
    result = slice_op(tensor, starts, ends, axes, steps)
    setattr(module, result.name, result)
    return result


def _slice_last_dim_strided(module, tensor, start, end, step, op_name):
    """Dynamically create a full-rank SliceF for tensor[..., start:end:step]."""
    shape = list(tensor.shape)
    ndim = len(shape)
    out_last = len(range(start, end, step))
    out_shape = shape[:-1] + [out_last]

    starts_arr = np.zeros(ndim, dtype=np.int64);  starts_arr[-1] = start
    ends_arr   = np.array(shape, dtype=np.int64);  ends_arr[-1]   = end
    axes_arr   = np.arange(ndim, dtype=np.int64)
    steps_arr  = np.ones(ndim, dtype=np.int64);    steps_arr[-1]  = step

    starts = F._from_data(f'{op_name}.starts', starts_arr, is_const=True)
    ends   = F._from_data(f'{op_name}.ends',   ends_arr,   is_const=True)
    axes   = F._from_data(f'{op_name}.axes',   axes_arr,   is_const=True)
    steps  = F._from_data(f'{op_name}.steps',  steps_arr,  is_const=True)
    for t in (starts, ends, axes, steps):
        setattr(module, t.name, t)

    slice_op = F.SliceF(op_name, out_shape=out_shape)
    setattr(module, slice_op.name, slice_op)
    slice_op.set_module(module)
    result = slice_op(tensor, starts, ends, axes, steps)
    setattr(module, result.name, result)
    return result


class BivariateGaussianActivation(SimNN.Module):
    """
    Activation function to output parameters of bivariate Gaussian distribution.

    Splits input [..., 5] into (mu_x, mu_y, sig_x, sig_y, rho),
    applies exp to sigmas and tanh to rho, then concatenates back.
    """

    def __init__(self, name='BivariateGaussianActivation'):
        super().__init__()
        self.name = name

        # Slicing is done dynamically in __call__ via _slice_last_dim

        # Activation ops
        self.exp_sig_x = F.Exp(f'{name}.exp_sig_x')
        self.exp_sig_y = F.Exp(f'{name}.exp_sig_y')
        self.tanh_rho  = F.Tanh(f'{name}.tanh_rho')

        # Concatenation
        self.concat = F.ConcatX(f'{name}.concat', axis=-1)

        super().link_op2module()

    def __call__(self, ip):
        mu_x  = _slice_last_dim(self, ip, 0, 1, f'{self.name}.slice_mu_x')
        mu_y  = _slice_last_dim(self, ip, 1, 2, f'{self.name}.slice_mu_y')
        sig_x = _slice_last_dim(self, ip, 2, 3, f'{self.name}.slice_sig_x')
        sig_y = _slice_last_dim(self, ip, 3, 4, f'{self.name}.slice_sig_y')
        rho   = _slice_last_dim(self, ip, 4, 5, f'{self.name}.slice_rho')

        sig_x = self.exp_sig_x(sig_x)
        sig_y = self.exp_sig_y(sig_y)
        rho   = self.tanh_rho(rho)

        out = self.concat(mu_x, mu_y, sig_x, sig_y, rho)
        return out

    def analytical_param_count(self):
        """No learnable parameters — pure activation ops."""
        return 0



class NormPoints(SimNN.Module):
    """
    Normalize the endpoint coordinates of a position tensor into [0, 1]
    using the given point cloud range.

    Args:
        pc_range: list of 6 floats [x_min, y_min, z_min, x_max, y_max, z_max]
    """

    def __init__(self, name, pc_range):
        super().__init__()
        self.name = name

        x_range = float(pc_range[3] - pc_range[0])
        y_range = float(pc_range[4] - pc_range[1])

        # Slicing is done dynamically in __call__ via _slice_last_dim

        # Constants for subtracting min
        self.x_min_const = F._from_data(f'{name}.x_min',
                                        np.array(float(pc_range[0]), dtype=np.float32).reshape(1),
                                        is_const=True)
        self.y_min_const = F._from_data(f'{name}.y_min',
                                        np.array(float(pc_range[1]), dtype=np.float32).reshape(1),
                                        is_const=True)
        self.x_range_const = F._from_data(f'{name}.x_range',
                                          np.array(x_range, dtype=np.float32).reshape(1),
                                          is_const=True)
        self.y_range_const = F._from_data(f'{name}.y_range',
                                          np.array(y_range, dtype=np.float32).reshape(1),
                                          is_const=True)

        # Arithmetic ops
        self.sub_x = F.Sub(f'{name}.sub_x')
        self.sub_y = F.Sub(f'{name}.sub_y')
        self.div_x = F.Div(f'{name}.div_x')
        self.div_y = F.Div(f'{name}.div_y')

        # Stack (concat along new last dim)
        self.concat = F.ConcatX(f'{name}.concat', axis=-1)

        super().link_op2module()

    def __call__(self, pos):
        x = _slice_last_dim(self, pos, 0, 1, f'{self.name}.slice_x')   # [..., 1]
        y = _slice_last_dim(self, pos, 1, 2, f'{self.name}.slice_y')   # [..., 1]

        x_norm = self.div_x(self.sub_x(x, self.x_min_const), self.x_range_const)
        y_norm = self.div_y(self.sub_y(y, self.y_min_const), self.y_range_const)

        return self.concat(x_norm, y_norm)  # [..., 2]

    def analytical_param_count(self):
        """No learnable parameters — constants and arithmetic ops only."""
        return 0




class Pos2PosEmb2D(SimNN.Module):
    """
    Convert 2D position into sinusoidal positional embeddings.

    Output shape: [..., 2 * num_pos_feats]

    Args:
        num_pos_feats: Number of positional features per coordinate (default 128).
        temperature:   Temperature scaling factor (default 10000).
    """

    def __init__(self, name='Pos2PosEmb2D', num_pos_feats=128, temperature=10000):
        super().__init__()
        self.name = name
        self.num_pos_feats = num_pos_feats
        self.half_feats = num_pos_feats // 2  # for sin/cos interleaving

        # Pre-compute frequency denominator: temperature^(2*(i//2)/num_pos_feats)
        scale = np.float32(2.0 * np.pi)
        dim_t = np.arange(num_pos_feats, dtype=np.float32)
        dim_t = (temperature ** (2.0 * (dim_t // 2) / num_pos_feats)).astype(np.float32)

        # Store constants
        self.scale_const = F._from_data(f'{name}.scale',
                                        np.array(scale, dtype=np.float32).reshape(1),
                                        is_const=True)
        self.dim_t_const = F._from_data(f'{name}.dim_t',
                                        dim_t.reshape(1, num_pos_feats),
                                        is_const=True)

        # Ops for scaling position
        self.mul_scale_x = F.Mul(f'{name}.mul_scale_x')
        self.mul_scale_y = F.Mul(f'{name}.mul_scale_y')

        # Slicing is done dynamically in __call__ via _slice_last_dim helpers

        # Division by frequency denominator
        self.div_x = F.Div(f'{name}.div_x')
        self.div_y = F.Div(f'{name}.div_y')

        # Sin/Cos for even/odd frequency indices
        self.sin_x = F.Sin(f'{name}.sin_x')
        self.cos_x = F.Cos(f'{name}.cos_x')
        self.sin_y = F.Sin(f'{name}.sin_y')
        self.cos_y = F.Cos(f'{name}.cos_y')

        # Even/odd slice ops are created dynamically in __call__

        # Interleave: unsqueeze sin/cos to [..., half, 1], concat to [..., half, 2],
        # then reshape to [..., num_pos_feats] to match torch.stack((sin,cos),dim=-1).flatten(-2)
        self.unsq_sin_x = F.Unsqueeze(f'{name}.unsq_sin_x')
        self.unsq_cos_x = F.Unsqueeze(f'{name}.unsq_cos_x')
        self.unsq_sin_y = F.Unsqueeze(f'{name}.unsq_sin_y')
        self.unsq_cos_y = F.Unsqueeze(f'{name}.unsq_cos_y')
        self.unsq_axis  = F._from_data(f'{name}.unsq_axis',
                                       np.array([-1], dtype=np.int64), is_const=True)

        self.concat_x_interleave = F.ConcatX(f'{name}.concat_x_interleave', axis=-1)
        self.concat_y_interleave = F.ConcatX(f'{name}.concat_y_interleave', axis=-1)

        # Reshape [..., half, 2] -> [..., num_pos_feats]  (flatten last 2 dims)
        self.reshape_x = F.Reshape(f'{name}.reshape_x')
        self.reshape_y = F.Reshape(f'{name}.reshape_y')

        # Final concat [pos_y, pos_x]
        self.concat_output = F.ConcatX(f'{name}.concat_output', axis=-1)

        super().link_op2module()

    def __call__(self, pos):
        """
        Args:
            pos: [..., 2] tensor of (x, y) positions

        Returns:
            [..., 2 * num_pos_feats] positional embeddings
        """
        # Extract x and y coordinates (dynamic SliceF)
        x = _slice_last_dim(self, pos, 0, 1, f'{self.name}.slice_x')  # [..., 1]
        y = _slice_last_dim(self, pos, 1, 2, f'{self.name}.slice_y')  # [..., 1]

        # Scale by 2*pi
        x_scaled = self.mul_scale_x(x, self.scale_const)  # [..., 1]
        y_scaled = self.mul_scale_y(y, self.scale_const)   # [..., 1]

        # Divide by frequency denominators -> [..., num_pos_feats]
        pos_x = self.div_x(x_scaled, self.dim_t_const)
        pos_y = self.div_y(y_scaled, self.dim_t_const)

        # Apply sin to even indices, cos to odd indices (dynamic SliceF)
        npf = self.num_pos_feats
        pos_x_even = _slice_last_dim_strided(self, pos_x, 0, npf, 2, f'{self.name}.slice_x_even')
        pos_x_odd  = _slice_last_dim_strided(self, pos_x, 1, npf, 2, f'{self.name}.slice_x_odd')
        sin_x = self.sin_x(pos_x_even)
        cos_x = self.cos_x(pos_x_odd)

        pos_y_even = _slice_last_dim_strided(self, pos_y, 0, npf, 2, f'{self.name}.slice_y_even')
        pos_y_odd  = _slice_last_dim_strided(self, pos_y, 1, npf, 2, f'{self.name}.slice_y_odd')
        sin_y = self.sin_y(pos_y_even)
        cos_y = self.cos_y(pos_y_odd)

        # Interleave: [sin_0, cos_0, sin_1, cos_1, ...]
        # Unsqueeze -> [..., half, 1], concat -> [..., half, 2], reshape -> [..., npf]
        sin_x_u = self.unsq_sin_x(sin_x, self.unsq_axis)   # [..., half, 1]
        cos_x_u = self.unsq_cos_x(cos_x, self.unsq_axis)   # [..., half, 1]
        paired_x = self.concat_x_interleave(sin_x_u, cos_x_u)  # [..., half, 2]
        # Flatten last two dims to get interleaved [..., npf]
        flat_shape_x = list(paired_x.shape[:-2]) + [npf]
        flat_shape_x_t = F._from_data(f'{self.name}.flat_shape_x',
                                       np.array(flat_shape_x, dtype=np.int64), is_const=True)
        setattr(self, flat_shape_x_t.name, flat_shape_x_t)
        pos_x_emb = self.reshape_x(paired_x, flat_shape_x_t)  # [..., npf]
        setattr(self, pos_x_emb.name, pos_x_emb)

        sin_y_u = self.unsq_sin_y(sin_y, self.unsq_axis)
        cos_y_u = self.unsq_cos_y(cos_y, self.unsq_axis)
        paired_y = self.concat_y_interleave(sin_y_u, cos_y_u)
        flat_shape_y = list(paired_y.shape[:-2]) + [npf]
        flat_shape_y_t = F._from_data(f'{self.name}.flat_shape_y',
                                       np.array(flat_shape_y, dtype=np.int64), is_const=True)
        setattr(self, flat_shape_y_t.name, flat_shape_y_t)
        pos_y_emb = self.reshape_y(paired_y, flat_shape_y_t)
        setattr(self, pos_y_emb.name, pos_y_emb)

        # Final output: [pos_y, pos_x] to match PyTorch original
        posemb = self.concat_output(pos_y_emb, pos_x_emb)  # [..., 2*num_pos_feats]
        return posemb

    def analytical_param_count(self):
        """No learnable parameters — pre-computed frequency constants only."""
        return 0




class Rot2D(SimNN.Module):
    """
    Compute 2D rotation matrices from yaw angles.

    Input:  yaw [...] (1-D or N-D tensor of angles)
    Output: [..., 2, 2] rotation matrices
    """

    def __init__(self, name='Rot2D'):
        super().__init__()
        self.name = name

        self.sin_op  = F.Sin(f'{name}.sin')
        self.cos_op  = F.Cos(f'{name}.cos')
        self.neg_op  = F.Neg(f'{name}.neg')

        # Build the 2x2 matrix: [[cy, -sy], [sy, cy]]
        self.concat_row0 = F.ConcatX(f'{name}.row0', axis=-1)   # [cy, -sy] -> [..., 2]
        self.concat_row1 = F.ConcatX(f'{name}.row1', axis=-1)   # [sy, cy]  -> [..., 2]

        # Unsqueeze each row [..., 2] -> [..., 1, 2] so ConcatX stacks properly
        self.unsq_row0 = F.Unsqueeze(f'{name}.unsq_row0')
        self.unsq_row1 = F.Unsqueeze(f'{name}.unsq_row1')
        self.unsq_axis = F._from_data(f'{name}.unsq_axis',
                                      np.array([-2], dtype=np.int64), is_const=True)

        # Concat along -2: two [..., 1, 2] -> [..., 2, 2]
        self.concat_mat  = F.ConcatX(f'{name}.mat', axis=-2)

        super().link_op2module()

    def __call__(self, yaw):
        sy = self.sin_op(yaw)   # [..., 1]
        cy = self.cos_op(yaw)   # [..., 1]
        neg_sy = self.neg_op(sy)

        row0 = self.concat_row0(cy, neg_sy)                  # [..., 2]
        row1 = self.concat_row1(sy, cy)                      # [..., 2]

        # Unsqueeze to [..., 1, 2] so ConcatX creates a new matrix dim
        row0 = self.unsq_row0(row0, self.unsq_axis)          # [..., 1, 2]
        row1 = self.unsq_row1(row1, self.unsq_axis)          # [..., 1, 2]

        mat = self.concat_mat(row0, row1)                    # [..., 2, 2]
        return mat

    def analytical_param_count(self):
        """No learnable parameters — sin/cos/neg ops only."""
        return 0



class AnchorCoordinateTransform(SimNN.Module):
    """
    Transform k-means anchor trajectories from canonical frame
    to each detected agent's local frame (rotate + translate).

    Uses Rot2D internally for rotation matrices.
    """

    def __init__(self, name='AnchorCoordinateTransform'):
        super().__init__()
        self.name = name

        self.pi_const = F._from_data(f'{name}.pi',
                                     np.array(3.1415953, dtype=np.float32).reshape(1),
                                     is_const=True)

        self.sub_pi    = F.Sub(f'{name}.sub_pi')
        self.rot2d     = Rot2D(f'{name}.rot2d')
        # Unsqueeze rot_mat [A,2,2] -> [A,1,1,2,2] for broadcast with [1,G,M,2,T]
        self.rot_unsq1 = F.Unsqueeze(f'{name}.rot_unsq1')
        self.rot_unsq2 = F.Unsqueeze(f'{name}.rot_unsq2')
        self.rot_unsq_ax1 = F._from_data(f'{name}.rot_unsq_ax1', np.array([1], dtype=np.int64), is_const=True)
        self.rot_unsq_ax2 = F._from_data(f'{name}.rot_unsq_ax2', np.array([2], dtype=np.int64), is_const=True)
        self.matmul    = F.MatMul(f'{name}.matmul')
        self.add_trans = F.Add(f'{name}.add_translation')
        self.transpose_anc = F.Transpose(f'{name}.transpose_anc', perm=[0, 1, 2, 4, 3])
        self.transpose_back = F.Transpose(f'{name}.transpose_back', perm=[0, 1, 2, 4, 3])

        super().link_op2module()

    def __call__(self, anchors, yaw, bbox_centers,
                 with_rotation_transform=True, with_translation_transform=True):
        """
        Args:
            anchors:     [num_groups, num_modes, steps, 2]
            yaw:         [num_agents, 1]
            bbox_centers:[num_agents, 3]  (only [:2] used)

        Returns:
            transformed: [num_agents, num_groups, num_modes, steps, 2]
        """
        transformed = anchors

        if with_rotation_transform:
            angle = self.sub_pi(yaw, self.pi_const)        # [A, 1]
            rot_mat = self.rot2d(angle)                     # [A, 2, 2]
            rot_mat = self.rot_unsq1(rot_mat, self.rot_unsq_ax1)  # [A, 1, 2, 2]
            rot_mat = self.rot_unsq2(rot_mat, self.rot_unsq_ax2)  # [A, 1, 1, 2, 2]
            # anchors: [1, G, M, T, 2] -> transpose last two -> [1, G, M, 2, T]
            transformed = self.transpose_anc(transformed)
            transformed = self.matmul(rot_mat, transformed) # broadcast matmul
            transformed = self.transpose_back(transformed)  # [A, G, M, T, 2]

        if with_translation_transform:
            transformed = self.add_trans(bbox_centers, transformed)

        return transformed

    def analytical_param_count(self):
        """No learnable parameters — delegates to Rot2D submodule."""
        return self.rot2d.analytical_param_count()

class TrajectoryCoordinateTransform(SimNN.Module):
    """
    Transform predicted trajectories from agent-local frame
    back to ego-centric coordinate frame (inverse rotation + translate).
    """

    def __init__(self, name='TrajectoryCoordinateTransform'):
        super().__init__()
        self.name = name

        self.pi_const = F._from_data(f'{name}.pi',
                                     np.array(3.1415953, dtype=np.float32).reshape(1),
                                     is_const=True)

        self.sub_pi    = F.Sub(f'{name}.sub_pi')
        self.neg_angle = F.Neg(f'{name}.neg_angle')
        self.rot2d     = Rot2D(f'{name}.rot2d')
        # Unsqueeze rot_mat [A,2,2] -> [A,1,1,2,2] for broadcast with [A,G,P,2,T]
        self.rot_unsq1 = F.Unsqueeze(f'{name}.rot_unsq1')
        self.rot_unsq2 = F.Unsqueeze(f'{name}.rot_unsq2')
        self.rot_unsq_ax1 = F._from_data(f'{name}.rot_unsq_ax1', np.array([1], dtype=np.int64), is_const=True)
        self.rot_unsq_ax2 = F._from_data(f'{name}.rot_unsq_ax2', np.array([2], dtype=np.int64), is_const=True)
        self.matmul    = F.MatMul(f'{name}.matmul')
        self.add_trans = F.Add(f'{name}.add_translation')
        self.transpose_traj = F.Transpose(f'{name}.transpose_traj', perm=[0, 1, 2, 4, 3])
        self.transpose_back = F.Transpose(f'{name}.transpose_back', perm=[0, 1, 2, 4, 3])

        super().link_op2module()

    def __call__(self, trajectory, yaw, bbox_centers,
                 with_rotation_transform=True, with_translation_transform=True):
        """
        Args:
            trajectory:  [A, G, P, T, 2]  (per-batch slice)
            yaw:         [A, 1]
            bbox_centers:[A, 3]

        Returns:
            transformed: [A, G, P, T, 2]
        """
        transformed = trajectory

        if with_rotation_transform:
            angle = self.sub_pi(yaw, self.pi_const)     # [A, 1]
            angle = self.neg_angle(angle)                # negate for inverse rotation
            rot_mat = self.rot2d(angle)                  # [A, 2, 2]
            rot_mat = self.rot_unsq1(rot_mat, self.rot_unsq_ax1)  # [A, 1, 2, 2]
            rot_mat = self.rot_unsq2(rot_mat, self.rot_unsq_ax2)  # [A, 1, 1, 2, 2]
            transformed = self.transpose_traj(transformed)  # [A, G, P, 2, T]
            transformed = self.matmul(rot_mat, transformed)
            transformed = self.transpose_back(transformed)  # [A, G, P, T, 2]

        if with_translation_transform:
            transformed = self.add_trans(bbox_centers, transformed)

        return transformed

    def analytical_param_count(self):
        """No learnable parameters -- delegates to Rot2D submodule."""
        return self.rot2d.analytical_param_count()
