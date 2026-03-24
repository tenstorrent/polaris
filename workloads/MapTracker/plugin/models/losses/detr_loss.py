#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
ttsim version of DETR losses for MapTracker
Structure matches maptracker/plugin/models/losses/detr_loss.py
"""

# -------------------------------PyTorch--------------------------------

# import torch
# from torch import nn as nn
# from torch.nn import functional as F
# from mmdet.models.losses import l1_loss, smooth_l1_loss
# from mmdet.models.losses.utils import weighted_loss
# import mmcv
#
# from mmdet.models.builder import LOSSES
#
#
# @LOSSES.register_module()
# class LinesL1Loss(nn.Module):
#
#     def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
#         """
#             L1 loss. The same as the smooth L1 loss
#             Args:
#                 reduction (str, optional): The method to reduce the loss.
#                     Options are "none", "mean" and "sum".
#                 loss_weight (float, optional): The weight of loss.
#         """
#
#         super().__init__()
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.beta = beta
#
#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None):
#         """Forward function.
#         Args:
#             pred (torch.Tensor): The prediction.
#                 shape: [bs, ...]
#             target (torch.Tensor): The learning target of the prediction.
#                 shape: [bs, ...]
#             weight (torch.Tensor, optional): The weight of loss for each
#                 prediction. Defaults to None.
#                 it's useful when the predictions are not all valid.
#             avg_factor (int, optional): Average factor that is used to average
#                 the loss. Defaults to None.
#             reduction_override (str, optional): The reduction method used to
#                 override the original reduction method of the loss.
#                 Defaults to None.
#         """
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#
#         if self.beta > 0:
#             loss = smooth_l1_loss(
#                 pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
#
#         else:
#             loss = l1_loss(
#                 pred, target, weight, reduction=reduction, avg_factor=avg_factor)
#
#         num_points = pred.shape[-1] // 2
#         loss = loss / num_points
#
#         return loss*self.loss_weight
#
#
# @mmcv.jit(derivate=True, coderize=True)
# @weighted_loss
# def bce(pred, label, class_weight=None):
#     """
#         pred: B,nquery,npts
#         label: B,nquery,npts
#     """
#
#     if label.numel() == 0:
#         return pred.sum() * 0
#     assert pred.size() == label.size()
#
#     loss = F.binary_cross_entropy_with_logits(
#         pred, label.float(), pos_weight=class_weight, reduction='none')
#
#     return loss
#
#
# @LOSSES.register_module()
# class MasksLoss(nn.Module):
#
#     def __init__(self, reduction='mean', loss_weight=1.0):
#         super(MasksLoss, self).__init__()
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#
#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None):
#         """Forward function.
#         Args:
#             xxx
#         """
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#
#         loss = bce(pred, target, weight, reduction=reduction,
#                    avg_factor=avg_factor)
#
#         return loss*self.loss_weight
#
# @mmcv.jit(derivate=True, coderize=True)
# @weighted_loss
# def ce(pred, label, class_weight=None):
#     """
#         pred: B*nquery,npts
#         label: B*nquery,
#     """
#
#     if label.numel() == 0:
#         return pred.sum() * 0
#
#     loss = F.cross_entropy(
#         pred, label, weight=class_weight, reduction='none')
#
#     return loss
#
#
# @LOSSES.register_module()
# class LenLoss(nn.Module):
#
#     def __init__(self, reduction='mean', loss_weight=1.0):
#         super(LenLoss, self).__init__()
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#
#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None):
#         """Forward function.
#         Args:
#             xxx
#         """
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#
#         loss = ce(pred, target, weight, reduction=reduction,
#                    avg_factor=avg_factor)
#
#         return loss*self.loss_weight

# -------------------------------TTSIM-----------------------------------


import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from functools import partial
import numpy as np


class LinesL1Loss(SimNN.Module):
    """
    L1 or Smooth L1 loss for line predictions (ttsim version)
    """

    def __init__(self, reduction="mean", loss_weight=1.0, beta=0.5):
        """
        Args:
            reduction (str): Reduction method. Options: 'none', 'mean', 'sum'
            loss_weight (float): Loss weight multiplier
            beta (float): Threshold for smooth L1. If beta > 0, use smooth L1, else use L1
        """
        super(LinesL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

        # Create operators in __init__
        self.sub_diff = F.Sub("l1_sub_diff")
        self.abs_diff = F.Abs("l1_abs")

        if beta > 0:
            # Smooth L1 operators
            self.pow_sq = F.Pow("l1_pow_sq")
            self.mul_half = F.Mul("l1_mul_half")
            self.div_beta = F.Div("l1_div_beta")
            self.sub_linear = F.Sub("l1_sub_linear")
            self.less = F.Less("l1_less")
            self.where = F.Where("l1_where")

        # Reduction operators
        if reduction == "mean":
            self.reduce = F.ReduceMean("l1_reduce_mean", axes=None)
        elif reduction == "sum":
            self.reduce = F.ReduceSum("l1_reduce_sum", axes=None)
        # else: no reduction operator needed for 'none'

        # Weight and normalization
        self.mul_weight = F.Mul("l1_mul_weight")
        self.div_num_points = F.Div("l1_div_num_points")

        self._tensors = {}

    def __call__(self, pred, target, weight=None, avg_factor=None):
        """
        Forward pass for L1/Smooth L1 loss.

        Args:
            pred (SimTensor): Predictions (bs, ..., coords)
            target (SimTensor): Targets (bs, ..., coords)
            weight (SimTensor, optional): Sample-wise weights
            avg_factor (float, optional): Average factor for loss reduction

        Returns:
            SimTensor: L1 loss
        """
        # Compute difference
        diff = self.sub_diff(pred, target)
        abs_diff = self.abs_diff(diff)

        if self.beta > 0:
            # Smooth L1: where(|diff| < beta, 0.5*diff^2/beta, |diff| - 0.5*beta)
            half = F._from_data(
                "l1_half", np.array([0.5], dtype=np.float32), is_const=True
            )
            two = F._from_data(
                "l1_two", np.array([2.0], dtype=np.float32), is_const=True
            )
            beta_tensor = F._from_data(
                "l1_beta", np.array([self.beta], dtype=np.float32), is_const=True
            )
            half_beta = F._from_data(
                "l1_half_beta",
                np.array([0.5 * self.beta], dtype=np.float32),
                is_const=True,
            )

            # Smooth part: 0.5 * diff^2 / beta
            diff_sq = self.pow_sq(diff, two)
            smooth_part = self.mul_half(half, diff_sq)
            smooth_part = self.div_beta(smooth_part, beta_tensor)

            # Linear part: |diff| - 0.5*beta
            linear_part = self.sub_linear(abs_diff, half_beta)

            # Conditional selection
            condition = self.less(abs_diff, beta_tensor)
            loss = self.where(condition, smooth_part, linear_part)
        else:
            # Standard L1
            loss = abs_diff

        # Apply sample-wise weight if provided
        if weight is not None:
            loss = self.mul_weight(loss, weight)

        # Reduce loss
        if self.reduction != "none":
            if avg_factor is not None:
                # Custom average factor
                sum_loss = F.ReduceSum("l1_sum_for_avg", axes=None)(loss)
                avg_factor_tensor = F._from_data(
                    "l1_avg_factor",
                    np.array([avg_factor], dtype=np.float32),
                    is_const=True,
                )
                div_avg = F.Div("l1_div_avg")
                loss = div_avg(sum_loss, avg_factor_tensor)
            else:
                loss = self.reduce(loss)

        # Normalize by number of points (coords / 2)
        num_points = pred.shape[-1] // 2
        num_points_tensor = F._from_data(
            "l1_num_points", np.array([num_points], dtype=np.float32), is_const=True
        )
        loss = self.div_num_points(loss, num_points_tensor)

        # Apply loss weight
        weight_tensor = F._from_data(
            "l1_weight", np.array([self.loss_weight], dtype=np.float32), is_const=True
        )
        final_mul = F.Mul("l1_final_mul")
        loss = final_mul(loss, weight_tensor)

        return loss

    def analytical_param_count(self, lvl):
        """
        Calculate total number of trainable parameters
        Loss modules have no trainable parameters

        Returns:
            int: 0 (no trainable parameters)
        """
        return 0


class MasksLoss(SimNN.Module):
    """
    Binary Cross Entropy loss for masks (ttsim version)
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        """
        Args:
            reduction (str): Reduction method. Options: 'none', 'mean', 'sum'
            loss_weight (float): Loss weight multiplier
        """
        super(MasksLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        # BCE = -(t*log(sigmoid(x)) + (1-t)*log(1-sigmoid(x)))
        # Using log-sum-exp trick for numerical stability:
        # BCE = max(x,0) - x*t + log(1 + exp(-|x|))

        self.relu = F.Relu("bce_relu")  # max(x, 0)
        self.mul_xt = F.Mul("bce_mul_xt")
        self.sub_max_xt = F.Sub("bce_sub_max_xt")
        self.abs_x = F.Abs("bce_abs")
        self.neg_abs = F.Neg("bce_neg_abs")
        self.exp = F.Exp("bce_exp")
        self.add_one = F.Add("bce_add_one")
        self.log = F.Log("bce_log")
        self.add_loss = F.Add("bce_add_loss")

        # Reduction
        if reduction == "mean":
            self.reduce = F.ReduceMean("bce_reduce_mean", axes=None)
        elif reduction == "sum":
            self.reduce = F.ReduceSum("bce_reduce_sum", axes=None)

        # Weight
        self.mul_weight = F.Mul("bce_mul_weight")

        self._tensors = {}

    def __call__(self, pred, target, weight=None, avg_factor=None):
        """
        Forward pass for BCE loss.

        Args:
            pred (SimTensor): Predictions (B, nquery, npts) - logits
            target (SimTensor): Targets (B, nquery, npts) - binary labels
            weight (SimTensor, optional): Sample-wise weights
            avg_factor (float, optional): Average factor

        Returns:
            SimTensor: BCE loss
        """
        # Create constants
        zero = F._from_data(
            "bce_zero", np.array([0.0], dtype=np.float32), is_const=True
        )
        one = F._from_data("bce_one", np.array([1.0], dtype=np.float32), is_const=True)

        # BCE with log-sum-exp trick: max(x,0) - x*t + log(1 + exp(-|x|))
        max_x_zero = self.relu(pred)  # ReLU is max(x, 0)
        x_times_t = self.mul_xt(pred, target)
        term1 = self.sub_max_xt(max_x_zero, x_times_t)

        abs_x = self.abs_x(pred)
        neg_abs_x = self.neg_abs(abs_x)
        exp_term = self.exp(neg_abs_x)
        one_plus_exp = self.add_one(one, exp_term)
        log_term = self.log(one_plus_exp)

        loss = self.add_loss(term1, log_term)

        # Apply sample-wise weight if provided
        if weight is not None:
            mul_sample_weight = F.Mul("bce_mul_sample_weight")
            loss = mul_sample_weight(loss, weight)

        # Reduce loss
        if self.reduction != "none":
            if avg_factor is not None:
                sum_loss = F.ReduceSum("bce_sum_for_avg", axes=None)(loss)
                avg_factor_tensor = F._from_data(
                    "bce_avg_factor",
                    np.array([avg_factor], dtype=np.float32),
                    is_const=True,
                )
                div_avg = F.Div("bce_div_avg")
                loss = div_avg(sum_loss, avg_factor_tensor)
            else:
                loss = self.reduce(loss)

        # Apply loss weight
        weight_tensor = F._from_data(
            "bce_weight", np.array([self.loss_weight], dtype=np.float32), is_const=True
        )
        loss = self.mul_weight(loss, weight_tensor)

        return loss

    def analytical_param_count(self, lvl):
        """
        Calculate total number of trainable parameters
        Loss modules have no trainable parameters

        Returns:
            int: 0 (no trainable parameters)
        """
        return 0


class LenLoss(SimNN.Module):
    """
    Cross Entropy loss for length classification (ttsim version)
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        """
        Args:
            reduction (str): Reduction method. Options: 'none', 'mean', 'sum'
            loss_weight (float): Loss weight multiplier
        """
        super(LenLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        # CrossEntropy = -log(exp(x[target]) / sum(exp(x)))
        # Using log-sum-exp for stability:
        # CE = log(sum(exp(x))) - x[target]
        #
        # Since ttsim doesn't have GatherElements for batched indexing,
        # we use one-hot encoding + sum to extract target logits

        self.exp = F.Exp("ce_exp")
        self.sum_exp = F.ReduceSum("ce_sum_exp", axes=[1], keepdims=0)
        self.log_sum = F.Log("ce_log_sum")

        # For extracting target logits: use one-hot multiplication
        self.mul_onehot = F.Mul("ce_mul_onehot")
        self.sum_selected = F.ReduceSum("ce_sum_selected", axes=[1], keepdims=0)

        self.sub_ce = F.Sub("ce_sub")

        # Reduction
        if reduction == "mean":
            self.reduce = F.ReduceMean("ce_reduce_mean", axes=None)
        elif reduction == "sum":
            self.reduce = F.ReduceSum("ce_reduce_sum", axes=None)

        # Weight
        self.mul_weight = F.Mul("ce_mul_weight")

        self._tensors = {}

    def __call__(self, pred, target, weight=None, avg_factor=None):
        """
        Forward pass for cross entropy loss.

        Args:
            pred (SimTensor): Predictions (B*nquery, npts) - logits
            target (SimTensor): Targets (B*nquery,) - class indices
            weight (SimTensor, optional): Sample-wise weights
            avg_factor (float, optional): Average factor

        Returns:
            SimTensor: CE loss
        """
        # Compute log(sum(exp(x)))
        exp_x = self.exp(pred)
        sum_exp_x = self.sum_exp(exp_x)
        log_sum_exp = self.log_sum(sum_exp_x)

        # Extract x[target] using one-hot encoding
        # Create one-hot from target indices: shape (N, npts)
        # Since we can't easily create one-hot in ttsim, we'll create it as input data
        npts = pred.shape[1] if len(pred.shape) > 1 else 20

        # Create one-hot encoding from target
        # This needs to be done with the actual target data
        import numpy as np

        if target.data is not None:
            one_hot = np.zeros((target.shape[0], npts), dtype=np.float32)
            one_hot[np.arange(target.shape[0]), target.data.astype(np.int64)] = 1.0
            one_hot_tensor = F._from_data("ce_onehot", one_hot, is_const=True)
        else:
            # Fallback if target.data is None - this shouldn't happen in normal use
            one_hot_tensor = F._from_shape(
                "ce_onehot", list(pred.shape), is_const=False
            )

        # Multiply pred by one-hot and sum to get target logits
        selected = self.mul_onehot(pred, one_hot_tensor)
        target_logits = self.sum_selected(selected)

        # CE = log_sum_exp - target_logits
        loss = self.sub_ce(log_sum_exp, target_logits)

        # Apply sample-wise weight if provided
        if weight is not None:
            mul_sample_weight = F.Mul("ce_mul_sample_weight")
            loss = mul_sample_weight(loss, weight)

        # Reduce loss
        if self.reduction != "none":
            if avg_factor is not None:
                sum_loss = F.ReduceSum("l1_sum_for_avg", axes=None)(loss)
                avg_factor_tensor = F._from_data(
                    "ce_avg_factor",
                    np.array([avg_factor], dtype=np.float32),
                    is_const=True,
                )
                div_avg = F.Div("ce_div_avg")
                loss = div_avg(sum_loss, avg_factor_tensor)
            else:
                loss = self.reduce(loss)

        # Apply loss weight
        weight_tensor = F._from_data(
            "ce_weight", np.array([self.loss_weight], dtype=np.float32), is_const=True
        )
        loss = self.mul_weight(loss, weight_tensor)

        return loss

    def analytical_param_count(self, lvl):
        """
        Calculate total number of trainable parameters
        Loss modules have no trainable parameters

        Returns:
            int: 0 (no trainable parameters)
        """
        return 0
