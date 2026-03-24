#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
ttsim version of segmentation losses for MapTracker
Structure matches maptracker/plugin/models/losses/seg_loss.py
"""

# -------------------------------PyTorch--------------------------------

# import torch
# from torch import nn as nn
# from torch.nn import functional as F
# import mmcv
#
# from mmdet.models.builder import LOSSES
# from mmdet.models.losses import FocalLoss, weight_reduce_loss
#
# from einops import rearrange
#
#
# def py_sigmoid_focal_loss(pred,
#                           target,
#                           weight=None,
#                           gamma=2.0,
#                           alpha=0.25,
#                           reduction='mean',
#                           avg_factor=None):
#     """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
#
#     Args:
#         pred (torch.Tensor): The prediction with shape (N, C), C is the
#             number of classes
#         target (torch.Tensor): The learning label of the prediction.
#         weight (torch.Tensor, optional): Sample-wise loss weight.
#         gamma (float, optional): The gamma for calculating the modulating
#             factor. Defaults to 2.0.
#         alpha (float, optional): A balanced form for Focal Loss.
#             Defaults to 0.25.
#         reduction (str, optional): The method used to reduce the loss into
#             a scalar. Defaults to 'mean'.
#         avg_factor (int, optional): Average factor that is used to average
#             the loss. Defaults to None.
#     """
#     pred_sigmoid = pred.sigmoid()
#     target = target.type_as(pred)
#     pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
#     focal_weight = (alpha * target + (1 - alpha) *
#                     (1 - target)) * pt.pow(gamma)
#     loss = F.binary_cross_entropy_with_logits(
#         pred, target, reduction='none') * focal_weight
#     if weight is not None:
#         if weight.shape != loss.shape:
#             if weight.size(0) == loss.size(0):
#                 # For most cases, weight is of shape (num_priors, ),
#                 #  which means it does not have the second axis num_class
#                 weight = weight.view(-1, 1)
#             else:
#                 # Sometimes, weight per anchor per class is also needed. e.g.
#                 #  in FSAF. But it may be flattened of shape
#                 #  (num_priors x num_class, ), while loss is still of shape
#                 #  (num_priors, num_class).
#                 assert weight.numel() == loss.numel()
#                 weight = weight.view(loss.size(0), -1)
#         assert weight.ndim == loss.ndim
#     loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
#     return loss
#
#
# @LOSSES.register_module()
# class MaskFocalLoss(FocalLoss):
#     def __init__(self,**kwargs):
#         super(MaskFocalLoss, self).__init__(**kwargs)
#
#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None):
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         if not self.use_sigmoid:
#             raise NotImplementedError
#
#         num_classes = pred.size(1)
#         loss = 0
#         for index in range(num_classes):
#             loss += self.loss_weight * py_sigmoid_focal_loss(
#                 pred[:,index],
#                 target[:,index],
#                 weight,
#                 gamma=self.gamma,
#                 alpha=self.alpha,
#                 reduction=reduction,
#                 avg_factor=avg_factor)
#
#         loss /= num_classes
#         return loss * self.loss_weight
#
#
# @LOSSES.register_module()
# class MaskDiceLoss(nn.Module):
#     """Dice Loss PyTorch
#         Created by: Zhang Shuai
#         Email: shuaizzz666@gmail.com
#         dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
#     Args:
#         weight: An array of shape [C,]
#         predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
#         target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
#     Return:
#         diceloss
#     """
#     def __init__(self, loss_weight):
#         super(MaskDiceLoss, self).__init__()
#         self.smooth = 1e-5
#         self.loss_weight = loss_weight
#
#     def forward(self, pred, target):
#         bs, num_classes = pred.shape[:2]
#         pred = rearrange(pred, 'b n h w -> b n (h w)')
#         target = rearrange(target, 'b n h w -> b n (h w)')
#         pred = pred.sigmoid()
#         intersection = torch.sum(pred * target, dim=2)  # (N, C)
#         union = torch.sum(pred.pow(2), dim=2) + torch.sum(target, dim=2)  # (N, C)
#         ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
#         dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)
#         dice_loss = 1 - torch.mean(dice_coef)  # 1
#
#         loss = self.loss_weight * dice_loss
#         return loss

# -------------------------------TTSIM-----------------------------------


import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class MaskFocalLoss(SimNN.Module):
    """
    Focal Loss for segmentation masks (ttsim version)

    focal_loss = -alpha * (1-p_t)^gamma * log(p_t)
    where p_t = p if target=1, else 1-p
    """

    def __init__(self, gamma=2.0, alpha=0.25, loss_weight=1.0):
        """
        Args:
            gamma (float): Focusing parameter. Default: 2.0
            alpha (float): Balancing parameter. Default: 0.25
            loss_weight (float): Loss weight multiplier. Default: 1.0
        """
        super(MaskFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

        # Create operators in __init__
        self.sigmoid = F.Sigmoid("focal_sigmoid")

        # For BCE: -[t*log(p) + (1-t)*log(1-p)]
        self.log = F.Log("focal_log")
        self.log1p = F.Log("focal_log1p")  # log(1-p)

        # For focal weight: alpha * t + (1-alpha) * (1-t)
        self.mul_alpha_t = F.Mul("focal_mul_alpha_t")
        self.mul_one_minus_alpha_t = F.Mul("focal_mul_one_minus_alpha_t")
        self.add_focal_alpha = F.Add("focal_add_alpha")

        # For modulating factor: (1-p)*t + p*(1-p)
        self.sub_one_minus_p = F.Sub("focal_sub_one_minus_p")
        self.sub_one_minus_t = F.Sub("focal_sub_one_minus_t")
        self.mul_pt1 = F.Mul("focal_mul_pt1")
        self.mul_pt2 = F.Mul("focal_mul_pt2")
        self.add_pt = F.Add("focal_add_pt")

        # Power and final multiplication
        self.pow_gamma = F.Pow("focal_pow_gamma")
        self.mul_focal_weight = F.Mul("focal_mul_weight")

        # BCE computation
        self.mul_bce_pos = F.Mul("focal_bce_pos")  # t * log(p)
        self.mul_bce_neg = F.Mul("focal_bce_neg")  # (1-t) * log(1-p)
        self.add_bce = F.Add("focal_add_bce")
        self.neg_bce = F.Neg("focal_neg_bce")

        # Final loss computation
        self.mul_bce_focal = F.Mul("focal_mul_bce_focal")
        self.reduce_mean_classes = F.ReduceMean("focal_reduce_classes", axes=[1])
        self.reduce_mean_batch = F.ReduceMean("focal_reduce_batch", axes=[0])
        self.mul_loss_weight = F.Mul("focal_mul_loss_weight")

        # Store constants as SimTensors
        self._tensors = {}

    def __call__(self, pred, target):
        """
        Forward pass for focal loss.

        Args:
            pred (SimTensor): Predictions (B, C, H, W) - logits
            target (SimTensor): Targets (B, C, H, W) - binary labels

        Returns:
            SimTensor: Scalar focal loss
        """
        # Create constants with proper shapes
        one = F._from_data(
            "focal_one", np.ones(pred.shape, dtype=np.float32), is_const=True
        )
        alpha_const = F._from_data(
            "focal_alpha",
            np.full(pred.shape, self.alpha, dtype=np.float32),
            is_const=True,
        )
        one_minus_alpha = F._from_data(
            "focal_one_minus_alpha",
            np.full(pred.shape, 1.0 - self.alpha, dtype=np.float32),
            is_const=True,
        )
        gamma_const = F._from_data(
            "focal_gamma", np.array([self.gamma], dtype=np.float32), is_const=True
        )
        loss_weight_const = F._from_data(
            "focal_loss_weight",
            np.array([self.loss_weight], dtype=np.float32),
            is_const=True,
        )

        # Apply sigmoid to get probabilities
        p = self.sigmoid(pred)

        # Compute p_t: (1-p)*t + p*(1-t)
        one_minus_p = self.sub_one_minus_p(one, p)
        one_minus_t = self.sub_one_minus_t(one, target)
        pt1 = self.mul_pt1(one_minus_p, target)
        pt2 = self.mul_pt2(p, one_minus_t)
        pt = self.add_pt(pt1, pt2)

        # Compute focal weight: alpha*t + (1-alpha)*(1-t)
        alpha_t = self.mul_alpha_t(alpha_const, target)
        one_minus_alpha_t = self.mul_one_minus_alpha_t(one_minus_alpha, one_minus_t)
        focal_alpha = self.add_focal_alpha(alpha_t, one_minus_alpha_t)

        # Modulating factor: (alpha*t + (1-alpha)*(1-t)) * pt^gamma
        pt_gamma = self.pow_gamma(pt, gamma_const)
        focal_weight = self.mul_focal_weight(focal_alpha, pt_gamma)

        # BCE: -[t*log(p) + (1-t)*log(1-p)]
        log_p = self.log(p)
        log_one_minus_p = self.log1p(one_minus_p)
        bce_pos = self.mul_bce_pos(target, log_p)
        bce_neg = self.mul_bce_neg(one_minus_t, log_one_minus_p)
        bce_sum = self.add_bce(bce_pos, bce_neg)
        bce = self.neg_bce(bce_sum)

        # Final loss: BCE * focal_weight
        loss = self.mul_bce_focal(bce, focal_weight)

        # Reduce: mean over classes, then mean over batch
        loss = self.reduce_mean_classes(loss)
        loss = self.reduce_mean_batch(loss)

        # Apply loss weight
        loss = self.mul_loss_weight(loss, loss_weight_const)

        return loss

    def analytical_param_count(self, lvl):
        """
        Calculate total number of trainable parameters
        Loss modules have no trainable parameters

        Returns:
            int: 0 (no trainable parameters)
        """
        return 0


class MaskDiceLoss(SimNN.Module):
    """
    Dice Loss for segmentation masks (ttsim version)

    dice_loss = 1 - 2*intersection / (pred^2 + target)
    """

    def __init__(self, loss_weight=1.0):
        """
        Args:
            loss_weight (float): Loss weight multiplier. Default: 1.0
        """
        super(MaskDiceLoss, self).__init__()
        self.smooth = 1e-5
        self.loss_weight = loss_weight

        # Create operators in __init__
        self.sigmoid = F.Sigmoid("dice_sigmoid")

        # Intersection: pred * target
        self.mul_intersection = F.Mul("dice_mul_intersection")
        self.sum_intersection = F.ReduceSum("dice_sum_intersection", axes=[2])

        # Union: pred^2 + target
        self.pow_pred = F.Pow("dice_pow_pred")
        self.sum_pred_sq = F.ReduceSum("dice_sum_pred_sq", axes=[2])
        self.sum_target = F.ReduceSum("dice_sum_target", axes=[2])
        self.add_union = F.Add("dice_add_union")

        # Dice coefficient: (2*intersection + smooth) / (union + smooth)
        self.mul_2 = F.Mul("dice_mul_2")
        self.add_smooth_num = F.Add("dice_add_smooth_num")
        self.add_smooth_denom = F.Add("dice_add_smooth_denom")
        self.div_dice = F.Div("dice_div")

        # Final loss: 1 - mean(dice)
        self.mean_dice = F.ReduceMean("dice_mean", axes=[0, 1])
        self.sub_loss = F.Sub("dice_sub_loss")

        # Apply loss weight
        self.mul_weight = F.Mul("dice_mul_weight")

        self._tensors = {}

    def __call__(self, pred, target):
        """
        Forward pass for dice loss.

        Args:
            pred (SimTensor): Predictions (B, C, H, W) - logits
            target (SimTensor): Targets (B, C, H, W) - binary labels

        Returns:
            SimTensor: Scalar dice loss
        """
        # Reshape to (B, C, H*W)
        B, C, H, W = pred.shape
        pred_shape = F._from_data(
            "dice_pred_shape", np.array([B, C, H * W], dtype=np.int64)
        )
        target_shape = F._from_data(
            "dice_target_shape", np.array([B, C, H * W], dtype=np.int64)
        )

        reshape_pred = F.Reshape("dice_reshape_pred")
        reshape_target = F.Reshape("dice_reshape_target")

        pred = reshape_pred(pred, pred_shape)
        target = reshape_target(target, target_shape)

        # Create constants
        two = F._from_data("dice_two", np.array([2.0], dtype=np.float32), is_const=True)
        one = F._from_data("dice_one", np.array([1.0], dtype=np.float32), is_const=True)
        smooth_tensor = F._from_data(
            "dice_smooth", np.array([self.smooth], dtype=np.float32), is_const=True
        )
        weight_tensor = F._from_data(
            "dice_weight", np.array([self.loss_weight], dtype=np.float32), is_const=True
        )

        # Apply sigmoid
        pred = self.sigmoid(pred)

        # Compute intersection: sum(pred * target) along spatial dim
        intersection = self.mul_intersection(pred, target)
        intersection = self.sum_intersection(intersection)

        # Compute union: sum(pred^2) + sum(target)
        pred_squared = self.pow_pred(pred, two)
        pred_sum = self.sum_pred_sq(pred_squared)
        target_sum = self.sum_target(target)
        union = self.add_union(pred_sum, target_sum)

        # Dice coefficient: (2*intersection + smooth) / (union + smooth)
        numerator = self.mul_2(two, intersection)
        numerator = self.add_smooth_num(numerator, smooth_tensor)
        denominator = self.add_smooth_denom(union, smooth_tensor)
        dice_coef = self.div_dice(numerator, denominator)

        # Mean dice coefficient over batch and classes
        mean_dice_coef = self.mean_dice(dice_coef)

        # Loss = 1 - mean_dice
        loss = self.sub_loss(one, mean_dice_coef)

        # Apply loss weight
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
