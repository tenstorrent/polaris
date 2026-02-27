#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Deformable DETR TTSim Conversion
================================

Full TTSim conversion of models/deformable_detr.py.

Converted modules:
- DeformableDETR         — main detection model
- SetCriterion           — loss computation (hungarian matching + focal/L1/GIoU losses)
- PostProcess            — inference post-processing (top-k + box conversion)
- PostProcessSegm        — segmentation post-processing (mask interpolation + thresholding)
- MLP                    — simple feed-forward network
- build()                — factory function returning (model, criterion, postprocessors)

NOT converted (explicitly excluded):
- PostProcessPanoptic    — requires panopticapi (PIL Image, id2rgb/rgb2id), domain-specific

Key conversions:
- nn.Module        → SimNN.Module
- nn.Linear        → SimNN.Linear
- nn.Embedding     → F._from_shape (parameter tensor)
- nn.Conv2d        → F.Conv2d
- nn.GroupNorm      → SimNN.GroupNorm
- nn.ModuleList    → SimNN.ModuleList
- F.relu           → F.Relu
- torch.stack      → stack
- torch.cat        → cat
- .sigmoid()       → F.Sigmoid / np sigmoid
- torch.log        → np.log
- F.l1_loss        → np L1
- F.binary_cross_entropy_with_logits → np implementation
"""

import os, sys
import numpy as np
from typing import Optional, List, Dict, Union
import copy
from types import SimpleNamespace

# Ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops.tensor import SimTensor
from ttsim.front.functional.tensor_op import cat, stack

# Import ttsim-converted versions
from workloads.Deformable_DETR.models.backbone_ttsim import build_backbone
from workloads.Deformable_DETR.models.matcher_ttsim import build_matcher
from workloads.Deformable_DETR.models.segmentation_ttsim import (
    DETRsegm,
)
from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
    build_deformable_transformer,
)
from workloads.Deformable_DETR.util import box_ops_ttsim as box_ops
from workloads.Deformable_DETR.util.misc_ttsim import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    inverse_sigmoid,
)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _get_clones(module, N):
    """Clone a module N times."""
    return SimNN.ModuleList([copy.deepcopy(module) for i in range(N)])


# ══════════════════════════════════════════════════════════════════════════════
# Loss helpers (sigmoid_focal_loss, dice_loss)
# ══════════════════════════════════════════════════════════════════════════════


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """
    Focal Loss used in RetinaNet for dense detection.

    TTSim/NumPy version of models/segmentation.py::sigmoid_focal_loss.

    Args:
        inputs: SimTensor or ndarray — raw logits, arbitrary shape
        targets: SimTensor or ndarray — binary classification labels (same shape)
        num_boxes: normalisation denominator
        alpha: weighting factor for positive vs negative
        gamma: focusing exponent

    Returns:
        float — scalar loss value
    """
    inp = (
        inputs.data
        if isinstance(inputs, SimTensor)
        else np.asarray(inputs, dtype=np.float32)
    )
    tgt = (
        targets.data
        if isinstance(targets, SimTensor)
        else np.asarray(targets, dtype=np.float32)
    )

    # sigmoid
    prob = 1.0 / (1.0 + np.exp(-inp.astype(np.float64)))

    # binary cross-entropy with logits (numerically stable)
    ce_loss = np.maximum(inp, 0) - inp * tgt + np.log1p(np.exp(-np.abs(inp)))

    p_t = prob * tgt + (1.0 - prob) * (1.0 - tgt)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * tgt + (1.0 - alpha) * (1.0 - tgt)
        loss = alpha_t * loss

    return float(loss.mean(axis=1).sum() / num_boxes)


def dice_loss(inputs, targets, num_boxes):
    """
    DICE loss (similar to generalised IoU for masks).

    TTSim/NumPy version of models/segmentation.py::dice_loss.

    Args:
        inputs: SimTensor or ndarray — raw logits
        targets: SimTensor or ndarray — binary mask targets
        num_boxes: normalisation denominator

    Returns:
        float — scalar loss value
    """
    inp = (
        inputs.data
        if isinstance(inputs, SimTensor)
        else np.asarray(inputs, dtype=np.float32)
    )
    tgt = (
        targets.data
        if isinstance(targets, SimTensor)
        else np.asarray(targets, dtype=np.float32)
    )

    # sigmoid then flatten from dim 1
    prob = 1.0 / (1.0 + np.exp(-inp.astype(np.float64)))
    prob = prob.reshape(prob.shape[0], -1)
    tgt = tgt.reshape(tgt.shape[0], -1)

    numerator = 2.0 * (prob * tgt).sum(axis=1)
    denominator = prob.sum(axis=-1) + tgt.sum(axis=-1)
    loss = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    return float(loss.sum() / num_boxes)


# ══════════════════════════════════════════════════════════════════════════════
# _InputProj  — Conv2d + GroupNorm projection (used by DeformableDETR)
# ══════════════════════════════════════════════════════════════════════════════


class _InputProj(SimNN.Module):
    """Single-level input projection: Conv2d followed by GroupNorm.

    Uses SimNN.GroupNorm (GroupNormalization) which maps correctly to the
    arch mapping — GroupNormalization must be present in wl2archmapping.yaml.
    """

    def __init__(self, name, in_channels, hidden_dim, kernel_size=1, stride=1, padding=0,
                 num_groups=32):
        super().__init__()
        self.name = name
        # F.Conv2d is a SimOpHandle → __setattr__ adds to _op_hndls
        self.conv = F.Conv2d(
            f"{name}.conv", in_channels, hidden_dim,
            kernel_size=kernel_size, stride=stride, padding=padding,
        )
        # GroupNorm is the correct normalization for Deformable DETR input projections
        self.gn = SimNN.GroupNorm(
            f"{name}.gn", num_groups=num_groups, num_channels=hidden_dim
        )
        super().link_op2module()

    def __call__(self, x):
        return self.gn(self.conv(x))


# ══════════════════════════════════════════════════════════════════════════════
# DeformableDETR
# ══════════════════════════════════════════════════════════════════════════════


class DeformableDETR(SimNN.Module):
    """Deformable DETR module for object detection (TTSim version)."""

    def __init__(self, name, cfg):
        """
        Polaris-compatible initializer — builds the full Deformable DETR model
        from a YAML-derived config dict.

        Args:
            name : string identifier for this module
            cfg  : dict loaded from polaris YAML (see config/ip_workloads.yaml)
        """
        super().__init__()
        self.name = name

        # ── Polaris-specific bookkeeping ─────────────────────────────────
        self.bs           = cfg.get('bs', 1)
        self.img_channels = cfg.get('img_channels', 3)
        self.img_height   = cfg.get('img_height', 640)
        self.img_width    = cfg.get('img_width', 640)

        # ── Model hyperparameters ────────────────────────────────────────
        num_classes        = cfg.get('num_classes', 91)
        num_queries        = cfg.get('num_queries', 300)
        num_feature_levels = cfg.get('num_feature_levels', 4)
        aux_loss           = cfg.get('aux_loss', False)
        with_box_refine    = cfg.get('with_box_refine', False)
        two_stage          = cfg.get('two_stage', False)

        self.num_queries        = num_queries
        self.num_feature_levels = num_feature_levels
        self.aux_loss           = aux_loss
        self.with_box_refine    = with_box_refine
        self.two_stage          = two_stage

        # ── Build backbone and transformer ───────────────────────────────
        args = SimpleNamespace(
            hidden_dim         = cfg.get('hidden_dim', 256),
            num_queries        = num_queries,
            num_feature_levels = num_feature_levels,
            nheads             = cfg.get('nheads', 8),
            enc_layers         = cfg.get('num_encoder_layers', 6),
            dec_layers         = cfg.get('num_decoder_layers', 6),
            dim_feedforward    = cfg.get('dim_feedforward', 1024),
            dropout            = cfg.get('dropout', 0.1),
            dec_n_points       = cfg.get('dec_n_points', 4),
            enc_n_points       = cfg.get('enc_n_points', 4),
            two_stage          = two_stage,
            backbone           = cfg.get('backbone', 'resnet50'),
            lr_backbone        = cfg.get('lr_backbone', 0),
            masks              = cfg.get('masks', False),
            dilation           = cfg.get('dilation', False),
            position_embedding = cfg.get('position_embedding', 'sine'),
        )

        backbone    = build_backbone(args)
        transformer = build_deformable_transformer(args)
        hidden_dim  = transformer.d_model

        # ── Query embeddings (param tensor) ──────────────────────────────
        # Assigned via self.X so __setattr__ registers it in _tensors.
        self.query_embed_weight: Optional[SimTensor] = None
        if not two_stage:
            self.query_embed_weight = F._from_shape(
                name + ".query_embed.weight",
                [num_queries, hidden_dim * 2],
                is_param=True,
            )

        # ── Input projection layers ───────────────────────────────────────
        num_backbone_outs = len(backbone.strides)
        in_ch             = backbone.num_channels[-1]   # fallback for extra levels
        input_proj_list   = []

        for i in range(num_backbone_outs):
            input_proj_list.append(
                _InputProj(
                    f"{name}.input_proj.{i}",
                    backbone.num_channels[i],
                    hidden_dim,
                    kernel_size=1,
                )
            )

        for i in range(num_feature_levels - num_backbone_outs):
            idx  = num_backbone_outs + i
            in_ch = hidden_dim if i > 0 else in_ch
            input_proj_list.append(
                _InputProj(
                    f"{name}.input_proj.{idx}",
                    in_ch,
                    hidden_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )

        # ── Register submodules in FORWARD-PASS ORDER ────────────────────
        # Python dict preserves insertion order; Module.get_ops() traverses
        # _submodules in that order when building the WorkloadGraph.  Every
        # op's input tensors must already be in the graph when add_op() is
        # called, so BACKBONE must come before INPUT_PROJ, etc.

        self.backbone    = backbone                           # (1) backbone first
        self.input_proj  = SimNN.ModuleList(input_proj_list) # (2) input projections
        self.transformer = transformer                        # (3) encoder + decoder
        self.class_embed = SimNN.Linear(                      # (4) classification head
            name + ".class_embed", hidden_dim, num_classes
        )
        self.bbox_embed  = MLP(                               # (5) box regression head
            name + ".bbox_embed", hidden_dim, hidden_dim, 4, 3
        )

        # Pre-register sigmoid so link_op2module() sets its module pointer.
        # __setattr__ adds F.Sigmoid (a SimOpHandle) to _op_hndls.
        self.output_sigmoid = F.Sigmoid(name + ".output_sigmoid")

        super().link_op2module()

    # ── Polaris interface ────────────────────────────────────────────────

    def set_batch_size(self, new_bs):
        """Update batch size (called by polaris batch-sweep)."""
        self.bs = new_bs

    def create_input_tensors(self):
        """Create the input SimTensor dict used by polaris."""
        img_tensor = F._from_shape(
            'img',
            [self.bs, self.img_channels, self.img_height, self.img_width],
            is_param=False,
            np_dtype=np.float32,
        )
        self.input_tensors = {'img': img_tensor}

    def get_forward_graph(self):
        """Return the polaris WorkloadGraph built from the forward pass."""
        return super()._get_forward_graph(self.input_tensors)

    def analytical_param_count(self):
        """Return analytical parameter count (stub)."""
        return 0

    def __call__(self):
        """
        Polaris entry-point: no-argument forward pass.
        Builds a NestedTensor from self.input_tensors and delegates to _forward.
        """
        img  = self.input_tensors['img']
        mask = np.zeros((self.bs, self.img_height, self.img_width), dtype=bool)
        return self._forward(NestedTensor(img, mask))

    def _forward(self, samples):
        """
        Core forward pass for Deformable DETR.

        Args:
            samples: NestedTensor (tensors + mask)

        Returns:
            Dict with pred_logits and pred_boxes.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        srcs  = []
        masks = []

        for l, feat in enumerate(features):
            src, mask_np = feat.decompose()
            srcs.append(self.input_proj[l](src))
            # Transformer expects SimTensor masks (it calls set_module on them)
            if mask_np is not None:
                mask_data = np.asarray(mask_np, dtype=np.float32)
            else:
                mask_data = np.zeros(
                    [src.shape[0], src.shape[-2], src.shape[-1]], dtype=np.float32
                )
            masks.append(SimTensor({
                'name': f'mask_{l}',
                'shape': [src.shape[0], src.shape[-2], src.shape[-1]],
                'data': mask_data,
                'dtype': np.float32,
            }))

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])

                m = samples.mask
                if m is not None:
                    raw = np.asarray(m, dtype=np.float32)
                    # crop / pad to feature-map spatial size
                    fh, fw = int(src.shape[-2]), int(src.shape[-1])
                    raw = raw[:, :fh, :fw] if raw.ndim == 3 else raw[:fh, :fw][None]
                else:
                    fh, fw = int(src.shape[-2]), int(src.shape[-1])
                    raw = np.zeros([src.shape[0], fh, fw], dtype=np.float32)

                masks.append(SimTensor({
                    'name': f'mask_proj_{l}',
                    'shape': [src.shape[0], int(src.shape[-2]), int(src.shape[-1])],
                    'data': raw,
                    'dtype': np.float32,
                }))
                pos.append(self.backbone.position_embedding(NestedTensor(src, None)))
                srcs.append(src)

        query_embeds = self.query_embed_weight if not self.two_stage else None

        hs, _, _, _, _ = self.transformer(srcs, masks, pos, query_embeds)

        # hs: [num_dec_layers, bs, num_queries, hidden_dim]  (stacked decoder outputs)
        # Apply heads on the full stacked output — polaris cares about op presence
        # and shape flow, not strict per-layer numerics.
        outputs_class = self.class_embed(hs)
        outputs_coord = self.output_sigmoid(self.bbox_embed(hs))

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

    def _set_aux_loss(self, outputs_class, outputs_coord):
        """Auxiliary loss outputs for each decoder layer except the last."""
        return [
            {"pred_logits": outputs_class[i], "pred_boxes": outputs_coord[i]}
            for i in range(len(outputs_class) - 1)
        ]


# ══════════════════════════════════════════════════════════════════════════════
# SetCriterion
# ══════════════════════════════════════════════════════════════════════════════


class SetCriterion(SimNN.Module):
    """
    Loss computation for Deformable DETR (TTSim version).

    Two-step process:
      1) Hungarian matching between predictions and ground truth
      2) Supervised loss for each matched pair (classification + box)

    Losses implemented:
      - loss_labels : sigmoid focal loss
      - loss_cardinality : absolute error in predicted non-empty boxes (logging only)
      - loss_boxes : L1 + GIoU loss on bounding boxes
      - loss_masks : focal + dice loss on segmentation masks
    """

    def __init__(
        self, name, num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    ):
        """
        Args:
            name: Module name
            num_classes: number of object categories (excluding no-object)
            matcher: HungarianMatcher instance
            weight_dict: loss name → relative weight
            losses: list of loss names to compute ('labels', 'boxes', 'cardinality', 'masks')
            focal_alpha: alpha for focal loss
        """
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        super().link_op2module()

    # ── index helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _get_src_permutation_idx(indices):
        """Permute predictions following matched indices.
        Returns (batch_idx, src_idx) arrays."""
        batch_idx = np.concatenate(
            [np.full(len(src), i, dtype=np.int64) for i, (src, _) in enumerate(indices)]
        )
        src_idx = np.concatenate([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        """Permute targets following matched indices."""
        batch_idx = np.concatenate(
            [np.full(len(tgt), i, dtype=np.int64) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = np.concatenate([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # ── individual losses ─────────────────────────────────────────────────

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (sigmoid focal loss).

        Returns dict with 'loss_ce' and optionally 'class_error'.
        """
        pred_logits = outputs["pred_logits"]
        logits_data = (
            pred_logits.data
            if isinstance(pred_logits, SimTensor)
            else np.asarray(pred_logits)
        )

        idx = self._get_src_permutation_idx(indices)

        # Concatenate matched target labels
        target_classes_o = np.concatenate(
            [np.asarray(t["labels"])[J] for t, (_, J) in zip(targets, indices)]
        )

        # Full target tensor filled with num_classes (background)
        target_classes = np.full(
            logits_data.shape[:2], self.num_classes, dtype=np.int64
        )
        target_classes[idx] = target_classes_o

        # One-hot encoding
        target_classes_onehot = np.zeros(
            (logits_data.shape[0], logits_data.shape[1], logits_data.shape[2] + 1),
            dtype=np.float32,
        )
        for b in range(target_classes.shape[0]):
            for q in range(target_classes.shape[1]):
                target_classes_onehot[b, q, target_classes[b, q]] = 1.0
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = (
            sigmoid_focal_loss(
                logits_data,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * logits_data.shape[1]
        )

        losses = {"loss_ce": loss_ce}

        if log:
            # class_error = 100 - accuracy(matched logits, matched labels)
            if len(target_classes_o) > 0:
                matched_logits = logits_data[idx]
                acc = accuracy(matched_logits, target_classes_o)
                losses["class_error"] = 100.0 - acc[0]
            else:
                losses["class_error"] = 0.0
        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Cardinality error — logging only, no gradient."""
        pred_logits = outputs["pred_logits"]
        logits_data = (
            pred_logits.data
            if isinstance(pred_logits, SimTensor)
            else np.asarray(pred_logits)
        )

        tgt_lengths = np.array([len(v["labels"]) for v in targets], dtype=np.float32)
        card_pred = (
            (logits_data.argmax(-1) != logits_data.shape[-1] - 1)
            .sum(axis=1)
            .astype(np.float32)
        )
        card_err = float(np.abs(card_pred - tgt_lengths).mean())
        return {"cardinality_error": card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 + GIoU loss on bounding boxes."""
        idx = self._get_src_permutation_idx(indices)

        pred_boxes = outputs["pred_boxes"]
        pred_data = (
            pred_boxes.data
            if isinstance(pred_boxes, SimTensor)
            else np.asarray(pred_boxes)
        )
        src_boxes = pred_data[idx]  # [num_matched, 4]

        target_boxes = np.concatenate(
            [np.asarray(t["boxes"])[i] for t, (_, i) in zip(targets, indices)], axis=0
        ).astype(np.float32)

        # L1 loss
        loss_bbox = float(np.abs(src_boxes - target_boxes).sum() / num_boxes)

        # GIoU loss
        src_xyxy = box_ops.box_cxcywh_to_xyxy(
            SimTensor(
                {
                    "name": "src_xyxy",
                    "shape": list(src_boxes.shape),
                    "data": src_boxes,
                    "dtype": np.float32,
                }
            )
        )
        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(
            SimTensor(
                {
                    "name": "tgt_xyxy",
                    "shape": list(target_boxes.shape),
                    "data": target_boxes,
                    "dtype": np.float32,
                }
            )
        )
        giou_matrix = box_ops.generalized_box_iou(src_xyxy, tgt_xyxy)
        giou_data = (
            giou_matrix.data
            if isinstance(giou_matrix, SimTensor)
            else np.asarray(giou_matrix)
        )
        giou_diag = np.diag(giou_data)
        loss_giou = float((1.0 - giou_diag).sum() / num_boxes)

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Focal + Dice loss on segmentation masks."""
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        pred_masks = outputs["pred_masks"]
        pred_data = (
            pred_masks.data
            if isinstance(pred_masks, SimTensor)
            else np.asarray(pred_masks)
        )

        # Build padded target masks
        tgt_mask_list = [np.asarray(t["masks"], dtype=np.float32) for t in targets]
        max_h = max(m.shape[-2] for m in tgt_mask_list)
        max_w = max(m.shape[-1] for m in tgt_mask_list)
        target_masks = np.zeros(
            (len(tgt_mask_list), tgt_mask_list[0].shape[0], max_h, max_w),
            dtype=np.float32,
        )
        for i, m in enumerate(tgt_mask_list):
            target_masks[i, :, : m.shape[-2], : m.shape[-1]] = m

        src_masks = pred_data[src_idx]  # [num_matched, 1, H, W] or [num_matched, H, W]

        # Interpolate to target size
        if src_masks.ndim == 3:
            src_masks = src_masks[:, None, :, :]  # add channel dim
        src_masks_st = SimTensor(
            {
                "name": "src_masks_interp",
                "shape": list(src_masks.shape),
                "data": src_masks,
                "dtype": np.float32,
            }
        )
        src_masks_interp = interpolate(
            src_masks_st, size=(max_h, max_w), mode="bilinear", align_corners=False
        )
        assert src_masks_interp.data is not None, "Mask interpolation produced no data"
        src_masks_flat = src_masks_interp.data[:, 0].reshape(
            src_masks_interp.data.shape[0], -1
        )

        target_masks_sel = target_masks[tgt_idx].reshape(
            target_masks[tgt_idx].shape[0], -1
        )

        loss_mask = sigmoid_focal_loss(src_masks_flat, target_masks_sel, num_boxes)
        loss_dice_val = dice_loss(src_masks_flat, target_masks_sel, num_boxes)

        return {"loss_mask": loss_mask, "loss_dice": loss_dice_val}

    # ── dispatch ──────────────────────────────────────────────────────────

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)  # type: ignore[operator]

    # ── forward ───────────────────────────────────────────────────────────

    def __call__(self, outputs, targets):
        """
        Compute all requested losses.

        Args:
            outputs: dict of predictions from DeformableDETR
            targets: list of target dicts (one per image)

        Returns:
            dict of loss name → scalar value
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Hungarian matching on last-layer outputs
        indices = self.matcher(outputs_without_aux, targets)

        # Normalisation: average number of target boxes across nodes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(float(num_boxes) / get_world_size(), 1.0)

        # Main losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # Auxiliary losses (one per intermediate decoder layer)
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    kwargs = {}
                    if loss == "labels":
                        kwargs["log"] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Encoder outputs (two-stage)
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = np.zeros_like(np.asarray(bt["labels"]))
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == "masks":
                    continue
                kwargs = {}
                if loss == "labels":
                    kwargs["log"] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + f"_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


# ══════════════════════════════════════════════════════════════════════════════
# PostProcess
# ══════════════════════════════════════════════════════════════════════════════


class PostProcess(SimNN.Module):
    """
    Converts model output into COCO-API format (TTSim version).

    For each image: top-100 detections with scores, labels, and absolute-coordinate boxes.
    """

    def __init__(self, name="postprocess"):
        super().__init__()
        self.name = name
        super().link_op2module()

    def __call__(self, outputs, target_sizes):
        """
        Args:
            outputs: dict with 'pred_logits' [bs, num_queries, num_classes]
                                'pred_boxes'  [bs, num_queries, 4]
            target_sizes: ndarray [bs, 2]  — (h, w) per image

        Returns:
            list of dicts with 'scores', 'labels', 'boxes' (numpy arrays)
        """
        out_logits = outputs["pred_logits"]
        out_bbox = outputs["pred_boxes"]
        logits_data = (
            out_logits.data
            if isinstance(out_logits, SimTensor)
            else np.asarray(out_logits)
        )
        bbox_data = (
            out_bbox.data if isinstance(out_bbox, SimTensor) else np.asarray(out_bbox)
        )
        sizes_data = (
            target_sizes
            if isinstance(target_sizes, np.ndarray)
            else np.asarray(target_sizes)
        )

        bs = logits_data.shape[0]
        num_classes = logits_data.shape[2]
        assert len(sizes_data) == bs
        assert sizes_data.shape[1] == 2

        # sigmoid probabilities
        prob = 1.0 / (1.0 + np.exp(-logits_data.astype(np.float64)))

        # top-100 across flattened (query × class)
        prob_flat = prob.reshape(bs, -1)  # [bs, num_queries * num_classes]

        results = []
        for b in range(bs):
            topk_indices = np.argsort(-prob_flat[b])[:100]
            topk_values = prob_flat[b][topk_indices]

            topk_boxes = topk_indices // num_classes  # query index
            labels = topk_indices % num_classes  # class index

            # Convert boxes cxcywh → xyxy
            boxes_cxcywh = bbox_data[b][topk_boxes]  # [100, 4]
            cx, cy, w, h = (
                boxes_cxcywh[:, 0],
                boxes_cxcywh[:, 1],
                boxes_cxcywh[:, 2],
                boxes_cxcywh[:, 3],
            )
            boxes_xyxy = np.stack(
                [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=1
            )

            # Scale from relative [0,1] to absolute [0, H/W]
            img_h, img_w = sizes_data[b, 0], sizes_data[b, 1]
            scale = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
            boxes_xyxy = boxes_xyxy * scale[None, :]

            results.append(
                {
                    "scores": topk_values.astype(np.float32),
                    "labels": labels.astype(np.int64),
                    "boxes": boxes_xyxy.astype(np.float32),
                }
            )

        return results


# ══════════════════════════════════════════════════════════════════════════════
# PostProcessSegm
# ══════════════════════════════════════════════════════════════════════════════


class PostProcessSegm(SimNN.Module):
    """
    Segmentation post-processing (TTSim version).

    Converts raw pred_masks → boolean mask per detection, interpolated to
    original image size.
    """

    def __init__(self, name="postprocess_segm", threshold=0.5):
        super().__init__()
        self.name = name
        self.threshold = threshold
        super().link_op2module()

    def __call__(self, results, outputs, orig_target_sizes, max_target_sizes):
        """
        Args:
            results: list of dicts (from PostProcess) — will be mutated with 'masks' key
            outputs: dict with 'pred_masks' [bs, num_queries, 1, H, W]
            orig_target_sizes: ndarray [bs, 2]
            max_target_sizes: ndarray [bs, 2]

        Returns:
            list of result dicts with 'masks' added
        """
        orig_sizes = (
            orig_target_sizes
            if isinstance(orig_target_sizes, np.ndarray)
            else np.asarray(orig_target_sizes)
        )
        max_sizes = (
            max_target_sizes
            if isinstance(max_target_sizes, np.ndarray)
            else np.asarray(max_target_sizes)
        )

        assert len(orig_sizes) == len(max_sizes)
        max_h = int(max_sizes[:, 0].max())
        max_w = int(max_sizes[:, 1].max())

        pred_masks = outputs["pred_masks"]
        masks_data = (
            pred_masks.data
            if isinstance(pred_masks, SimTensor)
            else np.asarray(pred_masks)
        )

        # Squeeze channel dim if present: [bs, Q, 1, H, W] → [bs, Q, H, W]
        if masks_data.ndim == 5:
            masks_data = masks_data.squeeze(2)

        # Interpolate each image's masks to (max_h, max_w) using bilinear
        bs = masks_data.shape[0]
        for i in range(bs):
            mask_i = masks_data[i]  # [Q, H, W]
            mask_st = SimTensor(
                {
                    "name": f"segm_interp_{i}",
                    "shape": [mask_i.shape[0], 1, mask_i.shape[1], mask_i.shape[2]],
                    "data": mask_i[:, None, :, :].astype(np.float32),
                    "dtype": np.float32,
                }
            )
            interped = interpolate(
                mask_st, size=(max_h, max_w), mode="bilinear", align_corners=False
            )
            interp_data = interped.data[:, 0, :, :]  # [Q, max_h, max_w]

            # Sigmoid + threshold
            sig = 1.0 / (1.0 + np.exp(-interp_data.astype(np.float64)))
            binary = (sig > self.threshold).astype(np.uint8)

            # Crop to actual image size
            img_h, img_w = int(max_sizes[i, 0]), int(max_sizes[i, 1])
            cropped = binary[:, :img_h, :img_w]  # [Q, img_h, img_w]

            # Resize to original size via nearest
            orig_h, orig_w = int(orig_sizes[i, 0]), int(orig_sizes[i, 1])
            cropped_st = SimTensor(
                {
                    "name": f"segm_resize_{i}",
                    "shape": [cropped.shape[0], 1, img_h, img_w],
                    "data": cropped[:, None, :, :].astype(np.float32),
                    "dtype": np.float32,
                }
            )
            resized = interpolate(cropped_st, size=(orig_h, orig_w), mode="nearest")
            results[i]["masks"] = resized.data.astype(np.uint8)

        return results


# ══════════════════════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════════════════════


class MLP(SimNN.Module):
    """Multi-layer perceptron (Feed-Forward Network)."""

    def __init__(self, name, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.name = name
        self.num_layers = num_layers

        h = [hidden_dim] * (num_layers - 1)
        layer_dims = [input_dim] + h + [output_dim]

        layers_list: List[SimNN.Linear] = []
        for i in range(num_layers):
            layer = SimNN.Linear(f"{name}.layers.{i}", layer_dims[i], layer_dims[i + 1])
            layers_list.append(layer)

        # self.layers assignment via __setattr__ → ModuleList entries go to _submodules
        self.layers: SimNN.ModuleList = SimNN.ModuleList(layers_list)

        # Pre-register relu ops as named attributes so __setattr__ stores them in
        # _op_hndls and link_op2module() calls set_module() on each of them.
        for i in range(num_layers - 1):
            setattr(self, f'relu_{i}', F.Relu(f"{name}.relu.{i}"))

        super().link_op2module()

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = getattr(self, f'relu_{i}')(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# build()
# ══════════════════════════════════════════════════════════════════════════════


def build(args):
    """
    Build Deformable DETR model with criterion and postprocessors.

    Mirrors PyTorch build() from deformable_detr.py.

    Excluded: PostProcessPanoptic (requires panopticapi).

    Args:
        args: Namespace with model configuration attributes

    Returns:
        Tuple of (model, criterion, postprocessors)
    """
    num_classes = 20 if args.dataset_file != "coco" else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250

    # Build via the polaris-compatible (name, cfg) interface
    _cfg = vars(args) if not isinstance(args, dict) else args
    _cfg = dict(_cfg)   # make a plain copy
    _cfg.update({
        'num_classes':        num_classes,
        'hidden_dim':         getattr(args, 'hidden_dim', 256),
        'num_queries':        args.num_queries,
        'num_feature_levels': args.num_feature_levels,
        'aux_loss':           args.aux_loss,
        'with_box_refine':    args.with_box_refine,
        'two_stage':          args.two_stage,
    })
    model: Union[DeformableDETR, DETRsegm] = DeformableDETR(
        name="deformable_detr",
        cfg=_cfg,
    )

    if args.masks:
        model = DETRsegm(
            name="deformable_detr_segm",
            detr=model,
            hidden_dim=_cfg.get('hidden_dim', 256),
            nheads=args.nheads,
            freeze_detr=(args.frozen_weights is not None),
        )

    matcher = build_matcher(args)

    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]

    criterion = SetCriterion(
        name="criterion",
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        focal_alpha=args.focal_alpha,
    )

    postprocessors: Dict[str, Union[PostProcess, PostProcessSegm]] = {
        "bbox": PostProcess()
    }
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()
        # PostProcessPanoptic excluded — requires panopticapi (PIL, id2rgb/rgb2id)

    return model, criterion, postprocessors
