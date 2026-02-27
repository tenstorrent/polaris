# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# SPDX-FileCopyrightText: (C) 2020 SenseTime. All Rights Reserved
# SPDX-License-Identifier: Apache-2.0
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np

# TTSim imports for polaris integration
import ttsim.front.functional.op as TF
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

from workloads.Deformable_DETR.reference import box_ops
from workloads.Deformable_DETR.reference.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    inverse_sigmoid,
)

# Use absolute imports for polaris compatibility
from workloads.Deformable_DETR.reference.backbone import build_backbone
from workloads.Deformable_DETR.reference.matcher import build_matcher
from workloads.Deformable_DETR.reference.segmentation import (
    DETRsegm,
    PostProcessPanoptic,
    PostProcessSegm,
    dice_loss,
    sigmoid_focal_loss,
)
from workloads.Deformable_DETR.reference.deformable_transformer import (
    build_deforamble_transformer,
)
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """This is the Deformable DETR module that performs object detection"""

    def __init__(
        self,
        name=None,
        cfg=None,
        backbone=None,
        transformer=None,
        num_classes=None,
        num_queries=None,
        num_feature_levels=None,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
    ):
        """Initializes the model.
        Parameters:
            name: model name (for polaris integration)
            cfg: dict with configuration for polaris integration (optional)
            backbone: torch module of the backbone to be used. See backbone.py (optional if cfg provided)
            transformer: torch module of the transformer architecture. See transformer.py (optional if cfg provided)
            num_classes: number of object classes (optional if cfg provided)
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries. (optional if cfg provided)
            num_feature_levels: number of feature levels (optional if cfg provided)
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        # Handle cfg-based initialization for polaris
        if cfg is not None and isinstance(cfg, dict):
            if name:
                cfg["name"] = name
            self._init_from_config(cfg)
            return
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (transformer.decoder.num_layers + 1)
            if two_stage
            else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def _init_from_config(self, cfg):
        """Initialize model from config dict for polaris integration"""
        # This is a simplified initialization for polaris
        # Extract config parameters
        self.name = cfg.get("name", "deformable_detr")
        self.bs = cfg.get("bs", 1)
        self.img_height = cfg.get("img_height", 800)
        self.img_width = cfg.get("img_width", 800)
        self.img_channels = cfg.get("img_channels", 3)
        self.num_classes = cfg.get("num_classes", 80)
        self.num_queries = cfg.get("num_queries", 300)
        self.hidden_dim = cfg.get("hidden_dim", 256)
        self.num_feature_levels = cfg.get("num_feature_levels", 4)

        # Initialize SimNN-compatible attributes
        self._tensors = {}
        self._op_hndls = {}
        self._submodules = {}

        # Build a simplified backbone using ttsim operations
        self._build_ttsim_backbone()

        # For polaris, we'll use a simplified architecture
        self.aux_loss = False
        self.with_box_refine = False
        self.two_stage = False

    def _build_ttsim_backbone(self):
        """Build simplified backbone using ttsim operations for polaris"""
        # Backbone stages
        self.conv1 = TF.Conv2d(
            f"{self.name}.backbone.conv1",
            self.img_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self._op_hndls[f"{self.name}.backbone.conv1"] = self.conv1

        self.bn1 = TF.BatchNorm2d(f"{self.name}.backbone.bn1", 64)
        self._op_hndls[f"{self.name}.backbone.bn1"] = self.bn1

        self.relu1 = TF.Relu(f"{self.name}.backbone.relu1")
        self._op_hndls[f"{self.name}.backbone.relu1"] = self.relu1

        self.maxpool1 = TF.MaxPool2d(
            f"{self.name}.backbone.maxpool1", kernel_size=3, stride=2, padding=1
        )
        self._op_hndls[f"{self.name}.backbone.maxpool1"] = self.maxpool1

        self.conv2 = TF.Conv2d(
            f"{self.name}.backbone.conv2", 64, 128, kernel_size=3, stride=2, padding=1
        )
        self._op_hndls[f"{self.name}.backbone.conv2"] = self.conv2

        self.bn2 = TF.BatchNorm2d(f"{self.name}.backbone.bn2", 128)
        self._op_hndls[f"{self.name}.backbone.bn2"] = self.bn2

        self.relu2 = TF.Relu(f"{self.name}.backbone.relu2")
        self._op_hndls[f"{self.name}.backbone.relu2"] = self.relu2

        self.conv3 = TF.Conv2d(
            f"{self.name}.backbone.conv3", 128, 256, kernel_size=3, stride=2, padding=1
        )
        self._op_hndls[f"{self.name}.backbone.conv3"] = self.conv3

        self.bn3 = TF.BatchNorm2d(f"{self.name}.backbone.bn3", 256)
        self._op_hndls[f"{self.name}.backbone.bn3"] = self.bn3

        self.relu3 = TF.Relu(f"{self.name}.backbone.relu3")
        self._op_hndls[f"{self.name}.backbone.relu3"] = self.relu3

        self.conv4 = TF.Conv2d(
            f"{self.name}.backbone.conv4", 256, 512, kernel_size=3, stride=2, padding=1
        )
        self._op_hndls[f"{self.name}.backbone.conv4"] = self.conv4

        self.bn4 = TF.BatchNorm2d(f"{self.name}.backbone.bn4", 512)
        self._op_hndls[f"{self.name}.backbone.bn4"] = self.bn4

        self.relu4 = TF.Relu(f"{self.name}.backbone.relu4")
        self._op_hndls[f"{self.name}.backbone.relu4"] = self.relu4

        # Input projection layers
        self.input_proj1 = TF.Conv2d(
            f"{self.name}.input_proj1", 128, self.hidden_dim, kernel_size=1
        )
        self._op_hndls[f"{self.name}.input_proj1"] = self.input_proj1

        self.input_proj2 = TF.Conv2d(
            f"{self.name}.input_proj2", 256, self.hidden_dim, kernel_size=1
        )
        self._op_hndls[f"{self.name}.input_proj2"] = self.input_proj2

        self.input_proj3 = TF.Conv2d(
            f"{self.name}.input_proj3", 512, self.hidden_dim, kernel_size=1
        )
        self._op_hndls[f"{self.name}.input_proj3"] = self.input_proj3

        # Simplified encoder
        self.encoder_attn = TF.Conv2d(
            f"{self.name}.encoder.attn", self.hidden_dim, self.hidden_dim, kernel_size=1
        )
        self._op_hndls[f"{self.name}.encoder.attn"] = self.encoder_attn

        self.encoder_ffn = TF.Conv2d(
            f"{self.name}.encoder.ffn", self.hidden_dim, self.hidden_dim, kernel_size=1
        )
        self._op_hndls[f"{self.name}.encoder.ffn"] = self.encoder_ffn

        self.encoder_norm = TF.BatchNorm2d(f"{self.name}.encoder.norm", self.hidden_dim)
        self._op_hndls[f"{self.name}.encoder.norm"] = self.encoder_norm

        # Detection heads
        self.cls_head_relu = TF.Relu(f"{self.name}.cls_head.relu")
        self._op_hndls[f"{self.name}.cls_head.relu"] = self.cls_head_relu

        self.bbox_head_relu = TF.Relu(f"{self.name}.bbox_head.relu")
        self._op_hndls[f"{self.name}.bbox_head.relu"] = self.bbox_head_relu

        # Link all operations to this "module"
        for op_name, op in self._op_hndls.items():
            op.set_module(self)

    def forward(self, samples: NestedTensor = None, x_in: SimTensor = None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        # Handle polaris/ttsim mode
        if x_in is not None:
            return self._forward_ttsim(x_in)

        # Original PyTorch mode
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out

    def _forward_ttsim(self, x):
        """Forward pass using ttsim operations for polaris integration"""
        batch = x.shape[0]

        # Backbone feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        feat2 = self.conv2(x)
        feat2 = self.bn2(feat2)
        feat2 = self.relu2(feat2)

        feat3 = self.conv3(feat2)
        feat3 = self.bn3(feat3)
        feat3 = self.relu3(feat3)

        feat4 = self.conv4(feat3)
        feat4 = self.bn4(feat4)
        feat4 = self.relu4(feat4)

        # Multi-scale feature projection
        proj_feat2 = self.input_proj1(feat2)
        proj_feat3 = self.input_proj2(feat3)
        proj_feat4 = self.input_proj3(feat4)

        # Simplified encoder
        enc_feat = proj_feat4
        attn_out = self.encoder_attn(enc_feat)
        enc_feat = enc_feat + attn_out
        ffn_out = self.encoder_ffn(enc_feat)
        enc_feat = enc_feat + ffn_out
        enc_feat = self.encoder_norm(enc_feat)

        # Flatten encoder output
        enc_h, enc_w = enc_feat.shape[2], enc_feat.shape[3]
        enc_feat_flat = enc_feat.reshape(batch, self.hidden_dim * enc_h * enc_w)

        # Query embeddings
        query_embed = TF._from_shape(
            f"{self.name}.query_embed",
            [batch, self.num_queries * self.hidden_dim],
            is_param=True,
            np_dtype=np.float32,
        )

        # Project encoder features to query dimension
        proj_enc_dim = self.num_queries * self.hidden_dim
        enc_proj = T.matmul(
            enc_feat_flat,
            TF._from_shape(
                f"{self.name}.enc_to_query_proj",
                [self.hidden_dim * enc_h * enc_w, proj_enc_dim],
                is_param=True,
                np_dtype=np.float32,
            ),
        )

        # Combine queries with encoder features
        decoder_feat = query_embed + enc_proj

        # Classification head
        cls_feat = T.matmul(
            decoder_feat,
            TF._from_shape(
                f"{self.name}.cls_head.weight",
                [
                    self.num_queries * self.hidden_dim,
                    self.num_queries * self.num_classes,
                ],
                is_param=True,
                np_dtype=np.float32,
            ),
        )
        cls_feat = self.cls_head_relu(cls_feat)
        cls_output = cls_feat.reshape(batch, self.num_queries, self.num_classes)

        # Bounding box head
        bbox_feat = T.matmul(
            decoder_feat,
            TF._from_shape(
                f"{self.name}.bbox_head.weight",
                [self.num_queries * self.hidden_dim, self.num_queries * 4],
                is_param=True,
                np_dtype=np.float32,
            ),
        )
        bbox_feat = self.bbox_head_relu(bbox_feat)
        bbox_output = bbox_feat.reshape(batch, self.num_queries, 4)

        # Concatenate outputs
        output = T.cat([cls_output, bbox_output], dim=2)
        final_output = output.reshape(batch, self.num_queries * (self.num_classes + 4))

        return final_output

    # ===== Polaris Integration Methods =====
    def set_batch_size(self, new_bs):
        """Set batch size for polaris"""
        self.bs = new_bs

    def create_input_tensors(self):
        """Create input tensors for polaris integration"""
        self.input_tensors = {
            "x_in": TF._from_shape(
                "x_in",
                [self.bs, self.img_channels, self.img_height, self.img_width],
                is_param=False,
                np_dtype=np.float32,
            ),
        }
        return

    def get_forward_graph(self):
        """Get forward computation graph for polaris integration"""
        # Import WorkloadGraph here to avoid circular imports
        from ttsim.graph import WorkloadGraph

        # Get tensors
        ttbl = {}
        for tname, t in self.input_tensors.items():
            if isinstance(t, SimTensor):
                ttbl[t.name] = t

        # Collect all tensors from operations
        self._collect_tensors(ttbl)

        # Get ops
        otbl = {}
        self._collect_ops(otbl)

        # Construct graph
        gg = WorkloadGraph(self.name)

        # Add tensors to graph
        for _, tensor in ttbl.items():
            gg.add_tensor(tensor)

        # Add ops to graph
        for _, op in otbl.items():
            gg.add_op(op)

        # Construct graph
        gg.construct_graph()

        return gg

    def _collect_tensors(self, tbl):
        """Helper to collect all tensors"""
        # Collect tensors from _tensors dict
        for k, v in self._tensors.items():
            tbl[v.name] = v
        # Collect tensors from op handles
        for k, v in self._op_hndls.items():
            # Collect parameters
            if hasattr(v, "params") and len(v.params) > 0:
                for _, ptensor in v.params:
                    tbl[ptensor.name] = ptensor
            # Collect implicit inputs
            if hasattr(v, "implicit_inputs") and len(v.implicit_inputs) > 0:
                for itensor in v.implicit_inputs:
                    tbl[itensor.name] = itensor

    def _collect_ops(self, tbl):
        """Helper to collect all ops"""
        # Collect all registered operation handles
        for k, v in self._op_hndls.items():
            if hasattr(v, "sim_op") and v.sim_op is not None:
                tbl[k] = v.sim_op

    def analytical_param_count(self):
        """Return parameter count for polaris integration"""
        return 0

    def __call__(self, x=None):
        """Call method for polaris integration"""
        if x is None:
            # Use input_tensors if available
            if hasattr(self, "input_tensors") and "x_in" in self.input_tensors:
                x = self.input_tensors["x_in"]
            else:
                raise ValueError("No input provided and input_tensors not created")

        return self.forward(x_in=x)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs["log"] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + f"_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), 100, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        return results


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != "coco" else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(
        num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha
    )
    criterion.to(device)
    postprocessors = {"bbox": PostProcess()}
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85
            )

    return model, criterion, postprocessors
