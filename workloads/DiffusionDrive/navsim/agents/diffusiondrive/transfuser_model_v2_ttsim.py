#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSIM conversion of transfuser_model_v2.py
Converted from PyTorch to TTSIM operations while preserving logic, inputs, and outputs.
# (Torch reference code: reference\torch_code\transfuser_model_v2.py)
"""


from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from loguru import logger
import numpy as np
import copy
import os, sys
import json

# Add polaris root to sys.path so workloads.DiffusionDrive... imports resolve
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

# TTSIM imports
import ttsim.front.functional.op as F
from ttsim.front.functional.sim_nn import Module as SimNN_Module
from ttsim.front.functional.sim_nn import ModuleList as SimNN_ModuleList

if TYPE_CHECKING:
    from ttsim.ops.tensor import SimTensor

# Keep the config and other non-torch imports
from workloads.DiffusionDrive.navsim.agents.diffusiondrive.transfuser_config import (
    TransfuserConfig,
)
from workloads.DiffusionDrive.navsim.agents.diffusiondrive.transfuser_backbone_ttsim import (
    TransfuserBackbone,
)
from workloads.DiffusionDrive.navsim.agents.diffusiondrive.modules.blocks import (
    GridSampleCrossBEVAttention_TTSIM,
)

from workloads.DiffusionDrive.navsim.agents.diffusiondrive.modules.conditional_unet1d_ttsim import (
    SinusoidalPosEmb_TTSIM,
)

def load_plan_anchor(base_path: str) -> np.ndarray:
    """
    Load plan_anchor from JSON.

    If base_path has no extension, appends `.json`.
    Expects JSON with a top-level dict containing a 'clusters' list, where
    each cluster has a 'waypoints' list of [x, y] pairs.
    """
    # If caller already passed a full path with extension, use it
    if os.path.splitext(base_path)[1]:
        path = base_path
    else:
        path = base_path + ".json"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find plan_anchor at '{path}'")

    # Support both .npy (raw array) and .json (clusters dict) formats
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)

    with open(path, "r") as f:
        data = json.load(f)

    clusters = data["clusters"]                    # list of dicts
    waypoints = [c["waypoints"] for c in clusters]  # list[list[[x, y], ...]]

    return np.array(waypoints, dtype=np.float32)


# Enum constants to avoid importing modules with fcntl dependencies (Windows compatibility)
# From navsim.agents.diffusiondrive.transfuser_features.BoundingBox2DIndex
class BoundingBox2DIndex:
    """Local constants for BoundingBox2D indexing (avoids fcntl import on Windows)."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4
    POINT = slice(0, 2)  # X, Y
    HEADING = 2

    @staticmethod
    def size():
        return 5  # X, Y, HEADING, LENGTH, WIDTH


# From navsim.common.enums.StateSE2Index
class StateSE2Index:
    """Local constants for StateSE2 indexing (avoids fcntl import on Windows)."""

    HEADING = 2


class V2TransfuserModel(SimNN_Module):
    """TTSIM module for Transfuser."""

    @staticmethod
    def _build_config_from_dict(cfg: dict) -> TransfuserConfig:
        """Build a TransfuserConfig from a polaris YAML cfg dict."""
        config = TransfuserConfig()
        # Map YAML keys to TransfuserConfig attributes
        if "image_architecture" in cfg:
            config.image_architecture = cfg["image_architecture"]
        if "lidar_architecture" in cfg:
            config.lidar_architecture = cfg["lidar_architecture"]
        if "camera_width" in cfg:
            config.camera_width = int(cfg["camera_width"])
        if "camera_height" in cfg:
            config.camera_height = int(cfg["camera_height"])
        if "img_width" in cfg:
            config.camera_width = int(cfg["img_width"])
        if "img_height" in cfg:
            config.camera_height = int(cfg["img_height"])
        if "lidar_resolution_width" in cfg:
            config.lidar_resolution_width = int(cfg["lidar_resolution_width"])
        if "lidar_resolution_height" in cfg:
            config.lidar_resolution_height = int(cfg["lidar_resolution_height"])
        if "lidar_w" in cfg:
            config.lidar_resolution_width = int(cfg["lidar_w"])
        if "lidar_h" in cfg:
            config.lidar_resolution_height = int(cfg["lidar_h"])
        if "tf_d_model" in cfg:
            config.tf_d_model = int(cfg["tf_d_model"])
        if "tf_d_ffn" in cfg:
            config.tf_d_ffn = int(cfg["tf_d_ffn"])
        if "tf_num_layers" in cfg:
            config.tf_num_layers = int(cfg["tf_num_layers"])
        if "tf_num_head" in cfg:
            config.tf_num_head = int(cfg["tf_num_head"])
        if "num_bounding_boxes" in cfg:
            config.num_bounding_boxes = int(cfg["num_bounding_boxes"])
        if "bev_features_channels" in cfg:
            config.bev_features_channels = int(cfg["bev_features_channels"])
        if "plan_anchor_path" in cfg:
            config.plan_anchor_path = cfg["plan_anchor_path"]
        else:
            # Default: look for the .json relative to this file (in navsim/ directory)
            _this_dir = os.path.dirname(os.path.abspath(__file__))
            _default = os.path.join(_this_dir, "..", "..", "kmeans_navsim_traj_20.json")
            if os.path.exists(_default):
                config.plan_anchor_path = os.path.normpath(_default)
        if "bkb_path" in cfg:
            config.bkb_path = cfg["bkb_path"]
        # Recompute derived anchor sizes
        config.img_vert_anchors = config.camera_height // 32
        config.img_horz_anchors = config.camera_width // 32
        config.lidar_vert_anchors = config.lidar_resolution_height // 32
        config.lidar_horz_anchors = config.lidar_resolution_width // 32
        config.bev_pixel_width = config.lidar_resolution_width
        config.bev_pixel_height = config.lidar_resolution_height // 2
        return config

    def __init__(self, config_or_name, cfg=None):
        """
        Initializes TransFuser TTSIM module.

        Two calling conventions:
          1) Direct:  V2TransfuserModel(config: TransfuserConfig)
          2) Polaris: V2TransfuserModel(name: str, cfg: dict)
        """

        super().__init__()

        # Detect calling convention
        if isinstance(config_or_name, str):
            # Polaris pattern: (name, cfg_dict)
            self.name = config_or_name
            self._polaris_cfg = cfg or {}
            config = self._build_config_from_dict(self._polaris_cfg)
            self._bs = self._polaris_cfg.get("bs", 1)
        else:
            # Direct pattern: (config: TransfuserConfig)
            config = config_or_name
            self.name = "V2TransfuserModel"
            self._polaris_cfg = None
            self._bs = 1

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone("backbone", config)

        # Embedding layers
        num_keyval = config.lidar_vert_anchors * config.lidar_horz_anchors + 1
        self._keyval_embedding = F.Embedding(
            "keyval_embedding", num_keyval, config.tf_d_model
        )
        # Create indices tensor for keyval embedding lookup
        self._keyval_indices = F._from_data(
            "keyval_indices", np.arange(num_keyval, dtype=np.int64)
        )
        self._keyval_indices.is_param = False
        self._keyval_indices.set_module(self)

        num_query = sum(self._query_splits)
        self._query_embedding = F.Embedding(
            "query_embedding", num_query, config.tf_d_model
        )
        # Create indices tensor for query embedding lookup
        self._query_indices = F._from_data(
            "query_indices", np.arange(num_query, dtype=np.int64)
        )
        self._query_indices.is_param = False
        self._query_indices.set_module(self)

        # BEV downscale convolution
        self._bev_downscale = F.Conv2d(
            "bev_downscale", 512, config.tf_d_model, kernel_size=1
        )
        self._bev_downscale_bias = F.Bias(
            "bev_downscale_bias", [1, config.tf_d_model, 1, 1]
        )

        # Status encoding linear layer
        self._status_encoding = F.Linear(
            "status_encoding", 4 + 2 + 2, config.tf_d_model, module=self
        )
        self._status_encoding_bias = F.Bias("status_encoding_bias", [config.tf_d_model])

        # BEV semantic head - using separate layers instead of Sequential
        self._bev_semantic_conv1 = F.Conv2d(
            "bev_semantic_conv1",
            config.bev_features_channels,
            config.bev_features_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._bev_semantic_conv1_bias = F.Bias(
            "bev_semantic_conv1_bias",
            [1, config.bev_features_channels, 1, 1],
        )
        self._bev_semantic_relu = F.Relu("bev_semantic_relu")
        self._bev_semantic_conv2 = F.Conv2d(
            "bev_semantic_conv2",
            config.bev_features_channels,
            config.num_bev_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self._bev_semantic_conv2_bias = F.Bias(
            "bev_semantic_conv2_bias",
            [1, config.num_bev_classes, 1, 1],
        )
        # Upsample
        upsample_size = (
            config.lidar_resolution_height // 2,
            config.lidar_resolution_width,
        )
        # Input spatial dims = backbone's bev_feature_upscale output
        input_h = config.lidar_resolution_height // config.bev_down_sample_factor
        input_w = config.lidar_resolution_width // config.bev_down_sample_factor
        scale_h = upsample_size[0] / input_h
        scale_w = upsample_size[1] / input_w
        self._bev_semantic_upsample = F.Resize(
            "bev_semantic_upsample",
            scale_factor=[scale_h, scale_w],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
        )

        # Transformer decoder layer components
        _tf_decoder_layer_list = []
        for i in range(config.tf_num_layers):
            layer = TransformerDecoderLayer_TTSIM(
                f"tf_decoder_layer_{i}",
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
            )
            _tf_decoder_layer_list.append(layer)
        self._tf_decoder_layers = SimNN_ModuleList(_tf_decoder_layer_list)

        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )

        # BEV projection: Linear(320→256), ReLU, LayerNorm(256)
        # Input is concat of downscaled bev (256) + backbone upscale (64) = 320 channels
        self.bev_proj_linear1 = F.Linear("bev_proj_linear1", 320, 256, module=self)
        self.bev_proj_bias1 = F.Bias("bev_proj_bias1", [256])
        self.bev_proj_relu = F.Relu("bev_proj_relu")
        self.bev_proj_ln = F.LayerNorm("bev_proj_ln", 256)

        # Pre-allocated inline ops used in __call__
        self.keyval_concat = F.ConcatX("keyval_concat", axis=1)
        self.keyval_add_emb = F.Add("keyval_add_emb")
        self.concat_cross_bev_upsample = F.Resize(
            "concat_cross_bev_upsample",
            scale_factor=[1.0, 1.0],  # placeholder — overwritten at call time
            mode="linear",
        )
        self.cross_bev_concat = F.ConcatX("cross_bev_concat", axis=1)

        super().link_op2module()

    # ── Polaris interface methods ────────────────────────────────────────

    def set_batch_size(self, new_bs):
        self._bs = new_bs

    def create_input_tensors(self):
        """Create placeholder input tensors for polaris graph tracing."""
        cfg = self._config
        # camera_feature: [bs, 3, camera_height, camera_width]
        cam_h = cfg.camera_height
        cam_w = cfg.camera_width
        # lidar_feature channels must match backbone encoder in_channels:
        #   use_ground_plane -> 2*lidar_seq_len, else lidar_seq_len
        if cfg.use_ground_plane:
            lidar_ch = 2 * cfg.lidar_seq_len
        else:
            lidar_ch = cfg.lidar_seq_len  # default 1
        lidar_h = cfg.lidar_resolution_height
        lidar_w = cfg.lidar_resolution_width
        # status_feature: [bs, 8] (4 + 2 + 2)

        self.input_tensors = {
            "camera_feature": F._from_shape(
                "camera_feature",
                [self._bs, 3, cam_h, cam_w],
                is_param=False,
                np_dtype=np.float32,
            ),
            "lidar_feature": F._from_shape(
                "lidar_feature",
                [self._bs, lidar_ch, lidar_h, lidar_w],
                is_param=False,
                np_dtype=np.float32,
            ),
            "status_feature": F._from_shape(
                "status_feature",
                [self._bs, 8],
                is_param=False,
                np_dtype=np.float32,
            ),
        }
        for _, t in self.input_tensors.items():
            t.is_param = False
            t.set_module(self)

    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)

    def analytical_param_count(self):
        return 0

    # ── Forward pass ─────────────────────────────────────────────────────

    def __call__(
        self,
        features: Optional[Dict[str, "SimTensor"]] = None,
        targets: Optional[Dict[str, "SimTensor"]] = None,
    ) -> Dict[str, "SimTensor"]:
        """TTSIM module forward pass."""
        self._debug = {}  # Store intermediates for debugging

        # Polaris calls __call__() with no args; tests pass features dict
        if features is None:
            features = self.input_tensors

        camera_feature = features["camera_feature"]
        lidar_feature = features["lidar_feature"]
        status_feature = features["status_feature"]

        assert status_feature.shape is not None
        batch_size = status_feature.shape[0]

        # Backbone forward
        bev_feature_upscale, bev_feature, _ = self._backbone(
            camera_feature, lidar_feature
        )
        self._debug["bev_feature_raw"] = (
            bev_feature.data.copy() if bev_feature.data is not None else None
        )
        self._debug["bev_feature_upscale"] = (
            bev_feature_upscale.data.copy()
            if bev_feature_upscale.data is not None
            else None
        )
        cross_bev_feature = bev_feature_upscale
        assert bev_feature_upscale.shape is not None
        assert bev_feature.shape is not None
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]

        # BEV downscale
        bev_feature = self._bev_downscale(bev_feature)
        bev_feature = self._bev_downscale_bias(bev_feature)
        bev_feature.link_module = self
        bev_feature = bev_feature.flatten(-2, -1)
        bev_feature.link_module = self
        bev_feature = bev_feature.permute([0, 2, 1])
        self._debug["bev_feature_flat"] = (
            bev_feature.data.copy() if bev_feature.data is not None else None
        )

        # Status encoding
        status_encoding = self._status_encoding(status_feature)
        status_encoding = self._status_encoding_bias(status_encoding)
        status_encoding.link_module = self
        self._debug["status_encoding"] = (
            status_encoding.data.copy() if status_encoding.data is not None else None
        )

        # Create keyval with concatenation
        # keyval = concatenate([bev_feature, status_encoding[:, None]], dim=1)
        status_encoding_expanded = status_encoding.unsqueeze(1)
        keyval = self.keyval_concat(bev_feature, status_encoding_expanded)

        # Add keyval embedding
        # Use pre-created indices tensor from __init__
        keyval_emb = self._keyval_embedding(self._keyval_indices)
        keyval_emb.link_module = self
        keyval_emb_expanded = keyval_emb.unsqueeze(0)  # Add batch dimension
        keyval = self.keyval_add_emb(keyval, keyval_emb_expanded)
        self._debug["keyval"] = keyval.data.copy() if keyval.data is not None else None

        # Prepare concat_cross_bev
        keyval.link_module = self
        concat_cross_bev = keyval[:, :-1]  # Slice to remove last element
        concat_cross_bev.link_module = self
        concat_cross_bev = concat_cross_bev.permute([0, 2, 1])
        concat_cross_bev.link_module = self
        concat_cross_bev = concat_cross_bev.contiguous()
        concat_cross_bev.link_module = self
        concat_cross_bev = concat_cross_bev.view(
            int(batch_size),
            -1,
            int(concat_cross_bev_shape[0]),
            int(concat_cross_bev_shape[1]),
        )

        # Upsample concat_cross_bev to match bev_feature_upscale
        scale_h = bev_spatial_shape[0] / concat_cross_bev_shape[0]
        scale_w = bev_spatial_shape[1] / concat_cross_bev_shape[1]
        self.concat_cross_bev_upsample.params[1][1].data = np.array(
            [scale_h, scale_w], dtype=np.float32
        )
        concat_cross_bev = self.concat_cross_bev_upsample(concat_cross_bev)

        # Concatenate concat_cross_bev and cross_bev_feature
        cross_bev_feature = self.cross_bev_concat(concat_cross_bev, cross_bev_feature)

        # Project cross_bev_feature
        cross_bev_feature.link_module = self
        cross_bev_feature = cross_bev_feature.flatten(-2, -1)
        cross_bev_feature.link_module = self
        cross_bev_feature = cross_bev_feature.permute([0, 2, 1])
        cross_bev_feature = self.bev_proj_linear1(cross_bev_feature)
        cross_bev_feature = self.bev_proj_bias1(cross_bev_feature)
        cross_bev_feature = self.bev_proj_relu(cross_bev_feature)
        cross_bev_feature = self.bev_proj_ln(cross_bev_feature)
        cross_bev_feature.link_module = self
        cross_bev_feature = cross_bev_feature.permute([0, 2, 1])
        cross_bev_feature.link_module = self
        cross_bev_feature = cross_bev_feature.contiguous()
        cross_bev_feature.link_module = self
        cross_bev_feature = cross_bev_feature.view(
            int(batch_size), -1, int(bev_spatial_shape[0]), int(bev_spatial_shape[1])
        )

        # Query embedding
        query = self._query_embedding(self._query_indices)
        query.link_module = self
        query = query.unsqueeze(0)  # Add batch dimension
        # Repeat for batch
        query.link_module = self
        query = query.repeat(batch_size, 1, 1)

        # Transformer decoder
        query_out = query
        self._debug["query_initial"] = (
            query_out.data.copy() if query_out.data is not None else None
        )
        for i_layer, layer in enumerate(self._tf_decoder_layers):
            query_out = layer(query_out, keyval)
            self._debug[f"query_after_layer{i_layer}"] = (
                query_out.data.copy() if query_out.data is not None else None
            )

        # BEV semantic head
        bev_semantic_map = self._bev_semantic_conv1(bev_feature_upscale)
        bev_semantic_map = self._bev_semantic_conv1_bias(bev_semantic_map)
        bev_semantic_map = self._bev_semantic_relu(bev_semantic_map)
        bev_semantic_map = self._bev_semantic_conv2(bev_semantic_map)
        bev_semantic_map = self._bev_semantic_conv2_bias(bev_semantic_map)
        bev_semantic_map = self._bev_semantic_upsample(bev_semantic_map)

        # Split query output
        query_out.link_module = self
        trajectory_query = query_out[:, : self._query_splits[0], :]
        agents_query = query_out[:, self._query_splits[0] :, :]

        output: Dict[str, "SimTensor"] = {"bev_semantic_map": bev_semantic_map}

        # Trajectory head
        trajectory = self._trajectory_head(
            trajectory_query,
            agents_query,
            cross_bev_feature,
            bev_spatial_shape,
            status_encoding_expanded,
            targets=targets,
            global_img=None,
        )
        output.update(trajectory)

        # Agent head (returns tuple: agent_states, agent_labels)
        agent_states, agent_labels = self._agent_head(agents_query)
        output["agent_states"] = agent_states
        output["agent_labels"] = agent_labels

        return output


class AgentHead(SimNN_Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()
        self.name = "AgentHead"

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        # MLP states
        self._mlp_states_linear1 = F.Linear(
            "mlp_states_linear1", self._d_model, self._d_ffn, module=self
        )
        self._mlp_states_bias1 = F.Bias("mlp_states_bias1", [self._d_ffn])
        self._mlp_states_relu = F.Relu("mlp_states_relu")
        self._mlp_states_linear2 = F.Linear(
            "mlp_states_linear2", self._d_ffn, BoundingBox2DIndex.size(), module=self
        )
        self._mlp_states_bias2 = F.Bias("mlp_states_bias2", [BoundingBox2DIndex.size()])

        # MLP label
        self._mlp_label_linear = F.Linear(
            "mlp_label_linear", self._d_model, 1, module=self
        )
        self._mlp_label_bias = F.Bias("mlp_label_bias", [1])

        # Ops for agent state post-processing
        self._tanh_point = F.Tanh("agent_states_point_tanh")
        self._mul_point = F.MulFixed("agent_states_point_mul", "scale", np.float32(32))
        self._tanh_heading = F.Tanh("agent_states_heading_tanh")
        self._mul_heading = F.MulFixed(
            "agent_states_heading_mul", "scale", np.float32(np.pi)
        )
        self._concat = F.ConcatX("agent_states_concat", axis=-1)

        super().link_op2module()

    def __call__(self, agent_queries) -> "Tuple[SimTensor, SimTensor]":
        """TTSIM module forward pass."""

        # MLP states
        agent_states = self._mlp_states_linear1(agent_queries)
        agent_states = self._mlp_states_bias1(agent_states)
        agent_states = self._mlp_states_relu(agent_states)
        agent_states = self._mlp_states_linear2(agent_states)
        agent_states = self._mlp_states_bias2(agent_states)

        # Apply tanh scaling to POINT and HEADING like PyTorch
        agent_states.link_module = self
        agent_states_point = agent_states[..., BoundingBox2DIndex.POINT]
        agent_states_point = self._tanh_point(agent_states_point)
        agent_states_point = self._mul_point(agent_states_point)

        agent_states_heading = agent_states[
            ..., BoundingBox2DIndex.HEADING : BoundingBox2DIndex.HEADING + 1
        ]
        agent_states_heading = self._tanh_heading(agent_states_heading)
        agent_states_heading = self._mul_heading(agent_states_heading)

        agent_states_dim = agent_states[..., BoundingBox2DIndex.HEADING + 1 :]
        agent_states = self._concat(
            agent_states_point, agent_states_heading, agent_states_dim
        )

        # MLP label
        agent_labels = self._mlp_label_linear(agent_queries)
        agent_labels = self._mlp_label_bias(agent_labels)

        # Return both outputs
        return agent_states, agent_labels


class DiffMotionPlanningRefinementModule(SimNN_Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
        name_prefix=None,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        p = name_prefix or "DiffMotionPlanningRefinementModule"
        self.name = p

        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        # Plan classification branch
        self.plan_cls_linear1 = F.Linear(
            f"{p}_cls_lin1", embed_dims, embed_dims, module=self
        )
        self.plan_cls_bias1 = F.Bias(f"{p}_cls_bias1", [embed_dims])
        self.plan_cls_relu1 = F.Relu(f"{p}_cls_relu1")
        self.plan_cls_ln1 = F.LayerNorm(f"{p}_cls_ln1", embed_dims)
        self.plan_cls_linear2 = F.Linear(
            f"{p}_cls_lin2", embed_dims, embed_dims, module=self
        )
        self.plan_cls_bias2 = F.Bias(f"{p}_cls_bias2", [embed_dims])
        self.plan_cls_relu2 = F.Relu(f"{p}_cls_relu2")
        self.plan_cls_ln2 = F.LayerNorm(f"{p}_cls_ln2", embed_dims)
        self.plan_cls_linear3 = F.Linear(f"{p}_cls_lin3", embed_dims, 1, module=self)
        self.plan_cls_bias3 = F.Bias(f"{p}_cls_bias3", [1])
        self.plan_cls_squeeze = F.Squeeze(f"{p}_cls_squeeze")

        # Plan regression branch
        self.plan_reg_linear1 = F.Linear(
            f"{p}_reg_lin1", embed_dims, embed_dims, module=self
        )
        self.plan_reg_bias1 = F.Bias(f"{p}_reg_bias1", [embed_dims])
        self.plan_reg_relu1 = F.Relu(f"{p}_reg_relu1")
        self.plan_reg_linear2 = F.Linear(
            f"{p}_reg_lin2", embed_dims, embed_dims, module=self
        )
        self.plan_reg_bias2 = F.Bias(f"{p}_reg_bias2", [embed_dims])
        self.plan_reg_relu2 = F.Relu(f"{p}_reg_relu2")
        self.plan_reg_linear3 = F.Linear(
            f"{p}_reg_lin3", embed_dims, ego_fut_ts * 3, module=self
        )
        self.plan_reg_bias3 = F.Bias(f"{p}_reg_bias3", [ego_fut_ts * 3])

        # Reshape operation for regression output
        self.plan_reg_reshape = F.Reshape(f"{p}_reg_reshape")

        self.if_zeroinit_reg = False

        super().link_op2module()

    def init_weight(self):
        # Weight initialization would be handled during weight injection, not in forward pass
        pass

    def __call__(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # Classification branch
        plan_cls = self.plan_cls_linear1(traj_feature)
        plan_cls = self.plan_cls_bias1(plan_cls)
        plan_cls = self.plan_cls_relu1(plan_cls)
        plan_cls = self.plan_cls_ln1(plan_cls)
        plan_cls = self.plan_cls_linear2(plan_cls)
        plan_cls = self.plan_cls_bias2(plan_cls)
        plan_cls = self.plan_cls_relu2(plan_cls)
        plan_cls = self.plan_cls_ln2(plan_cls)
        plan_cls = self.plan_cls_linear3(plan_cls)
        plan_cls = self.plan_cls_bias3(plan_cls)
        # Squeeze last dimension: [batch, num_modes, 1] -> [batch, num_modes]
        squeeze_axis = F._from_data(
            f"{self.name}_cls_squeeze_axis",
            np.array([-1], dtype=np.int64),
            is_const=True,
        )
        self._tensors[squeeze_axis.name] = squeeze_axis
        plan_cls = self.plan_cls_squeeze(plan_cls, squeeze_axis)

        # Regression branch
        traj_delta = self.plan_reg_linear1(traj_feature)
        traj_delta = self.plan_reg_bias1(traj_delta)
        traj_delta = self.plan_reg_relu1(traj_delta)
        traj_delta = self.plan_reg_linear2(traj_delta)
        traj_delta = self.plan_reg_bias2(traj_delta)
        traj_delta = self.plan_reg_relu2(traj_delta)
        traj_delta = self.plan_reg_linear3(traj_delta)
        traj_delta = self.plan_reg_bias3(traj_delta)

        # Reshape to [bs, ego_fut_mode, ego_fut_ts, 3]
        traj_delta.link_module = self
        reg_shape = F._from_data(
            f"{self.name}_reg_shape",
            np.array([bs, ego_fut_mode, self.ego_fut_ts, 3], dtype=np.int64),
            is_const=True,
        )
        self._tensors[reg_shape.name] = reg_shape
        plan_reg = self.plan_reg_reshape(traj_delta, reg_shape)

        return plan_reg, plan_cls


class ModulationLayer(SimNN_Module):

    def __init__(self, embed_dims: int, condition_dims: int, name_prefix=None):
        super(ModulationLayer, self).__init__()
        p = name_prefix or "ModulationLayer"
        self.name = p

        self.if_zeroinit_scale = False
        self.embed_dims = embed_dims

        # Scale shift MLP - Mish activation + Linear
        self.scale_shift_mish = F.Mish(f"{p}_ss_mish")
        self.scale_shift_linear = F.Linear(
            f"{p}_ss_linear", condition_dims, embed_dims * 2, module=self
        )
        self.scale_shift_bias = F.Bias(f"{p}_ss_bias", [embed_dims * 2])

        # Pre-allocate ops for __call__
        self.add_scale_one = F.Add(f"{p}_add_scale")
        self.mul_feature = F.Mul(f"{p}_mul")
        self.add_shift = F.Add(f"{p}_add_shift")
        self.concat1 = F.ConcatX(f"{p}_concat1", axis=-1)
        self.concat2 = F.ConcatX(f"{p}_concat2", axis=-1)

        # Link operations to module for .data computation
        super().link_op2module()

    def init_weight(self):
        # Weight initialization handled during weight injection
        pass

    def __call__(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        # Concatenate conditions
        if global_cond is not None:
            global_feature = self.concat1(global_cond, time_embed)
        else:
            global_feature = time_embed

        if global_img is not None:
            global_img_flat = global_img.flatten(2, 3)
            global_img_flat = global_img_flat.permute(0, 2, 1)
            global_img_flat = global_img_flat.contiguous()
            global_feature = self.concat2(global_img_flat, global_feature)

        # Apply scale shift MLP
        scale_shift = self.scale_shift_mish(global_feature)
        scale_shift = self.scale_shift_linear(scale_shift)
        scale_shift = self.scale_shift_bias(scale_shift)

        # Split into scale and shift
        mid_dim = self.embed_dims
        scale = scale_shift[..., :mid_dim]
        shift = scale_shift[..., mid_dim:]

        # Apply modulation: traj_feature * (1 + scale) + shift
        one_const = F._from_data(f"{self.name}_one", np.float32(1.0))
        self._tensors[one_const.name] = one_const
        scale_plus_one = self.add_scale_one(one_const, scale)
        traj_feature = self.mul_feature(traj_feature, scale_plus_one)
        traj_feature = self.add_shift(traj_feature, shift)

        # Ensure output has link_module set for downstream operations
        traj_feature.link_module = self

        return traj_feature


class CustomTransformerDecoderLayer(SimNN_Module):
    def __init__(
        self,
        num_poses,
        d_model,
        d_ffn,
        config,
        name_prefix=None,
    ):
        super().__init__()
        p = name_prefix or "CustomTransformerDecoderLayer"
        self.name = p

        self.dropout_prob = 0.1

        # Dropout layers
        self.dropout = F.Dropout(f"{p}_dropout", 0.1, False, module=self)
        self.dropout1 = F.Dropout(f"{p}_dropout1", 0.1, False, module=self)

        # Cross BEV attention - now using full TTSIM implementation
        self.cross_bev_attention = GridSampleCrossBEVAttention_TTSIM(
            embed_dims=config.tf_d_model,
            num_heads=config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
            name_prefix=f"{p}_bev_attn",
        )

        # Cross agent attention - MultiheadAttention implemented manually
        self.d_model = config.tf_d_model
        self.num_heads = config.tf_num_head
        self.head_dim = self.d_model // self.num_heads

        # Q, K, V projections for cross agent attention
        self.cross_agent_q_linear = F.Linear(
            f"{p}_cross_agent_q", config.tf_d_model, config.tf_d_model, module=self
        )
        self.cross_agent_q_bias = F.Bias(f"{p}_cross_agent_q_bias", [config.tf_d_model])
        self.cross_agent_k_linear = F.Linear(
            f"{p}_cross_agent_k", config.tf_d_model, config.tf_d_model, module=self
        )
        self.cross_agent_k_bias = F.Bias(f"{p}_cross_agent_k_bias", [config.tf_d_model])
        self.cross_agent_v_linear = F.Linear(
            f"{p}_cross_agent_v", config.tf_d_model, config.tf_d_model, module=self
        )
        self.cross_agent_v_bias = F.Bias(f"{p}_cross_agent_v_bias", [config.tf_d_model])
        self.cross_agent_out_proj = F.Linear(
            f"{p}_cross_agent_out", config.tf_d_model, config.tf_d_model, module=self
        )
        self.cross_agent_out_bias = F.Bias(
            f"{p}_cross_agent_out_bias", [config.tf_d_model]
        )

        # Cross ego attention - MultiheadAttention implemented manually
        self.cross_ego_q_linear = F.Linear(
            f"{p}_cross_ego_q", config.tf_d_model, config.tf_d_model, module=self
        )
        self.cross_ego_q_bias = F.Bias(f"{p}_cross_ego_q_bias", [config.tf_d_model])
        self.cross_ego_k_linear = F.Linear(
            f"{p}_cross_ego_k", config.tf_d_model, config.tf_d_model, module=self
        )
        self.cross_ego_k_bias = F.Bias(f"{p}_cross_ego_k_bias", [config.tf_d_model])
        self.cross_ego_v_linear = F.Linear(
            f"{p}_cross_ego_v", config.tf_d_model, config.tf_d_model, module=self
        )
        self.cross_ego_v_bias = F.Bias(f"{p}_cross_ego_v_bias", [config.tf_d_model])
        self.cross_ego_out_proj = F.Linear(
            f"{p}_cross_ego_out", config.tf_d_model, config.tf_d_model, module=self
        )
        self.cross_ego_out_bias = F.Bias(f"{p}_cross_ego_out_bias", [config.tf_d_model])

        # FFN
        self.ffn_linear1 = F.Linear(
            f"{p}_ffn_linear1", config.tf_d_model, config.tf_d_ffn, module=self
        )
        self.ffn_bias1 = F.Bias(f"{p}_ffn_bias1", [config.tf_d_ffn])
        self.ffn_relu = F.Relu(f"{p}_ffn_relu")
        self.ffn_linear2 = F.Linear(
            f"{p}_ffn_linear2", config.tf_d_ffn, config.tf_d_model, module=self
        )
        self.ffn_bias2 = F.Bias(f"{p}_ffn_bias2", [config.tf_d_model])

        # Layer norms
        self.norm1 = F.LayerNorm(f"{p}_norm1", config.tf_d_model)
        self.norm2 = F.LayerNorm(f"{p}_norm2", config.tf_d_model)
        self.norm3 = F.LayerNorm(f"{p}_norm3", config.tf_d_model)

        # Time modulation
        self.time_modulation = ModulationLayer(
            config.tf_d_model, 256, name_prefix=f"{p}_time_mod"
        )

        # Task decoder
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
            name_prefix=f"{p}_task_dec",
        )

        # Pre-allocate ops for _multi_head_attention (cross_agent)
        self.ca_reshape_q = F.Reshape(f"{p}_ca_reshape_q")
        self.ca_reshape_k = F.Reshape(f"{p}_ca_reshape_k")
        self.ca_reshape_v = F.Reshape(f"{p}_ca_reshape_v")
        self.ca_matmul_qk = F.MatMul(f"{p}_ca_matmul_qk")
        self.ca_scale = F.MulFixed(
            f"{p}_ca_scale", "scale", np.float32(1.0 / np.sqrt(self.head_dim))
        )
        self.ca_softmax = F.Softmax(f"{p}_ca_softmax", axis=-1)
        self.ca_matmul_av = F.MatMul(f"{p}_ca_matmul_av")
        self.ca_reshape_out = F.Reshape(f"{p}_ca_reshape_out")

        # Pre-allocate ops for _multi_head_attention (cross_ego)
        self.ce_reshape_q = F.Reshape(f"{p}_ce_reshape_q")
        self.ce_reshape_k = F.Reshape(f"{p}_ce_reshape_k")
        self.ce_reshape_v = F.Reshape(f"{p}_ce_reshape_v")
        self.ce_matmul_qk = F.MatMul(f"{p}_ce_matmul_qk")
        self.ce_scale = F.MulFixed(
            f"{p}_ce_scale", "scale", np.float32(1.0 / np.sqrt(self.head_dim))
        )
        self.ce_softmax = F.Softmax(f"{p}_ce_softmax", axis=-1)
        self.ce_matmul_av = F.MatMul(f"{p}_ce_matmul_av")
        self.ce_reshape_out = F.Reshape(f"{p}_ce_reshape_out")

        # Pre-allocate ops for __call__
        self.add_agent_residual = F.Add(f"{p}_add_agent_residual")
        self.add_ego_residual = F.Add(f"{p}_add_ego_residual")
        self.add_traj_op = F.Add(f"{p}_add_traj")
        self.heading_tanh = F.Tanh(f"{p}_heading_tanh")
        self.heading_mul = F.MulFixed(f"{p}_heading_mul", "scale", np.float32(np.pi))
        self.heading_squeeze = F.Reshape(f"{p}_heading_squeeze")
        self.heading_unsqueeze = F.Reshape(f"{p}_heading_unsqueeze")
        self.concat_poses = F.ConcatX(f"{p}_concat_poses", axis=-1)

        # Link operations to module for .data computation
        super().link_op2module()

    def _multi_head_attention(
        self,
        query,
        key,
        value,
        q_linear,
        q_bias,
        k_linear,
        k_bias,
        v_linear,
        v_bias,
        out_proj,
        out_bias,
        attn_type="cross_agent",
    ):
        """Manual multi-head attention implementation using pre-allocated ops."""
        batch_size = query.shape[0]
        tgt_len = query.shape[1]
        src_len = key.shape[1]

        # Select pre-allocated ops based on attention type
        if attn_type == "cross_agent":
            reshape_q, reshape_k, reshape_v = (
                self.ca_reshape_q,
                self.ca_reshape_k,
                self.ca_reshape_v,
            )
            matmul_qk, scale_op, softmax_op = (
                self.ca_matmul_qk,
                self.ca_scale,
                self.ca_softmax,
            )
            matmul_av, reshape_out = self.ca_matmul_av, self.ca_reshape_out
        else:
            reshape_q, reshape_k, reshape_v = (
                self.ce_reshape_q,
                self.ce_reshape_k,
                self.ce_reshape_v,
            )
            matmul_qk, scale_op, softmax_op = (
                self.ce_matmul_qk,
                self.ce_scale,
                self.ce_softmax,
            )
            matmul_av, reshape_out = self.ce_matmul_av, self.ce_reshape_out

        # Q, K, V projections
        Q = q_linear(query)
        Q = q_bias(Q)
        K = k_linear(key)
        K = k_bias(K)
        V = v_linear(value)
        V = v_bias(V)

        # Shape tensors — prefixed by self.name for uniqueness across layers
        q_shape = F._from_data(
            f"{self.name}_{attn_type}_q_shape",
            np.array(
                [batch_size, tgt_len, self.num_heads, self.head_dim], dtype=np.int64
            ),
            is_const=True,
        )
        self._tensors[q_shape.name] = q_shape
        kv_shape = F._from_data(
            f"{self.name}_{attn_type}_kv_shape",
            np.array(
                [batch_size, src_len, self.num_heads, self.head_dim], dtype=np.int64
            ),
            is_const=True,
        )
        self._tensors[kv_shape.name] = kv_shape

        Q.link_module = self
        Q = reshape_q(Q, q_shape)
        K.link_module = self
        K = reshape_k(K, kv_shape)
        V.link_module = self
        V = reshape_v(V, kv_shape)

        # Transpose to [batch, num_heads, length, head_dim]
        Q = Q.permute([0, 2, 1, 3])
        K = K.permute([0, 2, 1, 3])
        V = V.permute([0, 2, 1, 3])

        # scores = Q @ K^T / sqrt(head_dim)
        K_T = K.permute([0, 1, 3, 2])
        scores = matmul_qk(Q, K_T)
        scores = scale_op(scores)
        attn_weights = softmax_op(scores)

        # Apply attention to values
        attn_output = matmul_av(attn_weights, V)

        attn_output = attn_output.permute([0, 2, 1, 3])

        attn_output.link_module = self
        out_shape = F._from_data(
            f"{self.name}_{attn_type}_out_shape",
            np.array([batch_size, tgt_len, self.d_model], dtype=np.int64),
            is_const=True,
        )
        self._tensors[out_shape.name] = out_shape
        attn_output = reshape_out(attn_output, out_shape)

        output = out_proj(attn_output)
        output = out_bias(output)

        return output

    def __call__(
        self,
        traj_feature,
        noisy_traj_points,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        time_embed,
        status_encoding,
        global_img=None,
    ):
        """Forward pass of CustomTransformerDecoderLayer.

        Args:
            traj_feature: [B, 20, 256] trajectory features
            noisy_traj_points: [B, 20, 8, 2] noisy trajectory points
            bev_feature: [B, 256, 25, 25] BEV features
            bev_spatial_shape: (25, 25) spatial shape of BEV
            agents_query: [B, num_agents, 256] agent queries
            ego_query: [B, 1, 256] ego query
            time_embed: [B, 1, 256] time embedding
            status_encoding: [B, 1, 256] status encoding (unused in forward)
            global_img: optional global image features

        Returns:
            poses_reg: [B, 20, 8, 3] regression outputs (x, y, heading)
            poses_cls: [B, 20] classification outputs
        """
        # Cross BEV attention (only x, y — drop heading channel)
        noisy_traj_points.link_module = self
        traj_points_xy = noisy_traj_points[..., :2]
        traj_points_xy.link_module = self
        traj_feature = self.cross_bev_attention(
            traj_feature, traj_points_xy, bev_feature, bev_spatial_shape
        )
        traj_feature.link_module = self

        # Cross agent attention with residual
        attn_out = self._multi_head_attention(
            traj_feature,
            agents_query,
            agents_query,
            self.cross_agent_q_linear,
            self.cross_agent_q_bias,
            self.cross_agent_k_linear,
            self.cross_agent_k_bias,
            self.cross_agent_v_linear,
            self.cross_agent_v_bias,
            self.cross_agent_out_proj,
            self.cross_agent_out_bias,
            "cross_agent",
        )
        attn_out = self.dropout(attn_out)
        traj_feature = self.add_agent_residual(traj_feature, attn_out)
        traj_feature = self.norm1(traj_feature)
        traj_feature.link_module = self

        # Cross ego attention with residual
        attn_out2 = self._multi_head_attention(
            traj_feature,
            ego_query,
            ego_query,
            self.cross_ego_q_linear,
            self.cross_ego_q_bias,
            self.cross_ego_k_linear,
            self.cross_ego_k_bias,
            self.cross_ego_v_linear,
            self.cross_ego_v_bias,
            self.cross_ego_out_proj,
            self.cross_ego_out_bias,
            "cross_ego",
        )
        attn_out2 = self.dropout1(attn_out2)
        traj_feature = self.add_ego_residual(traj_feature, attn_out2)
        traj_feature = self.norm2(traj_feature)
        traj_feature.link_module = self

        # Feedforward network
        ffn_out = self.ffn_linear1(traj_feature)
        ffn_out = self.ffn_bias1(ffn_out)
        ffn_out = self.ffn_relu(ffn_out)
        ffn_out = self.ffn_linear2(ffn_out)
        ffn_out = self.ffn_bias2(ffn_out)
        traj_feature = self.norm3(ffn_out)
        traj_feature.link_module = self

        # Time modulation
        traj_feature = self.time_modulation(
            traj_feature, time_embed, global_cond=None, global_img=global_img
        )
        traj_feature.link_module = self

        # Task decoder - get regression and classification outputs
        poses_reg, poses_cls = self.task_decoder(traj_feature)
        poses_reg.link_module = self
        poses_cls.link_module = self

        # Post-process: add noisy_traj_xy to xy coordinates
        poses_reg_xy = poses_reg[..., :2]
        noisy_xy = noisy_traj_points[..., :2]
        noisy_xy.link_module = self
        poses_reg_xy = self.add_traj_op(poses_reg_xy, noisy_xy)
        poses_reg_xy.link_module = self

        # Post-process: apply tanh to heading and scale by pi
        poses_reg_heading = poses_reg[..., 2:3]  # [B, 20, 8, 1]

        # Squeeze out the last dimension: [B, 20, 8, 1] -> [B, 20, 8]
        squeeze_shape = F._from_data(
            f"{self.name}_heading_squeeze_shape",
            np.array(
                [
                    poses_reg_heading.shape[0],
                    poses_reg_heading.shape[1],
                    poses_reg_heading.shape[2],
                ],
                dtype=np.int64,
            ),
            is_const=True,
        )
        self._tensors[squeeze_shape.name] = squeeze_shape
        poses_reg_heading = self.heading_squeeze(poses_reg_heading, squeeze_shape)
        poses_reg_heading.link_module = self

        # Apply tanh
        poses_reg_heading = self.heading_tanh(poses_reg_heading)
        poses_reg_heading.link_module = self

        # Scale by pi
        poses_reg_heading = self.heading_mul(poses_reg_heading)
        poses_reg_heading.link_module = self

        # Unsqueeze back: [B, 20, 8] -> [B, 20, 8, 1]
        heading_shape = F._from_data(
            f"{self.name}_heading_unsqueeze_shape",
            np.array(
                [
                    poses_reg_heading.shape[0],
                    poses_reg_heading.shape[1],
                    poses_reg_heading.shape[2],
                    1,
                ],
                dtype=np.int64,
            ),
            is_const=True,
        )
        self._tensors[heading_shape.name] = heading_shape
        poses_reg_heading = self.heading_unsqueeze(poses_reg_heading, heading_shape)
        poses_reg_heading.link_module = self

        # Concatenate xy and heading: [B, 20, 8, 2] + [B, 20, 8, 1] -> [B, 20, 8, 3]
        poses_reg = self.concat_poses(poses_reg_xy, poses_reg_heading)
        poses_reg.link_module = self

        return poses_reg, poses_cls


def _recursive_relink(module):
    """Recursively re-link operations for module and all nested SimNN_Module instances."""
    # Re-link current module
    if hasattr(module, "link_op2module"):
        module.link_op2module()

    # Recursively re-link nested modules
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(module, attr_name)
            # Check if it's a SimNN_Module (has link_op2module method)
            if hasattr(attr, "link_op2module") and attr is not module:
                _recursive_relink(attr)
        except:
            pass


def _get_clones(module, N):
    """Deep copy modules and recursively re-link all operations."""
    clones = []
    for i in range(N):
        cloned = copy.deepcopy(module)
        # Recursively re-link all nested modules
        _recursive_relink(cloned)
        clones.append(cloned)
    return clones


class CustomTransformerDecoder(SimNN_Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
    ):
        super().__init__()
        self.name = "CustomTransformerDecoder"

        # Accept either a single layer or a list of layers
        if isinstance(decoder_layer, list):
            # List of pre-created layers provided
            assert (
                len(decoder_layer) == num_layers
            ), f"Expected {num_layers} layers, got {len(decoder_layer)}"
            layer_list = decoder_layer
        else:
            # Single layer provided - use it for all iterations (same as PyTorch with deepcopy)
            # Note: This means all layers share the same operations/weights
            layer_list = [decoder_layer] * num_layers

        self.num_layers = num_layers

        # Register each layer as a named attribute so __setattr__ routes
        # Module instances into _submodules (plain list assignment bypasses it)
        for i, layer in enumerate(layer_list):
            setattr(self, f"layer_{i}", layer)
        # Keep a reference list for iteration convenience
        self.layers = layer_list

    def __call__(
        self,
        traj_feature,
        noisy_traj_points,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        time_embed,
        status_encoding,
        global_img=None,
    ):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points

        # Each layer is a SEPARATE CustomTransformerDecoderLayer instance
        # with unique op names, so iterating through all layers is safe.
        for layer_idx, mod in enumerate(self.layers):
            # Set link_module for all input tensors to the current layer
            traj_feature.link_module = mod
            traj_points.link_module = mod
            bev_feature.link_module = mod
            agents_query.link_module = mod
            ego_query.link_module = mod
            time_embed.link_module = mod
            status_encoding.link_module = mod

            poses_reg, poses_cls = mod(
                traj_feature,
                traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
            )

            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)

            # Register intermediate tensors via setattr
            setattr(self, f"poses_reg_l{layer_idx}", poses_reg)
            setattr(self, f"poses_cls_l{layer_idx}", poses_cls)

            # Prepare traj_points for next iteration
            poses_reg.link_module = mod
            traj_points = poses_reg[..., :2]
            setattr(self, f"traj_points_l{layer_idx}", traj_points)

        return poses_reg_list, poses_cls_list


class GenSineEmbedPosition_TTSIM(SimNN_Module):
    """TTSIM implementation of gen_sineembed_for_position (DAB-DETR style).

    Matches PyTorch gen_sineembed_for_position(pos_tensor, hidden_dim=64):
      - Computes sinusoidal positional embeddings for x,y coordinates
      - Uses frequencies: 10000^(2*(i//2)/half_dim) as denominators
      - Interleaves sin(even_freq) and cos(odd_freq)
      - Outputs [pos_y, pos_x] concatenated
    """

    def __init__(self, hidden_dim=64, name=None):
        super().__init__()
        self.name = name or "GenSineEmbedPosition"
        self.hidden_dim = hidden_dim
        self.half_hidden_dim = hidden_dim // 2  # 32
        self.scale = np.float32(2.0 * np.pi)

        # Pre-compute frequency denominators: 10000^(2*(i//2)/half_dim)
        dim_t = np.arange(self.half_hidden_dim, dtype=np.float32)
        self.dim_t_np = (10000.0 ** (2.0 * (dim_t // 2) / self.half_hidden_dim)).astype(
            np.float32
        )

        # Store as constant tensors for graph path
        self.dim_t_const = F._from_data(
            f"{self.name}_dim_t",
            self.dim_t_np.reshape(1, self.half_hidden_dim),
            is_const=True,
        )
        self.scale_const = F._from_data(f"{self.name}_scale", self.scale, is_const=True)

        # Create ops for graph path
        self.mul_scale_x = F.Mul(f"{self.name}_mul_scale_x")
        self.mul_scale_y = F.Mul(f"{self.name}_mul_scale_y")
        self.div_x = F.Div(f"{self.name}_div_x")
        self.div_y = F.Div(f"{self.name}_div_y")
        self.sin_x = F.Sin(f"{self.name}_sin_x")
        self.cos_x = F.Cos(f"{self.name}_cos_x")
        self.sin_y = F.Sin(f"{self.name}_sin_y")
        self.cos_y = F.Cos(f"{self.name}_cos_y")
        self.concat_xy = F.ConcatX(f"{self.name}_concat", axis=-1)

        super().link_op2module()

    def __call__(self, pos_tensor):
        """
        Compute sinusoidal positional embedding for positions.
        Args:
            pos_tensor: [..., 2] or [..., 3] tensor (x, y[, heading])
        Returns:
            Tensor of shape [..., hidden_dim] containing sinusoidal embeddings
        """
        # Unique prefix per call
        if not hasattr(self, "_call_count"):
            self._call_count = 0
        self._call_count += 1
        _cc = self._call_count

        if pos_tensor.data is not None:
            # Numpy computation path (for data validation)
            data = pos_tensor.data
            x_embed = data[..., 0] * self.scale  # [...]
            y_embed = data[..., 1] * self.scale

            pos_x = x_embed[..., np.newaxis] / self.dim_t_np  # [..., 32]
            pos_y = y_embed[..., np.newaxis] / self.dim_t_np

            # sin on even indices, cos on odd indices, then interleave
            # stack([sin_even, cos_odd], dim=-1).flatten(-2)
            pos_x_sin = np.sin(pos_x[..., 0::2])  # [..., 16]
            pos_x_cos = np.cos(pos_x[..., 1::2])  # [..., 16]
            pos_x_embed = np.stack([pos_x_sin, pos_x_cos], axis=-1)  # [..., 16, 2]
            pos_x_embed = pos_x_embed.reshape(pos_x.shape)  # [..., 32]

            pos_y_sin = np.sin(pos_y[..., 0::2])
            pos_y_cos = np.cos(pos_y[..., 1::2])
            pos_y_embed = np.stack([pos_y_sin, pos_y_cos], axis=-1)
            pos_y_embed = pos_y_embed.reshape(pos_y.shape)

            # Concat [pos_y, pos_x] to match PyTorch
            result_np = np.concatenate([pos_y_embed, pos_x_embed], axis=-1).astype(
                np.float32
            )

            result = F._from_data(f"{self.name}_result_c{_cc}", result_np)
            self._tensors[result.name] = result
            return result

        # Shape inference path (when data is None)
        # Output shape is input_shape[:-1] + (hidden_dim,)
        pos_tensor.link_module = self
        out_shape = list(pos_tensor.shape[:-1]) + [self.hidden_dim]
        result = F._from_data(
            f"{self.name}_result_placeholder_c{_cc}",
            np.zeros(out_shape, dtype=np.float32),
        )
        self._tensors[result.name] = result
        return result


# ── Per-iteration normalization / denormalization modules ────────────────
# Each instance owns a unique set of ops so the denoising loop can be
# fully unrolled without any shared-op graph cycles.


class NormOdoModule(SimNN_Module):
    """Normalize odometry: result = 2*(x + offset)/range - 1 per axis."""

    def __init__(self, name_prefix):
        super().__init__()
        self.name = name_prefix
        p = name_prefix

        # Ops
        self.norm_x_add1 = F.Add(f"{p}_x_add1")
        self.norm_x_mul1 = F.MulFixed(f"{p}_x_mul1", "scale", np.float32(2.0))
        self.norm_x_div = F.MulFixed(f"{p}_x_div", "scale", np.float32(1.0 / 56.9))
        self.norm_x_sub = F.Sub(f"{p}_x_sub")

        self.norm_y_add1 = F.Add(f"{p}_y_add1")
        self.norm_y_mul1 = F.MulFixed(f"{p}_y_mul1", "scale", np.float32(2.0))
        self.norm_y_div = F.MulFixed(f"{p}_y_div", "scale", np.float32(1.0 / 46.0))
        self.norm_y_sub = F.Sub(f"{p}_y_sub")

        self.norm_head_add1 = F.Add(f"{p}_head_add1")
        self.norm_head_mul1 = F.MulFixed(f"{p}_head_mul1", "scale", np.float32(2.0))
        self.norm_head_div = F.MulFixed(f"{p}_head_div", "scale", np.float32(1.0 / 3.9))
        self.norm_head_sub = F.Sub(f"{p}_head_sub")

        self.norm_concat = F.ConcatX(f"{p}_concat", axis=-1)

        # Padding op for 2D inputs (x, y only → pad heading with zeros)
        self.pad_concat = F.ConcatX(f"{p}_pad_concat", axis=-1)

        # Constants
        self.x_offset = F._from_data(f"{p}_x_offset", np.float32(1.2), is_const=True)
        self.x_one = F._from_data(f"{p}_x_one", np.float32(1.0), is_const=True)
        self.y_offset = F._from_data(f"{p}_y_offset", np.float32(20.0), is_const=True)
        self.y_one = F._from_data(f"{p}_y_one", np.float32(1.0), is_const=True)
        self.head_offset = F._from_data(
            f"{p}_head_offset", np.float32(2.0), is_const=True
        )
        self.head_one = F._from_data(f"{p}_head_one", np.float32(1.0), is_const=True)

        super().link_op2module()

    def __call__(self, odo_info_fut):
        odo_info_fut.link_module = self

        # If 2-D input (x, y only) → pad heading with zeros
        if odo_info_fut.shape[-1] == 2:
            zeros_shape = list(odo_info_fut.shape)
            zeros_shape[-1] = 1
            zeros_heading = F._from_data(
                f"{self.name}_zeros_heading",
                np.zeros(zeros_shape, dtype=np.float32),
                is_const=True,
            )
            self._tensors[zeros_heading.name] = zeros_heading
            odo_info_fut = self.pad_concat(odo_info_fut, zeros_heading)

        odo_x = odo_info_fut[..., 0:1]
        odo_y = odo_info_fut[..., 1:2]
        odo_h = odo_info_fut[..., 2:3]

        nx = self.norm_x_sub(
            self.norm_x_div(self.norm_x_mul1(self.norm_x_add1(odo_x, self.x_offset))),
            self.x_one,
        )
        ny = self.norm_y_sub(
            self.norm_y_div(self.norm_y_mul1(self.norm_y_add1(odo_y, self.y_offset))),
            self.y_one,
        )
        nh = self.norm_head_sub(
            self.norm_head_div(
                self.norm_head_mul1(self.norm_head_add1(odo_h, self.head_offset))
            ),
            self.head_one,
        )

        result = self.norm_concat(nx, ny, nh)
        result.link_module = self
        return result


class DenormOdoModule(SimNN_Module):
    """Denormalize odometry: result = (x + 1)/2 * range - offset per axis."""

    def __init__(self, name_prefix):
        super().__init__()
        self.name = name_prefix
        p = name_prefix

        # Ops
        self.denorm_x_add1 = F.Add(f"{p}_x_add1")
        self.denorm_x_mul1 = F.MulFixed(f"{p}_x_mul1", "scale", np.float32(0.5))
        self.denorm_x_mul2 = F.MulFixed(f"{p}_x_mul2", "scale", np.float32(56.9))
        self.denorm_x_sub = F.Sub(f"{p}_x_sub")

        self.denorm_y_add1 = F.Add(f"{p}_y_add1")
        self.denorm_y_mul1 = F.MulFixed(f"{p}_y_mul1", "scale", np.float32(0.5))
        self.denorm_y_mul2 = F.MulFixed(f"{p}_y_mul2", "scale", np.float32(46.0))
        self.denorm_y_sub = F.Sub(f"{p}_y_sub")

        self.denorm_head_add1 = F.Add(f"{p}_head_add1")
        self.denorm_head_mul1 = F.MulFixed(f"{p}_head_mul1", "scale", np.float32(0.5))
        self.denorm_head_mul2 = F.MulFixed(f"{p}_head_mul2", "scale", np.float32(3.9))
        self.denorm_head_sub = F.Sub(f"{p}_head_sub")

        self.denorm_concat = F.ConcatX(f"{p}_concat", axis=-1)

        # Constants
        self.x_one = F._from_data(f"{p}_x_one", np.float32(1.0), is_const=True)
        self.x_offset = F._from_data(f"{p}_x_offset", np.float32(1.2), is_const=True)
        self.y_one = F._from_data(f"{p}_y_one", np.float32(1.0), is_const=True)
        self.y_offset = F._from_data(f"{p}_y_offset", np.float32(20.0), is_const=True)
        self.head_one = F._from_data(f"{p}_head_one", np.float32(1.0), is_const=True)
        self.head_offset = F._from_data(
            f"{p}_head_offset", np.float32(2.0), is_const=True
        )

        super().link_op2module()

    def __call__(self, odo_info_fut):
        odo_info_fut.link_module = self

        odo_x = odo_info_fut[..., 0:1]
        odo_y = odo_info_fut[..., 1:2]
        odo_h = odo_info_fut[..., 2:3]

        dx = self.denorm_x_sub(
            self.denorm_x_mul2(
                self.denorm_x_mul1(self.denorm_x_add1(odo_x, self.x_one))
            ),
            self.x_offset,
        )
        dy = self.denorm_y_sub(
            self.denorm_y_mul2(
                self.denorm_y_mul1(self.denorm_y_add1(odo_y, self.y_one))
            ),
            self.y_offset,
        )
        dh = self.denorm_head_sub(
            self.denorm_head_mul2(
                self.denorm_head_mul1(self.denorm_head_add1(odo_h, self.head_one))
            ),
            self.head_offset,
        )

        result = self.denorm_concat(dx, dy, dh)
        result.link_module = self
        return result


class TrajectoryHead(SimNN_Module):
    """Trajectory prediction head."""

    def __init__(
        self,
        num_poses: int,
        d_ffn: int,
        d_model: int,
        plan_anchor_path: str,
        config: TransfuserConfig,
    ):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()
        self.name = "TrajectoryHead"

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        # Note: Diffusion scheduler would need to be adapted for TTSIM
        # For now, we keep it as a Python object (not part of computation graph)
        self.diffusion_scheduler = None  # DDIMScheduler placeholder

        # Load plan anchor
        plan_anchor = load_plan_anchor(plan_anchor_path)
        
        # Load plan anchor
        # if os.path.exists(plan_anchor_path):
        #     plan_anchor = np.load(plan_anchor_path)
        # else:
        #     logger.warning(
        #         f"plan anchor file not found at {plan_anchor_path}; using deterministic fallback"
            # )
            
        fallback_seed = getattr(config, "plan_anchor_fallback_seed", 42)
        logger.warning(
            "plan anchor file not found at %s; using deterministic fallback with seed=%d",
            plan_anchor_path,
            fallback_seed,
        )
        rng = np.random.default_rng(fallback_seed)
        plan_anchor = rng.standard_normal(
            (self.ego_fut_mode, self._num_poses, 2)
        ).astype(np.float32)

        self.plan_anchor = F._from_data(
            "plan_anchor", plan_anchor.astype(np.float32), is_param=True
        )
        self.plan_anchor.link_module = self

        # Sinusoidal positional embedding for trajectory positions
        # Matches PyTorch gen_sineembed_for_position(hidden_dim=64)
        # Initial normalization (before denoising loop)
        self.norm_odo_init = NormOdoModule("norm_odo_init")

        # Per-iteration denoising modules (2 timestep iterations: [10, 0])
        for it in range(2):
            p = f"iter{it}"

            # Denormalization & normalization
            setattr(self, f"denorm_odo_{p}", DenormOdoModule(f"denorm_odo_{p}"))
            setattr(self, f"norm_odo_{p}", NormOdoModule(f"norm_odo_{p}"))

            # GenSineEmbedPosition
            setattr(
                self,
                f"gen_sineembed_{p}",
                GenSineEmbedPosition_TTSIM(hidden_dim=64, name=f"gen_sineembed_{p}"),
            )

            # Plan anchor encoder  (Linear → Bias → ReLU → LN → Linear → Bias)
            setattr(
                self,
                f"plan_enc_linear1_{p}",
                F.Linear(f"plan_enc_linear1_{p}", 512, d_model, module=self),
            )
            setattr(
                self, f"plan_enc_bias1_{p}", F.Bias(f"plan_enc_bias1_{p}", [d_model])
            )
            setattr(self, f"plan_enc_relu_{p}", F.Relu(f"plan_enc_relu_{p}"))
            setattr(self, f"plan_enc_ln_{p}", F.LayerNorm(f"plan_enc_ln_{p}", d_model))
            setattr(
                self,
                f"plan_enc_linear2_{p}",
                F.Linear(f"plan_enc_linear2_{p}", d_model, d_model, module=self),
            )
            setattr(
                self, f"plan_enc_bias2_{p}", F.Bias(f"plan_enc_bias2_{p}", [d_model])
            )

            # Time MLP  (SinusoidalPosEmb → Linear → Bias → Mish → Linear → Bias)
            setattr(
                self,
                f"time_sinusoidal_{p}",
                SinusoidalPosEmb_TTSIM(d_model, name=f"time_sinusoidal_{p}"),
            )
            setattr(
                self,
                f"time_linear1_{p}",
                F.Linear(f"time_linear1_{p}", d_model, d_model * 4, module=self),
            )
            setattr(self, f"time_bias1_{p}", F.Bias(f"time_bias1_{p}", [d_model * 4]))
            setattr(self, f"time_mish_{p}", F.Mish(f"time_mish_{p}"))
            setattr(
                self,
                f"time_linear2_{p}",
                F.Linear(f"time_linear2_{p}", d_model * 4, d_model, module=self),
            )
            setattr(self, f"time_bias2_{p}", F.Bias(f"time_bias2_{p}", [d_model]))

            # Diff decoder per iteration (each has 2 separate decoder layers)
            decoder_layers = []
            for lid in range(2):
                layer = CustomTransformerDecoderLayer(
                    num_poses=num_poses,
                    d_model=d_model,
                    d_ffn=d_ffn,
                    config=config,
                    name_prefix=f"decoder_{p}_l{lid}",
                )
                decoder_layers.append(layer)
            setattr(
                self, f"diff_decoder_{p}", CustomTransformerDecoder(decoder_layers, 2)
            )

        # Loss computer (not part of forward graph, used in training)
        self.loss_computer = None  # LossComputer placeholder

        # Pre-allocate ops used in forward_test
        self.add_noise_op = F.Add("add_noise")
        for it in range(2):
            p = f"iter{it}"
            setattr(
                self,
                f"sched_mul_model_{p}",
                F.MulFixed(f"sched_mul_model_{p}", "coeff", np.float32(0.9)),
            )
            setattr(
                self,
                f"sched_mul_sample_{p}",
                F.MulFixed(f"sched_mul_sample_{p}", "coeff", np.float32(0.1)),
            )
            setattr(self, f"sched_add_{p}", F.Add(f"sched_add_{p}"))

    def __call__(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        targets=None,
        global_img=None,
        noise=None,
    ) -> Dict[str, "SimTensor"]:
        """TTSIM module forward pass."""
        return self.forward_test(
            ego_query,
            agents_query,
            bev_feature,
            bev_spatial_shape,
            status_encoding,
            global_img,
            noise=noise,
        )

    def forward_train(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        targets=None,
        global_img=None,
    ) -> Dict[str, "SimTensor"]:
        """Training forward pass - placeholder for TTSIM."""
        # Training logic would need special handling for noise injection
        # and loss computation, which are not part of the inference graph
        raise NotImplementedError

    def forward_test(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        noise=None,
    ) -> Dict[str, "SimTensor"]:
        """Test forward pass — fully-unrolled 2-step denoising loop.

        Each denoising iteration uses SEPARATE module instances created in
        __init__ (denorm_odo_iter{N}, gen_sineembed_iter{N}, …) so no shared
        ops get called twice.  Intermediate tensors are registered with
        setattr for graph visibility.

        Pipeline:
          1. plan_anchor → norm_odo_init → add_noise
          2. For iter_idx in {0, 1} (timesteps [10, 0]):
             clamp → denorm_odo → gen_sineembed → plan_enc → time_mlp
             → diff_decoder → norm_odo → scheduler.step
          3. mode_select (argmax + gather)
        """
        bs = ego_query.shape[0]

        # Reset _call_count on all sub-modules
        self._reset_call_counts()

        # ─── 1. plan_anchor → norm_odo_init ───
        plan_anchor_expanded = self.plan_anchor.unsqueeze(0)
        plan_anchor_expanded = plan_anchor_expanded.repeat(bs, 1, 1, 1)
        img = self.norm_odo_init(plan_anchor_expanded)

        # ─── 2. add_noise: img = img + 0.1 * noise ───
        if noise is not None:
            if noise.data is not None:
                noise_data = noise.data
                if noise_data.shape[-1] != img.shape[-1]:
                    pad_shape = list(noise_data.shape)
                    pad_shape[-1] = img.shape[-1] - noise_data.shape[-1]
                    noise_data = np.concatenate(
                        [noise_data, np.zeros(pad_shape, dtype=np.float32)], axis=-1
                    )
                noise_scaled_data = (noise_data * 0.1).astype(np.float32)
            else:
                noise_scaled_data = np.zeros(img.shape, dtype=np.float32)
        else:
            noise_np = np.random.randn(*img.shape).astype(np.float32)
            noise_scaled_data = (noise_np * 0.1).astype(np.float32)

        noise_scaled = F._from_data("noise_scaled", noise_scaled_data)
        self._tensors[noise_scaled.name] = noise_scaled
        img = self.add_noise_op(img, noise_scaled)

        ego_fut_mode = img.shape[1]

        # ─── 3. Two-step denoising loop (timesteps [10, 0]) ───
        # Each iteration selects its own module copies via getattr so no
        # shared __init__ op is called more than once.
        roll_timesteps = [10, 0]
        assert roll_timesteps == [10, 0], (
            f"this implementation currently assumes timesteps [10, 0], got {roll_timesteps}"
        )

        for iter_idx, k in enumerate(roll_timesteps):
            p = f"iter{iter_idx}"

            # ── clamp(img, -1, 1) ──
            if img.data is not None:
                clamped_data = np.clip(img.data, -1.0, 1.0).astype(np.float32)
                x_boxes = F._from_data(f"x_boxes_{p}", clamped_data)
                x_boxes.link_module = self
                self._tensors[x_boxes.name] = x_boxes
            else:
                x_boxes = img

            # ── denorm_odo (per-iteration module) ──
            denorm_odo = getattr(self, f"denorm_odo_{p}")
            noisy_traj_points = denorm_odo(x_boxes)
            setattr(self, f"noisy_traj_{p}", noisy_traj_points)

            # ── gen_sineembed ──
            gen_sine = getattr(self, f"gen_sineembed_{p}")
            traj_pos_embed = gen_sine(noisy_traj_points)
            traj_pos_embed.link_module = self
            traj_pos_embed = traj_pos_embed.flatten(-2)
            setattr(self, f"traj_pos_embed_{p}", traj_pos_embed)

            # ── plan_anchor_encoder ──
            traj_feature = getattr(self, f"plan_enc_linear1_{p}")(traj_pos_embed)
            traj_feature = getattr(self, f"plan_enc_bias1_{p}")(traj_feature)
            traj_feature = getattr(self, f"plan_enc_relu_{p}")(traj_feature)
            traj_feature = getattr(self, f"plan_enc_ln_{p}")(traj_feature)
            traj_feature = getattr(self, f"plan_enc_linear2_{p}")(traj_feature)
            traj_feature = getattr(self, f"plan_enc_bias2_{p}")(traj_feature)
            traj_feature.link_module = self
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
            setattr(self, f"traj_feature_{p}", traj_feature)

            # ── time_mlp ──
            timestep_np = np.full(bs, k, dtype=np.int64)
            timestep_tensor = F._from_data(f"timestep_{p}", timestep_np)
            timestep_tensor.link_module = self
            self._tensors[timestep_tensor.name] = timestep_tensor

            time_embed = getattr(self, f"time_sinusoidal_{p}")(timestep_tensor)
            time_embed = getattr(self, f"time_linear1_{p}")(time_embed)
            time_embed = getattr(self, f"time_bias1_{p}")(time_embed)
            time_embed = getattr(self, f"time_mish_{p}")(time_embed)
            time_embed = getattr(self, f"time_linear2_{p}")(time_embed)
            time_embed = getattr(self, f"time_bias2_{p}")(time_embed)
            time_embed.link_module = self
            time_embed = time_embed.view(bs, 1, -1)
            setattr(self, f"time_embed_{p}", time_embed)

            # ── diff_decoder (per-iteration CustomTransformerDecoder) ──
            diff_decoder = getattr(self, f"diff_decoder_{p}")
            poses_reg_list, poses_cls_list = diff_decoder(
                traj_feature,
                noisy_traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
            )

            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            setattr(self, f"poses_reg_{p}", poses_reg)
            setattr(self, f"poses_cls_{p}", poses_cls)

            # ── x_start → norm_odo → scheduler.step ──
            poses_reg.link_module = self
            self._tensors[poses_reg.name] = poses_reg
            x_start = poses_reg[..., :2]
            x_start.link_module = self
            setattr(self, f"x_start_{p}", x_start)

            norm_odo = getattr(self, f"norm_odo_{p}")
            x_start = norm_odo(x_start)
            setattr(self, f"x_start_norm_{p}", x_start)

            # MockScheduler.step: img = 0.9 * x_start + 0.1 * img
            if x_start.data is not None and img.data is not None:
                step_data = (0.9 * x_start.data + 0.1 * img.data).astype(np.float32)
                img = F._from_data(f"scheduler_step_{p}", step_data)
                img.link_module = self
                self._tensors[img.name] = img
            else:
                mul_model = getattr(self, f"sched_mul_model_{p}")
                mul_sample = getattr(self, f"sched_mul_sample_{p}")
                add_step = getattr(self, f"sched_add_{p}")
                scaled_model = mul_model(x_start)
                scaled_sample = mul_sample(img)
                img = add_step(scaled_model, scaled_sample)
            setattr(self, f"img_{p}", img)

        # ─── 4. Mode selection: argmax + gather ───
        if poses_cls.data is not None and poses_reg.data is not None:
            mode_idx = np.argmax(poses_cls.data, axis=-1)
            best_reg_np = np.zeros((bs, self._num_poses, 3), dtype=np.float32)
            for b in range(bs):
                best_reg_np[b] = poses_reg.data[b, mode_idx[b]]
            best_reg = F._from_data("best_trajectory", best_reg_np)
            self._tensors[best_reg.name] = best_reg
        else:
            poses_reg.link_module = self
            best_reg = poses_reg[:, 0, :, :]

        return {"trajectory": best_reg}

    def _reset_call_counts(self):
        """Reset _call_count on all per-iteration sub-modules."""
        # Per-iteration decoder layers
        for it in range(2):
            p = f"iter{it}"
            decoder = getattr(self, f"diff_decoder_{p}", None)
            if decoder is not None:
                for layer in decoder.layers:
                    layer._call_count = 0
                    if hasattr(layer, "cross_bev_attention"):
                        layer.cross_bev_attention._call_count = 0
                    if hasattr(layer, "time_modulation"):
                        layer.time_modulation._call_count = 0
                    if hasattr(layer, "task_decoder"):
                        layer.task_decoder._call_count = 0
            # Per-iteration GenSineEmbed / SinPosEmb
            gs = getattr(self, f"gen_sineembed_{p}", None)
            if gs is not None:
                gs._call_count = 0
            ts = getattr(self, f"time_sinusoidal_{p}", None)
            if ts is not None:
                ts._call_count = 0


class TransformerDecoderLayer_TTSIM(SimNN_Module):
    """TTSIM implementation of Transformer Decoder Layer."""

    def __init__(self, name, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.nhead = nhead

        # Self attention components (simplified - full attention would need more components)
        self.self_attn_query = F.Linear(
            f"{name}_sa_query", d_model, d_model, module=self
        )
        self.self_attn_query_bias = F.Bias(f"{name}_sa_query_bias", [d_model])
        self.self_attn_key = F.Linear(f"{name}_sa_key", d_model, d_model, module=self)
        self.self_attn_key_bias = F.Bias(f"{name}_sa_key_bias", [d_model])
        self.self_attn_value = F.Linear(
            f"{name}_sa_value", d_model, d_model, module=self
        )
        self.self_attn_value_bias = F.Bias(f"{name}_sa_value_bias", [d_model])
        self.self_attn_out = F.Linear(f"{name}_sa_out", d_model, d_model, module=self)
        self.self_attn_out_bias = F.Bias(f"{name}_sa_out_bias", [d_model])

        # Cross attention components
        self.cross_attn_query = F.Linear(
            f"{name}_ca_query", d_model, d_model, module=self
        )
        self.cross_attn_query_bias = F.Bias(f"{name}_ca_query_bias", [d_model])
        self.cross_attn_key = F.Linear(f"{name}_ca_key", d_model, d_model, module=self)
        self.cross_attn_key_bias = F.Bias(f"{name}_ca_key_bias", [d_model])
        self.cross_attn_value = F.Linear(
            f"{name}_ca_value", d_model, d_model, module=self
        )
        self.cross_attn_value_bias = F.Bias(f"{name}_ca_value_bias", [d_model])
        self.cross_attn_out = F.Linear(f"{name}_ca_out", d_model, d_model, module=self)
        self.cross_attn_out_bias = F.Bias(f"{name}_ca_out_bias", [d_model])

        # Feed-forward network
        self.ffn_linear1 = F.Linear(
            f"{name}_ffn1", d_model, dim_feedforward, module=self
        )
        self.ffn_bias1 = F.Bias(f"{name}_ffn1_bias", [dim_feedforward])
        self.ffn_relu = F.Relu(f"{name}_ffn_relu")
        self.ffn_linear2 = F.Linear(
            f"{name}_ffn2", dim_feedforward, d_model, module=self
        )
        self.ffn_bias2 = F.Bias(f"{name}_ffn2_bias", [d_model])

        # Layer normalization
        self.norm1 = F.LayerNorm(f"{name}_norm1", d_model)
        self.norm2 = F.LayerNorm(f"{name}_norm2", d_model)
        self.norm3 = F.LayerNorm(f"{name}_norm3", d_model)

        # Dropout — separate per usage to avoid shared-op cycles
        self.dropout1 = F.Dropout(f"{name}_dropout1", dropout, False, module=self)
        self.dropout2 = F.Dropout(f"{name}_dropout2", dropout, False, module=self)
        self.dropout3 = F.Dropout(f"{name}_dropout3", dropout, False, module=self)

        # Residual add ops — pre-allocated so graph can discover them
        self.add1 = F.Add(f"{name}_add1")
        self.add2 = F.Add(f"{name}_add2")
        self.add3 = F.Add(f"{name}_add3")

        # Attention operations
        self.head_dim = d_model // nhead
        self.scale = 1.0 / (self.head_dim**0.5)

        # Pre-allocate multihead attention ops for both self_attn and cross_attn
        for attn_type in ["self_attn", "cross_attn"]:
            prefix = f"{name}_{attn_type}"
            setattr(self, f"{attn_type}_reshape_q", F.Reshape(f"{prefix}_reshape_q"))
            setattr(self, f"{attn_type}_reshape_k", F.Reshape(f"{prefix}_reshape_k"))
            setattr(self, f"{attn_type}_reshape_v", F.Reshape(f"{prefix}_reshape_v"))
            setattr(
                self,
                f"{attn_type}_perm_q",
                F.permute(f"{prefix}_permute_q", [0, 2, 1, 3]),
            )
            setattr(
                self,
                f"{attn_type}_perm_k",
                F.permute(f"{prefix}_permute_k", [0, 2, 1, 3]),
            )
            setattr(
                self,
                f"{attn_type}_perm_v",
                F.permute(f"{prefix}_permute_v", [0, 2, 1, 3]),
            )
            setattr(
                self,
                f"{attn_type}_perm_k_t",
                F.permute(f"{prefix}_permute_k_t", [0, 1, 3, 2]),
            )
            setattr(self, f"{attn_type}_matmul_qk", F.MatMul(f"{prefix}_matmul_qk"))
            setattr(self, f"{attn_type}_mul_scale", F.Mul(f"{prefix}_mul_scale"))
            setattr(
                self, f"{attn_type}_softmax", F.Softmax(f"{prefix}_softmax", axis=-1)
            )
            setattr(self, f"{attn_type}_matmul_av", F.MatMul(f"{prefix}_matmul_av"))
            setattr(
                self,
                f"{attn_type}_perm_back",
                F.permute(f"{prefix}_permute_back", [0, 2, 1, 3]),
            )
            setattr(
                self, f"{attn_type}_reshape_out", F.Reshape(f"{prefix}_reshape_out")
            )

        # Link operations to module for .data computation
        super().link_op2module()

    def _multihead_attention(self, query, key, value, prefix, attn_type):
        """Compute multi-head attention using pre-allocated ops."""
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_kv = key.shape[1]

        # Reshape to split heads: (batch, seq, d_model) -> (batch, seq, num_heads, head_dim)
        query.link_module = self
        key.link_module = self
        value.link_module = self

        q_shape = F._from_data(
            f"{prefix}_q_shape",
            np.array(
                [batch_size, seq_len_q, self.nhead, self.head_dim], dtype=np.int64
            ),
            is_const=True,
        )
        self._tensors[q_shape.name] = q_shape
        kv_shape = F._from_data(
            f"{prefix}_kv_shape",
            np.array(
                [batch_size, seq_len_kv, self.nhead, self.head_dim], dtype=np.int64
            ),
            is_const=True,
        )
        self._tensors[kv_shape.name] = kv_shape

        q = getattr(self, f"{attn_type}_reshape_q")(query, q_shape)
        k = getattr(self, f"{attn_type}_reshape_k")(key, kv_shape)
        v = getattr(self, f"{attn_type}_reshape_v")(value, kv_shape)

        # Transpose to (batch, num_heads, seq, head_dim)
        q = getattr(self, f"{attn_type}_perm_q")(q)
        k = getattr(self, f"{attn_type}_perm_k")(k)
        v = getattr(self, f"{attn_type}_perm_v")(v)

        # Transpose key for matmul: (batch, num_heads, head_dim, seq_kv)
        k_t = getattr(self, f"{attn_type}_perm_k_t")(k)

        # Compute attention scores: Q @ K.T
        scores = getattr(self, f"{attn_type}_matmul_qk")(q, k_t)

        # Scale
        scale_const = F._from_data(
            f"{prefix}_scale", np.float32(self.scale), is_const=True
        )
        self._tensors[scale_const.name] = scale_const
        scores = getattr(self, f"{attn_type}_mul_scale")(scores, scale_const)

        # Apply softmax
        attn_weights = getattr(self, f"{attn_type}_softmax")(scores)

        # Compute attention output: scores @ V
        attn = getattr(self, f"{attn_type}_matmul_av")(attn_weights, v)

        # Transpose back: (batch, seq_q, num_heads, head_dim)
        attn = getattr(self, f"{attn_type}_perm_back")(attn)

        # Reshape to (batch, seq_q, d_model)
        attn.link_module = self
        out_shape = F._from_data(
            f"{prefix}_out_shape",
            np.array([batch_size, seq_len_q, self.d_model], dtype=np.int64),
            is_const=True,
        )
        self._tensors[out_shape.name] = out_shape
        attn = getattr(self, f"{attn_type}_reshape_out")(attn, out_shape)

        return attn

    def __call__(self, tgt, memory):
        """Forward pass for transformer decoder layer."""

        # Self attention
        q = self.self_attn_query(tgt)
        q = self.self_attn_query_bias(q)
        k = self.self_attn_key(tgt)
        k = self.self_attn_key_bias(k)
        v = self.self_attn_value(tgt)
        v = self.self_attn_value_bias(v)

        attn_out = self._multihead_attention(
            q, k, v, f"{self.name}_self_attn", "self_attn"
        )
        attn_out = self.self_attn_out(attn_out)
        attn_out = self.self_attn_out_bias(attn_out)
        attn_out = self.dropout1(attn_out)

        tgt = self.add1(tgt, attn_out)
        tgt = self.norm1(tgt)

        # Cross attention
        q2 = self.cross_attn_query(tgt)
        q2 = self.cross_attn_query_bias(q2)
        k2 = self.cross_attn_key(memory)
        k2 = self.cross_attn_key_bias(k2)
        v2 = self.cross_attn_value(memory)
        v2 = self.cross_attn_value_bias(v2)

        attn_out2 = self._multihead_attention(
            q2, k2, v2, f"{self.name}_cross_attn", "cross_attn"
        )
        attn_out2 = self.cross_attn_out(attn_out2)
        attn_out2 = self.cross_attn_out_bias(attn_out2)
        attn_out2 = self.dropout2(attn_out2)

        tgt = self.add2(tgt, attn_out2)
        tgt = self.norm2(tgt)

        # Feed-forward
        ffn_out = self.ffn_linear1(tgt)
        ffn_out = self.ffn_bias1(ffn_out)
        ffn_out = self.ffn_relu(ffn_out)
        ffn_out = self.ffn_linear2(ffn_out)
        ffn_out = self.ffn_bias2(ffn_out)
        ffn_out = self.dropout3(ffn_out)

        tgt = self.add3(tgt, ffn_out)
        tgt = self.norm3(tgt)

        return tgt


# ── Standalone entry point ───────────────────────────────────────────────


def run_standalone(outdir: str = ".") -> None:
    dd_cfgs = {
        "dd_base": {
            "image_architecture": "resnet34",
            "lidar_architecture": "resnet34",
            "camera_height": 256,
            "camera_width": 1024,
            "lidar_h": 256,
            "lidar_w": 256,
            "bs": 1,
        },
    }

    for name, cfg in dd_cfgs.items():
        logger.info("Creating DiffusionDrive(%s)...", name)
        model = V2TransfuserModel(name, cfg)
        model.create_input_tensors()
        logger.info("Input shapes:")
        for k, v in model.input_tensors.items():
            logger.info(" %s: %s", k, v.shape)

        output = model()

        logger.info("Output keys: %s", list(output.keys()))
        for k, v in output.items():
            logger.info(" %s: %s", k, v.shape)

        gg = model.get_forward_graph()
        logger.info("Dumping ONNX...")
        gg.graph2onnx(f"{outdir}/{name}.onnx", do_model_check=True)
        logger.info("%s\n", "-" * 40)


if __name__ == "__main__":
    run_standalone()
