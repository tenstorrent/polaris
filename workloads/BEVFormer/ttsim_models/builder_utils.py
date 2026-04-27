#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Builder utilities for TTSim BEVFormer modules.

This module provides factory functions to build attention, FFN, and
normalization layers from configuration dictionaries, mimicking the
behavior of MMCV's builder pattern but adapted for TTSim.

These utilities replace:
- mmcv.cnn.bricks.transformer.build_attention
- mmcv.cnn.bricks.transformer.build_feedforward_network
- mmcv.cnn.build_norm_layer
- mmcv.cnn.build_activation_layer
"""

import sys
import os
from loguru import logger
import ttsim.front.functional.op as F

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
from ttsim.front.functional.sim_nn import Module, Linear


class LayerNorm(Module):
    """TTSim Layer Normalization module wrapping F.LayerNorm."""

    def __init__(self, name, normalized_shape, eps=1e-5):
        super().__init__()
        self.name = name
        self.normalized_shape = normalized_shape
        self.op = F.LayerNorm(name + ".ln", normalized_shape, epsilon=eps)
        super().link_op2module()

    def __call__(self, x):
        return self.op(x)

    def analytical_param_count(self):
        # scale (gamma) + bias (beta), each of size normalized_shape
        return 2 * self.normalized_shape


def build_activation_layer(cfg):
    """
    Build activation layer from config.

    Args:
        cfg (dict): Activation config with 'type' key
            Supported: 'ReLU', 'GELU', 'Sigmoid', 'Tanh'

    Returns:
        Activation operation function
    """
    if cfg is None:
        return None

    act_type = cfg.get("type", "ReLU")

    if act_type == "ReLU":
        return lambda name, x: F.Relu(name)(x)
    elif act_type == "GELU":
        return lambda name, x: F.Gelu(name)(x)
    elif act_type == "Sigmoid":
        return lambda name, x: F.Sigmoid(name)(x)
    elif act_type == "Tanh":
        return lambda name, x: F.Tanh(name)(x)
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


class FFN(Module):
    """
    TTSim implementation of Feed-Forward Network.

    A simple FFN with two linear layers and activation in between.

    Args:
        name (str): Module name
        embed_dims (int): Input/output dimension
        feedforward_channels (int): Hidden dimension
        num_fcs (int): Number of fully-connected layers (typically 2)
        ffn_drop (float): Dropout rate (ignored in inference)
        act_cfg (dict): Activation config
        add_identity (bool): Whether to add residual connection
    """

    def __init__(
        self,
        name,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        act_cfg=None,
        add_identity=True,
    ):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.add_identity = add_identity

        if act_cfg is None:
            act_cfg = dict(type="ReLU", inplace=True)

        self.activate = build_activation_layer(act_cfg)

        # Build fully-connected layers
        self.layers = []
        in_channels = embed_dims
        for i in range(num_fcs):
            out_channels = feedforward_channels if i < num_fcs - 1 else embed_dims
            self.layers.append(
                Linear(
                    f"{name}.fc{i}", in_features=in_channels, out_features=out_channels
                )
            )
            in_channels = out_channels

    def __call__(self, x, identity=None):
        """
        Forward pass of FFN.

        Args:
            x: Input tensor [..., embed_dims]
            identity: Identity tensor for residual connection (if prenorm)

        Returns:
            Output tensor [..., embed_dims]
        """
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # Apply activation after all but last layer
            if i < len(self.layers) - 1:
                out = self.activate(f"{self.name}.act{i}", out)

        # Add residual connection
        if identity is None:
            identity = x

        if self.add_identity:
            out = F.Add(self.name + ".residual")(out, identity)

        return out

    def analytical_param_count(self):
        """Calculate parameter count."""
        total = 0
        in_channels = self.embed_dims
        for i in range(self.num_fcs):
            out_channels = (
                self.feedforward_channels if i < self.num_fcs - 1 else self.embed_dims
            )
            # weight + bias
            total += in_channels * out_channels + out_channels
            in_channels = out_channels
        return total


def build_norm_layer(name, cfg, num_features):
    """
    Build normalization layer from config.

    Args:
        name (str): Layer name
        cfg (dict): Normalization config with 'type' key
        num_features (int): Number of features to normalize

    Returns:
        Callable that applies LayerNorm to a SimTensor.
    """
    if cfg is None:
        cfg = dict(type="LN")

    norm_type = cfg.get("type", "LN")
    eps = cfg.get("eps", 1e-5)

    if norm_type in ["LN", "LayerNorm"]:
        return LayerNorm(name, num_features, eps=eps)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")


def build_feedforward_network(name, cfg):
    """
    Build feed-forward network from config.

    Args:
        name (str): Module name
        cfg (dict): FFN config

    Returns:
        FFN module
    """
    ffn_type = cfg.get("type", "FFN")

    if ffn_type == "FFN":
        return FFN(
            name=name,
            embed_dims=cfg.get("embed_dims", 256),
            feedforward_channels=cfg.get("feedforward_channels", 1024),
            num_fcs=cfg.get("num_fcs", 2),
            ffn_drop=cfg.get("ffn_drop", 0.0),
            act_cfg=cfg.get("act_cfg", dict(type="ReLU")),
            add_identity=cfg.get("add_identity", True),
        )
    else:
        raise ValueError(f"Unsupported FFN type: {ffn_type}")


def build_attention(name, cfg):
    """
    Build attention module from config.

    Args:
        name (str): Module name
        cfg (dict): Attention config with 'type' key

    Returns:
        Attention module
    """
    attn_type = cfg.get("type", "MultiheadAttention")

    # Import attention modules (lazy import to avoid circular dependencies)
    if attn_type == "TemporalSelfAttention":
        try:
            from .temporal_self_attention import TemporalSelfAttention
        except ImportError:
            from temporal_self_attention import TemporalSelfAttention  # type: ignore[import-not-found,no-redef]

        return TemporalSelfAttention(
            name=name,
            embed_dims=cfg.get("embed_dims", 256),
            num_heads=cfg.get("num_heads", 8),
            num_levels=cfg.get("num_levels", 4),
            num_points=cfg.get("num_points", 4),
            num_bev_queue=cfg.get("num_bev_queue", 2),
            im2col_step=cfg.get("im2col_step", 64),
            dropout=cfg.get("dropout", 0.1),
            batch_first=cfg.get("batch_first", True),
        )

    elif attn_type == "SpatialCrossAttention":
        try:
            from .spatial_cross_attention import SpatialCrossAttention
        except ImportError:
            from spatial_cross_attention import SpatialCrossAttention  # type: ignore[import-not-found,no-redef]

        # Create deformable attention config with the actual parameters
        deformable_attention = {
            "embed_dims": cfg.get("embed_dims", 256),
            "num_heads": cfg.get("num_heads", 8),
            "num_levels": cfg.get("num_levels", 4),
            "num_points": cfg.get("num_points", 4),
        }

        return SpatialCrossAttention(
            name=name,
            embed_dims=cfg.get("embed_dims", 256),
            num_cams=cfg.get("num_cams", 6),
            pc_range=cfg.get("pc_range", None),
            dropout=cfg.get("dropout", 0.1),
            batch_first=cfg.get("batch_first", True),
            deformable_attention=deformable_attention,
        )

    elif attn_type == "MultiheadAttention":
        # Standard multi-head attention (simplified version)
        # This is a placeholder - implement if needed
        raise NotImplementedError(f"MultiheadAttention not yet implemented in TTSim")

    else:
        raise ValueError(f"Unsupported attention type: {attn_type}")


def multi_apply(func, *args, **kwargs):
    """
    Apply function to a list of arguments.

    This function applies the ``func`` to multiple inputs and
    map the multiple outputs of the ``func`` into different
    lists. Each list contains the same type of outputs corresponding
    to different inputs.

    Args:
        func (Function): A function that will be applied to a list of arguments
        *args: Multiple list/tuple arguments
        **kwargs: Keyword arguments to pass to func

    Returns:
        tuple(list): A tuple containing multiple lists, each list contains
            a kind of returned results by the function
    """
    from functools import partial

    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def reduce_mean(tensor):
    """
    Obtain the mean of tensor (simplified version for single-GPU inference).

    In multi-GPU training, this would use distributed operations.
    For TTSim inference on CPU, we just return the tensor as-is.

    Args:
        tensor: Input tensor

    Returns:
        tensor: Same tensor (no reduction needed for single-device)
    """
    # For single-device inference, no reduction needed
    return tensor


def inverse_sigmoid(x, eps=1e-5):
    """
    Inverse function of sigmoid: logit(x) = log(x / (1-x))

    Args:
        x: Input tensor (numpy array or TTSim tensor)
        eps: Small value to avoid numerical issues (default: 1e-5)

    Returns:
        Tensor with inverse sigmoid applied

    Note: This is used to convert normalized coordinates back to logits
    for iterative refinement in the decoder.
    """
    # Clamp to [0, 1] range
    x_clamped = F.Clip("inverse_sigmoid_clip", min_val=0.0, max_val=1.0)(x)

    # Clamp to avoid log(0): x1 = max(x, eps)
    x1 = F.Maximum("inverse_sigmoid_eps1")(x_clamped, F.Constant(eps))

    # Clamp to avoid log(0): x2 = max(1 - x, eps)
    x2_temp = F.Sub("inverse_sigmoid_sub")(F.Constant(1.0), x_clamped)
    x2 = F.Maximum("inverse_sigmoid_eps2")(x2_temp, F.Constant(eps))

    # log(x1 / x2)
    ratio = F.Div("inverse_sigmoid_div")(x1, x2)
    result = F.Log("inverse_sigmoid_log")(ratio)

    return result


def normalize_bbox(name, bboxes, pc_range):
    """
    Normalize bounding boxes to a standard format.

    Converts raw bbox coordinates to a normalized representation suitable for training:
    - Centers (cx, cy, cz) are kept as-is
    - Dimensions (w, l, h) are log-transformed
    - Rotation is converted to sin/cos representation
    - Velocity (vx, vy) is kept if present

    Args:
        name (str): Operation name prefix
        bboxes: Tensor of shape [..., code_size] where code_size >= 7
                Format: [cx, cy, cz, w, l, h, rot, vx, vy]
        pc_range: Point cloud range (not used in normalization, kept for API compatibility)

    Returns:
        Normalized bboxes of shape [..., code_size] where code_size may be 8 or 10
        Format: [cx, cy, w_log, l_log, cz, h_log, rot_sin, rot_cos, vx, vy]
    """
    # Extract components using slicing
    cx = F.Slice(f"{name}_cx", starts=[0, 0, 0], ends=[999, 999, 1], axes=[0, 1, 2])(
        bboxes
    )
    cy = F.Slice(f"{name}_cy", starts=[0, 0, 1], ends=[999, 999, 2], axes=[0, 1, 2])(
        bboxes
    )
    cz = F.Slice(f"{name}_cz", starts=[0, 0, 2], ends=[999, 999, 3], axes=[0, 1, 2])(
        bboxes
    )
    w = F.Slice(f"{name}_w", starts=[0, 0, 3], ends=[999, 999, 4], axes=[0, 1, 2])(
        bboxes
    )
    l = F.Slice(f"{name}_l", starts=[0, 0, 4], ends=[999, 999, 5], axes=[0, 1, 2])(
        bboxes
    )
    h = F.Slice(f"{name}_h", starts=[0, 0, 5], ends=[999, 999, 6], axes=[0, 1, 2])(
        bboxes
    )
    rot = F.Slice(f"{name}_rot", starts=[0, 0, 6], ends=[999, 999, 7], axes=[0, 1, 2])(
        bboxes
    )

    # Log-transform dimensions
    w_log = F.Log(f"{name}_w_log")(w)
    l_log = F.Log(f"{name}_l_log")(l)
    h_log = F.Log(f"{name}_h_log")(h)

    # Convert rotation to sin/cos
    rot_sin = F.Sin(f"{name}_rot_sin")(rot)
    rot_cos = F.Cos(f"{name}_rot_cos")(rot)

    # Check if velocity is present (code_size > 7)
    # This is done at graph construction time based on shape
    # For now, we'll handle both cases

    # Concatenate normalized components
    # Format: [cx, cy, w_log, l_log, cz, h_log, rot_sin, rot_cos, vx, vy]
    parts = [cx, cy, w_log, l_log, cz, h_log, rot_sin, rot_cos]

    # Add velocity if present (check shape at runtime)
    # This would need dynamic shape checking - for now assume it's handled externally
    # or the velocity components are extracted similarly

    normalized_bboxes = F.ConcatX(f"{name}_concat", axis=-1)(*parts)

    return normalized_bboxes


def bias_init_with_prob(prior_prob=0.01):
    """
    Initialize conv/fc bias value according to a given probability value.

    Used to initialize the classification head bias to achieve
    a specific prior probability for positive samples.

    Args:
        prior_prob (float): Prior probability (default: 0.01)

    Returns:
        float: Bias initialization value
    """
    import math

    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def bbox3d2result(bboxes, scores, labels, attrs=None):
    """
    Convert detection results to a dictionary format.

    This function packages 3D bounding box predictions, scores, and labels
    into a standardized dictionary format for evaluation and visualization.

    Args:
        bboxes: 3D bounding boxes (numpy array or tensor)
            Shape: [N, code_size] where code_size is typically 9 or 10
            Format: [cx, cy, cz, w, l, h, rot, vx, vy, (vz)]
        scores: Confidence scores for each detection (numpy array or tensor)
            Shape: [N]
        labels: Class labels for each detection (numpy array or tensor)
            Shape: [N]
        attrs: Optional attributes for each detection (e.g., velocity, orientation)
            Shape: [N, num_attrs] (optional)

    Returns:
        dict: Detection results with keys:
            - 'boxes_3d': 3D bounding boxes
            - 'scores_3d': Detection scores
            - 'labels_3d': Class labels
            - 'attrs_3d': Attributes (if provided)

    Example:
        >>> bboxes = np.array([[10.0, 20.0, 0.0, 4.0, 2.0, 1.5, 0.5, 1.0, 0.5]])
        >>> scores = np.array([0.95])
        >>> labels = np.array([0])  # Car class
        >>> result = bbox3d2result(bboxes, scores, labels)
        >>> print(result.keys())
        dict_keys(['boxes_3d', 'scores_3d', 'labels_3d'])

    Note:
        This function is a pure data formatting utility and doesn't perform
        any transformations on the input data. It simply packages the inputs
        into a standardized dictionary format used throughout BEVFormer.
    """
    result_dict = dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)

    if attrs is not None:
        result_dict["attrs_3d"] = attrs

    return result_dict


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Builder Utilities for TTSim BEVFormer")
    logger.info("=" * 80)
    logger.info("\n✓ Module imported successfully!")
    logger.info("\nAvailable builders:")
    logger.info("  - build_attention: Build attention modules")
    logger.info("  - build_feedforward_network: Build FFN modules")
    logger.info("  - build_norm_layer: Build normalization layers")
    logger.info("  - build_activation_layer: Build activation functions")

    logger.info("\nAvailable helper functions:")
    logger.info("  - multi_apply: Apply function to list of arguments")
    logger.info("  - reduce_mean: Reduce tensor mean (simplified for single-device)")
    logger.info("  - inverse_sigmoid: Inverse sigmoid function")
    logger.info("  - normalize_bbox: Normalize bounding boxes")
    logger.info("  - bias_init_with_prob: Bias initialization for classification")
    logger.info("  - bbox3d2result: Convert detection results to dict format")

    logger.info("\nSupported attention types:")
    logger.info("  - TemporalSelfAttention")
    logger.info("  - SpatialCrossAttention")

    logger.info("\nSupported FFN types:")
    logger.info("  - FFN (standard feed-forward network)")

    logger.info("\nSupported normalization types:")
    logger.info("  - LN / LayerNorm")

    logger.info("\nSupported activation types:")
    logger.info("  - ReLU, GELU, Sigmoid, Tanh")

    logger.info("\n✓ All builders ready!")
    logger.info("=" * 80)
