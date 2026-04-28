#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Builder utilities for TTSim modules.

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

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.sim_nn import Module, Linear


def build_activation_layer(cfg):
    """
    Build activation layer from config.
    
    Args:
        cfg (dict): Activation config with 'type' key
            Supported: 'ReLU', 'GELU', 'Sigmoid', 'Tanh'
    
    Returns:
        Activation op constructor function(name) -> SimOpHandle
    """
    if cfg is None:
        return None
    
    act_type = cfg.get('type', 'ReLU')
    
    if act_type == 'ReLU':
        return F.Relu
    elif act_type == 'GELU':
        return F.Gelu
    elif act_type == 'Sigmoid':
        return F.Sigmoid
    elif act_type == 'Tanh':
        return F.Tanh
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


class LayerNorm(Module):
    """
    TTSim implementation of Layer Normalization.
    
    Args:
        name (str): Module name
        normalized_shape (int or tuple): Input shape to normalize
        eps (float): Epsilon for numerical stability. Default: 1e-5
    """
    
    def __init__(self, name, normalized_shape, eps=1e-5):
        super().__init__()
        self.name = name
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Pre-create all ops in __init__ 
        self.mean_op = F.SimOpHandle(name + '.mean', 'ReduceMean', params=[], ipos=[0], keepdims=1)
        self.center_op = F.Sub(name + '.center')
        self.square_op = F.Mul(name + '.square')
        self.var_op = F.SimOpHandle(name + '.var', 'ReduceMean', params=[], ipos=[0], keepdims=1)
        self.var_eps_add = F.Add(name + '.var_eps')
        self.std_op = F.Sqrt(name + '.std')
        self.normalize_op = F.Div(name + '.normalize')
        
        # Pre-create constant tensors (scalar so it broadcasts to any rank)
        self.eps_tensor = F._from_data(name + '.eps',
            np.array([eps], dtype=np.float32), is_const=True)
        
        super().link_op2module()
    
    def __call__(self, x):
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor [..., normalized_shape]
        
        Returns:
            Normalized tensor
        """
        # Get the axes to normalize over (last len(normalized_shape) dimensions)
        ndim = len(x.shape)
        axes_to_normalize = list(range(ndim - len(self.normalized_shape), ndim))
        
        # Use self.xxx so __setattr__ tracks this SimTensor
        self.axes_tensor = F._from_data(
            self.name + '.axes',
            np.array(axes_to_normalize, dtype=np.int64),
            is_const=True)
        
        # Set axes as params on the pre-created ops (they need the axes tensor)
        self.mean_op.params = [(1, self.axes_tensor)]
        self.mean_op.implicit_inputs = [self.axes_tensor]
        self.var_op.params = [(1, self.axes_tensor)]
        self.var_op.implicit_inputs = [self.axes_tensor]
        
        mean = self.mean_op(x)
        x_centered = self.center_op(x, mean)
        x_squared = self.square_op(x_centered, x_centered)
        variance = self.var_op(x_squared)
        variance_eps = self.var_eps_add(variance, self.eps_tensor)
        std = self.std_op(variance_eps)
        normalized = self.normalize_op(x_centered, std)
        
        return normalized
    
    def analytical_param_count(self, lvl=0):
        """
        Calculate parameter count.
        In PyTorch LayerNorm, we have weight and bias parameters.
        
        Args:
            lvl (int): Verbosity level (unused, for API compatibility)
        """
        # weight + bias for normalized_shape
        num_elements = 1
        for dim in self.normalized_shape:
            num_elements *= dim
        return 2 * num_elements


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
    
    def __init__(self,
                 name,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 ffn_drop=0.,
                 act_cfg=None,
                 add_identity=True):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.add_identity = add_identity
        
        if act_cfg is None:
            act_cfg = dict(type='ReLU', inplace=True)
        
        self.activate = build_activation_layer(act_cfg)
        
        # Build fully-connected layers using ModuleList (tracked by __setattr__)
        fc_list = []
        in_channels = embed_dims
        for i in range(num_fcs):
            out_channels = feedforward_channels if i < num_fcs - 1 else embed_dims
            fc_list.append(Linear(f"{name}.fc{i}", in_features=in_channels, out_features=out_channels))
            in_channels = out_channels
        self.layers = SimNN.ModuleList(fc_list)
        
        # Pre-create activation ops for each intermediate layer
        for i in range(num_fcs - 1):
            act_name = f"{name}.act{i}"
            setattr(self, f'act_{i}', self.activate(act_name))
        
        # Residual add op
        self.residual_add = F.Add(name + '.residual')
    
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
            if i < len(self.layers) - 1:
                out = getattr(self, f'act_{i}')(out)
        
        if identity is None:
            identity = x
        
        if self.add_identity:
            out = self.residual_add(out, identity)
        
        return out
    
    def analytical_param_count(self, lvl=0):
        """
        Calculate parameter count.
        
        Args:
            lvl (int): Verbosity level (unused, for API compatibility)
        """
        total = 0
        in_channels = self.embed_dims
        for i in range(self.num_fcs):
            out_channels = self.feedforward_channels if i < self.num_fcs - 1 else self.embed_dims
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
        Normalization layer module
    """
    if cfg is None:
        cfg = dict(type='LN')
    
    norm_type = cfg.get('type', 'LN')
    eps = cfg.get('eps', 1e-5)
    
    if norm_type in ['LN', 'LayerNorm']:
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
    ffn_type = cfg.get('type', 'FFN')
    
    if ffn_type == 'FFN':
        return FFN(
            name=name,
            embed_dims=cfg.get('embed_dims', 256),
            feedforward_channels=cfg.get('feedforward_channels', 1024),
            num_fcs=cfg.get('num_fcs', 2),
            ffn_drop=cfg.get('ffn_drop', 0.),
            act_cfg=cfg.get('act_cfg', dict(type='ReLU')),
            add_identity=cfg.get('add_identity', True)
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
    attn_type = cfg.get('type', 'MultiheadAttention')
    
    # Import attention modules (lazy import to avoid circular dependencies)
    if attn_type == 'TemporalSelfAttention':
        from .temporal_self_attention import TemporalSelfAttention
        
        return TemporalSelfAttention(
            name=name,
            embed_dims=cfg.get('embed_dims', 256),
            num_heads=cfg.get('num_heads', 8),
            num_levels=cfg.get('num_levels', 4),
            num_points=cfg.get('num_points', 4),
            num_bev_queue=cfg.get('num_bev_queue', 2),
            im2col_step=cfg.get('im2col_step', 64),
            dropout=cfg.get('dropout', 0.1),
            batch_first=cfg.get('batch_first', True)
        )
    
    elif attn_type == 'SpatialCrossAttention':
        from .spatial_cross_attention import SpatialCrossAttention
        
        # Use the nested deformable_attention config if provided,
        # otherwise fall back to parent-level keys for backward compat.
        deformable_cfg = cfg.get('deformable_attention', {})
        deformable_attention = {
            'embed_dims': deformable_cfg.get('embed_dims', cfg.get('embed_dims', 256)),
            'num_heads': deformable_cfg.get('num_heads', cfg.get('num_heads', 8)),
            'num_levels': deformable_cfg.get('num_levels', cfg.get('num_levels', 4)),
            'num_points': deformable_cfg.get('num_points', cfg.get('num_points', 4))
        }
        
        return SpatialCrossAttention(
            name=name,
            embed_dims=cfg.get('embed_dims', 256),
            num_cams=cfg.get('num_cams', 6),
            pc_range=cfg.get('pc_range', None),
            dropout=cfg.get('dropout', 0.1),
            batch_first=cfg.get('batch_first', True),
            deformable_attention=deformable_attention
        )
    
    elif attn_type == 'PtsCrossAttention':
        from .pts_cross_attention import PtsCrossAttention

        return PtsCrossAttention(
            name=name,
            embed_dims=cfg.get('embed_dims', 256),
            num_heads=cfg.get('num_heads', 8),
            num_levels=cfg.get('num_levels', 1),
            num_points=cfg.get('num_points', 4),
            im2col_step=cfg.get('im2col_step', 64),
            dropout=cfg.get('dropout', 0.1),
            batch_first=cfg.get('batch_first', True)
        )

    elif attn_type == 'MultiheadAttention':
        from .multihead_attention import MultiheadAttention

        return MultiheadAttention(
            name=name,
            embed_dims=cfg.get('embed_dims', 256),
            num_heads=cfg.get('num_heads', 8),
            attn_drop=cfg.get('dropout', cfg.get('attn_drop', 0.0)),
            proj_drop=cfg.get('proj_drop', 0.0),
            batch_first=cfg.get('batch_first', False),
            bias=cfg.get('bias', True)
        )

    elif attn_type in ('CustomMSDeformableAttention',
                       'MultiScaleDeformableAttention'):
        from .decoder import CustomMSDeformableAttention

        return CustomMSDeformableAttention(
            name=name,
            embed_dims=cfg.get('embed_dims', 256),
            num_heads=cfg.get('num_heads', 8),
            num_levels=cfg.get('num_levels', 4),
            num_points=cfg.get('num_points', 4),
            im2col_step=cfg.get('im2col_step', 64),
            dropout=cfg.get('dropout', 0.1),
            batch_first=cfg.get('batch_first', False)
        )

    else:
        raise ValueError(f"Unsupported attention type: {attn_type}")


# ======================================================================
# Utility: inverse_sigmoid
# ======================================================================

def inverse_sigmoid_np(x, eps=1e-5):
    """
    Numpy implementation of inverse sigmoid (logit function).

    Used during reference-point refinement in the decoder.

    Args:
        x (np.ndarray): Input array, values should be in [0, 1].
        eps (float): Epsilon for numerical stability. Default: 1e-5.

    Returns:
        np.ndarray: log(x / (1 - x)), clamped for stability.
    """
    x = np.clip(x, 0.0, 1.0)
    x1 = np.clip(x, eps, None)
    x2 = np.clip(1.0 - x, eps, None)
    return np.log(x1 / x2)


class InverseSigmoid(Module):
    """
    TTSim implementation of inverse_sigmoid(x) = log(x / (1 - x)).

    Clamped for numerical stability: x clamped to [0, 1], then
    x1 = clamp(x, min=eps), x2 = clamp(1-x, min=eps), result = log(x1/x2).

    Args:
        name (str): Module name.
        eps (float): Epsilon for numerical stability. Default: 1e-5.
    """

    def __init__(self, name, eps=1e-5):
        super().__init__()
        self.name = name
        self.eps = eps

        # Clamp x to [0, 1] — ONNX Clip opset 13 requires min/max as tensor inputs
        self._clip01_min = F._from_data(name + '.clip_01.min', np.array([0.0], dtype=np.float32), is_const=True)
        self._clip01_max = F._from_data(name + '.clip_01.max', np.array([1.0], dtype=np.float32), is_const=True)
        self.clip_01 = F.SimOpHandle(name + '.clip_01', 'Clip',
                                     params=[(1, self._clip01_min), (2, self._clip01_max)],
                                     ipos=[0])

        # Clamp x1 = clamp(x, min=eps)
        self._clipx1_min = F._from_data(name + '.clip_x1.min', np.array([eps], dtype=np.float32), is_const=True)
        self.clip_x1 = F.SimOpHandle(name + '.clip_x1', 'Clip',
                                     params=[(1, self._clipx1_min)],
                                     ipos=[0])

        # 1 - x
        self.one_const = F._from_data(name + '.one',
                                      np.array([1.0], dtype=np.float32),
                                      is_const=True)
        self.sub_one = F.Sub(name + '.sub_one')

        # Clamp x2 = clamp(1-x, min=eps)
        self._clipx2_min = F._from_data(name + '.clip_x2.min', np.array([eps], dtype=np.float32), is_const=True)
        self.clip_x2 = F.SimOpHandle(name + '.clip_x2', 'Clip',
                                     params=[(1, self._clipx2_min)],
                                     ipos=[0])

        # x1 / x2
        self.div_op = F.Div(name + '.div')

        # log(x1 / x2)
        self.log_op = F.Log(name + '.log')

    def __call__(self, x):
        """
        Compute inverse_sigmoid(x).

        Args:
            x: Input tensor with values approximately in [0, 1].

        Returns:
            Tensor: log(clamp(x, eps) / clamp(1-x, eps))
        """
        x_clamped = self.clip_01(x)
        setattr(self, x_clamped.name, x_clamped)

        x1 = self.clip_x1(x_clamped)
        setattr(self, x1.name, x1)

        x2_pre = self.sub_one(self.one_const, x_clamped)
        setattr(self, x2_pre.name, x2_pre)

        x2 = self.clip_x2(x2_pre)
        setattr(self, x2.name, x2)

        ratio = self.div_op(x1, x2)
        setattr(self, ratio.name, ratio)

        result = self.log_op(ratio)
        setattr(self, result.name, result)

        return result


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Builder Utilities for TTSim BEVFormer")
    logger.info("=" * 80)
    logger.info("\n[OK] Module imported successfully!")
    logger.info("\nAvailable builders:")
    logger.info("  - build_attention: Build attention modules")
    logger.info("  - build_feedforward_network: Build FFN modules")
    logger.info("  - build_norm_layer: Build normalization layers")
    logger.info("  - build_activation_layer: Build activation functions")
    
    logger.info("\nSupported attention types:")
    logger.info("  - TemporalSelfAttention")
    logger.info("  - SpatialCrossAttention")
    logger.info("  - PtsCrossAttention")
    logger.info("  - MultiheadAttention")
    logger.info("  - CustomMSDeformableAttention")
    
    logger.info("\nSupported FFN types:")
    logger.info("  - FFN (standard feed-forward network)")
    
    logger.info("\nSupported normalization types:")
    logger.info("  - LN / LayerNorm")
    
    logger.info("\nSupported activation types:")
    logger.info("  - ReLU, GELU, Sigmoid, Tanh")
    
    logger.info("\n[OK] All builders ready!")
    logger.info("=" * 80)
