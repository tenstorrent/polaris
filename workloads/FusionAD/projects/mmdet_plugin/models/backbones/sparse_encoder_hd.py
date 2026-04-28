#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSIM version of SparseEncoderHD.

Converts mmdet3d sparse 3D-convolution backbone from PyTorch to TTSIM.
3D sparse convolutions (SubMConv3d / SparseConv3d) are approximated using
Conv2d on the H x W spatial plane with depth folded into the batch dimension.
Depth reduction for strided convolutions is handled via SliceF.

"""

# =============================================================================
# ORIGINAL TORCH CODE
# =============================================================================
# # Copyright (c) OpenMMLab. All rights reserved.
# from mmcv.runner import auto_fp16
# from torch import nn as nn
#
# from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
# from mmdet3d.ops import spconv as spconv
# from mmdet3d.models.builder import MIDDLE_ENCODERS
# from mmdet.models import BACKBONES
#
# @BACKBONES.register_module()
# class SparseEncoderHD(nn.Module):
#     r"""Sparse encoder for SECOND and Part-A2.
#
#     Args:
#         in_channels (int): The number of input channels.
#         sparse_shape (list[int]): The sparse shape of input tensor.
#         order (list[str]): Order of conv module. Defaults to ('conv',
#             'norm', 'act').
#         norm_cfg (dict): Config of normalization layer. Defaults to
#             dict(type='BN1d', eps=1e-3, momentum=0.01).
#         base_channels (int): Out channels for conv_input layer.
#             Defaults to 16.
#         output_channels (int): Out channels for conv_out layer.
#             Defaults to 128.
#         encoder_channels (tuple[tuple[int]]):
#             Convolutional channels of each encode block.
#         encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
#             Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
#         block_type (str): Type of the block to use. Defaults to 'conv_module'.
#     """
#
#     def __init__(self,
#                  in_channels,
#                  sparse_shape,
#                  order=('conv', 'norm', 'act'),
#                  norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
#                  base_channels=16,
#                  output_channels=128,
#                  encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
#                                                                         64)),
#                  encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
#                                                                  1)),
#                  encoder_strides=(2, 2, 2, 1),
#                  block_type='conv_module',
#                  keep_depth=False,
#                  fp16_enabled=False):
#         super().__init__()
#         assert block_type in ['conv_module', 'basicblock']
#         self.sparse_shape = sparse_shape
#         self.in_channels = in_channels
#         self.order = order
#         self.base_channels = base_channels
#         self.output_channels = output_channels
#         self.encoder_channels = encoder_channels
#         self.encoder_paddings = encoder_paddings
#         self.encoder_strides = encoder_strides
#         self.stage_num = len(self.encoder_channels)
#         self.keep_depth = keep_depth
#         if fp16_enabled:
#             self.fp16_enabled = fp16_enabled
#         # Spconv init all weight on its own
#
#         assert isinstance(order, tuple) and len(order) == 3
#         assert set(order) == {'conv', 'norm', 'act'}
#
#         if self.order[0] != 'conv':  # pre activate
#             self.conv_input = make_sparse_convmodule(
#                 in_channels,
#                 self.base_channels,
#                 3,
#                 norm_cfg=norm_cfg,
#                 padding=1,
#                 indice_key='subm1',
#                 conv_type='SubMConv3d',
#                 order=('conv', ))
#         else:  # post activate
#             self.conv_input = make_sparse_convmodule(
#                 in_channels,
#                 self.base_channels,
#                 3,
#                 norm_cfg=norm_cfg,
#                 padding=1,
#                 indice_key='subm1',
#                 conv_type='SubMConv3d')
#
#         encoder_out_channels = self.make_encoder_layers(
#             make_sparse_convmodule,
#             norm_cfg,
#             self.base_channels,
#             block_type=block_type)
#
#         self.conv_out = make_sparse_convmodule(
#             encoder_out_channels,
#             self.output_channels,
#             kernel_size=(1, 1, 1),
#             stride=(1, 1, 1),
#             norm_cfg=norm_cfg,
#             padding=0,
#             indice_key='spconv_down2',
#             conv_type='SparseConv3d')
#
#     @auto_fp16(apply_to=('voxel_features', ))
#     def forward(self, voxel_features, coors, batch_size):
#         """Forward of SparseEncoder.
#
#         Args:
#             voxel_features (torch.float32): Voxel features in shape (N, C).
#             coors (torch.int32): Coordinates in shape (N, 4), \
#                 the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
#             batch_size (int): Batch size.
#
#         Returns:
#             dict: Backbone features.
#         """
#         coors = coors.int()
#         input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
#                                                   self.sparse_shape,
#                                                   batch_size)
#         x = self.conv_input(input_sp_tensor)
#
#         encode_features = []
#         for encoder_layer in self.encoder_layers:
#             x = encoder_layer(x)
#             encode_features.append(x)
#
#         # for detection head
#         # [200, 176, 5] -> [200, 176, 5]
#         out = self.conv_out(encode_features[-1])
#         spatial_features = out.dense()
#
#         if not self.keep_depth:
#             spatial_features = spatial_features.sum(dim=2)
#
#         return spatial_features
#
#     def make_encoder_layers(self,
#                             make_block,
#                             norm_cfg,
#                             in_channels,
#                             block_type='conv_module',
#                             conv_cfg=dict(type='SubMConv3d')):
#         """make encoder layers using sparse convs.
#
#         Args:
#             make_block (method): A bounded function to build blocks.
#             norm_cfg (dict[str]): Config of normalization layer.
#             in_channels (int): The number of encoder input channels.
#             block_type (str): Type of the block to use. Defaults to
#                 'conv_module'.
#             conv_cfg (dict): Config of conv layer. Defaults to
#                 dict(type='SubMConv3d').
#
#         Returns:
#             int: The number of encoder output channels.
#         """
#         assert block_type in ['conv_module', 'basicblock']
#         self.encoder_layers = spconv.SparseSequential()
#
#         for i, blocks in enumerate(self.encoder_channels):
#             blocks_list = []
#             for j, out_channels in enumerate(tuple(blocks)):
#                 padding = tuple(self.encoder_paddings[i])[j]
#                 # each stage started with a spconv layer
#                 # except the first stage
#                 if i != 0 and j == 0 and block_type == 'conv_module':
#                     blocks_list.append(
#                         make_block(
#                             in_channels,
#                             out_channels,
#                             3,
#                             norm_cfg=norm_cfg,
#                             stride=self.encoder_strides[i],
#                             padding=padding,
#                             indice_key=f'spconv{i + 1}',
#                             conv_type='SparseConv3d'))
#                 elif block_type == 'basicblock':
#                     if j == len(blocks) - 1 and i != len(
#                             self.encoder_channels) - 1:
#                         blocks_list.append(
#                             make_block(
#                                 in_channels,
#                                 out_channels,
#                                 3,
#                                 norm_cfg=norm_cfg,
#                                 stride=self.encoder_strides[i],
#                                 padding=padding,
#                                 indice_key=f'spconv{i + 1}',
#                                 conv_type='SparseConv3d'))
#                     else:
#                         blocks_list.append(
#                             SparseBasicBlock(
#                                 out_channels,
#                                 out_channels,
#                                 norm_cfg=norm_cfg,
#                                 conv_cfg=conv_cfg))
#                 else:
#                     blocks_list.append(
#                         make_block(
#                             in_channels,
#                             out_channels,
#                             3,
#                             norm_cfg=norm_cfg,
#                             padding=padding,
#                             indice_key=f'subm{i + 1}',
#                             conv_type='SubMConv3d'))
#                 in_channels = out_channels
#             stage_name = f'encoder_layer{i + 1}'
#             stage_layers = spconv.SparseSequential(*blocks_list)
#             self.encoder_layers.add_module(stage_name, stage_layers)
#         return out_channels


# =============================================================================
# TTsim code
# =============================================================================
import os
import sys
import functools
import numpy as np

_POLARIS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
sys.path.insert(0, _POLARIS_DIR)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


# ---------------------------------------------------------------------------
# mmcv stub: auto_fp16 decorator (no-op)
# ---------------------------------------------------------------------------
def auto_fp16(apply_to=None, out_fp32=False):
    """No-op replacement for mmcv.runner.auto_fp16."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------
class SparseConvModule(SimNN.Module):
    """Models a single 3D sparse conv module (Conv + BN + ReLU).

    The 3D convolution is approximated as Conv2d on the H x W plane
    with depth folded into the batch dimension.

    Args:
        name (str): Module name.
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size for Conv2d.
        stride (int): Stride for Conv2d (applied to H, W).
        padding (int): Padding for Conv2d.
        order (tuple): Sequence of ('conv', 'norm', 'act').
    """

    def __init__(self, name, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, order=('conv', 'norm', 'act')):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.order = order

        self.conv = F.Conv2d(f'{name}.conv', in_channels, out_channels,
                             kernel_size, stride=stride, padding=padding)
        self.bn = F.BatchNorm2d(f'{name}.bn', out_channels)
        self.relu = F.Relu(f'{name}.relu')

        super().link_op2module()

    def __call__(self, x):
        """Process [B*D, C_in, H, W] -> [B*D, C_out, H', W']."""
        for layer_type in self.order:
            if layer_type == 'conv':
                x = self.conv(x)
                setattr(self, x.name, x)
            elif layer_type == 'norm':
                x = self.bn(x)
                setattr(self, x.name, x)
            elif layer_type == 'act':
                x = self.relu(x)
                setattr(self, x.name, x)
        return x

    def analytical_param_count(self):
        k = self.kernel_size
        params = self.in_channels * self.out_channels * k * k + self.out_channels
        if 'norm' in self.order:
            params += 2 * self.out_channels
        return params


class SparseBasicBlockTTSIM(SimNN.Module):
    """Models SparseBasicBlock: two conv blocks with residual connection.

    Args:
        name (str): Module name.
        channels (int): Number of channels (in == out for residual).
    """

    def __init__(self, name, channels):
        super().__init__()
        self.name = name
        self.channels = channels

        self.conv1 = F.Conv2d(f'{name}.conv1', channels, channels, 3, padding=1)
        self.bn1 = F.BatchNorm2d(f'{name}.bn1', channels)
        self.relu1 = F.Relu(f'{name}.relu1')

        self.conv2 = F.Conv2d(f'{name}.conv2', channels, channels, 3, padding=1)
        self.bn2 = F.BatchNorm2d(f'{name}.bn2', channels)

        self.add = F.Add(f'{name}.add')
        self.relu_out = F.Relu(f'{name}.relu_out')

        super().link_op2module()

    def __call__(self, x):
        """Process [B*D, C, H, W] -> [B*D, C, H, W] with residual."""
        identity = x

        x = self.conv1(x);     setattr(self, x.name, x)
        x = self.bn1(x);       setattr(self, x.name, x)
        x = self.relu1(x);     setattr(self, x.name, x)

        x = self.conv2(x);     setattr(self, x.name, x)
        x = self.bn2(x);       setattr(self, x.name, x)

        x = self.add(x, identity); setattr(self, x.name, x)
        x = self.relu_out(x);      setattr(self, x.name, x)
        return x

    def analytical_param_count(self):
        c = self.channels
        # Two Conv2d(3x3) + two BN
        return 2 * (c * c * 9 + c) + 2 * (2 * c)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class SparseEncoderHD(SimNN.Module):
    r"""TTSIM version of SparseEncoderHD backbone.

    3D sparse convolutions are approximated using Conv2d on the H x W plane.
    Depth is folded into the batch dimension and reduced via SliceF for
    strided layers.

    Args:
        name (str): Module name for ttsim graph.
        in_channels (int): Number of input channels.
        sparse_shape (list[int]): Sparse shape [D, H, W].
        order (tuple[str]): Order of conv module layers.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict): Config of normalization layer (kept for API compat).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        encoder_strides (tuple[int]): Strides of each encode stage.
        block_type (str): Type of the block to use.
            Defaults to 'conv_module'.
        keep_depth (bool): If False, sum along depth after conv_out.
        fp16_enabled (bool): Kept for API compatibility.
    """

    def __init__(self,
                 name,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=None,
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 encoder_strides=(2, 2, 2, 1),
                 block_type='conv_module',
                 keep_depth=False,
                 fp16_enabled=False):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.name = name
        self.sparse_shape = sparse_shape           # [D, H, W]
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.encoder_strides = encoder_strides
        self.stage_num = len(self.encoder_channels)
        self.keep_depth = keep_depth
        if fp16_enabled:
            self.fp16_enabled = fp16_enabled

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        # -- conv_input (SubMConv3d, stride=1) ------------------------------
        if self.order[0] != 'conv':  # pre-activate
            self.conv_input = SparseConvModule(
                f'{name}.conv_input', in_channels, self.base_channels, 3,
                padding=1, order=('conv',))
        else:  # post-activate
            self.conv_input = SparseConvModule(
                f'{name}.conv_input', in_channels, self.base_channels, 3,
                padding=1, order=order)

        # -- encoder layers -------------------------------------------------
        encoder_out_channels = self._make_encoder_layers(
            name, self.base_channels, block_type)

        # -- conv_out (SparseConv3d, kernel 1, stride 1) --------------------
        self.conv_out = SparseConvModule(
            f'{name}.conv_out', encoder_out_channels, self.output_channels,
            kernel_size=1, stride=1, padding=0, order=order)

        # -- Reshape ops for 5D / 4D conversions ----------------------------
        self.reshape_fold_in = F.Reshape(f'{name}.reshape_fold_in')

        # Pre-create reshape ops for each depth-reduction point
        for sid, bid, _ in self._strided_blocks:
            setattr(self, f'reshape_unfold_s{sid}b{bid}',
                    F.Reshape(f'{name}.reshape_unfold_s{sid}b{bid}'))
            setattr(self, f'reshape_fold_s{sid}b{bid}',
                    F.Reshape(f'{name}.reshape_fold_s{sid}b{bid}'))

        # Final reshape from batch-folded to 5D
        self.reshape_to_5d = F.Reshape(f'{name}.reshape_to_5d')

        # Transpose [B, D, C, H, W] -> [B, C, D, H, W]
        self.transpose_5d = F.Transpose(f'{name}.transpose_5d',
                                        perm=[0, 2, 1, 3, 4])

        # ReduceSum for depth collapse  (sum(dim=2))
        self.depth_reduce = F.ReduceSum(f'{name}.depth_reduce',
                                        axes=[2], keepdims=0)

        # Call counter for unique tensor names
        self._call_count = 0

        super().link_op2module()

    # -----------------------------------------------------------------------
    # Encoder layer construction
    # -----------------------------------------------------------------------
    def _make_encoder_layers(self, name, in_channels, block_type):
        """Build encoder layers using sparse conv modules.

        Populates:
            self._all_stage_blocks - list[list[(module, stride)]]
            self._strided_blocks   - list[(stage_idx, block_idx, stride)]

        Returns:
            int: Number of encoder output channels.
        """
        self._all_stage_blocks = []
        self._strided_blocks = []       # (stage_idx, block_idx, stride)

        for i, blocks in enumerate(self.encoder_channels):
            stage_blocks = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # 3D padding (d_pad, h_pad, w_pad) -> extract h_pad for Conv2d
                if isinstance(padding, (tuple, list)):
                    padding = padding[1]

                block_name = f'{name}.enc{i}_blk{j}'
                stride = 1

                if i != 0 and j == 0 and block_type == 'conv_module':
                    # SparseConv3d with stride (first block of stages 1+)
                    stride = self.encoder_strides[i]
                    block: SimNN.Module = SparseConvModule(
                        block_name, in_channels, out_channels, 3,
                        stride=stride, padding=padding, order=self.order)

                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        # Last block of non-last stage with stride
                        stride = self.encoder_strides[i]
                        block = SparseConvModule(
                            block_name, in_channels, out_channels, 3,
                            stride=stride, padding=padding, order=self.order)
                    elif in_channels != out_channels:
                        # Channel transition: use conv module instead of basic block
                        block = SparseConvModule(
                            block_name, in_channels, out_channels, 3,
                            stride=1, padding=padding, order=self.order)
                    else:
                        block = SparseBasicBlockTTSIM(block_name, out_channels)

                else:
                    # SubMConv3d (no stride change)
                    block = SparseConvModule(
                        block_name, in_channels, out_channels, 3,
                        stride=1, padding=padding, order=self.order)

                stage_blocks.append((block, stride))
                setattr(self, f'enc{i}_blk{j}', block)

                if stride > 1:
                    self._strided_blocks.append((i, j, stride))

                in_channels = out_channels

            self._all_stage_blocks.append(stage_blocks)

        return out_channels

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    @auto_fp16(apply_to=('voxel_features', ))
    def __call__(self, voxel_features):
        """Forward of SparseEncoderHD.

        3D sparse convolutions are modelled by folding the depth dimension
        into the batch dimension, running Conv2d on the H x W plane, and
        reducing depth via SliceF for strided stages.

        Args:
            voxel_features: Dense voxel tensor [B, C, D, H, W]
                where D, H, W correspond to sparse_shape.

        Returns:
            spatial_features:
                [B, output_channels, H', W']   if not keep_depth
                [B, output_channels, D', H', W'] if keep_depth
        """
        self._call_count += 1
        cc = self._call_count
        pfx = f'{self.name}_c{cc}'

        B, C, D, H, W = voxel_features.shape

        # ---- fold depth into batch: [B, C, D, H, W] -> [B*D, C, H, W] ----
        fold_in_shape = F._from_data(
            f'{pfx}.fold_in_shape',
            np.array([B * D, C, H, W], dtype=np.int64), is_const=True)
        setattr(self, fold_in_shape.name, fold_in_shape)

        x = self.reshape_fold_in(voxel_features, fold_in_shape)
        setattr(self, x.name, x)

        # ---- conv_input ---------------------------------------------------
        x = self.conv_input(x)
        setattr(self, x.name, x)

        curr_D = D

        # ---- encoder stages -----------------------------------------------
        for i, stage_blocks in enumerate(self._all_stage_blocks):
            for j, (block, stride) in enumerate(stage_blocks):
                x = block(x)
                setattr(self, x.name, x)

                if stride > 1:
                    # H, W already reduced by Conv2d stride.
                    # Now handle depth reduction.
                    _, C_curr, H_curr, W_curr = x.shape

                    # Unfold: [B*D_old, C, H', W'] -> [B, D_old, C, H', W']
                    unfold_shape = F._from_data(
                        f'{pfx}.unfold_s{i}b{j}',
                        np.array([B, curr_D, C_curr, H_curr, W_curr],
                                 dtype=np.int64), is_const=True)
                    setattr(self, unfold_shape.name, unfold_shape)

                    reshape_unfold = getattr(self, f'reshape_unfold_s{i}b{j}')
                    x = reshape_unfold(x, unfold_shape)
                    setattr(self, x.name, x)

                    # Depth slice with step=stride along axis=1
                    new_D = -(-curr_D // stride)  # ceiling division
                    slice_out = [B, new_D, C_curr, H_curr, W_curr]

                    starts_t = F._from_data(
                        f'{pfx}.ds_starts_s{i}',
                        np.array([0], dtype=np.int64), is_const=True)
                    ends_t = F._from_data(
                        f'{pfx}.ds_ends_s{i}',
                        np.array([curr_D], dtype=np.int64), is_const=True)
                    axes_t = F._from_data(
                        f'{pfx}.ds_axes_s{i}',
                        np.array([1], dtype=np.int64), is_const=True)
                    steps_t = F._from_data(
                        f'{pfx}.ds_steps_s{i}',
                        np.array([stride], dtype=np.int64), is_const=True)
                    for t in (starts_t, ends_t, axes_t, steps_t):
                        setattr(self, t.name, t)

                    depth_slice = F.SliceF(
                        f'{pfx}.depth_slice_s{i}', out_shape=slice_out)
                    setattr(self, f'{pfx}_depth_slice_s{i}', depth_slice)
                    x = depth_slice(x, starts_t, ends_t, axes_t, steps_t)
                    setattr(self, x.name, x)

                    curr_D = new_D

                    # Fold back: [B, D_new, C, H', W'] -> [B*D_new, C, H', W']
                    fold_shape = F._from_data(
                        f'{pfx}.fold_s{i}b{j}',
                        np.array([B * new_D, C_curr, H_curr, W_curr],
                                 dtype=np.int64), is_const=True)
                    setattr(self, fold_shape.name, fold_shape)

                    reshape_fold = getattr(self, f'reshape_fold_s{i}b{j}')
                    x = reshape_fold(x, fold_shape)
                    setattr(self, x.name, x)

        # ---- conv_out -----------------------------------------------------
        x = self.conv_out(x)
        setattr(self, x.name, x)

        _, C_out, H_out, W_out = x.shape

        # ---- reshape to 5D: [B*D', C, H, W] -> [B, D', C, H, W] ---------
        to_5d_shape = F._from_data(
            f'{pfx}.to_5d_shape',
            np.array([B, curr_D, C_out, H_out, W_out], dtype=np.int64),
            is_const=True)
        setattr(self, to_5d_shape.name, to_5d_shape)

        x = self.reshape_to_5d(x, to_5d_shape)
        setattr(self, x.name, x)

        # ---- transpose: [B, D', C, H, W] -> [B, C, D', H, W] ------------
        x = self.transpose_5d(x)
        setattr(self, x.name, x)

        # ---- collapse depth -----------------------------------------------
        if not self.keep_depth:
            x = self.depth_reduce(x)            # -> [B, C_out, H', W']
            setattr(self, x.name, x)

        return x

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    def analytical_param_count(self):
        """Return total trainable parameter count."""
        total = self.conv_input.analytical_param_count()
        for stage_blocks in self._all_stage_blocks:
            for block, _ in stage_blocks:
                if hasattr(block, 'analytical_param_count'):
                    total += block.analytical_param_count()
        total += self.conv_out.analytical_param_count()
        return total
