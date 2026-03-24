#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim implementation of MapSegHead - semantic segmentation head for BEV features
"""

# -------------------------------PyTorch--------------------------------

# import copy
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob, xavier_init
# from mmcv.runner import force_fp32
# from mmcv.cnn.bricks.transformer import build_positional_encoding
# from mmdet.models import build_loss
#
# from mmdet.models import HEADS
#
# from einops import repeat
#
#
# @HEADS.register_module(force=True)
# class MapSegHead(nn.Module):
#
#     def __init__(self,
#                  num_classes=3,
#                  in_channels=256,
#                  embed_dims=256,
#                  bev_size=(100,50),
#                  canvas_size=(200,100),
#                  loss_seg=dict(),
#                  loss_dice=dict(),
#         ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.in_channels = in_channels
#         self.embed_dims = embed_dims
#         self.bev_size = bev_size
#         self.canvas_size = canvas_size
#
#         self.loss_seg = build_loss(loss_seg)
#         self.loss_dice = build_loss(loss_dice)
#
#         if self.loss_seg.use_sigmoid:
#             self.cls_out_channels = num_classes
#         else:
#             self.cls_out_channels = num_classes + 1
#
#         assert canvas_size[0] % bev_size[0] == 0, 'canvas size must be a multiple of the bev size'
#         self.num_up_blocks = int(np.log2(canvas_size[0] // bev_size[0]))
#
#         self.conv_in = nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv_mid_layers = nn.ModuleList([])
#         self.downsample_layers = nn.ModuleList([])
#         for _ in range(self.num_up_blocks):
#             conv_mid = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='nearest'),
#                 nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#             )
#             self.conv_mid_layers.append(conv_mid)
#             self.downsample_layers.append(nn.Upsample(scale_factor=0.5, mode='bilinear'))
#
#         self.conv_out = nn.Conv2d(embed_dims, self.cls_out_channels, kernel_size=1, padding=0)
#
#
#         self.init_weights()
#
#     def init_weights(self):
#         if self.loss_seg.use_sigmoid:
#             bias_init = bias_init_with_prob(0.01)
#             m = self.conv_out
#             nn.init.constant_(m.bias, bias_init)
#
#     def forward_train(self, bev_features, gts, history_coords):
#         x = self.relu(self.conv_in(bev_features))
#         for conv_mid in self.conv_mid_layers:
#             x = conv_mid(x)
#         preds = self.conv_out(x)
#
#         seg_loss = self.loss_seg(preds, gts)
#         dice_loss = self.loss_dice(preds, gts)
#
#         # downsample the features to the original bev size
#         seg_feats = x
#         for downsample in self.downsample_layers:
#             seg_feats = downsample(seg_feats)
#
#         return preds, seg_feats, seg_loss, dice_loss
#
#     def forward_test(self, bev_features):
#         x = self.relu(self.conv_in(bev_features))
#         for conv_mid in self.conv_mid_layers:
#             x = conv_mid(x)
#         preds = self.conv_out(x)
#         seg_feats = x
#         for downsample in self.downsample_layers:
#             seg_feats = downsample(seg_feats)
#         return preds, seg_feats
#
#     def train(self, *args, **kwargs):
#         super().train(*args, **kwargs)
#
#     def eval(self):
#         super().eval()
#
#     def forward(self, *args, return_loss=True, **kwargs):
#         if return_loss:
#             return self.forward_train(*args, **kwargs)
#         else:
#             return self.forward_test(*args, **kwargs)

# -------------------------------TTSIM-----------------------------------

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class MapSegHead(SimNN.Module):
    """
    Semantic segmentation head for BEV map features.
    Upsamples BEV features (e.g., 100x50) to canvas size (e.g., 200x100)
    and outputs per-pixel class predictions.

    Args:
        name: Module name
        num_classes: Number of segmentation classes (default: 3)
        in_channels: Input feature channels (default: 256)
        embed_dims: Hidden layer channels (default: 256)
        bev_size: BEV feature map size (H, W) (default: (100, 50))
        canvas_size: Output canvas size (H, W) (default: (200, 100))
        max_batch_size: Maximum batch size for dynamic shapes
    """

    def __init__(
        self,
        name,
        num_classes=3,
        in_channels=256,
        embed_dims=256,
        bev_size=(100, 50),
        canvas_size=(200, 100),
        max_batch_size=4,
    ):
        super().__init__()

        self.name = name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.bev_size = bev_size  # (H, W)
        self.canvas_size = canvas_size  # (H, W)
        self.max_batch_size = max_batch_size

        # Calculate number of upsampling blocks needed
        # canvas_size must be power-of-2 multiple of bev_size
        assert (
            canvas_size[0] % bev_size[0] == 0
        ), "canvas size must be a multiple of bev size"
        assert (
            canvas_size[1] % bev_size[1] == 0
        ), "canvas size must be a multiple of bev size"
        scale_h = canvas_size[0] // bev_size[0]
        scale_w = canvas_size[1] // bev_size[1]
        assert scale_h == scale_w, "scale factor must be the same for H and W"
        assert (
            scale_h > 0 and (scale_h & (scale_h - 1)) == 0
        ), "scale must be power of 2"

        self.num_up_blocks = int(np.log2(scale_h))
        self.cls_out_channels = num_classes  # Using sigmoid activation

        # Initial convolution (no bias)
        self.conv_in = F.Conv2d(
            f"{name}.conv_in", in_channels, embed_dims, kernel_size=3, padding=1
        )

        self.relu = F.Relu(f"{name}.relu")

        # Upsampling blocks (each 2x upsampling)
        # Each block: Upsample -> Conv3x3 -> bias_reshape -> bias_add -> ReLU
        # Use SimOpHandleList for op collections (tracked by __setattr__)
        upsample_list = []
        conv_up_list = []
        bias_reshape_list = []
        bias_add_list = []
        relu_up_list = []
        downsample_list = []

        current_h, current_w = bev_size
        for i in range(self.num_up_blocks):
            conv_name = f"{name}.conv_up_{i}"

            upsample_list.append(F.Resize(f"{name}.upsample_{i}", scale_factor=2.0))
            current_h, current_w = current_h * 2, current_w * 2

            conv_up_list.append(
                F.Conv2d(conv_name, embed_dims, embed_dims, kernel_size=3, padding=1)
            )

            # Bias tensors stored as self.xxx so __setattr__ tracks them
            setattr(
                self,
                f"conv_up_bias_{i}",
                F._from_shape(f"{conv_name}.bias_param", [embed_dims], is_param=True),
            )
            setattr(
                self,
                f"conv_up_bias_shape_{i}",
                F._from_data(
                    f"{conv_name}.bias_shape",
                    np.array([1, embed_dims, 1, 1], dtype=np.int64),
                    is_const=True,
                ),
            )

            bias_reshape_list.append(F.Reshape(f"{conv_name}.bias_reshape"))
            bias_add_list.append(F.Add(f"{conv_name}.add"))
            relu_up_list.append(F.Relu(f"{name}.relu_up_{i}"))

        self.upsample_ops = F.SimOpHandleList(upsample_list)
        self.conv_ups = F.SimOpHandleList(conv_up_list)
        self.conv_up_bias_reshapes = F.SimOpHandleList(bias_reshape_list)
        self.conv_up_adds = F.SimOpHandleList(bias_add_list)
        self.relu_ups = F.SimOpHandleList(relu_up_list)

        # Output convolution (1x1, weight only)
        self.conv_out = F.Conv2d(
            f"{name}.conv_out",
            embed_dims,
            self.cls_out_channels,
            kernel_size=1,
            padding=0,
        )

        # Output conv bias (separate parameter)
        self.conv_out_bias = F._from_shape(
            f"{name}.conv_out.bias_param", [self.cls_out_channels], is_param=True
        )
        self.conv_out_bias_shape = F._from_data(
            f"{name}.conv_out.bias_shape",
            np.array([1, self.cls_out_channels, 1, 1], dtype=np.int64),
            is_const=True,
        )
        self.conv_out_bias_reshape = F.Reshape(f"{name}.conv_out.bias_reshape")
        self.conv_out_add = F.Add(f"{name}.conv_out.add")

        # Downsample ops for returning features at original BEV size
        # Use linear + align_corners to match PyTorch nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        current_h, current_w = canvas_size
        for i in range(self.num_up_blocks):
            downsample_list.append(
                F.Resize(
                    f"{name}.downsample_{i}",
                    scale_factor=0.5,
                    mode="linear",
                    coordinate_transformation_mode="align_corners",
                )
            )
            current_h, current_w = current_h // 2, current_w // 2

        self.downsample_ops = F.SimOpHandleList(downsample_list)

        super().link_op2module()

    def __call__(self, bev_features):
        """
        Forward pass for inference.

        Args:
            bev_features: BEV feature tensor [bs, C, bev_h, bev_w]

        Returns:
            preds: Segmentation predictions [bs, num_classes, canvas_h, canvas_w]
            seg_feats: Downsampled features [bs, embed_dims, bev_h, bev_w]
        """
        # Initial conv + ReLU (no bias for conv_in)
        x = self.relu(self.conv_in(bev_features))

        # Progressive upsampling
        for i in range(self.num_up_blocks):
            x = self.upsample_ops[i](x)
            x = self.conv_ups[i](x)
            # Add bias: reshape [embed_dims] -> [1, embed_dims, 1, 1] and add
            bias = getattr(self, f"conv_up_bias_{i}")
            bias_shape = getattr(self, f"conv_up_bias_shape_{i}")
            bias_reshaped = self.conv_up_bias_reshapes[i](bias, bias_shape)
            x = self.conv_up_adds[i](x, bias_reshaped)
            x = self.relu_ups[i](x)

        # Final output conv + bias
        preds = self.conv_out(x)
        bias_reshaped = self.conv_out_bias_reshape(
            self.conv_out_bias, self.conv_out_bias_shape
        )
        preds = self.conv_out_add(preds, bias_reshaped)

        # Downsample back to original BEV size for seg_feats
        seg_feats = x
        for i in range(self.num_up_blocks):
            seg_feats = self.downsample_ops[i](seg_feats)

        return preds, seg_feats

    def analytical_param_count(self, lvl=0):
        """Calculate total number of trainable parameters.

        Args:
            lvl: Verbosity level (0=silent, 1=summary, 2=detailed)

        Returns:
            int: Total parameter count
        """
        indent = "  " * lvl

        # conv_in: Conv2d(in_channels, embed_dims, 3x3), no bias
        conv_in_params = self.in_channels * self.embed_dims * 3 * 3

        # Upsample conv blocks: Conv2d(embed_dims, embed_dims, 3x3) + bias
        conv_up_params = 0
        for i in range(self.num_up_blocks):
            w = self.embed_dims * self.embed_dims * 3 * 3
            b = self.embed_dims
            conv_up_params += w + b

        # conv_out: Conv2d(embed_dims, num_classes, 1x1) + bias
        conv_out_params = (
            self.embed_dims * self.cls_out_channels * 1 * 1 + self.cls_out_channels
        )

        total = conv_in_params + conv_up_params + conv_out_params

        if lvl >= 2:
            print(f"{indent}MapSegHead '{self.name}':")
            print(f"{indent}  conv_in (3x3, no bias): {conv_in_params:,}")
            for i in range(self.num_up_blocks):
                w = self.embed_dims * self.embed_dims * 3 * 3
                b = self.embed_dims
                print(f"{indent}  conv_up_{i} (3x3 + bias): {w + b:,}")
            print(f"{indent}  conv_out (1x1 + bias): {conv_out_params:,}")

        if lvl >= 1:
            print(f"{indent}Total MapSegHead params: {total:,}")

        return total


if __name__ == "__main__":
    """Test MapSegHead construction and forward pass"""
    print("\n" + "=" * 80)
    print("MapSegHead TTSim Implementation Test")
    print("=" * 80)

    # Test parameters (small config for faster testing)
    bs = 1
    in_channels = 64
    embed_dims = 64
    num_classes = 3
    bev_h, bev_w = 50, 25
    canvas_h, canvas_w = 100, 50

    print(f"\nTest configuration:")
    print(f"  Batch size: {bs}")
    print(f"  Input: {in_channels} channels, {bev_h}x{bev_w}")
    print(f"  Output: {num_classes} classes, {canvas_h}x{canvas_w}")
    print(f"  Upsampling blocks: {int(np.log2(canvas_h // bev_h))}")

    # Create model
    model = MapSegHead(
        name="test_seg_head",
        num_classes=num_classes,
        in_channels=in_channels,
        embed_dims=embed_dims,
        bev_size=(bev_h, bev_w),
        canvas_size=(canvas_h, canvas_w),
        max_batch_size=4,
    )

    print(f"\n[OK] Model constructed successfully")

    # Create dummy input
    np.random.seed(42)
    bev_features_np = np.random.randn(bs, in_channels, bev_h, bev_w).astype(np.float32)
    bev_features = F._from_data("bev_features", bev_features_np, is_const=True)

    print(f"\n[OK] Input created: {bev_features.shape}")

    # Forward pass
    try:
        preds, seg_feats = model(bev_features)

        print(f"\n[OK] Forward pass successful!")
        print(
            f"  Predictions shape: {preds.shape} (expected: [{bs}, {num_classes}, {canvas_h}, {canvas_w}])"
        )
        print(
            f"  Seg features shape: {seg_feats.shape} (expected: [{bs}, {embed_dims}, {bev_h}, {bev_w}])"
        )

        # Verify shapes
        expected_preds_shape = [bs, num_classes, canvas_h, canvas_w]
        expected_feats_shape = [bs, embed_dims, bev_h, bev_w]

        assert (
            list(preds.shape) == expected_preds_shape
        ), f"Pred shape mismatch: {preds.shape} vs {expected_preds_shape}"
        assert (
            list(seg_feats.shape) == expected_feats_shape
        ), f"Feat shape mismatch: {seg_feats.shape} vs {expected_feats_shape}"

        print(f"\n[PASS] All tests passed!")

    except Exception as e:
        print(f"\n[FAIL] Forward pass failed: {str(e)}")
        import traceback

        traceback.print_exc()
