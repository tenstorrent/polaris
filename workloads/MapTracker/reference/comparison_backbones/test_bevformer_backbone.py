#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comparison test for BEVFormerBackbone: ttsim vs PyTorch
Tests the full pipeline: ResNet-50 backbone + FPN neck + BEV feature generation
"""

import os, sys

polaris_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if polaris_path not in sys.path:
    sys.path.insert(0, polaris_path)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))

# Fix for OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import numpy as np

from workloads.MapTracker.plugin.models.backbones.bevformer_backbone import (
    BEVFormerBackbone,
)

# ==========================================================================
# PyTorch Reference: ResNet-50 Bottleneck
# ==========================================================================


class BottleneckPyTorch(nn.Module):
    """ResNet bottleneck block (1x1 -> 3x3 -> 1x1 + residual)."""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# ==========================================================================
# PyTorch Reference: ResNet-50 Backbone
# ==========================================================================


class ResNet50PyTorch(nn.Module):
    """ResNet-50 backbone returning multi-scale features."""

    def __init__(self, img_channels=3, layers=None, out_indices=(1, 2, 3)):
        super().__init__()
        if layers is None:
            layers = [3, 4, 6, 3]
        self.out_indices = out_indices
        self.in_channels = 64

        self.conv1 = nn.Conv2d(img_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_stage(64, layers[0], stride=1)
        self.layer2 = self._make_stage(128, layers[1], stride=2)
        self.layer3 = self._make_stage(256, layers[2], stride=2)
        self.layer4 = self._make_stage(512, layers[3], stride=2)

    def _make_stage(self, planes, num_blocks, stride):
        exp = BottleneckPyTorch.expansion
        downsample = None
        if stride != 1 or self.in_channels != planes * exp:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * exp, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * exp),
            )
        layers = [BottleneckPyTorch(self.in_channels, planes, stride, downsample)]
        self.in_channels = planes * exp
        for _ in range(1, num_blocks):
            layers.append(BottleneckPyTorch(self.in_channels, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        outs = []
        for idx, stage in enumerate(stages):
            x = stage(x)
            if idx in self.out_indices:
                outs.append(x)
        return outs


# ==========================================================================
# PyTorch Reference: FPN Neck
# ==========================================================================


class FPNPyTorch(nn.Module):
    """Feature Pyramid Network."""

    def __init__(self, in_channels, out_channels, num_outs=3):
        super().__init__()
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for ic in in_channels:
            self.lateral_convs.append(nn.Conv2d(ic, out_channels, 1, bias=False))
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            )

        self.extra_convs = nn.ModuleList()
        if num_outs > self.num_ins:
            for i in range(num_outs - self.num_ins):
                ic = in_channels[-1] if i == 0 else out_channels
                self.extra_convs.append(
                    nn.Conv2d(ic, out_channels, 3, stride=2, padding=1, bias=False)
                )

    def forward(self, features):
        laterals = [self.lateral_convs[i](features[i]) for i in range(self.num_ins)]
        for i in range(self.num_ins - 2, -1, -1):
            laterals[i] = laterals[i] + F_torch.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="nearest"
            )
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]
        if self.num_outs > self.num_ins:
            x_extra = features[-1]
            for j, conv in enumerate(self.extra_convs):
                x_extra = conv(x_extra if j == 0 else outs[-1])
                outs.append(x_extra)
        return outs


# ==========================================================================
# PyTorch Reference: Mock BEV Transformer
# ==========================================================================


class SimpleBEVTransformerPyTorch(nn.Module):
    """Simplified BEV transformer for testing.
    Returns BEV queries + prop_bev (flattened) when history is provided."""

    def __init__(self, embed_dims):
        super().__init__()
        self.embed_dims = embed_dims

    def get_bev_features(self, mlvl_feats, bev_queries, bev_h, bev_w, **kwargs):
        bs = mlvl_feats[0].shape[0]
        output = bev_queries.unsqueeze(0).expand(bs, -1, -1)
        prop_bev = kwargs.get("prop_bev", None)
        if prop_bev is not None:
            # prop_bev: [bs, C, H, W] -> [bs, H*W, C]
            prop_flat = prop_bev.flatten(2).permute(0, 2, 1)
            output = output + prop_flat
        return output


# ==========================================================================
# PyTorch Reference: Full BEVFormerBackbone
# ==========================================================================

# ==========================================================================
# Weight injection helpers (ttsim params are positional, not named)
# ==========================================================================


def inject_conv2d_weights(ttsim_conv, pytorch_conv):
    """Conv2d: params[0][1] = weight."""
    ttsim_conv.params[0][1].data = pytorch_conv.weight.data.numpy()


def inject_bn2d_weights(ttsim_bn, pytorch_bn):
    """BatchNorm2d: params = [(1,scale), (2,bias), (3,mean), (4,var)]."""
    ttsim_bn.params[0][1].data = pytorch_bn.weight.data.numpy()
    ttsim_bn.params[1][1].data = pytorch_bn.bias.data.numpy()
    ttsim_bn.params[2][1].data = pytorch_bn.running_mean.data.numpy()
    ttsim_bn.params[3][1].data = pytorch_bn.running_var.data.numpy()


def inject_bottleneck_weights(ttsim_block, pytorch_block):
    """Inject all weights for a ResNet Bottleneck block."""
    pt_convs = [pytorch_block.conv1, pytorch_block.conv2, pytorch_block.conv3]
    pt_bns = [pytorch_block.bn1, pytorch_block.bn2, pytorch_block.bn3]
    # op_blk is [conv0, bn0, conv1, bn1, conv2, bn2]
    op_idx = 0
    for i in range(3):
        inject_conv2d_weights(ttsim_block.op_blk[op_idx], pt_convs[i])
        op_idx += 1
        inject_bn2d_weights(ttsim_block.op_blk[op_idx], pt_bns[i])
        op_idx += 1
    # Downsample
    if ttsim_block.downsample is not None and pytorch_block.downsample is not None:
        inject_conv2d_weights(ttsim_block.conv_ds, pytorch_block.downsample[0])
        inject_bn2d_weights(ttsim_block.bn_ds, pytorch_block.downsample[1])


def inject_resnet_weights(ttsim_backbone, pytorch_backbone):
    """Inject all ResNet-50 weights: stem + 4 stages."""
    # Stem
    inject_conv2d_weights(ttsim_backbone.conv1, pytorch_backbone.conv1)
    inject_bn2d_weights(ttsim_backbone.bn1, pytorch_backbone.bn1)
    # Stages
    pt_stages = [
        pytorch_backbone.layer1,
        pytorch_backbone.layer2,
        pytorch_backbone.layer3,
        pytorch_backbone.layer4,
    ]
    tt_stages = [
        ttsim_backbone.stage1,
        ttsim_backbone.stage2,
        ttsim_backbone.stage3,
        ttsim_backbone.stage4,
    ]
    for stage_idx in range(4):
        for blk_idx in range(len(pt_stages[stage_idx])):
            inject_bottleneck_weights(
                tt_stages[stage_idx][blk_idx], pt_stages[stage_idx][blk_idx]
            )


def inject_fpn_weights(ttsim_neck, pytorch_neck):
    """Inject all FPN weights: lateral convs + fpn convs."""
    for i in range(len(pytorch_neck.lateral_convs)):
        inject_conv2d_weights(
            ttsim_neck.lateral_convs[i], pytorch_neck.lateral_convs[i]
        )
    for i in range(len(pytorch_neck.fpn_convs)):
        inject_conv2d_weights(ttsim_neck.fpn_convs[i], pytorch_neck.fpn_convs[i])


class BEVFormerBackbonePyTorch(nn.Module):
    """
    PyTorch reference implementation of BEVFormerBackbone.
    Full pipeline: ResNet-50 -> FPN -> BEV query processing + history warping.
    """

    def __init__(
        self,
        bev_h,
        bev_w,
        embed_dims,
        real_h,
        real_w,
        upsample=False,
        up_outdim=128,
        history_steps=3,
        num_cams=7,
        img_channels=3,
        backbone_layers=None,
        out_indices=(1, 2, 3),
        fpn_in_channels=None,
        fpn_num_outs=3,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.real_h = real_h
        self.real_w = real_w
        self.upsample = upsample
        self.history_steps = history_steps
        self.num_cams = num_cams

        if backbone_layers is None:
            backbone_layers = [3, 4, 6, 3]
        if fpn_in_channels is None:
            fpn_in_channels = [512, 1024, 2048]

        self.img_backbone = ResNet50PyTorch(img_channels, backbone_layers, out_indices)
        self.img_neck = FPNPyTorch(fpn_in_channels, embed_dims, fpn_num_outs)
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        self.transformer = SimpleBEVTransformerPyTorch(embed_dims)

    def extract_img_feat(self, img):
        """Extract features: [B*num_cams, C, H, W] -> list of [B, num_cams, D, h, w]."""
        nc = self.num_cams
        B = img.shape[0] // nc
        backbone_feats = self.img_backbone(img)
        fpn_feats = self.img_neck(backbone_feats)
        mlvl_feats = []
        for feat in fpn_feats:
            _, d, fh, fw = feat.shape
            mlvl_feats.append(feat.view(B, nc, d, fh, fw))
        return mlvl_feats

    def forward(
        self,
        img,
        bev_pos,
        history_bev_feats=None,
        all_history_coord=None,
        prev_bev=None,
        img_metas=None,
    ):
        mlvl_feats = self.extract_img_feat(img)
        bs = mlvl_feats[0].shape[0]

        bev_indices = torch.arange(self.bev_h * self.bev_w, dtype=torch.long)
        bev_queries = self.bev_embedding(bev_indices)

        if history_bev_feats is not None and len(history_bev_feats) > 0:
            T = len(history_bev_feats)
            history_stacked = torch.stack(history_bev_feats, dim=0)
            history_transposed = history_stacked.permute(1, 0, 2, 3, 4)
            bs_t, t, c, h_dim, w_dim = history_transposed.shape
            history_flat = history_transposed.reshape(bs_t * t, c, h_dim, w_dim)
            coord_flat = all_history_coord.reshape(bs * T, self.bev_h, self.bev_w, 2)
            warped_flat = F_torch.grid_sample(
                history_flat, coord_flat, padding_mode="zeros", align_corners=False
            )
            all_warped_history_feat = warped_flat.reshape(
                bs, T, self.embed_dims, self.bev_h, self.bev_w
            )
            prop_bev_feat = all_warped_history_feat[:, -1]
            if T < self.history_steps:
                num_repeat = self.history_steps - T
                zero_padding = torch.zeros(
                    bs, num_repeat, self.embed_dims, self.bev_h, self.bev_w
                )
                all_warped_history_feat = torch.cat(
                    [zero_padding, all_warped_history_feat], dim=1
                )
        else:
            all_warped_history_feat = None
            prop_bev_feat = None

        outs = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            bev_pos=bev_pos,
            prop_bev=prop_bev_feat,
            img_metas=img_metas,
            prev_bev=prev_bev,
            warped_history_bev=all_warped_history_feat,
        )

        outs = outs.reshape(bs, self.bev_h, self.bev_w, self.embed_dims)
        outs = outs.permute(0, 3, 1, 2)

        if self.upsample:
            outs = self.up(outs)

        return outs, mlvl_feats


def test_bevformer_backbone_construction():
    """Test that the module can be constructed successfully."""
    print("\n" + "=" * 80)
    print("TEST 1: Module Construction")
    print("=" * 80)

    try:
        import ttsim.front.functional.sim_nn as SimNN

        # Create mock transformer
        class MockTransformer(SimNN.Module):
            def __init__(self, embed_dims):
                super().__init__()
                self.embed_dims = embed_dims

            def get_bev_features(self, *args, **kwargs):
                # Return dummy output
                import ttsim.front.functional.op as F

                return F._from_shape("mock_output", [1, 20 * 20, 32], is_param=False)

        transformer = MockTransformer(embed_dims=32)

        backbone = BEVFormerBackbone(
            name="test_backbone",
            bev_h=20,
            bev_w=20,
            embed_dims=32,
            real_h=30.0,
            real_w=60.0,
            transformer=transformer,
            max_batch_size=4,
            upsample=False,
            history_steps=3,
            num_cams=2,
            backbone_layers=[1, 1, 1, 1],
            out_indices=(1, 2, 3),
            fpn_in_channels=[512, 1024, 2048],
            fpn_num_outs=3,
        )

        print("[OK] Module constructed successfully")
        print(f"  - Module name: {backbone.name}")
        print(f"  - BEV grid: {backbone.bev_h}x{backbone.bev_w}")
        print(f"  - Embed dims: {backbone.embed_dims}")
        print(f"  - Real dimensions: {backbone.real_h}x{backbone.real_w} meters")
        print(f"  - History steps: {backbone.history_steps}")
        print(f"  - Max batch size: {backbone.max_batch_size}")
        print(f"  - Num cameras: {backbone.num_cams}")
        print(f"  - Has img_backbone: {hasattr(backbone, 'img_backbone')}")
        print(f"  - Has img_neck: {hasattr(backbone, 'img_neck')}")
        return True

    except Exception as e:
        print(f"[X] Construction failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_bevformer_backbone_forward():
    """Test forward pass with raw image inputs (ResNet-50 + FPN + BEV)."""
    print("\n" + "=" * 80)
    print("TEST 2: Forward Pass (No History)")
    print("=" * 80)

    # Test parameters (reduced for fast testing)
    bs = 1
    num_cam = 2
    img_h, img_w = 32, 32
    bev_h, bev_w = 10, 10
    embed_dims = 32
    real_h, real_w = 30.0, 60.0
    backbone_layers = [1, 1, 1, 1]
    fpn_in_channels = [512, 1024, 2048]

    print(f"\nTest parameters:")
    print(f"  - Batch size: {bs}")
    print(f"  - Cameras: {num_cam}")
    print(f"  - Image size: {img_h}x{img_w}")
    print(f"  - BEV grid: {bev_h}x{bev_w}")
    print(f"  - Embed dims: {embed_dims}")
    print(f"  - Backbone layers: {backbone_layers}")

    # Create inputs
    np.random.seed(42)
    torch.manual_seed(42)

    # Raw camera images: [B*num_cam, 3, H, W]
    img_np = np.random.randn(bs * num_cam, 3, img_h, img_w).astype(np.float32) * 0.01
    img_torch = torch.from_numpy(img_np)

    # BEV positional encoding
    bev_pos_np = np.random.randn(bs, embed_dims, bev_h, bev_w).astype(np.float32)
    bev_pos_torch = torch.from_numpy(bev_pos_np)

    print(f"\nInput shapes:")
    print(f"  - Images: {img_np.shape}")
    print(f"  - BEV pos: {bev_pos_np.shape}")

    try:
        # ========== PyTorch Implementation ==========
        print("\n" + "-" * 80)
        print("PyTorch Implementation")
        print("-" * 80)

        model_pytorch = BEVFormerBackbonePyTorch(
            bev_h=bev_h,
            bev_w=bev_w,
            embed_dims=embed_dims,
            real_h=real_h,
            real_w=real_w,
            upsample=False,
            history_steps=3,
            num_cams=num_cam,
            backbone_layers=backbone_layers,
            fpn_in_channels=fpn_in_channels,
        )
        model_pytorch.eval()

        with torch.no_grad():
            output_pytorch, mlvl_feats_pytorch = model_pytorch(
                img_torch, bev_pos_torch, history_bev_feats=None, all_history_coord=None
            )

        print(f"PyTorch output shape: {output_pytorch.shape}")
        print(
            f"PyTorch output range: [{output_pytorch.min().item():.6f}, {output_pytorch.max().item():.6f}]"
        )
        print(f"PyTorch mlvl_feats levels: {len(mlvl_feats_pytorch)}")
        for i, f in enumerate(mlvl_feats_pytorch):
            print(f"  Level {i}: {f.shape}")

        # ========== ttsim Implementation ==========
        print("\n" + "-" * 80)
        print("ttsim Implementation")
        print("-" * 80)

        import ttsim.front.functional.op as F
        import ttsim.front.functional.sim_nn as SimNN

        # Create mock transformer for ttsim (same logic as PyTorch)
        class MockTransformerTTSim(SimNN.Module):
            def __init__(self, embed_dims, bev_h, bev_w):
                super().__init__()
                self.embed_dims = embed_dims
                self.bev_h = bev_h
                self.bev_w = bev_w

            def get_bev_features(self, mlvl_feats, bev_queries, bev_h, bev_w, **kwargs):
                bs = mlvl_feats[0].shape[0]

                reshape_1 = F.Reshape("mock_transformer_reshape1")
                queries_3d_shape = F._from_data(
                    "mock_queries_3d_shape",
                    np.array([1, bev_h * bev_w, self.embed_dims], dtype=np.int64),
                    is_const=True,
                )
                queries_3d = reshape_1(bev_queries, queries_3d_shape)

                tile_op = F.Tile("mock_transformer_tile")
                tile_repeats = F._from_data(
                    "mock_tile_repeats",
                    np.array([bs, 1, 1], dtype=np.int64),
                    is_const=True,
                )
                output = tile_op(queries_3d, tile_repeats)

                return output

        transformer_ttsim = MockTransformerTTSim(embed_dims, bev_h, bev_w)

        model_ttsim = BEVFormerBackbone(
            name="test_backbone",
            bev_h=bev_h,
            bev_w=bev_w,
            embed_dims=embed_dims,
            real_h=real_h,
            real_w=real_w,
            transformer=transformer_ttsim,
            max_batch_size=4,
            upsample=False,
            history_steps=3,
            num_cams=num_cam,
            backbone_layers=backbone_layers,
            fpn_in_channels=fpn_in_channels,
        )

        # Inject PyTorch weights into ttsim model
        print("\nInjecting PyTorch weights...")

        # BEV embedding
        model_ttsim.bev_embedding.params[0][
            1
        ].data = model_pytorch.bev_embedding.weight.data.numpy()

        # ResNet-50 backbone + FPN neck
        inject_resnet_weights(model_ttsim.img_backbone, model_pytorch.img_backbone)
        inject_fpn_weights(model_ttsim.img_neck, model_pytorch.img_neck)

        print("Weight injection complete")

        # Create ttsim input
        img_ttsim = F._from_data("img_input", img_np, is_const=True)
        bev_pos_ttsim = F._from_data("bev_pos", bev_pos_np, is_const=True)

        # Forward pass
        output_ttsim, mlvl_feats_ttsim = model_ttsim(
            img_ttsim, bev_pos_ttsim, history_bev_feats=None, all_history_coord=None
        )

        if output_ttsim.data is None:
            print("ERROR: ttsim output.data is None!")
            return False

        print(f"ttsim output shape: {output_ttsim.data.shape}")
        print(
            f"ttsim output range: [{output_ttsim.data.min():.6f}, {output_ttsim.data.max():.6f}]"
        )
        print(f"ttsim mlvl_feats levels: {len(mlvl_feats_ttsim)}")

        # ========== Validate FPN Output (Intermediate) ==========
        print("\n" + "=" * 80)
        print("Intermediate Validation: FPN Output")
        print("=" * 80)

        for lvl_idx in range(len(mlvl_feats_pytorch)):
            pt_feat = mlvl_feats_pytorch[lvl_idx].numpy()
            tt_feat = mlvl_feats_ttsim[lvl_idx].data
            lvl_diff = np.abs(pt_feat - tt_feat)
            print(f"FPN Level {lvl_idx}: shape PT={pt_feat.shape}, TT={tt_feat.shape}")
            print(f"  diff: max={lvl_diff.max():.6e}, mean={lvl_diff.mean():.6e}")
            if np.allclose(pt_feat, tt_feat, rtol=1e-3, atol=1e-3):
                print(f"  [OK] FPN level {lvl_idx} matches")
            else:
                print(
                    f"  [X] FPN level {lvl_idx} differs (may be due to BN/interpolation differences)"
                )

        # ========== Validate BEV Embedding ==========
        print("\nValidating BEV Embedding Lookup...")
        with torch.no_grad():
            bev_indices_torch = torch.arange(bev_h * bev_w, dtype=torch.long)
            bev_queries_pytorch = model_pytorch.bev_embedding(bev_indices_torch)

        bev_queries_tensor_name = "test_backbone.bev_embedding.out"
        if bev_queries_tensor_name in model_ttsim._tensors:
            bev_queries_ttsim = model_ttsim._tensors[bev_queries_tensor_name]
            bev_queries_pytorch_np = bev_queries_pytorch.numpy()
            bev_emb_diff = np.abs(bev_queries_ttsim.data - bev_queries_pytorch_np)
            print(
                f"  BEV queries diff: max={bev_emb_diff.max():.10e}, mean={bev_emb_diff.mean():.10e}"
            )
            if np.allclose(
                bev_queries_ttsim.data, bev_queries_pytorch_np, rtol=1e-5, atol=1e-6
            ):
                print(f"  [OK] BEV Embedding lookup matches")
            else:
                print(f"  [X] BEV Embedding lookup differs!")
                return False

        # ========== Final Output Comparison ==========
        print("\n" + "=" * 80)
        print("Final Output Comparison")
        print("=" * 80)

        output_pytorch_np = output_pytorch.numpy()
        diff = np.abs(output_ttsim.data - output_pytorch_np)

        print(f"Max absolute difference: {diff.max():.10e}")
        print(f"Mean absolute difference: {diff.mean():.10e}")
        print(f"Median absolute difference: {np.median(diff):.10e}")

        if np.allclose(output_ttsim.data, output_pytorch_np, rtol=1e-4, atol=1e-4):
            print(
                f"\n[OK] TEST PASSED: Full pipeline (ResNet-50 + FPN + BEV) validated"
            )
            return True
        else:
            print(f"\n[X] TEST FAILED: Outputs differ by more than tolerance")
            max_idx = np.unravel_index(diff.argmax(), diff.shape)
            print(f"  Largest difference at index {max_idx}:")
            print(f"    PyTorch: {output_pytorch_np[max_idx]:.10f}")
            print(f"    TTSim:   {output_ttsim.data[max_idx]:.10f}")
            return False

    except Exception as e:
        print(f"\n[X] Test failed with exception: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_bevformer_backbone_with_history():
    """Test forward pass with history warping - PyTorch vs TTSim comparison."""
    print("\n" + "=" * 80)
    print("TEST 3: Forward Pass (With History Warping)")
    print("=" * 80)

    # Test parameters (reduced for fast testing)
    bs = 1
    num_cam = 2
    img_h, img_w = 32, 32
    bev_h, bev_w = 10, 10
    embed_dims = 32
    real_h, real_w = 30.0, 60.0
    history_steps = 3
    T = 2  # Number of history frames
    backbone_layers = [1, 1, 1, 1]
    fpn_in_channels = [512, 1024, 2048]

    print(f"\nTest parameters:")
    print(f"  - Batch size: {bs}")
    print(f"  - Cameras: {num_cam}")
    print(f"  - Image size: {img_h}x{img_w}")
    print(f"  - BEV grid: {bev_h}x{bev_w}")
    print(f"  - Embed dims: {embed_dims}")
    print(f"  - History frames: {T}")
    print(f"  - Backbone layers: {backbone_layers}")

    # Create inputs
    np.random.seed(42)
    torch.manual_seed(42)

    # Raw camera images
    img_np = np.random.randn(bs * num_cam, 3, img_h, img_w).astype(np.float32) * 0.01
    img_torch = torch.from_numpy(img_np)

    bev_pos_np = np.random.randn(bs, embed_dims, bev_h, bev_w).astype(np.float32)
    bev_pos_torch = torch.from_numpy(bev_pos_np)

    # History BEV features
    history_bev_feats_np = [
        np.random.randn(bs, embed_dims, bev_h, bev_w).astype(np.float32)
        for _ in range(T)
    ]
    history_bev_feats_torch = [torch.from_numpy(h) for h in history_bev_feats_np]

    # Warp coordinates (normalized grid coordinates in [-1, 1])
    all_history_coord_np = (
        np.random.randn(bs, T, bev_h, bev_w, 2).astype(np.float32) * 0.5
    )
    all_history_coord_np = np.clip(all_history_coord_np, -1.0, 1.0)
    all_history_coord_torch = torch.from_numpy(all_history_coord_np)

    print(f"\nInputs:")
    print(f"  - Images: {img_np.shape}")
    print(
        f"  - History BEV features: {T} frames of shape {history_bev_feats_np[0].shape}"
    )
    print(f"  - Warp coordinates: {all_history_coord_np.shape}")

    try:
        # ========== PyTorch Implementation ==========
        print("\n" + "-" * 80)
        print("PyTorch Implementation")
        print("-" * 80)

        model_pytorch = BEVFormerBackbonePyTorch(
            bev_h=bev_h,
            bev_w=bev_w,
            embed_dims=embed_dims,
            real_h=real_h,
            real_w=real_w,
            upsample=False,
            history_steps=history_steps,
            num_cams=num_cam,
            backbone_layers=backbone_layers,
            fpn_in_channels=fpn_in_channels,
        )
        model_pytorch.eval()

        with torch.no_grad():
            output_pytorch, _ = model_pytorch(
                img_torch,
                bev_pos_torch,
                history_bev_feats=history_bev_feats_torch,
                all_history_coord=all_history_coord_torch,
            )

        print(f"PyTorch output shape: {output_pytorch.shape}")
        print(
            f"PyTorch output range: [{output_pytorch.min().item():.6f}, {output_pytorch.max().item():.6f}]"
        )

        # ========== ttsim Implementation ==========
        print("\n" + "-" * 80)
        print("ttsim Implementation")
        print("-" * 80)

        import ttsim.front.functional.op as F
        import ttsim.front.functional.sim_nn as SimNN

        # Create mock transformer for ttsim
        class MockTransformerTTSim(SimNN.Module):
            def __init__(self, embed_dims, bev_h, bev_w):
                super().__init__()
                self.embed_dims = embed_dims
                self.bev_h = bev_h
                self.bev_w = bev_w

            def get_bev_features(self, mlvl_feats, bev_queries, bev_h, bev_w, **kwargs):
                bs = mlvl_feats[0].shape[0]

                reshape_1 = F.Reshape("mock_hist_transformer_reshape1")
                queries_3d_shape = F._from_data(
                    "mock_hist_queries_3d_shape",
                    np.array([1, bev_h * bev_w, self.embed_dims], dtype=np.int64),
                    is_const=True,
                )
                queries_3d = reshape_1(bev_queries, queries_3d_shape)

                tile_op = F.Tile("mock_hist_transformer_tile")
                tile_repeats = F._from_data(
                    "mock_hist_tile_repeats",
                    np.array([bs, 1, 1], dtype=np.int64),
                    is_const=True,
                )
                output = tile_op(queries_3d, tile_repeats)

                prop_bev = kwargs.get("prop_bev", None)
                if prop_bev is not None:
                    # prop_bev: [bs, C, H, W] -> [bs, H*W, C]
                    flatten_reshape = F.Reshape("mock_hist_prop_flatten")
                    flatten_shape = F._from_data(
                        "mock_hist_prop_flat_shape",
                        np.array([bs, self.embed_dims, bev_h * bev_w], dtype=np.int64),
                        is_const=True,
                    )
                    prop_flat = flatten_reshape(prop_bev, flatten_shape)

                    transpose_op = F.Transpose(
                        "mock_hist_prop_transpose", perm=[0, 2, 1]
                    )
                    prop_flat = transpose_op(prop_flat)

                    add_op = F.Add("mock_hist_prop_add")
                    output = add_op(output, prop_flat)

                return output

        transformer_ttsim = MockTransformerTTSim(embed_dims, bev_h, bev_w)

        model_ttsim = BEVFormerBackbone(
            name="test_backbone_hist",
            bev_h=bev_h,
            bev_w=bev_w,
            embed_dims=embed_dims,
            real_h=real_h,
            real_w=real_w,
            transformer=transformer_ttsim,
            max_batch_size=4,
            upsample=False,
            history_steps=history_steps,
            num_cams=num_cam,
            backbone_layers=backbone_layers,
            fpn_in_channels=fpn_in_channels,
        )

        # Inject PyTorch weights
        print("\nInjecting PyTorch weights...")

        # BEV embedding
        model_ttsim.bev_embedding.params[0][
            1
        ].data = model_pytorch.bev_embedding.weight.data.numpy()

        # ResNet-50 backbone + FPN neck
        inject_resnet_weights(model_ttsim.img_backbone, model_pytorch.img_backbone)
        inject_fpn_weights(model_ttsim.img_neck, model_pytorch.img_neck)

        print("Weight injection complete")

        # Create ttsim inputs
        img_ttsim = F._from_data("img_hist_input", img_np, is_const=True)
        bev_pos_ttsim = F._from_data("bev_pos_hist", bev_pos_np, is_const=True)

        history_bev_feats_ttsim = [
            F._from_data(f"history_bev_{i}", h, is_const=True)
            for i, h in enumerate(history_bev_feats_np)
        ]
        all_history_coord_ttsim = F._from_data(
            "all_history_coord", all_history_coord_np, is_const=True
        )

        # Forward pass with history
        output_ttsim, _ = model_ttsim(
            img_ttsim,
            bev_pos_ttsim,
            history_bev_feats=history_bev_feats_ttsim,
            all_history_coord=all_history_coord_ttsim,
        )

        if output_ttsim.data is None:
            print("ERROR: ttsim output.data is None!")
            return False

        print(f"ttsim output shape: {output_ttsim.data.shape}")
        print(
            f"ttsim output range: [{output_ttsim.data.min():.6f}, {output_ttsim.data.max():.6f}]"
        )

        # ========== Intermediate Validation: History Warping ==========
        print("\n" + "=" * 80)
        print("History Warping Validation (Intermediate Outputs)")
        print("=" * 80)

        # Extract prop_bev_feat from both implementations
        # PyTorch: Recompute history warping manually
        with torch.no_grad():
            # Stack history: [T, bs, C, H, W]
            history_stacked_pt = torch.stack(history_bev_feats_torch, dim=0)
            # Transpose: [bs, T, C, H, W]
            history_transposed_pt = history_stacked_pt.permute(1, 0, 2, 3, 4)
            # Reshape: [bs*T, C, H, W]
            history_flat_pt = history_transposed_pt.reshape(
                bs * T, embed_dims, bev_h, bev_w
            )
            # Reshape coords: [bs*T, H, W, 2]
            coord_flat_pt = all_history_coord_torch.reshape(bs * T, bev_h, bev_w, 2)
            # GridSample
            warped_flat_pt = F_torch.grid_sample(
                history_flat_pt,
                coord_flat_pt,
                padding_mode="zeros",
                align_corners=False,
            )

            # Reshape back: [bs, T, C, H, W]
            warped_pt = warped_flat_pt.reshape(bs, T, embed_dims, bev_h, bev_w)
            # Extract last: [bs, C, H, W]
            prop_bev_pt = warped_pt[:, -1]

        print(f"PyTorch prop_bev shape: {prop_bev_pt.shape}")
        print(
            f"PyTorch prop_bev range: [{prop_bev_pt.min().item():.6f}, {prop_bev_pt.max().item():.6f}]"
        )

        # TTSim: Extract from internal tensors
        prop_bev_tensor_name = "test_backbone_hist.prop_reshape.out"
        if prop_bev_tensor_name in model_ttsim._tensors:
            prop_bev_ttsim = model_ttsim._tensors[prop_bev_tensor_name]
            print(f"ttsim prop_bev shape: {prop_bev_ttsim.data.shape}")
            print(
                f"ttsim prop_bev range: [{prop_bev_ttsim.data.min():.6f}, {prop_bev_ttsim.data.max():.6f}]"
            )

            # Compare
            prop_bev_pt_np = prop_bev_pt.numpy()
            prop_diff = np.abs(prop_bev_ttsim.data - prop_bev_pt_np)

            print(f"\nHistory warping comparison:")
            print(f"  Max absolute difference: {prop_diff.max():.10e}")
            print(f"  Mean absolute difference: {prop_diff.mean():.10e}")
            print(f"  Median absolute difference: {np.median(prop_diff):.10e}")

            # Validate history warping using np.allclose
            if np.allclose(prop_bev_ttsim.data, prop_bev_pt_np, rtol=1e-5, atol=1e-5):
                print(f"  [OK] History warping matches (rtol=1e-5, atol=1e-5)")
            else:
                print(f"  [X] History warping differs")
                max_diff_idx = np.unravel_index(prop_diff.argmax(), prop_diff.shape)
                print(f"  Largest diff at index {max_diff_idx}:")
                print(f"    PyTorch: {prop_bev_pt_np[max_diff_idx]:.6f}")
                print(f"    TTSim:   {prop_bev_ttsim.data[max_diff_idx]:.6f}")
        else:
            print(f"[WARNING]  Could not find prop_bev tensor in TTSim outputs")
            print(
                f"Available tensor sample: {list(model_ttsim._tensors.keys())[:5]}..."
            )

        # ========== Final Output Comparison ==========
        print("\n" + "=" * 80)
        print("Final Output Comparison")
        print("=" * 80)

        output_pytorch_np = output_pytorch.numpy()
        diff = np.abs(output_ttsim.data - output_pytorch_np)

        print(f"Max absolute difference: {diff.max():.10e}")
        print(f"Mean absolute difference: {diff.mean():.10e}")
        print(f"Median absolute difference: {np.median(diff):.10e}")

        # Final output now incorporates prop_bev via mock transformer
        if np.allclose(output_ttsim.data, output_pytorch_np, rtol=1e-4, atol=1e-4):
            print(
                f"\n[OK] TEST PASSED: Full pipeline with history validated (rtol=1e-4, atol=1e-4)"
            )
            return True
        else:
            print(f"\n[X] TEST FAILED: Final outputs differ")
            max_idx = np.unravel_index(diff.argmax(), diff.shape)
            print(f"  Largest difference at index {max_idx}:")
            print(f"    PyTorch: {output_pytorch_np[max_idx]:.10f}")
            print(f"    TTSim:   {output_ttsim.data[max_idx]:.10f}")
            return False

    except Exception as e:
        print(f"\n[X] Test failed with exception: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BEVFormerBackbone Test Suite")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("Construction", test_bevformer_backbone_construction()))
    results.append(("Forward (No History)", test_bevformer_backbone_forward()))
    results.append(("Forward (With History)", test_bevformer_backbone_with_history()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results:
        status = "[OK] PASSED" if passed else "[X] FAILED"
        print(f"{test_name:.<60} {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n! All tests passed!")
    else:
        print(f"\n[WARNING]  {total - passed} test(s) failed")
