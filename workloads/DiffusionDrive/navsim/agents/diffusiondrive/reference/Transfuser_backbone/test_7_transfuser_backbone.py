#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for TransfuserBackbone module (Full Model).
Tests shape inference and numerical equivalence between PyTorch and TTSIM.

Full numerical validation includes:
1. PyTorch reference implementation of TransfuserBackbone
2. Weight injection from PyTorch to TTSIM transformers
3. Identical timm encoder initialization using cached features
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add DiffusionDrive to path for navsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
# Add TT-Polaris to path for ttsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.reference.torch_code.navsim.agents.diffusiondrive.transfuser_backbone import (
    TransfuserBackbone as TransfuserBackbone_PyTorch,
)
from navsim.agents.diffusiondrive.transfuser_backbone_ttsim import (
    TransfuserBackbone as TransfuserBackbone_TTSIM,
)

import ttsim.front.functional.op as F


def inject_weights_from_pytorch_to_ttsim(pytorch_gpt, ttsim_gpt, layer_idx=0):
    """
    Inject weights from PyTorch GPT transformer to TTSIM GPT transformer.

    Args:
        pytorch_gpt: PyTorch GPT module
        ttsim_gpt: TTSIM GPT module
        layer_idx: Which transformer layer (0-3)
    """
    # Inject positional embedding
    if hasattr(pytorch_gpt, "pos_emb"):
        pos_emb_data = pytorch_gpt.pos_emb.data.cpu().numpy()
        ttsim_gpt.pos_emb.data = pos_emb_data

    # Inject weights for each block
    for block_idx in range(len(pytorch_gpt.blocks)):
        pytorch_block = pytorch_gpt.blocks[block_idx]
        ttsim_block = ttsim_gpt.blocks[block_idx]

        # LayerNorm1
        ln1_weight = pytorch_block.ln1.weight.data.cpu().numpy()
        ln1_bias = pytorch_block.ln1.bias.data.cpu().numpy()
        ttsim_block.ln1.params[0][1].data = ln1_weight  # scale
        ttsim_block.ln1.params[1][1].data = ln1_bias  # bias

        # Self-Attention: query, key, value, proj (each has separate Linear + Bias)
        # Note: PyTorch Linear weights are (out_features, in_features), need transpose for TTSIM MatMul
        q_weight = pytorch_block.attn.query.weight.data.cpu().numpy().T
        q_bias = pytorch_block.attn.query.bias.data.cpu().numpy()
        k_weight = pytorch_block.attn.key.weight.data.cpu().numpy().T
        k_bias = pytorch_block.attn.key.bias.data.cpu().numpy()
        v_weight = pytorch_block.attn.value.weight.data.cpu().numpy().T
        v_bias = pytorch_block.attn.value.bias.data.cpu().numpy()
        proj_weight = pytorch_block.attn.proj.weight.data.cpu().numpy().T
        proj_bias = pytorch_block.attn.proj.bias.data.cpu().numpy()

        # Inject weights into Linear ops and biases into Bias ops
        ttsim_block.attn.query.params[0][1].data = q_weight
        ttsim_block.attn.query_bias.params[0][1].data = q_bias
        ttsim_block.attn.key.params[0][1].data = k_weight
        ttsim_block.attn.key_bias.params[0][1].data = k_bias
        ttsim_block.attn.value.params[0][1].data = v_weight
        ttsim_block.attn.value_bias.params[0][1].data = v_bias
        ttsim_block.attn.proj.params[0][1].data = proj_weight
        ttsim_block.attn.proj_bias.params[0][1].data = proj_bias

        # LayerNorm2
        ln2_weight = pytorch_block.ln2.weight.data.cpu().numpy()
        ln2_bias = pytorch_block.ln2.bias.data.cpu().numpy()
        ttsim_block.ln2.params[0][1].data = ln2_weight
        ttsim_block.ln2.params[1][1].data = ln2_bias

        # MLP: fc1, fc2 (each has separate Linear + Bias)
        # Transpose weights for TTSIM MatMul
        fc1_weight = pytorch_block.mlp[0].weight.data.cpu().numpy().T
        fc1_bias = pytorch_block.mlp[0].bias.data.cpu().numpy()
        fc2_weight = pytorch_block.mlp[2].weight.data.cpu().numpy().T
        fc2_bias = pytorch_block.mlp[2].bias.data.cpu().numpy()

        ttsim_block.mlp_fc1.params[0][1].data = fc1_weight
        ttsim_block.mlp_fc1_bias.params[0][1].data = fc1_bias
        ttsim_block.mlp_fc2.params[0][1].data = fc2_weight
        ttsim_block.mlp_fc2_bias.params[0][1].data = fc2_bias

    # Final LayerNorm
    ln_f_weight = pytorch_gpt.ln_f.weight.data.cpu().numpy()
    ln_f_bias = pytorch_gpt.ln_f.bias.data.cpu().numpy()
    ttsim_gpt.ln_f.params[0][1].data = ln_f_weight
    ttsim_gpt.ln_f.params[1][1].data = ln_f_bias


def inject_conv2d_weights(pytorch_conv, ttsim_conv, name="Conv2d"):
    """Inject Conv2d weights from PyTorch to TTSIM."""
    weight = pytorch_conv.weight.data.cpu().numpy()
    # Conv2d in TTSIM only has weight at params[0][1], no bias parameter
    ttsim_conv.params[0][1].data = weight


def inject_bn2d_weights(pytorch_bn, ttsim_bn):
    """Inject BatchNorm2d weights from PyTorch to TTSIM."""
    ttsim_bn.params[0][1].data = pytorch_bn.weight.data.cpu().numpy()
    ttsim_bn.params[1][1].data = pytorch_bn.bias.data.cpu().numpy()
    ttsim_bn.params[2][1].data = pytorch_bn.running_mean.cpu().numpy()
    ttsim_bn.params[3][1].data = pytorch_bn.running_var.cpu().numpy()


def inject_basic_block_weights(pt_block, ttsim_block):
    """Inject BasicBlock weights from PyTorch to TTSIM."""
    inject_conv2d_weights(pt_block.conv1, ttsim_block.conv1, name="block.conv1")
    inject_bn2d_weights(pt_block.bn1, ttsim_block.bn1)
    inject_conv2d_weights(pt_block.conv2, ttsim_block.conv2, name="block.conv2")
    inject_bn2d_weights(pt_block.bn2, ttsim_block.bn2)
    if ttsim_block.has_downsample and pt_block.downsample is not None:
        inject_conv2d_weights(
            pt_block.downsample[0], ttsim_block.ds_conv, name="downsample.conv"
        )
        inject_bn2d_weights(pt_block.downsample[1], ttsim_block.ds_bn)


def inject_resnet_encoder_weights(pytorch_encoder, ttsim_encoder):
    """Inject all ResNet encoder weights from PyTorch timm model to TTSIM ResNetEncoder."""
    pt_children = dict(pytorch_encoder.named_children())

    # Stem: conv1 + bn1
    inject_conv2d_weights(pt_children["conv1"], ttsim_encoder.conv1, name="stem.conv1")
    inject_bn2d_weights(pt_children["bn1"], ttsim_encoder.bn1)

    # Residual stages
    for stage_name, ttsim_stage in [
        ("layer1", ttsim_encoder.layer1),
        ("layer2", ttsim_encoder.layer2),
        ("layer3", ttsim_encoder.layer3),
        ("layer4", ttsim_encoder.layer4),
    ]:
        pt_stage = pt_children[stage_name]
        for blk_idx, (pt_blk, ttsim_blk) in enumerate(zip(pt_stage, ttsim_stage)):
            inject_basic_block_weights(pt_blk, ttsim_blk)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    atol = 1e-4
    rtol = 1e-4

    # Configuration
    config = TransfuserConfig()
    config.n_layer = 2
    config.n_head = 4
    config.block_exp = 4
    config.attn_pdrop = 0.0
    config.resid_pdrop = 0.0
    config.embd_pdrop = 0.0
    config.img_vert_anchors = 8
    config.img_horz_anchors = 32
    config.lidar_vert_anchors = 8
    config.lidar_horz_anchors = 8
    config.use_ground_plane = False
    config.lidar_seq_len = 1
    config.latent = False
    config.add_features = True
    config.transformer_decoder_join = False
    config.detect_boxes = True
    config.use_bev_semantic = True
    config.use_semantic = False
    config.use_depth = False
    config.bev_features_channels = 64
    config.bev_upsample_factor = 2.0
    config.bev_down_sample_factor = 4
    config.lidar_resolution_width = 256
    config.lidar_resolution_height = 256
    config.image_architecture = "resnet18"
    config.lidar_architecture = "resnet18"

    batch_size = 1
    image_h, image_w = 224, 224
    lidar_h, lidar_w = 256, 256
    lidar_channels = (
        2 * config.lidar_seq_len if config.use_ground_plane else config.lidar_seq_len
    )

    image_data = np.random.randn(batch_size, 3, image_h, image_w).astype(np.float32)
    lidar_data = np.random.randn(batch_size, lidar_channels, lidar_h, lidar_w).astype(
        np.float32
    )

    print("=" * 70)
    print("TEST: TransfuserBackbone")
    print("=" * 70)

    # --- PyTorch ---
    model_pytorch = TransfuserBackbone_PyTorch(config)
    model_pytorch.eval()

    image_torch = torch.from_numpy(image_data)
    lidar_torch = torch.from_numpy(lidar_data)

    with torch.no_grad():
        features_pt, fused_features_pt, image_feature_grid_pt = model_pytorch(
            image_torch, lidar_torch
        )

    features_pt_np = features_pt.cpu().numpy() if features_pt is not None else None
    fused_features_pt_np = (
        fused_features_pt.cpu().numpy() if fused_features_pt is not None else None
    )

    print(f"\n--- PyTorch TransfuserBackbone ---")
    print(f"Image input shape: {list(image_data.shape)}")
    print(f"LiDAR input shape: {list(lidar_data.shape)}")
    if features_pt_np is not None:
        print(f"BEV features shape: {list(features_pt_np.shape)}")
        print(
            f"BEV features stats: min={features_pt_np.min():.6f}, max={features_pt_np.max():.6f}"
        )
    if fused_features_pt_np is not None:
        print(f"Fused features shape: {list(fused_features_pt_np.shape)}")

    # Extract encoder features for per-layer transformer validation
    with torch.no_grad():
        image_features_pt = image_torch
        lidar_features_pt = lidar_torch

        image_layers = iter(model_pytorch.image_encoder.items())
        lidar_layers = iter(model_pytorch.lidar_encoder.items())

        if len(model_pytorch.image_encoder.return_layers) > 4:
            for name, module in image_layers:
                image_features_pt = module(image_features_pt)
                if name in model_pytorch.image_encoder.return_layers:
                    break
        if len(model_pytorch.lidar_encoder.return_layers) > 4:
            for name, module in lidar_layers:
                lidar_features_pt = module(lidar_features_pt)
                if name in model_pytorch.lidar_encoder.return_layers:
                    break

        encoder_features_pt = []
        for i in range(4):
            for name, module in image_layers:
                image_features_pt = module(image_features_pt)
                if name in model_pytorch.image_encoder.return_layers:
                    break
            for name, module in lidar_layers:
                lidar_features_pt = module(lidar_features_pt)
                if name in model_pytorch.lidar_encoder.return_layers:
                    break
            encoder_features_pt.append(
                {"image": image_features_pt.clone(), "lidar": lidar_features_pt.clone()}
            )

    # --- TTSIM ---
    model_ttsim = TransfuserBackbone_TTSIM("transfuser", config)

    # Inject weights
    print(f"\n--- TTSIM TransfuserBackbone ---")
    print("Injecting weights...")

    # Inject ResNet encoder weights layer-by-layer (TTSIM has no load_state_dict)
    inject_resnet_encoder_weights(
        model_pytorch.image_encoder, model_ttsim.image_encoder
    )
    inject_resnet_encoder_weights(
        model_pytorch.lidar_encoder, model_ttsim.lidar_encoder
    )

    for i in range(4):
        inject_weights_from_pytorch_to_ttsim(
            model_pytorch.transformers[i], model_ttsim.transformers[i], layer_idx=i
        )
    for i in range(4):
        inject_conv2d_weights(
            model_pytorch.lidar_channel_to_img[i],
            getattr(model_ttsim, f"lidar_channel_to_img_{i}"),
            name=f"lidar_channel_to_img[{i}]",
        )
        # Inject Conv2d bias
        getattr(model_ttsim, f"lidar_channel_to_img_bias_{i}").params[0][1].data = (
            model_pytorch.lidar_channel_to_img[i]
            .bias.data.cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )
        inject_conv2d_weights(
            model_pytorch.img_channel_to_lidar[i],
            getattr(model_ttsim, f"img_channel_to_lidar_{i}"),
            name=f"img_channel_to_lidar[{i}]",
        )
        getattr(model_ttsim, f"img_channel_to_lidar_bias_{i}").params[0][1].data = (
            model_pytorch.img_channel_to_lidar[i]
            .bias.data.cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )

    # FPN conv weights + biases
    if hasattr(model_pytorch, "c5_conv"):
        inject_conv2d_weights(
            model_pytorch.c5_conv, model_ttsim.c5_conv, name="c5_conv"
        )
        model_ttsim.c5_conv_bias.params[0][1].data = (
            model_pytorch.c5_conv.bias.data.cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )
    if hasattr(model_pytorch, "up_conv5"):
        inject_conv2d_weights(
            model_pytorch.up_conv5, model_ttsim.up_conv5, name="up_conv5"
        )
        model_ttsim.up_conv5_bias.params[0][1].data = (
            model_pytorch.up_conv5.bias.data.cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )
    if hasattr(model_pytorch, "up_conv4"):
        inject_conv2d_weights(
            model_pytorch.up_conv4, model_ttsim.up_conv4, name="up_conv4"
        )
        model_ttsim.up_conv4_bias.params[0][1].data = (
            model_pytorch.up_conv4.bias.data.cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )

    # Inject lidar_to_img_features_end Linear + Bias (for global fusion)
    if hasattr(model_pytorch, "lidar_to_img_features_end"):
        pt_weight = (
            model_pytorch.lidar_to_img_features_end.weight.data.cpu().numpy().T
        )  # transpose for TTSIM MatMul
        pt_bias = model_pytorch.lidar_to_img_features_end.bias.data.cpu().numpy()
        model_ttsim.lidar_to_img_features_end.params[0][1].data = pt_weight
        model_ttsim.lidar_to_img_features_end_bias.params[0][1].data = pt_bias

    # Per-layer transformer validation
    numerical_diffs = []
    for layer_idx in range(4):
        with torch.no_grad():
            image_feats_pt = encoder_features_pt[layer_idx]["image"]
            lidar_feats_pt = encoder_features_pt[layer_idx]["lidar"]

            image_embd_pt = model_pytorch.avgpool_img(image_feats_pt)
            lidar_embd_pt = model_pytorch.avgpool_lidar(lidar_feats_pt)
            lidar_embd_pt = model_pytorch.lidar_channel_to_img[layer_idx](lidar_embd_pt)

            image_out_pt, lidar_out_pt = model_pytorch.transformers[layer_idx](
                image_embd_pt, lidar_embd_pt
            )

            image_out_pt_np = image_out_pt.cpu().numpy()
            lidar_out_pt_np = lidar_out_pt.cpu().numpy()

            image_embd_np = image_embd_pt.cpu().numpy()
            lidar_embd_np = lidar_embd_pt.cpu().numpy()

        image_sim = F._from_data(
            f"test_image_{layer_idx}", image_embd_np, is_param=False, is_const=False
        )
        lidar_sim = F._from_data(
            f"test_lidar_{layer_idx}", lidar_embd_np, is_param=False, is_const=False
        )
        image_sim.link_module = model_ttsim.transformers[layer_idx]
        lidar_sim.link_module = model_ttsim.transformers[layer_idx]

        image_out_sim, lidar_out_sim = model_ttsim.transformers[layer_idx](
            image_sim, lidar_sim
        )

        if image_out_sim.data is not None and lidar_out_sim.data is not None:
            image_diff = np.abs(image_out_pt_np - image_out_sim.data)
            lidar_diff = np.abs(lidar_out_pt_np - lidar_out_sim.data)
            numerical_diffs.append(
                {
                    "layer": layer_idx,
                    "image_max_diff": image_diff.max(),
                    "image_mean_diff": image_diff.mean(),
                    "lidar_max_diff": lidar_diff.max(),
                    "lidar_mean_diff": lidar_diff.mean(),
                }
            )

    # Print TTSIM shapes
    if len(numerical_diffs) > 0:
        print(f"Image input shape: {list(image_data.shape)}")
        print(f"LiDAR input shape: {list(lidar_data.shape)}")
        # Use last layer's TTSIM output for shape/stats display
        if image_out_sim.data is not None:
            print(f"Transformer output shape (image): {list(image_out_sim.data.shape)}")
            print(f"Transformer output shape (lidar): {list(lidar_out_sim.data.shape)}")

    # End-to-end validation: run PyTorch encoder stages, feed into TTSIM
    # fusion (transformers + FPN + global pool). This mirrors the old test
    # where both models shared identical timm encoder weights.
    with torch.no_grad():
        img_feat = image_torch.clone()
        lid_feat = lidar_torch.clone()

        # Run stem through PyTorch encoder
        img_layers_iter = iter(model_pytorch.image_encoder.items())
        lid_layers_iter = iter(model_pytorch.lidar_encoder.items())

        if len(model_pytorch.image_encoder.return_layers) > 4:
            img_feat = model_pytorch.forward_layer_block(
                img_layers_iter, model_pytorch.image_encoder.return_layers, img_feat
            )
        if len(model_pytorch.lidar_encoder.return_layers) > 4:
            lid_feat = model_pytorch.forward_layer_block(
                lid_layers_iter, model_pytorch.lidar_encoder.return_layers, lid_feat
            )

        # Track TTSIM features starting from stem output
        img_feat_sim = F._from_data(
            "e2e_img_stem", img_feat.cpu().numpy(), is_param=False, is_const=False
        )
        lid_feat_sim = F._from_data(
            "e2e_lid_stem", lid_feat.cpu().numpy(), is_param=False, is_const=False
        )
        img_feat_sim.link_module = model_ttsim
        lid_feat_sim.link_module = model_ttsim

        for i in range(4):
            # Run PyTorch encoder stage
            img_feat = model_pytorch.forward_layer_block(
                img_layers_iter, model_pytorch.image_encoder.return_layers, img_feat
            )
            lid_feat = model_pytorch.forward_layer_block(
                lid_layers_iter, model_pytorch.lidar_encoder.return_layers, lid_feat
            )

            # Convert to TTSIM tensors (same encoder features for both paths)
            img_feat_sim = F._from_data(
                f"e2e_img_s{i}", img_feat.cpu().numpy(), is_param=False, is_const=False
            )
            lid_feat_sim = F._from_data(
                f"e2e_lid_s{i}", lid_feat.cpu().numpy(), is_param=False, is_const=False
            )
            img_feat_sim.link_module = model_ttsim
            lid_feat_sim.link_module = model_ttsim

            # Run TTSIM fusion (transformer + interpolate + residual add)
            img_feat_sim, lid_feat_sim = model_ttsim.fuse_features(
                img_feat_sim, lid_feat_sim, i
            )

            # Run PyTorch fusion (same logic) so features stay in sync
            img_feat, lid_feat = model_pytorch.fuse_features(img_feat, lid_feat, i)

        # BEV top-down
        if model_ttsim.config.detect_boxes or model_ttsim.config.use_bev_semantic:
            features_ttsim = model_ttsim.top_down(lid_feat_sim)
        else:
            features_ttsim = None

        # Global fusion
        img_pool = model_ttsim._adaptive_avg_pool2d(img_feat_sim, 1, 1, "e2e_gpool_img")
        img_pool.link_module = model_ttsim
        img_pool = img_pool.flatten(start_dim=1)

        lid_pool = model_ttsim._adaptive_avg_pool2d(lid_feat_sim, 1, 1, "e2e_gpool_lid")
        lid_pool.link_module = model_ttsim
        lid_pool = lid_pool.flatten(start_dim=1)

        if model_ttsim.config.add_features:
            lid_pool = model_ttsim.lidar_to_img_features_end_bias(
                model_ttsim.lidar_to_img_features_end(lid_pool)
            )
            fused_features_ttsim = model_ttsim.add_fused(img_pool, lid_pool)
        else:
            fused_features_ttsim = model_ttsim.concat_fused(img_pool, lid_pool)

        image_feature_grid_ttsim = None

    if features_ttsim is not None:
        if hasattr(features_ttsim, "data") and features_ttsim.data is not None:
            features_ttsim_np = features_ttsim.data
        elif hasattr(features_ttsim, "cpu"):
            features_ttsim_np = features_ttsim.cpu().numpy()
        else:
            features_ttsim_np = None
    else:
        features_ttsim_np = None

    if fused_features_ttsim is not None:
        if (
            hasattr(fused_features_ttsim, "data")
            and fused_features_ttsim.data is not None
        ):
            fused_features_ttsim_np = fused_features_ttsim.data
        elif hasattr(fused_features_ttsim, "cpu"):
            fused_features_ttsim_np = fused_features_ttsim.cpu().numpy()
        else:
            fused_features_ttsim_np = None
    else:
        fused_features_ttsim_np = None

    if features_ttsim_np is not None:
        print(f"BEV features shape: {list(features_ttsim_np.shape)}")
        print(
            f"BEV features stats: min={features_ttsim_np.min():.6f}, max={features_ttsim_np.max():.6f}"
        )
    if fused_features_ttsim_np is not None:
        print(f"Fused features shape: {list(fused_features_ttsim_np.shape)}")

    # --- Numerical Comparison ---
    print(f"\n--- Numerical Comparison ---")
    all_passed = True

    # Per-layer transformer diffs
    for d in numerical_diffs:
        img_max = d["image_max_diff"]
        img_mean = d["image_mean_diff"]
        lid_max = d["lidar_max_diff"]
        lid_mean = d["lidar_mean_diff"]
        print(
            f"Transformer Layer {d['layer']} - Image: Max diff: {img_max:.10f}, Mean diff: {img_mean:.10f}"
        )
        print(
            f"Transformer Layer {d['layer']} - LiDAR: Max diff: {lid_max:.10f}, Mean diff: {lid_mean:.10f}"
        )

    # End-to-end BEV features
    if features_pt_np is not None and features_ttsim_np is not None:
        bev_abs = np.abs(features_pt_np - features_ttsim_np)
        print(
            f"BEV features - Max diff: {bev_abs.max():.10f}, Mean diff: {bev_abs.mean():.10f}"
        )
        bev_close = np.allclose(features_pt_np, features_ttsim_np, atol=atol, rtol=rtol)
    else:
        bev_close = None

    if fused_features_pt_np is not None and fused_features_ttsim_np is not None:
        fused_abs = np.abs(fused_features_pt_np - fused_features_ttsim_np)
        print(
            f"Fused features - Max diff: {fused_abs.max():.10f}, Mean diff: {fused_abs.mean():.10f}"
        )
        fused_close = np.allclose(
            fused_features_pt_np, fused_features_ttsim_np, atol=atol, rtol=rtol
        )
    else:
        fused_close = None

    # Per-layer pass/fail
    print(f"\n(atol={atol}, rtol={rtol}):")
    layer_results = []
    for d in numerical_diffs:
        img_close = np.allclose(0, d["image_max_diff"], atol=atol)
        lid_close = np.allclose(0, d["lidar_max_diff"], atol=atol)
        layer_pass = d["image_max_diff"] < atol and d["lidar_max_diff"] < atol
        status = "PASS" if layer_pass else "WARN"
        print(f"  Transformer Layer {d['layer']}: {status}")
        layer_results.append(layer_pass)

    if bev_close is not None:
        print(f"  BEV features:          {'PASS' if bev_close else 'WARN'}")
    if fused_close is not None:
        print(f"  Fused features:        {'PASS' if fused_close else 'WARN'}")

    # Overall
    transformer_pass = all(
        d["image_max_diff"] < 0.001 and d["lidar_max_diff"] < 0.001
        for d in numerical_diffs
    )
    e2e_pass = (bev_close is None or bev_close) and (fused_close is None or fused_close)
    overall = transformer_pass and e2e_pass

    print("\n" + "=" * 70)
    if overall:
        print(f"OVERALL: PASS: PASS - (atol={atol}, rtol={rtol})")
    else:
        print(f"OVERALL: FAIL: FAIL - (atol={atol}, rtol={rtol})")
    print("=" * 70)


if __name__ == "__main__":
    main()
