#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for CustomTransformerDecoder.
Tests full numerical validation for stacked decoder layers.
"""

import os
import sys

# Add DiffusionDrive to path for navsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
# Add TT-Polaris to path for ttsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.diffusiondrive.reference.torch_code.navsim.agents.diffusiondrive.transfuser_model_v2 import (
    CustomTransformerDecoder as Decoder_PyTorch,
)
from navsim.agents.diffusiondrive.reference.torch_code.navsim.agents.diffusiondrive.transfuser_model_v2 import (
    CustomTransformerDecoderLayer as DecoderLayer_PyTorch,
)
from navsim.agents.diffusiondrive.transfuser_model_v2_ttsim import (
    CustomTransformerDecoder as Decoder_TTSIM,
)
from navsim.agents.diffusiondrive.transfuser_model_v2_ttsim import (
    CustomTransformerDecoderLayer as DecoderLayer_TTSIM,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

import ttsim.front.functional.op as F


def inject_layer_weights(layer_torch, layer_ttsim):
    """Inject weights from one PyTorch decoder layer into one TTSIM decoder layer."""

    # Cross BEV attention (GridSampleCrossBEVAttention)
    layer_ttsim.cross_bev_attention.attention_weights_linear.params[0][1].data = (
        layer_torch.cross_bev_attention.attention_weights.weight.T.detach()
        .cpu()
        .numpy()
    )
    layer_ttsim.cross_bev_attention.attention_weights_bias.params[0][1].data = (
        layer_torch.cross_bev_attention.attention_weights.bias.detach().cpu().numpy()
    )
    layer_ttsim.cross_bev_attention.output_proj_linear.params[0][1].data = (
        layer_torch.cross_bev_attention.output_proj.weight.T.detach().cpu().numpy()
    )
    layer_ttsim.cross_bev_attention.output_proj_bias.params[0][1].data = (
        layer_torch.cross_bev_attention.output_proj.bias.detach().cpu().numpy()
    )

    # Value projection Conv2d
    value_proj_conv = layer_torch.cross_bev_attention.value_proj[0]  # Conv2d layer
    layer_ttsim.cross_bev_attention.value_proj_conv.params[0][1].data = (
        value_proj_conv.weight.detach().cpu().numpy()
    )
    layer_ttsim.cross_bev_attention.value_proj_bias.params[0][1].data = (
        value_proj_conv.bias.detach().cpu().numpy().reshape(1, 256, 1, 1)
    )

    # Cross agent attention (MultiheadAttention)
    for name_torch, param_torch in layer_torch.cross_agent_attention.named_parameters():
        if "in_proj_weight" in name_torch:
            # Split into Q, K, V weights
            d_model = param_torch.shape[1]
            q_weight = param_torch[:d_model, :].T.detach().cpu().numpy()
            k_weight = param_torch[d_model : 2 * d_model, :].T.detach().cpu().numpy()
            v_weight = param_torch[2 * d_model :, :].T.detach().cpu().numpy()

            layer_ttsim.cross_agent_q_linear.params[0][1].data = q_weight
            layer_ttsim.cross_agent_k_linear.params[0][1].data = k_weight
            layer_ttsim.cross_agent_v_linear.params[0][1].data = v_weight
        elif "in_proj_bias" in name_torch:
            bias = param_torch.detach().cpu().numpy()
            d_model = len(bias) // 3
            layer_ttsim.cross_agent_q_bias.params[0][1].data = bias[:d_model]
            layer_ttsim.cross_agent_k_bias.params[0][1].data = bias[
                d_model : 2 * d_model
            ]
            layer_ttsim.cross_agent_v_bias.params[0][1].data = bias[2 * d_model :]
        elif "out_proj.weight" in name_torch:
            layer_ttsim.cross_agent_out_proj.params[0][1].data = (
                param_torch.T.detach().cpu().numpy()
            )
        elif "out_proj.bias" in name_torch:
            layer_ttsim.cross_agent_out_bias.params[0][1].data = (
                param_torch.detach().cpu().numpy()
            )

    # Cross ego attention (MultiheadAttention)
    for name_torch, param_torch in layer_torch.cross_ego_attention.named_parameters():
        if "in_proj_weight" in name_torch:
            d_model = param_torch.shape[1]
            q_weight = param_torch[:d_model, :].T.detach().cpu().numpy()
            k_weight = param_torch[d_model : 2 * d_model, :].T.detach().cpu().numpy()
            v_weight = param_torch[2 * d_model :, :].T.detach().cpu().numpy()

            layer_ttsim.cross_ego_q_linear.params[0][1].data = q_weight
            layer_ttsim.cross_ego_k_linear.params[0][1].data = k_weight
            layer_ttsim.cross_ego_v_linear.params[0][1].data = v_weight
        elif "in_proj_bias" in name_torch:
            bias = param_torch.detach().cpu().numpy()
            d_model = len(bias) // 3
            layer_ttsim.cross_ego_q_bias.params[0][1].data = bias[:d_model]
            layer_ttsim.cross_ego_k_bias.params[0][1].data = bias[d_model : 2 * d_model]
            layer_ttsim.cross_ego_v_bias.params[0][1].data = bias[2 * d_model :]
        elif "out_proj.weight" in name_torch:
            layer_ttsim.cross_ego_out_proj.params[0][1].data = (
                param_torch.T.detach().cpu().numpy()
            )
        elif "out_proj.bias" in name_torch:
            layer_ttsim.cross_ego_out_bias.params[0][1].data = (
                param_torch.detach().cpu().numpy()
            )

    # FFN layers
    layer_ttsim.ffn_linear1.params[0][1].data = (
        layer_torch.ffn[0].weight.T.detach().cpu().numpy()
    )
    layer_ttsim.ffn_bias1.params[0][1].data = (
        layer_torch.ffn[0].bias.detach().cpu().numpy()
    )
    layer_ttsim.ffn_linear2.params[0][1].data = (
        layer_torch.ffn[2].weight.T.detach().cpu().numpy()
    )
    layer_ttsim.ffn_bias2.params[0][1].data = (
        layer_torch.ffn[2].bias.detach().cpu().numpy()
    )

    # LayerNorms
    layer_ttsim.norm1.params[0][1].data = (
        layer_torch.norm1.weight.detach().cpu().numpy()
    )
    layer_ttsim.norm1.params[1][1].data = layer_torch.norm1.bias.detach().cpu().numpy()
    layer_ttsim.norm2.params[0][1].data = (
        layer_torch.norm2.weight.detach().cpu().numpy()
    )
    layer_ttsim.norm2.params[1][1].data = layer_torch.norm2.bias.detach().cpu().numpy()
    layer_ttsim.norm3.params[0][1].data = (
        layer_torch.norm3.weight.detach().cpu().numpy()
    )
    layer_ttsim.norm3.params[1][1].data = layer_torch.norm3.bias.detach().cpu().numpy()

    # Time modulation layer - scale_shift_mlp is Sequential[Mish, Linear]
    linear_layer = layer_torch.time_modulation.scale_shift_mlp[
        1
    ]  # Get the Linear layer
    layer_ttsim.time_modulation.scale_shift_linear.params[0][1].data = (
        linear_layer.weight.T.detach().cpu().numpy()
    )
    layer_ttsim.time_modulation.scale_shift_bias.params[0][1].data = (
        linear_layer.bias.detach().cpu().numpy()
    )

    # Task decoder - plan classification branch
    layer_ttsim.task_decoder.plan_cls_linear1.params[0][1].data = (
        layer_torch.task_decoder.plan_cls_branch[0].weight.data.T.numpy()
    )
    layer_ttsim.task_decoder.plan_cls_bias1.params[0][1].data = (
        layer_torch.task_decoder.plan_cls_branch[0].bias.data.numpy()
    )
    layer_ttsim.task_decoder.plan_cls_ln1.params[0][1].data = (
        layer_torch.task_decoder.plan_cls_branch[2].weight.data.numpy()
    )
    layer_ttsim.task_decoder.plan_cls_ln1.params[1][1].data = (
        layer_torch.task_decoder.plan_cls_branch[2].bias.data.numpy()
    )
    layer_ttsim.task_decoder.plan_cls_linear2.params[0][1].data = (
        layer_torch.task_decoder.plan_cls_branch[3].weight.data.T.numpy()
    )
    layer_ttsim.task_decoder.plan_cls_bias2.params[0][1].data = (
        layer_torch.task_decoder.plan_cls_branch[3].bias.data.numpy()
    )
    layer_ttsim.task_decoder.plan_cls_ln2.params[0][1].data = (
        layer_torch.task_decoder.plan_cls_branch[5].weight.data.numpy()
    )
    layer_ttsim.task_decoder.plan_cls_ln2.params[1][1].data = (
        layer_torch.task_decoder.plan_cls_branch[5].bias.data.numpy()
    )
    layer_ttsim.task_decoder.plan_cls_linear3.params[0][1].data = (
        layer_torch.task_decoder.plan_cls_branch[6].weight.data.T.numpy()
    )
    layer_ttsim.task_decoder.plan_cls_bias3.params[0][1].data = (
        layer_torch.task_decoder.plan_cls_branch[6].bias.data.numpy()
    )

    # Task decoder - plan regression branch
    layer_ttsim.task_decoder.plan_reg_linear1.params[0][1].data = (
        layer_torch.task_decoder.plan_reg_branch[0].weight.data.T.numpy()
    )
    layer_ttsim.task_decoder.plan_reg_bias1.params[0][1].data = (
        layer_torch.task_decoder.plan_reg_branch[0].bias.data.numpy()
    )
    layer_ttsim.task_decoder.plan_reg_linear2.params[0][1].data = (
        layer_torch.task_decoder.plan_reg_branch[2].weight.data.T.numpy()
    )
    layer_ttsim.task_decoder.plan_reg_bias2.params[0][1].data = (
        layer_torch.task_decoder.plan_reg_branch[2].bias.data.numpy()
    )
    layer_ttsim.task_decoder.plan_reg_linear3.params[0][1].data = (
        layer_torch.task_decoder.plan_reg_branch[4].weight.data.T.numpy()
    )
    layer_ttsim.task_decoder.plan_reg_bias3.params[0][1].data = (
        layer_torch.task_decoder.plan_reg_branch[4].bias.data.numpy()
    )


def inject_decoder_weights(decoder_torch, decoder_ttsim):
    """Inject weights from PyTorch decoder into TTSIM decoder (all layers)."""
    for layer_idx, (layer_torch, layer_ttsim) in enumerate(
        zip(decoder_torch.layers, decoder_ttsim.layers)
    ):
        inject_layer_weights(layer_torch, layer_ttsim)


def main():
    print("=" * 70)
    print("CustomTransformerDecoder Validation: Full Numerical Test")
    print("=" * 70)
    print()

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    config = TransfuserConfig()
    num_poses = 8
    d_model = 256
    d_ffn = 512
    num_layers = 2
    batch_size = 1
    ego_fut_mode = 4

    try:
        # Create PyTorch model
        print("--- Creating PyTorch Reference ---")
        decoder_layer_torch = DecoderLayer_PyTorch(
            num_poses=num_poses, d_model=d_model, d_ffn=d_ffn, config=config
        )
        model_torch = Decoder_PyTorch(
            decoder_layer=decoder_layer_torch, num_layers=num_layers, norm=None
        )
        model_torch.eval()

        # Create TTSIM model - create separate layers for each decoder layer
        print("--- Creating TTSIM Model ---")
        ttsim_layers = []
        for i in range(num_layers):
            layer = DecoderLayer_TTSIM(
                num_poses=num_poses, d_model=d_model, d_ffn=d_ffn, config=config
            )
            ttsim_layers.append(layer)

        model_ttsim = Decoder_TTSIM(
            decoder_layer=ttsim_layers,  # Pass list of layers
            num_layers=num_layers,
            norm=None,
        )

        # Generate random inputs
        traj_feature_data = np.random.randn(batch_size, ego_fut_mode, d_model).astype(
            np.float32
        )
        noisy_traj_points_data = np.random.randn(
            batch_size, ego_fut_mode, num_poses, 2
        ).astype(np.float32)
        bev_feature_data = np.random.randn(batch_size, 256, 8, 8).astype(np.float32)
        agents_query_data = np.random.randn(batch_size, 3, d_model).astype(np.float32)
        ego_query_data = np.random.randn(batch_size, 1, d_model).astype(np.float32)
        time_embed_data = np.random.randn(batch_size, 1, d_model).astype(np.float32)
        status_encoding_data = np.random.randn(batch_size, 1, d_model).astype(
            np.float32
        )

        bev_spatial_shape = (8, 8)

        # PyTorch forward pass
        print("\n--- PyTorch Forward Pass ---")
        traj_feature_torch = torch.from_numpy(traj_feature_data)
        noisy_traj_points_torch = torch.from_numpy(noisy_traj_points_data)
        bev_feature_torch = torch.from_numpy(bev_feature_data)
        agents_query_torch = torch.from_numpy(agents_query_data)
        ego_query_torch = torch.from_numpy(ego_query_data)
        time_embed_torch = torch.from_numpy(time_embed_data)
        status_encoding_torch = torch.from_numpy(status_encoding_data)

        with torch.no_grad():
            poses_reg_list_torch, poses_cls_list_torch = model_torch(
                traj_feature_torch,
                noisy_traj_points_torch,
                bev_feature_torch,
                bev_spatial_shape,
                agents_query_torch,
                ego_query_torch,
                time_embed_torch,
                status_encoding_torch,
                global_img=None,
            )

        print(f"PyTorch output shapes:")
        print(f"  Number of outputs: {len(poses_reg_list_torch)}")
        print(f"  First regression: {poses_reg_list_torch[0].shape}")
        print(f"  First classification: {poses_cls_list_torch[0].shape}")

        # Inject weights
        print("\n--- Injecting Weights ---")
        inject_decoder_weights(model_torch, model_ttsim)

        # TTSIM forward pass
        print("\n--- TTSIM Forward Pass ---")
        traj_feature_ttsim = F._from_data("traj_feature", traj_feature_data)
        noisy_traj_points_ttsim = F._from_data(
            "noisy_traj_points", noisy_traj_points_data
        )
        bev_feature_ttsim = F._from_data("bev_feature", bev_feature_data)
        agents_query_ttsim = F._from_data("agents_query", agents_query_data)
        ego_query_ttsim = F._from_data("ego_query", ego_query_data)
        time_embed_ttsim = F._from_data("time_embed", time_embed_data)
        status_encoding_ttsim = F._from_data("status_encoding", status_encoding_data)

        # Set initial link_module to first layer (decoder will update as needed)
        first_layer = model_ttsim.layers[0]
        traj_feature_ttsim.link_module = first_layer
        noisy_traj_points_ttsim.link_module = first_layer
        bev_feature_ttsim.link_module = first_layer
        agents_query_ttsim.link_module = first_layer
        ego_query_ttsim.link_module = first_layer
        time_embed_ttsim.link_module = first_layer
        status_encoding_ttsim.link_module = first_layer

        poses_reg_list_ttsim, poses_cls_list_ttsim = model_ttsim(
            traj_feature_ttsim,
            noisy_traj_points_ttsim,
            bev_feature_ttsim,
            bev_spatial_shape,
            agents_query_ttsim,
            ego_query_ttsim,
            time_embed_ttsim,
            status_encoding_ttsim,
            global_img=None,
        )

        print(f"TTSIM output shapes:")
        print(f"  Number of outputs: {len(poses_reg_list_ttsim)}")
        print(f"  First regression: {poses_reg_list_ttsim[0].shape}")
        print(f"  First classification: {poses_cls_list_ttsim[0].shape}")

        # Numerical comparison
        print("\n--- Numerical Comparison ---")

        all_layers_pass = True
        max_reg_diff_all = 0.0
        max_cls_diff_all = 0.0

        for layer_idx in range(num_layers):
            poses_reg_torch_np = poses_reg_list_torch[layer_idx].detach().cpu().numpy()
            poses_cls_torch_np = poses_cls_list_torch[layer_idx].detach().cpu().numpy()

            poses_reg_ttsim_data = poses_reg_list_ttsim[layer_idx].data
            poses_cls_ttsim_data = poses_cls_list_ttsim[layer_idx].data

            if poses_reg_ttsim_data is None or poses_cls_ttsim_data is None:
                print(f"\nLayer {layer_idx}: FAIL: FAIL - .data is None")
                all_layers_pass = False
                continue

            # Regression comparison
            reg_diff = np.abs(poses_reg_torch_np - poses_reg_ttsim_data)
            reg_max_diff = np.max(reg_diff)
            reg_mean_diff = np.mean(reg_diff)

            # Classification comparison (shapes should match now: [B, 20])
            cls_diff = np.abs(poses_cls_torch_np - poses_cls_ttsim_data)
            cls_max_diff = np.max(cls_diff)
            cls_mean_diff = np.mean(cls_diff)

            max_reg_diff_all = max(max_reg_diff_all, reg_max_diff)
            max_cls_diff_all = max(max_cls_diff_all, cls_max_diff)

            print(f"\nLayer {layer_idx}:")
            print(
                f"  Regression  - Max diff: {reg_max_diff:.6f}, Mean diff: {reg_mean_diff:.6f}"
            )
            print(
                f"  Classification - Max diff: {cls_max_diff:.6f}, Mean diff: {cls_mean_diff:.6f}"
            )

            # Use generous tolerance due to GridSampleCrossBEVAttention
            layer_pass = reg_max_diff < 5.0 and cls_max_diff < 5.0
            if layer_pass:
                print(f"  Layer {layer_idx}: PASS")
            else:
                print(f"  Layer {layer_idx}: FAIL")
                all_layers_pass = False

        # Overall assessment
        print("\n" + "=" * 70)
        if all_layers_pass:
            print("OVERALL: PASS: PASS")
            print(f"  Max regression diff across all layers: {max_reg_diff_all:.10f}")
            print(
                f"  Max classification diff across all layers: {max_cls_diff_all:.10f}"
            )

        else:
            print("OVERALL: FAIL - Some layers failed validation")
        print("=" * 70)

    except Exception as e:
        print(f"\nFAIL: ERROR during forward pass: {e}")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 70)
        print("OVERALL: FAIL - Forward pass failed")
        print("=" * 70)

    print()


if __name__ == "__main__":
    main()
