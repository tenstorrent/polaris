#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for TrajectoryHead.
Tests shape inference and numerical validation.
"""

import os
import sys

# Add DiffusionDrive to path for navsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
# Add TT-Polaris to path for ttsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.diffusiondrive.reference.torch_code.navsim.agents.diffusiondrive.transfuser_model_v2 import (
    TrajectoryHead as TrajectoryHead_PyTorch,
)
from navsim.agents.diffusiondrive.transfuser_model_v2_ttsim import (
    TrajectoryHead as TrajectoryHead_TTSIM,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

import ttsim.front.functional.op as F
from navsim.agents.diffusiondrive.reference.torch_code.navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position


def inject_linear_weights(ttsim_linear, ttsim_bias, pytorch_linear):
    """Inject weights from PyTorch Linear to TTSIM Linear+Bias."""
    weight = pytorch_linear.weight.detach().cpu().numpy().T
    bias = pytorch_linear.bias.detach().cpu().numpy()
    ttsim_linear.params[0][1].data = weight.astype(np.float32)
    ttsim_bias.params[0][1].data = bias.astype(np.float32)


def inject_layernorm_weights(ttsim_ln, pytorch_ln):
    """Inject weights from PyTorch LayerNorm to TTSIM LayerNorm."""
    if hasattr(pytorch_ln, "weight") and pytorch_ln.weight is not None:
        ttsim_ln.params[0][1].data = (
            pytorch_ln.weight.detach().cpu().numpy().astype(np.float32)
        )
    if hasattr(pytorch_ln, "bias") and pytorch_ln.bias is not None:
        ttsim_ln.params[1][1].data = (
            pytorch_ln.bias.detach().cpu().numpy().astype(np.float32)
        )


def inject_conv2d_weights(ttsim_conv, pytorch_conv):
    """Inject weights from PyTorch Conv2d to TTSIM Conv2d."""
    weight = pytorch_conv.weight.detach().cpu().numpy()
    ttsim_conv.params[0][1].data = weight.astype(np.float32)
    if pytorch_conv.bias is not None:
        bias = pytorch_conv.bias.detach().cpu().numpy()
        ttsim_conv.params[1][1].data = bias.astype(np.float32)


def inject_modulation_layer(ttsim_mod, pytorch_mod):
    """Inject ModulationLayer weights."""
    # pytorch_mod.scale_shift_mlp is Sequential([Mish(), Linear(condition_dims, embed_dims*2)])
    inject_linear_weights(
        ttsim_mod.scale_shift_linear,
        ttsim_mod.scale_shift_bias,
        pytorch_mod.scale_shift_mlp[1],
    )


def inject_multihead_attention_weights(
    ttsim_q_linear,
    ttsim_q_bias,
    ttsim_k_linear,
    ttsim_k_bias,
    ttsim_v_linear,
    ttsim_v_bias,
    ttsim_out_proj,
    ttsim_out_bias,
    pytorch_mha,
    d_model,
):
    """Inject weights from PyTorch MultiheadAttention to separate TTSIM Q, K, V projections."""
    # Split in_proj_weight into Q, K, V
    in_proj_weight = pytorch_mha.in_proj_weight.detach().cpu().numpy()
    in_proj_bias = pytorch_mha.in_proj_bias.detach().cpu().numpy()

    q_weight = in_proj_weight[:d_model, :].T
    k_weight = in_proj_weight[d_model : 2 * d_model, :].T
    v_weight = in_proj_weight[2 * d_model :, :].T

    q_bias = in_proj_bias[:d_model]
    k_bias = in_proj_bias[d_model : 2 * d_model]
    v_bias = in_proj_bias[2 * d_model :]

    # F.Linear stores weight in params[0][1].data, F.Bias stores bias in params[0][1].data
    ttsim_q_linear.params[0][1].data = q_weight.astype(np.float32)
    ttsim_q_bias.params[0][1].data = q_bias.astype(np.float32)

    ttsim_k_linear.params[0][1].data = k_weight.astype(np.float32)
    ttsim_k_bias.params[0][1].data = k_bias.astype(np.float32)

    ttsim_v_linear.params[0][1].data = v_weight.astype(np.float32)
    ttsim_v_bias.params[0][1].data = v_bias.astype(np.float32)

    # Inject output projection
    inject_linear_weights(ttsim_out_proj, ttsim_out_bias, pytorch_mha.out_proj)


def inject_decoder_layer_weights(ttsim_layer, pytorch_layer):
    """Inject weights for a single CustomTransformerDecoderLayer."""
    # BEV attention - access nested GridSampleCrossBEVAttention
    inject_linear_weights(
        ttsim_layer.cross_bev_attention.attention_weights_linear,
        ttsim_layer.cross_bev_attention.attention_weights_bias,
        pytorch_layer.cross_bev_attention.attention_weights,
    )
    inject_linear_weights(
        ttsim_layer.cross_bev_attention.output_proj_linear,
        ttsim_layer.cross_bev_attention.output_proj_bias,
        pytorch_layer.cross_bev_attention.output_proj,
    )

    # value_proj is Sequential([Conv2d(bias=True), ReLU])
    # In TTSIM, Conv2d doesn't have bias parameter, we use a separate Bias layer
    pytorch_conv = pytorch_layer.cross_bev_attention.value_proj[0]
    weight = pytorch_conv.weight.detach().cpu().numpy()
    ttsim_layer.cross_bev_attention.value_proj_conv.params[0][1].data = weight.astype(
        np.float32
    )
    # Inject bias separately into the Bias layer
    conv_bias = (
        pytorch_conv.bias.detach()
        .cpu()
        .numpy()
        .reshape(1, 256, 1, 1)
        .astype(np.float32)
    )
    ttsim_layer.cross_bev_attention.value_proj_bias.params[0][1].data = conv_bias

    # Cross agent attention - inject MultiheadAttention weights
    inject_multihead_attention_weights(
        ttsim_layer.cross_agent_q_linear,
        ttsim_layer.cross_agent_q_bias,
        ttsim_layer.cross_agent_k_linear,
        ttsim_layer.cross_agent_k_bias,
        ttsim_layer.cross_agent_v_linear,
        ttsim_layer.cross_agent_v_bias,
        ttsim_layer.cross_agent_out_proj,
        ttsim_layer.cross_agent_out_bias,
        pytorch_layer.cross_agent_attention,
        ttsim_layer.d_model,
    )

    # Cross ego attention - inject MultiheadAttention weights
    inject_multihead_attention_weights(
        ttsim_layer.cross_ego_q_linear,
        ttsim_layer.cross_ego_q_bias,
        ttsim_layer.cross_ego_k_linear,
        ttsim_layer.cross_ego_k_bias,
        ttsim_layer.cross_ego_v_linear,
        ttsim_layer.cross_ego_v_bias,
        ttsim_layer.cross_ego_out_proj,
        ttsim_layer.cross_ego_out_bias,
        pytorch_layer.cross_ego_attention,
        ttsim_layer.d_model,
    )

    # FFN - access Sequential layers
    inject_linear_weights(
        ttsim_layer.ffn_linear1, ttsim_layer.ffn_bias1, pytorch_layer.ffn[0]
    )
    inject_linear_weights(
        ttsim_layer.ffn_linear2, ttsim_layer.ffn_bias2, pytorch_layer.ffn[2]
    )

    # Layer norms
    inject_layernorm_weights(ttsim_layer.norm1, pytorch_layer.norm1)
    inject_layernorm_weights(ttsim_layer.norm2, pytorch_layer.norm2)
    inject_layernorm_weights(ttsim_layer.norm3, pytorch_layer.norm3)

    # Time modulation
    inject_modulation_layer(ttsim_layer.time_modulation, pytorch_layer.time_modulation)

    # Task decoder (DiffMotionPlanningRefinementModule)
    # plan_cls_branch is Sequential: [Linear, ReLU, LayerNorm, Linear, ReLU, LayerNorm, Linear]
    inject_linear_weights(
        ttsim_layer.task_decoder.plan_cls_linear1,
        ttsim_layer.task_decoder.plan_cls_bias1,
        pytorch_layer.task_decoder.plan_cls_branch[0],
    )
    inject_layernorm_weights(
        ttsim_layer.task_decoder.plan_cls_ln1,
        pytorch_layer.task_decoder.plan_cls_branch[2],
    )
    inject_linear_weights(
        ttsim_layer.task_decoder.plan_cls_linear2,
        ttsim_layer.task_decoder.plan_cls_bias2,
        pytorch_layer.task_decoder.plan_cls_branch[3],
    )
    inject_layernorm_weights(
        ttsim_layer.task_decoder.plan_cls_ln2,
        pytorch_layer.task_decoder.plan_cls_branch[5],
    )
    inject_linear_weights(
        ttsim_layer.task_decoder.plan_cls_linear3,
        ttsim_layer.task_decoder.plan_cls_bias3,
        pytorch_layer.task_decoder.plan_cls_branch[6],
    )

    # plan_reg_branch is Sequential: [Linear, ReLU, Linear, ReLU, Linear]
    inject_linear_weights(
        ttsim_layer.task_decoder.plan_reg_linear1,
        ttsim_layer.task_decoder.plan_reg_bias1,
        pytorch_layer.task_decoder.plan_reg_branch[0],
    )
    inject_linear_weights(
        ttsim_layer.task_decoder.plan_reg_linear2,
        ttsim_layer.task_decoder.plan_reg_bias2,
        pytorch_layer.task_decoder.plan_reg_branch[2],
    )
    inject_linear_weights(
        ttsim_layer.task_decoder.plan_reg_linear3,
        ttsim_layer.task_decoder.plan_reg_bias3,
        pytorch_layer.task_decoder.plan_reg_branch[4],
    )


def inject_trajectory_head_weights(ttsim_model, pytorch_model):
    """Inject all TrajectoryHead weights from PyTorch to TTSIM."""
    # TTSIM uses per-iteration unrolled modules (iter0, iter1).
    # PyTorch shares weights across iterations, so inject the same weights into each.
    for it in range(2):
        p = f"iter{it}"

        # Plan anchor encoder - Sequential([Linear(512,256), ReLU, LayerNorm(256), Linear(256,256)])
        inject_linear_weights(
            getattr(ttsim_model, f"plan_enc_linear1_{p}"),
            getattr(ttsim_model, f"plan_enc_bias1_{p}"),
            pytorch_model.plan_anchor_encoder[0],
        )
        inject_layernorm_weights(
            getattr(ttsim_model, f"plan_enc_ln_{p}"),
            pytorch_model.plan_anchor_encoder[2],
        )
        inject_linear_weights(
            getattr(ttsim_model, f"plan_enc_linear2_{p}"),
            getattr(ttsim_model, f"plan_enc_bias2_{p}"),
            pytorch_model.plan_anchor_encoder[3],
        )

        # Time MLP - Sequential([SinusoidalPosEmb, Linear(d_model, d_model*4), Mish, Linear(d_model*4, d_model)])
        inject_linear_weights(
            getattr(ttsim_model, f"time_linear1_{p}"),
            getattr(ttsim_model, f"time_bias1_{p}"),
            pytorch_model.time_mlp[1],
        )
        inject_linear_weights(
            getattr(ttsim_model, f"time_linear2_{p}"),
            getattr(ttsim_model, f"time_bias2_{p}"),
            pytorch_model.time_mlp[3],
        )

        # Decoder layers for this iteration
        ttsim_decoder = getattr(ttsim_model, f"diff_decoder_{p}")
        for i, (ttsim_layer, pytorch_layer) in enumerate(
            zip(ttsim_decoder.layers, pytorch_model.diff_decoder.layers)
        ):
            inject_decoder_layer_weights(ttsim_layer, pytorch_layer)


def main():
    print("=" * 70)
    print("TrajectoryHead Validation")
    print("=" * 70)

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    config = TransfuserConfig()
    num_poses = 8
    d_model = 256
    d_ffn = 512
    batch_size = 1
    num_agents = 3
    bev_h = 8
    bev_w = 8
    # Create plan anchor file
    plan_anchor_path = "test_plan_anchor.npy"
    # Plan anchor shape: (20, num_poses, 2) for (num_modes, num_poses, 2)
    # where 2 = (x, y) - heading is predicted by the model
    plan_anchor = np.random.randn(20, num_poses, 2).astype(np.float32)
    np.save(plan_anchor_path, plan_anchor)
    print(f"\nCreated plan anchor: {plan_anchor_path}")

    # Create models
    print("\n--- Creating Models ---")
    model_pytorch = TrajectoryHead_PyTorch(
        num_poses=num_poses,
        d_ffn=d_ffn,
        d_model=d_model,
        plan_anchor_path=plan_anchor_path,
        config=config,
    )
    model_pytorch.eval()

    model_ttsim = TrajectoryHead_TTSIM(
        num_poses=num_poses,
        d_ffn=d_ffn,
        d_model=d_model,
        plan_anchor_path=plan_anchor_path,
        config=config,
    )

    # Inject weights
    print("\n--- Injecting Weights ---")
    try:
        inject_trajectory_head_weights(model_ttsim, model_pytorch)
        print("PASS: Weight injection complete")
    except Exception as e:
        print(f"FAIL: Weight injection failed: {e}")
        import traceback

        traceback.print_exc()
        if os.path.exists(plan_anchor_path):
            os.remove(plan_anchor_path)
        return

    # Generate inputs
    print("\n--- Generating Inputs ---")
    ego_query_data = np.random.randn(batch_size, 1, d_model).astype(np.float32)
    agents_query_data = np.random.randn(batch_size, num_agents, d_model).astype(
        np.float32
    )
    bev_feature_data = np.random.randn(batch_size, 256, bev_h, bev_w).astype(np.float32)
    status_encoding_data = np.random.randn(batch_size, 1, d_model).astype(np.float32)

    bev_spatial_shape = (bev_h, bev_w)

    print(f"  Ego query: {ego_query_data.shape}")
    print(f"  Agents query: {agents_query_data.shape}")
    print(f"  BEV feature: {bev_feature_data.shape}")
    print(f"  Status encoding: {status_encoding_data.shape}")

    # PyTorch full 2-step denoising loop (matching TTSIM forward_test)
    print("\n--- PyTorch Forward Pass (full 2-step loop) ---")
    with torch.no_grad():
        ego_query_torch = torch.from_numpy(ego_query_data)
        agents_query_torch = torch.from_numpy(agents_query_data)
        bev_feature_torch = torch.from_numpy(bev_feature_data)
        status_encoding_torch = torch.from_numpy(status_encoding_data)

        try:
            bs = ego_query_torch.shape[0]

            # 1. plan_anchor → norm_odo → add_noise
            plan_anchor = model_pytorch.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
            img = model_pytorch.norm_odo(plan_anchor)

            # Generate shared noise matching img shape [bs, 20, 8, 3]
            noise_np = np.random.randn(*img.shape).astype(np.float32)
            noise_torch = torch.from_numpy(noise_np)
            print(f"  Noise shape: {noise_np.shape} (matches img after norm_odo)")

            # MockScheduler.add_noise: img = img + 0.1 * noise
            img = img + 0.1 * noise_torch

            ego_fut_mode = img.shape[1]

            # 2. Two-step denoising loop (timesteps [10, 0])
            roll_timesteps = [10, 0]
            for k in roll_timesteps:
                x_boxes = torch.clamp(img, min=-1, max=1)
                noisy_traj_points = model_pytorch.denorm_odo(x_boxes)

                traj_pos_embed = gen_sineembed_for_position(
                    noisy_traj_points, hidden_dim=64
                )
                traj_pos_embed = traj_pos_embed.flatten(-2)
                traj_feature = model_pytorch.plan_anchor_encoder(traj_pos_embed)
                traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

                timesteps = torch.tensor([k], dtype=torch.long).expand(bs)
                time_embed = model_pytorch.time_mlp(timesteps)
                time_embed = time_embed.view(bs, 1, -1)

                poses_reg_list, poses_cls_list = model_pytorch.diff_decoder(
                    traj_feature,
                    noisy_traj_points,
                    bev_feature_torch,
                    bev_spatial_shape,
                    agents_query_torch,
                    ego_query_torch,
                    time_embed,
                    status_encoding_torch,
                    None,
                )

                poses_reg = poses_reg_list[-1]
                poses_cls = poses_cls_list[-1]
                x_start = poses_reg[..., :2]
                x_start = model_pytorch.norm_odo(x_start)

                # MockScheduler.step: img = 0.9 * x_start + 0.1 * img
                img = 0.9 * x_start + 0.1 * img

            # 3. Mode selection: argmax + gather
            mode_idx = poses_cls.argmax(dim=-1)
            mode_idx = mode_idx[..., None, None, None].repeat(1, 1, num_poses, 3)
            best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
            trajectory_pytorch = best_reg.cpu().numpy()
            print(f"PASS: PyTorch output shape: {trajectory_pytorch.shape}")
        except Exception as e:
            print(f"FAIL: PyTorch forward failed: {e}")
            import traceback

            traceback.print_exc()
            if os.path.exists(plan_anchor_path):
                os.remove(plan_anchor_path)
            return

    # TTSIM forward pass (full 2-step denoising loop)
    print("\n--- TTSIM Forward Pass (full 2-step loop) ---")
    ego_query_ttsim = F._from_data("ego_query", ego_query_data)
    agents_query_ttsim = F._from_data("agents_query", agents_query_data)
    bev_feature_ttsim = F._from_data("bev_feature", bev_feature_data)
    status_encoding_ttsim = F._from_data("status_encoding", status_encoding_data)

    # Create noise SimTensor from the same numpy array
    noise_ttsim = F._from_data("noise", noise_np)

    ego_query_ttsim.link_module = model_ttsim

    try:
        output_ttsim = model_ttsim(
            ego_query_ttsim,
            agents_query_ttsim,
            bev_feature_ttsim,
            bev_spatial_shape,
            status_encoding_ttsim,
            targets=None,
            global_img=None,
            noise=noise_ttsim,
        )
        trajectory_ttsim_tensor = output_ttsim["trajectory"]
        print(f"PASS: TTSIM output shape: {trajectory_ttsim_tensor.shape}")
    except Exception as e:
        print(f"FAIL: TTSIM forward failed: {e}")
        import traceback

        traceback.print_exc()
        if os.path.exists(plan_anchor_path):
            os.remove(plan_anchor_path)
        return

    # Compare outputs - check if data is available
    print("\n--- Numerical Comparison ---")
    if trajectory_ttsim_tensor.data is not None:
        trajectory_ttsim = trajectory_ttsim_tensor.data
        diff = np.abs(trajectory_pytorch - trajectory_ttsim)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")

        atol = 1e-4
        rtol = 1e-4
        is_close = np.allclose(
            trajectory_pytorch, trajectory_ttsim, atol=atol, rtol=rtol
        )

        print("\n" + "=" * 70)
        if is_close:
            print(f"OVERALL: PASS - np.allclose(atol={atol}, rtol={rtol})")
        else:
            print(
                f"OVERALL: FAIL - np.allclose(atol={atol}, rtol={rtol}) returned False"
            )
            print(f"  Max abs diff: {max_diff:.6f}, Mean abs diff: {mean_diff:.6f}")
            # Show per-element breakdown of worst mismatches
            flat_diff = diff.flatten()
            worst_idx = np.argsort(flat_diff)[-5:][::-1]
            print(f"  Top-5 mismatches:")
            for idx in worst_idx:
                coord = np.unravel_index(idx, diff.shape)
                print(
                    f"    {coord}: pytorch={trajectory_pytorch[coord]:.6f} ttsim={trajectory_ttsim[coord]:.6f} diff={flat_diff[idx]:.6f}"
                )
        print("=" * 70)
    else:
        print("  WARN: SKIPPED: No TTSIM data available (graph created successfully)")
        print("=" * 70)

    # Cleanup
    if os.path.exists(plan_anchor_path):
        os.remove(plan_anchor_path)
        print(f"\nCleaned up: {plan_anchor_path}")

    print()


if __name__ == "__main__":
    main()
