#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for GPT module (Full Transformer).
Tests shape inference and numerical equivalence between PyTorch and TTSIM.
"""

import warnings

warnings.filterwarnings("ignore")


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

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone_ttsim import GPT as GPT_TTSIM

import ttsim.front.functional.op as F


class SelfAttention_PyTorch(nn.Module):
    """PyTorch SelfAttention."""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_embd = n_embd
        self.n_head = n_head

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        b, t, c = x.size()

        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))

        return y


class Block_PyTorch(nn.Module):
    """PyTorch Block."""

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention_PyTorch(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT_PyTorch(nn.Module):
    """PyTorch GPT transformer."""

    def __init__(self, n_embd, config, lidar_time_frames):
        super().__init__()
        self.n_embd = n_embd
        self.config = config

        pos_emb_size = (
            1 * config.img_vert_anchors * config.img_horz_anchors
            + lidar_time_frames * config.lidar_vert_anchors * config.lidar_horz_anchors
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, pos_emb_size, n_embd))

        self.drop = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.ModuleList(
            [
                Block_PyTorch(
                    n_embd,
                    config.n_head,
                    config.block_exp,
                    config.attn_pdrop,
                    config.resid_pdrop,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, image_tensor, lidar_tensor):
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        # Reshape
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous()
        image_tensor = image_tensor.view(bz, -1, self.n_embd)

        lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous()
        lidar_tensor = lidar_tensor.view(bz, -1, self.n_embd)

        # Concatenate and add positional embedding
        token_embeddings = torch.cat([image_tensor, lidar_tensor], dim=1)
        x = token_embeddings + self.pos_emb
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Split
        img_spatial_len = img_h * img_w
        image_out = x[:, :img_spatial_len, :]
        lidar_out = x[:, img_spatial_len:, :]

        # Reshape back
        image_tensor_out = image_out.view(bz, img_h, img_w, -1)
        image_tensor_out = image_tensor_out.permute(0, 3, 1, 2)

        lidar_tensor_out = lidar_out.view(bz, lidar_h, lidar_w, -1)
        lidar_tensor_out = lidar_tensor_out.permute(0, 3, 1, 2)

        return image_tensor_out, lidar_tensor_out


def inject_gpt_weights(ttsim_gpt, pytorch_gpt):
    """Inject PyTorch GPT weights into TTSIM GPT."""
    # Positional embedding
    ttsim_gpt.pos_emb.data = pytorch_gpt.pos_emb.data.numpy()

    # LayerNorm final
    ttsim_gpt.ln_f.params[0][1].data = pytorch_gpt.ln_f.weight.data.numpy()
    ttsim_gpt.ln_f.params[1][1].data = pytorch_gpt.ln_f.bias.data.numpy()

    # Blocks
    for i, (ttsim_block, pt_block) in enumerate(
        zip(ttsim_gpt.blocks, pytorch_gpt.blocks)
    ):
        # LayerNorm
        ttsim_block.ln1.params[0][1].data = pt_block.ln1.weight.data.numpy()
        ttsim_block.ln1.params[1][1].data = pt_block.ln1.bias.data.numpy()
        ttsim_block.ln2.params[0][1].data = pt_block.ln2.weight.data.numpy()
        ttsim_block.ln2.params[1][1].data = pt_block.ln2.bias.data.numpy()

        # Attention
        ttsim_block.attn.key.params[0][1].data = pt_block.attn.key.weight.data.T.numpy()
        ttsim_block.attn.key_bias.params[0][
            1
        ].data = pt_block.attn.key.bias.data.numpy()
        ttsim_block.attn.query.params[0][
            1
        ].data = pt_block.attn.query.weight.data.T.numpy()
        ttsim_block.attn.query_bias.params[0][
            1
        ].data = pt_block.attn.query.bias.data.numpy()
        ttsim_block.attn.value.params[0][
            1
        ].data = pt_block.attn.value.weight.data.T.numpy()
        ttsim_block.attn.value_bias.params[0][
            1
        ].data = pt_block.attn.value.bias.data.numpy()
        ttsim_block.attn.proj.params[0][
            1
        ].data = pt_block.attn.proj.weight.data.T.numpy()
        ttsim_block.attn.proj_bias.params[0][
            1
        ].data = pt_block.attn.proj.bias.data.numpy()

        # MLP
        ttsim_block.mlp_fc1.params[0][1].data = pt_block.mlp[0].weight.data.T.numpy()
        ttsim_block.mlp_fc1_bias.params[0][1].data = pt_block.mlp[0].bias.data.numpy()
        ttsim_block.mlp_fc2.params[0][1].data = pt_block.mlp[2].weight.data.T.numpy()
        ttsim_block.mlp_fc2_bias.params[0][1].data = pt_block.mlp[2].bias.data.numpy()


def main():
    print("=" * 70)
    print("TEST: GPT - Full Transformer (Multi-modal Fusion)")
    print("=" * 70)

    # Configuration
    config = TransfuserConfig()
    config.n_layer = 2
    config.n_head = 4
    config.block_exp = 4
    config.attn_pdrop = 0.1
    config.resid_pdrop = 0.1
    config.embd_pdrop = 0.1
    config.img_vert_anchors = 8
    config.img_horz_anchors = 32
    config.lidar_vert_anchors = 8
    config.lidar_horz_anchors = 8

    n_embd = 128
    batch_size = 2
    lidar_time_frames = 1

    # Create test inputs
    img_h, img_w = config.img_vert_anchors, config.img_horz_anchors
    lidar_h, lidar_w = config.lidar_vert_anchors, config.lidar_horz_anchors

    image_features = np.random.randn(batch_size, n_embd, img_h, img_w).astype(
        np.float32
    )
    lidar_features = np.random.randn(batch_size, n_embd, lidar_h, lidar_w).astype(
        np.float32
    )

    # PyTorch model
    print("\n--- PyTorch GPT ---")
    model_pt = GPT_PyTorch(n_embd, config, lidar_time_frames)
    model_pt.eval()

    with torch.no_grad():
        img_pt = torch.from_numpy(image_features)
        lidar_pt = torch.from_numpy(lidar_features)
        img_out_pt, lidar_out_pt = model_pt(img_pt, lidar_pt)

    print(f"Image input: {img_pt.shape} -> output: {img_out_pt.shape}")
    print(f"LiDAR input: {lidar_pt.shape} -> output: {lidar_out_pt.shape}")
    print(
        f"Image output stats: min={img_out_pt.numpy().min():.6f}, max={img_out_pt.numpy().max():.6f}"
    )
    print(
        f"LiDAR output stats: min={lidar_out_pt.numpy().min():.6f}, max={lidar_out_pt.numpy().max():.6f}"
    )

    # TTSIM model
    print("\n--- TTSIM GPT ---")
    model_ttsim = GPT_TTSIM("test_gpt", n_embd, config, lidar_time_frames)

    # Inject weights
    print("Injecting weights...")
    inject_gpt_weights(model_ttsim, model_pt)

    # Forward pass
    img_ttsim = F._from_data("img_input", image_features)
    lidar_ttsim = F._from_data("lidar_input", lidar_features)
    img_ttsim.link_module = model_ttsim
    lidar_ttsim.link_module = model_ttsim

    img_out_ttsim, lidar_out_ttsim = model_ttsim(img_ttsim, lidar_ttsim)

    print(f"Image input: {img_ttsim.shape} -> output: {img_out_ttsim.shape}")
    print(f"LiDAR input: {lidar_ttsim.shape} -> output: {lidar_out_ttsim.shape}")

    if img_out_ttsim.data is not None and lidar_out_ttsim.data is not None:
        print(
            f"Image output stats: min={img_out_ttsim.data.min():.6f}, max={img_out_ttsim.data.max():.6f}"
        )
        print(
            f"LiDAR output stats: min={lidar_out_ttsim.data.min():.6f}, max={lidar_out_ttsim.data.max():.6f}"
        )
    else:
        print("FAIL: FAIL: No data computed")
        return

    # Numerical comparison
    print("\n--- Numerical Comparison ---")
    img_diff = np.abs(img_out_pt.numpy() - img_out_ttsim.data)
    lidar_diff = np.abs(lidar_out_pt.numpy() - lidar_out_ttsim.data)

    print(f"Image - Max diff: {img_diff.max():.10f}, Mean diff: {img_diff.mean():.10f}")
    print(
        f"LiDAR - Max diff: {lidar_diff.max():.10f}, Mean diff: {lidar_diff.mean():.10f}"
    )

    # Validation
    max_diff = max(img_diff.max(), lidar_diff.max())
    if max_diff < 1e-4:
        print("PASS: PASS: GPT matches PyTorch (max diff < 1e-4)")
    elif max_diff < 1e-3:
        print("WARN: WARN: GPT close to PyTorch (max diff < 1e-3)")
    else:
        print(f"FAIL: FAIL: Large differences detected (max diff = {max_diff:.6f})")


if __name__ == "__main__":
    main()
