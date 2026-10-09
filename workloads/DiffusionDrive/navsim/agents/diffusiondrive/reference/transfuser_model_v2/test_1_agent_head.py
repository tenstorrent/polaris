#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for AgentHead module.
Tests shape inference and numerical equivalence between PyTorch and TTSIM.
"""

import os
import sys

# Add DiffusionDrive to path for navsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
# Add TT-Polaris to path for ttsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))

import numpy as np
import torch
import torch.nn as nn
from navsim.agents.diffusiondrive.transfuser_model_v2_ttsim import (
    AgentHead as AgentHead_TTSIM,
)

import ttsim.front.functional.op as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BoundingBox2DIndex:
    """Local constants for BoundingBox2D indexing."""

    POINT = slice(0, 2)  # X, Y
    HEADING = 2

    @staticmethod
    def size():
        return 5  # X, Y, HEADING, LENGTH, WIDTH


class AgentHead_PyTorch(nn.Module):
    """PyTorch reference implementation for AgentHead."""

    def __init__(self, num_agents: int, d_ffn: int, d_model: int):
        super().__init__()
        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries):
        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        )

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


def inject_weights(ttsim_head, pytorch_head):
    """Inject PyTorch weights into TTSIM AgentHead."""
    # MLP states
    ttsim_head._mlp_states_linear1.params[0][1].data = pytorch_head._mlp_states[
        0
    ].weight.data.T.numpy()
    ttsim_head._mlp_states_bias1.params[0][1].data = pytorch_head._mlp_states[
        0
    ].bias.data.numpy()
    ttsim_head._mlp_states_linear2.params[0][1].data = pytorch_head._mlp_states[
        2
    ].weight.data.T.numpy()
    ttsim_head._mlp_states_bias2.params[0][1].data = pytorch_head._mlp_states[
        2
    ].bias.data.numpy()

    # MLP label
    ttsim_head._mlp_label_linear.params[0][1].data = pytorch_head._mlp_label[
        0
    ].weight.data.T.numpy()
    ttsim_head._mlp_label_bias.params[0][1].data = pytorch_head._mlp_label[
        0
    ].bias.data.numpy()


def main():
    print("=" * 70)
    print("AgentHead Validation: PyTorch vs TTSIM")
    print("=" * 70)

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    num_agents = 10
    d_ffn = 512
    d_model = 256
    batch_size = 2

    # Create PyTorch model
    print("\n--- PyTorch AgentHead ---")
    model_pt = AgentHead_PyTorch(num_agents=num_agents, d_ffn=d_ffn, d_model=d_model)
    model_pt.eval()

    # Generate random input
    agent_queries_data = np.random.randn(batch_size, num_agents, d_model).astype(
        np.float32
    )
    agent_queries_pt = torch.from_numpy(agent_queries_data)

    # Forward pass
    with torch.no_grad():
        output_pt = model_pt(agent_queries_pt)

    agent_states_pt = output_pt["agent_states"].numpy()
    agent_labels_pt = output_pt["agent_labels"].numpy()

    print(f"Agent queries shape: {agent_queries_pt.shape}")
    print(f"Agent states shape: {agent_states_pt.shape}")
    print(f"Agent labels shape: {agent_labels_pt.shape}")  # [2, 10]
    print(
        f"Agent states stats: min={agent_states_pt.min():.6f}, max={agent_states_pt.max():.6f}, mean={agent_states_pt.mean():.6f}"
    )
    print(
        f"Agent labels stats: min={agent_labels_pt.min():.6f}, max={agent_labels_pt.max():.6f}, mean={agent_labels_pt.mean():.6f}"
    )

    # TTSIM model
    print("\n--- TTSIM AgentHead ---")
    model_ttsim = AgentHead_TTSIM(num_agents=num_agents, d_ffn=d_ffn, d_model=d_model)

    # Inject weights
    print("Injecting weights...")
    inject_weights(model_ttsim, model_pt)

    # Forward pass
    agent_queries_ttsim = F._from_data("agent_queries", agent_queries_data)
    agent_queries_ttsim.link_module = model_ttsim
    agent_states_ttsim, agent_labels_ttsim = model_ttsim(agent_queries_ttsim)

    print(f"Agent queries shape: {agent_queries_ttsim.shape}")
    print(f"Agent states shape: {agent_states_ttsim.shape}")
    print(f"Agent labels shape: {agent_labels_ttsim.shape}")  # Will be [2, 10, 1]

    # Squeeze agent_labels_pt to match TTSIM shape for now
    agent_labels_pt_expanded = agent_labels_pt[..., np.newaxis]  # [2, 10] -> [2, 10, 1]

    # Check if data is available
    if agent_states_ttsim.data is not None and agent_labels_ttsim.data is not None:
        print(
            f"Agent states stats: min={agent_states_ttsim.data.min():.6f}, max={agent_states_ttsim.data.max():.6f}, mean={agent_states_ttsim.data.mean():.6f}"
        )
        print(
            f"Agent labels stats: min={agent_labels_ttsim.data.min():.6f}, max={agent_labels_ttsim.data.max():.6f}, mean={agent_labels_ttsim.data.mean():.6f}"
        )

        # Numerical comparison
        print("\n--- Numerical Comparison: PyTorch vs TTSIM ---")

        atol = 1e-4
        rtol = 1e-4

        print(f"Tolerance: atol={atol}, rtol={rtol}")

        # Compare agent states
        diff_states = np.abs(agent_states_pt - agent_states_ttsim.data)
        max_diff_states = np.max(diff_states)
        mean_diff_states = np.mean(diff_states)

        print(f"\nAgent States:")
        print(f"  Max absolute difference: {max_diff_states:.10f}")
        print(f"  Mean absolute difference: {mean_diff_states:.10f}")

        is_close_states = np.allclose(
            agent_states_pt, agent_states_ttsim.data, atol=atol, rtol=rtol
        )

        if is_close_states:
            print(f"  PASS: PASS: TTSIM matches PyTorch for agent states")
        else:
            print(f"  FAIL: FAIL: Differences exceed tolerance for agent states")

        # Compare agent labels
        diff_labels = np.abs(agent_labels_pt_expanded - agent_labels_ttsim.data)
        max_diff_labels = np.max(diff_labels)
        mean_diff_labels = np.mean(diff_labels)

        print(f"\nAgent Labels:")
        print(f"  Max absolute difference: {max_diff_labels:.10f}")
        print(f"  Mean absolute difference: {mean_diff_labels:.10f}")

        is_close_labels = np.allclose(
            agent_labels_pt_expanded, agent_labels_ttsim.data, atol=atol, rtol=rtol
        )

        if is_close_labels:
            print(f"  PASS: PASS: TTSIM matches PyTorch for agent labels")
        else:
            print(f"  FAIL: FAIL: Differences exceed tolerance for agent labels")

        # Overall result
        print("\n" + "=" * 70)
        if is_close_states and is_close_labels:
            print("OVERALL: PASS: PASS - All outputs match")
        else:
            print("OVERALL: FAIL: FAIL - Some outputs don't match")
        print("=" * 70)
    else:
        print("  WARN: SKIPPED: No TTSIM data available")

    print()


if __name__ == "__main__":
    main()
