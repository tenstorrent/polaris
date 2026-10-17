#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim UniAD end-to-end model entry point.

Matches the workload entry format expected by TT-Polaris:
    module: UniAD_E2E@UniAD_E2E.py

YAML params (ip_workloads.yaml):
    embed_dims, num_query, num_classes, bev_h, bev_w,
    num_cameras, resnet_depth, num_enc_layers, num_heads, ffn_dim

Instance params:
    img_height, img_width, bs
"""

import os
import sys

_POLARIS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _POLARIS_ROOT not in sys.path:
    sys.path.insert(0, _POLARIS_ROOT)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.UniAD.projects.mmdet3d_plugin.uniad.detectors.uniad_e2e import UniAD


class UniAD_E2E(SimNN.Module):
    """
    TT-Polaris entry-point wrapper for UniAD.

    Delegates all logic to UniAD in detectors/uniad_e2e.py,
    keeping the module: UniAD_E2E@UniAD_E2E.py interface expected by TT-Polaris.

    Args:
        name: module name
        cfg : dict with all YAML params + instance params
    """

    def __init__(self, name: str, cfg: dict):
        super().__init__()
        self.name = name
        self._model = UniAD(name, cfg)

        # Mirror attributes used by tests / TT-Polaris
        self.bs = self._model.bs
        self.embed_dims = self._model.embed_dims
        self.bev_h = self._model.bev_h
        self.bev_w = self._model.bev_w
        self.num_cameras = self._model.num_cameras
        self.img_h = self._model.img_h
        self.img_w = self._model.img_w
        self.input_tensors: dict = {}

        super().link_op2module()

    def create_input_tensors(self):
        self._model.create_input_tensors()
        self.input_tensors = self._model.input_tensors

    def analytical_param_count(self):
        return 0

    def get_forward_graph(self):
        return self._model.get_forward_graph()

    def __call__(self, imgs=None, prev_bev=None):
        return self._model(imgs=imgs, prev_bev=prev_bev)


# ─── standalone runner ────────────────────────────────────────────────────────


def run_standalone(outdir: str = ".") -> None:
    cfg = {
        "embed_dims": 256,
        "num_query": 100,  # small for quick test
        "num_classes": 10,
        "bev_h": 10,
        "bev_w": 10,
        "num_cameras": 6,
        "resnet_depth": 50,
        "num_enc_layers": 2,
        "num_heads": 8,
        "ffn_dim": 512,
        "img_height": 64,
        "img_width": 64,
        "bs": 1,
        # head-specific
        "num_dec_layers": 2,
        "predict_steps": 6,
        "num_anchor": 4,
        "n_future": 2,
        "planning_steps": 4,
    }

    print("Building UniAD_E2E ...")
    model = UniAD_E2E("uniad", cfg)
    model.create_input_tensors()

    print("Running forward pass ...")
    out = model()
    print("plan_traj shape:", out["plan_traj"].shape)

    print("Building graph ...")
    gg = model.get_forward_graph()
    print(f"Graph: {gg.get_node_count()} nodes")
    print("Done.")


if __name__ == "__main__":
    run_standalone()
