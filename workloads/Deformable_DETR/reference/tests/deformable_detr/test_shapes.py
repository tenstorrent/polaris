#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive DeformableDETR Shape Validation: PyTorch vs TTSim.

Merges component-level shape tests and full end-to-end pipeline shape tests
into a single unified suite.

Component Shape Tests:
   1. MLP                — output shapes across configs
   2. PostProcess        — top-k scores, labels, boxes shapes
   3. sigmoid_focal_loss — scalar loss value match
   4. dice_loss          — scalar loss value match
   5. inverse_sigmoid    — output shape + numerical closeness
   6. SetCriterion       — loss key sets + value match
   7. PostProcessSegm    — mask output shapes
   8. SetCriterion + aux — auxiliary loss key matching
   9. PostProcess num    — numerical score/label/box agreement

End-to-End Pipeline Shape Tests:
  10. DeformableTransformer — hs + reference_points shapes
  11. MLP (bbox head)       — standalone output shape
  12. inverse_sigmoid       — standalone output shape
  13. Class + Bbox heads    — logits + coord output shapes
  14. Full DeformableDETR   — pred_logits, pred_boxes (1 enc + 1 dec)
  15. Full DeformableDETR   — pred_logits, pred_boxes, aux_outputs (3 enc + 3 dec)

Usage:
  python test_shapes.py
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np

# Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

# ── PyTorch imports ──────────────────────────────────────────────────────────
from workloads.Deformable_DETR.reference.deformable_transformer import (
    DeformableTransformer as TransformerPyTorch,
)
from workloads.Deformable_DETR.reference.deformable_detr import (
    DeformableDETR as DETRPyTorch,
    MLP as MLP_PT,
    PostProcess as PostProcess_PT,
    SetCriterion as SetCriterion_PT,
)
from workloads.Deformable_DETR.reference.segmentation import (
    PostProcessSegm as PostProcessSegm_PT,
    sigmoid_focal_loss as sigmoid_focal_loss_PT,
    dice_loss as dice_loss_PT,
)
from workloads.Deformable_DETR.reference.matcher import (
    HungarianMatcher as HungarianMatcher_PT,
)
from workloads.Deformable_DETR.reference.misc import (
    inverse_sigmoid as inverse_sigmoid_PT,
)

# ── TTSim imports ────────────────────────────────────────────────────────────
from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
    DeformableTransformer as TransformerTTSim,
)
from workloads.Deformable_DETR.models.deformable_detr_ttsim import (
    DeformableDETR as DETRTTSim,
    MLP as MLP_TT,
    PostProcess as PostProcess_TT,
    SetCriterion as SetCriterion_TT,
    PostProcessSegm as PostProcessSegm_TT,
    sigmoid_focal_loss as sigmoid_focal_loss_TT,
    dice_loss as dice_loss_TT,
)
from workloads.Deformable_DETR.models.matcher_ttsim import (
    HungarianMatcher as HungarianMatcher_TT,
)
from workloads.Deformable_DETR.util.misc_ttsim import (
    inverse_sigmoid as inverse_sigmoid_TT,
)
from ttsim.ops.tensor import SimTensor

# ═══════════════════════════════════════════════════════════════════════════════
# Globals
# ═══════════════════════════════════════════════════════════════════════════════

PASS_COUNT = 0
FAIL_COUNT = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════


def _sim(t, name="t"):
    """Torch tensor → SimTensor."""
    d = t.detach().cpu().numpy().copy()
    return SimTensor(
        {"name": name, "shape": list(d.shape), "data": d, "dtype": d.dtype}
    )


def torch_to_simtensor(torch_tensor, name="tensor", module=None):
    """Convert PyTorch tensor to SimTensor with optional link_module."""
    data = torch_tensor.detach().cpu().numpy().copy()
    tensor = SimTensor(
        {
            "name": name,
            "shape": list(torch_tensor.shape),
            "data": data,
            "dtype": data.dtype,
        }
    )
    if module is not None:
        tensor.link_module = module
        module._tensors[name] = tensor
    return tensor


def _check(label, expected, actual):
    """Compare two shapes (lists), print pass/fail, update counters."""
    global PASS_COUNT, FAIL_COUNT
    ok = list(expected) == list(actual)
    tag = "PASS" if ok else "FAIL"
    symbol = "✓" if ok else "✗"
    print(
        f"  {symbol} {tag} {label}:  PyTorch {list(expected)}  vs  TTSim {list(actual)}"
    )
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return ok


def _check_scalar(label, pt_val, tt_val, atol=1e-4):
    """Compare two scalar values."""
    global PASS_COUNT, FAIL_COUNT
    diff = abs(float(pt_val) - float(tt_val))
    ok = diff < atol
    tag = "PASS" if ok else "FAIL"
    symbol = "✓" if ok else "✗"
    print(
        f"  {symbol} {tag} {label}:  PyTorch {float(pt_val):.6f}  vs  TTSim {float(tt_val):.6f}  (diff={diff:.2e})"
    )
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return ok


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def subsection(title):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# Weight Sync Utilities (needed for end-to-end tests)
# ═══════════════════════════════════════════════════════════════════════════════


def sync_mlp_weights(pt_mlp, tt_mlp):
    """Copy weights from PyTorch MLP to TTSim MLP.
    SimNN.Linear stores param as [out_features, in_features] (same as PyTorch).
    """
    for i in range(pt_mlp.num_layers):
        pt_layer = pt_mlp.layers[i]
        tt_layer = tt_mlp.layers[i]
        tt_layer.param.data = pt_layer.weight.detach().numpy().copy()
        tt_layer.bias.data = pt_layer.bias.detach().numpy().copy()


def sync_linear_weights(pt_linear, tt_linear):
    """Sync nn.Linear → SimNN.Linear weights (same format, no transpose)."""
    tt_linear.param.data = pt_linear.weight.detach().numpy().copy()
    tt_linear.bias.data = pt_linear.bias.detach().numpy().copy()


def sync_encoder_layer_weights(pt_layer, tt_layer):
    """Copy weights from PyTorch encoder layer to TTSim encoder layer."""
    for proj_name in [
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
    ]:
        pt_proj = getattr(pt_layer.self_attn, proj_name)
        tt_proj = getattr(tt_layer.self_attn, proj_name)
        tt_proj.param.data = pt_proj.weight.detach().numpy().copy()
        tt_proj.bias.data = pt_proj.bias.detach().numpy().copy()

    tt_layer.linear1.param.data = pt_layer.linear1.weight.detach().numpy().copy()
    tt_layer.linear1.bias.data = pt_layer.linear1.bias.detach().numpy().copy()
    tt_layer.linear2.param.data = pt_layer.linear2.weight.detach().numpy().copy()
    tt_layer.linear2.bias.data = pt_layer.linear2.bias.detach().numpy().copy()

    tt_layer.norm1.params[0][1].data = pt_layer.norm1.weight.detach().numpy().copy()
    tt_layer.norm1.params[1][1].data = pt_layer.norm1.bias.detach().numpy().copy()
    tt_layer.norm2.params[0][1].data = pt_layer.norm2.weight.detach().numpy().copy()
    tt_layer.norm2.params[1][1].data = pt_layer.norm2.bias.detach().numpy().copy()


def sync_decoder_layer_weights(pt_layer, tt_layer):
    """Copy weights from PyTorch decoder layer to TTSim decoder layer (with norm swap)."""
    tt_layer.self_attn.in_proj_weight.data = (
        pt_layer.self_attn.in_proj_weight.detach().numpy().T.copy()
    )
    tt_layer.self_attn.in_proj_bias.data = (
        pt_layer.self_attn.in_proj_bias.detach().numpy().copy()
    )
    tt_layer.self_attn.out_proj.param.data = (
        pt_layer.self_attn.out_proj.weight.detach().numpy().T.copy()
    )
    tt_layer.self_attn.out_proj.bias.data = (
        pt_layer.self_attn.out_proj.bias.detach().numpy().copy()
    )

    for proj_name in [
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
    ]:
        pt_proj = getattr(pt_layer.cross_attn, proj_name)
        tt_proj = getattr(tt_layer.cross_attn, proj_name)
        tt_proj.param.data = pt_proj.weight.detach().numpy().copy()
        tt_proj.bias.data = pt_proj.bias.detach().numpy().copy()

    tt_layer.linear1.param.data = pt_layer.linear1.weight.detach().numpy().copy()
    tt_layer.linear1.bias.data = pt_layer.linear1.bias.detach().numpy().copy()
    tt_layer.linear2.param.data = pt_layer.linear2.weight.detach().numpy().copy()
    tt_layer.linear2.bias.data = pt_layer.linear2.bias.detach().numpy().copy()

    # LayerNorm SWAPPED naming for decoder
    tt_layer.norm1.params[0][1].data = pt_layer.norm2.weight.detach().numpy().copy()
    tt_layer.norm1.params[1][1].data = pt_layer.norm2.bias.detach().numpy().copy()
    tt_layer.norm2.params[0][1].data = pt_layer.norm1.weight.detach().numpy().copy()
    tt_layer.norm2.params[1][1].data = pt_layer.norm1.bias.detach().numpy().copy()
    tt_layer.norm3.params[0][1].data = pt_layer.norm3.weight.detach().numpy().copy()
    tt_layer.norm3.params[1][1].data = pt_layer.norm3.bias.detach().numpy().copy()


def sync_transformer_weights(pt_transformer, tt_transformer):
    """Sync all transformer weights."""
    tt_transformer.level_embed.data = pt_transformer.level_embed.detach().numpy().copy()

    if not pt_transformer.two_stage:
        tt_transformer.reference_points.param.data = (
            pt_transformer.reference_points.weight.detach().numpy().copy()
        )
        tt_transformer.reference_points.bias.data = (
            pt_transformer.reference_points.bias.detach().numpy().copy()
        )

    for pt_layer, tt_layer in zip(
        pt_transformer.encoder.layers, tt_transformer.encoder.layers
    ):
        sync_encoder_layer_weights(pt_layer, tt_layer)

    for pt_layer, tt_layer in zip(
        pt_transformer.decoder.layers, tt_transformer.decoder.layers
    ):
        sync_decoder_layer_weights(pt_layer, tt_layer)


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone DETR Wrappers (bypass backbone for end-to-end shape tests)
# ═══════════════════════════════════════════════════════════════════════════════


class DeformableDETR_Standalone_PT(torch.nn.Module):
    """Simplified PyTorch DETR that takes multi-scale features directly."""

    def __init__(
        self,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        hidden_dim,
        aux_loss=True,
    ):
        super().__init__()
        self.transformer = transformer
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss

        self.class_embed = torch.nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP_PT(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = torch.nn.Embedding(num_queries, hidden_dim * 2)

        num_pred = transformer.decoder.num_layers
        self.class_embed = torch.nn.ModuleList(
            [self.class_embed for _ in range(num_pred)]
        )
        self.bbox_embed = torch.nn.ModuleList(
            [self.bbox_embed for _ in range(num_pred)]
        )
        self.transformer.decoder.bbox_embed = None

    def forward(self, srcs, masks, pos_embeds):
        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, _, _ = self.transformer(
            srcs, masks, pos_embeds, query_embeds
        )

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_PT(reference)

            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return out


class DeformableDETR_Standalone_TT:
    """Simplified TTSim DETR that takes multi-scale features directly."""

    def __init__(
        self,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        hidden_dim,
        aux_loss=True,
    ):
        import ttsim.front.functional.sim_nn as SimNN

        self.transformer = transformer
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim

        self.class_embed = SimNN.Linear("class_embed", hidden_dim, num_classes)
        self.bbox_embed = MLP_TT("bbox_embed", hidden_dim, hidden_dim, 4, 3)

        self.query_embed_weight = SimTensor(
            {
                "name": "query_embed.weight",
                "shape": [num_queries, hidden_dim * 2],
                "data": np.random.randn(num_queries, hidden_dim * 2).astype(np.float32)
                * 0.1,
                "dtype": np.dtype(np.float32),
            }
        )

        num_pred = transformer.decoder.num_layers
        self.class_embed_layers = [self.class_embed] * num_pred
        self.bbox_embed_layers = [self.bbox_embed] * num_pred

    def __call__(self, srcs, masks, pos_embeds):
        query_embed = self.query_embed_weight

        hs, init_reference, inter_references, _, _ = self.transformer(
            srcs, masks, pos_embeds, query_embed
        )

        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                if isinstance(inter_references, SimTensor):
                    ref_data = (
                        inter_references.data[lvl - 1]
                        if inter_references.data is not None
                        else None
                    )
                    reference = SimTensor(
                        {
                            "name": f"inter_ref_{lvl-1}",
                            "shape": (
                                list(ref_data.shape)
                                if ref_data is not None
                                else inter_references.shape[1:]
                            ),
                            "data": ref_data,
                            "dtype": np.dtype(np.float32),
                        }
                    )
                else:
                    reference = inter_references[lvl - 1]

            reference = inverse_sigmoid_TT(reference)

            if isinstance(hs, SimTensor):
                hs_lvl_data = hs.data[lvl]
            else:
                hs_lvl_data = hs[lvl]

            hs_lvl = SimTensor(
                {
                    "name": f"hs_lvl_{lvl}",
                    "shape": list(hs_lvl_data.shape),
                    "data": hs_lvl_data,
                    "dtype": np.dtype(np.float32),
                }
            )

            outputs_class = self.class_embed_layers[lvl](hs_lvl)
            tmp = self.bbox_embed_layers[lvl](hs_lvl)

            if reference.shape[-1] == 4:
                if tmp.data is not None and reference.data is not None:
                    combined = tmp.data + reference.data
                else:
                    combined = None
            else:
                assert reference.shape[-1] == 2
                if tmp.data is not None and reference.data is not None:
                    combined = tmp.data.copy()
                    combined[..., :2] += reference.data
                else:
                    combined = None

            if combined is not None:
                coord_data = 1.0 / (1.0 + np.exp(-combined))
            else:
                coord_data = None

            outputs_coord = SimTensor(
                {
                    "name": f"coord_{lvl}",
                    "shape": list(tmp.shape),
                    "data": coord_data,
                    "dtype": np.dtype(np.float32),
                }
            )

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        out = {}
        out["pred_logits"] = outputs_classes[-1]
        out["pred_boxes"] = outputs_coords[-1]

        if self.aux_loss and len(outputs_classes) > 1:
            out["aux_outputs"] = []
            for i in range(len(outputs_classes) - 1):
                out["aux_outputs"].append(
                    {
                        "pred_logits": outputs_classes[i],
                        "pred_boxes": outputs_coords[i],
                    }
                )

        return out


def sync_standalone_detr_weights(pt_detr, tt_detr):
    """Sync weights for standalone DETR wrappers."""
    sync_transformer_weights(pt_detr.transformer, tt_detr.transformer)

    tt_detr.query_embed_weight.data = pt_detr.query_embed.weight.detach().numpy().copy()
    sync_linear_weights(pt_detr.class_embed[0], tt_detr.class_embed)
    sync_mlp_weights(pt_detr.bbox_embed[0], tt_detr.bbox_embed)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART A — Component Shape Tests
# ═══════════════════════════════════════════════════════════════════════════════


# ─── Test 1: MLP ─────────────────────────────────────────────────────────────


def test_01_mlp():
    section("Test 1: MLP (Multi-Layer Perceptron)")

    configs = [
        ("bbox_head", 256, 256, 4, 3, (2, 100, 256)),
        ("class_head", 256, 256, 91, 1, (2, 100, 256)),
        ("deep", 128, 512, 4, 5, (4, 50, 128)),
        ("2D_input", 256, 256, 4, 3, (8, 256)),
        ("4D_input", 256, 256, 4, 3, (6, 2, 100, 256)),
    ]

    all_ok = True
    for name, inp_d, hid_d, out_d, nlayers, shape in configs:
        print(
            f"\n  Config: {name}  input={shape}  MLP({inp_d},{hid_d},{out_d},{nlayers})"
        )

        torch.manual_seed(42)
        np.random.seed(42)
        x = torch.randn(*shape)

        pt = MLP_PT(inp_d, hid_d, out_d, nlayers)
        pt.eval()
        with torch.no_grad():
            y_pt = pt(x)

        tt = MLP_TT(f"mlp_{name}", inp_d, hid_d, out_d, nlayers)
        y_tt = tt(_sim(x, "x"))

        ok = _check(f"MLP.{name} output", y_pt.shape, y_tt.shape)

        for i in range(nlayers):
            pt_in = pt.layers[i].in_features
            pt_out = pt.layers[i].out_features
            tt_in = tt.layers[i].in_features
            tt_out = tt.layers[i].out_features
            if pt_in != tt_in or pt_out != tt_out:
                print(
                    f"    ✗ Layer {i} dim mismatch: PT ({pt_in},{pt_out}) vs TT ({tt_in},{tt_out})"
                )
                ok = False

        all_ok = all_ok and ok

    return all_ok


# ─── Test 2: PostProcess ─────────────────────────────────────────────────────


def test_02_postprocess():
    section("Test 2: PostProcess (top-k detection output)")

    configs = [
        ("standard", 2, 100, 91, [[800, 1200], [600, 800]]),
        ("single", 1, 50, 20, [[640, 640]]),
        ("large_cls", 2, 300, 250, [[1024, 1024], [512, 768]]),
    ]

    all_ok = True
    for name, bs, nq, nc, sizes in configs:
        print(f"\n  Config: {name}  bs={bs} queries={nq} classes={nc}")

        torch.manual_seed(42)
        np.random.seed(42)
        logits = torch.randn(bs, nq, nc)
        boxes = torch.sigmoid(torch.randn(bs, nq, 4))
        target_sizes = torch.tensor(sizes)

        pp_pt = PostProcess_PT()
        with torch.no_grad():
            res_pt = pp_pt({"pred_logits": logits, "pred_boxes": boxes}, target_sizes)

        pp_tt = PostProcess_TT()
        res_tt = pp_tt(
            {"pred_logits": _sim(logits, "logits"), "pred_boxes": _sim(boxes, "boxes")},
            target_sizes.numpy(),
        )

        assert len(res_pt) == len(res_tt) == bs

        ok = True
        for b in range(bs):
            ok &= _check(
                f"  img{b} scores", res_pt[b]["scores"].shape, res_tt[b]["scores"].shape
            )
            ok &= _check(
                f"  img{b} labels", res_pt[b]["labels"].shape, res_tt[b]["labels"].shape
            )
            ok &= _check(
                f"  img{b} boxes", res_pt[b]["boxes"].shape, res_tt[b]["boxes"].shape
            )

        all_ok = all_ok and ok

    return all_ok


# ─── Test 3: sigmoid_focal_loss ──────────────────────────────────────────────


def test_03_sigmoid_focal_loss():
    section("Test 3: sigmoid_focal_loss")

    configs = [
        ("small", 4, 10),
        ("medium", 16, 91),
        ("large", 64, 250),
    ]

    all_ok = True
    for name, N, C in configs:
        print(f"\n  Config: {name}  N={N} C={C}")

        torch.manual_seed(42)
        np.random.seed(42)
        inp = torch.randn(N, C)
        tgt = torch.zeros(N, C)
        for i in range(N):
            tgt[i, i % C] = 1.0
        num_boxes = max(N // 2, 1)

        val_pt = sigmoid_focal_loss_PT(inp, tgt, num_boxes, alpha=0.25, gamma=2).item()
        val_tt = sigmoid_focal_loss_TT(
            inp.numpy(), tgt.numpy(), num_boxes, alpha=0.25, gamma=2
        )

        ok = _check_scalar(f"focal_loss.{name}", val_pt, val_tt, atol=1e-3)
        all_ok = all_ok and ok

    return all_ok


# ─── Test 4: dice_loss ───────────────────────────────────────────────────────


def test_04_dice_loss():
    section("Test 4: dice_loss")

    configs = [
        ("small", 4, 100),
        ("medium", 8, 400),
    ]

    all_ok = True
    for name, N, HW in configs:
        print(f"\n  Config: {name}  N={N} HW={HW}")

        torch.manual_seed(42)
        np.random.seed(42)
        inp = torch.randn(N, HW)
        tgt = (torch.rand(N, HW) > 0.5).float()
        num_boxes = max(N, 1)

        val_pt = dice_loss_PT(inp, tgt, num_boxes).item()
        val_tt = dice_loss_TT(inp.numpy(), tgt.numpy(), num_boxes)

        ok = _check_scalar(f"dice_loss.{name}", val_pt, val_tt, atol=1e-3)
        all_ok = all_ok and ok

    return all_ok


# ─── Test 5: inverse_sigmoid ─────────────────────────────────────────────────


def test_05_inverse_sigmoid():
    section("Test 5: inverse_sigmoid")

    configs = [
        ("uniform", (2, 100, 2)),
        ("3d", (4, 50, 4)),
        ("1d", (256,)),
    ]

    all_ok = True
    for name, shape in configs:
        print(f"\n  Config: {name}  shape={shape}")

        torch.manual_seed(42)
        x = torch.sigmoid(torch.randn(*shape))

        y_pt = inverse_sigmoid_PT(x)
        y_tt = inverse_sigmoid_TT(x.numpy())

        ok = _check(f"inv_sigmoid.{name} output", y_pt.shape, y_tt.shape)

        diff = np.abs(y_pt.numpy() - y_tt.data).max()
        num_ok = diff < 1e-4
        if not num_ok:
            print(f"    ✗ numerical max_diff={diff:.2e}")
        else:
            print(f"    ✓ numerical max_diff={diff:.2e}")
        all_ok = all_ok and ok and num_ok

    return all_ok


# ─── Test 6: SetCriterion ────────────────────────────────────────────────────


def test_06_set_criterion():
    section("Test 6: SetCriterion (loss computation)")

    bs, nq, nc = 2, 10, 5
    num_gt = [3, 2]

    torch.manual_seed(42)
    np.random.seed(42)

    pred_logits = torch.randn(bs, nq, nc)
    pred_boxes = torch.sigmoid(torch.randn(bs, nq, 4))

    targets_pt, targets_tt = [], []
    for i, ng in enumerate(num_gt):
        labels_np = np.random.randint(0, nc, size=(ng,))
        boxes_np = np.random.rand(ng, 4).astype(np.float32)
        boxes_np[:, 2:] = np.clip(boxes_np[:, 2:], 0.05, 0.5)

        targets_pt.append(
            {
                "labels": torch.from_numpy(labels_np).long(),
                "boxes": torch.from_numpy(boxes_np.copy()),
            }
        )
        targets_tt.append({"labels": labels_np.copy(), "boxes": boxes_np.copy()})

    matcher_pt = HungarianMatcher_PT(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion_pt = SetCriterion_PT(
        nc,
        matcher_pt,
        weight_dict,
        losses=["labels", "boxes", "cardinality"],
        focal_alpha=0.25,
    )
    criterion_pt.eval()

    outputs_pt = {"pred_logits": pred_logits.clone(), "pred_boxes": pred_boxes.clone()}
    with torch.no_grad():
        losses_pt = criterion_pt(outputs_pt, targets_pt)

    matcher_tt = HungarianMatcher_TT(
        name="matcher", cost_class=2.0, cost_bbox=5.0, cost_giou=2.0
    )
    criterion_tt = SetCriterion_TT(
        name="criterion",
        num_classes=nc,
        matcher=matcher_tt,
        weight_dict=weight_dict,
        losses=["labels", "boxes", "cardinality"],
        focal_alpha=0.25,
    )

    outputs_tt = {
        "pred_logits": _sim(pred_logits, "pred_logits"),
        "pred_boxes": _sim(pred_boxes, "pred_boxes"),
    }
    losses_tt = criterion_tt(outputs_tt, targets_tt)

    print(f"\n  PyTorch loss keys: {sorted(losses_pt.keys())}")
    print(f"  TTSim loss keys:   {sorted(losses_tt.keys())}")

    all_ok = True
    common_keys = sorted(set(losses_pt.keys()) & set(losses_tt.keys()))
    if set(losses_pt.keys()) != set(losses_tt.keys()):
        missing_in_tt = set(losses_pt.keys()) - set(losses_tt.keys())
        missing_in_pt = set(losses_tt.keys()) - set(losses_pt.keys())
        if missing_in_tt:
            print(f"  ⚠ Keys in PyTorch but not TTSim: {missing_in_tt}")
        if missing_in_pt:
            print(f"  ⚠ Keys in TTSim but not PyTorch: {missing_in_pt}")

    for k in common_keys:
        pt_v = (
            losses_pt[k].item()
            if hasattr(losses_pt[k], "item")
            else float(losses_pt[k])
        )
        tt_v = float(losses_tt[k])
        ok = _check_scalar(f"loss[{k}]", pt_v, tt_v, atol=0.5)
        all_ok = all_ok and ok

    return all_ok


# ─── Test 7: PostProcessSegm ─────────────────────────────────────────────────


def test_07_postprocess_segm():
    section("Test 7: PostProcessSegm (mask post-processing)")

    bs, nq = 2, 5
    mask_h, mask_w = 16, 16

    torch.manual_seed(42)
    np.random.seed(42)

    pred_masks_pt = torch.randn(bs, nq, 1, mask_h, mask_w)
    pred_masks_np = pred_masks_pt.numpy().copy()

    orig_sizes_pt = torch.tensor([[200, 300], [150, 250]])
    max_sizes_pt = torch.tensor([[200, 300], [200, 300]])
    orig_sizes_np = orig_sizes_pt.numpy().copy()
    max_sizes_np = max_sizes_pt.numpy().copy()

    results_pt = [
        {
            "scores": torch.rand(nq),
            "labels": torch.randint(0, 10, (nq,)),
            "boxes": torch.rand(nq, 4),
        }
        for _ in range(bs)
    ]
    results_tt = [
        {
            "scores": r["scores"].numpy().copy(),
            "labels": r["labels"].numpy().copy(),
            "boxes": r["boxes"].numpy().copy(),
        }
        for r in results_pt
    ]

    pp_pt = PostProcessSegm_PT(threshold=0.5)
    with torch.no_grad():
        out_pt = pp_pt(
            results_pt, {"pred_masks": pred_masks_pt}, orig_sizes_pt, max_sizes_pt
        )

    pp_tt = PostProcessSegm_TT(threshold=0.5)
    out_tt = pp_tt(
        results_tt,
        {
            "pred_masks": SimTensor(
                {
                    "name": "pred_masks",
                    "shape": list(pred_masks_np.shape),
                    "data": pred_masks_np,
                    "dtype": np.dtype("float32"),
                }
            )
        },
        orig_sizes_np,
        max_sizes_np,
    )

    all_ok = True
    for b in range(bs):
        pt_mask = out_pt[b]["masks"]
        tt_mask = out_tt[b]["masks"]
        pt_shape = list(pt_mask.shape)
        tt_shape = list(tt_mask.shape)
        ok = _check(f"img{b} mask", pt_shape, tt_shape)
        all_ok = all_ok and ok

    return all_ok


# ─── Test 8: SetCriterion with aux_outputs ───────────────────────────────────


def test_08_set_criterion_aux():
    section("Test 8: SetCriterion with auxiliary outputs")

    bs, nq, nc = 2, 10, 5
    num_dec_layers = 3
    num_gt = [2, 3]

    torch.manual_seed(42)
    np.random.seed(42)

    pred_logits = torch.randn(bs, nq, nc)
    pred_boxes = torch.sigmoid(torch.randn(bs, nq, 4))

    aux_outputs_pt, aux_outputs_tt = [], []
    for lvl in range(num_dec_layers - 1):
        aux_logits = torch.randn(bs, nq, nc)
        aux_boxes = torch.sigmoid(torch.randn(bs, nq, 4))
        aux_outputs_pt.append({"pred_logits": aux_logits, "pred_boxes": aux_boxes})
        aux_outputs_tt.append(
            {
                "pred_logits": _sim(aux_logits, f"aux_logits_{lvl}"),
                "pred_boxes": _sim(aux_boxes, f"aux_boxes_{lvl}"),
            }
        )

    targets_pt, targets_tt = [], []
    for ng in num_gt:
        labels_np = np.random.randint(0, nc, size=(ng,))
        boxes_np = np.random.rand(ng, 4).astype(np.float32)
        boxes_np[:, 2:] = np.clip(boxes_np[:, 2:], 0.05, 0.5)
        targets_pt.append(
            {
                "labels": torch.from_numpy(labels_np).long(),
                "boxes": torch.from_numpy(boxes_np.copy()),
            }
        )
        targets_tt.append({"labels": labels_np.copy(), "boxes": boxes_np.copy()})

    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    for i in range(num_dec_layers - 2):
        weight_dict.update(
            {
                k + f"_{i}": v
                for k, v in {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}.items()
            }
        )

    matcher_pt = HungarianMatcher_PT(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    criterion_pt = SetCriterion_PT(
        nc,
        matcher_pt,
        weight_dict,
        losses=["labels", "boxes", "cardinality"],
        focal_alpha=0.25,
    )
    criterion_pt.eval()

    outputs_full_pt = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "aux_outputs": aux_outputs_pt,
    }
    with torch.no_grad():
        losses_pt = criterion_pt(outputs_full_pt, targets_pt)

    matcher_tt = HungarianMatcher_TT(
        name="matcher", cost_class=2.0, cost_bbox=5.0, cost_giou=2.0
    )
    criterion_tt = SetCriterion_TT(
        name="criterion",
        num_classes=nc,
        matcher=matcher_tt,
        weight_dict=weight_dict,
        losses=["labels", "boxes", "cardinality"],
        focal_alpha=0.25,
    )

    outputs_full_tt = {
        "pred_logits": _sim(pred_logits, "logits"),
        "pred_boxes": _sim(pred_boxes, "boxes"),
        "aux_outputs": aux_outputs_tt,
    }
    losses_tt = criterion_tt(outputs_full_tt, targets_tt)

    print(f"\n  PyTorch: {len(losses_pt)} loss keys")
    print(f"  TTSim:   {len(losses_tt)} loss keys")

    all_ok = True
    pt_keys = set(losses_pt.keys())
    tt_keys = set(losses_tt.keys())

    ok = pt_keys == tt_keys
    if ok:
        print(f"  ✓ PASS  Loss key sets match ({len(pt_keys)} keys)")
    else:
        missing_in_tt = pt_keys - tt_keys
        extra_in_tt = tt_keys - pt_keys
        print(f"  ✗ FAIL  Loss key mismatch")
        if missing_in_tt:
            print(f"    Missing in TTSim: {sorted(missing_in_tt)}")
        if extra_in_tt:
            print(f"    Extra in TTSim: {sorted(extra_in_tt)}")

    global PASS_COUNT, FAIL_COUNT
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    all_ok = all_ok and ok

    for i in range(num_dec_layers - 2):
        expected_key = f"loss_ce_{i}"
        has_pt = expected_key in losses_pt
        has_tt = expected_key in losses_tt
        tag = "✓" if (has_pt and has_tt) else "✗"
        print(f"  {tag} aux key '{expected_key}': PT={has_pt} TT={has_tt}")
        if has_pt and has_tt:
            PASS_COUNT += 1
        else:
            FAIL_COUNT += 1
            all_ok = False

    return all_ok


# ─── Test 9: PostProcess numerical ───────────────────────────────────────────


def test_09_postprocess_numerical():
    section("Test 9: PostProcess numerical comparison")

    bs, nq, nc = 2, 20, 10

    torch.manual_seed(42)
    np.random.seed(42)
    logits = torch.randn(bs, nq, nc)
    boxes = torch.sigmoid(torch.randn(bs, nq, 4))
    target_sizes = torch.tensor([[400, 600], [300, 500]])

    pp_pt = PostProcess_PT()
    with torch.no_grad():
        res_pt = pp_pt({"pred_logits": logits, "pred_boxes": boxes}, target_sizes)

    pp_tt = PostProcess_TT()
    res_tt = pp_tt(
        {"pred_logits": _sim(logits, "logits"), "pred_boxes": _sim(boxes, "boxes")},
        target_sizes.numpy(),
    )

    all_ok = True
    for b in range(bs):
        pt_scores = res_pt[b]["scores"].numpy()
        tt_scores = res_tt[b]["scores"]
        score_diff = np.abs(pt_scores - tt_scores).max()
        ok_scores = score_diff < 1e-4
        tag = "✓" if ok_scores else "✗"
        print(f"  {tag} img{b} scores max_diff={score_diff:.2e}")

        pt_labels = res_pt[b]["labels"].numpy()
        tt_labels = res_tt[b]["labels"]
        labels_match = np.array_equal(pt_labels, tt_labels)
        tag = "✓" if labels_match else "✗"
        print(f"  {tag} img{b} labels match={labels_match}")

        pt_boxes = res_pt[b]["boxes"].numpy()
        tt_boxes = res_tt[b]["boxes"]
        box_diff = np.abs(pt_boxes - tt_boxes).max()
        ok_boxes = box_diff < 0.5
        tag = "✓" if ok_boxes else "✗"
        print(f"  {tag} img{b} boxes max_diff={box_diff:.2f}")

        all_ok = all_ok and ok_scores and labels_match and ok_boxes

    global PASS_COUNT, FAIL_COUNT
    if all_ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
#  PART B — End-to-End Pipeline Shape Tests
# ═══════════════════════════════════════════════════════════════════════════════


# ─── Test 10: DeformableTransformer shapes ───────────────────────────────────


def test_10_transformer_shapes():
    """Transformer output shapes: hs [n_dec, B, Q, D] and reference_points [B, Q, 2]."""

    section("Test 10: DeformableTransformer (Output Shapes)")

    B, Q, D = 1, 10, 64
    n_levels, n_heads = 2, 4
    n_enc, n_dec = 1, 1
    d_ffn, n_points = 128, 4
    spatial_dims = [(5, 5), (3, 3)]

    torch.manual_seed(42)
    np.random.seed(42)

    srcs_pt = [torch.randn(B, D, h, w) * 0.1 for h, w in spatial_dims]
    masks_pt = [torch.zeros(B, h, w, dtype=torch.bool) for h, w in spatial_dims]
    pos_pt = [torch.randn(B, D, h, w) * 0.1 for h, w in spatial_dims]
    qe_pt = torch.randn(Q, D * 2) * 0.1

    pt = TransformerPyTorch(
        d_model=D,
        nhead=n_heads,
        num_encoder_layers=n_enc,
        num_decoder_layers=n_dec,
        dim_feedforward=d_ffn,
        dropout=0.0,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=n_levels,
        dec_n_points=n_points,
        enc_n_points=n_points,
        two_stage=False,
    )
    pt.eval()

    with torch.no_grad():
        hs_pt, ref_pt, inter_ref_pt, _, _ = pt(srcs_pt, masks_pt, pos_pt, qe_pt)

    tt = TransformerTTSim(
        name="xfmr_test",
        d_model=D,
        nhead=n_heads,
        num_encoder_layers=n_enc,
        num_decoder_layers=n_dec,
        dim_feedforward=d_ffn,
        dropout=0.0,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=n_levels,
        dec_n_points=n_points,
        enc_n_points=n_points,
        two_stage=False,
    )
    sync_transformer_weights(pt, tt)

    srcs_sim = [torch_to_simtensor(s, f"src_{i}", tt) for i, s in enumerate(srcs_pt)]
    masks_sim = [
        torch_to_simtensor(m.float(), f"mask_{i}", tt) for i, m in enumerate(masks_pt)
    ]
    pos_sim = [torch_to_simtensor(p, f"pos_{i}", tt) for i, p in enumerate(pos_pt)]
    qe_sim = torch_to_simtensor(qe_pt, "qe", tt)

    hs_tt, ref_tt, _, _, _ = tt(srcs_sim, masks_sim, pos_sim, qe_sim)

    all_ok = True
    ok = _check("hs output", list(hs_pt.shape), list(hs_tt.shape))
    all_ok = all_ok and ok
    ok = _check("reference_points", list(ref_pt.shape), list(ref_tt.shape))
    all_ok = all_ok and ok

    # Verify expected dimensions
    expected_hs = [n_dec, B, Q, D]
    expected_ref = [B, Q, 2]
    ok = _check("hs expected dims", expected_hs, list(hs_pt.shape))
    all_ok = all_ok and ok
    ok = _check("ref expected dims", expected_ref, list(ref_pt.shape))
    all_ok = all_ok and ok

    return all_ok


# ─── Test 11: MLP (bbox head, standalone shape) ─────────────────────────────


def test_11_mlp_standalone_shape():
    """Verify MLP output shape matches."""

    section("Test 11: MLP (Bbox Head — Standalone Shape)")

    D = 64
    torch.manual_seed(42)
    np.random.seed(42)

    pt_mlp = MLP_PT(D, D, 4, 3)
    pt_mlp.eval()

    tt_mlp = MLP_TT("mlp_test", D, D, 4, 3)
    sync_mlp_weights(pt_mlp, tt_mlp)

    x_pt = torch.randn(1, 10, D) * 0.1
    x_sim = torch_to_simtensor(x_pt, "mlp_input")

    with torch.no_grad():
        y_pt = pt_mlp(x_pt)
    y_tt = tt_mlp(x_sim)

    all_ok = True
    ok = _check("MLP output", list(y_pt.shape), list(y_tt.shape))
    all_ok = all_ok and ok

    expected = [1, 10, 4]
    ok = _check("MLP expected [B,Q,4]", expected, list(y_pt.shape))
    all_ok = all_ok and ok

    return all_ok


# ─── Test 12: inverse_sigmoid (standalone shape) ────────────────────────────


def test_12_inverse_sigmoid_shape():
    """Verify inverse_sigmoid preserves shape."""

    section("Test 12: inverse_sigmoid (Standalone Shape)")

    configs = [
        ("2d", (1, 10, 2)),
        ("4d", (2, 5, 10, 4)),
        ("1d", (100,)),
    ]

    all_ok = True
    for name, shape in configs:
        torch.manual_seed(42)
        x_pt = torch.sigmoid(torch.randn(*shape))

        y_pt = inverse_sigmoid_PT(x_pt)
        y_tt = inverse_sigmoid_TT(x_pt.numpy())

        ok = _check(f"inv_sigmoid.{name}", list(y_pt.shape), list(y_tt.shape))
        all_ok = all_ok and ok

        # Shape should be preserved
        ok = _check(f"inv_sigmoid.{name} == input shape", list(shape), list(y_pt.shape))
        all_ok = all_ok and ok

    return all_ok


# ─── Test 13: Class + Bbox heads shapes ─────────────────────────────────────


def test_13_heads_shapes():
    """Verify class_embed and bbox_embed head output shapes."""

    section("Test 13: Class + Bbox Head Output Shapes")

    D = 64
    num_classes = 10
    B, Q = 1, 10

    torch.manual_seed(42)
    np.random.seed(42)

    hs_data = torch.randn(B, Q, D) * 0.1
    ref_data = torch.sigmoid(torch.randn(B, Q, 2))

    pt_class = torch.nn.Linear(D, num_classes)
    pt_bbox = MLP_PT(D, D, 4, 3)
    pt_class.eval()
    pt_bbox.eval()

    import ttsim.front.functional.sim_nn as SimNN

    tt_class = SimNN.Linear("class_embed", D, num_classes)
    tt_bbox = MLP_TT("bbox_embed", D, D, 4, 3)

    sync_linear_weights(pt_class, tt_class)
    sync_mlp_weights(pt_bbox, tt_bbox)

    with torch.no_grad():
        cls_pt = pt_class(hs_data)
        bbox_pt = pt_bbox(hs_data)

    hs_sim = torch_to_simtensor(hs_data, "hs")
    cls_tt = tt_class(hs_sim)
    bbox_tt = tt_bbox(hs_sim)

    all_ok = True
    ok = _check("class_embed output", list(cls_pt.shape), list(cls_tt.shape))
    all_ok = all_ok and ok
    ok = _check("bbox_embed output", list(bbox_pt.shape), list(bbox_tt.shape))
    all_ok = all_ok and ok

    # Verify expected shapes
    ok = _check(
        "class expected [B,Q,num_classes]", [B, Q, num_classes], list(cls_pt.shape)
    )
    all_ok = all_ok and ok
    ok = _check("bbox expected [B,Q,4]", [B, Q, 4], list(bbox_pt.shape))
    all_ok = all_ok and ok

    return all_ok


# ─── Test 14: Full DeformableDETR (1 enc + 1 dec) ───────────────────────────


def test_14_full_detr_shapes():
    """Full DETR output shapes: pred_logits [B,Q,nc], pred_boxes [B,Q,4]."""

    section("Test 14: Full DeformableDETR Shapes (1 enc + 1 dec)")

    B, Q, D = 1, 10, 64
    num_classes, n_levels, n_heads = 10, 2, 4
    n_enc, n_dec, d_ffn, n_points = 1, 1, 128, 4
    spatial_dims = [(5, 5), (3, 3)]

    torch.manual_seed(42)
    np.random.seed(42)

    pt_transformer = TransformerPyTorch(
        d_model=D,
        nhead=n_heads,
        num_encoder_layers=n_enc,
        num_decoder_layers=n_dec,
        dim_feedforward=d_ffn,
        dropout=0.0,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=n_levels,
        dec_n_points=n_points,
        enc_n_points=n_points,
        two_stage=False,
    )
    pt_detr = DeformableDETR_Standalone_PT(
        transformer=pt_transformer,
        num_classes=num_classes,
        num_queries=Q,
        num_feature_levels=n_levels,
        hidden_dim=D,
        aux_loss=True,
    )
    pt_detr.eval()

    tt_transformer = TransformerTTSim(
        name="xfmr",
        d_model=D,
        nhead=n_heads,
        num_encoder_layers=n_enc,
        num_decoder_layers=n_dec,
        dim_feedforward=d_ffn,
        dropout=0.0,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=n_levels,
        dec_n_points=n_points,
        enc_n_points=n_points,
        two_stage=False,
    )
    tt_detr = DeformableDETR_Standalone_TT(
        transformer=tt_transformer,
        num_classes=num_classes,
        num_queries=Q,
        num_feature_levels=n_levels,
        hidden_dim=D,
        aux_loss=True,
    )

    sync_standalone_detr_weights(pt_detr, tt_detr)

    srcs_pt = [torch.randn(B, D, h, w) * 0.1 for h, w in spatial_dims]
    masks_pt = [torch.zeros(B, h, w, dtype=torch.bool) for h, w in spatial_dims]
    pos_pt = [torch.randn(B, D, h, w) * 0.1 for h, w in spatial_dims]

    with torch.no_grad():
        out_pt = pt_detr(srcs_pt, masks_pt, pos_pt)

    srcs_sim = [
        torch_to_simtensor(s, f"src_{i}", tt_transformer) for i, s in enumerate(srcs_pt)
    ]
    masks_sim = [
        torch_to_simtensor(m.float(), f"mask_{i}", tt_transformer)
        for i, m in enumerate(masks_pt)
    ]
    pos_sim = [
        torch_to_simtensor(p, f"pos_{i}", tt_transformer) for i, p in enumerate(pos_pt)
    ]
    out_tt = tt_detr(srcs_sim, masks_sim, pos_sim)

    all_ok = True

    ok = _check(
        "pred_logits",
        list(out_pt["pred_logits"].shape),
        list(out_tt["pred_logits"].shape),
    )
    all_ok = all_ok and ok
    ok = _check(
        "pred_boxes", list(out_pt["pred_boxes"].shape), list(out_tt["pred_boxes"].shape)
    )
    all_ok = all_ok and ok

    # Expected shapes
    ok = _check(
        "pred_logits expected [B,Q,nc]",
        [B, Q, num_classes],
        list(out_pt["pred_logits"].shape),
    )
    all_ok = all_ok and ok
    ok = _check(
        "pred_boxes expected [B,Q,4]", [B, Q, 4], list(out_pt["pred_boxes"].shape)
    )
    all_ok = all_ok and ok

    # With 1 dec layer, aux_outputs should be empty (0 entries) or absent
    has_aux_pt = "aux_outputs" in out_pt
    has_aux_tt = "aux_outputs" in out_tt
    print(f"\n  aux_outputs present: PT={has_aux_pt}  TT={has_aux_tt}")
    if has_aux_pt:
        print(f"  PT aux count: {len(out_pt['aux_outputs'])}")
    if has_aux_tt:
        print(f"  TT aux count: {len(out_tt['aux_outputs'])}")

    return all_ok


# ─── Test 15: Full DeformableDETR (3 enc + 3 dec layers) ────────────────────


def test_15_full_detr_multi_shapes():
    """Full DETR with 3 dec layers: pred shapes + aux_outputs count."""

    section("Test 15: Full DeformableDETR Shapes (3 enc + 3 dec)")

    B, Q, D = 1, 10, 64
    num_classes, n_levels, n_heads = 10, 2, 4
    n_enc, n_dec, d_ffn, n_points = 3, 3, 128, 4
    spatial_dims = [(5, 5), (3, 3)]

    torch.manual_seed(123)
    np.random.seed(123)

    pt_transformer = TransformerPyTorch(
        d_model=D,
        nhead=n_heads,
        num_encoder_layers=n_enc,
        num_decoder_layers=n_dec,
        dim_feedforward=d_ffn,
        dropout=0.0,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=n_levels,
        dec_n_points=n_points,
        enc_n_points=n_points,
        two_stage=False,
    )
    pt_detr = DeformableDETR_Standalone_PT(
        transformer=pt_transformer,
        num_classes=num_classes,
        num_queries=Q,
        num_feature_levels=n_levels,
        hidden_dim=D,
        aux_loss=True,
    )
    pt_detr.eval()

    tt_transformer = TransformerTTSim(
        name="xfmr_multi",
        d_model=D,
        nhead=n_heads,
        num_encoder_layers=n_enc,
        num_decoder_layers=n_dec,
        dim_feedforward=d_ffn,
        dropout=0.0,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=n_levels,
        dec_n_points=n_points,
        enc_n_points=n_points,
        two_stage=False,
    )
    tt_detr = DeformableDETR_Standalone_TT(
        transformer=tt_transformer,
        num_classes=num_classes,
        num_queries=Q,
        num_feature_levels=n_levels,
        hidden_dim=D,
        aux_loss=True,
    )

    sync_standalone_detr_weights(pt_detr, tt_detr)

    srcs_pt = [torch.randn(B, D, h, w) * 0.1 for h, w in spatial_dims]
    masks_pt = [torch.zeros(B, h, w, dtype=torch.bool) for h, w in spatial_dims]
    pos_pt = [torch.randn(B, D, h, w) * 0.1 for h, w in spatial_dims]

    with torch.no_grad():
        out_pt = pt_detr(srcs_pt, masks_pt, pos_pt)

    srcs_sim = [
        torch_to_simtensor(s, f"src_{i}", tt_transformer) for i, s in enumerate(srcs_pt)
    ]
    masks_sim = [
        torch_to_simtensor(m.float(), f"mask_{i}", tt_transformer)
        for i, m in enumerate(masks_pt)
    ]
    pos_sim = [
        torch_to_simtensor(p, f"pos_{i}", tt_transformer) for i, p in enumerate(pos_pt)
    ]
    out_tt = tt_detr(srcs_sim, masks_sim, pos_sim)

    all_ok = True

    ok = _check(
        "pred_logits",
        list(out_pt["pred_logits"].shape),
        list(out_tt["pred_logits"].shape),
    )
    all_ok = all_ok and ok
    ok = _check(
        "pred_boxes", list(out_pt["pred_boxes"].shape), list(out_tt["pred_boxes"].shape)
    )
    all_ok = all_ok and ok

    # Expected shapes
    ok = _check(
        "pred_logits expected [B,Q,nc]",
        [B, Q, num_classes],
        list(out_pt["pred_logits"].shape),
    )
    all_ok = all_ok and ok
    ok = _check(
        "pred_boxes expected [B,Q,4]", [B, Q, 4], list(out_pt["pred_boxes"].shape)
    )
    all_ok = all_ok and ok

    # aux_outputs: should have n_dec - 1 = 2 entries
    global PASS_COUNT, FAIL_COUNT
    if "aux_outputs" in out_pt and "aux_outputs" in out_tt:
        n_aux_pt = len(out_pt["aux_outputs"])
        n_aux_tt = len(out_tt["aux_outputs"])
        expected_aux = n_dec - 1
        print(
            f"\n  aux_outputs count: PyTorch={n_aux_pt}  TTSim={n_aux_tt}  expected={expected_aux}"
        )

        ok = n_aux_pt == n_aux_tt == expected_aux
        if ok:
            print(f"  ✓ PASS  aux_outputs count matches ({expected_aux})")
            PASS_COUNT += 1
        else:
            print(f"  ✗ FAIL  aux_outputs count mismatch")
            FAIL_COUNT += 1
        all_ok = all_ok and ok

        # Check shapes of each aux output
        for i in range(min(n_aux_pt, n_aux_tt)):
            ok1 = _check(
                f"aux[{i}].pred_logits",
                list(out_pt["aux_outputs"][i]["pred_logits"].shape),
                list(out_tt["aux_outputs"][i]["pred_logits"].shape),
            )
            ok2 = _check(
                f"aux[{i}].pred_boxes",
                list(out_pt["aux_outputs"][i]["pred_boxes"].shape),
                list(out_tt["aux_outputs"][i]["pred_boxes"].shape),
            )
            all_ok = all_ok and ok1 and ok2
    else:
        print(
            f"\n  ✗ FAIL  aux_outputs missing (PT={'aux_outputs' in out_pt} TT={'aux_outputs' in out_tt})"
        )
        FAIL_COUNT += 1
        all_ok = False

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    global PASS_COUNT, FAIL_COUNT

    print("\n" + "=" * 80)
    print("  DeformableDETR — Comprehensive Shape Validation")
    print("  PyTorch vs TTSim (Components + End-to-End)")
    print("=" * 80)

    results = {}

    # Part A: Component shape tests
    results["01_MLP"] = test_01_mlp()
    results["02_PostProcess"] = test_02_postprocess()
    results["03_sigmoid_focal_loss"] = test_03_sigmoid_focal_loss()
    results["04_dice_loss"] = test_04_dice_loss()
    results["05_inverse_sigmoid"] = test_05_inverse_sigmoid()
    results["06_SetCriterion"] = test_06_set_criterion()
    results["07_PostProcessSegm"] = test_07_postprocess_segm()
    results["08_SetCriterion_aux"] = test_08_set_criterion_aux()
    results["09_PostProcess_num"] = test_09_postprocess_numerical()

    # Part B: End-to-end shape tests
    results["10_Transformer"] = test_10_transformer_shapes()
    results["11_MLP_standalone"] = test_11_mlp_standalone_shape()
    results["12_inverse_sigmoid"] = test_12_inverse_sigmoid_shape()
    results["13_Heads"] = test_13_heads_shapes()
    results["14_Full_DETR_1x1"] = test_14_full_detr_shapes()
    results["15_Full_DETR_3x3"] = test_15_full_detr_multi_shapes()

    # Summary
    section("SUMMARY")

    print("\n  Part A — Component Shape Tests:")
    for name, ok in list(results.items())[:9]:
        symbol = "✓" if ok else "✗"
        print(f"    {symbol}  {name}")

    print("\n  Part B — End-to-End Pipeline Shape Tests:")
    for name, ok in list(results.items())[9:]:
        symbol = "✓" if ok else "✗"
        print(f"    {symbol}  {name}")

    all_ok = all(results.values())
    print(
        f"\n  Total checks: {PASS_COUNT + FAIL_COUNT}  |  Passed: {PASS_COUNT}  |  Failed: {FAIL_COUNT}"
    )
    print(f"\n  OVERALL: {'ALL PASSED ✓' if all_ok else 'SOME FAILED ✗'}")

    return all_ok


class _TeeStream:
    """Write to both a file and the original stream simultaneously."""

    def __init__(self, original, filepath):
        self._original = original
        self._file = open(filepath, "w", encoding="utf-8")

    def write(self, text):
        self._original.write(text)
        self._file.write(text)

    def flush(self):
        self._original.flush()
        self._file.flush()

    def close(self):
        self._file.close()


if __name__ == "__main__":
    REPORT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    )
    os.makedirs(REPORT_DIR, exist_ok=True)
    REPORT_PATH = os.path.join(REPORT_DIR, "deformable_DETR_shape_validation.md")

    tee = _TeeStream(sys.stdout, REPORT_PATH)
    sys.stdout = tee

    try:
        success = main()
        print(f"\n\n*Report saved to: {REPORT_PATH}*")
        tee.close()
        sys.stdout = tee._original
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        tee.close()
        sys.stdout = tee._original
        sys.exit(1)
