#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive DeformableDETR Numerical Validation: PyTorch vs TTSim.

Merges component-level tests and full end-to-end pipeline tests into
a single unified suite.

Component Tests (from deformable_detr modules):
   1. MLP                — with synced weights, multiple configs
   2. PostProcess        — top-k scores, labels, boxes
   3. sigmoid_focal_loss — scalar loss value
   4. dice_loss          — scalar loss value
   5. inverse_sigmoid    — element-wise logit, multiple shapes
   6. SetCriterion       — all loss values (labels, boxes, cardinality)
   7. PostProcessSegm    — mask values
   8. SetCriterion + aux — auxiliary outputs loss matching
   9. MLP intermediates  — layer-by-layer intermediate outputs

End-to-End Pipeline Tests:
  10. DeformableTransformer (standalone, 1 enc + 1 dec)
  11. MLP (bbox head, standalone)
  12. inverse_sigmoid (standalone)
  13. Class + Bbox heads on shared transformer output
  14. Full DeformableDETR (1 enc + 1 dec layer, bypass backbone)
  15. Full DeformableDETR (3 enc + 3 dec layers, bypass backbone)

Usage:
  python test_numerical.py
"""

import os
import sys
import copy
from typing import Any

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
    """Torch tensor → SimTensor (compact helper)."""
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


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def subsection(title):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def compare_numerical(pytorch_out, ttsim_out, name="Output", rtol=1e-2, atol=1e-3):
    """Compare PyTorch and TTSim outputs numerically (rich output)."""
    global PASS_COUNT, FAIL_COUNT

    if isinstance(pytorch_out, torch.Tensor):
        pt_np = pytorch_out.detach().cpu().numpy()
    else:
        pt_np = np.asarray(pytorch_out)

    if isinstance(ttsim_out, SimTensor):
        if ttsim_out.data is None:
            print(f"  ✗ FAIL - {name}: TTSim output has no data!")
            FAIL_COUNT += 1
            return False
        tt_np = ttsim_out.data
    else:
        tt_np = np.asarray(ttsim_out)

    if pt_np.shape != tt_np.shape:
        print(
            f"  ✗ FAIL - {name}: Shape mismatch PyTorch={pt_np.shape} TTSim={tt_np.shape}"
        )
        FAIL_COUNT += 1
        return False

    diff = np.abs(pt_np - tt_np)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    is_close = np.allclose(pt_np, tt_np, rtol=rtol, atol=atol, equal_nan=True)

    status = "✓ PASS" if is_close else "✗ FAIL"
    print(f"  {status} - {name}")
    print(
        f"    Shape: {list(pt_np.shape)}  Max diff: {max_diff:.2e}  Mean diff: {mean_diff:.2e}"
    )
    print(f"    PyTorch (first 8): {pt_np.flatten()[:8]}")
    print(f"    TTSim   (first 8): {tt_np.flatten()[:8]}")

    if not is_close:
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(
            f"    Max diff at {max_idx}: PT={pt_np[max_idx]:.6f} TT={tt_np[max_idx]:.6f}"
        )
        FAIL_COUNT += 1
    else:
        PASS_COUNT += 1

    return is_close


def _report(label, pt_val, tt_val, atol=1e-4):
    """Compare numerical values with flattened array comparison."""
    global PASS_COUNT, FAIL_COUNT

    if isinstance(pt_val, torch.Tensor):
        pt_arr = pt_val.detach().cpu().numpy()
    elif isinstance(pt_val, np.ndarray):
        pt_arr = pt_val
    else:
        pt_arr = np.array(pt_val)

    if isinstance(tt_val, SimTensor):
        tt_arr = tt_val.data
    elif isinstance(tt_val, np.ndarray):
        tt_arr = tt_val
    else:
        tt_arr = np.array(tt_val)

    pt_arr = pt_arr.astype(np.float64).ravel()
    tt_arr = tt_arr.astype(np.float64).ravel()

    if pt_arr.shape != tt_arr.shape:
        print(
            f"  ✗ FAIL {label}: shape mismatch PT {pt_arr.shape} vs TT {tt_arr.shape}"
        )
        FAIL_COUNT += 1
        return False

    max_diff = np.abs(pt_arr - tt_arr).max() if pt_arr.size > 0 else 0.0
    mean_diff = np.abs(pt_arr - tt_arr).mean() if pt_arr.size > 0 else 0.0
    ok = max_diff < atol

    tag = "PASS" if ok else "FAIL"
    symbol = "✓" if ok else "✗"

    print(
        f"  {symbol} {tag} {label}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}"
    )

    n = min(8, len(pt_arr))
    print(f"         PT: {np.array2string(pt_arr[:n], precision=6, separator=', ')}")
    print(f"         TT: {np.array2string(tt_arr[:n], precision=6, separator=', ')}")
    if len(pt_arr) > n:
        print(f"         ... ({len(pt_arr)} total elements)")

    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return ok


def _report_scalar(label, pt_val, tt_val, atol=1e-4):
    """Compare two scalar values."""
    global PASS_COUNT, FAIL_COUNT
    pt_f = float(pt_val)
    tt_f = float(tt_val)
    diff = abs(pt_f - tt_f)
    ok = diff < atol

    tag = "PASS" if ok else "FAIL"
    symbol = "✓" if ok else "✗"
    print(f"  {symbol} {tag} {label}:  PT={pt_f:.8f}  TT={tt_f:.8f}  diff={diff:.2e}")

    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Weight Sync Utilities
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

    print(
        f"  Synced: transformer ({len(pt_transformer.encoder.layers)} enc + "
        f"{len(pt_transformer.decoder.layers)} dec layers)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone DETR Wrappers (bypass backbone for end-to-end tests)
# ═══════════════════════════════════════════════════════════════════════════════


class DeformableDETR_Standalone_PT(torch.nn.Module):
    """
    Simplified PyTorch DETR that takes multi-scale features directly
    (bypasses the backbone + input_proj).
    """

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

        _class_embed = torch.nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP_PT(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = torch.nn.Embedding(num_queries, hidden_dim * 2)

        num_pred = transformer.decoder.num_layers
        self.class_embed = torch.nn.ModuleList([_class_embed for _ in range(num_pred)])
        self.bbox_embed = torch.nn.ModuleList([_bbox_embed for _ in range(num_pred)])
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

        out: dict[str, Any] = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return out


class DeformableDETR_Standalone_TT:
    """
    Simplified TTSim DETR that takes multi-scale features directly
    (bypasses backbone + input_proj).
    """

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

        # Query embedding (learnable parameter)
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

        # Shared heads (mirrors PyTorch without box_refine)
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
                # inter_references is a SimTensor [n_layers, B, Q, 2]; extract layer slice
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

            # Class head
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

            # Sigmoid
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

        # Use last decoder layer outputs directly (no stack needed)
        out = {}
        out["pred_logits"] = outputs_classes[-1]
        out["pred_boxes"] = outputs_coords[-1]

        if self.aux_loss and len(outputs_classes) > 1:
            out["aux_outputs"] = []
            for i in range(len(outputs_classes) - 1):
                aux_cls = outputs_classes[i]
                aux_coord = outputs_coords[i]
                out["aux_outputs"].append(
                    {
                        "pred_logits": aux_cls,
                        "pred_boxes": aux_coord,
                    }
                )

        return out


def sync_standalone_detr_weights(pt_detr, tt_detr):
    """Sync weights for standalone DETR wrappers."""
    subsection("Weight Sync (Standalone DETR)")

    # Transformer
    sync_transformer_weights(pt_detr.transformer, tt_detr.transformer)

    # Query embed
    tt_detr.query_embed_weight.data = pt_detr.query_embed.weight.detach().numpy().copy()
    print("  Synced: query_embed")

    # Class embed (shared — sync the single underlying Linear)
    sync_linear_weights(pt_detr.class_embed[0], tt_detr.class_embed)
    print("  Synced: class_embed")

    # Bbox embed (shared — sync the single underlying MLP)
    sync_mlp_weights(pt_detr.bbox_embed[0], tt_detr.bbox_embed)
    print("  Synced: bbox_embed")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART A — Component Tests (deformable_detr modules)
# ═══════════════════════════════════════════════════════════════════════════════


# ─── Test 1: MLP (with synced weights) ───────────────────────────────────────


def test_01_mlp_numerical():
    section("Test 1: MLP — Numerical (synced weights)")

    configs = [
        ("bbox_head", 256, 256, 4, 3, (2, 10, 256)),
        ("class_head", 256, 256, 91, 1, (2, 10, 256)),
        ("deep_mlp", 128, 512, 4, 5, (1, 5, 128)),
    ]

    all_ok = True
    for name, inp_d, hid_d, out_d, nlayers, shape in configs:
        print(
            f"\n  Config: {name}  input={shape}  MLP({inp_d},{hid_d},{out_d},{nlayers})"
        )

        torch.manual_seed(42)
        np.random.seed(42)

        pt = MLP_PT(inp_d, hid_d, out_d, nlayers)
        pt.eval()
        tt = MLP_TT(f"mlp_{name}", inp_d, hid_d, out_d, nlayers)
        sync_mlp_weights(pt, tt)

        x = torch.randn(*shape)
        with torch.no_grad():
            y_pt = pt(x)
        y_tt = tt(_sim(x, "x"))

        ok = _report(f"MLP.{name} output", y_pt, y_tt, atol=1e-4)
        all_ok = all_ok and ok

    return all_ok


# ─── Test 2: PostProcess ─────────────────────────────────────────────────────


def test_02_postprocess_numerical():
    section("Test 2: PostProcess — Numerical")

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
        print(f"\n  Image {b}:")
        ok_s = _report(
            f"  scores[{b}]", res_pt[b]["scores"], res_tt[b]["scores"], atol=1e-4
        )
        ok_l = _report(
            f"  labels[{b}]",
            res_pt[b]["labels"].float(),
            res_tt[b]["labels"].astype(np.float32),
            atol=0.5,
        )
        ok_b = _report(
            f"  boxes[{b}]", res_pt[b]["boxes"], res_tt[b]["boxes"], atol=0.01
        )
        all_ok = all_ok and ok_s and ok_l and ok_b

    return all_ok


# ─── Test 3: sigmoid_focal_loss ──────────────────────────────────────────────


def test_03_sigmoid_focal_loss_numerical():
    section("Test 3: sigmoid_focal_loss — Numerical")

    configs = [
        ("small", 4, 10, 2),
        ("medium", 16, 91, 8),
        ("large", 64, 250, 32),
    ]

    all_ok = True
    for name, N, C, num_boxes in configs:
        print(f"\n  Config: {name}  N={N} C={C} num_boxes={num_boxes}")

        torch.manual_seed(42)
        np.random.seed(42)

        inp = torch.randn(N, C)
        tgt = torch.zeros(N, C)
        for i in range(N):
            tgt[i, i % C] = 1.0

        val_pt = sigmoid_focal_loss_PT(inp, tgt, num_boxes, alpha=0.25, gamma=2).item()
        val_tt = sigmoid_focal_loss_TT(
            inp.numpy(), tgt.numpy(), num_boxes, alpha=0.25, gamma=2
        )

        ok = _report_scalar(f"focal_loss.{name}", val_pt, val_tt, atol=1e-6)
        all_ok = all_ok and ok

    return all_ok


# ─── Test 4: dice_loss ───────────────────────────────────────────────────────


def test_04_dice_loss_numerical():
    section("Test 4: dice_loss — Numerical")

    configs = [
        ("small", 4, 100, 4),
        ("medium", 8, 400, 8),
        ("ones", 2, 50, 2),
    ]

    all_ok = True
    for name, N, HW, num_boxes in configs:
        print(f"\n  Config: {name}  N={N} HW={HW}")

        torch.manual_seed(42)
        np.random.seed(42)

        inp = torch.randn(N, HW)
        tgt = (torch.rand(N, HW) > 0.5).float()

        val_pt = dice_loss_PT(inp, tgt, num_boxes).item()
        val_tt = dice_loss_TT(inp.numpy(), tgt.numpy(), num_boxes)

        ok = _report_scalar(f"dice_loss.{name}", val_pt, val_tt, atol=1e-6)
        all_ok = all_ok and ok

    return all_ok


# ─── Test 5: inverse_sigmoid (multiple shapes) ──────────────────────────────


def test_05_inverse_sigmoid_numerical():
    section("Test 5: inverse_sigmoid — Numerical")

    configs = [
        ("uniform", (2, 10, 2)),
        ("3d", (4, 5, 4)),
        ("1d", (20,)),
    ]

    all_ok = True
    for name, shape in configs:
        print(f"\n  Config: {name}  shape={shape}")

        torch.manual_seed(42)
        x = torch.sigmoid(torch.randn(*shape))

        y_pt = inverse_sigmoid_PT(x)
        y_tt = inverse_sigmoid_TT(x.numpy())

        ok = _report(f"inv_sigmoid.{name}", y_pt, y_tt, atol=1e-5)
        all_ok = all_ok and ok

    return all_ok


# ─── Test 6: SetCriterion ────────────────────────────────────────────────────


def test_06_set_criterion_numerical():
    section("Test 6: SetCriterion — Numerical")

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

    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}

    matcher_pt = HungarianMatcher_PT(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
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

    print(f"\n  PyTorch losses:")
    for k in sorted(losses_pt.keys()):
        v = (
            losses_pt[k].item()
            if hasattr(losses_pt[k], "item")
            else float(losses_pt[k])
        )
        print(f"    {k:25s} = {v:.8f}")

    print(f"\n  TTSim losses:")
    for k in sorted(losses_tt.keys()):
        v = float(losses_tt[k])
        print(f"    {k:25s} = {v:.8f}")

    print(f"\n  Comparison:")
    all_ok = True
    common_keys = sorted(set(losses_pt.keys()) & set(losses_tt.keys()))
    for k in common_keys:
        pt_v = (
            losses_pt[k].item()
            if hasattr(losses_pt[k], "item")
            else float(losses_pt[k])
        )
        tt_v = float(losses_tt[k])
        ok = _report_scalar(f"loss[{k}]", pt_v, tt_v, atol=1e-4)
        all_ok = all_ok and ok

    missing = set(losses_pt.keys()) - set(losses_tt.keys())
    extra = set(losses_tt.keys()) - set(losses_pt.keys())
    if missing:
        print(f"  ⚠ Missing in TTSim: {sorted(missing)}")
    if extra:
        print(f"  ⚠ Extra in TTSim: {sorted(extra)}")

    return all_ok


# ─── Test 7: PostProcessSegm ─────────────────────────────────────────────────


def test_07_postprocess_segm_numerical():
    section("Test 7: PostProcessSegm — Numerical")

    bs, nq = 2, 3
    mask_h, mask_w = 8, 8

    torch.manual_seed(42)
    np.random.seed(42)

    pred_masks_pt = torch.randn(bs, nq, 1, mask_h, mask_w)
    pred_masks_np = pred_masks_pt.numpy().copy()

    orig_sizes_pt = torch.tensor([[100, 150], [80, 120]])
    max_sizes_pt = torch.tensor([[100, 150], [100, 150]])
    orig_sizes_np = orig_sizes_pt.numpy().copy()
    max_sizes_np = max_sizes_pt.numpy().copy()

    results_pt, results_tt = [], []
    for _ in range(bs):
        scores_pt = torch.rand(nq)
        labels_pt = torch.randint(0, 10, (nq,))
        boxes_pt = torch.rand(nq, 4)
        results_pt.append({"scores": scores_pt, "labels": labels_pt, "boxes": boxes_pt})
        results_tt.append(
            {
                "scores": scores_pt.numpy().copy(),
                "labels": labels_pt.numpy().copy(),
                "boxes": boxes_pt.numpy().copy(),
            }
        )

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
        pt_mask = (
            out_pt[b]["masks"].numpy()
            if isinstance(out_pt[b]["masks"], torch.Tensor)
            else out_pt[b]["masks"]
        )
        tt_mask = out_tt[b]["masks"]

        print(f"\n  Image {b}: mask shape PT={pt_mask.shape} TT={tt_mask.shape}")

        pt_flat = pt_mask.ravel().astype(np.float32)
        tt_flat = tt_mask.ravel().astype(np.float32)

        if pt_flat.shape == tt_flat.shape:
            agreement = (pt_flat == tt_flat).mean() * 100
            max_diff = np.abs(pt_flat - tt_flat).max()
            n = min(20, len(pt_flat))
            print(f"    Agreement: {agreement:.1f}%  max_diff={max_diff:.0f}")
            print(f"    PT: {pt_flat[:n].astype(int)}")
            print(f"    TT: {tt_flat[:n].astype(int)}")
            ok = agreement > 95.0
        else:
            print(f"    ✗ Shape mismatch!")
            ok = False

        global PASS_COUNT, FAIL_COUNT
        if ok:
            print(f"    ✓ PASS")
            PASS_COUNT += 1
        else:
            print(f"    ✗ FAIL")
            FAIL_COUNT += 1
        all_ok = all_ok and ok

    return all_ok


# ─── Test 8: SetCriterion with aux_outputs ───────────────────────────────────


def test_08_set_criterion_aux_numerical():
    section("Test 8: SetCriterion with aux_outputs — Numerical")

    bs, nq, nc = 2, 8, 5
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

    outputs_pt = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "aux_outputs": aux_outputs_pt,
    }
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
        "pred_logits": _sim(pred_logits, "logits"),
        "pred_boxes": _sim(pred_boxes, "boxes"),
        "aux_outputs": aux_outputs_tt,
    }
    losses_tt = criterion_tt(outputs_tt, targets_tt)

    all_keys = sorted(set(losses_pt.keys()) | set(losses_tt.keys()))
    print(f"\n  {'Loss Key':30s}  {'PyTorch':>14s}  {'TTSim':>14s}  {'Diff':>12s}")
    print(f"  {'-'*30}  {'-'*14}  {'-'*14}  {'-'*12}")

    all_ok = True
    for k in all_keys:
        pt_v = (
            losses_pt[k].item()
            if k in losses_pt and hasattr(losses_pt[k], "item")
            else (float(losses_pt[k]) if k in losses_pt else None)
        )
        tt_v = float(losses_tt[k]) if k in losses_tt else None

        if pt_v is not None and tt_v is not None:
            diff = abs(pt_v - tt_v)
            ok = diff < 1e-3
            symbol = "✓" if ok else "✗"
            print(f"  {symbol} {k:28s}  {pt_v:14.8f}  {tt_v:14.8f}  {diff:12.2e}")
            if not ok:
                all_ok = False
        elif pt_v is None:
            print(f"  ⚠ {k:28s}  {'N/A':>14s}  {tt_v:14.8f}  {'(TT only)':>12s}")
        else:
            print(f"  ⚠ {k:28s}  {pt_v:14.8f}  {'N/A':>14s}  {'(PT only)':>12s}")

    global PASS_COUNT, FAIL_COUNT
    if all_ok:
        PASS_COUNT += 1
        print(f"\n  ✓ PASS  All {len(all_keys)} loss values match")
    else:
        FAIL_COUNT += 1
        print(f"\n  ✗ FAIL  Some loss values diverge")

    return all_ok


# ─── Test 9: MLP intermediate layers ─────────────────────────────────────────


def test_09_mlp_intermediates():
    section("Test 9: MLP — Layer-by-layer Intermediate Outputs")

    inp_d, hid_d, out_d, nlayers = 256, 256, 4, 3
    shape = (2, 5, 256)

    torch.manual_seed(42)
    np.random.seed(42)

    pt = MLP_PT(inp_d, hid_d, out_d, nlayers)
    pt.eval()
    tt = MLP_TT(f"mlp_debug", inp_d, hid_d, out_d, nlayers)
    sync_mlp_weights(pt, tt)

    x_pt = torch.randn(*shape)

    all_ok = True
    cur_pt = x_pt.clone()
    cur_tt_data = x_pt.numpy().copy()

    for i in range(nlayers):
        with torch.no_grad():
            cur_pt = pt.layers[i](cur_pt)
            if i < nlayers - 1:
                cur_pt = TF.relu(cur_pt)

        cur_tt_sim = SimTensor(
            {
                "name": f"layer_{i}_in",
                "shape": list(cur_tt_data.shape),
                "data": cur_tt_data,
                "dtype": np.dtype("float32"),
            }
        )
        cur_tt_sim.set_module(tt)
        cur_tt_out = tt.layers[i](cur_tt_sim)
        if i < nlayers - 1:
            import ttsim.front.functional.op as F_sim

            relu_op = F_sim.Relu(f"debug_relu_{i}")
            cur_tt_out = relu_op(cur_tt_out)

        cur_tt_data = cur_tt_out.data.copy()

        print(f"\n  Layer {i} ({'ReLU' if i < nlayers-1 else 'linear'}):")
        ok = _report(f"after layer {i}", cur_pt, cur_tt_out, atol=1e-4)
        all_ok = all_ok and ok

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
#  PART B — End-to-End Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════


# ─── Test 10: DeformableTransformer (standalone) ─────────────────────────────


def test_10_transformer_numerical():
    """Quick sanity check: transformer only (no DETR heads)."""

    section("Test 10: DeformableTransformer (Standalone)")

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

    ok_hs = compare_numerical(
        hs_pt, hs_tt, "Transformer hs output", rtol=0.05, atol=0.01
    )
    ok_ref = compare_numerical(
        ref_pt, ref_tt, "Transformer reference_points", rtol=0.05, atol=0.01
    )

    return ok_hs and ok_ref


# ─── Test 11: MLP (bbox head, standalone) ────────────────────────────────────


def test_11_mlp_standalone():
    """Verify MLP (bbox head) produces identical outputs."""

    section("Test 11: MLP (Bbox Head — Standalone)")

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

    return compare_numerical(y_pt, y_tt, "MLP output", rtol=1e-4, atol=1e-5)


# ─── Test 12: inverse_sigmoid (standalone) ───────────────────────────────────


def test_12_inverse_sigmoid():
    """Verify inverse_sigmoid matches."""

    section("Test 12: inverse_sigmoid (Standalone)")

    torch.manual_seed(42)
    x_pt = torch.sigmoid(torch.randn(1, 10, 2))
    x_np = x_pt.numpy().copy()

    with torch.no_grad():
        y_pt = inverse_sigmoid_PT(x_pt)
    y_tt = inverse_sigmoid_TT(x_np)

    return compare_numerical(y_pt, y_tt, "inverse_sigmoid", rtol=1e-4, atol=1e-5)


# ─── Test 13: Class + Bbox heads on transformer output ──────────────────────


def test_13_heads_on_transformer_output():
    """
    Test that class_embed and bbox_embed heads produce identical results
    when fed the same transformer output + reference_points.
    """

    section("Test 13: Class + Bbox Heads on Shared Transformer Output")

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
        tmp_pt = pt_bbox(hs_data)
        ref_inv = inverse_sigmoid_PT(ref_data)
        tmp_pt[..., :2] += ref_inv
        coord_pt = tmp_pt.sigmoid()

    hs_sim = torch_to_simtensor(hs_data, "hs")
    ref_sim = torch_to_simtensor(ref_data, "ref")

    cls_tt = tt_class(hs_sim)
    tmp_tt_raw = tt_bbox(hs_sim)
    ref_inv_tt = inverse_sigmoid_TT(ref_sim)

    if tmp_tt_raw.data is not None and ref_inv_tt.data is not None:
        tmp_combined = tmp_tt_raw.data.copy()
        tmp_combined[..., :2] += ref_inv_tt.data
        coord_data = 1.0 / (1.0 + np.exp(-tmp_combined))
    else:
        coord_data = None

    coord_tt = SimTensor(
        {
            "name": "coord",
            "shape": list(tmp_tt_raw.shape),
            "data": coord_data,
            "dtype": np.dtype(np.float32),
        }
    )

    ok_cls = compare_numerical(
        cls_pt, cls_tt, "Classification logits", rtol=1e-4, atol=1e-5
    )
    ok_coord = compare_numerical(
        coord_pt, coord_tt, "Bbox coordinates", rtol=1e-4, atol=1e-5
    )

    return ok_cls and ok_coord


# ─── Test 14: Full DeformableDETR (1 enc + 1 dec) ───────────────────────────


def test_14_full_detr_numerical():
    """
    Full end-to-end numerical validation of DeformableDETR.
    Config: B=1, Q=10, d_model=64, 2 levels, 4 heads,
            1 enc layer, 1 dec layer, num_classes=10, dropout=0.0
    """

    section("Test 14: Full DeformableDETR (1 enc + 1 dec)")

    B, Q, D = 1, 10, 64
    num_classes, n_levels, n_heads = 10, 2, 4
    n_enc, n_dec, d_ffn, n_points = 1, 1, 128, 4
    spatial_dims = [(5, 5), (3, 3)]

    torch.manual_seed(42)
    np.random.seed(42)

    subsection("Building PyTorch DeformableDETR")
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

    subsection("Building TTSim DeformableDETR")
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

    subsection("Creating inputs")
    srcs_pt = [torch.randn(B, D, h, w) * 0.1 for h, w in spatial_dims]
    masks_pt = [torch.zeros(B, h, w, dtype=torch.bool) for h, w in spatial_dims]
    pos_pt = [torch.randn(B, D, h, w) * 0.1 for h, w in spatial_dims]

    print(f"  Multi-scale features:")
    for i, s in enumerate(srcs_pt):
        print(f"    Level {i}: {list(s.shape)}")

    subsection("PyTorch forward")
    with torch.no_grad():
        out_pt = pt_detr(srcs_pt, masks_pt, pos_pt)
    print(f"  pred_logits: {list(out_pt['pred_logits'].shape)}")
    print(f"  pred_boxes:  {list(out_pt['pred_boxes'].shape)}")
    if "aux_outputs" in out_pt:
        print(f"  aux_outputs:  {len(out_pt['aux_outputs'])} layers")

    subsection("TTSim forward")
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
    print(f"  pred_logits: {out_tt['pred_logits'].shape}")
    print(f"  pred_boxes:  {out_tt['pred_boxes'].shape}")
    if "aux_outputs" in out_tt:
        print(f"  aux_outputs:  {len(out_tt['aux_outputs'])} layers")

    subsection("Numerical Comparison")
    all_ok = True

    ok = compare_numerical(
        out_pt["pred_logits"],
        out_tt["pred_logits"],
        "pred_logits",
        rtol=0.05,
        atol=0.01,
    )
    all_ok = all_ok and ok
    ok = compare_numerical(
        out_pt["pred_boxes"], out_tt["pred_boxes"], "pred_boxes", rtol=0.05, atol=0.01
    )
    all_ok = all_ok and ok

    if "aux_outputs" in out_pt and "aux_outputs" in out_tt:
        for i, (aux_pt, aux_tt) in enumerate(
            zip(out_pt["aux_outputs"], out_tt["aux_outputs"])
        ):
            ok = compare_numerical(
                aux_pt["pred_logits"],
                aux_tt["pred_logits"],
                f"aux_outputs[{i}].pred_logits",
                rtol=0.05,
                atol=0.01,
            )
            all_ok = all_ok and ok
            ok = compare_numerical(
                aux_pt["pred_boxes"],
                aux_tt["pred_boxes"],
                f"aux_outputs[{i}].pred_boxes",
                rtol=0.05,
                atol=0.01,
            )
            all_ok = all_ok and ok

    return all_ok


# ─── Test 15: Full DeformableDETR (3 enc + 3 dec layers) ────────────────────


def test_15_full_detr_multi_layer():
    """
    Full DeformableDETR with 3 encoder + 3 decoder layers.
    Tests aux_outputs at each decoder layer.
    """

    section("Test 15: Full DeformableDETR (3 enc + 3 dec layers)")

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

    ok = compare_numerical(
        out_pt["pred_logits"],
        out_tt["pred_logits"],
        "pred_logits (3 dec layers)",
        rtol=0.05,
        atol=0.01,
    )
    all_ok = all_ok and ok
    ok = compare_numerical(
        out_pt["pred_boxes"],
        out_tt["pred_boxes"],
        "pred_boxes (3 dec layers)",
        rtol=0.05,
        atol=0.01,
    )
    all_ok = all_ok and ok

    if "aux_outputs" in out_pt and "aux_outputs" in out_tt:
        n_aux_pt = len(out_pt["aux_outputs"])
        n_aux_tt = len(out_tt["aux_outputs"])
        print(f"\n  aux_outputs count: PyTorch={n_aux_pt}  TTSim={n_aux_tt}")
        if n_aux_pt != n_aux_tt:
            print(f"  ✗ FAIL — count mismatch")
            all_ok = False
        for i in range(min(n_aux_pt, n_aux_tt)):
            ok1 = compare_numerical(
                out_pt["aux_outputs"][i]["pred_logits"],
                out_tt["aux_outputs"][i]["pred_logits"],
                f"aux[{i}].pred_logits",
                rtol=0.05,
                atol=0.01,
            )
            ok2 = compare_numerical(
                out_pt["aux_outputs"][i]["pred_boxes"],
                out_tt["aux_outputs"][i]["pred_boxes"],
                f"aux[{i}].pred_boxes",
                rtol=0.05,
                atol=0.01,
            )
            all_ok = all_ok and ok1 and ok2

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    global PASS_COUNT, FAIL_COUNT

    print("\n" + "=" * 80)
    print("  DeformableDETR — Comprehensive Numerical Validation")
    print("  PyTorch vs TTSim (Components + End-to-End)")
    print("=" * 80)

    results = {}

    # Part A: Component tests
    results["01_MLP"] = test_01_mlp_numerical()
    results["02_PostProcess"] = test_02_postprocess_numerical()
    results["03_sigmoid_focal_loss"] = test_03_sigmoid_focal_loss_numerical()
    results["04_dice_loss"] = test_04_dice_loss_numerical()
    results["05_inverse_sigmoid"] = test_05_inverse_sigmoid_numerical()
    results["06_SetCriterion"] = test_06_set_criterion_numerical()
    results["07_PostProcessSegm"] = test_07_postprocess_segm_numerical()
    results["08_SetCriterion_aux"] = test_08_set_criterion_aux_numerical()
    results["09_MLP_intermediates"] = test_09_mlp_intermediates()

    # Part B: End-to-end pipeline tests
    results["10_Transformer"] = test_10_transformer_numerical()
    results["11_MLP_standalone"] = test_11_mlp_standalone()
    results["12_inverse_sigmoid"] = test_12_inverse_sigmoid()
    results["13_Heads"] = test_13_heads_on_transformer_output()
    results["14_Full_DETR_1x1"] = test_14_full_detr_numerical()
    results["15_Full_DETR_3x3"] = test_15_full_detr_multi_layer()

    # Summary
    section("SUMMARY")

    print("\n  Part A — Component Tests:")
    for name, ok in list(results.items())[:9]:
        symbol = "✓" if ok else "✗"
        print(f"    {symbol}  {name}")

    print("\n  Part B — End-to-End Pipeline Tests:")
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
    REPORT_PATH = os.path.join(REPORT_DIR, "deformable_DETR_numerical_validation.md")

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
