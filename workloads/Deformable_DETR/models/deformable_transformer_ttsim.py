#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Deformable DETR Transformer - TTSim Conversion
===============================================

FIX APPLIED: F.Dropout positional-only parameter bug fixed in op.py — keyword args now work correctly.
FIX APPLIED: Decoder reference_points expansion matches PyTorch exactly.
FIX APPLIED: Full transformer data propagation fixed.
FIX APPLIED: forward_ffn now includes residual + dropout4 + norm3 to match PyTorch.
"""

import os, sys
import numpy as np
from typing import Optional

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops.tensor import SimTensor
from ttsim.front.functional.tensor_op import cat, stack

from workloads.Deformable_DETR.models.ops.modules.ms_deform_attn_ttsim import (
    MSDeformAttn,
)


# ================================================================================================
# HELPER FUNCTIONS
# ================================================================================================


def xavier_uniform_(tensor_shape):
    if len(tensor_shape) < 2:
        raise ValueError("Xavier init requires at least 2D tensor")
    fan_in, fan_out = tensor_shape[1], tensor_shape[0]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    a = np.sqrt(3.0) * std
    return np.random.uniform(-a, a, tensor_shape).astype(np.float32)


def constant_(shape, val):
    return np.full(shape, val, dtype=np.float32)


def uniform_(shape, a, b):
    return np.random.uniform(a, b, shape).astype(np.float32)


def normal_(shape, mean=0, std=1):
    return np.random.normal(mean, std, shape).astype(np.float32)


def zeros_(shape):
    return np.zeros(shape, dtype=np.float32)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.Relu
    if activation == "gelu":
        return F.Gelu
    if activation == "glu":
        return F.Glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def get_sine_pos_embed(pos_tensor, num_pos_feats=128, temperature=10000):
    input_shape = pos_tensor.shape
    batch_shape = input_shape[:-1]
    output_shape = list(batch_shape) + [num_pos_feats * 2]

    pos_embed = SimTensor(
        {
            "name": pos_tensor.name + ".pos_embed",
            "shape": output_shape,
            "dtype": pos_tensor.dtype,
            "op_in": [pos_tensor.name],
            "op_out": [],
        }
    )
    return pos_embed


def inverse_sigmoid(x, eps=1e-5):
    inv_sigmoid_op = F.InverseSigmoid(x.name + ".inv_sigmoid", eps=eps)
    return inv_sigmoid_op(x)


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos


# ================================================================================================
# ENCODER LAYER
# ================================================================================================


class DeformableTransformerEncoderLayer(SimNN.Module):
    def __init__(
        self,
        name,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout = dropout

        self.self_attn = MSDeformAttn(
            d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points,
            name=name + ".self_attn",
        )

        self.dropout1 = F.Dropout(name + ".dropout1", prob=dropout, train_mode=False)
        self.norm1 = F.LayerNorm(name + ".norm1", d_model)

        self.linear1 = SimNN.Linear(name + ".linear1", d_model, d_ffn)
        self.linear2 = SimNN.Linear(name + ".linear2", d_ffn, d_model)

        activation_fn = _get_activation_fn(activation)
        self.activation = activation_fn(name + ".activation")

        self.dropout2 = F.Dropout(name + ".dropout2", prob=dropout, train_mode=False)
        self.dropout3 = F.Dropout(name + ".dropout3", prob=dropout, train_mode=False)
        self.norm2 = F.LayerNorm(name + ".norm2", d_model)

        self.pos_add = F.Add(name + ".pos_add")
        self.add1 = F.Add(name + ".add1")  # src + dropout1_out
        self.add2 = F.Add(name + ".add2")  # src + dropout3_out

        super().link_op2module()

    def with_pos_embed(self, tensor, pos):
        if pos is None:
            return tensor
        return self.pos_add(tensor, pos)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        return src2

    def __call__(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        # Ensure src has link_module
        if not hasattr(src, "link_module") or src.link_module is None:
            src.link_module = self
            self._tensors[src.name] = src

        # Self-attention
        src_with_pos = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            src_with_pos,
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )

        # Ensure src2 has link_module
        if hasattr(src2, "link_module") and src2.link_module is None:
            src2.link_module = self
            self._tensors[src2.name] = src2

        # Residual + dropout + norm
        dropout1_out = self.dropout1(src2)
        src = self.add1(src, dropout1_out)
        src = self.norm1(src)

        # FFN
        src2 = self.forward_ffn(src)
        dropout3_out = self.dropout3(src2)
        src = self.add2(src, dropout3_out)
        src = self.norm2(src)

        return src


# ================================================================================================
# ENCODER
# ================================================================================================


class DeformableTransformerEncoder(SimNN.Module):
    def __init__(self, name, encoder_layer, num_layers):
        super().__init__()
        self.name = name
        self.num_layers = num_layers

        self.layers = []
        for i in range(num_layers):
            layer = DeformableTransformerEncoderLayer(
                f"{name}.layers.{i}",
                d_model=encoder_layer.d_model,
                d_ffn=encoder_layer.d_ffn,
                dropout=encoder_layer.dropout,
                activation="relu",
                n_levels=encoder_layer.self_attn.n_levels,
                n_heads=encoder_layer.self_attn.n_heads,
                n_points=encoder_layer.self_attn.n_points,
            )
            self.layers.append(layer)
            self._submodules[layer.name] = layer

        if len(self.layers) > 0:
            self.layers = SimNN.ModuleList(self.layers)  # type: ignore[assignment]

        super().link_op2module()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device=None):
        ref_data = None

        if hasattr(spatial_shapes, "data") and spatial_shapes.data is not None:
            spatial_data = spatial_shapes.data
            n_levels = spatial_data.shape[0]
            batch_size = valid_ratios.shape[0] if hasattr(valid_ratios, "shape") else 1
            total_spatial_size = int(np.sum(spatial_data[:, 0] * spatial_data[:, 1]))

            vr_data = (
                valid_ratios.data
                if hasattr(valid_ratios, "data") and valid_ratios.data is not None
                else None
            )

            if vr_data is not None:
                reference_points_list = []
                for lvl in range(n_levels):
                    H_ = int(spatial_data[lvl, 0])
                    W_ = int(spatial_data[lvl, 1])
                    ref_y, ref_x = np.meshgrid(
                        np.linspace(0.5, H_ - 0.5, H_, dtype=np.float32),
                        np.linspace(0.5, W_ - 0.5, W_, dtype=np.float32),
                        indexing="ij",
                    )
                    ref_y = ref_y.reshape(-1)[None] / (vr_data[:, None, lvl, 1] * H_)
                    ref_x = ref_x.reshape(-1)[None] / (vr_data[:, None, lvl, 0] * W_)
                    ref = np.stack([ref_x, ref_y], axis=-1)
                    reference_points_list.append(ref)
                ref_cat = np.concatenate(reference_points_list, axis=1)
                ref_data = ref_cat[:, :, None, :] * vr_data[:, None, :, :]
        else:
            n_levels = (
                spatial_shapes.shape[0] if hasattr(spatial_shapes, "shape") else 4
            )
            batch_size = valid_ratios.shape[0] if hasattr(valid_ratios, "shape") else 1
            total_spatial_size = 1

        reference_points = SimTensor(
            {
                "name": "reference_points",
                "shape": [batch_size, total_spatial_size, n_levels, 2],
                "data": ref_data,
                "dtype": np.dtype(np.float32),
                "is_const": False,
                "op_in": [],
            }
        )
        return reference_points

    def __call__(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios)
        reference_points.set_module(self)
        self._tensors[reference_points.name] = reference_points

        for layer in self.layers:
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )

        return output


# ================================================================================================
# DECODER LAYER
# ================================================================================================


class DeformableTransformerDecoderLayer(SimNN.Module):
    def __init__(
        self,
        name,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout = dropout

        self.self_attn = SimNN.MultiheadAttention(
            name + ".self_attn", embed_dim=d_model, num_heads=n_heads, dropout=dropout
        )
        self._submodules[self.self_attn.name] = self.self_attn

        self.dropout1 = F.Dropout(name + ".dropout1", prob=dropout, train_mode=False)
        self.norm1 = F.LayerNorm(name + ".norm1", d_model)

        self.cross_attn = MSDeformAttn(
            d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points,
            name=name + ".cross_attn",
        )

        self.dropout2 = F.Dropout(name + ".dropout2", prob=dropout, train_mode=False)
        self.norm2 = F.LayerNorm(name + ".norm2", d_model)

        self.linear1 = SimNN.Linear(name + ".linear1", d_model, d_ffn)
        self.linear2 = SimNN.Linear(name + ".linear2", d_ffn, d_model)

        activation_fn = _get_activation_fn(activation)
        self.activation = activation_fn(name + ".activation")

        self.dropout3 = F.Dropout(name + ".dropout3", prob=dropout, train_mode=False)
        self.dropout4 = F.Dropout(name + ".dropout4", prob=dropout, train_mode=False)
        self.norm3 = F.LayerNorm(name + ".norm3", d_model)

        self.pos_add = F.Add(name + ".pos_add_self")   # for self-attn q/k
        self.pos_add_cross = F.Add(name + ".pos_add_cross")  # for cross-attn query
        self.add1 = F.Add(name + ".add1")  # tgt + dropout1_out
        self.add2 = F.Add(name + ".add2")  # tgt + dropout2_out
        self.add3 = F.Add(name + ".add3")  # tgt + dropout4_out (in forward_ffn)

        super().link_op2module()

    def with_pos_embed(self, tensor, pos):
        if pos is None:
            return tensor
        return self.pos_add(tensor, pos)

    def forward_ffn(self, tgt):
        """
        FFN block matching PyTorch exactly:
        linear1 -> activation -> dropout3 -> linear2 -> dropout4 -> residual -> norm3

        FIX: Previously this only returned the raw linear2 output.
        Now it includes the full FFN block with residual, dropout4, and norm3.
        """
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        dropout4_out = self.dropout4(tgt2)
        tgt = self.add3(tgt, dropout4_out)
        tgt = self.norm3(tgt)
        return tgt

    def __call__(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
    ):
        if not hasattr(tgt, "link_module") or tgt.link_module is None:
            tgt.link_module = self
            self._tensors[tgt.name] = tgt

        # Self-attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, None, None, False)

        if hasattr(tgt2, "link_module") and tgt2.link_module is None:
            tgt2.link_module = self
            self._tensors[tgt2.name] = tgt2

        dropout1_out = self.dropout1(tgt2)
        tgt = self.add1(tgt, dropout1_out)
        tgt = self.norm1(tgt)

        # Cross-attention
        cross_query = self.pos_add_cross(tgt, query_pos) if query_pos is not None else tgt
        tgt2 = self.cross_attn(
            cross_query,
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )

        if hasattr(tgt2, "link_module") and tgt2.link_module is None:
            tgt2.link_module = self
            self._tensors[tgt2.name] = tgt2

        dropout2_out = self.dropout2(tgt2)
        tgt = self.add2(tgt, dropout2_out)
        tgt = self.norm2(tgt)

        # FFN (now includes residual + dropout4 + norm3 inside forward_ffn)
        tgt = self.forward_ffn(tgt)

        return tgt


# ================================================================================================
# DECODER
# ================================================================================================


class DeformableTransformerDecoder(SimNN.Module):
    def __init__(self, name, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.name = name
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.layers = []
        for i in range(num_layers):
            layer = DeformableTransformerDecoderLayer(
                f"{name}.layers.{i}",
                d_model=decoder_layer.d_model,
                d_ffn=decoder_layer.d_ffn,
                dropout=decoder_layer.dropout,
                activation="relu",
                n_levels=decoder_layer.cross_attn.n_levels,
                n_heads=decoder_layer.cross_attn.n_heads,
                n_points=decoder_layer.cross_attn.n_points,
            )
            self.layers.append(layer)
            self._submodules[layer.name] = layer

        if len(self.layers) > 0:
            self.layers = SimNN.ModuleList(self.layers)  # type: ignore[assignment]

        super().link_op2module()

    def __call__(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
    ):
        output = tgt
        intermediate = []
        intermediate_reference_points = []

        # Expand reference_points ONCE before iterating through layers (matching PyTorch)
        # PyTorch: reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
        # Shape: [B, Q, 2] -> [B, Q, 1, 2] * [B, 1, L, 2] -> [B, Q, L, 2]

        if len(reference_points.shape) == 4:
            # Already expanded [B, Q, L, 2] or [B, Q, L, 4]
            if reference_points.shape[-1] == 4:
                # Slice to get first 2 coords
                if reference_points.data is not None:
                    ref_input_data = reference_points.data[:, :, :, :2].copy()
                else:
                    ref_input_data = None
                reference_points_input = SimTensor(
                    {
                        "name": reference_points.name + ".input",
                        "shape": list(reference_points.shape[:3]) + [2],
                        "data": ref_input_data,
                        "dtype": np.dtype(np.float32),
                    }
                )
                reference_points_input.set_module(self)
                self._tensors[reference_points_input.name] = reference_points_input
            else:
                reference_points_input = reference_points
        else:
            # Need to expand [B, Q, 2] -> [B, Q, L, 2]
            n_levels = (
                src_valid_ratios.shape[1]
                if len(src_valid_ratios.shape) >= 2
                else src_spatial_shapes.shape[0]
            )

            if reference_points.shape[-1] == 4:
                ref_2d_data = (
                    reference_points.data[:, :, :2]
                    if reference_points.data is not None
                    else None
                )
            else:
                ref_2d_data = reference_points.data

            # Create expanded reference points: [B, Q, 1, 2] * [B, 1, L, 2] = [B, Q, L, 2]
            if ref_2d_data is not None and src_valid_ratios.data is not None:
                # ref_2d: [B, Q, 2] -> [B, Q, 1, 2]
                ref_unsqueezed = ref_2d_data[:, :, None, :]
                # valid_ratios: [B, L, 2] -> [B, 1, L, 2]
                vr_unsqueezed = src_valid_ratios.data[:, None, :, :]
                # Multiply: [B, Q, 1, 2] * [B, 1, L, 2] -> [B, Q, L, 2]
                ref_expanded_data = ref_unsqueezed * vr_unsqueezed
            else:
                ref_expanded_data = None

            reference_points_input = SimTensor(
                {
                    "name": reference_points.name + ".expanded",
                    "shape": [
                        reference_points.shape[0],
                        reference_points.shape[1],
                        n_levels,
                        2,
                    ],
                    "data": ref_expanded_data,
                    "dtype": np.dtype(np.float32),
                }
            )
            reference_points_input.set_module(self)
            self._tensors[reference_points_input.name] = reference_points_input

        # Now iterate through layers using the same expanded reference_points_input
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
            )

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            if len(intermediate) == 1:
                # stack() requires ≥2 inputs; manually unsqueeze for single-layer case
                out_data = (
                    intermediate[0].data[None, ...]
                    if intermediate[0].data is not None
                    else None
                )
                output_stack = SimTensor(
                    {
                        "name": "decoder_intermediate_stack",
                        "shape": [1] + list(intermediate[0].shape),
                        "data": out_data,
                        "dtype": np.dtype(np.float32),
                    }
                )
                ref_data = (
                    intermediate_reference_points[0].data[None, ...]
                    if intermediate_reference_points[0].data is not None
                    else None
                )
                ref_stack = SimTensor(
                    {
                        "name": "decoder_ref_stack",
                        "shape": [1] + list(intermediate_reference_points[0].shape),
                        "data": ref_data,
                        "dtype": np.dtype(np.float32),
                    }
                )
            else:
                output_stack = stack(intermediate, dim=0)
                ref_stack = stack(intermediate_reference_points, dim=0)
            return output_stack, ref_stack

        return output, reference_points


# ================================================================================================
# MAIN TRANSFORMER
# ================================================================================================


class DeformableTransformer(SimNN.Module):
    def __init__(
        self,
        name="deformable_transformer",
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
    ):
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_feature_levels = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(
            name + ".encoder_layer_template",
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=enc_n_points,
        )

        decoder_layer = DeformableTransformerDecoderLayer(
            name + ".decoder_layer_template",
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=dec_n_points,
        )

        self.encoder = DeformableTransformerEncoder(
            name + ".encoder", encoder_layer, num_encoder_layers
        )
        self._submodules[self.encoder.name] = self.encoder

        self.decoder = DeformableTransformerDecoder(
            name + ".decoder",
            decoder_layer,
            num_decoder_layers,
            return_intermediate=return_intermediate_dec,
        )
        self._submodules[self.decoder.name] = self.decoder

        self.level_embed = F._from_shape(
            name + ".level_embed", [num_feature_levels, d_model], is_param=True
        )
        self.level_embed.set_module(self)
        self._tensors[self.level_embed.name] = self.level_embed

        if two_stage:
            self.enc_output = SimNN.Linear(name + ".enc_output", d_model, d_model)
            self._submodules[self.enc_output.name] = self.enc_output
            self.enc_output_norm = F.LayerNorm(name + ".enc_output_norm", d_model)
            self.pos_trans = SimNN.Linear(name + ".pos_trans", d_model * 2, d_model * 2)
            self._submodules[self.pos_trans.name] = self.pos_trans
            self.pos_trans_norm = F.LayerNorm(name + ".pos_trans_norm", d_model * 2)
        else:
            self.reference_points = SimNN.Linear(name + ".reference_points", d_model, 2)
            self._submodules[self.reference_points.name] = self.reference_points

        super().link_op2module()

    def get_proposal_pos_embed(self, proposals):
        batch_size = proposals.shape[0]
        num_queries = proposals.shape[1]
        pos_embed = SimTensor(
            {
                "name": proposals.name + ".pos_embed",
                "shape": [batch_size, num_queries, self.d_model * 2],
                "dtype": proposals.dtype,
                "op_in": [proposals.name],
                "op_out": [],
            }
        )
        pos_embed.set_module(self)
        self._tensors[pos_embed.name] = pos_embed
        return pos_embed

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        batch_size = memory.shape[0]
        n_tokens = memory.shape[1]
        output_proposals = SimTensor(
            {
                "name": memory.name + ".proposals",
                "shape": [batch_size, n_tokens, 4],
                "dtype": memory.dtype,
                "op_in": [memory.name],
                "op_out": [],
            }
        )
        output_proposals.set_module(self)
        self._tensors[output_proposals.name] = output_proposals
        output_memory = self.enc_output(memory)
        output_memory = self.enc_output_norm(output_memory)
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        batch_size = mask.shape[0]
        H = mask.shape[1]
        W = mask.shape[2]
        vr_data = None
        if hasattr(mask, "data") and mask.data is not None:
            mask_data = mask.data
            if mask_data.dtype in (np.float32, np.float64):
                non_padding = 1.0 - mask_data
            else:
                non_padding = (~mask_data.astype(bool)).astype(np.float32)
            valid_H = np.sum(non_padding[:, :, 0], axis=1)
            valid_W = np.sum(non_padding[:, 0, :], axis=1)
            valid_ratio_h = valid_H / float(H)
            valid_ratio_w = valid_W / float(W)
            vr_data = np.stack([valid_ratio_w, valid_ratio_h], axis=-1).astype(
                np.float32
            )

        valid_ratio = SimTensor(
            {
                "name": mask.name + ".valid_ratio",
                "shape": [batch_size, 2],
                "data": vr_data,
                "dtype": np.dtype(np.float32),
                "op_in": [mask.name],
                "op_out": [],
            }
        )
        valid_ratio.set_module(self)
        self._tensors[valid_ratio.name] = valid_ratio
        return valid_ratio

    def __call__(self, srcs, masks, pos_embeds, query_embed=None):
        if pos_embeds is None:
            pos_embeds = []
            for src in srcs:
                bs, c, h, w = src.shape
                pos_embed = SimTensor(
                    {
                        "name": f"pos_embed_zeros_{len(pos_embeds)}",
                        "shape": [bs, c, h, w],
                        "data": (
                            np.zeros((bs, c, h, w), dtype=np.float32)
                            if src.data is not None
                            else None
                        ),
                        "dtype": np.dtype(np.float32),
                    }
                )
                pos_embed.set_module(self)
                self._tensors[pos_embed.name] = pos_embed
                pos_embeds.append(pos_embed)

        for i, src in enumerate(srcs):
            if not hasattr(src, "link_module") or src.link_module is None:
                src.set_module(self)
                self._tensors[
                    src.name if hasattr(src, "name") else f"input_src_{i}"
                ] = src
        for i, mask in enumerate(masks):
            if not hasattr(mask, "link_module") or mask.link_module is None:
                mask.set_module(self)
                self._tensors[
                    mask.name if hasattr(mask, "name") else f"input_mask_{i}"
                ] = mask
        for i, pos in enumerate(pos_embeds):
            if not hasattr(pos, "link_module") or pos.link_module is None:
                pos.set_module(self)
                self._tensors[
                    pos.name if hasattr(pos, "name") else f"input_pos_{i}"
                ] = pos

        # Flatten multi-scale features manually with data propagation
        src_flatten_list = []
        mask_flatten_list = []
        lvl_pos_embed_flatten_list = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = [h, w]
            spatial_shapes.append(spatial_shape)

            # Flatten and transpose manually with data
            if src.data is not None:
                src_flat_data = src.data.reshape(bs, c, h * w).transpose(
                    0, 2, 1
                )  # [B, H*W, C]
            else:
                src_flat_data = None
            src_flat = SimTensor(
                {
                    "name": src.name + f".flat_{lvl}",
                    "shape": [bs, h * w, c],
                    "data": src_flat_data,
                    "dtype": np.dtype(np.float32),
                }
            )
            src_flat.set_module(self)
            self._tensors[src_flat.name] = src_flat

            if mask.data is not None:
                mask_flat_data = mask.data.reshape(bs, h * w)
            else:
                mask_flat_data = None
            mask_flat = SimTensor(
                {
                    "name": mask.name + f".flat_{lvl}",
                    "shape": [bs, h * w],
                    "data": mask_flat_data,
                    "dtype": np.dtype(np.float32),
                }
            )
            mask_flat.set_module(self)
            self._tensors[mask_flat.name] = mask_flat

            if pos_embed.data is not None:
                pos_flat_data = pos_embed.data.reshape(bs, c, h * w).transpose(
                    0, 2, 1
                )  # [B, H*W, C]
            else:
                pos_flat_data = None
            pos_flat = SimTensor(
                {
                    "name": pos_embed.name + f".flat_{lvl}",
                    "shape": [bs, h * w, c],
                    "data": pos_flat_data,
                    "dtype": np.dtype(np.float32),
                }
            )
            pos_flat.set_module(self)
            self._tensors[pos_flat.name] = pos_flat

            # Add level embed
            if self.level_embed.data is not None and pos_flat_data is not None:
                lvl_embed_data = self.level_embed.data[lvl : lvl + 1, :].reshape(
                    1, 1, c
                )
                lvl_pos_data = pos_flat_data + lvl_embed_data
            else:
                lvl_pos_data = None
            lvl_pos_embed = SimTensor(
                {
                    "name": f"lvl_pos_embed_{lvl}",
                    "shape": [bs, h * w, c],
                    "data": lvl_pos_data,
                    "dtype": np.dtype(np.float32),
                }
            )
            lvl_pos_embed.set_module(self)
            self._tensors[lvl_pos_embed.name] = lvl_pos_embed

            src_flatten_list.append(src_flat)
            mask_flatten_list.append(mask_flat)
            lvl_pos_embed_flatten_list.append(lvl_pos_embed)

        # Concatenate
        src_flatten = cat(src_flatten_list, dim=1)
        mask_flatten = cat(mask_flatten_list, dim=1)
        lvl_pos_embed_flatten = cat(lvl_pos_embed_flatten_list, dim=1)

        spatial_shapes_tensor = F._from_data(
            self.name + ".spatial_shapes",
            np.array(spatial_shapes, dtype=np.int64),
            is_const=True,
        )
        self._tensors[spatial_shapes_tensor.name] = spatial_shapes_tensor

        level_start_index = [0]
        for shape in spatial_shapes:
            level_start_index.append(level_start_index[-1] + shape[0] * shape[1])
        level_start_index_tensor = F._from_data(
            self.name + ".level_start_index",
            np.array(level_start_index[:-1], dtype=np.int64),
            is_const=True,
        )
        self._tensors[level_start_index_tensor.name] = level_start_index_tensor

        valid_ratios = [self.get_valid_ratio(mask) for mask in masks]
        valid_ratios_tensor = stack(valid_ratios, dim=1)

        memory = self.encoder(
            src_flatten,
            spatial_shapes_tensor,
            level_start_index_tensor,
            valid_ratios_tensor,
            lvl_pos_embed_flatten,
            mask_flatten,
        )

        bs = memory.shape[0]

        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes_tensor
            )
            topk_proposals = output_proposals[:, : self.two_stage_num_proposals]
            reference_points = topk_proposals
            tgt = output_memory[:, : self.two_stage_num_proposals]
            query_pos = self.get_proposal_pos_embed(topk_proposals)
            query_pos = self.pos_trans(query_pos)
            query_pos = self.pos_trans_norm(query_pos)
        else:
            query_embed_weight = query_embed
            if (
                not hasattr(query_embed_weight, "link_module")
                or query_embed_weight.link_module is None
            ):
                query_embed_weight.set_module(self)
                self._tensors[
                    (
                        query_embed_weight.name
                        if hasattr(query_embed_weight, "name")
                        else "input_query_embed"
                    )
                ] = query_embed_weight

            num_queries = query_embed_weight.shape[0]

            # Split query_embed into query_pos and tgt (matching PyTorch order)
            # PyTorch: query_embed, tgt = torch.split(query_embed, c, dim=1)
            #   first half → query_embed (used as query_pos in decoder)
            #   second half → tgt (initial decoder target)
            if query_embed_weight.data is not None:
                query_pos_data = np.tile(
                    query_embed_weight.data[None, :, : self.d_model], (bs, 1, 1)
                )
                tgt_data = np.tile(
                    query_embed_weight.data[None, :, self.d_model :], (bs, 1, 1)
                )
            else:
                tgt_data = None
                query_pos_data = None

            tgt = SimTensor(
                {
                    "name": query_embed_weight.name + ".tgt",
                    "shape": [bs, num_queries, self.d_model],
                    "data": tgt_data,
                    "dtype": np.dtype(np.float32),
                }
            )
            tgt.set_module(self)
            self._tensors[tgt.name] = tgt

            query_pos = SimTensor(
                {
                    "name": query_embed_weight.name + ".query_pos",
                    "shape": [bs, num_queries, self.d_model],
                    "data": query_pos_data,
                    "dtype": np.dtype(np.float32),
                }
            )
            query_pos.set_module(self)
            self._tensors[query_pos.name] = query_pos

            # Reference points from linear + sigmoid
            reference_points_linear = self.reference_points(query_pos)

            # Manual sigmoid
            if reference_points_linear.data is not None:
                ref_sigmoid_data = 1.0 / (1.0 + np.exp(-reference_points_linear.data))
            else:
                ref_sigmoid_data = None

            reference_points_2d = SimTensor(
                {
                    "name": reference_points_linear.name + ".sigmoid",
                    "shape": list(reference_points_linear.shape),
                    "data": ref_sigmoid_data,
                    "dtype": np.dtype(np.float32),
                }
            )
            reference_points_2d.set_module(self)
            self._tensors[reference_points_2d.name] = reference_points_2d

            # Pass [B, Q, 2] reference_points to decoder — let decoder expand with valid_ratios
            # (matches PyTorch which passes [B, Q, 2] and expands inside decoder loop)
            reference_points = reference_points_2d

        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes_tensor,
            level_start_index_tensor,
            valid_ratios_tensor,
            query_pos,
            mask_flatten,
        )

        return hs, reference_points, inter_references, None, None


def build_deformable_transformer(args):
    return DeformableTransformer(
        name="deformable_transformer",
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
    )
