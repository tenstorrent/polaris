#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
import ttsim.front.functional.sim_nn as SimNN
from workloads.ttnn.vadv2.tt.tt_base_transformer_layer import TtBaseTransformerLayer
from workloads.ttnn.vadv2.tt.tt_utils import inverse_sigmoid
from workloads.ttnn.vadv2.tt.tt_utils import DictAsAttr

class TtDetectionTransformerDecoder(SimNN.Module):
    def __init__(self, num_layers, embed_dim, num_heads, params, params_branches, device):
        super().__init__()
        self.name = "TtDetectionTransformerDecoder"
        self.return_intermediate = True
        self.device = device
        self.params = params
        params = DictAsAttr(params)
        self.params_branches = params_branches
        self.layers = [
            TtBaseTransformerLayer(
                params.layers[f"layer{i}"],
                self.device,
                attn_cfgs=[
                    {
                        "type": "MultiheadAttention",
                        "embed_dims": embed_dim,
                        "num_heads": num_heads,
                    },
                    {
                        "type": "CustomMSDeformableAttention",
                        "embed_dims": embed_dim,
                        "num_levels": 1,
                    },
                ],
                ffn_cfgs={
                    "type": "FFN",
                    "embed_dims": embed_dim,
                    "feedforward_channels": 512,
                    "num_fcs": 2,
                    "ffn_drop": 0.0,
                    "act_cfg": {"type": "ReLU", "inplace": True},
                },
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                norm_cfg={"type": "LN"},
                init_cfg=None,
                batch_first=False,
                kwargs={
                    "feedforward_channels": 512,
                    "act_cfg": {"type": "ReLU", "inplace": True},
                    "ffn_num_fcs": 2,
                },
            )
            for i in range(num_layers)
        ]

    def __call__(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        reg_branches=None,
        cls_branches=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        N, W, C = reference_points.shape
        # reference_points = ttnn.Tensor(shape=reference_points.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, data=reference_points.data)
        # reference_points.set_module(self)
        # self._op_hndls['reference_points'] = reference_points
        for lid, layer in enumerate(self.layers):
            # reference_points_input = reference_points[..., :2]
            reference_points_input = ttnn._rand(shape=(N, W, 2), dtype=ttnn.bfloat16, device=self.device)
            reference_points_input = ttnn.unsqueeze(reference_points_input, 2)
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
            )
            output = ttnn.permute(output, (1, 0, 2))
            ttnn.ReadDeviceProfiler(self.device)

            if reg_branches is not None:
                # Select reg_branch layers for current lid
                layers = self.params_branches["reg_branches"][str(lid)]

                tmp = output
                for i in range(3):
                    tmp = ttnn.linear(
                        tmp, layers[str(i)]["weight"], bias=layers[str(i)]["bias"], memory_config=ttnn.L1_MEMORY_CONFIG
                    )
                    if i < 2:
                        tmp = ttnn.relu(tmp)
                assert reference_points.shape[-1] == 3

                tmp_shape = tmp.shape
                ref_pt_shape = reference_points.shape
                t2 = ttnn.Tensor(shape=(tmp_shape[0], tmp_shape[1], 2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
                r2 = ttnn.Tensor(shape=(ref_pt_shape[0], ref_pt_shape[1], 2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
                updated_xy = t2 + inverse_sigmoid(r2)  # shape (..., 2)

                t45 = ttnn.Tensor(shape=(tmp_shape[0], tmp_shape[1], 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
                r3 = ttnn.Tensor(shape=(ref_pt_shape[0], ref_pt_shape[1], 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
                updated_z = t45 + inverse_sigmoid(r3)  # shape (..., 1)

                new_reference_points = ttnn.concat(updated_xy, updated_z, axis=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(tmp)
                new_reference_points = ttnn.sigmoid(new_reference_points, memory_config=ttnn.L1_MEMORY_CONFIG)

                reference_points = new_reference_points

            output = ttnn.permute(output, (1, 0, 2))
            # print('Decoder output shape', output)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            a = ttnn.stack(intermediate, dim=0)
            b = ttnn.stack(intermediate_reference_points, dim=0)
            return a, b
        return output, reference_points


class TtMapDetectionTransformerDecoder(SimNN.Module):
    def __init__(self, num_layers, embed_dim, num_heads, params, params_branches, device):
        super().__init__()
        self.return_intermediate = True
        self.name = "TtMapDetectionTransformerDecoder"
        self.device = device
        self.params = params
        params = DictAsAttr(params)
        self.params_branches = params_branches
        self.layers = [
            TtBaseTransformerLayer(
                params.layers[f"layer{i}"],
                self.device,
                attn_cfgs=[
                    {
                        "type": "MultiheadAttention",
                        "embed_dims": embed_dim,
                        "num_heads": num_heads,
                    },
                    {
                        "type": "CustomMSDeformableAttention",
                        "embed_dims": embed_dim,
                        "num_levels": 1,
                    },
                ],
                ffn_cfgs={
                    "type": "FFN",
                    "embed_dims": embed_dim,
                    "feedforward_channels": 512,
                    "num_fcs": 2,
                    "ffn_drop": 0.0,
                    "act_cfg": {"type": "ReLU", "inplace": True},
                },
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                norm_cfg={"type": "LN"},
                init_cfg=None,
                batch_first=False,
                kwargs={
                    "feedforward_channels": 512,
                    "act_cfg": {"type": "ReLU", "inplace": True},
                    "ffn_num_fcs": 2,
                },
            )
            for i in range(num_layers)
        ]

    def __call__(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        map_reg_branches=None,
        cls_branches=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points # reference_points[..., :2]
            reference_points_input = ttnn.unsqueeze(reference_points_input, 2)
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
            )
            output = ttnn.permute(output, (1, 0, 2))

            if map_reg_branches is not None:
                layers = self.params_branches.map_reg_branches[str(lid)]

                tmp = output
                for i in range(3):
                    tmp = ttnn.linear(
                        tmp, layers[str(i)].weight, bias=layers[str(i)].bias, memory_config=ttnn.L1_MEMORY_CONFIG
                    )
                    if i < 2:  # Apply ReLU after the first two layers
                        tmp = ttnn.relu(tmp)

                assert reference_points.shape[-1] == 2

                updated_xy = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])

                new_reference_points = ttnn.concat([updated_xy], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

                ttnn.deallocate(tmp)
                new_reference_points = ttnn.sigmoid(new_reference_points)

                reference_points = new_reference_points

            output = ttnn.permute(output, (1, 0, 2))
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            a = ttnn.stack(intermediate, dim=0)
            b = ttnn.stack(intermediate_reference_points, dim=0)
            return a, b
        return output, reference_points


class TtCustomTransformerDecoder:
    def __init__(self, params, device, num_layers, return_intermediate=False, embed_dim=256, num_heads=8):
        self.device = device
        self.params = params
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        params = DictAsAttr(params)
        self.layers = [
            TtBaseTransformerLayer(
                params.layers[f"layer{i}"],
                self.device,
                attn_cfgs=[
                    {
                        "type": "MultiheadAttention",
                        "embed_dims": 256,
                        "num_heads": 8,
                    }
                ],
                ffn_cfgs={
                    "type": "FFN",
                    "embed_dims": 256,
                    "feedforward_channels": 512,
                    "num_fcs": 2,
                    "act_cfg": {"type": "ReLU", "inplace": True},
                },
                operation_order=("cross_attn", "norm", "ffn", "norm"),
                norm_cfg={"type": "LN"},
                init_cfg=None,
                batch_first=False,
                kwargs={"feedforward_channels": 512},
            )
            for i in range(num_layers)
        ]

    def __call__(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        key_padding_mask=None,
        *args,
        **kwargs,
    ):
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return ttnn.stack(intermediate, dim=0)

        return query
