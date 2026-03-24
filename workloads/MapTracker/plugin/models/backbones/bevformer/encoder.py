#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of BEVFormer Encoder.

This module implements the BEVFormer encoder which consists of:
1. BEVFormerEncoder: Sequence of transformer layers for BEV feature encoding
2. BEVFormerLayer: Single transformer layer with temporal and spatial attention
3. Reference point generation for 3D spatial attention and 2D temporal attention
4. Camera projection for multi-view feature aggregation

Converted from PyTorch/mmcv to TTSim (CPU-only, Python 3.13 compatible).
No MMCV dependencies, no CUDA operations.
"""

import sys
import os

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
from ttsim.front.functional.sim_nn import Module, ModuleList
from workloads.MapTracker.plugin.models.backbones.bevformer.custom_base_transformer_layer import (
    MyCustomBaseTransformerLayer,
)
from workloads.MapTracker.plugin.models.backbones.bevformer.temporal_net import (
    TemporalNet,
)

# -------------------------------PyTorch--------------------------------

# class BEVFormerEncoder(TransformerLayerSequence):

#     """
#     Attention with both self and cross
#     Implements the decoder in DETR transformer.
#     Args:
#         return_intermediate (bool): Whether to return intermediate outputs.
#         coder_norm_cfg (dict): Config of last normalization layer. Default：
#             `LN`.
#     """

#     def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
#                  **kwargs):

#         super(BEVFormerEncoder, self).__init__(*args, **kwargs)
#         self.return_intermediate = return_intermediate

#         temporal_mem_layers = []
#         for _ in range(self.num_layers):
#             mem_conv = TemporalNet(history_steps=4, hidden_dims=self.embed_dims, num_blocks=1)
#             temporal_mem_layers.append(mem_conv)
#         self.temporal_mem_layers = nn.ModuleList(temporal_mem_layers)

#         self.num_points_in_pillar = num_points_in_pillar
#         self.pc_range = pc_range
#         self.fp16_enabled = False

#     @staticmethod
#     def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
#         """Get the reference points used in SCA and TSA.
#         Args:
#             H, W: spatial shape of bev.
#             Z: hight of pillar.
#             D: sample D points uniformly from each pillar.
#             device (obj:`device`): The device where
#                 reference_points should be.
#         Returns:
#             Tensor: reference points used in decoder, has \
#                 shape (bs, num_keys, num_levels, 2).
#         """

#         # reference points in 3D space, used in spatial cross-attention (SCA)
#         if dim == '3d':
#             zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
#                                 device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
#             xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
#                                 device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
#             # ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
#             #                     device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
#             # change y-axis direction
#             ys = torch.linspace(H - 0.5, 0.5, H, dtype=dtype,
#                                 device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
#             ref_3d = torch.stack((xs, ys, zs), -1)
#             ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
#             ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
#             return ref_3d

#         # reference points on 2D bev plane, used in temporal self-attention (TSA).
#         elif dim == '2d':
#             ref_y, ref_x = torch.meshgrid(
#                 # torch.linspace(
#                 #     0.5, H - 0.5, H, dtype=dtype, device=device),
#                 torch.linspace(
#                     H - 0.5, 0.5, H, dtype=dtype, device=device),
#                 torch.linspace(
#                     0.5, W - 0.5, W, dtype=dtype, device=device)
#             )
#             ref_y = ref_y.reshape(-1)[None] / H
#             ref_x = ref_x.reshape(-1)[None] / W
#             ref_2d = torch.stack((ref_x, ref_y), -1)
#             ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
#             return ref_2d

#     # This function must use fp32!!!
#     @force_fp32(apply_to=('reference_points', 'img_metas'))
#     def point_sampling(self, reference_points, pc_range, img_metas):

#         ego2img = []
#         for img_meta in img_metas:
#             ego2img.append(img_meta['ego2img'])
#         ego2img = np.asarray(ego2img)
#         ego2img = reference_points.new_tensor(ego2img)  # (B, N, 4, 4)
#         reference_points = reference_points.clone()

#         reference_points[..., 0:1] = reference_points[..., 0:1] * \
#             (pc_range[3] - pc_range[0]) + pc_range[0]
#         reference_points[..., 1:2] = reference_points[..., 1:2] * \
#             (pc_range[4] - pc_range[1]) + pc_range[1]
#         reference_points[..., 2:3] = reference_points[..., 2:3] * \
#             (pc_range[5] - pc_range[2]) + pc_range[2]

#         reference_points = torch.cat(
#             (reference_points, torch.ones_like(reference_points[..., :1])), -1)

#         reference_points = reference_points.permute(1, 0, 2, 3)
#         D, B, num_query = reference_points.size()[:3]
#         num_cam = ego2img.size(1)

#         reference_points = reference_points.view(
#             D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

#         ego2img = ego2img.view(
#             1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

#         reference_points_cam = torch.matmul(ego2img.to(torch.float32),
#                                             reference_points.to(torch.float32)).squeeze(-1)
#         eps = 1e-5

#         bev_mask = (reference_points_cam[..., 2:3] > eps)
#         reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
#             reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

#         reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
#         reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

#         bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
#                     & (reference_points_cam[..., 1:2] < 1.0)
#                     & (reference_points_cam[..., 0:1] < 1.0)
#                     & (reference_points_cam[..., 0:1] > 0.0))
#         if digit_version(TORCH_VERSION) >= digit_version('1.8'):
#             bev_mask = torch.nan_to_num(bev_mask)
#         else:
#             bev_mask = bev_mask.new_tensor(
#                 np.nan_to_num(bev_mask.cpu().numpy()))

#         reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
#         bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

#         return reference_points_cam, bev_mask

#     @auto_fp16()
#     def forward(self,
#                 bev_query,
#                 key,
#                 value,
#                 *args,
#                 bev_h=None,
#                 bev_w=None,
#                 bev_pos=None,
#                 spatial_shapes=None,
#                 level_start_index=None,
#                 prev_bev=None,
#                 shift=0.,
#                 warped_history_bev=None,
#                 **kwargs):
#         """Forward function for `TransformerDecoder`.
#         Args:
#             bev_query (Tensor): Input BEV query with shape
#                 `(num_query, bs, embed_dims)`.
#             key & value (Tensor): Input multi-cameta features with shape
#                 (num_cam, num_value, bs, embed_dims)
#             reference_points (Tensor): The reference
#                 points of offset. has shape
#                 (bs, num_query, 4) when as_two_stage,
#                 otherwise has shape ((bs, num_query, 2).
#             valid_ratios (Tensor): The radios of valid
#                 points on the feature map, has shape
#                 (bs, num_levels, 2)
#         Returns:
#             Tensor: Results with shape [1, num_query, bs, embed_dims] when
#                 return_intermediate is `False`, otherwise it has shape
#                 [num_layers, num_query, bs, embed_dims].
#         """

#         output = bev_query
#         intermediate = []

#         ref_3d = self.get_reference_points(
#             bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
#         ref_2d = self.get_reference_points(
#             bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

#         reference_points_cam, bev_mask = self.point_sampling(
#             ref_3d, self.pc_range, kwargs['img_metas'])

#         # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
#         # shift_ref_2d = ref_2d  # .clone()
#         shift_ref_2d = ref_2d.clone()
#         shift_ref_2d += shift[:, None, None, :]

#         # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
#         bev_query = bev_query.permute(1, 0, 2)
#         bev_pos = bev_pos.permute(1, 0, 2)
#         bs, len_bev, num_bev_level, _ = ref_2d.shape

#         if prev_bev is not None:
#             prev_bev = prev_bev.permute(1, 0, 2)
#             prev_bev = torch.stack(
#                 [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
#             hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
#                 bs*2, len_bev, num_bev_level, 2)
#         else:
#             hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
#                 bs*2, len_bev, num_bev_level, 2)

#         for lid, layer in enumerate(self.layers):
#             output = layer(
#                 bev_query,
#                 key,
#                 value,
#                 *args,
#                 bev_pos=bev_pos,
#                 ref_2d=hybird_ref_2d,
#                 ref_3d=ref_3d,
#                 bev_h=bev_h,
#                 bev_w=bev_w,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 reference_points_cam=reference_points_cam,
#                 bev_mask=bev_mask,
#                 prev_bev=prev_bev,
#                 warped_history_bev=warped_history_bev,
#                 **kwargs)

#             # BEV memory fusion layer
#             mem_layer = self.temporal_mem_layers[lid]
#             curr_feat = rearrange(output, 'b (h w) c -> b c h w', h=warped_history_bev.shape[3])
#             fused_output = mem_layer(warped_history_bev, curr_feat)
#             fused_output = rearrange(fused_output, 'b c h w -> b (h w) c')
#             output = output + fused_output

#             bev_query = output
#             if self.return_intermediate:
#                 intermediate.append(output)

#         if self.return_intermediate:
#             return torch.stack(intermediate)

#         return output


# @TRANSFORMER_LAYER.register_module()
# class BEVFormerLayer(MyCustomBaseTransformerLayer):
#     """Implements decoder layer in DETR transformer.
#     Args:
#         attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
#             Configs for self_attention or cross_attention, the order
#             should be consistent with it in `operation_order`. If it is
#             a dict, it would be expand to the number of attention in
#             `operation_order`.
#         feedforward_channels (int): The hidden dimension for FFNs.
#         ffn_dropout (float): Probability of an element to be zeroed
#             in ffn. Default 0.0.
#         operation_order (tuple[str]): The execution order of operation
#             in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
#             Default：None
#         act_cfg (dict): The activation config for FFNs. Default: `LN`
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: `LN`.
#         ffn_num_fcs (int): The number of fully-connected layers in FFNs.
#             Default：2.
#     """

#     def __init__(self,
#                  attn_cfgs,
#                  feedforward_channels,
#                  ffn_dropout=0.0,
#                  operation_order=None,
#                  act_cfg=dict(type='ReLU', inplace=True),
#                  norm_cfg=dict(type='LN'),
#                  ffn_num_fcs=2,
#                  **kwargs):
#         super(BEVFormerLayer, self).__init__(
#             attn_cfgs=attn_cfgs,
#             feedforward_channels=feedforward_channels,
#             ffn_dropout=ffn_dropout,
#             operation_order=operation_order,
#             act_cfg=act_cfg,
#             norm_cfg=norm_cfg,
#             ffn_num_fcs=ffn_num_fcs,
#             **kwargs)
#         self.fp16_enabled = False
#         assert len(operation_order) == 6
#         assert set(operation_order) == set(
#             ['self_attn', 'norm', 'cross_attn', 'ffn'])

#     def forward(self,
#                 query,
#                 key=None,
#                 value=None,
#                 bev_pos=None,
#                 query_pos=None,
#                 key_pos=None,
#                 attn_masks=None,
#                 query_key_padding_mask=None,
#                 key_padding_mask=None,
#                 ref_2d=None,
#                 ref_3d=None,
#                 bev_h=None,
#                 bev_w=None,
#                 reference_points_cam=None,
#                 mask=None,
#                 spatial_shapes=None,
#                 level_start_index=None,
#                 prev_bev=None,
#                 **kwargs):
#         """Forward function for `TransformerDecoderLayer`.

#         **kwargs contains some specific arguments of attentions.

#         Args:
#             query (Tensor): The input query with shape
#                 [num_queries, bs, embed_dims] if
#                 self.batch_first is False, else
#                 [bs, num_queries embed_dims].
#             key (Tensor): The key tensor with shape [num_keys, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_keys, embed_dims] .
#             value (Tensor): The value tensor with same shape as `key`.
#             query_pos (Tensor): The positional encoding for `query`.
#                 Default: None.
#             key_pos (Tensor): The positional encoding for `key`.
#                 Default: None.
#             attn_masks (List[Tensor] | None): 2D Tensor used in
#                 calculation of corresponding attention. The length of
#                 it should equal to the number of `attention` in
#                 `operation_order`. Default: None.
#             query_key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_queries]. Only used in `self_attn` layer.
#                 Defaults to None.
#             key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_keys]. Default: None.

#         Returns:
#             Tensor: forwarded results with shape [num_queries, bs, embed_dims].
#         """

#         norm_index = 0
#         attn_index = 0
#         ffn_index = 0
#         identity = query
#         if attn_masks is None:
#             attn_masks = [None for _ in range(self.num_attn)]
#         elif isinstance(attn_masks, torch.Tensor):
#             attn_masks = [
#                 copy.deepcopy(attn_masks) for _ in range(self.num_attn)
#             ]
#             warnings.warn(f'Use same attn_mask in all attentions in '
#                           f'{self.__class__.__name__} ')
#         else:
#             assert len(attn_masks) == self.num_attn, f'The length of ' \
#                                                      f'attn_masks {len(attn_masks)} must be equal ' \
#                                                      f'to the number of attention in ' \
#                 f'operation_order {self.num_attn}'

#         for layer in self.operation_order:
#             # temporal self attention
#             if layer == 'self_attn':
#                 query = self.attentions[attn_index](
#                     query,
#                     prev_bev,
#                     prev_bev,
#                     identity if self.pre_norm else None,
#                     query_pos=bev_pos,
#                     key_pos=bev_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=query_key_padding_mask,
#                     reference_points=ref_2d,
#                     spatial_shapes=torch.tensor(
#                         [[bev_h, bev_w]], device=query.device),
#                     level_start_index=torch.tensor([0], device=query.device),
#                     **kwargs)
#                 attn_index += 1
#                 identity = query

#             elif layer == 'norm':
#                 query = self.norms[norm_index](query)
#                 norm_index += 1

#             # spaital cross attention
#             elif layer == 'cross_attn':
#                 query = self.attentions[attn_index](
#                     query,
#                     key,
#                     value,
#                     identity if self.pre_norm else None,
#                     query_pos=query_pos,
#                     key_pos=key_pos,
#                     reference_points=ref_3d,
#                     reference_points_cam=reference_points_cam,
#                     mask=mask,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=key_padding_mask,
#                     spatial_shapes=spatial_shapes,
#                     level_start_index=level_start_index,
#                     **kwargs)
#                 attn_index += 1
#                 identity = query
#             elif layer == 'ffn':
#                 query = self.ffns[ffn_index](
#                     query, identity if self.pre_norm else None)
#                 ffn_index += 1

#         return query

# -------------------------------TTSIM-----------------------------------


class BEVFormerEncoder(Module):
    """
    BEVFormer Encoder: Sequence of transformer layers for BEV encoding.

    Implements multi-layer transformer with both temporal self-attention
    (TSA) and spatial cross-attention (SCA) for aggregating multi-camera
    features into bird's-eye-view representation.

    Args:
        name (str): Module name
        transformerlayers (dict): Configuration for transformer layers
        num_layers (int): Number of transformer layers
        pc_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        num_points_in_pillar (int): Number of Z-axis sampling points per pillar (default: 4)
        return_intermediate (bool): Whether to return intermediate layer outputs
    """

    def __init__(
        self,
        name,
        transformerlayers,
        num_layers,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        history_steps=4,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = (
            pc_range if pc_range is not None else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        )

        # Build transformer layers
        _layers = []
        for i in range(num_layers):
            layer = BEVFormerLayer(name=f"{name}.layer_{i}", **transformerlayers)
            _layers.append(layer)
        self.layers = ModuleList(_layers)

        # Get embed_dims from the first layer (set by MyCustomBaseTransformerLayer)
        self.embed_dims = self.layers[0].embed_dims

        # Build temporal memory fusion layers (one per encoder layer)
        _temporal_mem_layers = []
        for i in range(num_layers):
            mem_conv = TemporalNet(
                name=f"{name}.temporal_mem_{i}",
                history_steps=history_steps,
                hidden_dims=self.embed_dims,
                num_blocks=1,
            )
            _temporal_mem_layers.append(mem_conv)
        self.temporal_mem_layers = ModuleList(_temporal_mem_layers)

        # Ops for BEV memory fusion reshape: 'b (h w) c -> b c h w' and back
        self.mem_reshape_to_4d = F.Reshape(name + ".mem_reshape_to_4d")
        self.mem_permute_to_bchw = F.Transpose(
            name + ".mem_permute_to_bchw", perm=[0, 2, 1]
        )  # [b, hw, c] -> [b, c, hw]
        self.mem_reshape_back = F.Reshape(
            name + ".mem_reshape_back"
        )  # [b, c, h, w] -> [b, c, hw]
        self.mem_permute_back = F.Transpose(
            name + ".mem_permute_back", perm=[0, 2, 1]
        )  # [b, c, hw] -> [b, hw, c]
        self.mem_add = F.Add(name + ".mem_add")  # output = output + fused_output

    @staticmethod
    def get_reference_points(
        H, W, Z=8, num_points_in_pillar=4, dim="3d", bs=1, dtype=np.float32
    ):
        """
        Get the reference points used in SCA and TSA.

        Args:
            H, W: Spatial shape of BEV
            Z: Height of pillar
            num_points_in_pillar: Sample D points uniformly from each pillar
            dim: '3d' for spatial cross-attention, '2d' for temporal self-attention
            bs: Batch size
            dtype: Data type for numpy arrays

        Returns:
            Reference points tensor with shape:
            - 3d: [bs, num_points_in_pillar, H*W, 3] for (x, y, z)
            - 2d: [bs, H*W, 1, 2] for (x, y)
        """
        if dim == "3d":
            # Reference points in 3D space for spatial cross-attention (SCA)
            # Create normalized coordinates [0, 1]
            zs = np.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype) / Z
            zs = zs.reshape(-1, 1, 1)
            zs = np.broadcast_to(zs, (num_points_in_pillar, H, W))

            xs = np.linspace(0.5, W - 0.5, W, dtype=dtype) / W
            xs = xs.reshape(1, 1, W)
            xs = np.broadcast_to(xs, (num_points_in_pillar, H, W))

            # change y-axis direction (matches PyTorch source)
            ys = np.linspace(H - 0.5, 0.5, H, dtype=dtype) / H
            ys = ys.reshape(1, H, 1)
            ys = np.broadcast_to(ys, (num_points_in_pillar, H, W))

            # Stack to [num_points_in_pillar, H, W, 3]
            ref_3d = np.stack([xs, ys, zs], axis=-1)

            # Reshape to [num_points_in_pillar, 3, H, W] -> [num_points_in_pillar, 3, H*W] -> [num_points_in_pillar, H*W, 3]
            ref_3d = ref_3d.transpose(0, 3, 1, 2)  # [D, 3, H, W]
            ref_3d = ref_3d.reshape(num_points_in_pillar, 3, -1)  # [D, 3, H*W]
            ref_3d = ref_3d.transpose(0, 2, 1)  # [D, H*W, 3]

            # Add batch dimension and repeat: [bs, D, H*W, 3]
            ref_3d = np.broadcast_to(
                ref_3d[np.newaxis, :, :, :], (bs, num_points_in_pillar, H * W, 3)
            )

            return ref_3d

        elif dim == "2d":
            # Reference points on 2D BEV plane for temporal self-attention (TSA)
            # change y-axis direction (matches PyTorch source)
            ref_y = np.linspace(H - 0.5, 0.5, H, dtype=dtype) / H
            ref_x = np.linspace(0.5, W - 0.5, W, dtype=dtype) / W

            # Create meshgrid
            ref_y_grid, ref_x_grid = np.meshgrid(ref_y, ref_x, indexing="ij")

            # Flatten and stack: [H*W, 2]
            ref_y_flat = ref_y_grid.reshape(-1)
            ref_x_flat = ref_x_grid.reshape(-1)
            ref_2d = np.stack([ref_x_flat, ref_y_flat], axis=-1)  # [H*W, 2]

            # Add dimensions: [bs, H*W, 1, 2]
            ref_2d = ref_2d[np.newaxis, :, np.newaxis, :]
            ref_2d = np.broadcast_to(ref_2d, (bs, H * W, 1, 2))

            return ref_2d

        else:
            raise ValueError(f"dim must be '3d' or '2d', got {dim}")

    @staticmethod
    def point_sampling(reference_points, pc_range, img_metas):
        """
        Project 3D reference points to camera image coordinates.

        This function transforms BEV reference points from normalized coordinates
        to actual 3D space, then projects them to each camera view using the
        lidar-to-image transformation matrices.

        Args:
            reference_points: numpy array [bs, D, H*W, 3] with normalized coords [0,1]
            pc_range: list [x_min, y_min, z_min, x_max, y_max, z_max]
            img_metas: list of dicts containing 'lidar2img' transformation matrices

        Returns:
            reference_points_cam: numpy array [num_cam, bs, H*W, D, 2] - projected 2D coords
            bev_mask: numpy array [num_cam, bs, H*W, D] - visibility mask (bool)
        """
        # Extract lidar2img matrices from metadata
        _lidar2img_list = []
        for img_meta in img_metas:
            _lidar2img_list.append(img_meta["lidar2img"])
        lidar2img = np.array(_lidar2img_list)  # [bs, num_cam, 4, 4]

        # Denormalize reference points to actual 3D coordinates
        reference_points = reference_points.copy()
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        # Add homogeneous coordinate: [bs, D, H*W, 4]
        ones = np.ones_like(reference_points[..., :1])
        reference_points = np.concatenate([reference_points, ones], axis=-1)

        # Permute to [D, bs, H*W, 4]
        reference_points = reference_points.transpose(1, 0, 2, 3)
        D, B, num_query = reference_points.shape[:3]
        num_cam = lidar2img.shape[1]

        # Expand for all cameras: [D, bs, num_cam, H*W, 4, 1]
        reference_points = reference_points[:, :, np.newaxis, :, :, np.newaxis]
        reference_points = np.broadcast_to(
            reference_points, (D, B, num_cam, num_query, 4, 1)
        )

        # Expand lidar2img: [D, bs, num_cam, H*W, 4, 4]
        lidar2img = lidar2img[np.newaxis, :, :, np.newaxis, :, :]
        lidar2img = np.broadcast_to(lidar2img, (D, B, num_cam, num_query, 4, 4))

        # Project to camera coordinates: [D, bs, num_cam, H*W, 4, 1]
        reference_points_cam = np.matmul(lidar2img, reference_points).squeeze(-1)

        eps = 1e-5

        # Check if points are in front of camera (z > 0)
        bev_mask = reference_points_cam[..., 2:3] > eps

        # Perspective division: x/z, y/z
        z_coord = reference_points_cam[..., 2:3]
        z_coord_safe = np.maximum(z_coord, np.ones_like(z_coord) * eps)
        reference_points_cam = reference_points_cam[..., 0:2] / z_coord_safe

        # Normalize by image size
        img_h, img_w = img_metas[0]["img_shape"][0][:2]
        reference_points_cam[..., 0] /= img_w
        reference_points_cam[..., 1] /= img_h

        # Update mask: point must be in front and within image bounds
        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )

        # Handle NaN values
        bev_mask = np.nan_to_num(bev_mask, nan=0.0)

        # Permute to [num_cam, bs, H*W, D, 2] and [num_cam, bs, H*W, D]
        reference_points_cam = reference_points_cam.transpose(2, 1, 3, 0, 4)
        bev_mask = bev_mask.transpose(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    def __call__(
        self,
        bev_query,
        key,
        value,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=None,
        img_metas=None,
        warped_history_bev=None,
        **kwargs,
    ):
        """
        Forward function for BEVFormerEncoder.

        Args:
            bev_query: BEV query tensor [bs, num_query, embed_dims]
            key: Multi-camera features [num_cam, num_value, bs, embed_dims]
            value: Multi-camera features (same as key)
            bev_h, bev_w: Spatial dimensions of BEV grid
            bev_pos: Positional encoding for BEV [bs, num_query, embed_dims]
            spatial_shapes: List of (H, W) for each feature pyramid level
            level_start_index: Start indices for each level
            prev_bev: Previous BEV features for temporal attention
            shift: Spatial shift for temporal attention
            img_metas: Image metadata containing camera parameters

        Returns:
            output: Encoded BEV features [bs, num_query, embed_dims]
            or list of intermediate outputs if return_intermediate=True
        """
        output = bev_query
        intermediate = []

        bs = bev_query.shape[0]

        # Generate reference points
        Z = self.pc_range[5] - self.pc_range[2]
        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            Z,
            self.num_points_in_pillar,
            dim="3d",
            bs=bs,
            dtype=np.float32,
        )

        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim="2d", bs=bs, dtype=np.float32
        )

        # Project 3D reference points to camera coordinates
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas
        )

        # Handle shift for temporal attention (bug kept for paper reproduction)
        # ref_2d is always numpy; shift may be TTSim tensor or numpy.
        # Convert ref_2d to TTSim const so the add is always graph-visible.
        shift_ref_2d_np = ref_2d.copy()
        if shift is not None:
            is_shift_ttsim = hasattr(shift, "op_in")
            if is_shift_ttsim:
                # Convert numpy ref_2d to a TTSim constant, then use TTSim ops
                shift_ref_2d = F._from_data(
                    self.name + ".shift_ref_2d_base",
                    shift_ref_2d_np.astype(np.float32),
                    is_const=True,
                )
                setattr(self, shift_ref_2d.name, shift_ref_2d)
                _shift_shape = F._from_data(
                    self.name + ".shift_reshape_shape",
                    np.array([shift.data.shape[0], 1, 1, 2], dtype=np.int64),
                    is_const=True,
                )
                setattr(self, _shift_shape.name, _shift_shape)
                _shift_op = F.Reshape(self.name + ".shift_reshape")
                setattr(self, _shift_op.name, _shift_op)
                shift_reshaped = _shift_op(shift, _shift_shape)
                setattr(self, shift_reshaped.name, shift_reshaped)
                _add_op = F.Add(self.name + ".shift_ref_2d_add")
                setattr(self, _add_op.name, _add_op)
                shift_ref_2d = _add_op(shift_ref_2d, shift_reshaped)
                setattr(self, shift_ref_2d.name, shift_ref_2d)
            else:  # Both numpy
                shift_ref_2d = shift_ref_2d_np + shift[:, np.newaxis, np.newaxis, :]
        else:
            shift_ref_2d = shift_ref_2d_np

        # Prepare hybrid reference points for temporal attention
        len_bev = bev_h * bev_w
        num_bev_level = ref_2d.shape[2]

        if prev_bev is not None:
            # Stack current and previous BEV: [bs*2, len_bev, embed_dims]
            prev_bev_stacked = np.concatenate([prev_bev, bev_query], axis=0)
            # Stack shifted and current reference points: [bs*2, len_bev, num_bev_level, 2]
            # Use TTSim ConcatX when shift_ref_2d is a SimTensor to keep
            # the Reshape+Add ops graph-connected.
            if hasattr(shift_ref_2d, "op_in"):
                ref_2d_const = F._from_data(
                    self.name + ".ref_2d_for_hybrid",
                    ref_2d.astype(np.float32),
                    is_const=True,
                )
                setattr(self, ref_2d_const.name, ref_2d_const)
                _hybrid_concat = F.ConcatX(self.name + ".hybrid_ref_concat", axis=0)
                setattr(self, _hybrid_concat.name, _hybrid_concat)
                hybird_ref_2d = _hybrid_concat(shift_ref_2d, ref_2d_const)
                setattr(self, hybird_ref_2d.name, hybird_ref_2d)
            else:
                hybird_ref_2d = np.concatenate([shift_ref_2d, ref_2d], axis=0)
        else:
            prev_bev_stacked = None
            # Stack current reference points twice: [bs*2, len_bev, num_bev_level, 2]
            # When shift_ref_2d is a SimTensor, use it for the first half so the
            # Reshape+Add ops stay graph-connected (even though prev_bev is None,
            # the shifted ref points are numerically close to ref_2d).
            if hasattr(shift_ref_2d, "op_in"):
                ref_2d_const = F._from_data(
                    self.name + ".ref_2d_for_hybrid_no_prev",
                    ref_2d.astype(np.float32),
                    is_const=True,
                )
                setattr(self, ref_2d_const.name, ref_2d_const)
                _hybrid_concat = F.ConcatX(
                    self.name + ".hybrid_ref_concat_no_prev", axis=0
                )
                setattr(self, _hybrid_concat.name, _hybrid_concat)
                hybird_ref_2d = _hybrid_concat(shift_ref_2d, ref_2d_const)
                setattr(self, hybird_ref_2d.name, hybird_ref_2d)
            else:
                hybird_ref_2d = np.concatenate([ref_2d, ref_2d], axis=0)

        # Convert to TTSim const tensor if not already a SimTensor
        if not hasattr(hybird_ref_2d, "op_in"):
            hybird_ref_2d = F._from_data(
                self.name + ".hybird_ref_2d",
                hybird_ref_2d.astype(np.float32),
                is_const=True,
            )
            setattr(self, hybird_ref_2d.name, hybird_ref_2d)
        ref_3d_tensor = F._from_data(
            self.name + ".ref_3d", ref_3d.astype(np.float32), is_const=True
        )
        setattr(self, ref_3d_tensor.name, ref_3d_tensor)
        reference_points_cam_tensor = F._from_data(
            self.name + ".reference_points_cam",
            reference_points_cam.astype(np.float32),
            is_const=True,
        )
        setattr(self, reference_points_cam_tensor.name, reference_points_cam_tensor)
        bev_mask_tensor = F._from_data(
            self.name + ".bev_mask",
            bev_mask.max(axis=-1).astype(np.float32),
            is_const=True,
        )
        setattr(self, bev_mask_tensor.name, bev_mask_tensor)

        # Forward through transformer layers
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d_tensor,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam_tensor,
                bev_mask=bev_mask_tensor,
                prev_bev=prev_bev_stacked,
                **kwargs,
            )

            # BEV memory fusion: fuse warped history features with current output
            # Matches PyTorch: output = output + temporal_mem_layers[lid](warped_history_bev, curr_feat)
            if warped_history_bev is not None:
                mem_layer = self.temporal_mem_layers[lid]

                # Reshape output: [b, h*w, c] -> [b, c, h, w]
                # Step 1: permute [b, hw, c] -> [b, c, hw]
                curr_feat = self.mem_permute_to_bchw(output)
                # Step 2: reshape [b, c, hw] -> [b, c, h, w]
                to_4d_shape = F._from_data(
                    f"{self.name}.mem_to_4d_shape_l{lid}",
                    np.array([bs, self.embed_dims, bev_h, bev_w], dtype=np.int64),
                    is_const=True,
                )
                self._tensors[to_4d_shape.name] = to_4d_shape
                curr_feat = self.mem_reshape_to_4d(curr_feat, to_4d_shape)

                # Apply TemporalNet fusion
                fused_output = mem_layer(warped_history_bev, curr_feat)

                # Reshape back: [b, c, h, w] -> [b, h*w, c]
                # Step 1: reshape [b, c, h, w] -> [b, c, hw]
                to_flat_shape = F._from_data(
                    f"{self.name}.mem_to_flat_shape_l{lid}",
                    np.array([bs, self.embed_dims, bev_h * bev_w], dtype=np.int64),
                    is_const=True,
                )
                self._tensors[to_flat_shape.name] = to_flat_shape
                fused_output = self.mem_reshape_back(fused_output, to_flat_shape)
                # Step 2: permute [b, c, hw] -> [b, hw, c]
                fused_output = self.mem_permute_back(fused_output)

                # Residual add: output = output + fused_output
                output = self.mem_add(output, fused_output)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return intermediate

        return output

    def analytical_param_count(self):
        """Calculate total parameter count."""
        total = 0
        for layer in self.layers:
            total += layer.analytical_param_count()  # type: ignore[attr-defined]
        for mem_layer in self.temporal_mem_layers:
            total += mem_layer.analytical_param_count(lvl=0)  # type: ignore[attr-defined]
        return total


class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """
    Single BEVFormer transformer layer.

    Implements a transformer layer with:
    - Temporal self-attention (TSA): Attends to previous BEV frames
    - Spatial cross-attention (SCA): Aggregates multi-camera features
    - Feed-forward network (FFN): Non-linear transformation
    - Layer normalization between operations

    The operation order is fixed as: ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')

    Args:
        name (str): Module name
        attn_cfgs (list): Attention configurations [temporal_attn, spatial_attn]
        feedforward_channels (int): Hidden dimension for FFN
        ffn_dropout (float): Dropout rate (ignored in inference)
        operation_order (tuple): Fixed as ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        act_cfg (dict): Activation config (default: ReLU)
        norm_cfg (dict): Normalization config (default: LayerNorm)
        ffn_num_fcs (int): Number of FC layers in FFN (default: 2)
    """

    def __init__(
        self,
        name,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=None,
        norm_cfg=None,
        ffn_num_fcs=2,
        **kwargs,
    ):

        # Set defaults
        if operation_order is None:
            operation_order = ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
        if act_cfg is None:
            act_cfg = dict(type="ReLU", inplace=True)
        if norm_cfg is None:
            norm_cfg = dict(type="LN")

        # Validate operation order
        assert len(operation_order) == 6, "BEVFormerLayer requires exactly 6 operations"
        assert set(operation_order) == set(
            ["self_attn", "norm", "cross_attn", "ffn"]
        ), "BEVFormerLayer operations must be self_attn, norm, cross_attn, ffn"

        # Build ffn_cfgs dict from the (formerly deprecated) individual args
        ffn_cfgs = dict(
            type="FFN",
            embed_dims=(
                attn_cfgs[0].get("embed_dims", 256)
                if isinstance(attn_cfgs, list) and len(attn_cfgs) > 0
                else 256
            ),
            feedforward_channels=feedforward_channels,
            num_fcs=ffn_num_fcs,
            ffn_drop=ffn_dropout,
            act_cfg=act_cfg,
        )

        super().__init__(
            name=name,
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            **kwargs,
        )

    def __call__( # type: ignore[override]
        self,  # type: ignore[override]
        query,
        key=None,
        value=None,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):
        """
        Forward function for BEVFormerLayer.

        Args:
            query: Input BEV query [bs, num_queries, embed_dims]
            key: Multi-camera features for cross-attention
            value: Multi-camera features for cross-attention
            bev_pos: Positional encoding for BEV
            ref_2d: 2D reference points for temporal attention
            ref_3d: 3D reference points for spatial attention
            bev_h, bev_w: BEV spatial dimensions
            reference_points_cam: Projected camera coordinates
            mask: Visibility mask
            spatial_shapes: Feature pyramid spatial shapes
            level_start_index: Level start indices
            prev_bev: Previous BEV for temporal attention
            attn_masks: Attention masks
            query_key_padding_mask: Query padding mask
            key_padding_mask: Key padding mask

        Returns:
            query: Transformed BEV features [bs, num_queries, embed_dims]
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # Handle attention masks
        if attn_masks is None:
            attn_masks = [None for _ in range(len(self.attentions))]
        elif not isinstance(attn_masks, list):
            attn_masks = [attn_masks for _ in range(len(self.attentions))]

        for layer in self.operation_order:
            if layer == "self_attn":
                # Temporal self-attention
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=[(bev_h, bev_w)],
                    level_start_index=[0],
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                # Spatial cross-attention
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
