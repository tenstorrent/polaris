#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.utils.common import parse_yaml
from ttsim.ops import SimTensor

from workloads.bevdepth.layers.backbones.ResNet import ResNet, BasicBlock
from workloads.bevdepth.layers.backbones.SecondFPN import SecondFPN

import numpy as np

def build_backbone(name, cfg):
    '''
    Explanation of fields
        frozen_stages
            -1: No stages are frozen
             0: Only Stem is frozen,
             1: Stem + Stage-1 is frozen
             2: Stem + Stage-1 + Stage-2 frozen
        out_indices = utputs used in next stage (neck)
             0: o/p from Stage-1: (bs,  256, H/4,  W/4)
             1: o/p from Stage-2: (bs,  512, H/8,  W/8)
             2: o/p from Stage-3: (bs, 1024, H/16, W/16)
             3: o/p from Stage-4: (bs, 2048, H/32, W/32)
             [0,1,2,3] means o/p's from all stages in the next stage
        norm_eval
             False: use BN in training mode, compute mean/var
             True: use BN in eval mode
    '''
    bb_type          = cfg['type']  #'ResNet',
    bb_norm_eval     = cfg['norm_eval']#: False,
    bb_frozen_stages = cfg.get('frozen_stages', 0)  #: 0,
    bb_in_channels   = cfg.get('in_channels', 64)  #: 0,
    bb_num_channels  = cfg.get('num_channels', 3)  #: 0,
    bb_depth         = cfg['depth']
    bb_out_indices   = cfg['out_indices']

    assert bb_type          == 'ResNet', f"build_backbone ERR-1"
    assert bb_frozen_stages == 0,        f"build_backbone ERR-2"
    assert bb_norm_eval     == False,    f"build_backbone ERR-3"
    count = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in bb_out_indices: count[i] += 1
    assert all(c <= 1 for _,c in count.items()), f"build_backbone ERR-4, {count}"

    depth2layers = { 18: [2,2,2,2], 50: [3,4,6,3], }

    resnet_cfg = {
            'layers'      : depth2layers[bb_depth],
            'out_indices' : bb_out_indices,
            'num_classes' : 2, #output classes are irrelevant for BEVDepth
            'num_channels': bb_num_channels,
            'in_channels': bb_in_channels,
            }
    return ResNet(name, resnet_cfg)

def build_neck(name, cfg):
    nt = cfg['type']
    ic = cfg['in_channels']
    us = cfg['upsample_strides']
    oc = cfg['out_channels']

    assert nt == 'SECONDFPN',            f"build_neck ERR-1"
    assert ic == [256, 512, 1024, 2048], f"build_neck ERR-2"
    #assert us == [0.25, 0.5, 1, 2],      f"build_neck ERR-3"
    #assert oc == [128, 128, 128, 128],   f"build_neck ERR-4"

    return SecondFPN(name, in_channels=ic, upsample_strides=us, out_channels=oc)

class _ASPPModule(SimNN.Module):

    def __init__(self, name, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.name        = name
        self.bn          = F.BatchNorm2d(name + '.bn', planes)
        self.relu        = F.Relu(name + '.relu')
        self.atrous_conv = F.Conv2d(name + '.atrous_conv',
                                    inplanes, planes,
                                    kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        super().link_op2module()
        return


    def __call__(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

class ASPP(SimNN.Module):

    def __init__(self, name, inplanes, mid_channels=256):
        super(ASPP, self).__init__()
        self.name  = name
        dilations  = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(name + '.aspp1', inplanes, mid_channels, 1, padding=0,            dilation=dilations[0])
        self.aspp2 = _ASPPModule(name + '.aspp2', inplanes, mid_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(name + '.aspp3', inplanes, mid_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(name + '.aspp4', inplanes, mid_channels, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = F.SimOpHandleList([
            F.AveragePool2d(name + '.avgpool0', output_size=(1, 1), adaptive=True),
            F.Conv2d(name + '.conv0', inplanes, mid_channels, 1, stride=1, bias=False),
            F.BatchNorm2d(name + '.bn0', mid_channels),
            F.Relu(name + '.relu0', ),
            ])
        self.conv1    = F.Conv2d(name + '.conv1', int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1      = F.BatchNorm2d(name + '.bn1', mid_channels)
        self.relu1    = F.Relu(name + '.relu1')
        self.dropout1 = F.Dropout(name + '.drop1', 0.5)

        super().link_op2module()
        return

    def __call__(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = T.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x  = T.cat((x1, x2, x3, x4, x5), dim=1)
        x  = self.conv1(x)
        x  = self.bn1(x)
        x  = self.relu1(x)
        x  = self.dropout1(x)
        return x

class Mlp(SimNN.Module):

    def __init__(self, name, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.name       = name
        self.fc1        = SimNN.Linear(name + 'fc1', in_features, hidden_features)
        self.relu1      = F.Relu(name + '.relu1')
        self.drop1      = F.Dropout(name + 'drop1', drop)
        self.fc2        = SimNN.Linear(name + '.fc2', hidden_features, out_features)
        self.drop2      = F.Dropout(name + '.drop2', drop)
        super().link_op2module()
        return

    def __call__(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(SimNN.Module):

    def __init__(self, name, channels):
        super().__init__()
        self.name            = name
        self.conv_reduce     = F.Conv2d(name + 'conv_reduce', channels, channels, 1, bias=True)
        self.act1            = F.Relu(name + '.relu')
        self.conv_expand     = F.Conv2d(name + 'conv_expand', channels, channels, 1, bias=True)
        self.gate            = F.Sigmoid(name + '.sigmoid')
        super().link_op2module()
        return

    def __call__(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(SimNN.Module):

    def __init__(self, name, in_channels, mid_channels, context_channels, depth_channels):
        super().__init__()
        self.name        = name
        self.reduce_conv = F.SimOpHandleList([
            F.Conv2d(name + '.reduce_conv.conv', in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            F.BatchNorm2d(name + '.reduce_conv.bn', mid_channels),
            F.Relu(name + '.reduce_conv.relu')
            ])
        self.context_conv = F.Conv2d(name + '.context_conv', mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn           = F.BatchNorm2d(name + '.bn1d', 27) #self.bn = nn.BatchNorm1d(27)
        self.depth_mlp    = Mlp(name + '.depth_mlp', 27, mid_channels, mid_channels)
        self.depth_se     = SELayer(name + '.depth_se', mid_channels)  # NOTE: add camera-aware
        self.context_mlp  = Mlp(name + '.context_mlp', 27, mid_channels, mid_channels)
        self.context_se   = SELayer(name + '.context_se', mid_channels)  # NOTE: add camera-aware
        self.bb0          = BasicBlock(name + '.bb0', mid_channels, mid_channels)
        self.bb1          = BasicBlock(name + '.bb1', mid_channels, mid_channels)
        self.bb2          = BasicBlock(name + '.bb2', mid_channels, mid_channels)
        self.aspp         = ASPP(name + '.aspp', mid_channels, mid_channels)
        #TODO: add build_conv_layer
        #self.dcn          = build_conv_layer(cfg=dict(type='DCN', in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1, groups=4, im2col_step=128,))
        self.conv         = F.Conv2d(name + '.conv', mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
        super().link_op2module()
        return

    def __call__(self, x, intrin_mats, ida_mats, sensor2ego_mats, bda_mat):
        intrins    = intrin_mats[:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams   = intrins.shape[2]
        ida        = ida_mats[:, 0:1, ...]
        sensor2ego = sensor2ego_mats[:, 0:1, ..., :3, :]
        bda        = bda_mat.view(batch_size, 1, 1, 4, 4).repeat(1, 1, num_cams, 1, 1)
        mlp_input  = T.cat([
            T.stack([
                intrins[:, 0:1, ..., 0, 0],
                intrins[:, 0:1, ..., 1, 1],
                intrins[:, 0:1, ..., 0, 2],
                intrins[:, 0:1, ..., 1, 2],
                ida[:, 0:1, ..., 0, 0],
                ida[:, 0:1, ..., 0, 1],
                ida[:, 0:1, ..., 0, 3],
                ida[:, 0:1, ..., 1, 0],
                ida[:, 0:1, ..., 1, 1],
                ida[:, 0:1, ..., 1, 3],
                bda[:, 0:1, ..., 0, 0],
                bda[:, 0:1, ..., 0, 1],
                bda[:, 0:1, ..., 1, 0],
                bda[:, 0:1, ..., 1, 1],
                bda[:, 0:1, ..., 2, 2],
                ], dim=-1,),
            sensor2ego.view(batch_size, 1, num_cams, -1)], -1,)

        mlp_input   = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x           = self.reduce_conv(x)
        context_se  = self.context_mlp(mlp_input)[..., None, None]
        context     = self.context_se(x, context_se)
        context     = self.context_conv(context)
        depth_se    = self.depth_mlp(mlp_input)[..., None, None]
        depth       = self.depth_se(x, depth_se)
        depth       = self.bb0(depth) #depth = self.depth_conv(depth) begin
        depth       = self.bb1(depth)
        depth       = self.bb2(depth)
        depth       = self.aspp(depth)
        #depth       = self.dcn(depth)
        depth       = self.conv(depth)
        final_result= T.cat([depth, context], dim=1)
        return final_result

class DepthAggregation(SimNN.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, name, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()
        self.name = name

        self.reduce_conv = F.SimOpHandleList([
            F.Conv2d(name + '.reduce_conv', in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            F.BatchNorm2d(name + '.reduce_conv_bn', mid_channels),
            F.Relu(name + '.reduce_conv_relu')
            ])

        self.conv = F.SimOpHandleList([
            F.Conv2d(name + '.conv0', mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            F.BatchNorm2d(name + '.bn0', mid_channels),
            F.Relu(name + '.relu0'),
            F.Conv2d(name + '.conv1', mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            F.BatchNorm2d(name + '.bn1', mid_channels),
            F.Relu(name + '.relu1'),
            ])

        self.out_conv = F.SimOpHandleList([
            F.Conv2d(name + '.out_conv', mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            ])

        super().link_op2module()
        return

    def __call__(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class BaseLSSFPN(SimNN.Module):

    def __init__(self, name,
                 x_bound, y_bound, z_bound, d_bound,
                 final_dim,
                 downsample_factor,
                 output_channels,
                 img_backbone_conf, img_neck_conf, depth_net_conf,
                 use_da=False, training=False):

        super().__init__()
        self.name              = name
        self.downsample_factor = downsample_factor
        self.d_bound           = d_bound
        self.final_dim         = final_dim
        self.output_channels   = output_channels
        self.training          = training
        self.use_da            = use_da
        self.frustum           = self.create_frustum()
        self.depth_channels    = self.frustum.shape[0]

        #submodules
        self.img_backbone      = build_backbone(name + '.img_backbone', img_backbone_conf)
        self.img_neck          = build_neck(name + '.img_neck', img_neck_conf)
        self.depth_net         = self._configure_depth_net(name + '.depth_net', depth_net_conf)
        self.voxel_pooling_inf = F.VoxelPooling(name + '.voxel_pooling_inference')
        self.depth_aggregation = self._configure_depth_aggregation_net(name + '.depth_aggregation_net') if self.use_da else None
        #tensors
        self.voxel_size        = F._from_data(name + '.voxel_size',  data=np.array([row[2]
                                                                            for row in [x_bound, y_bound, z_bound]]),
                                              is_const=True)
        self.voxel_coord       = F._from_data(name + '.voxel_coord', data=np.array([row[0] + row[2] / 2.0
                                                                            for row in [x_bound, y_bound, z_bound]]),
                                              is_const=True)
        self.voxel_num         = F._from_data(name + '.voxel_num',   data=np.array([(row[1] - row[0]) / row[2]
                                                                            for row in [x_bound, y_bound, z_bound]]),
                                              is_const=True)
        self.voxel_size.set_module(self)
        self.voxel_coord.set_module(self)
        self.voxel_num.set_module(self)
        super().link_op2module()
        return

    def _configure_depth_net(self, name, depth_net_conf):
        return DepthNet(name,
                        depth_net_conf['in_channels'],
                        depth_net_conf['mid_channels'],
                        self.output_channels,
                        self.depth_channels,)

    def _configure_depth_aggregation_net(self, name):
        return DepthAggregation(name, self.output_channels, self.output_channels, self.output_channels)

    def _forward_voxel_net(self, img_feat_with_depth):
        if self.use_da:
            # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
            img_feat_with_depth = img_feat_with_depth.permute(0, 3, 1, 4, 2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d       = img_feat_with_depth.shape
            img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
            img_feat_with_depth = (self.depth_aggregation(img_feat_with_depth).view(n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous())
        return img_feat_with_depth

    def create_frustum(self):
        ogfH, ogfW = self.final_dim
        fH, fW     = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords   = np.arange(*self.d_bound, dtype=np.float32).reshape(-1, 1, 1)
        d_coords   = np.broadcast_to(d_coords, d_coords.shape[:1] + (fH, fW))
        D, _, _    = d_coords.shape
        x_coords   = np.linspace(0, ogfW - 1, fW, dtype=np.float32).reshape(1, 1, fW)
        x_coords   = np.broadcast_to(x_coords, (D, fH, fW))
        y_coords   = np.linspace(0, ogfH - 1, fH, dtype=np.float32).reshape(1, fH, 1)
        y_coords   = np.broadcast_to(y_coords, (D, fH, fW))
        paddings   = np.ones_like(d_coords)
        frustum    = np.stack((x_coords, y_coords, d_coords, paddings), -1) # D x H x W x 3
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        #points  = self.frustum
        #ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        #points  = ida_mat.inverse().matmul(points.unsqueeze(-1))
        points = F._from_data('frustum', self.frustum)
        points.set_module(self)
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        ida_mat_inv = F._from_shape('ida_mat_invers', ida_mat.shape)
        ida_mat_inv.set_module(self)
        points = T.matmul(ida_mat_inv, points.unsqueeze(-1))

        # cam_to_ego
        points = T.cat(
                (
                    points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                    points[:, :, :, :, :, 2:]),
                5)

        intrin_mat_inv = F._from_shape('intrin_mat_inverse', intrin_mat.shape)
        intrin_mat_inv.set_module(self)
        combine = T.matmul(sensor2ego_mat, intrin_mat_inv)
        combine_view = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = T.matmul(combine_view, points)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(batch_size, num_cams, 1, 1, 1, 4, 4)
            points = T.matmul(bda_mat, points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_cam_feats(self, imgs):
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape
        imgs      = imgs.flatten().view(batch_size * num_sweeps * num_cams, num_channels, imH, imW)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats.set_module(self)
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1],
                                      img_feats.shape[2],
                                      img_feats.shape[3])
        return img_feats

    def _forward_depth_net(self, feat, sensor2ego_mats, intrin_mats, ida_mats, sensor2sensor_mats, bda_mat):
        return self.depth_net(feat, intrin_mats, ida_mats, sensor2sensor_mats, bda_mat)

    def _forward_single_sweep(self, sweep_index, sweep_imgs,
                              sensor2ego_mats, intrin_mats, ida_mats, sensor2sensor_mats, bda_mat,
                              is_return_depth=False):
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(batch_size * num_cams,
                                    source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4]),
            #mats_dict,
            sensor2ego_mats, intrin_mats, ida_mats, sensor2sensor_mats, bda_mat
        )
        depth = depth_feature[:, :self.depth_channels].softmax(dim=1, dtype=depth_feature.dtype)
        geom_xyz = self.get_geometry(
            sensor2ego_mats[:, sweep_index, ...],
            intrin_mats[:, sweep_index, ...],
            ida_mats[:, sweep_index, ...],
            bda_mat
        )
        _2p0 = F._from_data('_2p0', data=np.array([2.0]))
        geom_xyz = (geom_xyz - (self.voxel_coord - self.voxel_size / _2p0)) / self.voxel_size
        if self.training or self.use_da:
            assert False, "\nREACHED voxel_pooling_train -- not implemented yet\n"
            img_feat_with_depth = depth.unsqueeze(1) * \
                    depth_feature[:, self.depth_channels:(self.depth_channels + self.output_channels)].unsqueeze(2)

            img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

            img_feat_with_depth = img_feat_with_depth.reshape(
                batch_size,
                num_cams,
                img_feat_with_depth.shape[1],
                img_feat_with_depth.shape[2],
                img_feat_with_depth.shape[3],
                img_feat_with_depth.shape[4],
            )

            img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)

            feature_map = voxel_pooling_train(geom_xyz, img_feat_with_depth.contiguous(), self.voxel_num.cuda())
        else:
            feature_map = self.voxel_pooling_inf(geom_xyz,
                                                  depth,
                                                  depth_feature[:,
                                                                self.depth_channels:(self.depth_channels + self.output_channels)
                                                                ].contiguous(),
                                                  #self.voxel_num.cuda(),
                                                  self.voxel_num,
                                                  )
        if is_return_depth:
            # final_depth has to be fp32, otherwise the depth
            # loss will colapse during the traing process.
            return feature_map.contiguous(), depth_feature[:, :self.depth_channels].softmax(dim=1)
        return feature_map.contiguous()

    def __call__(self,
                 sweep_imgs,
                 sensor2ego_mats,
                 intrin_mats,
                 ida_mats,
                 sensor2sensor_mats,
                 bda_mat=None,
                 timestamps=None,
                 is_return_depth=False):

        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(0, sweep_imgs[:, 0:1, ...],
                                                   sensor2ego_mats,
                                                   intrin_mats,
                                                   ida_mats,
                                                   sensor2sensor_mats,
                                                   bda_mat,
                                                   is_return_depth=is_return_depth)

        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]

        #TTSIM Magic: Instead of running the model "num_sweeps" times,
        # we'll just clone the output for the required number of times
        # and update the repeat counts for all operators in the graph...

        # clone tensors...
        for sweep_index in range(1, num_sweeps):
            ret_feature_list.append(key_frame_feature.clone(sweep_index))

        # Since all sim_ops in the graph have been set, 
        # we can update the repeat counts
        repeated_ops: dict[str, Any] = {}
        self.get_ops(repeated_ops)
        for op_num, (op_name,op_obj) in enumerate(repeated_ops.items()):
            op_obj.repeat_count = num_sweeps

        if is_return_depth:
            return T.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return T.cat(ret_feature_list, 1)

if __name__ == '__main__':
    H         = 900
    W         = 1600
    final_dim = (256, 704)
    img_conf  = dict(img_mean=[123.675, 116.28, 103.53], img_std=[58.395, 57.12, 57.375], to_rgb=True)

    bb_cfg = {
        'x_bound'  : [-51.2, 51.2, 0.8],
        'y_bound'  : [-51.2, 51.2, 0.8],
        'z_bound'  : [-5, 3, 8],
        'd_bound'  : [2.0, 58.0, 0.5],
        'final_dim': final_dim,
        'output_channels'  : 80,
        'downsample_factor': 16,
        'img_backbone_conf': dict(
            type='ResNet',
            depth=50,
            frozen_stages=0,
            out_indices=[0, 1, 2, 3],
            norm_eval=False,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
            ),
        'img_neck_conf': dict(
            type='SECONDFPN',
            in_channels=[256, 512, 1024, 2048],
            upsample_strides=[0.25, 0.5, 1, 2],
            out_channels=[128, 128, 128, 128],
            ),
        #'depth_net_conf': dict(in_channels=512, mid_channels=512)
        'depth_net_conf': dict(in_channels=128, mid_channels=512)
        }

    EB  = 5  # Batch
    ENS = 2  # Num Sweeps
    ENC = 6  # Num Cameras
    EIC = 3  # Img Channels 3: RGB
    EH  = 64 # Img Height
    EW  = 64 # Img Width
    sweep_imgs         = F._from_shape('sweep_imgs', [EB, ENS, ENC, EIC, EH, EW])
    sensor2ego_mats    = F._from_shape('sensor2ego_mats', [EB, ENS, ENC, 4, 4])
    intrin_mats        = F._from_shape('intrin_mats', [EB, ENS, ENC, 4, 4])
    ida_mats           = F._from_shape('ida_mats', [EB, ENS, ENC, 4, 4])
    sensor2sensor_mats = F._from_shape('sensor2sensor_mats', [EB, ENS, ENC, 4, 4])
    bda_mat            = F._from_shape('bda_mat', [EB, 4, 4])


    ############ DepthNet ########################
    #dn_x= F._from_shape('dn_x', [30, 128, 16, 16])
    #dncfg = dict(in_channels= 128, mid_channels= 512, context_channels=80, depth_channels=112)
    #dn    = DepthNet('toy_dn', **dncfg)
    #for t in [sweep_imgs, sensor2ego_mats, intrin_mats, ida_mats, sensor2sensor_mats, bda_mat]:
    #    t.set_module(dn)
    #y = dn(dn_x, intrin_mats, ida_mats, sensor2sensor_mats, bda_mat)
    #print(y)
    #gg = dn._get_forward_graph({t.name: t for t in [dn_x, intrin_mats, ida_mats, sensor2sensor_mats, bda_mat]})
    #gg.graph2onnx(f'bevdepth_dn.onnx', do_model_check=False)
    #print("DONE")
    #exit(0)

    ############ BaseLSSFPN ########################
    backbone = BaseLSSFPN('base_lss_fpn', **bb_cfg)

    for t in [sweep_imgs, sensor2ego_mats, intrin_mats, ida_mats, sensor2sensor_mats, bda_mat]:
        t.set_module(backbone)

    y = backbone(sweep_imgs, sensor2ego_mats, intrin_mats, ida_mats, sensor2sensor_mats, bda_mat)
    print(y)
    print("DNONE")
    gg = backbone._get_forward_graph({t.name: t
                                      for t in [sweep_imgs,
                                                sensor2ego_mats, intrin_mats, ida_mats, sensor2sensor_mats, bda_mat
                                                ]})
    gg.graph2onnx(f'bevdepth_backbone.onnx', do_model_check=False)
