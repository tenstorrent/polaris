#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
import numpy as np
from loguru import logger


class Mlp(SimNN.Module):
    """Simplified TT-Sim MLP"""
    def __init__(self, name, in_features, hidden_features=None, out_features=None, act_layer=F.Gelu, drop=0.):
        super().__init__()
        
        self.name = name
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = SimNN.Linear(name + '.fc1', in_features, hidden_features)
        self.act = act_layer(name + '.gelu')
        self.drop1 = F.Identity(name + '.drop1') if drop == 0.0 else F.Dropout(name + '.drop1', prob=drop, train_mode=False)
        self.fc2 = SimNN.Linear(name + '.fc2', hidden_features, out_features)
        self.drop2 = F.Identity(name + '.drop2') if drop == 0.0 else F.Dropout(name + '.drop2', prob=drop, train_mode=False)

        super().link_op2module()

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(x, window_size, module=None):
    """Simplified window partition"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute([0, 1, 3, 2, 4, 5]).contiguous() # type: ignore[attr-defined]
    windows.set_module(module)
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, module=None):
    """Simplified window reverse"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute([0, 1, 3, 2, 4, 5]).contiguous() # type: ignore[attr-defined]
    x.set_module(module)
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(SimNN.Module):
    def __init__(self, name, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.name = name  
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        ch, cw = np.meshgrid(coords_h, coords_w)
        coords = np.stack([ch, cw])  # 2, Wh, Ww
        coords_flatten = np.reshape(coords, (2, -1))  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = np.transpose(relative_coords, axes=(1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.relative_position_index = F._from_data(name + '.relative_position_index',
                                                  relative_position_index.astype(np.int64), is_param=False)
        self.relative_position_bias = F._from_shape(name + '.relative_position_bias',
                                                  [window_size[0] * window_size[1], window_size[0] * window_size[1], num_heads])
        self.qkv = SimNN.Linear(self.name + '.qkv', dim, dim * 3, bias=qkv_bias)
        self.qkv_split = F.SplitOpHandle(self.name +'.qkv_split', count=3, axis=0)
        self.attn_drop = F.Dropout(self.name + '.attn_drop', attn_drop, False)
        self.proj = SimNN.Linear(self.name + '.proj', dim, dim)
        self.proj_drop = F.Dropout(self.name + '.proj_drop', proj_drop, False)
        self.softmax = F.Softmax(self.name + '.softmax')
        self.matmulop = F.MatMul(self.name + '.matmul')
        
        super().link_op2module()

    def __call__(self, x, mask=None):
        """        
        Args:
            x: input features with shape of (num_windows*B, N, C)  
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) 
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute([2, 0, 3, 1, 4])
        assert qkv.shape == [3, B_, self.num_heads, N, self.head_dim], f"qkv shape {qkv.shape} incorrect!!"
        q,k,v = self.qkv_split(qkv)
        scale_tensor = F._from_shape(self.name + '.scale', q.shape)
        q.set_module(self)
        scale_tensor.set_module(self)
        q = q * scale_tensor    # type: ignore[operator]
        k.set_module(self)
        attn = self.matmulop(q, k.transpose(-2, -1))    # type: ignore[attr-defined]
        self.relative_position_bias.set_module(self)
        perm_relative_position_bias = self.relative_position_bias.permute([2, 0, 1])  # type: ignore[attr-defined]
        perm_relative_position_bias.set_module(self)
        attn = attn + perm_relative_position_bias.unsqueeze(0)   # type: ignore[attr-defined]
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = T.matmul(attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        # Extract parameters from config dictionary
        self.name = name
        self.dim = cfg.get('dim', 96)
        self.input_resolution = cfg.get('input_resolution', [56, 56])
        self.num_heads = cfg.get('num_heads', 3)
        self.window_size = cfg.get('window_size', 7)
        self.shift_size = cfg.get('shift_size', 0)
        self.mlp_ratio = cfg.get('mlp_ratio', 4.0)
        self.drop = cfg.get('drop', 0.0)
        self.attn_drop = cfg.get('attn_drop', 0.0)
        self.drop_path = cfg.get('drop_path', 0.0)
        self.qkv_bias = cfg.get('qkv_bias', True)
        self.bs = cfg.get('bs', 1)
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = F.LayerNorm(name + '.norm1', self.dim)
        self.attn = WindowAttention(
            name + '.windowattn', self.dim, [self.window_size, self.window_size], self.num_heads,
            qkv_bias=self.qkv_bias, attn_drop=self.attn_drop, proj_drop=self.drop)

        self.drop_path1 = F.Dropout(name + '.drop_path1', prob=self.drop_path, train_mode=False) if self.drop_path > 0. else F.Identity(name + '.identity1')
        self.norm2 = F.LayerNorm(name + '.norm2', self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = Mlp(name + '.mlp', in_features=self.dim, hidden_features=mlp_hidden_dim, drop=self.drop)
        self.drop_path2 = F.Dropout(name + '.drop_path2', prob=self.drop_path, train_mode=False) if self.drop_path > 0. else F.Identity(name + '.identity2')

        # Skip complex attention mask for simplicity
        self.attn_mask = None
        super().link_op2module()

    def __call__(self, x=None):
        if x is None:
            x = self.input_tensors['x_in']
        
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        #  window partition
        x_windows = window_partition(x, self.window_size, module=self)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, module=self)

        # Skip reverse cyclic shift - convert back to flat format for residual addition
        x = shifted_x.view(B, L, C)  

        # Skip drop path - now both shortcut and x are [B, L, C]
        x = shortcut + self.drop_path1(x)

        # FFN
        x = x + self.drop_path2(self.norm2(self.mlp(x)))

        return x

    def create_input_tensors(self):
        """Create input tensors for Polaris framework compatibility"""
        H, W = self.input_resolution
        num_patches = H * W
        bs = getattr(self, 'bs', 1)  
        
        self.input_tensors = {
            'x_in': F._from_shape('x_in', [bs, num_patches, self.dim],
                               is_param=False, np_dtype=np.float32),
        }
        return

    def get_forward_graph(self):
        """Get forward graph for Polaris framework"""
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def set_batch_size(self, new_bs):
        """Set batch size for Polaris framework"""
        self.bs = new_bs

    def analytical_param_count(self):
        """Return analytical parameter count for Polaris framework"""
        return 0


class PatchMerging(SimNN.Module):
    def __init__(self, name, input_resolution, dim, norm_layer=F.LayerNorm):
        super().__init__()
        self.name = name
        self.input_resolution = input_resolution
        self.dim = dim
        self.catop = F.ConcatX(self.name + '_concat', axis=-1)
        self.reduction = SimNN.Linear(self.name + '_reduction', 4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(self.name + '_norm', 4 * dim)
        super().link_op2module()

    def __call__(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = self.catop(x0, x1, x2, x3)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(SimNN.Module):
    def __init__(self, name, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=F.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.name = name
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        cfg = []
        for i in range(depth):
            cfg.append({
                'dim': dim,
                'input_resolution': input_resolution,
                'num_heads': num_heads,
                'window_size': window_size,
                'shift_size': (0 if (i % 2 == 0) else window_size // 2),
                'mlp_ratio': 4.,
                'qkv_bias': True,
                'drop': 0.0,
                'attn_drop': 0.0,
                'drop_path': 0.0
            })
        self.blocks = SimNN.ModuleList([
            SwinTransformerBlock(self.name + f'_SwinTxBlock{i}', cfg[i])
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(self.name + '_downsample', input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(SimNN.Module):
    def __init__(self, name, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.name = name
        img_size = [img_size, img_size]
        patch_size = [patch_size, patch_size]
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = F.Conv2d(self.name + '_projConv2d', in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        if norm_layer is not None:
            self.norm = norm_layer(self.name + '_normPatchEmbed', embed_dim)
        else:
            self.norm = None

        super().link_op2module()

    def __call__(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformer(SimNN.Module):
    def __init__(self, name, cfg): 
        super().__init__()
        self.name = name
        self.img_size = cfg.get('img_size', 224)
        self.patch_size = cfg.get('patch_size', 4)
        self.in_chans = cfg.get('in_chans', 3)
        self.num_classes = cfg.get('num_classes', 1000)
        self.embed_dim = cfg.get('embed_dim', 96)
        self.depths = cfg.get('depths', [2, 2, 6, 2])
        self.num_heads = cfg.get('num_heads', [3, 6, 12, 24])
        self.window_size = cfg.get('window_size', 7)
        self.mlp_ratio = cfg.get('mlp_ratio', 4.)
        self.qkv_bias = cfg.get('qkv_bias', True)
        self.qk_scale = cfg.get('qk_scale', None)
        self.drop_rate = cfg.get('drop_rate', 0.)
        self.attn_drop_rate = cfg.get('attn_drop_rate', 0.)
        self.drop_path_rate = cfg.get('drop_path_rate', 0.1)
        self.norm_layer = cfg.get('norm_layer', F.LayerNorm)
        self.ape = cfg.get('ape', False)
        self.patch_norm = cfg.get('patch_norm', True)
        self.use_checkpoint = cfg.get('use_checkpoint', False)
        self.fused_window_process = cfg.get('fused_window_process', False)
        self.num_layers = len(self.depths)
        self.embed_dim = self.embed_dim
        self.ape = self.ape
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = self.mlp_ratio
        self.bs = cfg.get('bs', 1)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(self.name + '_patch_embed',
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = F.Dropout(self.name + '_pos_drop', self.drop_rate)

        # stochastic depth - Ignore for simplicity
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = SimNN.ModuleList([BasicLayer(self.name + '_layer' + str(i_layer),
                            dim=int(self.embed_dim * 2 ** i_layer),
                            input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                            depth=self.depths[i_layer],
                            num_heads=self.num_heads[i_layer],
                            window_size=self.window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                            drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                            drop_path=0.0,
                            norm_layer=self.norm_layer,
                            downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                            use_checkpoint=self.use_checkpoint,
                            fused_window_process=self.fused_window_process) for i_layer in range(self.num_layers)])

        self.norm = self.norm_layer(self.name + '_normSwinTransformer', self.num_features)
        self.avgpool = F.AdaptiveAvgPool1d(self.name + '_AdaptiveAvgPool1d', adaptive=True, output_size=1)
        self.head = F.Linear(self.name + '_head', self.num_features, self.num_classes) if self.num_classes > 0 else F.Identity(self.name + '_headIdentity')
        super().link_op2module()

    def forward_features(self, x):
        x = self.patch_embed(x)
        # assume self.ape is False for simplicity
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = x.flatten(1)
        return x

    def __call__(self, x=None):
        if x is None:
            x = self.input_tensors['x_in']
        logger.debug(f"Input shape: {x.shape}")
        self._tensors[x.name] = x
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def create_input_tensors(self):
        """Create input tensors for Polaris framework compatibility"""
        self.input_tensors = {
            'x_in': F._from_shape('x_in', [self.bs, self.in_chans, self.img_size, self.img_size],
                               is_param=False, np_dtype=np.float32),
        }
        return

    def get_forward_graph(self):
        """Get forward graph for Polaris framework"""
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def set_batch_size(self, new_bs):
        """Set batch size for Polaris framework"""
        self.bs = new_bs

    def analytical_param_count(self):
        """Return analytical parameter count for Polaris framework"""
        return 0


def run_swin_transformer_test():
    model = SwinTransformer(name='swin_transformer', cfg={'num_classes': 3000, 'img_size': 224, 'in_chans': 3, 'bs': 3})
    model.create_input_tensors()
    output = model()
    print(f"Output shape: {output.shape}")
    assert output.shape == [model.bs, model.num_classes], f"Output shape {output.shape} incorrect!!"
    print("SwinTransformer forward pass successful!")

    # gg = model.get_forward_graph()
    # onnx_filename = 'swin_transformer.onnx'
    # gg.graph2onnx(onnx_filename, do_model_check=False)

if __name__ == '__main__':
    run_swin_transformer_test()
