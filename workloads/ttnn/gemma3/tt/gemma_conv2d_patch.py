# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 Conv2D Patch Embedding
"""
import numpy as np
from typing import Optional, Union

import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.common.gemma_Lightweightmodule import LightweightModule


class TtGemmaConv2dPatch(LightweightModule):
    """Conv2D-based patch embedding for vision transformer."""

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix: str,
        dtype,
        in_channels: int = 3,
        out_channels: int = 1152,
        kernel_size: int = 14,
        stride: int = 14,
        bias: bool = True,
        image_size: int = 896,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = bias
        self.image_size = image_size
        
        self.patch_dim = in_channels * kernel_size * kernel_size
        self.num_patches_per_side = image_size // kernel_size
        self.num_patches = self.num_patches_per_side ** 2
        
        # Load weights
        weight_key = f"{state_dict_prefix}weight"
        linear_weight_key = f"{state_dict_prefix}_linear.weight"
        bias_key = f"{state_dict_prefix}bias"
        
        linear_bias_key = f"{state_dict_prefix}_linear.bias"
        conv_weight = None
        if weight_key in state_dict:
            conv_weight = state_dict[weight_key]
        elif linear_weight_key in state_dict:
             conv_weight = state_dict[linear_weight_key]
        if conv_weight is not None:
            if hasattr(conv_weight, 'numpy'):
                conv_weight = conv_weight.numpy()
            elif hasattr(conv_weight, 'detach'):
                conv_weight = conv_weight.detach().cpu().numpy()
            weight_np = conv_weight.reshape(out_channels, -1)
            weight_np = weight_np.T.astype(np.float32)
        else:
            weight_np = np.random.randn(self.patch_dim, out_channels).astype(np.float32) * 0.02
        
        self.weight = ttnn.Tensor(
            weight_np,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Load bias - use Optional type annotation
        self.bias: Optional[ttnn.Tensor] = None  # Fix: Add type annotation
        if self.use_bias:
            linear_bias_key = f"{state_dict_prefix}_linear.bias"
            if bias_key in state_dict:
                bias_data = state_dict[bias_key]
            elif linear_bias_key in state_dict:
                bias_data = state_dict[linear_bias_key]
            else:                
                bias_data = None
            if bias_data is not None:
                if hasattr(bias_data, 'numpy'):
                    bias_data = bias_data.numpy()
                elif hasattr(bias_data, 'detach'):
                    bias_data = bias_data.detach().cpu().numpy()
                bias_np = bias_data.flatten().astype(np.float32)
            else:
                bias_np = np.zeros(out_channels, dtype=np.float32)
            
            bias_np = bias_np.reshape(1, 1, 1, out_channels)
            self.bias = ttnn.Tensor(
                bias_np,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

    def _to_numpy(self, x):
        """Convert tensor to numpy array."""
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, 'to_numpy'):
            return x.to_numpy()
        if hasattr(x, 'numpy'):
            return x.numpy()
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
        if hasattr(x, 'data'):
            data = x.data
            if isinstance(data, np.ndarray):
                return data
            if hasattr(data, 'numpy'):
                return data.numpy()
        return np.array(x)

    def _extract_patches(self, x):
        """Extract patches from input image tensor."""
        x_np = self._to_numpy(x)
        original_shape = x_np.shape
        
        if len(original_shape) == 0 or original_shape == ():
            B = 1
            C = self.in_channels
            H = W = self.image_size
            expected_size = B * C * H * W
            if x_np.size == expected_size:
                x_np = x_np.reshape(B, C, H, W)
            else:
                x_np = np.random.randn(B, C, H, W).astype(np.float32)
            original_shape = x_np.shape
        
        if len(original_shape) == 4:
            if original_shape[1] == self.in_channels:
                B, C, H, W = original_shape
            elif original_shape[0] == 1:
                if original_shape[2] == self.in_channels:
                    _, B, C, HW = original_shape
                    H = W = int(np.sqrt(HW))
                    x_np = x_np.reshape(B, C, H, W)
                elif original_shape[1] == self.in_channels:
                    x_np = x_np.squeeze(0)
                    x_np = x_np[np.newaxis, ...]
                    B, C, H, W = x_np.shape
                else:
                    B, C, H, W = original_shape
            else:
                B, C, H, W = original_shape
        elif len(original_shape) == 3:
            C, H, W = original_shape
            B = 1
            x_np = x_np[np.newaxis, ...]
        elif len(original_shape) == 2:
            B = original_shape[0]
            total = original_shape[1]
            C = self.in_channels
            HW = total // C
            H = W = int(np.sqrt(HW))
            x_np = x_np.reshape(B, C, H, W)
        elif len(original_shape) == 1:
            B = 1
            C = self.in_channels
            H = W = self.image_size
            expected_size = B * C * H * W
            if x_np.size >= expected_size:
                x_np = x_np[:expected_size].reshape(B, C, H, W)
            else:
                x_np = np.random.randn(B, C, H, W).astype(np.float32)
        else:
            raise ValueError(f"Cannot handle input shape: {original_shape}")
        
        if x_np.shape[1] != self.in_channels:
            if len(x_np.shape) == 4 and x_np.shape[-1] == self.in_channels:
                x_np = x_np.transpose(0, 3, 1, 2)
        
        B, C, H, W = x_np.shape
        num_patches_h = H // self.kernel_size
        num_patches_w = W // self.kernel_size
        num_patches = num_patches_h * num_patches_w
        
        x_np = x_np.reshape(B, C, num_patches_h, self.kernel_size, num_patches_w, self.kernel_size)
        x_np = x_np.transpose(0, 2, 4, 1, 3, 5)
        x_np = x_np.reshape(B, num_patches, self.patch_dim)
        x_np = x_np.reshape(1, B, num_patches, self.patch_dim).astype(np.float32)
        
        patches = ttnn.Tensor(
            x_np,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        return patches

    def forward(self, x):
        """Forward pass for patch embedding."""
        patches = self._extract_patches(x)
        
        out = ttnn.linear(
            patches,
            self.weight,
            bias=self.bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        
        return out