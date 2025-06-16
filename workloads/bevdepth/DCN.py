#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.utils.common import parse_yaml
from ttsim.ops import SimTensor
import math

class DCN(SimNN.Module):  # DeformableConv2d
    def __init__( self, name, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=4,):
        super().__init__()
        self.name         = name
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.groups       = groups

        # Convolution for offset prediction (2 * K * K channels)
        self.offset_conv = F.Conv2d(name + '.conv_offset', in_channels,
                                    2 * kernel_size * kernel_size,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    #bias=True,
                                    )

        # Convolution for modulation (K * K channels)
        self.modulator_conv = F.Conv2d(name + '.conv_mod',
                                       in_channels,
                                       kernel_size * kernel_size,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       #bias=True,
                                       )

        # Convolution for main convolution
        self.conv = F.Conv2d( name + '.conv',
                             in_channels * kernel_size * kernel_size,  # Input channels = C * K * K
                             out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             groups=groups,
                             #bias=True,
                             )

        # Initialize weights
        #self._reset_parameters()

#   def _reset_parameters(self):
#       nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
#       if self.conv.bias is not None:
#           fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
#           bound = 1 / math.sqrt(fan_in)
#           nn.init.uniform_(self.conv.bias, -bound, bound)
#       nn.init.constant_(self.offset_conv.weight, 0.0)
#       nn.init.constant_(self.offset_conv.bias, 0.0)
#       nn.init.constant_(self.modulator_conv.weight, 0.0)
#       nn.init.constant_(self.modulator_conv.bias, 0.0)

    def _get_offset_grid(self, height, width, device):
        y, x = torch.meshgrid(
            torch.arange(self.kernel_size, device=device),
            torch.arange(self.kernel_size, device=device),
            indexing='ij',
        )
        y = (y - self.kernel_size // 2).float()
        x = (x - self.kernel_size // 2).float()
        grid = torch.stack([x, y], dim=-1).view(1, -1, 2)  # [1, K*K, 2]
        grid = grid.expand(height, width, -1, -1)  # [H, W, K*K, 2]
        return grid

    def __call__(self, x):
        batch, channels, height, width = x.size()
        k2 = self.kernel_size * self.kernel_size  # K*K, e.g., 9 for 3x3 kernel

        # Predict offsets
        offsets = self.offset_conv(x)                         # [B, 2*K*K, H, W]
        offsets = offsets.view(batch, 2, k2, height, width)   # [B, 2, K*K, H, W]
        offsets = offsets.permute(0, 3, 4, 2, 1)              # [B, H, W, K*K, 2]

        # Predict modulation scalars
        modulators = torch.sigmoid(self.modulator_conv(x))         # [B, K*K, H, W]
        modulators = modulators.view(batch, 1, height, width, k2)  # [B, 1, H, W, K*K]

        # Get base grid
        grid = self._get_offset_grid(height, width, x.device) # [H, W, K*K, 2]

        # Add offsets to grid
        sample_grid = grid + offsets  # [B, H, W, K*K, 2]

        # Normalize grid to [-1, 1] for grid_sample
        sample_grid = sample_grid / torch.tensor( [width - 1, height - 1], device=x.device).view(1, 1, 1, 1, 2) * 2.0 - 1.0

        # Initialize output tensor for sampled features
        sampled_features = torch.zeros(batch, channels, height, width, k2, device=x.device)  # [B, C, H, W, K*K]

        # Loop over K*K points for sampling
        for i in range(k2):
            # Extract grid for the i-th kernel point: [B, H, W, 2]
            grid_i = sample_grid[:, :, :, i, :]  # [B, H, W, 2]
            # Sample features for this kernel point
            sampled_i = F.grid_sample(
                x, grid_i, align_corners=True, mode='bilinear'
            )  # [B, C, H, W]
            # Store in output tensor
            sampled_features[:, :, :, :, i] = sampled_i

        # Apply modulation
        sampled_features = sampled_features * modulators  # [B, C, H, W, K*K]

        # Reshape for grouped convolution
        sampled_features = sampled_features.view(batch, -1, height, width)  # [B, C*K*K, H, W]

        # Apply grouped convolution
        out = self.conv(sampled_features)  # [B, C_out, H, W]
        return out

# Example usage and ONNX export
if __name__ == "__main__":
    # Model parameters
    mid_channels = 256
    kernel_size = 3
    padding = 1
    groups = 4

    # Instantiate the model
    model = DeformableConv2dONNX(
        in_channels=mid_channels,
        out_channels=mid_channels,
        kernel_size=kernel_size,
        padding=padding,
        groups=groups,
    )

    # Set model to evaluation mode
    model.eval()

    # Create a dummy input
    batch_size = 2
    height, width = 64, 64
    dummy_input = torch.randn(batch_size, mid_channels, height, width)

    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")

    # Export to ONNX
    onnx_path = "deformable_conv.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=16,  # GridSample requires opset >= 16
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )

    print(f"Model exported to {onnx_path}")

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")
