#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import numpy as np

#Reference Naive Implementation fo VoxelPooling used in BEVDepth
class NaiveVoxelPooling(nn.Module):
    def __init__(self, num_voxel_x, num_voxel_y, num_channels):
        """
        Args:
            num_voxel_x: Number of voxels in X dimension (width)
            num_voxel_y: Number of voxels in Y dimension (height)
            num_channels: Feature dimension per point
        """
        super().__init__()
        self.num_voxel_x = num_voxel_x
        self.num_voxel_y = num_voxel_y
        self.num_channels = num_channels

    def forward(
        self,
        geom_xyz,         # (total_samples, 3) - voxel indices [x, y, z] per point
        depth_features,   # (total_samples,) - scalar per sample
        context_features, # (B, num_cams, num_channels, num_height, num_width)
        batch_size,
        num_cams,
        num_depth,
        num_height,
        num_width,
    ):
        """
        Returns:
            output_features: (B, num_voxel_y, num_voxel_x, num_channels)
        """
        # Initialize output grid to zeros
        output_features = torch.zeros(
            batch_size, self.num_voxel_y, self.num_voxel_x, self.num_channels,
            device=context_features.device, dtype=context_features.dtype
        )

        total_samples = batch_size * num_cams * num_depth * num_height * num_width

        # Loop over all "samples" (points) in the flattened tensor
        for sample_idx in range(total_samples):
            # Extract voxel coordinates for this sample
            sample_x = geom_xyz[sample_idx, 0].item()
            sample_y = geom_xyz[sample_idx, 1].item()
            sample_z = geom_xyz[sample_idx, 2].item()

            # Skip if outside valid grid
            if (sample_x < 0 or sample_x >= self.num_voxel_x or
                sample_y < 0 or sample_y >= self.num_voxel_y):
                continue

            # Decode the multi-dimensional index from flattened sample_idx
            width_idx  = sample_idx % num_width
            height_idx = (sample_idx // num_width) % num_height
            depth_idx  = (sample_idx // (num_width * num_height)) % num_depth
            cam_idx    = (sample_idx // (num_width * num_height * num_depth)) % num_cams
            batch_idx  = sample_idx // (num_cams * num_depth * num_height * num_width)

            # Get the scalar depth value for this sample
            depth_val = depth_features[sample_idx]

            for ch in range(self.num_channels):
                # Access context feature: [B, num_cams, num_channels, num_height, num_width]
                context_val = context_features[
                    batch_idx, cam_idx, ch, height_idx, width_idx
                ]
                res = depth_val * context_val

                # Accumulate in the output voxel grid [B, Y, X, C]
                output_features[batch_idx, sample_y, sample_x, ch] += res

        return output_features

####
def test_voxel_pooling():
    # Settings for the test
    batch_size   = 2
    num_cams     = 1
    num_depth    = 2
    num_height   = 2
    num_width    = 3
    num_channels = 4
    num_voxel_x  = 5
    num_voxel_y  = 6

    total_samples = batch_size * num_cams * num_depth * num_height * num_width

    # Generate geom_xyz: random integer voxel indices in [0, num_voxel_x), [0, num_voxel_y)
    # For realism, also add some out-of-bounds indices for edge-case testing
    np.random.seed(0)
    geom_xyz = np.random.randint(
        low=[-1, -1, 0],  # allow -1 for oob test
        high=[num_voxel_x + 1, num_voxel_y + 1, 1],
        size=(total_samples, 3),
    )
    geom_xyz = torch.tensor(geom_xyz, dtype=torch.long) #type: ignore

    # depth_features: random positive floats, shape (total_samples,)
    depth_features = torch.rand(total_samples, dtype=torch.float32)

    # context_features: random floats, shape (B, Cams, Channels, Height, Width)
    context_features = torch.rand(
        batch_size, num_cams, num_channels, num_height, num_width, dtype=torch.float32
    )

    # Instantiate the module
    pooling = NaiveVoxelPooling(num_voxel_x, num_voxel_y, num_channels)

    # Call the pooling operator
    output = pooling(
        geom_xyz, depth_features, context_features,
        batch_size, num_cams, num_depth, num_height, num_width
    )

    # Print some shapes and a sample of the outputs for inspection
    print("geom_xyz.shape:", geom_xyz.shape)
    print("depth_features.shape:", depth_features.shape)
    print("context_features.shape:", context_features.shape)
    print("output.shape:", output.shape)

    # Print nonzero entries for visual verification
    nonzero_idx = torch.nonzero(output)
    print("Nonzero output indices:", nonzero_idx)
    print("Sample output values (nonzero):", output[output != 0])

if __name__ == '__main__':
    test_voxel_pooling()


