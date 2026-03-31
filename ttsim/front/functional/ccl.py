#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
CCL (Collective Communication Library) operations for Polaris
Implements distributed tensor operations across multiple devices
"""

def all_reduce(tensor, mesh_device, cluster_axis=0, dim=3, **kwargs):
    """
    All-reduce operation that sums tensors across all devices in the mesh.
    
    In Mixtral MoE:
    - 8 devices total, each processes 1 expert
    - Each device input: [1, 32, 4096] 
    - Each device output: [1, 1, 32, 512] (4096/8 = 512 per expert)
    - After all-reduce: sum of all expert outputs
    
    Args:
        tensor: Input tensor from each expert
        mesh_device: The device mesh for distributed computation  
        cluster_axis: Axis for reduction (0 = across all devices)
        dim: Dimension to reduce along
        **kwargs: Additional arguments
    
    Returns:
        Tensor with summed values across devices
    """
    # Log the operation for debugging
    if hasattr(tensor, 'shape'):
        print(f"[CCL] all_reduce: input shape={tensor.shape}, dim={dim}, cluster_axis={cluster_axis}")
    
    # In Polaris simulation, return the input tensor
    # The actual hardware would perform the sum reduction across devices
    return tensor

# Alias for compatibility with tt-metal naming convention
tt_all_reduce = all_reduce

def all_gather(tensor, mesh_device, cluster_axis=0, dim=0, num_links=1, **kwargs):
    """
    All-gather operation that concatenates tensors from all devices.
    
    Args:
        tensor: Input tensor from each device
        mesh_device: The device mesh
        cluster_axis: Axis for gathering
        dim: Dimension to concatenate along
        num_links: Number of communication links
        **kwargs: Additional arguments
    
    Returns:
        Concatenated tensor from all devices
    """
    if hasattr(tensor, 'shape'):
        print(f"[CCL] all_gather: input shape={tensor.shape}, dim={dim}, cluster_axis={cluster_axis}")
    
    return tensor

# Alias for compatibility
tt_all_gather = all_gather
