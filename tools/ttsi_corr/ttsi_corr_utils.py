#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for Metal Tensix performance metrics processing.

This module contains shared functions used by both parse_ttsi_perf_results.py
and run_ttsi_corr.py for consistent handling of workload names,
hardware configurations, and model-to-benchmark mappings.
"""

# Valid tags for TTSI reference data
TTSI_REF_VALID_TAGS = ['15oct25']
TTSI_REF_DEFAULT_TAG = TTSI_REF_VALID_TAGS[0]


def parse_hardware_to_device(hardware: str) -> str:
    """
    Parse hardware string to device name.
    
    Examples:
        'n150 (Wormhole)' -> 'n150'
        'n300 (Wormhole)' -> 'n300'
        'n150' -> 'n150'
        'QuietBox (Wormhole)' -> 'quietbox'
        'Galaxy (Wormhole)' -> 'galaxy'
    
    Args:
        hardware (str): Hardware string from MD metrics.
        
    Returns:
        str: Normalized device name.
    """
    # Remove parentheses and architecture info
    device = hardware.split('(')[0].strip().lower()
    return device


def get_workload_name_from_model(model: str) -> str:
    """
    Derive a short workload name from the model name.
    
    Args:
        model (str): Model name from MD metrics.
        
    Returns:
        str: Short workload identifier.
    """
    model_lower = model.lower()
    
    # BERT variants
    if 'bert-large' in model_lower:
        return 'bert'
    elif 'bert' in model_lower:
        return 'bert'
    
    # ResNet variants
    if 'resnet-50' in model_lower or 'resnet50' in model_lower:
        return 'resnet50'
    
    # Llama variants
    if 'llama' in model_lower:
        return 'llama'
    
    # Mamba
    if 'mamba' in model_lower:
        return 'mamba'
    
    # YOLO variants
    if 'yolo' in model_lower:
        if 'v8' in model_lower:
            return 'yolov8'
        elif 'v7' in model_lower:
            return 'yolov7'
        return 'yolo'
    
    # Stable Diffusion / UNet
    if 'stable diffusion' in model_lower or 'unet' in model_lower:
        return 'unet'
    
    # Default: use first word of model name
    return model.split()[0].lower()


def get_benchmark_from_model(model: str) -> str:
    """
    Map model name to benchmark identifier.
    
    Args:
        model (str): Model name from MD metrics.
        
    Returns:
        str: Benchmark identifier (e.g., 'Benchmark.BERT').
    """
    model_lower = model.lower()
    
    if 'bert' in model_lower:
        return 'Benchmark.BERT'
    elif 'resnet' in model_lower:
        return 'Benchmark.ResNet50'
    elif 'llama' in model_lower:
        return 'Benchmark.Llama'
    elif 'mamba' in model_lower:
        return 'Benchmark.Mamba'
    elif 'yolo' in model_lower:
        return 'Benchmark.YOLO'
    elif 'stable diffusion' in model_lower or 'unet' in model_lower:
        return 'Benchmark.UNet'
    else:
        # Default: use model name as benchmark
        return f'Benchmark.{model.split()[0]}'

