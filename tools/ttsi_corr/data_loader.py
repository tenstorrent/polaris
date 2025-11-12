# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Data Loader Module - Load and normalize performance metrics from various sources.

This module handles loading performance metrics from HTML and Markdown format
files, normalizing them into a consistent format for correlation analysis.

Responsibilities:
-----------------
- Load metrics from HTML format files
- Load metrics from Markdown format files
- Normalize MD metrics to standard format
- Read and parse metadata files
- Validate metric configurations

Functions:
----------
- load_metrics_from_sources() - Load metrics based on source type
- load_html_metrics() - Load HTML format metrics
- load_md_metrics() - Load MD format metrics
- normalize_md_metric() - Normalize MD metric model to standard format
- read_metadata() - Read metadata from data directory

Migration Status:
-----------------
🟢 IMPLEMENTED - Phase 4 complete
  ✅ All data loading functions migrated
  ✅ Backward compatibility maintained
  🔴 Tests to be added

Usage:
------
    from ttsi_corr.data_loader import load_metrics_from_sources
    
    metrics = load_metrics_from_sources(
        tensix_perf_data_dir=Path('data/metal/inf/15oct25'),
        data_source='md'
    )

See Also:
---------
- tools/run_ttsi_corr.py: Main correlation script
- tools/parsers/md_parser.py: MD parsing utilities
- tools/ttsi_corr/ttsi_corr_utils.py: Utility functions
"""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from tools.parsers.md_parser import TensixMdPerfMetricModel
from tools.ttsi_corr.ttsi_corr_utils import (get_benchmark_from_model, get_workload_name_from_model,
                                             parse_hardware_to_device)
from ttsim.utils.common import parse_yaml


def load_html_metrics(data_dir: Path) -> list[dict[str, Any]]:
    """
    Load HTML format metrics from data directory.
    
    Loads performance metrics from HTML-format YAML files. Currently supports
    'bert' and 'resnet50' workloads.
    
    Args:
        data_dir: Directory containing metric files
        
    Returns:
        list: Loaded configurations with metadata
        
    Example:
        >>> metrics = load_html_metrics(Path('data/metal/inf/tag'))
        >>> len(metrics)
        42
        
    Note:
        Migrated from tools.run_ttsi_corr._load_html_metrics() in Phase 4.
    """
    html_configs = []
    html_workloads = ['bert', 'resnet50']
    
    for wl in html_workloads:
        data_file = data_dir / f'tensix_perf_metrics_{wl}.yaml'
        if not data_file.exists():
            logger.warning('HTML metrics file not found: {}', data_file)
            continue
        
        data_obj = parse_yaml(data_file)
        for tensix_cfg in data_obj:
            # Validate required fields
            required_fields = ['benchmark', 'gpu', 'gpu_batch_size', 'perf', 
                             'system', 'precision', 'metric']
            missing = [f for f in required_fields if f not in tensix_cfg]
            
            if missing:
                logger.warning('Skipping entry due to missing fields {}: {}', 
                             missing, tensix_cfg.get('wlname', 'unknown'))
                continue
            
            # Add metadata
            tensix_cfg['_source'] = 'html'
            tensix_cfg['_wl_type'] = wl
            html_configs.append(tensix_cfg)
    
    return html_configs


def load_md_metrics(data_dir: Path) -> list[dict[str, Any]]:
    """
    Load MD format metrics from data directory.
    
    Loads performance metrics from Markdown-format YAML files. Supports
    multiple metric types: llm, vision, nlp, detection, diffusion.
    
    Args:
        data_dir: Directory containing metric files
        
    Returns:
        list: Loaded and normalized configurations with metadata
        
    Example:
        >>> metrics = load_md_metrics(Path('data/metal/inf/tag'))
        >>> len(metrics)
        137
        
    Note:
        Migrated from tools.run_ttsi_corr._load_md_metrics() in Phase 4.
    """
    md_configs = []
    md_metric_types = ['llm', 'vision', 'nlp', 'detection', 'diffusion']
    
    for metric_type in md_metric_types:
        data_file = data_dir / f'tensix_md_perf_metrics_{metric_type}.yaml'
        if not data_file.exists():
            logger.warning('MD metrics file not found: {}', data_file)
            continue
        
        data_obj = parse_yaml(data_file)
        for md_cfg_dict in data_obj:
            # Validate required fields
            required_fields = ['model', 'gpu', 'batch', 'hardware', 'precision']
            missing = [f for f in required_fields if f not in md_cfg_dict]
            
            if missing:
                logger.warning('Skipping MD entry due to missing fields {}: {}',
                             missing, md_cfg_dict.get('model', 'unknown'), once=True)
                continue
            
            # Parse and normalize
            try:
                md_model = TensixMdPerfMetricModel(**md_cfg_dict)
                normalized_cfg = normalize_md_metric(md_model)
                
                if normalized_cfg is None:
                    logger.warning('Skipping MD entry due to missing performance metrics: {}',
                                 md_model.model, once=True)
                    continue
                
                # Add metadata
                normalized_cfg['_source'] = 'md'
                normalized_cfg['_wl_type'] = get_workload_name_from_model(
                    normalized_cfg['wlname']
                )
                md_configs.append(normalized_cfg)
                
            except Exception as e:
                logger.warning('Failed to parse MD entry: {} - {}',
                             md_cfg_dict.get('model', 'unknown'), e)
                continue
    
    return md_configs


def normalize_md_metric(md_model: TensixMdPerfMetricModel) -> dict[str, Any] | None:
    """
    Convert MD metric model to correlation-compatible format.
    
    Uses the model's accessor methods to get performance metrics and
    normalizes them to match the HTML format structure.
    
    Args:
        md_model: MD metric model instance
        
    Returns:
        dict | None: Normalized configuration compatible with HTML format,
                    or None if no valid performance metrics
                    
    Example:
        >>> from tools.parsers.md_parser import TensixMdPerfMetricModel
        >>> md = TensixMdPerfMetricModel(
        ...     model='BERT-Large',
        ...     gpu='H100',
        ...     batch=16,
        ...     hardware='n300',
        ...     precision='bf16',
        ...     throughput_infs=100.0,
        ...     target_throughput_infs=120.0
        ... )
        >>> result = normalize_md_metric(md)
        >>> result['perf']
        100.0
        >>> result['target_perf']
        120.0
        
    Note:
        Migrated from tools.run_ttsi_corr.normalize_md_metric() in Phase 4.
    """
    # Use the model's accessor methods to get performance metrics
    perf = md_model.get_perf()
    target_perf = md_model.get_target_perf()
    metric_name = md_model.get_metric_name()
    
    # Check if we have valid performance metrics
    if perf is None or target_perf is None:
        return None
    
    normalized = {
        'benchmark': get_benchmark_from_model(md_model.model),
        'wlname': md_model.model,  # Keep full model name as wlname
        'gpu': md_model.gpu,
        'gpu_batch_size': md_model.batch,
        'system': parse_hardware_to_device(md_model.hardware),
        'precision': md_model.precision,
        'perf': perf,
        'target_perf': target_perf,
        'metric': metric_name,
        'ttft_ms': md_model.ttft_ms,  # Include TTFT if available
    }
    
    return normalized


def read_metadata(data_dir: Path) -> dict[str, Any] | None:
    """
    Read metadata from the data directory if it exists.
    
    Looks for a '_metadata.yaml' file in the specified directory and
    returns its contents as a dictionary.
    
    Args:
        data_dir: Directory containing the _metadata.yaml file
        
    Returns:
        dict | None: Metadata dictionary if file exists and is valid,
                    None otherwise
                    
    Example:
        >>> metadata = read_metadata(Path('data/metal/inf/tag'))
        >>> metadata['data_source']
        'md'
        >>> metadata['tag']
        '15oct25'
        
    Note:
        Migrated from tools.run_ttsi_corr.read_metadata() in Phase 4.
    """
    metadata_file = data_dir / '_metadata.yaml'
    
    if not metadata_file.exists():
        logger.debug('No metadata file found at {}', metadata_file)
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = yaml.safe_load(f)
        logger.debug('Loaded metadata from {}', metadata_file)
        return metadata
    except Exception as e:
        logger.warning('Failed to read metadata from {}: {}', metadata_file, e)
        return None


def load_metrics_from_sources(
    tensix_perf_data_dir: Path,
    data_source: str
) -> list[dict[str, Any]]:
    """
    Load metrics from HTML or MD format files.
    
    High-level facade for loading performance metrics. Delegates to
    appropriate loader based on data_source parameter.
    
    Args:
        tensix_perf_data_dir: Directory containing metric files
        data_source: Source type ('html' or 'md')
        
    Returns:
        list: All loaded configurations with metadata
        
    Raises:
        ValueError: If no valid configurations found or invalid data_source
        
    Example:
        >>> metrics = load_metrics_from_sources(
        ...     Path('data/metal/inf/15oct25'),
        ...     'md'
        ... )
        >>> len(metrics) > 0
        True
        
    Note:
        Migrated from tools.run_ttsi_corr.load_metrics_from_sources() in Phase 4.
    """
    # Load metrics based on source type
    if data_source == 'html':
        all_configs = load_html_metrics(tensix_perf_data_dir)
        logger.debug('Loaded {} HTML format configurations', len(all_configs))
    elif data_source == 'md':
        all_configs = load_md_metrics(tensix_perf_data_dir)
        logger.debug('Loaded {} MD format configurations', len(all_configs))
    else:
        raise ValueError(f'Invalid data_source: {data_source}. Must be "html" or "md".')
    
    if not all_configs:
        raise ValueError(f'No valid configurations found in {tensix_perf_data_dir}')
    
    logger.info('Loaded {} total configurations', len(all_configs))
    return all_configs
