# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Workload Processor Module - Process and validate workload configurations.

This module handles processing of workload configurations, including mapping
workload names, creating workload specifications, and extracting module configs.

Responsibilities:
-----------------
- Process workload configurations from metrics
- Map short workload names to full configurations
- Create workload specifications for simulation
- Extract module and instance configurations
- Validate and filter workloads
- Handle duplicate detection

Functions:
----------
- process_workload_configs() - Process all configurations
- process_single_config() - Process a single configuration
- find_workload_config() - Find workload by name
- get_workload_module_config() - Get module configuration

Migration Status:
-----------------
🟢 IMPLEMENTED - Phase 5 complete
  ✅ All workload processing functions migrated
  ✅ Backward compatibility maintained

Usage:
------
    from ttsi_corr.workload_processor import process_workload_configs
    
    ttsim_wlspec, metal_ref_scores, uniq_devs = process_workload_configs(
        all_configs=metrics,
        workloads_file=wl_file,
        workload_filter=None,
        default_precision='fp16',
        device_table=DEVICE_TABLE,
        correlation_instance_name='corr'
    )

See Also:
---------
- tools/run_ttsi_corr.py: Main correlation script
- tools/workloads.py: Workload configuration types
"""

from pathlib import Path
from typing import Any

from loguru import logger

from tools.workloads import WorkloadConfig, WorkloadsFile

type ScoreTuple = tuple[str, str, str, str]
type ScoreDict = dict[ScoreTuple, dict[str, float]]


def find_workload_config(workloads_file: WorkloadsFile, wl_name: str) -> WorkloadConfig | None:
    """
    Find a workload configuration by short workload name.
    
    Maps short workload names (bert, resnet50, llama, etc.) to actual workload
    configurations in the workloads file using fuzzy matching.
    
    Args:
        workloads_file: Loaded workloads file
        wl_name: Short workload name (e.g., 'bert', 'resnet50')
        
    Returns:
        WorkloadConfig | None: Matching workload configuration or None if not found
        
    Example:
        >>> wl_file = load_workloads_file('config/all_workloads.yaml')
        >>> config = find_workload_config(wl_file, 'bert')
        >>> config.name
        'BERT'
        
    Note:
        Migrated from tools.run_ttsi_corr.find_workload_config() in Phase 5.
    """
    # Create mapping from short names to workload config names
    name_mapping = {
        'bert': ['bert', 'BERT'],
        'resnet50': ['resnet50', 'RESNET50', 'ResNet', 'resnet'],
        'resnet': ['resnet', 'ResNet', 'RESNET50', 'resnet50'],
        'llama': ['llama', 'llama2', 'Llama'],
        'yolov7': ['yolov7', 'YOLOv7'],
        'yolov8': ['yolov8', 'YOLOv8'],
        'unet': ['unet', 'UNet'],
        'mamba': ['mamba', 'Mamba'],
    }
    
    # Get possible names for this workload (always include the original name first)
    possible_names = name_mapping.get(wl_name.lower(), [wl_name])
    
    # Search for matching workload
    for possible_name in possible_names:
        wl_config = workloads_file.get_workload_by_name(possible_name)
        if wl_config is not None:
            return wl_config
    
    return None


def get_workload_module_config(
    wl_name: str,
    model_name: str,
    batch_size: int,
    workloads_file: WorkloadsFile,
    correlation_instance_name: str = 'corr'
) -> dict[str, Any] | None:
    """
    Get workload module and instance configuration from workloads file.
    
    Loads configuration from the workloads file. Returns None if not found.
    
    Args:
        wl_name: Short workload name (e.g., 'bert', 'resnet50')
        model_name: Full model name for additional context
        batch_size: Batch size
        workloads_file: Loaded workloads file (required)
        correlation_instance_name: Instance name to use for correlation
        
    Returns:
        dict | None: Configuration with 'module', 'basedir', and 'instance_config' keys,
                    or None if workload not found in file
                    
    Example:
        >>> config = get_workload_module_config(
        ...     'bert', 'BERT-Large', 16, wl_file
        ... )
        >>> config['module']
        'basicbert.BasicBERT'
        
    Note:
        Migrated from tools.run_ttsi_corr.get_workload_module_config() in Phase 5.
    """
    # Find configuration from workloads file
    wl_config = find_workload_config(workloads_file, wl_name)
    if wl_config is None:
        logger.warning('Workload {} not found in workloads file', wl_name)
        return None
    
    logger.debug('Using workload configuration from file: {}', wl_config.name)
    
    # Extract instance config from workload
    # Use params as base config and merge with first instance if available
    instance_config = {'bs': batch_size}
    
    # Add params to instance config if present
    if wl_config.params:
        instance_config.update(wl_config.params)
    
    # If there are instances defined, use correlation_instance_name if available
    # (in correlation, we override batch size anyway)
    if wl_config.instances:
        # Prefer correlation_instance_name instance if it exists
        if correlation_instance_name in wl_config.instances:
            selected_instance_name = correlation_instance_name
            logger.debug('Using correlation instance for workload: {}', wl_config.name)
        else:
            logger.warning('Skipped workload {} as instance named "{}" not found in instances', 
                          wl_config.name, correlation_instance_name, once=True)
            return None
        
        selected_instance = wl_config.instances[selected_instance_name]
        # Merge instance config, keeping batch_size override
        instance_config.update(selected_instance)
        instance_config['bs'] = batch_size
    
    return {
        'module': wl_config.module,
        'basedir': wl_config.basedir,
        'instance_config': instance_config
    }


def process_single_config(
    tensix_cfg: dict[str, Any],
    workloads_file: WorkloadsFile,
    default_precision: str | None,
    existing_scores: ScoreDict,
    device_table: dict[str, str],
    correlation_instance_name: str = 'corr'
) -> tuple[dict[str, Any], ScoreTuple, dict[str, Any]] | None:
    """
    Process a single configuration into workload spec and reference scores.
    
    Args:
        tensix_cfg: Configuration dictionary with workload metadata
        workloads_file: Workload configurations file
        default_precision: Default precision override
        existing_scores: Existing reference scores to check for duplicates
        device_table: Mapping of system names to device names
        correlation_instance_name: Instance name to use for correlation
        
    Returns:
        tuple: (xrec, instance_key, ref_scores) or None if should be skipped
        
    Note:
        Migrated from tools.run_ttsi_corr._process_single_config() in Phase 5.
    """
    wl = tensix_cfg['_wl_type']
    model_name = tensix_cfg.get('wlname', wl)
    
    # Extract configuration fields
    bs = tensix_cfg['gpu_batch_size']
    perf = tensix_cfg['perf']
    target_perf = tensix_cfg.get('target_perf', perf)
    
    # Defensive check
    if perf is None or target_perf is None:
        logger.warning('Skipping workload {} due to None perf or target_perf', model_name)
        return None
    
    system = tensix_cfg['system']
    prec = tensix_cfg['precision'] if default_precision is None else default_precision
    
    # Get device name
    if system not in device_table:
        logger.warning('Unknown system "{}", adding to device table', system)
        device_table[system] = system
    gpu_dev = device_table[system]
    
    # Create workload specification
    wl_identifier = model_name.replace(' ', '_').replace('(', '').replace(')', '') \
                               .replace('=', '').replace('/', '_')
    instance_name = f'b{bs}'
    
    xrec = {
        'api': 'TTSIM',
        'basedir': 'workloads',
        'scenario': 'offline',
        'benchmark': tensix_cfg['benchmark'],
        'name': wl_identifier,
        'gpu': tensix_cfg['gpu'],
        'nodes': 1,
        'num_gpu': 1,
        'perf': perf,
        'target_perf': target_perf,
        'system': system,
        'prec': prec,
        'metric': tensix_cfg['metric'],
        'ref_perf': perf,
        'gpu_dev': gpu_dev,
        'instances': {instance_name: {'bs': bs}},
    }
    
    instance_key = (wl_identifier, gpu_dev, xrec['api'], instance_name)
    
    # Check for duplicates
    if instance_key in existing_scores:
        existing = existing_scores[instance_key]
        logger.warning('Duplicate instance key detected, skipping duplicate:')
        logger.warning('  Key: {}', instance_key)
        logger.warning('  Existing: model={}, system={}, prec={}, metric={}, perf={}, target_perf={}',
                      model_name, system, existing['precision'], tensix_cfg['metric'], 
                      existing['perf'], existing['target_perf'])
        logger.warning('  New (skipped): model={}, system={}, prec={}, metric={}, perf={}, target_perf={}',
                      model_name, system, prec, tensix_cfg['metric'], perf, target_perf)
        return None
    
    # Get workload-specific module configuration
    wl_config = get_workload_module_config(
        wl, model_name, bs, workloads_file, correlation_instance_name
    )
    if wl_config is None:
        return None
    
    xrec['module'] = wl_config['module']
    xrec['basedir'] = wl_config.get('basedir', 'workloads')
    xrec['instances'][instance_name].update(wl_config['instance_config'])
    
    logger.debug('{} Instance {}', wl, xrec['instances'])
    
    ref_scores = {
        'perf': perf,
        'target_perf': target_perf,
        'precision': tensix_cfg['precision'],
        'ttft_ms': tensix_cfg.get('ttft_ms'),
        'model_name': model_name,
        'system': system,
        'metric': tensix_cfg['metric'],
    }
    
    return xrec, instance_key, ref_scores


def process_workload_configs(
    all_configs: list[dict[str, Any]],
    workloads_file: WorkloadsFile,
    workload_filter: set[str] | None,
    default_precision: str | None,
    device_table: dict[str, str],
    correlation_instance_name: str = 'corr'
) -> tuple[list[dict[str, Any]], ScoreDict, set[str]]:
    """
    Process configurations to create workload specifications and reference scores.
    
    Args:
        all_configs: Validated configurations
        workloads_file: Loaded workload configurations
        workload_filter: Workload filter (None = all)
        default_precision: Default precision override
        device_table: Mapping of system names to device names
        correlation_instance_name: Instance name to use for correlation
        
    Returns:
        tuple: (ttsim_wlspec, metal_ref_scores, uniq_devs)
            - ttsim_wlspec: List of workload specifications for simulation
            - metal_ref_scores: Dictionary of reference scores
            - uniq_devs: Set of unique device names
            
    Example:
        >>> ttsim_wlspec, scores, devs = process_workload_configs(
        ...     all_configs=metrics,
        ...     workloads_file=wl_file,
        ...     workload_filter=None,
        ...     default_precision='fp16',
        ...     device_table={'n300': 'n300'},
        ...     correlation_instance_name='corr'
        ... )
        >>> len(ttsim_wlspec)
        42
        
    Note:
        Migrated from tools.run_ttsi_corr.process_workload_configs() in Phase 5.
    """
    available_workloads = set(tmp.name.lower() for tmp in workloads_file.workloads)
    ttsim_wlspec = []
    metal_ref_scores: ScoreDict = {}
    uniq_devs = set()
    
    for tensix_cfg in all_configs:
        wl = tensix_cfg['_wl_type']
        
        # Apply filters
        if workload_filter and wl not in workload_filter:
            continue

        if wl.lower() not in available_workloads:
            logger.warning('Skipping workload {} as it is not found in workloads file', wl, once=True)
            continue

        if tensix_cfg['system'] not in device_table:
            logger.warning('Skipping config {} as it is not found in device table', tensix_cfg, once=True)
            continue

        # Process single configuration
        result = process_single_config(
            tensix_cfg, workloads_file, default_precision, metal_ref_scores,
            device_table, correlation_instance_name
        )
        
        if result is None:
            continue
        
        xrec, instance_key, ref_scores = result
        
        # Store results
        metal_ref_scores[instance_key] = ref_scores
        uniq_devs.add(xrec['gpu_dev'])
        ttsim_wlspec.append(xrec)
    
    logger.info('Configured {} workload specifications', len(ttsim_wlspec))
    return ttsim_wlspec, metal_ref_scores, uniq_devs
