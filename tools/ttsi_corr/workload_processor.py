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

from typing import Any

from loguru import logger

from tools.workloads import WorkloadConfig, WorkloadsFile

type ScoreTuple = tuple[str, str, str, str]
type ScoreDict = dict[ScoreTuple, dict[str, float]]

# Original model name to workload instance mapping
__MODEL_NAME_TO_WL_CONFIG: dict[str, dict[str, str | None]] = {
    'bert': {'BERT-Large': 'bert_large', 'BERT-Base': 'bert_base', 'Sentence-Bert (backbone: bert-base)': 'bert_base'},
    'bevdepth': {},
    'resnet50': {'ResNet-50 (224x224)': 'rn50_224x224'},
    'swin': {},
    'unet': {},
    'vgg_unet': {
        'UNet - VGG19 (256x256)': 'vgg_unet_256x256'
    },
    'stablediffusion': {
        'Stable Diffusion 1.4 (512x512)': 'sd_b1',
    },
    'yolov7': {},
    'yolov8': {'YOLOv8s (640x640)': 'yolov8s'},
    'llama2': {},
    'llama3': {'Llama 3.1 8B': 'llama3_8b_decode'},
    'vit': {'ViT-Base (224x224)': 'vit_base_224x224'},
}

# Convert to lowercase for case-insensitive matching
MODEL_NAME_TO_WL_CONFIG: dict[str, dict[str, str | None]] = {k.lower(): {k2.lower(): v2 for k2, v2 in v.items()} for k, v in __MODEL_NAME_TO_WL_CONFIG.items()}  # type: ignore[assignment]


def find_workload_config(workloads_file: WorkloadsFile, wl_name: str, model_name: str) -> tuple[WorkloadConfig, str] | None:
    """
    Find a workload configuration and instance name by short workload name and model name.

    Maps short workload names (bert, resnet50, llama, etc.) and model names to actual
    workload configurations and instance names using MODEL_NAME_TO_WL_CONFIG mapping.

    Args:
        workloads_file: Loaded workloads file
        wl_name: Short workload name (e.g., 'bert', 'resnet50')
        model_name: Full model name (e.g., 'BERT-Large', 'ResNet-50 (224x224)')

    Returns:
        tuple[WorkloadConfig, str] | None: Tuple of (workload_config, instance_name) or None if not found

    Example:
        >>> wl_file = load_workloads_file('config/all_workloads.yaml')
        >>> result = find_workload_config(wl_file, 'bert', 'BERT-Large')
        >>> wl_config, instance_name = result
        >>> wl_config.name
        'bert'
        >>> instance_name
        'bert_large'

    Note:
        Migrated from tools.run_ttsi_corr.find_workload_config() in Phase 5.
    """
    wl_name_lower = wl_name.lower()
    model_name_lower = model_name.lower()
    if wl_name_lower not in MODEL_NAME_TO_WL_CONFIG:
        logger.warning('Unknown workload type: {}', wl_name, once=True)
        return None

    wl_instance = MODEL_NAME_TO_WL_CONFIG.get(wl_name_lower, {}).get(model_name_lower, None)
    if wl_instance is None:
        logger.warning('No workload mapping found for workload "{}" model "{}"', wl_name, model_name, once=True)
        return None

    wl = workloads_file.get_workload_by_name(wl_name_lower)
    if wl is None:
        logger.warning('Workload "{}" not found in workloads file', wl_name)
        return None

    if wl_instance not in wl.instances:
        logger.warning('No instance named "{}" found for workload "{}" (instances {}), skipping', wl_instance, wl_name, wl.instances.keys(), once=True)
        return None
    logger.info('Found workload "{}" with instance "{}"', wl_name, wl_instance)
    return wl, wl_instance


def get_workload_module_config(
    wl_name: str,
    model_name: str,
    batch_size: int,
    workloads_file: WorkloadsFile
) -> dict[str, Any] | None:
    """
    Get workload module and instance configuration from workloads file.

    Loads configuration from the workloads file and selects the appropriate

    Args:
        wl_name: Short workload name (e.g., 'bert', 'resnet50')
        model_name: Full model name used to determine instance (e.g., 'BERT-Large')
        batch_size: Batch size
        workloads_file: Loaded workloads file (required)

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
        The instance name is automatically determined from MODEL_NAME_TO_WL_CONFIG
        based on the model_name parameter.
    """
    # Find configuration from workloads file
    result = find_workload_config(workloads_file, wl_name, model_name)
    if result is None:
        # logger.warning('Workload {} not found in workloads file', wl_name)
        return None

    # Unpack the tuple returned by find_workload_config
    wl_config, selected_instance_name = result

    logger.debug('Using workload configuration from file: {} with instance: {}',
                wl_config.name, selected_instance_name)

    # Extract instance config from workload
    # Use params as base config and merge with selected instance
    instance_config = {'bs': batch_size}

    # Add params to instance config if present
    if wl_config.params:
        instance_config.update(wl_config.params)

    # Get the selected instance configuration
    if selected_instance_name in wl_config.instances:
        selected_instance = wl_config.instances[selected_instance_name]
        # Merge instance config, keeping batch_size override
        instance_config.update(selected_instance)
        instance_config['bs'] = batch_size
    else:
        logger.error('Instance "{}" not found in workload "{}" (this should not happen)',
                    selected_instance_name, wl_config.name)
        return None

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
    wl_config_result: tuple[WorkloadConfig, str] | None = None
) -> tuple[dict[str, Any], ScoreTuple, dict[str, Any]] | None:
    """
    Process a single configuration into workload spec and reference scores.

    Instance selection is automatically determined by MODEL_NAME_TO_WL_CONFIG
    based on the model name in the configuration.

    Args:
        tensix_cfg: Configuration dictionary with workload metadata
        workloads_file: Workload configurations file
        default_precision: Default precision override
        existing_scores: Existing reference scores to check for duplicates
        device_table: Mapping of system names to device names
        wl_config_result: Optional pre-computed workload config result to avoid duplicate lookup

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

    # Find workload configuration to get API (use pre-computed result if available)
    if wl_config_result is None:
        result = find_workload_config(workloads_file, wl, model_name)
        if result is None:
            # Cannot determine API without workload config
            logger.warning('Skipping workload {} (model: {}) - workload config not found, cannot determine API', wl, model_name)
            return None
        wl_config, _ = result
    else:
        wl_config, _ = wl_config_result
    api = wl_config.api

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
        'api': api,
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
    module_config = get_workload_module_config(
        wl, model_name, bs, workloads_file
    )
    if module_config is None:
        return None

    xrec['module'] = module_config['module']
    xrec['basedir'] = module_config.get('basedir', 'workloads')
    xrec['instances'][instance_name].update(module_config['instance_config'])

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
    workload_group_filter: set[str] | None,
    default_precision: str | None,
    device_table: dict[str, str]
) -> tuple[list[dict[str, Any]], ScoreDict, set[str]]:
    """
    Process configurations to create workload specifications and reference scores.

    Instance selection is automatically determined by MODEL_NAME_TO_WL_CONFIG
    based on the model names in the configurations.

    Args:
        all_configs: Validated configurations
        workloads_file: Loaded workload configurations
        workload_filter: Workload filter (None = all)
        workload_group_filter: Workload API group filter (None = all). Case-insensitive.
        default_precision: Default precision override
        device_table: Mapping of system names to device names

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
        ...     workload_group_filter=None,
        ...     default_precision='fp16',
        ...     device_table={'n300': 'n300'}
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
            logger.warning('Skipping config {} as it is not found in device table', tensix_cfg['system'], once=True)
            continue

        # Apply workload-group filter by checking API
        wl_config_result: tuple[WorkloadConfig, str] | None = None
        if workload_group_filter is not None:
            model_name = tensix_cfg.get('wlname', wl)
            wl_config_result = find_workload_config(workloads_file, wl, model_name)
            if wl_config_result is None:
                # Skip if workload config not found (can't determine API)
                continue
            wl_config, _ = wl_config_result
            api = wl_config.api
            if api.upper() not in workload_group_filter:
                continue

        # Process single configuration (pass wl_config_result to avoid duplicate lookup)
        result = process_single_config(
            tensix_cfg, workloads_file, default_precision, metal_ref_scores,
            device_table, wl_config_result
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
