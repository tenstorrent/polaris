# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Simulator Module - Orchestrate Polaris simulation execution.

This module handles the orchestration of Polaris simulations, including
configuration validation, workload filtering, and simulation execution.

Responsibilities:
-----------------
- Validate and filter workload configurations
- Write simulation configuration files
- Execute Polaris simulation
- Handle dry-run mode

Functions:
----------
- validate_and_filter_configs() - Filter configurations by valid metrics
- run_polaris_simulation() - Execute Polaris simulation
- validate_workload_filter() - Validate workload filter has matches

Migration Status:
-----------------
🟢 IMPLEMENTED - Phase 7 complete
  ✅ All simulation functions migrated
  ✅ Backward compatibility maintained

Usage:
------
    from ttsi_corr.simulator import validate_and_filter_configs, run_polaris_simulation
    
    # Validate configurations
    valid_configs = validate_and_filter_configs(
        all_configs=metrics,
        workload_filter={'bert', 'resnet50'}
    )
    
    # Run simulation
    ret = run_polaris_simulation(
        ttsim_wlspec=workload_specs,
        uniq_devs={'n300'},
        opath=Path('output'),
        args=args,
        dry_run=False
    )

See Also:
---------
- tools/run_ttsi_corr.py: Main correlation script
- tools/ttsi_corr/workload_processor.py: Workload processing
"""

import argparse
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from ttsim.utils.common import print_yaml

# Constants
POLPROJ_STUDY_NAME = 'details_correlation'


def validate_workload_filter(
    configs: list[dict[str, Any]],
    workload_filter: set[str]
) -> None:
    """
    Validate that workload filter has matches.
    
    Checks that at least one configuration matches the workload filter.
    Logs available and matching workloads for debugging.
    
    Args:
        configs: List of configuration dictionaries
        workload_filter: Set of workload names to match
        
    Raises:
        ValueError: If no configurations match the filter
        
    Example:
        >>> validate_workload_filter(configs, {'bert', 'resnet50'})
        # Logs: Workload filter matches: ['bert', 'resnet50']
        
    Note:
        Migrated from tools.run_ttsi_corr._validate_workload_filter() in Phase 7.
    """
    matching_workloads = set()
    available_workloads = set()
    
    for cfg in configs:
        wl_type = cfg['_wl_type']
        available_workloads.add(wl_type)
        if wl_type in workload_filter:
            matching_workloads.add(wl_type)
    
    if not matching_workloads:
        logger.error('Workload filter resulted in no matches')
        logger.error('  Requested workloads: {}', sorted(workload_filter))
        logger.error('  Available workloads: {}', sorted(available_workloads))
        raise ValueError('Workload filter resulted in no matches')
    
    logger.info('Workload filter matches: {} (requested: {}, available: {})',
               sorted(matching_workloads), sorted(workload_filter),
               sorted(available_workloads))


def validate_and_filter_configs(
    all_configs: list[dict[str, Any]],
    workload_filter: set[str] | None
) -> list[dict[str, Any]]:
    """
    Filter configurations by valid performance metrics and workload filter.
    
    Removes configurations with None perf or target_perf values, then validates
    that the workload filter (if provided) has matches.
    
    Args:
        all_configs: All loaded configurations
        workload_filter: Set of workload names to include (None = all)
        
    Returns:
        list: Filtered configurations with valid metrics
        
    Raises:
        ValueError: If no valid configurations remain after filtering
        
    Example:
        >>> valid_configs = validate_and_filter_configs(
        ...     all_configs=metrics,
        ...     workload_filter={'bert', 'resnet50'}
        ... )
        >>> len(valid_configs)
        42
        
    Note:
        Migrated from tools.run_ttsi_corr.validate_and_filter_configs() in Phase 7.
    """
    # Filter out configurations with None performance metrics
    logger.debug('Filtering configurations with valid performance metrics...')
    valid_configs = []
    skipped_configs = []
    
    for idx, cfg in enumerate(all_configs):
        perf = cfg.get('perf')
        target_perf = cfg.get('target_perf')
        
        if perf is None or target_perf is None:
            skipped_configs.append({
                'index': idx,
                'workload': cfg.get('wlname', 'unknown'),
                'source': cfg.get('_source', 'unknown'),
                'perf': perf,
                'target_perf': target_perf
            })
        else:
            valid_configs.append(cfg)
    
    if skipped_configs:
        logger.warning('Skipping {} configuration(s) with None performance metrics:',
                      len(skipped_configs))
        for entry in skipped_configs:
            logger.warning('  - Workload: {}, Source: {}, perf={}, target_perf={}',
                          entry['workload'], entry['source'],
                          entry['perf'], entry['target_perf'])
    
    if not valid_configs:
        raise ValueError('No configurations with valid performance metrics found')
    
    # Validate workload filter if specified
    if workload_filter:
        validate_workload_filter(valid_configs, workload_filter)
    
    logger.info('Processing {} valid configurations (skipped {})...',
               len(valid_configs), len(skipped_configs))
    
    return valid_configs


def run_polaris_simulation(
    ttsim_wlspec: list[dict[str, Any]],
    uniq_devs: set[str],
    opath: Path,
    args: argparse.Namespace,
    dry_run: bool
) -> int:
    """
    Write configuration files and execute Polaris simulation.
    
    Creates YAML configuration files for workloads and run settings, then
    executes the Polaris simulation via polproj.py.
    
    Args:
        ttsim_wlspec: Workload specifications
        uniq_devs: Unique device names for filtering
        opath: Output path for configuration files
        args: Command-line arguments containing arch_config and precision
        dry_run: Whether to skip actual execution
        
    Returns:
        int: 0 on success, non-zero on failure
        
    Side Effects:
        - Writes tensix_workloads.yaml to output directory
        - Writes tensix_runcfg.yaml to output directory
        - Executes subprocess for Polaris simulation
        
    Example:
        >>> ret = run_polaris_simulation(
        ...     ttsim_wlspec=workload_specs,
        ...     uniq_devs={'n300'},
        ...     opath=Path('output'),
        ...     args=args,
        ...     dry_run=False
        ... )
        >>> ret
        0
        
    Note:
        Migrated from tools.run_ttsi_corr.run_polaris_simulation() in Phase 7.
    """
    tensix_workloads_yaml_file = 'tensix_workloads.yaml'
    tensix_runcfg_file = 'tensix_runcfg.yaml'
    
    # Write workloads YAML
    workloads_output_file = opath / tensix_workloads_yaml_file
    if dry_run:
        logger.info('[DRY-RUN] Would write workloads YAML to: {}', workloads_output_file)
        logger.debug('[DRY-RUN] Workloads content: {} entries', len(ttsim_wlspec))
    else:
        print_yaml({'workloads': ttsim_wlspec}, workloads_output_file)
    
    # Write run configuration YAML
    wlmapspec = 'config/' + ('wl2archmapping.yaml' if args.precision is None 
                            else f'wl2archmapping_{args.precision}.yaml')
    
    runcfg_dict = {
        'title': 'Metal Tensix Correlation',
        'study': POLPROJ_STUDY_NAME,
        'odir': str(opath),
        'wlspec': str(workloads_output_file),
        'archspec': args.arch_config,
        'wlmapspec': wlmapspec,
        'filterarch': ','.join(uniq_devs),
        'dump_stats_csv': True,
    }
    
    runcfg_file_path = opath / tensix_runcfg_file
    if dry_run:
        logger.info('[DRY-RUN] Would write run configuration YAML to: {}', runcfg_file_path)
        logger.debug('[DRY-RUN] Configuration: {}', runcfg_dict)
        logger.info('[DRY-RUN] Would execute Polaris simulation')
        logger.info('[DRY-RUN] Skipping actual execution and correlation comparison')
        return 0
    
    print_yaml(runcfg_dict, runcfg_file_path)
    
    # Execute simulation
    cmd = ['python', 'polproj.py', '--config', str(runcfg_file_path)]
    cmdstr = ' '.join(cmd)
    
    logger.debug('Executing command: {}', cmdstr)
    proc_result = subprocess.run(cmdstr, shell=True, stderr=subprocess.STDOUT)
    
    if proc_result.returncode != 0:
        logger.error('Command "{}" failed with exit code {}', cmd, proc_result.returncode)
        return proc_result.returncode
    
    return 0
