# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Correlation Module - Compare and analyze correlation between reference and projected scores.

This module handles the core correlation logic: comparing silicon reference scores
against HLM projected scores, calculating correlation statistics, and generating
comparison reports.

Responsibilities:
-----------------
- Read scores from simulation output
- Compare reference vs projected scores
- Calculate correlation ratios
- Compute geometric mean statistics
- Generate correlation comparison tables

Functions:
----------
- compare_scores() - Compare reference and actual scores
- read_scores() - Read scores from JSON file
- calculate_and_save_geomean() - Calculate and save geometric mean

Migration Status:
-----------------
🟢 IMPLEMENTED - Phase 6 complete
  ✅ All correlation functions migrated
  ✅ Backward compatibility maintained

Usage:
------
    from ttsi_corr.correlation import compare_scores, calculate_and_save_geomean
    
    # Compare scores
    comparison = compare_scores(
        ref_scores=silicon_scores,
        actual_scores=hlm_scores,
        include_override_precision=True
    )
    
    # Calculate geometric mean
    geomean = calculate_and_save_geomean(
        comparison=comparison,
        output_path=Path('output/correlation_geomean.json')
    )

See Also:
---------
- tools/run_ttsi_corr.py: Main correlation script
- tools/ttsi_corr/excel_writer.py: Output generation
"""

import json
import math
from pathlib import Path
from typing import Any

from loguru import logger

type ScoreTuple = tuple[str, str, str, str]
type ScoreDict = dict[ScoreTuple, dict[str, float]]


def read_scores(filepath: Path, default_precision: str | None) -> ScoreDict:
    """
    Read actual scores from Polaris simulation output JSON file.
    
    Parses the simulation summary JSON and extracts performance scores,
    creating a dictionary keyed by (workload, device, api, instance).
    
    Args:
        filepath: Path to the JSON file containing simulation results
        default_precision: Default precision if not specified in results
        
    Returns:
        ScoreDict: Dictionary mapping (wlname, devname, wlgroup, wlinstance) tuples
                  to score dictionaries containing projection scores and metadata
                  
    Example:
        >>> scores = read_scores(Path('output/study-summary.json'), 'fp16')
        >>> key = ('BERT', 'n300', 'TTSIM', 'b16')
        >>> scores[key]['projection']
        1250.5
        
    Note:
        Migrated from tools.run_ttsi_corr.read_scores() in Phase 6.
    """
    actual_scores: ScoreDict = {}
    logger.debug('===============================================')
    logger.debug('Actual scores from {}', filepath)
    
    with open(filepath) as fin:
        actual_results = json.load(fin)['summary']
        for actual_res in actual_results:
            actual_key = tuple([
                actual_res['wlname'],
                actual_res['devname'],
                actual_res['wlgroup'],
                actual_res['wlinstance']
            ])
            actual_scores[actual_key] = {
                'ideal projection': actual_res['ideal_throughput'],
                'projection': actual_res['perf_projection'],
                'precision': default_precision,  # type: ignore[dict-item]
                'ttft_ms': actual_res.get('ttft_ms'),
            }
    
    return actual_scores


def compare_scores(
    ref_scores: ScoreDict,
    actual_scores: ScoreDict,
    include_override_precision: bool = True
) -> list[dict[str, Any]]:
    """
    Compare reference and actual scores to generate correlation data.
    
    Compares silicon reference scores against HLM simulation scores, calculating
    correlation ratios and organizing the data for output. Conditionally includes
    columns based on data availability and configuration.
    
    Args:
        ref_scores: Reference scores from silicon measurements
        actual_scores: Actual scores from HLM simulation
        include_override_precision: Whether to include Override-Precision column.
                               Set to True when precision override was specified on CLI.
                               
    Returns:
        List of dictionaries containing correlation data for CSV/XLSX output.
        Each dictionary contains columns for identification, silicon metrics,
        HLM metrics, and ratio comparisons. The Override-Precision column is
        included only if include_override_precision is True. TTFT columns are
        included only if at least one TTFT value is non-None.
        
    Example:
        >>> comparison = compare_scores(
        ...     ref_scores=silicon_scores,
        ...     actual_scores=hlm_scores,
        ...     include_override_precision=True
        ... )
        >>> len(comparison)
        42
        >>> comparison[0]['Ratio-HLM-to-Si']
        0.95
        
    Note:
        Migrated from tools.run_ttsi_corr.compare_scores() in Phase 6.
    """
    result = []
    logger.debug('===============================================')
    logger.debug('Tensix Correlation Results')
    
    common_keys = set(ref_scores.keys()).intersection(set(actual_scores.keys()))
    only_ref_keys = set(ref_scores.keys()).difference(set(actual_scores.keys()))
    only_actual_keys = set(actual_scores.keys()).difference(set(ref_scores.keys()))
    
    if only_ref_keys:
        logger.info('{} keys present in reference scores but not in actual scores', len(only_ref_keys))
        logger.debug('Keys in reference but not in actual: {}', only_ref_keys)
    if only_actual_keys:
        logger.info('{} keys present in actual scores but not in reference scores', len(only_actual_keys))
        logger.debug('Keys in actual but not in reference: {}', only_actual_keys)
    
    # First pass: determine if we should include TTFT columns
    has_ttft_data = False
    for key in common_keys:
        ref_ttft_ms = ref_scores[key].get('ttft_ms')
        projected_ttft_ms = actual_scores[key].get('ttft_ms')
        if ref_ttft_ms is not None or projected_ttft_ms is not None:
            has_ttft_data = True
            break
    
    if has_ttft_data:
        logger.debug('TTFT data found, including TTFT-ms and HLM-TTFT-ms columns')
    else:
        logger.debug('No TTFT data found, excluding TTFT-ms and HLM-TTFT-ms columns')
    
    for key in sorted(common_keys):
        ref_score = ref_scores[key]['perf']
        ref_target_score = ref_scores[key]['target_perf']
        ref_precision = ref_scores[key]['precision']
        ref_ttft_ms = ref_scores[key].get('ttft_ms')
        
        # Defensive check: skip if ref_score or ref_target_score is None
        if ref_score is None or ref_target_score is None:
            logger.warning('Skipping correlation for key {} due to None ref_score ({}) or ref_target_score ({})',  # type: ignore[unreachable]
                          key, ref_score, ref_target_score)  # type: ignore[unreachable]
            continue  # type: ignore[unreachable]
        
        projection_precision = actual_scores[key]['precision'] if actual_scores[key]['precision'] is not None else ref_precision
        projected_score = actual_scores[key]['projection']
        projected_ideal_score = actual_scores[key].get('ideal projection', projected_score)
        projected_ttft_ms = actual_scores[key].get('ttft_ms')
        ratio_hlm_to_si = projected_score / ref_score if ref_score != 0 else None
        ratio_hlm_to_target = projected_score / ref_target_score if ref_target_score != 0 else None
        
        # Build result dictionary with conditional columns
        # Convert Instance to integer (strip 'b' prefix)
        instance_str = str(key[3])  # Ensure it's a string
        try:
            instance_value: int | str = int(instance_str.strip('b'))
        except (ValueError, AttributeError):
            instance_value = instance_str  # Fallback to original if conversion fails
        
        row_data: dict[str, Any] = {
            'Workload': key[0],
            'Arch': 'Wormhole',  # key[2],  # TODO: Use actual architecture name
            'Batch': instance_value,  # Renamed from Instance, converted to int
            'Device': key[1],
            'Data-Precision': ref_precision,
        }
        
        # Conditionally add TTFT columns
        if has_ttft_data:
            row_data['TTFT-ms'] = ref_ttft_ms
        
        row_data.update({
            'Si-Score': ref_score,
            'Si-Target-Score': ref_target_score,
        })
        
        # Conditionally add Override-Precision column
        if include_override_precision:
            row_data['Override-Precision'] = projection_precision
        
        # Conditionally add HLM-TTFT-ms column
        if has_ttft_data:
            row_data['HLM-TTFT-ms'] = projected_ttft_ms
        
        # Add remaining columns
        row_data.update({
            'HLM-Score': projected_score,
            'HLM-Ideal-Score': projected_ideal_score,
            'Ratio-HLM-to-Si': ratio_hlm_to_si,
            'Ratio-HLM-to-SiTarget': ratio_hlm_to_target,
        })
        
        result.append(row_data)
    
    return result


def calculate_and_save_geomean(comparison: list[dict[str, Any]], output_path: Path) -> float | None:
    """
    Calculate geometric mean of correlation ratios and save to JSON.
    
    Computes the geometric mean of 'Ratio-HLM-to-SiTarget' values from the
    correlation comparison data. Only includes positive, non-None values.
    The geometric mean is calculated as: exp(mean(log(values)))
    
    Saves results to a JSON file containing:
    - geomean_ratio_hlm_to_sitarget: The calculated geometric mean
    - num_valid_ratios: Count of valid ratios used in calculation
    - total_comparisons: Total number of comparison entries
    
    Args:
        comparison: List of dictionaries containing correlation data.
                   Each dict should have a 'Ratio-HLM-to-SiTarget' key.
        output_path: Path where the JSON file will be saved.
    
    Returns:
        float | None: The geometric mean value, or None if no valid ratios found.
    
    Side Effects:
        - Creates a JSON file at output_path with geometric mean data
        - Logs progress, results, and warnings
        
    Example:
        >>> geomean = calculate_and_save_geomean(
        ...     comparison=correlation_data,
        ...     output_path=Path('output/geomean.json')
        ... )
        >>> geomean
        0.9832
        
    Note:
        Migrated from tools.run_ttsi_corr.calculate_and_save_geomean() in Phase 6.
    """
    # Extract valid ratio scores (positive, non-None values)
    ratio_scores = [
        row['Ratio-HLM-to-SiTarget'] 
        for row in comparison 
        if row['Ratio-HLM-to-SiTarget'] is not None and row['Ratio-HLM-to-SiTarget'] > 0
    ]
    
    if not ratio_scores:
        logger.warning('No valid Ratio-HLM-to-SiTarget values found for geometric mean calculation')
        return None
    
    # Calculate geometric mean: (product of all values)^(1/n)
    # Using log space for numerical stability: exp(mean(log(values)))
    geomean_ratio = math.exp(sum(math.log(x) for x in ratio_scores) / len(ratio_scores))
    logger.info(
        'Geometric mean of Ratio-HLM-to-SiTarget: {:.4f} (based on {} valid ratios)', 
        geomean_ratio, 
        len(ratio_scores)
    )
    
    # Prepare data for JSON output
    geomean_data = {
        'geomean_ratio_hlm_to_sitarget': geomean_ratio,
        'num_valid_ratios': len(ratio_scores),
        'total_comparisons': len(comparison)
    }
    
    # Write geometric mean to JSON file
    with open(output_path, 'w') as fout:
        json.dump(geomean_data, fout, indent=2)
    logger.info('Geometric mean written to: {}', output_path)
    
    return geomean_ratio
