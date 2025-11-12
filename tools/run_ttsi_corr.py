#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
from pathlib import Path
from typing import Any

from loguru import logger
from openpyxl import Workbook

from tools.ttsi_corr.chart_builder import add_scurve_chart
from tools.ttsi_corr.correlation import calculate_and_save_geomean, compare_scores, read_scores
from tools.ttsi_corr.data_loader import load_metrics_from_sources, read_metadata
from tools.ttsi_corr.excel_writer import ExcelFormatter, write_csv
from tools.ttsi_corr.simulator import run_polaris_simulation, validate_and_filter_configs
from tools.ttsi_corr.ttsi_corr_utils import TTSI_REF_VALID_TAGS
from tools.ttsi_corr.workload_processor import process_workload_configs
from tools.workloads import WorkloadsFile, load_workloads_file
from ttsim.config import get_arspec_from_yaml
from ttsim.utils.common import setup_logger

type ScoreTuple = tuple[str, str, str, str]
type ScoreDict = dict[ScoreTuple, dict[str, float]]


OUTPUT_DIR: Path = Path('__CORR_TENSIX_HLM_OUT')
POLPROJ_STUDY_NAME: str = 'details_correlation'
CORRELATION_INSTANCE_NAME: str = 'corr'

DEVICE_TABLE: dict[str, str] = {}


# Note: Workload processing functions migrated to ttsi_corr.workload_processor in Phase 5
# Note: Data loading functions migrated to ttsi_corr.data_loader in Phase 4
# Note: Chart functions migrated to ttsi_corr.chart_builder in Phase 3
# Note: Excel writing functions migrated to ttsi_corr.excel_writer in Phase 2


def write_correlation_xlsx(comparison: list[dict[str, Any]], output_path: Path) -> None:
    """
    Write correlation results to an Excel file with formulas and formatting.

    Creates an XLSX file with:
    - Excel formulas for calculated fields (ratios)
    - Comma number formatting for float values
    - Integer formatting for Batch column (0 decimal places)
    - Borders around all cells
    - Auto-filter on header row
    - Frozen panes for easier navigation

    Args:
        comparison: List of dictionaries containing correlation data.
                   Each dict should have keys matching the expected columns.
        output_path: Path where the XLSX file will be saved.

    Returns:
        None

    Side Effects:
        - Creates an XLSX file at output_path
        - Logs progress and completion

    Note:
        Uses ExcelFormatter utility class from ttsi_corr.excel_writer for
        formatting operations (Phase 2 refactoring).
    """
    logger.debug('Writing XLSX file to: {}', output_path)

    # Create Excel formatter instance
    formatter = ExcelFormatter()

    wb = Workbook()
    ws = wb.active
    if ws is None:
        raise RuntimeError('Failed to create worksheet')
    ws.title = 'Correlation Results'

    # Write header row
    headers = list(comparison[0].keys())
    ws.append(headers)

    # Create column index map for formula generation
    col_map = {header: idx + 1 for idx, header in enumerate(headers)}

    # Define which columns should use formulas
    formula_columns = {
        'Ratio-HLM-to-Si',
        'Ratio-HLM-to-SiTarget',
    }

    # Write data rows with formulas for calculated fields
    for row_idx, row in enumerate(comparison, start=2):  # Start at row 2 (after header)
        row_data = []
        for col_idx, header in enumerate(headers, start=1):
            value = row[header]

            # Generate formulas for calculated columns
            if header in formula_columns:
                # Get column letters for source values using ExcelFormatter
                ref_score_col = formatter.col_letter(col_map['Si-Score'])
                target_score_col = formatter.col_letter(col_map['Si-Target-Score'])
                projected_score_col = formatter.col_letter(col_map['HLM-Score'])

                # Generate appropriate formula based on column name
                if header == 'Ratio-HLM-to-Si':
                    formula = f'=IF({ref_score_col}{row_idx}=0, "", {projected_score_col}{row_idx}/{ref_score_col}{row_idx})'
                elif header == 'Ratio-HLM-to-SiTarget':
                    formula = f'=IF({target_score_col}{row_idx}=0, "", {projected_score_col}{row_idx}/{target_score_col}{row_idx})'
                else:
                    formula = value

                row_data.append(formula)
            else:
                row_data.append(value)

        ws.append(row_data)

    # Apply number formatting using ExcelFormatter
    formatter.apply_number_formats(ws, headers, comparison)

    # Apply borders using ExcelFormatter
    formatter.apply_borders(ws, num_rows=len(comparison) + 1, num_cols=len(headers))

    # Enable auto-filter on the header row
    ws.auto_filter.ref = ws.dimensions

    # Freeze panes: freeze after row 1 (header) and after HLM-Ideal-Score column
    if 'HLM-Ideal-Score' in col_map:
        projected_ideal_score_col_idx = col_map['HLM-Ideal-Score']
        # Freeze at the next column after HLM-Ideal-Score, row 2
        freeze_col = formatter.col_letter(projected_ideal_score_col_idx + 1)
        formatter.apply_freeze_panes(ws, freeze_col=freeze_col, freeze_row=2)
    else:
        # Fallback: just freeze after row 1 if column not found
        ws.freeze_panes = 'A2'
        logger.warning('HLM-Ideal-Score column not found, freezing at A2')

    # Add S-curve analysis sheet
    add_scurve_chart(wb, comparison)

    # Save workbook
    wb.save(output_path)
    logger.info('XLSX file written to: {}', output_path)


# Note: Correlation functions migrated to ttsi_corr.correlation in Phase 6
# calculate_and_save_geomean(), read_scores(), and compare_scores() are already imported above


# Note: normalize_md_metric() migrated to ttsi_corr.data_loader in Phase 4
# This is already imported above, so this is just a comment for documentation


# Note: find_workload_config() and get_workload_module_config() migrated to ttsi_corr.workload_processor in Phase 5
# These are already imported above, so this is just a comment for documentation


# Note: read_scores() and compare_scores() migrated to ttsi_corr.correlation in Phase 6
# These are already imported above

def setup_environment(args: argparse.Namespace) -> tuple[Path, bool]:
    """
    Set up the execution environment and output directory.

    Args:
        args: Parsed command-line arguments

    Returns:
        tuple: (output_path, dry_run_flag)
    """
    setup_logger(level=args.loglevel)

    opath = Path(args.output_dir)
    dry_run = args.dry_run

    if dry_run:
        logger.info('[DRY-RUN] Would create output directory: {}', opath)
    else:
        os.makedirs(opath, exist_ok=True)
        logger.debug('Created output directory: {}', opath)

    return opath, dry_run


# Note: read_metadata() migrated to ttsi_corr.data_loader in Phase 4
# This is already imported above, so this is just a comment for documentation


def get_data_source_from_metadata(
    data_dir: Path,
    dry_run: bool
) -> str:
    """
    Get the data source from metadata file.

    Args:
        data_dir: Directory containing the metadata
        dry_run: Whether to run in dry-run mode

    Returns:
        str: Data source ('html' or 'md'), defaults to 'md' if no metadata found

    Raises:
        RuntimeError: If metadata file exists but data_source field is missing
    """
    if dry_run:
        logger.debug('[DRY-RUN] Would read data source from metadata')
        return 'md'  # Default for dry-run

    metadata = read_metadata(data_dir)

    if metadata is None:
        logger.warning(
            'No metadata found in {}. Defaulting to data_source=md. '
            'Consider re-parsing the data with parse_ttsi_perf_results.py to generate metadata.',
            data_dir
        )
        return 'md'

    data_source = metadata.get('data_source')

    if data_source is None:
        raise RuntimeError(
            f'Metadata file exists in {data_dir} but does not contain "data_source" field. '
            f'Please re-parse the data with parse_ttsi_perf_results.py.'
        )

    if data_source not in ['html', 'md']:
        raise RuntimeError(
            f'Invalid data_source in metadata: "{data_source}". Must be "html" or "md".'
        )

    logger.info('Data source from metadata: {} (tag={}, parsed={})',
                data_source, metadata.get('tag'), metadata.get('parsed_date'))

    return data_source


def get_data_directory(
    input_dir: str,
    tag: str,
    dry_run: bool
) -> Path:
    """
    Get and validate the data directory containing Tensix metrics.

    Constructs the full path as input_dir / tag.

    Args:
        input_dir: Base directory containing tagged metric subdirectories
        tag: Tag subdirectory name (e.g., '15oct25')
        dry_run: Whether to run in dry-run mode

    Returns:
        Path: Directory containing Tensix metrics (input_dir / tag)

    Raises:
        RuntimeError: If directory doesn't exist (in non-dry-run mode)
    """
    tensix_perf_data_dir = Path(input_dir) / tag

    if dry_run:
        logger.info('[DRY-RUN] Would use data directory: {}', tensix_perf_data_dir)
        return tensix_perf_data_dir

    if not tensix_perf_data_dir.exists():
        raise RuntimeError(f'Data directory does not exist: {tensix_perf_data_dir}')

    logger.info('Using data directory: {} (from input-dir={}, tag={})',
                tensix_perf_data_dir, input_dir, tag)
    return tensix_perf_data_dir


def load_workload_configs(
    workloads_config_path: str,
    workload_filter_str: str | None
) -> tuple[WorkloadsFile, set[str] | None]:
    """
    Load workload configurations and parse filter if provided.

    Args:
        workloads_config_path: Path to workloads configuration file
        workload_filter_str: Comma-separated workload names to filter

    Returns:
        tuple: (workloads_file, workload_filter_set)

    Raises:
        Exception: If workloads configuration cannot be loaded
    """
    logger.debug('Loading workloads configuration from: {}', workloads_config_path)
    workloads_file = load_workloads_file(workloads_config_path)
    logger.debug('Loaded {} workload configurations', len(workloads_file.workloads))

    workload_filter = None
    if workload_filter_str:
        workload_filter = set(wl.strip() for wl in workload_filter_str.split(','))
        logger.debug('Filtering workloads: {}', workload_filter)

    return workloads_file, workload_filter


# Note: Simulator functions migrated to ttsi_corr.simulator in Phase 7
# validate_and_filter_configs(), run_polaris_simulation(), and validate_workload_filter() are imported above




def generate_correlation_outputs(
    metal_ref_scores: ScoreDict,
    opath: Path,
    default_precision: str | None,
    dry_run: bool
) -> int:
    """
    Compare scores and generate correlation output files.

    Args:
        metal_ref_scores: Reference scores from Tensix
        opath: Output path
        default_precision: Default precision setting
        dry_run: Whether in dry-run mode

    Returns:
        int: 0 on success, non-zero on failure
    """
    if dry_run:
        logger.info('[DRY-RUN] Would read scores from: {}', opath / POLPROJ_STUDY_NAME / 'SUMMARY' / 'study-summary.json')
        logger.info('[DRY-RUN] Would write correlation results to: {}', opath / 'correlation_result.csv')
        logger.info('[DRY-RUN] Would write correlation results to: {}', opath / 'correlation_result.xlsx')
        logger.info('[DRY-RUN] Dry-run completed successfully')
        return 0

    # Read simulation scores
    actual_scores = read_scores(
        opath / POLPROJ_STUDY_NAME / 'SUMMARY' / 'study-summary.json',
        default_precision
    )

    # Compare scores (include Override-Precision column only if precision was specified on CLI)
    comparison = compare_scores(metal_ref_scores, actual_scores, include_override_precision=(default_precision is not None))

    if not comparison:
        logger.warning('No valid comparisons generated (all workloads may have been skipped)')
        logger.info('Correlation completed with no results')
        return 0

    # Write CSV
    correlation_csv_path = opath / 'correlation_result.csv'
    write_csv(comparison, correlation_csv_path)

    # Write XLSX
    correlation_xlsx_path = opath / 'correlation_result.xlsx'
    write_correlation_xlsx(comparison, correlation_xlsx_path)

    # Calculate and save geometric mean
    geomean_json_path = opath / 'correlation_geomean.json'
    calculate_and_save_geomean(comparison, geomean_json_path)

    logger.info('Correlation completed successfully with {} results', len(comparison))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    # Create parser with custom formatter
    parser = argparse.ArgumentParser(
        description='Run Tensix Metal correlation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Reference data group
    ref_data_group = parser.add_argument_group('Reference Data')
    ref_data_group.add_argument('--input-dir', type=str, default='data/metal/inf',
                                help='Base directory containing tagged metric subdirectories (default: %(default)s)')
    ref_data_group.add_argument('--tag', '-t', type=str, choices=TTSI_REF_VALID_TAGS, default=TTSI_REF_VALID_TAGS[0],
                                help=f'Tag subdirectory name (required). Valid values: {TTSI_REF_VALID_TAGS}. (default: %(default)s)')

    # WL/Arch configurations group
    wl_arch_group = parser.add_argument_group('WL/Arch configurations')
    wl_arch_group.add_argument('--workloads-config', type=str, default='config/ttsi_correlation_workloads.yaml',
                               help='Workloads configuration file (default: %(default)s)')
    wl_arch_group.add_argument('--arch-config', type=str, default='config/tt_wh.yaml',
                               help='Architecture specification file (default: %(default)s)')
    wl_arch_group.add_argument('--arch-name', type=str, default='wormhole',
                               help='Architecture name to correlate against (default: %(default)s)')
    wl_arch_group.add_argument('--workload-filter', type=str,
                               help='Comma-separated workload names to process. '
                                    'Common workloads: bert, llama, mamba, resnet50, unet, yolo, yolov7, yolov8. '
                                    'Additional workloads may be available depending on data source (e.g., falcon, mistral, mixtral, qwen, whisper, vit-base, etc.). '
                                    'Example: --workload-filter bert,llama,resnet50')
    wl_arch_group.add_argument('--precision', type=str,
                               help='Override precision for all workloads (e.g., bf8, bf16, fp32). '
                                    'If not specified, uses precision from parsed metrics data')

    # Other arguments
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR.as_posix(),
                        help='Output directory for results (default: %(default)s)')
    parser.add_argument('--dry-run', '--dryrun', action='store_true',
                        help='Show what would be done without executing')
    parser.add_argument('--loglevel', '--log-level', type=str.upper, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging level (default: %(default)s)')

    args = parser.parse_args(argv)

    return args


def main(argv: list[str]) -> int:
    """
    Main entry point for Metal-Tensix correlation analysis.

    Orchestrates the complete correlation workflow:
    1. Setup environment and parse arguments
    2. Construct data directory from input-dir and tag
    3. Read data source format (html/md) from metadata file
    4. Load workload configurations
    5. Load metrics from specified data source
    6. Validate and filter configurations
    7. Process workloads and create specs
    8. Run Polaris simulation
    9. Generate correlation outputs (CSV, XLSX, JSON)

    Required Arguments:
        --tag: Tag identifying the parsed metrics dataset

    Optional Arguments:
        --input-dir: Base directory containing tagged metrics (default: data/metal/inf)
        --workloads-config: Workload configuration file
        --arch-config: Architecture specification file
        --workload-filter: Comma-separated workload names to process
        --precision: Override precision for all workloads
        --output-dir: Output directory for results
        --dry-run: Preview actions without executing

    Note:
        Requires that metrics have been pre-parsed using parse_ttsi_perf_results.py
        which creates the metadata file containing data source information.

    Args:
        argv: Command-line arguments

    Returns:
        int: 0 on success, non-zero on failure
    """
    # Parse arguments and setup
    args = parse_args(argv[1:])
    opath, dry_run = setup_environment(args)

    try:
        # Get data directory (constructed from input_dir / tag)
        arch_specs = get_arspec_from_yaml(args.arch_config)
        arch_specs_packages = arch_specs[1]

        # Build device table from arch config based on requested arch name
        arch_name_lower = args.arch_name.lower()
        device_names = [
            pkgname for pkgname in arch_specs_packages
            if arch_specs_packages[pkgname].devname.lower() == arch_name_lower
        ]

        if not device_names:
            logger.error('Architecture name "{}" not found in arch config: {}',
                        args.arch_name, args.arch_config)
            logger.error('Available architectures: {}',
                        sorted(set(pkg.devname for pkg in arch_specs_packages.values())))
            return 1

        # Populate device table
        for devname in device_names:
            DEVICE_TABLE[devname] = devname

        logger.debug('Device table populated for arch "{}": {}', args.arch_name, device_names)

        tensix_perf_data_dir = get_data_directory(args.input_dir, args.tag, dry_run)

        # Get data source from metadata
        data_source = get_data_source_from_metadata(tensix_perf_data_dir, dry_run)

        # Load workload configurations
        workloads_file, workload_filter = load_workload_configs(
            args.workloads_config,
            args.workload_filter
        )

        # Load metrics from data source specified in metadata
        all_configs = load_metrics_from_sources(tensix_perf_data_dir, data_source)

        # Validate and filter configurations
        valid_configs = validate_and_filter_configs(all_configs, workload_filter)

        # Process workload configurations
        ttsim_wlspec, metal_ref_scores, uniq_devs = process_workload_configs(
            valid_configs,
            workloads_file,
            workload_filter,
            args.precision,
            DEVICE_TABLE
        )

        # Run Polaris simulation
        ret = run_polaris_simulation(ttsim_wlspec, uniq_devs, opath, args, dry_run)
        if ret != 0:
            return ret

        # Generate correlation outputs
        return generate_correlation_outputs(
            metal_ref_scores,
            opath,
            args.precision,
            dry_run
        )

    except Exception as e:
        logger.error('Correlation failed: {}', e)
        return 1


if __name__ == '__main__':
    exit(main(sys.argv))
