#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import argparse
import ast
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import yaml
from loguru import logger


class AggregatedResultDict(TypedDict):
    """TypedDict for aggregated operator performance results."""
    optype: str
    precision: str
    input_shape_0: Tuple[int, ...]
    input_shape_1: Tuple[int, ...]
    msecs: float


class JsonOutputDict(TypedDict):
    """TypedDict for JSON output format (shapes as lists)."""
    optype: str
    precision: str
    input_shape_0: List[int]
    input_shape_1: List[int]
    msecs: float


def parse_logical_shape(value: str) -> int:
    """
    Extract logical dimension from format like '1[1]' or '224[224]'.

    Args:
        value: String in format 'logical[padded]' or just a number

    Returns:
        Integer logical dimension value

    Raises:
        ValueError: If value cannot be parsed
    """
    if not value or value.strip() == '':
        raise ValueError("Empty shape value")

    # Handle format like "1[1]" or "224[224]"
    match = re.match(r'^(\d+)(?:\[.*\])?$', str(value).strip())
    if match:
        return int(match.group(1))

    # Try to parse as plain integer
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Cannot parse shape value: {value}")

NORMALIZE_PRECISION = {
    'BFLOAT16': 'BF16',
    'BFLOAT8_B': 'BF8',
    'BFLOAT4_B': 'BF4',
    'BFLOAT2_B': 'BF2',
    'FLOAT32': 'FP32',
    'FLOAT16': 'FP16',
    'FLOAT8': 'FP8',
    'INT32': 'INT32',
    'INT16': 'INT16',
    'INT8': 'INT8',
    'BOOL': 'BOOL',
}


def extract_shape(row: Dict[str, str], prefix: str) -> Tuple[int, int, int, int]:
    """
    Extract shape tuple (W, Z, Y, X) from INPUT_0 or INPUT_1 columns.

    Args:
        row: CSV row dictionary
        prefix: Either 'INPUT_0' or 'INPUT_1'

    Returns:
        Tuple of (W, Z, Y, X) integers

    Raises:
        KeyError: If required columns are missing
        ValueError: If shape values cannot be parsed
    """
    w_col = f'{prefix}_W_PAD[LOGICAL]'
    z_col = f'{prefix}_Z_PAD[LOGICAL]'
    y_col = f'{prefix}_Y_PAD[LOGICAL]'
    x_col = f'{prefix}_X_PAD[LOGICAL]'

    required_cols = [w_col, z_col, y_col, x_col]
    missing_cols = [col for col in required_cols if col not in row]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    try:
        w = parse_logical_shape(row[w_col])
        z = parse_logical_shape(row[z_col])
        y = parse_logical_shape(row[y_col])
        x = parse_logical_shape(row[x_col])
        return (w, z, y, x)
    except ValueError as e:
        raise ValueError(f"Error parsing shape for {prefix}: {e}")


def _normalize_shapes_to_tuples(
    shape_0: Any,
    shape_1: Any,
    source_type: str
) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Normalize shapes to tuples, handling different input formats.

    Args:
        shape_0: First shape (can be list, tuple, or other)
        shape_1: Second shape (can be list, tuple, or other)
        source_type: Source type ('yaml', 'json', or 'csv') for error messages

    Returns:
        Tuple of (shape_0, shape_1) as tuples, or None if conversion fails
    """
    # YAML files store shapes as lists; require lists
    if source_type == 'yaml':
        if isinstance(shape_0, list):
            shape_0 = tuple(shape_0)
        else:
            logger.warning(f"Skipping {source_type.upper()} entry with non-list shape_0: {type(shape_0)}")
            return None

        if isinstance(shape_1, list):
            shape_1 = tuple(shape_1)
        else:
            logger.warning(f"Skipping {source_type.upper()} entry with non-list shape_1: {type(shape_1)}")
            return None
    # JSON files can have lists or tuples
    elif source_type == 'json':
        if isinstance(shape_0, list):
            shape_0 = tuple(shape_0)
        elif not isinstance(shape_0, tuple):
            return None

        if isinstance(shape_1, list):
            shape_1 = tuple(shape_1)
        elif not isinstance(shape_1, tuple):
            return None
    # CSV shapes are already tuples after parsing
    elif source_type == 'csv':
        if not isinstance(shape_0, tuple) or not isinstance(shape_1, tuple):
            return None
    else:
        return None

    return (shape_0, shape_1)


def _process_dict_entry(
    entry: Dict[str, Any],
    existing_data: Dict[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], float],
    source_type: str
) -> bool:
    """
    Process a dictionary entry and add it to existing_data if valid.

    Args:
        entry: Dictionary entry with optype field (or op_code for backward compatibility), precision, input_shape_0, input_shape_1, msecs
        existing_data: Dictionary to add the entry to
        source_type: Source type ('yaml' or 'json') for normalization

    Returns:
        True if entry was successfully processed and added, False otherwise
    """
    if not isinstance(entry, dict):
        return False   # type: ignore[unreachable]

    # Read optype from output files (with fallback to op_code for backward compatibility)
    optype = str(entry.get('optype', entry.get('op_code', '')))
    precision = NORMALIZE_PRECISION.get(str(entry.get('precision', '')).upper(), '')
    shape_0 = entry.get('input_shape_0')
    shape_1 = entry.get('input_shape_1')
    msecs = entry.get('msecs')

    if not optype or not precision or shape_0 is None or shape_1 is None or msecs is None:
        return False   # type: ignore[unreachable]

    normalized = _normalize_shapes_to_tuples(shape_0, shape_1, source_type)
    if normalized is None:
        return False

    shape_0_tuple, shape_1_tuple = normalized
    key = (optype, precision, shape_0_tuple, shape_1_tuple)
    existing_data[key] = float(msecs)
    return True


def _process_csv_row(
    row: Dict[str, str],
    existing_data: Dict[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], float]
) -> bool:
    """
    Process a CSV row and add it to existing_data if valid.

    Args:
        row: CSV row dictionary
        existing_data: Dictionary to add the entry to

    Returns:
        True if row was successfully processed and added, False otherwise
    """
    # Read optype from output CSV files (with fallback to op_code for backward compatibility)
    optype = row.get('optype', row.get('op_code', '')).strip()
    precision = NORMALIZE_PRECISION.get(str(row.get('precision', '')).upper(), '')
    shape_0_str = row.get('input_shape_0', '').strip()
    shape_1_str = row.get('input_shape_1', '').strip()
    msecs_str = row.get('msecs', '').strip()

    if not optype or not precision or not shape_0_str or not shape_1_str or not msecs_str:
        return False

    try:
        # Parse shape strings like "(8, 224, 768)"
        shape_0 = ast.literal_eval(shape_0_str)
        shape_1 = ast.literal_eval(shape_1_str)

        normalized = _normalize_shapes_to_tuples(shape_0, shape_1, 'csv')
        if normalized is None:
            return False

        shape_0_tuple, shape_1_tuple = normalized
        msecs = float(msecs_str)
        key = (optype, precision, shape_0_tuple, shape_1_tuple)
        existing_data[key] = msecs
        return True
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Failed to parse CSV row: {e}, skipping")
        return False


def _load_from_list_format(
    data_list: List[Any],
    file_path: Path,
    source_type: str,
    existing_data: Dict[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], float]
) -> bool:
    """
    Load data from a list-based format (YAML or JSON).

    Args:
        data_list: List of dictionary entries
        file_path: Path to the file being loaded (for logging)
        source_type: Source type ('yaml' or 'json')
        existing_data: Dictionary to populate

    Returns:
        True if data was successfully loaded, False otherwise
    """
    if not isinstance(data_list, list):
        return False   # type: ignore[unreachable]

    initial_count = len(existing_data)
    for entry in data_list:
        _process_dict_entry(entry, existing_data, source_type)

    loaded_count = len(existing_data) - initial_count
    if loaded_count > 0:
        logger.info(f"Loaded {loaded_count} entries from existing {source_type.upper()} file")
        return True
    return False


def load_existing_data(output_path: Path) -> Dict[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], float]:
    """
    Load existing performance data from output files if they exist.

    Tries to load from YAML first, then JSON, then CSV (in order of preference).

    Args:
        output_path: Base path for output files (without extension)

    Returns:
        Dictionary mapping (optype, precision, input_shape_0, input_shape_1) to msecs
    """
    existing_data: Dict[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], float] = {}

    yaml_path = output_path.with_suffix('.yaml')
    json_path = output_path.with_suffix('.json')
    csv_path = output_path.with_suffix('.csv')

    # Try YAML first
    if yaml_path.exists():
        logger.info(f"Loading existing data from YAML: {yaml_path}")
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
                if _load_from_list_format(yaml_data, yaml_path, 'yaml', existing_data):
                    return existing_data
        except Exception as e:
            logger.warning(f"Failed to load YAML file {yaml_path}: {e}")

    # Try JSON
    if json_path.exists():
        logger.info(f"Loading existing data from JSON: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if _load_from_list_format(json_data, json_path, 'json', existing_data):
                    return existing_data
        except Exception as e:
            logger.warning(f"Failed to load JSON file {json_path}: {e}")

    # Try CSV
    if csv_path.exists():
        logger.info(f"Loading existing data from CSV: {csv_path}")
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                initial_count = len(existing_data)
                for row in reader:
                    _process_csv_row(row, existing_data)
                loaded_count = len(existing_data) - initial_count
                if loaded_count > 0:
                    logger.info(f"Loaded {loaded_count} entries from existing CSV file")
                    return existing_data
        except Exception as e:
            logger.warning(f"Failed to load CSV file {csv_path}: {e}")

    return existing_data


def detect_conflicts(
    existing_data: Dict[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], float],
    new_data: Dict[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], float],
    tolerance: float = 1e-6
) -> List[Tuple[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], float, float]]:
    """
    Detect conflicts between existing and new data.

    A conflict occurs when the same key exists in both datasets but with
    different msecs values (beyond the tolerance threshold).

    Args:
        existing_data: Existing performance data
        new_data: New performance data
        tolerance: Tolerance for floating point comparison

    Returns:
        List of conflicts as tuples of (key, existing_msecs, new_msecs)
    """
    conflicts = []
    for key, new_msecs in new_data.items():
        if key in existing_data:
            existing_msecs = existing_data[key]
            if abs(existing_msecs - new_msecs) > tolerance:
                conflicts.append((key, existing_msecs, new_msecs))
    return conflicts


def process_operator_perf(input_csv_file: str, output_file: str, force: bool = False) -> None:
    """
    Process TT-NN profiler CSV to extract operator performance mapping.

    If output files exist and force is False, they will be read and new data will be appended.
    If conflicts are detected (same key with different values), warnings will
    be issued and files will not be updated.
    If force is True, existing files will be overwritten without reading them.

    Args:
        input_csv_file: Path to input CSV file
        output_file: Basename for output files (will create .csv, .json, and .yaml)
        force: If True, overwrite existing files without reading them
    """
    # Validate input file
    input_path = Path(input_csv_file)
    if not input_path.exists():
        logger.error(f"Input CSV file not found: {input_csv_file}")
        sys.exit(1)

    # Determine output paths
    output_path = Path(output_file)
    csv_output = output_path.with_suffix('.csv')
    json_output = output_path.with_suffix('.json')
    yaml_output = output_path.with_suffix('.yaml')

    # Load existing data if files exist and force is False
    existing_data: Dict[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], float] = {}
    if not force:
        existing_data = load_existing_data(output_path)
        if existing_data:
            logger.info(f"Found existing data with {len(existing_data)} entries")
    else:
        logger.info("Force mode enabled: existing files will be overwritten")

    # Read and filter operator rows
    operator_rows : List[AggregatedResultDict] = []
    required_columns = [
        'OP CODE',
        'DEVICE FW DURATION [ns]',
        'INPUT_0_DATATYPE',
        'INPUT_1_DATATYPE',
        'INPUT_0_W_PAD[LOGICAL]',
        'INPUT_0_Z_PAD[LOGICAL]',
        'INPUT_0_Y_PAD[LOGICAL]',
        'INPUT_0_X_PAD[LOGICAL]',
        'INPUT_1_W_PAD[LOGICAL]',
        'INPUT_1_Z_PAD[LOGICAL]',
        'INPUT_1_Y_PAD[LOGICAL]',
        'INPUT_1_X_PAD[LOGICAL]',
    ]

    logger.info(f"Reading CSV file: {input_csv_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Check for required columns
        if reader.fieldnames is None:
            logger.error("CSV file has no header row")
            sys.exit(1)

        missing_cols = [col for col in required_columns if col not in reader.fieldnames]
        if missing_cols:
            logger.error(f"Missing required columns in CSV: {missing_cols}")
            sys.exit(1)

        # Filter operator rows and extract data
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (row 1 is header)

            try:
                # Extract OP CODE
                optype = row.get('OP CODE', '').strip()
                if not optype:
                    logger.warning(f"Row {row_num}: Missing OP CODE, skipping")
                    continue

                # Extract precision
                dtype_0 = NORMALIZE_PRECISION.get(str(row.get('INPUT_0_DATATYPE', '')).upper(), '')
                dtype_1 = NORMALIZE_PRECISION.get(str(row.get('INPUT_1_DATATYPE', '')).upper(), '')

                if not dtype_0:
                    logger.warning(f"Row {row_num}: Missing INPUT_0_DATATYPE, skipping")
                    continue

                if dtype_0 != dtype_1:
                    logger.warning(
                        f"Row {row_num}: Input datatypes differ: "
                        f"INPUT_0={dtype_0}, INPUT_1={dtype_1}. Using INPUT_0."
                    )

                precision = dtype_0

                # Extract shapes
                shape_0 = extract_shape(row, 'INPUT_0')
                shape_1 = extract_shape(row, 'INPUT_1')

                # Extract duration and convert to msecs
                duration_ns_str = row.get('DEVICE FW DURATION [ns]', '').strip()
                if not duration_ns_str:
                    logger.warning(f"Row {row_num}: Missing DEVICE FW DURATION [ns], skipping")
                    continue

                try:
                    duration_ns = float(duration_ns_str)
                    msecs = duration_ns / 1_000_000.0
                except ValueError:
                    logger.warning(f"Row {row_num}: Invalid DEVICE FW DURATION [ns] value: {duration_ns_str}, skipping")
                    continue

                operator_row : AggregatedResultDict = {
                    'optype': optype,
                    'precision': precision,
                    'input_shape_0': shape_0,
                    'input_shape_1': shape_1,
                    'msecs': msecs
                }
                operator_rows.append(operator_row)

            except (KeyError, ValueError) as e:
                logger.warning(f"Row {row_num}: Error processing row: {e}, skipping")
                continue

    if not operator_rows:
        logger.error("No operator rows found in CSV file")
        sys.exit(1)

    logger.info(f"Found {len(operator_rows)} operator rows")

    grouped_data: defaultdict[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], List[float]]
    grouped_data = defaultdict(list)

    tmp_row: AggregatedResultDict
    for tmp_row in operator_rows:
        key = (tmp_row['optype'], tmp_row['precision'], tmp_row['input_shape_0'], tmp_row['input_shape_1'])
        assert isinstance(tmp_row['msecs'], float)
        grouped_data[key].append(tmp_row['msecs'])

    # Calculate statistics and check for high variance
    new_data: Dict[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], float] = {}
    aggregated_results: List[AggregatedResultDict] = []
    for (optype, precision, shape_0, shape_1), msecs_values in grouped_data.items():  # type: ignore[assignment]
        # msecs_values is a list of floats (type checker may not infer this correctly)
        mean_msecs: float = statistics.mean(msecs_values)  # type: ignore[arg-type,assignment]

        # Calculate standard deviation (need at least 2 values)
        if len(msecs_values) > 1:
            std_msecs: float = statistics.stdev(msecs_values)  # type: ignore[arg-type,assignment]
            # Threshold: 10% of mean or 0.1 msecs, whichever is larger
            threshold: float = max(mean_msecs * 0.10, 0.1)  # type: ignore[operator]

            if std_msecs > threshold:  # type: ignore[operator]
                logger.warning(
                    f"High variance in msecs for (optype={optype}, precision={precision}, "
                    f"shape0={shape_0}, shape1={shape_1}): "
                    f"mean={mean_msecs:.6f} msecs, std={std_msecs:.6f} msecs, "
                    f"count={len(msecs_values)}"
                )
        else:
            std_msecs = 0.0

        key = (optype, precision, shape_0, shape_1)
        new_data[key] = mean_msecs

        aggregated_results.append({
            'optype': optype,
            'precision': precision,
            'input_shape_0': shape_0,
            'input_shape_1': shape_1,
            'msecs': mean_msecs,
        })

    logger.info(f"Aggregated to {len(aggregated_results)} unique configurations")

    # Check for conflicts with existing data (only if not in force mode)
    if existing_data:
        conflicts = detect_conflicts(existing_data, new_data)
        if conflicts:
            logger.error("Conflicts detected between existing and new data. Files will not be updated.")
            for key, existing_msecs, new_msecs in conflicts:
                optype, precision, shape_0, shape_1 = key  # type: ignore[assignment]
                logger.error(
                    f"Conflict for (optype={optype}, precision={precision}, shape0={shape_0}, shape1={shape_1}): "
                    f"existing={existing_msecs:.6f} msecs, new={new_msecs:.6f} msecs, "
                    f"difference={abs(existing_msecs - new_msecs):.6f} msecs"
                )
            sys.exit(1)

        # Merge existing and new data
        merged_data = existing_data.copy()
        for key, msecs in new_data.items():
            if key not in merged_data:
                merged_data[key] = msecs

        # Rebuild aggregated_results from merged data
        aggregated_results = []
        for (optype, precision, shape_0, shape_1), msecs in merged_data.items():  # type: ignore[assignment]
            aggregated_results.append({
                'optype': optype,  # type: ignore[assignment]
                'precision': precision,  # type: ignore[assignment]
                'input_shape_0': shape_0,  # type: ignore[assignment]
                'input_shape_1': shape_1,  # type: ignore[assignment]
                'msecs': msecs,
            })

        logger.info(f"Merged data: {len(existing_data)} existing + {len(new_data)} new = {len(merged_data)} total entries")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV output
    logger.info(f"Writing CSV output: {csv_output}")
    with open(csv_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['optype', 'precision', 'input_shape_0', 'input_shape_1', 'msecs'])
        writer.writeheader()
        for result in aggregated_results:
            row_dict: Dict[str, str] = {
                'optype': result['optype'],
                'precision': result['precision'],
                'input_shape_0': str(result['input_shape_0']),
                'input_shape_1': str(result['input_shape_1']),
                'msecs': f"{result['msecs']:.6f}",
            }
            writer.writerow(row_dict)

    # Write JSON output
    logger.info(f"Writing JSON output: {json_output}")
    json_data: List[JsonOutputDict] = []
    for result in aggregated_results:
        json_entry: JsonOutputDict = {
            'optype': result['optype'],
            'precision': result['precision'],
            'input_shape_0': list(result['input_shape_0']),
            'input_shape_1': list(result['input_shape_1']),
            'msecs': result['msecs'],
        }
        json_data.append(json_entry)

    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    # Write YAML output
    logger.info(f"Writing YAML output: {yaml_output}")
    yaml_data = []
    for result in aggregated_results:
        yaml_data.append({
            'optype': result['optype'],
            'precision': result['precision'],
            'input_shape_0': list(result['input_shape_0']),  # Convert to list for standard YAML
            'input_shape_1': list(result['input_shape_1']),  # Convert to list for standard YAML
            'msecs': result['msecs'],
        })

    with open(yaml_output, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"Successfully processed {len(operator_rows)} operator rows into {len(aggregated_results)} unique configurations")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process TT-NN profiler CSV to extract operator performance mapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input-csv-file',
        required=True,
        type=str,
        help='Path to TT-NN profiler output CSV file',
    )
    parser.add_argument(
        '--output-file',
        required=True,
        type=str,
        help='Basename for output files (will create .csv, .json, and .yaml files)',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files without reading or merging with existing data',
    )

    args = parser.parse_args()

    # Setup logger
    logger.remove()
    logger.add(sys.stderr, level='INFO', format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>')

    try:
        process_operator_perf(args.input_csv_file, args.output_file, force=args.force)
    except Exception as e:
        logger.exception(f"Error processing CSV: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

