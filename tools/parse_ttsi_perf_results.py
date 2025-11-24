#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, TypeAlias
from urllib.parse import urlparse

import yaml
from loguru import logger
from lxml import html
from pydantic import BaseModel, TypeAdapter

from tools.parsers.md_parser import extract_table_from_md_link, save_md_metrics
from tools.ttsi_corr.ttsi_corr_utils import TTSI_REF_VALID_TAGS
from tools.ttsi_corr.data_loader import load_metrics_from_sources
from ttsim.utils.common import setup_logger
from ttsim.utils.readfromurl import read_from_url

DEFAULT_OUTPUTDIR_BASE = 'data/metal/inf'


class TensixNwPerfMetricModel(BaseModel):
    """
    A Pydantic model for network performance metrics (HTML parsing).
    """

    benchmark: str
    wlname: str
    gpu: str = 'Tensix'
    gpu_batch_size: int

    id: str = 'metal'
    input_dtype: str = 'bf8'  # TODO: confirm if this is always bf8
    metric: str
    perf: float
    precision: str = 'bf8'    # TODO: confirm if this is always bf8
    system: str
    target_perf: float
    ttft_ms: float | None = None

    class Config:
        extra = 'forbid'  # Disallow extra fields not defined in the model
        populate_by_name = True  # Allow population by field names
        use_enum_values = True  # Use enum values if any are defined


MetricList: TypeAlias = List[TensixNwPerfMetricModel]
MetricListModel = TypeAdapter(MetricList)


ATTRIBUTES = [
    'ttft_ms',
    'target_tokens_per_second_per_user',
    'tokens_per_second_per_user',
    'tokens_per_second',
    'sentences_per_second',
    'target_sentences_per_second',
    'tt_metalium_release',
    'vllm_tenstorrent_repo_release',
    'target_perf',
]


COLNAME_MAP: dict[str, str] = {
    'ttft_(ms)': 'ttft_ms',
    'ttft(ms)': 'ttft_ms',
    'ttft': 'ttft_ms',
    'target_t/s/u': 'target_tokens_per_second_per_user',
    'targett/s/u': 'target_tokens_per_second_per_user',
    't/s/u': 'tokens_per_second_per_user',
    't/s': 'tokens_per_second',
    'sen/sec': 'perf',
    'sentence/sec': 'perf',
    'target_sen/sec': 'target_perf',
    'target_sentence/sec': 'target_perf',
    'model': 'benchmark',
    'batch': 'gpu_batch_size',
    'tt-metalium_release': 'tt_metallium_release',
    'vllm_tenstorrent_repo_release': 'vllm_tenstorrent_repo_release',
    'hardware': 'system',
    'fps': 'perf',
    'target_fps': 'target_perf',
}
def get_colname_map(colname: str) -> str:
    c = colname.lower().strip().replace(' ', '_').replace('-', '_')
    return COLNAME_MAP.get(c, c).lower()


def detect_content_format(content: str) -> str:
    """
    Detect if content is HTML or Markdown format.

    Args:
        content (str): Content to analyze.

    Returns:
        str: 'html' if HTML tags detected, 'md' if MD table syntax detected, 'unknown' otherwise.
    """
    # Check for HTML tags
    html_patterns = [r'<table[^>]*>', r'<tr[^>]*>', r'<td[^>]*>', r'<th[^>]*>']
    for pattern in html_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return 'html'

    # Check for MD table syntax
    md_patterns = [r'\|.*\|.*\|', r'\|[-:]+\|']
    for pattern in md_patterns:
        if re.search(pattern, content):
            return 'md'

    return 'unknown'


type ValueType = str
type ValueDict = dict[str, ValueType]
type ValueRow = list[ValueType]
type RowList = list[ValueRow]
type ColNames = list[str]


def parse_html_table(table: html.HtmlElement) -> tuple[ColNames, RowList]:
    """
    Parses an HTML table and returns a tuple of column names and rows.

    Args:
        table (lxml.html.HtmlElement): The HTML table element to parse.

    Returns:
        tuple[list[str], list[list[str]]]: A tuple containing the column names and the rows of the table.
    """
    rows: RowList = []
    column_names: ColNames = []
    for element in table.findall('.//tr'):
        colnames = element.findall('.//th')
        if colnames:
            # If the first row contains column names, use them
            column_names = [get_colname_map(col.text_content().strip()) for col in colnames]
            continue
        cols = element.findall('.//td')
        if not cols:  # pragma: no cover  # Skip empty rows
            continue
        valrow: list[ValueType] = [col.text_content().strip() for col in cols]
        rows.append(valrow)

    return column_names, rows


def process_bert_row(row: ValueDict) -> ValueDict:
    """
    Process a row for BERT benchmark.

    Args:
        row (dict): The row data to process.

    Returns:
        dict: The processed row data.
    """
    row['wlname'] = row.pop('benchmark', 'resnet50')
    row['benchmark'] = 'Benchmark.BERT'
    row['metric'] = 'Samples/s'
    return row


def process_resnet_row(row: ValueDict) -> ValueDict:
    """
    Process a row for ResNet benchmarks.

    Args:
        row (dict): The row data to process.

    Returns:
        dict: The processed row data.
    """
    row['wlname'] = row.pop('benchmark', 'resnet50')
    row['benchmark'] = 'Benchmark.ResNet50'
    row['metric'] = 'Samples/s'
    return row


def extract_table_from_html_link(link: str, use_cache: bool = True) -> List[TensixNwPerfMetricModel]:
    """
    Extracts a table from an HTML link.

    Args:
        link (str): The URL of the HTML page containing the table.
        use_cache (bool): Whether to use cache for fetching content. Defaults to True.
        When set to False, it will always fetch the content from the URL.

    Returns:
        List[TensixNwPerfMetricModel]: The extracted table as a list of TensixNWPerfMetricModel entries, one per row.
    """
    # Read content
    html_content: str = read_from_url(link, use_cache=use_cache)

    # Verify it's actually HTML format
    content_format = detect_content_format(html_content)
    if content_format == 'md':
        raise ValueError(f'Expected HTML format but found MD content in {link}')
    elif content_format == 'unknown':
        raise ValueError(f'Unable to detect valid HTML table format in {link}')

    doc: html.HtmlElement = html.fromstring(html_content)

    all_tables = doc.findall('.//table')

    if not all_tables:
        raise ValueError(f'No HTML tables found in {link}')

    tablelines = [table.sourceline for table in doc.findall('.//table')]
    allh2_containing_tables = [h2 for h2 in doc.findall('.//h2') if h2.sourceline + 1 in tablelines]
    h2_line_text = {h2.sourceline: h2.text_content() for h2 in allh2_containing_tables}
    relevant_tables = [table for table in all_tables if table.sourceline - 1 in h2_line_text]
    hw_data = []
    for table in relevant_tables:
        column_names, parsed_tab = parse_html_table(table)
        if len(column_names) == 2 and 'release' in column_names:
            # Such a table appears at the beginning about release history
            continue
        rows: list[ValueDict] = []
        for htmlrow in parsed_tab:
            new_row = {column_names[ndx]: htmlrow[ndx] for ndx in range(len(column_names))}
            rows.append(new_row)

        for row in rows:
            assert isinstance(row['benchmark'], str), f'Expected "benchmark" to be a string, got {type(row["benchmark"])}'
            assert isinstance(row['system'], str), f"Expected 'system' to be a string, got {type(row['system'])}"
            if not re.search('[a-z][0-9][0-9][0-9]', row['system'].lower()):
                continue

            trimmed_row: ValueDict = {k: v for k, v in row.items() if k != 'release' and v is not None}
            for attr in ['perf', 'target_perf', 'ttft_ms']:
                attr_value = trimmed_row.get(attr, '')
                if isinstance(attr_value, str) and ',' in attr_value:
                    trimmed_row[attr] = trimmed_row[attr].replace(',', '')

            if 'bert-large' in trimmed_row['benchmark'].lower():
                trimmed_row = process_bert_row(trimmed_row)
            else:
                trimmed_row = process_resnet_row(trimmed_row)
            for k in trimmed_row:
                if isinstance(trimmed_row[k], str) and trimmed_row[k].endswith('*'):  # pragma: no cover
                    trimmed_row[k] = trimmed_row[k][:-1]
            try:
                metric: TensixNwPerfMetricModel = TensixNwPerfMetricModel(**trimmed_row)  # type: ignore # unidentifiable error in this call
            except Exception as e:  # pragma: no cover
                logger.error(f'Error parsing row {trimmed_row}: {e}')
                raise
            hw_data.append(metric)

    if not hw_data:
        raise ValueError(f'No supported workloads found in HTML tables from {link}.')

    return hw_data


def report_systems_of_interest(metrics: List[TensixNwPerfMetricModel]) -> None:
    """
    Reports the systems of interest based on the extracted metrics.

    Args:
        metrics (list): List of extracted metrics.
    """
    systems = {metric.system for metric in metrics if metric.system}
    systems_of_interest = sorted([s for s in systems if re.search('[a-z][0-9][0-9][0-9]', s)])
    logger.debug('Unique systems found: {}', len(systems))
    logger.debug('Systems: {}', systems)
    logger.debug('Systems of interest: {}', systems_of_interest)


def save_html_metrics(metrics: List[TensixNwPerfMetricModel], output_dir: Path) -> None:
    """
    Saves the extracted HTML metrics to a YAML file.

    Args:
        metrics (List[TensixNwPerfMetricModel]): List of extracted HTML metrics.
        output_dir (Path): Directory to save the metrics.
    """
    wl: str
    metrics_by_wl: dict[str, List[TensixNwPerfMetricModel]] = {}
    for metric in metrics:
        wl = metric.wlname.lower()
        if 'resnet-50' in wl:
            wl = 'resnet50'
        elif 'bert' in wl:
            wl = 'bert'
        else:
            raise NotImplementedError(f'Unknown workload name {wl} in metric {metric.model_dump()}')
        if wl not in metrics_by_wl:
            metrics_by_wl[wl] = []
        metrics_by_wl[wl].append(metric)
    wlentry: List[TensixNwPerfMetricModel]
    for wl, wlentry in metrics_by_wl.items():
        filename: str = f'tensix_perf_metrics_{wl}.yaml'
        filepath: Path = Path(output_dir) / filename
        modeldump = yaml.dump([wlmodel.model_dump(mode='yaml') for wlmodel in wlentry], indent=4)
        with open(filepath, 'w') as f:
            print(modeldump, file=f)
        logger.info('Saved {} metrics for workload {} to {}', len(wlentry), wl, filepath)
    logger.info('HTML metrics saved to {}', output_dir)


def save_metadata(
    output_dir: Path,
    tag: str,
    data_source: str,
    input_url: str,
    use_cache: bool
) -> None:
    """
    Save metadata about the parsing operation to a YAML file.

    Only writes the file if tag, data_source, or input_url have changed.
    The file is preserved if only parsed_date or use_cache differ.

    Args:
        output_dir (Path): Directory where metadata file will be saved.
        tag (str): Tag identifying this data snapshot.
        data_source (str): Data source format used ('html' or 'md').
        input_url (str): Source URL that was parsed.
        use_cache (bool): Whether caching was enabled during parsing.
    """
    metadata = {
        'tag': tag,
        'data_source': data_source,
        'input_url': input_url,
        'parsed_date': datetime.now().isoformat(),
        'use_cache': use_cache,
    }

    metadata_file = output_dir / '_metadata.yaml'

    # Check if metadata file already exists
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                existing_metadata = yaml.safe_load(f)

            # Compare only the fields that should trigger an overwrite: tag, data_source, input_url
            # Exclude parsed_date and use_cache from comparison
            significant_fields = ['tag', 'data_source', 'input_url']

            # Significant fields is a subset of metadata fields, so do not need to check for missing keys
            metadata_significant = {k: metadata[k] for k in significant_fields}
            # Existing metadata may have additional fields, so use get() to avoid KeyError
            existing_significant = {k: existing_metadata.get(k) for k in significant_fields}

            if metadata_significant == existing_significant:
                logger.info('Metadata unchanged (tag, data_source, input_url), preserving existing file: {}', metadata_file)
                return
        except Exception as e:
            logger.warning('Could not read existing metadata file: {}. Will overwrite.', e)

    # Write metadata if it doesn't exist or has changed
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    logger.info('Saved metadata to {}', metadata_file)


def compare_metrics(orig_output_dir: Path, output_dir: Path) -> bool:
    """
    Compare the metrics in the original output directory with the metrics in the temporary directory.

    Args:
        orig_output_dir (Path): Path to the original/baseline output directory containing metrics to compare against.
        output_dir (Path): Path to the new output directory with freshly parsed metrics.
    Returns:
        bool: True if there are any differences (additions, removals, or changes), False if identical.
    """
    try:
        orig_metrics = load_metrics_from_sources(orig_output_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.warning("No baseline found at '{}': {}. Treating as all new metrics.", orig_output_dir, e)
        orig_metrics = []
    try:
        new_metrics = load_metrics_from_sources(output_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load newly parsed metrics from '{}': {}", output_dir, e)
        return False

    # Create unique keys for each metric configuration
    def create_metric_key(metric: dict) -> str:
        """Create a unique key for a metric configuration dict."""
        key_fields = ['benchmark', 'wlname', 'gpu', 'gpu_batch_size', 'system', 'precision']
        key_parts = []
        for field in key_fields:
            value = metric.get(field, 'unknown')
            key_parts.append(f'{field}={value}')
        return '|'.join(key_parts)

    # Create dictionaries mapping keys to metrics for easier lookup
    orig_metrics_dict = {}
    for metric in orig_metrics:
        key = create_metric_key(metric)
        if key in orig_metrics_dict:
            logger.warning("Duplicate metric key found in original metrics: {}", key)
        orig_metrics_dict[key] = metric
    new_metrics_dict = {}
    for metric in new_metrics:
        key = create_metric_key(metric)
        if key in new_metrics_dict:
            logger.warning("Duplicate metric key found in new metrics: {}", key)
        new_metrics_dict[key] = metric

    # Create sets of keys for comparison
    orig_keys = set(orig_metrics_dict.keys())
    new_keys = set(new_metrics_dict.keys())

    # Find common, only original, and only new keys
    common_keys = orig_keys & new_keys
    only_orig_keys = orig_keys - new_keys
    only_new_keys = new_keys - orig_keys

    # Compare values for common keys
    changed_configs = []
    identical_configs = []

    for key in common_keys:
        orig_metric = orig_metrics_dict[key]
        new_metric = new_metrics_dict[key]

        # Compare all fields except metadata fields (starting with _)
        differences = []
        all_fields = set(orig_metric.keys()) | set(new_metric.keys())

        for field in sorted(all_fields):
            if field.startswith('_'):  # Skip metadata fields
                continue

            orig_value = orig_metric.get(field)
            new_value = new_metric.get(field)

            if orig_value != new_value:
                differences.append(f'{field}: {orig_value} -> {new_value}')

        if differences:
            changed_configs.append((key, differences))
        else:
            identical_configs.append(key)

    # Log comparison results
    logger.info('Loaded {} original metrics and {} new metrics for comparison',
                len(orig_metrics), len(new_metrics))
    logger.info('Common configurations: {}', len(common_keys))
    logger.info('  - Identical: {}', len(identical_configs))
    logger.info('  - Changed: {}', len(changed_configs))
    logger.info('Only in original: {}', len(only_orig_keys))
    logger.info('Only in new: {}', len(only_new_keys))

    if changed_configs:
        logger.info('Configurations with changes:')
        for key, differences in changed_configs:
            logger.info('  ~ {}', key)
            for diff in differences:
                logger.info('    {}', diff)

    if only_orig_keys:
        logger.info('Configurations removed:')
        for key in sorted(only_orig_keys):
            logger.info('  - {}', key)

    if only_new_keys:
        logger.info('Configurations added:')
        for key in sorted(only_new_keys):
            logger.info('  + {}', key)

    # Determine if there are any differences
    has_differences = bool(changed_configs or only_orig_keys or only_new_keys)

    if has_differences:
        logger.info('RESULT: Differences detected between original and new metrics')
    else:
        logger.info('RESULT: No differences found - metrics are identical')

    return has_differences


def create_args(argv: List[str] | None = None) -> argparse.Namespace:
    """
    Creates command line arguments for the script.

    Args:
        argv (List[str] | None): List of command line arguments. If None, uses sys.argv.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Parse and extract metrics from TT-Metal Tensix results.')
    parser.add_argument('--input', '-i', dest='input_url', type=str,
                        default='https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/models/README.md',
                        help='Input URL to parse (HTML or MD file). Defaults to TT-Metal models README.md raw content')
    parser.add_argument('--output-dir', '-o', dest='output_dir', type=str, default=DEFAULT_OUTPUTDIR_BASE,
                        help='Directory to save the extracted metrics')
    parser.add_argument('--tag', '-t', dest='tag', type=str, required=True, choices=TTSI_REF_VALID_TAGS,
                        help=f'Tag for the output subdirectory (required). Valid values: {TTSI_REF_VALID_TAGS}')
    parser.add_argument('--use-cache', '-c', dest='use_cache', action=argparse.BooleanOptionalAction, default=True,
                        help='Use cache for fetching content. Defaults to True. When set to False, it will always fetch the content from the URL.')
    parser.add_argument('--check-uptodate', dest='check_uptodate', action='store_true',
                        help='Check if metrics have changed compared to existing baseline without saving.'
                             'Returns exit code 1 if differences detected, 0 if identical or 2 if an error occurred.')
    parser.add_argument('--loglevel', '-l', dest='loglevel', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level. Defaults to INFO.')
    return parser.parse_args(argv)


def parse_ttsi_perf_results(argv: List[str] | None = None) -> int:
    """
    Main function to parse and extract metrics from TTSI performance results.

    Args:
        argv (List[str] | None): List of command line arguments. If None, uses sys.argv.
        The arguments include options for output directory, tag (mandatory), and caching behavior.

    Returns:
        int: Exit code of the script.
            0: Success (no differences in check-uptodate mode, or successful processing otherwise)
            1: Differences detected (only in check-uptodate mode)
            2: Error during processing
    """
    args = create_args(argv)
    setup_logger(level=args.loglevel)
    link = args.input_url

    # Use temporary directory if check-uptodate is True, otherwise use specified output directory
    if args.check_uptodate:
        with tempfile.TemporaryDirectory() as temp_dir:
            orig_output_dir = Path(args.output_dir) / args.tag
            output_dir = Path(temp_dir) / args.tag
            logger.info('Using temporary directory for check-up-to-date mode: {}', temp_dir)
            os.makedirs(output_dir, exist_ok=True)

            result = _process_metrics(link, args, output_dir)
            if result != 0:
                return result

            # Compare the metrics in the temporary directory with the metrics in the original output directory
            try:
                has_differences = compare_metrics(orig_output_dir, output_dir)
                return 1 if has_differences else 0
            except Exception as e:
                logger.error('Error comparing metrics: {}', e)
                return 2
    else:
        output_dir = Path(args.output_dir) / args.tag
        os.makedirs(output_dir, exist_ok=True)
        return _process_metrics(link, args, output_dir)


def _process_metrics(link: str, args: argparse.Namespace, output_dir: Path) -> int:
    """Process metrics from the given link and save to output directory.

    Args:
        link: URL to fetch metrics from
        args: Parsed command line arguments
        output_dir: Directory to save metrics to

    Returns:
        int: Exit code (0 for success, 2 for error)
    """
    # Track which data source was actually used
    actual_data_source = None

    try:
        # Determine file format based on extension
        parsed_url = urlparse(link)
        path = parsed_url.path.lower()

        if path.endswith('.md'):
            # MD file - parse as markdown
            logger.debug('Detected MD file format from extension')
            md_metrics = extract_table_from_md_link(link, use_cache=args.use_cache)

            # Check if any metrics were extracted
            if not md_metrics:
                logger.error('No metrics found in {}. Verify the file contains valid metric data.', link)
                return 2

            save_md_metrics(md_metrics, output_dir)
            logger.debug('Extracted {} MD metrics from {}', len(md_metrics), link)
            actual_data_source = 'md'

        else:
            # No extension or other - assume HTML, with MD fallback
            logger.info('Assuming HTML file format (no .md extension)')

            try:
                html_metrics = extract_table_from_html_link(link, use_cache=args.use_cache)
                report_systems_of_interest(html_metrics)
                save_html_metrics(html_metrics, output_dir)
                logger.info('Extracted {} HTML metrics from {}', len(html_metrics), link)
                actual_data_source = 'html'

            except (ValueError, Exception) as html_error:
                logger.warning('HTML parsing failed: {}. Trying MD format as fallback...', html_error)

                try:
                    md_metrics = extract_table_from_md_link(link, use_cache=args.use_cache)
                    save_md_metrics(md_metrics, output_dir)
                    logger.info('Successfully extracted {} MD metrics from {} using fallback', len(md_metrics), link)
                    actual_data_source = 'md'

                except Exception as md_error:
                    logger.error('Both HTML and MD parsing failed. HTML error: {}. MD error: {}', html_error, md_error)
                    return 2

        # Save metadata about this parsing operation
        if actual_data_source:
            save_metadata(output_dir, args.tag, actual_data_source, link, args.use_cache)

        return 0

    except Exception as e:
        logger.error('Error processing {}: {}', link, e)
        return 2


def main(argv: List[str] | None = None) -> int:
    return parse_ttsi_perf_results(argv)


if __name__ == '__main__':
    exit(main())
