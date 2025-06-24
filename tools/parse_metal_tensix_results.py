#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import re
from pathlib import Path
from typing import List, TypeAlias, Union

import yaml
from loguru import logger
from lxml import html
from pydantic import BaseModel, TypeAdapter

from ttsim.utils.readfromurl import read_from_url

DEFAULT_OUTPUTDIR = Path('data/metal/inf/closed')


class TensixNwPerfMetricModel(BaseModel):
    """
    A Pydantic model for network performance metrics.
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

    class Config:
        extra = 'forbid'  # Disallow extra fields not defined in the model
        populate_by_name = True  # Allow population by field names
        use_enum_values = True  # Use enum values if any are defined


MetricList: TypeAlias = List[TensixNwPerfMetricModel]
MetricListModel = TypeAdapter(MetricList)


ATTRIBUTES = [
    'time_to_first_token_ms',
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
    'ttft_(ms)': 'time_to_first_token_ms',
    'target_t/s/u': 'target_tokens_per_second_per_user',
    'targett/s/u': 'target_tokens_per_second_per_user',
    't/s/u': 'tokens_per_second_per_user',
    't/s': 'tokens_per_second',
    'sen/sec': 'perf',
    'target_sen/sec': 'target_perf',
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


type ValueType = str
type ValueDict = dict[str, ValueType]
type ValueRow = list[ValueType]
type RowList = list[ValueRow]
type ColNames = list[str]


def parse_html_table(table: html.HtmlElement) -> tuple[ColNames, RowList]:
    """
    Parses an HTML table and returns it as a DataFrame.

    Args:
        table (lxml.html.HtmlElement): The HTML table element to parse.

    Returns:
        pd.DataFrame: The parsed table as a DataFrame.
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


def extract_table_from_html_link(link: str) -> list[TensixNwPerfMetricModel]:
    """
    Extracts a table from an HTML link.

    Args:
        link (str): The URL of the HTML page containing the table.

    Returns:
        str: The extracted table in a string format.
    """
    # Placeholder for actual implementation
    html_content: str = read_from_url(link)
    doc: html.HtmlElement = html.fromstring(html_content)

    all_tables = doc.findall('.//table')
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
            if 'bert' not in row['benchmark'].lower() and 'resnet' not in row['benchmark'].lower():
                continue
            assert isinstance(row['system'], str), f"Expected 'system' to be a string, got {type(row['system'])}"
            if not re.search('[a-z][0-9][0-9][0-9]', row['system'].lower()):
                continue

            trimmed_row: ValueDict = {k: v for k, v in row.items() if k != 'release' and v is not None}
            for attr in ['perf', 'target_perf']:
                attr_value = trimmed_row.get(attr, '')
                if isinstance(attr_value, str) and ',' in attr_value:
                    trimmed_row[attr] = trimmed_row[attr].replace(',', '')

            if 'bert' in trimmed_row['benchmark'].lower():
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
    return hw_data


def setup_logger() -> None:
    logger.remove()
    logger.add(sys.stdout, format='{level}:{name}:{line:4}:{message}', level='INFO')


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


def save_metrics(metrics: List[TensixNwPerfMetricModel], output_dir: Path) -> None:
    """
    Saves the extracted metrics to a YAML file.

    Args:
        metrics (list): List of extracted metrics.
        output_dir (Path): Directory to save the metrics.
    """
    wl: str
    metrics_by_wl: dict[str, list[TensixNwPerfMetricModel]] = {}
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
    wlentry: list[TensixNwPerfMetricModel]
    for wl, wlentry in metrics_by_wl.items():
        filename: str = f'tensix_perf_metrics_{wl}.yaml'
        filepath: Path = Path(output_dir) / filename
        modeldump = yaml.dump([wlmodel.model_dump(mode='yaml') for wlmodel in wlentry], indent=4)
        with open(filepath, 'w') as f:
            print(modeldump, file=f)
        logger.info('Saved {} metrics for workload {} to {}', len(wlentry), wl, filepath)
    logger.info('Metrics saved to {}', output_dir)



def create_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Creates command line arguments for the script.

    Args:
        argv (list[str] | None): List of command line arguments. If None, uses sys.argv.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Parse and extract metrics from TT-Metal Tensix results.')
    parser.add_argument('--output-dir', '-o', dest='output_dir', type=str, default=DEFAULT_OUTPUTDIR,
                        help='Directory to save the extracted metrics')
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = create_args(argv)
    setup_logger()
    link = 'https://github.com/tenstorrent/tt-metal/tree/main'
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = extract_table_from_html_link(link)
    report_systems_of_interest(metrics)
    save_metrics(metrics, args.output_dir)
    logger.info('Extracted {} metrics from {}', len(metrics), link)
    return 0


if __name__ == '__main__':
    exit(main())
