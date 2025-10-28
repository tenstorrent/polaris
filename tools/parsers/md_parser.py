#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Parser for Markdown tables containing Tensix Metal performance metrics.
Uses markdown-it-py library for robust, standards-compliant parsing.
"""
import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeAlias

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import yaml
from loguru import logger
from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import BaseModel, TypeAdapter

from ttsim.utils.readfromurl import read_from_url


class TensixMdPerfMetricModel(BaseModel):
    """
    A Pydantic model for network performance metrics from MD files.
    Uses exact field names from MD tables with optional fields for different metric types.
    """

    # Common fields across all MD tables
    model: str
    batch: int
    hardware: str
    release: Optional[str] = None

    # LLM and Speech-to-Text specific fields
    ttft_ms: Optional[float] = None
    tokens_per_sec_per_user: Optional[float] = None
    target_tokens_per_sec_per_user: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    vllm_repo_release: Optional[str] = None

    # Diffusion model specific fields
    sec_per_image: Optional[float] = None
    target_sec_per_image: Optional[float] = None

    # Classification model specific fields
    images_per_sec: Optional[float] = None
    target_images_per_sec: Optional[float] = None

    # Object Detection and Segmentation specific fields
    fps: Optional[float] = None
    target_fps: Optional[float] = None

    # NLP model specific fields
    sentences_per_sec: Optional[float] = None
    target_sentences_per_sec: Optional[float] = None

    # Derived fields for compatibility
    gpu: str = 'Tensix'
    id: str = 'metal'
    input_dtype: str = 'bf8'
    precision: str = 'bf8'

    class Config:
        extra = 'forbid'
        populate_by_name = True
        use_enum_values = True

    # Generic performance accessors
    def get_perf(self) -> Optional[float]:
        """Get the primary performance metric value."""
        if self.tokens_per_sec_per_user is not None:
            return self.tokens_per_sec_per_user
        elif self.images_per_sec is not None:
            return self.images_per_sec
        elif self.fps is not None:
            return self.fps
        elif self.sentences_per_sec is not None:
            return self.sentences_per_sec
        elif self.sec_per_image is not None:
            return 1.0 / self.sec_per_image if self.sec_per_image > 0 else None
        return None

    def get_target_perf(self) -> Optional[float]:
        """Get the target performance metric value."""
        if self.target_tokens_per_sec_per_user is not None:
            return self.target_tokens_per_sec_per_user
        elif self.target_images_per_sec is not None:
            return self.target_images_per_sec
        elif self.target_fps is not None:
            return self.target_fps
        elif self.target_sentences_per_sec is not None:
            return self.target_sentences_per_sec
        elif self.target_sec_per_image is not None:
            return 1.0 / self.target_sec_per_image if self.target_sec_per_image > 0 else None
        return None

    def get_metric_name(self) -> str:
        """Get the metric name based on available fields."""
        if self.tokens_per_sec_per_user is not None:
            return 'tokens/sec/user'
        elif self.images_per_sec is not None:
            return 'images/sec'
        elif self.fps is not None:
            return 'fps'
        elif self.sentences_per_sec is not None:
            return 'sentences/sec'
        elif self.sec_per_image is not None:
            return 'images/sec'  # Convert to images/sec for consistency
        return 'performance'


MdMetricList: TypeAlias = List[TensixMdPerfMetricModel]
MdMetricListModel = TypeAdapter(MdMetricList)


# Column mapping for MD files - maps MD column names to model field names
MD_COLNAME_MAP: dict[str, str] = {
    'ttft (ms)': 'ttft_ms',
    'ttft_(ms)': 'ttft_ms',
    't/s/u': 'tokens_per_sec_per_user',
    'tokens/sec/user': 'tokens_per_sec_per_user',
    'target t/s/u': 'target_tokens_per_sec_per_user',
    'target_t/s/u': 'target_tokens_per_sec_per_user',
    'targett/s/u': 'target_tokens_per_sec_per_user',  # Handle actual column name from TT-Metal README
    't/s': 'tokens_per_sec',
    'tokens/sec': 'tokens_per_sec',
    'sec/image': 'sec_per_image',
    'target sec/image': 'target_sec_per_image',
    'target_sec/image': 'target_sec_per_image',
    'image/sec': 'images_per_sec',
    'images/sec': 'images_per_sec',
    'target image/sec': 'target_images_per_sec',
    'target_image/sec': 'target_images_per_sec',
    'target images/sec': 'target_images_per_sec',
    'frame/sec (fps)': 'fps',
    'fps': 'fps',
    'target fps': 'target_fps',
    'target_fps': 'target_fps',
    'sentence/sec': 'sentences_per_sec',
    'sentences/sec': 'sentences_per_sec',
    'target sentence/sec': 'target_sentences_per_sec',
    'target_sentence/sec': 'target_sentences_per_sec',
    'tt-metalium release': 'release',
    'tt_metalium_release': 'release',
    'release': 'release',
    'vllm tenstorrent repo release': 'vllm_repo_release',
    'vllm_tenstorrent_repo_release': 'vllm_repo_release',
    'model': 'model',
    'batch': 'batch',
    'hardware': 'hardware',
}


def get_md_colname_map(colname: str) -> str:
    """Map MD table column names to model field names."""
    c = colname.lower().strip()
    return MD_COLNAME_MAP.get(
        c, 
        colname.lower()
               .replace(' ', '_')
               .replace('-', '_')
               .replace('/', '_')
               .replace('(', '')
               .replace(')', '')
    )


def parse_numeric_md_value(value: str) -> Optional[float]:
    """
    Parse numeric value from markdown cell, handling various formats.
    
    Args:
        value (str): String value to parse.
        
    Returns:
        Optional[float]: Parsed numeric value or None if not parseable.
    """
    if not value or value == '-' or value == '':
        return None

    # Remove asterisks and other non-numeric characters except decimal points
    cleaned = re.sub(r'[^\d.,]', '', value)
    if not cleaned:
        return None

    # Handle comma as thousands separator
    cleaned = cleaned.replace(',', '')

    try:
        return float(cleaned)
    except ValueError:
        return None


def clean_cell_value(value: str) -> str:
    """
    Clean markdown cell value by removing markdown formatting.
    
    Args:
        value (str): Raw cell value.
        
    Returns:
        str: Cleaned cell value.
    """
    # Remove markdown links but keep the text
    value = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', value)
    # Remove HTML tags
    value = re.sub(r'<[^>]+>', '', value)
    # Clean up whitespace
    value = value.strip()
    return value


def extract_table_data_from_tokens(tokens: List[Token]) -> List[Dict[str, Any]]:
    """
    Extract table data from markdown-it parsed tokens.
    
    Args:
        tokens (List[Token]): List of markdown-it tokens.
        
    Returns:
        List[Dict[str, Any]]: List of table row dictionaries.
    """
    tables_data: List[Dict[str, Any]] = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        
        # Look for table_open token
        if token.type == 'table_open':
            logger.debug('Found table at token index {}', i)
            i += 1
            
            # Next should be thead_open
            if i >= len(tokens) or tokens[i].type != 'thead_open':
                logger.warning('Expected thead_open after table_open')
                i += 1
                continue
            
            i += 1  # Move past thead_open
            
            # Extract headers
            headers: List[str] = []

            # TODO: (Review suggestion) Consider using iterators rather than manual indexing
            if i < len(tokens) and tokens[i].type == 'tr_open':
                i += 1  # Move past tr_open
                
                while i < len(tokens) and tokens[i].type != 'tr_close':
                    if tokens[i].type == 'th_open':
                        i += 1  # Move to inline content
                        if i < len(tokens) and tokens[i].type == 'inline':
                            header_text = clean_cell_value(tokens[i].content)
                            headers.append(header_text)
                            logger.debug('Found header: {}', header_text)
                        i += 1  # Move past inline
                        if i < len(tokens) and tokens[i].type == 'th_close':
                            i += 1  # Move past th_close
                    else:
                        i += 1
                
                i += 1  # Move past tr_close
            
            i += 1  # Move past thead_close
            
            # Skip if no valid headers or 'model' not in headers
            # This also filters out release tables which don't have 'model' column
            if not headers or 'model' not in [h.lower() for h in headers]:
                logger.debug('Skipping table - no headers or no "model" column')
                continue
            
            # Map headers to field names
            mapped_headers = [get_md_colname_map(h) for h in headers]
            logger.debug('Mapped headers: {}', mapped_headers)
            
            # Extract tbody data
            if i >= len(tokens) or tokens[i].type != 'tbody_open':
                logger.warning('Expected tbody_open after thead')
                continue
            
            i += 1  # Move past tbody_open
            
            # Process data rows
            row_count = 0
            while i < len(tokens) and tokens[i].type != 'tbody_close':
                if tokens[i].type == 'tr_open':
                    i += 1  # Move past tr_open
                    
                    # Extract row cells
                    cells: List[str] = []
                    while i < len(tokens) and tokens[i].type != 'tr_close':
                        if tokens[i].type == 'td_open':
                            i += 1  # Move to inline content
                            if i < len(tokens) and tokens[i].type == 'inline':
                                cell_text = clean_cell_value(tokens[i].content)
                                cells.append(cell_text)
                            i += 1  # Move past inline
                            if i < len(tokens) and tokens[i].type == 'td_close':
                                i += 1  # Move past td_close
                        else:
                            i += 1
                    
                    i += 1  # Move past tr_close
                    
                    # Create row dictionary
                    if len(cells) >= len(mapped_headers):
                        row_data: Dict[str, Any] = {}
                        skip_row = False
                        
                        for idx, field_name in enumerate(mapped_headers):
                            if idx < len(cells):
                                cell_value = cells[idx]
                                
                                # Parse values based on field type
                                if field_name == 'batch':
                                    numeric_val = parse_numeric_md_value(cell_value)
                                    if numeric_val is None:
                                        # Batch is required and must be valid - skip this row
                                        logger.warning('Skipping row with invalid batch value: {}', cell_value)
                                        skip_row = True
                                        break
                                    row_data[field_name] = int(numeric_val)
                                elif field_name in ['ttft_ms', 'tokens_per_sec_per_user', 'target_tokens_per_sec_per_user',
                                                   'tokens_per_sec', 'sec_per_image', 'target_sec_per_image',
                                                   'images_per_sec', 'target_images_per_sec', 'fps', 'target_fps',
                                                   'sentences_per_sec', 'target_sentences_per_sec']:
                                    row_data[field_name] = parse_numeric_md_value(cell_value)
                                else:
                                    row_data[field_name] = cell_value if cell_value else None
                        
                        # Only add if has required fields and row is valid
                        if not skip_row and 'model' in row_data and 'batch' in row_data and 'hardware' in row_data:
                            tables_data.append(row_data)
                            row_count += 1
                    
                else:
                    i += 1
            
            logger.debug('Extracted {} rows from table', row_count)
            i += 1  # Move past tbody_close
            
        else:
            i += 1
    
    return tables_data


def extract_table_from_md_link(link: str, use_cache: bool = True) -> List[TensixMdPerfMetricModel]:
    """
    Extracts tables from a Markdown link using markdown-it-py parser.
    
    Args:
        link (str): The URL of the MD page containing the tables.
        use_cache (bool): Whether to use cache for fetching content. Defaults to True.
        
    Returns:
        List[TensixMdPerfMetricModel]: The extracted tables as a list of TensixMdPerfMetricModel entries.
    """
    logger.debug('Parsing markdown tables from: {}', link)
    
    # Read content
    md_content: str = read_from_url(link, use_cache=use_cache)
    
    # Parse with markdown-it-py
    md = MarkdownIt().enable('table')
    tokens = md.parse(md_content)
    
    logger.debug('Parsed {} tokens from markdown content', len(tokens))
    
    # Extract table data from tokens
    table_rows = extract_table_data_from_tokens(tokens)
    
    logger.debug('Extracted {} raw table rows', len(table_rows))
    
    # Convert to Pydantic models
    md_data: List[TensixMdPerfMetricModel] = []
    for row_data in table_rows:
        try:
            metric = TensixMdPerfMetricModel(**row_data)
            md_data.append(metric)
        except Exception as e:
            logger.warning('Failed to create model from row {}: {}', row_data.get('model', 'unknown'), e)
            continue
    
    if not md_data:
        raise ValueError(f'No valid table data extracted from {link}')
    
    logger.info('Extracted {} metrics from MD file {}', len(md_data), link)
    return md_data


def save_md_metrics(metrics: List[TensixMdPerfMetricModel], output_dir: Path) -> None:
    """
    Saves the extracted MD metrics to YAML files grouped by metric type.
    
    Args:
        metrics (List[TensixMdPerfMetricModel]): List of extracted MD metrics.
        output_dir (Path): Directory to save the metrics.
    """
    # Group metrics by model type for organization
    metrics_by_type: dict[str, List[TensixMdPerfMetricModel]] = {}

    for metric in metrics:
        # Determine metric type based on available performance fields
        metric_type = 'unknown'
        if metric.tokens_per_sec_per_user is not None:
            metric_type = 'llm'
        elif metric.images_per_sec is not None:
            metric_type = 'vision'
        elif metric.fps is not None:
            metric_type = 'detection'
        elif metric.sentences_per_sec is not None:
            metric_type = 'nlp'
        elif metric.sec_per_image is not None:
            metric_type = 'diffusion'

        if metric_type not in metrics_by_type:
            metrics_by_type[metric_type] = []
        metrics_by_type[metric_type].append(metric)

    for metric_type, type_metrics in metrics_by_type.items():
        filename: str = f'tensix_md_perf_metrics_{metric_type}.yaml'
        filepath: Path = Path(output_dir) / filename
        modeldump = yaml.dump([metric.model_dump(mode='yaml') for metric in type_metrics], indent=4)
        with open(filepath, 'w') as f:
            print(modeldump, file=f)
        logger.debug('Saved {} MD metrics for type {} to {}', len(type_metrics), metric_type, filepath)

    logger.debug('MD metrics saved to {}', output_dir)

