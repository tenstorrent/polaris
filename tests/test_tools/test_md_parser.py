#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the markdown parser (tools/parsers/md_parser.py).

Tests cover:
- Pydantic model validation
- Column name mapping
- Numeric value parsing
- Cell value cleaning
- Table extraction from markdown
- YAML output generation
"""
from unittest.mock import patch

import pytest

from tools.parsers.md_parser import (MD_COLNAME_MAP, TensixMdPerfMetricModel, clean_cell_value,
                                     extract_table_from_md_link, get_md_colname_map, parse_numeric_md_value,
                                     save_md_metrics)


class TestTensixMdPerfMetricModel:
    """Tests for the Pydantic model."""

    def test_model_creation_llm(self):
        """Test creating a model for LLM metrics."""
        metric = TensixMdPerfMetricModel(
            model='Llama 3.1 8B',
            batch=32,
            hardware='n150 (Wormhole)',
            release='v0.55.0',
            tokens_per_sec_per_user=23.5,
            target_tokens_per_sec_per_user=26.0,
            ttft_ms=89.0
        )
        
        assert metric.model == 'Llama 3.1 8B'
        assert metric.batch == 32
        assert metric.hardware == 'n150 (Wormhole)'
        assert metric.tokens_per_sec_per_user == 23.5
        assert metric.gpu == 'Tensix'  # Default value
        assert metric.id == 'metal'  # Default value

    def test_model_creation_vision(self):
        """Test creating a model for vision metrics."""
        metric = TensixMdPerfMetricModel(
            model='ResNet-50',
            batch=32,
            hardware='n300',
            images_per_sec=1500.0,
            target_images_per_sec=1600.0
        )
        
        assert metric.model == 'ResNet-50'
        assert metric.images_per_sec == 1500.0

    def test_model_creation_detection(self):
        """Test creating a model for detection metrics."""
        metric = TensixMdPerfMetricModel(
            model='YOLOv8',
            batch=1,
            hardware='n150',
            fps=45.0,
            target_fps=50.0
        )
        
        assert metric.fps == 45.0

    def test_model_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            TensixMdPerfMetricModel(  # type: ignore[call-arg]
                model='Test Model'
                # Missing required fields: batch, hardware
            )

    def test_model_extra_forbid(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            TensixMdPerfMetricModel(  # type: ignore[call-arg]
                model='Test',
                batch=1,
                hardware='n150',
                unknown_field='value'  # This should be rejected
            )

    def test_get_perf_llm(self):
        """Test get_perf() for LLM metrics."""
        metric = TensixMdPerfMetricModel(
            model='Test',
            batch=1,
            hardware='n150',
            tokens_per_sec_per_user=25.0
        )
        assert metric.get_perf() == 25.0

    def test_get_perf_vision(self):
        """Test get_perf() for vision metrics."""
        metric = TensixMdPerfMetricModel(
            model='Test',
            batch=1,
            hardware='n150',
            images_per_sec=1500.0
        )
        assert metric.get_perf() == 1500.0

    def test_get_perf_diffusion(self):
        """Test get_perf() for diffusion metrics (sec_per_image)."""
        metric = TensixMdPerfMetricModel(
            model='Test',
            batch=1,
            hardware='n150',
            sec_per_image=2.0
        )
        assert metric.get_perf() == 0.5  # 1/2.0

    def test_get_target_perf(self):
        """Test get_target_perf()."""
        metric = TensixMdPerfMetricModel(
            model='Test',
            batch=1,
            hardware='n150',
            tokens_per_sec_per_user=25.0,
            target_tokens_per_sec_per_user=30.0
        )
        assert metric.get_target_perf() == 30.0

    def test_get_metric_name(self):
        """Test get_metric_name()."""
        metric = TensixMdPerfMetricModel(
            model='Test',
            batch=1,
            hardware='n150',
            fps=45.0
        )
        assert metric.get_metric_name() == 'fps'


class TestColumnMapping:
    """Tests for column name mapping."""

    def test_get_md_colname_map_exact_match(self):
        """Test exact matches in MD_COLNAME_MAP."""
        assert get_md_colname_map('TTFT (ms)') == 'ttft_ms'
        assert get_md_colname_map('t/s/u') == 'tokens_per_sec_per_user'
        assert get_md_colname_map('fps') == 'fps'

    def test_get_md_colname_map_case_insensitive(self):
        """Test case insensitivity."""
        assert get_md_colname_map('FPS') == 'fps'
        assert get_md_colname_map('Model') == 'model'

    def test_get_md_colname_map_with_spaces(self):
        """Test handling of spaces."""
        assert get_md_colname_map('  fps  ') == 'fps'

    def test_get_md_colname_map_fallback(self):
        """Test fallback for unknown column names."""
        result = get_md_colname_map('Some New Column')
        assert result == 'some_new_column'  # Spaces to underscores, lowercase

    def test_md_colname_map_completeness(self):
        """Test that all important column types are mapped."""
        assert 'ttft (ms)' in MD_COLNAME_MAP
        assert 't/s/u' in MD_COLNAME_MAP
        assert 'model' in MD_COLNAME_MAP
        assert 'batch' in MD_COLNAME_MAP
        assert 'hardware' in MD_COLNAME_MAP
        assert 'fps' in MD_COLNAME_MAP


class TestNumericParsing:
    """Tests for numeric value parsing."""

    def test_parse_numeric_md_value_integer(self):
        """Test parsing integer values."""
        assert parse_numeric_md_value('32') == 32.0
        assert parse_numeric_md_value('1500') == 1500.0

    def test_parse_numeric_md_value_float(self):
        """Test parsing float values."""
        assert parse_numeric_md_value('23.5') == 23.5
        assert parse_numeric_md_value('0.95') == 0.95

    def test_parse_numeric_md_value_with_comma(self):
        """Test parsing values with comma separators."""
        assert parse_numeric_md_value('1,500') == 1500.0
        assert parse_numeric_md_value('10,000.5') == 10000.5

    def test_parse_numeric_md_value_with_asterisk(self):
        """Test parsing values with asterisks (footnote markers)."""
        assert parse_numeric_md_value('23.5*') == 23.5
        assert parse_numeric_md_value('*32') == 32.0

    def test_parse_numeric_md_value_empty(self):
        """Test parsing empty or dash values."""
        assert parse_numeric_md_value('') is None
        assert parse_numeric_md_value('-') is None
        assert parse_numeric_md_value('   ') is None

    def test_parse_numeric_md_value_invalid(self):
        """Test parsing invalid values."""
        assert parse_numeric_md_value('N/A') is None
        assert parse_numeric_md_value('TBD') is None
        assert parse_numeric_md_value('abc') is None


class TestCellCleaning:
    """Tests for cell value cleaning."""

    def test_clean_cell_value_simple(self):
        """Test cleaning simple values."""
        assert clean_cell_value('  test  ') == 'test'
        assert clean_cell_value('value') == 'value'

    def test_clean_cell_value_markdown_link(self):
        """Test removing markdown links but keeping text."""
        assert clean_cell_value('[text](url)') == 'text'
        assert clean_cell_value('[Llama 3.1](https://example.com)') == 'Llama 3.1'

    def test_clean_cell_value_html_tags(self):
        """Test removing HTML tags."""
        assert clean_cell_value('<b>bold</b>') == 'bold'
        assert clean_cell_value('<em>emphasis</em>') == 'emphasis'

    def test_clean_cell_value_complex(self):
        """Test cleaning complex values."""
        result = clean_cell_value('  [text](url) <b>bold</b>  ')
        assert result == 'text bold'


class TestTableExtraction:
    """Tests for table extraction from markdown."""

    @patch('tools.parsers.md_parser.read_from_url')
    def test_extract_table_from_md_link_basic(self, mock_read):
        """Test extracting a basic table."""
        mock_md = """
| Model | Batch | Hardware | FPS |
|-------|-------|----------|-----|
| YOLOv8 | 1 | n150 | 45.0 |
| ResNet-50 | 32 | n300 | 1500.0 |
"""
        mock_read.return_value = mock_md
        
        metrics = extract_table_from_md_link('http://test.com/test.md')
        
        assert len(metrics) == 2
        assert metrics[0].model == 'YOLOv8'
        assert metrics[0].batch == 1
        assert metrics[0].hardware == 'n150'
        assert metrics[0].fps == 45.0

    @patch('tools.parsers.md_parser.read_from_url')
    def test_extract_table_no_model_column(self, mock_read):
        """Test that tables without 'model' column are skipped."""
        mock_md = """
| Version | Date | Release |
|---------|------|---------|
| v1.0 | 2025-01 | First |
"""
        mock_read.return_value = mock_md
        
        with pytest.raises(ValueError, match='No valid table data'):
            extract_table_from_md_link('http://test.com/test.md')

    @patch('tools.parsers.md_parser.read_from_url')
    def test_extract_table_invalid_row(self, mock_read):
        """Test handling of invalid rows."""
        mock_md = """
| Model | Batch | Hardware | FPS |
|-------|-------|----------|-----|
| Valid | 32 | n150 | 45.0 |
| Invalid | notanumber | n150 | 45.0 |
"""
        mock_read.return_value = mock_md
        
        metrics = extract_table_from_md_link('http://test.com/test.md')
        
        # Should skip invalid row
        assert len(metrics) == 1
        assert metrics[0].model == 'Valid'


class TestSaveMetrics:
    """Tests for saving metrics to YAML files."""

    def test_save_md_metrics_llm(self, tmp_path):
        """Test saving LLM metrics."""
        metrics = [
            TensixMdPerfMetricModel(
                model='Llama 3.1 8B',
                batch=32,
                hardware='n150',
                tokens_per_sec_per_user=23.5
            ),
            TensixMdPerfMetricModel(
                model='Llama 3.2 1B',
                batch=32,
                hardware='n300',
                tokens_per_sec_per_user=45.0
            )
        ]
        
        save_md_metrics(metrics, tmp_path)
        
        # Check that LLM file was created
        llm_file = tmp_path / 'tensix_md_perf_metrics_llm.yaml'
        assert llm_file.exists()
        
        # Check file content
        content = llm_file.read_text()
        assert 'Llama 3.1 8B' in content
        assert 'tokens_per_sec_per_user' in content

    def test_save_md_metrics_multiple_types(self, tmp_path):
        """Test saving metrics of multiple types."""
        metrics = [
            TensixMdPerfMetricModel(
                model='Llama',
                batch=32,
                hardware='n150',
                tokens_per_sec_per_user=23.5
            ),
            TensixMdPerfMetricModel(
                model='ResNet',
                batch=32,
                hardware='n150',
                images_per_sec=1500.0
            ),
            TensixMdPerfMetricModel(
                model='YOLO',
                batch=1,
                hardware='n150',
                fps=45.0
            )
        ]
        
        save_md_metrics(metrics, tmp_path)
        
        # Check that all expected files were created
        assert (tmp_path / 'tensix_md_perf_metrics_llm.yaml').exists()
        assert (tmp_path / 'tensix_md_perf_metrics_vision.yaml').exists()
        assert (tmp_path / 'tensix_md_perf_metrics_detection.yaml').exists()


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the parser."""

    @patch('tools.parsers.md_parser.read_from_url')
    def test_full_parse_workflow(self, mock_read, tmp_path):
        """Test complete workflow from markdown to YAML files."""
        # Realistic markdown content
        mock_md = """
# Model Performance

## LLM Models

| Model | Batch | Hardware | T/S/U | Target T/S/U | TTFT (ms) |
|-------|-------|----------|-------|--------------|-----------|
| Llama 3.1 8B | 32 | n150 (Wormhole) | 23.5 | 26.0 | 89.0 |
| Llama 3.2 1B | 32 | n300 (Wormhole) | 45.0 | 48.0 | 45.0 |

## Vision Models

| Model | Batch | Hardware | Images/sec | Target Images/sec |
|-------|-------|----------|------------|-------------------|
| ResNet-50 | 32 | n150 | 1500.0 | 1600.0 |
"""
        mock_read.return_value = mock_md
        
        # Extract and save
        metrics = extract_table_from_md_link('http://test.com')
        save_md_metrics(metrics, tmp_path)
        
        # Verify results
        assert len(metrics) == 3
        
        llm_file = tmp_path / 'tensix_md_perf_metrics_llm.yaml'
        vision_file = tmp_path / 'tensix_md_perf_metrics_vision.yaml'
        
        assert llm_file.exists()
        assert vision_file.exists()
        
        # Check content
        llm_content = llm_file.read_text()
        assert 'Llama 3.1 8B' in llm_content
        assert 'Llama 3.2 1B' in llm_content
        
        vision_content = vision_file.read_text()
        assert 'ResNet-50' in vision_content

