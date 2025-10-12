#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for parse_ttsi_perf_results script.

Tests cover:
- Basic functionality with default URL
- Custom input handling
- URL caching
- Error handling
- Output file generation
"""
from unittest.mock import MagicMock, patch

import pytest

from tools.ttsi_corr.ttsi_corr_utils import TTSI_REF_DEFAULT_TAG
from tools.parse_ttsi_perf_results import parse_ttsi_perf_results


class TestParseMetalTensixResultsBasic:
    """Basic functionality tests."""

    @pytest.mark.tools_secondary
    def test_parse_with_default_url(self, tmp_path):
        """Test parsing with default URL and no caching."""
        output_dir = tmp_path / 'output'
        res = parse_ttsi_perf_results([
            '--output-dir', str(output_dir),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--no-use-cache'
        ])
        assert res == 0, f'parse_ttsi_perf_results failed, got result: {res}'
        
        # Check that output directory with tag subdirectory was created
        assert (output_dir / TTSI_REF_DEFAULT_TAG).exists()

    @pytest.mark.tools_secondary
    def test_parse_output_files_created(self, tmp_path):
        """Test that expected output files are created."""
        output_dir = tmp_path / 'output'
        res = parse_ttsi_perf_results([
            '--output-dir', str(output_dir),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--no-use-cache'
        ])
        
        assert res == 0
        
        # Check for expected YAML files (at least one should be created)
        yaml_files = list((output_dir / TTSI_REF_DEFAULT_TAG).glob('tensix_md_perf_metrics_*.yaml'))
        assert len(yaml_files) > 0, 'No YAML files created'


class TestParseMetalTensixResultsInputs:
    """Tests for different input types."""

    @patch('tools.parse_ttsi_perf_results.extract_table_from_md_link')
    @patch('tools.parse_ttsi_perf_results.save_md_metrics')
    def test_parse_with_custom_url(self, mock_save, mock_extract, tmp_path):
        """Test parsing with custom input URL."""
        # Mock the extraction
        mock_metric = MagicMock()
        mock_metric.model = 'TestModel'
        mock_metric.batch = 32
        mock_extract.return_value = [mock_metric]
        
        output_dir = tmp_path / 'output'
        custom_url = 'https://custom.com/metrics.md'
        
        res = parse_ttsi_perf_results([
            '--input', custom_url,
            '--output-dir', str(output_dir),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--no-use-cache'
        ])
        
        assert res == 0
        mock_extract.assert_called_once()
        # Check that custom URL was used
        call_args = mock_extract.call_args
        assert call_args[0][0] == custom_url

    @patch('tools.parse_ttsi_perf_results.extract_table_from_md_link')
    def test_parse_with_use_cache(self, mock_extract, tmp_path):
        """Test that caching flag is passed correctly."""
        mock_extract.return_value = []
        
        output_dir = tmp_path / 'output'
        
        # Test with cache
        parse_ttsi_perf_results([
            '--output-dir', str(output_dir),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--use-cache'
        ])
        
        call_args = mock_extract.call_args
        assert call_args[1]['use_cache'] is True

    @patch('tools.parse_ttsi_perf_results.extract_table_from_md_link')
    def test_parse_with_no_use_cache(self, mock_extract, tmp_path):
        """Test that no-cache flag is passed correctly."""
        mock_extract.return_value = []
        
        output_dir = tmp_path / 'output'
        
        # Test without cache
        parse_ttsi_perf_results([
            '--output-dir', str(output_dir),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--no-use-cache'
        ])
        
        call_args = mock_extract.call_args
        assert call_args[1]['use_cache'] is False


class TestParseMetalTensixResultsErrorHandling:
    """Tests for error handling."""

    @patch('tools.parse_ttsi_perf_results.extract_table_from_md_link')
    def test_parse_no_metrics_found(self, mock_extract, tmp_path):
        """Test handling when no metrics are extracted."""
        mock_extract.return_value = []
        
        output_dir = tmp_path / 'output'
        
        res = parse_ttsi_perf_results([
            '--output-dir', str(output_dir),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--no-use-cache'
        ])
        
        # Should return error code
        assert res != 0

    @patch('tools.parse_ttsi_perf_results.extract_table_from_md_link')
    def test_parse_extraction_exception(self, mock_extract, tmp_path):
        """Test handling of extraction exceptions."""
        mock_extract.side_effect = ValueError('Invalid markdown format')
        
        output_dir = tmp_path / 'output'
        
        res = parse_ttsi_perf_results([
            '--output-dir', str(output_dir),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--no-use-cache'
        ])
        
        # Should return error code
        assert res != 0


class TestParseMetalTensixResultsArguments:
    """Tests for command-line argument parsing."""

    def test_missing_tag_argument(self):
        """Test that missing --tag argument raises an error."""
        with pytest.raises(SystemExit):
            parse_ttsi_perf_results(['--no-use-cache'])

    def test_invalid_tag_value(self):
        """Test that invalid tag value raises an error."""
        with pytest.raises(SystemExit):
            parse_ttsi_perf_results(['--tag', 'invalid_tag', '--no-use-cache'])

    def test_missing_output_dir(self):
        """Test that default output-dir is used when not specified."""
        # Should succeed with default output directory
        res = parse_ttsi_perf_results(['--tag', TTSI_REF_DEFAULT_TAG, '--no-use-cache'])
        assert res == 0

    @patch('tools.parse_ttsi_perf_results.extract_table_from_md_link')
    def test_all_arguments(self, mock_extract, tmp_path):
        """Test with all possible arguments."""
        mock_extract.return_value = []
        
        output_dir = tmp_path / 'output'
        custom_url = 'https://test.com/metrics.md'
        
        parse_ttsi_perf_results([
            '--input', custom_url,
            '--output-dir', str(output_dir),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--use-cache'
        ])
        
        # Should handle all arguments correctly
        assert mock_extract.called


@pytest.mark.integration
class TestParseMetalTensixResultsIntegration:
    """Integration tests."""

    @pytest.mark.tools_secondary
    def test_full_workflow_no_cache(self, tmp_path_factory):
        """
        Full integration test: parse default URL without caching.
        
        This is the original test, kept for backward compatibility.
        """
        tmpdir = tmp_path_factory.mktemp('parse_ttsi_perf_results')
        res = parse_ttsi_perf_results([
            '--output-dir', tmpdir.as_posix(),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--no-use-cache'
        ])
        assert res == 0, f'parse_ttsi_perf_results failed, got result: {res}'
        
        # Verify output structure
        tag_dir = tmpdir / TTSI_REF_DEFAULT_TAG
        assert tag_dir.exists()
        yaml_files = list(tag_dir.glob('*.yaml'))
        assert len(yaml_files) > 0, 'Expected YAML output files'

    @pytest.mark.tools_secondary
    @pytest.mark.slow
    def test_full_workflow_with_cache(self, tmp_path):
        """Test full workflow with caching (slow test)."""
        output_dir = tmp_path / 'output'
        
        # First run - populates cache
        res1 = parse_ttsi_perf_results([
            '--output-dir', str(output_dir),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--use-cache'
        ])
        assert res1 == 0
        
        # Second run - uses cache (should be faster)
        output_dir2 = tmp_path / 'output2'
        res2 = parse_ttsi_perf_results([
            '--output-dir', str(output_dir2),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--use-cache'
        ])
        assert res2 == 0
        
        # Both should produce same files
        files1 = sorted([f.name for f in (output_dir / TTSI_REF_DEFAULT_TAG).glob('*.yaml')])
        files2 = sorted([f.name for f in (output_dir2 / TTSI_REF_DEFAULT_TAG).glob('*.yaml')])
        assert files1 == files2
