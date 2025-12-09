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
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest
import yaml

from tools.parse_ttsi_perf_results import parse_ttsi_perf_results, save_metadata
from tools.parsers.md_parser import TensixMdPerfMetricModel
from tools.ttsi_corr.ttsi_corr_utils import TTSI_REF_DEFAULT_TAG


@pytest.fixture(autouse=True)
def check_test_does_not_modify_files():
    """
    Check that no files in the repository are modified by any test.

    Currently, this fixture has been added to this test file only, but it can be moved
    to conftest.py if desired to apply to all tests. However, doing so may slow down
    the test suite as it runs git commands after each test.

    This fixture was added after observing that a test was modifying
    reference data files, causing subsequent tests to fail. This check helps catch such
    issues in the future.
    """
    # Run this check around each test; no specific setup needed prior to the test
    yield
    # Teardown: Check for uncommitted changes in the git repository
    # Flag an error if any files were modified
    if subprocess.run(['git', 'diff', '--exit-code', 'data']).returncode != 0:
        pytest.fail("The test modified the files in the repository.")


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


class TestSaveMetadata:
    """Tests for metadata file handling."""

    def test_save_metadata_creates_new_file(self, tmp_path):
        """Test that save_metadata creates a new metadata file."""
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        save_metadata(
            output_dir=output_dir,
            tag='test_tag',
            data_source='md',
            input_url='https://test.com/data.md',
            use_cache=True
        )

        metadata_file = output_dir / '_metadata.yaml'
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = yaml.safe_load(f)

        assert metadata['tag'] == 'test_tag'
        assert metadata['data_source'] == 'md'
        assert metadata['input_url'] == 'https://test.com/data.md'
        assert metadata['use_cache'] is True
        assert 'parsed_date' in metadata

    def test_save_metadata_preserves_file_when_only_timestamp_differs(self, tmp_path):
        """Test that metadata file is not overwritten when only timestamp differs."""
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # First save
        save_metadata(
            output_dir=output_dir,
            tag='test_tag',
            data_source='md',
            input_url='https://test.com/data.md',
            use_cache=True
        )

        metadata_file = output_dir / '_metadata.yaml'
        first_mtime = metadata_file.stat().st_mtime

        with open(metadata_file) as f:
            first_metadata = yaml.safe_load(f)
        first_timestamp = first_metadata['parsed_date']

        # Wait a bit to ensure timestamp would be different
        time.sleep(0.1)

        # Second save with same significant fields (but different current time)
        save_metadata(
            output_dir=output_dir,
            tag='test_tag',
            data_source='md',
            input_url='https://test.com/data.md',
            use_cache=True
        )

        # File modification time should be unchanged
        second_mtime = metadata_file.stat().st_mtime
        assert first_mtime == second_mtime, 'File should not have been modified'

        # Timestamp in file should be preserved
        with open(metadata_file) as f:
            second_metadata = yaml.safe_load(f)
        assert second_metadata['parsed_date'] == first_timestamp

    def test_save_metadata_preserves_file_when_only_use_cache_differs(self, tmp_path):
        """Test that metadata file is not overwritten when only use_cache differs."""
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # First save with use_cache=True
        save_metadata(
            output_dir=output_dir,
            tag='test_tag',
            data_source='md',
            input_url='https://test.com/data.md',
            use_cache=True
        )

        metadata_file = output_dir / '_metadata.yaml'
        first_mtime = metadata_file.stat().st_mtime

        with open(metadata_file) as f:
            first_metadata = yaml.safe_load(f)
        first_timestamp = first_metadata['parsed_date']
        first_use_cache = first_metadata['use_cache']

        time.sleep(0.1)

        # Second save with use_cache=False (only use_cache changed)
        save_metadata(
            output_dir=output_dir,
            tag='test_tag',
            data_source='md',
            input_url='https://test.com/data.md',
            use_cache=False
        )

        # File modification time should be unchanged
        second_mtime = metadata_file.stat().st_mtime
        assert first_mtime == second_mtime, 'File should not have been modified when only use_cache changes'

        # Original values should be preserved
        with open(metadata_file) as f:
            second_metadata = yaml.safe_load(f)
        assert second_metadata['parsed_date'] == first_timestamp
        assert second_metadata['use_cache'] == first_use_cache

    def test_save_metadata_updates_when_data_source_changes(self, tmp_path):
        """Test that metadata file is updated when data_source changes."""
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # First save
        save_metadata(
            output_dir=output_dir,
            tag='test_tag',
            data_source='md',
            input_url='https://test.com/data.md',
            use_cache=True
        )

        metadata_file = output_dir / '_metadata.yaml'
        first_mtime = metadata_file.stat().st_mtime

        time.sleep(0.1)

        # Second save with different data_source
        save_metadata(
            output_dir=output_dir,
            tag='test_tag',
            data_source='html',  # Changed from 'md'
            input_url='https://test.com/data.md',
            use_cache=True
        )

        # File should have been updated
        second_mtime = metadata_file.stat().st_mtime
        assert second_mtime > first_mtime, 'File should have been modified'

        # Data source should be updated
        with open(metadata_file) as f:
            updated_metadata = yaml.safe_load(f)
        assert updated_metadata['data_source'] == 'html'

    def test_save_metadata_updates_when_url_changes(self, tmp_path):
        """Test that metadata file is updated when URL changes."""
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # First save
        save_metadata(
            output_dir=output_dir,
            tag='test_tag',
            data_source='md',
            input_url='https://test.com/data1.md',
            use_cache=True
        )

        metadata_file = output_dir / '_metadata.yaml'
        first_mtime = metadata_file.stat().st_mtime

        time.sleep(0.1)

        # Second save with different URL
        save_metadata(
            output_dir=output_dir,
            tag='test_tag',
            data_source='md',
            input_url='https://test.com/data2.md',  # Changed URL
            use_cache=True
        )

        # File should have been updated
        second_mtime = metadata_file.stat().st_mtime
        assert second_mtime > first_mtime, 'File should have been modified'

        # URL should be updated
        with open(metadata_file) as f:
            updated_metadata = yaml.safe_load(f)
        assert updated_metadata['input_url'] == 'https://test.com/data2.md'

    def test_save_metadata_handles_missing_fields_in_existing(self, tmp_path):
        """Test graceful handling when existing file has missing fields."""
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # Create metadata file with only significant fields (no parsed_date or use_cache)
        metadata_file = output_dir / '_metadata.yaml'
        with open(metadata_file, 'w') as f:
            yaml.dump({
                'tag': 'test_tag',
                'data_source': 'md',
                'input_url': 'https://test.com/data.md',
            }, f)

        first_mtime = metadata_file.stat().st_mtime
        time.sleep(0.1)

        # Should not update since significant fields are the same
        save_metadata(
            output_dir=output_dir,
            tag='test_tag',
            data_source='md',
            input_url='https://test.com/data.md',
            use_cache=True
        )

        second_mtime = metadata_file.stat().st_mtime
        assert first_mtime == second_mtime, 'File should not have been modified'

    def test_save_metadata_updates_when_tag_changes(self, tmp_path):
        """Test that metadata file is updated when tag changes."""
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # First save
        save_metadata(
            output_dir=output_dir,
            tag='tag1',
            data_source='md',
            input_url='https://test.com/data.md',
            use_cache=True
        )

        metadata_file = output_dir / '_metadata.yaml'
        first_mtime = metadata_file.stat().st_mtime

        time.sleep(0.1)

        # Second save with different tag
        save_metadata(
            output_dir=output_dir,
            tag='tag2',  # Changed tag
            data_source='md',
            input_url='https://test.com/data.md',
            use_cache=True
        )

        # File should have been updated
        second_mtime = metadata_file.stat().st_mtime
        assert second_mtime > first_mtime, 'File should have been modified'

        # Tag should be updated
        with open(metadata_file) as f:
            updated_metadata = yaml.safe_load(f)
        assert updated_metadata['tag'] == 'tag2'

    def test_save_metadata_handles_corrupt_existing_file(self, tmp_path):
        """Test handling when existing metadata file is corrupt."""
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # Create corrupt metadata file
        metadata_file = output_dir / '_metadata.yaml'
        with open(metadata_file, 'w') as f:
            f.write('this is not valid yaml: [}')

        # Should handle gracefully and overwrite
        save_metadata(
            output_dir=output_dir,
            tag='test_tag',
            data_source='md',
            input_url='https://test.com/data.md',
            use_cache=True
        )

        # File should now be valid
        with open(metadata_file) as f:
            metadata = yaml.safe_load(f)
        assert metadata['tag'] == 'test_tag'


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

    @patch('tools.parse_ttsi_perf_results.extract_table_from_md_link')
    def test_successful_parse_creates_output_dir(self, mock_extract, tmp_path):
        """Test that successful parsing creates the expected output directory structure."""
        # Mock the extraction to return minimal valid data to avoid modifying actual reference data
        # Any modification to reference data would cause test failures in subsequent tests
        # Return non-empty list to avoid the "No metrics found" error (return code 2)
        mock_extract.return_value = [TensixMdPerfMetricModel(
            model='test_model',
            batch=1,
            hardware='test_hw',
            tokens_per_sec=100.0
        )]

        # Use temporary directory to avoid modifying actual reference data
        output_dir = tmp_path / 'output'
        res = parse_ttsi_perf_results([
            '--output-dir', str(output_dir),
            '--tag', TTSI_REF_DEFAULT_TAG,
            '--no-use-cache'
        ])
        assert res == 0

        # Verify the output directory was created
        assert (output_dir / TTSI_REF_DEFAULT_TAG).exists()

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
