#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for run_ttsi_corr script.

Tests cover:
- Basic correlation workflow
- Argument parsing
- Output file generation
- Workload configuration handling
- Error handling
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.run_ttsi_corr import main as run_ttsi_corr
from tools.ttsi_corr.workload_processor import get_workload_module_config


class TestGetWorkloadModuleConfig:
    """Tests for workload configuration extraction."""

    def test_get_config_with_corr_instance(self):
        """Test extracting config with mapped instance."""
        from tools.workloads import WorkloadConfig, WorkloadsFile

        workload = WorkloadConfig(
            api='TTSIM',
            name='resnet50',
            basedir='workloads',
            module='BasicResNet@basicresnet.py',
            params={'layers': 50},
            instances={
                'default': {'bs': 32},
                'rn50_224x224': {'bs': 32, 'layers': 50}  # Instance name from MODEL_NAME_TO_WL_CONFIG
            }
        )
        workloads_file = WorkloadsFile(workloads=[workload])

        # Use valid model name from MODEL_NAME_TO_WL_CONFIG mapping
        config = get_workload_module_config(
            'resnet50', 'ResNet-50 (224x224)', 32, workloads_file
        )

        assert config is not None
        assert 'instance_config' in config
        assert config['instance_config']['bs'] == 32
        assert config['instance_config']['layers'] == 50
        assert config['module'] == 'BasicResNet@basicresnet.py'

    def test_get_config_missing_corr_instance(self):
        """Test that None is returned when mapped instance is missing in workload file."""
        from tools.workloads import WorkloadConfig, WorkloadsFile

        workload = WorkloadConfig(
            api='TTSIM',
            name='resnet50',
            basedir='workloads',
            module='resnet.py',
            params=None,
            instances={
                'default': {'bs': 32}
                # No 'rn50_224x224' instance (mapped from 'ResNet-50 (224x224)')
            }
        )
        workloads_file = WorkloadsFile(workloads=[workload])

        # Use valid model name from MODEL_NAME_TO_WL_CONFIG mapping
        config = get_workload_module_config(
            'resnet50', 'ResNet-50 (224x224)', 32, workloads_file
        )

        # When workload is found in file but lacks mapped instance, returns None
        assert config is None

    def test_get_config_workload_not_in_file(self):
        """Test that None is returned when workload not in config file."""
        from tools.workloads import WorkloadConfig, WorkloadsFile

        workload = WorkloadConfig(
            api='TTSIM',
            name='bert',
            basedir='workloads',
            module='bert.py',
            params=None,
            instances={'bert_large': {'bs': 1}}
        )
        workloads_file = WorkloadsFile(workloads=[workload])

        # Looking for 'resnet50' but only 'bert' is defined - should return None
        # Also 'resnet50' is in MODEL_NAME_TO_WL_CONFIG but 'bert' workload file doesn't have it
        config = get_workload_module_config(
            'resnet50', 'ResNet-50 (224x224)', 32, workloads_file
        )

        # Should return None when workload not found in file
        assert config is None

    def test_get_config_merges_params(self):
        """Test that params are merged with instance config."""
        from tools.workloads import WorkloadConfig, WorkloadsFile

        workload = WorkloadConfig(
            api='TTSIM',
            name='resnet50',
            basedir='workloads',
            module='resnet.py',
            params={'layers': 50, 'width': 1.0},  # Params at workload level
            instances={
                'rn50_224x224': {'bs': 32}  # Instance-specific param (mapped from MODEL_NAME_TO_WL_CONFIG)
            }
        )
        workloads_file = WorkloadsFile(workloads=[workload])

        # Use valid model name from MODEL_NAME_TO_WL_CONFIG mapping
        config = get_workload_module_config(
            'resnet50', 'ResNet-50 (224x224)', 32, workloads_file
        )

        assert config is not None
        assert 'instance_config' in config
        assert config['instance_config']['layers'] == 50
        assert config['instance_config']['width'] == 1.0
        assert config['instance_config']['bs'] == 32
        assert config['module'] == 'resnet.py'


class TestCorrelationBasic:
    """Basic correlation functionality tests."""

    @patch('tools.run_ttsi_corr.load_metrics_from_sources')
    @patch('tools.run_ttsi_corr.load_workload_configs')
    @patch('tools.run_ttsi_corr.run_polaris_simulation')
    def test_minimal_run(self, mock_polaris, mock_load_wl, mock_load_metrics, tmp_path):
        """Test minimal successful run with mocked dependencies."""
        # Mock data directory loading (no metrics)
        mock_load_metrics.return_value = []

        # Mock workloads file
        from tools.workloads import WorkloadConfig, WorkloadsFile
        mock_wl = WorkloadsFile(workloads=[
            WorkloadConfig(
                api='TTSIM',
                name='test',
                basedir='workloads',
                module='test.py',
                params=None,
                instances={'corr': {'bs': 1}}
            )
        ])
        mock_load_wl.return_value = (mock_wl, None)  # Returns tuple (workloads_file, workload_filter)

        # Mock polaris run
        mock_polaris.return_value = 0

        output_dir = tmp_path / 'output'
        base_dir = tmp_path / 'data'
        tag_dir = base_dir / '15oct25'
        tag_dir.mkdir(parents=True)

        # Note: This will skip workload processing since no metrics match
        res = run_ttsi_corr([
            'run_ttsi_corr',  # argv[0]
            '--input-dir', str(base_dir),
            '--tag', '15oct25',
            '--output-dir', str(output_dir),
            '--workloads-config', 'config/ttsi_correlation_workloads.yaml',  # Will be mocked
            '--arch-config', 'config/tt_wh.yaml'
        ])

        # Should return error when no matching workloads (updated behavior after refactoring)
        assert res != 0


class TestCorrelationOutputs:
    """Tests for output file generation."""

    def test_output_directory_created(self, tmp_path):
        """Test that output directory is created."""
        # This is a lightweight test without full execution
        _ = tmp_path / 'nonexistent' / 'nested' / 'output'

        # The function should create the directory structure
        # (Testing this requires partial execution or mocking)
        # TODO: Implement this
        pass  # Skip for now, requires integration test

    @pytest.mark.integration
    def test_expected_outputs_generated(self, tmp_path):
        """Test that expected output files are generated (integration test)."""
        # This would be a full integration test
        # Requires actual data files and configuration
        # TODO: Implement this
        pytest.skip('Requires full test environment with data files')


class TestCorrelationArguments:
    """Tests for command-line argument handling."""

    def test_missing_required_args(self):
        """Test that script runs with default arguments."""
        # Script now has default values for --input-dir and --tag, so it should run
        # (though it may not find data, it shouldn't raise SystemExit from argparse)
        # This test verifies the defaults are properly set
        result = run_ttsi_corr(['run_ttsi_corr'])
        # Should either succeed or fail gracefully (not with SystemExit from argparse)
        assert result in (0, 1)  # 0 for success, 1 for expected failures

    @patch('tools.run_ttsi_corr.load_metrics_from_sources')
    @patch('tools.run_ttsi_corr.load_workload_configs')
    def test_data_dir_argument(self, mock_load_wl, mock_load_metrics, tmp_path):
        """Test --input-dir and --tag arguments."""
        mock_load_metrics.return_value = []

        from tools.workloads import WorkloadConfig, WorkloadsFile
        mock_load_wl.return_value = (WorkloadsFile(workloads=[
            WorkloadConfig(
                api='TTSIM',
                name='test',
                basedir='workloads',
                module='test.py',
                params=None,
                instances={'corr': {'bs': 1}}
            )
        ]), None)  # Returns tuple (workloads_file, workload_filter)

        base_dir = tmp_path / 'data'
        tag_dir = base_dir / '15oct25'
        tag_dir.mkdir(parents=True)
        output_dir = tmp_path / 'output'

        # Should attempt to load from input-dir / tag
        with patch('tools.run_ttsi_corr.run_polaris_simulation') as mock_polaris:
            mock_polaris.return_value = 0
            run_ttsi_corr([
                'run_ttsi_corr',  # argv[0]
                '--input-dir', str(base_dir),
                '--tag', '15oct25',
                '--output-dir', str(output_dir),
                '--workloads-config', 'config/test.yaml',
                '--arch-config', 'config/tt_wh.yaml'
            ])

            # Verify input-dir/tag was used
            mock_load_metrics.assert_called()
            call_path = Path(mock_load_metrics.call_args[0][0])
            assert call_path == tag_dir


class TestCorrelationErrorHandling:
    """Tests for error handling."""

    @patch('tools.run_ttsi_corr.load_workload_configs')
    def test_invalid_workloads_config(self, mock_load, tmp_path):
        """Test handling of invalid workloads configuration."""
        mock_load.side_effect = FileNotFoundError('Config not found')

        output_dir = tmp_path / 'output'

        res = run_ttsi_corr([
            'run_ttsi_corr',  # argv[0]
            '--input-dir', 'data',
            '--tag', '15oct25',
            '--output-dir', str(output_dir),
            '--workloads-config', 'nonexistent.yaml',
            '--arch-config', 'config/tt_wh.yaml'
        ])

        # Should handle error gracefully
        assert res != 0

    @patch('tools.run_ttsi_corr.load_metrics_from_sources')
    def test_invalid_data_dir(self, mock_load, tmp_path):
        """Test handling of invalid data directory."""
        mock_load.side_effect = FileNotFoundError('Data dir not found')

        output_dir = tmp_path / 'output'

        res = run_ttsi_corr([
            'run_ttsi_corr',  # argv[0]
            '--input-dir', 'nonexistent',
            '--tag', '15oct25',
            '--output-dir', str(output_dir),
            '--workloads-config', 'config/test.yaml',
            '--arch-config', 'config/tt_wh.yaml'
        ])

        # Should handle error gracefully
        assert res != 0


@pytest.mark.integration
class TestCorrelationIntegration:
    """Integration tests for full correlation workflow."""

    def test_minimal_integration(self, tmp_path_factory):
        """
        Minimal integration test (original test).

        Note: This test requires a data directory to be present.
        """
        input_dir = 'data/metal/inf'
        tag = '15oct25'
        # Now requires --input-dir and --tag arguments
        res = run_ttsi_corr(['run_ttsi_corr', '--input-dir', input_dir, '--tag', tag])
        # Current implementation behavior
        assert res == 0, 'run_ttsi_corr failed'

    def test_full_correlation_workflow(self, tmp_path):
        """
        Full integration test with actual data.

        Requires:
        - Valid data directory with YAML metrics
        - Valid workloads configuration
        - Valid architecture configuration

        Example usage when enabled:
            output_dir = tmp_path / 'correlation_output'
            res = run_ttsi_corr([
                '--input-dir', 'data/metal/inf',
                '--tag', '15oct25',
                '--workloads-config', 'config/ttsi_correlation_workloads.yaml',
                '--arch-config', 'config/tt_wh.yaml',
                '--output-dir', str(output_dir)
            ])
            assert res == 0
            assert (output_dir / 'correlation_result.csv').exists()
            assert (output_dir / 'correlation_result.xlsx').exists()
            assert (output_dir / 'correlation_geomean.json').exists()
        """
        # TODO: Resolve whether to retain or not
        pytest.skip('Requires full test environment with data files')

