#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the workloads module (tools/workloads.py).

Tests cover:
- WorkloadConfig Pydantic model validation
- WorkloadsFile validation
- Module format validation
- File loading utilities
"""
from pathlib import Path

import pytest

from tools.workloads import WorkloadConfig, WorkloadsFile, load_workloads_file


class TestWorkloadConfig:
    """Tests for WorkloadConfig Pydantic model."""

    def test_valid_workload_python_with_class(self):
        """Test valid workload with Python file and class name."""
        config = WorkloadConfig(
            api='TTSIM',
            name='resnet50',
            basedir='workloads',
            module='BasicResNet@basicresnet.py',
            params=None,
            instances={'default': {'bs': 32}}
        )
        
        assert config.api == 'TTSIM'
        assert config.name == 'resnet50'
        assert config.module == 'BasicResNet@basicresnet.py'
        assert 'default' in config.instances

    def test_valid_workload_python_without_class(self):
        """Test valid workload with Python file only."""
        config = WorkloadConfig(
            api='TTSIM',
            name='test',
            basedir='workloads',
            module='test.py',
            params=None,
            instances={'default': {'bs': 1}}
        )
        
        assert config.module == 'test.py'

    def test_valid_workload_onnx(self):
        """Test valid workload with ONNX file."""
        config = WorkloadConfig(
            api='ONNX',
            name='model',
            basedir='models',
            module='model.onnx',
            params=None,
            instances={'default': {}}
        )
        
        assert config.module == 'model.onnx'

    def test_valid_workload_html(self):
        """Test valid workload with HTML file."""
        config = WorkloadConfig(
            api='HTML',
            name='graph',
            basedir='graphs',
            module='graph.html',
            params=None,
            instances={'default': {}}
        )
        
        assert config.module == 'graph.html'

    def test_invalid_module_extension(self):
        """Test that invalid file extensions are rejected."""
        with pytest.raises(ValueError, match='Unsupported model file extension'):
            WorkloadConfig(
                api='TTSIM',
                name='test',
                basedir='workloads',
                module='test.txt',  # Invalid extension
                params=None,
                instances={'default': {}}
            )

    def test_invalid_module_format_onnx_with_at(self):
        """Test that ONNX files cannot use ClassName@file format."""
        with pytest.raises(ValueError, match='must use "file.onnx" format'):
            WorkloadConfig(
                api='ONNX',
                name='model',
                basedir='models',
                module='ClassName@model.onnx',  # Invalid for .onnx
                params=None,
                instances={'default': {}}
            )

    def test_invalid_module_format_html_with_at(self):
        """Test that HTML files cannot use ClassName@file format."""
        with pytest.raises(ValueError, match='must use "file.html" format'):
            WorkloadConfig(
                api='HTML',
                name='graph',
                basedir='graphs',
                module='ClassName@graph.html',  # Invalid for .html
                params=None,
                instances={'default': {}}
            )

    def test_invalid_module_empty_class(self):
        """Test that empty class name is rejected."""
        with pytest.raises(ValueError, match='Class name cannot be empty'):
            WorkloadConfig(
                api='TTSIM',
                name='test',
                basedir='workloads',
                module='@test.py',  # Empty class name
                params=None,
                instances={'default': {}}
            )

    def test_invalid_module_empty_filename(self):
        """Test that empty filename is rejected."""
        with pytest.raises(ValueError, match='Filename cannot be empty'):
            WorkloadConfig(
                api='TTSIM',
                name='test',
                basedir='workloads',
                module='ClassName@',  # Empty filename
                params=None,
                instances={'default': {}}
            )

    def test_invalid_module_no_extension(self):
        """Test that files without extension are rejected."""
        with pytest.raises(ValueError, match='must have an extension'):
            WorkloadConfig(
                api='TTSIM',
                name='test',
                basedir='workloads',
                module='test',  # No extension
                params=None,
                instances={'default': {}}
            )

    def test_invalid_module_multiple_at(self):
        """Test that multiple @ symbols are rejected."""
        with pytest.raises(ValueError, match='must have exactly one @'):
            WorkloadConfig(
                api='TTSIM',
                name='test',
                basedir='workloads',
                module='Class@Name@file.py',  # Multiple @
                params=None,
                instances={'default': {}}
            )

    def test_workload_with_params(self):
        """Test workload with optional params."""
        config = WorkloadConfig(
            api='TTSIM',
            name='resnet',
            basedir='workloads',
            module='resnet.py',
            params={'layers': 50},
            instances={'default': {'bs': 32}}
        )
        
        assert config.params == {'layers': 50}

    def test_workload_without_params(self):
        """Test workload without params."""
        config = WorkloadConfig(
            api='TTSIM',
            name='test',
            basedir='workloads',
            module='test.py',
            params=None,
            instances={'default': {}}
        )
        
        assert config.params is None

    def test_workload_multiple_instances(self):
        """Test workload with multiple instances."""
        config = WorkloadConfig(
            api='TTSIM',
            name='resnet',
            basedir='workloads',
            module='resnet.py',
            params=None,
            instances={
                'small': {'bs': 1, 'layers': 18},
                'medium': {'bs': 32, 'layers': 50},
                'large': {'bs': 64, 'layers': 101}
            }
        )
        
        assert len(config.instances) == 3
        assert config.instances['small']['bs'] == 1
        assert config.instances['large']['layers'] == 101

    def test_workload_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            WorkloadConfig(  # type: ignore[call-arg]
                api='TTSIM',
                name='test',
                basedir='workloads',
                module='test.py',
                params=None,
                instances={'default': {}},
                unknown_field='value'  # This should be rejected
            )


class TestWorkloadsFile:
    """Tests for WorkloadsFile Pydantic model."""

    def test_valid_workloads_file(self):
        """Test valid workloads file with multiple workloads."""
        workloads_file = WorkloadsFile(
            workloads=[
                WorkloadConfig(
                    api='TTSIM',
                    name='resnet50',
                    basedir='workloads',
                    module='BasicResNet@basicresnet.py',
                    params=None,
                    instances={'default': {'bs': 32}}
                ),
                WorkloadConfig(
                    api='TTSIM',
                    name='bert',
                    basedir='workloads',
                    module='bert.py',
                    params=None,
                    instances={'default': {'bs': 1}}
                )
            ]
        )
        
        assert len(workloads_file.workloads) == 2
        assert workloads_file.workloads[0].name == 'resnet50'
        assert workloads_file.workloads[1].name == 'bert'

    def test_empty_workloads_file(self):
        """Test that empty workloads list is rejected."""
        with pytest.raises(ValueError, match='at least one workload'):
            WorkloadsFile(workloads=[])

    def test_duplicate_workload_names(self):
        """Test that duplicate workload names are rejected."""
        with pytest.raises(ValueError, match='Duplicate workload name'):
            WorkloadsFile(
                workloads=[
                    WorkloadConfig(
                        api='TTSIM',
                        name='resnet',
                        basedir='workloads',
                        module='resnet.py',
                        params=None,
                        instances={'default': {}}
                    ),
                    WorkloadConfig(
                        api='TTSIM',
                        name='resnet',  # Duplicate
                        basedir='workloads',
                        module='resnet2.py',
                        params=None,
                        instances={'default': {}}
                    )
                ]
            )


class TestLoadWorkloadsFile:
    """Tests for load_workloads_file utility."""

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid YAML file."""
        yaml_content = """
workloads:
  - api: TTSIM
    name: resnet50
    basedir: workloads
    module: BasicResNet@basicresnet.py
    instances:
      default:
        bs: 32
        layers: 50
  - api: TTSIM
    name: bert
    basedir: workloads
    module: bert.py
    instances:
      default:
        bs: 1
"""
        yaml_file = tmp_path / 'workloads.yaml'
        yaml_file.write_text(yaml_content)
        
        workloads_file = load_workloads_file(yaml_file)
        
        assert len(workloads_file.workloads) == 2
        assert workloads_file.workloads[0].name == 'resnet50'
        assert workloads_file.workloads[1].name == 'bert'

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_workloads_file(Path('/nonexistent/file.yaml'))

    def test_load_invalid_yaml_structure(self, tmp_path):
        """Test loading YAML with invalid structure."""
        yaml_content = """
not_workloads:  # Wrong key
  - api: TTSIM
    name: test
"""
        yaml_file = tmp_path / 'invalid.yaml'
        yaml_file.write_text(yaml_content)
        
        with pytest.raises(Exception):  # Pydantic ValidationError
            load_workloads_file(yaml_file)

    def test_load_yaml_with_invalid_module(self, tmp_path):
        """Test loading YAML with invalid module format."""
        yaml_content = """
workloads:
  - api: TTSIM
    name: test
    basedir: workloads
    module: test.txt  # Invalid extension
    instances:
      default: {}
"""
        yaml_file = tmp_path / 'invalid_module.yaml'
        yaml_file.write_text(yaml_content)
        
        with pytest.raises(ValueError, match='Unsupported model file extension'):
            load_workloads_file(yaml_file)

    def test_load_yaml_with_path_object(self, tmp_path):
        """Test that Path objects are accepted."""
        yaml_content = """
workloads:
  - api: TTSIM
    name: test
    basedir: workloads
    module: test.py
    instances:
      default: {}
"""
        yaml_file = tmp_path / 'workloads.yaml'
        yaml_file.write_text(yaml_content)
        
        # Should accept Path object
        workloads_file = load_workloads_file(yaml_file)
        assert len(workloads_file.workloads) == 1

    def test_load_yaml_with_string_path(self, tmp_path):
        """Test that string paths are accepted."""
        yaml_content = """
workloads:
  - api: TTSIM
    name: test
    basedir: workloads
    module: test.py
    instances:
      default: {}
"""
        yaml_file = tmp_path / 'workloads.yaml'
        yaml_file.write_text(yaml_content)
        
        # Should accept string path
        workloads_file = load_workloads_file(str(yaml_file))
        assert len(workloads_file.workloads) == 1


@pytest.mark.integration
class TestIntegration:
    """Integration tests for workloads module."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow from YAML to validated objects."""
        # Create a realistic workloads configuration
        yaml_content = """
workloads:
  - api: TTSIM
    name: resnet50
    basedir: workloads
    module: BasicResNet@basicresnet.py
    params:
      layers: 50
      width: 1.0
    instances:
      default:
        bs: 32
      corr:
        bs: 32
        layers: 50
      
  - api: TTSIM
    name: bert
    basedir: workloads
    module: bert.py
    instances:
      default:
        bs: 1
        seq_len: 512
      corr:
        bs: 1
"""
        yaml_file = tmp_path / 'workloads.yaml'
        yaml_file.write_text(yaml_content)
        
        # Load and validate
        workloads_file = load_workloads_file(yaml_file)
        
        # Verify structure
        assert len(workloads_file.workloads) == 2
        
        resnet = workloads_file.workloads[0]
        assert resnet.name == 'resnet50'
        assert resnet.params is not None
        assert resnet.params.get('layers') == 50
        assert 'corr' in resnet.instances
        
        bert = workloads_file.workloads[1]
        assert bert.name == 'bert'
        assert bert.params is None
        assert bert.instances['corr']['bs'] == 1

    def test_error_messages_are_helpful(self, tmp_path):
        """Test that validation errors have helpful messages."""
        yaml_content = """
workloads:
  - api: TTSIM
    name: test
    basedir: workloads
    module: ClassName@file.onnx  # Invalid - can't use @ with .onnx
    instances:
      default: {}
"""
        yaml_file = tmp_path / 'bad.yaml'
        yaml_file.write_text(yaml_content)
        
        with pytest.raises(ValueError) as exc_info:
            load_workloads_file(yaml_file)
        
        # Check that error message is informative
        error_msg = str(exc_info.value)
        assert 'onnx' in error_msg.lower()
        assert 'format' in error_msg.lower()

