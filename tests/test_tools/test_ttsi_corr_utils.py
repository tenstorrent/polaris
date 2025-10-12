#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ttsi_corr_utils module (tools/ttsi_corr/ttsi_corr_utils.py).

Tests cover:
- Hardware string parsing
- Workload name derivation
- Benchmark mapping
"""
import pytest

from tools.ttsi_corr.ttsi_corr_utils import get_benchmark_from_model, get_workload_name_from_model, parse_hardware_to_device


class TestParseHardwareToDevice:
    """Tests for hardware string parsing."""

    def test_parse_n150_simple(self):
        """Test parsing simple n150."""
        assert parse_hardware_to_device('n150') == 'n150'

    def test_parse_n150_with_arch(self):
        """Test parsing n150 with architecture."""
        assert parse_hardware_to_device('n150 (Wormhole)') == 'n150'

    def test_parse_n300_simple(self):
        """Test parsing simple n300."""
        assert parse_hardware_to_device('n300') == 'n300'

    def test_parse_n300_with_arch(self):
        """Test parsing n300 with architecture."""
        assert parse_hardware_to_device('n300 (Wormhole)') == 'n300'

    def test_parse_quietbox(self):
        """Test parsing QuietBox."""
        assert parse_hardware_to_device('QuietBox (Wormhole)') == 'quietbox'
        assert parse_hardware_to_device('quietbox') == 'quietbox'

    def test_parse_galaxy(self):
        """Test parsing Galaxy."""
        assert parse_hardware_to_device('Galaxy (Wormhole)') == 'galaxy'
        assert parse_hardware_to_device('GALAXY') == 'galaxy'

    def test_parse_with_extra_spaces(self):
        """Test parsing with extra whitespace."""
        assert parse_hardware_to_device('  n150  ') == 'n150'
        assert parse_hardware_to_device('n150  (Wormhole)') == 'n150'

    def test_parse_case_insensitive(self):
        """Test that parsing is case insensitive."""
        assert parse_hardware_to_device('N150') == 'n150'
        assert parse_hardware_to_device('N300 (WORMHOLE)') == 'n300'


class TestGetWorkloadNameFromModel:
    """Tests for workload name derivation from model names."""

    def test_bert_base(self):
        """Test BERT base model."""
        assert get_workload_name_from_model('BERT-Base') == 'bert'
        assert get_workload_name_from_model('bert-base') == 'bert'

    def test_bert_large(self):
        """Test BERT large model."""
        assert get_workload_name_from_model('BERT-Large') == 'bert'
        assert get_workload_name_from_model('bert-large') == 'bert'

    def test_resnet50(self):
        """Test ResNet-50 variants."""
        assert get_workload_name_from_model('ResNet-50') == 'resnet50'
        assert get_workload_name_from_model('resnet-50') == 'resnet50'
        assert get_workload_name_from_model('ResNet50') == 'resnet50'
        assert get_workload_name_from_model('resnet50') == 'resnet50'

    def test_llama_variants(self):
        """Test Llama model variants."""
        assert get_workload_name_from_model('Llama 3.1 8B') == 'llama'
        assert get_workload_name_from_model('Llama 3.2 1B') == 'llama'
        assert get_workload_name_from_model('llama-7b') == 'llama'

    def test_mamba(self):
        """Test Mamba models."""
        assert get_workload_name_from_model('Mamba-2.8B') == 'mamba'
        assert get_workload_name_from_model('mamba') == 'mamba'

    def test_yolo_variants(self):
        """Test YOLO variants."""
        assert get_workload_name_from_model('YOLOv8') == 'yolov8'
        assert get_workload_name_from_model('YOLOv7') == 'yolov7'
        assert get_workload_name_from_model('yolo-v8') == 'yolov8'
        assert get_workload_name_from_model('yolo-v7') == 'yolov7'

    def test_stable_diffusion(self):
        """Test Stable Diffusion / UNet."""
        assert get_workload_name_from_model('Stable Diffusion') == 'unet'
        assert get_workload_name_from_model('stable diffusion') == 'unet'
        assert get_workload_name_from_model('UNet') == 'unet'

    def test_unknown_model(self):
        """Test unknown model defaults to first word."""
        assert get_workload_name_from_model('SomeNewModel v1.0') == 'somenewmodel'
        assert get_workload_name_from_model('CustomModel') == 'custommodel'

    def test_case_insensitivity(self):
        """Test that model name matching is case insensitive."""
        assert get_workload_name_from_model('BERT') == 'bert'
        assert get_workload_name_from_model('LLAMA') == 'llama'
        assert get_workload_name_from_model('YOLO') == 'yolo'


class TestGetBenchmarkFromModel:
    """Tests for benchmark identifier mapping."""

    def test_bert_benchmark(self):
        """Test BERT benchmark mapping."""
        assert get_benchmark_from_model('BERT-Base') == 'Benchmark.BERT'
        assert get_benchmark_from_model('bert-large') == 'Benchmark.BERT'

    def test_resnet_benchmark(self):
        """Test ResNet benchmark mapping."""
        assert get_benchmark_from_model('ResNet-50') == 'Benchmark.ResNet50'
        assert get_benchmark_from_model('resnet50') == 'Benchmark.ResNet50'

    def test_llama_benchmark(self):
        """Test Llama benchmark mapping."""
        assert get_benchmark_from_model('Llama 3.1 8B') == 'Benchmark.Llama'
        assert get_benchmark_from_model('llama-7b') == 'Benchmark.Llama'

    def test_mamba_benchmark(self):
        """Test Mamba benchmark mapping."""
        assert get_benchmark_from_model('Mamba-2.8B') == 'Benchmark.Mamba'
        assert get_benchmark_from_model('mamba') == 'Benchmark.Mamba'

    def test_yolo_benchmark(self):
        """Test YOLO benchmark mapping."""
        assert get_benchmark_from_model('YOLOv8') == 'Benchmark.YOLO'
        assert get_benchmark_from_model('yolo-v7') == 'Benchmark.YOLO'

    def test_unet_benchmark(self):
        """Test UNet/Stable Diffusion benchmark mapping."""
        assert get_benchmark_from_model('Stable Diffusion') == 'Benchmark.UNet'
        assert get_benchmark_from_model('UNet') == 'Benchmark.UNet'

    def test_unknown_benchmark(self):
        """Test unknown model benchmark mapping."""
        result = get_benchmark_from_model('CustomModel v1.0')
        assert result == 'Benchmark.CustomModel'
        assert result.startswith('Benchmark.')

    def test_case_insensitivity(self):
        """Test that benchmark mapping is case insensitive."""
        assert get_benchmark_from_model('BERT') == 'Benchmark.BERT'
        assert get_benchmark_from_model('resnet') == 'Benchmark.ResNet50'
        assert get_benchmark_from_model('LLAMA') == 'Benchmark.Llama'


@pytest.mark.integration
class TestIntegration:
    """Integration tests for utility functions."""

    def test_complete_parsing_workflow(self):
        """Test complete workflow of parsing model information."""
        # Simulate extracting info from a table row
        model_name = 'Llama 3.1 8B'
        hardware = 'n150 (Wormhole)'
        
        # Parse components
        device = parse_hardware_to_device(hardware)
        workload_name = get_workload_name_from_model(model_name)
        benchmark = get_benchmark_from_model(model_name)
        
        # Verify results
        assert device == 'n150'
        assert workload_name == 'llama'
        assert benchmark == 'Benchmark.Llama'

    def test_multiple_models_consistent(self):
        """Test that related models map to same workload/benchmark."""
        models = [
            'Llama 3.1 8B',
            'Llama 3.2 1B',
            'Llama 2 7B',
        ]
        
        for model in models:
            assert get_workload_name_from_model(model) == 'llama'
            assert get_benchmark_from_model(model) == 'Benchmark.Llama'

    def test_all_supported_models(self):
        """Test all supported model types."""
        test_cases = [
            ('BERT-Base', 'bert', 'Benchmark.BERT'),
            ('ResNet-50', 'resnet50', 'Benchmark.ResNet50'),
            ('Llama 3.1 8B', 'llama', 'Benchmark.Llama'),
            ('Mamba-2.8B', 'mamba', 'Benchmark.Mamba'),
            ('YOLOv8', 'yolov8', 'Benchmark.YOLO'),
            ('Stable Diffusion', 'unet', 'Benchmark.UNet'),
        ]
        
        for model, expected_workload, expected_benchmark in test_cases:
            assert get_workload_name_from_model(model) == expected_workload
            assert get_benchmark_from_model(model) == expected_benchmark

