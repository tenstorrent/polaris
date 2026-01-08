#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test that tensor shape information is correctly added to STAT outputs."""

import csv
import json

import pytest
import yaml

import polaris
from tests.common import reset_typespec  # noqa: F401


def parse_tensor_string(s):
    """Parse tensor string format: name[dim1xdim2]:precision;name2[dim1xdim2]:precision"""
    if not s:
        return []

    tensors = []
    for part in s.split(';'):
        if '[' not in part:
            continue
        name_shape, precision = part.split(':')
        name, shape_str = name_shape.split('[', 1)  # Split only on first '[', for the unlikely case name has '['
        shape_str = shape_str.rstrip(']')
        if shape_str:
            shape = [int(d) for d in shape_str.split('x')]
        else:
            shape = []
        tensors.append({'name': name, 'shape': shape, 'precision': precision})
    return tensors


@pytest.mark.unit
def test_tensor_shapes_in_json_output(reset_typespec, tmp_path):  # noqa: F811
    """Test that tensor shape information appears correctly in JSON output."""

    # Run polaris with a simple workload
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = polaris.main([
        '--odir', str(output_dir),
        '--study', 'test_shapes',
        '--wlspec', 'config/mlperf_inference.yaml',
        '--archspec', 'config/all_archs.yaml',
        '--wlmapspec', 'config/wl2archmapping.yaml',
        '--filterwl', 'BERT_SQUAD_v1p1',
        '--filterwli', 'bert_large_b1',
        '--filterarch', 'Q1_A1',
        '--outputformat', 'json'
    ])

    assert result == 0, "Polaris should run successfully"

    # Find the generated JSON stats file
    stats_dir = output_dir / "test_shapes" / "STATS"
    assert stats_dir.exists(), f"STATS directory should exist at {stats_dir}"

    json_files = list(stats_dir.glob("*-opstats.json"))
    assert len(json_files) > 0, "At least one JSON stats file should be generated"

    # Read and validate the JSON file
    with open(json_files[0], 'r') as f:
        data = json.load(f)

    # Check that operatorstats exists and has the new fields
    assert 'operatorstats' in data, "JSON should contain 'operatorstats' field"
    operatorstats = data['operatorstats']
    assert len(operatorstats) > 0, "Should have at least one operator"

    # Check the first operator for the new tensor fields
    first_op = operatorstats[0]
    assert 'input_tensors' in first_op, "Operator should have 'input_tensors' field"
    assert 'output_tensors' in first_op, "Operator should have 'output_tensors' field"
    assert 'weight_tensors' in first_op, "Operator should have 'weight_tensors' field"

    # Verify the fields are strings
    assert isinstance(first_op['input_tensors'], str), "input_tensors should be a string"
    assert isinstance(first_op['output_tensors'], str), "output_tensors should be a string"
    assert isinstance(first_op['weight_tensors'], str), "weight_tensors should be a string"

    # Parse and verify structure
    input_tensors = parse_tensor_string(first_op['input_tensors'])
    output_tensors = parse_tensor_string(first_op['output_tensors'])

    assert len(input_tensors) > 0, "Should have at least one input tensor"
    assert len(output_tensors) > 0, "Should have at least one output tensor"
    assert 'name' in input_tensors[0], "Tensor should have 'name' field"
    assert 'shape' in input_tensors[0], "Tensor should have 'shape' field"

    # Find an operator with weights (like Gather or MatMul)
    ops_with_weights = [op for op in operatorstats if op['weight_tensors']]
    assert len(ops_with_weights) > 0, "Should have at least one operator with weights"

    weight_op = ops_with_weights[0]
    weight_tensors = parse_tensor_string(weight_op['weight_tensors'])
    assert len(weight_tensors) > 0, "Weight operator should have at least one weight"
    assert 'name' in weight_tensors[0], "Weight tensor should have 'name' field"
    assert 'shape' in weight_tensors[0], "Weight tensor should have 'shape' field"
    assert len(weight_tensors[0]['shape']) > 0, "Weight tensor should have non-empty shape"


@pytest.mark.unit
def test_tensor_shapes_in_csv_output(reset_typespec, tmp_path):  # noqa: F811
    """Test that tensor shape information appears correctly in CSV output."""

    # Run polaris with CSV output enabled
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = polaris.main([
        '--odir', str(output_dir),
        '--study', 'test_shapes_csv',
        '--wlspec', 'config/mlperf_inference.yaml',
        '--archspec', 'config/all_archs.yaml',
        '--wlmapspec', 'config/wl2archmapping.yaml',
        '--filterwl', 'BERT_SQUAD_v1p1',
        '--filterwli', 'bert_large_b1',
        '--filterarch', 'Q1_A1',
        '--dump_stats_csv'
    ])

    assert result == 0, "Polaris should run successfully"

    # Find the generated CSV stats file
    stats_dir = output_dir / "test_shapes_csv" / "STATS"
    assert stats_dir.exists(), f"STATS directory should exist at {stats_dir}"

    csv_files = list(stats_dir.glob("*-opstats.csv"))
    assert len(csv_files) > 0, "At least one CSV stats file should be generated"

    # Read and validate the CSV file
    with open(csv_files[0], 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0, "CSV should have at least one data row"

    # Check that the new columns exist
    first_row = rows[0]
    assert 'input_tensors' in first_row, "CSV should have 'input_tensors' column"
    assert 'output_tensors' in first_row, "CSV should have 'output_tensors' column"
    assert 'weight_tensors' in first_row, "CSV should have 'weight_tensors' column"

    # Verify that the tensor data is in string format (no commas in the string)
    assert ',' not in first_row['input_tensors'], "input_tensors should not contain commas"
    assert ',' not in first_row['output_tensors'], "output_tensors should not contain commas"
    assert ',' not in first_row['weight_tensors'], "weight_tensors should not contain commas"

    # Parse and verify structure
    input_tensors = parse_tensor_string(first_row['input_tensors'])

    assert len(input_tensors) > 0, "Should have at least one input tensor"
    assert 'name' in input_tensors[0], "Tensor should have 'name' field"
    assert 'shape' in input_tensors[0], "Tensor should have 'shape' field"


@pytest.mark.unit
def test_tensor_shapes_in_yaml_output(reset_typespec, tmp_path):  # noqa: F811
    """Test that tensor shape information appears correctly in YAML output."""

    # Run polaris with YAML output
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = polaris.main([
        '--odir', str(output_dir),
        '--study', 'test_shapes_yaml',
        '--wlspec', 'config/mlperf_inference.yaml',
        '--archspec', 'config/all_archs.yaml',
        '--wlmapspec', 'config/wl2archmapping.yaml',
        '--filterwl', 'BERT_SQUAD_v1p1',
        '--filterwli', 'bert_large_b1',
        '--filterarch', 'Q1_A1',
        '--outputformat', 'yaml'
    ])

    assert result == 0, "Polaris should run successfully"

    # Find the generated YAML stats file
    stats_dir = output_dir / "test_shapes_yaml" / "STATS"
    assert stats_dir.exists(), f"STATS directory should exist at {stats_dir}"

    yaml_files = list(stats_dir.glob("*-opstats.yaml"))
    assert len(yaml_files) > 0, "At least one YAML stats file should be generated"

    # Read and validate the YAML file
    with open(yaml_files[0], 'r') as f:
        data = yaml.safe_load(f)

    # Check structure
    assert 'operatorstats' in data, "YAML should contain 'operatorstats' field"
    operatorstats = data['operatorstats']
    assert len(operatorstats) > 0, "Should have at least one operator"

    # Verify tensor fields are strings
    first_op = operatorstats[0]
    assert 'input_tensors' in first_op, "Operator should have 'input_tensors' field"
    assert 'output_tensors' in first_op, "Operator should have 'output_tensors' field"
    assert 'weight_tensors' in first_op, "Operator should have 'weight_tensors' field"

    # Verify that the fields are strings
    assert isinstance(first_op['input_tensors'], str), "input_tensors should be a string"
    assert isinstance(first_op['output_tensors'], str), "output_tensors should be a string"
    assert isinstance(first_op['weight_tensors'], str), "weight_tensors should be a string"


@pytest.mark.unit
def test_model_dump_called_correctly(reset_typespec, tmp_path):  # noqa: F811
    """Test that model_dump() is called as a method (not passed as object) in YAML output.
    
    This verifies the fix for the issue where yaml.dump(model.model_dump) without
    parentheses would dump the method object itself instead of the model data.
    """
    # Run polaris with YAML output
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = polaris.main([
        '--odir', str(output_dir),
        '--study', 'test_model_dump',
        '--wlspec', 'config/mlperf_inference.yaml',
        '--archspec', 'config/all_archs.yaml',
        '--wlmapspec', 'config/wl2archmapping.yaml',
        '--filterwl', 'BERT_SQUAD_v1p1',
        '--filterwli', 'bert_large_b1',
        '--filterarch', 'Q1_A1',
        '--outputformat', 'yaml'
    ])

    assert result == 0, "Polaris should run successfully"

    # Find the generated YAML stats file
    stats_dir = output_dir / "test_model_dump" / "STATS"
    yaml_files = list(stats_dir.glob("*-opstats.yaml"))
    assert len(yaml_files) > 0, "At least one YAML stats file should be generated"

    # Read the YAML file as text to check for method object indicators
    with open(yaml_files[0], 'r') as f:
        yaml_text = f.read()

    # Check that the YAML doesn't contain method object indicators
    # If model_dump() is not called, YAML would contain something like:
    # "bound method BaseModel.model_dump" or similar Python object representation
    assert 'bound method' not in yaml_text.lower(), \
        "YAML should not contain 'bound method' (indicates method object, not data)"
    assert '<function' not in yaml_text.lower(), \
        "YAML should not contain '<function' (indicates function object, not data)"
    assert 'method object' not in yaml_text.lower(), \
        "YAML should not contain 'method object' (indicates method reference, not data)"

    # Load YAML and verify it's actual data (dict structure)
    with open(yaml_files[0], 'r') as f:
        data = yaml.safe_load(f)

    # Verify the loaded data is a dictionary with expected fields
    assert isinstance(data, dict), "YAML should load as a dictionary, not a method object"
    assert 'archname' in data, "YAML should contain 'archname' field from model data"
    assert 'operatorstats' in data, "YAML should contain 'operatorstats' field from model data"
    assert isinstance(data['operatorstats'], list), "operatorstats should be a list"
    assert len(data['operatorstats']) > 0, "operatorstats should contain data"

    # Verify operatorstats contain actual operator data, not method references
    first_op = data['operatorstats'][0]
    assert isinstance(first_op, dict), "Each operator stat should be a dict"
    assert 'opname' in first_op, "Operator should have 'opname' field"
    assert 'optype' in first_op, "Operator should have 'optype' field"
    assert isinstance(first_op['opname'], str), "opname should be a string"


@pytest.mark.unit
def test_input_weight_tensor_overlap(reset_typespec, tmp_path):  # noqa: F811
    """Test to check if input tensors and weight tensors have duplicates.
    
    By design, weight/parameter tensors appear in both input_tensors and weight_tensors
    since they are inputs to operations. This test verifies this expected behavior.
    """
    # Run polaris with JSON output for easy parsing
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = polaris.main([
        '--odir', str(output_dir),
        '--study', 'test_overlap',
        '--wlspec', 'config/mlperf_inference.yaml',
        '--archspec', 'config/all_archs.yaml',
        '--wlmapspec', 'config/wl2archmapping.yaml',
        '--filterwl', 'BERT_SQUAD_v1p1',
        '--filterwli', 'bert_large_b1',
        '--filterarch', 'Q1_A1',
        '--outputformat', 'json'
    ])

    assert result == 0, "Polaris should run successfully"

    # Read the JSON stats file
    stats_dir = output_dir / "test_overlap" / "STATS"
    json_files = list(stats_dir.glob("*-opstats.json"))
    with open(json_files[0], 'r') as f:
        data = json.load(f)

    operatorstats = data['operatorstats']
    
    # Find operators with weights
    ops_with_weights = [op for op in operatorstats if op['weight_tensors']]
    
    if len(ops_with_weights) > 0:
        # Check if weight tensors are also in input tensors
        duplicates_found = False
        for op in ops_with_weights:
            input_tensors = parse_tensor_string(op['input_tensors'])
            weight_tensors = parse_tensor_string(op['weight_tensors'])
            
            input_names = {t['name'] for t in input_tensors}
            weight_names = {t['name'] for t in weight_tensors}
            
            overlap = input_names & weight_names
            if overlap:
                duplicates_found = True
                # This is expected behavior - weights are inputs to the operation
                print(f"\nOperator '{op['opname']}' ({op['optype']}):")
                print(f"  Input tensors: {op['input_tensors']}")
                print(f"  Weight tensors: {op['weight_tensors']}")
                print(f"  Overlap: {overlap}")
        
        # Verify that duplicates exist (this is expected by design)
        assert duplicates_found, \
            "Expected to find weight tensors also appearing in input tensors " \
            "(weights are inputs to operations like MatMul, Gather, etc.)"
    else:
        # If no weights found, that's also valid but we should note it
        print("\nNo operators with weight tensors found in this workload")

