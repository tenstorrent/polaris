#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pydantic import BaseModel

from ttsim.stats.hlmstats import (NUMPY_ARRAY_SIZE_THRESHOLD, HLMStats, OutputFormat, prepare_model_for_json,
                                  process_numpy_attr, save_data)


class MockOpStats(BaseModel):
    optype: str
    opname: str
    attrs: dict


class MockModel(BaseModel):
    operatorstats: list[MockOpStats] = []


@pytest.mark.unit
def test_process_numpy_attr_small_array():
    """Test processing of small numpy arrays."""
    v = np.array([1, 2, 3])
    opstats = MockOpStats(optype='Constant', opname='test_op', attrs={'value': v})

    with patch('ttsim.stats.hlmstats.logger') as mock_logger:
        process_numpy_attr(v, 0, opstats, 'value')

        # Should convert to descriptive string
        assert isinstance(opstats.attrs['value'], str)
        assert 'shape=(3,)' in opstats.attrs['value']
        assert 'value=[1, 2, 3]' in opstats.attrs['value']
        # Should log warning for Constant
        mock_logger.warning.assert_called_once()


@pytest.mark.unit
def test_process_numpy_attr_large_array():
    """Test processing of large numpy arrays."""
    v = np.ones(150)  # Larger than threshold
    opstats = MockOpStats(optype='Constant', opname='test_op', attrs={'value': v})

    with patch('ttsim.stats.hlmstats.logger') as mock_logger:
        process_numpy_attr(v, 0, opstats, 'value')

        # Should convert to truncated string
        assert isinstance(opstats.attrs['value'], str)
        assert '...' in opstats.attrs['value']  # Truncated
        # Should log warning for Constant
        mock_logger.warning.assert_called_once()


@pytest.mark.unit
def test_process_numpy_attr_non_constant_operator():
    """Test no warning for unexpected operator types with numpy arrays."""
    v = np.array([1.0])
    opstats = MockOpStats(optype='Add', opname='test_op', attrs={'value': v})

    with patch('ttsim.stats.hlmstats.logger') as mock_logger:
        process_numpy_attr(v, 0, opstats, 'value')

        # Should convert to descriptive string
        assert isinstance(opstats.attrs['value'], str)
        assert 'value=[1.0]' in opstats.attrs['value']
        # Should not log warning for unexpected type
        mock_logger.warning.assert_not_called()


@pytest.mark.unit
def test_save_data_json_with_numpy_arrays(tmp_path):
    """Test save_data with JSON output containing numpy arrays."""
    # Create mock operator stats with numpy array
    v = np.array([1, 2, 3])
    opstats = MockOpStats(optype='Constant', opname='test_op', attrs={'value': v})
    model = MockModel(operatorstats=[opstats])

    json_file = tmp_path / 'test.json'
    with patch('ttsim.stats.hlmstats.logger'):
        save_data(model, str(json_file), OutputFormat.FMT_JSON)

    # Read back and verify
    with open(json_file, 'r') as f:
        data = json.load(f)

    assert 'operatorstats' in data
    assert len(data['operatorstats']) == 1
    value = data['operatorstats'][0]['attrs']['value']
    assert isinstance(value, str)
    assert 'value=[1, 2, 3]' in value


@pytest.mark.unit
def test_numpy_array_size_threshold():
    """Test that the threshold constant is defined and reasonable."""
    assert isinstance(NUMPY_ARRAY_SIZE_THRESHOLD, int)
    assert NUMPY_ARRAY_SIZE_THRESHOLD > 0


@pytest.mark.unit
def test_process_numpy_attr_empty_array():
    """Test processing of empty numpy arrays."""
    v = np.array([])
    opstats = MockOpStats(optype='Add', opname='test_op', attrs={'value': v})

    with patch('ttsim.stats.hlmstats.logger') as mock_logger:
        process_numpy_attr(v, 0, opstats, 'value')

        # Should convert to descriptive string
        assert isinstance(opstats.attrs['value'], str)
        assert 'value=[]' in opstats.attrs['value']
        # Should not log warning for non-Constant
        mock_logger.warning.assert_not_called()


@pytest.mark.unit
def test_process_numpy_attr_multidimensional_small():
    """Test processing of small multi-dimensional numpy arrays."""
    v = np.array([[1, 2], [3, 4]])
    opstats = MockOpStats(optype='Add', opname='test_op', attrs={'value': v})

    with patch('ttsim.stats.hlmstats.logger') as mock_logger:
        process_numpy_attr(v, 0, opstats, 'value')

        # Should convert to descriptive string
        assert isinstance(opstats.attrs['value'], str)
        assert 'value=[[1, 2], [3, 4]]' in opstats.attrs['value']
        # Should not log warning for non-Constant
        mock_logger.warning.assert_not_called()


@pytest.mark.unit
def test_process_numpy_attr_multidimensional_large():
    """Test processing of large multi-dimensional numpy arrays."""
    v = np.ones((15, 15))  # Larger than threshold (225 > 100)
    opstats = MockOpStats(optype='Add', opname='test_op', attrs={'value': v})

    with patch('ttsim.stats.hlmstats.logger') as mock_logger:
        process_numpy_attr(v, 0, opstats, 'value')

        # Should convert to truncated string
        assert isinstance(opstats.attrs['value'], str)
        assert '...' in opstats.attrs['value']  # Truncated
        assert 'shape=(15, 15)' in opstats.attrs['value']
        # Should not log warning for non-Constant
        mock_logger.warning.assert_not_called()


@pytest.mark.unit
def test_process_numpy_attr_special_values_small():
    """Test processing of small numpy arrays with special values (NaN, inf)."""
    v = np.array([np.nan, np.inf, -np.inf, 1.0])
    opstats = MockOpStats(optype='Add', opname='test_op', attrs={'value': v})

    with patch('ttsim.stats.hlmstats.logger') as mock_logger:
        process_numpy_attr(v, 0, opstats, 'value')

        # Should convert to descriptive string
        result = opstats.attrs['value']
        assert isinstance(result, str)
        assert 'value=[nan, inf, -inf, 1.0]' in result
        # Should not log warning for non-Constant
        mock_logger.warning.assert_not_called()


@pytest.mark.unit
def test_process_numpy_attr_special_values_large():
    """Test processing of large numpy arrays with special values (NaN, inf)."""
    v = np.full(150, np.nan)  # Larger than threshold
    opstats = MockOpStats(optype='Add', opname='test_op', attrs={'value': v})

    with patch('ttsim.stats.hlmstats.logger') as mock_logger:
        process_numpy_attr(v, 0, opstats, 'value')

        # Should convert to truncated string
        assert isinstance(opstats.attrs['value'], str)
        assert '...' in opstats.attrs['value']  # Truncated
        assert 'shape=(150,)' in opstats.attrs['value']
        # Should not log warning for non-Constant
        mock_logger.warning.assert_not_called()


@pytest.mark.unit
def test_save_data_yaml_with_numpy_arrays(tmp_path):
    """Test save_data with YAML output containing numpy arrays."""
    # Create mock operator stats with numpy array
    v = np.array([1, 2, 3])
    opstats = MockOpStats(optype='Constant', opname='test_op', attrs={'value': v})
    model = MockModel(operatorstats=[opstats])

    yaml_file = tmp_path / 'test.yaml'
    # YAML should handle numpy arrays (possibly converting to lists)
    save_data(model, str(yaml_file), OutputFormat.FMT_YAML)

    # Verify the file was created and contains expected content
    assert yaml_file.exists()
    with open(yaml_file, 'r') as f:
        content = f.read()
    assert 'operatorstats' in content


@pytest.mark.unit
def test_save_data_pickle_with_numpy_arrays(tmp_path):
    """Test save_data with Pickle output containing numpy arrays."""
    # Create mock operator stats with numpy array
    v = np.array([1, 2, 3])
    opstats = MockOpStats(optype='Constant', opname='test_op', attrs={'value': v})
    model = MockModel(operatorstats=[opstats])

    pickle_file = tmp_path / 'test.pkl'
    # Pickle should handle numpy arrays correctly
    save_data(model, str(pickle_file), OutputFormat.FMT_PICKLE)

    # Verify by loading back
    with open(pickle_file, 'rb') as f:
        loaded_model = pickle.load(f)

    assert np.array_equal(loaded_model.operatorstats[0].attrs['value'], v)


@pytest.mark.unit
def test_csv_dumping_with_numpy_arrays(tmp_path):
    """Test that CSV dumping handles numpy arrays correctly when flag_dump_stats_csv is True."""
    # Mock the dependencies for HLMStats
    mock_dev = Mock()
    mock_dev.devname = 'test_dev'
    mock_dev.name = 'test_dev'
    mock_dev.freq_MHz = 1000
    mock_dev.simconfig_obj = Mock()
    mock_dev.get_exec_stats = Mock(return_value={})

    mock_opstats = Mock()
    mock_opstats.optype = 'Add'
    mock_opstats.opname = 'test_op'
    mock_opstats.attrs = {'value': np.array([1, 2, 3])}
    mock_opstats.uses_compute_pipe = 'test_pipe'
    mock_opstats.precision = 'test_prec'
    mock_opstats.repeat_count = 1
    mock_opstats.inList = []
    mock_opstats.outList = []
    mock_opstats.domain = 'test_domain'
    mock_opstats.opclass_str = 'test_class'
    mock_opstats.removed_in_optimization = False
    mock_opstats.fused_in_optimization = False
    mock_opstats.fused_with_op = None
    mock_opstats.perf_stats = {
        'inElems': 1, 'outElems': 1, 'inBytes': 1, 'outBytes': 1,
        'instrs': {}, 'inParamCount': 1, 'inActCount': 1, 'outActCount': 1
    }
    mock_opstats.compute_cycles = 1
    mock_opstats.mem_rd_cycles = 1
    mock_opstats.mem_wr_cycles = 1
    mock_opstats.exec_stats = {}

    mock_wlgraph = Mock()
    mock_wlgraph._tensors = {}
    mock_wlgraph.get_ordered_nodes = Mock(return_value=['test_op'])
    mock_wlgraph.get_op = Mock(return_value=mock_opstats)
    mock_wlgraph.is_input_node = Mock(return_value=False)
    mock_wlgraph.is_output_node = Mock(return_value=False)

    mock_wlinfo = {
        'wlg': 'test_group',
        'wln': 'test_workload',
        'wli': 'test_instance',
        'wlb': 1
    }

    mock_sinfo = {
        'flag_dump_stats_csv': True,
        'outputfmt': OutputFormat.FMT_NONE,
        'stat_dir': tmp_path,
        'config_dir': tmp_path,
        'odir': tmp_path,
        'saved_devices': set()
    }

    # Create HLMStats instance
    hlmstats = HLMStats(mock_dev, mock_wlgraph, mock_wlinfo, mock_sinfo)

    # Mock the model creation
    with patch('ttsim.stats.hlmstats.TTSimHLWlDevRunPerfStats') as mock_model_class:
        mock_model = Mock()
        mock_model.operatorstats = [mock_opstats]
        mock_model_class.return_value = mock_model

        # Call dump_stats
        hlmstats.dump_stats(None)

        # Check that the CSV file was created
        statF = 'test_dev-test_group-test_workload-test_instance-b1-opstats.csv'
        statP = tmp_path / statF
        assert statP.exists(), f"CSV file {statP} was not created"


@pytest.mark.unit
def test_prepare_model_for_json_no_operatorstats():
    """Test prepare_model_for_json with a model that has no operatorstats attribute."""
    class ModelWithoutOperatorStats(BaseModel):
        name: str = "test"

    model = ModelWithoutOperatorStats()
    result = prepare_model_for_json(model)

    # Should return the same model instance
    assert result is model
    assert result.name == "test"


@pytest.mark.unit
def test_prepare_model_for_json_empty_operatorstats():
    """Test prepare_model_for_json with a model that has empty operatorstats."""
    model = MockModel(operatorstats=[])
    result = prepare_model_for_json(model)

    # Should return the same model instance since no numpy arrays
    assert result is model
    assert result.operatorstats == []


@pytest.mark.unit
def test_prepare_model_for_json_mixed_attributes():
    """Test prepare_model_for_json with mixed numpy and non-numpy attributes."""
    # Create operator stats with mixed attributes
    opstats1 = MockOpStats(
        optype='Add',
        opname='op1',
        attrs={'scalar': 42, 'array': np.array([1, 2, 3])}
    )
    opstats2 = MockOpStats(
        optype='Mul',
        opname='op2',
        attrs={'string': 'hello', 'matrix': np.array([[1, 2], [3, 4]])}
    )
    model = MockModel(operatorstats=[opstats1, opstats2])

    result = prepare_model_for_json(model)

    # Should return a deep copy since there are numpy arrays
    assert result is not model
    assert len(result.operatorstats) == 2

    # Check that non-numpy attributes remain unchanged
    assert result.operatorstats[0].attrs['scalar'] == 42
    assert result.operatorstats[1].attrs['string'] == 'hello'

    # Check that numpy arrays are converted to strings
    assert isinstance(result.operatorstats[0].attrs['array'], str)
    assert 'value=[1, 2, 3]' in result.operatorstats[0].attrs['array']

    assert isinstance(result.operatorstats[1].attrs['matrix'], str)
    assert 'value=[[1, 2], [3, 4]]' in result.operatorstats[1].attrs['matrix']

    # Original model should be unchanged
    assert isinstance(model.operatorstats[0].attrs['array'], np.ndarray)
    assert isinstance(model.operatorstats[1].attrs['matrix'], np.ndarray)


@pytest.mark.unit
def test_prepare_model_for_json_no_numpy_arrays():
    """Test prepare_model_for_json with operatorstats but no numpy arrays."""
    opstats1 = MockOpStats(
        optype='Add',
        opname='op1',
        attrs={'scalar': 42, 'string': 'hello'}
    )
    opstats2 = MockOpStats(
        optype='Mul',
        opname='op2',
        attrs={'list': [1, 2, 3], 'dict': {'key': 'value'}}
    )
    model = MockModel(operatorstats=[opstats1, opstats2])

    result = prepare_model_for_json(model)

    # Should return the same model instance since no numpy arrays
    assert result is model
    assert len(result.operatorstats) == 2

    # Attributes should remain unchanged
    assert result.operatorstats[0].attrs['scalar'] == 42
    assert result.operatorstats[0].attrs['string'] == 'hello'
    assert result.operatorstats[1].attrs['list'] == [1, 2, 3]
    assert result.operatorstats[1].attrs['dict'] == {'key': 'value'}