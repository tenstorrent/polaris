# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from contextlib import chdir

import pytest

from tools.spdxchecker import (ConfigFileModel, SPDXHeaderStatus, analyze_file, classify_file, collect_all_files,
                               collect_git_status_files, create_args, ext_2_lang, get_active_files, load_config,
                               validate_config)


@pytest.mark.parametrize("extension,expected_language", [
    (".py", "python"),
    (".js", "javascript"),
    (".html", "html"),
    (".css", "css"),
    (".yaml", "yaml"),
    (".unknown", "unknown"),
])
def test_validates_file_extension_to_language_mapping(extension, expected_language):
    assert ext_2_lang(extension) == expected_language


@pytest.mark.parametrize("filename,expected_result", [
    ("example.py", (".py", "python")),
    ("example.js", (".js", "javascript")),
    ("unknownfile.xyz", (".xyz", "unknown")),
])
def test_classifies_file_based_on_extension(filename, expected_result):
    assert classify_file(filename) == expected_result

lic_header = '# SPDX-License-Identifier: Apache-2.0'
copyright_header = '# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC'


@pytest.mark.parametrize("filename,allowed_licenses,allowed_copyrights,content,expected_status", [
    ("valid_file.py", ["Apache-2.0"], ["Tenstorrent AI ULC"], '\n'.join([lic_header, copyright_header]), (SPDXHeaderStatus.ST_OK, SPDXHeaderStatus.ST_OK)),
    ("missing_license.py", ["Apache-2.0"], ["Tenstorrent AI ULC"], copyright_header, (SPDXHeaderStatus.ST_MISSING, SPDXHeaderStatus.ST_OK)),
    ("missing_copyright.py", ["Apache-2.0"], ["Tenstorrent AI ULC"], lic_header, (SPDXHeaderStatus.ST_OK, SPDXHeaderStatus.ST_MISSING)),
    ("incorrect_license.py", ["Apache-2.0"], ["Tenstorrent AI ULC"], '\n'.join([lic_header + 'force-diff', copyright_header]), (SPDXHeaderStatus.ST_INCORRECT, SPDXHeaderStatus.ST_OK)),
    ("incorrect_copyright.py", ["Apache-2.0"], ["Tenstorrent AI ULC"], '\n'.join([lic_header, copyright_header+'force-diff']), (SPDXHeaderStatus.ST_OK, SPDXHeaderStatus.ST_INCORRECT)),
])
def test_analyzes_file_for_spdx_headers(filename, allowed_licenses, allowed_copyrights, expected_status, content, mocker):
    mocker.patch("builtins.open", mocker.mock_open(read_data=content))
    assert analyze_file(filename, allowed_licenses, allowed_copyrights) == expected_status


def test_collects_all_files_in_directory(tmp_path):
    (tmp_path / "file1.py").write_text("")
    (tmp_path / "file2.js").write_text("")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored_file").write_text("")
    assert collect_all_files(str(tmp_path)) == ["file1.py", "file2.js"]

def test_collects_git_status_files(tmp_path):
    res = get_active_files(True, ConfigFileModel()) # collect_git_status_files(True) != []
    assert isinstance(res, list)
    with chdir(tmp_path):
        with pytest.raises(FileNotFoundError, match=".git directory not found"):
            collect_git_status_files(True)

def test_create_args():
    args = create_args().parse_args([])
    assert args is not None
    assert hasattr(args, 'allowed_licenses')
    assert hasattr(args, 'allowed_copyrights')
    assert hasattr(args, 'config')


# New tests for enhanced configuration support

def test_config_model_with_all_fields():
    """Test ConfigFileModel with all fields populated."""
    config = ConfigFileModel(
        ignore=['*.json', '*.md'],
        warning=['*.html'],
        allowed_licenses=['Apache-2.0'],
        allowed_copyrights=['Tenstorrent AI ULC', '(C) 2025 Tenstorrent AI ULC']
    )
    assert len(config.ignore) == 2
    assert len(config.allowed_licenses) == 1
    assert len(config.allowed_copyrights) == 2


def test_config_model_with_empty_fields():
    """Test ConfigFileModel with empty optional fields."""
    config = ConfigFileModel()
    assert config.ignore == []
    assert config.warning == []
    assert config.allowed_licenses == []
    assert config.allowed_copyrights == []


def test_validate_config_with_valid_licenses():
    """Test configuration validation with valid SPDX licenses."""
    config = ConfigFileModel(
        allowed_licenses=['Apache-2.0'],
        allowed_copyrights=['Tenstorrent AI ULC']
    )
    # Should not raise any exceptions
    validate_config(config)


def test_validate_config_errors_on_invalid_licenses():
    """Test configuration validation raises error for invalid SPDX licenses."""
    config = ConfigFileModel(
        allowed_licenses=['Apache-2.0', 'InvalidLicense'],
        allowed_copyrights=['Tenstorrent AI ULC']
    )
    with pytest.raises(ValueError, match='Invalid SPDX license identifiers'):
        validate_config(config)


def test_validate_config_warns_no_licenses():
    """Test configuration validation warns when no licenses specified."""
    config = ConfigFileModel(
        allowed_copyrights=['Tenstorrent AI ULC']
    )
    # Should not raise an exception, just warn
    validate_config(config)
    # Test passes if no exception is raised


def test_validate_config_warns_no_copyrights():
    """Test configuration validation warns when no copyrights specified."""
    config = ConfigFileModel(
        allowed_licenses=['Apache-2.0']
    )
    # Should not raise an exception, just warn
    validate_config(config)
    # Test passes if no exception is raised


def test_validate_config_with_validation_disabled():
    """Test that validation can be disabled for invalid SPDX licenses."""
    config = ConfigFileModel(
        allowed_licenses=['Apache-2.0', 'MIT', 'BSD-3-Clause', 'CustomLicense'],
        allowed_copyrights=['Tenstorrent AI ULC']
    )
    # Should not raise an exception when validation is disabled
    validate_config(config, validate_spdx=False)
    # Test passes if no exception is raised


def test_validate_config_with_validation_enabled():
    """Test that validation rejects invalid licenses when enabled."""
    config = ConfigFileModel(
        allowed_licenses=['Apache-2.0', 'MIT'],
        allowed_copyrights=['Tenstorrent AI ULC']
    )
    # Should raise an exception when validation is enabled (default)
    with pytest.raises(ValueError, match='Invalid SPDX license identifiers'):
        validate_config(config, validate_spdx=True)


def test_load_config_file_not_found():
    """Test load_config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match='Configuration file not found'):
        load_config('/nonexistent/path/config.yml')


def test_load_config_valid_file(tmp_path):
    """Test load_config successfully loads a valid configuration file."""
    config_file = tmp_path / "test_config.yml"
    config_file.write_text("""
ignore:
  - '*.json'
  - '*.md'
warning:
  - '*.html'
allowed_licenses:
  - Apache-2.0
allowed_copyrights:
  - Tenstorrent AI ULC
  - (C) 2025 Tenstorrent AI ULC
""")
    config = load_config(str(config_file))
    assert len(config.ignore) == 2
    assert len(config.allowed_licenses) == 1
    assert len(config.allowed_copyrights) == 2


def test_load_config_empty_file(tmp_path):
    """Test load_config handles empty YAML file."""
    config_file = tmp_path / "empty_config.yml"
    config_file.write_text("")
    config = load_config(str(config_file))
    assert config.ignore == []
    assert config.allowed_licenses == []


def test_load_config_invalid_yaml(tmp_path):
    """Test load_config handles invalid YAML syntax."""
    config_file = tmp_path / "invalid_config.yml"
    config_file.write_text("invalid: yaml: syntax: [")
    with pytest.raises(Exception):  # YAMLError
        load_config(str(config_file))


def test_load_config_with_validation_disabled(tmp_path):
    """Test load_config with SPDX validation disabled allows any license."""
    config_file = tmp_path / "config_with_custom_license.yml"
    config_file.write_text("""
allowed_licenses:
  - MIT
  - BSD-3-Clause
  - CustomLicense
allowed_copyrights:
  - Tenstorrent AI ULC
""")
    # Should not raise an exception when validation is disabled
    config = load_config(str(config_file), validate_spdx=False)
    assert len(config.allowed_licenses) == 3
    assert 'MIT' in config.allowed_licenses
    assert 'CustomLicense' in config.allowed_licenses


def test_load_config_with_validation_enabled_rejects_invalid(tmp_path):
    """Test load_config with SPDX validation enabled rejects invalid licenses."""
    config_file = tmp_path / "config_with_invalid_license.yml"
    config_file.write_text("""
allowed_licenses:
  - MIT
  - InvalidLicense
allowed_copyrights:
  - Tenstorrent AI ULC
""")
    # Should raise an exception when validation is enabled
    with pytest.raises(ValueError, match='Invalid SPDX license identifiers'):
        load_config(str(config_file), validate_spdx=True)


def test_multiple_copyright_holders():
    """Test that multiple copyright holders are supported."""
    content = '\n'.join([lic_header, copyright_header])

    # Mock the file reading
    import unittest.mock as mock
    with mock.patch("builtins.open", mock.mock_open(read_data=content)):
        license_status, copyright_status = analyze_file(
            "test.py",
            ["Apache-2.0"],
            ["Tenstorrent AI ULC", "Another Company"]
        )
        assert license_status == SPDXHeaderStatus.ST_OK
        assert copyright_status == SPDXHeaderStatus.ST_OK


def test_multiple_license_and_copyright_lines():
    """Test that multiple license and copyright lines in a single file are all processed."""
    # Test with multiple valid license and copyright lines
    # Build test strings dynamically to avoid SPDX checker detecting them as actual headers
    spdx_copyright = "# SPDX-FileCopyrightText: "
    spdx_license = "# SPDX-License-Identifier: "

    multiple_valid_content = f'''#!/usr/bin/env python3
{spdx_copyright}(C) 2025 Tenstorrent AI ULC
{spdx_license}Apache-2.0
{spdx_copyright}(C) 2024 Another Company
{spdx_license}MIT

def hello():
    print("Hello world")
'''

    # Test with mixed valid and invalid lines
    mixed_content = f'''#!/usr/bin/env python3
{spdx_copyright}(C) 2025 Tenstorrent AI ULC
{spdx_license}Apache-2.0
{spdx_copyright}(C) 2024 Unknown Company
{spdx_license}InvalidLicense

def hello():
    print("Hello world")
'''

    import unittest.mock as mock

    # Test case 1: All valid entries should result in OK status
    with mock.patch("builtins.open", mock.mock_open(read_data=multiple_valid_content)):
        license_status, copyright_status = analyze_file(
            "test.py",
            ["Apache-2.0", "MIT"],
            ["Tenstorrent AI ULC", "Another Company"]
        )
        assert license_status == SPDXHeaderStatus.ST_OK
        assert copyright_status == SPDXHeaderStatus.ST_OK

    # Test case 2: Mixed valid/invalid entries should result in INCORRECT status
    with mock.patch("builtins.open", mock.mock_open(read_data=mixed_content)):
        license_status, copyright_status = analyze_file(
            "test.py",
            ["Apache-2.0", "MIT"],
            ["Tenstorrent AI ULC", "Another Company"],
            warn_flag=True  # Use warn_flag to avoid error logging in tests
        )
        assert license_status == SPDXHeaderStatus.ST_INCORRECT
        assert copyright_status == SPDXHeaderStatus.ST_INCORRECT


def test_cli_config_parameter():
    """Test that CLI includes --config parameter."""
    args = create_args().parse_args(['--config', 'custom_config.yml'])
    assert args.config == 'custom_config.yml'


def test_cli_validate_spdx_licenses_flag():
    """Test that CLI includes --validate-spdx-licenses flag."""
    # Test default value (enabled)
    args_default = create_args().parse_args([])
    assert args_default.validate_spdx_licenses is True

    # Test explicit enable
    args_enabled = create_args().parse_args(['--validate-spdx-licenses'])
    assert args_enabled.validate_spdx_licenses is True

    # Test explicit disable
    args_disabled = create_args().parse_args(['--no-validate-spdx-licenses'])
    assert args_disabled.validate_spdx_licenses is False


def test_cli_override_licenses():
    """Test that CLI can override allowed licenses."""
    args = create_args().parse_args(['--allowed-licenses', 'Apache-2.0', 'MIT'])
    assert args.allowed_licenses == ['Apache-2.0', 'MIT']


def test_cli_override_copyrights():
    """Test that CLI can override allowed copyrights."""
    args = create_args().parse_args(['--allowed-copyrights', 'Company A', 'Company B'])
    assert args.allowed_copyrights == ['Company A', 'Company B']
