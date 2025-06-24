# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from contextlib import chdir
from tools.spdxchecker import ext_2_lang, classify_file, analyze_file, collect_all_files, collect_git_status_files, SPDXHeaderStatus, create_args, \
    IgnoreFileModel, get_active_files


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


@pytest.mark.parametrize("filename,allowed_licenses,allowed_copyright,content,expected_status", [
    ("valid_file.py", ["Apache-2.0"], "Tenstorrent AI ULC", '\n'.join([lic_header, copyright_header]), (SPDXHeaderStatus.ST_OK, SPDXHeaderStatus.ST_OK)),
    ("missing_license.py", ["Apache-2.0"], "Tenstorrent AI ULC", copyright_header, (SPDXHeaderStatus.ST_MISSING, SPDXHeaderStatus.ST_OK)),
    ("missing_copyright.py", ["Apache-2.0"], "Tenstorrent AI ULC", lic_header, (SPDXHeaderStatus.ST_OK, SPDXHeaderStatus.ST_MISSING)),
    ("incorrect_license.py", ["Apache-2.0"], "Tenstorrent AI ULC", '\n'.join([lic_header + 'force-diff', copyright_header]), (SPDXHeaderStatus.ST_INCORRECT, SPDXHeaderStatus.ST_OK)),
    ("incorrect_copyright.py", ["Apache-2.0"], "Tenstorrent AI ULC", '\n'.join([lic_header, copyright_header+'force-diff']), (SPDXHeaderStatus.ST_OK, SPDXHeaderStatus.ST_INCORRECT)),
])
def test_analyzes_file_for_spdx_headers(filename, allowed_licenses, allowed_copyright, expected_status, content, mocker):
    mocker.patch("builtins.open", mocker.mock_open(read_data=content))
    assert analyze_file(filename, allowed_licenses, allowed_copyright) == expected_status


def test_collects_all_files_in_directory(tmp_path):
    (tmp_path / "file1.py").write_text("")
    (tmp_path / "file2.js").write_text("")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored_file").write_text("")
    assert collect_all_files(str(tmp_path)) == ["file1.py", "file2.js"]

def test_collects_git_status_files(tmp_path):
    res = get_active_files(True, IgnoreFileModel()) # collect_git_status_files(True) != []
    assert isinstance(res, list)
    with chdir(tmp_path):
        with pytest.raises(FileNotFoundError, match=".git directory not found"):
            collect_git_status_files(True)

def test_create_args():
    args = create_args().parse_args([])
    assert args is not None
    assert hasattr(args, 'allowed_licenses')
    assert hasattr(args, 'allowed_copyright')
