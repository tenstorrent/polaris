# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import time
import urllib.error
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from ttsim.utils.lfc import CACHE_AGE_SECONDS, resolve_lfc_path


@pytest.mark.unit
def test_resolve_lfc_path_invalid_prefix():
    with pytest.raises(ValueError, match="Path must start with 'lfc://'"):
        resolve_lfc_path("invalid/path.yaml")

@pytest.mark.unit
def test_resolve_lfc_path_cached():
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).touch()
            # Set mtime to now
            os.utime(local_path, (time.time(), time.time()))

            result = resolve_lfc_path("lfc://test/file.yaml")
            assert result == local_path
    finally:
        os.chdir(original_cwd)

@pytest.mark.unit
def test_resolve_lfc_path_stale_download_success():
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).touch()
            # Set mtime to old
            old_time = time.time() - CACHE_AGE_SECONDS - 1
            os.utime(local_path, (old_time, old_time))
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_response = mock_open(read_data=b"content").return_value
                mock_response.status = 200
                mock_urlopen.return_value.__enter__.return_value = mock_response
                result = resolve_lfc_path("lfc://test/file.yaml")
                assert result == local_path
                mock_urlopen.assert_called_once()
                # Check content was written
                with open(local_path, 'rb') as f:
                    assert f.read() == b"content"
    finally:
        os.chdir(original_cwd)

@pytest.mark.unit
def test_resolve_lfc_path_download_fail_fallback():
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).write_text("old content")
            # Set mtime to be stale (older than CACHE_AGE_SECONDS)
            old_time = time.time() - CACHE_AGE_SECONDS - 1
            os.utime(local_path, (old_time, old_time))
            with patch('urllib.request.urlopen', side_effect=Exception("Network error")) as mock_urlopen:
                result = resolve_lfc_path("lfc://test/file.yaml")
                assert result == local_path
                # Verify download was attempted
                mock_urlopen.assert_called()
                # Content should remain old (preserved after failed download)
                assert Path(local_path).read_text() == "old content"
    finally:
        os.chdir(original_cwd)

@pytest.mark.unit
def test_resolve_lfc_path_download_fail_no_local():
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            with patch('urllib.request.urlopen', side_effect=Exception("Network error")):
                with pytest.raises(RuntimeError, match="Download failed and no local file exists"):
                    resolve_lfc_path("lfc://test/file.yaml")
    finally:
        os.chdir(original_cwd)

@pytest.mark.unit
def test_resolve_lfc_path_auth_required():
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_response = mock_open().return_value
                mock_response.status = 401
                mock_urlopen.return_value.__enter__.return_value = mock_response
                with pytest.raises(RuntimeError, match="Authentication required"):
                    resolve_lfc_path("lfc://test/file.yaml")
    finally:
        os.chdir(original_cwd)


# Security validation tests

@pytest.mark.unit
def test_resolve_lfc_path_rejects_empty_path():
    with pytest.raises(ValueError, match="Server path cannot be empty"):
        resolve_lfc_path("lfc://")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_absolute_path():
    with pytest.raises(ValueError, match="Absolute paths not allowed"):
        resolve_lfc_path("lfc:///etc/passwd")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_path_traversal():
    with pytest.raises(ValueError, match="Path traversal.*not allowed"):
        resolve_lfc_path("lfc://../../../etc/passwd")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_path_traversal_midpath():
    with pytest.raises(ValueError, match="Path traversal.*not allowed"):
        resolve_lfc_path("lfc://test/../../../etc/passwd")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_backslashes():
    with pytest.raises(ValueError, match="Backslashes not allowed"):
        resolve_lfc_path("lfc://test\\file.yaml")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_windows_drive_letter():
    with pytest.raises(ValueError, match="Windows drive letters not allowed"):
        resolve_lfc_path("lfc://C:/Windows/System32/file.txt")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_unc_path():
    with pytest.raises(ValueError, match="UNC paths not allowed"):
        resolve_lfc_path("lfc://\\\\server\\share\\file.txt")


@pytest.mark.unit
def test_resolve_lfc_path_allows_valid_nested_path():
    """Test that valid nested paths are allowed"""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/deep/nested/path/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).touch()
            os.utime(local_path, (time.time(), time.time()))
            
            result = resolve_lfc_path("lfc://deep/nested/path/file.yaml")
            assert result == local_path
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
def test_resolve_lfc_path_http_404_with_local_fallback():
    """Test that 404 errors fall back to stale local cache"""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).write_text("cached content")
            # Set mtime to be stale
            old_time = time.time() - CACHE_AGE_SECONDS - 1
            os.utime(local_path, (old_time, old_time))
            
            http_error = urllib.error.HTTPError(
                url="http://test.com/file.yaml",
                code=404,
                msg="Not Found",
                hdrs={},
                fp=None
            )
            with patch('urllib.request.urlopen', side_effect=http_error) as mock_urlopen:
                result = resolve_lfc_path("lfc://test/file.yaml")
                assert result == local_path
                # Verify download was attempted
                mock_urlopen.assert_called()
                # Content should be preserved
                assert Path(local_path).read_text() == "cached content"
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
def test_resolve_lfc_path_http_500_with_local_fallback():
    """Test that 500 errors fall back to stale local cache"""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).write_text("cached content")
            # Set mtime to be stale
            old_time = time.time() - CACHE_AGE_SECONDS - 1
            os.utime(local_path, (old_time, old_time))
            
            http_error = urllib.error.HTTPError(
                url="http://test.com/file.yaml",
                code=500,
                msg="Internal Server Error",
                hdrs={},
                fp=None
            )
            with patch('urllib.request.urlopen', side_effect=http_error) as mock_urlopen:
                result = resolve_lfc_path("lfc://test/file.yaml")
                assert result == local_path
                # Verify download was attempted
                mock_urlopen.assert_called()
                # Content should be preserved
                assert Path(local_path).read_text() == "cached content"
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
def test_resolve_lfc_path_http_404_without_local_fallback():
    """Test that 404 errors raise when no local cache exists"""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            http_error = urllib.error.HTTPError(
                url="http://test.com/file.yaml",
                code=404,
                msg="Not Found",
                hdrs={},
                fp=None
            )
            with patch('urllib.request.urlopen', side_effect=http_error):
                with pytest.raises(RuntimeError, match="Download failed.*HTTP 404.*no local file exists"):
                    resolve_lfc_path("lfc://test/file.yaml")
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
def test_resolve_lfc_path_http_401_no_fallback():
    """Test that 401 errors are raised immediately without fallback"""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).write_text("cached content")
            # Set mtime to be stale
            old_time = time.time() - CACHE_AGE_SECONDS - 1
            os.utime(local_path, (old_time, old_time))
            
            http_error = urllib.error.HTTPError(
                url="http://test.com/file.yaml",
                code=401,
                msg="Unauthorized",
                hdrs={},
                fp=None
            )
            with patch('urllib.request.urlopen', side_effect=http_error):
                # 401 should raise immediately, not fall back to cache
                with pytest.raises(RuntimeError, match="Authentication required"):
                    resolve_lfc_path("lfc://test/file.yaml")
    finally:
        os.chdir(original_cwd)