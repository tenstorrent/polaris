#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

from ttsim.utils.cache import CacheManager


def test_cache_manager() -> None:
    """
    Test the CacheManager functionality.
    """
    # Test setting and getting content
    CacheManager.set_content('test_key', 'This is a test content.')
    content = CacheManager.get_content('test_key')
    assert content == 'This is a test content.', "Content mismatch after caching and retrieving."

    # Test retrieving non-existent content
    non_existent_content = CacheManager.get_content('non_existent_key')
    assert non_existent_content is None, "Expected None for non-existent key."



def test_retrieves_cached_filepath_when_file_exists(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    cache_dir = tmp_path / '.cache' / 'urlcontent'
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / 'test_key.cache'
    cache_file.write_text('test content')

    result = CacheManager.get_cached_filepath('test_key')

    assert result == cache_file


def test_returns_none_when_cached_filepath_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))

    result = CacheManager.get_cached_filepath('nonexistent_key')

    assert result is None


def test_retrieves_cached_content_when_file_exists(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    cache_dir = tmp_path / '.cache' / 'urlcontent'
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / 'test_key.cache'
    cache_file.write_text('test content')

    result = CacheManager.get_content('test_key')

    assert result == 'test content'


def test_returns_none_when_cached_content_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))

    result = CacheManager.get_content('nonexistent_key')

    assert result is None


def test_stores_content_in_cache_and_returns_filepath(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    cache_dir = tmp_path / '.cache' / 'urlcontent'

    result = CacheManager.set_content('test_key', 'test content')

    assert result == cache_dir / 'test_key.cache'
    assert result.read_text() == 'test content'

def test_handles_empty_key_for_get_cached_filepath(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))

    result = CacheManager.get_cached_filepath('')

    assert result is None


def test_handles_empty_key_for_get_content(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))

    result = CacheManager.get_content('')

    assert result is None


def test_diagnoses_empty_key_for_set_content(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))

    with pytest.raises(ValueError, match="Key cannot be empty"):
        CacheManager.set_content('', 'test content')


def test_handles_large_content_for_set_content(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    large_content = 'a' * 10**6  # 1 MB of data

    result = CacheManager.set_content('large_key', large_content)

    assert result.read_text() == large_content


def test_handles_special_characters_in_key(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    special_key = 'key_with_special_chars_!@#$%^&*()'

    CacheManager.set_content(special_key, 'special content')
    result = CacheManager.get_content(special_key)

    assert result == 'special content'
