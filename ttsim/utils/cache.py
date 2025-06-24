#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

CACHE_DIR: str = '.cache/urlcontent'


class CacheManager:
    """
    A simple cache manager to store and retrieve content based on a key.
    The content is stored in a file named after the key in a directory named '__temp'.
    """

    @staticmethod
    def get_cached_filepath(key: str) -> Path | None:
        """
        Retrieve cached content by key.

        Args:
            key (str): The name of the cache file.

        Returns:
            str: The cached content if found, otherwise None.
        """
        cache_path = Path(os.environ['HOME']) / CACHE_DIR
        os.makedirs(cache_path, exist_ok=True)
        cache_file = cache_path / f"{key}.cache"
        if os.path.exists(cache_file):
            return cache_file
        else:
            return None

    @staticmethod
    def get_content(key: str) -> str | None:
        """
        Retrieve cached content by key.

        Args:
            key (str): The name of the cache file.

        Returns:
            str: The cached content if found, otherwise None.
        """
        cached_file = CacheManager.get_cached_filepath(key)
        if cached_file is None:
            return None
        with open(cached_file, 'r') as f:
            return f.read()

    @staticmethod
    def set_content(key: str, content: str) -> Path:
        """
        Store content in the cache with the given key.

        Args:
            key (str): The name of the cache file.
            content (str): The content to be cached.
        """
        if not key:
            raise ValueError("Key cannot be empty")
        cache_path = Path(os.environ['HOME']) / CACHE_DIR
        os.makedirs(cache_path, exist_ok=True)
        cache_file = cache_path / f"{key}.cache"
        with open(cache_file, 'w') as f:
            f.write(content)
        return cache_file
