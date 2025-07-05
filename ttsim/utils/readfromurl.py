#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from urllib.parse import urlparse

import requests

from ttsim.utils.cache import CacheManager


def url_2_key(url: str) -> str:
    """
    Converts a URL to a cache key by replacing certain characters.

    Args:
        url (str): The URL to convert.

    Returns:
        str: The cache key derived from the URL.
    """
    return url.replace('https://', '').replace('http://', '').replace('/', '_')


class FileLocator:
    SUPPORTED_SCHEMES = ['http', 'https']
    def __init__(self, filename: Path | str, use_cache: bool = True):
        """
            Initialize the FileLocator with a base path.
            :param filename: Filename.
            :param use_cache: Whether to use cache for fetching content.
        """
        if isinstance(filename, Path):
            self._filename = filename
            return
        res = urlparse(filename)
        if res.scheme == '':
            # If the scheme is empty, treat it as a local file path
            self._filename = Path(filename)
            return
        if res.scheme not in FileLocator.SUPPORTED_SCHEMES:
            raise NotImplementedError(f'Unsupported URL scheme {res.scheme} for {filename}')
        self._use_cache = use_cache
        self._filename = self.__fetch_url(filename)


    @property
    def path(self) -> Path:
        """
        Returns the path of the file.
        If the filename is a URL, it returns the URL as a Path object.
        """
        return Path(self._filename)

    def __fetch_url(self, url: str) -> Path:
        """
        Fetches the content from a URL and returns it as a string.
        Caches the content for future use.
        """
        cache_key = url_2_key(url)
        if (not self._use_cache) or (filepath := CacheManager.get_cached_filepath(cache_key)) is None:
            if not self._use_cache:
                logging.debug(f'cache force-skipped for reading {url}')
            else:
                logging.debug(f'no cached content for {url}, fetching from the URL')
            (response := requests.get(url)).raise_for_status() # Raise an error for bad responses
            filepath = CacheManager.set_content(cache_key, response.text)
            assert filepath is not None, f'Failed to cache content for {url}'
        else:
            logging.debug(f'Cached content for {url} being reused')
        return filepath


def locator_handle(filename: Path | str) -> Path:
    """
        Read the contents of a file and return it as a string.
        :param filename: Path to the file to read.
        :return: Contents of the file as a string.
    """
    locator: FileLocator = FileLocator(filename)
    return locator.path

def read_from_url(url: str, use_cache: bool = True) -> str:
    """
    Reads content from a URL and caches it.

    Args:
        url (str): The URL to read from.
        use_cache (bool): Whether to use cache for fetching content. Defaults to True.
        When set to False, it will always fetch the content from the URL.
    Returns:
        str: The content read from the URL.
    """
    with open(FileLocator(url, use_cache=use_cache).path) as fin:
        return fin.read()
