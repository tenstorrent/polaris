# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lightweight HuggingFace config downloader with local JSON cache fallback.

Uses only stdlib (``urllib`` + ``json``) -- no torch, transformers, or
huggingface_hub dependency.
"""

from __future__ import annotations

import json
import pathlib
import urllib.request

from loguru import logger

_HF_RESOLVE = "https://huggingface.co/{repo_id}/resolve/main/{filename}"


def load_hf_config(
    repo_id: str,
    filename: str = "config.json",
    cache_dir: pathlib.Path | str | None = None,
    *,
    timeout: int = 10,
) -> dict:
    """Load a JSON file from a local cache, falling back to a HuggingFace download.

    Checks *cache_dir* first so that workloads run reproducibly without network
    access when a cached copy is present.  On a successful download the result
    is written to *cache_dir* for subsequent offline use.

    Args:
        repo_id:   HuggingFace repo, e.g. ``"google/vit-base-patch16-224"``.
        filename:  File inside the repo to fetch (default ``"config.json"``).
        cache_dir: Directory for the local JSON cache file.
                   Defaults to the current working directory.
        timeout:   HTTP timeout in seconds.

    Returns:
        Parsed ``dict`` from the JSON file.
    """
    if cache_dir is None:
        cache_dir = pathlib.Path.cwd()
    else:
        cache_dir = pathlib.Path(cache_dir)

    safe_name = repo_id.replace("/", "--") + "--" + filename
    cache_path = cache_dir / safe_name

    if cache_path.exists():
        logger.debug("Loaded {}/{} from local cache {}", repo_id, filename, cache_path)
        return json.loads(cache_path.read_text())

    url = _HF_RESOLVE.format(repo_id=repo_id, filename=filename)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2))
        logger.debug("Loaded {}/{} from HuggingFace (cached to {})", repo_id, filename, cache_path)
        return data
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load {filename} for {repo_id}: "
            f"no local cache at {cache_path} and download from {url} failed: {exc}"
        ) from exc
