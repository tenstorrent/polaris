# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

CACHE_AGE_SECONDS = 604800  # 1 week

def _validate_server_path(server_path: str) -> None:
    """
    Validate server path to prevent path traversal attacks.

    Args:
        server_path: Path after 'lfc://' prefix

    Raises:
        ValueError: If path contains security risks (absolute paths, .., or backslashes)
    """
    if not server_path:
        raise ValueError("Server path cannot be empty")
    
    # Reject absolute paths
    if server_path.startswith('/'):
        raise ValueError(f"Absolute paths not allowed in LFC paths: {server_path}")
    
    # Reject Windows absolute paths (e.g., C:/, \\server\share)
    if len(server_path) >= 2 and server_path[1] == ':':
        raise ValueError(f"Windows drive letters not allowed in LFC paths: {server_path}")
    
    if server_path.startswith('\\\\'):
        raise ValueError(f"UNC paths not allowed in LFC paths: {server_path}")
    
    # Reject backslashes (potential path separator confusion)
    if '\\' in server_path:
        raise ValueError(f"Backslashes not allowed in LFC paths: {server_path}")
    
    # Reject path traversal attempts
    path_parts = server_path.split('/')
    for part in path_parts:
        if part == '..':
            raise ValueError(f"Path traversal (..) not allowed in LFC paths: {server_path}")
    
    # Additional safety: normalize and verify the resolved path stays within __ext
    try:
        # Construct the full local path and resolve it
        local_path = Path('__ext') / server_path
        resolved = local_path.resolve()
        expected_base = Path('__ext').resolve()
        
        # Check that resolved path is under __ext
        try:
            resolved.relative_to(expected_base)
        except ValueError:
            raise ValueError(f"Path would escape __ext directory: {server_path}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Invalid path format: {server_path}") from e

def resolve_lfc_path(lfc_path: str) -> str:
    """
    Resolve an LFC path to a local file path, downloading if necessary.

    Args:
        lfc_path: Path starting with 'lfc://', e.g., 'lfc://hlm-lut/whb0_n150_lut.yaml'

    Returns:
        Local file path relative to workspace, e.g., '__ext/hlm-lut/whb0_n150_lut.yaml'

    Raises:
        ValueError: If path does not start with 'lfc://' or contains invalid/unsafe components
        RuntimeError: If download fails and no local file exists
    """
    if not lfc_path.startswith('lfc://'):
        raise ValueError(f"Path must start with 'lfc://': {lfc_path}")

    # Translate to local path
    server_path = lfc_path[6:]  # Remove 'lfc://'
    
    # Validate server path for security (prevent path traversal)
    _validate_server_path(server_path)
    
    local_path = f"__ext/{server_path}"

    local_file = Path(local_path)
    if local_file.exists():
        mtime = os.path.getmtime(local_path)
        if time.time() - mtime < CACHE_AGE_SECONDS:
            return local_path  # Fresh, use it

    # Need to download
    download_lfc_file(server_path, local_path)
    return local_path

def download_lfc_file(server_path: str, local_path: str):
    """
    Download a file from LFC server to local path.

    Args:
        server_path: Path relative to simulators-ai-perf, e.g., 'hlm-lut/whb0_n150_lut.yaml'
        local_path: Local path, e.g., '__ext/hlm-lut/whb0_n150_lut.yaml'
    """
    # Determine server base URL
    is_ci = os.getenv('GITHUB_ACTIONS') == 'true'
    if is_ci:
        base_url = 'http://large-file-cache.large-file-cache.svc.cluster.local/simulators-ai-perf/'
    else:
        base_url = 'http://aus2-lfcache.aus2.tenstorrent.com/simulators-ai-perf/'

    url = base_url + server_path

    # Create local directory
    local_file = Path(local_path)
    local_file.parent.mkdir(parents=True, exist_ok=True)

    # Use a lock file to prevent concurrent downloads
    lock_path = f"{local_path}.lock"
    lock_file = None
    try:
        if HAS_FCNTL:
            lock_file = open(lock_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        
        # Check again if file exists and is fresh (another process may have downloaded it)
        if local_file.exists():
            mtime = os.path.getmtime(local_path)
            if time.time() - mtime < CACHE_AGE_SECONDS:
                return  # Fresh file exists, no need to download

        # Try download with retries
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    if response.status == 401:
                        raise RuntimeError(f"Authentication required for {url}")
                    
                    # Download to temporary file in same directory (ensures atomic rename)
                    with tempfile.NamedTemporaryFile(
                        mode='wb',
                        dir=local_file.parent,
                        prefix=f'.{local_file.name}.',
                        suffix='.tmp',
                        delete=False
                    ) as tmp_file:
                        tmp_path = tmp_file.name
                        tmp_file.write(response.read())
                    
                    # Atomically replace the target file
                    os.replace(tmp_path, local_path)
                return  # Success
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    raise RuntimeError(f"Authentication required for {url}")
                if attempt == 2:
                    # On final attempt, check for local fallback (except 401)
                    if local_file.exists():
                        print(f"Download failed after 3 attempts (HTTP {e.code}), using existing local file: {local_path}", file=sys.stderr)
                        return
                    else:
                        # Provide Tailscale diagnostic for non-CI environments
                        if not is_ci:
                            print("Direct access failed. Ensure Tailscale VPN is connected.", file=sys.stderr)
                        raise RuntimeError(f"Download failed (HTTP {e.code}) and no local file exists: {local_path}") from e
            except Exception as e:
                # Clean up temp file if it exists
                if 'tmp_path' in locals():
                    try:
                        os.unlink(tmp_path)
                    except FileNotFoundError:
                        pass
                
                if isinstance(e, RuntimeError) and "Authentication required" in str(e):
                    raise  # Re-raise auth errors immediately
                if attempt == 2:
                    # Check if local exists
                    if local_file.exists():
                        print(f"Download failed after 3 attempts, using existing local file: {local_path}", file=sys.stderr)
                        return
                    else:
                        # Provide Tailscale diagnostic for non-CI environments
                        if not is_ci:
                            print("Direct access failed. Ensure Tailscale VPN is connected.", file=sys.stderr)
                        raise RuntimeError(f"Download failed and no local file exists: {local_path}") from e
    finally:
        # Release lock and clean up lock file
        if lock_file:
            try:
                if HAS_FCNTL:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                os.unlink(lock_path)
            except Exception:
                pass  # Best effort cleanup