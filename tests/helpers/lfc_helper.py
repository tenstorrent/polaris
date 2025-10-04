#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Simple helper utilities for LFC file management in tests.

This module provides basic functions to check for LFC files and download them
if needed, using the centralized __ext directory structure.

Directory Structure:
    __ext/
    ├── tests/
    │   ├── models/           # ONNX and other model files
    │   └── data_files/       # ELF and other test data files
    └── rtl/                  # RTL test data (existing)

Developer Workflow (Manual Setup):
    For efficient development across multiple repository clones, developers can:
    1. Create a common directory outside Git repos: mkdir -p ~/lfc_cache/__ext/tests
    2. Download LFC files once: ./tools/ci/lfc_downloader.sh --extract <archive>
    3. Symlink from each clone: ln -s ~/lfc_cache/__ext __ext
    4. Tests will then use the shared files without re-downloading
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Project root directory (scripts are expected to be run from project root)
PROJECT_ROOT = Path.cwd()
LFC_DOWNLOADER_SCRIPT = PROJECT_ROOT / "tools" / "ci" / "lfc_downloader.sh"
EXT_DIR = PROJECT_ROOT / "__ext"
DOWNLOAD_TIMEOUT = 300   # 5-minute Timeout for the LFC download command

def ensure_lfc_file_exists(file_path: str, lfc_archive: str) -> bool:
    """
    Ensure LFC file exists, downloading if necessary.
    
    Args:
        file_path: Path to the required file relative to project root
        lfc_archive: LFC archive file name (e.g., 'ext_test_onnx_models.tar.gz')
    
    Returns:
        True if file exists after all attempts, False otherwise
    """
    full_path = PROJECT_ROOT / file_path
    
    # Check if file already exists
    if full_path.exists():
        return True
    
    # Try to download the archive
    if download_lfc_archive(lfc_archive):
        # Check again after download
        return full_path.exists()
    
    return False


def download_lfc_archive(archive_name: str) -> bool:
    """
    Download and extract LFC archive using the lfc_downloader.sh script.

    This function will create any necessary directories if they do not exist,
    and will extract the files from the archive to the project root directory.
    
    Args:
        archive_name: Name of the tar.gz file on LFC server
    
    Returns:
        True if successful, False otherwise
    """
    if not LFC_DOWNLOADER_SCRIPT.exists():
        print(f"Error: LFC downloader script not found: {LFC_DOWNLOADER_SCRIPT}")
        return False
    
    try:
        # Run the LFC downloader with extract option
        cmd = ["bash", str(LFC_DOWNLOADER_SCRIPT), "--extract", archive_name]
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=DOWNLOAD_TIMEOUT
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"LFC downloader failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"LFC downloader timed out for archive: {archive_name}")
        return False
    except Exception as e:
        print(f"Error running LFC downloader: {e}")
        return False


def get_ext_path(relative_path: str) -> Path:
    """
    Get the full path to a file in the __ext directory.
    
    Args:
        relative_path: Path relative to __ext (e.g., 'tests/models/onnx/inference/gpt_nano.onnx')
        
    Returns:
        Full path to the file
    """
    return EXT_DIR / relative_path


def require_lfc_file(file_path: str, lfc_archive: str, description: str = ""):
    """
    Require an LFC file to exist, downloading if necessary, or fail the test.
    
    Args:
        file_path: Path to the required file relative to project root
        lfc_archive: LFC archive file name
        description: Description of the file for error messages
    
    Raises:
        FileNotFoundError: If file cannot be obtained
    """
    if not ensure_lfc_file_exists(file_path, lfc_archive):
        error_msg = f"Required LFC file not available: {file_path}"
        if description:
            error_msg += f" ({description})"
        error_msg += f"\nTried to download from: {lfc_archive}"
        error_msg += "\nPlease check LFC connectivity and file availability."
        raise FileNotFoundError(error_msg)
