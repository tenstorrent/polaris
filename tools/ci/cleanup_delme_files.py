#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Cleanup script for DELETEME_ prefixed files in GitHub gists.

This script automatically identifies and removes temporary files with the DELETEME_
prefix from GitHub gists. These files are created by the CI/CD badge generation
system for non-main branches and should be periodically cleaned up.
"""

import os
import sys
import requests
import json
from typing import List, Dict, Any
from loguru import logger


def cleanup_delme_files(gist_id: str, token: str) -> None:
    """
    Clean up DELETEME_ prefixed files from a GitHub gist.
    
    Args:
        gist_id: The GitHub gist ID
        token: GitHub personal access token
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        # Get current gist data
        logger.info(f"Fetching gist data for gist ID: {gist_id}")
        response = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch gist: HTTP {response.status_code}")
            logger.error(f"Response: {response.text}")
            return
        
        gist_data = response.json()
        files = gist_data.get("files", {})
        
        # Find DELETEME_ files
        delme_files = [filename for filename in files.keys() if filename.startswith("DELETEME_")]
        
        if not delme_files:
            logger.info("No DELETEME_ files found in gist")
            return
        
        logger.info(f"Found {len(delme_files)} DELETEME_ files to clean up:")
        for filename in delme_files:
            logger.info(f"  - {filename}")
        
        # Prepare update data (remove DELETEME_ files)
        update_data: Dict[str, Any] = {
            "files": {}
        }
        
        for filename in delme_files:
            update_data["files"][filename] = None  # Setting to None deletes the file
        
        # Update gist to remove files
        logger.info("Deleting DELETEME_ files...")
        response = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers=headers,
            data=json.dumps(update_data)
        )
        
        if response.status_code == 200:
            logger.success(f"Successfully cleaned up {len(delme_files)} DELETEME_ files")
        else:
            logger.error(f"Failed to update gist: HTTP {response.status_code}")
            logger.error(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during cleanup: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {e}")


def main():
    """Main function."""
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Get environment variables
    gist_id = os.getenv("GIST_ID")
    token = os.getenv("GIST_TOKEN")
    
    if not gist_id:
        logger.error("Error: GIST_ID environment variable not set")
        logger.info("Set GIST_ID to the GitHub gist ID you want to clean up")
        sys.exit(1)
    
    if not token:
        logger.error("Error: GIST_TOKEN environment variable not set")
        logger.info("Set GIST_TOKEN to your GitHub personal access token")
        sys.exit(1)
    
    logger.info("Starting DELETEME_ file cleanup process")
    logger.info(f"Target gist ID: {gist_id}")
    
    cleanup_delme_files(gist_id, token)
    
    logger.info("Cleanup process completed")


if __name__ == "__main__":
    main()
