#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Update metadata files to include git commit hash for data provenance.

This utility fetches the latest commit hash from the source repository
and updates the _metadata.yaml file to include source_commit field.
"""

import sys
import argparse
import re
import urllib.request
from pathlib import Path
from typing import Optional

import yaml
import json


def extract_repo_info(url: str) -> Optional[tuple[str, str, str]]:
    """
    Extract repository owner, name, and branch from GitHub URL.

    Args:
        url: GitHub URL

    Returns:
        Tuple of (owner, repo, branch) or None if not a GitHub URL
    """
    # Pattern: https://raw.githubusercontent.com/{owner}/{repo}/refs/heads/{branch}/{path}
    pattern = r'https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/refs/heads/([^/]+)/'

    match = re.match(pattern, url)
    if match:
        return (match.group(1), match.group(2), match.group(3))

    # Alternative pattern: https://github.com/{owner}/{repo}/blob/{branch}/{path}
    pattern2 = r'https://github\.com/([^/]+)/([^/]+)/(?:blob|tree)/([^/]+)/'

    match = re.match(pattern2, url)
    if match:
        return (match.group(1), match.group(2), match.group(3))

    return None


def get_latest_commit_hash(owner: str, repo: str, branch: str) -> Optional[str]:
    """
    Get the latest commit hash for a repository branch.

    Args:
        owner: Repository owner
        repo: Repository name
        branch: Branch name

    Returns:
        Commit hash or None on error
    """
    # Use GitHub API to get branch info
    api_url = f'https://api.github.com/repos/{owner}/{repo}/branches/{branch}'

    try:
        with urllib.request.urlopen(api_url) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                return data['commit']['sha']
    except Exception as e:
        print(f'Warning: Could not fetch commit hash: {e}', file=sys.stderr)

    return None


def update_metadata_with_commit(
    metadata_file: Path, commit_hash: Optional[str] = None, dry_run: bool = False
) -> bool:
    """
    Update metadata file to include source commit hash.

    Args:
        metadata_file: Path to _metadata.yaml
        commit_hash: Commit hash to add (if None, will try to fetch)
        dry_run: If True, only show what would be done

    Returns:
        True if successful
    """
    if not metadata_file.exists():
        print(f'Error: Metadata file not found: {metadata_file}', file=sys.stderr)
        return False

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)

    if not metadata:
        print(f'Error: Empty metadata file: {metadata_file}', file=sys.stderr)
        return False

    # Check if already has commit hash
    if 'source_commit' in metadata and not commit_hash:
        print(f'Metadata already has source_commit: {metadata["source_commit"]}')
        return True

    # Try to fetch commit hash if not provided
    if not commit_hash and 'input_url' in metadata:
        url = metadata['input_url']
        repo_info = extract_repo_info(url)

        if repo_info:
            owner, repo, branch = repo_info
            print(f'Fetching latest commit for {owner}/{repo}/{branch}...')
            commit_hash = get_latest_commit_hash(owner, repo, branch)

            if commit_hash:
                print(f'Found commit: {commit_hash[:8]}')
            else:
                print('Warning: Could not fetch commit hash', file=sys.stderr)
        else:
            print(f'Warning: Could not parse repository info from URL: {url}')

    if not commit_hash:
        print('Error: No commit hash provided or fetched', file=sys.stderr)
        return False

    # Update metadata
    metadata['source_commit'] = commit_hash

    if dry_run:
        print(f'\n[DRY RUN] Would update {metadata_file}:')
        print(yaml.dump(metadata, default_flow_style=False, sort_keys=False))
        return True

    # Write updated metadata
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    print(f'✓ Updated {metadata_file}')
    return True


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description='Update metadata files with git commit hash for provenance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update specific tag metadata (will fetch latest commit)
  python update_metadata_provenance.py --tag 03nov25

  # Update with specific commit hash
  python update_metadata_provenance.py --tag 03nov25 --commit abc123def

  # Dry run to see what would change
  python update_metadata_provenance.py --tag 03nov25 --dry-run

  # Update all tags
  python update_metadata_provenance.py --all
        """,
    )

    parser.add_argument(
        '--tag', type=str, help='Tag to update (e.g., 03nov25, 15oct25)'
    )

    parser.add_argument(
        '--commit', type=str, help='Git commit hash to add (if not provided, will fetch latest)'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data' / 'metal' / 'inf',
        help='Base directory for reference data (default: data/metal/inf)',
    )

    parser.add_argument(
        '--all', action='store_true', help='Update all tags in data directory'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes',
    )

    args = parser.parse_args()

    if not args.tag and not args.all:
        print('Error: Either --tag or --all must be specified', file=sys.stderr)
        parser.print_help()
        return 1

    if args.tag and args.all:
        print('Error: Cannot use --tag and --all together', file=sys.stderr)
        return 1

    success = True

    if args.all:
        # Update all tag directories
        tag_dirs = [d for d in args.data_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]

        if not tag_dirs:
            print(f'No tag directories found in {args.data_dir}', file=sys.stderr)
            return 1

        print(f'Found {len(tag_dirs)} tag(s) to update\n')

        for tag_dir in sorted(tag_dirs):
            print(f'Processing tag: {tag_dir.name}')
            metadata_file = tag_dir / '_metadata.yaml'
            if not update_metadata_with_commit(metadata_file, args.commit, args.dry_run):
                success = False
            print()
    else:
        # Update specific tag
        tag_dir = args.data_dir / args.tag
        if not tag_dir.exists():
            print(f'Error: Tag directory not found: {tag_dir}', file=sys.stderr)
            return 1

        metadata_file = tag_dir / '_metadata.yaml'
        if not update_metadata_with_commit(metadata_file, args.commit, args.dry_run):
            success = False

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

