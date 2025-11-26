#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Compare reference performance data between different tags.

This tool provides detailed comparison and diff analysis between two versions
of TT-Metal reference performance data, helping track changes, additions, and
updates across data snapshots.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

# Add project root to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tools.ttsi_corr.ttsi_corr_utils import TTSI_REF_DEFAULT_TAG, TTSI_REF_VALID_TAGS


# ANSI color codes for terminal output
class Colors:
    """Terminal color codes."""

    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def load_tag_data(tag_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all YAML files for a given tag directory.

    Args:
        tag_dir: Path to tag directory

    Returns:
        Dict mapping filename to list of benchmark entries
    """
    data = {}
    for yaml_file in tag_dir.glob('tensix_md_perf_metrics_*.yaml'):
        with open(yaml_file, 'r') as f:
            content = yaml.safe_load(f)
            if content:  # Skip empty files
                data[yaml_file.name] = content

    return data


def load_metadata(tag_dir: Path) -> Dict[str, Any]:
    """
    Load metadata for a given tag.

    Args:
        tag_dir: Path to tag directory

    Returns:
        Metadata dictionary
    """
    metadata_file = tag_dir / '_metadata.yaml'
    if not metadata_file.exists():
        return {}

    with open(metadata_file, 'r') as f:
        return yaml.safe_load(f) or {}


def create_entry_key(entry: Dict[str, Any]) -> str:
    """
    Create a unique key for a benchmark entry.

    Args:
        entry: Benchmark entry

    Returns:
        Unique key string
    """
    model = entry.get('model', 'unknown')
    hardware = entry.get('hardware', 'unknown')
    batch = entry.get('batch', 'unknown')
    precision = entry.get('precision', 'unknown')

    return f'{model}|{hardware}|{batch}|{precision}'


def extract_hardware_platforms(data: Dict[str, List[Dict[str, Any]]]) -> Set[str]:
    """
    Extract unique hardware platforms from data.

    Args:
        data: Tag data

    Returns:
        Set of hardware platform names
    """
    platforms = set()
    for entries in data.values():
        for entry in entries:
            hw = entry.get('hardware', '')
            # Extract device name before parentheses
            device = hw.split('(')[0].strip()
            platforms.add(device)

    return platforms


def extract_models(data: Dict[str, List[Dict[str, Any]]]) -> Set[str]:
    """
    Extract unique model names from data.

    Args:
        data: Tag data

    Returns:
        Set of model names
    """
    models = set()
    for entries in data.values():
        for entry in entries:
            model = entry.get('model', '')
            # Remove DP= suffix for grouping
            base_model = model.split(' (DP=')[0] if ' (DP=' in model else model
            models.add(base_model)

    return models


def get_metric_value(entry: Dict[str, Any]) -> Tuple[str, float]:
    """
    Get the primary metric value from an entry.

    Args:
        entry: Benchmark entry

    Returns:
        Tuple of (metric_name, metric_value)
    """
    metrics = [
        ('images_per_sec', entry.get('images_per_sec')),
        ('tokens_per_sec', entry.get('tokens_per_sec')),
        ('sentences_per_sec', entry.get('sentences_per_sec')),
        ('fps', entry.get('fps')),
        ('tokens_per_sec_per_user', entry.get('tokens_per_sec_per_user')),
    ]

    for metric_name, value in metrics:
        if value is not None:
            return (metric_name, value)

    return ('unknown', 0.0)


def compare_tags(
    old_tag: str, new_tag: str, data_dir: Path, verbose: bool = False
) -> None:
    """
    Compare two tags and report differences.

    Args:
        old_tag: Old tag identifier
        new_tag: New tag identifier
        data_dir: Base directory for reference data
        verbose: Show detailed differences
    """
    old_dir = data_dir / old_tag
    new_dir = data_dir / new_tag

    if not old_dir.exists():
        print(f'{Colors.RED}Error: Tag directory {old_tag} not found{Colors.RESET}')
        sys.exit(1)

    if not new_dir.exists():
        print(f'{Colors.RED}Error: Tag directory {new_tag} not found{Colors.RESET}')
        sys.exit(1)

    # Load data
    print(f'{Colors.CYAN}Loading data...{Colors.RESET}')
    old_data = load_tag_data(old_dir)
    new_data = load_tag_data(new_dir)
    old_metadata = load_metadata(old_dir)
    new_metadata = load_metadata(new_dir)

    # Print header
    print(f'\n{Colors.BOLD}{"=" * 80}{Colors.RESET}')
    print(
        f'{Colors.BOLD}Reference Data Comparison: '
        f'{old_tag} → {new_tag}{Colors.RESET}'
    )
    print(f'{Colors.BOLD}{"=" * 80}{Colors.RESET}\n')

    # Print metadata comparison
    print(f'{Colors.BLUE}Metadata:{Colors.RESET}')
    print(f'  Old tag: {old_tag}')
    print(f'    Parsed: {old_metadata.get("parsed_date", "unknown")}')
    print(f'    Source: {old_metadata.get("input_url", "unknown")}')
    if 'source_commit' in old_metadata:
        print(f'    Commit: {old_metadata["source_commit"]}')

    print(f'\n  New tag: {new_tag}')
    print(f'    Parsed: {new_metadata.get("parsed_date", "unknown")}')
    print(f'    Source: {new_metadata.get("input_url", "unknown")}')
    if 'source_commit' in new_metadata:
        print(f'    Commit: {new_metadata["source_commit"]}')
    else:
        print(
            f'    {Colors.YELLOW}⚠️  No source_commit in metadata '
            f'(recommended for traceability){Colors.RESET}'
        )

    # Count entries
    old_count = sum(len(entries) for entries in old_data.values())
    new_count = sum(len(entries) for entries in new_data.values())

    print(f'\n{Colors.BLUE}Summary:{Colors.RESET}')
    print(f'  Total benchmarks: {old_count} → {new_count}')

    if new_count > old_count:
        diff = new_count - old_count
        print(
            f'  {Colors.GREEN}✓ Added {diff} new '
            f'benchmark{"s" if diff > 1 else ""}{Colors.RESET}'
        )
    elif new_count < old_count:
        diff = old_count - new_count
        print(
            f'  {Colors.RED}✗ Removed {diff} '
            f'benchmark{"s" if diff > 1 else ""}{Colors.RESET}'
        )
    else:
        print(f'  {Colors.YELLOW}= No change in benchmark count{Colors.RESET}')

    # Compare hardware platforms
    old_platforms = extract_hardware_platforms(old_data)
    new_platforms = extract_hardware_platforms(new_data)

    added_platforms = new_platforms - old_platforms
    removed_platforms = old_platforms - new_platforms

    if added_platforms or removed_platforms:
        print(f'\n{Colors.BLUE}Hardware Platforms:{Colors.RESET}')

    if added_platforms:
        print(
            f'  {Colors.GREEN}✓ Added: '
            f'{", ".join(sorted(added_platforms))}{Colors.RESET}'
        )

    if removed_platforms:
        print(
            f'  {Colors.RED}✗ Removed: '
            f'{", ".join(sorted(removed_platforms))}{Colors.RESET}'
        )

    # Compare models
    old_models = extract_models(old_data)
    new_models = extract_models(new_data)

    added_models = new_models - old_models
    removed_models = old_models - new_models

    if added_models or removed_models:
        print(f'\n{Colors.BLUE}Models:{Colors.RESET}')

    if added_models:
        print(f'  {Colors.GREEN}✓ Added models:{Colors.RESET}')
        for model in sorted(added_models):
            print(f'    - {model}')

    if removed_models:
        print(f'  {Colors.RED}✗ Removed models:{Colors.RESET}')
        for model in sorted(removed_models):
            print(f'    - {model}')

    # Detailed entry comparison
    if verbose:
        print(f'\n{Colors.BLUE}Detailed Changes:{Colors.RESET}')

        # Create entry maps
        old_entries_map = {}
        for entries in old_data.values():
            for entry in entries:
                key = create_entry_key(entry)
                old_entries_map[key] = entry

        new_entries_map = {}
        for entries in new_data.values():
            for entry in entries:
                key = create_entry_key(entry)
                new_entries_map[key] = entry

        # Find added entries
        added_keys = set(new_entries_map.keys()) - set(old_entries_map.keys())
        if added_keys:
            print(f'\n  {Colors.GREEN}Added Benchmarks ({len(added_keys)}):{Colors.RESET}')
            for key in sorted(added_keys):
                entry = new_entries_map[key]
                metric_name, metric_value = get_metric_value(entry)
                print(
                    f'    + {entry["model"]} | {entry["hardware"]} | '
                    f'batch={entry["batch"]} | {metric_name}={metric_value}'
                )

        # Find removed entries
        removed_keys = set(old_entries_map.keys()) - set(new_entries_map.keys())
        if removed_keys:
            print(f'\n  {Colors.RED}Removed Benchmarks ({len(removed_keys)}):{Colors.RESET}')
            for key in sorted(removed_keys):
                entry = old_entries_map[key]
                metric_name, metric_value = get_metric_value(entry)
                print(
                    f'    - {entry["model"]} | {entry["hardware"]} | '
                    f'batch={entry["batch"]} | {metric_name}={metric_value}'
                )

        # Find modified entries
        common_keys = set(old_entries_map.keys()) & set(new_entries_map.keys())
        modified = []

        for key in common_keys:
            old_entry = old_entries_map[key]
            new_entry = new_entries_map[key]

            old_metric_name, old_metric_value = get_metric_value(old_entry)
            new_metric_name, new_metric_value = get_metric_value(new_entry)

            if old_metric_value != new_metric_value or old_metric_name != new_metric_name:
                modified.append((key, old_entry, new_entry))

        if modified:
            print(
                f'\n  {Colors.YELLOW}Modified Benchmarks '
                f'({len(modified)}):{Colors.RESET}'
            )
            for key, old_entry, new_entry in modified:
                old_metric_name, old_metric_value = get_metric_value(old_entry)
                new_metric_name, new_metric_value = get_metric_value(new_entry)

                if old_metric_value > 0:
                    change_pct = (
                        (new_metric_value - old_metric_value) / old_metric_value * 100
                    )
                    change_str = f' ({change_pct:+.1f}%)'
                else:
                    change_str = ''

                print(
                    f'    ~ {old_entry["model"]} | {old_entry["hardware"]} | '
                    f'batch={old_entry["batch"]}'
                )
                print(
                    f'      {old_metric_name}: {old_metric_value} → '
                    f'{new_metric_value}{change_str}'
                )

    # File-by-file breakdown
    print(f'\n{Colors.BLUE}File Breakdown:{Colors.RESET}')
    all_files = set(old_data.keys()) | set(new_data.keys())

    for filename in sorted(all_files):
        old_count = len(old_data.get(filename, []))
        new_count = len(new_data.get(filename, []))

        if old_count != new_count:
            diff = new_count - old_count
            if diff > 0:
                color = Colors.GREEN
                symbol = '+'
            else:
                color = Colors.RED
                symbol = ''
            print(
                f'  {color}{filename}: {old_count} → {new_count} '
                f'({symbol}{diff}){Colors.RESET}'
            )
        else:
            print(f'  {filename}: {old_count} (no change)')

    print(f'\n{Colors.BOLD}{"=" * 80}{Colors.RESET}\n')


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description='Compare reference performance data between tags',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Compare default with previous tag
  python {Path(__file__).name} --old 15oct25 --new 03nov25

  # Detailed comparison with verbose output
  python {Path(__file__).name} --old 15oct25 --new 03nov25 --verbose

  # Compare using default tags
  python {Path(__file__).name}

Valid tags: {', '.join(TTSI_REF_VALID_TAGS)}
        """,
    )

    parser.add_argument(
        '--old',
        type=str,
        default=TTSI_REF_VALID_TAGS[1] if len(TTSI_REF_VALID_TAGS) > 1 else None,
        help=f'Old tag to compare (default: {TTSI_REF_VALID_TAGS[1] if len(TTSI_REF_VALID_TAGS) > 1 else "N/A"})',
    )

    parser.add_argument(
        '--new',
        type=str,
        default=TTSI_REF_DEFAULT_TAG,
        help=f'New tag to compare (default: {TTSI_REF_DEFAULT_TAG})',
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data' / 'metal' / 'inf',
        help='Base directory for reference data (default: data/metal/inf)',
    )

    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Show detailed differences (added, removed, modified entries)',
    )

    args = parser.parse_args()

    if args.old is None:
        print(f'{Colors.RED}Error: --old tag required (only one tag available){Colors.RESET}')
        return 1

    compare_tags(args.old, args.new, args.data_dir, args.verbose)

    return 0


if __name__ == '__main__':
    sys.exit(main())

