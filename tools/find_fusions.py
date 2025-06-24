#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import csv
import argparse
from loguru import logger
from typing import Any


def find_files_recursively(directory, extension='.csv'):
    """
    Recursively find all files with the given extension in the specified directory.
    """
    found_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                found_files.append(os.path.join(root, file))
    return sorted(found_files)


def find_fusions(fname, rows):
    """
    Find all fusions in the rows.
    """

    def update_opnum(rows):
        for row in rows:
            row['opnum'] = int(row['opnum'])
        for ndx, row in enumerate(rows):
            assert row['opnum'] == ndx

    def initialize_blocks(rows):
        name_to_opnum: dict[str, int] = {row['opname']: row['opnum'] for row in rows}
        assert len(name_to_opnum) == len(rows), f'Found duplicate opname in {fname}'
        for row3 in rows:
            if row3['removed'].lower() == 'true':
                row3['blocks'] = []
                continue
            if row3['fused'].lower() != 'true':
                row3['parent'] = None
                row3['blocks'] = [row3['opnum']]
            else:
                row3['parent'] = name_to_opnum[row3['fused_with_op']]
                row3['blocks'] = []

    update_opnum(rows)
    initialize_blocks(rows)
    for row in rows:
        if row['fused'].lower() == 'true':
            parent_root = row['opnum']
            while parent_root is not None and rows[parent_root]['parent'] is not None:
                parent_root = rows[parent_root]['parent']
            rows[parent_root]['blocks'].append(row['opnum'])
    block_set = set()
    for row in rows:
        if not row['blocks']:
            continue
        op_list = [rows[opnum]['optype'] for opnum in row['blocks']]
        block_set.add(tuple(op_list))

    return block_set


def find_directories_recursively(directory):
    """
    Recursively find all directories in the specified directory.
    """
    found_directories = []
    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            found_directories.append(os.path.join(root, dir_name))
    return found_directories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, required=True, help='Directory containing the CSV files')

    if not sys.argv[1:]:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    dirs = find_directories_recursively(args.inputdir)
    logger.debug('Found directories: {}', dirs)
    stat_dirs = [d for d in dirs if d.endswith('/STATS')]
    logger.debug('Found stat directories: {}', stat_dirs)
    if not stat_dirs:
        logger.error('No STATS directories found in the input directory.')
        return 1
    if len(stat_dirs) > 1:
        logger.error('Multiple STATS directories found: {}', stat_dirs)
        return 1
    fusion_sets: dict[str, Any] = dict()
    files = find_files_recursively(stat_dirs[0])
    for f in files:
        logger.debug('Processing file: {}', f)
        rows = [row for row in csv.DictReader(open(f, 'r'))]
        fusions = find_fusions(os.path.basename(f), rows)
        existing_file = {file for file, fset in fusion_sets.items() if fset['fusions'] == fusions}
        if existing_file:
            fusion_sets[existing_file.pop()]['files'].append(os.path.basename(f))
            continue
        fusion_sets[os.path.basename(f)] = {'fusions': fusions, 'files': []}
        continue
    for file in fusion_sets:
        logger.info('File: {}', file)
        for file2 in fusion_sets[file]['files']:
            logger.info('          and file: {}', file2)
        for fusion in sorted(fusion_sets[file]['fusions']):
            logger.info('      Fusion: {}', fusion)
        logger.info('=====================================')
        continue
    return 0


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stdout, level='INFO', format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>')
    main()
