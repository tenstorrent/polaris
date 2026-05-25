#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import argparse
import re

from loguru import logger


def tt_smi_find_indices():
    with os.popen('tt-smi -ls | iconv -f UTF-8 -t ASCII -c') as fin:
        lines = [l.rstrip() for l in fin]
    can_be_reset_index = [tmp for tmp, l in enumerate(lines) if 'can be reset' in l]
    if not can_be_reset_index:
        raise ValueError('tt-smi output does not identify boards that can be reset')
    if len(can_be_reset_index) > 1:
        raise ValueError('tt-smi output has multiple lines matching boards that can be reset')
    indices = []
    for line in lines[can_be_reset_index[0] + 1:]:
        m = re.search('^ [0-9]+', line)
        if not m:
            continue
        indices.append(m.group(0).strip())
    if len(indices) == 0:
        raise ValueError('tt-smi output does not identify boards that can be reset')
    return indices

    # tt-smi -ls | iconv -f UTF-8 -t ASCII -c  | sed '1,/can be reset/d' | grep '^ [0-9]' | sed 's/^ \([0-9]*\).*/\1/'


def reset_boards(indices: list[str]) -> int:
    index_str = ','.join(indices)
    cmd = f'tt-smi -r {index_str}'
    logger.info(f'resetting board(s) #{index_str}')
    ret = os.system(cmd)
    if ret != 0:
        raise ValueError(f'{cmd} failed, exit code {ret}')
    logger.success(f'Executed {cmd} successfully, board(s) #{index_str} reset successfully')
    return ret


def main() -> int:
    parser = argparse.ArgumentParser(description='Reset Tensix board(s) via tt-smi')
    parser.add_argument('--multiple-boards', action='store_true',
                        help='allow resetting more than one board when multiple are found')
    args = parser.parse_args()

    indices = tt_smi_find_indices()
    if len(indices) > 1 and not args.multiple_boards:
        raise ValueError(
            f'found {len(indices)} boards ({", ".join(f"#{i}" for i in indices)}); '
            'pass --multiple-boards to reset all of them'
        )
    return reset_boards(indices)


if __name__ == '__main__':
    exit(main())

