#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import os
import argparse
import deepdiff
import yaml
import csv
from pprint import pprint


def print_entries(header, entries):
    print(header, type(entries))
    pprint(entries)
    return


def print_type_changes(entries):
    print('Type Changes')
    for k, kentry in entries.items():
        print_entries(k, kentry)

def print_dictionary_item_added(entries):
    print_entries('Additional Dictionary Items', entries)

def print_values_changed(entries):
    print_entries('Values different', entries)


dispatch = {
    'type_changes': print_type_changes,
    'dictionary_item_added': print_dictionary_item_added,
    'values_changed': print_values_changed,
}


def read_file(filename):
    if filename.endswith('.yaml'):
        with open(filename) as fin:
            return yaml.load(fin, Loader=yaml.CLoader)
    if filename.endswith('.csv'):
        with open(filename) as fin:
            reader = csv.DictReader(fin)
            rows = [row for row in reader]
        for row in rows:
            for k in row:
                v = row[k]
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        v = v
                row[k] = v
        return rows
    raise NotImplementedError('should not be here')

def compare_yamls(file1: str, file2: str) -> int:
    obj1 = read_file(file1)
    obj2 = read_file(file2)

    comparison = deepdiff.DeepDiff(obj1, obj2, ignore_numeric_type_changes=True)
    result = 0
    for cmpk, cmpentry in comparison.items():
        result += 1
        try:
            dispatch[cmpk](cmpentry)
        except KeyError:
            print('error: unhandled result type ' + cmpk)
            exit(1)
        print('====================')
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose output')
    parser.add_argument('file1', help='YAML file (1)')
    parser.add_argument('file2', help='YAML file (2)')
    args = parser.parse_args()
    res = compare_yamls(args.file1, args.file2)
    if res:
        print('mismatch: ', os.path.basename(args.file1))
    elif args.verbose:
        print('match: ', os.path.basename(args.file1))
    return res

if __name__ == '__main__':
    exit(main())
