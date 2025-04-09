#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import yaml
import csv
import json
from typing import Any
from importlib import import_module, util as importlib_util
from pathlib import Path
import logging
from copy import deepcopy
from functools import lru_cache

openpyxl = None

class dict2obj:
    def __init__(self, d):
        for k,v in d.items():
            setattr(self, k, dict2obj(v) if isinstance(v, dict) else v)
    def __getattr__(self, item):
        raise AttributeError(f"Attribute '{item}' not found!!")

def parse_csv(csvfilename):
    with open(csvfilename) as fcsv:
        rowlines = [row.strip() for row in fcsv]

    # Skip rows beginning with '#', and blank rows
    rowlines = [rowlines[0]] + [row for row in rowlines[1:] if row != '' and not row.startswith('#')]

    reader = csv.DictReader(rowlines)
    rows = [row for row in reader]
    # DictReader has fieldnames attribute
    cols = reader.fieldnames

    return rows, cols

def parse_xlsx(filename, sheetname=None):
    global openpyxl
    if openpyxl is None:
        openpyxl = import_module('openpyxl')
    wb = openpyxl.load_workbook(filename, read_only=True, data_only=True)
    if sheetname is not None:
        try:
            wsheet = wb[sheetname]
        except KeyError:
            raise RuntimeError(f'no worksheet {sheetname} in {filename}')
    else:
        sheetnames = wb.sheetnames
        if len(sheetnames) == 1:
            sheetname = sheetnames[0]
            wsheet = wb[sheetname]
            logging.info('using sheet %s', sheetname)
        else:
            raise RuntimeError(f'too many sheets in {filename}, choose one of {sheetnames}')

    # get the columnnames, and headermap from first row
    header = dict()
    column_names = []
    for input_row in wsheet.iter_rows(min_row=1, max_row=1):
        for cell in input_row:
            if cell.value is None:  # empty
                continue
            header[cell.column_letter] = cell.value
            column_names.append(cell.value)
    # table is a list, each entry will be a dict corresponding to a row
    table = []
    for input_row in wsheet.iter_rows(min_row=2, max_row=wsheet.max_row + 1):
        row = {col: None for col in column_names}
        is_empty = True
        for cell in input_row:
            if cell.value is None:  # empty
                continue
            if cell.column_letter == 'A' and cell.value.startswith('#'):
                break
            try:
                row[header[cell.column_letter]] = cell.value
            except KeyError:
                # Cell with blank column header - looks like remark - skip
                break
            is_empty = False
        if is_empty:
            continue
        table.append(row)
    return table, column_names

def parse_worksheet(filename):
    if filename.endswith('csv'):
        return parse_csv(filename)
    elif filename.endswith('xls') or filename.endswith('xlsx'):
        if '@' in filename:
            sheet_name, input_file = filename.split('@')
        else:
            sheet_name, input_file = None, filename
        return parse_xlsx(input_file, sheet_name)
    else:
        raise RuntimeError(f'reading worksheet file "{filename} not supported')

def parse_yaml(yamlfile):
    res = None
    with open(yamlfile) as yamlf:
        res = yaml.safe_load(yamlf)
    return res

#TODO: Check if this sucks for large YAMLs: convert to streaming generator instead...
def parse_multidoc_yaml(yamlfile):
    res = None
    with open(yamlfile) as yamlf:
        res = []
        for rec in yaml.safe_load_all(yamlf):
            res.append(rec)
    return res

def print_csv(outcols, outrows, filename):
    # newline should be '' for DictWriter
    with open(filename, 'w', newline='') as ocsv:
        writer = csv.DictWriter(ocsv, fieldnames=outcols)
        writer.writeheader()
        for xrow in outrows:
            writer.writerow(xrow)

def print_json(jsdata, jsfilename):
    with open(jsfilename, 'w') as jsf:
        json.dump(jsdata, jsf)

def get_ttsim_functional_instance(python_module_path, python_instance_name, python_instance_cfg):    
    module_path = Path(python_module_path)
    python_module_name = module_path.stem
    if '@' in python_module_name:              # name@file means import name from file
        for tmp in module_path.parent.parts:   # Ensure '@' only in filename, not in intermediate parts
            assert '@' not in tmp
        name_parts = module_path.name.split('@')  # Split name, not stem, so as not to lose the extension
        assert len(name_parts) == 2            # Ensure only one '@'
        python_module_name = name_parts[0]
        module_path = module_path.parent / name_parts[1]
    spec = importlib_util.spec_from_file_location(python_module_name, module_path)
    assert spec is not None
    mod = importlib_util.module_from_spec(spec)
    assert mod is not None
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    wl_cls = getattr(mod, python_module_name)
    python_instance = wl_cls(python_instance_name, python_instance_cfg)
    return python_instance

def str_to_bool(s):
    if isinstance(s, bool):
        return s
    if isinstance(s, int):
        return s != 0
    if isinstance(s, float):
        return s != 0
    if s.lower() in ['true', 't', 'yes', 'y', 'on', 'enable', '1']:
        return True
    elif s.lower() in ['false', 'f', 'no', 'n', 'off', 'disable', '0']:
        return False
    else:
        raise ValueError('expecting boolean value')

def convert_units(val, from_unit, to_unit):
    from_unit, to_unit = from_unit.upper(), to_unit.upper()
    from_unit = ' B' if from_unit == 'B' else from_unit
    to_unit   = ' B' if to_unit   == 'B' else to_unit

    supported_factors = ['T', 'G', 'M', 'K', ' ']
    supported_types   = ['HZ', 'B', 'FLOPS', 'OPS']
    supported_units   = [f + t for f in supported_factors for t in supported_types]
    assert from_unit in supported_units, f"from_unit: {from_unit} should be one of {supported_units}!!"
    assert to_unit   in supported_units, f"to_unit: {to_unit} should be one of {supported_units}!!"
    ff, ft = from_unit[0], from_unit[1:]
    tf, tt = to_unit[0],   to_unit[1:]
    assert ft == tt, f"Illegal unit conversion {from_unit} --> {to_unit}!!"

    exp_tbl = {
            ('T', 'T'):  0,  ('T', 'G'):  1, ('T', 'M'):  2, ('T', 'K'):  3, ('T', ' '): 4,
            ('G', 'T'): -1,  ('G', 'G'):  0, ('G', 'M'):  1, ('G', 'K'):  2, ('G', ' '): 3,
            ('M', 'T'): -2,  ('M', 'G'): -1, ('M', 'M'):  0, ('M', 'K'):  1, ('M', ' '): 2,
            ('K', 'T'): -3,  ('K', 'G'): -2, ('K', 'M'): -1, ('K', 'K'):  0, ('K', ' '): 1,
            (' ', 'T'): -4,  (' ', 'G'): -3, (' ', 'M'): -2, (' ', 'K'): -1, (' ', ' '): 0,
            }
    try:
        exp = exp_tbl[(ff,tf)]
    except KeyError:
        print(f"convert_units: {from_unit} --> {to_unit} not supported!!")
        raise

    new_val = val * (1024 ** exp) if ft == 'B' else val * (1000 ** exp)
    return new_val

def make_tuple(value, tuple_len):
    if isinstance(value, tuple([tuple, list])):
        if len(value) != tuple_len:
            raise RuntimeError(f'incompatible tuple/list with {len(value)} instead of {tuple_len} elements')
        return value
    if isinstance(value, int):
        return tuple([value for _ in range(tuple_len)])
    raise NotImplementedError(f'{value} for make_tuple')

def check_known_args(opname: str, /,
                     args: dict[str, Any], 
                     default_args: dict[str, Any]) -> None:
    unknown_args = set(args.keys()) - set(default_args.keys())
    if unknown_args:
         raise AssertionError(f'Unknown args {unknown_args} used with operator {opname}')

def get_kwargs_with_defaults(opname: str, /,
                             args: dict[str, Any], 
                             default_args: dict[str, Any]) -> dict[str, Any]:
    """
        Utility function to get the values of arguments, with defaults for missing ones.
        Also asserts that there are no unknown / unsupported arguments
    """
    check_known_args(opname, args, default_args)
    eff_args = deepcopy(default_args)
    eff_args.update(args)
    return eff_args

@lru_cache(128)
def warnonce(msg, *args, **kwargs):
    logging.warning(msg, *args, **kwargs)
