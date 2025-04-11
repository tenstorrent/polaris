#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
import logging
from ..utils.common import parse_yaml, parse_worksheet
from .simconfig import SimConfig, XlsxConfig, create_ipblock, create_package, WorkloadGroup, AWorkload
from .validators import PYDWlMapDataSpecValidator, PYDWlMapResourceSpecValidator, PYDWlMapSpecValidator, PYDPkgMemoryValidator, PYDPkgComputeValidator, PYDComputePipeValidator, \
     PYDL2CacheValidator, PYDMemoryBlockValidator, PYDComputeBlockValidator, PYDWorkloadListValidator


def get_arspec_from_yaml(cfg_yaml_file):
    cfg_dict = parse_yaml(cfg_yaml_file)
    for k in ['packages', 'ipblocks']:
        assert k in cfg_dict, f"no {k} field in architecture spec {cfg_yaml_file}"

    ipblocks = {}
    for ip_type, ip_dict in cfg_dict['ipblocks'].items():
        for ip_name, ip_cfg in ip_dict.items():
            try:
                if ip_type == 'compute':
                    validated_result_1 = PYDComputeBlockValidator(**ip_cfg)
                elif ip_type == 'memory':
                    validated_result_2 = PYDMemoryBlockValidator(**ip_cfg)
                else:
                    raise AssertionError('should not reach here')
            except Exception as e:
                logging.error('%s: error validating %s IP block named %s, configuration=%s: %s', cfg_yaml_file, ip_type, ip_name, ip_cfg, e)
                raise
            ipblocks[ip_name] = create_ipblock(ip_type, ip_name, ip_cfg)

    packages = {}
    for pkg_type, pkg_dict in cfg_dict['packages'].items():
        for pkg_name, pkg_cfg in pkg_dict.items():
            try:
                validated_result_3 = PYDPkgComputeValidator(**pkg_cfg['ipgroups']['compute'])
            except Exception as e:
                logging.error('%s: error validating package/compute %s of %s, %s: %s', cfg_yaml_file, pkg_type, pkg_name, pkg_cfg, e)
                raise
            try:
                validated_result_4 = PYDPkgMemoryValidator(**pkg_cfg['ipgroups']['memory'])
            except Exception as e:
                logging.error('%s: error validating package/memory %s of %s, %s: %s', cfg_yaml_file, pkg_type, pkg_name, pkg_cfg, e)
                raise
            packages[pkg_name] = create_package(pkg_type, pkg_name, pkg_cfg, ipblocks)

    return ipblocks, packages


def get_wlspec_from_yaml(cfg_yaml_file):
    cfg_dict = parse_yaml(cfg_yaml_file)
    validated_workloads = PYDWorkloadListValidator(**cfg_dict)
    assert 'workloads' in cfg_dict, f"Attribute(workloads) missing in {cfg_yaml_file}"
    wldb = {}
    for wlg_cfg in cfg_dict['workloads']:
        wldb[wlg_cfg['name']] = AWorkload.create_workload(wlg_cfg['api'], **wlg_cfg)
    return wldb


def get_wlmapspec_from_yaml(cfg_yaml_file):
    cfg_dict   = parse_yaml(cfg_yaml_file)
    cfg_object = PYDWlMapSpecValidator(**cfg_dict)
    
    required_fields = ['op_data_type_spec', 'op_removal_spec', 'op_fusion_spec', 'op_rsrc_spec']
    for ff in required_fields:
        assert ff in cfg_dict, f"required attribute: {ff} missing in workload map file: {cfg_yaml_file}"

    op2dt = []
    for k,v in cfg_dict['op_data_type_spec'].items():
        if k == 'global_type':
            op2dt.append(['*', v.lower()])
        elif k == 'override':
            for kk, vv in v.items():
                op2dt.append([kk.upper(), vv.lower()])
        else: # pragma: no cover
            pass

    op2rsrc = {}
    assert 'compute' in cfg_dict['op_rsrc_spec'], f"Attribute(compute) missing in op_rsrc_spec"
    for op_pipe, op_list in cfg_dict['op_rsrc_spec']['compute'].items():
        op2rsrc.update({o.upper(): op_pipe.lower() for o in op_list})

    null_ops       = [x.upper() for x in cfg_dict['op_removal_spec']]
    op_fusion_list = [[y.upper() for y in x] for x in cfg_dict['op_fusion_spec']]

    return op2dt, op2rsrc, null_ops, op_fusion_list


def parse_xlsx_config(xlsx_worksheet):
    IGNORE_COLUMNS = ['comments', 'remarks']
    rows, cols = parse_worksheet(xlsx_worksheet)
    cols = [col for col in cols if col.lower() not in IGNORE_COLUMNS]
    # DictReader returns column names in order of appearance, hence cols.pop(0) is indeed
    # the first column
    firstcol = cols.pop(0)

    archcfg_counts = Counter(cols)
    duplicate_archcfgs = [tmp for tmp, count in archcfg_counts.items() if count > 1]
    if duplicate_archcfgs:  # pragma: no cover
        for tmp in duplicate_archcfgs:
            logging.error(f'Architecture Config {tmp} defined {archcfg_counts[tmp]} times in {xlsx_worksheet}')
        raise Exception('some arch config names are defined multiple times')

    cfgTbl = {col: XlsxConfig(xlsx_worksheet + ':' + col) for col in cols}

    for row in rows:
        param = row[firstcol]
        for col in cols:
            cfgTbl[col].set_value(param, row[col])

    return cfgTbl


# TODO: Move to tools which generate YAML from XLSX
# def get_arspec_from_xlsx(xlsx_worksheet):
#     devices = {}
#     xlsx_cfg = parse_xlsx_config(xlsx_worksheet)
#     for archname, archcfg in xlsx_cfg.items():
#         devices[archname] = NvidiaGPU(archname, archcfg.values)
#     return devices
