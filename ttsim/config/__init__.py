#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
import logging
import copy
from ..utils.common import parse_yaml, parse_worksheet
from .simconfig import SimConfig, XlsxConfig, WorkloadGroup, AWorkload
from pydantic import ValidationError
from .validators import PYDWlMapDataSpecValidator, PYDWlMapResourceSpecValidator, PYDWlMapSpecValidator, PYDPkgMemoryValidator, PYDPkgComputeValidator, PYDComputePipeValidator, \
     PYDL2CacheValidator, PYDMemoryBlockValidator, PYDComputeBlockValidator, PYDWorkloadListValidator, TTSimHLWlDevRunOpCSVPerfStats, \
    TTSimHLWlDevRunPerfStats, TTSimHLRunSummary, TTSimHLRunSummaryRow
from .simconfig import IPBlocksModel, PackageInstanceModel


def get_child(base, key, idattr='name'):
    if isinstance(base, dict):
        return base.get(key, None)
    if isinstance(base, list):
        subbases = [entry for entry in base if entry[idattr] == key]
        if not subbases:
            raise ValueError(f'{key} not found')
        if len(subbases) != 1:
            raise ValueError(f'{key} has multiple occurrences')
        return subbases[0]
    raise ValueError(f'{key} can not be searched in {base}')

def get_arspec_from_yaml(cfg_yaml_file):
    arch_dict = parse_yaml(cfg_yaml_file)

    for k in ['packages', 'ipblocks']:
        assert k in arch_dict, f"no {k} field in architecture spec {cfg_yaml_file}"
    ipblocks_dict = arch_dict['ipblocks']
    ipblocks_db = IPBlocksModel(**{'ipblocks': ipblocks_dict})
    ipblocks_name_2_block = {ipblock_entry['name']: ipblock_entry for ipblock_entry in ipblocks_dict}

    pkg_instance_db = dict()
    for pkgentry in arch_dict['packages']:
        for pkginstance in pkgentry['instances']:
            pkginstance['devname'] = pkgentry['name']
            ipgroups = []
            for ipgroup_base in pkginstance['ipgroups']:
                ipgroup = {x: ipgroup_base[x] for x in ipgroup_base if x != 'ip_overrides'}
                ipgroups.append(ipgroup)
                ipobj = copy.deepcopy(ipblocks_name_2_block[ipgroup_base['ipname']])
                ipgroup['ipobj'] = ipobj
                overrides = ipgroup_base.get('ip_overrides', None)
                if overrides is None:
                    overrides = {}
                for override_key, override_value in overrides.items():
                    base = ipobj
                    override_key_parts = override_key.split('.')
                    for ovkey_part in override_key_parts[:-1]:
                        newbase = get_child(base, ovkey_part)
                        if newbase is None:
                            raise ValueError(f'child for {ovkey_part} not found in {base}')
                        base = newbase
                        continue
                    logging.debug(f'{override_key=} {override_value=} {base=} {override_key_parts[-1]=}')
                    last_key = override_key_parts[-1]
                    old_value = base.get(last_key, None)
                    if old_value is None:
                        raise ValueError(f'attribute {last_key} not defined in {base}')
                    if old_value == override_value:
                        logging.warning('device %s ipgroup %s overrode value of %s from %s to %s (NO DIFFERENCE)',
                                        pkgentry['name'], ipgroup_base['ipname'], override_key, old_value, override_value)
                    else:
                        logging.info('device %s ipgroup %s overrode value of %s from %s to %s',
                                        pkgentry['name'], ipgroup_base['ipname'], override_key, old_value, override_value)
                    base[last_key] = override_value
            pkginstance['ipgroups'] = ipgroups
            try:
                _tmp = PackageInstanceModel(**pkginstance)
            except ValidationError as e:
                logging.error('validation error when creating %s', pkginstance['name'])
                raise
            logging.info('created instance %s', _tmp.name)
            pkg_instance_db[_tmp.name] = _tmp
    return ipblocks_db, pkg_instance_db


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
    assert 'compute' in cfg_dict['op_rsrc_spec'], "Attribute(compute) missing in op_rsrc_spec"
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
