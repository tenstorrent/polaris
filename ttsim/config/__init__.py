#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from loguru import logger
import copy
from ..utils.common import parse_yaml, parse_worksheet
from .simconfig import SimConfig as SimConfig, XlsxConfig as XlsxConfig, WorkloadGroup as WorkloadGroup, AWorkload as AWorkload
from pydantic import ValidationError
from .validators import PYDWlMapDataSpecValidator, PYDWlMapResourceSpecValidator, PYDWlMapSpecValidator, PYDPkgMemoryValidator, PYDPkgComputeValidator, PYDComputePipeValidator, \
     PYDL2CacheValidator, PYDMemoryBlockValidator, PYDComputeBlockValidator, PYDWorkloadListValidator, TTSimHLWlDevRunOpCSVPerfStats, \
    TTSimHLWlDevRunPerfStats, TTSimHLRunSummary, TTSimHLRunSummaryRow
from .simconfig import IPBlocksModel, PackageInstanceModel, TypeWorkload as TypeWorkload
from .wl2archmap import get_wlmapspec_from_yaml as get_wlmapspec_from_yaml

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
                    logger.debug('override_key={} override_value={} base={} override_key_parts[-1]={}',
                                 override_key, override_value, base, override_key_parts[-1])
                    last_key = override_key_parts[-1]
                    old_value = base.get(last_key, None)
                    if old_value is None:
                        raise ValueError(f'attribute {last_key} not defined in {base}')
                    if old_value == override_value:
                        logger.warning('device {} ipgroup {} overrode value of {} from {} to {} (NO DIFFERENCE)',
                                        pkgentry['name'], ipgroup_base['ipname'], override_key, old_value, override_value, once=True)
                    else:
                        logger.info('device {} ipgroup {} overrode value of {} from {} to {}',
                                        pkgentry['name'], ipgroup_base['ipname'], override_key, old_value, override_value, once=True)
                    base[last_key] = override_value
            pkginstance['ipgroups'] = ipgroups
            try:
                _tmp = PackageInstanceModel(**pkginstance)
            except ValidationError as e:
                logger.error('validation error when creating {}', pkginstance['name'])
                raise
            logger.info('created instance {}', _tmp.name)
            pkg_instance_db[_tmp.name] = _tmp
    return ipblocks_db, pkg_instance_db


def get_wlspec_from_yaml(cfg_yaml_file: str) -> dict[str, list[TypeWorkload]]:
    cfg_dict = parse_yaml(cfg_yaml_file)
    validated_workloads = PYDWorkloadListValidator(**cfg_dict)
    assert 'workloads' in cfg_dict, f"Attribute(workloads) missing in {cfg_yaml_file}"
    workload_names: set[str] = {wlg_cfg['name'].upper() for wlg_cfg in cfg_dict['workloads']}
    wldb: dict[str, list[TypeWorkload]] = {name: [] for name in workload_names}
    for wlg_cfg in cfg_dict['workloads']:
        # assert wlg_cfg['name'] not in wldb, f"Duplicate workload name {wlg_cfg['name']} in {cfg_yaml_file}"
        wldb[wlg_cfg['name'].upper()].append(AWorkload.create_workload(wlg_cfg['api'], **wlg_cfg))
    return wldb


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
            logger.error(f'Architecture Config {tmp} defined {archcfg_counts[tmp]} times in {xlsx_worksheet}')
        raise Exception('some arch config names are defined multiple times')

    cfgTbl = {col: XlsxConfig(xlsx_worksheet + ':' + col) for col in cols}

    for row in rows:
        param = row[firstcol]
        for col in cols:
            cfgTbl[col].set_value(param, row[col])

    return cfgTbl
