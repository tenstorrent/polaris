#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import csv
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
from loguru import logger
import subprocess
from pathlib import Path
from typing import Any

from ttsim.utils.common import parse_yaml, print_yaml

type ScoreTuple = tuple[str, str, str, str]
type ScoreDict = dict[ScoreTuple, float]


OUTPUT_DIR: Path = Path('__TMP_TENSIX_METAL_CORR_OUT')


def read_scores(filepath: Path) -> ScoreDict:
    actual_scores: ScoreDict = dict()
    logger.info('===============================================')
    logger.info('Actual scores from {}', filepath)
    with open(filepath) as fin:
        actual_results = json.load(fin)['summary']
        for actual_res in actual_results:
            actual_key = tuple([actual_res['devname'], actual_res['wlcls'], actual_res['wlname'], actual_res['wlinstance']])
            actual_scores[actual_key] = actual_res['ideal_throughput']
    return actual_scores


def compare_scores(ref_scores: ScoreDict, actual_scores: ScoreDict) -> list[dict[str, Any]]:
    result = []
    logger.info('===============================================')
    logger.info('Tensix Correlation Results')
    common_keys = set(ref_scores.keys()).intersection(set(actual_scores.keys()))
    only_ref_keys = set(ref_scores.keys()).difference(set(actual_scores.keys()))
    only_actual_keys = set(actual_scores.keys()).difference(set(ref_scores.keys()))
    if only_ref_keys:
        logger.warning('Keys present in reference scores but not in actual scores: {}', only_ref_keys)
    if only_actual_keys:
        logger.warning('Keys present in actual scores but not in reference scores: {}', only_actual_keys)
    for key in common_keys:
        ref_score = ref_scores[key]
        actual = actual_scores[key]
        ratio = ref_score / actual
        result.append(
            {
                'Arch': key[0],
                'Workload': key[2],
                'Instance': key[3],
                'Api': key[1],
                'Tensix-Score': ref_score,
                'Actual-Score': actual,
                'Diff': ref_score - actual,
                'Ratio': ratio,
            }
        )
    return result

def main() -> int:
    tensix_workloads_yaml_file: str = 'tensix_workloads.yaml'
    tensix_runcfg_file: str = 'tensix_runcfg.yaml'
    odir: Path = OUTPUT_DIR
    sname: str = 'SIMPLE'
    tensix_perf_data_dir: Path = Path('data/metal/inf/closed')  # Tensix metal results
    gpu_dev_tbl = {
        'n150': 'n150',
        'n300': 'n300',
    }

    uniq_devs = set()
    ttsim_wlspec = []

    metal_ref_scores: ScoreDict = dict()

    opath = Path(odir)
    os.makedirs(opath, exist_ok=True)

    wls = ['bert', 'resnet50']
    for wl in wls:
        data_file = tensix_perf_data_dir / ('tensix_perf_metrics_' + wl + '.yaml')
        data_obj = parse_yaml(data_file)
        for tensix_cfg in data_obj:
            # if tensix_cfg['scenario'] != 'Scenario.Offline':
            #     continue
            missing = [f for f in ['benchmark', 'gpu', 'gpu_batch_size', 'perf', 'system', 'precision', 'metric'] if f not in tensix_cfg]
            if missing:
                raise ValueError(f'Missing fields {missing} in {data_file} for workload {wl} in {tensix_cfg.keys()}')
            scenario = 'offline'  # Hardcoded for simplicity    # TODO: confirm if this is always 'offline' for Tensix
            benchmark = tensix_cfg['benchmark']
            gpu = tensix_cfg['gpu']
            bs = tensix_cfg['gpu_batch_size']  # [wl]
            nodes = 1
            num_gpu = 1
            perf = tensix_cfg['perf']
            system = tensix_cfg['system']
            prec = tensix_cfg['precision']
            metric = tensix_cfg['metric']
            ref_perf = perf / num_gpu / nodes
            gpu_dev = gpu_dev_tbl[system]

            instance_name = f'b{bs}'
            xrec = {
                'api': 'TTSIM',
                'basedir': 'workloads',
                'scenario': scenario,
                'benchmark': benchmark,
                'name': wl,
                'gpu': gpu,
                'nodes': nodes,
                'num_gpu': num_gpu,
                'perf': perf,
                'sys': sys,
                'prec': prec,
                'metric': metric,
                'ref_perf': ref_perf,
                'gpu_dev': gpu_dev,
                'instances': {instance_name: {'bs': bs}},
                # 'cp_streams'  : cp_streams,
                # 'inf_streams' : inf_streams,
            }
            instance_key = tuple([xrec['gpu_dev'], xrec['api'], xrec['name'], instance_name])
            if instance_key in metal_ref_scores:
                raise ValueError(f'Duplicate Instance key {instance_key} in {data_file}')
            metal_ref_scores[instance_key] = ref_perf

            if wl == 'bert':
                # seqlen = tensix_cfg['bert_opt_seqlen']
                seqlen = 384  # Hardcoded for simplicity    # TODO: confirm; picked up from metal repo code (following the landing page URL)
                xrec['module'] = 'BasicLLM@BasicLLM.py'
                xrec['instances'][instance_name].update(
                    {'nL': 24, 'nH': 16, 'dE': 1024, 'nW': seqlen, 'vocab_sz': 30522}
                )
            elif wl == 'resnet50':
                xrec['module'] = 'ResNet@basicresnet.py'
                xrec['instances'][instance_name].update(
                    {
                        'layers': [3, 4, 6, 3],
                        'num_classes': 1000,
                        'num_channels': 3,
                    }
                )
            else:
                pass
            uniq_devs.add(gpu_dev)
            ttsim_wlspec.append(xrec)

            """
            ostr = ""
            if wl == 'bert':
                ostr += f'seqlen      = {seqlen}\n'
            ostr += \
            f'sys         = {sys        }\n' + \
            f'precision   = {prec       }\n' + \
            f'benchmark   = {benchmark  }\n' + \
            f'metric      = {metric     }\n' + \
            f'scenario    = {scenario   }\n' + \
            f'gpu         = {gpu        }\n' + \
            f'bs          = {bs         }\n' + \
            f'nodes       = {nodes      }\n' + \
            f'num_gpu     = {num_gpu    }\n' + \
            f'cp_streams  = {cp_streams }\n' + \
            f'inf_streams = {inf_streams}\n' + \
            f'perf        = {perf:.0f}\n' + f'ref_perf    = {ref_perf:.0f}\n'
            print()
            print(ostr)
            print()
            """

    devstr = '--filterarch ' + ','.join(uniq_devs)
    print(devstr)
    print_yaml({'workloads': ttsim_wlspec}, opath / tensix_workloads_yaml_file)

    runcfg_dict = {
        'title': 'Metal Tensix Correlation',
        'study': sname,
        'odir': odir.as_posix(),
        'wlspec': (opath / tensix_workloads_yaml_file).as_posix(),
        'archspec': 'config/tt_wh.yaml',
        'wlmapspec': 'config/wl2archmapping.yaml',
        'filterarch': ','.join(uniq_devs),
        'dumpstatscsv': True,
    }
    print_yaml(runcfg_dict, opath / tensix_runcfg_file)

    cmd = ['python', 'polproj.py', '--config', (opath / tensix_runcfg_file).as_posix()]
    cmdstr = ' '.join(cmd)
    print(cmdstr)
    ret = subprocess.run(cmdstr, shell=True, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        logger.error('command "{}" failed with exit code {}', cmd, ret.returncode)
        return ret.returncode
    actual_scores = read_scores(opath / sname / 'SUMMARY' / 'study-summary.json')
    comparison = compare_scores(metal_ref_scores, actual_scores)
    with open(opath / 'correlation_result.csv', 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=comparison[0].keys())
        writer.writeheader()
        for row in comparison:
            writer.writerow(row)

    return 0


if __name__ == '__main__':
    exit(main())
