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


OUTPUT_DIR: Path = Path('__TMP_NV_MLPERF_CORR_OUT')


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
    logger.info('MLPerf NV Correlation Results')
    for key in ref_scores:
        ref_score = ref_scores[key]
        actual = actual_scores[key]
        ratio = ref_score / actual
        result.append(
            {
                'Arch': key[0],
                'Workload': key[2],
                'Instance': key[3],
                'Api': key[1],
                'MLPerf-Score': ref_score,
                'Actual-Score': actual,
                'Diff': ref_score - actual,
                'Ratio': ratio,
            }
        )
    return result

def main() -> int:
    nv_workloads_yaml_file: str = 'nv_workloads.yaml'
    nv_runcfg_file: str         = 'nv_runcfg.yaml'
    odir: Path = OUTPUT_DIR
    sname: str = 'SIMPLE'
    nv_perf_data_dir: Path       =  Path('data/mlperf/inf/closed') #MLPerf v4.1 results
    gpu_dev_tbl = {
            'NVIDIA H200-SXM-141GB': 'H200-SXM5',
            }

    uniq_devs    = set()
    ttsim_wlspec = []

    mlperf_ref_scores: ScoreDict = dict()

    opath                  = Path(odir)
    os.makedirs(opath, exist_ok=True)

    wls = ['bert', 'resnet50']
    for wl in wls:
        data_file = nv_perf_data_dir / ('nv_perf_metrics_' + wl + '.yaml')
        data_obj  = parse_yaml(data_file)
        for nv_cfg in data_obj:
            if nv_cfg['scenario'] != 'Scenario.Offline':
                continue
            scenario    = nv_cfg['scenario']
            benchmark   = nv_cfg['benchmark']
            gpu         = nv_cfg['gpu']
            bs          = nv_cfg['gpu_batch_size'][wl]
            nodes       = nv_cfg['nodes']
            num_gpu     = nv_cfg['num gpu']
            perf        = nv_cfg['perf']
            system      = nv_cfg['sys']
            prec        = nv_cfg['precision']
            metric      = nv_cfg['metric']
            cp_streams  = nv_cfg['gpu_copy_streams']
            inf_streams = nv_cfg['gpu_inference_streams']
            ref_perf    = perf/num_gpu/nodes
            gpu_dev     = gpu_dev_tbl[gpu]

            instance_name = f'b{bs}'
            xrec = {
                'api'         : 'TTSIM',
                'basedir'     : 'workloads',
                'scenario'    : scenario,
                'benchmark'   : benchmark,
                'name'        : wl,
                'gpu'         : gpu,
                'nodes'       : nodes,
                'num_gpu'     : num_gpu,
                'perf'        : perf,
                'sys'         : system,
                'prec'        : prec,
                'metric'      : metric,
                'cp_streams'  : cp_streams,
                'inf_streams' : inf_streams,
                'ref_perf'    : ref_perf,
                'gpu_dev'     : gpu_dev,
                'instances'   : { instance_name: { 'bs': bs } },
            }
            instance_key = tuple([xrec['gpu_dev'], xrec['api'], xrec['name'], instance_name])
            if instance_key in mlperf_ref_scores:
                raise AssertionError(f"Duplicate MLPerf key {instance_key} in {data_file}")
            mlperf_ref_scores[instance_key] = ref_perf

            if wl == 'bert':
                seqlen = nv_cfg['bert_opt_seqlen']
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
                raise NotImplementedError(f'Unknown workload {wl} in {data_file}')
            uniq_devs.add(gpu_dev)
            ttsim_wlspec.append(xrec)

            '''
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
            '''

    devstr = '--filterarch ' + ','.join(uniq_devs)
    print(devstr)
    print_yaml({'workloads': ttsim_wlspec}, opath / nv_workloads_yaml_file)

    runcfg_dict = {
        'title': 'MLPerf NV Correlation',
        'study': sname,
        'odir': odir.as_posix(),
        'wlspec': (opath / nv_workloads_yaml_file).as_posix(),
        'archspec': 'config/all_archs.yaml',
        'wlmapspec': 'config/wl2archmapping.yaml',
        'filterarch': ','.join(uniq_devs),
        'dump_stats_csv': True,
    }
    print_yaml(runcfg_dict, opath / nv_runcfg_file)

    cmd = ['python', 'polproj.py', '--config', (opath / nv_runcfg_file).as_posix()]
    cmdstr = ' '.join(cmd)
    print(cmdstr)
    ret = subprocess.run(cmdstr, shell=True, stderr=subprocess.STDOUT)
    if ret.returncode != 0:  # pragma: no cover
        logger.error('command "{}" failed with exit code {}', cmd, ret.returncode)
        return ret.returncode
    actual_scores = read_scores(opath / sname / 'SUMMARY' / 'study-summary.json')
    comparison =  compare_scores(mlperf_ref_scores, actual_scores)
    with open(opath / 'correlation_result.csv', 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=comparison[0].keys())
        writer.writeheader()
        for row in comparison:
            writer.writerow(row)

    return 0


if __name__ == '__main__':
    exit(main())
