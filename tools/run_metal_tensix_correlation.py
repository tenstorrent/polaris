#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from ttsim.utils.common import parse_yaml, print_yaml

type ScoreTuple = tuple[str, str, str, str]
type ScoreDict = dict[ScoreTuple, dict[str, float]]


OUTPUT_DIR: Path = Path('__TMP_TENSIX_METAL_CORR_OUT')


def read_scores(filepath: Path, default_precision) -> ScoreDict:
    actual_scores: ScoreDict = dict()
    logger.info('===============================================')
    logger.info('Actual scores from {}', filepath)
    with open(filepath) as fin:
        actual_results = json.load(fin)['summary']
        for actual_res in actual_results:
            actual_key = tuple([actual_res['wlname'], actual_res['devname'], actual_res['wlcls'], actual_res['wlinstance']])
            actual_scores[actual_key] = {'ideal projection': actual_res['ideal_throughput'],
                                         'projection': actual_res['perf_projection'],
                                         'precision': default_precision
            }
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
    for key in sorted(common_keys):
        ref_score = ref_scores[key]['perf']
        ref_target_score = ref_scores[key]['target_perf']
        ref_precision = ref_scores[key]['precision']
        projection_precision = actual_scores[key]['precision'] if actual_scores[key]['precision'] is not None else ref_precision
        projected_score = actual_scores[key]['projection']
        projected_ideal_score = actual_scores[key].get('ideal projection', projected_score)
        ratio_ideal_to_score = projected_ideal_score / ref_score if ref_score != 0 else None
        ratio_ideal_to_target = projected_ideal_score / ref_target_score if ref_target_score != 0 else None
        ratio_score_to_score = projected_score / ref_score if ref_score != 0 else None
        ratio_score_to_target = projected_score / ref_target_score if ref_target_score != 0 else None
        result.append(
            {
                'Workload': key[0],
                'Arch': key[2],
                'Instance': key[3],
                'Api': key[1],
                'Tensix-Precision': ref_precision,
                'Tensix-Ref-Score': ref_score,
                'Tensix-Target-Score': ref_target_score,
                'Projection-Precision': projection_precision,
                'Projected-Score': projected_score,
                'Projected-Ideal-Score': projected_ideal_score,
                'Ratio-Ideal-Score-to-Ref': ratio_ideal_to_score,
                'Ratio-Ideal-Score-to-Target': ratio_ideal_to_target,
                'Ratio-Score-to-Ref': ratio_score_to_score,
                'Ratio-Score-to-Target': ratio_score_to_target,
                'Diff-Ideal-Score-to-Ref': projected_ideal_score - ref_score,
                'Diff-Ideal-Score-to-Target': projected_ideal_score - ref_target_score,
                'Diff-Score-to-Ref': projected_score - ref_score,
                'Diff-Score-to-Target': projected_score - ref_target_score,
            }
        )
    return result

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Tensix Metal correlation')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR.as_posix(),
                        help='Output directory for the results')
    parser.add_argument('--precision', type=str,
                        help='Precision to use for the correlation')
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    tensix_workloads_yaml_file: str = 'tensix_workloads.yaml'
    tensix_runcfg_file: str = 'tensix_runcfg.yaml'
    odir: Path = Path(args.output_dir)
    default_precision: str = args.precision
    sname: str = 'SIMPLE'
    tensix_perf_data_dir: Path = Path('data/metal/inf')  # Tensix metal results
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
            target_perf = tensix_cfg.get('target_perf', perf)  # Optional, use perf if not present
            system = tensix_cfg['system']
            prec = tensix_cfg['precision'] if default_precision is None else default_precision
            metric = tensix_cfg['metric']
            ref_perf = perf / num_gpu / nodes
            ref_target_perf = target_perf / num_gpu / nodes
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
                'target_perf': target_perf,
                'system': system,
                'prec': prec,
                'metric': metric,
                'ref_perf': ref_perf,
                'gpu_dev': gpu_dev,
                'instances': {instance_name: {'bs': bs}},
                # 'cp_streams'  : cp_streams,
                # 'inf_streams' : inf_streams,
            }
            instance_key = tuple([xrec['name'], xrec['gpu_dev'], xrec['api'], instance_name])
            if instance_key in metal_ref_scores:
                raise ValueError(f'Duplicate Instance key {instance_key} in {data_file}')
            metal_ref_scores[instance_key] = {'perf': ref_perf, 'target_perf': ref_target_perf, 'precision': tensix_cfg['precision']}

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

    print_yaml({'workloads': ttsim_wlspec}, opath / tensix_workloads_yaml_file)

    # TODO: Implement a way to pass default precision from CLI to polproj / polaris
    #       And then remove choosing wlmapspec based on default precision

    wlmapspec = 'config/' + ('wl2archmapping.yaml' if default_precision is None else f'wl2archmapping_{default_precision}.yaml')
    runcfg_dict = {
        'title': 'Metal Tensix Correlation',
        'study': sname,
        'odir': odir.as_posix(),
        'wlspec': (opath / tensix_workloads_yaml_file).as_posix(),
        'archspec': 'config/tt_wh.yaml',
        'wlmapspec': wlmapspec,
        'filterarch': ','.join(uniq_devs),
        'dump_stats_csv': True,
    }
    print_yaml(runcfg_dict, opath / tensix_runcfg_file)

    cmd = ['python', 'polproj.py', '--config', (opath / tensix_runcfg_file).as_posix()]
    cmdstr = ' '.join(cmd)
    logger.info('executing {}', cmdstr)
    ret = subprocess.run(cmdstr, shell=True, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        logger.error('command "{}" failed with exit code {}', cmd, ret.returncode)
        return ret.returncode
    actual_scores = read_scores(opath / sname / 'SUMMARY' / 'study-summary.json', default_precision)
    comparison = compare_scores(metal_ref_scores, actual_scores)
    with open(opath / 'correlation_result.csv', 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=comparison[0].keys())
        writer.writeheader()
        for row in comparison:
            writer.writerow(row)

    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
