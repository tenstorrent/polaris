#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ttsim.utils.common import parse_yaml, print_yaml
from pathlib import Path
import subprocess
import yaml

if __name__ == '__main__':
    nv_workloads_yaml_file = 'nv_workloads.yaml'
    odir                   = '__TMP_NV_MLPERF_CORR_OUT'
    sname                  = 'SIMPLE'
    nv_perf_data_dir       =  Path('data/mlperf/inf/closed') #MLPerf v4.1 results
    gpu_dev_tbl = {
            'NVIDIA H200-SXM-141GB': 'H200-SXM5',
            }

    uniq_devs    = set()
    ttsim_wlspec = []

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
            sys         = nv_cfg['sys']
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
                    'sys'         : sys,
                    'prec'        : prec,
                    'metric'      : metric,
                    'cp_streams'  : cp_streams,
                    'inf_streams' : inf_streams,
                    'ref_perf'    : ref_perf,
                    'gpu_dev'     : gpu_dev,
                    'instances'   : { instance_name: { 'bs': bs } }
                    }

            if wl == 'bert':
                seqlen = nv_cfg['bert_opt_seqlen']
                xrec['module'] = 'BasicLLM@BasicLLM.py'
                xrec['instances'][instance_name].update({
                        'nL': 24, 'nH': 16, 'dE': 1024,
                        'nW':  seqlen,
                        'vocab_sz': 30522
                        })
            elif wl == 'resnet50':
                xrec['module'] = 'ResNet@basicresnet.py'
                xrec['instances'][instance_name].update({
                        'layers': [3, 4, 6, 3],
                        'num_classes': 1000,
                        'num_channels': 3,
                        })
            else:
                pass
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

    devstr = "--filterarch " + ",".join(uniq_devs)
    print(devstr)
    print_yaml({'workloads': ttsim_wlspec}, nv_workloads_yaml_file)
    cmd  = ['python polaris.py',
            '-a config/all_archs.yaml',
            '-m config/wl2archmapping.yaml']
    cmd += [ f'-w {nv_workloads_yaml_file}', f'-o {odir} -s {sname}']
    cmd += [f'{devstr}']
    cmdstr = " ".join(cmd)
    print(cmdstr)
    ret = subprocess.run(cmdstr, shell=True, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        print('command "%s" failed with exit code %d' % (cmd, ret.returncode))
