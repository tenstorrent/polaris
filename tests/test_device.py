#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from ttsim.utils.common import parse_csv
from ttsim.config import get_arspec_from_yaml

class ArchInfo:
    def __init__(self, archyaml, refcsv):
        self.ipblocks, self.packages = get_arspec_from_yaml(archyaml)

        precisions = ['int4', 'int8', 'int32', 'bf16', 'fp16', 'tf32', 'fp32', 'fp64']
        self.stats_tbl = []
        for pn, package in self.packages.items():
            unsupported_precisions: dict[str, set] = {}
            pmemory = package.get_ipgroup('memory')
            pcompute = package.get_ipgroup('compute')
            rec = {
                    'pkg_name' : pn,
                    'mem_units': pmemory.num_units,
                    'mem_size' : package.mem_size(),
                    'mem_bw'   : package.peak_bandwidth(),
                    'mem_freq' : package.mem_frequency(),
                    'mem_ip'   : pmemory.ipname.upper(),
                    }
            compute_units = pcompute.num_units
            rec['compute_units'] = compute_units
            for pipe_obj in pcompute.ipobj.pipes:
                pipe_name = pipe_obj.name
                rec[pipe_name + '.systolic_depth'] = pipe_obj.systolic_depth
                rec[pipe_name + '.freq']           = pipe_obj.frequency()

            for pipe_name in ['matrix', 'vector']:
                pipe_obj = pcompute.ipobj.get_pipe(pipe_name)
                for prec in precisions:
                    try:
                        rec['tot.' + pipe_name + '.mac.' + prec.upper() + '.ipc']   = \
                                package.peak_ipc(pipe_name, 'mac', prec)
                        rec['tot.' + pipe_name + '.mac.' + prec.upper() + '.flops'] = \
                                package.peak_flops(pipe_name, 'mac', prec, mul_factor=2)
                        rec[pipe_name + '.mac.' + prec.upper() + '.ipc']            = \
                                pipe_obj.peak_ipc('mac', prec) / pipe_obj.num_units / pipe_obj.systolic_depth
                    except (KeyError, AssertionError):
                        try:
                            unsupported_precisions[pipe_name].add(prec)
                        except KeyError:
                            unsupported_precisions[pipe_name] = {prec}

            self.stats_tbl.append(rec)

            if unsupported_precisions:
                for pipe, precs in unsupported_precisions.items():
                    print(f'{pn} Pipe {pipe}, unsupported precisions {precs}')
        rows, cols = parse_csv(refcsv)
        self.ref_stats = {}
        for r in rows:
            self.ref_stats[r['stat']] = r


# # TODO: After fixing failures, rename next function to remove wip_


def wip_test_device():
    archinfo = ArchInfo('config/all_archs.yaml','config/nvidia_ref_metrics.csv')
    results = []
    for x in archinfo.stats_tbl:
        devname  = x['pkg_name']
        if devname.startswith('Q'): continue
        for s,v in x.items():
            if s in ['pkg_name']:
                continue
            if s not in archinfo.ref_stats:
                assert False, f"stat: {s} not found in nvidia_ref_stats"
            else:
                rx = archinfo.ref_stats[s]
                rv = rx[devname]

                if s in ['mem_ip']:
                    match = v == rv
                else:
                    rv = float(rv)
                    if rv == 0.0:
                        match = v == rv
                    else:
                        match =  abs(v - rv)/rv < 0.1
                results.append([s, v, rv, match])

    failures = [row for row in results if not row[-1]]
    if failures:
        for row in failures:
            s, v, rv, match = row
            print(f'{s=} {v=} {rv=} failure')
    print(f'{len(failures)} failures out of {len(results)}')
    assert not failures

