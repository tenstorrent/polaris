#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
from ttsim.utils.common import parse_yaml, parse_csv
from ttsim.config import create_ipblock, create_package

class ArchInfo:
    def __init__(self, archyaml, refcsv):
        self.cfg_dict = parse_yaml(archyaml)
        self.ipblocks = {}
        for ip_type, ip_dict in self.cfg_dict['ipblocks'].items():
            for ip_name, ip_cfg in ip_dict.items():
                self.ipblocks[ip_name] = create_ipblock(ip_type, ip_name, ip_cfg)

        self.packages = {}
        for pkg_type, pkg_dict in self.cfg_dict['packages'].items():
            for pkg_name, pkg_cfg in pkg_dict.items():
                self.packages[pkg_name] = create_package(pkg_type, pkg_name, pkg_cfg, self.ipblocks)

        precisions = ['int4', 'int8', 'int32', 'bf16', 'fp16', 'tf32', 'fp32', 'fp64']
        self.stats_tbl = []
        for pn,po in self.packages.items():
            compute_units = po.ipgroups['compute'].num_units
            rec = {
                    'pkg_name' : pn,
                    'mem_units': po.ipgroups['memory'].num_units,
                    'mem_size' : po.mem_size(),
                    'mem_bw'   : po.peak_bandwidth(),
                    'mem_freq' : po.mem_frequency(),
                    'mem_ip'   : po.ipgroups['memory'].ip.upper(),
                    }
            compute_units = po.ipgroups['compute'].num_units
            rec['compute_units'] = compute_units
            for pipe_name, pipe_obj in po.ipgroups['compute'].ipobj.pipes.items():
                rec[pipe_name + '.systolic_depth'] = pipe_obj.systolic_depth
                rec[pipe_name + '.freq']           = pipe_obj.frequency()

            for pipe_name in ['matrix', 'vector']:
                pipe_obj = po.ipgroups['compute'].ipobj.pipes[pipe_name]
                for prec in precisions:
                    try:
                        rec['tot.' + pipe_name + '.mac.' + prec.upper() + '.ipc']   = \
                                po.peak_ipc(pipe_name, 'mac', prec)
                        rec['tot.' + pipe_name + '.mac.' + prec.upper() + '.flops'] = \
                                po.peak_flops(pipe_name, 'mac', prec, mul_factor=2)
                        rec[pipe_name + '.mac.' + prec.upper() + '.ipc']            = \
                                pipe_obj.peak_ipc('mac', prec) / pipe_obj.num_units / pipe_obj.systolic_depth
                    except KeyError:
                        print(f'precision {prec} not supported for pipe {pipe_name}')

            self.stats_tbl.append(rec)

        rows, cols = parse_csv(refcsv)
        self.ref_stats = {}
        for r in rows:
            self.ref_stats[r['stat']] = r



def pytest_generate_tests(metafunc):
    if 's' in metafunc.fixturenames:
        archinfo = ArchInfo('config/all_archs.yaml','config/nvidia_ref_metrics.csv')
        results = []
        for x in archinfo.stats_tbl:
            devname  = x['pkg_name']
            print(devname)
            if devname.startswith('Q'): continue
            for s,v in x.items():
                print(s)
                if s in ['pkg_name']:
                    continue
                if s not in archinfo.ref_stats:
                    assert False, f"stat: {s} not found in nvidia_ref_stats"
                else:
                    # print(s, v)
                    # test_count += 1
                    print(f'Check {devname}.{s}.....')
                    rx = archinfo.ref_stats[s]
                    rv = rx[devname]

                    if s in ['mem_ip']:
                        match = (v == rv)
                    else:
                        rv = float(rv)
                        if rv == 0.0:
                            match = (v == rv)
                        else:
                            match = abs(v - rv)/rv < 0.1
                    results.append([s, v, rv, match])

        metafunc.parametrize('s,v,rv,match', results)


# TODO: After fixing failures, rename next function to remove wip_
def wip_test_reference_match(s, v, rv, match):
    assert match, f'{s}: {v} != expected {rv}'
