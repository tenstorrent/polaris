#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
import copy
import cProfile
import logging
import pickle
import time
import tracemalloc
from collections import defaultdict
from enum import Enum, auto
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from ttsim.config import TTSimHLWlDevRunPerfStats, TTSimHLRunSummary, get_arspec_from_yaml, get_wlmapspec_from_yaml, get_wlspec_from_yaml
from ttsim.front import onnx2graph
from ttsim.utils.common import get_ttsim_functional_instance, print_csv, str_to_bool

""" Polaris top-level executable. """

LOG   = logging.getLogger(__name__)
INFO  = LOG.info
DEBUG = LOG.debug

class OutputFormat(Enum):
    FMT_NONE = auto()
    FMT_YAML = auto()
    FMT_JSON = auto()
    FMT_PICKLE = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return OutputFormat['FMT_' + s.upper()]

    @property
    @lru_cache(4)
    def cname(self)->str:
        return self.name.replace('FMT_', '').lower()

class ReducedStats:
    # Type hints for instance attributes
    rsrc_bound: dict[Any, int]

    def __init__(self, _devname, _wlcls, _wlname, _wlinstance, _dev):
        self.devname      = _devname
        self.wlcls        = _wlcls
        self.wlname       = _wlname
        self.wlinstance   = _wlinstance
        self.device       = _dev

    def summarize(self, _stats, _guardband=0.25):
        self.freq_Mhz= _stats[0]['freq_MHz']
        self.bs      = _stats[0]['batch']

        #F1 : fields that scale with op repeat count
        #F2 : fields that are independent of op repeat count
        F1 = ['op_rpt_count', 'cycles', 'msecs', 'inParamCount', 'precision', 'rsrc_bnck']
        F2 = ['inActCount', 'outActCount', 'precision']

        L1 = [tuple([s[f] for f in F1]) for s in _stats]
        L2 = [tuple([s[f] for f in F2]) for s in _stats if not (s['removed'] or s['fused'])]
        BPE_TBL = {
                'FP64'  : 8,
                'FP32'  : 4,
                'TF32'  : 4,
                'FP16'  : 2,
                'BF16'  : 2,
                'FP8'   : 1,
                'INT32' : 4,
                'INT8'  : 1,
                }


        self.tot_cycles   = sum([r*c for r,c,m,p,b,rb in L1])
        self.tot_msecs    = sum([r*m for r,c,m,p,b,rb in L1])
        self.inParamCount = sum([r*p for r,c,m,p,b,rb in L1])
        self.inParamBytes = sum([r*p*BPE_TBL[b] for r,c,m,p,b,rb in L1])
        self.inActCount   = sum([ia                 for ia,oa,p in L2])
        self.outActCount  = sum([oa                 for ia,oa,p in L2])
        self.maxActCount  = max([ia+oa              for ia,oa,p in L2])
        self.inActBytes   = sum([ia*BPE_TBL[p]      for ia,oa,p in L2])
        self.outActBytes  = sum([oa*BPE_TBL[p]      for ia,oa,p in L2])
        self.maxActBytes  = max([(ia+oa)*BPE_TBL[p] for ia,oa,p in L2])

        #check if fits device memory...
        self.mem_size_GB   = (self.inParamBytes + self.maxActBytes) / 1024 / 1024 / 1024
        self.device_mem_GB = self.device.mem_size(units='GB')
        self.fits_device   = self.mem_size_GB <= self.device_mem_GB

        self.rsrc_bound   = defaultdict(int)
        for r,c,m,p,b,rb in L1:
            if rb != 'NA':
                self.rsrc_bound['rsrc_' + rb] += r*c

        #arch resource bottleneck stats...
        for r in ['COMP', 'MEM']:
            self.rsrc_bound['rsrc_' + r.lower()] /= self.tot_cycles

        sxrec = {
                'devname'      : self.devname,
                'freq_Mhz'     : self.freq_Mhz,
                'wlcls'        : self.wlcls,
                'wlname'       : self.wlname,
                'wlinstance'   : self.wlinstance,
                'bs'           : self.bs,
                'inParams'     : self.inParamCount,
                'inActs'       : self.inActCount,
                'outActs'      : self.outActCount,
                'maxActs'      : self.maxActCount,
                'inParamBytes' : self.inParamBytes,
                'inActBytes'   : self.inActBytes,
                'outActBytes'  : self.outActBytes,
                'maxActBytes'  : self.maxActBytes,
                'tot_cycles'   : self.tot_cycles,
                'tot_msecs'    : self.tot_msecs,
                'ideal_throughput'   : self.bs * 1000 / self.tot_msecs,
                'mem_size_GB'  : self.mem_size_GB,
                'device_memsize_GB': self.device_mem_GB,
                'fits_device'  : self.fits_device,
                'device_peak_bw_GBps': self.device.peak_bandwidth(),
                'device_peak_fp8_tflops': self.device.peak_flops('matrix', 'mac', 'fp8', mul_factor=2),
                }
        sxrec['perf_projection'] = (1 - _guardband) * sxrec['ideal_throughput'] #25% guardband for SW/Host overhead
        sxrec.update(self.rsrc_bound)

        return sxrec

class RangeArgument:
    def __init__(self, name, arg, range_type='add'):
        self.vals = []
        if arg is not None:
            argx = arg[0]
            assert len(argx) == 3, f"range-arg: {name} = {argx} needs 3 numbers to specify a range"
            start, end, step = argx[0], argx[1], argx[2]

            assert start != end, f"Illegal RangeArgument: {name} [start({start}) == end({end})]"
            if range_type == 'mul' and step == 1:
                assert False, f"Illegal RangeArgument: {name} is of 'mul' type, step cannot be 1"
            elif range_type == 'add' and step == 0:
                assert False, f"Illegal RangeArgument: {name} is of 'add' type, step cannot be 0"
            else:
                pass

            if start > end:
                start, end, step = argx[1], argx[0], -1 * argx[2]
            assert start >= 1, f"range-arg: {name} = {argx} start cannot be < 1"

            assert range_type in ['add', 'mul'], \
                    f"range-arg: {name}, range_type({range_type}) can only be (add|mul)"
            x = start
            while (x <= end):
                self.vals.append(x)
                x = x + step if range_type == 'add' else x * step
            if self.vals[-1] < end:
                self.vals.append(end)
            assert len(self.vals) > 0, f"range-arg: {name} = {arg} specifies an empty range"

    def check(self):
        return len(self.vals) > 0

    def getvals(self):
        return self.vals

def check_args(args):
    assert args.inference != args.training, \
            f"Cannot run inference({args.inference}) & training({args.training}) together"
    return

def apply_filter(L, filter_csv_str, get_param_func):
    if filter_csv_str is not None:
        filter_fields = filter_csv_str.split(',')
        L = [x for x in L if get_param_func(x) in filter_fields]
    return L

def get_wlgraph(TBL, wlg, wln, wli, gcfg, wpath, enable_memalloc):
    xrows = [xrec for xrec in TBL if xrec[0] == wlg and xrec[1] == wln and xrec[2] == wli]
    wlb  = gcfg['bs']
    num_xrows = len(xrows)
    if num_xrows == 1 and xrows[0][3] is None:
        # we did not have the workload batch-size when we created TBL because batchsweep
        # was not set; now that the workload is instantiated, we have the batch-size
        # so, update the TBL accordingly
        del TBL[(wlg, wln, wli, None)]
        TBL[(wlg, wln, wli, wlb)] = None
    else:
        assert (wlg, wln, wli, wlb) in TBL, \
            f"Workload= {wlg}.{wln}.{wli}.b{wlb} missing in workloads graph table!!"

    if TBL[(wlg,wln,wli,wlb)] is None:
        if wlg == 'TTSIM':
            ttsim_wl  = get_ttsim_functional_instance(wpath, wln, gcfg) #<--- This is slow...

            if enable_memalloc:
                tracemalloc.start()
            ttsim_wl.create_input_tensors()
            ttsim_wl_out   = ttsim_wl() # noqa: F841 # we execute the graph and all the nodes are well formed
            if enable_memalloc:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics("lineno")
                for stat in top_stats[:10]:
                    print("TRACEMALLOC:", stat)
                print("TRACEMALLOC: ttsim", "="*50, "\n")

            ttsim_wl_graph = ttsim_wl.get_forward_graph() # we should have a valid workload graph at this point
            TBL[(wlg,wln,wli,wlb)] = (ttsim_wl, ttsim_wl_graph)
            DEBUG(f">>ttsim-wl analytical parameter count {wlg}.{wln}.{wli}.b{wlb}= {ttsim_wl.analytical_param_count():,d}")
        elif wlg == 'ONNX':
            print(f">>onnx-wl = {wlg}.{wln}.{wli}.b{wlb} = {wpath}")
            onnx_graph = onnx2graph(wli, wpath)
            TBL[(wlg,wln,wli,wlb)] = (None, onnx_graph)
            for _,op in onnx_graph._ops.items():
                itensors = [onnx_graph._tensors[x] for x in op.inList]
                otensors = [onnx_graph._tensors[x] for x in op.outList]
                print("CALLING get_perf_counts for:", op.name)
                #for x in itensors: print(x)
                #for x in otensors: print(x)
                op.get_perf_counts(itensors, otensors)
                op.update_tensor_counts(itensors,otensors)
                #print(op.perf_stats)
        else:
            assert False, f"Workload Group: {wlg} is not supported; Current Support only for (TTSIM,ONNX)"

    return TBL[(wlg,wln,wli,wlb)]

def do_instr_profile(_WLG, _ODIR):
    ITBL: dict[tuple[str, str, str, int], Any] = {}
    for _wlx, (wlobj, wlgraph) in _WLG.items():
        if _wlx not in ITBL: ITBL[_wlx] = {}  # noqa: E701
        for i,x in enumerate(wlgraph.get_ordered_nodes()):
            fwd_op = wlgraph._ops[x]
            for k,v in fwd_op.perf_stats['instrs'].items():
                if k not in ITBL[_wlx]: ITBL[_wlx][k] = 0  # noqa: E701
                ITBL[_wlx][k] += v

    profile_data = []
    for wlg, wln, wli, wlb in ITBL:
        instr_tbl =  ITBL[(wlg, wln, wli, wlb)]
        for instr, count in instr_tbl.items():
            profile_data.append({
                'group'      : wlg,
                'workload'   : wln,
                'instance'   : wli,
                'batchsize'  : wlb,
                'instruction': instr,
                'count'      : count
                })
    instr_profile_file = _ODIR / 'workload_instruction_profile.csv'
    print_csv(profile_data[0].keys(), profile_data, instr_profile_file)

def setup_cmdline_args():
    logging_levels = [ 'debug', 'info', 'warning', 'error', 'critical' ]
    data_types     = [ 'fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8', 'int32', 'int8' ]  # noqa: F841
    parser = argparse.ArgumentParser('polaris')

    parser.add_argument('--dryrun',    '-n', action='store_true', default=False, help='show but do not run')
    parser.add_argument('--instr_profile',   action='store_true', default=False, help='Collect Instruction Profile for Workloads')
    parser.add_argument('--dump_ttsim_onnx', action='store_true', default=False, help='Dump ONNX graph for TTSIM Workload')
    parser.add_argument('--enable_memalloc', action='store_true', default=False, help='Enable Memory Allocation Stats')
    parser.add_argument('--enable_cprofile', action='store_true', default=False, help='Enable CProfiler Stats ')

    parser.add_argument('--training',  '-t', type=str_to_bool, default='false', help='Training run')
    parser.add_argument('--inference', '-i', type=str_to_bool, default='true',  help='Inference run')
    parser.add_argument('--log_level', '-l', type=str,         default='info',  help="set logging level", choices=logging_levels)

    parser.add_argument('--odir',      '-o', required=True, help="Output Directory Name")
    parser.add_argument('--study',     '-s', required=True, help="Study Name")
    parser.add_argument('--wlspec',    '-w', required=True, help="Workloads Specification")
    parser.add_argument('--archspec',  '-a', required=True, help="Architecture Specification")
    parser.add_argument('--wlmapspec', '-m', required=True, help="Workload To Architecture Mapping Specification")


    parser.add_argument('--filterwlg',  default=None, help="use only workload-groups specified in filterwlg (comma sep list)")
    parser.add_argument('--filterwl',   default=None, help="use only workloads specified in filterwl (comma sep list)")
    parser.add_argument('--filterwli',  default=None, help="use only workload instances specified in filterwli (comma sep list)")
    parser.add_argument('--filterarch', default=None, help="use only architectures specified in filterarch (comma sep list)")
    parser.add_argument('--frequency', nargs=3, metavar=('start', 'end', 'step'), type=int,
                        action='append', help='frequency (in MHz) range specification (arith-seq)')
    parser.add_argument('--batchsize', nargs=3, metavar=('start', 'end', 'step'), type=int,
                        action='append', help='batchsize range specification (geom-seq)')
    parser.add_argument('--outputformat', choices=['none', 'yaml', 'json', 'pickle'], default='json', type=str.lower)
    parser.add_argument('--dumpstatscsv', dest='dump_stats_csv', action='store_true', 
                        default=False, help='Dump stats in CSV format')

    #cmdline args processing
    args = parser.parse_args()
    check_args(args)

    #set logging level...
    numeric_level = getattr(logging, args.log_level.upper(), None)
    assert isinstance(numeric_level, int), f'Invalid log level: {args.log}'
    logging_format = "%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s"
    logging.basicConfig(level=numeric_level, format=logging_format)

    #print cmdline for easy debug
    DEBUG("CMD=  python " + " ".join(sys.argv))

    #frequency/batchsize sweep...
    fsweep = RangeArgument('frequency', args.frequency)
    bsweep = RangeArgument('batchsize', args.batchsize, range_type='mul')

    #data types
    #dt = [['*', args.data_type.upper()]]
    #if args.override_data_type:
    #    dt += [[y.upper() for y in x.split(',')] for x in args.override_data_type.split('/')]

    return args, fsweep, bsweep

def dump_ttsim_onnx(TBL, _odir):
    onnx_dir = _odir / 'ONNX'
    os.makedirs(onnx_dir, exist_ok=True)

    #we pick the smallest batch to dump the ONNX...
    min_batchsize = min([wlb for _,_,_,wlb in TBL])
    for wlg, wln, wli, wlb in TBL:
        if wlb == min_batchsize:
            wlobj, wlgraph = TBL[(wlg, wln, wli, wlb)]
            onnx_ofilename = ".".join([wlg, wln, wli]) + f'.b{min_batchsize}' + '.onnx'
            wlgraph.graph2onnx(onnx_dir / onnx_ofilename)
    return

def get_devices(devspec, fsweep, filterarch):
    # Collect Device Specifications
    ipblks, devs = get_arspec_from_yaml(devspec)
    if fsweep.check():
        device_list = [(d,f) for d in devs for f in fsweep.getvals()]
    else:
        device_list = [(d,None) for d in devs]
    devlist = apply_filter(device_list, filterarch, lambda x: x[0])

    INFO(f'reading device specification {devspec}: found {len(devs):4d} #devices')
    if fsweep.check():
        INFO(f'reading frequency sweep {" "*26}: found {len(fsweep.getvals()):4d} #frequencies')

    return devlist, devs

def get_workloads(wlspec, bsweep, filterwlg, filterwl, filterwli):
    # Collect Workload Specifications
    workload_specs = get_wlspec_from_yaml(wlspec)
    wl_list = []
    for wlname, wl in workload_specs.items():
        wlapi = wl.api
        for wli_name, wli_cfg in wl.get_instances().items():
            if bsweep.check():
                wl_list += [(wlapi, wl.name, wli_name, wli_cfg, b) for b in bsweep.getvals()]
            else:
                wl_list += [(wlapi, wl.name, wli_name, wli_cfg, None)]
    wl_list = apply_filter(wl_list, filterwlg, lambda x: x[0])
    wl_list = apply_filter(wl_list, filterwl,  lambda x: x[1])
    wl_list = apply_filter(wl_list, filterwli, lambda x: x[2])

    num_batches   = len(bsweep.getvals()) if bsweep.check() else 1
    num_workloads = len(wl_list) // num_batches
    INFO(f'reading workloads specification {wlspec}: found {num_workloads:4d} #devices')
    if bsweep.check():
        INFO(f'reading batch sweep                   : found {num_batches} #batch-sizes')
    return wl_list, workload_specs

def create_uniq_workloads_tbl(WL_LIST):
    TBL = {(wlg, wln, wli, wlb): None for wlg, wln, wli, _, wlb in WL_LIST}
    return TBL

def do_dryrun(_wl, _dl):
    ALL_EXPS = product(_wl, _dl)

    #set field widths for diagnostic prints
    DEVNF = max([len(x) for x,_ in _dl])            + 1
    DEVFF = max([len(f"{x}") for _,x in _dl])       + 1
    WLGF  = max([len(x) for x,_,_,_,_ in _wl])      + 1
    WLNF  = max([len(x) for _,x,_,_,_ in _wl])      + 1
    WLIF  = max([len(x) for _,_,x,_,_ in _wl])      + 1
    WLB   = max([len(f"{x}") for _,_,_,_,x in _wl]) + 1

    for exp_no, (exp_wl, exp_dev) in enumerate(ALL_EXPS):
        wlg, wln, wli, wlicfg, wlb = exp_wl
        devname, devfreq           = exp_dev
        #diagnostics
        xstr  = f"  ..exp:{exp_no:3d} dev:{devname:{DEVNF}s} freq:"
        xstr += f"{devfreq}" if devfreq is None else f"{devfreq:{DEVFF}d}"
        xstr += f" wlg:{wlg:{WLGF}s} wl:{wln:{WLNF}s}"
        xstr += f"wli:{wli:{WLIF}s} wlb:"
        xstr += f"{wlb}" if wlb is None else f"{wlb:{WLB}d}"
        INFO(xstr)


    return

def save_data(model: BaseModel, filename, outputfmt: OutputFormat)->None:
    if outputfmt == OutputFormat.FMT_NONE:
        return
    elif outputfmt == OutputFormat.FMT_YAML:
        with open(filename, 'w') as fout:
            yaml.dump(model.model_dump, fout, indent=4, Dumper=yaml.CDumper)
    elif outputfmt == OutputFormat.FMT_JSON:
        with open(filename, 'w') as fout:
            print(model.model_dump_json(indent=4), file=fout)
    elif outputfmt == OutputFormat.FMT_PICKLE:
        with open(filename, 'wb') as foutbin:
            pickle.dump(model, foutbin)

def execute_wl_on_dev(_wl, _dl, _wspec, _dspec, _op2dt, _op2rsrc, _null_ops, _op_fusion_list, _WLG,
                      _odir, study, _enable_memalloc, outputfmt, flag_dump_stats_csv):
    # TODO: Reduce number of arguments to this function
    study_dir = _odir / study
    stat_dir    = study_dir / 'STATS'
    config_dir  = study_dir / 'CONFIG'
    os.makedirs(stat_dir,    exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    saved_devices = set()
    _summary_stats = []
    job_summaries: list[Any] = []
    ALL_EXPS = product(_wl, _dl)
    for exp_no, (exp_wl, exp_dev) in enumerate(ALL_EXPS):
        wlgroup, wlname, wlins_name, wlins_cfg, wlbatch = exp_wl
        devname, devfreq                                = exp_dev

        dev_obj   = _dspec[devname]
        if devfreq is not None:
            dev_obj.set_frequency(devfreq) #override device frequency is we have freqsweep

        wlpath    = wlins_cfg['path']
        wlcfg     = wlins_cfg['cfg']
        if wlbatch is not None:
            wlcfg['bs'] = wlbatch #override batch_size if we have batchsweep

        try:
            wlobj, wlgraph = get_wlgraph(_WLG, wlgroup, wlname, wlins_name, wlcfg, wlpath,
                                         _enable_memalloc)
        except Exception as e:
            logging.error('workload %s failed with %s', exp_wl, e)
            raise

        wlgraph.set_precision (_op2dt)
        wlgraph.map_resources (_op2rsrc)
        wlgraph.execute       (dev_obj)
        wlgraph.remove_nodes  (_null_ops)
        wlgraph.fuse_nodes    (_op_fusion_list)

        #publish stats
        rows   = []
        model_rows = []
        for i,x in enumerate(wlgraph.get_ordered_nodes()):
            is_inode = x in wlgraph._input_nodes
            is_onode = x in wlgraph._output_nodes
            fwd_op   = wlgraph._ops[x]
            dev_freq_MHz = dev_obj.frequency(fwd_op.uses_compute_pipe, units='MHz')
            val    = {
                    'devname'       : devname,
                    'freq_MHz'      : dev_freq_MHz,
                    'pipe'          : fwd_op.uses_compute_pipe.upper(),
                    'precision'     : fwd_op.precision.upper(),
                    'wlgroup'       : wlgroup,
                    'wlname'        : wlname,
                    'wlinstance'    : wlins_name,
                    'batch'         : wlcfg['bs'],
                    'opnum'         : i,
                    'opname'        : x,
                    'is_input_node' : is_inode,
                    'is_output_node': is_onode,
                    'optype'        : fwd_op.optype,
                    'op_rpt_count'  : fwd_op.repeat_count,
                    'attrs'         : fwd_op.attrs,
                    'inList'        : fwd_op.inList,
                    'outList'       : fwd_op.outList,
                    'domain'        : fwd_op.domain,
                    'opclass'       : fwd_op.opclass_str,
                    'removed'       : fwd_op.removed_in_optimization,
                    'fused'         : fwd_op.fused_in_optimization,
                    'fused_with_op' : 'NA' if fwd_op.fused_with_op is None else fwd_op.fused_with_op
                    }
            val.update(fwd_op.perf_stats)
            TOT_INSTR_COUNT = sum([v for k,v in fwd_op.perf_stats['instrs'].items()])
            val.update({
                'instr_count'   : TOT_INSTR_COUNT,
                'compute_cycles': fwd_op.compute_cycles,
                'mem_rd_cycles' : fwd_op.mem_rd_cycles,
                'mem_wr_cycles' : fwd_op.mem_wr_cycles,
                'ramp_penalty'  : dev_obj.ramp_penalty()
                })

            if fwd_op.fused_op_cycles is None:
                compute_cycles = fwd_op.compute_cycles
                mem_cycles     = fwd_op.mem_rd_cycles + fwd_op.mem_wr_cycles
            else:
                compute_cycles = fwd_op.fused_op_cycles['compute_cycles']
                mem_cycles     = fwd_op.fused_op_cycles['mem_rd_cycles'] + fwd_op.fused_op_cycles['mem_wr_cycles']
            cycles = max(compute_cycles, mem_cycles) + dev_obj.ramp_penalty()
            msecs  = cycles / dev_freq_MHz / 1e3

            if fwd_op.removed_in_optimization or fwd_op.fused_in_optimization:
                rsrc_bnck = 'NA'
                cycles    = 0
                msecs     = 0.0
            elif compute_cycles >= mem_cycles:
                rsrc_bnck = 'COMP'.lower()
            else:
                rsrc_bnck = 'MEM'.lower()
            val.update({
                'rsrc_bnck' : rsrc_bnck,
                'cycles'    : cycles,
                'msecs'     : msecs
                })

            rows.append(val)
            opval = copy.deepcopy(val)
            for tmp in ['devname', 'freq_MHz', 'wlgroup', 'wlname', 'wlinstance', 'batch']:
                del opval[tmp]
            model_rows.append(opval)

        model_dict = {
            'devname': devname,
            'freq_MHz': dev_freq_MHz,
            'wlgroup': wlgroup,
            'wlname': wlname,
            'wlinstance': wlins_name,
            'batch': wlcfg['bs'],
            'operatorstats': model_rows
        }
        model = TTSimHLWlDevRunPerfStats(**model_dict)
        statF_parts  = [f"{devname}"]
        statF_parts += [] if devfreq is None else [f"f{devfreq}"]
        statF_parts += [f"{wlgroup}", f"{wlname}", f"{wlins_name}"]
        statF_parts += [] if wlbatch is None else [f"b{wlbatch}"]
        statF = "-".join(statF_parts) + '-opstats.csv'
        statP = stat_dir / statF
        if flag_dump_stats_csv:
            print_csv(rows[0].keys(), rows, statP)
        if outputfmt != OutputFormat.FMT_NONE:
            statyamlP = stat_dir / (statP.stem + '.' + outputfmt.cname)
            save_data(model, statyamlP, outputfmt)

            if devname not in saved_devices:
                devF = config_dir / f'{devname}.{outputfmt.cname}'
                save_data(dev_obj, devF, outputfmt)
                saved_devices.add(devname)

        reduced_stat  = ReducedStats(devname, wlgroup, wlname, wlins_name, dev_obj)
        summary_dict = reduced_stat.summarize(rows)
        if outputfmt != OutputFormat.FMT_NONE:
            summary_dict['stat_filename'] = statyamlP.relative_to(_odir).as_posix()
        else:
            summary_dict['stat_filename'] = ''
        _summary_stats.append(summary_dict)
        logging.info('ran job #%d %s %s %s', exp_no, wlins_name, devname, devfreq)

    return _summary_stats

def main() -> int:
    args, freqsweep, batchsweep = setup_cmdline_args()

    if args.enable_cprofile:
        profiler   = cProfile.Profile()
        profiler.enable()

    outputformat = OutputFormat.enumvalue(args.outputformat)
    #create outdir & output book-keeping assets
    odir        = Path(args.odir)
    studydir    = odir / args.study
    summary_dir = studydir / 'SUMMARY'
    os.makedirs(odir,        exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    device_list, devspec  = get_devices(args.archspec, freqsweep, args.filterarch)
    workload_list, wlspec = get_workloads(args.wlspec, batchsweep, args.filterwlg, args.filterwl, args.filterwli)
    OP2DT, OP2RSRC, NULL_OPS, OP_FUSION_LIST = get_wlmapspec_from_yaml(args.wlmapspec)

    if args.dryrun:
        do_dryrun(workload_list, device_list)
        tot_exp_run = 0
    else:
        workload_graphs = create_uniq_workloads_tbl(workload_list)

        INFO('simulation: workload+ --> device+')
        summary_stats = execute_wl_on_dev(workload_list, device_list, wlspec, devspec,
                                          OP2DT, OP2RSRC, NULL_OPS, OP_FUSION_LIST,
                                          workload_graphs, odir, args.study, args.enable_memalloc,
                                          outputformat, args.dump_stats_csv)

        summary_stat_filename = summary_dir / f'study-summary.{outputformat.cname}'
        save_data(TTSimHLRunSummary(**{'summary': summary_stats}), summary_stat_filename, outputformat)
        if args.dump_stats_csv:
            summary_stat_csv_filename = summary_dir / (summary_stat_filename.stem + '.csv')
            print_csv(summary_stats[0].keys(), summary_stats, summary_stat_csv_filename)


        if args.instr_profile:
            do_instr_profile(workload_graphs, odir)

        if args.dump_ttsim_onnx:
            dump_ttsim_onnx(workload_graphs, odir)

        tot_exp_run = len(summary_stats)

    if args.enable_cprofile:
        profiler.disable()
        profiler.dump_stats("polaris_cprofile_stats.prof")

    return tot_exp_run

if __name__ == '__main__':
    start_time = time.perf_counter()
    num_exps   = main()
    end_time   = time.perf_counter()
    del_time   = end_time - start_time

    if num_exps > 0:
        print(f"Completed {num_exps} jobs in {del_time:0.2f} seconds @ {num_exps/del_time:.0f} jobs per sec")
    else:
        print()
