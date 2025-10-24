#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.utils.common import print_csv
from ttsim.config import TTSimHLWlDevRunPerfStats
from ttsim.utils.types import get_sim_dtype, get_bpe
import copy

from typing import Any, List, Optional, Dict, Set
from collections import defaultdict
from enum import Enum, auto
from functools import lru_cache
from pydantic import BaseModel
from loguru import logger
import math
import yaml
import pickle

LOG     = logger
INFO    = LOG.info
DEBUG   = LOG.debug
ERROR   = LOG.error
WARNING = LOG.warning

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

class HLMStats:
    def __init__(self, _dev, _wlgraph, _wlinfo, _sinfo):
        self.device                  = _dev
        # Note: For backward compatibility, Device stores architecture package name in 'devname'
        # and device instance name in 'name'. Here we map them to their output field names:
        self.archname                = _dev.devname  # Architecture package name (e.g., "Grendel", "Wormhole")
        self.devname                 = _dev.name     # Device instance name (e.g., "Q1_A1", "n150")
        self.devFreqMHz              = _dev.freq_MHz
        self.wlgraph                 = _wlgraph
        self.wlgroup                 = _wlinfo['wlg']
        self.wlname                  = _wlinfo['wln']
        self.wlinstance              = _wlinfo['wli']
        self.batchsize               = _wlinfo['wlb']
        self.flag_dump_stats_csv     = _sinfo['flag_dump_stats_csv']
        self.outputfmt               = _sinfo['outputfmt']
        self.stat_dir                = _sinfo['stat_dir']
        self.config_dir              = _sinfo['config_dir']
        self.odir                    = _sinfo['odir']
        self.saved_devices: Set[str] = _sinfo['saved_devices']

        self.check_precision()

        return

    def dump_stats(self, dfreq):
        summary_dict = self.device.get_exec_stats(self.wlgraph, self.batchsize)

        graph_ordered_nodes = self.wlgraph.get_ordered_nodes()

        opstats_tbl = []
        for opnum,opname in enumerate(graph_ordered_nodes):
            op  = self.wlgraph.get_op(opname)
            val = {
                    'archname'         : self.archname,
                    'devname'          : self.devname,
                    'freq_MHz'         : self.devFreqMHz,
                    'pipe'             : op.uses_compute_pipe.upper(),
                    'precision'        : op.precision.upper(),
                    'wlgroup'          : self.wlgroup,
                    'wlname'           : self.wlname,
                    'wlinstance'       : self.wlinstance,
                    'batch'            : self.batchsize,
                    'opnum'            : opnum,
                    'opname'           : opname,
                    'is_input_node'    : self.wlgraph.is_input_node(opname),
                    'is_output_node'   : self.wlgraph.is_output_node(opname),
                    'optype'           : op.optype,
                    'op_rpt_count'     : op.repeat_count,
                    'attrs'            : op.attrs,
                    'inList'           : op.inList,
                    'outList'          : op.outList,
                    'domain'           : op.domain,
                    'opclass'          : op.opclass_str,
                    'removed'          : op.removed_in_optimization,
                    'fused'            : op.fused_in_optimization,
                    'fused_with_op'    : 'NA' if op.fused_with_op is None else op.fused_with_op,
                    'inElems'          : op.perf_stats['inElems'],
                    'outElems'         : op.perf_stats['outElems'],
                    'inBytes'          : op.perf_stats['inBytes'],
                    'outBytes'         : op.perf_stats['outBytes'],
                    'instrs'           : op.perf_stats['instrs'],
                    'inParamCount'     : op.perf_stats['inParamCount'],
                    'inActCount'       : op.perf_stats['inActCount'],
                    'outActCount'      : op.perf_stats['outActCount'],
                    'instr_count'      : sum([v for k,v in op.perf_stats['instrs'].items()]),
                    'compute_cycles'   : op.compute_cycles,
                    'mem_rd_cycles'    : op.mem_rd_cycles,
                    'mem_wr_cycles'    : op.mem_wr_cycles,
                    }
            val.update(op.exec_stats)
            opstats_tbl.append(val)

        model_rows = copy.deepcopy(opstats_tbl)
        for rec in model_rows:
            for tmp in ['archname', 'devname', 'freq_MHz', 'wlgroup', 'wlname', 'wlinstance', 'batch']:
                del rec[tmp]

        model_dict = {
                'archname'     : self.archname,
                'devname'      : self.devname,
                'freq_MHz'     : self.devFreqMHz,
                'wlgroup'      : self.wlgroup,
                'wlname'       : self.wlname,
                'wlinstance'   : self.wlinstance,
                'batch'        : self.batchsize,
                'operatorstats': model_rows,
                }
        model = TTSimHLWlDevRunPerfStats(**model_dict)

        #dumps stats
        statF_parts  = [f"{self.devname}"]
        statF_parts += [] if dfreq is None else [f"f{dfreq}"]
        statF_parts += [f"{self.wlgroup}", f"{self.wlname}", f"{self.wlinstance}"]
        statF_parts += [] if self.batchsize is None else [f"b{self.batchsize}"]
        statF = "-".join(statF_parts) + '-opstats.csv'
        statP = self.stat_dir / statF

        if self.flag_dump_stats_csv:
            print_csv(opstats_tbl[0].keys(), opstats_tbl, statP)

        if self.outputfmt != OutputFormat.FMT_NONE:
            statyamlP = self.stat_dir / (statP.stem + '.' + self.outputfmt.cname)
            save_data(model, statyamlP, self.outputfmt)

            if self.devname not in self.saved_devices:
                devF = self.config_dir / f'{self.devname}.{self.outputfmt.cname}'
                save_data(self.device.simconfig_obj, devF, self.outputfmt)
                self.saved_devices.add(self.devname)
        else:
            statyamlP = None

        #collect and return dump summary stats
        final_summary_dict = {
                'archname'     : self.archname,
                'devname'      : self.devname,
                'freq_Mhz'     : self.devFreqMHz,
                'wlgroup'      : self.wlgroup,
                'wlname'       : self.wlname,
                'wlinstance'   : self.wlinstance,
                'bs'           : self.batchsize,
                }
        final_summary_dict.update(summary_dict)

        if self.outputfmt != OutputFormat.FMT_NONE:
            final_summary_dict['stat_filename'] = statyamlP.relative_to(self.odir).as_posix()
        else:
            final_summary_dict['stat_filename'] = ''

        return final_summary_dict

    def check_precision(self):
        _graph_ordered_nodes = self.wlgraph.get_ordered_nodes()
        for opnum, opname in enumerate(_graph_ordered_nodes):
            op = self.wlgraph.get_op(opname)
            val_in_bpe = op.perf_stats['inBytes'] // op.perf_stats['inElems']
            if (not op.removed_in_optimization) and val_in_bpe != get_bpe(get_sim_dtype(op.precision)):
                WARNING(
                        f"device={self.devname} workload={self.wlname} instance={self.wlinstance}" + \
                                f" optype={op.optype} opclass={op.opclass_str} input bpe mismatch:" + \
                                f" bytes/elems {val_in_bpe}  != operator precision {op.precision}" + \
                                f" bpe {get_bpe(get_sim_dtype(op.precision))}",
                    once=True
                )
        return
