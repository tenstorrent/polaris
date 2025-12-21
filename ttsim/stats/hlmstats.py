#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import copy
import pickle
from enum import Enum, auto
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Set, TypeVar

import numpy
import yaml
from loguru import logger
from pydantic import BaseModel

from ttsim.config import TTSimHLWlDevRunPerfStats
from ttsim.utils.common import print_csv
from ttsim.utils.types import get_bpe, get_sim_dtype

BaseModel_SubType = TypeVar('BaseModel_SubType', bound=BaseModel)

LOG     = logger
INFO    = LOG.info
DEBUG   = LOG.debug
ERROR   = LOG.error
WARNING = LOG.warning

# Threshold for numpy array size to determine truncation in logging/output
NUMPY_ARRAY_SIZE_THRESHOLD = 100

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

def process_numpy_attr(v: numpy.ndarray, op_index: int, opstats: Any, k: str) -> None:
    """Process a numpy array attribute for JSON serialization.

    Converts numpy arrays to descriptive strings containing shape, dtype, and value
    information (truncated for large arrays as needed)

    Args:
        v: The numpy array to process.
        op_index: The index of the operator in the model.
        opstats: The operator statistics object containing the attribute.
        k: The key of the attribute in opstats.attrs.
    """
    # Truncate large arrays for logging
    if v.size > NUMPY_ARRAY_SIZE_THRESHOLD:
        truncated = numpy.array2string(v, threshold=10, edgeitems=3)
        value_for_output = f"shape={v.shape}, dtype={v.dtype}, truncated: {truncated}"
    else:
        value_for_output = f"shape={v.shape}, dtype={v.dtype}, value={v.tolist()}"
    if opstats.optype in ['Constant', 'ConstantOfShape']:
        logger.warning(
            f"Unexpected numpy.ndarray value for operator op#{op_index} "
            f"(opname {opstats.opname}, optype {opstats.optype}): {k} = {value_for_output}"
        )
    opstats.attrs[k] = value_for_output

def prepare_model_for_json(model: BaseModel_SubType) -> BaseModel_SubType:
    """Prepare a Pydantic model for JSON serialization by handling numpy arrays.

    Checks for numpy arrays in operatorstats attributes and creates a deep copy
    if any are found to avoid mutating the original model. Then processes each
    numpy array to make it JSON-serializable.

    Args:
        model: The Pydantic BaseModel to prepare.

    Returns:
        The prepared model (original or deep copy) with numpy arrays converted.
    """
    if not hasattr(model, 'operatorstats'):
        return model
    has_numpy_arrays = any(
        isinstance(v, numpy.ndarray)
        for opstats in model.operatorstats
        if hasattr(opstats, 'attrs')
        for v in opstats.attrs.values()
    )
    if not has_numpy_arrays:
        return model
    model_to_dump = copy.deepcopy(model)
    if TYPE_CHECKING:
        assert hasattr(model_to_dump, 'operatorstats')
    for op_index, opstats in enumerate(model_to_dump.operatorstats):
        if hasattr(opstats, 'attrs'):
            for k, v in opstats.attrs.items():
                if isinstance(v, numpy.ndarray):
                    process_numpy_attr(v, op_index, opstats, k)
    return model_to_dump

def save_data(model: BaseModel, filename, outputfmt: OutputFormat)->None:
    if outputfmt == OutputFormat.FMT_NONE:
        return
    elif outputfmt == OutputFormat.FMT_YAML:
        with open(filename, 'w') as fout:
            # Note: model_dump is a method and must be called to get the model data.
            # Dumping model.model_dump without parentheses would dump the method object itself.
            yaml.dump(model.model_dump(), fout, indent=4, Dumper=yaml.CDumper)
    elif outputfmt == OutputFormat.FMT_JSON:
        # Handle any numpy arrays in attrs by converting to lists
        # This is needed because pydantic's model_dump_json does not
        # automatically convert numpy arrays to JSON-serializable types
        model_to_dump = prepare_model_for_json(model)
        with open(filename, 'w') as fout:
            print(model_to_dump.model_dump_json(indent=4), file=fout)
    elif outputfmt == OutputFormat.FMT_PICKLE:
        with open(filename, 'wb') as foutbin:
            pickle.dump(model, foutbin)

def format_tensor_for_stats(tensor) -> str:
    """Format a single tensor as name[dim1xdim2xdim3]:precision for stats output."""
    name = tensor.name
    shape = list(tensor.shape) if tensor.shape is not None else []
    # Get precision as string
    if hasattr(tensor.dtype, 'name'):
        precision = tensor.dtype.name.lower()
    elif isinstance(tensor.dtype, str):
        precision = tensor.dtype.lower()
    else:
        precision = str(tensor.dtype).lower()
    if shape:
        shape_str = 'x'.join(str(d) for d in shape)
        return f"{name}[{shape_str}]:{precision}"
    return f"{name}[]:{precision}"

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

    def _extract_tensor_info(self, op):
        """Extract tensor information (name and shape) for input, output, and weight tensors.

        Returns a dict with three keys (all strings in format: name[dim1xdim2]:precision;name2[dim1xdim2]:precision):
        - input_tensors: String representation of input tensors
        - output_tensors: String representation of output tensors
        - weight_tensors: String representation of weight/parameter tensors
        """
        input_parts = []
        output_parts = []
        weight_parts = []

        # Process input tensors
        for tensor_name in op.inList:
            if tensor_name in self.wlgraph._tensors:
                tensor = self.wlgraph._tensors[tensor_name]
                input_parts.append(format_tensor_for_stats(tensor))

                # If tensor is a parameter/weight, also add to weight_tensors
                if tensor.is_param:
                    weight_parts.append(format_tensor_for_stats(tensor))
            else:
                WARNING(
                    f"Tensor '{tensor_name}' not found in workload graph tensors for op '{op.name}'.",
                    once=True
                )

        # Process output tensors
        for tensor_name in op.outList:
            if tensor_name in self.wlgraph._tensors:
                tensor = self.wlgraph._tensors[tensor_name]
                output_parts.append(format_tensor_for_stats(tensor))
            else:
                WARNING(
                    f"Tensor '{tensor_name}' not found in workload graph tensors for op '{op.name}'.",
                    once=True
                )

        return {
            'input_tensors': ';'.join(input_parts),
            'output_tensors': ';'.join(output_parts),
            'weight_tensors': ';'.join(weight_parts)
        }

    def dump_stats(self, dfreq):
        summary_dict = self.device.get_exec_stats(self.wlgraph, self.batchsize)

        graph_ordered_nodes = self.wlgraph.get_ordered_nodes()

        opstats_tbl = []
        for opnum,opname in enumerate(graph_ordered_nodes):
            op  = self.wlgraph.get_op(opname)

            # Extract tensor shape information
            tensor_info = self._extract_tensor_info(op)

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
                    'input_tensors'    : tensor_info['input_tensors'],
                    'output_tensors'   : tensor_info['output_tensors'],
                    'weight_tensors'   : tensor_info['weight_tensors'],
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
