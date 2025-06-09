#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import logging
import re
import os
from copy import deepcopy
from typing import Any, Type, Optional
import yaml
import sys
from pydantic import BaseModel, model_validator, ValidationError
from typing import Optional, List, Union, NewType, TYPE_CHECKING
from collections import Counter
import deepdiff
from ttsim.utils.common import convert_units, warnonce

LOG   = logging.getLogger(__name__)
INFO  = LOG.info
DEBUG = LOG.debug


class SimConfig:
    def __init__(self, cfg):
        #currently only supports dicts and scalars
        for k, v in cfg.items():
            setattr(self, k, SimConfig(v) if isinstance(v, dict) else v)

    def __iter__(self):
        for i in vars(self).keys():
            yield i


class XlsxConfig:
    def __init__(self, cfgname):
        self.values         = dict()
        self.cfgname        = cfgname
        self.defaulted_keys = set()

    @staticmethod
    def normalize_param(param):
        return re.sub('[ ()/\\[\\].-]', '', param.strip()).lower()

    def set_value(self, name, value):
        k = XlsxConfig.normalize_param(name)
        if k in self.values and self.values[k] != value:
            raise RuntimeError(
                f'name "{name}"="{value}", already defined as {self.values[k]}; name normalized as "{k}"')
        self.values[k] = value

    def value(self, name, defvalue=None):
        try:
            return self.values[XlsxConfig.normalize_param(name)]
        except KeyError:
            if defvalue is None:
                raise AttributeError(f'undefined value for "{name}" in config "{self.cfgname}"')
            self.defaulted_keys.add(name)
            return defvalue


class SimCfgBlk: #Generic Sim Configuration Block
    def __init__(self, blkname, **kwargs):
        self.name = blkname

        required_fields = getattr(self, 'required_fields', [])
        optional_fields = getattr(self, 'optional_fields', {})
        missing_fields  = [f for f in required_fields if f not in kwargs]
        if len(missing_fields) > 0:
            raise ValueError(f"Missing required_fields: {missing_fields}")

        #update required_fields
        for f in required_fields:
            setattr(self, f, kwargs[f])

        #update optional fields, may need default values
        for f, dv in optional_fields.items():
            setattr(self, f, kwargs.get(f, dv))

    def kind(self):
        return type(self).__name__

    def set_param(self, pname, pvalue):
        if len(pname) == 0:
            INFO(f"??? CHECK THIS in {self.kind()}.{self.name}.set_param()")
            return False

        parts = pname.split('.')
        if len(parts) == 1:
            attr = parts[0]
            if hasattr(self, attr):
                setattr(self, attr, pvalue)
                return True
            else:
                return False

        cur_attr, next_pname = parts[0], ".".join(parts[1:])

        #search the obj hierachy for name match. child objects may be SimCfgBlk or dict
        cur_child_obj = getattr(self, cur_attr)
        cur_child_pos = 1
        while not isinstance(cur_child_obj, SimCfgBlk) and cur_child_pos < len(pname):
            if isinstance(cur_child_obj, dict):
                if '.' in next_pname:
                    cur_attr       = parts[cur_child_pos]
                    cur_child_obj  = cur_child_obj[cur_attr]
                    cur_child_pos += 1
                    next_pname     = ".".join(parts[cur_child_pos:])
                elif next_pname in cur_child_obj:
                    old_val      = cur_child_obj[next_pname]
                    old_val_type = type(old_val)
                    new_val_type = type(pvalue)
                    if old_val_type == new_val_type:
                        cur_child_obj[next_pname] = pvalue
                        return True
                    else:
                        INFO(f"WARNING: Type Mismatch: old({old_val}) != new({pvalue}), IGNORED!!")
                        return False
                else:
                    return False
            else:
                assert False, "WAAAAAH"

        return cur_child_obj.set_param(next_pname, pvalue)

    def __str__(self):

        def str_helper(obj, indent=0):
            lines = []
            prefix = ' ' * indent
            if isinstance(obj, SimCfgBlk):
                lines.append(f"{prefix}{obj.kind()}({getattr(obj, 'name', 'UNK')}):")
                for key, val in obj.__dict__.items():
                    assert not isinstance(val, list), f"List[SimCfgBlk] not supported yet!!"
                    if isinstance(val, dict):
                        lines.append(f"{prefix}  {key}:")
                        for item_name, item_val in val.items():
                            lines.append(f"{prefix}    {item_name}:")
                            lines.append(str_helper(item_val, indent + 6))
                    elif isinstance(val, SimCfgBlk):
                        lines.append(f"{prefix}  {key}:")
                        lines.append(str_helper(val, indent + 4))
                    else:
                        lines.append(f"{prefix}  {key}: {val}")
            elif isinstance(obj, dict):
                for ikey, ival in obj.items():
                    if isinstance(ival, SimCfgBlk) or isinstance(ival, list) or isinstance(ival, dict):
                        assert False, f"DISASTER2 {ikey} {ival}"
                    else:
                        lines.append(f"{prefix} {ikey}: {ival}")
            else:
                DEBUG('... %s', obj)


            return '\n'.join(lines)
        return str_helper(self)


#####################
# Workloads
#####################


class WorkloadCfgBlk(SimCfgBlk):

    def __init__(self, wlname, **kwargs):
        super().__init__(wlname, **kwargs)

    def get_instances(self):
        pass


class WorkloadTTSIM(WorkloadCfgBlk):
    # Type declarations for INSTANCE attributes
    instances: dict
    params: dict
    module: str
    basedir: str
    # Class attributex
    required_fields = ['basedir', 'instances', 'module']
    optional_fields: dict[str, Any] = {'params': {}}

    def __init__(self, wlname, **kwargs):
        super().__init__(wlname, **kwargs)
        self.api = 'TTSIM'

    def get_instances(self):
        result = {}
        for iname, icfg in self.instances.items():
            xcfg = {}
            if self.params:
                xcfg.update(self.params)
            for xx,xv in icfg.items():
                xcfg[xx] = xv
            result[iname] = {'group': self.name, 'module': self.module, 'cfg': xcfg}
            result[iname]['path'] = os.path.join(self.basedir, self.module)
        return result


class WorkloadONNX(WorkloadCfgBlk):
    # Type declarations for INSTANCE attributes
    instances: dict
    basedir: str
    # Class attributex
    required_fields = ['basedir', 'instances']

    def __init__(self, wlname, **kwargs):
        super().__init__(wlname, **kwargs)
        self.api = 'ONNX'

    def get_instances(self):
        result = {}
        for iname, icfg in self.instances.items():
            xcfg = {xx: icfg[xx] for xx in icfg}
            result[iname] = {'group': self.name, 'cfg': xcfg}
            result[iname]['path'] = os.path.join(self.basedir, xcfg['path'])
        return result


TypeWorkloadClass = Type[WorkloadTTSIM] | Type[WorkloadONNX]
TypeWorkload = WorkloadTTSIM | WorkloadONNX


class AWorkload:
    WLCLS_TBL: dict[str, TypeWorkloadClass] = {'TTSIM': WorkloadTTSIM, 'ONNX': WorkloadONNX}

    @staticmethod
    def create_workload(apiname: str, **kwargs) -> TypeWorkload:
        wlclass = AWorkload.WLCLS_TBL[apiname.upper()]
        return wlclass(kwargs['name'], **kwargs)


class WorkloadGroup(WorkloadCfgBlk):
    # Type hints for instance attributes
    workloads: dict[str, WorkloadCfgBlk]
    # Class attributes
    WLCLS_TBL = {'TTSIM': WorkloadTTSIM, 'ONNX': WorkloadONNX}

    def __init__(self, apiname, **kwargs):
        WLCLS = self.WLCLS_TBL[apiname]
        self.workloads = {}
        for wl in kwargs:
            self.workloads[wl] = WLCLS(wl, **kwargs)
        super().__init__(apiname)

    def get_workload_names(self):
        return self.workloads.keys()

    def get_workload_instances(self, wl_name):
        wl_obj = self.workloads[wl_name]
        return wl_obj.get_instances()


#####################
# IPBlocks
#####################


class ComputeInsnModel(BaseModel, extra='forbid'):
    name: str
    tpt: dict[str, float]

class ComputePipeModel(BaseModel, extra='forbid'):
    name: str
    num_units: int
    freq_MHz: float
    systolic_depth: Optional[int] = 1
    instructions: List[ComputeInsnModel]

    def get_insn(self, instr: str) -> ComputeInsnModel:
        matches = [insn for insn in self.instructions if insn.name == instr]
        if len(matches) != 1:
            raise AssertionError(f'non-unique matches for instruction {instr} in {self.name} pipe')
        return matches[0]

    def frequency(self, units="MHz"):
        return convert_units(self.freq_MHz, 'MHz', units)

    def set_frequency(self, newfreq, units="MHZ"):
        self.freq_MHz = convert_units(newfreq, units, 'MHz')
        return


    def handle_missing_precision(self, insn: ComputeInsnModel, prec: str) -> Union[str, None]:
        tblx = {
            'int4' : ['int8', 'int16', 'int32'],
            'int8' : ['int16', 'int32'],
            'int16': ['int32'],
            'int32': [],
            'fp8'  : ['fp16', 'fp32', 'fp64'],
            'bf16' : ['fp32', 'fp64'],
            'fp16' : ['fp32', 'fp64'],
            'fp32' : ['fp64'],
            'tf32' : ['fp32', 'fp64'],
            'fp64' : [],
        }
        if prec not in tblx:
            raise AssertionError(f'unable to handle missing precision: {prec} for instruction= {insn.name} in ComputePipe.{self.name}!!')
        common_precisions = [new_prec for new_prec in tblx[prec] if new_prec in insn.tpt]
        if not common_precisions:
            raise AssertionError(f'no substitute precisions for {prec} for instruction {insn.name} under pipe {self.name}')
        return common_precisions[0]


    def peak_ipc(self, instr: str, prec: str) -> float:
        insn = self.get_insn(instr)
        if insn is None:
            raise AssertionError(f"Throughput for instruction= {instr} for ComputePipe.{self.name} not specified!!")
        try:
            ipc = insn.tpt[prec]
        except KeyError:
            warnonce(f"WARNING: Missing Support for Precision={prec} with Instruction={instr}@ComputePipe={self.name}")
            upgraded_prec = self.handle_missing_precision(insn, prec)
            warnonce(f">>>> upgraded_prec=   {upgraded_prec}")
            if upgraded_prec is None:
                raise AssertionError(f"Missing Support for Precision={prec} with Instruction={instr}@ComputePipe={self.name}")
            else:
                warnonce(f"WARNING: Using Precision={upgraded_prec} with Instruction={instr}@ComputePipe={self.name} instead!!")
                ipc = insn.tpt[upgraded_prec]
        if TYPE_CHECKING:
            assert self.systolic_depth is not None
        return ipc * self.systolic_depth * self.num_units

    def peak_flops(self, instr: str, prec: str, /, units:str='TFLOPS', mul_factor:int=1) -> float:
        mflops = self.peak_ipc(instr, prec) * mul_factor * self.frequency(units="MHZ")
        tflops = convert_units(mflops, 'MFLOPS', 'TFLOPS')
        return tflops


class L2CacheModel(BaseModel, extra='forbid'):
    num_banks: int
    bytes_per_clk_per_bank: int


class ComputeBlockModel(BaseModel, extra='forbid'):
    name: str
    iptype: str
    l2_cache: Optional[L2CacheModel] = None
    pipes: List[ComputePipeModel]

    def get_pipe(self, pipename: str) -> ComputePipeModel:
        matches = [pipe for pipe in self.pipes if pipe.name == pipename]
        if len(matches) != 1:
            raise AssertionError(f'non-unique matches for pipe {pipename} in {self.name}')
        return matches[0]

    def set_frequency(self, newfreq, units="MHZ")->None:
        for pipe in self.pipes:
            pipe.set_frequency(newfreq, units)


class MemoryBlockModel(BaseModel, extra='forbid'):
    name: str
    iptype: str
    technology: str
    data_bits: int
    freq_MHz: float
    size_GB: int
    stacks: Optional[int] = 1
    data_rate: Optional[int] = 1

    def size(self, /, units="GB"):
        return convert_units(self.size_GB, 'GB', units)

    def frequency(self, units="MHz"):
        return convert_units(self.freq_MHz, 'MHZ', units)

    def peak_bandwidth(self, freq_units="GHz"):
        freq = self.frequency(freq_units)
        Tps  = 2 * freq * self.stacks * self.data_rate #transfers-per-sec
        bw   = Tps * self.data_bits / 8
        return bw


type BlockModelType = Union[ComputeBlockModel, MemoryBlockModel]  # type: ignore[no-redef]

class IPGroupComponentModel(BaseModel, extra='forbid'):
    ipname: str
    iptype: str
    num_units: int
    freq_MHz: Optional[float] = None
    ramp_penalty: Optional[float] = 0.0
    ipobj: BlockModelType | None = None

class IPGroupComputeModel(IPGroupComponentModel):
    systolic_depth: Optional[int] = None

class IPGroupMemoryModel(IPGroupComponentModel):
    size_GB: Optional[float] = None

# IPBlocksModel = NewType('IPBlocksModel', BaseModel)
type IPGroupModel = Union[IPGroupComputeModel, IPGroupMemoryModel]


# #####################
# # Packages
# #####################


class PackageInstanceModel(BaseModel, extra='forbid'):
    devname: str
    name: str
    ipgroups: List[IPGroupModel]

    def get_ipgroup(self, iptype: str) -> IPGroupModel:
        matching = [ipgroup for ipgroup in self.ipgroups if ipgroup.iptype == iptype]
        if len(matching) != 1:
            raise AssertionError(f'non-unique matches for {iptype} in {self.name}')
        return matching[0]

    # Compute group interfaces
    def set_frequency(self, newfreq, units="MHZ"):
        compute_group = self.get_ipgroup(iptype='compute')
        if TYPE_CHECKING:
            assert compute_group.ipobj is not None
            assert isinstance(compute_group, ComputeBlockModel)
        compute_group.ipobj.set_frequency(newfreq, units)


    def peak_ipc(self, pipe: str, instr: str, precision: str):
        compute_group = self.get_ipgroup(iptype='compute')
        if TYPE_CHECKING:
            assert compute_group.ipobj is not None
            assert isinstance(compute_group, ComputeBlockModel)
        N        = compute_group.num_units
        pipe_obj = compute_group.ipobj.get_pipe(pipe)
        ipc      = pipe_obj.peak_ipc(instr, precision)
        return N * ipc

    def peak_flops(self, pipe, instr, precision, mul_factor=1, units='TFLOPS') -> float:
        compute_group = self.get_ipgroup(iptype='compute')
        if TYPE_CHECKING:
            assert compute_group.ipobj is not None
            assert isinstance(compute_group, ComputeBlockModel)
        pipe_obj = compute_group.ipobj.get_pipe(pipe)
        flops    = pipe_obj.peak_flops(instr, precision, mul_factor=mul_factor, units=units)
        N        = compute_group.num_units
        return N * flops

    def frequency(self, pipe, units="MHz"):
        compute_group = self.get_ipgroup(iptype='compute')
        if TYPE_CHECKING:
            assert compute_group.ipobj is not None
            assert isinstance(compute_group, ComputeBlockModel)
        pipe_obj = compute_group.ipobj.get_pipe(pipe)
        freq     = pipe_obj.frequency(units)
        return freq

    def ramp_penalty(self):
        compute_group = self.get_ipgroup(iptype='compute')
        if TYPE_CHECKING:
            assert compute_group.ipobj is not None
            assert isinstance(compute_group, ComputeBlockModel)
        return compute_group.ramp_penalty

    # Memory group interfaces
    def mem_size(self, units='GB'):
        memory_group =  self.get_ipgroup('memory')
        if TYPE_CHECKING:
            assert memory_group.ipobj is not None
            assert isinstance(memory_group, MemoryBlockModel)
        N = memory_group.num_units
        S = memory_group.ipobj.size(units)
        return N * S

    def peak_bandwidth(self, freq_units='GHz'):
        memory_group =  self.get_ipgroup('memory')
        if TYPE_CHECKING:
            assert memory_group.ipobj is not None
            assert isinstance(memory_group, MemoryBlockModel)
        N = memory_group.num_units
        B = memory_group.ipobj.peak_bandwidth(freq_units=freq_units)
        return N * B

    def mem_frequency(self, units="MHz"):
        memory_group =  self.get_ipgroup('memory')
        if TYPE_CHECKING:
            assert memory_group.ipobj is not None
            assert isinstance(memory_group, MemoryBlockModel)
        pipe_obj = memory_group.ipobj
        freq     = pipe_obj.frequency(units)
        return freq


def find_duplicates(counters: Counter):
    dups = {_key: _value for _key, _value in counters.items() if _value > 1}
    return dups


class IPBlocksModel(BaseModel, extra='forbid'): # type: ignore[no-redef]
    ipblocks: List[BlockModelType]

    @model_validator(mode='after')
    def validate_ipblocks(self):
        ipblocks_frequencies: Counter = Counter()
        for _tmp in self.ipblocks:
            ipblocks_frequencies[_tmp.name] += 1
        duplicates = find_duplicates(ipblocks_frequencies)
        if duplicates:
            raise AssertionError(f'ipblocks have multiple definitions: {duplicates}')
        return self

    def get_ipblock(self, blockname: str) -> BlockModelType:
        for _tmp in self.ipblocks:
            if _tmp.name == blockname:
                return _tmp
        raise AssertionError(f'IPBlock {blockname} not defined')

