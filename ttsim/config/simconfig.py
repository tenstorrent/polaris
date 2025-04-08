#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
import logging
import re
import os

from copy import deepcopy

from ttsim.utils.common import convert_units, warnonce
from typing import Any, Type

from typing import Optional

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
            xcfg = {xx: icfg[xx] for xx in icfg}
            if self.params:
                xcfg.update(self.params)
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


class ComputePipe(SimCfgBlk):
    # Type declaration of INSTANCE attributes
    num_units: int
    freq_MHz: float
    systolic_depth: int
    instructions: dict[str, Any]
    # Class attributes
    required_fields = ['num_units', 'freq_MHz', 'instructions']
    optional_fields = {'systolic_depth': 1}

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def frequency(self, units="MHz"):
        return convert_units(self.freq_MHz, 'MHz', units)

    def set_frequency(self, newfreq, units="MHZ"):
        self.freq_MHz = convert_units(newfreq, units, 'MHz')
        return

    def handle_missing_precision(self, instr, prec):
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
        assert prec in tblx, f"unable to handle missing precision: {prec} for instruction= {instr} in ComputePipe.{self.name}!!"
        for new_prec in tblx[prec]:
            if new_prec in self.instructions[instr]:
                return new_prec
        return None

    def peak_ipc(self, instr, prec):
        assert instr in self.instructions, f"Throughput for instruction= {instr} for ComputePipe.{self.name} not specified!!"
        try:
            ipc = self.instructions[instr][prec]
        except KeyError:
            warnonce(f"WARNING: Missing Support for Precision={prec} with Instruction={instr}@ComputePipe={self.name}")
            upgraded_prec = self.handle_missing_precision(instr, prec)
            warnonce(f">>>> upgraded_prec=   {upgraded_prec}")
            if upgraded_prec is None:
                raise
            else:
                warnonce(f"WARNING: Using Precision={upgraded_prec} with Instruction={instr}@ComputePipe={self.name} instead!!")
                ipc = self.instructions[instr][upgraded_prec]
        return ipc * self.systolic_depth * self.num_units

    def peak_flops(self, instr, prec, /, units='TFLOPS', mul_factor=1):
        mflops = self.peak_ipc(instr, prec) * mul_factor * self.frequency(units="MHZ")
        tflops = convert_units(mflops, 'MFLOPS', 'TFLOPS')
        return tflops


class ComputeIP(SimCfgBlk):

    def __init__(self, name, **kwargs):
        self.pipes = {}
        for pp in kwargs['pipes']:
            self.pipes[pp] = ComputePipe(pp, **kwargs['pipes'][pp])
        super().__init__(name)

        #make sure required compute pipes are specified
        for p in ['vector', 'matrix']:
            assert p in self.pipes, f"missing compute pipe: {p} for {self.kind()}.{self.name}"

    def set_frequency(self, newfreq, units="MHZ"):
        for pn, po in self.pipes.items():
            po.set_frequency(newfreq, units)


class MemoryIP(SimCfgBlk):
    # Type declaration of INSTANCE attributes
    data_bits: int
    freq_MHz: float
    size_GB: int
    stacks: Optional[int] = 1
    data_rate: Optional[int] = 1
    # Class attributes
    required_fields = ['technology', 'data_bits', 'freq_MHz', 'size_GB']
    optional_fields = {'stacks': 1, 'data_rate': 1}
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def size(self, /, units="GB"):
        return convert_units(self.size_GB, 'GB', units)

    def frequency(self, units="MHz"):
        return convert_units(self.freq_MHz, 'MHZ', units)

    def peak_bandwidth(self, freq_units="GHz"):
        freq = self.frequency(freq_units)
        Tps  = 2 * freq * self.stacks * self.data_rate #transfers-per-sec
        bw   = Tps * self.data_bits / 8
        return bw

def create_ipblock(ip_type, ip_name, ip_cfg):
    cls_tbl = {
        'compute': ComputeIP,
        'memory':  MemoryIP,
    }
    ip_obj = cls_tbl[ip_type.lower()](ip_name, **ip_cfg)
    return ip_obj

#####################
# Packages
#####################
class IPGroup(SimCfgBlk):
    # Type declarations of INSTANCE attributes
    ip: str
    num_units: int
    ramp_penalty: float
    # Class attributes
    required_fields = ['ip', 'num_units']
    optional_fields = {'ramp_penalty': 0}

    def __init__(self, name, ipdb, **kwargs):
        super().__init__(name, **kwargs)
        try:
            self.ipobj = deepcopy(ipdb[self.ip])
        except KeyError:
            DEBUG(f"Error: ip={self.ip} specification not found for {self.kind()}.{self.name}!!")
            raise
        if 'ip_overrides' in kwargs:
            for ok, ov in kwargs['ip_overrides'].items():
                res = self.ipobj.set_param(ok, ov)
                if not res:
                    DEBUG(f"WARNING: Illegal ip_override specification {ok}, IGNORED!!")


class Package(SimCfgBlk):

    def __init__(self, name, ipdb, **kwargs):
        self.ipgroups = {}
        for pg in kwargs['ipgroups']:
            self.ipgroups[pg] = IPGroup(pg, ipdb, **kwargs['ipgroups'][pg])
        super().__init__(name)

        #make sure required ip-groups are specified
        for g in ['compute', 'memory']:
            assert g in self.ipgroups, f"missing ip-group: {g} for {self.kind()}.{self.name}"

    #compute api
    def set_frequency(self, newfreq, units="MHZ"):
        self.ipgroups['compute'].ipobj.set_frequency(newfreq, units)

    def peak_ipc(self, pipe, instr, precision):
        N        = self.ipgroups['compute'].num_units
        pipe_obj = self.ipgroups['compute'].ipobj.pipes[pipe]
        ipc      = pipe_obj.peak_ipc(instr, precision)
        return N * ipc

    def peak_flops(self, pipe, instr, precision, mul_factor=1, units='TFLOPS'):
        pipe_obj = self.ipgroups['compute'].ipobj.pipes[pipe]
        flops    = pipe_obj.peak_flops(instr, precision, mul_factor=mul_factor, units=units)
        N        = self.ipgroups['compute'].num_units
        return N * flops

    def ramp_penalty(self):
        return self.ipgroups['compute'].ramp_penalty

    def frequency(self, pipe, units="MHz"):
        pipe_obj = self.ipgroups['compute'].ipobj.pipes[pipe]
        freq     = pipe_obj.frequency(units)
        return freq

    #memory api
    def mem_size(self, units='GB'):
        N = self.ipgroups['memory'].num_units
        S = self.ipgroups['memory'].ipobj.size(units)
        return N * S

    def peak_bandwidth(self, freq_units='GHz'):
        N = self.ipgroups['memory'].num_units
        B = self.ipgroups['memory'].ipobj.peak_bandwidth(freq_units=freq_units)
        return N * B

    def mem_frequency(self, units="MHz"):
        pipe_obj = self.ipgroups['memory'].ipobj
        freq     = pipe_obj.frequency(units)
        return freq


def create_package(pkg_type, pkg_name, pkg_cfg, ip_db):
    return Package(pkg_name, ip_db, **pkg_cfg)
