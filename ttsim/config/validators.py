#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional, List, Literal, Annotated
from pydantic import BaseModel, Field

type TypeFrequency = float
type TypeTimeMS    = float
type TypeMemsizeGB = float

class PYDWlMapDataSpecValidator(BaseModel, extra='forbid'):
    global_type: Optional[str] = None
    override: Optional[dict[str, str]] = None


class PYDWlMapResourceSpecValidator(BaseModel, extra='forbid'):
    compute: dict[str, list[str]]


class PYDWlMapSpecValidator(BaseModel, extra='forbid'):
    op_data_type_spec: PYDWlMapDataSpecValidator
    op_removal_spec: list[str]
    op_fusion_spec: list[list[str]]
    op_rsrc_spec: PYDWlMapResourceSpecValidator


class PYDPkgMemoryValidator(BaseModel, extra='forbid'):
    ip: str
    num_units: int
    freq_MHz: Optional[float] = None
    ip_overrides: Optional[dict[str, int | float]] = {}


class PYDPkgComputeValidator(BaseModel, extra='forbid'):
    ip: str
    num_units: int
    ramp_penalty: Optional[float] = 0.0
    freq_MHz: Optional[float] = None
    systolic_depth: Optional[int] = None
    ip_overrides: Optional[dict[str, int | float]] = {}


class PYDComputePipeValidator(BaseModel, extra='forbid'):
    num_units: int
    freq_MHz: float
    systolic_depth: Optional[int] = 1
    instructions: dict[str, dict[str, float]]


class PYDL2CacheValidator(BaseModel, extra='forbid'):
    num_banks: int
    bytes_per_clk_per_bank: int

class PYDMemoryBlockValidator(BaseModel, extra='forbid'):
    technology: str
    data_bits: int
    freq_MHz: float
    size_GB: float
    data_rate: Optional[float] = 1
    stacks: Optional[int] = 1


class PYDComputeBlockValidator(BaseModel, extra='forbid'):
    pipes: dict[str, PYDComputePipeValidator]
    l2_cache: Optional[PYDL2CacheValidator] = None

class PYDWorkloadBaseModel(BaseModel):
    name: str


class PYDWorkloadTTSIMModelValidator(PYDWorkloadBaseModel):
    api: Literal['TTSIM']
    name: str
    module: str
    basedir: str
    instances: dict
    params: Optional[dict] = {}



class PYDWorkloadONNXModelValidator(PYDWorkloadBaseModel):
    api: Literal['ONNX']
    name: str
    basedir: str
    instances: dict

    def get_instances(self):
        result = {}
        for iname, icfg in self.instances.items():
            xcfg = {xx: icfg[xx] for xx in icfg}
            result[iname] = {'group': self.name, 'cfg': xcfg }
            result[iname]['path'] = os.path.join(self.basedir, xcfg['path'])
        return result


AnyWorkload = Annotated[PYDWorkloadTTSIMModelValidator | PYDWorkloadONNXModelValidator, Field(discriminator='api')]


class PYDWorkloadListValidator(BaseModel):
    workloads: List[AnyWorkload]

type TypeDeviceName = str
type TypePipeName = str
type TypePrecision = str
type TypeOpType = str
type TypeOpClass = str
type TypeResourceName = str
type TypeDomain = str
type TypeInstrName = str

# Option 1 : Direct representation of output CSV
class TTSimHLWlDevRunOpCSVPerfStats(BaseModel, extra='forbid'):
    devname: TypeDeviceName = Field(
        description = 'Device Name'
    )
    freq_MHz: float = Field(
        description = 'Frequency in MHz'
    )
    pipe: TypePipeName = Field(
        description = 'Pipe name'
    )
    precision: TypePrecision = Field(
        description = 'Precision'
    )
    wlgroup: str = Field(
        description = 'Workload group (API - TTSIM, ONNX etc)'
    )
    wlname: str = Field(
        description = 'Workload name'
    )
    wlinstance: str = Field(
        description = 'Workload instance (specific workload configuration)'
    )
    batch: int = Field(
        description = 'Batch Size'
    )
    opnum: int = Field(
        description = 'Operator sequence number'
    )
    opname: str = Field(
        description = 'Operator name'
    )
    is_input_node: bool = Field(
        description = 'Is this a network input node? (boolean)' # TODO: P1 Is this right?
    )
    is_output_node: bool = Field(
        description = 'Is this a network output node? (boolean)' # TODO: P1 Is this right?
    )
    optype: TypeOpType = Field(
        description = 'Operator type'
    )
    op_rpt_count: int = Field(
        description = 'Repeat count' # TODO: P1 What is this?
    )
    attrs: dict = Field(
        description = 'Operator attributes'
    )
    inList: list = Field(
        description = 'List of tensors input to this operator'
    )
    outList: list = Field(
        description = 'List of tensors output by this operator'
    )
    domain: TypeDomain = Field(
        description = '???' # TODO: P1 What is this?
    )
    opclass: TypeOpClass = Field(
        description = 'Operator class' # TODO: P1 Distinguish between operator type and class
    )
    removed: bool = Field(
        description = 'Is this operator removed'
    )
    fused: bool = Field(
        description = 'Is this operator fused with another'
    )
    fused_with_op: str = Field(
        description = 'The operator (name) with which this operator is fused'
    )
    inElems: int = Field(
        description = 'Count of input elements'
    )
    outElems: int = Field(
        description = 'Count of output elements'
    )
    inBytes: int = Field(
        description = 'Size in bytes of input elements'
    )
    outBytes: int = Field(
        description = 'Size in bytes of output elements'
    )
    instrs: dict[TypeInstrName, int] = Field(
        description = 'Map of instruction name to its count, within this operator'
    )
    inParamCount: int = Field(
        description = 'Count of (input) parameters'
    )
    inActCount: int = Field(
        description = 'Count if input activation elements'
    )
    outActCount:int = Field(
        description = 'Count if output activation elements'
    )
    instr_count: int = Field(
        description = 'Count of instructions' # TODO: elaborate
    )
    compute_cycles: float = Field(
        description = 'Number of cycles required for compute during the execution this operator'
    )
    mem_rd_cycles: float = Field(
        description = 'Number of cycles required for reading memory during the execution this operator'
    )
    mem_wr_cycles: float = Field(
        description = 'Number of cycles required for writing memory during the execution this operator'
    )
    ramp_penalty: float = Field(
        description = 'Ramp Penalty cycles (roughly speaking, the time required for ramping the operator execution till the entire system is loaded)'
    )
    rsrc_bnck: TypeResourceName = Field(
        description = 'Bottleneck resource for this operator; among all resources, the operator required most cycles in this resource'
    )
    cycles: float = Field(
        description = 'Number of cycles required to execute the operator'
    )
    msecs: float = Field(
        description = 'Time (msec) required to execute the operator'
    )

# Option 2 - Structured Stats

class TTSimHLWlDevRunOperatorPerfStats(BaseModel, extra='forbid'):
    pipe: TypePipeName
    precision: TypePrecision
    opnum: int
    opname: str
    is_input_node: bool
    is_output_node: bool
    optype: TypeOpType
    op_rpt_count: int
    attrs: dict
    inList: list
    outList: list
    domain: TypeDomain
    opclass: TypeOpClass
    removed: bool
    fused: bool
    fused_with_op: str
    inElems: int
    outElems: int
    inBytes: int
    outBytes: int
    instrs: dict[TypeInstrName, int]
    inParamCount: int
    inActCount: int
    outActCount:int
    instr_count: int
    compute_cycles: float
    mem_rd_cycles: float
    mem_wr_cycles: float
    ramp_penalty: float
    rsrc_bnck: TypeResourceName
    cycles: float
    msecs: float

class TTSimHLWlDevRunPerfStats(BaseModel, extra='forbid'):
    """
        This model represents a "run" of the high level simulator on:
           - one specific workload instance,
           - one specific hardware

        The performance metrics of individual operators comprising the workload appear as a list in this model
    """
    devname: TypeDeviceName
    freq_MHz: float
    wlgroup: str
    wlname: str
    wlinstance: str
    batch: int    # TODO: Run or Operator?
    operatorstats: list[TTSimHLWlDevRunOperatorPerfStats]

class TTSimHLRunSummaryRow(BaseModel, extra='forbid'):
    devname      : str
    freq_Mhz     : TypeFrequency
    wlcls        : str
    wlname       : str
    wlinstance   : str
    bs           : int
    inParams     : int
    inActs       : int
    outActs      : int
    maxActs      : int
    inParamBytes : int
    inActBytes   : int
    outActBytes  : int
    maxActBytes  : int
    tot_cycles   : int
    tot_msecs    : TypeTimeMS
    throughput   : float
    mem_size_GB  : TypeMemsizeGB
    device_mem_GB: int
    fits_device  : bool
    rsrc_mem     : float
    rsrc_comp    : float
    stat_filename: str

class TTSimHLRunSummary(BaseModel, extra='forbid'):
    summary: list[TTSimHLRunSummaryRow]
