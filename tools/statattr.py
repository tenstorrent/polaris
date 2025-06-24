#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

category_2_seq: dict[str, int] = {'key': 0, 'info': 1, 'perf': 2, 'comp': 3, 'mem': 4, 'cache': 5, 'misc': 6}


class StatAttributeDescriptor(BaseModel, extra='forbid'):
    sr: int = 0
    seq: int = 0
    name: str
    attrtype: Literal['int', 'float', 'str', 'bool', 'dict', 'list']
    catg: Literal['key', 'info', 'perf', 'comp', 'mem', 'cache', 'misc']
    prio: int
    is_filter: bool = False
    label: str | None = None

    @property
    def is_numeric(self):
        return self.attrtype in ['int', 'float']

    @property
    def is_nonnumeric(self):
        return not self.is_numeric


type StatAttributeDescriptorList = list[StatAttributeDescriptor]


class StatAttributeDescriptors:
    job_attribute_list: StatAttributeDescriptorList
    op_attribute_list: StatAttributeDescriptorList

    @classmethod
    def setup_attribute_descriptors(cls) -> None:
        cls.setup_job_attribute_descriptors()
        cls.setup_op_attribute_descriptors()

    @classmethod
    def setup_job_attribute_descriptors(cls) -> None:
        tmp_job_attributes_list: list[dict[str, Any]] = [
            {'name': 'devname', 'catg':'key', 'prio':0, 'attrtype': 'str', 'is_filter': True, 'label': 'Device'},
            {'name': 'freq_Mhz', 'catg':'info', 'prio':0, 'attrtype': 'float', 'label': 'Freq'},
            {'name': 'wlcls', 'catg':'info', 'prio':0, 'attrtype': 'str', 'is_filter': True, 'label': 'API'},
            {'name': 'wlname', 'catg':'key', 'prio':0, 'attrtype': 'str', 'is_filter': True, 'label': 'Workload'},
            {'name': 'wlinstance', 'catg':'key', 'prio':0, 'attrtype': 'str', 'label': 'WL/Instance'},
            {'name': 'bs', 'catg':'info', 'prio':0, 'attrtype': 'int', 'label': 'Batch'},
            {'name': 'inParams', 'catg':'info', 'prio':1, 'attrtype': 'int'},
            {'name': 'inActs', 'catg':'info', 'prio':1, 'attrtype': 'int'},
            {'name': 'outActs', 'catg':'info', 'prio':1, 'attrtype': 'int'},
            {'name': 'maxActs', 'catg':'info', 'prio':1, 'attrtype': 'int'},
            {'name': 'inParamBytes', 'catg':'mem', 'prio':1, 'attrtype': 'int'},
            {'name': 'inActBytes', 'catg':'mem', 'prio':1, 'attrtype': 'int'},
            {'name': 'outActBytes', 'catg':'mem', 'prio':1, 'attrtype': 'int'},
            {'name': 'maxActBytes', 'catg':'mem', 'prio':1, 'attrtype': 'int'},
            {'name': 'tot_cycles', 'catg':'key', 'prio':0, 'attrtype': 'int', 'is_filter': True, 'label': 'Cycles'},
            {'name': 'tot_msecs', 'catg':'perf', 'prio':0, 'attrtype': 'float', 'label': 'Time (ms)'},
            {'name': 'throughput', 'catg':'perf', 'prio':0, 'attrtype': 'float'},
            {'name': 'mem_size_GB', 'catg':'info', 'prio':0, 'attrtype': 'float', 'label': 'Memory footprint (GB)'},
            {'name': 'device_mem_GB', 'catg':'info', 'prio':0, 'attrtype': 'float', 'label': 'Device Memory (GB)'},
            {'name': 'fits_device', 'catg':'info', 'prio':0, 'attrtype': 'bool'},
            {'name': 'rsrc_mem', 'catg':'perf', 'prio':0, 'attrtype': 'float', 'label': 'MemBound', 'is_filter': True},
            {'name': 'rsrc_comp', 'catg':'perf', 'prio':0, 'attrtype': 'float', 'label': 'CmpBound', 'is_filter': True},
        ]
        cls.job_attribute_list = [StatAttributeDescriptor(**x) for x in tmp_job_attributes_list]
        StatAttributeDescriptors.update_attribute_descriptors(cls.job_attribute_list)

    @classmethod
    def setup_op_attribute_descriptors(cls) -> None:
        tmp_op_attributes_list: list[dict[str, Any]] = [
            {'name': 'opnum',          'attrtype': 'int',   'catg': 'key',   'prio': 0, 'label': 'Op#'},
            {'name': 'opname',         'attrtype': 'str',   'catg': 'key',   'prio': 0, 'label': 'Op'},
            {'name': 'pipe',           'attrtype': 'str',   'catg': 'comp',  'prio': 0, 'is_filter': True},
            {'name': 'precision',      'attrtype': 'str',   'catg': 'comp',  'prio': 0},
            {'name': 'is_input_node',  'attrtype': 'bool',  'catg': 'info',  'prio': 0, 'label': 'Is NW Input'},
            {'name': 'is_output_node', 'attrtype': 'bool',  'catg': 'info',  'prio': 0, 'label': 'Is NW Output'},
            {'name': 'optype',         'attrtype': 'str',   'catg': 'info',  'prio': 0, 'is_filter': True, 'label': 'Op Type'},
            {'name': 'op_rpt_count',   'attrtype': 'int',   'catg': 'info',  'prio': 0, 'label': 'Op Rpt Count'},
            {'name': 'attrs',          'attrtype': 'dict',  'catg': 'info',  'prio': 0, 'label': 'Attributes'},
            {'name': 'inList',         'attrtype': 'list',  'catg': 'info',  'prio': 0},
            {'name': 'outList',        'attrtype': 'list',  'catg': 'info',  'prio': 0},
            {'name': 'domain',         'attrtype': 'str',   'catg': 'info',  'prio': 0},
            {'name': 'opclass',        'attrtype': 'str',   'catg': 'info',  'prio': 0},
            {'name': 'removed',        'attrtype': 'bool',  'catg': 'info',  'prio': 0, 'label': 'Is Removed'},
            {'name': 'fused',          'attrtype': 'bool',  'catg': 'info',  'prio': 0, 'is_filter': True, 'label': 'Is Fused'},
            {'name': 'fused_with_op',  'attrtype': 'str',   'catg': 'info',  'prio': 0, 'label': 'Fused With'},
            {'name': 'inElems',        'attrtype': 'int',   'catg': 'info',  'prio': 0},
            {'name': 'outElems',       'attrtype': 'int',   'catg': 'info',  'prio': 0},
            {'name': 'inBytes',        'attrtype': 'int',   'catg': 'mem',   'prio': 0},
            {'name': 'outBytes',       'attrtype': 'int',   'catg': 'mem',   'prio': 0},
            {'name': 'instrs',         'attrtype': 'dict',  'catg': 'comp',  'prio': 0, 'label': 'Instructions'},
            {'name': 'inParamCount',   'attrtype': 'int',   'catg': 'info',  'prio': 0},
            {'name': 'inActCount',     'attrtype': 'int',   'catg': 'info',  'prio': 0},
            {'name': 'outActCount',    'attrtype': 'int',   'catg': 'info',  'prio': 0},
            {'name': 'instr_count',    'attrtype': 'int',   'catg': 'comp',  'prio': 0},
            {'name': 'compute_cycles', 'attrtype': 'float', 'catg': 'comp',  'prio': 0, 'label': 'Compute Cycles'},
            {'name': 'mem_rd_cycles',  'attrtype': 'float', 'catg': 'mem',   'prio': 0, 'label': 'Read Cycles'},
            {'name': 'mem_wr_cycles',  'attrtype': 'float', 'catg': 'mem',   'prio': 0, 'label': 'Write Cycles'},
            {'name': 'ramp_penalty',   'attrtype': 'float', 'catg': 'misc',  'prio': 0, 'label': 'Ramp Penalty'},
            {'name': 'rsrc_bnck',      'attrtype': 'str',   'catg': 'perf',  'prio': 0, 'is_filter': True, 'label': 'Resource Bound'},
            {'name': 'cycles',         'attrtype': 'float', 'catg': 'key',  'prio': 0, 'is_filter': True, 'label': 'Cycles'},
            {'name': 'msecs',          'attrtype': 'float',  'catg': 'perf', 'prio': 0, 'label': 'Time (ms)'},
        ]
        cls.op_attribute_list = []
        for x in tmp_op_attributes_list:
            try:
                cls.op_attribute_list.append(StatAttributeDescriptor(**x))
            except ValidationError:
                logging.error('Invalid attribute descriptor for attribute %s', x)
                raise
        StatAttributeDescriptors.update_attribute_descriptors(cls.op_attribute_list)

    @staticmethod
    def update_attribute_descriptors(attrlist: list[StatAttributeDescriptor]) -> None:
        """
        Update the attribute descriptors, setting serial number (sr) and sequence number (seq)
        """
        filter_attrs = [attrdesc.name for attrdesc in attrlist if attrdesc.is_filter]
        if not filter_attrs:
            raise NotImplementedError('comparison summary without any filter attribute not supported')
        for ndx, attrdesc in enumerate(attrlist):
            attrdesc.sr = ndx
            if attrdesc.label is None:
                attrdesc.label = attrdesc.name
        for ndx, attrdesc in enumerate(sorted(attrlist, key=lambda x: (category_2_seq[x.catg], x.prio, x.sr))):
            attrdesc.seq = ndx
