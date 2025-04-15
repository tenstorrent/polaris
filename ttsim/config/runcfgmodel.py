#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
from typing import Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator

type TYPE_SWEEP = Literal['all', 'first-parent']
type TYPE_RUN = Literal['inference', 'training']
type TYPE_LOGLEVEL = Literal['critical', 'error', 'warning', 'info', 'debug']
type TYPE_frequency = Tuple[int, int, int]
type TYPE_batchsize = Tuple[int, int, int]

class PolarisRunConfig(BaseModel, extra='forbid'):
    title: str = Field(description='Human readable itle for the run')
    output: str = Field(
        description='This value is considered to be a Jinja2 template, and expanded as one, with the variables that '
        'Kochab automatically defines prior to each run. Hence, the value can use the complete expressive power of '
        'Jinja2 templates to construct the output directory names. '
        '(TBD: provide pointer to the list of variables defined by Kochab that can be used in the template)'
    )
    study: str = Field(
        description='Study name'
    )
    wlspec: str = Field(
        description='Workload Specification yaml file'
    )
    archspec: str = Field(
        description='Architecture Specification yaml file'
    )
    wlmapspec: str = Field(
        description='Workload To Architecture Mapping Specification yaml file'
    )

    filterapi: Optional[str] = Field(
        default=None,
        description='APIs to be considered for the run'
    )

    filterwl: Optional[str] = Field(
        default=None,
        description='use only workloads specified in filterwl (comma sep list)'
    )

    filterwli: Optional[str] = Field(
        default=None,
        description='use only workload instances specified in filterwli (comma sep list)'
    )

    filterarch: Optional[str] = Field(
        default=None,
        description='use only architectures specified in filterarch (comma sep list)'
    )

    filterrun: Optional[TYPE_RUN] = Field(
        default='inference',
        description='Filter either inference or training runs',
    )

    filter: Optional[str] = Field(
        default=None,
        description='A regular expression to filter, occurring anywhere in the command'
    )

    frequency: Optional[TYPE_frequency] = Field(
        default=None,
        description='frequency (in MHz) range specification (arith-seq)')

    @field_validator('frequency')
    def validate_frequency(cls, v: TYPE_frequency, info: ValidationInfo) -> TYPE_frequency:
        if v is None:
            return v
        if any([v[0] <= 0, v[1] <= 0, v[2] <= 0]):
            raise AssertionError(f'frequency values should be positive in {v}')
        if v[0] >= v[1]:
            raise AssertionError(f'frequency first value {v[0]} should be <= last value {v[1]} in {v}')
        return v

    @field_validator('batchsize')
    def validate_batchsize(cls, v: TYPE_batchsize, info: ValidationInfo) -> TYPE_batchsize:
        if v is None:
            return v
        if any([v[0] <= 0, v[1] <= 0, v[2] <= 0]):
            raise AssertionError(f'batchsize values should be positive in {v}')
        if v[2] == 1:
            raise AssertionError(f'batchsize step {v[2]} can not be 1')
        if v[0] >= v[1]:
            raise AssertionError(f'batchsize first value {v[0]} should be < last value {v[1]} in {v}')
        return v

    batchsize: Optional[TYPE_batchsize] = Field(default=None, description='batchsize range specification (geom-seq)')

    knobs: Optional[str] = Field(default='', description='Additional knobs to be used the runs')

    instr_profile: Optional[bool] = Field(default=False, description='Collect instruction profile for workloads')

    dump_ttsim_onnx: Optional[bool] = Field(default=False, description='Dump ONNX graph for TTSIM Workload')

    ignorefails: Optional[bool] = Field(default=False,
                                        description='If true, run will continue even if some command(s) fails')

    loglevel: Optional[TYPE_LOGLEVEL] = Field(
        default='info', description='Logging level (error,warning,info,debug) for the run'
    )

    githash: str | None = Field(
        default=None, description='Particular git commit'
    )

    saved_copy: Optional[bool] = Field(
        default=False,
        description='Only for internal use. This attribute should be true for yaml copies saved by polaris system only'
    )


type TYPE_GITHASH = Union[str, None]
