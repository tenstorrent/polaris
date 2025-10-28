#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.graph import WorkloadGraph

from enum import Enum, auto
from dataclasses import dataclass

from typing import Any

################################x################################x################################x################################
# utility types ---  mostly from tt-umd
ChipId       = int
EthChannelID = int
CoreCoord    = tuple[int, int] #xy-pair

@dataclass
class EthCoord:
    cluster_id : int
    x          : int
    y          : int
    rack       : int
    shelf      : int

class ARCH(Enum):
    GRAYSKULL   = auto()
    WORMHOLE_B0 = auto()
    BLACKHOLE   = auto()
    QUASAR      = auto()
    UNKNOWN     = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return ARCH[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()

class BoardType(Enum):
    N150    = auto()
    N300    = auto()
    P100    = auto()
    P150    = auto()
    P300    = auto()
    GALAXY  = auto()
    UNKNOWN = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return BoardType[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()
################################x################################x################################x################################

class Device:
    def __init__(self, **kwargs):
        self.device_id            : (Any|int) = kwargs.get('device_id')
        self.l1_small_size        : int = kwargs.get('l1_small_size',        0)
        self.trace_region_size    : int = kwargs.get('trace_region_size',    0)
        self.worker_l1_size       : int = kwargs.get('worker_l1_size',       0)
        self.dispatch_core_config : int = kwargs.get('dispatch_core_config', 0) #DispatchCoreConfig

        #Placeholders
        self.core_grid            : int = 0
        self.grid_x               : int = 0
        self.grid_y               : int = 0
        self.architecture         : ARCH = ARCH.WORMHOLE_B0

        self.args    = kwargs
        self.tensors = {}
        self.ops     = {}

        return

    def get_arch_name(self):
        return str(self.architecture)

    def arch(self):
        return self.architecture

    def get_num_devices(self):
        #TODO: Check this logic
        return 1

    def compute_with_storage_grid_size(self, grid_x: (Any | int) = None, grid_y: (Any | int) = None):

        if grid_x is not None: self.grid_x = grid_x
        if grid_y is not None: self.grid_y = grid_y

        return (self.grid_x, self.grid_y)

    def add_tensor(self, t):
        if t.name not in self.tensors:
            self.tensors[t.name] = t

    def add_op(self, o):
        if o.name not in self.ops:
            self.ops[o.name] = o

    def get_graph(self):
        gg = WorkloadGraph('xxx')
        for _,t in self.tensors.items():
            gg.add_tensor(t)
        for _,o in self.ops.items():
            gg.add_op(o)
        gg.construct_graph()
        return gg

    def __str__(self):
        return f"(Device: {self.args})"


def open_device(**kwargs):
    return Device(**kwargs) # Normally returns ttnn.multi_device.MeshDevice

def close_device(device: Device):
    return

def num_cores_to_corerangeset(*args, **kwargs):
    return (1,1) # dummy implementation

def create_sharded_memory_config(*args, **kwargs):
    return None

def interleaved_to_sharded(input_tensor, *args, **kwargs):
    return input_tensor
