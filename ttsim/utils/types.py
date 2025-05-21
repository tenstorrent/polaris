#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from typing import NewType, NamedTuple, Union
from enum import Enum, StrEnum, auto
from dataclasses import dataclass
from collections import namedtuple

###################################################################################
# Useful Types
###################################################################################

class FrameworkType(Enum):
    ONNX    = auto()
    METAL   = auto()
    UNKNOWN = auto()

class SimDataType(Enum):
    BOOL       = auto()
    INT4       = auto()
    INT8       = auto()
    INT16      = auto()
    INT32      = auto()
    INT64      = auto()
    UINT4      = auto()
    UINT8      = auto()
    UINT16     = auto()
    UINT32     = auto()
    UINT64     = auto()
    BFLOAT8    = auto()
    BFLOAT16   = auto()
    FLOAT16    = auto()
    FLOAT32    = auto()
    FLOAT64    = auto()
    UNKNOWN    = auto()

class plTensorDimType(Enum):
    RANKED   = auto() #Ranked Tensores have dims e.g. [1], [1,2,3,4]
    SCALAR   = auto() #Single Entry Scalar Values, e.g. int, float, ...
    NULL     = auto() #no dims, contain no data, some FWs have optional inputs
    UNKNOWN  = auto()

class plTensorUsageType(Enum):
    DATA       = auto()
    PARAM      = auto()
    ATTRIBUTE  = auto()
    CONSTANT   = auto()
    UNKNOWN    = auto()

plGraphNodeIDType = NewType("plGraphNodeIDType", str)
class plGraphEdgeKey(NamedTuple):
    name: str
    src: plGraphNodeIDType
    outportnum: int

SOCNodeID  = namedtuple('SOCNodeID',  'x y')
SOCGridDim = namedtuple('SOCGridDim', 'x_size y_size')
DRAMChnlID = namedtuple('DRAMChnlID', 'chnl sub_chnl')

class SOCNodeType(Enum):
    CORE   = auto()
    PCIE   = auto()
    DRAM   = auto()
    ETH    = auto()
    ROUTER = auto()
    ARC    = auto()
    DUMMY  = auto()

def socnodetype2str(n: SOCNodeType) -> str:
    tbl = {
            SOCNodeType.CORE   : 'C',
            SOCNodeType.PCIE   : 'P',
            SOCNodeType.DRAM   : 'D',
            SOCNodeType.ETH    : 'E',
            SOCNodeType.ROUTER : 'R',
            SOCNodeType.ARC    : 'A',
            SOCNodeType.DUMMY  : '-',
            }
    return tbl[n]

class NOCTopologyType(StrEnum):
    MESH  = auto()
    TORUS = auto()

class NOCRoutingAlgoType(StrEnum):
    XY       = auto()
    ADAPTIVE = auto()


#Address Types
# X,Y are 6b, so in principle we can address upto 64 routers per direction
# Local Offset is 32b for Grayskull and 36b for Wormhole
# Therefore, Unicast Addr = 6 + 6 + 32 = 44b for Grayskull and 48b for Wormhole
# Similarly, Multicast Addr = 6 + 6 + 6 + 6 + 32 = 56b for Grayskull and 60b for Wormhole
plRouterId      = tuple[int, int] #(X, Y)
plUnicastAddr   = tuple[plRouterId, int] #Destination, Local-Offset
plMulticastAddr = tuple[plRouterId, plRouterId, int] #Start, End, Local-Offset
plAddress       = plUnicastAddr | plMulticastAddr #Python 3.10+ this means Union[UnicastAddr,MulticastAddr]

class DataFormat(Enum):
    #Floating Point
    FP64  = auto()
    FP32  = auto()
    TF32  = auto()
    FP16A = auto()
    FP16B = auto()
    LF8   = auto()
    #Integer Point
    INT32 = auto()
    INT16 = auto()
    INT8  = auto()
    #Block Floating Point -- shared exponent
    BFP8A = auto()
    BFP8B = auto()
    BFP4A = auto()
    BFP4B = auto()
    BFP2A = auto()
    BFP2B = auto()
    #Boolean
    BOOL  = auto()
    UNDEFINED = auto()

def str2df(x: Union[str, list[str]]) -> Union[DataFormat, list[DataFormat]]:
    tbl = {
            "XXX01"  : DataFormat.FP64,
            "XXX03"  : DataFormat.TF32,
            "float16"  : DataFormat.FP16A,
            "float16_b"  : DataFormat.FP16B,
            "XXX06"  : DataFormat.LF8,
            "XXX07"  : DataFormat.INT32,
            "XXX08"  : DataFormat.INT16,
            "XXX09"  : DataFormat.INT8,
            "XXX10"  : DataFormat.BFP8A,
            "XXX11"  : DataFormat.BFP8B,
            "XXX12"  : DataFormat.BFP4A,
            "XXX13"  : DataFormat.BFP4B,
            "XXX14"  : DataFormat.BFP2A,
            "XXX16"  : DataFormat.BOOL,

            #p1 workloads....
            "bfp2_b" : DataFormat.BFP2B,
            "rawuint32"  : DataFormat.INT32,
            "float32"  : DataFormat.FP32,

            #"XXX17"  : DataFormat.UNDEFINED
            }
    if isinstance(x, list):
        vals = []
        for d in x:
            d = d.lower()
            v = tbl[d]
            vals.append(v)
        return vals
    else:
        x = x.lower()
        val = tbl[x]
        return val

class MathFidelity(Enum):
    LoFi  = auto()
    HiFi2 = auto()
    HiFi3 = auto()
    HiFi4 = auto()
    UNDEFINED = auto()

def str2mf(x: str) -> MathFidelity:
    x = x.lower()
    tbl = {
            "lofi"  : MathFidelity.LoFi,
            "hifi2" : MathFidelity.HiFi2,
            "hifi3" : MathFidelity.HiFi3,
            "hifi4" : MathFidelity.HiFi4
            }
    return tbl[x]

def get_sim_dtype(dtype: str) -> SimDataType:
    tbl = {
            'BOOL'    : SimDataType.BOOL,
            'INT4'    : SimDataType.INT4,
            'INT8'    : SimDataType.INT8,
            'INT16'   : SimDataType.INT16,
            'INT32'   : SimDataType.INT32,
            'INT64'   : SimDataType.INT64,
            'UINT4'   : SimDataType.UINT4,
            'UINT8 '  : SimDataType.UINT8,
            'UINT16'  : SimDataType.UINT16,
            'UINT32'  : SimDataType.UINT32,
            'UINT64'  : SimDataType.UINT64,
            'BFLOAT8' : SimDataType.BFLOAT8,
            'BFLOAT16': SimDataType.BFLOAT16,
            'FLOAT16' : SimDataType.FLOAT16,
            'FLOAT32' : SimDataType.FLOAT32,
            'FLOAT64' : SimDataType.FLOAT64,
            }
    try:
        res = tbl[dtype.upper()]
    except KeyError:
        res = SimDataType.UNKNOWN
    return res

def get_bpe(dtype: SimDataType) -> int:
    tbl = {
            SimDataType.BOOL      : 1,
            SimDataType.INT4      : 1,
            SimDataType.INT8      : 1,
            SimDataType.INT16     : 2,
            SimDataType.INT32     : 4,
            SimDataType.INT64     : 8,
            SimDataType.UINT4     : 1,
            SimDataType.UINT8     : 1,
            SimDataType.UINT16    : 2,
            SimDataType.UINT32    : 4,
            SimDataType.UINT64    : 8,
            SimDataType.BFLOAT8   : 1,
            SimDataType.BFLOAT16  : 2,
            SimDataType.FLOAT16   : 2,
            SimDataType.FLOAT32   : 4,
            SimDataType.FLOAT64   : 8,
            SimDataType.UNKNOWN   : -1,
            }
    return tbl[dtype]
