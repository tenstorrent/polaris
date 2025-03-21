import os
from typing import Optional, List, Literal, Annotated
from pydantic import BaseModel, Field

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
    ip_overrides: Optional[dict[str, int | float]] = {}


class PYDPkgComputeValidator(BaseModel, extra='forbid'):
    ip: str
    num_units: int
    ramp_penalty: Optional[float] = 0.0
    freq_MHz: Optional[float] = None
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

