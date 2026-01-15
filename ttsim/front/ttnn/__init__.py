#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .device import open_device, close_device, ARCH, num_cores_to_corerangeset, create_sharded_memory_config, ReadDeviceProfiler
from .device import interleaved_to_sharded
from .tensor import _rand, full, zeros, ones, from_torch, to_torch, to_layout, to_device, DataType, ShardTensor2dMesh, typecast, pad
from .tensor import Layout, Shape, as_tensor, arange, stack, ShardStrategy, unsqueeze_to_4D, ReplicateTensorToMesh
from .config import Conv2dConfig, WormholeComputeKernelConfig, init_device_compute_kernel_config
from .config import MatmulMultiCoreReuseMultiCast1DProgramConfig
from .buffer import TensorMemoryLayout, ShardOrientation, BufferType
from .memory import MemoryConfig, create_sharded_memory_config_, get_memory_config, to_memory_config
from .core   import CoreCoord, CoreRange, CoreRangeSet, CoreGrid
from .op     import *

float32  = DataType.FLOAT32
bfloat16 = DataType.BFLOAT16
int64    = DataType.INT64
uint32   = DataType.UINT32
bfloat8_b = DataType.BFLOAT8_B
bool      = DataType.BOOL
int32     = DataType.INT32

ROW_MAJOR_LAYOUT = Layout.ROW_MAJOR_LAYOUT
TILE_LAYOUT      = Layout.TILE_LAYOUT
TILE_SIZE        = 32

DRAM_MEMORY_CONFIG = MemoryConfig.DRAM
L1_MEMORY_CONFIG   = MemoryConfig.L1

L1_WIDTH_SHARDED_MEMORY_CONFIG = 0

#placeholders

def name_to_datatype(dtype_name: str) -> DataType:
    try:
        return DataType[dtype_name.upper()]
    except KeyError:
        valid_dtypes = [dt.name.lower() for dt in DataType]
        raise ValueError(
            f"Invalid dtype name '{dtype_name}'. Valid options are: {', '.join(valid_dtypes)}"
        )

def get_arch_name():
    return ARCH.WORMHOLE_B0.cname

def is_tensor_storage_on_device(ttnn_tensor_like):
    return True

def prepare_conv_weights(weight_tensor, weights_format, input_memory_config, input_layout,
                         has_bias, input_dtype, **kwargs):
    return weight_tensor

def prepare_conv_bias(bias_tensor, input_memory_config, input_layout, input_dtype, **kwargs,):
    return bias_tensor


def deallocate(x): pass
def reallocate(x): return x

def untilize_with_unpadding(x, *args, **kwargs): return x
def tilize_with_val_padding(x, *args, **kwargs): return x
