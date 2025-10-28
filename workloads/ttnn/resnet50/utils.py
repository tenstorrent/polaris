#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttsim.front.ttnn as ttnn
from ttsim.utils.common import dict2obj

from workloads.ttnn.common import create_ttnn_tensor, update_param_attrs

import math

def create_rn50_params(*, device, dtype):
    params = {
            'conv1': {
                'weight': create_ttnn_tensor(shape=[64, 3, 4, 4]),
                'bias'  : create_ttnn_tensor(shape=[64]),
                },
            'fc': {
                'weight': create_ttnn_tensor(shape=[1000, 2048]),
                'bias'  : create_ttnn_tensor(shape=[1000]),
                },
            }

    params['layer1'] = [ #type: ignore
            { #0
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[64, 64, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[64]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[64, 64, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[64]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[256, 64, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'downsample': {
                 'weight': create_ttnn_tensor(shape=[256, 64, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             },

             { #1
              'conv1': {
                  'weight': create_ttnn_tensor(shape=[64, 256, 1, 1]),
                  'bias'  : create_ttnn_tensor(shape=[64]),
                  },
              'conv2': {
                  'weight': create_ttnn_tensor(shape=[64, 64, 3, 3]),
                  'bias'  : create_ttnn_tensor(shape=[64]),
                  },
              'conv3': {
                  'weight': create_ttnn_tensor(shape=[256, 64, 1, 1]),
                  'bias'  : create_ttnn_tensor(shape=[256]),
                  },
              },

             { #2
              'conv1': {
                  'weight': create_ttnn_tensor(shape=[64, 256, 1, 1]),
                  'bias'  : create_ttnn_tensor(shape=[64]),
                  },
              'conv2': {
                  'weight': create_ttnn_tensor(shape=[64, 64, 3, 3]),
                  'bias'  : create_ttnn_tensor(shape=[64]),
                  },
              'conv3': {
                  'weight': create_ttnn_tensor(shape=[256, 64, 1, 1]),
                  'bias'  : create_ttnn_tensor(shape=[256]),
                  },
              },
             ]

    params['layer2'] = [ #type: ignore
            { #0
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[128, 256, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[128]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[128, 128, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[128]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[512, 128, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[512]),
                 },

             'downsample': {
                 'weight': create_ttnn_tensor(shape=[512, 256, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[512]),
                 },
             },
            { #1
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[128, 512, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[128]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[128, 128, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[128]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[512, 128, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[512]),
                 },
             },
            { #2
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[128, 512, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[128]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[128, 128, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[128]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[512, 128, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[512]),
                 },
             },
            { #3
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[128, 512, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[128]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[128, 128, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[128]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[512, 128, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[512]),
                 },
             },
            ]

    params['layer3'] = [ #type: ignore
            { #0
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[256, 512, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[256, 256, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[1024, 256, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[1024]),
                 },
             'downsample': {
                 'weight': create_ttnn_tensor(shape=[1024, 512, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[1024]),
                 }, #1
             },
            { #1
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[256, 1024, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[256, 256, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[1024, 256, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[1024]),
                 },
             },
            { #2
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[256, 1024, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[256, 256, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[1024, 256, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[1024]),
                 },
             },
            { #3
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[256, 1024, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[256, 256, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[1024, 256, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[1024]),
                 },
             },
            { #4
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[256, 1024, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[256, 256, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[1024, 256, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[1024]),
                 },
             },
            { #5
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[256, 1024, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[256, 256, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[256]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[1024, 256, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[1024]),
                 },
             },
            ]

    params['layer4'] = [ #type: ignore
            { #0
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[512, 1024, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[512]),},
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[512, 512, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[512]),},
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[2048, 512, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[2048]),},
             'downsample': {
                 'weight': create_ttnn_tensor(shape=[2048, 1024, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[2048]),
                 },
             },
            { #1
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[512, 2048, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[512]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[512, 512, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[512]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[2048, 512, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[2048]),
                 },
             },
            { #2
             'conv1': {
                 'weight': create_ttnn_tensor(shape=[512, 2048, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[512]),
                 },
             'conv2': {
                 'weight': create_ttnn_tensor(shape=[512, 512, 3, 3]),
                 'bias'  : create_ttnn_tensor(shape=[512]),
                 },
             'conv3': {
                 'weight': create_ttnn_tensor(shape=[2048, 512, 1, 1]),
                 'bias'  : create_ttnn_tensor(shape=[2048]),
                 },
             },
            ]

    update_param_attrs(params, device=device, dtype=dtype)

    return dict2obj(params)

############################x############################x############################x############################
def get_core_grid_from_num_cores(num_cores: int, grid_rows: int, grid_cols: int):
    """
    columns = num_cores // grid_rows
    assert columns <= grid_cols, "Not enough cores for specified core grid"
    ranges = []
    if columns != 0:
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_rows - 1, columns - 1),
            )
        )
    remainder = num_cores % grid_rows
    if remainder != 0:
        assert columns + 1 <= grid_cols, "Not enough cores for specified core grid"
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, columns),
                ttnn.CoreCoord(remainder - 1, columns),
            )
        )
    return ttnn.CoreRangeSet({*ranges})
    """
    raise NotImplementedError('get_core_grid_from_num_cores not implemented yet')

def find_closest_largest_divisor(num: int, start_divisor: int) -> int:
    divisor = start_divisor
    while num % divisor != 0:
        divisor -= 1
    return divisor

# Determins input memory config for a height sharded conv operation.
# If override_num_cores is set to True, the number of cores will be overriden to the closest largest divisor of the number of tiles
# This will avoid default conv codepath which can pad-up the nhw num tiles and produce padded output
# This can lead to issues with data-movment ops not handling padding correctly
def get_conv_input_memory_config(
        batch_size              : int,
        input_channels          : int,
        input_height            : int,
        input_width             : int,
        output_channels         : int,
        output_height           : int,
        output_width            : int,
        compute_grid            : tuple[int, int],#ttnn.CoreGrid,
        input_channels_alignment: int,
        override_num_cores      : bool,

        ) -> ttnn.MemoryConfig:

    """
    parallel_config = ttnn._ttnn.operations.conv.determine_parallel_config(
            shard_layout            = ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            batch_size              = batch_size,
            input_channels          = input_channels,
            output_height           = output_height,
            output_width            = output_width,
            output_channels         = output_channels,
            compute_grid_size       = compute_grid,
            block_shard_orientation = ttnn.ShardOrientation.ROW_MAJOR,
            enable_channels_padding = True,
            )

    if override_num_cores:
        nhw_ntiles           = math.ceil(batch_size * output_height * output_width / 32)
        num_cores_nwh        = find_closest_largest_divisor(nhw_ntiles, compute_grid.x * compute_grid.y)
        parallel_config.grid = get_core_grid_from_num_cores(num_cores_nwh, compute_grid.x, compute_grid.y)

    memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config (
            tensor_shape=ttnn.Shape([1, 1,
                                     input_width * input_height * batch_size,
                                     nearest_y(input_channels, input_channels_alignment,),
                                     ]),
            parallel_config=parallel_config,
            tile_size=32,
            )
    return memory_config
    """
    return ttnn.MemoryConfig.L1 #L1_WIDTH_SHARDED_MEMORY_CONFIG


def _nearest_y(x, y): return math.ceil(x / y) * y

def _nearest_32(x): return math.ceil(x / 32) * 32

def is_blackhole():
    ARCH_NAME = ttnn.get_arch_name()
    return "blackhole" in ARCH_NAME

def is_wormhole_b0():
    ARCH_NAME = ttnn.get_arch_name()
    return "wormhole_b0" in ARCH_NAME

def is_grayskull():
    ARCH_NAME = ttnn.get_arch_name()
    return "grayskull" in ARCH_NAME
