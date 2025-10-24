#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice
from workloads.ttnn.vgg_unet.model_preprocessing import create_vgg_unet_model_parameters
from loguru import logger
from ttsim.utils.common import setup_logger

# shard concat function
def sharded_concat(input_tensors, num_cores=64, dim=1):
    return ttnn.concat(input_tensors[0], input_tensors[1], axis=dim)

# TTNN conv class
class Conv:
    def __init__(
        self,
        device,
        conv_param,
        conv_pth,
        input_tensor_layout=ttnn.TILE_LAYOUT,
    ) -> None:
        self.conv_param = conv_param
        self.conv_pth = conv_pth
        self.device = device
        self.cache = {} # type: ignore[var-annotated]

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        self.conv_output_dtype = conv_param.dtype
        output_layout = ttnn.ROW_MAJOR_LAYOUT

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            activation=conv_param.activation,
            shard_layout=ttnn.ShardOrientation.ROW_MAJOR, # type: ignore[arg-type]
            reshard_if_not_optimal=False,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            output_layout=output_layout,
        )

        self.bias = conv_pth.bias
        self.weight = conv_pth.weight

        self.conv_kwargs = {
            "in_channels": conv_param.in_channels,
            "out_channels": conv_param.out_channels,
            "batch_size": conv_param.batch_size,
            "input_height": conv_param.input_height,
            "input_width": conv_param.input_width,
            "kernel_size": conv_param.kernel_size,
            "stride": conv_param.stride,
            "padding": conv_param.padding,
            "dilation": conv_param.dilation,
            "groups": conv_param.groups,
            "device": device,
            "conv_config": self.conv_config,
        }

    def __str__(self) -> str:
        return f"Conv: {self.weight.shape} {self.bias.shape} {self.conv_kwargs['kernel_size']}"

    def __call__(self, x):
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            padding=self.conv_kwargs["padding"],
            compute_config=self.compute_config,
            kernel_size=self.conv_kwargs["kernel_size"],
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.conv_output_dtype,
        )
        return x


# TTNN Conv Transpose class
class Conv_transpose:
    def __init__(
        self,
        device,
        conv_param,
        conv_pth,
    ) -> None:
        self.conv_param = conv_param
        self.conv_pth = conv_pth
        self.device = device
        self.cache = {} # type: ignore[var-annotated]

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        output_layout = ttnn.ROW_MAJOR_LAYOUT

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=conv_param.shard_layout,
            reshard_if_not_optimal=conv_param.reshard_if_not_optimal,
            deallocate_activation=conv_param.deallocate_activation,
            enable_act_double_buffer=conv_param.enable_act_double_buffer,
            enable_weights_double_buffer=True,
            output_layout=output_layout,
        )
        if conv_param.act_block_h is not None:
            self.conv_config.act_block_h_override = conv_param.act_block_h

        self.bias = conv_pth.bias
        self.weight = conv_pth.weight

        self.conv_kwargs = {
            "in_channels": conv_param.in_channels,
            "out_channels": conv_param.out_channels,
            "batch_size": conv_param.batch_size,
            "input_height": conv_param.input_height,
            "input_width": conv_param.input_width,
            "kernel_size": conv_param.kernel_size,
            "stride": conv_param.stride,
            "padding": conv_param.padding,
            "dilation": conv_param.dilation,
            "groups": conv_param.groups,
            "device": device,
            "conv_config": self.conv_config,
            "output_padding": conv_param.output_padding,
        }

    def __str__(self) -> str:
        return f"Conv: {self.weight.shape} {self.bias.shape} {self.conv_kwargs['kernel_size']}"

    def __call__(self, x):
        x = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            padding=self.conv_kwargs["padding"],
            compute_config=self.compute_config,
            kernel_size=self.conv_kwargs["kernel_size"],
            stride=self.conv_kwargs["stride"],
            return_output_dim=True,
            return_weights_and_bias=True,
            mirror_kernel=True,
            dtype=self.conv_param.dtype,
        )
        return x


class Tt_decoder_block:
    def __init__(self, device, conv_args, parameters) -> None:
        self.conv_args = conv_args
        self.up = Conv_transpose(device, conv_args.up, parameters.up)
        self.conv1 = Conv(device, conv_args.conv_block.conv1, parameters.conv1)
        self.conv2 = Conv(device, conv_args.conv_block.conv2, parameters.conv2)

    def __call__(self, x, cat_in):
        x = self.up(x)
        x = sharded_concat([x, cat_in])
        if self.conv_args.conv_block.conv1.do_sharded_to_interleaved:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = self.conv1(x)
        if self.conv_args.conv_block.conv2.do_sharded_to_interleaved:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = self.conv2(x)
        return x


class Tt_vgg_unet:
    def __init__(self, device, parameters, conv_args) -> None:
        self.conv_args = conv_args
        self.parameters = parameters
        self.s1_0 = Conv(device, conv_args.s1["0"], parameters["0"], input_tensor_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.s1_2 = Conv(device, conv_args.s1["2"], parameters["2"])
        self.s2_5 = Conv(device, conv_args.s2["5"], parameters["5"])
        self.s2_7 = Conv(device, conv_args.s2["7"], parameters["7"])
        self.s3_10 = Conv(device, conv_args.s3["10"], parameters["10"])
        self.s3_12 = Conv(device, conv_args.s3["12"], parameters["12"])
        self.s3_14 = Conv(device, conv_args.s3["14"], parameters["14"])
        self.s3_16 = Conv(device, conv_args.s3["16"], parameters["16"])
        self.s4_19 = Conv(device, conv_args.s4["19"], parameters["19"])
        self.s4_21 = Conv(device, conv_args.s4["21"], parameters["21"])
        self.s4_23 = Conv(device, conv_args.s4["23"], parameters["23"])
        self.s4_25 = Conv(device, conv_args.s4["25"], parameters["25"])
        self.b1_28 = Conv(device, conv_args.b1["28"], parameters["28"])
        self.b1_30 = Conv(device, conv_args.b1["30"], parameters["30"])
        self.b1_32 = Conv(device, conv_args.b1["32"], parameters["32"])
        self.b1_34 = Conv(device, conv_args.b1["34"], parameters["34"])
        self.d1 = Tt_decoder_block(device, conv_args.d1, parameters.d1)
        self.d2 = Tt_decoder_block(device, conv_args.d2, parameters.d2)
        self.d3 = Tt_decoder_block(device, conv_args.d3, parameters.d3)
        self.d4 = Tt_decoder_block(device, conv_args.d4, parameters.d4)
        self.out = Conv(device, conv_args.out, parameters.out)

    def __call__(self, input, min_channels=3):
        n, c, h, w = input.shape
        channel_padding_needed = min_channels - c
        if channel_padding_needed > 0:
            x = ttnn.pad(input, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
            ttnn.deallocate(input)
            input = x
        x = input
        x = self.s1_0(x)
        x = self.s1_2(x)
        s1 = x

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.conv_args.s2["4"].batch_size,
            input_h=self.conv_args.s2["4"].input_height,
            input_w=self.conv_args.s2["4"].input_width,
            channels=x.shape[3],
            kernel_size=[self.conv_args.s2["4"].kernel_size, self.conv_args.s2["4"].kernel_size],
            stride=[self.conv_args.s2["4"].stride, self.conv_args.s2["4"].stride],
            padding=[self.conv_args.s2["4"].padding, self.conv_args.s2["4"].padding],
            dilation=[self.conv_args.s2["4"].dilation, self.conv_args.s2["4"].dilation],
        )
        x = self.s2_5(x)
        x = self.s2_7(x)
        s2 = x

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.conv_args.s3["9"].batch_size,
            input_h=self.conv_args.s3["9"].input_height,
            input_w=self.conv_args.s3["9"].input_width,
            channels=x.shape[3],
            kernel_size=[self.conv_args.s3["9"].kernel_size, self.conv_args.s3["9"].kernel_size],
            stride=[self.conv_args.s3["9"].stride, self.conv_args.s3["9"].stride],
            padding=[self.conv_args.s3["9"].padding, self.conv_args.s3["9"].padding],
            dilation=[self.conv_args.s3["9"].dilation, self.conv_args.s3["9"].dilation],
        )

        x = self.s3_10(x)

        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        sharded_memory_config = ttnn.create_sharded_memory_config(
            [
                512,
                32,
            ],
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.to_memory_config(x, sharded_memory_config)

        x = self.s3_12(x)
        x = self.s3_14(x)
        x = self.s3_16(x)

        s3 = x

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.conv_args.s4["18"].batch_size,
            input_h=self.conv_args.s4["18"].input_height,
            input_w=self.conv_args.s4["18"].input_width,
            channels=x.shape[3],
            kernel_size=[self.conv_args.s4["18"].kernel_size, self.conv_args.s4["18"].kernel_size],
            stride=[self.conv_args.s4["18"].stride, self.conv_args.s4["18"].stride],
            padding=[self.conv_args.s4["18"].padding, self.conv_args.s4["18"].padding],
            dilation=[self.conv_args.s4["18"].dilation, self.conv_args.s4["18"].dilation],
        )

        x = self.s4_19(x)
        x = self.s4_21(x)
        x = self.s4_23(x)
        x = self.s4_25(x)
        s4 = x

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.conv_args.b1["27"].batch_size,
            input_h=self.conv_args.b1["27"].input_height,
            input_w=self.conv_args.b1["27"].input_width,
            channels=x.shape[3],
            kernel_size=[self.conv_args.b1["27"].kernel_size, self.conv_args.b1["27"].kernel_size],
            stride=[self.conv_args.b1["27"].stride, self.conv_args.b1["27"].stride],
            padding=[self.conv_args.b1["27"].padding, self.conv_args.b1["27"].padding],
            dilation=[self.conv_args.b1["27"].dilation, self.conv_args.b1["27"].dilation],
        )

        x = self.b1_28(x)
        x = self.b1_30(x)
        x = self.b1_32(x)
        x = self.b1_34(x)

        x = self.d1(x, s4)
        ttnn.deallocate(s4)
        x = self.d2(x, s3)
        ttnn.deallocate(s3)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = self.d3(x, s2)
        ttnn.deallocate(s2)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = self.d4(x, s1)
        ttnn.deallocate(s1)
        x = self.out(x)
        return x

def test_vgg_unet(wlname: str, device: TTNNDevice, gcfg: dict):
    input_tensor = ttnn._rand(shape=[1, 3, 256, 256], device=device, dtype=ttnn.float32)
    parameters = create_vgg_unet_model_parameters(device)
    ttnn_model = Tt_vgg_unet(device, parameters, parameters.conv_args)
    result = ttnn_model(input_tensor)
    logger.info("VGG UNet input shape: {}", input_tensor.shape)
    logger.info("VGG UNet output shape: {}", result.shape)
    return result

if __name__ == "__main__":
    setup_logger(level='INFO')
    device = ttnn.open_device(device_id=0)
    test_vgg_unet('vgg_unet', device, {})
