#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn

def create_vgg_unet_model_parameters(device):
    class Parameters:
        class WeightBias:
            def __init__(self, w, b):
                self.weight = w
                self.bias   = b
                self.layout = ttnn.Layout.ROW_MAJOR
                self.dtype  = ttnn.DataType.FLOAT32

        class ConvArgs:

            class DecodeBlockArgs:

                class ConvBlockArgs:

                    class WeightBiasArgs:
                        def __init__(self, w, b):
                            self._data = {}
                            self.weight = w
                            self.bias   = b
                            self.layout = ttnn.Layout.ROW_MAJOR
                            self.dtype  = ttnn.DataType.FLOAT32
                            self.in_channels = None
                            self.out_channels = None
                            self.batch_size = None
                            self.input_height = None
                            self.input_width = None
                            self.kernel_size = [3, 3]
                            self.stride = [1, 1]
                            self.padding = [1, 1]
                            self.groups = 1
                            self.dilation = 1
                            self.activation = "relu"
                            self.shard_layout = None
                            self.do_sharded_to_interleaved = False
                        def __getitem__(self, key):
                            return self._data[key]
                        def __setitem__(self, key, value):
                            self._data[key] = value

                    def __init__(self, w1, b1, w2, b2):
                        self._data = {}
                        self.conv1 = self.WeightBiasArgs(w1, b1)
                        self.conv2 = self.WeightBiasArgs(w2, b2)
                    def __getitem__(self, key):
                        return self._data[key]
                    def __setitem__(self, key, value):
                        self._data[key] = value

                class UpArgs:
                    def __init__(self, w, b):
                        self._data = {}
                        self.weight = w
                        self.bias   = b
                        self.layout = ttnn.Layout.ROW_MAJOR
                        self.dtype  = ttnn.DataType.FLOAT32
                        self.in_channels = None
                        self.out_channels = None
                        self.batch_size = None
                        self.input_height = None
                        self.input_width = None
                        self.kernel_size = [3, 3]
                        self.stride = [1, 1]
                        self.padding = [1, 1]
                        self.groups = 1
                        self.dilation = 1
                        self.shard_layout = None
                        self.reshard_if_not_optimal = False
                        self.deallocate_activation = False
                        self.enable_act_double_buffer = False
                        self.act_block_h = None
                        self.output_padding = 0

                    def __getitem__(self, key):
                        return self._data[key]
                    def __setitem__(self, key, value):
                        self._data[key] = value

                def __init__(self, w, b):
                    self.up = self.UpArgs(w, b)
                    self.conv_block = self.ConvBlockArgs(None, None, None, None)
                    self.conv1 = Parameters.WeightBias(None, None)
                    self.conv2 = Parameters.WeightBias(None, None)

            class ConvDictArgs:
                def __init__(self):
                    self._data = {}
                    self.dtype =  ttnn.DataType.BFLOAT16
                    self.in_channels = None
                    self.out_channels = None
                    self.batch_size = None
                    self.input_height = None
                    self.input_width = None
                    self.kernel_size = [3, 3]
                    self.stride = [1, 1]
                    self.padding = [1, 1]
                    self.groups = 1
                    self.dilation = 1
                    self.activation = "relu"
                def __getitem__(self, key):
                    return self._data[key]
                def __setitem__(self, key, value):
                    self._data[key] = value

            def __init__(self, w, b):
                self.s1 = {"0": self.ConvDictArgs(), "2": self.ConvDictArgs()}
                self.s2 = {"4": self.ConvDictArgs(), "5": self.ConvDictArgs(), "7": self.ConvDictArgs()}
                self.s3 = {"9": self.ConvDictArgs(), "10": self.ConvDictArgs(), "12": self.ConvDictArgs(), "14": self.ConvDictArgs(), "16": self.ConvDictArgs()}
                self.s4 = {"18": self.ConvDictArgs(), "19": self.ConvDictArgs(), "21": self.ConvDictArgs(), "23": self.ConvDictArgs(), "25": self.ConvDictArgs()}
                self.b1 = {"27": self.ConvDictArgs(), "28": self.ConvDictArgs(), "30": self.ConvDictArgs(), "32": self.ConvDictArgs(), "34": self.ConvDictArgs()}
                self.d1 = self.DecodeBlockArgs(w, b)
                self.d2 = self.DecodeBlockArgs(w, b)
                self.d3 = self.DecodeBlockArgs(w, b)
                self.d4 = self.DecodeBlockArgs(w, b)
                self.out = self.ConvDictArgs()

        def __init__(self):
            self._data = {"0": self.WeightBias(None, None),
                          "2": self.WeightBias(None, None),
                          "5": self.WeightBias(None, None),
                          "7": self.WeightBias(None, None),
                          "10": self.WeightBias(None, None),
                          "12": self.WeightBias(None, None),
                          "14": self.WeightBias(None, None),
                          "16": self.WeightBias(None, None),
                          "19": self.WeightBias(None, None),
                          "21": self.WeightBias(None, None),
                          "23": self.WeightBias(None, None),
                          "25": self.WeightBias(None, None),
                          "28": self.WeightBias(None, None),
                          "30": self.WeightBias(None, None),
                          "32": self.WeightBias(None, None),
                          "34": self.WeightBias(None, None)}
            self.d1 = self.ConvArgs.DecodeBlockArgs(None, None)
            self.d2 = self.ConvArgs.DecodeBlockArgs(None, None)
            self.d3 = self.ConvArgs.DecodeBlockArgs(None, None)
            self.d4 = self.ConvArgs.DecodeBlockArgs(None, None)
            self.out = self.WeightBias(None, None)

            self._data["0"].weight = ttnn.Tensor(shape=ttnn.Shape([64, 3, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["0"].bias   = ttnn.Tensor(shape=ttnn.Shape([64]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["2"].weight = ttnn.Tensor(shape=ttnn.Shape([64, 64, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["2"].bias   = ttnn.Tensor(shape=ttnn.Shape([64]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["5"].weight = ttnn.Tensor(shape=ttnn.Shape([128, 64, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["5"].bias   = ttnn.Tensor(shape=ttnn.Shape([128]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["7"].weight = ttnn.Tensor(shape=ttnn.Shape([128, 128, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["7"].bias   = ttnn.Tensor(shape=ttnn.Shape([128]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["10"].weight = ttnn.Tensor(shape=ttnn.Shape([256, 128, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["10"].bias   = ttnn.Tensor(shape=ttnn.Shape([256]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["12"].weight = ttnn.Tensor(shape=ttnn.Shape([256, 256, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["12"].bias   = ttnn.Tensor(shape=ttnn.Shape([256]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["14"].weight = ttnn.Tensor(shape=ttnn.Shape([256, 256, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["14"].bias   = ttnn.Tensor(shape=ttnn.Shape([256]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["16"].weight = ttnn.Tensor(shape=ttnn.Shape([256, 256, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["16"].bias   = ttnn.Tensor(shape=ttnn.Shape([256]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["19"].weight = ttnn.Tensor(shape=ttnn.Shape([512, 256, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["19"].bias   = ttnn.Tensor(shape=ttnn.Shape([512]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["21"].weight = ttnn.Tensor(shape=ttnn.Shape([512, 512, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["21"].bias   = ttnn.Tensor(shape=ttnn.Shape([512]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["23"].weight = ttnn.Tensor(shape=ttnn.Shape([512, 512, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["23"].bias   = ttnn.Tensor(shape=ttnn.Shape([512]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["25"].weight = ttnn.Tensor(shape=ttnn.Shape([512, 512, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["25"].bias   = ttnn.Tensor(shape=ttnn.Shape([512]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["28"].weight = ttnn.Tensor(shape=ttnn.Shape([512, 512, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["28"].bias   = ttnn.Tensor(shape=ttnn.Shape([512]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["30"].weight = ttnn.Tensor(shape=ttnn.Shape([512, 512, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["30"].bias   = ttnn.Tensor(shape=ttnn.Shape([512]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["32"].weight = ttnn.Tensor(shape=ttnn.Shape([512, 512, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["32"].bias   = ttnn.Tensor(shape=ttnn.Shape([512]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["34"].weight = ttnn.Tensor(shape=ttnn.Shape([512, 512, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self._data["34"].bias   = ttnn.Tensor(shape=ttnn.Shape([512]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)

            self.conv_args = self.ConvArgs(None, None)

            self.d1.up.weight    = ttnn.Tensor(shape=ttnn.Shape([512, 512, 2, 2]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d1.up.bias      = ttnn.Tensor(shape=ttnn.Shape([512]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d1.conv1.weight = ttnn.Tensor(shape=ttnn.Shape([512, 1024, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d1.conv1.bias   = ttnn.Tensor(shape=ttnn.Shape([512]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d1.conv2.weight = ttnn.Tensor(shape=ttnn.Shape([512, 512, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d1.conv2.bias   = ttnn.Tensor(shape=ttnn.Shape([512]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d2.up.weight    = ttnn.Tensor(shape=ttnn.Shape([512, 256, 2, 2]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d2.up.bias      = ttnn.Tensor(shape=ttnn.Shape([256]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d2.conv1.weight = ttnn.Tensor(shape=ttnn.Shape([256, 512, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d2.conv1.bias   = ttnn.Tensor(shape=ttnn.Shape([256]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d2.conv2.weight = ttnn.Tensor(shape=ttnn.Shape([256, 256, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d2.conv2.bias   = ttnn.Tensor(shape=ttnn.Shape([256]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d3.up.weight    = ttnn.Tensor(shape=ttnn.Shape([256, 128, 2, 2]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d3.up.bias      = ttnn.Tensor(shape=ttnn.Shape([128]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d3.conv1.weight = ttnn.Tensor(shape=ttnn.Shape([128, 256, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d3.conv1.bias   = ttnn.Tensor(shape=ttnn.Shape([128]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d3.conv2.weight = ttnn.Tensor(shape=ttnn.Shape([128, 128, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d3.conv2.bias   = ttnn.Tensor(shape=ttnn.Shape([128]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d4.up.weight    = ttnn.Tensor(shape=ttnn.Shape([128, 64, 2, 2]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d4.up.bias      = ttnn.Tensor(shape=ttnn.Shape([64]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d4.conv1.weight = ttnn.Tensor(shape=ttnn.Shape([64, 128, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d4.conv1.bias   = ttnn.Tensor(shape=ttnn.Shape([64]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d4.conv2.weight = ttnn.Tensor(shape=ttnn.Shape([64, 64, 3, 3]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.d4.conv2.bias   = ttnn.Tensor(shape=ttnn.Shape([64]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.out.weight = ttnn.Tensor(shape=ttnn.Shape([1, 64, 1, 1]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)
            self.out.bias   = ttnn.Tensor(shape=ttnn.Shape([1]), layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.FLOAT32, device=device)

        def __getitem__(self, key):
            return self._data[key]
        def __setitem__(self, key, value):
            self._data[key] = value

    parameters = Parameters()
    parameters.conv_args.s1["0"]["act_block_h"] = None
    parameters.conv_args.s1["0"]["enable_split_reader"] = False
    parameters.conv_args.s1["0"]["enable_act_double_buffer"] = False
    parameters.conv_args.s1["0"]["deallocate_activation"] = True
    parameters.conv_args.s1["0"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s1["0"]["shard_layout"] = None
    parameters.conv_args.s1["0"]["activation"] = "relu"
    parameters.conv_args.s1["0"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s1["0"].in_channels = 3
    parameters.conv_args.s1["0"].out_channels = 64
    parameters.conv_args.s1["0"].batch_size = 1
    parameters.conv_args.s1["0"].input_height = 256
    parameters.conv_args.s1["0"].input_width = 256

    parameters.conv_args.s1["2"]["act_block_h"] = None
    parameters.conv_args.s1["2"]["enable_split_reader"] = False
    parameters.conv_args.s1["2"]["enable_act_double_buffer"] = False
    parameters.conv_args.s1["2"]["deallocate_activation"] = True
    parameters.conv_args.s1["2"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s1["2"]["shard_layout"] = None
    parameters.conv_args.s1["2"]["activation"] = "relu"
    parameters.conv_args.s1["2"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s1["2"].in_channels = 64
    parameters.conv_args.s1["2"].out_channels = 64
    parameters.conv_args.s1["2"].batch_size = 1
    parameters.conv_args.s1["2"].input_height = 256
    parameters.conv_args.s1["2"].input_width = 256

    parameters.conv_args.s2["4"].in_channels = 64
    parameters.conv_args.s2["4"].batch_size = 1
    parameters.conv_args.s2["4"].input_height = 256
    parameters.conv_args.s2["4"].input_width = 256
    parameters.conv_args.s2["4"].kernel_size = 2 # type: ignore[assignment]
    parameters.conv_args.s2["4"].stride = 2 # type: ignore[assignment]
    parameters.conv_args.s2["4"].padding = 0 # type: ignore[assignment]
    parameters.conv_args.s2["4"].dilation = 1
    parameters.conv_args.s2["4"].dtype = ttnn.DataType.BFLOAT16

    parameters.conv_args.s2["5"]["act_block_h"] = None
    parameters.conv_args.s2["5"]["enable_split_reader"] = False
    parameters.conv_args.s2["5"]["enable_act_double_buffer"] = False
    parameters.conv_args.s2["5"]["deallocate_activation"] = True
    parameters.conv_args.s2["5"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s2["5"]["shard_layout"] = None
    parameters.conv_args.s2["5"]["activation"] = "relu"
    parameters.conv_args.s2["5"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s2["5"].in_channels = 64
    parameters.conv_args.s2["5"].out_channels = 128
    parameters.conv_args.s2["5"].batch_size = 1
    parameters.conv_args.s2["5"].input_height = 128
    parameters.conv_args.s2["5"].input_width = 128

    parameters.conv_args.s2["7"]["act_block_h"] = None
    parameters.conv_args.s2["7"]["enable_split_reader"] = False
    parameters.conv_args.s2["7"]["enable_act_double_buffer"] = False
    parameters.conv_args.s2["7"]["deallocate_activation"] = True
    parameters.conv_args.s2["7"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s2["7"]["shard_layout"] = None
    parameters.conv_args.s2["7"]["activation"] = "relu"
    parameters.conv_args.s2["7"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s2["7"].in_channels = 128
    parameters.conv_args.s2["7"].out_channels = 128
    parameters.conv_args.s2["7"].batch_size = 1
    parameters.conv_args.s2["7"].input_height = 128
    parameters.conv_args.s2["7"].input_width = 128

    parameters.conv_args.s3["9"].in_channels = 128
    parameters.conv_args.s3["9"].input_height = 128
    parameters.conv_args.s3["9"].input_width = 128
    parameters.conv_args.s3["9"].kernel_size = 2 # type: ignore[assignment]
    parameters.conv_args.s3["9"].stride = 2 # type: ignore[assignment]
    parameters.conv_args.s3["9"].padding = 0 # type: ignore[assignment]
    parameters.conv_args.s3["9"].dilation = 1
    parameters.conv_args.s3["9"].batch_size = 1
    parameters.conv_args.s3["9"].dtype = ttnn.DataType.BFLOAT16

    parameters.conv_args.s3["10"]["act_block_h"] = None
    parameters.conv_args.s3["10"]["enable_split_reader"] = False
    parameters.conv_args.s3["10"]["enable_act_double_buffer"] = False
    parameters.conv_args.s3["10"]["deallocate_activation"] = True
    parameters.conv_args.s3["10"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s3["10"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.s3["10"]["activation"] = "relu"
    parameters.conv_args.s3["10"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s3["10"].in_channels = 128
    parameters.conv_args.s3["10"].out_channels = 256
    parameters.conv_args.s3["10"].batch_size = 1
    parameters.conv_args.s3["10"].input_height = 64
    parameters.conv_args.s3["10"].input_width = 64

    parameters.conv_args.s3["12"]["act_block_h"] = None
    parameters.conv_args.s3["12"]["enable_split_reader"] = False
    parameters.conv_args.s3["12"]["enable_act_double_buffer"] = False
    parameters.conv_args.s3["12"]["deallocate_activation"] = True
    parameters.conv_args.s3["12"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s3["12"]["shard_layout"] = None
    parameters.conv_args.s3["12"]["activation"] = "relu"
    parameters.conv_args.s3["12"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s3["12"].in_channels = 256
    parameters.conv_args.s3["12"].out_channels = 256
    parameters.conv_args.s3["12"].batch_size = 1
    parameters.conv_args.s3["12"].input_height = 64
    parameters.conv_args.s3["12"].input_width = 64

    parameters.conv_args.s3["14"]["act_block_h"] = None
    parameters.conv_args.s3["14"]["enable_split_reader"] = False
    parameters.conv_args.s3["14"]["enable_act_double_buffer"] = False
    parameters.conv_args.s3["14"]["deallocate_activation"] = True
    parameters.conv_args.s3["14"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s3["14"]["shard_layout"] = None
    parameters.conv_args.s3["14"]["activation"] = "relu"
    parameters.conv_args.s3["14"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s3["14"].in_channels = 256
    parameters.conv_args.s3["14"].out_channels = 256
    parameters.conv_args.s3["14"].batch_size = 1
    parameters.conv_args.s3["14"].input_height = 64
    parameters.conv_args.s3["14"].input_width = 64

    parameters.conv_args.s3["16"]["act_block_h"] = None
    parameters.conv_args.s3["16"]["enable_split_reader"] = False
    parameters.conv_args.s3["16"]["enable_act_double_buffer"] = False
    parameters.conv_args.s3["16"]["deallocate_activation"] = True
    parameters.conv_args.s3["16"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s3["16"]["shard_layout"] = None
    parameters.conv_args.s3["16"]["activation"] = "relu"
    parameters.conv_args.s3["16"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s3["16"].in_channels = 256
    parameters.conv_args.s3["16"].out_channels = 256
    parameters.conv_args.s3["16"].batch_size = 1
    parameters.conv_args.s3["16"].input_height = 64
    parameters.conv_args.s3["16"].input_width = 64

    parameters.conv_args.s4["18"].in_channels = 256
    parameters.conv_args.s4["18"].input_height = 64
    parameters.conv_args.s4["18"].input_width = 64
    parameters.conv_args.s4["18"].kernel_size = 2 # type: ignore[assignment]
    parameters.conv_args.s4["18"].stride = 2 # type: ignore[assignment]
    parameters.conv_args.s4["18"].padding = 0 # type: ignore[assignment]
    parameters.conv_args.s4["18"].dilation = 1
    parameters.conv_args.s4["18"].batch_size = 1
    parameters.conv_args.s4["18"].dtype = ttnn.DataType.BFLOAT16

    parameters.conv_args.s4["19"]["act_block_h"] = None
    parameters.conv_args.s4["19"]["enable_split_reader"] = False
    parameters.conv_args.s4["19"]["enable_act_double_buffer"] = False
    parameters.conv_args.s4["19"]["deallocate_activation"] = True
    parameters.conv_args.s4["19"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s4["19"]["shard_layout"] = None
    parameters.conv_args.s4["19"]["activation"] = "relu"
    parameters.conv_args.s4["19"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s4["19"].in_channels = 256
    parameters.conv_args.s4["19"].out_channels = 512
    parameters.conv_args.s4["19"].batch_size = 1
    parameters.conv_args.s4["19"].input_height = 32
    parameters.conv_args.s4["19"].input_width = 32

    parameters.conv_args.s4["21"]["act_block_h"] = None
    parameters.conv_args.s4["21"]["enable_split_reader"] = False
    parameters.conv_args.s4["21"]["enable_act_double_buffer"] = False
    parameters.conv_args.s4["21"]["deallocate_activation"] = True
    parameters.conv_args.s4["21"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s4["21"]["shard_layout"] = None
    parameters.conv_args.s4["21"]["activation"] = "relu"
    parameters.conv_args.s4["21"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s4["21"].in_channels = 512
    parameters.conv_args.s4["21"].out_channels = 512
    parameters.conv_args.s4["21"].batch_size = 1
    parameters.conv_args.s4["21"].input_height = 32
    parameters.conv_args.s4["21"].input_width = 32

    parameters.conv_args.s4["23"]["act_block_h"] = None
    parameters.conv_args.s4["23"]["enable_split_reader"] = False
    parameters.conv_args.s4["23"]["enable_act_double_buffer"] = False
    parameters.conv_args.s4["23"]["deallocate_activation"] = True
    parameters.conv_args.s4["23"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s4["23"]["shard_layout"] = None
    parameters.conv_args.s4["23"]["activation"] = "relu"
    parameters.conv_args.s4["23"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s4["23"].in_channels = 512
    parameters.conv_args.s4["23"].out_channels = 512
    parameters.conv_args.s4["23"].batch_size = 1
    parameters.conv_args.s4["23"].input_height = 32
    parameters.conv_args.s4["23"].input_width = 32

    parameters.conv_args.s4["25"]["act_block_h"] = None
    parameters.conv_args.s4["25"]["enable_split_reader"] = False
    parameters.conv_args.s4["25"]["enable_act_double_buffer"] = False
    parameters.conv_args.s4["25"]["deallocate_activation"] = True
    parameters.conv_args.s4["25"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s4["25"]["shard_layout"] = None
    parameters.conv_args.s4["25"]["activation"] = "relu"
    parameters.conv_args.s4["25"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.s4["25"].in_channels = 512
    parameters.conv_args.s4["25"].out_channels = 512
    parameters.conv_args.s4["25"].batch_size = 1
    parameters.conv_args.s4["25"].input_height = 32
    parameters.conv_args.s4["25"].input_width = 32

    parameters.conv_args.b1["27"].in_channels = 512
    parameters.conv_args.b1["27"].input_height = 32
    parameters.conv_args.b1["27"].input_width = 32
    parameters.conv_args.b1["27"].kernel_size = 2 # type: ignore[assignment]
    parameters.conv_args.b1["27"].stride = 2 # type: ignore[assignment]
    parameters.conv_args.b1["27"].padding = 0 # type: ignore[assignment]
    parameters.conv_args.b1["27"].dilation = 1
    parameters.conv_args.b1["27"].batch_size = 1
    parameters.conv_args.b1["27"].dtype = ttnn.DataType.BFLOAT16

    parameters.conv_args.b1["28"]["act_block_h"] = None
    parameters.conv_args.b1["28"]["enable_split_reader"] = False
    parameters.conv_args.b1["28"]["enable_act_double_buffer"] = False
    parameters.conv_args.b1["28"]["deallocate_activation"] = True
    parameters.conv_args.b1["28"]["reshard_if_not_optimal"] = False
    parameters.conv_args.b1["28"]["shard_layout"] = None
    parameters.conv_args.b1["28"]["activation"] = "relu"
    parameters.conv_args.b1["28"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.b1["28"].in_channels = 512
    parameters.conv_args.b1["28"].out_channels = 512
    parameters.conv_args.b1["28"].batch_size = 1
    parameters.conv_args.b1["28"].input_height = 16
    parameters.conv_args.b1["28"].input_width = 16

    parameters.conv_args.b1["30"]["act_block_h"] = None
    parameters.conv_args.b1["30"]["enable_split_reader"] = False
    parameters.conv_args.b1["30"]["enable_act_double_buffer"] = False
    parameters.conv_args.b1["30"]["deallocate_activation"] = True
    parameters.conv_args.b1["30"]["reshard_if_not_optimal"] = False
    parameters.conv_args.b1["30"]["shard_layout"] = None
    parameters.conv_args.b1["30"]["activation"] = "relu"
    parameters.conv_args.b1["30"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.b1["30"].in_channels = 512
    parameters.conv_args.b1["30"].out_channels = 512
    parameters.conv_args.b1["30"].batch_size = 1
    parameters.conv_args.b1["30"].input_height = 16
    parameters.conv_args.b1["30"].input_width = 16

    parameters.conv_args.b1["32"]["act_block_h"] = None
    parameters.conv_args.b1["32"]["enable_split_reader"] = False
    parameters.conv_args.b1["32"]["enable_act_double_buffer"] = False
    parameters.conv_args.b1["32"]["deallocate_activation"] = True
    parameters.conv_args.b1["32"]["reshard_if_not_optimal"] = False
    parameters.conv_args.b1["32"]["shard_layout"] = None
    parameters.conv_args.b1["32"]["activation"] = "relu"
    parameters.conv_args.b1["32"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.b1["32"].in_channels = 512
    parameters.conv_args.b1["32"].out_channels = 512
    parameters.conv_args.b1["32"].batch_size = 1
    parameters.conv_args.b1["32"].input_height = 16
    parameters.conv_args.b1["32"].input_width = 16

    parameters.conv_args.b1["34"]["act_block_h"] = None
    parameters.conv_args.b1["34"]["enable_split_reader"] = False
    parameters.conv_args.b1["34"]["enable_act_double_buffer"] = False
    parameters.conv_args.b1["34"]["deallocate_activation"] = True
    parameters.conv_args.b1["34"]["reshard_if_not_optimal"] = False
    parameters.conv_args.b1["34"]["shard_layout"] = None
    parameters.conv_args.b1["34"]["activation"] = "relu"
    parameters.conv_args.b1["34"]["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.b1["34"].in_channels = 512
    parameters.conv_args.b1["34"].out_channels = 512
    parameters.conv_args.b1["34"].batch_size = 1
    parameters.conv_args.b1["34"].input_height = 16
    parameters.conv_args.b1["34"].input_width = 16

    parameters.conv_args.d1.up["act_block_h"] = None
    parameters.conv_args.d1.up["enable_split_reader"] = False
    parameters.conv_args.d1.up["enable_act_double_buffer"] = False
    parameters.conv_args.d1.up["deallocate_activation"] = False
    parameters.conv_args.d1.up["reshard_if_not_optimal"] = False
    parameters.conv_args.d1.up["shard_layout"] = None
    parameters.conv_args.d1.up["dtype"] = ttnn.bfloat16
    parameters.conv_args.d1.up.in_channels = 512
    parameters.conv_args.d1.up.out_channels = 512
    parameters.conv_args.d1.up.batch_size = 1
    parameters.conv_args.d1.up.input_height = 16
    parameters.conv_args.d1.up.input_width = 16
    parameters.conv_args.d1.up.kernel_size = (2, 2) # type: ignore[assignment]
    parameters.conv_args.d1.up.stride = (2, 2) # type: ignore[assignment]
    parameters.conv_args.d1.up.padding = (0, 0) # type: ignore[assignment]

    parameters.conv_args.d1.conv_block.conv1["act_block_h"] = None
    parameters.conv_args.d1.conv_block.conv1["enable_split_reader"] = False
    parameters.conv_args.d1.conv_block.conv1["enable_act_double_buffer"] = False
    parameters.conv_args.d1.conv_block.conv1["deallocate_activation"] = True
    parameters.conv_args.d1.conv_block.conv1["reshard_if_not_optimal"] = False
    parameters.conv_args.d1.conv_block.conv1["shard_layout"] = None
    parameters.conv_args.d1.conv_block.conv1["activation"] = "relu"
    parameters.conv_args.d1.conv_block.conv1["padding"] = (1, 1)
    parameters.conv_args.d1.conv_block.conv1["do_sharded_to_interleaved"] = True
    parameters.conv_args.d1.conv_block.conv1.in_channels = 1024
    parameters.conv_args.d1.conv_block.conv1.out_channels = 512
    parameters.conv_args.d1.conv_block.conv1.batch_size = 1
    parameters.conv_args.d1.conv_block.conv1.input_height = 32
    parameters.conv_args.d1.conv_block.conv1.input_width = 32

    parameters.conv_args.d1.conv_block.conv2["act_block_h"] = None
    parameters.conv_args.d1.conv_block.conv2["enable_split_reader"] = False
    parameters.conv_args.d1.conv_block.conv2["enable_act_double_buffer"] = False
    parameters.conv_args.d1.conv_block.conv2["deallocate_activation"] = True
    parameters.conv_args.d1.conv_block.conv2["reshard_if_not_optimal"] = False
    parameters.conv_args.d1.conv_block.conv2["shard_layout"] = None
    parameters.conv_args.d1.conv_block.conv2["activation"] = "relu"
    parameters.conv_args.d1.conv_block.conv2["padding"] = (1, 1)
    parameters.conv_args.d1.conv_block.conv2["do_sharded_to_interleaved"] = True
    parameters.conv_args.d1.conv_block.conv2.in_channels = 512
    parameters.conv_args.d1.conv_block.conv2.out_channels = 512
    parameters.conv_args.d1.conv_block.conv2.batch_size = 1
    parameters.conv_args.d1.conv_block.conv2.input_height = 32
    parameters.conv_args.d1.conv_block.conv2.input_width = 32

    parameters.conv_args.d2.up["act_block_h"] = None
    parameters.conv_args.d2.up["enable_split_reader"] = False
    parameters.conv_args.d2.up["enable_act_double_buffer"] = False
    parameters.conv_args.d2.up["deallocate_activation"] = True
    parameters.conv_args.d2.up["reshard_if_not_optimal"] = False
    parameters.conv_args.d2.up["shard_layout"] = None
    parameters.conv_args.d2.up["dtype"] = ttnn.bfloat16
    parameters.conv_args.d2.up.in_channels = 512
    parameters.conv_args.d2.up.out_channels = 256
    parameters.conv_args.d2.up.batch_size = 1
    parameters.conv_args.d2.up.input_height = 32
    parameters.conv_args.d2.up.input_width = 32
    parameters.conv_args.d2.up.kernel_size = (2, 2) # type: ignore[assignment]
    parameters.conv_args.d2.up.stride = (2, 2) # type: ignore[assignment]
    parameters.conv_args.d2.up.padding = (0, 0) # type: ignore[assignment]

    parameters.conv_args.d2.conv_block.conv1["act_block_h"] = None
    parameters.conv_args.d2.conv_block.conv1["enable_split_reader"] = False
    parameters.conv_args.d2.conv_block.conv1["enable_act_double_buffer"] = False
    parameters.conv_args.d2.conv_block.conv1["deallocate_activation"] = True
    parameters.conv_args.d2.conv_block.conv1["reshard_if_not_optimal"] = False
    parameters.conv_args.d2.conv_block.conv1["shard_layout"] = None
    parameters.conv_args.d2.conv_block.conv1["activation"] = "relu"
    parameters.conv_args.d2.conv_block.conv1["padding"] = (1, 1)
    parameters.conv_args.d2.conv_block.conv1["do_sharded_to_interleaved"] = True
    parameters.conv_args.d2.conv_block.conv1.in_channels = 512
    parameters.conv_args.d2.conv_block.conv1.out_channels = 256
    parameters.conv_args.d2.conv_block.conv1.batch_size = 1
    parameters.conv_args.d2.conv_block.conv1.input_height = 64
    parameters.conv_args.d2.conv_block.conv1.input_width = 64

    parameters.conv_args.d2.conv_block.conv2["act_block_h"] = None
    parameters.conv_args.d2.conv_block.conv2["enable_split_reader"] = False
    parameters.conv_args.d2.conv_block.conv2["enable_act_double_buffer"] = False
    parameters.conv_args.d2.conv_block.conv2["deallocate_activation"] = True
    parameters.conv_args.d2.conv_block.conv2["reshard_if_not_optimal"] = False
    parameters.conv_args.d2.conv_block.conv2["shard_layout"] = None
    parameters.conv_args.d2.conv_block.conv2["activation"] = "relu"
    parameters.conv_args.d2.conv_block.conv2["padding"] = (1, 1)
    parameters.conv_args.d2.conv_block.conv2["do_sharded_to_interleaved"] = False
    parameters.conv_args.d2.conv_block.conv2.in_channels = 256
    parameters.conv_args.d2.conv_block.conv2.out_channels = 256
    parameters.conv_args.d2.conv_block.conv2.batch_size = 1
    parameters.conv_args.d2.conv_block.conv2.input_height = 64
    parameters.conv_args.d2.conv_block.conv2.input_width = 64

    parameters.conv_args.d3.up["act_block_h"] = None
    parameters.conv_args.d3.up["enable_split_reader"] = False
    parameters.conv_args.d3.up["enable_act_double_buffer"] = False
    parameters.conv_args.d3.up["deallocate_activation"] = True
    parameters.conv_args.d3.up["reshard_if_not_optimal"] = False
    parameters.conv_args.d3.up["shard_layout"] = None
    parameters.conv_args.d3.up["dtype"] = ttnn.bfloat16
    parameters.conv_args.d3.up.in_channels = 256
    parameters.conv_args.d3.up.out_channels = 128
    parameters.conv_args.d3.up.batch_size = 1
    parameters.conv_args.d3.up.input_height = 64
    parameters.conv_args.d3.up.input_width = 64
    parameters.conv_args.d3.up.kernel_size = (2, 2) # type: ignore[assignment]
    parameters.conv_args.d3.up.stride = (2, 2) # type: ignore[assignment]
    parameters.conv_args.d3.up.padding = (0, 0) # type: ignore[assignment]

    parameters.conv_args.d3.conv_block.conv1["act_block_h"] = None
    parameters.conv_args.d3.conv_block.conv1["enable_split_reader"] = False
    parameters.conv_args.d3.conv_block.conv1["enable_act_double_buffer"] = False
    parameters.conv_args.d3.conv_block.conv1["deallocate_activation"] = True
    parameters.conv_args.d3.conv_block.conv1["reshard_if_not_optimal"] = False
    parameters.conv_args.d3.conv_block.conv1["shard_layout"] = None
    parameters.conv_args.d3.conv_block.conv1["activation"] = "relu"
    parameters.conv_args.d3.conv_block.conv1["padding"] = (1, 1)
    parameters.conv_args.d3.conv_block.conv1["do_sharded_to_interleaved"] = True
    parameters.conv_args.d3.conv_block.conv1.in_channels = 256
    parameters.conv_args.d3.conv_block.conv1.out_channels = 128
    parameters.conv_args.d3.conv_block.conv1.batch_size = 1
    parameters.conv_args.d3.conv_block.conv1.input_height = 128
    parameters.conv_args.d3.conv_block.conv1.input_width = 128

    parameters.conv_args.d3.conv_block.conv2["act_block_h"] = None
    parameters.conv_args.d3.conv_block.conv2["enable_split_reader"] = False
    parameters.conv_args.d3.conv_block.conv2["enable_act_double_buffer"] = False
    parameters.conv_args.d3.conv_block.conv2["deallocate_activation"] = True
    parameters.conv_args.d3.conv_block.conv2["reshard_if_not_optimal"] = False
    parameters.conv_args.d3.conv_block.conv2["shard_layout"] = None
    parameters.conv_args.d3.conv_block.conv2["activation"] = "relu"
    parameters.conv_args.d3.conv_block.conv2["padding"] = (1, 1)
    parameters.conv_args.d3.conv_block.conv2["do_sharded_to_interleaved"] = True
    parameters.conv_args.d3.conv_block.conv2.in_channels = 128
    parameters.conv_args.d3.conv_block.conv2.out_channels = 128
    parameters.conv_args.d3.conv_block.conv2.batch_size = 1
    parameters.conv_args.d3.conv_block.conv2.input_height = 128
    parameters.conv_args.d3.conv_block.conv2.input_width = 128

    parameters.conv_args.d4.up["act_block_h"] = None
    parameters.conv_args.d4.up["enable_split_reader"] = False
    parameters.conv_args.d4.up["enable_act_double_buffer"] = False
    parameters.conv_args.d4.up["deallocate_activation"] = True
    parameters.conv_args.d4.up["reshard_if_not_optimal"] = False
    parameters.conv_args.d4.up["shard_layout"] = None
    parameters.conv_args.d4.up["dtype"] = ttnn.bfloat16
    parameters.conv_args.d4.up.in_channels = 128
    parameters.conv_args.d4.up.out_channels = 64
    parameters.conv_args.d4.up.batch_size = 1
    parameters.conv_args.d4.up.input_height = 128
    parameters.conv_args.d4.up.input_width = 128
    parameters.conv_args.d4.up.kernel_size = (2, 2) # type: ignore[assignment]
    parameters.conv_args.d4.up.stride = (2, 2) # type: ignore[assignment]
    parameters.conv_args.d4.up.padding = (0, 0) # type: ignore[assignment]

    parameters.conv_args.d4.conv_block.conv1["act_block_h"] = None
    parameters.conv_args.d4.conv_block.conv1["enable_split_reader"] = False
    parameters.conv_args.d4.conv_block.conv1["enable_act_double_buffer"] = False
    parameters.conv_args.d4.conv_block.conv1["deallocate_activation"] = True
    parameters.conv_args.d4.conv_block.conv1["reshard_if_not_optimal"] = False
    parameters.conv_args.d4.conv_block.conv1["shard_layout"] = None
    parameters.conv_args.d4.conv_block.conv1["activation"] = "relu"
    parameters.conv_args.d4.conv_block.conv1["padding"] = (1, 1)
    parameters.conv_args.d4.conv_block.conv1["do_sharded_to_interleaved"] = True
    parameters.conv_args.d4.conv_block.conv1.in_channels = 128
    parameters.conv_args.d4.conv_block.conv1.out_channels = 64
    parameters.conv_args.d4.conv_block.conv1.batch_size = 1
    parameters.conv_args.d4.conv_block.conv1.input_height = 256
    parameters.conv_args.d4.conv_block.conv1.input_width = 256

    parameters.conv_args.d4.conv_block.conv2["act_block_h"] = None
    parameters.conv_args.d4.conv_block.conv2["enable_split_reader"] = False
    parameters.conv_args.d4.conv_block.conv2["enable_act_double_buffer"] = False
    parameters.conv_args.d4.conv_block.conv2["deallocate_activation"] = True
    parameters.conv_args.d4.conv_block.conv2["reshard_if_not_optimal"] = False
    parameters.conv_args.d4.conv_block.conv2["shard_layout"] = None
    parameters.conv_args.d4.conv_block.conv2["activation"] = "relu"
    parameters.conv_args.d4.conv_block.conv2["padding"] = (1, 1)
    parameters.conv_args.d4.conv_block.conv2["do_sharded_to_interleaved"] = True
    parameters.conv_args.d4.conv_block.conv2.in_channels = 64
    parameters.conv_args.d4.conv_block.conv2.out_channels = 64
    parameters.conv_args.d4.conv_block.conv2.batch_size = 1
    parameters.conv_args.d4.conv_block.conv2.input_height = 256
    parameters.conv_args.d4.conv_block.conv2.input_width = 256

    parameters.conv_args.out["act_block_h"] = None
    parameters.conv_args.out["enable_split_reader"] = False
    parameters.conv_args.out["enable_act_double_buffer"] = False
    parameters.conv_args.out["deallocate_activation"] = True
    parameters.conv_args.out["reshard_if_not_optimal"] = False
    parameters.conv_args.out["shard_layout"] = None
    parameters.conv_args.out["activation"] = ""
    parameters.conv_args.out["padding"] = (0, 0)
    parameters.conv_args.out["dtype"] = ttnn.DataType.BFLOAT16
    parameters.conv_args.out.in_channels = 64
    parameters.conv_args.out.out_channels = 1
    parameters.conv_args.out.batch_size = 1
    parameters.conv_args.out.input_height = 256
    parameters.conv_args.out.input_width = 256
    parameters.conv_args.out.kernel_size = (1, 1) # type: ignore[assignment]
    parameters.conv_args.out.stride = (1, 1) # type: ignore[assignment]
    parameters.conv_args.out.padding = (0, 0) # type: ignore[assignment]

    return parameters
