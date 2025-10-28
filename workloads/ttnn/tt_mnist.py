#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.ttnn as ttnn

def mnist(device, batch_size, x, parameters):
    x = ttnn.reshape(x, (x.shape[0], -1))
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.linear(
        x,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
    )
    x = ttnn.linear(
        x,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
    )
    x = ttnn.linear(
        x,
        parameters.fc3.weight,
        bias=parameters.fc3.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
    )

    x = ttnn.softmax(x)

    return x


def create_inputs(device):
    from ttsim.utils import dict2obj

    batch_size = 128
    img_h      = 28
    img_w      = 28
    embed_dims = [28*28, 256, 128, 10]
    itensor    = ttnn.Tensor(name='X', shape=(batch_size, img_h, img_w), dtype=ttnn.float32, device=device)
    params     = dict2obj({
        f"fc{i+1}":  {
            'weight': ttnn.Tensor(name=f"W{i+1}", shape=(embed_dims[i], embed_dims[i+1]), dtype=ttnn.float32, device=device),
            'bias'  : ttnn.Tensor(name=f"b{i+1}", shape=(embed_dims[i+1],), dtype=ttnn.float32, device=device),
            } for i in range(3)
        })
    return batch_size, itensor, params

if __name__ == '__main__':
    try:
        D       = ttnn.open_device(device_id=0)
        B, X, P = create_inputs(D)
        Y       = mnist(D, B, X, P)

        g = D.get_graph()
        g.graph2onnx('tt_mnist.onnx')

    finally:
        ttnn.close_device(D)
