#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import ttsim.front.ttnn as ttnn
from ttsim.utils.common import dict2obj

def create_ttnn_tensor(**kwargs):
    assert 'shape' in kwargs, f"shape not defined during ttnn.Tensor creation: {kwargs}"
    return ttnn.Tensor(**kwargs)

def update_param_attrs(P, *, device, dtype, prefix=""):
    if isinstance(P, dict):
        for k,v in P.items():
            update_param_attrs(v, device=device, dtype=dtype, prefix= f"{prefix}.{k}")
    elif isinstance(P, list):
        for i,x in enumerate(P):
            update_param_attrs(x, device=device, dtype=dtype, prefix= f"{prefix}.{i}")
    elif isinstance(P, dict2obj):
        for k,v in vars(P).items():
            update_param_attrs(v, device=device, dtype=dtype, prefix= f"{prefix}.{k}")
    elif isinstance(P, ttnn.Tensor):
        P.name   = prefix
        P.device = device
        P.dtype  = dtype
        device.add_tensor(P)
    else:
        assert False, f"Unknown type for {prefix} = {type(P)}!!"
    return

