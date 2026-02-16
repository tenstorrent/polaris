# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time
import math
from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

def compute_conv2d_flops(N, C_in, C_out, H_in, W_in, K_h, K_w, stride_h=1, stride_w=1, pad_h=1, pad_w=1):
    H_out = math.floor((H_in + 2 * pad_h - K_h) / stride_h) + 1
    W_out = math.floor((W_in + 2 * pad_w - K_w) / stride_w) + 1
    flops = N * H_out * W_out * C_out * (C_in * K_h * K_w * 2)
    return flops, H_out, W_out

def run_conv2d(wlname: str, device: TTNNDevice, cfg: dict):
    """
    Polaris TTNN workload entry for a *simple* NCHW Conv2D benchmark.

    We avoid fancy reshape/permute and just use:
      X: (N, C_in, H, W)
      W: (C_out, C_in, K_h, K_w)
    """
    # 1) Read config
    bs = int(cfg.get("bs", 1))

    N = int(cfg.get("N", bs))
    C_in = int(cfg.get("C_in", cfg.get("c_in", 64)))
    C_out = int(cfg.get("C_out", cfg.get("c_out", 64)))
    H = int(cfg.get("H", cfg.get("h", 64)))
    W = int(cfg.get("W", cfg.get("w", 64)))

    K_h = int(cfg.get("kernel_h", 3))
    K_w = int(cfg.get("kernel_w", 3))
    kernel_size = (K_h, K_w)

    stride = (1, 1)
    padding = (1, 1)

    dtype_str = cfg.get("dtype", "bf16")
    warmup = int(cfg.get("warmup", 5))
    iters = int(cfg.get("iters", 10))

    flops, H_out, W_out = compute_conv2d_flops(
        N, C_in, C_out, H, W, K_h, K_w,
        stride_h=stride[0], stride_w=stride[1],
        pad_h=padding[0], pad_w=padding[1],
    )

    logger.info("=== TTNN Polaris Conv2D Workload: {} ===", wlname)
    logger.info("Input NCHW        : (N={}, C_in={}, H={}, W={})", N, C_in, H, W)
    logger.info("Kernel            : (C_out={}, C_in={}, K_h={}, K_w={})", C_out, C_in, K_h, K_w)
    logger.info("Stride            : {}", stride)
    logger.info("Padding           : {}", padding)
    logger.info("Estimated output  : (N={}, C_out={}, H_out={}, W_out={})", N, C_out, H_out, W_out)
    logger.info("Logical DType     : {} (device uses bfloat16)", dtype_str)
    logger.info("Warmup iters      : {}", warmup)
    logger.info("Timed iters       : {}", iters)

    # 2) Create input/weights/bias on device (N, C_in, H, W) and (C_out, C_in, K_h, K_w)
    X = ttnn._rand(
        (N, C_in, H, W),
        dtype=ttnn.bfloat16,
        device=device,
    )

    W_tt = ttnn._rand(
        (C_out, C_in, K_h, K_w),
        dtype=ttnn.bfloat16,
        device=device,
    )

    # Bias: [C_out] or [1, C_out, 1, 1] – depends on your TTNN version.
    # Many TTNN convs accept [C_out], so we use that.
    B_tt = ttnn.zeros(
        (C_out,),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    def forward(input_tensor, weight_tensor, bias_tensor):
        # Direct NCHW conv2d call; match in_channels to C_in
        out = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            in_channels=C_in,
            out_channels=C_out,
            device=device,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            batch_size=N,
            input_height=H,
            input_width=W,
            groups=1,
        )
        return out

    # 3) Warmup
    logger.info("Running warmup iterations...")
    for _ in range(warmup):
        _ = forward(X, W_tt, B_tt)

    # 4) Timed runs
    logger.info("Running timed iterations...")
    start = time.time()
    out = None
    for _ in range(iters):
        out = forward(X, W_tt, B_tt)
    end = time.time()

    avg_time_ms = (end - start) * 1000.0 / iters
    tflops_per_s = flops / (avg_time_ms * 1e-3) / 1e12

    logger.info("Average time       : {:.3f} ms", avg_time_ms)
    logger.info("Estimated FLOPs    : {:.3f} GFLOPs per conv", flops / 1e9)
    logger.info("Achieved throughput: {:.2f} TFLOPs/s", tflops_per_s)
    logger.info("Conv2D TTNN output shape: {}", getattr(out, "shape", "unknown"))

    return out



'''
Workload

- api: TTNN
  name: Conv2DTest
  basedir: tests/Conv2D
  module: run_conv2d@ttnn_functional_conv2d.py
  instances:
    default:
      bs: 1
      N: 1
      C_in: 64
      C_out: 64
      H: 64
      W: 64
      kernel_h: 3
      kernel_w: 3
      dtype: bf16
      warmup: 5
      iters: 10

command to run:
 
python polaris.py \
  --archspec  config/tt_wh.yaml \
  --wlspec    config/all_workloads.yaml \
  --wlmapspec config/wl2archmapping.yaml \
  --filterwl  Conv2DTest \
  --filterwli default \
  --filterarch n150 \
  --study     CONV2D_WH \
  --odir      __CONV2D_WH \
  --dump_stats_csv



'''
