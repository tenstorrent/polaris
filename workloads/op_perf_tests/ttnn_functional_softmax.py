# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time
from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

def run_softmax(wlname: str, device: TTNNDevice, cfg: dict):
    """
    Polaris TTNN workload entry for a simple NCHW softmax benchmark.

    Expected cfg keys (with defaults):
        bs      : int, batch size alias; used as N if N not given
        N       : int, batch size (default bs or 1)
        C       : int, channels (default 64)
        H       : int, height   (default 32)
        W       : int, width    (default 32)
        axis    : int, axis for softmax over [N, C, H, W], supports negatives (default -1 = W)
        warmup  : int, warmup iterations (default 5)
        iters   : int, timed iterations (default 10)
    """
    # 1) Read config
    bs = int(cfg.get("bs", 1))

    N = int(cfg.get("N", bs))
    C = int(cfg.get("C", cfg.get("c", 64)))
    H = int(cfg.get("H", cfg.get("h", 32)))
    W = int(cfg.get("W", cfg.get("w", 32)))

    axis = int(cfg.get("axis", -1))
    if axis < 0:
        axis = 4 + axis  # for shape [N, C, H, W]
    if axis not in (0, 1, 2, 3):
        raise ValueError("axis must be one of 0,1,2,3 or the corresponding negative index")

    warmup = int(cfg.get("warmup", 5))
    iters = int(cfg.get("iters", 10))

    num_elements = N * C * H * W

    logger.info("=== TTNN Polaris Softmax Workload: {} ===", wlname)
    logger.info("Input shape       : (N={}, C={}, H={}, W={})", N, C, H, W)
    logger.info("Softmax axis      : {} (0=N, 1=C, 2=H, 3=W)", axis)
    logger.info("Warmup iters      : {}", warmup)
    logger.info("Timed iters       : {}", iters)
    logger.info("Elements per op   : {}", num_elements)

    # 2) Create input tensor on device
    X = ttnn._rand(
        (N, C, H, W),
        dtype=ttnn.bfloat16,
        device=device,
    )

    def softmax_op(t):
        # Use dim=axis as in your standalone script
        return ttnn.softmax(t, dim=axis)

    # 3) Warmup
    logger.info("Running warmup iterations...")
    for _ in range(warmup):
        _ = softmax_op(X)

    # 4) Timed runs
    logger.info("Running timed iterations...")
    start = time.time()
    out = None
    for _ in range(iters):
        out = softmax_op(X)
    end = time.time()

    avg_time_ms = (end - start) * 1000.0 / iters
    elems_per_s = num_elements / (avg_time_ms * 1e-3)

    logger.info("Average time       : {:.3f} ms", avg_time_ms)
    logger.info("Throughput         : {:.3f} Gelems/s", elems_per_s / 1e9)
    logger.info("Output shape       : {}", getattr(out, "shape", "unknown"))

    return out


'''
Workload:

  - api: TTNN
    name: SoftmaxTest
    basedir: tests/softmax
    module: run_softmax@ttnn_functional_softmax.py
    instances:
      default:
        bs: 1
        N: 1
        C: 64
        H: 32
        W: 32
        axis: -1      # -1 → last dim (W)
        warmup: 5
        iters: 10

command to run:

python polaris.py \
  --archspec  config/tt_wh.yaml \
  --wlspec    config/all_workloads.yaml \
  --wlmapspec config/wl2archmapping.yaml \
  --filterwl  SoftmaxTest \
  --filterwli default \
  --filterarch n150 \
  --study     SOFTMAX_WH \
  --odir      __SOFTMAX_WH \
  --dump_stats_csv

'''