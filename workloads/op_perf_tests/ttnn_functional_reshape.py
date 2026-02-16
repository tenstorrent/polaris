# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time
import math
from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

def run_reshape(wlname: str, device: TTNNDevice, cfg: dict):
    """
    Polaris TTNN workload entry for a simple reshape benchmark.

    Expected cfg keys (with defaults):
        bs           : int, batch size alias; used as N if N not given
        N            : int, batch size (default bs or 1)
        C            : int, channels (default 64)
        H            : int, height   (default 32)
        W            : int, width    (default 32)
        target_shape : list/tuple of ints, e.g. [1, 1, 2048, 32]  (REQUIRED)
        warmup       : int, warmup iterations (default 5)
        iters        : int, timed iterations (default 10)
    """
    # 1) Read config
    bs = int(cfg.get("bs", 1))

    N = int(cfg.get("N", bs))
    C = int(cfg.get("C", cfg.get("c", 64)))
    H = int(cfg.get("H", cfg.get("h", 32)))
    W = int(cfg.get("W", cfg.get("w", 32)))

    target_shape_cfg = cfg.get("target_shape", None)
    if target_shape_cfg is None:
        raise ValueError("cfg['target_shape'] must be provided for reshape workload")
    # Ensure it's a tuple/list of ints
    target_shape = tuple(int(v) for v in target_shape_cfg)

    warmup = int(cfg.get("warmup", 5))
    iters = int(cfg.get("iters", 10))

    in_elems = N * C * H * W
    out_elems = math.prod(target_shape)
    if in_elems != out_elems:
        raise ValueError(
            f"Input elements ({in_elems}) != target elements ({out_elems}). "
            f"Shapes must have same number of elements."
        )

    logger.info("=== TTNN Polaris Reshape Workload: {} ===", wlname)
    logger.info("Input shape       : (N={}, C={}, H={}, W={})", N, C, H, W)
    logger.info("Target shape      : {}", target_shape)
    logger.info("Total elements    : {}", in_elems)
    logger.info("Warmup iters      : {}", warmup)
    logger.info("Timed iters       : {}", iters)

    # 2) Create input tensor on device
    # Use simple TTNN random init: _rand(shape, dtype, device)
    X = ttnn._rand(
        (N, C, H, W),
        dtype=ttnn.bfloat16,
        device=device,
    )

    def reshape_op(t):
        return ttnn.reshape(t, target_shape)

    # 3) Warmup
    logger.info("Running warmup iterations...")
    for _ in range(warmup):
        _ = reshape_op(X)

    # 4) Timed runs
    logger.info("Running timed iterations...")
    start = time.time()
    out = None
    for _ in range(iters):
        out = reshape_op(X)
    end = time.time()

    avg_time_ms = (end - start) * 1000.0 / iters
    elems_per_s = in_elems / (avg_time_ms * 1e-3)

    logger.info("Average time       : {:.3f} ms", avg_time_ms)
    logger.info("Elements per op    : {}", in_elems)
    logger.info("Throughput         : {:.3f} Gelems/s", elems_per_s / 1e9)
    logger.info("Output shape       : {}", getattr(out, "shape", "unknown"))

    return out
	

'''
Workload:

  - api: TTNN
    name: ReshapeTest
    basedir: tests/Reshape
    module: run_reshape@ttnn_functional_reshape.py
    instances:
      default:
        bs: 1
        N: 1
        C: 64
        H: 32
        W: 32
        target_shape: [1, 1, 2048, 32]  # must have same total elements as N*C*H*W
        warmup: 5
        iters: 10

Command to run:

 python polaris.py \
  --archspec  config/tt_wh.yaml \
  --wlspec    config/all_workloads.yaml \
  --wlmapspec config/wl2archmapping.yaml \
  --filterwl  ReshapeTest \
  --filterwli default \
  --filterarch n150 \
  --study     RESHAPE_WH \
  --odir      __RESHAPE_WH \
  --dump_stats_csv



'''
