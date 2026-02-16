# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time
from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

def parse_dtype(dtype_str: str):
    """
    Map 'fp16' / 'bf16' strings to simple tags.
    """
    s = str(dtype_str).lower()
    if s in ("fp16", "float16"):
        return "fp16"
    if s in ("bf16", "bfloat16"):
        return "bf16"
    raise ValueError(f"Unsupported host dtype string for TTNN add workload: {dtype_str}")

def run_binary_add(wlname: str, device: TTNNDevice, cfg: dict):
    """
    Polaris TTNN workload entry for a simple NCHW binary add benchmark.
    """
    # 1) Read config
    bs = int(cfg.get("bs", 1))
    N = int(cfg.get("N", bs))
    C = int(cfg.get("C", cfg.get("c", 64)))
    H = int(cfg.get("H", cfg.get("h", 64)))
    W = int(cfg.get("W", cfg.get("w", 64)))

    dtype_str = cfg.get("dtype", "fp16")
    warmup = int(cfg.get("warmup", 5))
    iters = int(cfg.get("iters", 10))

    host_dtype_tag = parse_dtype(dtype_str)
    ttnn_dtype = ttnn.bfloat16  # device dtype

    num_elements = N * C * H * W
    bytes_per_element = 2  # bf16/fp16
    approx_bytes_per_op = 3 * num_elements * bytes_per_element  # 2 inputs + 1 output

    logger.info("=== TTNN Polaris Binary Add Workload: {} ===", wlname)
    logger.info("Input shape       : (N={}, C={}, H={}, W={})", N, C, H, W)
    logger.info("Host dtype tag    : {} (device uses bfloat16)", host_dtype_tag)
    logger.info("Warmup iters      : {}", warmup)
    logger.info("Timed iters       : {}", iters)
    logger.info("Approx bytes/op   : {:.3f} MB", approx_bytes_per_op / 1e6)

    # 2) Create tensors directly on device
    a_tt = ttnn._rand(
        (N, C, H, W),
        dtype=ttnn_dtype,
        device=device,
    )
    b_tt = ttnn._rand(
        (N, C, H, W),
        dtype=ttnn_dtype,
        device=device,
    )

    def binary_add(x, y):
        return ttnn.add(x, y)

    # 3) Warmup (no explicit synchronize needed in TTSIM)
    logger.info("Running warmup iterations...")
    for _ in range(warmup):
        _ = binary_add(a_tt, b_tt)

    # 4) Timed runs
    logger.info("Running timed iterations...")
    start = time.time()
    out = None
    for _ in range(iters):
        out = binary_add(a_tt, b_tt)
    end = time.time()

    avg_time_ms = (end - start) * 1000.0 / iters
    gb_per_s = (approx_bytes_per_op / (avg_time_ms * 1e-3)) / 1e9

    logger.info("Average time       : {:.3f} ms", avg_time_ms)
    logger.info("Effective bandwidth: {:.2f} GB/s", gb_per_s)

    return out
    
    
    
'''

Workload
  - api: TTNN
    name: BinaryAdd
    basedir: tests/BinaryAdd
    module: run_binary_add@ttnn_functional_add.py
    instances:
      default:
        bs: 1       # used as N when N not provided
        N: 1        # or omit to use bs
        C: 1
        H: 120
        W: 120
        dtype: fp16
        warmup: 5
        iters: 10

Command to run:
        
python polaris.py \
  --archspec  config/tt_wh.yaml \
  --wlspec    config/all_workloads.yaml \
  --wlmapspec config/wl2archmapping.yaml \
  --filterwl  BinaryAdd \
  --filterwli default \
  --filterarch n150 \
  --study     BINARY_ADD_WH \
  --odir      __BINARY_check_WH \
  --dump_stats_csv

'''