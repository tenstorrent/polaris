# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

def parse_dtype(dtype_str: str):
    """
    Map a string like 'bfloat8_b', 'bfp8', 'bf16', 'float32' to ttnn dtype.
    Extend based on what your ttnn build actually supports.
    """
    s = str(dtype_str).lower()

    if s in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if s in ("fp32", "float32"):
        return ttnn.float32

    if s in ("bfp8", "bfloat8_b", "bfp8_b"):
        return ttnn.bfloat8_b

    raise ValueError(f"Unsupported dtype string: {dtype_str}")

def run_transpose(wlname: str, device: TTNNDevice, cfg: dict):
    """
    Polaris TTNN workload entry for a simple 2D transpose benchmark.

    Expected cfg keys (with defaults):
        M        : int, number of rows (default 1024)
        N        : int, number of cols (default 1024)
        dtype    : str, 'bfloat8_b', 'bf16', 'float32', ... (default 'bfloat8_b')
        num_runs : int, how many times to run transpose (default 10)
    """
    # 1) Read config
    M = int(cfg.get("M", 1024))
    N = int(cfg.get("N", 1024))
    dtype_str = cfg.get("dtype", "bfloat8_b")
    num_runs = int(cfg.get("num_runs", 10))

    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive integers")

    dtype = parse_dtype(dtype_str)

    logger.info("=== TTNN Polaris Transpose Workload: {} ===", wlname)
    logger.info("Input shape       : {} x {}", M, N)
    logger.info("Data type         : {}", dtype_str)
    logger.info("Num runs          : {}", num_runs)

    # 2) Create input tensor on device: shape (M, N)
    a_tt = ttnn._rand(
        (M, N),
        dtype=dtype,
        device=device,
    )

    # 3) Run transpose multiple times
    output = None
    for i in range(num_runs):
        logger.info("Run {}/{}: ttnn.transpose", i + 1, num_runs)
        # Swap dim 0 and 1: (M, N) -> (N, M)
        output = ttnn.transpose(a_tt, 0, 1)

    logger.info("Original shape (device): {}", (M, N))
    logger.info("Transposed shape (device): {}", getattr(output, "shape", "unknown"))
    logger.info("Returning output tensor for Polaris graph capture.")

    return output
	
	
'''
workload
- api: TTNN
  name: TransposeTest
  basedir: tests/Transpose
  module: run_transpose@ttnn_functional_transpose.py
  instances:
    default:
      bs: 1           # <-- add this
      M: 1024
      N: 1024
      dtype: bfloat8_b
      num_runs: 10

Command to run:

python polaris.py \
  --archspec  config/tt_wh.yaml \
  --wlspec    config/all_workloads.yaml \
  --wlmapspec config/wl2archmapping.yaml \
  --filterwl  TransposeTest \
  --filterwli default \
  --filterarch n150 \
  --study     TRANSPOSE_WH \
  --odir      __TRANSPOSE_WH \
  --dump_stats_csv



'''