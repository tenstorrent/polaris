# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

# -------- Helpers to map strings to ttnn enums / dtypes --------

def parse_dtype(dtype_str: str):
    """
    Map a string like 'bfloat8_b', 'bfp8', 'bf16', 'float32' to ttnn dtype.
    Adjust symbols to match your TTNN build if needed.
    """
    s = str(dtype_str).lower()

    if s in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if s in ("fp32", "float32"):
        return ttnn.float32

    if s in ("bfp8", "bfloat8_b", "bfp8_b", "bf8"):
        # Change to ttnn.bfloat8 or ttnn.bfp8_b if your build uses that
        return ttnn.bfloat8_b

    raise ValueError(f"Unsupported dtype string for TTNN matmul workload: {dtype_str}")

def parse_math_fidelity(fidelity_str: str):
    """
    Map a string like 'LoFi', 'HiFi2', 'HiFi3', 'HiFi4' to ttnn.MathFidelity.
    """
    s = str(fidelity_str).lower()
    if s == "lofi":
        return ttnn.MathFidelity.LoFi
    if s == "hifi2":
        return ttnn.MathFidelity.HiFi2
    if s == "hifi3":
        return ttnn.MathFidelity.HiFi3
    if s == "hifi4":
        return ttnn.MathFidelity.HiFi4

    raise ValueError(f"Unsupported math fidelity for TTNN matmul workload: {fidelity_str}")

# -------- Polaris TTNN workload entry point --------

def run_matmul_test(wlname: str, device: TTNNDevice, cfg: dict):
    """
    Polaris TTNN workload entry for a simple MxK @ KxN matmul sweep.

    Args:
        wlname: Workload identifier from YAML (e.g., "Matmul2Tensors").
        device: TTNN device managed by Polaris (do NOT open/close here).
        cfg:    Merged configuration dict from YAML + CLI overrides.

    Expected cfg keys (with defaults):
        bs        : int, batch size (required by Polaris infra, but unused here; default 1)
        M         : int, rows of A  (default 1024)
        K         : int, cols of A / rows of B (default 1024)
        N         : int, cols of B (default 1024)
        dtype     : str, device dtype ("bfloat8_b", "bf16", "float32", ...) (default "bfloat8_b")
        fidelity  : str, math fidelity ("HiFi4", "HiFi3", "HiFi2", "LoFi") (default "HiFi4")
        num_runs  : int, how many times to run matmul (default 10)

    Returns:
        TTNN tensor result of the last matmul, for Polaris graph capture.
    """
    # 1) Read config with safe defaults
    batch_size = int(cfg.get("bs", 1))  # present to satisfy polaris.py; not used in math

    M = int(cfg.get("M", cfg.get("m", 1024)))
    K = int(cfg.get("K", cfg.get("k", 1024)))
    N = int(cfg.get("N", cfg.get("n", 1024)))

    num_runs = int(cfg.get("num_runs", cfg.get("runs", 10)))

    dtype_str = cfg.get("dtype", "bfloat8_b")
    fidelity_str = cfg.get("fidelity", "HiFi4")

    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError("M, K, N must be positive integers")

    logger.info(f"=== TTNN Polaris Matmul Workload: {wlname} ===")
    logger.info(f"M, K, N        : {M}, {K}, {N}")
    logger.info(f"Data type      : {dtype_str}")
    logger.info(f"Math fidelity  : {fidelity_str}")
    logger.info(f"Num runs       : {num_runs}")
    logger.info(f"Batch size (unused here) : {batch_size}")

    # 2) Map string options to TTNN types/enums
    dtype = parse_dtype(dtype_str)
    math_fidelity = parse_math_fidelity(fidelity_str)

    # 3) Create input tensors directly on device (TTNN style)
    logger.info(
        f"Creating TTNN tensors A[{M}, {K}] and B[{K}, {N}] on device "
        f"with dtype={dtype_str}..."
    )

    # Your _rand signature appears to be: _rand(shape, dtype, device)
    a_tt = ttnn._rand(
        (M, K),
        dtype=dtype,
        device=device,
    )

    b_tt = ttnn._rand(
        (K, N),
        dtype=dtype,
        device=device,
    )

    # 4) Compute kernel config
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    logger.info(f"Requested math_fidelity: {compute_config.math_fidelity}")

    # 5) Run matmul multiple times; result of last run is returned
    output_tensor = None
    for i in range(num_runs):
        logger.info(
            f"Run {i + 1}/{num_runs}: matmul ({dtype_str}, {fidelity_str})"
        )
        output_tensor = ttnn.matmul(
            a_tt,
            b_tt,
            compute_kernel_config=compute_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # No core_grid argument – let TTNN choose
        )

    # 6) Optional: log result shape/precision for sanity
    try:
        out_shape = output_tensor.shape # type: ignore[union-attr]
    except Exception:
        out_shape = "unknown"

    logger.info(f"Matmul completed; output shape (TTNN): {out_shape}")
    logger.info("Returning output tensor for Polaris graph capture.")

    return output_tensor
	
	
	


'''
Workload:

  - api: TTNN
    name: Matmul2Tensors
    basedir: tests/Matmultest
    module: run_matmul_test@ttnn_functional_matmul.py
    instances:
      default: { bs: 1, M: 1024, K: 1024, N: 1024, dtype: bfloat8_b, fidelity: HiFi4, num_runs: 10 } 
  
Command to Run code:

   python polaris.py \
  --archspec  config/tt_wh.yaml \
  --wlspec    config/all_workloads.yaml \
  --wlmapspec config/wl2archmapping.yaml \
  --filterwl  Matmul2Tensors \
  --filterwli default \
  --filterarch n150 \
  --study     MATMUL2_WH \
  --odir      __MATMULPPPP_WH \
  --dump_stats_csv

'''
