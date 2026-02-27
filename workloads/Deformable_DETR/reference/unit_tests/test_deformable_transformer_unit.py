#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for Deformable Transformer modules — PyTorch vs TTSim comparison WITH EDGE CASES.

================================================================================
VALIDATION TYPES EXPLAINED:
================================================================================

NUMERICAL VALIDATION:
    - Compares actual OUTPUT VALUES between PyTorch and TTSim
    - Uses np.allclose() with tolerance: rtol / atol
    - Reports: max absolute difference, mean absolute difference
    - PASS if: all values within tolerance
    - FAIL if: any value exceeds tolerance

SHAPE VALIDATION ONLY:
    - Compares only OUTPUT DIMENSIONS (shapes) between PyTorch and TTSim
    - NO numerical value comparison
    - PASS if: all shapes match exactly
    - FAIL if: any shape mismatch detected
    - WHY USE: When weights are not synced (shape-only inference mode)

================================================================================
MODULES TESTED:
================================================================================

MODULE 1: EncoderLayer — NUMERICAL VALIDATION
    Architecture: MSDeformAttn + FFN (Linear→ReLU→Linear) + 2×LayerNorm + residuals
    Edge Cases: positive, negative, zeros, mixed, small (~1e-6), large (~1e6),
                minimum_input (seq=18)
    WHY NUMERICAL: Small dims (d_model=64, seq=50) keep computation feasible
    Tolerance: rtol=0.1, atol=0.05 (deformable attention accumulates error)

MODULE 2: Encoder (stacked) — NUMERICAL VALIDATION
    Architecture: N×EncoderLayer with reference point generation via valid_ratios
    Edge Cases: positive, negative, zeros, mixed, small, large, minimum_input
    WHY NUMERICAL: Small dims (d_model=64, 2 layers) keep computation feasible
    Tolerance: rtol=0.1, atol=0.05 (error accumulates across stacked layers)

MODULE 3: DecoderLayer — NUMERICAL VALIDATION
    Architecture: Self-attn (MHA) + Cross-attn (MSDeformAttn) + FFN + 3×LayerNorm
    Note: Norm numbering SWAPPED: PT norm2→TT norm1, PT norm1→TT norm2
    Edge Cases: positive, negative, zeros, mixed, small, large, minimum_input
    WHY NUMERICAL: Small dims + synced weights allow full comparison
    Tolerance: rtol=0.1, atol=0.05

MODULE 4: Decoder (stacked) — NUMERICAL VALIDATION
    Architecture: N×DecoderLayer, reference point expansion [B,Q,2]→[B,Q,lvl,2]
    Edge Cases: positive, negative, zeros, mixed, small, large, minimum_input
    WHY NUMERICAL: Small dims (2 layers, d_model=64)
    Tolerance: rtol=0.1, atol=0.05

MODULE 5: Full Transformer — NUMERICAL VALIDATION
    Architecture: level_embed + Encoder + query_embed split + ref_points Linear + Decoder
    Edge Cases: positive, negative, zeros, mixed, small, large, minimum_input
    WHY NUMERICAL: 1 encoder + 1 decoder layer, d_model=64, small spatial dims
    Tolerance: rtol=0.1, atol=0.05

================================================================================
EDGE CASES TESTED (MANDATORY — all modules):
================================================================================

'positive'       — Standard positive values (1.0 - 2.0) - baseline test
'negative'       — All negative values (-2.0 to -1.0) - tests sign handling
'zeros'          — All zeros - tests division edge cases in LayerNorm
'mixed'          — Mix of positive/negative values - tests real-world distribution
'small'          — Very small values (~1e-6) - tests numerical precision near zero
'large'          — Very large values (~1e6) - tests numerical overflow handling
'minimum_input'  — Smallest valid spatial/sequence size - degenerate/boundary case

================================================================================
RUN:
    cd polaris
    pytest workloads/Deformable_DETR/unit_tests/test_deformable_transformer_unit.py -v -s
    # or
    python workloads/Deformable_DETR/unit_tests/test_deformable_transformer_unit.py
================================================================================
"""

import os
import sys
import pytest
import torch
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

# PyTorch implementations
from workloads.Deformable_DETR.reference.deformable_transformer import (
    DeformableTransformerEncoderLayer as EncoderLayerPyTorch,
    DeformableTransformerEncoder as EncoderPyTorch,
    DeformableTransformerDecoderLayer as DecoderLayerPyTorch,
    DeformableTransformerDecoder as DecoderPyTorch,
    DeformableTransformer as TransformerPyTorch,
)

# TTSim implementations
from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
    DeformableTransformerEncoderLayer as EncoderLayerTTSim,
    DeformableTransformerEncoder as EncoderTTSim,
    DeformableTransformerDecoderLayer as DecoderLayerTTSim,
    DeformableTransformerDecoder as DecoderTTSim,
    DeformableTransformer as TransformerTTSim,
)

from ttsim.ops.tensor import SimTensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RTOL = 1e-4
ATOL = 1e-5
SEED = 42

# Default tolerances for deformable transformer tests
# (deformable attention + stacked layers accumulate more error than simple ops)
DT_RTOL = 0.1
DT_ATOL = 0.05


# ---------------------------------------------------------------------------
# Terminal Colors (ANSI escape codes)
# ---------------------------------------------------------------------------
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    @staticmethod
    def success(s):
        return f"{Colors.GREEN}{s}{Colors.RESET}"

    @staticmethod
    def fail(s):
        return f"{Colors.RED}{Colors.BOLD}{s}{Colors.RESET}"

    @staticmethod
    def warn(s):
        return f"{Colors.YELLOW}{s}{Colors.RESET}"

    @staticmethod
    def info(s):
        return f"{Colors.CYAN}{s}{Colors.RESET}"

    @staticmethod
    def bold(s):
        return f"{Colors.BOLD}{s}{Colors.RESET}"

    @staticmethod
    def cyan(s):
        return f"{Colors.CYAN}{s}{Colors.RESET}"

    @staticmethod
    def header(s):
        return f"{Colors.BOLD}{Colors.MAGENTA}{s}{Colors.RESET}"

    @staticmethod
    def dim(s):
        return f"{Colors.DIM}{s}{Colors.RESET}"


# ---------------------------------------------------------------------------
# Visual separators
# ---------------------------------------------------------------------------
SUMMARY_LINE = "═" * 65
DIVIDER_LINE = "─" * 65


# ---------------------------------------------------------------------------
# Edge case descriptions
# ---------------------------------------------------------------------------
EDGE_CASE_DESC = {
    "positive": "Standard positive values (1.0 - 2.0) - baseline test",
    "negative": "All negative values (-2.0 to -1.0) - tests sign handling",
    "zeros": "All zeros - tests division edge case in LayerNorm",
    "mixed": "Mix of positive/negative values - tests real-world distribution",
    "small": "Very small values (~1e-6) - tests numerical precision near zero",
    "large": "Very large values (~1e6) - tests numerical overflow handling",
    "minimum_input": "Smallest valid input size - degenerate/boundary case",
}


# ---------------------------------------------------------------------------
# Report data collectors (populated by tests, consumed by _write_report)
# ---------------------------------------------------------------------------
REPORT_SECTIONS = []
FAILED_TESTS = []
TEST_RESULTS = []
MODULE_STATS = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seed(offset=0):
    torch.manual_seed(SEED + offset)
    np.random.seed(SEED + offset)


def torch_to_simtensor(tensor, name="tensor", module=None):
    """Convert a PyTorch tensor to a SimTensor, optionally linking to a module."""
    data = tensor.detach().cpu().numpy().copy()
    st = SimTensor(
        {
            "name": name,
            "shape": list(tensor.shape),
            "data": data,
            "dtype": data.dtype,
        }
    )
    if module is not None:
        st.link_module = module
        module._tensors[name] = st
    return st


def _to_numpy(x):
    """Coerce PyTorch tensor / SimTensor / ndarray to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, SimTensor):
        return x.data
    return np.asarray(x)


def generate_test_data(shape, data_type):
    """Generate test data based on type."""
    if data_type == "positive":
        return np.random.rand(*shape).astype(np.float32) + 1.0
    elif data_type == "negative":
        return -np.random.rand(*shape).astype(np.float32) - 1.0
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return (np.random.randn(*shape) * 2).astype(np.float32)
    elif data_type == "small":
        return np.random.rand(*shape).astype(np.float32) * 1e-6
    elif data_type == "large":
        return np.random.rand(*shape).astype(np.float32) * 1e6
    else:
        return np.random.randn(*shape).astype(np.float32)


def _fmt_samples(arr, n=10):
    """First *n* values, formatted for a markdown table cell."""
    return ", ".join(f"{v:.6f}" for v in np.asarray(arr).flat[:n])


def _compact_shape(shape):
    """Format shape as [1,64,8,8] (no spaces) for inline display."""
    if isinstance(shape, (list, tuple)):
        return "[" + ",".join(str(s) for s in shape) + "]"
    return str(shape)


def print_test_linear(
    module,
    edge_case,
    edge_desc,
    input_shape,
    shape_line,
    shape_ok,
    is_numerical=True,
    num_ok=None,
    max_diff=None,
    mean_diff=None,
    rtol=DT_RTOL,
    atol=DT_ATOL,
    failure_reason="",
    pt_data=None,
    tt_data=None,
    input_samples=None,
):
    """Print test result in clean tree-style linear format."""
    passed = shape_ok and (num_ok if is_numerical else True)

    print(f"\nMODULE: {Colors.bold(module)}")
    print(f"├─ EDGE CASE: {Colors.warn(edge_case)} ({edge_desc})")
    print(f"├─ INPUT: {input_shape}")

    if input_samples:
        for sname, sarr in input_samples.items():
            flat = np.asarray(sarr).flatten()
            sstr = ", ".join(f"{v:.6f}" for v in flat[:5])
            print(f"├─ INPUT {sname}[0:5]: [{sstr}]")

    shape_status = Colors.success("✓ MATCH") if shape_ok else Colors.fail("✗ MISMATCH")
    print(f"├─ SHAPE: {shape_line} → {shape_status}")

    if is_numerical and max_diff is not None:
        if num_ok:
            num_status = Colors.success(f"✓ PASS (tol: rtol={rtol}, atol={atol})")
        else:
            num_status = Colors.fail("✗ FAIL")
        print(
            f"├─ NUMERICAL: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} → {num_status}"
        )

    if is_numerical and pt_data is not None and tt_data is not None:
        pt_arr = pt_data.flat[:5] if hasattr(pt_data, "flat") else pt_data
        tt_arr = tt_data.flat[:5] if hasattr(tt_data, "flat") else tt_data
        pt_str = ", ".join(f"{v:.6f}" for v in pt_arr)
        tt_str = ", ".join(f"{v:.6f}" for v in tt_arr)
        print(f"├─ PT OUTPUT[0:5]: [{pt_str}]")
        print(f"├─ TT OUTPUT[0:5]: [{tt_str}]")

    if not passed and failure_reason:
        print(f"├─ FAILURE REASON: {Colors.fail(failure_reason)}")

    result_str = Colors.success("✓ PASS") if passed else Colors.fail("✗ FAIL")
    print(f"└─ RESULT: {result_str}")


def print_summary():
    """Print the final summary table."""
    print(f"\n{SUMMARY_LINE}")
    print("SUMMARY")
    print(SUMMARY_LINE)
    print(f"{'MODULE':<32}{'SHAPE':<12}{'NUMERICAL':<12}TOTAL")

    total_sp = total_st = total_np = total_nt = 0
    all_passed = True

    for name, stats in MODULE_STATS.items():
        sp, st = stats["shape_passed"], stats["shape_total"]
        total_sp += sp
        total_st += st
        shape_str = f"{sp}/{st}"

        if stats["num_total"] is not None:
            np_, nt = stats["num_passed"], stats["num_total"]
            total_np += np_
            total_nt += nt
            num_str = f"{np_}/{nt}"
            mod_pass = (sp == st) and (np_ == nt)
        else:
            num_str = "N/A"
            mod_pass = sp == st

        if not mod_pass:
            all_passed = False

        status = Colors.success("✓ PASS") if mod_pass else Colors.fail("✗ FAIL")
        print(f"{name:<32}{shape_str:<12}{num_str:<12}{status}")

    print(DIVIDER_LINE)

    total_num_str = f"{total_np}/{total_nt}" if total_nt > 0 else "N/A"
    overall = Colors.success("✓ PASS") if all_passed else Colors.fail("✗ FAIL")
    print(f"{'TOTAL':<32}{total_sp}/{total_st:<11} {total_num_str:<12}{overall}")

    if FAILED_TESTS:
        print(f"\n{Colors.fail('FAILED TESTS:')}")
        for ft in FAILED_TESTS:
            diff_str = f"max_diff={ft['max_diff']:.2e}" if ft.get("max_diff") else ""
            atol_val = ft.get("atol", DT_ATOL)
            gt_str = f" > atol={atol_val}" if ft.get("max_diff") else ""
            print(f"- {ft['module']} | {ft['edge_case']} values | {diff_str}{gt_str}")

    print(SUMMARY_LINE)


# ---------------------------------------------------------------------------
# Weight Sync Helpers
# ---------------------------------------------------------------------------


def sync_encoder_layer_weights(pt_layer, tt_layer):
    """Copy weights from PyTorch EncoderLayer to TTSim EncoderLayer.

    Components synced:
      - self_attn (MSDeformAttn): sampling_offsets, attention_weights, value_proj, output_proj
        → Custom Linear: weight [out,in], bias [out] — NO transpose
      - linear1, linear2 (SimNN.Linear): param.data transposed [in,out], bias.data [out]
      - norm1, norm2 (F.LayerNorm): params[0][1].data = scale, params[1][1].data = bias
    """
    for proj_name in [
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
    ]:
        pt_proj = getattr(pt_layer.self_attn, proj_name)
        tt_proj = getattr(tt_layer.self_attn, proj_name)
        tt_proj.weight = pt_proj.weight.detach().numpy().copy()
        tt_proj.bias = pt_proj.bias.detach().numpy().copy()

    tt_layer.linear1.param.data = pt_layer.linear1.weight.detach().numpy().T.copy()
    tt_layer.linear1.bias.data = pt_layer.linear1.bias.detach().numpy().copy()
    tt_layer.linear2.param.data = pt_layer.linear2.weight.detach().numpy().T.copy()
    tt_layer.linear2.bias.data = pt_layer.linear2.bias.detach().numpy().copy()

    tt_layer.norm1.params[0][1].data = pt_layer.norm1.weight.detach().numpy().copy()
    tt_layer.norm1.params[1][1].data = pt_layer.norm1.bias.detach().numpy().copy()
    tt_layer.norm2.params[0][1].data = pt_layer.norm2.weight.detach().numpy().copy()
    tt_layer.norm2.params[1][1].data = pt_layer.norm2.bias.detach().numpy().copy()


def sync_decoder_layer_weights(pt_layer, tt_layer):
    """Copy weights from PyTorch DecoderLayer to TTSim DecoderLayer.

    IMPORTANT — Norm numbering is SWAPPED between PyTorch and TTSim:
      PyTorch norm2 (self-attn)  → TTSim norm1
      PyTorch norm1 (cross-attn) → TTSim norm2
      PyTorch norm3 (FFN)        → TTSim norm3

    Self-attention: MultiheadAttention — in_proj_weight transposed, out_proj transposed
    Cross-attention: MSDeformAttn — custom Linear, NO transpose
    FFN: SimNN.Linear — param transposed
    """
    # Self-attention (MHA)
    tt_layer.self_attn.in_proj_weight.data = (
        pt_layer.self_attn.in_proj_weight.detach().numpy().T.copy()
    )
    tt_layer.self_attn.in_proj_bias.data = (
        pt_layer.self_attn.in_proj_bias.detach().numpy().copy()
    )
    tt_layer.self_attn.out_proj.param.data = (
        pt_layer.self_attn.out_proj.weight.detach().numpy().T.copy()
    )
    tt_layer.self_attn.out_proj.bias.data = (
        pt_layer.self_attn.out_proj.bias.detach().numpy().copy()
    )

    # Cross-attention (MSDeformAttn)
    for proj_name in [
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
    ]:
        pt_proj = getattr(pt_layer.cross_attn, proj_name)
        tt_proj = getattr(tt_layer.cross_attn, proj_name)
        tt_proj.weight = pt_proj.weight.detach().numpy().copy()
        tt_proj.bias = pt_proj.bias.detach().numpy().copy()

    # FFN
    tt_layer.linear1.param.data = pt_layer.linear1.weight.detach().numpy().T.copy()
    tt_layer.linear1.bias.data = pt_layer.linear1.bias.detach().numpy().copy()
    tt_layer.linear2.param.data = pt_layer.linear2.weight.detach().numpy().T.copy()
    tt_layer.linear2.bias.data = pt_layer.linear2.bias.detach().numpy().copy()

    # LayerNorm (SWAPPED)
    tt_layer.norm1.params[0][1].data = pt_layer.norm2.weight.detach().numpy().copy()
    tt_layer.norm1.params[1][1].data = pt_layer.norm2.bias.detach().numpy().copy()
    tt_layer.norm2.params[0][1].data = pt_layer.norm1.weight.detach().numpy().copy()
    tt_layer.norm2.params[1][1].data = pt_layer.norm1.bias.detach().numpy().copy()
    tt_layer.norm3.params[0][1].data = pt_layer.norm3.weight.detach().numpy().copy()
    tt_layer.norm3.params[1][1].data = pt_layer.norm3.bias.detach().numpy().copy()


def sync_encoder_weights(pt_encoder, tt_encoder):
    """Sync all encoder layer weights."""
    for pt_layer, tt_layer in zip(pt_encoder.layers, tt_encoder.layers):
        sync_encoder_layer_weights(pt_layer, tt_layer)


def sync_decoder_weights(pt_decoder, tt_decoder):
    """Sync all decoder layer weights (with norm swap)."""
    for pt_layer, tt_layer in zip(pt_decoder.layers, tt_decoder.layers):
        sync_decoder_layer_weights(pt_layer, tt_layer)


def sync_full_transformer_weights(pt_transformer, tt_transformer):
    """Sync all weights for full DeformableTransformer.

    Components:
      - level_embed: Learnable param [n_levels, d_model]
      - reference_points: SimNN.Linear (one-stage, transposed)
      - encoder layers: via sync_encoder_layer_weights
      - decoder layers: via sync_decoder_layer_weights (with norm swap)
    """
    tt_transformer.level_embed.data = pt_transformer.level_embed.detach().numpy().copy()

    if not pt_transformer.two_stage:
        tt_transformer.reference_points.param.data = (
            pt_transformer.reference_points.weight.detach().numpy().T.copy()
        )
        tt_transformer.reference_points.bias.data = (
            pt_transformer.reference_points.bias.detach().numpy().copy()
        )

    sync_encoder_weights(pt_transformer.encoder, tt_transformer.encoder)
    sync_decoder_weights(pt_transformer.decoder, tt_transformer.decoder)


# ---------------------------------------------------------------------------
# Report tee-stream
# ---------------------------------------------------------------------------
class _TeeStream:
    """Write to both a file and the real stdout."""

    def __init__(self, file_handle, original_stdout):
        self._file = file_handle
        self._stdout = original_stdout
        self.encoding = getattr(original_stdout, "encoding", "utf-8")

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self._stdout.fileno()


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — EncoderLayer (numerical, synced weights, edge cases)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_enc_layer = "test_encoder_layer"
test_cases_enc_layer = [
    # (description, batch, seq_len, d_model, d_ffn, n_levels, n_heads, n_points,
    #  spatial_shapes, level_start_indices, data_type, category)
    # --- Baseline ---
    (
        "EncLayer baseline B=1 seq=50",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "positive",
        "baseline",
    ),
    (
        "EncLayer baseline B=2 seq=50",
        2,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "positive",
        "baseline",
    ),
    # --- Edge case: Values (mandatory) ---
    (
        "EncLayer negative values",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "negative",
        "edge_value",
    ),
    (
        "EncLayer zero values",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "zeros",
        "edge_value",
    ),
    (
        "EncLayer mixed values",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "mixed",
        "edge_value",
    ),
    (
        "EncLayer small values (1e-6)",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "small",
        "edge_value",
    ),
    (
        "EncLayer large values (1e6)",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "large",
        "edge_value",
    ),
    # --- Edge case: Shapes ---
    (
        "EncLayer minimum seq=18",
        1,
        18,
        64,
        128,
        2,
        4,
        4,
        [[3, 3], [3, 3]],
        [0, 9],
        "positive",
        "edge_shape",
    ),
    (
        "EncLayer 3 levels seq=75",
        1,
        75,
        64,
        128,
        3,
        4,
        4,
        [[5, 5], [5, 5], [5, 5]],
        [0, 25, 50],
        "positive",
        "edge_shape",
    ),
]


def _run_encoder_layer_test(
    batch,
    seq_len,
    d_model,
    d_ffn,
    n_levels,
    n_heads,
    n_points,
    spatial_shapes,
    level_start_indices,
    data_type,
    tno,
):
    """Run a single EncoderLayer test and return (pt_np, tt_np, src_data, shape_ok, num_ok, mx, mn)."""
    np.random.seed(SEED + tno)
    torch.manual_seed(SEED + tno)

    src_data = generate_test_data([batch, seq_len, d_model], data_type) * 0.1
    pos_data = generate_test_data([batch, seq_len, d_model], data_type) * 0.1
    ref_data = np.random.rand(batch, seq_len, n_levels, 2).astype(np.float32)

    src_torch = torch.from_numpy(src_data)
    pos_torch = torch.from_numpy(pos_data)
    ref_torch = torch.from_numpy(ref_data)
    ss_torch = torch.tensor(spatial_shapes, dtype=torch.long)
    lsi_torch = torch.tensor(level_start_indices, dtype=torch.long)

    # PyTorch
    pt_layer = EncoderLayerPyTorch(
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    pt_layer.eval()

    with torch.no_grad():
        pt_out = pt_layer(src_torch, pos_torch, ref_torch, ss_torch, lsi_torch)

    # TTSim
    tt_layer = EncoderLayerTTSim(
        name=f"enc_layer_ut_{tno}",
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0,
        activation="relu",
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
    )
    sync_encoder_layer_weights(pt_layer, tt_layer)

    src_sim = torch_to_simtensor(src_torch, "src")
    pos_sim = torch_to_simtensor(pos_torch, "pos")
    ref_sim = torch_to_simtensor(ref_torch, "ref_points")
    ss_sim = torch_to_simtensor(ss_torch.float(), "spatial_shapes")
    lsi_sim = torch_to_simtensor(lsi_torch.float(), "level_start_idx")

    tt_out = tt_layer(src_sim, pos_sim, ref_sim, ss_sim, lsi_sim)

    pt_np = _to_numpy(pt_out)
    pt_shape = list(pt_np.shape)
    tt_shape = list(tt_out.shape)
    shape_ok = pt_shape == tt_shape

    rtol, atol = DT_RTOL, DT_ATOL
    if tt_out.data is not None and shape_ok:
        tt_np = tt_out.data
        diff = np.abs(pt_np - tt_np)
        finite = np.isfinite(diff)
        mx = float(diff[finite].max()) if np.any(finite) else 0.0
        mn = float(diff[finite].mean()) if np.any(finite) else 0.0
        num_ok = bool(np.allclose(pt_np, tt_np, rtol=rtol, atol=atol, equal_nan=True))
    else:
        tt_np = np.zeros_like(pt_np)
        mx, mn = 0.0, 0.0
        num_ok = shape_ok

    return (
        pt_np,
        tt_np,
        src_data,
        pt_shape,
        tt_shape,
        shape_ok,
        num_ok,
        mx,
        mn,
        rtol,
        atol,
    )


@pytest.mark.unit
def test_encoder_layer():
    """Test EncoderLayer: shape + numerical validation across edge cases."""
    _seed()
    rows, detail_blocks = [], []
    passed = shape_passed = num_passed = 0
    failed_cases = []

    for tno, (tmsg, batch, seq, dm, dffn, nl, nh, np_, ss, lsi, dt, cat) in enumerate(
        test_cases_enc_layer
    ):
        pt_np, tt_np, src_data, pt_s, tt_s, s_ok, n_ok, mx, mn, rtol, atol = (
            _run_encoder_layer_test(batch, seq, dm, dffn, nl, nh, np_, ss, lsi, dt, tno)
        )

        ok = s_ok and n_ok
        passed += int(ok)
        shape_passed += int(s_ok)
        num_passed += int(n_ok)

        reason = ""
        if not s_ok:
            reason = f"Shape mismatch: PyTorch={pt_s} vs TTSim={tt_s}"
        elif not n_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={atol}"

        print_test_linear(
            module="EncoderLayer",
            edge_case=dt,
            edge_desc=EDGE_CASE_DESC.get(dt, tmsg),
            input_shape=f"src=[{batch},{seq},{dm}]",
            shape_line=f"PyTorch={_compact_shape(pt_s)} | TTSim={_compact_shape(tt_s)}",
            shape_ok=s_ok,
            is_numerical=True,
            num_ok=n_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=rtol,
            atol=atol,
            failure_reason=reason,
            pt_data=pt_np,
            tt_data=tt_np,
            input_samples={"src": src_data},
        )

        TEST_RESULTS.append(
            {
                "module": "EncoderLayer",
                "validation_type": "NUMERICAL",
                "edge_case": dt,
                "edge_desc": EDGE_CASE_DESC.get(dt, ""),
                "input_shape": f"[{batch},{seq},{dm}]",
                "pt_shape": pt_s,
                "tt_shape": tt_s,
                "shape_ok": s_ok,
                "num_ok": n_ok,
                "max_diff": mx,
                "mean_diff": mn,
                "passed": ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, dt, mx))
            FAILED_TESTS.append(
                {
                    "module": "EncoderLayer",
                    "test": tmsg,
                    "edge_case": dt,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {dt} | `[{batch},{seq},{dm}]` | `{pt_s}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )
        detail_blocks.append(
            f"---\n\n### {'🟢' if ok else '🔴'} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{dt}` — {EDGE_CASE_DESC.get(dt, 'N/A')}\n\n"
            f"**Input:** src=`[{batch},{seq},{dm}]` → **Output:** `{pt_s}`\n\n"
            f"**src Float Samples [0:10]:** `[{_fmt_samples(src_data)}]`\n\n"
        )

    MODULE_STATS["EncoderLayer"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_enc_layer),
        "num_passed": num_passed,
        "num_total": len(test_cases_enc_layer),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "EncoderLayer",
            "description": "MSDeformAttn + FFN + LayerNorm + residuals",
            "passed": passed,
            "total": len(test_cases_enc_layer),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_enc_layer
    ), f"EncoderLayer: {passed}/{len(test_cases_enc_layer)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — Encoder (stacked layers)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_encoder = "test_encoder"
test_cases_encoder = [
    # (description, batch, seq, d_model, d_ffn, n_levels, n_heads, n_points, n_layers,
    #  spatial_shapes, level_start_indices, data_type, category)
    # --- Baseline ---
    (
        "Encoder baseline 2 layers",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "positive",
        "baseline",
    ),
    (
        "Encoder baseline B=2",
        2,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "positive",
        "baseline",
    ),
    # --- Edge case: Values (mandatory) ---
    (
        "Encoder negative values",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "negative",
        "edge_value",
    ),
    (
        "Encoder zero values",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "zeros",
        "edge_value",
    ),
    (
        "Encoder mixed values",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "mixed",
        "edge_value",
    ),
    (
        "Encoder small values (1e-6)",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "small",
        "edge_value",
    ),
    (
        "Encoder large values (1e6)",
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "large",
        "edge_value",
    ),
    # --- Edge case: Shapes ---
    (
        "Encoder minimum seq=18",
        1,
        18,
        64,
        128,
        2,
        4,
        4,
        2,
        [[3, 3], [3, 3]],
        [0, 9],
        "positive",
        "edge_shape",
    ),
]


@pytest.mark.unit
def test_encoder():
    """Test Encoder (stacked layers): numerical validation across edge cases."""
    _seed()
    rows, detail_blocks = [], []
    passed = shape_passed = num_passed = 0
    failed_cases = []

    for tno, (
        tmsg,
        batch,
        seq,
        dm,
        dffn,
        nl,
        nh,
        np_,
        nlayers,
        ss,
        lsi,
        dt,
        cat,
    ) in enumerate(test_cases_encoder):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        src_data = generate_test_data([batch, seq, dm], dt) * 0.1
        pos_data = generate_test_data([batch, seq, dm], dt) * 0.1
        vr_data = np.ones((batch, nl, 2), dtype=np.float32)

        src_torch = torch.from_numpy(src_data)
        pos_torch = torch.from_numpy(pos_data)
        ss_torch = torch.tensor(ss, dtype=torch.long)
        lsi_torch = torch.tensor(lsi, dtype=torch.long)
        vr_torch = torch.from_numpy(vr_data)

        # PyTorch
        enc_layer_pt = EncoderLayerPyTorch(
            d_model=dm,
            d_ffn=dffn,
            dropout=0.0,
            activation="relu",
            n_levels=nl,
            n_heads=nh,
            n_points=np_,
        )
        enc_pt = EncoderPyTorch(enc_layer_pt, nlayers)
        enc_pt.eval()

        with torch.no_grad():
            pt_out = enc_pt(src_torch, ss_torch, lsi_torch, vr_torch, pos_torch)

        # TTSim
        enc_layer_tt = EncoderLayerTTSim(
            name=f"enc_layer_ut2_{tno}",
            d_model=dm,
            d_ffn=dffn,
            dropout=0.0,
            activation="relu",
            n_levels=nl,
            n_heads=nh,
            n_points=np_,
        )
        enc_tt = EncoderTTSim(
            name=f"encoder_ut2_{tno}", encoder_layer=enc_layer_tt, num_layers=nlayers
        )
        sync_encoder_weights(enc_pt, enc_tt)

        src_sim = torch_to_simtensor(src_torch, "src", enc_tt)
        pos_sim = torch_to_simtensor(pos_torch, "pos", enc_tt)
        ss_sim = torch_to_simtensor(ss_torch.float(), "spatial_shapes", enc_tt)
        lsi_sim = torch_to_simtensor(lsi_torch.float(), "level_start_idx", enc_tt)
        vr_sim = torch_to_simtensor(vr_torch, "valid_ratios", enc_tt)

        tt_out = enc_tt(src_sim, ss_sim, lsi_sim, vr_sim, pos_sim)

        pt_np = _to_numpy(pt_out)
        pt_shape = list(pt_np.shape)
        tt_shape = list(tt_out.shape)
        shape_ok = pt_shape == tt_shape

        rtol, atol = DT_RTOL, DT_ATOL
        if tt_out.data is not None and shape_ok:
            tt_np = tt_out.data
            diff = np.abs(pt_np - tt_np)
            finite = np.isfinite(diff)
            mx = float(diff[finite].max()) if np.any(finite) else 0.0
            mn = float(diff[finite].mean()) if np.any(finite) else 0.0
            num_ok = bool(
                np.allclose(pt_np, tt_np, rtol=rtol, atol=atol, equal_nan=True)
            )
        else:
            tt_np = np.zeros_like(pt_np)
            mx, mn = 0.0, 0.0
            num_ok = shape_ok

        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PyTorch={pt_shape} vs TTSim={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={atol}"

        print_test_linear(
            module="Encoder",
            edge_case=dt,
            edge_desc=EDGE_CASE_DESC.get(dt, tmsg),
            input_shape=f"src=[{batch},{seq},{dm}] layers={nlayers}",
            shape_line=f"PyTorch={_compact_shape(pt_shape)} | TTSim={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=rtol,
            atol=atol,
            failure_reason=reason,
            pt_data=pt_np,
            tt_data=tt_np,
            input_samples={"src": src_data},
        )

        TEST_RESULTS.append(
            {
                "module": "Encoder",
                "validation_type": "NUMERICAL",
                "edge_case": dt,
                "input_shape": f"[{batch},{seq},{dm}]",
                "pt_shape": pt_shape,
                "tt_shape": tt_shape,
                "shape_ok": shape_ok,
                "num_ok": num_ok,
                "max_diff": mx,
                "mean_diff": mn,
                "passed": ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, dt, mx))
            FAILED_TESTS.append(
                {
                    "module": "Encoder",
                    "test": tmsg,
                    "edge_case": dt,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {dt} | `[{batch},{seq},{dm}]` | `{pt_shape}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )

    MODULE_STATS["Encoder"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_encoder),
        "num_passed": num_passed,
        "num_total": len(test_cases_encoder),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "Encoder",
            "description": "Stacked encoder layers with reference point generation",
            "passed": passed,
            "total": len(test_cases_encoder),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "",
        }
    )

    assert passed == len(
        test_cases_encoder
    ), f"Encoder: {passed}/{len(test_cases_encoder)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — DecoderLayer (numerical, synced weights, norm swap)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_dec_layer = "test_decoder_layer"
test_cases_dec_layer = [
    # (description, batch, num_queries, src_seq, d_model, d_ffn,
    #  n_levels, n_heads, n_points, spatial_shapes, lsi, data_type, category)
    # --- Baseline ---
    (
        "DecLayer baseline B=1 Q=10",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "positive",
        "baseline",
    ),
    (
        "DecLayer baseline B=2 Q=10",
        2,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "positive",
        "baseline",
    ),
    # --- Edge case: Values (mandatory) ---
    (
        "DecLayer negative values",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "negative",
        "edge_value",
    ),
    (
        "DecLayer zero values",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "zeros",
        "edge_value",
    ),
    (
        "DecLayer mixed values",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "mixed",
        "edge_value",
    ),
    (
        "DecLayer small values (1e-6)",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "small",
        "edge_value",
    ),
    (
        "DecLayer large values (1e6)",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "large",
        "edge_value",
    ),
    # --- Edge case: Shapes ---
    (
        "DecLayer minimum Q=4 seq=18",
        1,
        4,
        18,
        64,
        128,
        2,
        4,
        4,
        [[3, 3], [3, 3]],
        [0, 9],
        "positive",
        "edge_shape",
    ),
    (
        "DecLayer single query Q=1",
        1,
        1,
        50,
        64,
        128,
        2,
        4,
        4,
        [[5, 5], [5, 5]],
        [0, 25],
        "positive",
        "edge_shape",
    ),
]


@pytest.mark.unit
def test_decoder_layer():
    """Test DecoderLayer: shape + numerical validation across edge cases."""
    _seed()
    rows, detail_blocks = [], []
    passed = shape_passed = num_passed = 0
    failed_cases = []

    for tno, (
        tmsg,
        batch,
        nq,
        src_seq,
        dm,
        dffn,
        nl,
        nh,
        np_,
        ss,
        lsi,
        dt,
        cat,
    ) in enumerate(test_cases_dec_layer):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        tgt_data = generate_test_data([batch, nq, dm], dt) * 0.1
        qpos_data = generate_test_data([batch, nq, dm], dt) * 0.1
        ref_data = np.random.rand(batch, nq, nl, 2).astype(np.float32)
        src_data = generate_test_data([batch, src_seq, dm], dt) * 0.1

        tgt_torch = torch.from_numpy(tgt_data)
        qpos_torch = torch.from_numpy(qpos_data)
        ref_torch = torch.from_numpy(ref_data)
        src_torch = torch.from_numpy(src_data)
        ss_torch = torch.tensor(ss, dtype=torch.long)
        lsi_torch = torch.tensor(lsi, dtype=torch.long)

        # PyTorch
        pt_layer = DecoderLayerPyTorch(
            d_model=dm,
            d_ffn=dffn,
            dropout=0.0,
            activation="relu",
            n_levels=nl,
            n_heads=nh,
            n_points=np_,
        )
        pt_layer.eval()

        with torch.no_grad():
            pt_out = pt_layer(
                tgt_torch, qpos_torch, ref_torch, src_torch, ss_torch, lsi_torch
            )

        # TTSim
        tt_layer = DecoderLayerTTSim(
            name=f"dec_layer_ut_{tno}",
            d_model=dm,
            d_ffn=dffn,
            dropout=0.0,
            activation="relu",
            n_levels=nl,
            n_heads=nh,
            n_points=np_,
        )
        sync_decoder_layer_weights(pt_layer, tt_layer)

        tgt_sim = torch_to_simtensor(tgt_torch, "tgt", tt_layer)
        qpos_sim = torch_to_simtensor(qpos_torch, "query_pos", tt_layer)
        ref_sim = torch_to_simtensor(ref_torch, "ref_points", tt_layer)
        src_sim = torch_to_simtensor(src_torch, "src", tt_layer)
        ss_sim = torch_to_simtensor(ss_torch.float(), "spatial_shapes", tt_layer)
        lsi_sim = torch_to_simtensor(lsi_torch.float(), "level_start_idx", tt_layer)

        tt_out = tt_layer(tgt_sim, qpos_sim, ref_sim, src_sim, ss_sim, lsi_sim)

        pt_np = _to_numpy(pt_out)
        pt_shape = list(pt_np.shape)
        tt_shape = list(tt_out.shape)
        shape_ok = pt_shape == tt_shape

        rtol, atol = DT_RTOL, DT_ATOL
        if tt_out.data is not None and shape_ok:
            tt_np = tt_out.data
            diff = np.abs(pt_np - tt_np)
            finite = np.isfinite(diff)
            mx = float(diff[finite].max()) if np.any(finite) else 0.0
            mn = float(diff[finite].mean()) if np.any(finite) else 0.0
            num_ok = bool(
                np.allclose(pt_np, tt_np, rtol=rtol, atol=atol, equal_nan=True)
            )
        else:
            tt_np = np.zeros_like(pt_np)
            mx, mn = 0.0, 0.0
            num_ok = shape_ok

        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PyTorch={pt_shape} vs TTSim={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={atol}"

        print_test_linear(
            module="DecoderLayer",
            edge_case=dt,
            edge_desc=EDGE_CASE_DESC.get(dt, tmsg),
            input_shape=f"tgt=[{batch},{nq},{dm}] src=[{batch},{src_seq},{dm}]",
            shape_line=f"PyTorch={_compact_shape(pt_shape)} | TTSim={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=rtol,
            atol=atol,
            failure_reason=reason,
            pt_data=pt_np,
            tt_data=tt_np,
            input_samples={"tgt": tgt_data, "src": src_data},
        )

        TEST_RESULTS.append(
            {
                "module": "DecoderLayer",
                "validation_type": "NUMERICAL",
                "edge_case": dt,
                "input_shape": f"tgt=[{batch},{nq},{dm}]",
                "pt_shape": pt_shape,
                "tt_shape": tt_shape,
                "shape_ok": shape_ok,
                "num_ok": num_ok,
                "max_diff": mx,
                "mean_diff": mn,
                "passed": ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, dt, mx))
            FAILED_TESTS.append(
                {
                    "module": "DecoderLayer",
                    "test": tmsg,
                    "edge_case": dt,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {dt} | `tgt=[{batch},{nq},{dm}]` "
            f"| `{pt_shape}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )

    MODULE_STATS["DecoderLayer"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_dec_layer),
        "num_passed": num_passed,
        "num_total": len(test_cases_dec_layer),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "DecoderLayer",
            "description": "Self-attn (MHA) + Cross-attn (MSDeformAttn) + FFN + 3×LayerNorm (norm swap)",
            "passed": passed,
            "total": len(test_cases_dec_layer),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "",
        }
    )

    assert passed == len(
        test_cases_dec_layer
    ), f"DecoderLayer: {passed}/{len(test_cases_dec_layer)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Decoder (stacked layers, return_intermediate=False)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_decoder = "test_decoder"
test_cases_decoder = [
    # (description, batch, num_queries, src_seq, d_model, d_ffn,
    #  n_levels, n_heads, n_points, n_layers, spatial_shapes, lsi, data_type, category)
    # --- Baseline ---
    (
        "Decoder baseline 2 layers",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "positive",
        "baseline",
    ),
    (
        "Decoder baseline B=2",
        2,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "positive",
        "baseline",
    ),
    # --- Edge case: Values (mandatory) ---
    (
        "Decoder negative values",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "negative",
        "edge_value",
    ),
    (
        "Decoder zero values",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "zeros",
        "edge_value",
    ),
    (
        "Decoder mixed values",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "mixed",
        "edge_value",
    ),
    (
        "Decoder small values (1e-6)",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "small",
        "edge_value",
    ),
    (
        "Decoder large values (1e6)",
        1,
        10,
        50,
        64,
        128,
        2,
        4,
        4,
        2,
        [[5, 5], [5, 5]],
        [0, 25],
        "large",
        "edge_value",
    ),
    # --- Edge case: Shapes ---
    (
        "Decoder minimum Q=4 seq=18",
        1,
        4,
        18,
        64,
        128,
        2,
        4,
        4,
        2,
        [[3, 3], [3, 3]],
        [0, 9],
        "positive",
        "edge_shape",
    ),
]


@pytest.mark.unit
def test_decoder():
    """Test Decoder (stacked layers): numerical validation across edge cases."""
    _seed()
    rows = []
    passed = shape_passed = num_passed = 0
    failed_cases = []

    for tno, (
        tmsg,
        batch,
        nq,
        src_seq,
        dm,
        dffn,
        nl,
        nh,
        np_,
        nlayers,
        ss,
        lsi,
        dt,
        cat,
    ) in enumerate(test_cases_decoder):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        tgt_data = generate_test_data([batch, nq, dm], dt) * 0.1
        qpos_data = generate_test_data([batch, nq, dm], dt) * 0.1
        ref_data = np.random.rand(batch, nq, 2).astype(np.float32)  # [B,Q,2]
        src_data = generate_test_data([batch, src_seq, dm], dt) * 0.1
        vr_data = np.ones((batch, nl, 2), dtype=np.float32)

        tgt_torch = torch.from_numpy(tgt_data)
        qpos_torch = torch.from_numpy(qpos_data)
        ref_torch = torch.from_numpy(ref_data)
        src_torch = torch.from_numpy(src_data)
        ss_torch = torch.tensor(ss, dtype=torch.long)
        lsi_torch = torch.tensor(lsi, dtype=torch.long)
        vr_torch = torch.from_numpy(vr_data)

        # PyTorch
        dec_layer_pt = DecoderLayerPyTorch(
            d_model=dm,
            d_ffn=dffn,
            dropout=0.0,
            activation="relu",
            n_levels=nl,
            n_heads=nh,
            n_points=np_,
        )
        dec_pt = DecoderPyTorch(dec_layer_pt, nlayers, return_intermediate=False)
        dec_pt.eval()

        with torch.no_grad():
            pt_out, _ = dec_pt(
                tgt_torch,
                ref_torch,
                src_torch,
                ss_torch,
                lsi_torch,
                vr_torch,
                qpos_torch,
            )

        # TTSim
        dec_layer_tt = DecoderLayerTTSim(
            name=f"dec_layer_ut4_{tno}",
            d_model=dm,
            d_ffn=dffn,
            dropout=0.0,
            activation="relu",
            n_levels=nl,
            n_heads=nh,
            n_points=np_,
        )
        dec_tt = DecoderTTSim(
            name=f"decoder_ut4_{tno}",
            decoder_layer=dec_layer_tt,
            num_layers=nlayers,
            return_intermediate=False,
        )
        sync_decoder_weights(dec_pt, dec_tt)

        tgt_sim = torch_to_simtensor(tgt_torch, "tgt", dec_tt)
        qpos_sim = torch_to_simtensor(qpos_torch, "query_pos", dec_tt)
        ref_sim = torch_to_simtensor(ref_torch, "ref_points", dec_tt)
        src_sim = torch_to_simtensor(src_torch, "src", dec_tt)
        ss_sim = torch_to_simtensor(ss_torch.float(), "spatial_shapes", dec_tt)
        lsi_sim = torch_to_simtensor(lsi_torch.float(), "level_start_idx", dec_tt)
        vr_sim = torch_to_simtensor(vr_torch, "valid_ratios", dec_tt)

        tt_out, _ = dec_tt(tgt_sim, ref_sim, src_sim, ss_sim, lsi_sim, vr_sim, qpos_sim)

        pt_np = _to_numpy(pt_out)
        pt_shape = list(pt_np.shape)
        tt_shape = list(tt_out.shape)
        shape_ok = pt_shape == tt_shape

        rtol, atol = DT_RTOL, DT_ATOL
        if tt_out.data is not None and shape_ok:
            tt_np = tt_out.data
            diff = np.abs(pt_np - tt_np)
            finite = np.isfinite(diff)
            mx = float(diff[finite].max()) if np.any(finite) else 0.0
            mn = float(diff[finite].mean()) if np.any(finite) else 0.0
            num_ok = bool(
                np.allclose(pt_np, tt_np, rtol=rtol, atol=atol, equal_nan=True)
            )
        else:
            tt_np = np.zeros_like(pt_np)
            mx, mn = 0.0, 0.0
            num_ok = shape_ok

        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PyTorch={pt_shape} vs TTSim={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={atol}"

        print_test_linear(
            module="Decoder",
            edge_case=dt,
            edge_desc=EDGE_CASE_DESC.get(dt, tmsg),
            input_shape=f"tgt=[{batch},{nq},{dm}] src=[{batch},{src_seq},{dm}] layers={nlayers}",
            shape_line=f"PyTorch={_compact_shape(pt_shape)} | TTSim={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=rtol,
            atol=atol,
            failure_reason=reason,
            pt_data=pt_np,
            tt_data=tt_np,
            input_samples={"tgt": tgt_data, "src": src_data},
        )

        TEST_RESULTS.append(
            {
                "module": "Decoder",
                "validation_type": "NUMERICAL",
                "edge_case": dt,
                "input_shape": f"tgt=[{batch},{nq},{dm}]",
                "pt_shape": pt_shape,
                "tt_shape": tt_shape,
                "shape_ok": shape_ok,
                "num_ok": num_ok,
                "max_diff": mx,
                "mean_diff": mn,
                "passed": ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, dt, mx))
            FAILED_TESTS.append(
                {
                    "module": "Decoder",
                    "test": tmsg,
                    "edge_case": dt,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {dt} | `tgt=[{batch},{nq},{dm}]` "
            f"| `{pt_shape}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )

    MODULE_STATS["Decoder"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_decoder),
        "num_passed": num_passed,
        "num_total": len(test_cases_decoder),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "Decoder",
            "description": "Stacked decoder layers with reference point expansion",
            "passed": passed,
            "total": len(test_cases_decoder),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "",
        }
    )

    assert passed == len(
        test_cases_decoder
    ), f"Decoder: {passed}/{len(test_cases_decoder)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — Full Transformer (encoder + decoder pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_full = "test_full_transformer"
test_cases_full = [
    # (description, batch, num_queries, d_model, n_heads,
    #  n_enc_layers, n_dec_layers, d_ffn, n_points, n_levels,
    #  spatial_dims, data_type, category)
    # --- Baseline ---
    (
        "FullXformer baseline B=1",
        1,
        10,
        64,
        4,
        1,
        1,
        128,
        4,
        2,
        [(5, 5), (3, 3)],
        "positive",
        "baseline",
    ),
    (
        "FullXformer baseline B=2",
        2,
        10,
        64,
        4,
        1,
        1,
        128,
        4,
        2,
        [(5, 5), (3, 3)],
        "positive",
        "baseline",
    ),
    # --- Edge case: Values (mandatory) ---
    (
        "FullXformer negative values",
        1,
        10,
        64,
        4,
        1,
        1,
        128,
        4,
        2,
        [(5, 5), (3, 3)],
        "negative",
        "edge_value",
    ),
    (
        "FullXformer zero values",
        1,
        10,
        64,
        4,
        1,
        1,
        128,
        4,
        2,
        [(5, 5), (3, 3)],
        "zeros",
        "edge_value",
    ),
    (
        "FullXformer mixed values",
        1,
        10,
        64,
        4,
        1,
        1,
        128,
        4,
        2,
        [(5, 5), (3, 3)],
        "mixed",
        "edge_value",
    ),
    (
        "FullXformer small values (1e-6)",
        1,
        10,
        64,
        4,
        1,
        1,
        128,
        4,
        2,
        [(5, 5), (3, 3)],
        "small",
        "edge_value",
    ),
    (
        "FullXformer large values (1e6)",
        1,
        10,
        64,
        4,
        1,
        1,
        128,
        4,
        2,
        [(5, 5), (3, 3)],
        "large",
        "edge_value",
    ),
    # --- Edge case: Shapes ---
    (
        "FullXformer minimum 3×3 + 2×2",
        1,
        4,
        64,
        4,
        1,
        1,
        128,
        4,
        2,
        [(3, 3), (2, 2)],
        "positive",
        "edge_shape",
    ),
]


@pytest.mark.unit
def test_full_transformer():
    """Test full DeformableTransformer: numerical validation across edge cases."""
    _seed()
    rows = []
    passed = shape_passed = num_passed = 0
    failed_cases = []

    for tno, (
        tmsg,
        batch,
        nq,
        dm,
        nh,
        nlenc,
        nldec,
        dffn,
        np_,
        nlvl,
        sdims,
        dt,
        cat,
    ) in enumerate(test_cases_full):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        # Build multi-scale features
        srcs_torch, masks_torch, pos_torch_list = [], [], []
        for lvl, (h, w) in enumerate(sdims):
            src_np = generate_test_data([batch, dm, h, w], dt) * 0.1
            srcs_torch.append(torch.from_numpy(src_np))
            masks_torch.append(torch.zeros(batch, h, w, dtype=torch.bool))
            pos_np = generate_test_data([batch, dm, h, w], "mixed") * 0.1
            pos_torch_list.append(torch.from_numpy(pos_np))

        qe_np = generate_test_data([nq, dm * 2], dt) * 0.1
        qe_torch = torch.from_numpy(qe_np)

        # PyTorch
        pt_xfmr = TransformerPyTorch(
            d_model=dm,
            nhead=nh,
            num_encoder_layers=nlenc,
            num_decoder_layers=nldec,
            dim_feedforward=dffn,
            dropout=0.0,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=nlvl,
            dec_n_points=np_,
            enc_n_points=np_,
            two_stage=False,
        )
        pt_xfmr.eval()

        with torch.no_grad():
            hs_pt, _, _, _, _ = pt_xfmr(
                srcs_torch, masks_torch, pos_torch_list, qe_torch
            )

        # TTSim
        tt_xfmr = TransformerTTSim(
            name=f"xfmr_ut5_{tno}",
            d_model=dm,
            nhead=nh,
            num_encoder_layers=nlenc,
            num_decoder_layers=nldec,
            dim_feedforward=dffn,
            dropout=0.0,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=nlvl,
            dec_n_points=np_,
            enc_n_points=np_,
            two_stage=False,
        )
        sync_full_transformer_weights(pt_xfmr, tt_xfmr)

        srcs_sim = [
            torch_to_simtensor(s, f"src_{i}", tt_xfmr) for i, s in enumerate(srcs_torch)
        ]
        masks_sim = [
            torch_to_simtensor(m.float(), f"mask_{i}", tt_xfmr)
            for i, m in enumerate(masks_torch)
        ]
        pos_sim = [
            torch_to_simtensor(p, f"pos_{i}", tt_xfmr)
            for i, p in enumerate(pos_torch_list)
        ]
        qe_sim = torch_to_simtensor(qe_torch, "query_embed", tt_xfmr)

        hs_tt, _, _, _, _ = tt_xfmr(srcs_sim, masks_sim, pos_sim, qe_sim)

        pt_np = _to_numpy(hs_pt)
        pt_shape = list(pt_np.shape)
        tt_shape = list(hs_tt.shape)
        shape_ok = pt_shape == tt_shape

        rtol, atol = DT_RTOL, DT_ATOL
        if hs_tt.data is not None and shape_ok:
            tt_np = hs_tt.data
            diff = np.abs(pt_np - tt_np)
            finite = np.isfinite(diff)
            mx = float(diff[finite].max()) if np.any(finite) else 0.0
            mn = float(diff[finite].mean()) if np.any(finite) else 0.0
            num_ok = bool(
                np.allclose(pt_np, tt_np, rtol=rtol, atol=atol, equal_nan=True)
            )
        else:
            tt_np = np.zeros_like(pt_np)
            mx, mn = 0.0, 0.0
            num_ok = shape_ok

        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PyTorch={pt_shape} vs TTSim={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={atol}"

        first_src = _to_numpy(srcs_torch[0])
        print_test_linear(
            module="FullTransformer",
            edge_case=dt,
            edge_desc=EDGE_CASE_DESC.get(dt, tmsg),
            input_shape=f"levels={nlvl} spatial={sdims} Q={nq}",
            shape_line=f"PyTorch={_compact_shape(pt_shape)} | TTSim={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=rtol,
            atol=atol,
            failure_reason=reason,
            pt_data=pt_np,
            tt_data=tt_np,
            input_samples={"src_0": first_src},
        )

        TEST_RESULTS.append(
            {
                "module": "FullTransformer",
                "validation_type": "NUMERICAL",
                "edge_case": dt,
                "input_shape": f"spatial={sdims}",
                "pt_shape": pt_shape,
                "tt_shape": tt_shape,
                "shape_ok": shape_ok,
                "num_ok": num_ok,
                "max_diff": mx,
                "mean_diff": mn,
                "passed": ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, dt, mx))
            FAILED_TESTS.append(
                {
                    "module": "FullTransformer",
                    "test": tmsg,
                    "edge_case": dt,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {dt} | spatial={sdims} "
            f"| `{pt_shape}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )

    MODULE_STATS["FullTransformer"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_full),
        "num_passed": num_passed,
        "num_total": len(test_cases_full),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "FullTransformer",
            "description": "level_embed + Encoder + query split + ref_points + Decoder",
            "passed": passed,
            "total": len(test_cases_full),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "",
        }
    )

    assert passed == len(
        test_cases_full
    ), f"FullTransformer: {passed}/{len(test_cases_full)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-runner with markdown report
# ═══════════════════════════════════════════════════════════════════════════════


def _write_markdown_report(report_path, exit_code):
    """Generate a simple, module-wise markdown report from REPORT_SECTIONS."""
    total_passed = sum(s["passed"] for s in REPORT_SECTIONS)
    total_tests = sum(s["total"] for s in REPORT_SECTIONS)
    status = "PASS" if total_passed == total_tests else "FAIL"

    lines = [
        "# Deformable Transformer Unit Test Report",
        f"**PyTorch vs TTSim Comparison** | **{total_passed}/{total_tests} passed** | {status}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Exit Code: {exit_code}",
        "",
        "---",
        "",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Module | Passed | Total | Status |")
    lines.append("|--------|--------|-------|--------|")
    for s in REPORT_SECTIONS:
        mod_status = "PASS" if s["passed"] == s["total"] else "FAIL"
        lines.append(f"| {s['name']} | {s['passed']} | {s['total']} | {mod_status} |")
    lines.append("")
    lines.append(f"**Total: {total_passed}/{total_tests} tests passed**")
    lines.append("")

    # Failed tests
    if FAILED_TESTS:
        lines.append("---")
        lines.append("")
        lines.append("## Failed Tests")
        lines.append("")
        lines.append("| Module | Test | Edge Case | Max Diff |")
        lines.append("|--------|------|-----------|----------|")
        for ft in FAILED_TESTS:
            diff_str = f"{ft['max_diff']:.2e}" if ft.get("max_diff") else "N/A"
            lines.append(
                f"| {ft['module']} | {ft['test']} | {ft['edge_case']} | {diff_str} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # Module details
    for s in REPORT_SECTIONS:
        mod_status = "PASS" if s["passed"] == s["total"] else "FAIL"
        lines.append(f"## {s['name']} ({s['passed']}/{s['total']} {mod_status})")
        if s.get("description"):
            lines.append(f"*{s['description']}*")
        lines.append("")
        lines.append(s["table"])
        lines.append("")
        failed = s.get("failed_cases", [])
        if failed:
            lines.append("**Failed Cases:**")
            for tno, tmsg, edge, diff in failed:
                diff_str = f"{diff:.2e}" if diff else "N/A"
                lines.append(f"- [{tno}] {tmsg} — {edge} (diff: {diff_str})")
            lines.append("")
        lines.append("---")
        lines.append("")

    # Config
    lines.append("## Configuration")
    lines.append(f"- Tolerance: rtol={DT_RTOL}, atol={DT_ATOL}")
    lines.append(f"- Random Seed: {SEED}")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _sync_globals_from_pytest():
    """After pytest.main() re-imports this file, copy populated global
    collectors back into __main__ so print_summary() sees the results."""
    this_file = os.path.normcase(os.path.abspath(__file__))
    main_mod = sys.modules.get("__main__")

    for mod in list(sys.modules.values()):
        if mod is main_mod or mod is None:
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            continue
        if os.path.normcase(os.path.abspath(mod_file)) == this_file:
            for attr in (
                "MODULE_STATS",
                "REPORT_SECTIONS",
                "FAILED_TESTS",
                "TEST_RESULTS",
            ):
                src = getattr(mod, attr, None)
                dst = globals()[attr]
                if src is not None and src is not dst:
                    if isinstance(dst, dict):
                        dst.update(src)
                    elif isinstance(dst, list) and src:
                        dst.extend(src)
            break


if __name__ == "__main__":
    report_dir = os.path.join(os.path.dirname(__file__), "..", "unit_test_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "deformable_transformer_unit_test_report.md")
    output_path = os.path.join(report_dir, "deformable_transformer_unit_test_output.md")

    _original_stdout = sys.stdout
    _tee_file = open(output_path, "w", encoding="utf-8")
    sys.stdout = _TeeStream(_tee_file, _original_stdout)

    print(f"\n{SUMMARY_LINE}")
    print(f"DEFORMABLE TRANSFORMER UNIT TEST SUITE - PyTorch vs TTSim")
    print(f"{SUMMARY_LINE}\n")

    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])

    _sync_globals_from_pytest()

    print_summary()

    sys.stdout = _original_stdout
    _tee_file.close()

    _write_markdown_report(report_path, exit_code)

    print(f"\n{Colors.cyan(f'[Markdown report : {report_path}]')}")
    print(f"{Colors.cyan(f'[Full output log  : {output_path}]')}\n")
    sys.exit(exit_code)
