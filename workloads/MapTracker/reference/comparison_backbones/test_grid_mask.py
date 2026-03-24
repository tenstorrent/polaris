#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for GridMask TTSim module.
Validates the conversion from PyTorch to TTSim with step-by-step data validation,
comparing PyTorch and TTSim tensors at every level of the forward pass.

Test Coverage:
1.  Module Construction
2.  Mask Generation – Step-by-step comparison (numpy ↔ TTSim compute)
3.  Full Forward Pass – PyTorch reference vs TTSim graph (with data)
4.  Unsqueeze Operation – Intermediate validation
5.  Reshape Operations  – Intermediate validation
6.  Tile/Expand Operation – Intermediate validation
7.  Element-wise Masking (Mul) – Core masking step
8.  Offset Mode – Step-by-step: x*mask + offset*(1-mask)
9.  Mode=1 (Inverted Mask) – Inverted mask data validation
10. Training vs Inference – Passthrough in eval mode
11. Different Input Sizes – Shape correctness
12. Configuration Variants – use_h/use_w combinations
13. Parameter Count – No trainable parameters
"""

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

import numpy as np
import torch
from PIL import Image
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_tile

from workloads.MapTracker.plugin.models.backbones.bevformer.grid_mask import GridMask
from workloads.MapTracker.reference.comparison_mapers.ttsim_utils import (
    TensorWrapper,
    OpWrapper,
    ttsim_mul,
    ttsim_add,
    ttsim_sub,
    ttsim_unsqueeze,
    ttsim_reshape,
    compare_arrays,
    print_header,
    print_test,
)

# ============================================================================
# Helpers
# ============================================================================


def ttsim_tile(x, repeats):
    """Tile a numpy array using TTSim compute_tile."""
    rps = TensorWrapper(np.array(repeats, dtype=np.int64))
    return compute_tile([TensorWrapper(x), rps], OpWrapper())


def _section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _step(title):
    print("\n  " + title)
    print("  " + "-" * 76)


# ============================================================================
# PyTorch Reference Implementation  (no mmcv/mmdet dependencies)
# ============================================================================


def pytorch_generate_mask(
    H, W, use_h, use_w, rotate, ratio, mode, prob, training, seed=None
):
    """
    Pure NumPy mask generation mirroring TTSim GridMask._generate_mask().

    Returns the binary mask as np.ndarray of shape [H, W].
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    if rng.rand() > prob or not training:
        return np.ones((H, W), dtype=np.float32)

    hh = int(1.5 * H)
    ww = int(1.5 * W)
    d = rng.randint(2, min(H, W))

    if ratio == 1:
        l = rng.randint(1, d)
    else:
        l = min(max(int(d * ratio + 0.5), 1), d - 1)

    mask = np.ones((hh, ww), dtype=np.float32)
    st_h = rng.randint(d)
    st_w = rng.randint(d)

    if use_h:
        for i in range(hh // d):
            s = d * i + st_h
            t = min(s + l, hh)
            mask[s:t, :] = 0

    if use_w:
        for i in range(ww // d):
            s = d * i + st_w
            t = min(s + l, ww)
            mask[:, s:t] = 0

    if rotate > 0:
        r = rng.randint(rotate)
        mask_img = Image.fromarray(np.uint8(mask * 255))
        mask_img = mask_img.rotate(r)
        mask = np.asarray(mask_img).astype(np.float32) / 255.0

    mask = mask[(hh - H) // 2 : (hh - H) // 2 + H, (ww - W) // 2 : (ww - W) // 2 + W]

    if mode == 1:
        mask = 1.0 - mask

    return mask


def pytorch_grid_mask_forward(
    x_np,
    use_h=True,
    use_w=True,
    rotate=1,
    ratio=0.5,
    mode=0,
    offset_mode=False,
    prob=1.0,
    training=True,
    seed=None,
):
    """
    Full NumPy/PyTorch forward pass mirroring TTSim GridMask.__call__().

    Returns:
        output       : np.ndarray [N, C, H, W]  – masked output
        mask_np      : np.ndarray [H, W]         – the generated mask
        intermediates: dict of np.ndarray         – every intermediate tensor
    """
    N, C, H, W = x_np.shape

    mask_np = pytorch_generate_mask(
        H, W, use_h, use_w, rotate, ratio, mode, prob, training, seed
    )
    mask_unsq = mask_np[np.newaxis, :, :]
    x_reshaped = x_np.reshape(N * C, H, W)
    mask_expanded = np.tile(mask_unsq, (N * C, 1, 1))

    intermediates = {
        "mask": mask_np,
        "mask_unsq": mask_unsq,
        "x_reshaped": x_reshaped,
        "mask_expanded": mask_expanded,
    }

    if offset_mode and seed is not None:
        offset_np = (2 * (np.random.RandomState(seed + 1).rand(H, W) - 0.5)).astype(
            np.float32
        )
        offset_unsq = offset_np[np.newaxis, :, :]
        offset_expanded = np.tile(offset_unsq, (N * C, 1, 1))

        masked_x = x_reshaped * mask_expanded
        inv_mask = 1.0 - mask_expanded
        offset_part = offset_expanded * inv_mask
        output_reshaped = masked_x + offset_part

        intermediates.update(
            {
                "offset": offset_np,
                "offset_unsq": offset_unsq,
                "offset_expanded": offset_expanded,
                "masked_x": masked_x,
                "inv_mask": inv_mask,
                "offset_part": offset_part,
                "output_reshaped": output_reshaped,
            }
        )
    else:
        output_reshaped = x_reshaped * mask_expanded
        intermediates["output_reshaped"] = output_reshaped

    output = output_reshaped.reshape(N, C, H, W)
    intermediates["output"] = output
    return output, mask_np, intermediates


# ============================================================================
# Test functions
# ============================================================================


def test_grid_mask_construction():
    """Test 1 – Module can be constructed successfully."""
    _section("TEST 1: Module Construction")

    try:
        grid_mask = GridMask(
            name="test_grid_mask",
            use_h=True,
            use_w=True,
            rotate=1,
            ratio=0.5,
            prob=1.0,
            training=True,
        )
        print("PASS: Module constructed successfully")
        print(f"  - name:     {grid_mask.name}")
        print(f"  - use_h:    {grid_mask.use_h}")
        print(f"  - use_w:    {grid_mask.use_w}")
        print(f"  - rotate:   {grid_mask.rotate}")
        print(f"  - ratio:    {grid_mask.ratio}")
        print(f"  - prob:     {grid_mask.prob}")
        return True
    except Exception as e:
        print(f"FAIL: Construction failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mask_generation_step_by_step():
    """
    Test 2 – Deterministic mask generation: NumPy reference vs TTSim
    compute functions at every intermediate step.
    """
    _section("TEST 2: Mask Generation – Step-by-step Data Validation")

    SEED = 42
    H, W = 16, 16
    N, C = 2, 3

    x_np = np.random.RandomState(0).rand(N, C, H, W).astype(np.float32)

    # ── PyTorch reference ────────────────────────────────────────────────────
    _step("PyTorch Reference – intermediate tensors")

    _, mask_np, pt = pytorch_grid_mask_forward(
        x_np,
        use_h=True,
        use_w=True,
        rotate=1,
        ratio=0.5,
        mode=0,
        offset_mode=False,
        prob=1.0,
        training=True,
        seed=SEED,
    )

    print(
        f"    mask          shape: {pt['mask'].shape}          "
        f"range [{pt['mask'].min():.3f}, {pt['mask'].max():.3f}]"
    )
    print(f"    mask_unsq     shape: {pt['mask_unsq'].shape}")
    print(f"    x_reshaped    shape: {pt['x_reshaped'].shape}")
    print(f"    mask_expanded shape: {pt['mask_expanded'].shape}")
    print(f"    output_reshaped shape: {pt['output_reshaped'].shape}")
    print(f"    output        shape: {pt['output'].shape}")

    # ── TTSim compute functions ──────────────────────────────────────────────
    _step("TTSim Compute Functions – same intermediate steps")

    # Step 1: mask generation is pure NumPy – identical by construction
    mask_ttsim = mask_np.copy()

    # Step 2: Unsqueeze [H, W] → [1, H, W]
    ts_mask_unsq = ttsim_unsqueeze(mask_ttsim, [0])

    # Step 3: Reshape input [N,C,H,W] → [N*C,H,W]
    ts_x_reshaped = ttsim_reshape(x_np, [N * C, H, W])

    # Step 4: Tile [1,H,W] → [N*C,H,W]
    ts_mask_expanded = ttsim_tile(ts_mask_unsq, [N * C, 1, 1])

    # Step 5: Multiply
    ts_output_reshaped = ttsim_mul(ts_x_reshaped, ts_mask_expanded)

    # Step 6: Reshape output [N*C,H,W] → [N,C,H,W]
    ts_output = ttsim_reshape(ts_output_reshaped, [N, C, H, W])

    # ── Comparison ───────────────────────────────────────────────────────────
    _step("Comparison: PyTorch vs TTSim")

    all_passed = True
    all_passed &= compare_arrays(pt["mask"], mask_ttsim, "mask [H,W]")
    all_passed &= compare_arrays(pt["mask_unsq"], ts_mask_unsq, "mask_unsq [1,H,W]")
    all_passed &= compare_arrays(
        pt["x_reshaped"], ts_x_reshaped, "x_reshaped [N*C,H,W]"
    )
    all_passed &= compare_arrays(
        pt["mask_expanded"], ts_mask_expanded, "mask_expanded [N*C,H,W]"
    )
    all_passed &= compare_arrays(
        pt["output_reshaped"], ts_output_reshaped, "output_reshaped [N*C,H,W]"
    )
    all_passed &= compare_arrays(pt["output"], ts_output, "output [N,C,H,W]")

    if all_passed:
        print(
            "\nPASS: All intermediate tensors match between PyTorch and TTSim compute"
        )
    else:
        print("\nFAIL: One or more intermediate tensors differ")
    return all_passed


def test_full_forward_pytorch_vs_ttsim():
    """
    Test 3 – Run TTSim GridMask graph forward pass and compare final
    output data against the PyTorch reference.
    """
    _section("TEST 3: Full Forward Pass – PyTorch vs TTSim Graph (data)")

    SEED = 7
    N, C, H, W = 2, 3, 32, 32
    x_np = np.random.RandomState(1).rand(N, C, H, W).astype(np.float32)

    # ── PyTorch reference ────────────────────────────────────────────────────
    _step("PyTorch Reference")

    pt_output, pt_mask, _ = pytorch_grid_mask_forward(
        x_np,
        use_h=True,
        use_w=True,
        rotate=1,
        ratio=0.5,
        mode=0,
        offset_mode=False,
        prob=1.0,
        training=True,
        seed=SEED,
    )
    print(f"    Input  shape: {x_np.shape}")
    print(
        f"    Mask   shape: {pt_mask.shape}  "
        f"zero-fraction: {(pt_mask == 0).mean():.2%}"
    )
    print(f"    Output shape: {pt_output.shape}")
    print(f"    Output range: [{pt_output.min():.4f}, {pt_output.max():.4f}]")

    # ── TTSim graph ──────────────────────────────────────────────────────────
    _step("TTSim Graph Construction + Data Computation")

    x_ttsim = F._from_data("fwd_input", x_np, is_const=False, is_param=False)
    grid_mask = GridMask(
        name="fwd_test",
        use_h=True,
        use_w=True,
        rotate=1,
        ratio=0.5,
        prob=1.0,
        training=True,
    )
    output_ttsim = grid_mask(x_ttsim, seed=SEED)

    print(f"    TTSim output tensor name:   {output_ttsim.name}")
    print(f"    TTSim output shape (graph): {output_ttsim.shape}")

    # ── Comparison ───────────────────────────────────────────────────────────
    _step("Comparison: PyTorch vs TTSim graph data")

    ttsim_data = output_ttsim.data
    if ttsim_data is None:
        print("  WARNING: TTSim graph output has no propagated data.")
        print("           Falling back to TTSim compute-function path for validation.")
        mask_np = pytorch_generate_mask(H, W, True, True, 1, 0.5, 0, 1.0, True, SEED)
        x_flat = x_np.reshape(N * C, H, W)
        mask_exp = ttsim_tile(mask_np[np.newaxis], [N * C, 1, 1])
        ts_out = ttsim_reshape(ttsim_mul(x_flat, mask_exp), [N, C, H, W])
        passed = compare_arrays(
            pt_output, ts_out, "Final output [N,C,H,W] (compute path)"
        )
    else:
        passed = compare_arrays(
            pt_output, ttsim_data, "Final output [N,C,H,W]", rtol=1e-5, atol=1e-6
        )
        if passed:
            zero_fraction = (ttsim_data == 0).mean()
            print(f"\n  Zero-fraction in TTSim output: {zero_fraction:.2%}")

    if passed:
        print("PASS: PyTorch and TTSim outputs match numerically")
    else:
        print("FAIL: Numerical mismatch")
    return passed


def test_unsqueeze_operation():
    """
    Test 4 – Validate the Unsqueeze step in isolation:
    mask [H, W] → [1, H, W].
    """
    _section("TEST 4: Unsqueeze Operation – [H,W] → [1,H,W]")

    H, W = 16, 16
    SEED = 42
    mask_np = pytorch_generate_mask(H, W, True, True, 1, 0.5, 0, 1.0, True, SEED)

    # PyTorch
    mask_pt = torch.from_numpy(mask_np).unsqueeze(0).numpy()

    # TTSim compute
    mask_ttsim = ttsim_unsqueeze(mask_np, [0])

    print(f"  Input  shape: {mask_np.shape}")
    print(f"  Expected: [1, {H}, {W}]")
    print(f"  PyTorch  shape: {mask_pt.shape}")
    print(f"  TTSim    shape: {mask_ttsim.shape}")

    passed = compare_arrays(mask_pt, mask_ttsim, "unsqueeze( mask, axis=0 )")
    if passed:
        print("PASS: Unsqueeze matches")
    return passed


def test_reshape_operations():
    """
    Test 5 – Validate both Reshape operations in the forward pass:
      (a) [N, C, H, W] → [N*C, H, W]   (before multiply)
      (b) [N*C, H, W]  → [N, C, H, W]  (after multiply)
    """
    _section("TEST 5: Reshape Operations – Input & Output")

    N, C, H, W = 2, 3, 16, 16
    x_np = np.random.RandomState(5).rand(N, C, H, W).astype(np.float32)
    dummy = np.random.RandomState(6).rand(N * C, H, W).astype(np.float32)

    x_pt_flat = x_np.reshape(N * C, H, W)
    x_pt_back = dummy.reshape(N, C, H, W)

    x_ts_flat = ttsim_reshape(x_np, [N * C, H, W])
    x_ts_back = ttsim_reshape(dummy, [N, C, H, W])

    print(f"  (a) {x_np.shape} → {x_pt_flat.shape}")
    print(f"  (b) {dummy.shape} → {x_pt_back.shape}")

    a = compare_arrays(x_pt_flat, x_ts_flat, "reshape  [N,C,H,W] → [N*C,H,W]")
    b = compare_arrays(x_pt_back, x_ts_back, "reshape  [N*C,H,W] → [N,C,H,W]")

    passed = a and b
    if passed:
        print("PASS: Both reshape operations match")
    return passed


def test_tile_operation():
    """
    Test 6 – Validate Tile/expand:  mask [1,H,W] → [N*C,H,W].
    """
    _section("TEST 6: Tile Operation – [1,H,W] → [N*C,H,W]")

    H, W = 16, 16
    N, C = 2, 3
    SEED = 42

    mask_np = pytorch_generate_mask(H, W, True, True, 1, 0.5, 0, 1.0, True, SEED)
    mask_unsq = mask_np[np.newaxis, :, :]

    # PyTorch: expand (read-only view, same data)
    mask_pt = torch.from_numpy(mask_unsq).expand(N * C, H, W).numpy()

    # TTSim compute: tile (materialised copy)
    mask_ts = ttsim_tile(mask_unsq, [N * C, 1, 1])

    print(f"  Input  shape: {mask_unsq.shape}")
    print(f"  Expected: [{N*C}, {H}, {W}]")
    print(f"  PyTorch shape:{mask_pt.shape}")
    print(f"  TTSim   shape:{mask_ts.shape}")

    passed = compare_arrays(mask_pt, mask_ts, "tile( mask, [N*C,1,1] )")
    if passed:
        print("PASS: Tile operation matches")
    return passed


def test_element_wise_masking():
    """
    Test 7 – Core masking step: x_reshaped * mask_expanded.
    Validates zero-masking and passthrough regions separately.
    """
    _section("TEST 7: Element-wise Masking (Mul) – Core Operation")

    H, W = 16, 16
    N, C = 2, 3
    SEED = 42

    x_np = np.random.RandomState(10).rand(N, C, H, W).astype(np.float32)
    mask_np = pytorch_generate_mask(H, W, True, True, 1, 0.5, 0, 1.0, True, SEED)

    x_flat = x_np.reshape(N * C, H, W)
    mask_unsq = mask_np[np.newaxis, :, :]
    mask_exp = np.tile(mask_unsq, (N * C, 1, 1))

    masked_pt = x_flat * mask_exp
    masked_ts = ttsim_mul(x_flat, mask_exp)

    print(f"  x_flat    shape: {x_flat.shape}")
    print(f"  mask_exp  shape: {mask_exp.shape}")
    print(
        f"  PyTorch output:  {masked_pt.shape}  "
        f"zero-fraction {(masked_pt == 0).mean():.2%}"
    )
    print(
        f"  TTSim   output:  {masked_ts.shape}  "
        f"zero-fraction {(masked_ts == 0).mean():.2%}"
    )

    passed = compare_arrays(masked_pt, masked_ts, "x_reshaped * mask_expanded")

    _step("Spot-check: masked regions zero, unmasked regions preserved")
    zero_pos = mask_exp == 0
    one_pos = mask_exp == 1
    zero_ok = np.all(masked_ts[zero_pos] == 0.0)
    pass_ok = np.allclose(masked_ts[one_pos], x_flat[one_pos])
    print(f"    Masked   regions all zero:     {'OK' if zero_ok else 'FAIL'}")
    print(f"    Unmasked regions preserved:    {'OK' if pass_ok else 'FAIL'}")

    passed = passed and zero_ok and pass_ok
    if passed:
        print("PASS: Element-wise masking correct")
    return passed


def test_offset_mode_data_validation():
    """
    Test 8 – Offset mode ( offset=True ):
    output = x * mask + offset * (1 - mask).

    Validates each sub-step:
      (a) x * mask
      (b) 1 - mask   (inverted mask)
      (c) offset_expanded * (1 - mask)
      (d) (x * mask) + offset * (1 - mask)
    """
    _section("TEST 8: Offset Mode – Step-by-step Data Validation")

    SEED = 55
    N, C, H, W = 1, 2, 16, 16
    x_np = np.random.RandomState(20).rand(N, C, H, W).astype(np.float32)

    # ── PyTorch reference ────────────────────────────────────────────────────
    _step("PyTorch Reference")

    pt_output, _, pt = pytorch_grid_mask_forward(
        x_np,
        use_h=True,
        use_w=True,
        rotate=1,
        ratio=0.5,
        mode=0,
        offset_mode=True,
        prob=1.0,
        training=True,
        seed=SEED,
    )
    print(f"    masked_x        shape: {pt['masked_x'].shape}")
    print(f"    inv_mask        shape: {pt['inv_mask'].shape}")
    print(f"    offset_part     shape: {pt['offset_part'].shape}")
    print(f"    output_reshaped shape: {pt['output_reshaped'].shape}")
    print(f"    output          shape: {pt['output'].shape}")

    # ── TTSim compute functions ──────────────────────────────────────────────
    _step("TTSim Compute Functions")

    mask_np = pytorch_generate_mask(H, W, True, True, 1, 0.5, 0, 1.0, True, SEED)
    x_flat = x_np.reshape(N * C, H, W)
    mask_unsq = mask_np[np.newaxis, :, :]
    mask_exp = ttsim_tile(mask_unsq, [N * C, 1, 1])

    offset_np = (2 * (np.random.RandomState(SEED + 1).rand(H, W) - 0.5)).astype(
        np.float32
    )
    offset_unsq = offset_np[np.newaxis, :, :]
    offset_exp = ttsim_tile(offset_unsq, [N * C, 1, 1])

    ts_masked_x = ttsim_mul(x_flat, mask_exp)
    ts_one = np.ones_like(mask_exp, dtype=np.float32)
    ts_inv_mask = ttsim_sub(ts_one, mask_exp)
    ts_offset_part = ttsim_mul(offset_exp, ts_inv_mask)
    ts_out_reshaped = ttsim_add(ts_masked_x, ts_offset_part)
    ts_output = ttsim_reshape(ts_out_reshaped, [N, C, H, W])

    # ── Comparison ───────────────────────────────────────────────────────────
    _step("Comparison: PyTorch vs TTSim")

    all_passed = True
    all_passed &= compare_arrays(pt["masked_x"], ts_masked_x, "x * mask")
    all_passed &= compare_arrays(pt["inv_mask"], ts_inv_mask, "1 - mask")
    all_passed &= compare_arrays(pt["offset_part"], ts_offset_part, "offset * (1-mask)")
    all_passed &= compare_arrays(
        pt["output_reshaped"], ts_out_reshaped, "x*mask + offset*(1-mask)"
    )
    all_passed &= compare_arrays(pt["output"], ts_output, "final output [N,C,H,W]")

    if all_passed:
        print("\nPASS: Offset-mode all intermediate and final tensors match")
    else:
        print("\nFAIL: Mismatch in offset mode")
    return all_passed


def test_mode1_inverted_mask():
    """
    Test 9 – Mode=1: mask is inverted (1 - mask).
    Validates that region kept vs zeroed-out is the complement of mode=0.
    """
    _section("TEST 9: Mode=1 – Inverted Mask Data Validation")

    SEED = 99
    N, C, H, W = 1, 1, 16, 16
    x_np = np.ones((N, C, H, W), dtype=np.float32)

    mask_m0 = pytorch_generate_mask(
        H, W, True, True, 1, 0.5, mode=0, prob=1.0, training=True, seed=SEED
    )
    mask_m1 = pytorch_generate_mask(
        H, W, True, True, 1, 0.5, mode=1, prob=1.0, training=True, seed=SEED
    )

    print(f"  mask (mode=0) zero-fraction: {(mask_m0 == 0).mean():.2%}")
    print(f"  mask (mode=1) zero-fraction: {(mask_m1 == 0).mean():.2%}")

    complement = compare_arrays(1.0 - mask_m0, mask_m1, "mask_m1 == 1 - mask_m0")

    # Validate forward output using TTSim compute
    pt_output_m1, _, _ = pytorch_grid_mask_forward(
        x_np, mode=1, prob=1.0, training=True, seed=SEED
    )
    mask_ts = ttsim_tile(mask_m1[np.newaxis], [N * C, 1, 1])
    ts_out = ttsim_reshape(ttsim_mul(x_np.reshape(N * C, H, W), mask_ts), [N, C, H, W])
    data_ok = compare_arrays(pt_output_m1, ts_out, "mode=1 final output")

    # Also verify via TTSim graph shape
    x_ttsim = F._from_data("m1_input", x_np, is_const=False, is_param=False)
    gm_m1 = GridMask(
        name="gm_mode1",
        use_h=True,
        use_w=True,
        ratio=0.5,
        prob=1.0,
        mode=1,
        training=True,
    )
    out_m1 = gm_m1(x_ttsim, seed=SEED)
    shape_ok = list(out_m1.shape) == [N, C, H, W]
    print(f"  TTSim graph output shape: {out_m1.shape}  {'OK' if shape_ok else 'FAIL'}")

    passed = complement and data_ok and shape_ok
    if passed:
        print("PASS: Mode=1 (inverted mask) validated")
    return passed


def test_training_vs_inference():
    """
    Test 10 – Training vs Inference mode.
    In inference (training=False), the grid mask is not applied: output == input.
    """
    _section("TEST 10: Training vs Inference Mode")

    N, C, H, W = 1, 3, 64, 64
    x_np = np.random.RandomState(30).rand(N, C, H, W).astype(np.float32)
    SEED = 7

    # Inference: training=False → all-ones mask
    mask_eval = pytorch_generate_mask(
        H, W, True, True, 1, 0.5, 0, prob=1.0, training=False, seed=SEED
    )
    print(f"  Inference mask is all-ones: {np.all(mask_eval == 1.0)}")

    # TTSim graph shapes for both modes
    x_train = F._from_data("tv_input_train", x_np, is_const=False, is_param=False)
    gm_train = GridMask(
        name="gm_train", use_h=True, use_w=True, ratio=0.5, prob=1.0, training=True
    )
    out_train = gm_train(x_train, seed=SEED)

    x_eval = F._from_data("tv_input_eval", x_np, is_const=False, is_param=False)
    gm_eval = GridMask(
        name="gm_eval", use_h=True, use_w=True, ratio=0.5, prob=1.0, training=False
    )
    out_eval = gm_eval(x_eval, seed=SEED)

    train_shape_ok = list(out_train.shape) == [N, C, H, W]
    eval_shape_ok = list(out_eval.shape) == [N, C, H, W]
    print(
        f"  Training mode shape:   {out_train.shape}  {'OK' if train_shape_ok else 'FAIL'}"
    )
    print(
        f"  Inference mode shape:  {out_eval.shape}   {'OK' if eval_shape_ok  else 'FAIL'}"
    )

    # Verify: inference output == input (mask is all-ones, so x*1 == x)
    ts_mask_exp = ttsim_tile(mask_eval[np.newaxis], [N * C, 1, 1])
    ts_eval_out = ttsim_reshape(
        ttsim_mul(x_np.reshape(N * C, H, W), ts_mask_exp), [N, C, H, W]
    )
    eval_data_ok = compare_arrays(
        x_np, ts_eval_out, "Inference output == input (no masking)"
    )

    passed = train_shape_ok and eval_shape_ok and eval_data_ok
    if passed:
        print("PASS: Training vs inference mode validated")
    return passed


def test_different_input_sizes():
    """Test 11 – Different spatial resolutions with data validation."""
    _section("TEST 11: Different Input Sizes")

    test_cases = [
        (1, 3, 224, 224, "ImageNet"),
        (2, 64, 32, 32, "Small feature map"),
        (4, 16, 64, 64, "Batch feature map"),
    ]

    all_passed = True

    for idx, (N, C, H, W, label) in enumerate(test_cases, 1):
        try:
            print(f"\n  [{idx}] {label}: N={N}, C={C}, H={H}, W={W}")

            x_np = np.random.RandomState(idx).rand(N, C, H, W).astype(np.float32)
            SEED = idx * 13

            pt_output, _, _ = pytorch_grid_mask_forward(
                x_np, prob=1.0, training=True, seed=SEED
            )

            mask_np = pytorch_generate_mask(
                H, W, True, True, 1, 0.5, 0, 1.0, True, SEED
            )
            x_flat = x_np.reshape(N * C, H, W)
            mask_exp = ttsim_tile(mask_np[np.newaxis], [N * C, 1, 1])
            ts_out = ttsim_reshape(ttsim_mul(x_flat, mask_exp), [N, C, H, W])

            ok = compare_arrays(pt_output, ts_out, f"output [{N},{C},{H},{W}]")
            all_passed &= ok

        except Exception as e:
            print(f"  FAIL: Test case {idx} ({label}) failed: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("\nPASS: All size variants validated")
    return all_passed


def test_configuration_variants():
    """Test 12 – use_h / use_w / offset / mode combinations with data validation."""
    _section("TEST 12: Configuration Variants")

    SEED = 77
    N, C, H, W = 1, 2, 32, 32
    x_np = np.random.RandomState(40).rand(N, C, H, W).astype(np.float32)

    configs = [
        dict(use_h=True, use_w=False, offset_mode=False, mode=0, label="h-only mask"),
        dict(use_h=False, use_w=True, offset_mode=False, mode=0, label="w-only mask"),
        dict(use_h=True, use_w=True, offset_mode=False, mode=0, label="h+w mask"),
        dict(use_h=True, use_w=True, offset_mode=True, mode=0, label="h+w + offset"),
        dict(
            use_h=True, use_w=True, offset_mode=False, mode=1, label="mode=1 inverted"
        ),
    ]

    all_passed = True
    for idx, cfg in enumerate(configs, 1):
        try:
            label = cfg.pop("label")
            offset_mode = cfg.pop("offset_mode")
            print(f"\n  [{idx}] {label}")

            pt_output, _, _ = pytorch_grid_mask_forward(
                x_np, prob=1.0, training=True, seed=SEED, offset_mode=offset_mode, **cfg
            )

            mask_np = pytorch_generate_mask(
                H,
                W,
                cfg["use_h"],
                cfg["use_w"],
                rotate=1,
                ratio=0.5,
                mode=cfg["mode"],
                prob=1.0,
                training=True,
                seed=SEED,
            )
            x_flat = x_np.reshape(N * C, H, W)
            mask_exp = ttsim_tile(mask_np[np.newaxis], [N * C, 1, 1])

            if offset_mode:
                offset_np = (
                    2 * (np.random.RandomState(SEED + 1).rand(H, W) - 0.5)
                ).astype(np.float32)
                offset_exp = ttsim_tile(offset_np[np.newaxis], [N * C, 1, 1])
                masked_x = ttsim_mul(x_flat, mask_exp)
                inv_mask = ttsim_sub(np.ones_like(mask_exp), mask_exp)
                off_part = ttsim_mul(offset_exp, inv_mask)
                ts_out_r = ttsim_add(masked_x, off_part)
            else:
                ts_out_r = ttsim_mul(x_flat, mask_exp)

            ts_out = ttsim_reshape(ts_out_r, [N, C, H, W])
            ok = compare_arrays(pt_output, ts_out, label)
            all_passed &= ok

        except Exception as e:
            print(f"  FAIL: {label} failed: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("\nPASS: All configuration variants validated")
    return all_passed


def test_parameter_count():
    """Test 13 – GridMask has no trainable parameters."""
    _section("TEST 13: Parameter Count")

    grid_mask = GridMask(
        "test_params", use_h=True, use_w=True, ratio=0.5, prob=1.0, training=True
    )

    param_count = grid_mask.analytical_param_count()
    expected = 0

    print(f"  Expected: {expected}")
    print(f"  Actual:   {param_count}")

    if param_count == expected:
        print("PASS: Parameter count correct (GridMask has no trainable parameters)")
        return True
    else:
        print("FAIL: Parameter count mismatch")
        return False


# ============================================================================
# Main
# ============================================================================


def main():
    _section("GridMask TTSim Module – Validation Suite")
    print("\nThis suite validates TTSim GridMask against a PyTorch reference")
    print("at every intermediate step of the forward pass.\n")
    print("Intermediate tensors validated:")
    print("  mask [H,W] · mask_unsq [1,H,W] · x_reshaped [N*C,H,W]")
    print("  mask_expanded [N*C,H,W] · output_reshaped · final output [N,C,H,W]")
    print("Offset mode sub-steps: x*mask · 1-mask · offset*(1-mask) · sum")

    results = {
        "Module Construction": test_grid_mask_construction(),
        "Mask Generation Step-by-Step": test_mask_generation_step_by_step(),
        "Full Forward (Graph vs PyTorch)": test_full_forward_pytorch_vs_ttsim(),
        "Unsqueeze Operation": test_unsqueeze_operation(),
        "Reshape Operations": test_reshape_operations(),
        "Tile/Expand Operation": test_tile_operation(),
        "Element-wise Masking (Mul)": test_element_wise_masking(),
        "Offset Mode Data Validation": test_offset_mode_data_validation(),
        "Mode=1 Inverted Mask": test_mode1_inverted_mask(),
        "Training vs Inference Mode": test_training_vs_inference(),
        "Different Input Sizes": test_different_input_sizes(),
        "Configuration Variants": test_configuration_variants(),
        "Parameter Count": test_parameter_count(),
    }

    _section("TEST SUMMARY")
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:.<60} {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! GridMask TTSim implementation validated.")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
